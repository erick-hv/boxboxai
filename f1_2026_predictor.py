"""
F1 2026 - Predictor de Ganador de Carrera
==========================================
Versión: 7.0 — Bayesian + Momentum + Component Risk
Modelo híbrido: Pesos manuales (<6 carreras) → XGBoost automático (>=6 carreras)

Fuentes de datos:
  - Jolpica-F1 API  : resultados, clasificación, pitstops, penalizaciones, abandonos
  - OpenF1 API      : clima, mensajes de control de carrera
  - FastF1          : tiempos de vuelta, telemetría, sectores, stint data

Instalar dependencias:
  pip install fastf1 pandas numpy tabulate requests xgboost scikit-learn

Uso:
  python f1_2026_predictor.py
"""

import warnings
warnings.filterwarnings("ignore")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
import fastf1
import pandas as pd
import numpy as np
from tabulate import tabulate
from pathlib import Path
import time
import sys

# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
SEASON          = 2026
CACHE_DIR       = "./f1_cache"
OUTPUT_CSV      = "f1_2026_predicciones.csv"
JOLPICA_BASE    = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE     = "https://api.openf1.org/v1"
XGB_MIN_RACES   = 6          # Carreras mínimas para activar XGBoost

# ═════════════════════════════════════════════════════════════
#  CAPA DE DATOS OPENF1 — Reemplaza FastF1 para datos 2026
#  Gratis, sin auth, datos históricos disponibles inmediatamente
#  Docs: https://openf1.org/docs
# ═════════════════════════════════════════════════════════════

# Mapeo de número de piloto → código 2026
DRIVER_NUMBER_MAP = {
    1: "NOR",  63: "RUS",  44: "HAM",  16: "LEC",  12: "ANT",
    6: "LAW",  87: "BEA",  10: "GAS",  30: "LIN",  55: "SAI",
    3: "RIC",  81: "PIA",  14: "ALO",  18: "STR",  27: "HUL",
    5: "HAD",  43: "COL",  77: "BOT",  31: "OCO",  23: "ALB",
    41: "BOR",  11: "PER",
}

# Cache de session_keys de OpenF1 por ronda y tipo
_OF1_SESSION_CACHE: dict = {}


SESSION_NAME_MAP = {
    "Race"              : "R",
    "Qualifying"        : "Q",
    "Sprint"            : "S",
    "Sprint Qualifying" : "SQ",
    "Practice 1"        : "FP1",
    "Practice 2"        : "FP2",
    "Practice 3"        : "FP3",
}

# Cache de sessions para evitar llamadas repetidas
_OF1_SESSIONS_CACHE: dict = {}


def of1_get_sessions(year: int = SEASON) -> pd.DataFrame:
    """
    Obtiene todos los session_keys de OpenF1 para la temporada.
    Asigna round_num basado en el orden de los Race meetings por fecha.
    """
    global _OF1_SESSIONS_CACHE
    if year in _OF1_SESSIONS_CACHE:
        return _OF1_SESSIONS_CACHE[year]

    data = api_get(f"{OPENF1_BASE}/sessions", params={"year": year})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["session_code"] = df["session_name"].map(SESSION_NAME_MAP)
    df["date_start"]   = pd.to_datetime(df["date_start"], errors="coerce")

    # Asignar round_num: ordenar Race sessions por fecha y numerar desde 1
    race_sessions = df[df["session_code"] == "R"].sort_values("date_start").reset_index(drop=True)
    meeting_round_map = {
        int(row["meeting_key"]): i + 1
        for i, row in race_sessions.iterrows()
    }
    df["round_num"] = df["meeting_key"].map(meeting_round_map)

    _OF1_SESSIONS_CACHE[year] = df
    return df


def of1_session_key(sessions_df: pd.DataFrame,
                    round_num: int, session_code: str) -> int | None:
    """
    Obtiene el session_key de OpenF1 para una ronda y tipo de sesión.
    Usa round_num asignado desde el orden de Race meetings.
    """
    if sessions_df.empty:
        return None
    hits = sessions_df[
        (sessions_df["round_num"] == round_num) &
        (sessions_df["session_code"] == session_code)
    ]
    if hits.empty:
        return None
    return int(hits.iloc[0]["session_key"])


def of1_get_rounds_with_data(sessions_df: pd.DataFrame,
                              session_code: str = "R") -> list:
    """
    Retorna lista de round_nums que tienen datos disponibles
    en OpenF1 para el tipo de sesión dado.
    """
    if sessions_df.empty:
        return []
    hits = sessions_df[sessions_df["session_code"] == session_code].dropna(subset=["round_num"])
    return sorted(hits["round_num"].astype(int).tolist())


def of1_get_laps(session_key: int,
                  enrich_with_stints: bool = True) -> pd.DataFrame:
    """
    Descarga todos los tiempos de vuelta de una sesión desde OpenF1.
    Si enrich_with_stints=True, joinea con /stints para agregar
    compound y stint_number (que no vienen en /laps directamente).
    """
    data = api_get(f"{OPENF1_BASE}/laps",
                   params={"session_key": session_key})
    time.sleep(REQ_DELAY)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["code"] = df["driver_number"].map(DRIVER_NUMBER_MAP).fillna(
        df["driver_number"].astype(str))

    # Enriquecer con datos de stints (compound + stint_number)
    if enrich_with_stints and "lap_number" in df.columns:
        stints = of1_get_stints(session_key)
        if not stints.empty and "lap_start" in stints.columns:
            # Expandir stints a filas individuales por vuelta
            stint_rows = []
            for _, s in stints.iterrows():
                lap_s = int(s.get("lap_start", 0) or 0)
                lap_e = int(s.get("lap_end",   0) or 0)
                for lap in range(lap_s, lap_e + 1):
                    stint_rows.append({
                        "driver_number": s["driver_number"],
                        "lap_number"   : lap,
                        "compound"     : s.get("compound", None),
                        "stint_number" : s.get("stint_number", None),
                        "tyre_age"     : s.get("tyre_age_at_start", 0) + (lap - lap_s),
                    })
            if stint_rows:
                stint_laps = pd.DataFrame(stint_rows)
                df = df.merge(stint_laps,
                              on=["driver_number","lap_number"],
                              how="left")
    return df


def of1_get_stints(session_key: int) -> pd.DataFrame:
    """
    Descarga datos de stints (compuesto, vueltas en neumático) desde OpenF1.
    """
    data = api_get(f"{OPENF1_BASE}/stints",
                   params={"session_key": session_key})
    time.sleep(REQ_DELAY)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["code"] = df["driver_number"].map(DRIVER_NUMBER_MAP).fillna(
        df["driver_number"].astype(str))
    return df


def of1_get_position(session_key: int) -> pd.DataFrame:
    """
    Descarga datos de posición vuelta por vuelta desde OpenF1.
    """
    data = api_get(f"{OPENF1_BASE}/position",
                   params={"session_key": session_key})
    time.sleep(REQ_DELAY)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["code"] = df["driver_number"].map(DRIVER_NUMBER_MAP).fillna(
        df["driver_number"].astype(str))
    return df


def of1_get_pit(session_key: int) -> pd.DataFrame:
    """Descarga datos de pit stops desde OpenF1."""
    data = api_get(f"{OPENF1_BASE}/pit",
                   params={"session_key": session_key})
    time.sleep(REQ_DELAY)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["code"] = df["driver_number"].map(DRIVER_NUMBER_MAP).fillna(
        df["driver_number"].astype(str))
    return df


def of1_get_race_control(session_key: int) -> pd.DataFrame:
    """Descarga mensajes de race control (SC, VSC, banderas) desde OpenF1."""
    data = api_get(f"{OPENF1_BASE}/race_control",
                   params={"session_key": session_key})
    time.sleep(REQ_DELAY)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def of1_get_intervals(session_key: int) -> pd.DataFrame:
    """Descarga gaps/intervalos entre pilotos desde OpenF1."""
    data = api_get(f"{OPENF1_BASE}/intervals",
                   params={"session_key": session_key})
    time.sleep(REQ_DELAY)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["code"] = df["driver_number"].map(DRIVER_NUMBER_MAP).fillna(
        df["driver_number"].astype(str))
    return df


# ─────────────────────────────────────────────────────────────
#  FUNCIONES DE FEATURES USANDO OPENF1
# ─────────────────────────────────────────────────────────────

def of1_collect_lap_consistency(completed: list,
                                 sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Consistencia de vueltas via OpenF1 /laps."""
    print("📈  Consistencia de vueltas (OpenF1)...")
    records = {}
    for rnd in completed:
        sk = of1_session_key(sessions_df, rnd, "R")
        if not sk:
            continue
        laps = of1_get_laps(sk)
        if laps.empty or "lap_duration" not in laps.columns:
            continue
        if "lap_duration" not in laps.columns:
            continue
        valid = laps.dropna(subset=["lap_duration"])
        valid = valid[valid["lap_duration"] > 0]
        for code, grp in valid.groupby("code"):
            times = grp["lap_duration"].tolist()
            if len(times) >= 5:
                records.setdefault(code, []).extend(times)
    rows = [{"code": c, "lap_std": np.std(t)} for c, t in records.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code","lap_std"])


def of1_collect_sector_times(completed: list,
                               sessions_df: pd.DataFrame,
                               next_circuit: str) -> pd.DataFrame:
    """
    Tiempos por sector via OpenF1 /laps.
    Incluye perfil de sectores para el próximo circuito.
    """
    print("🔢  Tiempos por sector (OpenF1)...")
    key_sector = CIRCUIT_SECTOR_PROFILE.get(next_circuit, "S1")
    sector_col = {"S1": "duration_sector_1",
                  "S2": "duration_sector_2",
                  "S3": "duration_sector_3"}[key_sector]
    records = {}
    circuit_records = {}
    for rnd in completed:
        sk = of1_session_key(sessions_df, rnd, "Q")
        if not sk:
            continue
        laps = of1_get_laps(sk)
        if laps.empty:
            continue
        for s_col, s_name in [("duration_sector_1","S1"),
                               ("duration_sector_2","S2"),
                               ("duration_sector_3","S3")]:
            if s_col not in laps.columns:
                continue
            valid = laps[laps[s_col].notna() & (laps[s_col] > 0)]
            best_per_driver = valid.groupby("code")[s_col].min()
            best_ref = best_per_driver.min()
            for code, t in best_per_driver.items():
                records.setdefault(code, {}).setdefault(s_name, []).append(t - best_ref)
        # Circuit sector profile
        if sector_col in laps.columns:
            valid = laps.dropna(subset=[sector_col])
            valid = valid[valid[sector_col] > 0]
            best_drv = valid.groupby("code")[sector_col].min()
            best_all = best_drv.min()
            for code, t in best_drv.items():
                circuit_records.setdefault(code, []).append(t - best_all)

    # avg_sector_delta = promedio de todos los sectores
    rows = []
    for code, sect_data in records.items():
        all_deltas = [d for s in sect_data.values() for d in s]
        rows.append({"code": code, "avg_sector_delta": np.mean(all_deltas)})
    sector_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code","avg_sector_delta"])

    # circuit_score (sector específico del próximo circuito)
    if circuit_records:
        cs_rows = [{"code": c, "circuit_score_raw": np.mean(d)}
                   for c, d in circuit_records.items()]
        cs_df = pd.DataFrame(cs_rows)
        mn, mx = cs_df["circuit_score_raw"].min(), cs_df["circuit_score_raw"].max()
        cs_df["circuit_score"] = (1 - (cs_df["circuit_score_raw"] - mn) / (mx - mn + 1e-9))
        sector_df = sector_df.merge(cs_df[["code","circuit_score"]], on="code", how="left")

    return sector_df


def of1_collect_tyre_degradation(completed: list,
                                  sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Degradación de neumáticos via OpenF1 /stints + /laps."""
    print("🔴  Degradación de neumáticos (OpenF1)...")
    records = {}
    compound_records = {}  # {code: {compound: [deltas]}}

    for rnd in completed:
        sk = of1_session_key(sessions_df, rnd, "R")
        if not sk:
            continue
        laps_df   = of1_get_laps(sk)
        stints_df = of1_get_stints(sk)
        if laps_df.empty:
            continue

        avail_cols = [c for c in ["lap_duration","stint_number"] if c in laps_df.columns]
        valid_laps = laps_df.dropna(subset=avail_cols)
        valid_laps = valid_laps[valid_laps["lap_duration"] > 0]
        best_overall = valid_laps["lap_duration"].min()

        # Degradación por stint (si stint_number disponible)
        if "stint_number" not in valid_laps.columns:
            valid_laps["stint_number"] = 1   # tratar toda la carrera como un stint
        for (code, stint), grp in valid_laps.groupby(["code","stint_number"]):
            times = grp.sort_values("lap_number")["lap_duration"].tolist()
            if len(times) >= 5:
                slope = np.polyfit(np.arange(len(times)), times, 1)[0]
                records.setdefault(code, []).append(slope)

        # Compound performance
        compound_col = next((c for c in ["compound","tyre_compound"] if c in laps_df.columns), None)
        if compound_col:
            for compound in ["SOFT","MEDIUM","HARD"]:
                comp_laps = valid_laps[valid_laps[compound_col] == compound]
                for code, grp in comp_laps.groupby("code"):
                    best_drv = grp["lap_duration"].min()
                    delta    = best_drv - best_overall
                    compound_records.setdefault(code, {}).setdefault(
                        compound, []).append(delta)

    rows = [{"code": c, "tyre_deg_slope": np.mean(s)} for c, s in records.items()]
    deg_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code","tyre_deg_slope"])

    # Compound score
    comp_rows = []
    for code, comp_data in compound_records.items():
        soft_d   = np.mean(comp_data.get("SOFT",   [np.nan]))
        medium_d = np.mean(comp_data.get("MEDIUM", [np.nan]))
        hard_d   = np.mean(comp_data.get("HARD",   [np.nan]))
        available = [d for d in [soft_d, medium_d, hard_d] if not np.isnan(d)]
        avg_delta = np.mean(available) if available else np.nan
        comp_rows.append({
            "code": code, "compound_score": avg_delta,
            "soft_delta": soft_d, "medium_delta": medium_d, "hard_delta": hard_d,
            "compound_versatility": np.std(available) if len(available)>=2 else 0.5,
        })
    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        mn, mx = comp_df["compound_score"].min(), comp_df["compound_score"].max()
        if mx > mn:
            comp_df["compound_score"] = 1 - (comp_df["compound_score"] - mn)/(mx - mn)
        else:
            comp_df["compound_score"] = 1.0
        deg_df = deg_df.merge(comp_df, on="code", how="outer")

    return deg_df


def of1_collect_lap1_gain(completed: list,
                           sessions_df: pd.DataFrame,
                           race_df: pd.DataFrame) -> pd.DataFrame:
    """Ganancia en vuelta 1 via OpenF1 /position."""
    print("🚀  Rendimiento vuelta 1 (OpenF1)...")
    records = {}
    for rnd in completed:
        sk = of1_session_key(sessions_df, rnd, "R")
        if not sk:
            continue
        pos_df = of1_get_position(sk)
        if pos_df.empty or "position" not in pos_df.columns:
            continue
        # Grid de salida desde race_df (Jolpica)
        rnd_race = race_df[race_df["round"] == rnd]
        if rnd_race.empty:
            continue
        grid_map = dict(zip(rnd_race["code"], rnd_race["grid"]))
        # Posición al final de la vuelta 1 (primer registro por piloto)
        pos_df["date"] = pd.to_datetime(pos_df["date"], errors="coerce")
        if "position" not in pos_df.columns:
            continue
        lap1_pos = pos_df.sort_values("date").groupby("code").first()["position"]
        for code, pos1 in lap1_pos.items():
            grid = grid_map.get(code, np.nan)
            if not np.isnan(safe_float(pos1)) and not np.isnan(safe_float(grid)) and safe_float(grid) > 0:
                records.setdefault(code, []).append(safe_float(grid) - safe_float(pos1))
    rows = [{"code": c, "lap1_gain": np.mean(g)} for c, g in records.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code","lap1_gain"])


def of1_collect_practice_pace(completed: list,
                               sessions_df: pd.DataFrame) -> pd.DataFrame:
    """FP pace normalizado via OpenF1 /laps."""
    print("🏋   Ritmo de prácticas libres normalizado (OpenF1)...")
    SESSION_WEIGHTS = {"FP1": 0.20, "FP2": 0.50, "FP3": 0.30}
    quick_records   = {}
    longrun_records = {}

    for rnd in completed:
        fp_sessions = ["FP1"] if rnd in SPRINT_ROUNDS else ["FP1","FP2","FP3"]
        for sname in fp_sessions:
            sk = of1_session_key(sessions_df, rnd, sname)
            if not sk:
                continue
            laps = of1_get_laps(sk)
            if laps.empty or "lap_duration" not in laps.columns:
                continue
            valid = laps.dropna(subset=["lap_duration"])
            valid = valid[valid["lap_duration"] > 0]
            # Filtrar pit out laps si la columna existe
            if "is_pit_out_lap" in valid.columns:
                valid = valid[valid["is_pit_out_lap"] != True]
            if valid.empty:
                continue
            best_per_driver = valid.groupby("code")["lap_duration"].min()
            session_median  = best_per_driver.median()
            weight = SESSION_WEIGHTS.get(sname, 0.33)
            for code, t in best_per_driver.items():
                delta = (t - session_median) / session_median * 100
                quick_records.setdefault(code, []).append((delta, weight))

            # FP2 long runs (requiere stint_number)
            if sname == "FP2" and "stint_number" in valid.columns:
                stint_len = valid.groupby(["code","stint_number"]).size()
                long = stint_len[stint_len >= 8].index
                lr_ref = valid["lap_duration"].median()
                for (code, stint) in long:
                    stint_laps = valid[(valid["code"]==code) &
                                       (valid["stint_number"]==stint)]["lap_duration"]
                    mid = stint_laps.iloc[len(stint_laps)//4: len(stint_laps)*3//4]
                    if len(mid) >= 3:
                        longrun_records.setdefault(code, []).append(mid.mean())
            elif sname == "FP2":
                # Sin stint_number: usar todas las vueltas del medio de la sesión
                lr_ref = valid["lap_duration"].median()
                for code, grp in valid.groupby("code"):
                    sorted_laps = grp.sort_values("lap_number")["lap_duration"] if "lap_number" in grp.columns else grp["lap_duration"]
                    mid = sorted_laps.iloc[len(sorted_laps)//4: len(sorted_laps)*3//4]
                    if len(mid) >= 5:
                        longrun_records.setdefault(code, []).append(mid.mean())

    rows = []
    all_codes = set(quick_records) | set(longrun_records)
    lr_ref = np.median([t for ts in longrun_records.values() for t in ts]) if longrun_records else None
    for code in all_codes:
        fp_delta = np.nan
        if code in quick_records:
            deltas  = [d for d,_ in quick_records[code]]
            weights = [w for _,w in quick_records[code]]
            fp_delta = np.average(deltas, weights=weights)
        fp2_lr = np.nan
        if code in longrun_records and lr_ref:
            fp2_lr = (np.mean(longrun_records[code]) - lr_ref) / lr_ref * 100
        rows.append({"code": code, "fp_avg_delta": fp_delta,
                     "fp2_longrun_delta": fp2_lr})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code","fp_avg_delta","fp2_longrun_delta"])


def of1_collect_safety_car(completed: list,
                            sessions_df: pd.DataFrame,
                            race_df: pd.DataFrame) -> pd.DataFrame:
    """Safety Car performance via OpenF1 /race_control + /position."""
    print("🟡  Rendimiento bajo Safety Car (OpenF1)...")
    records = {}
    for rnd in completed:
        sk = of1_session_key(sessions_df, rnd, "R")
        if not sk:
            continue
        rc_df  = of1_get_race_control(sk)
        pos_df = of1_get_position(sk)
        if rc_df.empty or pos_df.empty:
            continue
        # Detectar períodos de SC/VSC
        flag_col = next((c for c in ["flag","message","category"] if c in rc_df.columns), None)
        if not flag_col:
            continue
        sc_msgs = rc_df[rc_df[flag_col].astype(str).str.contains("SAFETY CAR|VIRTUAL|SC", na=False)]
        if sc_msgs.empty:
            continue
        pos_df["date"] = pd.to_datetime(pos_df["date"], errors="coerce")
        pos_df = pos_df.sort_values("date")
        for _, sc in sc_msgs.iterrows():
            sc_time = pd.to_datetime(sc.get("date",""), errors="coerce")
            if pd.isna(sc_time):
                continue
            # Posición 30s antes y 60s después del SC
            before = pos_df[pos_df["date"] <= sc_time].groupby("code")["position"].last()
            after  = pos_df[pos_df["date"] >= sc_time + pd.Timedelta("60s")].groupby("code")["position"].first()
            for code in before.index:
                if code in after.index:
                    gain = safe_float(before[code]) - safe_float(after[code])
                    if not np.isnan(gain):
                        records.setdefault(code, []).append(gain)
    rows = [{"code": c, "sc_gain_avg": np.mean(g)} for c, g in records.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code","sc_gain_avg"])


def of1_collect_quali_gap_teammate(completed: list,
                                    sessions_df: pd.DataFrame,
                                    driver_standings: pd.DataFrame) -> pd.DataFrame:
    """Gap a compañero en clasificación via OpenF1 /laps."""
    print("🔑  Gap qualifying vs compañero (OpenF1)...")
    if driver_standings.empty:
        return pd.DataFrame(columns=["code","quali_gap_teammate"])
    team_map = dict(zip(driver_standings["code"], driver_standings["TeamName"]))
    records  = {}
    for rnd in completed:
        sk = of1_session_key(sessions_df, rnd, "Q")
        if not sk:
            continue
        laps = of1_get_laps(sk)
        if laps.empty or "lap_duration" not in laps.columns:
            continue
        valid = laps.dropna(subset=["lap_duration"])
        valid = valid[valid["lap_duration"] > 0]
        best  = valid.groupby("code")["lap_duration"].min().to_dict()
        # Agrupar por equipo
        team_drivers = {}
        for code, team in team_map.items():
            if code in best:
                team_drivers.setdefault(team, []).append((code, best[code]))
        for team, drivers in team_drivers.items():
            if len(drivers) < 2:
                continue
            for i, (ca, ta) in enumerate(drivers):
                for cb, tb in drivers[i+1:]:
                    records.setdefault(ca, []).append(tb - ta)
                    records.setdefault(cb, []).append(ta - tb)
    rows = [{"code": c, "quali_gap_teammate": np.mean(g)} for c, g in records.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code","quali_gap_teammate"])


def of1_collect_pitstop_performance(completed: list,
                                     sessions_df: pd.DataFrame,
                                     driver_standings: pd.DataFrame) -> pd.DataFrame:
    """Tiempo promedio de pit stop por equipo via OpenF1 /pit."""
    print("🔧  Pit stops (OpenF1)...")
    if driver_standings.empty:
        return pd.DataFrame(columns=["TeamName","avg_pitstop"])
    team_map  = dict(zip(driver_standings["code"], driver_standings["TeamName"]))
    team_stops = {}
    for rnd in completed:
        sk = of1_session_key(sessions_df, rnd, "R")
        if not sk:
            continue
        pit_df = of1_get_pit(sk)
        pit_dur_col = next((c for c in ["pit_duration","duration"] if c in pit_df.columns), None)
        if pit_df.empty or not pit_dur_col:
            continue
        pit_df = pit_df.rename(columns={pit_dur_col: "pit_duration"})
        valid = pit_df.dropna(subset=["pit_duration"])
        valid = valid[(valid["pit_duration"] > 1.5) & (valid["pit_duration"] < 60)]
        for _, row in valid.iterrows():
            code = row.get("code","")
            team = team_map.get(code,"")
            if team:
                team_stops.setdefault(team, []).append(row["pit_duration"])
    rows = [{"TeamName": t, "avg_pitstop": np.mean(s)} for t, s in team_stops.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["TeamName","avg_pitstop"])


def of1_check_available(sessions_df: pd.DataFrame,
                         completed: list) -> bool:
    """
    Verifica si OpenF1 tiene datos de laps para las carreras completadas.
    Retorna True si hay datos reales disponibles.
    """
    if sessions_df.empty or not completed:
        return False
    sk = of1_session_key(sessions_df, completed[-1], "R")
    if not sk:
        return False
    laps = of1_get_laps(sk)
    return not laps.empty and "lap_duration" in laps.columns and len(laps) > 50


SPRINT_ROUNDS   = {2, 6, 7, 10, 13, 18}   # Rondas con sprint en 2026
REQ_TIMEOUT     = 15
REQ_DELAY       = 0.4        # Segundos entre llamadas a la API

fastf1.Cache.enable_cache(CACHE_DIR)

# ─────────────────────────────────────────────────────────────
#  UTILIDADES GENERALES
# ─────────────────────────────────────────────────────────────
def api_get(url: str, params: dict = None) -> dict | None:
    """GET con reintentos y manejo de errores."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    print(f"   ⚠  JSON inválido en {url}: {e}")
                    return None
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"   ⏳  Rate limit — esperando {wait}s...")
                time.sleep(wait)
            elif r.status_code == 404:
                return None   # Datos no disponibles aún — silencioso
            else:
                print(f"   ⚠  HTTP {r.status_code} en {url}")
                time.sleep(1)
        except requests.exceptions.ConnectionError:
            print(f"   ❌  Sin conexión — verifica tu internet.")
            time.sleep(2)
        except Exception as e:
            print(f"   ⚠  Error en {url}: {e}")
            time.sleep(1)
    return None


def safe_float(val, default=np.nan):
    try:
        return float(val)
    except Exception:
        return default


def first_valid_col(df: pd.DataFrame, candidates: list, default=None):
    """Retorna el nombre de la primera columna que existe en df."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


# ─────────────────────────────────────────────────────────────
#  1. CALENDARIO Y PRÓXIMA CARRERA
# ─────────────────────────────────────────────────────────────
def fetch_schedule() -> pd.DataFrame:
    print("📅  Descargando calendario 2026...")
    # limit=100 para traer todas las carreras en una sola llamada
    data = api_get(f"{JOLPICA_BASE}/{SEASON}.json", params={"limit": 100})
    if not data:
        return pd.DataFrame()
    mr    = data.get("MRData", {})
    races = mr.get("RaceTable", {}).get("Races", [])
    if not races:
        print("   ⚠  La API no devolvió carreras para 2026. Verifica tu conexión.")
        return pd.DataFrame()
    rows = []
    for r in races:
        # "round" viene como string desde la API — convertir con fallback
        try:
            rnd = int(r.get("round", 0))
        except (ValueError, TypeError):
            continue
        if rnd == 0:
            continue
        rows.append({
            "round"   : rnd,
            "name"    : r.get("raceName", f"Ronda {rnd}"),
            "circuit" : r.get("Circuit", {}).get("circuitName", "Desconocido"),
            "locality": r.get("Circuit", {}).get("Location", {}).get("locality", ""),
            "country" : r.get("Circuit", {}).get("Location", {}).get("country", ""),
            "date"    : r.get("date", ""),
        })
    df = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    print(f"   ✅  {len(df)} carreras encontradas en el calendario.")
    return df


def get_completed_and_next(schedule: pd.DataFrame):
    """Determina qué rondas están completas y cuál es la siguiente."""
    print("🔍  Verificando rondas completadas...")
    completed = []
    for rnd in schedule["round"].tolist():
        data = api_get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/results.json")
        time.sleep(REQ_DELAY)
        races = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races and races[0].get("Results"):
            race_name = races[0].get("raceName", f"Ronda {rnd}")
            print(f"   ✅  Ronda {rnd}: {race_name} — completa")
            completed.append(int(rnd))
        else:
            print(f"   ⏳  Ronda {rnd} — aún no disputada")
    all_rounds = [int(r) for r in schedule["round"].tolist()]
    next_round = next((r for r in all_rounds if r not in completed), None)
    return completed, next_round


# ─────────────────────────────────────────────────────────────
#  2. STANDINGS (puntos de campeonato)
# ─────────────────────────────────────────────────────────────
def fetch_driver_standings() -> pd.DataFrame:
    print("🏆  Descargando standings de pilotos...")
    data = api_get(f"{JOLPICA_BASE}/{SEASON}/driverStandings.json")
    if not data:
        return pd.DataFrame()
    standings = (data.get("MRData", {})
                     .get("StandingsTable", {})
                     .get("StandingsLists", [{}])[0]
                     .get("DriverStandings", []))
    rows = []
    for s in standings:
        drv = s.get("Driver", {})
        cons = s.get("Constructors", [{}])[0]
        rows.append({
            "code"      : drv.get("code", drv.get("driverId", "???").upper()[:3]),
            "FullName"  : f"{drv.get('givenName','')} {drv.get('familyName','')}".strip(),
            "TeamName"  : cons.get("name", "Desconocido"),
            "champ_pts" : safe_float(s.get("points", 0)),
        })
    return pd.DataFrame(rows)


def fetch_constructor_standings() -> dict:
    print("🏗   Descargando standings de constructores...")
    data = api_get(f"{JOLPICA_BASE}/{SEASON}/constructorStandings.json")
    if not data:
        return {}
    standings = (data.get("MRData", {})
                     .get("StandingsTable", {})
                     .get("StandingsLists", [{}])[0]
                     .get("ConstructorStandings", []))
    return {s["Constructor"]["name"]: safe_float(s.get("points", 0))
            for s in standings}


# ─────────────────────────────────────────────────────────────
#  3. RESULTADOS DE CARRERA (API)
# ─────────────────────────────────────────────────────────────
def fetch_api_race_results(completed: list) -> pd.DataFrame:
    print(f"🏁  Descargando resultados de {len(completed)} carrera(s)...")
    rows = []
    for rnd in completed:
        data = api_get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/results.json")
        time.sleep(REQ_DELAY)
        races = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            continue
        race_name = races[0].get("raceName", f"Ronda {rnd}")
        for res in races[0].get("Results", []):
            drv   = res.get("Driver", {})
            code  = drv.get("code", drv.get("driverId", "???").upper()[:3])
            pos   = safe_float(res.get("position", np.nan))
            grid  = safe_float(res.get("grid", np.nan))
            pts   = safe_float(res.get("points", 0))
            fl    = 1 if res.get("FastestLap", {}).get("rank") == "1" else 0
            status = res.get("status", "")
            dnf   = 0 if status in ("Finished", "+1 Lap", "+2 Laps") else 1
            rows.append({"round": rnd, "race_name": race_name, "code": code,
                         "pos": pos, "grid": grid, "pts": pts,
                         "fastest_lap": fl, "dnf": dnf})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  4. CLASIFICACIÓN (API)
# ─────────────────────────────────────────────────────────────
def fetch_api_qualifying(completed: list) -> pd.DataFrame:
    print("⏱   Descargando resultados de clasificación...")
    rows = []
    for rnd in completed:
        data = api_get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/qualifying.json")
        time.sleep(REQ_DELAY)
        races = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            continue
        for res in races[0].get("QualifyingResults", []):
            drv  = res.get("Driver", {})
            code = drv.get("code", drv.get("driverId", "???").upper()[:3])
            pos  = safe_float(res.get("position", np.nan))
            rows.append({"round": rnd, "code": code, "quali_pos": pos})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  5. SPRINT RESULTS (API)
# ─────────────────────────────────────────────────────────────
def fetch_api_sprint_results(completed: list) -> pd.DataFrame:
    print("🏃  Descargando resultados de sprint...")
    rows = []
    sprint_rounds_done = [r for r in completed if r in SPRINT_ROUNDS]
    for rnd in sprint_rounds_done:
        data = api_get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/sprint.json")
        time.sleep(REQ_DELAY)
        races = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            continue
        for res in races[0].get("SprintResults", []):
            drv  = res.get("Driver", {})
            code = drv.get("code", drv.get("driverId", "???").upper()[:3])
            pos  = safe_float(res.get("position", np.nan))
            pts  = safe_float(res.get("points", 0))
            rows.append({"round": rnd, "code": code,
                         "sprint_pos": pos, "sprint_pts": pts})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  6. PIT STOPS (tiempo promedio por equipo — API)
# ─────────────────────────────────────────────────────────────
def fetch_pitstop_performance(completed: list, driver_team: dict) -> pd.DataFrame:
    """Calcula tiempo promedio de pit stop por equipo."""
    print("🔧  Descargando datos de pit stops...")
    team_stops = {}
    for rnd in completed:
        data = api_get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/pitstops.json")
        time.sleep(REQ_DELAY)
        races = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            continue
        for stop in races[0].get("PitStops", []):
            code     = stop.get("driverId", "").upper()[:3]
            duration = safe_float(stop.get("duration", np.nan))
            team     = driver_team.get(code, "")
            if team and not np.isnan(duration) and duration < 60:
                team_stops.setdefault(team, []).append(duration)
    rows = []
    for team, stops in team_stops.items():
        rows.append({"TeamName": team,
                     "avg_pitstop": np.mean(stops),
                     "pitstop_count": len(stops)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["TeamName", "avg_pitstop"])


# ─────────────────────────────────────────────────────────────
#  7. TASA DE ABANDONO (API)
# ─────────────────────────────────────────────────────────────
def calc_dnf_rate(race_df: pd.DataFrame) -> pd.DataFrame:
    if race_df.empty:
        return pd.DataFrame(columns=["code", "dnf_rate"])
    grp = race_df.groupby("code").agg(
        races=("dnf", "count"),
        dnfs=("dnf", "sum")
    ).reset_index()
    grp["dnf_rate"] = grp["dnfs"] / grp["races"]
    return grp[["code", "dnf_rate"]]


# ─────────────────────────────────────────────────────────────
#  8. PENALIZACIONES (OpenF1 race_control)
# ─────────────────────────────────────────────────────────────
def fetch_grid_penalties(next_round: int, schedule: pd.DataFrame) -> dict:
    """
    Busca penalizaciones de parrilla para la próxima carrera.
    Usa la parrilla de salida (grid) vs posición en clasificación para detectar penalizaciones.
    """
    print("🚦  Buscando penalizaciones de parrilla...")
    penalties = {}
    # Compara posición de clasificación vs posición de salida en la carrera más reciente
    # Si grid > quali_pos en más de 3 posiciones, asumimos penalización
    try:
        quali_data = api_get(f"{JOLPICA_BASE}/{SEASON}/{next_round}/qualifying.json")
        time.sleep(REQ_DELAY)
        race_data  = api_get(f"{JOLPICA_BASE}/{SEASON}/{next_round}/results.json")
        time.sleep(REQ_DELAY)
        if not quali_data or not race_data:
            return penalties
        q_races = quali_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        r_races = race_data.get("MRData",  {}).get("RaceTable", {}).get("Races", [])
        if not q_races or not r_races:
            return penalties
        quali_pos = {
            res["Driver"]["code"]: int(res["position"])
            for res in q_races[0].get("QualifyingResults", [])
        }
        for res in r_races[0].get("Results", []):
            code = res["Driver"]["code"]
            grid = safe_float(res.get("grid", 0))
            qpos = quali_pos.get(code, 0)
            if qpos > 0 and grid > 0 and (grid - qpos) >= 3:
                penalties[code] = int(grid - qpos)
    except Exception as e:
        print(f"   ⚠  No se pudieron detectar penalizaciones: {e}")
    return penalties


def fetch_season_penalties(completed: list) -> pd.DataFrame:
    """
    Penalizaciones acumuladas: detecta pilotos que salieron más atrás que su posición
    de clasificación (diferencia >= 3), indicando penalización de parrilla.
    """
    print("📋  Calculando penalizaciones acumuladas...")
    rows = []
    for rnd in completed:
        quali_data = api_get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/qualifying.json")
        time.sleep(REQ_DELAY)
        race_data  = api_get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/results.json")
        time.sleep(REQ_DELAY)
        if not quali_data or not race_data:
            continue
        q_races = quali_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        r_races = race_data.get("MRData",  {}).get("RaceTable", {}).get("Races", [])
        if not q_races or not r_races:
            continue
        quali_pos = {
            res["Driver"]["code"]: int(res["position"])
            for res in q_races[0].get("QualifyingResults", [])
        }
        for res in r_races[0].get("Results", []):
            code = res["Driver"].get("code", "")
            grid = safe_float(res.get("grid", 0))
            qpos = quali_pos.get(code, 0)
            if qpos > 0 and grid > 0 and (grid - qpos) >= 3:
                rows.append({"round": rnd, "driver_number": code})
    if not rows:
        return pd.DataFrame(columns=["driver_number", "penalty_count"])
    df = pd.DataFrame(rows)
    return df.groupby("driver_number").size().reset_index(name="penalty_count")


# ─────────────────────────────────────────────────────────────
#  9. CLIMA (OpenF1)
# ─────────────────────────────────────────────────────────────
# Coordenadas de cada circuito del calendario 2026
CIRCUIT_COORDS = {
    "Albert Park Grand Prix Circuit"        : (-37.8497,  144.9680),
    "Shanghai International Circuit"        : ( 31.3389,  121.2198),
    "Suzuka Circuit"                        : ( 34.8431,  136.5407),
    "Bahrain International Circuit"         : ( 26.0325,   50.5106),
    "Jeddah Corniche Circuit"               : ( 21.6319,   39.1044),
    "Miami International Autodrome"         : ( 25.9581,  -80.2389),
    "Autodromo Enzo e Dino Ferrari"         : ( 44.3439,   11.7167),
    "Circuit de Monaco"                     : ( 43.7347,    7.4205),
    "Circuit de Barcelona-Catalunya"        : ( 41.5700,    2.2611),
    "Circuit Gilles Villeneuve"             : ( 45.5000,  -73.5228),
    "Red Bull Ring"                         : ( 47.2197,   14.7647),
    "Silverstone Circuit"                   : ( 52.0786,   -1.0169),
    "Hungaroring"                           : ( 47.5789,   19.2486),
    "Circuit de Spa-Francorchamps"          : ( 50.4372,    5.9714),
    "Circuit Park Zandvoort"                : ( 52.3888,    4.5409),
    "Autodromo Nazionale di Monza"          : ( 45.6156,    9.2811),
    "Baku City Circuit"                     : ( 40.3725,   49.8533),
    "Marina Bay Street Circuit"             : (  1.2914,  103.8640),
    "Circuit of the Americas"               : ( 30.1328,  -97.6411),
    "Autodromo Hermanos Rodriguez"          : ( 19.4042,  -99.0907),
    "Autodromo Jose Carlos Pace"            : (-23.7036,  -46.6997),
    "Las Vegas Strip Circuit"               : ( 36.1147, -115.1728),
    "Losail International Circuit"          : ( 25.4900,   51.4542),
    "Yas Marina Circuit"                    : ( 24.4672,   54.6031),
}


def fetch_weather_for_race(next_round: int, schedule: pd.DataFrame) -> dict:
    """Obtiene clima del circuito usando Open-Meteo (gratuito, sin auth, sin SSL issues)."""
    print("🌤   Consultando datos de clima (Open-Meteo)...")
    row = schedule[schedule["round"] == next_round]
    if row.empty:
        return {}

    circuit  = row.iloc[0].get("circuit", "")
    locality = row.iloc[0].get("locality", "")
    country  = row.iloc[0].get("country", "")

    # Buscar coordenadas — primero por nombre de circuito exacto, luego parcial
    coords = CIRCUIT_COORDS.get(circuit)
    if not coords:
        for key, val in CIRCUIT_COORDS.items():
            if locality.lower() in key.lower() or any(
                    w in key.lower() for w in circuit.lower().split() if len(w) > 4):
                coords = val
                break

    if not coords:
        print(f"   ⚠  Coordenadas no encontradas para {circuit} — agrega al diccionario CIRCUIT_COORDS")
        return {}

    lat, lon = coords
    try:
        # Open-Meteo: pronóstico de 7 días, variables horarias de temperatura y lluvia
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"          : lat,
            "longitude"         : lon,
            "hourly"            : "temperature_2m,precipitation,relative_humidity_2m",
            "forecast_days"     : 7,
            "timezone"          : "auto",
        }
        r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            print(f"   ⚠  Open-Meteo HTTP {r.status_code}")
            return {}
        data    = r.json()
        hourly  = data.get("hourly", {})
        temps   = [t for t in hourly.get("temperature_2m", []) if t is not None]
        precips = [p for p in hourly.get("precipitation", [])  if p is not None]
        humids  = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]

        avg_temp = np.mean(temps[:72])   if temps   else None   # próximos 3 días
        avg_hum  = np.mean(humids[:72])  if humids  else None
        rain_exp = any(p > 0.5 for p in precips[:72]) if precips else False

        print(f"   ✅  Clima obtenido para {circuit} ({locality}, {country})")
        return {
            "avg_track_temp": (avg_temp + 10) if avg_temp is not None else None,
            "avg_humidity"  : avg_hum,
            "rain_expected" : rain_exp,
            "location"      : f"{locality}, {country}",
        }
    except Exception as e:
        print(f"   ⚠  Error obteniendo clima: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
#  10. FASTF1 — CONSISTENCIA DE VUELTAS
# ─────────────────────────────────────────────────────────────
# Cache de sesiones fallidas — evita doble intento de carga
_FAILED_SESSIONS: set = set()


def load_session(year, rnd, session_type, need_laps=False):
    """
    Carga una sesión FastF1 de forma segura.
    Si ya falló antes, no reintenta (elimina mensajes duplicados y tiempo perdido).
    """
    key = (year, rnd, session_type)
    if key in _FAILED_SESSIONS:
        return None
    try:
        sess = fastf1.get_session(year, rnd, session_type)
        sess.load(laps=need_laps, telemetry=False, weather=False, messages=False)
        if need_laps:
            laps = sess.laps
            if len(laps) == 0:
                _FAILED_SESSIONS.add(key)
                return None
        return sess
    except Exception as e:
        err = str(e).lower()
        # Suprimir el error conocido de timing data 2026 — es esperado
        if "timing" not in err and "not been loaded" not in err and "ergast" not in err:
            print(f"   ⚠  FastF1 [{session_type} R{rnd}]: {e}")
        _FAILED_SESSIONS.add(key)
        return None


def safe_laps(sess):
    """
    Retorna el objeto Laps de FastF1 o None.
    NO reintenta carga — load_session ya lo intentó.
    """
    if sess is None:
        return None
    try:
        laps = sess.laps
        if len(laps) == 0:
            return None
        return laps
    except Exception:
        return None


def collect_lap_consistency(completed: list) -> pd.DataFrame:
    """Desviación estándar de tiempos de vuelta — mide consistencia."""
    print("📈  Calculando consistencia de vueltas (FastF1)...")
    records = {}
    for rnd in completed:
        sess = load_session(SEASON, rnd, "R", need_laps=True)
        if not sess:
            continue
        raw = safe_laps(sess)
        if raw is None or raw.empty:
            continue
        laps = raw.pick_quicklaps() if hasattr(raw, "pick_quicklaps") else raw
        if laps.empty:
            continue
        for code, grp in laps.groupby("Driver"):
            times = grp["LapTime"].dt.total_seconds().dropna()
            if len(times) >= 5:
                records.setdefault(code, []).extend(times.tolist())
    rows = []
    for code, times in records.items():
        rows.append({"code": code, "lap_std": np.std(times)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code", "lap_std"])


# ─────────────────────────────────────────────────────────────
#  11. FASTF1 — TIEMPOS POR SECTOR
# ─────────────────────────────────────────────────────────────
def collect_sector_times(completed: list) -> pd.DataFrame:
    """Posición relativa en cada sector durante clasificación."""
    print("🔢  Calculando tiempos por sector (FastF1)...")
    rows = []
    for rnd in completed:
        sess = load_session(SEASON, rnd, "Q", need_laps=True)
        if not sess:
            continue
        raw = safe_laps(sess)
        if raw is None or raw.empty:
            continue
        laps = raw.pick_quicklaps() if hasattr(raw, "pick_quicklaps") else raw
        if laps.empty:
            continue
        for sector_col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            if sector_col not in laps.columns:
                continue
            grp = laps.groupby("Driver")[sector_col].apply(
                lambda x: x.dt.total_seconds().min()
            ).dropna().reset_index()
            grp.columns = ["code", "best"]
            best_all = grp["best"].min()
            grp["delta"] = grp["best"] - best_all
            grp["round"] = rnd
            grp["sector"] = sector_col
            rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["code", "avg_sector_delta"])
    df = pd.concat(rows, ignore_index=True)
    return df.groupby("code")["delta"].mean().reset_index(
        ).rename(columns={"delta": "avg_sector_delta"})


# ─────────────────────────────────────────────────────────────
#  12. FASTF1 — DEGRADACIÓN DE NEUMÁTICOS (stint medio)
# ─────────────────────────────────────────────────────────────
def collect_tyre_degradation(completed: list) -> pd.DataFrame:
    """
    Calcula la degradación de neumáticos midiendo la pendiente de tiempo
    de vuelta por vuelta dentro de cada stint.
    """
    print("🔴  Analizando degradación de neumáticos (FastF1)...")
    records = {}
    for rnd in completed:
        sess = load_session(SEASON, rnd, "R", need_laps=True)
        if not sess:
            continue
        raw = safe_laps(sess)
        if raw is None or raw.empty:
            continue
        laps = raw.pick_quicklaps() if hasattr(raw, "pick_quicklaps") else raw
        if laps.empty or "Stint" not in laps.columns:
            continue
        for (driver, stint), grp in laps.groupby(["Driver", "Stint"]):
            times = grp["LapTime"].dt.total_seconds().dropna()
            lap_nums = np.arange(len(times))
            if len(times) >= 5:
                try:
                    slope = np.polyfit(lap_nums, times, 1)[0]
                    records.setdefault(driver, []).append(slope)
                except Exception:
                    pass
    rows = [{"code": c, "tyre_deg_slope": np.mean(s)}
            for c, s in records.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "tyre_deg_slope"])


# ─────────────────────────────────────────────────────────────
#  13. FASTF1 — POSICIÓN VUELTA 1 VS SALIDA
# ─────────────────────────────────────────────────────────────
def collect_lap1_gain(completed: list) -> pd.DataFrame:
    """Ganancia/pérdida de posiciones en la vuelta 1 vs posición de salida."""
    print("🚀  Calculando rendimiento en vuelta 1 (FastF1)...")
    records = {}
    for rnd in completed:
        sess = load_session(SEASON, rnd, "R", need_laps=True)
        if not sess:
            continue
        laps = safe_laps(sess)
        if laps is None or laps.empty:
            continue
        lap1 = laps[laps["LapNumber"] == 1][["Driver", "Position"]].dropna()
        results = sess.results
        if results is None or results.empty:
            continue
        grid_col = first_valid_col(results, ["GridPosition", "grid"])
        code_col  = first_valid_col(results, ["Abbreviation", "Driver"])
        if not grid_col or not code_col:
            continue
        grid_map = dict(zip(results[code_col], results[grid_col]))
        for _, row in lap1.iterrows():
            drv   = row["Driver"]
            pos1  = safe_float(row["Position"])
            grid  = safe_float(grid_map.get(drv, np.nan))
            if not np.isnan(pos1) and not np.isnan(grid) and grid > 0:
                records.setdefault(drv, []).append(grid - pos1)
    rows = [{"code": c, "lap1_gain": np.mean(g)} for c, g in records.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "lap1_gain"])


# ─────────────────────────────────────────────────────────────
#  14. FASTF1 — PRÁCTICAS LIBRES (ritmo relativo)
# ─────────────────────────────────────────────────────────────
def collect_practice_pace(completed: list) -> pd.DataFrame:
    """
    FP pace delta normalizado por sesión — Mejora 6.

    Cambios vs versión anterior:
      1. Delta normalizado vs MEDIANA de sesión (no vs el mejor absoluto)
         → elimina el efecto de pistas que mejoran con el goma
      2. Peso diferente por sesión: FP2 > FP3 > FP1
         → FP2 tiene long runs representativos del ritmo de carrera
      3. FP2 long run pace por separado: stints de 8+ vueltas en FP2
         → el predictor más valioso de ritmo real de carrera
      4. Ajuste por evolución de pista: normaliza FP1 vs FP2 delta
    """
    print("🏋   Descargando ritmo de prácticas libres normalizado (FastF1)...")

    # Pesos por sesión (FP2 es la más valiosa por long runs)
    SESSION_WEIGHTS = {"FP1": 0.20, "FP2": 0.50, "FP3": 0.30}

    quick_records  = {}   # delta en vueltas rápidas (normalizado vs mediana)
    longrun_records = {}  # long run pace FP2 (stints 8+ vueltas)

    for rnd in completed:
        sessions = ["FP1"] if rnd in SPRINT_ROUNDS else ["FP1", "FP2", "FP3"]

        fp2_median = None  # referencia cruzada FP1→FP2 evolution

        for sname in sessions:
            sess = load_session(SEASON, rnd, sname, need_laps=True)
            if not sess:
                continue
            try:
                raw = safe_laps(sess)
                if raw is None or raw.empty:
                    continue
                laps = raw.pick_quicklaps() if hasattr(raw, "pick_quicklaps") else raw
                if laps.empty:
                    continue

                best_per_driver = laps.groupby("Driver")["LapTime"].min(
                    ).dt.total_seconds()

                # Normalizar vs MEDIANA de sesión (no vs el mejor)
                session_median = best_per_driver.median()
                if sname == "FP2":
                    fp2_median = session_median

                weight = SESSION_WEIGHTS.get(sname, 0.33)
                for drv, t in best_per_driver.items():
                    delta = (t - session_median) / session_median * 100  # % vs mediana
                    quick_records.setdefault(drv, []).append((delta, weight))

                # ── FP2 Long Run Analysis ──────────────────────────────
                if sname == "FP2":
                    all_laps = raw  # usar TODAS las vueltas, no solo quick
                    if all_laps is None or all_laps.empty:
                        continue
                    if "Stint" not in all_laps.columns:
                        continue
                    all_laps = all_laps.copy()
                    all_laps["LapTime_s"] = all_laps["LapTime"].dt.total_seconds()
                    all_laps = all_laps.dropna(subset=["LapTime_s"])
                    all_laps = all_laps[all_laps["LapTime_s"] > 0]

                    # Filtrar stints largos (8+ vueltas = representativo de carrera)
                    stint_lengths = all_laps.groupby(
                        ["Driver", "Stint"])["LapTime_s"].count()
                    long_stints   = stint_lengths[stint_lengths >= 8].index

                    for (drv, stint) in long_stints:
                        stint_laps = all_laps[
                            (all_laps["Driver"] == drv) &
                            (all_laps["Stint"] == stint)
                        ]["LapTime_s"]
                        # Usar el tiempo del medio del stint (elimina in/out lap efecto)
                        mid_start = len(stint_laps) // 4
                        mid_end   = len(stint_laps) * 3 // 4
                        mid_laps  = stint_laps.iloc[mid_start:mid_end]
                        if len(mid_laps) >= 3:
                            avg_pace  = mid_laps.mean()
                            longrun_records.setdefault(drv, []).append(avg_pace)

            except Exception:
                pass

    # ── Construir DataFrame ────────────────────────────────────────────
    rows = []
    all_codes = set(quick_records.keys()) | set(longrun_records.keys())

    # Referencia global de long run para normalizar
    all_longruns = [t for times in longrun_records.values() for t in times]
    lr_ref       = np.median(all_longruns) if all_longruns else None

    for code in all_codes:
        # Delta ponderado de sesiones quick
        if code in quick_records:
            deltas  = [d for d, _ in quick_records[code]]
            weights = [w for _, w in quick_records[code]]
            fp_delta = np.average(deltas, weights=weights)
        else:
            fp_delta = np.nan

        # Long run FP2 normalizado vs mediana global
        if code in longrun_records and lr_ref:
            lr_avg = np.mean(longrun_records[code])
            fp2_longrun_delta = (lr_avg - lr_ref) / lr_ref * 100
        else:
            fp2_longrun_delta = np.nan

        rows.append({
            "code"             : code,
            "fp_avg_delta"     : fp_delta,          # % vs mediana sesión (ponderado)
            "fp2_longrun_delta": fp2_longrun_delta,  # % vs mediana long runs FP2
        })

    if not rows:
        return pd.DataFrame(columns=["code", "fp_avg_delta", "fp2_longrun_delta"])
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  15. FASTF1 — SPRINT QUALIFYING
# ─────────────────────────────────────────────────────────────
def collect_sprint_qualifying(completed: list) -> pd.DataFrame:
    print("⚡  Descargando sprint qualifying (FastF1)...")
    rows = []
    sprint_rounds_done = [r for r in completed if r in SPRINT_ROUNDS]
    for rnd in sprint_rounds_done:
        sess = load_session(SEASON, rnd, "SQ", need_laps=False)
        if not sess:
            continue
        results = sess.results
        if results is None or results.empty:
            continue
        code_col = first_valid_col(results, ["Abbreviation", "Driver"])
        pos_col  = first_valid_col(results, ["Position"])
        if not code_col or not pos_col:
            continue
        for _, row in results.iterrows():
            rows.append({"round": rnd,
                         "code": str(row[code_col])[:3].upper(),
                         "sq_pos": safe_float(row[pos_col])})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "sq_pos"])



# ─────────────────────────────────────────────────────────────
#  16b. GAP A COMPAÑERO EN CLASIFICACIÓN (décimas)
# ─────────────────────────────────────────────────────────────
def collect_quali_gap_teammate(completed: list,
                               driver_standings: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el gap en segundos entre cada piloto y su compañero de equipo
    en cada clasificación completada.

    Ventajas vs delta de carrera:
      - Disponible CADA fin de semana (incluso sprints)
      - Elimina el ruido de estrategia, safety car, tráfico
      - Mide directamente velocidad pura del piloto vs mismo auto

    Positivo = más rápido que el compañero (mejor)
    """
    if driver_standings.empty:
        return pd.DataFrame(columns=["code", "quali_gap_teammate"])

    print("🔑  Calculando gap de clasificación vs compañero de equipo...")
    team_map = dict(zip(driver_standings["code"], driver_standings["TeamName"]))
    records  = {}

    for rnd in completed:
        sess = load_session(SEASON, rnd, "Q", need_laps=True)
        if not sess:
            continue
        raw = safe_laps(sess)
        if raw is None or raw.empty:
            continue
        laps = raw.pick_quicklaps() if hasattr(raw, "pick_quicklaps") else raw
        if laps.empty:
            continue

        # Mejor vuelta por piloto en segundos
        best = laps.groupby("Driver")["LapTime"].min().dt.total_seconds().to_dict()

        # Agrupar por equipo y calcular gap entre compañeros
        team_drivers = {}
        for code, team in team_map.items():
            # Buscar el código en los laps (puede estar en formato distinto)
            matching = [k for k in best.keys()
                        if k[:3].upper() == code[:3].upper() or k == code]
            if matching:
                team_drivers.setdefault(team, []).append((code, best[matching[0]]))

        for team, drivers in team_drivers.items():
            if len(drivers) < 2:
                continue
            for i, (code_a, time_a) in enumerate(drivers):
                for code_b, time_b in drivers[i+1:]:
                    gap_a = time_b - time_a   # positivo = A más rápido
                    gap_b = time_a - time_b   # positivo = B más rápido
                    records.setdefault(code_a, []).append(gap_a)
                    records.setdefault(code_b, []).append(gap_b)

    if not records:
        return pd.DataFrame(columns=["code", "quali_gap_teammate"])

    rows = [{"code": c, "quali_gap_teammate": np.mean(g)}
            for c, g in records.items()]
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
#  16. DELTA VS COMPAÑERO DE EQUIPO
# ─────────────────────────────────────────────────────────────
def calc_teammate_delta(race_df: pd.DataFrame,
                        driver_standings: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula qué tan adelante termina cada piloto vs su compañero de equipo
    en promedio. Valor positivo = mejor que el compañero.
    """
    if race_df.empty or driver_standings.empty:
        return pd.DataFrame(columns=["code", "teammate_delta"])
    team_map = dict(zip(driver_standings["code"], driver_standings["TeamName"]))
    race_df  = race_df.copy()
    race_df["team"] = race_df["code"].map(team_map)
    rows = []
    for rnd, grp in race_df.groupby("round"):
        grp = grp.dropna(subset=["pos", "team"])
        for _, r in grp.iterrows():
            teammates = grp[(grp["team"] == r["team"]) & (grp["code"] != r["code"])]
            if not teammates.empty:
                tm_pos   = teammates["pos"].mean()
                my_pos   = r["pos"]
                rows.append({"round": rnd, "code": r["code"],
                              "delta": tm_pos - my_pos})
    if not rows:
        return pd.DataFrame(columns=["code", "teammate_delta"])
    df = pd.DataFrame(rows)
    return df.groupby("code")["delta"].mean().reset_index(
        ).rename(columns={"delta": "teammate_delta"})


# ─────────────────────────────────────────────────────────────
#  17. SAFETY CAR Y POSICIÓN DURANTE SC
# ─────────────────────────────────────────────────────────────
def collect_safety_car_performance(completed: list) -> pd.DataFrame:
    """
    Mide si un piloto tiende a aprovechar el SC para hacer pit y ganar posiciones.
    (Ratio: posiciones ganadas después de SC / número de SC en las que estuvo)
    """
    print("🟡  Analizando rendimiento bajo Safety Car (FastF1)...")
    records = {}
    for rnd in completed:
        sess = load_session(SEASON, rnd, "R", need_laps=True)
        if not sess:
            continue
        try:
            laps = safe_laps(sess)
            if laps is None or laps.empty or "TrackStatus" not in laps.columns:
                continue
            sc_laps = laps[laps["TrackStatus"].astype(str).str.contains("4|6",
                           na=False)]["LapNumber"].unique()
            if len(sc_laps) == 0:
                continue
            for driver, grp in laps.groupby("Driver"):
                grp = grp.sort_values("LapNumber")
                gains = []
                for sc_lap in sc_laps:
                    before = grp[grp["LapNumber"] == sc_lap - 1]["Position"]
                    after  = grp[grp["LapNumber"] == sc_lap + 2]["Position"]
                    if not before.empty and not after.empty:
                        gains.append(before.iloc[0] - after.iloc[0])
                if gains:
                    records.setdefault(driver, []).extend(gains)
        except Exception:
            pass
    rows = [{"code": c, "sc_gain_avg": np.mean(g)} for c, g in records.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "sc_gain_avg"])



# ─────────────────────────────────────────────────────────────
#  18b. PERFIL DE SECTORES POR TIPO DE CIRCUITO
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  CLASIFICACIÓN DE TIPO DE CIRCUITO
# ─────────────────────────────────────────────────────────────
CIRCUIT_TYPE = {
    # high_speed: rectas largas, curvas rápidas, poco overtaking técnico
    "Albert Park Grand Prix Circuit"   : "high_speed",
    "Shanghai International Circuit"   : "technical",
    "Suzuka Circuit"                   : "high_speed",
    "Bahrain International Circuit"    : "technical",
    "Jeddah Corniche Circuit"          : "street",
    "Miami International Autodrome"    : "mixed",
    "Autodromo Enzo e Dino Ferrari"    : "technical",
    "Circuit de Monaco"                : "street",
    "Circuit de Barcelona-Catalunya"   : "mixed",
    "Circuit Gilles Villeneuve"        : "street",
    "Red Bull Ring"                    : "high_speed",
    "Silverstone Circuit"              : "high_speed",
    "Hungaroring"                      : "technical",
    "Circuit de Spa-Francorchamps"     : "high_speed",
    "Circuit Park Zandvoort"           : "technical",
    "Autodromo Nazionale di Monza"     : "high_speed",
    "Baku City Circuit"                : "street",
    "Marina Bay Street Circuit"        : "street",
    "Circuit of the Americas"          : "mixed",
    "Autodromo Hermanos Rodriguez"     : "mixed",
    "Autodromo Jose Carlos Pace"       : "mixed",
    "Las Vegas Strip Circuit"          : "street",
    "Losail International Circuit"     : "high_speed",
    "Yas Marina Circuit"               : "mixed",
}

# Índice de dificultad de adelantamiento por circuito (0=fácil, 1=imposible)
# Afecta directamente el peso de qualifying en el modelo
OVERTAKING_DIFFICULTY = {
    "Albert Park Grand Prix Circuit"   : 0.50,
    "Shanghai International Circuit"   : 0.40,
    "Suzuka Circuit"                   : 0.55,
    "Bahrain International Circuit"    : 0.35,
    "Jeddah Corniche Circuit"          : 0.60,
    "Miami International Autodrome"    : 0.50,
    "Autodromo Enzo e Dino Ferrari"    : 0.45,
    "Circuit de Monaco"                : 0.95,  # casi imposible adelantar
    "Circuit de Barcelona-Catalunya"   : 0.65,
    "Circuit Gilles Villeneuve"        : 0.45,
    "Red Bull Ring"                    : 0.40,
    "Silverstone Circuit"              : 0.45,
    "Hungaroring"                      : 0.80,  # muy difícil
    "Circuit de Spa-Francorchamps"     : 0.35,
    "Circuit Park Zandvoort"           : 0.70,
    "Autodromo Nazionale di Monza"     : 0.30,  # fácil adelantar
    "Baku City Circuit"                : 0.50,
    "Marina Bay Street Circuit"        : 0.75,
    "Circuit of the Americas"          : 0.45,
    "Autodromo Hermanos Rodriguez"     : 0.50,
    "Autodromo Jose Carlos Pace"       : 0.55,
    "Las Vegas Strip Circuit"          : 0.40,
    "Losail International Circuit"     : 0.45,
    "Yas Marina Circuit"               : 0.40,
}


def collect_circuit_type_profile(completed: list,
                                 schedule: pd.DataFrame,
                                 race_df: pd.DataFrame,
                                 driver_standings: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada piloto, calcula su rendimiento promedio por tipo de circuito
    (high_speed, street, technical, mixed) usando solo datos de 2026.

    Retorna circuit_type_score: qué tan bien le va al piloto en circuitos
    del mismo tipo que el próximo GP. Normalizado 0-1.
    """
    if race_df.empty or schedule.empty:
        return pd.DataFrame(columns=["code", "circuit_type_score"])

    print("🗂   Calculando perfil por tipo de circuito...")

    # Mapear cada ronda completada a su tipo de circuito
    rnd_type = {}
    for _, row in schedule.iterrows():
        circuit  = row.get("circuit", "")
        rnd_type[int(row["round"])] = CIRCUIT_TYPE.get(circuit, "mixed")

    # Calcular posición promedio por tipo de circuito para cada piloto
    team_map = dict(zip(driver_standings["code"], driver_standings["TeamName"]))
    rows = []
    for _, r in race_df.iterrows():
        rnd  = int(r["round"])
        ctype = rnd_type.get(rnd, "mixed")
        rows.append({"code": r["code"], "pos": r["pos"], "circuit_type": ctype})

    if not rows:
        return pd.DataFrame(columns=["code", "circuit_type_score"])

    df = pd.DataFrame(rows).dropna(subset=["pos"])

    # Determinar tipo del próximo circuito
    # (se pasa externamente como parámetro next_circuit_type)
    # Por ahora devolvemos el perfil completo; el filtro se hace en build_features
    profile = df.groupby(["code", "circuit_type"])["pos"].mean().reset_index()
    return profile


def get_circuit_type_score(profile_df: pd.DataFrame,
                           next_circuit_type: str,
                           driver_codes: list) -> pd.DataFrame:
    """
    Extrae el score del tipo de circuito correspondiente al próximo GP.
    Si el piloto no tiene datos de ese tipo, usa su promedio general.
    """
    if profile_df.empty:
        return pd.DataFrame(columns=["code", "circuit_type_score"])

    rows = []
    for code in driver_codes:
        drv_data = profile_df[profile_df["code"] == code]
        type_data = drv_data[drv_data["circuit_type"] == next_circuit_type]
        if not type_data.empty:
            avg_pos = type_data["pos"].mean()
        elif not drv_data.empty:
            avg_pos = drv_data["pos"].mean()  # fallback: promedio general
        else:
            avg_pos = np.nan
        rows.append({"code": code, "circuit_type_score": avg_pos})

    df = pd.DataFrame(rows)
    # Normalizar: menor posición = mejor = mayor score
    mn, mx = df["circuit_type_score"].min(), df["circuit_type_score"].max()
    if mx > mn and not np.isnan(mn):
        df["circuit_type_score"] = 1 - (df["circuit_type_score"] - mn) / (mx - mn)
    else:
        df["circuit_type_score"] = df["circuit_type_score"].apply(
            lambda x: 0.5 if np.isnan(x) else 0.5)
    return df

# Clasificación de circuitos por qué sector importa más
CIRCUIT_SECTOR_PROFILE = {
    # (circuito, sector_key) — qué sector domina el tiempo de vuelta
    "Albert Park Grand Prix Circuit"   : "S1",   # alta velocidad S1
    "Shanghai International Circuit"   : "S3",   # chicane final S3
    "Suzuka Circuit"                   : "S1",   # esses de Suzuka, S1 dominante
    "Bahrain International Circuit"    : "S2",   # sector medio técnico
    "Jeddah Corniche Circuit"          : "S1",   # alta velocidad
    "Miami International Autodrome"    : "S2",
    "Autodromo Enzo e Dino Ferrari"    : "S2",
    "Circuit de Monaco"                : "S3",   # Loews y chicane, S3 crítico
    "Circuit de Barcelona-Catalunya"   : "S1",
    "Circuit Gilles Villeneuve"        : "S2",
    "Red Bull Ring"                    : "S1",
    "Silverstone Circuit"              : "S1",   # Copse, Maggots, alta vel
    "Hungaroring"                      : "S2",   # muy técnico
    "Circuit de Spa-Francorchamps"     : "S1",   # Eau Rouge, Raidillon
    "Circuit Park Zandvoort"           : "S1",
    "Autodromo Nazionale di Monza"     : "S1",   # rectas largas
    "Baku City Circuit"                : "S3",   # zona del castillo
    "Marina Bay Street Circuit"        : "S3",
    "Circuit of the Americas"          : "S1",   # vuelta 1 y S1 curvas
    "Autodromo Hermanos Rodriguez"     : "S2",
    "Autodromo Jose Carlos Pace"       : "S2",
    "Las Vegas Strip Circuit"          : "S1",
    "Losail International Circuit"     : "S1",
    "Yas Marina Circuit"               : "S2",
}

SECTOR_COL_MAP = {"S1": "Sector1Time", "S2": "Sector2Time", "S3": "Sector3Time"}


def collect_circuit_sector_score(completed: list,
                                 next_circuit: str) -> pd.DataFrame:
    """
    Para cada piloto calcula qué tan fuerte es en el sector que domina
    el circuito de la próxima carrera, usando datos de carreras similares en 2026.

    Lógica:
      1. Identificar el sector clave del próximo circuito
      2. Para cada carrera completada, extraer el delta de cada piloto en ese sector
      3. Pilotos con menor delta (= más rápidos) en ese sector reciben mayor score
    """
    print(f"🗺   Calculando perfil de sectores para: {next_circuit}...")
    key_sector = CIRCUIT_SECTOR_PROFILE.get(next_circuit, "S1")
    sector_col = SECTOR_COL_MAP[key_sector]
    print(f"   Sector clave: {key_sector} ({sector_col})")

    records = {}
    for rnd in completed:
        # Usar datos de clasificación — el sector más puro sin tráfico
        sess = load_session(SEASON, rnd, "Q", need_laps=True)
        if not sess:
            continue
        raw = safe_laps(sess)
        if raw is None or raw.empty:
            continue
        laps = raw.pick_quicklaps() if hasattr(raw, "pick_quicklaps") else raw
        if sector_col not in laps.columns:
            continue
        grp = laps.groupby("Driver")[sector_col].apply(
            lambda x: x.dt.total_seconds().min()).dropna()
        if grp.empty:
            continue
        best_ref = grp.min()
        for drv, t in grp.items():
            # delta negativo o 0 = igual al mejor, positivo = más lento
            records.setdefault(drv, []).append(t - best_ref)

    if not records:
        return pd.DataFrame(columns=["code", "circuit_score"])

    rows = []
    for code, deltas in records.items():
        # Menor delta promedio = más fuerte en ese sector = mayor score
        avg_delta = np.mean(deltas)
        rows.append({"code": code, "circuit_score": avg_delta,
                     "key_sector": key_sector})
    df = pd.DataFrame(rows)
    # Invertir: menor delta → mayor score (normalizar 0-1)
    mn, mx = df["circuit_score"].min(), df["circuit_score"].max()
    if mx > mn:
        df["circuit_score"] = 1 - (df["circuit_score"] - mn) / (mx - mn)
    else:
        df["circuit_score"] = 1.0
    return df[["code", "circuit_score"]]


# ─────────────────────────────────────────────────────────────
#  18c. ESTRATEGIA DE COMPUESTOS (Tire Compound Strategy)
# ─────────────────────────────────────────────────────────────
def collect_compound_strategy(completed: list) -> pd.DataFrame:
    """
    Analiza cómo se comporta cada piloto con cada compuesto de neumático:
      - Ritmo en SOFT  vs la mejor vuelta de la sesión
      - Ritmo en MEDIUM vs la mejor vuelta
      - Ritmo en HARD  vs la mejor vuelta
      - compound_score: índice compuesto de versatilidad y rendimiento

    Un piloto que es rápido en todos los compuestos tiene mayor compound_score.
    Un piloto dependiente de un solo compuesto tiene menor score en condiciones
    de estrategia variada.
    """
    print("🟠  Analizando estrategia de compuestos (FastF1)...")
    compound_records = {}  # {code: {compound: [deltas]}}

    for rnd in completed:
        sess = load_session(SEASON, rnd, "R", need_laps=True)
        if not sess:
            continue
        raw = safe_laps(sess)
        if raw is None or raw.empty:
            continue
        laps = raw.pick_quicklaps() if hasattr(raw, "pick_quicklaps") else raw
        if "Compound" not in laps.columns:
            continue

        for compound in ["SOFT", "MEDIUM", "HARD"]:
            comp_laps = laps[laps["Compound"] == compound]
            if comp_laps.empty:
                continue
            best_overall = laps["LapTime"].dt.total_seconds().min()
            for drv, grp in comp_laps.groupby("Driver"):
                best_drv = grp["LapTime"].dt.total_seconds().min()
                delta    = best_drv - best_overall
                compound_records.setdefault(drv, {}).setdefault(
                    compound, []).append(delta)

    if not compound_records:
        return pd.DataFrame(columns=["code", "compound_score",
                                     "soft_delta", "medium_delta", "hard_delta"])

    rows = []
    for code, comp_data in compound_records.items():
        soft_d   = np.mean(comp_data.get("SOFT",   [np.nan]))
        medium_d = np.mean(comp_data.get("MEDIUM", [np.nan]))
        hard_d   = np.mean(comp_data.get("HARD",   [np.nan]))

        # Versatilidad: menor desviación entre compuestos = más versátil
        available = [d for d in [soft_d, medium_d, hard_d] if not np.isnan(d)]
        versatility = np.std(available) if len(available) >= 2 else 0.5
        avg_delta   = np.mean(available) if available else np.nan

        rows.append({
            "code"          : code,
            "soft_delta"    : soft_d,
            "medium_delta"  : medium_d,
            "hard_delta"    : hard_d,
            "compound_score": avg_delta,      # se normaliza en build_features
            "compound_versatility": versatility,
        })

    df = pd.DataFrame(rows)
    # Normalizar compound_score (menor delta = mejor = mayor score)
    mn, mx = df["compound_score"].min(), df["compound_score"].max()
    if mx > mn:
        df["compound_score"] = 1 - (df["compound_score"] - mn) / (mx - mn)
    else:
        df["compound_score"] = 1.0
    return df[["code", "compound_score", "soft_delta", "medium_delta",
               "hard_delta", "compound_versatility"]]



def of1_collect_next_race_qualifying(next_round: int,
                                      sessions_df: pd.DataFrame,
                                      driver_standings: pd.DataFrame) -> pd.DataFrame:
    """
    Descarga los datos de clasificación del PRÓXIMO GP si ya están disponibles.
    Esto es el dato más valioso: en 2026, pole → victoria en alta tasa.

    Retorna DataFrame con:
      - quali_pos_next    : posición en clasificación del próximo GP
      - quali_time_next   : tiempo de vuelta en Q3 (segundos)
      - s1_next, s2_next, s3_next : tiempos de sector en clasificación
    """
    sk = of1_session_key(sessions_df, next_round, "Q")
    if not sk:
        return pd.DataFrame(columns=["code","quali_pos_next","quali_time_next"])

    print(f"🏆  Descargando clasificación del próximo GP (sk={sk})...")
    laps = of1_get_laps(sk, enrich_with_stints=False)
    if laps.empty or "lap_duration" not in laps.columns:
        print("   ⚠  Sin datos de clasificación para el próximo GP")
        return pd.DataFrame(columns=["code","quali_pos_next","quali_time_next"])

    # Mejor vuelta por piloto
    valid = laps.dropna(subset=["lap_duration"])
    valid = valid[valid["lap_duration"] > 0]
    best  = valid.groupby("code").agg(
        quali_time_next = ("lap_duration", "min"),
        s1_next         = ("duration_sector_1", "min"),
        s2_next         = ("duration_sector_2", "min"),
        s3_next         = ("duration_sector_3", "min"),
    ).reset_index()

    # Asignar posición por tiempo
    best = best.sort_values("quali_time_next").reset_index(drop=True)
    best["quali_pos_next"] = best.index + 1

    n_drivers = len(best)
    print(f"   ✅  {n_drivers} pilotos en clasificación del próximo GP")
    if not best.empty:
        pole = best.iloc[0]
        print(f"   🏎  POLE: {pole['code']} — {pole['quali_time_next']:.3f}s")

    return best


def of1_collect_next_race_fp(next_round: int,
                               sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Descarga ritmo de FP1/FP2/FP3 del PRÓXIMO GP si ya están disponibles.
    Normalizado vs mediana de sesión. FP2 long run tiene peso mayor.
    """
    SESSION_WEIGHTS = {"FP1": 0.20, "FP2": 0.50, "FP3": 0.30}
    quick_records   = {}
    longrun_records = {}
    sessions_found  = []

    for sname in ["FP1", "FP2", "FP3"]:
        sk = of1_session_key(sessions_df, next_round, sname)
        if not sk:
            continue
        laps = of1_get_laps(sk, enrich_with_stints=(sname == "FP2"))
        if laps.empty or "lap_duration" not in laps.columns:
            continue
        sessions_found.append(sname)
        valid = laps.dropna(subset=["lap_duration"])
        valid = valid[valid["lap_duration"] > 0]
        if "is_pit_out_lap" in valid.columns:
            valid = valid[valid["is_pit_out_lap"] != True]
        if valid.empty:
            continue
        best_per_driver = valid.groupby("code")["lap_duration"].min()
        session_median  = best_per_driver.median()
        weight = SESSION_WEIGHTS.get(sname, 0.33)
        for code, t in best_per_driver.items():
            delta = (t - session_median) / session_median * 100
            quick_records.setdefault(code, []).append((delta, weight))

        # FP2 long runs
        if sname == "FP2" and "stint_number" in valid.columns:
            lr_ref     = valid["lap_duration"].median()
            stint_len  = valid.groupby(["code","stint_number"]).size()
            long_stints = stint_len[stint_len >= 8].index
            for (code, stint) in long_stints:
                sl = valid[(valid["code"]==code) &
                            (valid["stint_number"]==stint)]["lap_duration"]
                mid = sl.iloc[len(sl)//4: len(sl)*3//4]
                if len(mid) >= 3:
                    longrun_records.setdefault(code, []).append(mid.mean())

    if not sessions_found:
        return pd.DataFrame(columns=["code","fp_next_delta","fp2_next_longrun"])

    print(f"   ✅  Prácticas del próximo GP disponibles: {', '.join(sessions_found)}")
    rows = []
    lr_ref = np.median([t for ts in longrun_records.values() for t in ts]) if longrun_records else None
    all_codes = set(quick_records) | set(longrun_records)
    for code in all_codes:
        fp_delta = np.nan
        if code in quick_records:
            ds  = [d for d,_ in quick_records[code]]
            ws  = [w for _,w in quick_records[code]]
            fp_delta = np.average(ds, weights=ws)
        fp2_lr = np.nan
        if code in longrun_records and lr_ref:
            fp2_lr = (np.mean(longrun_records[code]) - lr_ref) / lr_ref * 100
        rows.append({"code": code, "fp_next_delta": fp_delta,
                     "fp2_next_longrun": fp2_lr})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
#  18. CONSTRUCCIÓN DE FEATURES
# ─────────────────────────────────────────────────────────────
def build_features(driver_standings, constructor_pts, race_df, quali_df,
                   sprint_df, sq_df, lap_std_df, fp_df, sector_df,
                   tyre_deg_df, lap1_df, dnf_df, teammate_df,
                   pitstop_df, penalty_df, sc_df,
                   compound_df=None, circuit_df=None,
                   circuit_type_df=None, quali_gap_df=None,
                   overtaking_difficulty=0.55,
                   next_quali_df=None, next_fp_df=None) -> pd.DataFrame:
    compound_df      = compound_df      if compound_df      is not None else pd.DataFrame()
    circuit_df       = circuit_df       if circuit_df       is not None else pd.DataFrame()
    circuit_type_df  = circuit_type_df  if circuit_type_df  is not None else pd.DataFrame()
    quali_gap_df     = quali_gap_df     if quali_gap_df     is not None else pd.DataFrame()
    next_quali_df    = next_quali_df    if next_quali_df    is not None else pd.DataFrame()
    next_fp_df       = next_fp_df       if next_fp_df       is not None else pd.DataFrame()
    """Combina todas las fuentes en un DataFrame de features por piloto."""

    feat = driver_standings[["code", "FullName", "TeamName", "champ_pts"]].copy()
    n    = len(feat)

    # ── Resultados de carrera ──────────────────────────────────────────
    if not race_df.empty:
        agg = race_df.groupby("code").agg(
            avg_finish=("pos",  "mean"),
            avg_grid  =("grid", "mean"),
            fl_rate   =("fastest_lap", "mean"),
        ).reset_index()
        feat = feat.merge(agg, on="code", how="left")
        # ── MEJORA 11: Momentum con decaimiento exponencial ──────────────
        # Pesos: última carrera 50%, penúltima 30%, antepenúltima 20%
        # Más carreras disponibles → mejor señal de forma real
        sorted_rounds = sorted(race_df["round"].unique())
        decay_weights = {0: 0.50, 1: 0.30, 2: 0.20}   # índice desde el final
        momentum_rows = []
        for code in race_df["code"].unique():
            drv_results = race_df[race_df["code"] == code].sort_values("round")
            recent      = drv_results.tail(3).reset_index(drop=True)
            n_races     = len(recent)
            if n_races == 0:
                continue
            # Normalizar pesos según cuántas carreras hay disponibles
            if n_races == 1:
                w_map = {0: 1.0}
            elif n_races == 2:
                w_map = {0: 0.40, 1: 0.60}   # más peso a la más reciente
            else:
                w_map = {0: 0.20, 1: 0.30, 2: 0.50}
            weighted_pts = sum(
                recent.iloc[i]["pts"] * w_map.get(i, 0)
                for i in range(n_races)
            )
            # También calcular posición ponderada (menor = mejor)
            weighted_pos = sum(
                recent.iloc[i]["pos"] * w_map.get(i, 0)
                for i in range(n_races)
                if not np.isnan(recent.iloc[i]["pos"])
            )
            momentum_rows.append({
                "code"           : code,
                "recent_form"    : weighted_pts,    # pts ponderados exponencialmente
                "momentum_pos"   : weighted_pos,    # pos ponderada (menor = mejor)
                "last_race_pts"  : recent.iloc[-1]["pts"] if n_races > 0 else 0,
                "last_race_pos"  : recent.iloc[-1]["pos"] if n_races > 0 else np.nan,
            })
        if momentum_rows:
            mom_df = pd.DataFrame(momentum_rows)
            feat   = feat.merge(mom_df, on="code", how="left")
        else:
            feat["recent_form"]  = np.nan
            feat["momentum_pos"] = np.nan
            feat["last_race_pts"]= np.nan
            feat["last_race_pos"]= np.nan
    else:
        for col in ["avg_finish", "avg_grid", "fl_rate", "recent_form"]:
            feat[col] = np.nan

    # ── Clasificación ─────────────────────────────────────────────────
    if not quali_df.empty:
        qgg = quali_df.groupby("code")["quali_pos"].mean().reset_index()
        feat = feat.merge(qgg, on="code", how="left")
    else:
        feat["quali_pos"] = np.nan

    # ── Sprint race ────────────────────────────────────────────────────
    if not sprint_df.empty:
        sgg = sprint_df.groupby("code").agg(
            sprint_avg_pos=("sprint_pos", "mean"),
            sprint_pts    =("sprint_pts", "sum"),
        ).reset_index()
        feat = feat.merge(sgg, on="code", how="left")
    else:
        feat["sprint_avg_pos"] = np.nan
        feat["sprint_pts"]     = np.nan

    # ── Sprint qualifying ──────────────────────────────────────────────
    if not sq_df.empty:
        sqgg = sq_df.groupby("code")["sq_pos"].mean().reset_index()
        feat = feat.merge(sqgg, on="code", how="left")
    else:
        feat["sq_pos"] = np.nan

    # ── Consistencia de vueltas ────────────────────────────────────────
    feat = feat.merge(lap_std_df, on="code", how="left") \
               if not lap_std_df.empty else feat.assign(lap_std=np.nan)

    # ── Prácticas libres ───────────────────────────────────────────────
    feat = feat.merge(fp_df, on="code", how="left") \
               if not fp_df.empty else feat.assign(fp_avg_delta=np.nan)

    # ── Sectores ───────────────────────────────────────────────────────
    feat = feat.merge(sector_df, on="code", how="left") \
               if not sector_df.empty else feat.assign(avg_sector_delta=np.nan)

    # ── Degradación neumáticos ─────────────────────────────────────────
    feat = feat.merge(tyre_deg_df, on="code", how="left") \
               if not tyre_deg_df.empty else feat.assign(tyre_deg_slope=np.nan)

    # ── Vuelta 1 ───────────────────────────────────────────────────────
    feat = feat.merge(lap1_df, on="code", how="left") \
               if not lap1_df.empty else feat.assign(lap1_gain=np.nan)

    # ── Tasa de abandono ───────────────────────────────────────────────
    feat = feat.merge(dnf_df, on="code", how="left") \
               if not dnf_df.empty else feat.assign(dnf_rate=np.nan)

    # ── Delta vs compañero ────────────────────────────────────────────
    feat = feat.merge(teammate_df, on="code", how="left") \
               if not teammate_df.empty else feat.assign(teammate_delta=np.nan)

    # ── Pit stops ─────────────────────────────────────────────────────
    if not pitstop_df.empty:
        feat = feat.merge(pitstop_df[["TeamName", "avg_pitstop"]],
                          on="TeamName", how="left")
    else:
        feat["avg_pitstop"] = np.nan

    # ── Penalizaciones acumuladas ─────────────────────────────────────
    feat = feat.merge(penalty_df, left_on="code", right_on="driver_number",
                      how="left").drop(columns=["driver_number"], errors="ignore") \
               if not penalty_df.empty else feat.assign(penalty_count=np.nan)
    feat["penalty_count"] = feat.get("penalty_count", pd.Series(dtype=float)).fillna(0)

    # ── Safety Car ────────────────────────────────────────────────────
    feat = feat.merge(sc_df, on="code", how="left") \
               if not sc_df.empty else feat.assign(sc_gain_avg=np.nan)

    # ── Compound strategy ─────────────────────────────────────────────
    feat = feat.merge(compound_df, on="code", how="left") \
               if not compound_df.empty else feat.assign(
                   compound_score=np.nan, compound_versatility=np.nan,
                   soft_delta=np.nan, medium_delta=np.nan, hard_delta=np.nan)

    # ── Circuit sector score ──────────────────────────────────────────
    feat = feat.merge(circuit_df, on="code", how="left") \
               if not circuit_df.empty else feat.assign(circuit_score=np.nan)

    # ── Clasificación del próximo GP (si disponible) ───────────────────
    feat = feat.merge(next_quali_df, on="code", how="left") \
               if not next_quali_df.empty else feat.assign(
                   quali_pos_next=np.nan, quali_time_next=np.nan,
                   s1_next=np.nan, s2_next=np.nan, s3_next=np.nan)

    # ── Prácticas del próximo GP (si disponible) ──────────────────────
    feat = feat.merge(next_fp_df, on="code", how="left") \
               if not next_fp_df.empty else feat.assign(
                   fp_next_delta=np.nan, fp2_next_longrun=np.nan)

    # ── Circuit type score (head-to-head por tipo de circuito) ────────
    feat = feat.merge(circuit_type_df, on="code", how="left") \
               if not circuit_type_df.empty else feat.assign(circuit_type_score=np.nan)

    # ── Gap a compañero en qualifying (décimas) ───────────────────────
    feat = feat.merge(quali_gap_df, on="code", how="left") \
               if not quali_gap_df.empty else feat.assign(quali_gap_teammate=np.nan)

    # ── fp2_longrun_delta (si fp_df lo tiene) ────────────────────────
    if "fp2_longrun_delta" not in feat.columns:
        feat["fp2_longrun_delta"] = np.nan

    # ── Overtaking difficulty — guardada como atributo del DataFrame ──
    feat.attrs["overtaking_difficulty"] = overtaking_difficulty

    # ── Constructor pts ───────────────────────────────────────────────
    feat["constructor_pts"] = feat["TeamName"].map(constructor_pts).fillna(0)

    # ── Relleno de NaN ────────────────────────────────────────────────
    # Posiciones: NaN → peor posición + 1 (no mediana, para no dar ventaja falsa)
    position_cols = ["quali_pos_next", "quali_pos", "avg_grid", "avg_finish",
                     "momentum_pos", "quali_gap_teammate"]
    numeric_cols = feat.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col in position_cols and col in feat.columns:
            worst = feat[col].max()
            fill_val = (worst + 1) if (not pd.isna(worst)) else 22.0
            feat[col] = feat[col].fillna(fill_val)
        else:
            median = feat[col].median()
            feat[col] = feat[col].fillna(median if not np.isnan(median) else 0)

    return feat


# ─────────────────────────────────────────────────────────────
#  19. MODELO DE PESOS MANUALES
# ─────────────────────────────────────────────────────────────
def score_manual(feat: pd.DataFrame,
                 xgb_weights: dict = None) -> pd.Series:
    """
    Modelo de pesos manuales — 2026 specific.
    Qualifying es el predictor dominante (pole→victoria en alta tasa en 2026).
    Si xgb_weights está disponible, blendea pesos aprendidos con manuales.
    """
    s = pd.Series(0.0, index=feat.index)
    n = len(feat)

    def rank_asc(col):
        return (n + 1 - feat[col].rank(ascending=True, method="min")) / n
    def rank_desc(col):
        return feat[col].rank(ascending=False, method="min").rsub(n + 1) / n

    # ── Peso dinámico de qualifying según dificultad de adelantar ──
    ot_diff         = feat.attrs.get("overtaking_difficulty", 0.55)
    quali_w_dynamic = round(0.22 + ot_diff * 0.16, 3)
    delta_w         = quali_w_dynamic - 0.28
    finish_w        = max(0.04, 0.10 - delta_w * 0.5)
    recent_w        = max(0.04, 0.10 - delta_w * 0.5)

    # ¿Tenemos clasificación real del próximo GP?
    # Si sí → es el predictor más valioso con diferencia
    has_next_quali = (
        "quali_pos_next" in feat.columns and
        feat["quali_pos_next"].notna().sum() >= 10  # al menos 10 pilotos con dato
    )
    if has_next_quali:
        print("   ⭐  Clasificación del próximo GP disponible → peso dominante 45%")

    w = {
        # ── DATOS DEL PRÓXIMO GP (los más relevantes) ──────────────────
        "quali_pos_next"   : 0.45 if has_next_quali else 0.00,
        "fp_next_delta"    : 0.08 if has_next_quali else 0.00,
        "fp2_next_longrun" : 0.05 if has_next_quali else 0.00,
        # ── HISTORIAL DE TEMPORADA ──────────────────────────────────────
        "quali_pos"        : 0.05 if has_next_quali else quali_w_dynamic,
        "quali_gap_teammate": 0.04 if has_next_quali else 0.06,
        "avg_grid"         : 0.03 if has_next_quali else 0.06,
        "champ_pts"        : 0.06 if has_next_quali else 0.11,
        "avg_finish"       : 0.04 if has_next_quali else finish_w,
        "constructor_pts"  : 0.05 if has_next_quali else 0.08,
        "recent_form"      : 0.05 if has_next_quali else recent_w,
        "momentum_pos"     : 0.02 if has_next_quali else 0.03,
        "fl_rate"          : 0.02 if has_next_quali else 0.04,
        "lap_std"          : 0.02 if has_next_quali else 0.04,
        "sprint_pts"       : 0.02 if has_next_quali else 0.03,
        "fp_avg_delta"     : 0.01,
        "fp2_longrun_delta": 0.01,
        "tyre_deg_slope"   : 0.02 if has_next_quali else 0.03,
        "compound_score"   : 0.01 if has_next_quali else 0.03,
        "circuit_score"    : 0.02 if has_next_quali else 0.03,
        "circuit_type_score": 0.01 if has_next_quali else 0.03,
        "lap1_gain"        : 0.01 if has_next_quali else 0.02,
        "teammate_delta"   : 0.01,
        "avg_pitstop"      : 0.01 if has_next_quali else 0.02,
        "sc_gain_avg"      : 0.01,
        "avg_sector_delta" : 0.01 if has_next_quali else 0.02,
    }

    # Auto-normalizar para que sumen exactamente 1.0
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    # Feedback loop: blend XGBoost importancias con pesos manuales
    if xgb_weights:
        print("   🔄  Aplicando feedback XGBoost a pesos manuales...")
        total_xgb = sum(xgb_weights.values()) or 1
        for k in list(w.keys()):
            if k in xgb_weights:
                w[k] = 0.6 * (xgb_weights[k] / total_xgb) + 0.4 * w[k]
        total_w = sum(w.values())
        w = {k: v / total_w for k, v in w.items()}

    def apply(col, direction):
        if col in feat.columns and feat[col].notna().any():
            fn = rank_asc if direction == "asc" else rank_desc
            return fn(col) * w.get(col, 0)
        return pd.Series(0.0, index=feat.index)

    s += apply("quali_pos_next",      "asc")   # ★★ clasificación del próximo GP
    s += apply("fp_next_delta",       "asc")   # ★★ ritmo FP del próximo GP
    s += apply("fp2_next_longrun",    "asc")   # ★★ long run FP2 próximo GP
    s += apply("quali_pos",           "asc")
    s += apply("quali_gap_teammate",  "desc")  # mayor gap = más dominante que compañero
    s += apply("avg_grid",            "asc")
    s += apply("champ_pts",           "desc")
    s += apply("avg_finish",          "asc")
    s += apply("constructor_pts",     "desc")
    s += apply("recent_form",         "desc")
    s += apply("momentum_pos",        "asc")   # posición ponderada exp. (menor=mejor)
    s += apply("fl_rate",             "desc")
    s += apply("lap_std",             "asc")
    s += apply("sprint_pts",          "desc")
    s += apply("fp_avg_delta",        "asc")
    s += apply("fp2_longrun_delta",   "asc")   # menor delta = más rápido en carrera
    s += apply("tyre_deg_slope",      "asc")
    s += apply("compound_score",      "desc")
    s += apply("circuit_score",       "desc")
    s += apply("circuit_type_score",  "desc")
    s += apply("lap1_gain",           "desc")
    s += apply("teammate_delta",      "desc")
    s += apply("avg_pitstop",         "asc")
    s += apply("sc_gain_avg",         "desc")
    s += apply("avg_sector_delta",    "asc")

    s -= feat.get("dnf_rate",      pd.Series(0.0, index=feat.index)).fillna(0) * 0.10
    s -= (feat.get("penalty_count", pd.Series(0.0, index=feat.index)).fillna(0)
          / (feat.get("penalty_count", pd.Series(1.0, index=feat.index)).max() + 1)) * 0.05
    return s


# ─────────────────────────────────────────────────────────────
#  20. MODELO XGBOOST
# ─────────────────────────────────────────────────────────────
def score_xgboost(feat: pd.DataFrame, race_df: pd.DataFrame) -> pd.Series | None:
    """Entrena XGBoost con los datos disponibles usando Leave-One-Out CV."""
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("   ⚠  xgboost no instalado — usando modelo de pesos.")
        return None

    if race_df.empty:
        return None

    feature_cols = [c for c in [
        "champ_pts", "avg_finish", "avg_grid", "fl_rate", "lap_std",
        "constructor_pts", "recent_form", "momentum_pos", "sprint_pts",
        "quali_pos_next", "fp_next_delta", "fp2_next_longrun", "fp_avg_delta",
        "avg_sector_delta", "tyre_deg_slope", "lap1_gain", "teammate_delta",
        "avg_pitstop", "sc_gain_avg", "dnf_rate", "penalty_count",
    ] if c in feat.columns]

    # Construir dataset de entrenamiento por carrera
    X_rows, y_rows = [], []
    for rnd in race_df["round"].unique():
        rnd_res = race_df[race_df["round"] == rnd][["code", "pos"]].dropna()
        for _, r in rnd_res.iterrows():
            driver_feat = feat[feat["code"] == r["code"]][feature_cols]
            if not driver_feat.empty:
                X_rows.append(driver_feat.iloc[0].values)
                y_rows.append(r["pos"])

    if len(X_rows) < 10:
        return None

    X = np.array(X_rows)
    y = np.array(y_rows)

    model = XGBRegressor(n_estimators=100, max_depth=3,
                         learning_rate=0.1, subsample=0.8,
                         random_state=42, verbosity=0)
    try:
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        errors = []
        for train_idx, test_idx in loo.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            errors.append(abs(pred[0] - y[test_idx][0]))
        mae = np.mean(errors)
        print(f"   ✅  XGBoost MAE (LOO): {mae:.2f} posiciones")
    except Exception:
        pass

    # Entrenamiento final con todos los datos
    model.fit(X, y)
    pred_pos = model.predict(feat[feature_cols].values)
    score    = pd.Series(1 / (pred_pos + 0.1), index=feat.index)

    # ── FEEDBACK LOOP: exportar importancias para reponderar modelo manual ──
    try:
        importances = dict(zip(feature_cols, model.feature_importances_))
        # Mostrar top-5 features más importantes aprendidas
        top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        print("   📊  Top features aprendidos por XGBoost:")
        for fname, fimp in top5:
            bar = "█" * int(fimp * 100)
            print(f"      {fname:<22} {fimp:.3f}  {bar}")
        return score, importances
    except Exception:
        return score, {}



# ─────────────────────────────────────────────────────────────
#  21b. MODELO DE FIABILIDAD
# ─────────────────────────────────────────────────────────────
# Clasificación de causas de DNF por componente
# Usamos palabras clave del campo "status" de la API de Jolpica
DNF_COMPONENT_KEYWORDS = {
    "power_unit" : ["power unit", "engine", "electrical", "turbo",
                    "hybrid", "mgu", "ers", "battery", "fuel system"],
    "gearbox"    : ["gearbox", "transmission", "driveshaft"],
    "hydraulics" : ["hydraulics", "hydraulic"],
    "suspension" : ["suspension", "wheel", "upright", "steering"],
    "accident"   : ["accident", "collision", "spin", "damage",
                    "retired", "debris"],
    "brakes"     : ["brakes", "brake"],
    "other"      : [],   # fallback
}


def classify_dnf_component(status: str) -> str:
    """Clasifica el status de DNF en una categoría de componente."""
    s = status.lower()
    for component, keywords in DNF_COMPONENT_KEYWORDS.items():
        if component == "other":
            continue
        if any(kw in s for kw in keywords):
            return component
    return "other"


def build_component_retirement_risk(race_df: pd.DataFrame,
                                    driver_standings: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza el historial de abandonos por COMPONENTE para cada piloto y equipo.

    Diferencia vs DNF rate simple:
      - Un piloto con 2 abandonos mecánicos de power unit tiene mayor riesgo
        que uno con 1 abandono por accidente (que puede no repetirse)
      - Problemas mecánicos en regs nuevas tienden a REPETIRSE en el mismo equipo
      - Accidentes son más aleatorios y no predicen bien el futuro

    Retorna para cada piloto:
      - mechanical_risk   : probabilidad bayesiana de DNF mecánico
      - accident_risk     : probabilidad de DNF por accidente
      - finish_prob       : probabilidad ajustada de terminar
      - dominant_failure  : componente que más falla en su equipo
      - team_pu_failures  : fallos de power unit del equipo en 2026
    """
    PRIOR_RACES    = 8
    PRIOR_MECH_DNF = 0.5   # prior mecánico más bajo que accidente
    PRIOR_ACC_DNF  = 0.3

    # Construir mapa de equipo por piloto
    team_map = {}
    if not driver_standings.empty:
        team_map = dict(zip(driver_standings["code"], driver_standings["TeamName"]))

    # Analizar fallos por equipo (los mecánicos se repiten en el mismo constructor)
    team_failures = {}  # {team: {component: count}}
    if not race_df.empty and "status" in race_df.columns:
        dnf_races = race_df[race_df["dnf"] == 1].copy()
        dnf_races["component"] = dnf_races["status"].apply(classify_dnf_component)
        dnf_races["team"]      = dnf_races["code"].map(team_map)
        for _, row in dnf_races.iterrows():
            team = row.get("team", "")
            comp = row.get("component", "other")
            if team:
                team_failures.setdefault(team, {}).setdefault(comp, 0)
                team_failures[team][comp] += 1

    rel_rows = []
    for _, r in driver_standings.iterrows():
        code = r["code"]
        team = r.get("TeamName", team_map.get(code, ""))

        driver_races = race_df[race_df["code"] == code] if not race_df.empty else pd.DataFrame()
        total_races  = len(driver_races)

        # Separar DNFs mecánicos vs accidentes
        mech_dnfs = 0
        acc_dnfs  = 0
        if not driver_races.empty and "status" in driver_races.columns:
            dnfs = driver_races[driver_races["dnf"] == 1]["status"]
            for s in dnfs:
                comp = classify_dnf_component(str(s))
                if comp == "accident":
                    acc_dnfs += 1
                else:
                    mech_dnfs += 1

        # Estimación bayesiana separada para mecánicos y accidentes
        mech_rate = (mech_dnfs + PRIOR_MECH_DNF) / (total_races + PRIOR_RACES)
        acc_rate  = (acc_dnfs  + PRIOR_ACC_DNF)  / (total_races + PRIOR_RACES)

        # Riesgo del equipo — mecánicos se correlacionan entre compañeros
        team_mech = sum(
            v for k, v in team_failures.get(team, {}).items()
            if k not in ("accident", "other")
        )
        team_total_races = sum(
            len(race_df[race_df["code"] == c])
            for c in driver_standings.get("code", [])
            if team_map.get(c) == team
        )
        team_mech_rate = team_mech / max(team_total_races, 1)

        # Blend: 60% riesgo individual + 40% riesgo de equipo
        blended_mech_rate = 0.60 * mech_rate + 0.40 * team_mech_rate

        # Probabilidad total de terminar
        finish_prob = max(0.0, min(1.0,
            1.0 - blended_mech_rate - acc_rate * 0.7))  # accidente descuenta menos

        # Componente dominante de fallo en el equipo
        team_comp_counts = team_failures.get(team, {})
        dominant_failure = (max(team_comp_counts, key=team_comp_counts.get)
                            if team_comp_counts else "ninguno")

        rel_rows.append({
            "code"            : code,
            "real_races"      : total_races,
            "real_dnfs"       : mech_dnfs + acc_dnfs,
            "mech_dnfs"       : mech_dnfs,
            "acc_dnfs"        : acc_dnfs,
            "mechanical_risk" : round(blended_mech_rate, 4),
            "accident_risk"   : round(acc_rate, 4),
            "finish_prob"     : round(finish_prob, 4),
            "reliability"     : round(finish_prob, 4),
            "dominant_failure": dominant_failure,
            "team_mech_rate"  : round(team_mech_rate, 4),
        })

    return pd.DataFrame(rel_rows)


def build_reliability(feat: pd.DataFrame, race_df: pd.DataFrame,
                      driver_standings: pd.DataFrame = None) -> pd.DataFrame:
    """
    Wrapper: usa build_component_retirement_risk si hay datos de status,
    sino usa el modelo bayesiano simple como fallback.
    """
    ds = driver_standings if driver_standings is not None else feat[["code"]].copy()
    if not race_df.empty and "status" in race_df.columns:
        return build_component_retirement_risk(race_df, ds)
    # Fallback bayesiano simple
    PRIOR_RACES, PRIOR_DNFS = 8, 0.8
    rows = []
    for _, r in feat.iterrows():
        code        = r["code"]
        drv         = race_df[race_df["code"] == code] if not race_df.empty else pd.DataFrame()
        real_races  = len(drv)
        real_dnfs   = int(drv["dnf"].sum()) if not drv.empty else 0
        finish_prob = max(0.0, min(1.0,
            1.0 - (real_dnfs + PRIOR_DNFS) / (real_races + PRIOR_RACES)))
        rows.append({"code": code, "real_races": real_races, "real_dnfs": real_dnfs,
                     "mech_dnfs": 0, "acc_dnfs": real_dnfs,
                     "mechanical_risk": 0.06, "accident_risk": 0.04,
                     "finish_prob": round(finish_prob, 4),
                     "reliability": round(finish_prob, 4),
                     "dominant_failure": "desconocido", "team_mech_rate": 0.0})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  21c. SIMULACIÓN MONTE CARLO
# ─────────────────────────────────────────────────────────────
def monte_carlo_simulation(
        feat        : pd.DataFrame,
        base_scores : pd.Series,
        reliability : pd.DataFrame,
        weather     : dict,
        n_sims      : int = 10_000,
) -> pd.DataFrame:
    """
    Corre N simulaciones de la carrera inyectando variaciones aleatorias en:
      1. Ritmo base del piloto          (ruido gaussiano ±15%)
      2. Fiabilidad / DNF               (Bernoulli por piloto)
      3. Safety Car                     (Poisson, ~1.2 SC por carrera en 2026)
      4. Pit stop variance              (ruido uniforme ±2s por parada)
      5. Clima / lluvia                 (si se espera lluvia, mezcla ranking)
      6. Vuelta 1 incidente             (prob 18% de que alguien pierda 2-4 pos)

    Retorna un DataFrame con:
      - win_mc_pct    : % de simulaciones en que el piloto ganó
      - podium_mc_pct : % de veces en top-3
      - finish_mc_pct : % de veces que terminó la carrera
      - avg_mc_pos    : posición promedio en las simulaciones
      - p10_pos / p90_pos : percentiles 10 y 90 de posición (intervalo de confianza)
    """
    print(f"🎲  Corriendo {n_sims:,} simulaciones Monte Carlo...")

    rng      = np.random.default_rng(42)
    n        = len(feat)
    codes    = feat["code"].tolist()
    scores_v = base_scores.values.copy().astype(float)

    # Mapa de fiabilidad por código
    rel_map      = dict(zip(reliability["code"], reliability["finish_prob"]))
    mech_risk_map = dict(zip(reliability["code"],
                             reliability.get("mechanical_risk",
                             pd.Series(0.06, index=reliability.index))))

    # ¿Lluvia esperada?
    rain_expected = weather.get("rain_expected", False)

    # Acumuladores
    wins    = np.zeros(n, dtype=int)
    podiums = np.zeros(n, dtype=int)
    finishes= np.zeros(n, dtype=int)
    pos_all = np.zeros((n, n_sims), dtype=np.float32)

    for sim in range(n_sims):
        # 1. Ruido de ritmo base (±15% gaussiano)
        noise_scale = 0.22 if rain_expected else 0.15   # más caos con lluvia
        sim_scores  = scores_v * (1 + rng.normal(0, noise_scale, n))

        # 2. Safety Car — redistribuye ligeramente el campo
        #    Poisson(1.2): en promedio 1-2 SC por carrera
        n_sc = rng.poisson(1.2)
        if n_sc > 0:
            # SC comprime el pelotón → añade ruido extra al 50% trasero
            bottom_half = np.argsort(sim_scores)[:n//2]
            sim_scores[bottom_half] *= (1 + rng.uniform(0, 0.08, len(bottom_half)))

        # 3. Lluvia — mezcla aleatoria extra si hay lluvia
        if rain_expected and rng.random() < 0.6:
            shuffle_idx = rng.choice(n, size=n//3, replace=False)
            sim_scores[shuffle_idx] = rng.permutation(sim_scores[shuffle_idx])

        # 4. Pit stop variance — penaliza aleatoriamente a algún piloto
        pit_victim = rng.integers(0, n)
        sim_scores[pit_victim] *= (1 - rng.uniform(0.02, 0.08))

        # 5. Vuelta 1 incidente (prob 18%)
        if rng.random() < 0.18:
            victim = rng.integers(0, min(8, n))   # más probable en top-8
            sim_scores[victim] *= (1 - rng.uniform(0.05, 0.15))

        # 6. DNF por fiabilidad — separado mecánico (se repite) vs accidente
        dnf_mask = np.array([
            rng.random() > rel_map.get(c, 0.90)
            for c in codes
        ])
        # Extra: riesgo mecánico adicional calibrado a ~1.5 DNF/carrera en 2026
        # Aplicamos solo si el mech_risk del equipo es realmente elevado (>15%)
        extra_mech = np.array([
            rng.random() < max(0, mech_risk_map.get(c, 0.06) - 0.10) * 0.3
            for c in codes
        ])
        dnf_mask = dnf_mask | extra_mech
        # Los DNF salen al fondo con score muy bajo
        sim_scores[dnf_mask] = -rng.uniform(0, 0.01, dnf_mask.sum())

        # Convertir scores a posiciones (menor pos = mejor)
        order    = np.argsort(-sim_scores)   # descendente → 1er lugar primero
        pos_sim  = np.empty(n, dtype=float)
        pos_sim[order] = np.arange(1, n + 1, dtype=float)
        # DNF → posición >= n+1 (no terminó)
        pos_sim[dnf_mask] = n + rng.integers(1, 5, dnf_mask.sum())

        pos_all[:, sim] = pos_sim

        # Acumular métricas
        non_dnf = [idx for idx in order if not dnf_mask[idx]]
        if non_dnf:
            wins[non_dnf[0]] += 1
        for idx in non_dnf[:3]:
            podiums[idx] += 1
        finishes += (~dnf_mask).astype(int)

    # Calcular estadísticas finales
    n_drivers = len(codes)
    rows = []
    for i, code in enumerate(codes):
        p = pos_all[i]
        # Separar simulaciones donde terminó vs DNF
        finished_pos = p[p <= n_drivers]   # posiciones reales (1 a n)
        dnf_pos      = p[p > n_drivers]    # posiciones de DNF (n+1 en adelante)

        # Percentiles SOLO sobre carreras terminadas
        if len(finished_pos) >= 10:
            p10 = float(np.percentile(finished_pos, 10))
            p90 = float(np.percentile(finished_pos, 90))
            avg = float(np.mean(finished_pos))
        else:
            # Muy pocos terminos — usar todos
            p10 = float(np.percentile(p, 10))
            p90 = float(np.percentile(p, 90))
            avg = float(np.mean(p))

        rows.append({
            "code"          : code,
            "win_mc_pct"    : round(wins[i]    / n_sims * 100, 2),
            "podium_mc_pct" : round(podiums[i] / n_sims * 100, 2),
            "finish_mc_pct" : round(finishes[i]/ n_sims * 100, 2),
            "avg_mc_pos"    : round(avg, 2),
            "p10_pos"       : round(p10, 1),
            "p90_pos"       : round(p90, 1),
        })

    mc_df = pd.DataFrame(rows).sort_values("win_mc_pct", ascending=False
                         ).reset_index(drop=True)
    print(f"   ✅  Simulaciones completadas.")
    return mc_df

# ─────────────────────────────────────────────────────────────
#  21. NORMALIZAR A PROBABILIDAD
# ─────────────────────────────────────────────────────────────
def scores_to_probability(scores: pd.Series) -> pd.Series:
    s = scores - scores.min()
    total = s.sum()
    if total == 0:
        return pd.Series(1 / len(scores), index=scores.index)
    return s / total * 100


# ─────────────────────────────────────────────────────────────
#  22. REPORTE FINAL
# ─────────────────────────────────────────────────────────────
def print_report(scored: pd.DataFrame, next_info: pd.Series,
                 n_completed: int, model_used: str,
                 weather: dict, penalties: dict,
                 mc_df: pd.DataFrame = None,
                 reliability: pd.DataFrame = None):
    """Imprime el reporte formateado en consola — dos tablas limpias."""

    circuit   = next_info.get("circuit",  "Desconocido")
    country   = next_info.get("country",  "")
    race_date = next_info.get("date",     "?")
    race_name = next_info.get("name",     "Próxima carrera")

    # ── Merge Monte Carlo y fiabilidad ────────────────────────
    df = scored.copy()
    if mc_df is not None and not mc_df.empty:
        df = df.merge(mc_df[["code","win_mc_pct","podium_mc_pct",
                              "finish_mc_pct","avg_mc_pos",
                              "p10_pos","p90_pos"]],
                      on="code", how="left")
    if reliability is not None and not reliability.empty:
        df = df.merge(reliability[["code","finish_prob","real_dnfs","real_races"]],
                      on="code", how="left")

    use_mc = mc_df is not None and not mc_df.empty

    # ── Clima ─────────────────────────────────────────────────
    if weather:
        temp = weather.get("avg_track_temp")
        hum  = weather.get("avg_humidity")
        rain = weather.get("rain_expected", False)
        clima_str = (
            f"{'🌧  Lluvia posible' if rain else '☀  Seco esperado'}"
            + (f"  |  Pista ~{temp:.0f}°C" if temp else "")
            + (f"  |  Humedad {hum:.0f}%"  if hum  else "")
        )
    else:
        clima_str = "N/D"

    sep  = "═" * 70
    sep2 = "─" * 70

    # ══════════════════════════════════════════════════════════
    #  ENCABEZADO
    # ══════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print(f"  🏎   F1 {SEASON}  ·  PREDICCIÓN DE CARRERA")
    print(sep)
    print(f"  {'Carrera':<16}: {race_name}")
    print(f"  {'Circuito':<16}: {circuit}  —  {country}")
    print(f"  {'Fecha':<16}: {race_date}")
    print(f"  {'Carreras usadas':<16}: {n_completed}")
    print(f"  {'Modelo':<16}: {model_used}")
    print(f"  {'Clima':<16}: {clima_str}")
    if penalties:
        pen_str = "  |  ".join(f"{k}: -{v} pos" for k, v in penalties.items())
        print(f"  {'⚠ Penalizaciones':<16}: {pen_str}")
    print(sep)

    # ══════════════════════════════════════════════════════════
    #  TABLA 1 — PREDICCIÓN PRINCIPAL (lo más importante)
    # ══════════════════════════════════════════════════════════
    medals = {1:"🥇", 2:"🥈", 3:"🥉"}
    print(f"\n  📊  TABLA 1 — PREDICCIÓN DE VICTORIA\n")

    t1_headers = ["", "Piloto", "Equipo", "Pts"]
    if use_mc:
        t1_headers += ["🎲 Ganar MC", "🏆 Podio MC", "Pos Esp.", "Intervalo"]
    else:
        t1_headers += ["Win %"]

    t1_rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        medal   = medals.get(i, f"{i:>2}.")
        name    = r.get("FullName", r["code"])
        team    = r.get("TeamName", "?")
        # Acortar nombre de equipos largos
        team_short = (team.replace(" F1 Team","").replace(" Racing","")
                          .replace("Aston Martin Aramco","Aston Martin"))
        pts     = int(r.get("champ_pts", 0))

        if use_mc:
            win_mc  = r.get("win_mc_pct",    0)
            pod_mc  = r.get("podium_mc_pct", 0)
            avg_pos = r.get("avg_mc_pos",    0)
            p10     = r.get("p10_pos",       0)
            p90     = r.get("p90_pos",       0)

            # Barra visual de probabilidad (max 20 chars)
            bar_len = min(20, int(win_mc / 2))
            bar     = "█" * bar_len + "░" * (20 - bar_len)

            t1_rows.append([
                medal, name, team_short, pts,
                f"{win_mc:>5.1f}%  {bar}",
                f"{pod_mc:>5.1f}%",
                f"{avg_pos:.1f}",
                f"P{p10:.0f} – P{p90:.0f}",
            ])
        else:
            win_pct = r.get("win_pct", 0)
            bar_len = min(20, int(win_pct / 2))
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            t1_rows.append([medal, name, team_short, pts,
                             f"{win_pct:>5.1f}%  {bar}"])

    print(tabulate(t1_rows, headers=t1_headers,
                   tablefmt="rounded_outline", colalign=("center",)))

    # ══════════════════════════════════════════════════════════
    #  TABLA 2 — DETALLE TÉCNICO (solo top 10)
    # ══════════════════════════════════════════════════════════
    print(f"\n  🔬  TABLA 2 — DETALLE TÉCNICO  (Top 10)\n")

    t2_headers = ["", "Piloto", "Qualy Next", "FP Next%", "Fin Prom",
                  "Forma(3)", "Fiab%", "DNFs/Carr", "Deg Neum", "Δ Compañero"]

    t2_rows = []
    for i, (_, r) in enumerate(df.head(10).iterrows(), 1):
        medal    = medals.get(i, f"{i:>2}.")
        name     = r.get("FullName", r["code"])
        fin_avg   = r.get("avg_finish",       np.nan)
        forma     = r.get("recent_form",      np.nan)
        fiab      = r.get("finish_prob",      np.nan)
        real_d    = r.get("real_dnfs",        0)
        real_r    = r.get("real_races",       0)
        deg       = r.get("tyre_deg_slope",   np.nan)
        tm_delta  = r.get("teammate_delta",   np.nan)
        q_next    = r.get("quali_pos_next",   np.nan)
        fp_next   = r.get("fp_next_delta",    np.nan)

        def fmt(v, dec=1, suffix=""):
            return f"{v:.{dec}f}{suffix}" if not pd.isna(v) else "—"

        dnf_str  = f"{int(real_d)}/{int(real_r)}" if real_r > 0 else "—"
        fiab_str = f"{fiab*100:.1f}%" if not pd.isna(fiab) else "—"
        deg_str  = fmt(deg, 3)
        tm_str   = (f"+{tm_delta:.2f}" if (not pd.isna(tm_delta) and tm_delta > 0)
                    else fmt(tm_delta, 2))
        q_str    = f"P{int(q_next)}" if not pd.isna(q_next) else "—"
        fp_str   = (f"{fp_next:+.2f}%" if not pd.isna(fp_next) else "—")

        t2_rows.append([medal, name,
                         q_str, fp_str, fmt(fin_avg), fmt(forma, 0),
                         fiab_str, dnf_str, deg_str, tm_str])

    print(tabulate(t2_rows, headers=t2_headers,
                   tablefmt="rounded_outline", colalign=("center",)))

    # ══════════════════════════════════════════════════════════
    #  RESUMEN GANADOR
    # ══════════════════════════════════════════════════════════
    winner = df.iloc[0]
    w_name = winner.get("FullName", winner["code"])
    w_team = winner.get("TeamName", "?")
    print(f"\n  {sep2}")
    print(f"  ✅  PROBABLE GANADOR  :  {w_name}  ({w_team})")
    if use_mc:
        wmc  = winner.get("win_mc_pct",    0)
        pmc  = winner.get("podium_mc_pct", 0)
        fmc  = winner.get("finish_mc_pct", 0)
        pavg = winner.get("avg_mc_pos",    0)
        p10w = winner.get("p10_pos",       0)
        p90w = winner.get("p90_pos",       0)
        print(f"  🎲  Monte Carlo       :  {wmc:.1f}% ganar  |  {pmc:.1f}% podio  |  {fmc:.1f}% terminar")
        print(f"  📊  Posición esperada :  {pavg:.1f}  (intervalo P10={p10w:.0f} – P90={p90w:.0f})")
    else:
        print(f"  📊  Prob. victoria    :  {winner['win_pct']:.1f}%")
    print(f"{sep2}")
    print(f"  ⚠   Modelo estadístico + 10,000 simulaciones Monte Carlo.")
    print(f"      Estrategia, clima y mecánica pueden cambiar cualquier predicción.\n")



# ─────────────────────────────────────────────────────────────
#  MEJORA 10 — ACTUALIZACIÓN BAYESIANA ENTRE FINES DE SEMANA
# ─────────────────────────────────────────────────────────────
import json
import os

PRIORS_FILE = os.environ.get("F1_PRIORS_FILE", "./f1_2026_bayesian_priors.json")


def load_priors() -> dict:
    """Carga priors persistidos de ejecuciones anteriores."""
    try:
        with open(PRIORS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_priors(priors: dict):
    """Persiste priors actualizados para la próxima ejecución."""
    try:
        with open(PRIORS_FILE, "w") as f:
            json.dump(priors, f, indent=2)
    except Exception as e:
        print(f"   ⚠  No se pudieron guardar priors: {e}")


def update_bayesian_priors(priors: dict,
                            scored: pd.DataFrame,
                            race_df: pd.DataFrame,
                            last_completed_round: int) -> dict:
    """
    Después de cada carrera, compara la predicción del modelo con el resultado real
    y ajusta los priors de cada piloto para la próxima carrera.

    Lógica de actualización:
      - Si el modelo predijo bien (pos_pred ≈ pos_real): refuerza el score actual
      - Si predijo peor de lo real (piloto sorprendió): sube su prior
      - Si predijo mejor de lo real (piloto decepcionó): baja su prior

    Los priors son multiplicadores que se aplican al score base:
      - prior > 1.0: el modelo históricamente subestima a este piloto
      - prior < 1.0: el modelo históricamente sobreestima a este piloto
      - prior = 1.0: predicciones históricamente precisas (neutral)

    El sistema usa Bayes con learning rate para no over-ajustar con pocas carreras.
    """
    if race_df.empty:
        return priors

    last_race = race_df[race_df["round"] == last_completed_round]
    if last_race.empty:
        return priors

    LEARNING_RATE = 0.15   # qué tan rápido actualiza (0=nunca, 1=solo último resultado)
    MAX_PRIOR     = 1.40   # tope superior del multiplicador
    MIN_PRIOR     = 0.60   # tope inferior del multiplicador

    # Mapa de posición real en la última carrera
    real_pos = dict(zip(last_race["code"], last_race["pos"]))

    # Rank-based predicted positions: sort all drivers by win_pct descending,
    # assign 1-indexed rank as pred_pos. This ensures the predicted P1 is the
    # driver with highest win_pct, P2 the next, etc. — rather than the old
    # linear formula which placed a 10% win-pct driver at ~P20 and made every
    # error massive and negative, inflating all priors to the ceiling.
    n_drivers = len(scored)
    _sorted = scored.sort_values("win_pct", ascending=False).reset_index(drop=True)
    pred_pos_map = {row["code"]: rank + 1 for rank, (_, row) in enumerate(_sorted.iterrows())}

    for _, row in scored.iterrows():
        code     = row["code"]
        pred_pos = pred_pos_map.get(code, n_drivers)

        actual_pos = real_pos.get(code, np.nan)
        if np.isnan(actual_pos):
            continue

        # Error = posición real - posición predicha
        # Positivo = piloto terminó peor de lo esperado (modelo lo sobreestimó)
        # Negativo = piloto terminó mejor de lo esperado (modelo lo subestimó)
        error = actual_pos - pred_pos

        # Ajuste: normalizar error por número de pilotos
        normalized_error = error / n_drivers   # rango aprox -1 a +1

        # Prior actual (default 1.0 si es nuevo)
        current_prior = priors.get(code, {}).get("score_multiplier", 1.0)

        # Actualización bayesiana: si error positivo (sobreestimado), baja el prior
        adjustment = -normalized_error * LEARNING_RATE
        new_prior  = current_prior * (1 + adjustment)
        new_prior  = max(MIN_PRIOR, min(MAX_PRIOR, new_prior))

        # Acumular historial de errores para tracking
        history = priors.get(code, {}).get("error_history", [])
        history.append(round(error, 2))
        history = history[-10:]  # conservar solo los últimos 10

        priors[code] = {
            "score_multiplier" : round(new_prior, 4),
            "last_error"       : round(error, 2),
            "avg_error"        : round(np.mean(history), 2),
            "error_history"    : history,
            "races_tracked"    : priors.get(code, {}).get("races_tracked", 0) + 1,
        }

    return priors


def apply_bayesian_priors(scores: pd.Series,
                           feat: pd.DataFrame,
                           priors: dict) -> pd.Series:
    """
    Aplica los multiplicadores bayesianos al score base.
    Pilotos consistentemente subestimados reciben un boost;
    pilotos consistentemente sobreestimados reciben una penalización.

    El efecto se suaviza según cuántas carreras se han rastreado:
    con pocas carreras el prior tiene poco efecto, con muchas tiene más.
    """
    if not priors:
        return scores

    adjusted = scores.copy()
    for i, row in feat.reset_index(drop=True).iterrows():
        code    = row["code"]
        p_data  = priors.get(code, {})
        if not p_data:
            continue
        multiplier    = p_data.get("score_multiplier", 1.0)
        races_tracked = p_data.get("races_tracked", 0)
        # Suavizar: el prior tiene menos efecto con pocas carreras rastreadas
        # Con 1 carrera: 20% del efecto; con 5+: 100%
        blend_factor  = min(1.0, races_tracked / 5.0)
        blended_mult  = 1.0 + (multiplier - 1.0) * blend_factor
        adjusted.iloc[i] = scores.iloc[i] * blended_mult

    n_changed = sum(1 for c in feat["code"] if c in priors)
    if n_changed:
        print(f"   🧠  Priors bayesianos aplicados a {n_changed} pilotos")

    return adjusted

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    Path(CACHE_DIR).mkdir(exist_ok=True)

    # 1. Calendario y rondas
    schedule = fetch_schedule()
    if schedule.empty:
        print("❌  No se pudo descargar el calendario. Verifica tu conexión.")
        sys.exit(1)

    print(f"\n📅  Calendario {SEASON}: {len(schedule)} carreras")
    completed, next_round = get_completed_and_next(schedule)
    print(f"   Completadas : {completed}")
    print(f"   Próxima     : Ronda {next_round}")

    if next_round is None:
        print("🏁  Temporada finalizada — no hay próxima carrera.")
        sys.exit(0)

    next_info = schedule[schedule["round"] == next_round].iloc[0]

    # 2. Standings
    driver_standings = fetch_driver_standings()
    constructor_pts  = fetch_constructor_standings()
    if driver_standings.empty:
        print("❌  No se pudieron obtener los standings.")
        sys.exit(1)

    # 3. Datos de carrera y clasificación (API)
    race_df   = fetch_api_race_results(completed) if completed else pd.DataFrame()
    quali_df  = fetch_api_qualifying(completed)   if completed else pd.DataFrame()
    sprint_df = fetch_api_sprint_results(completed) if completed else pd.DataFrame()

    # Mapa piloto → equipo para pitstops
    driver_team = dict(zip(driver_standings["code"], driver_standings["TeamName"]))

    # 3b. Cargar session keys de OpenF1
    print("\n🔍  Descargando session keys de OpenF1...")
    of1_sessions = of1_get_sessions(SEASON)
    if not of1_sessions.empty:
        print(f"   ✅  {len(of1_sessions)} sesiones encontradas en OpenF1")
    else:
        print("   ⚠  OpenF1 no devolvió sesiones")

    # 3c. Verificar disponibilidad de datos de vuelta
    print("🔍  Verificando disponibilidad de datos de telemetría...")
    of1_available = of1_check_available(of1_sessions, completed)
    if of1_available:
        print("   ✅  OpenF1 tiene datos de vueltas — usando OpenF1 como fuente primaria")
    else:
        print("   ⚠  OpenF1 sin datos de vuelta todavía, verificando FastF1...")
        ff1_available = False
        if completed:
            test_sess = load_session(SEASON, completed[-1], "R", need_laps=True)
            if test_sess is not None:
                test_laps = safe_laps(test_sess)
                ff1_available = test_laps is not None and len(test_laps) > 0
        if ff1_available:
            print("   ✅  FastF1 disponible como fallback")
        else:
            print("   ⚠  Ni OpenF1 ni FastF1 tienen datos de vuelta todavía")
            print("      El modelo usará solo datos API (resultados, qualy, standings)")

    # 4. Datos de alto impacto
    dnf_df = calc_dnf_rate(race_df)
    if of1_available:
        # ── OPENF1 como fuente primaria (más actualizado) ──────────────
        tyre_deg_df = of1_collect_tyre_degradation(completed, of1_sessions)
        lap1_df     = of1_collect_lap1_gain(completed, of1_sessions, race_df)
        lap_std_df  = of1_collect_lap_consistency(completed, of1_sessions)
        fp_df       = of1_collect_practice_pace(completed, of1_sessions)
        sc_df       = of1_collect_safety_car(completed, of1_sessions, race_df)
    elif completed and (ff1_available if "ff1_available" in dir() else False):
        # ── FASTF1 como fallback ────────────────────────────────────────
        tyre_deg_df = collect_tyre_degradation(completed)
        lap1_df     = collect_lap1_gain(completed)
        lap_std_df  = collect_lap_consistency(completed)
        fp_df       = collect_practice_pace(completed)
        sc_df       = collect_safety_car_performance(completed)
    else:
        tyre_deg_df = pd.DataFrame(columns=["code","tyre_deg_slope"])
        lap1_df     = pd.DataFrame(columns=["code","lap1_gain"])
        lap_std_df  = pd.DataFrame(columns=["code","lap_std"])
        fp_df       = pd.DataFrame(columns=["code","fp_avg_delta","fp2_longrun_delta"])
        sc_df       = pd.DataFrame(columns=["code","sc_gain_avg"])

    # 5. Datos de medio impacto
    if of1_available:
        pitstop_df = of1_collect_pitstop_performance(completed, of1_sessions, driver_standings)
    elif completed:
        pitstop_df = fetch_pitstop_performance(completed, driver_team)
    else:
        pitstop_df = pd.DataFrame(columns=["TeamName","avg_pitstop"])
    teammate_df  = calc_teammate_delta(race_df, driver_standings)
    penalty_df   = fetch_season_penalties(completed) if completed else pd.DataFrame(columns=["driver_number","penalty_count"])

    # 6. Datos de bajo impacto
    sq_df = collect_sprint_qualifying(completed) if (completed and not of1_available) else pd.DataFrame(columns=["code","sq_pos"])

    # 6b. Sector / circuito / compuesto
    next_circuit      = schedule[schedule["round"] == next_round].iloc[0].get("circuit", "")
    next_circuit_type = CIRCUIT_TYPE.get(next_circuit, "mixed")
    ot_difficulty     = OVERTAKING_DIFFICULTY.get(next_circuit, 0.55)

    if of1_available:
        sector_and_circuit = of1_collect_sector_times(completed, of1_sessions, next_circuit)
        sector_df  = sector_and_circuit[["code","avg_sector_delta"]].copy() if "avg_sector_delta" in sector_and_circuit.columns else pd.DataFrame(columns=["code","avg_sector_delta"])
        circuit_df = sector_and_circuit[["code","circuit_score"]].copy() if "circuit_score" in sector_and_circuit.columns else pd.DataFrame(columns=["code","circuit_score"])
        compound_df = tyre_deg_df[["code","compound_score","soft_delta","medium_delta","hard_delta","compound_versatility"]].copy() if "compound_score" in tyre_deg_df.columns else pd.DataFrame(columns=["code","compound_score"])
        # Limpiar tyre_deg_df para que solo tenga las columnas esperadas
        tyre_cols = ["code","tyre_deg_slope"]
        tyre_deg_df = tyre_deg_df[[c for c in tyre_cols if c in tyre_deg_df.columns]]
    else:
        sector_df   = pd.DataFrame(columns=["code","avg_sector_delta"])
        circuit_df  = pd.DataFrame(columns=["code","circuit_score"])
        compound_df = pd.DataFrame(columns=["code","compound_score"])

    # 6c. Circuit type + quali gap
    ct_profile_df  = collect_circuit_type_profile(
                        completed, schedule, race_df, driver_standings) if completed else pd.DataFrame()
    circuit_type_df = get_circuit_type_score(
                        ct_profile_df, next_circuit_type,
                        driver_standings["code"].tolist())
    if of1_available:
        quali_gap_df = of1_collect_quali_gap_teammate(completed, of1_sessions, driver_standings)
    else:
        quali_gap_df = pd.DataFrame(columns=["code","quali_gap_teammate"])

    data_source = "OpenF1" if of1_available else ("FastF1" if (completed and ("ff1_available" in dir() and ff1_available)) else "Solo API")
    print(f"   📊  Fuente de datos de vuelta: {data_source}")

    print(f"   🏁  Circuito: {next_circuit}")
    print(f"   🗂   Tipo: {next_circuit_type}  |  Dificultad adelantar: {ot_difficulty:.2f}")
    print(f"   ⚖   Peso qualifying dinámico: {round(0.22 + ot_difficulty * 0.16, 3):.1%}")

    # 6d. Datos del próximo GP — clasificación y prácticas ya disponibles
    print("\n🎯  Descargando datos pre-carrera del próximo GP (Ronda {next_round})...")
    if of1_available:
        next_quali_df = of1_collect_next_race_qualifying(
                            next_round, of1_sessions, driver_standings)
        next_fp_df    = of1_collect_next_race_fp(next_round, of1_sessions)
        if not next_quali_df.empty:
            print(f"   ✅  Clasificación del próximo GP cargada ({len(next_quali_df)} pilotos)")
        else:
            print("   ⏳  Clasificación del próximo GP aún no disponible")
        if not next_fp_df.empty:
            print(f"   ✅  Prácticas del próximo GP cargadas ({len(next_fp_df)} pilotos)")
        else:
            print("   ⏳  Prácticas del próximo GP aún no disponibles")
    else:
        next_quali_df = pd.DataFrame(columns=["code","quali_pos_next","quali_time_next"])
        next_fp_df    = pd.DataFrame(columns=["code","fp_next_delta","fp2_next_longrun"])

    # 7. Clima y penalizaciones para próxima carrera
    weather  = fetch_weather_for_race(next_round, schedule)
    grid_pen = fetch_grid_penalties(next_round, schedule)

    # 8. Build features
    print("\n🔨  Construyendo matriz de features...")
    feat = build_features(
        driver_standings, constructor_pts, race_df, quali_df,
        sprint_df, sq_df, lap_std_df, fp_df, sector_df,
        tyre_deg_df, lap1_df, dnf_df, teammate_df,
        pitstop_df, penalty_df, sc_df,
        compound_df=compound_df,
        circuit_df=circuit_df,
        circuit_type_df=circuit_type_df,
        quali_gap_df=quali_gap_df,
        overtaking_difficulty=ot_difficulty,
        next_quali_df=next_quali_df,
        next_fp_df=next_fp_df,
    )

    # 9. Cargar priors bayesianos de ejecuciones anteriores
    print("\n🧠  Cargando priors bayesianos...")
    priors = load_priors()
    if priors:
        n_tracked = len([c for c in driver_standings["code"] if c in priors])
        print(f"   📂  {n_tracked} pilotos con historial de predicciones previas")
    else:
        print("   📂  Primera ejecución — sin priors previos")

    # 9b. Scoring
    xgb_weights = {}   # importancias aprendidas por XGBoost (feedback loop)
    use_xgb     = len(completed) >= XGB_MIN_RACES

    if use_xgb:
        print(f"\n🤖  Activando XGBoost ({len(completed)} carreras ≥ {XGB_MIN_RACES})...")
        result = score_xgboost(feat, race_df)
        if result is not None:
            xgb_score, xgb_weights = result
            feat["xgb_pos"] = xgb_score
            scores     = xgb_score
            model_used = "XGBoost + Feedback Loop"
        else:
            scores     = score_manual(feat)
            model_used = "Pesos manuales (XGBoost falló)"
    else:
        remaining = XGB_MIN_RACES - len(completed)
        print(f"\n⚖   Usando modelo de pesos ({len(completed)}/{XGB_MIN_RACES} para XGBoost)...")
        # Aunque no hay XGBoost aún, intentar pre-entrenar para extraer importancias
        result = score_xgboost(feat, race_df)
        if result is not None:
            _, xgb_weights = result
            print(f"   💡  Importancias XGBoost extraídas ({len(xgb_weights)} features)")
        scores     = score_manual(feat, xgb_weights=xgb_weights if xgb_weights else None)
        model_used = f"Pesos manuales{' + Feedback XGB' if xgb_weights else ''} (faltan {remaining} carreras para XGBoost)"

    # 10. Aplicar priors bayesianos al score
    if priors and completed:
        scores = apply_bayesian_priors(scores, feat, priors)

    # 10b. Convertir a probabilidad
    feat["win_pct"] = scores_to_probability(scores).values

    # 11. Ordenar
    scored = feat.sort_values("win_pct", ascending=False).reset_index(drop=True)

    # 12. Modelo de fiabilidad por componente
    print("\n🔩  Calculando fiabilidad y riesgo por componente...")
    reliability = build_reliability(feat, race_df, driver_standings=driver_standings)
    # Mostrar equipos con mayor riesgo mecánico
    if not reliability.empty and "mechanical_risk" in reliability.columns:
        high_risk = reliability[reliability["mechanical_risk"] > 0.15].sort_values(
            "mechanical_risk", ascending=False)
        if not high_risk.empty:
            print("   ⚠  Alto riesgo mecánico:")
            for _, r in high_risk.head(4).iterrows():
                print(f"      {r['code']}: {r['mechanical_risk']*100:.1f}% "
                      f"({r.get('dominant_failure','?')})")

    # 13. Simulación Monte Carlo
    mc_df = monte_carlo_simulation(
        feat        = scored,
        base_scores = scores.reindex(scored.index),
        reliability = reliability,
        weather     = weather,
        n_sims      = 10_000,
    )

    # 14. Actualizar priors bayesianos con última carrera completada
    if completed:
        last_rnd = max(completed)
        print(f"\n🧠  Actualizando priors bayesianos con carrera {last_rnd}...")
        priors = update_bayesian_priors(priors, scored, race_df, last_rnd)
        save_priors(priors)
        print(f"   💾  Priors guardados en: {PRIORS_FILE}")
        # Mostrar pilotos con mayor ajuste
        adjustments = [(c, d["score_multiplier"]) for c, d in priors.items()
                       if d.get("races_tracked", 0) >= 1]
        adjustments.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)
        if adjustments[:3]:
            print("   📊  Mayores ajustes acumulados:")
            for code, mult in adjustments[:3]:
                direction = "⬆ subestimado" if mult > 1.0 else "⬇ sobreestimado"
                print(f"      {code}: ×{mult:.3f} ({direction})")

    # 15. Reporte
    print_report(scored, next_info, len(completed), model_used,
                 weather, grid_pen, mc_df=mc_df, reliability=reliability)

    # 13. Guardar CSV
    # Merge MC into scored for CSV
    if mc_df is not None and not mc_df.empty:
        scored = scored.merge(mc_df, on="code", how="left")
    # Stamp round so the bot can detect stale CSVs before serving predictions
    scored["round_num"] = int(next_round)
    scored["race_name"] = str(next_info.get("name", ""))
    save_cols = [c for c in [
        "round_num", "race_name",
        "code", "FullName", "TeamName", "champ_pts", "avg_finish", "avg_grid",
        "recent_form", "sprint_pts", "fp_avg_delta", "avg_sector_delta",
        "tyre_deg_slope", "lap1_gain", "teammate_delta", "dnf_rate",
        "avg_pitstop", "penalty_count", "sc_gain_avg", "win_pct",
        "win_mc_pct", "podium_mc_pct", "finish_mc_pct", "avg_mc_pos",
        "p10_pos", "p90_pos",
        "compound_score", "circuit_score", "soft_delta", "medium_delta", "hard_delta",
        "circuit_type_score", "quali_gap_teammate", "fp2_longrun_delta",
        "momentum_pos", "last_race_pts", "last_race_pos",
        "mechanical_risk", "accident_risk", "dominant_failure", "team_mech_rate",
        "quali_pos_next", "quali_time_next", "fp_next_delta", "fp2_next_longrun",
    ] if c in scored.columns]
    scored[save_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"💾  Resultados guardados en: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
    

#-------run script using these commands:----------
#-------source .venv/bin/activate-----------------
#-------python f1_2026_prediction3.py-------------