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
import re
from datetime import datetime, timezone, timedelta
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
OUTPUT_CSV               = "f1_2026_predicciones.csv"
XGB_MODEL_FILE           = "./f1_2026_xgb_race_model.json"
LGBM_MODEL_FILE          = "./f1_2026_lgbm_race_model.txt"
_INCREMENTAL_TREES       = 20
_INCREMENTAL_WEIGHT_MULT = 3.0
_META_CTYPES             = ["high_speed", "street", "technical", "mixed"]
_META_MIN_ROUNDS         = 6     # minimum LOO rounds to trust stacking weights
_GP_WEIGHT               = 0.20  # GP gets 20% of final blend; XGB+LGBM share remaining 80%
JOLPICA_BASE    = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE     = "https://api.openf1.org/v1"
XGB_MIN_RACES   = 6          # Carreras mínimas para activar XGBoost
LSTM_MIN_RACES       = 12         # Carreras mínimas para activar LSTM (more sequences needed)
TELEMETRY_MIN_RACES  = 12         # Same threshold — corner telemetry needs a full data set
TYRE_INVENTORY_CACHE     = Path("./f1_2026_tyre_inventory.json")
CORNER_TELEMETRY_CACHE   = Path("./f1_2026_corner_telemetry.json")

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

# meeting_key lookup — mirrors RACE_CALENDAR_2026 in models.py
# Fixes OpenF1's 24-race calendar offset (Bahrain/Saudi not in Jolpica's 22 races)
JOLPICA_ROUND_TO_MEETING_KEY: dict[int, int] = {
     1: 1279,  2: 1280,  3: 1281,  4: 1284,  5: 1285,
     6: 1286,  7: 1287,  8: 1288,  9: 1289, 10: 1290,
    11: 1291, 12: 1292, 13: 1293, 14: 1294, 15: 1296,
    16: 1295, 17: 1297, 18: 1298, 19: 1299, 20: 1300,
    21: 1301, 22: 1302,
}
_MEETING_KEY_TO_JOLPICA_ROUND: dict[int, int] = {
    v: k for k, v in JOLPICA_ROUND_TO_MEETING_KEY.items()
}


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

    # Asignar round_num via meeting_key → Jolpica round (corrige offset Bahrain/Arabia en OpenF1)
    # Sesiones de carreras no en Jolpica (Bahrain, Arabia) quedan como NaN — correcto
    df["round_num"] = df["meeting_key"].map(_MEETING_KEY_TO_JOLPICA_ROUND)

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
    val = hits.iloc[0]["session_key"]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return int(val)


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


SPRINT_ROUNDS   = {2, 4, 5, 9, 12, 16}    # China, Miami, Canada, British, Dutch, Singapore (Jolpica round #)
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
            "time"    : r.get("time", ""),   # UTC race start, e.g. "13:00:00Z"
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
#  Team name normalization — maps legacy/API names to 2026 official names.
#  Applied at every point where Constructor.name comes in from Jolpica.
# ─────────────────────────────────────────────────────────────
_TEAM_NAME_ALIASES: dict[str, str] = {
    "RB F1 Team"           : "Racing Bulls",
    "Visa Cash App RB"     : "Racing Bulls",
    "VCARB"                : "Racing Bulls",
    "AlphaTauri"           : "Racing Bulls",
    "Scuderia AlphaTauri"  : "Racing Bulls",
}

def normalize_team_name(name: str) -> str:
    return _TEAM_NAME_ALIASES.get(name, name)


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
            "TeamName"  : normalize_team_name(cons.get("name", "Desconocido")),
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
    return {normalize_team_name(s["Constructor"]["name"]): safe_float(s.get("points", 0))
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
    """Obtiene clima del circuito usando Open-Meteo (gratuito, sin auth, sin SSL issues).
    Precipitation is narrowed to a 3-hour race window (1h before → 2h after start)
    when the race start time is available; falls back to a 72-hour window otherwise.
    """
    print("🌤   Consultando datos de clima (Open-Meteo)...")
    row = schedule[schedule["round"] == next_round]
    if row.empty:
        return {}

    circuit        = row.iloc[0].get("circuit", "")
    locality       = row.iloc[0].get("locality", "")
    country        = row.iloc[0].get("country", "")
    race_date_str  = row.iloc[0].get("date", "")
    race_time_str  = row.iloc[0].get("time", "")   # "HH:MM:SSZ" UTC or ""

    # Parse race start time in UTC from Jolpica schedule
    race_start_utc = None
    if race_date_str and race_time_str:
        try:
            clean = race_time_str.rstrip("Z")
            d_parts = list(map(int, race_date_str.split("-")))
            t_parts = list(map(int, clean.split(":")))
            race_start_utc = datetime(*d_parts, *t_parts, tzinfo=timezone.utc)
        except Exception:
            race_start_utc = None

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
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"     : lat,
            "longitude"    : lon,
            "hourly"       : "temperature_2m,precipitation,relative_humidity_2m",
            "forecast_days": 7,
            "timezone"     : "auto",
        }
        r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            print(f"   ⚠  Open-Meteo HTTP {r.status_code}")
            return {}
        data    = r.json()
        hourly  = data.get("hourly", {})
        temps   = [t for t in hourly.get("temperature_2m", [])        if t is not None]
        precips = [p for p in hourly.get("precipitation", [])          if p is not None]
        humids  = [h for h in hourly.get("relative_humidity_2m", [])  if h is not None]

        avg_temp = np.mean(temps[:72])  if temps  else None
        avg_hum  = np.mean(humids[:72]) if humids else None

        # ── Race-window precipitation ─────────────────────────────────────
        # Default: 72-hour window (entire weekend)
        window       = precips[:72]
        utc_offset   = data.get("utc_offset_seconds", 0)
        tz_name      = data.get("timezone", "local")
        hourly_times = hourly.get("time", [])   # "YYYY-MM-DDTHH:MM" in local time

        if race_start_utc is not None and hourly_times:
            try:
                race_local     = race_start_utc + timedelta(seconds=utc_offset)
                local_hour_str = race_local.strftime("%Y-%m-%dT%H:00")
                idx = hourly_times.index(local_hour_str)
                # 3-hour window: 1h before start, start, +1h, +2h (covers ~race distance)
                w_start = max(0, idx - 1)
                w_end   = min(len(precips), idx + 3)
                window  = precips[w_start:w_end]

                t_from = (race_local - timedelta(hours=1)).strftime("%H:%M")
                t_to   = (race_local + timedelta(hours=2)).strftime("%H:%M")
                print(f"   🕐  Inicio de carrera: "
                      f"{race_start_utc.strftime('%H:%M')} UTC  /  "
                      f"{race_local.strftime('%H:%M')} local ({tz_name})")
                print(f"   📡  Ventana de pronóstico: {t_from}–{t_to} local "
                      f"(3h: 1h antes + 2h después de la salida)")
            except (ValueError, IndexError):
                print("   ⚠  Hora de carrera fuera del rango del pronóstico — usando ventana 72h")
        else:
            if not race_time_str:
                print("   ⚠  Hora de carrera no disponible (API) — usando ventana 72h")
            elif not hourly_times:
                print("   ⚠  Open-Meteo no devolvió índice de horas — usando ventana 72h")

        rain_prob = (sum(1 for p in window if p > 0.2) / max(len(window), 1)
                     ) if window else 0.0

        # ── Nowcast override ──────────────────────────────────────────────────
        # Attempt minutely_15 precision if race is ≤6 h away.
        # When available, its rain_prob supersedes the hourly reading.
        nowcast_result = {"nowcast_available": False}
        if race_start_utc is not None:
            nowcast_result = fetch_weather_nowcast(lat, lon, race_start_utc, utc_offset)
            if nowcast_result.get("nowcast_available"):
                slots     = nowcast_result["nowcast_slots"]
                rain_prob = nowcast_result["rain_prob"]
                print(f"   🎯  Nowcast minutely_15 activo "
                      f"({slots} intervalos ×15 min) — rain_prob={rain_prob:.0%}")
            else:
                hours_away = (
                    (race_start_utc - datetime.now(timezone.utc)).total_seconds() / 3600
                    if race_start_utc else 99
                )
                print(f"   📊  Pronóstico horario "
                      f"(nowcast omitido: carrera en {hours_away:.0f}h > 6h)")
        else:
            print(f"   📊  Pronóstico horario (hora de carrera desconocida)")

        print(f"   ✅  Clima obtenido para {circuit} ({locality}, {country})")
        return {
            "avg_track_temp"   : (avg_temp + 10) if avg_temp is not None else None,
            "avg_humidity"     : avg_hum,
            "rain_prob"        : round(rain_prob, 3),
            "location"         : f"{locality}, {country}",
            "nowcast_available": nowcast_result.get("nowcast_available", False),
        }
    except Exception as e:
        print(f"   ⚠  Error obteniendo clima: {e}")
        return {}


def fetch_weather_nowcast(
    lat: float,
    lon: float,
    race_start_utc: datetime,
    utc_offset: int = 0,
) -> dict:
    """
    Fetches Open-Meteo minutely_15 precipitation for high-resolution nowcasting.

    Only meaningful within 6 hours of race start — hourly forecasts are already
    adequate beyond that window. Returns {"nowcast_available": False} when the
    race is too far away or the endpoint returns no usable data.

    Rain threshold: same 0.2 mm/15min as the hourly 0.2 mm/h gate, so
    rain_prob values are directly comparable to the hourly reading.
    """
    _NO_CAST = {"nowcast_available": False}

    now_utc       = datetime.now(timezone.utc)
    hours_to_race = (race_start_utc - now_utc).total_seconds() / 3600
    if hours_to_race > 6:
        return _NO_CAST

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude"     : lat,
                "longitude"    : lon,
                "minutely_15"  : "precipitation",
                "forecast_days": 3,
                "timezone"     : "auto",
            },
            timeout=REQ_TIMEOUT,
        )
        if r.status_code != 200:
            return _NO_CAST

        m15     = r.json().get("minutely_15", {})
        times   = m15.get("time",          [])
        precips = m15.get("precipitation", [])
        if not times or not precips:
            return _NO_CAST

        # 3-hour race window: 1 h before start → 2 h after (= 12 × 15-min slots)
        race_local = race_start_utc + timedelta(seconds=utc_offset)
        win_start  = (race_local - timedelta(hours=1)).replace(tzinfo=None)
        win_end    = (race_local + timedelta(hours=2)).replace(tzinfo=None)

        window = []
        for i, ts in enumerate(times):
            try:
                t = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
            except ValueError:
                continue
            if win_start <= t < win_end and i < len(precips) and precips[i] is not None:
                window.append(float(precips[i]))

        if not window:
            return _NO_CAST

        rain_prob = sum(1 for p in window if p > 0.2) / len(window)
        return {
            "rain_prob"        : round(rain_prob, 3),
            "nowcast_available": True,
            "nowcast_slots"    : len(window),
        }
    except Exception:
        return _NO_CAST


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
    "Autodromo Enzo e Dino Ferrari"    : 0.70,  # Imola: very low overtaking, T1 chicane blocks passes
    "Circuit de Monaco"                : 0.95,  # casi imposible adelantar
    "Circuit de Barcelona-Catalunya"   : 0.65,
    "Circuit Gilles Villeneuve"        : 0.45,
    "Red Bull Ring"                    : 0.65,
    "Silverstone Circuit"              : 0.55,  # fast corners limit passing despite 2 DRS zones
    "Hungaroring"                      : 0.80,  # muy difícil
    "Circuit de Spa-Francorchamps"     : 0.35,
    "Circuit Park Zandvoort"           : 0.70,
    "Autodromo Nazionale di Monza"     : 0.30,  # fácil adelantar
    "Baku City Circuit"                : 0.50,
    "Marina Bay Street Circuit"        : 0.75,
    "Circuit of the Americas"          : 0.45,
    "Autodromo Hermanos Rodriguez"     : 0.50,
    "Autodromo Jose Carlos Pace"       : 0.45,  # Interlagos: 2 DRS zones, Senna-S creates overtaking
    "Las Vegas Strip Circuit"          : 0.40,
    "Losail International Circuit"     : 0.45,
    "Yas Marina Circuit"               : 0.40,
}

# Standard race distance in laps per circuit — used for endurance embedding dimension.
# Source: official 2026 F1 race schedule (varies 44–78 laps).
CIRCUIT_RACE_LAPS = {
    "Albert Park Grand Prix Circuit"   : 58,
    "Shanghai International Circuit"   : 56,
    "Suzuka Circuit"                   : 53,
    "Bahrain International Circuit"    : 57,
    "Jeddah Corniche Circuit"          : 50,
    "Miami International Autodrome"    : 57,
    "Autodromo Enzo e Dino Ferrari"    : 63,
    "Circuit de Monaco"                : 78,
    "Circuit de Barcelona-Catalunya"   : 66,
    "Circuit Gilles Villeneuve"        : 70,
    "Red Bull Ring"                    : 71,
    "Silverstone Circuit"              : 52,
    "Hungaroring"                      : 70,
    "Circuit de Spa-Francorchamps"     : 44,
    "Circuit Park Zandvoort"           : 72,
    "Autodromo Nazionale di Monza"     : 53,
    "Baku City Circuit"                : 51,
    "Marina Bay Street Circuit"        : 62,
    "Circuit of the Americas"          : 56,
    "Autodromo Hermanos Rodriguez"     : 71,
    "Autodromo Jose Carlos Pace"       : 71,
    "Las Vegas Strip Circuit"          : 50,
    "Losail International Circuit"     : 57,
    "Yas Marina Circuit"               : 58,
}

# Historical probability of at least one SC/VSC per race at each circuit.
# Source: F1 race data 2018-2025.  Used to derive Poisson lambda for MC:
#   lambda = -ln(1 - prob)  →  P(≥1 SC) = 1 - e^(-lambda) matches historical rate.
SAFETY_CAR_PROB = {
    "Circuit de Monaco"                : 0.85,   # most likely SC on calendar
    "Baku City Circuit"                : 0.80,
    "Marina Bay Street Circuit"        : 0.75,   # Singapore
    "Albert Park Grand Prix Circuit"   : 0.70,   # Melbourne
    "Jeddah Corniche Circuit"          : 0.65,
    "Hungaroring"                      : 0.60,   # Hungary
    "Circuit Park Zandvoort"           : 0.60,
    "Autodromo Enzo e Dino Ferrari"    : 0.55,   # Imola
    "Circuit Gilles Villeneuve"        : 0.55,   # Canada
    "Autodromo Jose Carlos Pace"       : 0.55,   # Interlagos
    "Circuit of the Americas"          : 0.50,   # COTA
    "Silverstone Circuit"              : 0.50,
    "Circuit de Spa-Francorchamps"     : 0.50,
    "Las Vegas Strip Circuit"          : 0.50,
    "Suzuka Circuit"                   : 0.45,
    "Red Bull Ring"                    : 0.45,
    "Miami International Autodrome"    : 0.45,
    "Autodromo Hermanos Rodriguez"     : 0.45,   # Mexico
    "Shanghai International Circuit"   : 0.45,
    "Autodromo Nazionale di Monza"     : 0.40,
    "Bahrain International Circuit"    : 0.40,
    "Losail International Circuit"     : 0.40,   # Qatar
    "Circuit de Barcelona-Catalunya"   : 0.40,   # Spain
    "Yas Marina Circuit"               : 0.35,   # Abu Dhabi: fewest SC historically
}

# Historically optimal undercut lap window and probability per circuit.
# "laps" = (earliest, latest) lap the undercut tends to fire; (0,0) = no undercut.
UNDERCUT_WINDOW = {
    "Bahrain International Circuit"    : {"laps": (14, 18), "prob": 0.45},
    "Jeddah Corniche Circuit"          : {"laps": (15, 20), "prob": 0.35},
    "Albert Park Grand Prix Circuit"   : {"laps": (18, 22), "prob": 0.40},
    "Suzuka Circuit"                   : {"laps": (20, 25), "prob": 0.35},
    "Shanghai International Circuit"   : {"laps": (16, 20), "prob": 0.45},
    "Miami International Autodrome"    : {"laps": (16, 20), "prob": 0.40},
    "Autodromo Enzo e Dino Ferrari"    : {"laps": (22, 27), "prob": 0.40},
    "Circuit de Monaco"                : {"laps": ( 0,  0), "prob": 0.05},
    "Circuit de Barcelona-Catalunya"   : {"laps": (18, 22), "prob": 0.40},
    "Circuit Gilles Villeneuve"        : {"laps": (20, 25), "prob": 0.45},
    "Red Bull Ring"                    : {"laps": (16, 20), "prob": 0.50},
    "Silverstone Circuit"              : {"laps": (16, 20), "prob": 0.50},
    "Hungaroring"                      : {"laps": (25, 30), "prob": 0.35},
    "Circuit de Spa-Francorchamps"     : {"laps": (12, 16), "prob": 0.55},
    "Circuit Park Zandvoort"           : {"laps": (20, 25), "prob": 0.30},
    "Autodromo Nazionale di Monza"     : {"laps": (18, 23), "prob": 0.45},
    "Baku City Circuit"                : {"laps": (15, 20), "prob": 0.50},
    "Marina Bay Street Circuit"        : {"laps": (20, 25), "prob": 0.10},
    "Circuit of the Americas"          : {"laps": (18, 22), "prob": 0.45},
    "Autodromo Hermanos Rodriguez"     : {"laps": (20, 25), "prob": 0.40},
    "Autodromo Jose Carlos Pace"       : {"laps": (18, 22), "prob": 0.45},
    "Las Vegas Strip Circuit"          : {"laps": (16, 22), "prob": 0.40},
    "Losail International Circuit"     : {"laps": (20, 25), "prob": 0.40},
    "Yas Marina Circuit"               : {"laps": (22, 27), "prob": 0.35},
}

# Average stationary pit time by constructor (seconds).
# Faster crews → smaller score penalty when selected as pit victim in MC.
PIT_STOP_LOSS = {
    "Mercedes"    : 2.3,
    "McLaren"     : 2.3,
    "Red Bull"    : 2.4,
    "Ferrari"     : 2.5,
    "Haas"        : 2.6,
    "RB"          : 2.6,
    "Racing Bulls": 2.6,
    "Williams"    : 2.7,
    "Aston Martin": 2.7,
    "Alpine"      : 2.8,
    "Cadillac"    : 2.8,
    "Audi"        : 2.9,
    "Sauber"      : 2.9,
}
_PIT_LOSS_DEFAULT = 2.6
_PIT_LOSS_BEST    = min(PIT_STOP_LOSS.values())   # 2.3
_PIT_LOSS_WORST   = max(PIT_STOP_LOSS.values())   # 2.9

# ── Pirelli compound selections for 2026 (historical defaults / confirmed) ────
# Source: Pirelli press releases (press.pirelli.com), pitpass.com, f1network.net
# Format: (hard_C, medium_C, soft_C) — C6 was dropped for 2026; range is C1–C5.
# ✓ = confirmed from live Pirelli announcement; est. = estimated from circuit type.
# CONFIDENCE: lower than API-sourced features — scraped from inconsistent HTML.
# These serve as fallback when fetch_compound_selection() cannot reach live sources.
CIRCUIT_COMPOUND_DEFAULTS_2026: dict[str, tuple[int, int, int]] = {
    "Albert Park Grand Prix Circuit"   : (3, 4, 5),  # ✓ R1 confirmed
    "Shanghai International Circuit"   : (2, 3, 4),  # ✓ R2 confirmed
    "Suzuka Circuit"                   : (1, 2, 3),  # ✓ R3 confirmed
    "Bahrain International Circuit"    : (1, 2, 3),  # est. (demanding, high wear)
    "Jeddah Corniche Circuit"          : (2, 3, 4),  # est.
    "Miami International Autodrome"    : (2, 3, 4),  # est.
    "Autodromo Enzo e Dino Ferrari"    : (2, 3, 4),  # est.
    "Circuit de Monaco"                : (3, 4, 5),  # est. (street, softest)
    "Circuit de Barcelona-Catalunya"   : (2, 3, 4),  # est.
    "Circuit Gilles Villeneuve"        : (2, 3, 4),  # est.
    "Red Bull Ring"                    : (3, 4, 5),  # ✓ R8 confirmed
    # Silverstone: majority of sources say C1/C2/C3; one source (The Race) says
    # C2/C3/C4 ("one step softer than before"). Using C1/C2/C3 as default —
    # confirmed flag is False so the print output signals the uncertainty.
    "Silverstone Circuit"              : (1, 2, 3),  # majority: C1/C2/C3 (conflict: one src C2/C3/C4)
    "Hungaroring"                      : (2, 3, 4),  # est.
    "Circuit de Spa-Francorchamps"     : (1, 2, 3),  # est. (high-speed like Silverstone)
    "Circuit Park Zandvoort"           : (2, 3, 4),  # ✓ confirmed
    "Autodromo Nazionale di Monza"     : (3, 4, 5),  # ✓ confirmed
    "Baku City Circuit"                : (3, 4, 5),  # est. (was C4/C5/C6 in 2025; C6 dropped)
    "Marina Bay Street Circuit"        : (3, 4, 5),  # ✓ confirmed
    "Circuit of the Americas"          : (1, 3, 4),  # ✓ confirmed
    "Autodromo Hermanos Rodriguez"     : (2, 4, 5),  # ✓ confirmed
    "Autodromo Jose Carlos Pace"       : (2, 3, 4),  # ✓ confirmed
    "Las Vegas Strip Circuit"          : (3, 4, 5),  # ✓ confirmed
    "Losail International Circuit"     : (1, 2, 3),  # ✓ confirmed
    "Yas Marina Circuit"               : (3, 4, 5),  # ✓ confirmed
}

# Realistic weekend softness range: C1/C2/C3 avg=2.0 (hardest) → C3/C4/C5 avg=4.0 (softest).
# Used to normalize compound_softness to [0, 1] for the MC undercut multiplier.
_COMPOUND_AVG_MIN = 2.0   # C1/C2/C3
_COMPOUND_AVG_MAX = 4.0   # C3/C4/C5

# ── Corner telemetry segments — APPROXIMATE distance ranges (meters from lap start) ──
# IMPORTANT: These values are estimated from published circuit maps and GPS overlays.
# Accuracy is ±50–100 m. FastF1 telemetry is sampled at ~4 Hz (~70–80 km/h per sample
# at racing speeds), so precision is limited to ~5–15 m at best.
# Covers 8 strategically significant circuits only; others return neutral (NaN).
# DO NOT treat as authoritative track engineering data.
CIRCUIT_CORNERS: dict[str, dict[str, tuple[int, int]]] = {
    "Silverstone Circuit": {
        "Copse"             : (300,   550),   # T1 — fast right, entry speed diagnostic
        "Maggotts-Becketts" : (1400, 2100),   # T3-5 complex — most demanding sector
        "Stowe"             : (4700, 5000),   # T15 — high-load right hander
        "Club"              : (5100, 5450),   # T16 — final complex before pits
    },
    "Circuit de Monaco": {
        "Sainte Devote"     : (50,    220),   # T1 — opening right, sets up tunnel straight
        "Casino"            : (850,  1050),   # T6 — cambered right, tricky exit
        "Mirabeau"          : (1300, 1550),   # T8 — hairpin entry commitment
        "Rascasse"          : (2900, 3100),   # T18 — final chicane, last overtake spot
    },
    "Circuit de Spa-Francorchamps": {
        "La Source"         : (50,    200),   # T1 — hairpin, braking zone
        "Eau Rouge-Raidillon": (700,  1100),  # T2-3 — iconic uphill sequence
        "Pouhon"            : (3600, 3900),   # T11 — double left, highest-G flat corner
        "Blanchimont"       : (5700, 6050),   # T15-16 — high-speed left before Bus Stop
    },
    "Autodromo Nazionale di Monza": {
        "Variante del Rettifilo": (200,  600),  # T1-2 — first chicane, key braking zone
        "Variante della Roggia" : (1400, 1750), # T4-5 — second chicane, braking commitment
        "Lesmo 2"               : (3200, 3550), # T9 — fast right, circuit character
        "Parabolica"            : (5100, 5550), # T11 — final corner, last lap opportunity
    },
    "Suzuka Circuit": {
        "Turns 1-2"         : (100,   380),  # opening complex, race-start incident zone
        "Esses"             : (900,  1600),  # T3-7 — iconic S-curves, driver skill test
        "Hairpin"           : (2850, 3050),  # T11 — slow, key undercut pivot
        "Spoon"             : (4000, 4450),  # T13-14 — double left apex commitment
        "130R"              : (5050, 5250),  # T17 — fastest corner, bravery test
    },
    "Hungaroring": {
        "Turn 1"            : (50,    250),  # opening hairpin — only real overtake spot
        "Turn 2"            : (300,   500),  # tight right, exit sets up sector 1
        "Turns 6-7"         : (1800, 2150),  # mid-corner complex — traction key
        "Turn 11"           : (3200, 3500),  # final hairpin — pivotal for lap time
    },
    "Red Bull Ring": {
        "Turn 1"            : (100,   370),  # steep uphill braking zone
        "Turn 3"            : (950,  1150),  # crest exit — car balance critical
        "Turn 6"            : (2700, 2900),  # tight right hander, slow
        "Turns 9-10"        : (3800, 4100),  # final complex — DRS activation corner
    },
    "Bahrain International Circuit": {
        "Turns 1-2"         : (100,   400),  # first braking zone — race start critical
        "Turn 4"            : (650,   870),  # hairpin — main overtaking spot
        "Turns 8-10"        : (2100, 2500),  # flowing mid-circuit complex
        "Turns 14-15"       : (4200, 4580),  # tight final sector — tyre stress
    },
}


def fetch_compound_selection(
        circuit_name: str,
        round_num: int,
        race_name: str = "",
) -> dict:
    """
    Fetch the confirmed Pirelli dry compound selection (Hard/Medium/Soft C-numbers)
    for the upcoming race. Tries live web sources first; falls back to
    CIRCUIT_COMPOUND_DEFAULTS_2026 when scraping fails or the preview has not
    been published yet (typically <2 weeks before FP1).

    DATA RELIABILITY WARNING: This uses HTML scraping from inconsistent public
    sources, not a clean API. Confidence is intentionally lower than API-derived
    features. Source conflicts are flagged explicitly in the output.

    Returns:
        {
          "hard": int, "medium": int, "soft": int,   # C-numbers (1–5)
          "softness": float,  # normalised [0,1]: 0 = hardest weekend, 1 = softest
          "confirmed": bool,  # True = live web source; False = historical default
          "conflict": bool,   # True = sources disagree (do not treat as ground truth)
          "source": str,      # URL or "historical_default"
        }
    """
    import json as _json

    NOM_CACHE = Path("./f1_2026_compound_noms.json")
    cache_key = str(round_num)

    def _softness(h: int, m: int, s: int) -> float:
        avg = (h + m + s) / 3.0
        return float(np.clip((avg - _COMPOUND_AVG_MIN) / (_COMPOUND_AVG_MAX - _COMPOUND_AVG_MIN), 0.0, 1.0))

    # ── Check cache ───────────────────────────────────────────────────────
    if NOM_CACHE.exists():
        try:
            cached = _json.loads(NOM_CACHE.read_text(encoding="utf-8"))
            if cache_key in cached:
                e = cached[cache_key]
                result = {
                    "hard": e["hard"], "medium": e["medium"], "soft": e["soft"],
                    "softness": _softness(e["hard"], e["medium"], e["soft"]),
                    "confirmed": e.get("confirmed", False),
                    "conflict":  e.get("conflict", False),
                    "source":    e.get("source", "cache"),
                }
                tag = "confirmed ✓" if result["confirmed"] else (
                    "⚠ source conflict — treat as estimate" if result["conflict"]
                    else "historical default")
                print(f"   🏎️  Compounds (cached): C{e['hard']}/C{e['medium']}/C{e['soft']} "
                      f"→ softness {result['softness']:.2f}  [{tag}]")
                return result
        except Exception:
            pass

    # ── Scrape live sources ───────────────────────────────────────────────
    # Pirelli preview articles follow no consistent URL pattern; we try known
    # pages that were found to contain 2026 compound data for AUT+GBR.
    # This list should be updated as new press releases are published.
    # WARNING: HTML structure changes silently — always verify scraped results.
    _SCRAPE_URLS = [
        "https://www.f1network.net/main/s107/st208612.htm",
        "https://www.pitpass.com/82890/Pirelli-reveals-Austrian-and-British-GP-tyre-compounds",
    ]

    def _fetch_text(url: str) -> str:
        try:
            hdrs = {"User-Agent": "Mozilla/5.0 (compatible; F1Predictor/1.0)"}
            r = requests.get(url, headers=hdrs, timeout=10)
            if r.status_code != 200:
                return ""
            stripped = re.sub(r"<[^>]+>", " ", r.text)
            return re.sub(r"\s+", " ", stripped)
        except Exception:
            return ""

    def _extract_triple(text: str) -> tuple[int, int, int] | None:
        """
        Look for three consecutive C-numbers (C_n, C_{n+1}, C_{n+2}) in the text.
        Returns (hard, medium, soft) or None.
        Only accepts clean consecutive triples to avoid false matches.
        """
        nums = [int(m) for m in re.findall(r"\bC([1-5])\b", text)]
        seen: list[int] = []
        for n in nums:
            if n not in seen:
                seen.append(n)
        for i in range(len(seen) - 2):
            a, b, c = seen[i], seen[i+1], seen[i+2]
            if b == a + 1 and c == b + 1:
                return (a, b, c)
        return None

    # Check if this URL is relevant to this circuit
    circuit_kw = circuit_name.lower().split()[0]  # e.g. "silverstone"
    race_kw    = race_name.lower().split()[-2] if len(race_name.split()) >= 2 else race_name.lower()

    candidates: list[tuple[int, int, int]] = []
    found_source = ""
    for url in _SCRAPE_URLS:
        text = _fetch_text(url)
        if not text:
            continue
        tl = text.lower()
        if circuit_kw not in tl and race_kw not in tl:
            continue   # page doesn't mention this circuit
        triple = _extract_triple(text)
        if triple:
            candidates.append(triple)
            if not found_source:
                found_source = url

    # ── Resolve ───────────────────────────────────────────────────────────
    conflict  = False
    confirmed = False

    if len(candidates) >= 2 and candidates[0] == candidates[1]:
        # Two sources agree → confirmed
        hard, medium, soft = candidates[0]
        confirmed = True
    elif len(candidates) == 1:
        # Single source — use it but don't mark confirmed
        hard, medium, soft = candidates[0]
    elif len(candidates) >= 2:
        # Sources disagree → flag conflict, use majority or first
        conflict  = True
        hard, medium, soft = candidates[0]
        print(f"   ⚠️  Compound source conflict: {candidates} — using first, flagged as uncertain")
    else:
        # No live data found — fall back to hardcoded defaults
        default = CIRCUIT_COMPOUND_DEFAULTS_2026.get(circuit_name)
        if default:
            hard, medium, soft = default
        else:
            hard, medium, soft = 2, 3, 4   # mid-range neutral
        found_source = "historical_default"

    softness = _softness(hard, medium, soft)
    result = {
        "hard": hard, "medium": medium, "soft": soft,
        "softness": softness, "confirmed": confirmed,
        "conflict": conflict, "source": found_source,
    }

    # ── Print summary ──────────────────────────────────────────────────────
    if found_source == "historical_default":
        is_known_conflict = circuit_name == "Silverstone Circuit"
        note = ("⚠ one source says C2/C3/C4 — conflict not resolved"
                if is_known_conflict else "historical default")
        print(f"   🏎️  Compounds (fallback): C{hard}/C{medium}/C{soft} "
              f"→ softness {softness:.2f}  [{note}]")
    elif conflict:
        print(f"   🏎️  Compounds (conflict): C{hard}/C{medium}/C{soft} "
              f"→ softness {softness:.2f}  [⚠ sources disagree]")
    else:
        tag = "confirmed ✓" if confirmed else "single source"
        print(f"   🏎️  Compounds ({tag}): C{hard}/C{medium}/C{soft} "
              f"→ softness {softness:.2f}")

    # ── Cache result (atomic write) ────────────────────────────────────────
    try:
        cache: dict = {}
        if NOM_CACHE.exists():
            cache = _json.loads(NOM_CACHE.read_text(encoding="utf-8"))
        cache[cache_key] = {
            "hard": hard, "medium": medium, "soft": soft,
            "confirmed": confirmed, "conflict": conflict, "source": found_source,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = NOM_CACHE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(cache, indent=2), encoding="utf-8")
        tmp.replace(NOM_CACHE)
    except Exception as exc:
        print(f"   ⚠  Could not save compound cache: {exc}")

    return result


CIRCUIT_ID_MAP = {
    "Albert Park Grand Prix Circuit"   : "albert_park",
    "Shanghai International Circuit"   : "shanghai",
    "Suzuka Circuit"                   : "suzuka",
    "Bahrain International Circuit"    : "bahrain",
    "Jeddah Corniche Circuit"          : "jeddah",
    "Miami International Autodrome"    : "miami",
    "Autodromo Enzo e Dino Ferrari"    : "imola",
    "Circuit de Monaco"                : "monaco",
    "Circuit de Barcelona-Catalunya"   : "catalunya",
    "Circuit Gilles Villeneuve"        : "villeneuve",
    "Red Bull Ring"                    : "red_bull_ring",
    "Silverstone Circuit"              : "silverstone",
    "Hungaroring"                      : "hungaroring",
    "Circuit de Spa-Francorchamps"     : "spa",
    "Circuit Park Zandvoort"           : "zandvoort",
    "Autodromo Nazionale di Monza"     : "monza",
    "Baku City Circuit"                : "baku",
    "Marina Bay Street Circuit"        : "marina_bay",
    "Circuit of the Americas"          : "americas",
    "Autodromo Hermanos Rodriguez"     : "rodriguez",
    "Autodromo Jose Carlos Pace"       : "interlagos",
    "Las Vegas Strip Circuit"          : "vegas",
    "Losail International Circuit"     : "losail",
    "Yas Marina Circuit"               : "yas_marina",
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


def of1_collect_next_sprint_qualifying(next_round: int,
                                        sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Descarga datos de Sprint Qualifying del PRÓXIMO GP (solo fines de semana sprint).
    Retorna sq_pos_next (posición en SQ) y sq_time_next (tiempo de vuelta).
    Devuelve DataFrame vacío en fines de semana normales o si SQ no está disponible aún.
    """
    if next_round not in SPRINT_ROUNDS:
        return pd.DataFrame(columns=["code", "sq_pos_next", "sq_time_next"])
    sk = of1_session_key(sessions_df, next_round, "SQ")
    if not sk:
        return pd.DataFrame(columns=["code", "sq_pos_next", "sq_time_next"])

    print(f"⚡  Descargando Sprint Qualifying del próximo GP (sk={sk})...")
    laps = of1_get_laps(sk, enrich_with_stints=False)
    if laps.empty or "lap_duration" not in laps.columns:
        print("   ⚠  Sin datos de Sprint Qualifying para el próximo GP")
        return pd.DataFrame(columns=["code", "sq_pos_next", "sq_time_next"])

    valid = laps.dropna(subset=["lap_duration"])
    valid = valid[valid["lap_duration"] > 0]
    best  = valid.groupby("code").agg(
        sq_time_next=("lap_duration", "min"),
    ).reset_index()

    best = best.sort_values("sq_time_next").reset_index(drop=True)
    best["sq_pos_next"] = best.index + 1

    n_drivers = len(best)
    print(f"   ✅  {n_drivers} pilotos en Sprint Qualifying del próximo GP")
    if not best.empty:
        sq_pole = best.iloc[0]
        print(f"   ⚡  SQ POLE: {sq_pole['code']} — {sq_pole['sq_time_next']:.3f}s")

    return best[["code", "sq_pos_next", "sq_time_next"]]


def of1_collect_next_race_fp(next_round: int,
                               sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Descarga ritmo de FP1/FP2/FP3 del PRÓXIMO GP si ya están disponibles.
    Normalizado vs mediana de sesión. FP2 long run tiene peso mayor.
    Also extracts per-compound long-run pace (soft_pace_delta, medium_pace_delta,
    hard_pace_delta) from FP2 long stints when compound data is available.
    compound_preference = soft_pace_delta - medium_pace_delta
      (negative → driver is relatively faster on softs, positive → better on mediums)

    Sprint weekends (FP1 only): SESSION_WEIGHTS = {"FP1": 1.00}; FP1 long stints
    (≥8 laps, >1.5% slower than driver's best) are promoted to fp2_next_longrun as
    the race simulation proxy (no FP2/FP3 sessions exist on sprint weekends).
    """
    # Sprint weekends: FP1 is the only practice session — FP2/FP3 do not exist
    is_sprint = next_round in SPRINT_ROUNDS
    if is_sprint:
        print(f"   🏃  Sprint weekend (R{next_round}): FP1 is the only practice session — "
              f"FP2/FP3 do not exist")
    SESSION_WEIGHTS   = {"FP1": 1.00} if is_sprint else {"FP1": 0.20, "FP2": 0.50, "FP3": 0.30}
    quick_records     = {}
    longrun_records   = {}
    compound_longrun  = {}   # {code: {compound: [avg_pace_seconds]}}
    deg_records       = {}   # {code: {compound: [(slope_s_per_lap, n_laps)]}}
    speed_records     = {}   # {code: {"i1": [km/h, ...], "i2": [km/h, ...]}}
    race_sim_records  = {}   # {code: [(pace_s, deg_slope, n_laps)]}
    sessions_found    = []

    fp_session_list = ["FP1"] if is_sprint else ["FP1", "FP2", "FP3"]
    for sname in fp_session_list:
        sk = of1_session_key(sessions_df, next_round, sname)
        if not sk:
            continue
        # FP1 and FP2 get stint enrichment: FP1 for race sim detection, FP2 for compound/deg
        laps = of1_get_laps(sk, enrich_with_stints=(sname in ("FP1", "FP2")))
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

        # Speed trap readings (i1, i2) — already in /laps, no extra API call
        # Collect across all FP sessions; st_speed is a straight trap, excluded
        for trap_col, trap_key in (("i1_speed", "i1"), ("i2_speed", "i2")):
            if trap_col not in valid.columns:
                continue
            trap_valid = valid[valid[trap_col].notna() & (valid[trap_col] > 80)]
            for code, grp in trap_valid.groupby("code"):
                speed_records.setdefault(code, {}).setdefault(trap_key, []).extend(
                    grp[trap_col].tolist()
                )

        # FP1 race simulation detection
        # A long stint (≥8 laps) whose mid-pace is >1.5% slower than the driver's
        # fastest FP1 lap is classified as a race simulation (heavy fuel signature).
        if sname == "FP1" and "stint_number" in valid.columns:
            has_lapnum_fp1 = "lap_number" in valid.columns
            fp1_best       = valid.groupby("code")["lap_duration"].min()
            stint_len_fp1  = valid.groupby(["code", "stint_number"]).size()
            long_fp1       = stint_len_fp1[stint_len_fp1 >= 8].index
            for (code, stint) in long_fp1:
                if code not in fp1_best:
                    continue
                mask = (valid["code"] == code) & (valid["stint_number"] == stint)
                grp  = valid[mask].sort_values("lap_number") if has_lapnum_fp1 \
                       else valid[mask]
                sl   = grp["lap_duration"].reset_index(drop=True)
                mid  = sl.iloc[len(sl)//4: len(sl)*3//4]
                if len(mid) < 3:
                    continue
                pace     = mid.mean()
                best_lap = fp1_best[code]
                # Race sim signature: >1.5% slower than driver's best lap (heavy fuel)
                if pace > best_lap * 1.015:
                    slope = np.polyfit(np.arange(len(sl)), sl.values, 1)[0]
                    race_sim_records.setdefault(code, []).append((pace, slope, len(sl)))

        # FP2 long runs (or FP1 on sprint weekends as race sim proxy) —
        # extract compound pace + degradation slope.
        # On sprint weekends FP1 long stints serve as the race simulation data
        # (heavy fuel signature: >1.5% slower than driver best) → fp2_next_longrun.
        _is_longrun_session = (sname == "FP2") or (is_sprint and sname == "FP1")
        if is_sprint and sname == "FP1":
            print("   ⚠  Sprint weekend: using FP1 long runs as race sim proxy "
                  "(fp2_next_longrun derived from FP1)")
        if _is_longrun_session and "stint_number" in valid.columns:
            has_cmp     = "compound" in valid.columns
            has_lapnum  = "lap_number" in valid.columns
            stint_len   = valid.groupby(["code","stint_number"]).size()
            long_stints = stint_len[stint_len >= 8].index
            for (code, stint) in long_stints:
                mask = (valid["code"] == code) & (valid["stint_number"] == stint)
                # Sort by lap number for correct slope direction
                grp  = valid[mask].sort_values("lap_number") if has_lapnum \
                       else valid[mask]
                sl   = grp["lap_duration"].reset_index(drop=True)
                mid  = sl.iloc[len(sl)//4: len(sl)*3//4]
                if len(mid) >= 3:
                    pace = mid.mean()
                    longrun_records.setdefault(code, []).append(pace)

                    # Degradation rate: linear slope on full sorted stint
                    # Units: seconds per lap (positive = getting slower)
                    slope = np.polyfit(np.arange(len(sl)), sl.values, 1)[0]

                    if has_cmp:
                        cmp_vals = valid[mask]["compound"].dropna()
                        if not cmp_vals.empty:
                            cmp = cmp_vals.mode().iloc[0]
                            if cmp in ("SOFT", "MEDIUM", "HARD"):
                                compound_longrun.setdefault(code, {}) \
                                                .setdefault(cmp, []).append(pace)
                                deg_records.setdefault(code, {}) \
                                           .setdefault(cmp, []).append((slope, len(sl)))
                    else:
                        # No compound info — track under a generic key
                        deg_records.setdefault(code, {}) \
                                   .setdefault("ALL", []).append((slope, len(sl)))

    if not sessions_found:
        return pd.DataFrame(columns=["code", "fp_next_delta", "fp2_next_longrun",
                                     "soft_pace_delta", "medium_pace_delta",
                                     "hard_pace_delta", "compound_preference",
                                     "tyre_deg_rate", "deg_rate_soft",
                                     "deg_rate_medium", "deg_rate_hard",
                                     "high_speed_delta", "medium_speed_delta",
                                     "low_speed_delta", "corner_balance",
                                     "race_sim_delta", "race_sim_deg"])

    print(f"   ✅  Prácticas del próximo GP disponibles: {', '.join(sessions_found)}")

    # Per-compound field reference: median long-run pace on each compound
    _cmp_ref = {}
    for cmp in ("SOFT", "MEDIUM", "HARD"):
        all_times = [t for cd in compound_longrun.values() for t in cd.get(cmp, [])]
        if all_times:
            _cmp_ref[cmp] = np.median(all_times)

    # Per-driver degradation rates from FP2 long stints
    # tyre_deg_rate = weighted-average slope across all compound-stint pairs
    # Units: seconds/lap; positive = getting slower (more deg); lower = better tyre manager
    _deg_summary = {}
    for code, cmp_data in deg_records.items():
        all_pairs = [(slope, n) for pairs in cmp_data.values() for slope, n in pairs]
        if all_pairs:
            slopes, ns        = zip(*all_pairs)
            tyre_deg_rate     = float(np.average(slopes, weights=ns))
        else:
            tyre_deg_rate     = np.nan
        dr_soft = dr_med = dr_hard = np.nan
        for cmp, pairs in cmp_data.items():
            slopes, ns = zip(*pairs)
            val = float(np.average(slopes, weights=ns))
            if cmp == "SOFT":     dr_soft = val
            elif cmp == "MEDIUM": dr_med  = val
            elif cmp == "HARD":   dr_hard = val
        _deg_summary[code] = {
            "tyre_deg_rate":   tyre_deg_rate,
            "deg_rate_soft":   dr_soft,
            "deg_rate_medium": dr_med,
            "deg_rate_hard":   dr_hard,
        }

    # ── Corner speed profile from i1/i2 speed traps ──────────────────────
    # Classify each trap by session-wide median: <160=low, 160-220=medium, >=220=high
    # delta = (field_median - driver_avg) / field_median * 100 → negative = faster
    def _speed_class(median_kmh: float) -> str:
        if median_kmh < 160: return "low"
        if median_kmh < 220: return "medium"
        return "high"

    _trap_classes = {}  # "i1" | "i2" → "low" | "medium" | "high"
    _trap_field   = {}  # "i1" | "i2" → field median km/h
    for trap_key in ("i1", "i2"):
        all_spds = [s for recs in speed_records.values() for s in recs.get(trap_key, [])]
        if all_spds:
            fmed = float(np.median(all_spds))
            _trap_field[trap_key]   = fmed
            _trap_classes[trap_key] = _speed_class(fmed)

    _drv_trap_avg = {}  # code → {trap_key: avg_speed}
    for code, recs in speed_records.items():
        _drv_trap_avg[code] = {
            tk: float(np.mean(spds)) for tk, spds in recs.items() if spds
        }

    # Build per-driver speed class deltas
    # Accumulate (delta, weight) per class; weight = count of readings in that class
    _corner_deltas = {}  # code → {class: [deltas]}
    for code, trap_avgs in _drv_trap_avg.items():
        for tk, drv_avg in trap_avgs.items():
            cls = _trap_classes.get(tk)
            fmed = _trap_field.get(tk)
            if cls is None or fmed is None or fmed == 0:
                continue
            delta = (fmed - drv_avg) / fmed * 100   # negative = driver faster
            _corner_deltas.setdefault(code, {}).setdefault(cls, []).append(delta)

    _corner_summary = {}  # code → {high_speed_delta, medium_speed_delta, low_speed_delta, corner_balance}
    for code in _drv_trap_avg:
        cls_data = _corner_deltas.get(code, {})
        class_avgs = {cls: float(np.mean(vals)) for cls, vals in cls_data.items()}
        high_d   = class_avgs.get("high",   np.nan)
        medium_d = class_avgs.get("medium", np.nan)
        low_d    = class_avgs.get("low",    np.nan)
        avail    = [v for v in (high_d, medium_d, low_d) if not np.isnan(v)]
        balance  = (max(avail) - min(avail)) if len(avail) >= 2 else np.nan
        _corner_summary[code] = {
            "high_speed_delta":   high_d,
            "medium_speed_delta": medium_d,
            "low_speed_delta":    low_d,
            "corner_balance":     balance,
        }

    if _trap_classes:
        print(f"   🔷  Speed trap classes: "
              f"i1={_trap_classes.get('i1','?')} "
              f"({_trap_field.get('i1',0):.0f} km/h), "
              f"i2={_trap_classes.get('i2','?')} "
              f"({_trap_field.get('i2',0):.0f} km/h)")

    # ── FP1 race simulation summary ───────────────────────────────────────
    # race_sim_delta = (driver_avg_pace - field_median) / field_median * 100
    # Negative = driver's race sim is faster than field median → better race pace
    _race_sim_summary = {}
    if race_sim_records:
        all_rs_paces = [p for recs in race_sim_records.values() for p, _, _ in recs]
        rs_field     = float(np.median(all_rs_paces))
        n_detected   = len(race_sim_records)
        print(f"   🏁  Race sims FP1 detectados: {n_detected} pilotos "
              f"(ref campo: {rs_field:.3f}s)")
        for code, recs in race_sim_records.items():
            paces  = [p for p, _, _ in recs]
            slopes = [s for _, s, _ in recs]
            ns_arr = [n for _, _, n in recs]
            avg_pace  = float(np.average(paces,  weights=ns_arr))
            avg_slope = float(np.average(slopes, weights=ns_arr))
            _race_sim_summary[code] = {
                "race_sim_delta": (avg_pace - rs_field) / rs_field * 100,
                "race_sim_deg":   avg_slope,
            }

    rows     = []
    lr_ref   = (np.median([t for ts in longrun_records.values() for t in ts])
                if longrun_records else None)
    all_codes = set(quick_records) | set(longrun_records)
    for code in all_codes:
        fp_delta = np.nan
        if code in quick_records:
            ds, ws   = zip(*quick_records[code])
            fp_delta = np.average(ds, weights=ws)
        fp2_lr = np.nan
        if code in longrun_records and lr_ref:
            fp2_lr = (np.mean(longrun_records[code]) - lr_ref) / lr_ref * 100

        # Compound-specific pace delta vs field median (negative = faster than field)
        cdata  = compound_longrun.get(code, {})
        soft_d = medium_d = hard_d = np.nan
        for cmp, ref in _cmp_ref.items():
            drv_avg = np.mean(cdata[cmp]) if cmp in cdata else np.nan
            delta   = (drv_avg - ref) / ref * 100 if not np.isnan(drv_avg) else np.nan
            if cmp == "SOFT":     soft_d   = delta
            elif cmp == "MEDIUM": medium_d = delta
            elif cmp == "HARD":   hard_d   = delta
        comp_pref = (soft_d - medium_d) if not (np.isnan(soft_d) or np.isnan(medium_d)) \
                    else np.nan

        dr = _deg_summary.get(code, {})
        cs = _corner_summary.get(code, {})
        rs = _race_sim_summary.get(code, {})
        rows.append({"code": code, "fp_next_delta": fp_delta,
                     "fp2_next_longrun":   fp2_lr,
                     "soft_pace_delta":    soft_d,
                     "medium_pace_delta":  medium_d,
                     "hard_pace_delta":    hard_d,
                     "compound_preference": comp_pref,
                     "tyre_deg_rate":      dr.get("tyre_deg_rate",   np.nan),
                     "deg_rate_soft":      dr.get("deg_rate_soft",   np.nan),
                     "deg_rate_medium":    dr.get("deg_rate_medium", np.nan),
                     "deg_rate_hard":      dr.get("deg_rate_hard",   np.nan),
                     "high_speed_delta":   cs.get("high_speed_delta",   np.nan),
                     "medium_speed_delta": cs.get("medium_speed_delta", np.nan),
                     "low_speed_delta":    cs.get("low_speed_delta",    np.nan),
                     "corner_balance":     cs.get("corner_balance",     np.nan),
                     "race_sim_delta":     rs.get("race_sim_delta", np.nan),
                     "race_sim_deg":       rs.get("race_sim_deg",   np.nan)})
    return pd.DataFrame(rows)


# 2026 dry tyre allocation per driver per race weekend (Article 30, Sporting Regs)
# Standard: 8 soft / 3 medium / 2 hard (13 total)
# Sprint:   6 soft / 4 medium / 2 hard (12 total; different distribution)
_TYRE_ALLOC_STANDARD : dict[str, int] = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}
_TYRE_ALLOC_SPRINT   : dict[str, int] = {"SOFT": 6, "MEDIUM": 4, "HARD": 2}


def compute_tyre_inventory(
        sessions_df : pd.DataFrame,
        next_round  : int,
) -> pd.DataFrame:
    """
    Estimates per-driver new tyre set inventory for the upcoming race by
    counting fresh-set starts (tyre_age_at_start == 0) across all completed
    practice sessions of the current race weekend, then subtracting from the
    standard 2026 allocation.

    Tracked sessions:
      Standard weekend : FP1, FP2, FP3
      Sprint weekend   : FP1, S (Sprint Race)

    Returns DataFrame[code, soft_new_remaining, medium_new_remaining,
                      hard_new_remaining, total_new_remaining,
                      tyre_inventory_score]
    where tyre_inventory_score is the z-score of soft_new_remaining vs field.
    Returns empty DataFrame when no practice data is available yet.

    Cache: f1_2026_tyre_inventory.json, keyed by "{round}_{session_code}".
    Cache is per-session so already-completed sessions are never re-fetched.
    """
    import json as _json

    is_sprint   = next_round in SPRINT_ROUNDS
    alloc       = _TYRE_ALLOC_SPRINT if is_sprint else _TYRE_ALLOC_STANDARD
    sessions    = ["FP1", "S"] if is_sprint else ["FP1", "FP2", "FP3"]

    # Load cache
    cache: dict = {}
    if TYRE_INVENTORY_CACHE.exists():
        try:
            cache = _json.loads(TYRE_INVENTORY_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # fresh_used[driver_code][compound] = count of fresh sets used across sessions
    fresh_used: dict[str, dict[str, int]] = {}
    any_data = False

    for sess_code in sessions:
        cache_key = f"{next_round}_{sess_code}"

        if cache_key in cache:
            sess_data: dict = cache[cache_key]
            if sess_data:
                print(f"   💾  Tyre inventory {sess_code} R{next_round}: "
                      f"{len(sess_data)} pilotos (cache)")
        else:
            sk = of1_session_key(sessions_df, next_round, sess_code)
            if sk is None:
                continue  # session not yet completed or not in calendar

            stints_df = of1_get_stints(sk)
            sess_data = {}

            if not stints_df.empty and "tyre_age_at_start" in stints_df.columns:
                for _, row in stints_df.iterrows():
                    code     = str(row.get("code", "")).strip()
                    compound = str(row.get("compound", "")).upper()
                    age      = row.get("tyre_age_at_start", None)
                    if not code or compound not in ("SOFT", "MEDIUM", "HARD"):
                        continue
                    if age is None or (isinstance(age, float) and np.isnan(age)):
                        continue
                    if int(age) == 0:
                        sess_data.setdefault(code, {})
                        sess_data[code][compound] = sess_data[code].get(compound, 0) + 1

            # Persist to cache (even empty, so we don't re-fetch a completed session
            # that genuinely returned no usable stints data)
            cache[cache_key] = sess_data
            try:
                _tmp = TYRE_INVENTORY_CACHE.with_suffix(".tmp")
                _tmp.write_text(_json.dumps(cache, indent=2, ensure_ascii=False),
                                encoding="utf-8")
                _tmp.replace(TYRE_INVENTORY_CACHE)
            except Exception as _e:
                print(f"   ⚠  No se pudo escribir caché de inventario: {_e}")

            if sess_data:
                print(f"   📊  Tyre inventory {sess_code} R{next_round}: "
                      f"{len(sess_data)} pilotos (fetched)")

        if not sess_data:
            continue

        any_data = True
        for code, compounds in sess_data.items():
            fresh_used.setdefault(code, {})
            for compound, count in compounds.items():
                fresh_used[code][compound] = fresh_used[code].get(compound, 0) + count

    if not any_data:
        return pd.DataFrame()

    rows = []
    for code, used in fresh_used.items():
        soft_rem   = max(0, alloc["SOFT"]   - used.get("SOFT",   0))
        medium_rem = max(0, alloc["MEDIUM"] - used.get("MEDIUM", 0))
        hard_rem   = max(0, alloc["HARD"]   - used.get("HARD",   0))
        rows.append({
            "code"                 : code,
            "soft_new_remaining"   : soft_rem,
            "medium_new_remaining" : medium_rem,
            "hard_new_remaining"   : hard_rem,
            "total_new_remaining"  : soft_rem + medium_rem + hard_rem,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # tyre_inventory_score = z-score of soft_new_remaining across field
    # Positive = more fresh softs than average = strategic flexibility advantage
    _mean = df["soft_new_remaining"].mean()
    _std  = df["soft_new_remaining"].std()
    if _std and _std > 0:
        df["tyre_inventory_score"] = (df["soft_new_remaining"] - _mean) / _std
    else:
        df["tyre_inventory_score"] = 0.0

    return df


def fetch_corner_telemetry(
        next_round   : int,
        next_circuit : str,
        completed    : list,
) -> pd.DataFrame:
    """
    Loads qualifying telemetry for next_round via FastF1 and computes per-driver
    corner mastery metrics for each named segment in CIRCUIT_CORNERS[next_circuit].

    Returns DataFrame[code, corner_mastery_score, corner_detail] or empty DataFrame.

    CONFIDENCE notes (lower than OpenF1 API features):
    - Corner distance boundaries are APPROXIMATE (±50–100 m from circuit maps).
    - FastF1 telemetry is ~4 Hz → one sample every ~15–20 m at racing speed.
    - Throttle/Brake channels vary by session: float 0–100 or bool 0/1.
    - Only 8 circuits have corner maps; all others return empty → neutral fallback.
    - Dormant below TELEMETRY_MIN_RACES (12) — not enough history to calibrate.

    Metrics per corner:
      min_speed    : minimum speed (km/h) in the corner distance window
      throttle_pct : fraction through window where Throttle first exceeds 50 %
                     (earlier = better exit; NaN if channel unavailable)
      brake_pct    : fraction through window where Brake last exceeds 50 %
                     (later = deeper braking commitment; NaN if unavailable)

    corner_mastery_score = weighted mean of (driver_min_speed / session_best_min_speed)
    across all corners, weighted by corner length (distance range).
    Score near 1.0 = matching the fastest driver through corners.

    Cache: CORNER_TELEMETRY_CACHE, keyed by "{SEASON}_{round}_Q".
    FastF1 telemetry loads take several seconds per driver — cache is essential.
    """
    import json as _json

    if len(completed) < TELEMETRY_MIN_RACES:
        print(f"   🏎️  Corner telemetry dormant "
              f"({len(completed)}/{TELEMETRY_MIN_RACES} races)")
        return pd.DataFrame()

    corners = CIRCUIT_CORNERS.get(next_circuit)
    if corners is None:
        print(f"   ℹ️  Corner telemetry: '{next_circuit}' not in CIRCUIT_CORNERS — neutral")
        return pd.DataFrame()

    # Load per-session cache
    cache: dict = {}
    if CORNER_TELEMETRY_CACHE.exists():
        try:
            cache = _json.loads(
                CORNER_TELEMETRY_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    cache_key = f"{SEASON}_{next_round}_Q"
    if cache_key in cache:
        cached_rows = cache[cache_key]
        if not cached_rows:
            print(f"   ℹ️  Corner telemetry R{next_round}: no data (cached)")
            return pd.DataFrame()
        df = pd.DataFrame(cached_rows)
        # corner_detail is stored as dict string — restore
        if "corner_detail" in df.columns:
            df["corner_detail"] = df["corner_detail"].apply(
                lambda x: x if isinstance(x, dict) else {})
        print(f"   💾  Corner telemetry R{next_round}: "
              f"{len(df)} pilotos (cache)")
        return df

    # ── Load qualifying session with telemetry via FastF1 ─────────────────
    print(f"   📡  Cargando telemetría FastF1 Q R{next_round} "
          f"({next_circuit}) — puede tardar ~30–60 s...")
    try:
        sess = fastf1.get_session(SEASON, next_round, "Q")
        sess.load(laps=True, telemetry=True, weather=False, messages=False)
    except Exception as _e:
        print(f"   ⚠  FastF1 telemetry load failed: {_e}")
        cache[cache_key] = []
        _write_json_atomic(CORNER_TELEMETRY_CACHE, cache)
        return pd.DataFrame()

    try:
        laps = sess.laps
        if laps is None or len(laps) == 0:
            raise ValueError("no laps in session")
    except Exception as _e:
        print(f"   ⚠  FastF1 Q session has no lap data: {_e}")
        cache[cache_key] = []
        _write_json_atomic(CORNER_TELEMETRY_CACHE, cache)
        return pd.DataFrame()

    corner_names   = list(corners.keys())
    corner_ranges  = list(corners.values())
    corner_lengths = [r[1] - r[0] for r in corner_ranges]
    total_length   = sum(corner_lengths)
    corner_weights = [ln / total_length for ln in corner_lengths]

    # session-best minimum speed per corner (across all drivers)
    session_best: dict[str, float | None] = {n: None for n in corner_names}
    # per-driver metrics: code → {corner_name → {min_speed, throttle_pct, brake_pct}}
    driver_metrics: dict[str, dict] = {}

    drivers_in_session = laps["Driver"].dropna().unique().tolist()
    n_processed = 0

    for drv in drivers_in_session:
        try:
            drv_laps = laps.pick_driver(drv)
            if len(drv_laps) == 0:
                continue
            # Filter to representative fast laps before picking fastest
            fast = drv_laps.pick_quicklaps(threshold=1.07)
            fastest_lap = fast.pick_fastest() if len(fast) > 0 \
                         else drv_laps.pick_fastest()
            if fastest_lap is None:
                continue

            tel = fastest_lap.get_telemetry()
            if tel is None or tel.empty:
                continue
            required = {"Distance", "Speed"}
            if not required.issubset(tel.columns):
                continue

            # Map FastF1 driver abbr → our 3-letter code
            # FastF1 "Driver" field is already the 3-letter abbreviation
            code = str(drv).upper()

            corner_data: dict[str, dict] = {}
            for name, (d_start, d_end), w in zip(
                    corner_names, corner_ranges, corner_weights):
                seg = tel[(tel["Distance"] >= d_start) &
                          (tel["Distance"] <= d_end)]
                if len(seg) < 2:
                    continue

                min_speed = float(seg["Speed"].min())

                throttle_pct = np.nan
                brake_pct    = np.nan

                if "Throttle" in seg.columns:
                    t_vals   = seg["Throttle"].values.astype(float)
                    t_thresh = 0.5 if t_vals.max() <= 1.0 else 50.0
                    on_mask  = seg["Throttle"] > t_thresh
                    if on_mask.any():
                        first_on_d   = float(seg.loc[on_mask, "Distance"].iloc[0])
                        throttle_pct = float(np.clip(
                            (first_on_d - d_start) / (d_end - d_start), 0.0, 1.0))

                if "Brake" in seg.columns:
                    b_vals   = seg["Brake"].values.astype(float)
                    b_thresh = 0.5 if b_vals.max() <= 1.0 else 50.0
                    brk_mask = seg["Brake"] > b_thresh
                    if brk_mask.any():
                        last_brk_d = float(seg.loc[brk_mask, "Distance"].iloc[-1])
                        brake_pct  = float(np.clip(
                            (last_brk_d - d_start) / (d_end - d_start), 0.0, 1.0))

                corner_data[name] = {
                    "min_speed"   : round(min_speed, 1),
                    "throttle_pct": round(throttle_pct, 3)
                                    if not np.isnan(throttle_pct) else None,
                    "brake_pct"   : round(brake_pct, 3)
                                    if not np.isnan(brake_pct) else None,
                }
                # Track session best (highest min speed = least scrubbing)
                if (session_best[name] is None or
                        min_speed > session_best[name]):
                    session_best[name] = min_speed

            if corner_data:
                driver_metrics[code] = corner_data
                n_processed += 1

        except Exception:
            continue  # skip drivers with missing/malformed telemetry

    if not driver_metrics:
        print("   ⚠  Corner telemetry: no usable data extracted")
        cache[cache_key] = []
        _write_json_atomic(CORNER_TELEMETRY_CACHE, cache)
        return pd.DataFrame()

    print(f"   📊  Corner telemetry: {n_processed} pilotos procesados "
          f"({next_circuit}, {len(corner_names)} corners)")

    # ── Compute corner_mastery_score per driver ────────────────────────────
    rows = []
    for code, c_data in driver_metrics.items():
        ratios   = []
        weights  = []
        per_name = {}

        for name, (d_start, d_end), w in zip(
                corner_names, corner_ranges, corner_weights):
            best = session_best.get(name)
            if best is None or best == 0 or name not in c_data:
                continue
            ratio = c_data[name]["min_speed"] / best
            ratios.append(ratio)
            weights.append(w)
            per_name[name] = round(ratio, 4)

        if not ratios:
            continue

        w_total = sum(weights)
        mastery = float(
            sum(r * wt / w_total for r, wt in zip(ratios, weights))
        ) if w_total > 0 else float(np.mean(ratios))

        rows.append({
            "code"                : code,
            "corner_mastery_score": round(mastery, 4),
            "corner_detail"       : per_name,
        })

    if not rows:
        cache[cache_key] = []
        _write_json_atomic(CORNER_TELEMETRY_CACHE, cache)
        return pd.DataFrame()

    cache[cache_key] = rows
    _write_json_atomic(CORNER_TELEMETRY_CACHE, cache)

    return pd.DataFrame(rows)


def _write_json_atomic(path: Path, data: dict) -> None:
    """Atomic JSON write: write to .tmp then rename to avoid partial reads."""
    import json as _json
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
    except Exception as _e:
        print(f"   ⚠  JSON write failed ({path.name}): {_e}")


def fetch_circuit_affinity(circuit_name: str) -> pd.DataFrame:
    """
    Fetch race results at circuit_name for the last 3 seasons from Jolpica.
    Returns DataFrame[code, circuit_affinity] — mean finishing position over
    all visits in those years. Drivers with < 2 visits get NaN so the caller
    can substitute the field median (neutral, no bonus or penalty).
    DNFs counted as P18 (light penalty, not catastrophic).
    One request per year to avoid ergast's oldest-first default pagination.
    """
    circuit_id = CIRCUIT_ID_MAP.get(circuit_name)
    if not circuit_id:
        print(f"   ⚠  Circuit ID desconocido para '{circuit_name}' — sin affinity")
        return pd.DataFrame(columns=["code", "circuit_affinity"])

    print(f"📍  Descargando historial de {circuit_name} ({circuit_id}) — últimos 3 años...")
    rows = []
    for year in range(int(SEASON) - 3, int(SEASON)):   # 2023, 2024, 2025
        url  = f"{JOLPICA_BASE}/{year}/circuits/{circuit_id}/results.json"
        data = api_get(url)
        if not data:
            continue
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            for res in race.get("Results", []):
                drv    = res.get("Driver", {})
                code   = drv.get("code", drv.get("driverId", "???").upper()[:3])
                status = res.get("status", "")
                try:
                    pos = int(res.get("position", 18))
                except (ValueError, TypeError):
                    pos = 18
                if status not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps"):
                    pos = 18
                rows.append({"code": code, "pos": pos, "year": year})

    if not rows:
        print(f"   ⚠  Sin historial disponible para {circuit_id} en los últimos 3 años")
        return pd.DataFrame(columns=["code", "circuit_affinity"])

    df  = pd.DataFrame(rows)
    agg = df.groupby("code")["pos"].agg(mean="mean", count="count").reset_index()
    agg.columns = ["code", "circuit_affinity", "_visits"]
    agg.loc[agg["_visits"] < 2, "circuit_affinity"] = np.nan
    good = agg["_visits"].ge(2).sum()
    print(f"   ✅  {len(rows)} resultados en {df['year'].nunique()} temporadas — "
          f"{good} pilotos con ≥2 visitas")
    return agg[["code", "circuit_affinity"]]


# ─────────────────────────────────────────────────────────────
#  18a. PERFILES COMPORTAMENTALES HISTÓRICOS 2023-2025
# ─────────────────────────────────────────────────────────────
# Race names that were run in significantly wet conditions
_WET_RACE_KEYWORDS = [
    (2023, "British"), (2023, "Qatar"),
    (2024, "British"), (2024, "Canada"), (2024, "Belgian"), (2024, "Japan"),
    (2025, "British"), (2025, "Belgian"), (2025, "Japan"), (2025, "Australia"),
    (2025, "Brazil"),
]

_PROFILES_CACHE_DAYS = 7


def fetch_driver_behavioral_profiles(current_drivers: list) -> dict:
    """
    Fetch 2023-2025 Jolpica data and compute 5 regulation-independent
    driver skill fingerprints.  Results are cached in PROFILES_FILE
    and only refreshed when the file is > 7 days old.

    Returned dict: {driver_code: {metric: value, ...}}
    """
    if os.path.exists(PROFILES_FILE):
        age_days = (time.time() - os.path.getmtime(PROFILES_FILE)) / 86400
        if age_days < _PROFILES_CACHE_DAYS:
            with open(PROFILES_FILE) as f:
                cached = json.load(f)
            print(f"   📂  Perfiles cargados desde caché ({age_days:.1f}d antigüedad, "
                  f"{len(cached)} pilotos)")
            return cached

    print("🧬  Construyendo perfiles comportamentales 2023–2025 (primera vez ~90s)...")
    years = [2023, 2024, 2025]

    overtaking_data = {}   # code → [positions_gained from midfield grid P6-P15]
    quali_deltas    = {}   # code → [quali_pos - teammate_quali_pos per race]
    wet_data        = {}   # code → [finish_pos - season_avg_finish  (neg = better)]
    dnf_counts      = {}   # code → {mech, acc, total}
    tyre_data       = {}   # code → [pos_at_80pct - final_pos  (pos = gained laps)]

    for year in years:
        sched_data = api_get(f"{JOLPICA_BASE}/{year}.json", params={"limit": 100})
        if not sched_data:
            continue
        races_sched = sched_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        n_rounds = len(races_sched)

        # Map round → wet flag from known-wet race names
        wet_rounds: set = set()
        for race in races_sched:
            rnum  = int(race.get("round", 0))
            rname = race.get("raceName", "")
            for (wy, kw) in _WET_RACE_KEYWORDS:
                if wy == year and kw.lower() in rname.lower():
                    wet_rounds.add(rnum)

        # ── Race results ──────────────────────────────────────────────────
        print(f"   📥  {year}: resultados ({n_rounds} carreras)...")
        year_results: dict = {}
        driver_finishes: dict = {}

        for rnd in range(1, n_rounds + 1):
            data = api_get(f"{JOLPICA_BASE}/{year}/{rnd}/results.json")
            time.sleep(REQ_DELAY)
            if not data:
                continue
            r_races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not r_races:
                continue
            results = r_races[0].get("Results", [])
            year_results[rnd] = results

            for res in results:
                drv    = res.get("Driver", {})
                code   = drv.get("code", drv.get("driverId", "???").upper()[:3])
                status = res.get("status", "")
                try:
                    pos = int(res.get("position", 20))
                except (ValueError, TypeError):
                    pos = 20
                if status not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps"):
                    pos = 20   # DNF → P20 for season average
                driver_finishes.setdefault(code, []).append(pos)

        season_avg = {code: float(np.mean(ps)) for code, ps in driver_finishes.items()}

        for rnd, results in year_results.items():
            is_wet = rnd in wet_rounds
            for res in results:
                drv    = res.get("Driver", {})
                code   = drv.get("code", drv.get("driverId", "???").upper()[:3])
                status = res.get("status", "")
                dnf    = status not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps")
                try:
                    grid = int(res.get("grid", 0))
                    pos  = int(res.get("position", 20))
                except (ValueError, TypeError):
                    continue

                # DNF classification
                dc = dnf_counts.setdefault(code, {"mech": 0, "acc": 0, "total": 0})
                dc["total"] += 1
                if dnf:
                    st_lo = status.lower()
                    is_acc = any(kw in st_lo for kw in
                                 ["accident", "collision", "spin", "damage", "retired",
                                  "debris"])
                    dc["acc" if is_acc else "mech"] += 1

                if not dnf:
                    if 6 <= grid <= 15:   # midfield start → overtaking ability
                        overtaking_data.setdefault(code, []).append(float(grid - pos))
                    if is_wet and code in season_avg:
                        wet_data.setdefault(code, []).append(
                            float(pos - season_avg[code]))

        # ── Qualifying — teammate consistency ─────────────────────────────
        print(f"   📥  {year}: clasificación (consistencia vs compañero)...")
        for rnd in range(1, n_rounds + 1):
            data = api_get(f"{JOLPICA_BASE}/{year}/{rnd}/qualifying.json")
            time.sleep(REQ_DELAY)
            if not data:
                continue
            q_races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not q_races:
                continue
            by_team: dict = {}
            for res in q_races[0].get("QualifyingResults", []):
                drv  = res.get("Driver", {})
                code = drv.get("code", drv.get("driverId", "???").upper()[:3])
                team = res.get("Constructor", {}).get("constructorId", "?")
                try:
                    qpos = int(res.get("position", 20))
                except (ValueError, TypeError):
                    continue
                by_team.setdefault(team, []).append((code, qpos))
            for drivers in by_team.values():
                if len(drivers) == 2:
                    (c1, p1), (c2, p2) = drivers[0], drivers[1]
                    quali_deltas.setdefault(c1, []).append(float(p1 - p2))
                    quali_deltas.setdefault(c2, []).append(float(p2 - p1))

        # ── Laps — tyre management (position at 80% vs final) ────────────
        print(f"   📥  {year}: vueltas (tyre management)...")
        for rnd, results in year_results.items():
            # Build driverId → code map from race results
            id_to_code = {
                res.get("Driver", {}).get("driverId", ""): res.get("Driver", {}).get(
                    "code", res.get("Driver", {}).get("driverId", "???").upper()[:3])
                for res in results
            }

            data = api_get(f"{JOLPICA_BASE}/{year}/{rnd}/laps.json",
                           params={"limit": 100})
            time.sleep(REQ_DELAY)
            if not data:
                continue
            l_races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not l_races:
                continue
            laps_list = l_races[0].get("Laps", [])
            if not laps_list:
                continue

            total_laps = max(int(l["number"]) for l in laps_list)
            lap_80     = round(0.80 * total_laps)
            pos_at_80: dict  = {}
            pos_final: dict  = {}

            for lap_entry in laps_list:
                lap_num = int(lap_entry["number"])
                for t in lap_entry.get("Timings", []):
                    did = t.get("driverId", "")
                    try:
                        lp = int(t.get("position", 99))
                    except (ValueError, TypeError):
                        continue
                    if lap_num == lap_80:
                        pos_at_80[did] = lp
                    if lap_num == total_laps:
                        pos_final[did] = lp

            for did, p80 in pos_at_80.items():
                if did in pos_final and did in id_to_code:
                    code = id_to_code[did]
                    # positive = gained positions in final 20% of laps
                    tyre_data.setdefault(code, []).append(float(p80 - pos_final[did]))

    # ── Aggregate into per-driver profile ────────────────────────────────
    profiles: dict = {}
    all_codes = (set(overtaking_data) | set(quali_deltas) | set(wet_data)
                 | set(dnf_counts) | set(tyre_data))

    for code in all_codes:
        ot = overtaking_data.get(code, [])
        qd = quali_deltas.get(code, [])
        ww = wet_data.get(code, [])
        dc = dnf_counts.get(code, {"mech": 0, "acc": 0, "total": 0})
        tm = tyre_data.get(code, [])
        n_tot = dc.get("total", 0)

        profiles[code] = {
            "overtaking_ability"    : round(float(np.mean(ot)), 3) if len(ot) >= 3 else None,
            "quali_consistency"     : round(float(np.std(qd)),  3) if len(qd) >= 5 else None,
            "wet_weather_delta"     : round(float(np.mean(ww)), 3) if len(ww) >= 2 else None,
            "historical_dnf_rate"   : round((dc["mech"] + dc["acc"]) / n_tot, 3)
                                       if n_tot > 0 else None,
            "mech_dnf_rate"         : round(dc["mech"] / n_tot, 3) if n_tot > 0 else None,
            "acc_dnf_rate"          : round(dc["acc"]  / n_tot, 3) if n_tot > 0 else None,
            "tyre_management_index" : round(float(np.mean(tm)), 3) if len(tm) >= 5 else None,
            "n_races"               : n_tot,
            "n_wet"                 : len(ww),
            "n_quali"               : len(qd),
            "n_overtaking"          : len(ot),
        }

    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"   💾  {len(profiles)} perfiles guardados → {PROFILES_FILE}")
    return profiles


# ─────────────────────────────────────────────────────────────
#  18. CONSTRUCCIÓN DE FEATURES
# ─────────────────────────────────────────────────────────────
def fetch_press_conference_sentiment(
    next_round: int,
    next_circuit: str,
    schedule: pd.DataFrame,
) -> dict:
    """
    Downloads the FIA/F1 Thursday or Friday press conference transcript for
    *next_round* and uses claude-haiku-4-5-20251001 to extract a per-driver
    confidence score in [-1.0, +1.0].

    Returns {driver_code: float}. Caches by round in SENTIMENT_FILE.
    Falls back to {} at every error point.
    """
    # ── 1. Cache check ────────────────────────────────────────────────────
    _cache: dict = {}
    if os.path.exists(SENTIMENT_FILE):
        try:
            with open(SENTIMENT_FILE) as _f:
                _cache = json.load(_f)
            if str(next_round) in _cache:
                print(f"   📋  Sentimiento R{next_round}: cargado desde caché "
                      f"({len(_cache[str(next_round)])} pilotos)")
                return _cache[str(next_round)]
        except Exception:
            _cache = {}

    # ── 2. Build candidate URLs ───────────────────────────────────────────
    _row = schedule[schedule["round"] == next_round]
    _race_name = (_row.iloc[0].get("name", next_circuit)
                  if not _row.empty else next_circuit)
    # "Austrian Grand Prix" → "austrian"
    _slug_word = (_race_name.lower()
                  .replace("grand prix", "").replace("gp", "")
                  .strip().replace(" ", "-").rstrip("-"))
    _year = SEASON  # module-level constant

    _candidates = [
        # formula1.com Thursday
        f"https://www.formula1.com/en/latest/article/{_year}-{_slug_word}-grand-prix-thursday-press-conference",
        # formula1.com Friday
        f"https://www.formula1.com/en/latest/article/{_year}-{_slug_word}-grand-prix-friday-press-conference",
        # FIA Thursday
        f"https://www.fia.com/news/{_year}-{_slug_word}-grand-prix-thursday-press-conference",
        # FIA Friday
        f"https://www.fia.com/news/{_year}-{_slug_word}-grand-prix-friday-press-conference",
    ]

    _headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    # ── 3. Fetch transcript text ───────────────────────────────────────────
    import re as _re
    _transcript = ""

    def _strip_html(html: str) -> str:
        text = _re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=_re.S)
        text = _re.sub(r"<style[^>]*>.*?</style>",  " ", text,  flags=_re.S)
        text = _re.sub(r"<[^>]+>", " ", text)
        text = _re.sub(r"\s+", " ", text).strip()
        return text

    def _looks_like_transcript(text: str) -> bool:
        upper = text.upper()
        return (
            len(text) > 800
            and sum(1 for code in
                    ["VERSTAPPEN", "HAMILTON", "NORRIS", "RUSSELL", "LECLERC",
                     "ANTONELLI", "PIASTRI", "HADJAR"]
                    if code in upper) >= 2
        )

    for _url in _candidates:
        try:
            _resp = requests.get(_url, headers=_headers, timeout=8, allow_redirects=True)
            if _resp.ok:
                _text = _strip_html(_resp.text)
                if _looks_like_transcript(_text):
                    _transcript = _text[:9000]
                    print(f"   🌐  Transcripción encontrada: {_url[:70]}...")
                    break
        except Exception:
            continue

    # ── 3b. DuckDuckGo fallback if no direct hit ──────────────────────────
    if not _transcript:
        try:
            import urllib.parse as _urlp
            _q = f"F1 2026 {_race_name} press conference transcript thursday site:formula1.com OR site:fia.com"
            _ddg = f"https://html.duckduckgo.com/html/?q={_urlp.quote(_q)}"
            _dresp = requests.get(_ddg, headers=_headers, timeout=8)
            if _dresp.ok:
                _links = _re.findall(
                    r'https?://(?:www\.formula1\.com|www\.fia\.com)[^"&\s]+', _dresp.text
                )
                _links = [u for u in _links
                          if "press" in u.lower() or "conference" in u.lower()]
                for _link in _links[:3]:
                    try:
                        _p = requests.get(_link, headers=_headers, timeout=8,
                                          allow_redirects=True)
                        if _p.ok:
                            _t = _strip_html(_p.text)
                            if _looks_like_transcript(_t):
                                _transcript = _t[:9000]
                                print(f"   🌐  Transcripción (DDG): {_link[:70]}...")
                                break
                    except Exception:
                        continue
        except Exception as _e:
            print(f"   ⚠  DDG fallback falló: {_e}")

    if not _transcript:
        print(f"   ℹ️   Sin transcripción para R{next_round} — sentimiento neutral")
        return {}

    # ── 4. claude-haiku sentiment analysis ───────────────────────────────
    if not _ANTHROPIC_KEY:
        print("   ⚠  ANTHROPIC_API_KEY no configurado — sin análisis de sentimiento")
        return {}

    try:
        import anthropic as _ant
        _client = _ant.Anthropic(api_key=_ANTHROPIC_KEY)

        _prompt = (
            "You are analyzing an F1 press conference transcript to score each "
            "driver's confidence level.\n\n"
            f"TRANSCRIPT:\n{_transcript}\n\n"
            "TASK: For every driver who speaks or is quoted, assign a sentiment score:\n"
            " +1.0 = very confident (\"found something special\", \"car was amazing\")\n"
            "  0.0 = neutral / factual\n"
            " -1.0 = very negative (\"difficult weekend\", \"lot of problems\")\n\n"
            "Return ONLY a JSON object. Keys = standard 3-letter F1 codes "
            "(VER, HAM, NOR, PIA, RUS, LEC, SAI, ANT, HAD, LAW, GAS, ALO, STR, "
            "OCO, BEA, HUL, ALB, BOT, COL, LIN, BOR, PER). "
            "Values = float in [-1.0, 1.0]. "
            "Only include drivers with actual quotes. Example: "
            "{\"HAM\": 0.7, \"VER\": -0.3}"
        )

        _msg = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": _prompt}],
        )
        _raw = _msg.content[0].text.strip()

        # Extract JSON (may be wrapped in markdown fences)
        _m = _re.search(r"\{[^{}]+\}", _raw, _re.DOTALL)
        if _m:
            _scores: dict = json.loads(_m.group())
            _scores = {
                k: float(max(-1.0, min(1.0, v)))
                for k, v in _scores.items()
                if isinstance(v, (int, float))
            }
            if _scores:
                _cache[str(next_round)] = _scores
                with open(SENTIMENT_FILE, "w") as _f:
                    json.dump(_cache, _f, indent=2)
                print(f"   💾  Sentimiento R{next_round}: {len(_scores)} pilotos "
                      f"→ {SENTIMENT_FILE}")
                return _scores

    except Exception as _e:
        print(f"   ⚠  Análisis de sentimiento falló: {_e}")

    return {}


def detect_pace_anomalies(feat: pd.DataFrame,
                           is_sprint_weekend: bool = False) -> pd.DataFrame:
    """
    IsolationForest pace anomaly detector with corrected F1 logic.

    F1 session pairing key:
      fp_next_delta    = FP1/FP3 qualifying-simulation pace → comparable to quali_time_next
      fp2_next_longrun = FP2 long-run race pace (heavy fuel) → comparable to grid position

    Sprint weekend override (is_sprint_weekend=True):
      FP3 does not exist → skip FP3 vs GP Qualifying comparison.
      Instead compare Sprint Qualifying position vs GP Qualifying position:
        sprint_quali_delta > 0 → driver improved SQ→GP (was hiding pace in SQ)
        sprint_quali_delta < 0 → driver regressed SQ→GP (genuine struggle in GP quali)
      Race threat detection unchanged: FP1 long-run (promoted to fp2_next_longrun) vs grid slot.

    Returns per driver:
      anomaly_score    — positive = normal, negative = anomalous (IsolationForest)
      sandbagging_flag — hid pace in practice vs actual qualifying
      struggling_flag  — underperformed vs practice pace signal
      race_threat_flag — race pace rank much better than grid slot (will charge)

    Called BEFORE the NaN fill so raw NaN values identify drivers without real session data.
    """
    _FEATS = ["fp_next_delta", "quali_time_next", "fp2_next_longrun", "sector_balance"]

    _result = pd.DataFrame({
        "code":             feat["code"].values,
        "anomaly_score":    np.zeros(len(feat)),
        "sandbagging_flag": np.zeros(len(feat), dtype=bool),
        "struggling_flag":  np.zeros(len(feat), dtype=bool),
        "race_threat_flag": np.zeros(len(feat), dtype=bool),
    }, index=feat.index)

    # Need ≥2 feature columns that each have ≥5 real values
    _avail = [c for c in _FEATS
              if c in feat.columns and feat[c].notna().sum() >= 5]
    if len(_avail) < 2:
        _have = sum(1 for c in _FEATS if c in feat.columns and feat[c].notna().any())
        print(f"   🔍  IsolationForest: sin datos suficientes ({_have}/{len(_FEATS)} features) — neutral")
        return _result

    _min_ok   = max(2, len(_avail) // 2)
    _row_mask = feat[_avail].notna().sum(axis=1) >= _min_ok
    if _row_mask.sum() < 5:
        print(f"   🔍  IsolationForest: solo {_row_mask.sum()} pilotos con datos — neutral")
        return _result

    _sub = feat.loc[_row_mask, _avail].copy()
    for _c in _avail:
        _sub[_c] = _sub[_c].fillna(_sub[_c].median())

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest

        _X      = StandardScaler().fit_transform(_sub.values)
        _iso    = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
        _iso.fit(_X)
        _scores = _iso.decision_function(_X)

        _row_idx = feat.index[_row_mask]
        _result.loc[_row_idx, "anomaly_score"] = _scores

    except Exception as _e:
        print(f"   ⚠  IsolationForest falló: {_e}")
        return _result

    # ── Sandbagging / struggling ─────────────────────────────────────────
    _has_fp   = ("fp_next_delta"       in feat.columns and feat["fp_next_delta"].notna().any())
    _has_qt   = ("quali_time_next"     in feat.columns and feat["quali_time_next"].notna().any())
    _has_qpos = ("quali_pos_next"      in feat.columns and feat["quali_pos_next"].notna().any())
    _has_fp2  = ("fp2_next_longrun"    in feat.columns and feat["fp2_next_longrun"].notna().any())
    _has_sqdelta = ("sprint_quali_delta" in feat.columns and
                    feat["sprint_quali_delta"].abs().sum() > 0)

    _is_anom = (_result["anomaly_score"].values < 0)

    if is_sprint_weekend and _has_sqdelta:
        # Sprint weekend: FP3 doesn't exist — compare Sprint Qualifying vs GP Qualifying.
        # sprint_quali_delta = sq_pos - gp_pos (positive = improved from SQ to GP = hid pace)
        _sq_delta = feat["sprint_quali_delta"].values
        _sq_real  = feat["sprint_quali_delta"].abs().gt(0).values  # non-zero means both SQ+Q exist
        _result["sandbagging_flag"] = _is_anom & _sq_real & (_sq_delta >  3)
        _result["struggling_flag"]  = _is_anom & _sq_real & (_sq_delta < -3)
    elif _has_fp and _has_qt:
        # Normal weekend: compare FP3/FP1 qualifying simulation pace vs actual GP qualifying.
        # fp_next_delta rank ascending: rank 1 = fastest qualifying sim (most-negative delta)
        # quali_time_next rank ascending: rank 1 = fastest qualifying time
        _fp_rank    = feat["fp_next_delta"].rank(ascending=True, na_option="bottom")
        _qt_rank    = feat["quali_time_next"].rank(ascending=True, na_option="bottom")
        # positive diff → FP3 rank worse than quali rank → driver hid pace (sandbagging)
        # negative diff → FP3 rank better than quali rank → driver underdelivered (struggling)
        _fp_qt_diff = (_fp_rank - _qt_rank).values

        _fp_real  = feat["fp_next_delta"].notna().values
        _qt_real  = feat["quali_time_next"].notna().values

        _result["sandbagging_flag"] = _is_anom & _fp_real & _qt_real & (_fp_qt_diff >  3)
        _result["struggling_flag"]  = _is_anom & _fp_real & _qt_real & (_fp_qt_diff < -3)

    # ── Race threat: FP2 race pace vs qualifying grid position ───────────
    # fp2_next_longrun = FP2 long-run race pace (heavy fuel — race simulation)
    # quali_pos_next   = qualifying grid position
    # If FP2 race pace rank is much better than grid slot → driver will charge during race
    # No anomaly flag required — this is a pure pace-vs-position gap signal
    if _has_fp2 and _has_qpos:
        _fp2_rank  = feat["fp2_next_longrun"].rank(ascending=True, na_option="bottom")
        _qpos      = feat["quali_pos_next"]
        # negative gap → fp2 pace rank better than grid slot → race threat
        _race_gap  = (_fp2_rank - _qpos).values

        _fp2_real  = feat["fp2_next_longrun"].notna().values
        _qpos_real = feat["quali_pos_next"].notna().values

        _result["race_threat_flag"] = _fp2_real & _qpos_real & (_race_gap < -3)

    n_sand  = _result["sandbagging_flag"].sum()
    n_stru  = _result["struggling_flag"].sum()
    n_race  = _result["race_threat_flag"].sum()
    n_anom  = (_result["anomaly_score"] < 0).sum()
    if n_anom or n_race:
        print(f"   🔍  IsolationForest: {n_anom} anomalías "
              f"({n_sand} sandbagging, {n_stru} struggling, {n_race} race threats)")

    return _result


def build_features(driver_standings, constructor_pts, race_df, quali_df,
                   sprint_df, sq_df, lap_std_df, fp_df, sector_df,
                   tyre_deg_df, lap1_df, dnf_df, teammate_df,
                   pitstop_df, penalty_df, sc_df,
                   compound_df=None, circuit_df=None,
                   circuit_type_df=None, quali_gap_df=None,
                   overtaking_difficulty=0.55,
                   next_quali_df=None, next_fp_df=None,
                   circuit_affinity_df=None,
                   behavioral_df=None,
                   sentiment_df=None,
                   next_circuit_laps: int = 57,
                   next_circuit_type: str = "mixed",
                   sprint_quali_df=None,
                   is_sprint_weekend: bool = False) -> pd.DataFrame:
    compound_df          = compound_df          if compound_df          is not None else pd.DataFrame()
    circuit_df           = circuit_df           if circuit_df           is not None else pd.DataFrame()
    circuit_type_df      = circuit_type_df      if circuit_type_df      is not None else pd.DataFrame()
    quali_gap_df         = quali_gap_df         if quali_gap_df         is not None else pd.DataFrame()
    next_quali_df        = next_quali_df        if next_quali_df        is not None else pd.DataFrame()
    next_fp_df           = next_fp_df           if next_fp_df           is not None else pd.DataFrame()
    circuit_affinity_df  = circuit_affinity_df  if circuit_affinity_df  is not None else pd.DataFrame()
    sprint_quali_df      = sprint_quali_df      if sprint_quali_df      is not None else pd.DataFrame()
    """Combina todas las fuentes en un DataFrame de features por piloto."""

    feat = driver_standings[["code", "FullName", "TeamName", "champ_pts"]].copy()
    n    = len(feat)

    # ── Resultados de carrera ──────────────────────────────────────────
    if not race_df.empty:
        agg = race_df.groupby("code").agg(
            avg_grid  =("grid", "mean"),
            fl_rate   =("fastest_lap", "mean"),
        ).reset_index()
        # EWM finish: cap DNF positions at P12 (mechanical retirement ≠ genuine P16+)
        # then use span=5 so one bad race doesn't dominate (vs span=3 which weights last race ~50%)
        _ewm_pos = race_df.copy()
        _ewm_pos.loc[_ewm_pos["dnf"] == 1, "pos"] = \
            _ewm_pos.loc[_ewm_pos["dnf"] == 1, "pos"].clip(upper=12)
        ewm_finish = (_ewm_pos.sort_values("round")
                      .groupby("code")["pos"]
                      .apply(lambda s: s.ewm(span=5).mean().iloc[-1])
                      .reset_index(name="avg_finish"))
        agg = agg.merge(ewm_finish, on="code", how="left")
        feat = feat.merge(agg, on="code", how="left")
        # ── MEJORA 11: Momentum con decaimiento exponencial ──────────────
        # Pesos: última carrera 50%, penúltima 30%, antepenúltima 20%
        # Más carreras disponibles → mejor señal de forma real
        sorted_rounds = sorted(race_df["round"].unique())
        decay_weights = {0: 0.50, 1: 0.30, 2: 0.20}   # índice desde el final
        momentum_rows = []
        for code in race_df["code"].unique():
            drv_results = race_df[race_df["code"] == code].sort_values("round")
            recent      = drv_results.tail(4).reset_index(drop=True)
            n_races     = len(recent)
            if n_races == 0:
                continue
            # Normalizar pesos según cuántas carreras hay disponibles
            if n_races == 1:
                w_map = {0: 1.0}
            elif n_races == 2:
                w_map = {0: 0.40, 1: 0.60}
            elif n_races == 3:
                w_map = {0: 0.20, 1: 0.30, 2: 0.50}
            else:  # 4+ races: last=50%, -2=30%, -3=15%, -4=5%
                w_map = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.50}
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
        # ── Constructor momentum: rolling 3-race win rate per team ──────────
        _team_map    = dict(zip(driver_standings["code"], driver_standings["TeamName"]))
        _last3_set   = set(sorted(race_df["round"].unique())[-3:])
        _wins_r      = race_df[race_df["pos"] == 1].copy()
        _wins_r["team"] = _wins_r["code"].map(_team_map)
        _wins_last3  = _wins_r[_wins_r["round"].isin(_last3_set)].groupby("team").size()
        cm_rows = [{"code": c,
                    "constructor_momentum": _wins_last3.get(_team_map.get(c, ""), 0) / 3.0}
                   for c in feat["code"]]
        feat = feat.merge(pd.DataFrame(cm_rows), on="code", how="left")
        feat["constructor_momentum"] = feat["constructor_momentum"].fillna(0.0)
        # ── Podium streak score ────────────────────────────────────────
        streak_rows = []
        for code in feat["code"]:
            drv = race_df[race_df["code"] == code].sort_values("round", ascending=False)
            streak = 0
            for _, row in drv.iterrows():
                if row.get("dnf", 0) == 1 or row.get("pos", 99) > 3:
                    break
                streak += 1
            streak_rows.append({"code": code, "streak_score": min(streak * 0.15, 1.0)})
        feat = feat.merge(pd.DataFrame(streak_rows), on="code", how="left")
        feat["streak_score"] = feat["streak_score"].fillna(0.0)
        _active_streaks = feat[feat["streak_score"] > 0][["code", "streak_score"]].sort_values(
            "streak_score", ascending=False)
        if not _active_streaks.empty:
            print("   🔥  Rachas de podio activas:")
            for _, r in _active_streaks.iterrows():
                n_consec = round(r["streak_score"] / 0.15)
                print(f"      {r['code']:<5}  {n_consec} carreras top-3 consecutivas (score={r['streak_score']:.2f})")
        # ── Post-DNF bounce indicator ──────────────────────────────────
        _last_round_all = race_df["round"].max()
        _last_race_df   = race_df[race_df["round"] == _last_round_all]
        _dnf_codes      = set(_last_race_df[_last_race_df["dnf"] == 1]["code"].tolist())
        feat["post_dnf_bounce"] = feat["code"].apply(lambda c: 1.0 if c in _dnf_codes else 0.0)
        if _dnf_codes:
            print(f"   💥  Post-DNF bounce (última carrera): {', '.join(sorted(_dnf_codes))}")
        # ── Championship pressure coefficient ─────────────────────────
        _n_completed     = race_df["round"].nunique()
        _n_remaining     = max(22 - _n_completed, 0)
        _total_avail_pts = max(_n_remaining * 26, 1)
        _leader_pts      = feat["champ_pts"].max()
        feat["championship_pressure"] = feat["champ_pts"].apply(
            lambda pts: max(0.0, 1.0 - max(0.0, _leader_pts - pts) / _total_avail_pts) * 0.5
        )
        # ── Circuit affinity: avg finish at next circuit over last 3 seasons ──
        if not circuit_affinity_df.empty and "circuit_affinity" in circuit_affinity_df.columns:
            feat = feat.merge(circuit_affinity_df, on="code", how="left")
            field_med = feat["circuit_affinity"].median()
            feat["circuit_affinity"] = feat["circuit_affinity"].fillna(
                field_med if not np.isnan(field_med) else 11.0)
            top5 = feat[["code", "circuit_affinity"]].sort_values("circuit_affinity").head(5)
            print("   🏎  Affinity top 5 (menor pos. media = más afinidad al circuito):")
            for _, r in top5.iterrows():
                print(f"      {r['code']:<5}  {r['circuit_affinity']:.2f}")
        else:
            feat["circuit_affinity"] = 11.0  # neutral midfield fallback
    else:
        for col in ["avg_finish", "avg_grid", "fl_rate", "recent_form", "constructor_momentum",
                    "circuit_affinity"]:
            feat[col] = np.nan
        feat["streak_score"]    = 0.0
        feat["post_dnf_bounce"] = 0.0
        _leader_pts_e = feat["champ_pts"].max()
        feat["championship_pressure"] = feat["champ_pts"].apply(
            lambda pts: max(0.0, 1.0 - max(0.0, _leader_pts_e - pts) / (22 * 26)) * 0.5
        )

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

    # ── Sprint Qualifying data (sprint weekends only) ──────────────────────
    # sq_pos_next / sq_time_next: performance in Friday's Sprint Qualifying
    # sprint_quali_delta = sq_pos - gp_quali_pos:
    #   positive → driver improved from SQ to GP quali (hid pace / SQ setback)
    #   negative → driver regressed from SQ to GP quali (genuine pace step-back)
    if not sprint_quali_df.empty and "sq_pos_next" in sprint_quali_df.columns:
        feat = feat.merge(sprint_quali_df[["code", "sq_pos_next", "sq_time_next"]],
                          on="code", how="left")
        if "quali_pos_next" in feat.columns and feat["quali_pos_next"].notna().any():
            feat["sprint_quali_delta"] = feat["sq_pos_next"] - feat["quali_pos_next"]
            _sq_avail = feat["sprint_quali_delta"].notna().sum()
            if _sq_avail:
                _sq_top = (feat[["code", "sprint_quali_delta"]]
                           .dropna(subset=["sprint_quali_delta"])
                           .sort_values("sprint_quali_delta", ascending=False))
                print(f"   ⚡  Sprint quali delta ({_sq_avail} pilotos) — "
                      f"top improvers SQ→GP: "
                      + ", ".join(f"{r['code']} (+{r['sprint_quali_delta']:.0f})"
                                  for _, r in _sq_top.head(3).iterrows()
                                  if r["sprint_quali_delta"] > 0))
        else:
            feat["sprint_quali_delta"] = np.nan
    else:
        feat["sq_pos_next"]        = np.nan
        feat["sq_time_next"]       = np.nan
        feat["sprint_quali_delta"] = np.nan
    feat["sprint_quali_delta"] = feat["sprint_quali_delta"].fillna(0.0)

    # ── Sector balance: σ of S1/S2/S3 deltas vs field best in next quali ──
    _s_cols = ["s1_next", "s2_next", "s3_next"]
    if all(c in feat.columns for c in _s_cols) and feat[_s_cols].notna().any().any():
        _s1b = feat["s1_next"].min()
        _s2b = feat["s2_next"].min()
        _s3b = feat["s3_next"].min()
        _d1  = feat["s1_next"] - _s1b
        _d2  = feat["s2_next"] - _s2b
        _d3  = feat["s3_next"] - _s3b
        feat["sector_balance"] = pd.concat(
            [_d1.rename("d1"), _d2.rename("d2"), _d3.rename("d3")], axis=1
        ).std(axis=1)
        _sb = (pd.DataFrame({
                   "code": feat["code"].values,
                   "sector_balance": feat["sector_balance"].values,
                   "ΔS1": _d1.values, "ΔS2": _d2.values, "ΔS3": _d3.values,
               })
               .dropna(subset=["sector_balance"])
               .sort_values("sector_balance"))
        if not _sb.empty:
            print("   ⚡  Top 5 sector balance (σ deltas S1/S2/S3 — menor = más consistente):")
            for _, r in _sb.head(5).iterrows():
                print(f"      {r['code']:<5}  σ={r['sector_balance']:.3f}s  "
                      f"ΔS1:{r['ΔS1']:+.3f}  ΔS2:{r['ΔS2']:+.3f}  ΔS3:{r['ΔS3']:+.3f}")
    else:
        feat["sector_balance"] = np.nan
        print("   ⚠  Sin datos de sector del próximo GP — sector_balance neutral")

    # ── Prácticas del próximo GP (si disponible) ──────────────────────
    _FP_FALLBACK = dict(fp_next_delta=np.nan, fp2_next_longrun=np.nan,
                        soft_pace_delta=np.nan, medium_pace_delta=np.nan,
                        hard_pace_delta=np.nan, compound_preference=np.nan,
                        tyre_deg_rate=np.nan, deg_rate_soft=np.nan,
                        deg_rate_medium=np.nan, deg_rate_hard=np.nan,
                        high_speed_delta=np.nan, medium_speed_delta=np.nan,
                        low_speed_delta=np.nan, corner_balance=np.nan,
                        race_sim_delta=np.nan, race_sim_deg=np.nan)
    feat = feat.merge(next_fp_df, on="code", how="left") \
               if not next_fp_df.empty else feat.assign(**_FP_FALLBACK)
    # Ensure all FP columns exist even when next_fp_df was fetched before FP2
    for _col, _val in _FP_FALLBACK.items():
        if _col not in feat.columns:
            feat[_col] = _val

    # Fill NaN compound deltas with field median (neutral — no compound advantage)
    for _cname in ("soft_pace_delta", "medium_pace_delta", "hard_pace_delta"):
        if feat[_cname].notna().any():
            feat[_cname] = feat[_cname].fillna(feat[_cname].median())

    # Fill NaN tyre_deg_rate with field median (neutral — no deg advantage)
    if feat["tyre_deg_rate"].notna().any():
        feat["tyre_deg_rate"] = feat["tyre_deg_rate"].fillna(
            feat["tyre_deg_rate"].median())

    # Corner profile score — combine speed class deltas with circuit-type weighting
    # Weights per class for each circuit type (must sum to 1.0 across available classes)
    _SPEED_WEIGHTS = {
        "high_speed": {"high": 0.60, "medium": 0.40, "low": 0.00},
        "street":     {"high": 0.10, "medium": 0.50, "low": 0.40},
        "technical":  {"high": 0.20, "medium": 0.50, "low": 0.30},
        "mixed":      {"high": 0.40, "medium": 0.40, "low": 0.20},
    }
    _sw = _SPEED_WEIGHTS.get(next_circuit_type, _SPEED_WEIGHTS["mixed"])
    _col_map = {"high": "high_speed_delta", "medium": "medium_speed_delta", "low": "low_speed_delta"}

    def _corner_score(row):
        avail_w = {cls: w for cls, w in _sw.items()
                   if w > 0 and not np.isnan(row.get(_col_map[cls], np.nan))}
        if not avail_w:
            return np.nan
        total_w = sum(avail_w.values())
        return sum(row[_col_map[cls]] * (w / total_w) for cls, w in avail_w.items())

    feat["corner_profile_score"] = feat.apply(
        lambda r: _corner_score(r.to_dict()), axis=1)
    # Neutral fill for drivers with no speed trap data
    if feat["corner_profile_score"].notna().any():
        feat["corner_profile_score"] = feat["corner_profile_score"].fillna(
            feat["corner_profile_score"].median())

    # Print top-5 when compound data is available
    for _cname, _label in [("soft_pace_delta", "SOFT"), ("medium_pace_delta", "MEDIUM")]:
        if feat[_cname].notna().any() and feat[_cname].nunique() > 1:
            _top5 = (feat[["code", _cname]].dropna()
                       .sort_values(_cname).head(5))
            print(f"   🔴  Top 5 ritmo {_label} vs campo (negativo = más rápido):")
            for _, r in _top5.iterrows():
                print(f"      {r['code']:<5}  {r[_cname]:+.3f}%")

    # Print top-5 tyre deg rates when FP2 long-run data is available
    if feat["tyre_deg_rate"].notna().any() and feat["tyre_deg_rate"].nunique() > 1:
        _top5_deg = (feat[["code", "tyre_deg_rate"]].dropna()
                       .sort_values("tyre_deg_rate").head(5))
        print("   🏎  Top 5 gestión de neumáticos FP2 (s/vuelta, menor = mejor):")
        for _, r in _top5_deg.iterrows():
            print(f"      {r['code']:<5}  {r['tyre_deg_rate']:+.4f} s/lap")

    # Print top-5 corner profile scores when speed trap data is available
    if feat["corner_profile_score"].notna().any() and feat["corner_profile_score"].nunique() > 1:
        _top5_cps = (feat[["code", "corner_profile_score"]].dropna()
                       .sort_values("corner_profile_score").head(5))
        print(f"   🔷  Top 5 perfil de curvas ({next_circuit_type}, negativo = más rápido):")
        for _, r in _top5_cps.iterrows():
            print(f"      {r['code']:<5}  {r['corner_profile_score']:+.3f}%")

    # Fill NaN race_sim_delta with field median (neutral when no FP1 race sim detected)
    if feat["race_sim_delta"].notna().any():
        feat["race_sim_delta"] = feat["race_sim_delta"].fillna(
            feat["race_sim_delta"].median())

    # Print top-5 race sim delta when FP1 data is available
    if feat["race_sim_delta"].notna().any() and feat["race_sim_delta"].nunique() > 1:
        _top5_rs = (feat[["code", "race_sim_delta"]].dropna()
                      .sort_values("race_sim_delta").head(5))
        print("   🏁  Top 5 ritmo carrera FP1 (negativo = más rápido vs campo):")
        for _, r in _top5_rs.iterrows():
            print(f"      {r['code']:<5}  {r['race_sim_delta']:+.3f}%")

    # ── Circuit type score (head-to-head por tipo de circuito) ────────
    feat = feat.merge(circuit_type_df, on="code", how="left") \
               if not circuit_type_df.empty else feat.assign(circuit_type_score=np.nan)

    # ── Gap a compañero en qualifying (décimas) ───────────────────────
    feat = feat.merge(quali_gap_df, on="code", how="left") \
               if not quali_gap_df.empty else feat.assign(quali_gap_teammate=np.nan)

    # ── fp2_longrun_delta (si fp_df lo tiene) ────────────────────────
    if "fp2_longrun_delta" not in feat.columns:
        feat["fp2_longrun_delta"] = np.nan

    # ── Perfiles comportamentales históricos (2023-2025) ──────────────
    _b_cols = ["overtaking_ability", "quali_consistency", "wet_weather_delta",
               "historical_dnf_rate", "tyre_management_index"]
    if behavioral_df is not None and not behavioral_df.empty:
        _avail = [c for c in _b_cols if c in behavioral_df.columns]
        if _avail:
            feat = feat.merge(behavioral_df[["code"] + _avail], on="code", how="left")
            _n_matched = feat[_avail[0]].notna().sum()
            print(f"   🧬  Perfiles comportamentales: {_n_matched}/{len(feat)} pilotos "
                  f"con datos 2023-2025")
    else:
        for col in _b_cols:
            feat[col] = np.nan

    # ── Press conference sentiment (NLP via claude-haiku) ─────────────────
    if sentiment_df is not None and not sentiment_df.empty and "press_sentiment" in sentiment_df.columns:
        feat = feat.merge(sentiment_df[["code", "press_sentiment"]], on="code", how="left")
        feat["press_sentiment"] = feat["press_sentiment"].fillna(0.0)
        _n_sent = (feat["press_sentiment"] != 0.0).sum()
        print(f"   🎙️  Sentimiento de prensa: {_n_sent}/{len(feat)} pilotos con datos")
    else:
        feat["press_sentiment"] = 0.0

    # ── Driver-circuit style compatibility (3D embedding cosine similarity) ──
    _emb_req = ["overtaking_ability", "quali_consistency",
                "tyre_management_index", "historical_dnf_rate"]
    if (behavioral_df is not None and not behavioral_df.empty and
            all(c in behavioral_df.columns for c in _emb_req)):
        # Normalization ranges from full behavioral pool (all historical drivers)
        _oa_pool = behavioral_df["overtaking_ability"].dropna()
        _qc_pool = behavioral_df["quali_consistency"].dropna()
        _tm_pool = behavioral_df["tyre_management_index"].dropna()
        _dr_pool = behavioral_df["historical_dnf_rate"].dropna()

        def _n01s(s, pool):
            lo, hi = pool.min(), pool.max()
            if hi <= lo:
                return pd.Series(0.5, index=s.index)
            return ((s - lo) / (hi - lo)).clip(0, 1).fillna(0.5)

        # 3D driver embeddings per driver
        _agg  = _n01s(feat["overtaking_ability"],    _oa_pool)          # aggression
        _con  = 1.0 - _n01s(feat["quali_consistency"], _qc_pool)        # consistency (inverted)
        _tm_n = _n01s(feat["tyre_management_index"], _tm_pool)
        _dr_n = 1.0 - _n01s(feat["historical_dnf_rate"], _dr_pool)
        _end  = (_tm_n + _dr_n) / 2.0                                   # endurance

        # 3D circuit embedding
        _ctype_enc = {"street": 1.0, "technical": 0.8, "mixed": 0.5, "high_speed": 0.2}
        _lap_lo    = min(CIRCUIT_RACE_LAPS.values())
        _lap_hi    = max(CIRCUIT_RACE_LAPS.values())
        _circ_end  = float(np.clip((next_circuit_laps - _lap_lo) / max(_lap_hi - _lap_lo, 1), 0, 1))
        _circ_vec  = np.array([
            1.0 - overtaking_difficulty,
            _ctype_enc.get(next_circuit_type, 0.5),
            _circ_end,
        ])

        # Vectorised cosine similarity: (n_drivers,3) @ (3,) / (|d| * |c|)
        _D       = np.column_stack([_agg.values, _con.values, _end.values])
        _d_norms = np.linalg.norm(_D, axis=1)
        _c_norm  = float(np.linalg.norm(_circ_vec))
        _dots    = _D @ _circ_vec
        _denom   = _d_norms * _c_norm
        _compat  = np.where(_denom > 1e-9, _dots / _denom, 0.0)

        feat["compatibility_score"] = np.round(_compat, 4)

        # Store embeddings so main() can print/save them
        feat.attrs["driver_embeddings"] = {
            code: [round(float(_agg.iloc[i]), 4),
                   round(float(_con.iloc[i]), 4),
                   round(float(_end.iloc[i]), 4)]
            for i, code in enumerate(feat["code"].values)
        }
        feat.attrs["circuit_embedding"] = [round(float(x), 4) for x in _circ_vec]
    else:
        feat["compatibility_score"] = 0.0

    # ── Overtaking difficulty — guardada como atributo del DataFrame ──
    feat.attrs["overtaking_difficulty"] = overtaking_difficulty

    # ── Constructor pts ───────────────────────────────────────────────
    feat["constructor_pts"] = feat["TeamName"].map(constructor_pts).fillna(0)

    # ── Pace anomaly detection — run BEFORE NaN fill to keep raw NaN pattern ──
    _anomaly_result = detect_pace_anomalies(feat, is_sprint_weekend=is_sprint_weekend)

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

    # ── Merge anomaly results (after NaN fill — keeps bool cols out of fill loop) ──
    feat = feat.merge(
        _anomaly_result[["code", "anomaly_score", "sandbagging_flag",
                         "struggling_flag", "race_threat_flag"]],
        on="code", how="left",
    )
    feat["anomaly_score"]    = feat["anomaly_score"].fillna(0.0)
    feat["sandbagging_flag"] = feat["sandbagging_flag"].fillna(False)
    feat["struggling_flag"]  = feat["struggling_flag"].fillna(False)
    feat["race_threat_flag"] = feat["race_threat_flag"].fillna(False)

    # Print driver pace anomaly flags
    _fp_ranks  = (feat["fp_next_delta"].rank(ascending=True)
                  if "fp_next_delta"    in feat.columns else None)
    _fp2_ranks = (feat["fp2_next_longrun"].rank(ascending=True)
                  if "fp2_next_longrun" in feat.columns else None)
    _flagged = feat[feat["sandbagging_flag"] | feat["struggling_flag"] | feat["race_threat_flag"]]
    if not _flagged.empty:
        print("   🚨  PACE ANOMALY FLAGS:")
    for _, _fr in _flagged.iterrows():
        _qpos   = int(_fr.get("quali_pos_next", 22))
        _fp_est = (int(round(_fp_ranks.loc[_fr.name]))
                   if _fp_ranks is not None else "?")
        if _fr["sandbagging_flag"]:
            print(f"   ⚠  {_fr['code']} — SANDBAGGING: FP3 suggested P{_fp_est}, "
                  f"qualified P{_qpos} (hiding pace)")
        elif _fr["struggling_flag"]:
            print(f"   ⚠  {_fr['code']} — STRUGGLING: FP3 suggested P{_fp_est}, "
                  f"qualified P{_qpos} (setup issue)")
        if _fr["race_threat_flag"] and _fp2_ranks is not None:
            _fp2_r = int(round(_fp2_ranks.loc[_fr.name]))
            _fp2_label = ("top-3" if _fp2_r <= 3 else
                          "top-5" if _fp2_r <= 5 else
                          "top-10" if _fp2_r <= 10 else f"P{_fp2_r}")
            print(f"   🚀  {_fr['code']} — RACE THREAT: FP2 race pace {_fp2_label}, "
                  f"starts P{_qpos} (will charge)")

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
        feat["quali_pos_next"].notna().sum() >= 10 and
        feat["quali_pos_next"].nunique() >= 5        # reject corrupt fallback (e.g. all 22.0)
    )
    if has_next_quali:
        print("   ⭐  Clasificación del próximo GP disponible → peso dominante 55%")

    w = {
        # ── DATOS DEL PRÓXIMO GP (los más relevantes) ──────────────────
        "quali_pos_next"   : 0.55 if has_next_quali else 0.00,
        "fp_next_delta"    : 0.08 if has_next_quali else 0.00,
        "fp2_next_longrun"  : 0.05 if has_next_quali else 0.00,
        "sector_balance"    : 0.03 if has_next_quali else 0.00,
        "soft_pace_delta"      : 0.02 if has_next_quali else 0.00,
        "medium_pace_delta"    : 0.02 if has_next_quali else 0.00,
        "tyre_deg_rate"        : 0.03 if has_next_quali else 0.00,
        "corner_profile_score" : 0.02 if has_next_quali else 0.00,
        "race_sim_delta"       : 0.03 if has_next_quali else 0.00,
        # ── HISTORIAL DE TEMPORADA ──────────────────────────────────────
        "quali_pos"        : 0.05 if has_next_quali else quali_w_dynamic,
        "quali_gap_teammate": 0.04 if has_next_quali else 0.06,
        "avg_grid"         : 0.03 if has_next_quali else 0.06,
        "champ_pts"        : 0.10 if has_next_quali else 0.11,
        "avg_finish"       : 0.04 if has_next_quali else finish_w,
        "constructor_pts"      : 0.05 if has_next_quali else 0.08,
        "constructor_momentum" : 0.05 if has_next_quali else 0.05,
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
        "circuit_affinity"  : 0.04 if has_next_quali else 0.06,
        "lap1_gain"        : 0.01 if has_next_quali else 0.02,
        "teammate_delta"   : 0.01,
        "avg_pitstop"      : 0.01 if has_next_quali else 0.02,
        "sc_gain_avg"      : 0.01,
        "avg_sector_delta" : 0.01 if has_next_quali else 0.02,
        "streak_score"          : 0.03,
        "post_dnf_bounce"       : 0.02,
        "championship_pressure" : 0.02,
        # ── Perfiles históricos 2023-2025 ───────────────────────────────
        "overtaking_ability"    : 0.03,
        "quali_consistency"     : 0.02,
        "wet_weather_delta"     : round(0.05 * feat.attrs.get("rain_prob", 0.0), 4),
        "historical_dnf_rate"   : 0.01,
        "tyre_management_index" : 0.02,
        "compatibility_score"   : 0.04 if has_next_quali else 0.03,
        # ── NLP sentiment — only active when press conf has happened (≈ post-quali) ──
        "press_sentiment"       : 0.03 if has_next_quali else 0.00,
        # ── Sprint weekend: SQ→GP qualifying improvement signal ────────────
        # Only meaningful when both Sprint Quali and GP Quali data are available
        "sprint_quali_delta"    : 0.02 if (has_next_quali and
                                            "sprint_quali_delta" in feat.columns and
                                            feat["sprint_quali_delta"].abs().sum() > 0)
                                       else 0.00,
        # Fresh soft sets remaining vs field avg — strategic flexibility signal.
        # Only meaningful once at least one practice session has completed.
        "tyre_inventory_score"  : 0.02 if has_next_quali else 0.00,
        # Corner mastery from FastF1 telemetry — approximate, 8 circuits only.
        # Active only when data was computed (col non-zero) AND has_next_quali.
        # Weight is intentionally low: telemetry boundaries are ±50–100 m estimates.
        "corner_mastery_score"  : 0.03 if (has_next_quali and
                                            "corner_mastery_score" in feat.columns and
                                            feat["corner_mastery_score"].abs().sum() > 0.01)
                                       else 0.00,
    }

    # When qualifying is available, rescale season-history features so raw dict sums to 1.0
    # This makes quali_pos_next effective weight exactly 55% (auto-norm below becomes a no-op)
    if has_next_quali:
        _next_keys = {"quali_pos_next", "fp_next_delta", "fp2_next_longrun",
                      "sector_balance", "soft_pace_delta", "medium_pace_delta",
                      "tyre_deg_rate", "corner_profile_score", "race_sim_delta",
                      "tyre_inventory_score", "corner_mastery_score"}
        season_sum = sum(v for k, v in w.items() if k not in _next_keys)
        season_target = 1.0 - sum(w[k] for k in _next_keys if k in w)
        if season_sum > 0:
            _sf = season_target / season_sum
            for k in list(w.keys()):
                if k not in _next_keys:
                    w[k] *= _sf

    # Auto-normalizar para que sumen exactamente 1.0
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    # Feedback loop: blend XGBoost importancias con pesos manuales
    # _next features are excluded: XGBoost never trained on them, so it has no valid opinion
    _NEXT_RACE_WEIGHTS = {"quali_pos_next", "fp_next_delta", "fp2_next_longrun",
                          "sector_balance", "soft_pace_delta", "medium_pace_delta",
                          "tyre_deg_rate", "corner_profile_score", "race_sim_delta",
                          "tyre_inventory_score", "corner_mastery_score"}
    if xgb_weights:
        print("   🔄  Aplicando feedback XGBoost a pesos manuales...")
        total_xgb = sum(xgb_weights.values()) or 1
        for k in list(w.keys()):
            if k in xgb_weights and k not in _NEXT_RACE_WEIGHTS:
                w[k] = 0.6 * (xgb_weights[k] / total_xgb) + 0.4 * w[k]
        # Preserve next-race feature weights (qualifying, FP) — XGB inflates season weights
        # so a full re-normalization would dilute the 55% qualifying weight to ~39%.
        # Instead, rescale only season features so next-race weights stay at their intended sum.
        _next_total  = sum(w[k] for k in _NEXT_RACE_WEIGHTS if k in w)
        _season_sum  = sum(v for k, v in w.items() if k not in _NEXT_RACE_WEIGHTS)
        _season_tgt  = 1.0 - _next_total
        if _season_sum > 0 and _season_tgt > 0:
            _sf = _season_tgt / _season_sum
            for k in list(w.keys()):
                if k not in _NEXT_RACE_WEIGHTS:
                    w[k] *= _sf

    def apply(col, direction):
        if col in feat.columns and feat[col].notna().any():
            fn = rank_asc if direction == "asc" else rank_desc
            return fn(col) * w.get(col, 0)
        return pd.Series(0.0, index=feat.index)

    s += apply("quali_pos_next",      "asc")   # ★★ clasificación del próximo GP
    s += apply("fp_next_delta",       "asc")   # ★★ ritmo FP del próximo GP
    s += apply("fp2_next_longrun",    "asc")   # ★★ long run FP2 próximo GP
    s += apply("sector_balance",      "asc")   # ★  consistencia S1/S2/S3 (menor σ = mejor)
    s += apply("soft_pace_delta",     "asc")   # ★  ritmo SOFT vs campo en long run FP2
    s += apply("medium_pace_delta",   "asc")   # ★  ritmo MEDIUM vs campo en long run FP2
    s += apply("tyre_deg_rate",       "asc")   # ★  degradación neumático FP2 (s/vuelta, menor = mejor)
    s += apply("corner_profile_score","asc")   # ★  perfil de curva: negativo = más rápido en clase dominante del circuito
    s += apply("race_sim_delta",      "asc")   # ★  ritmo simulación carrera FP1 (negativo = más rápido)
    s += apply("quali_pos",           "asc")
    s += apply("quali_gap_teammate",  "desc")  # mayor gap = más dominante que compañero
    s += apply("avg_grid",            "asc")
    s += apply("champ_pts",           "desc")
    s += apply("avg_finish",          "asc")
    s += apply("constructor_pts",          "desc")
    s += apply("constructor_momentum",     "desc")  # wins in last 3 races (per team)
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
    s += apply("circuit_affinity",    "asc")   # lower avg finish = stronger affinity
    s += apply("lap1_gain",           "desc")
    s += apply("teammate_delta",      "desc")
    s += apply("avg_pitstop",         "asc")
    s += apply("sc_gain_avg",         "desc")
    s += apply("avg_sector_delta",    "asc")
    s += apply("streak_score",          "desc")
    s += apply("post_dnf_bounce",       "desc")
    s += apply("championship_pressure", "desc")
    s += apply("overtaking_ability",    "desc")
    s += apply("quali_consistency",     "asc")   # lower std = more consistent
    s += apply("wet_weather_delta",     "asc")   # negative = better in wet vs dry avg
    s += apply("tyre_management_index", "desc")
    s += apply("compatibility_score",   "desc")  # driver-circuit style match
    s += apply("press_sentiment",       "desc")  # pre-race confidence from press conf NLP
    s += apply("sprint_quali_delta",   "desc")  # sprint wknd: improved SQ→GP = latent pace
    s += apply("tyre_inventory_score", "desc")  # more fresh softs remaining = strategic advantage
    s += apply("corner_mastery_score", "desc")  # higher ratio vs session best = better corners

    s -= feat.get("dnf_rate",      pd.Series(0.0, index=feat.index)).fillna(0) * 0.10
    s -= feat.get("historical_dnf_rate",
                  pd.Series(0.0, index=feat.index)).fillna(0) * 0.03
    s -= (feat.get("penalty_count", pd.Series(0.0, index=feat.index)).fillna(0)
          / (feat.get("penalty_count", pd.Series(1.0, index=feat.index)).max() + 1)) * 0.05
    return s


# ─────────────────────────────────────────────────────────────
#  20. MODELOS XGBoost / LightGBM — arquitectura Quali + Race
# ─────────────────────────────────────────────────────────────

# Qualifying model inputs — features correlated with grid position
_QUALI_FEAT_COLS = [
    "champ_pts", "avg_grid", "constructor_pts", "recent_form",
    "fp_avg_delta", "teammate_delta",
]

# Race model inputs — all season features + predicted qualifying position
_RACE_FEAT_COLS = [
    "champ_pts", "avg_finish", "avg_grid", "fl_rate", "lap_std",
    "constructor_pts", "recent_form", "momentum_pos", "sprint_pts",
    "fp_avg_delta",
    "avg_sector_delta", "tyre_deg_slope", "lap1_gain", "teammate_delta",
    "avg_pitstop", "sc_gain_avg", "dnf_rate", "penalty_count",
    "overtaking_ability", "quali_consistency", "tyre_management_index",
]


def _round_weights(race_df: pd.DataFrame):
    """(sorted_rounds, {round: recency_rank}) for exponential sample weighting."""
    rnds = sorted(race_df["round"].unique())
    return rnds, {rnd: i for i, rnd in enumerate(rnds)}


def _build_quali_Xyw(feat, quali_df, feat_cols, round_rank):
    """Build (X, y, w) for qualifying-position prediction. Returns None if < 10 rows."""
    rows_X, rows_y, rows_w = [], [], []
    for rnd in sorted(quali_df["round"].unique()):
        q_rnd = quali_df[quali_df["round"] == rnd][["code", "quali_pos"]].dropna()
        rnd_w = 1.5 ** round_rank.get(int(rnd), 0)
        for _, r in q_rnd.iterrows():
            drv = feat[feat["code"] == r["code"]][feat_cols]
            if not drv.empty:
                rows_X.append(drv.iloc[0].values)
                rows_y.append(float(r["quali_pos"]))
                rows_w.append(rnd_w)
    if len(rows_X) < 10:
        return None
    return np.array(rows_X), np.array(rows_y), np.array(rows_w)


def _build_race_Xyw(feat, race_df, quali_df, feat_cols, round_rank):
    """
    Build (X, y, w, col_names) for race-position prediction.
    Appends the ACTUAL qualifying position for each (round, driver) pair as an
    extra column when quali_df is available, so the race model learns how much
    grid position matters.  Returns None if < 10 rows.
    """
    use_q = (quali_df is not None and not quali_df.empty
             and "round" in quali_df.columns)
    q_lookup = {}
    if use_q:
        for _, r in quali_df.iterrows():
            q_lookup[(int(r["round"]), r["code"])] = float(r["quali_pos"])

    field_qpos = 11.5   # midfield fallback for missing qualifying data
    rows_X, rows_y, rows_w = [], [], []
    for rnd in sorted(race_df["round"].unique()):
        rnd_res = race_df[race_df["round"] == rnd][["code", "pos"]].dropna()
        rnd_w   = 1.5 ** round_rank[rnd]
        for _, r in rnd_res.iterrows():
            drv = feat[feat["code"] == r["code"]][feat_cols]
            if not drv.empty:
                row = drv.iloc[0].values.tolist()
                if use_q:
                    row.append(q_lookup.get((int(rnd), r["code"]), field_qpos))
                rows_X.append(row)
                rows_y.append(float(r["pos"]))
                rows_w.append(rnd_w)

    if len(rows_X) < 10:
        return None
    all_cols = feat_cols + (["predicted_quali_pos"] if use_q else [])
    return np.array(rows_X), np.array(rows_y), np.array(rows_w), all_cols


def score_xgboost(feat: pd.DataFrame, race_df: pd.DataFrame,
                  quali_df: pd.DataFrame = None,
                  warm_start_file: str = None,
                  warm_race_df: pd.DataFrame = None):
    """
    Two-stage XGBoost pipeline:
      Stage 1 — Qualifying model: predicts qualifying grid position from season features.
      Stage 2 — Race model: predicts finishing position using season features +
                            predicted_quali_pos from stage 1.
    Returns (score, importances) or None.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("   ⚠  xgboost no instalado — usando modelo de pesos.")
        return None

    if race_df.empty:
        return None

    n = len(feat)
    _, round_rank  = _round_weights(race_df)
    q_feat_cols    = [c for c in _QUALI_FEAT_COLS if c in feat.columns]
    r_feat_cols    = [c for c in _RACE_FEAT_COLS  if c in feat.columns]

    def _xgb():
        return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                             subsample=0.8, random_state=42, verbosity=0)

    # ── Stage 1: Qualifying model ────────────────────────────────────────
    predicted_quali_pos = None
    use_quali = (quali_df is not None and not quali_df.empty
                 and "round" in quali_df.columns and len(q_feat_cols) >= 2)
    if use_quali:
        qdata = _build_quali_Xyw(feat, quali_df, q_feat_cols, round_rank)
        if qdata is not None:
            Xq, yq, wq = qdata
            qm = _xgb()
            qm.fit(Xq, yq, sample_weight=wq)
            pq = np.clip(qm.predict(feat[q_feat_cols].values), 1, n)
            predicted_quali_pos = pq
            pole_code = feat.iloc[int(np.argmin(pq))]["code"]
            print(f"   🏎  XGB quali model → pole predicted: {pole_code} "
                  f"(P{pq.min():.1f})")

    # ── Stage 2: Race model ──────────────────────────────────────────────
    rdata = _build_race_Xyw(feat, race_df, quali_df, r_feat_cols, round_rank)
    if rdata is None:
        return None
    X, y, w_arr, all_race_cols = rdata

    # ── Training mode: full retrain vs incremental warm start ────────────
    _do_warm = (warm_race_df is not None
                and warm_start_file is not None
                and os.path.exists(warm_start_file))
    if _do_warm:
        inc_rdata = _build_race_Xyw(feat, warm_race_df, quali_df,
                                    r_feat_cols, round_rank)
        if inc_rdata is None:
            _do_warm = False  # too few rows — fall back to full retrain

    if _do_warm:
        Xi, yi, wi, _ = inc_rdata
        wi = wi * _INCREMENTAL_WEIGHT_MULT
        try:
            from xgboost import Booster as _XGBBooster
            _saved = _XGBBooster(); _saved.load_model(warm_start_file)
            if _saved.num_features() != Xi.shape[1]:
                raise ValueError(f"feature count mismatch: saved={_saved.num_features()} vs new={Xi.shape[1]}")
            model = XGBRegressor(n_estimators=_INCREMENTAL_TREES, max_depth=3,
                                 learning_rate=0.1, subsample=0.8,
                                 random_state=42, verbosity=0)
            model.fit(Xi, yi, sample_weight=wi, xgb_model=warm_start_file)
            model.save_model(warm_start_file)
            print(f"   🔄  XGBoost: warm start (+{_INCREMENTAL_TREES} árboles, "
                  f"×{_INCREMENTAL_WEIGHT_MULT} peso, {len(Xi)} muestras nuevas)")
        except Exception as _e:
            print(f"   ⚠  XGBoost warm start falló ({_e}) — retrain completo")
            _do_warm = False
    if not _do_warm:
        model = _xgb()
        try:
            from sklearn.model_selection import LeaveOneOut
            loo, errors = LeaveOneOut(), []
            for tr, te in loo.split(X):
                model.fit(X[tr], y[tr], sample_weight=w_arr[tr])
                errors.append(abs(model.predict(X[te])[0] - y[te][0]))
            print(f"   ✅  XGBoost Race MAE (LOO): {np.mean(errors):.2f} posiciones")
        except Exception:
            pass
        model.fit(X, y, sample_weight=w_arr)
        if warm_start_file:
            model.save_model(warm_start_file)
        print(f"   ✅  XGBoost: entrenamiento completo ({len(X)} muestras)")

    # Inference: build feature matrix that matches training column order
    X_pred = feat[r_feat_cols].values
    if "predicted_quali_pos" in all_race_cols:
        if predicted_quali_pos is not None:
            X_pred = np.hstack([X_pred, predicted_quali_pos.reshape(-1, 1)])
        else:
            fallback = (feat["avg_grid"].values.reshape(-1, 1)
                        if "avg_grid" in feat.columns else np.full((n, 1), 11.5))
            X_pred = np.hstack([X_pred, fallback])

    pred_pos = model.predict(X_pred)
    score    = pd.Series(1 / (pred_pos + 0.1), index=feat.index)

    importances = dict(zip(all_race_cols, model.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("   📊  Top features XGBoost (race model):")
    for fname, fimp in top5:
        print(f"      {fname:<22} {fimp:.3f}  {'█' * int(fimp * 100)}")
    return score, importances


def score_lightgbm(feat: pd.DataFrame, race_df: pd.DataFrame,
                   quali_df: pd.DataFrame = None,
                   warm_start_file: str = None,
                   warm_race_df: pd.DataFrame = None):
    """
    Two-stage LightGBM pipeline mirroring score_xgboost().
    Returns (score, importances) or None.
    Importances are normalised to sum=1 so they blend directly with XGB importances.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("   ⚠  lightgbm no instalado — omitiendo LGBM")
        return None

    if race_df.empty:
        return None

    n = len(feat)
    _, round_rank = _round_weights(race_df)
    q_feat_cols   = [c for c in _QUALI_FEAT_COLS if c in feat.columns]
    r_feat_cols   = [c for c in _RACE_FEAT_COLS  if c in feat.columns]

    def _lgbm():
        return lgb.LGBMRegressor(n_estimators=100, max_depth=3, num_leaves=8,
                                  learning_rate=0.1, subsample=0.8,
                                  random_state=42, verbose=-1)

    # ── Stage 1: Qualifying model ────────────────────────────────────────
    predicted_quali_pos = None
    use_quali = (quali_df is not None and not quali_df.empty
                 and "round" in quali_df.columns and len(q_feat_cols) >= 2)
    if use_quali:
        qdata = _build_quali_Xyw(feat, quali_df, q_feat_cols, round_rank)
        if qdata is not None:
            Xq, yq, wq = qdata
            qm = _lgbm()
            qm.fit(Xq, yq, sample_weight=wq)
            predicted_quali_pos = np.clip(
                qm.predict(feat[q_feat_cols].values), 1, n)

    # ── Stage 2: Race model ──────────────────────────────────────────────
    rdata = _build_race_Xyw(feat, race_df, quali_df, r_feat_cols, round_rank)
    if rdata is None:
        return None
    X, y, w_arr, all_race_cols = rdata

    try:
        # ── Training mode: full retrain vs incremental warm start ────────
        _do_warm_lgb = (warm_race_df is not None
                        and warm_start_file is not None
                        and os.path.exists(warm_start_file))
        if _do_warm_lgb:
            inc_rdata_lgb = _build_race_Xyw(feat, warm_race_df, quali_df,
                                            r_feat_cols, round_rank)
            if inc_rdata_lgb is None:
                _do_warm_lgb = False

        if _do_warm_lgb:
            Xi, yi, wi, _ = inc_rdata_lgb
            wi = wi * _INCREMENTAL_WEIGHT_MULT
            try:
                _saved_lgb = lgb.Booster(model_file=warm_start_file)
                if _saved_lgb.num_feature() != Xi.shape[1]:
                    raise ValueError(f"feature count mismatch: saved={_saved_lgb.num_feature()} vs new={Xi.shape[1]}")
                model = lgb.LGBMRegressor(n_estimators=_INCREMENTAL_TREES,
                                          max_depth=3, num_leaves=8,
                                          learning_rate=0.1, subsample=0.8,
                                          random_state=42, verbose=-1)
                model.fit(Xi, yi, sample_weight=wi, init_model=warm_start_file)
                model.booster_.save_model(warm_start_file)
                print(f"   🔄  LightGBM: warm start (+{_INCREMENTAL_TREES} árboles, "
                      f"×{_INCREMENTAL_WEIGHT_MULT} peso, {len(Xi)} muestras nuevas)")
            except Exception as _e:
                print(f"   ⚠  LightGBM warm start falló ({_e}) — retrain completo")
                _do_warm_lgb = False
        if not _do_warm_lgb:
            model = _lgbm()
            model.fit(X, y, sample_weight=w_arr)
            if warm_start_file:
                model.booster_.save_model(warm_start_file)
            print(f"   ✅  LightGBM: entrenamiento completo ({len(X)} muestras)")

        X_pred = feat[r_feat_cols].values
        if "predicted_quali_pos" in all_race_cols:
            if predicted_quali_pos is not None:
                X_pred = np.hstack([X_pred, predicted_quali_pos.reshape(-1, 1)])
            else:
                fallback = (feat["avg_grid"].values.reshape(-1, 1)
                            if "avg_grid" in feat.columns else np.full((n, 1), 11.5))
                X_pred = np.hstack([X_pred, fallback])

        pred_pos = model.predict(X_pred)
        score    = pd.Series(1 / (pred_pos + 0.1), index=feat.index)

        raw_imp   = model.feature_importances_.astype(float)
        total_imp = raw_imp.sum() or 1.0
        importances = dict(zip(all_race_cols, raw_imp / total_imp))
        return score, importances
    except Exception as e:
        print(f"   ⚠  LightGBM error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
#  21b. GAUSSIAN PROCESS REGRESSION — incertidumbre explícita
# ─────────────────────────────────────────────────────────────
def score_gaussian_process(feat: pd.DataFrame, race_df: pd.DataFrame,
                            quali_df: pd.DataFrame = None):
    """
    Single-stage Gaussian Process Regression race model.

    Unlike tree ensembles, GP explicitly quantifies uncertainty: feature-space
    regions with sparse training data receive wider prediction intervals.  This
    makes it especially valuable at the start of a season (6-12 completed races)
    where some regions of the feature space are poorly covered.

    Kernel: RBF (smooth covariance structure) + WhiteKernel (observation noise).
    Features are StandardScaled before fitting — mandatory for distance-based kernels.
    GPR doesn't support sample_weight, so recency-weighting is omitted here; the
    RBF kernel naturally clusters structurally-similar rounds anyway.

    Returns (score, gp_std) or None.
      score   — pd.Series of 1/(pred_pos + 0.1), same convention as XGB/LGBM
      gp_std  — pd.Series of prediction σ per driver (higher = model more uncertain)
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("   ⚠  sklearn GP no disponible — omitiendo GP")
        return None

    if race_df.empty:
        return None

    _, round_rank = _round_weights(race_df)
    r_feat_cols   = [c for c in _RACE_FEAT_COLS if c in feat.columns]
    if len(r_feat_cols) < 3:
        return None

    # Training data — no qualifying stage (keeps GP focused on race-pace space)
    rdata = _build_race_Xyw(feat, race_df, None, r_feat_cols, round_rank)
    if rdata is None:
        return None
    X, y, _, _ = rdata   # sample_weight unused: GPR doesn't support it

    # Scale — RBF kernel computes L2 distances, so scale matters
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = (RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
              WhiteKernel(noise_level=0.5, noise_level_bounds=(0.1, 5.0)))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2,
        normalize_y=True,   # normalize target to zero-mean: important for GP
        random_state=42,
    )
    try:
        gpr.fit(X_scaled, y)
    except Exception as e:
        print(f"   ⚠  GP entrenamiento falló: {e}")
        return None

    X_pred = scaler.transform(feat[r_feat_cols].fillna(0).values)
    pred_pos, pred_std = gpr.predict(X_pred, return_std=True)
    pred_pos = np.clip(pred_pos, 0.5, None)

    score  = pd.Series(1.0 / (pred_pos + 0.1), index=feat.index)
    gp_std = pd.Series(pred_std, index=feat.index)

    # Print top-5 with uncertainty bars
    _gp_df = feat[["code"]].copy()
    _gp_df["pos"] = np.clip(pred_pos, 1.0, 22.0).round(1)
    _gp_df["std"] = gp_std.values.round(2)
    _top5 = _gp_df.nsmallest(5, "pos")
    print(f"   🌊  GP top-5 predicciones (σ = incertidumbre por datos escasos):")
    for _, r in _top5.iterrows():
        bar = "▪" * min(int(r["std"] * 8), 20)
        print(f"      {r['code']:<5} P{r['pos']:4.1f}  σ={r['std']:.2f}  {bar}")
    print(f"   🔧  Kernel optimizado: {gpr.kernel_}")
    print(f"   ✅  GP: {len(X)} muestras, {len(r_feat_cols)} features")

    return score, gp_std


# ─────────────────────────────────────────────────────────────
#  21b-2. LSTM MOMENTUM MODEL
# ─────────────────────────────────────────────────────────────

def score_lstm(feat: pd.DataFrame,
               race_df: pd.DataFrame,
               completed: list,
               seq_len: int = 5) -> "pd.Series | None":
    """
    LSTM sequential momentum model — captures trend signals XGBoost cannot.

    Architecture:
      Input  : (batch, seq_len=5, 4) — [pos_norm, dnf, grid_norm, pts_norm]
      LSTM   : hidden_size=32, num_layers=2, dropout=0.2, bidirectional=False
      Linear : 32 → 1  (predicted next finishing position, lower = better)

    Training:
      Sliding window over each driver's race history.
      Sample weights: recency-scaled 1×→3× (mirrors XGBoost treatment).
      Adam lr=0.01, 50 epochs, weighted MSE loss.

    Returns pd.Series (higher score = better predicted outcome) aligned to
    feat.index, or None when dormant or data is insufficient.

    Active only when len(completed) >= LSTM_MIN_RACES so the sliding-window
    dataset has enough samples per driver to generalise.
    """
    if len(completed) < LSTM_MIN_RACES:
        print(f"   🧠  LSTM dormant ({len(completed)}/{LSTM_MIN_RACES} races)")
        return None

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("   ⚠  torch not installed — LSTM skipped  (pip install torch)")
        return None

    if race_df.empty:
        return None

    # ── Normalisation constants ──────────────────────────────────────────
    _N_POS  = 22.0
    _N_GRID = 22.0
    _N_PTS  = 26.0   # 25 pts for win + 1 fastest lap

    _pts_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

    def _row_feats(r) -> list:
        pos  = float(r.get("pos",  22.0)) / _N_POS
        dnf  = float(r.get("dnf",   0.0))
        grid = float(r.get("grid", 11.0)) / _N_GRID
        pts  = float(_pts_map.get(int(r.get("pos", 22)), 0)) / _N_PTS
        return [pos, dnf, grid, pts]

    # ── Build (X, y, weight) training tuples ────────────────────────────
    sorted_df = race_df.sort_values("round")
    X_list, y_list, w_list = [], [], []

    for code, grp in sorted_df.groupby("code"):
        rows = [_row_feats(r) for _, r in grp.iterrows()]
        n    = len(rows)
        if n < 2:
            continue
        for i in range(n - 1):
            start  = max(0, i - seq_len + 1)
            window = rows[start : i + 1]
            pad    = seq_len - len(window)
            seq    = [[0.0, 0.0, 0.0, 0.0]] * pad + window
            target = rows[i + 1][0]          # next-race pos_norm

            recency = (i + 1) / max(n - 1, 1)   # 0..1 → more recent = 1
            weight  = 1.0 + 2.0 * recency        # range [1, 3]

            X_list.append(seq)
            y_list.append(target)
            w_list.append(weight)

    if len(X_list) < 10:
        print(f"   ⚠  LSTM: only {len(X_list)} training sequences — skipped")
        return None

    X = torch.tensor(X_list, dtype=torch.float32)          # (N, seq_len, 4)
    y = torch.tensor(y_list, dtype=torch.float32)          # (N,)
    w = torch.tensor(w_list, dtype=torch.float32)          # (N,)

    # ── Model definition ─────────────────────────────────────────────────
    class _RaceLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=2,
                                dropout=0.2, batch_first=True, bidirectional=False)
            self.fc   = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(1)   # last timestep → scalar

    model = _RaceLSTM()
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    for _ in range(50):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - y) ** 2 * w).mean()
        loss.backward()
        opt.step()
    final_loss = float(loss.item())

    # ── Inference: predict next position for each current driver ─────────
    model.eval()
    all_codes   = feat["code"].tolist()
    pred_by_code: dict[str, float] = {}

    with torch.no_grad():
        for code in all_codes:
            grp = sorted_df[sorted_df["code"] == code]
            if grp.empty:
                pred_by_code[code] = 0.5   # midfield default (normalised)
                continue
            rows_inf = [_row_feats(r) for _, r in grp.tail(seq_len).iterrows()]
            pad      = seq_len - len(rows_inf)
            seq      = [[0.0, 0.0, 0.0, 0.0]] * pad + rows_inf
            x_inf    = torch.tensor([seq], dtype=torch.float32)
            pred_by_code[code] = float(model(x_inf).item())

    # ── Convert to score (higher = better) ───────────────────────────────
    # Predicted position (lower = better) → negate → normalize to [0, 1]
    pred_s = feat["code"].map(pred_by_code)
    score  = -pred_s                                          # invert direction
    s_min, s_max = score.min(), score.max()
    if s_max > s_min:
        score = (score - s_min) / (s_max - s_min)
    else:
        score = pd.Series(0.5, index=feat.index)

    score.index = feat.index

    n_seqs = len(X_list)
    print(f"   🧠  LSTM: {n_seqs} sequences, {len(completed)} races, "
          f"loss={final_loss:.4f}")
    top3 = score.nlargest(3)
    _top_codes = feat.loc[top3.index, "code"].tolist()
    print(f"   🧠  LSTM top-3 momentum: {', '.join(_top_codes)}")

    return score


# ─────────────────────────────────────────────────────────────
#  21b-3. GRAPH NEURAL NETWORK — OVERTAKING PROBABILITY
# ─────────────────────────────────────────────────────────────

def score_gnn_overtaking(
        feat: pd.DataFrame,
        race_df: pd.DataFrame,
        completed: list,
        next_circuit_type: str = "mixed",
) -> "tuple[pd.Series, dict] | None":
    """
    Manual 2-layer Graph Convolutional Network for overtaking-potential modeling.

    Activation: len(completed) >= LSTM_MIN_RACES (same 12-race threshold as LSTM).

    ── DATA HONESTY ────────────────────────────────────────────────────────
    The ideal GNN would use per-lap position swaps with gap < 2 s from
    OpenF1 /intervals as edge labels.  Those data are not in the current
    pipeline — fetching 8 historical interval streams would add ~8 API
    round-trips that are not yet cached, and the /intervals endpoint
    returns timestamps rather than lap-number-aligned records, requiring
    non-trivial join logic to obtain per-lap proximity windows.

    Instead we approximate with what race_df provides:
      · Adjacency: drivers who started within 3 grid positions = "proximity
        contest" (proxy for "within 2 seconds" at lap 1 of the race).
      · Training signal: grid_pos − final_pos per driver (overall position
        gain), not per-lap swap events.

    The GCN's genuine added value here is NEIGHBOURHOOD CONTEXT: a driver
    surrounded by aggressive overtakers on the grid faces different dynamics
    than one surrounded by conservative pacers, even after adjusting for
    raw pace.  With 12+ races and per-lap gap data this model would be
    substantially more discriminating.  At 12 races it adds marginal but
    real signal on top of `overtaking_ability`.

    Returns:
        (driver_scores, overtake_matrix) — or None when dormant/insufficient
        driver_scores  : pd.Series [0,1], higher = more overtaking potential
        overtake_matrix: {(code_a, code_b): float} — lap-1 pass probability
                         for adjacent grid pairs (code_a starts ahead;
                         code_b is the passer)
    ────────────────────────────────────────────────────────────────────────
    """
    if len(completed) < LSTM_MIN_RACES:
        print(f"   🕸️  GNN dormant ({len(completed)}/{LSTM_MIN_RACES} races)")
        return None

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("   ⚠  torch not installed — GNN skipped  (pip install torch)")
        return None

    if race_df.empty or "grid" not in race_df.columns:
        return None

    # ── Index: all drivers in the current feature frame ──────────────────
    all_codes = feat["code"].tolist()   # preserves order; aligns with feat.index
    code_idx  = {c: i for i, c in enumerate(all_codes)}
    n         = len(all_codes)

    # ── Historical adjacency matrix ───────────────────────────────────────
    # pass_matrix[i, j]  = times driver j passed driver i
    #                       (j started behind i AND finished ahead of i)
    # opp_matrix[i, j]   = times they were within 3 grid positions
    #                       (i started ahead — opportunity for j to pass i)
    pass_matrix = np.zeros((n, n), dtype=np.float32)
    opp_matrix  = np.zeros((n, n), dtype=np.float32)

    for rnd, grp in race_df.groupby("round"):
        grp = grp.dropna(subset=["grid", "pos"])
        grp = grp[grp["code"].isin(code_idx)]
        if len(grp) < 4:
            continue
        grid_map = {r["code"]: int(r["grid"]) for _, r in grp.iterrows()}
        pos_map  = {r["code"]: int(r["pos"])  for _, r in grp.iterrows()}
        codes_r  = list(grid_map)

        for k, code_a in enumerate(codes_r):
            for code_b in codes_r[k + 1:]:
                g_a, g_b = grid_map[code_a], grid_map[code_b]
                p_a, p_b = pos_map[code_a],  pos_map[code_b]
                if abs(g_a - g_b) > 3:
                    continue
                i, j = code_idx[code_a], code_idx[code_b]
                if g_a < g_b:              # a starts ahead — can b pass a?
                    opp_matrix[i, j] += 1
                    if p_b < p_a:          # b finished ahead = overtake
                        pass_matrix[i, j] += 1
                else:                      # b starts ahead — can a pass b?
                    opp_matrix[j, i] += 1
                    if p_a < p_b:
                        pass_matrix[j, i] += 1

    # Empirical pass rate with Laplace smoothing (prior = 0.10).
    # Without smoothing, 0/0 → NaN and 1/1 → 1.0 — both unreliable at low N.
    prior = 0.10
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.where(
            opp_matrix > 0,
            (pass_matrix + prior) / (opp_matrix + 1.0),
            0.0,
        ).astype(np.float32)

    A = adj + np.eye(n, dtype=np.float32) * 0.5   # self-loops for stability
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    A_norm   = (A / row_sums).astype(np.float32)
    A_t      = torch.tensor(A_norm, dtype=torch.float32)

    # ── Node features [pace, overtaking_ability, grid_slot] ──────────────
    def _norm01(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return ((arr - lo) / max(float(hi - lo), 1e-9)).astype(np.float32)

    fi = feat.set_index("code")

    pace_col = "raw_score" if "raw_score" in feat.columns else "avg_finish"
    pace_n   = _norm01(fi[pace_col].reindex(all_codes).fillna(0.5).values.astype(np.float32))

    oa_n = _norm01(
        fi["overtaking_ability"].reindex(all_codes).fillna(0.0).values.astype(np.float32)
        if "overtaking_ability" in feat.columns
        else np.full(n, 0.0, dtype=np.float32)
    )

    if "quali_pos_next" in feat.columns:
        gp_raw = fi["quali_pos_next"].reindex(all_codes).fillna(11.0).values.astype(np.float32)
        gp_n   = 1.0 - _norm01(gp_raw)   # invert: P1 → 1.0, P22 → 0.0
    else:
        gp_n   = np.full(n, 0.5, dtype=np.float32)

    X   = np.column_stack([pace_n, oa_n, gp_n]).astype(np.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # ── Training target: mean position gain (grid − final) ────────────────
    gains = {}
    for code in all_codes:
        drv = race_df[race_df["code"] == code].dropna(subset=["grid", "pos"])
        if not drv.empty:
            gains[code] = float((drv["grid"] - drv["pos"]).mean())
    y_raw  = np.array([gains.get(c, 0.0) for c in all_codes], dtype=np.float32)
    y_t    = torch.tensor(_norm01(y_raw), dtype=torch.float32)

    # ── Manual 2-layer GCN (no torch_geometric dependency) ───────────────
    # Kipf & Welling (2017): H_{l+1} = ReLU(Â @ H_l @ W_l)
    # where Â = D^{-1} A (simplified symmetric normalisation).
    class _ManualGCN(nn.Module):
        def __init__(self, in_dim: int, hidden: int):
            super().__init__()
            self.W1   = nn.Linear(in_dim, hidden, bias=False)
            self.W2   = nn.Linear(hidden, hidden, bias=False)
            self.head = nn.Linear(hidden, 1, bias=True)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            h1 = self.relu(a @ self.W1(x))    # (n, hidden)
            h2 = self.relu(a @ self.W2(h1))   # (n, hidden)
            return self.head(h2).squeeze(1)    # (n,)

    model = _ManualGCN(in_dim=3, hidden=32)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    model.train()
    for _ in range(100):
        opt.zero_grad()
        pred = model(X_t, A_t)
        loss = nn.functional.mse_loss(pred, y_t)
        loss.backward()
        opt.step()
    final_loss = float(loss.item())

    # ── Inference ─────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        raw_out = model(X_t, A_t).numpy()   # (n,) predicted normalised gain

    scores_norm = _norm01(raw_out)
    driver_scores = pd.Series(scores_norm, index=feat.index)

    # ── Pairwise lap-1 overtake probability matrix for Monte Carlo ─────────
    # Only driver pairs adjacent in the NEXT race qualifying grid (within 2
    # positions).  For each pair where code_a starts ahead, we estimate the
    # probability that code_b passes code_a on lap 1.
    # Formula: base empirical rate (Laplace-smoothed) ± GCN score differential.
    overtake_matrix: dict[tuple, float] = {}

    if "quali_pos_next" in feat.columns:
        next_grid = {
            row["code"]: int(row.get("quali_pos_next", 22))
            for _, row in feat.iterrows()
        }
        for code_a in all_codes:
            for code_b in all_codes:
                if code_a == code_b:
                    continue
                ga = next_grid.get(code_a, 22)
                gb = next_grid.get(code_b, 22)
                if ga >= gb or gb - ga > 2:   # only: a ahead AND b within 2 slots
                    continue
                i, j = code_idx[code_a], code_idx[code_b]
                base_p     = float(adj[i, j])
                score_diff = (scores_norm[j] - scores_norm[i]) * 0.15
                p          = float(np.clip(base_p + score_diff, 0.02, 0.35))
                overtake_matrix[(code_a, code_b)] = p

    sparsity = float((adj > 0).mean())
    n_pairs  = len(overtake_matrix)
    print(f"   🕸️  GNN: {n}×{n} adj ({sparsity:.0%} non-zero), "
          f"{n_pairs} lap-1 pairs, loss={final_loss:.4f}")
    top3 = [all_codes[i] for i in np.argsort(scores_norm)[::-1][:3]]
    print(f"   🕸️  GNN top-3 overtakers: {', '.join(top3)}")

    return driver_scores, overtake_matrix


# ─────────────────────────────────────────────────────────────
#  21c. MODELO DE FIABILIDAD
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
        feat                   : pd.DataFrame,
        base_scores            : pd.Series,
        reliability            : pd.DataFrame,
        weather                : dict,
        n_sims                 : int = 10_000,
        circuit_name           : str = "",
        gnn_overtake           : "dict | None" = None,
        compound_softness      : float = 0.5,   # normalised [0,1]: 0=C1/C2/C3, 1=C3/C4/C5
        tyre_inventory_scores  : "dict | None" = None,  # {code: z-score} fresh soft advantage
) -> pd.DataFrame:
    """
    Corre N simulaciones de la carrera inyectando variaciones aleatorias en:
      1. Ritmo base del piloto          (ruido gaussiano ±15%)
      2. Fiabilidad / DNF               (Bernoulli por piloto)
      3. Safety Car                     (Poisson, lambda derivado de SAFETY_CAR_PROB)
      4. Pit stop time loss             (team-specific stationary time via PIT_STOP_LOSS)
      5. Clima / lluvia                 (si se espera lluvia, mezcla ranking)
      6. Vuelta 1 incidente             (GNN pairwise probs when active; generic 18% fallback)
      7. Tyre strategy / undercut       (circuit-specific prob from UNDERCUT_WINDOW, P3-P10)
         Traffic model: 30% of undercuts exit into traffic → partial gain recovery
      8. Final stint boost              (2-stop: +0.02 pace over last 20 laps)

    gnn_overtake: {(code_a, code_b): float} — pairwise lap-1 overtake probability
                  produced by score_gnn_overtaking().  When supplied, replaces the
                  generic 18% lap-1 incident with per-pair sampling for adjacent
                  grid pairs in the top-8.  Falls back to generic logic when None.

    Retorna un DataFrame con:
      - win_mc_pct    : % de simulaciones en que el piloto ganó
      - podium_mc_pct : % de veces en top-3
      - finish_mc_pct : % de veces que terminó la carrera
      - avg_mc_pos    : posición promedio en las simulaciones
      - p10_pos / p90_pos : percentiles 10 y 90 de posición (intervalo de confianza)
    """
    # Circuit-specific SC/VSC probability → Poisson lambda
    # lambda = -ln(1 - p)  so that P(≥1 SC) = 1 - e^(-lambda) = p
    _DEFAULT_SC_PROB = 0.50
    sc_prob   = SAFETY_CAR_PROB.get(circuit_name, _DEFAULT_SC_PROB)
    sc_lambda = -np.log(1.0 - sc_prob) if sc_prob < 1.0 else 3.0
    _uw          = UNDERCUT_WINDOW.get(circuit_name, {"laps": (16, 20), "prob": 0.40})
    _uw_laps      = _uw["laps"]
    # Scale undercut probability by compound softness.
    # Softer compounds degrade faster → larger strategy windows → more undercuts.
    # compound_softness=0 (C1/C2/C3) → ×0.90; =0.5 (C2/C3/C4) → ×1.00; =1.0 (C3/C4/C5) → ×1.10
    # Capped at [0.05, 0.80] to stay physically plausible.
    # CONFIDENCE: lower than other inputs — depends on scraped compound data.
    _compound_scale = 1.0 + 0.20 * (compound_softness - 0.5)
    undercut_prob   = float(np.clip(float(_uw["prob"]) * _compound_scale, 0.05, 0.80))
    print(f"🎲  Corriendo {n_sims:,} simulaciones Monte Carlo...")
    print(f"   🚦  SC/VSC prob: {sc_prob:.0%}  "
          f"(λ={sc_lambda:.2f}, ~{sc_lambda:.1f} SC esperados por carrera)")
    if _uw_laps[0] > 0:
        print(f"   🔁  Undercut prob: {undercut_prob:.0%}  "
              f"(ventana óptima laps {_uw_laps[0]}–{_uw_laps[1]}, "
              f"compound scale ×{_compound_scale:.2f})")
    else:
        print(f"   🔁  Undercut prob: {undercut_prob:.0%}  "
              f"(circuito urbano — sin ventana, compound scale ×{_compound_scale:.2f})")

    rng      = np.random.default_rng(42)
    n        = len(feat)
    codes    = feat["code"].tolist()
    scores_v = base_scores.values.copy().astype(float)

    # Rank-based pace mapping — decouples pace spread from raw_score magnitude
    # Rank 1 (best model score) → 1.030, Rank 22 → 0.970, log curve between
    # Prevents outlier raw_scores from monopolising the pace ceiling
    ranks = pd.Series(scores_v).rank(ascending=False, method='min').values
    scores_v = 1.03 - 0.06 * np.log(ranks) / np.log(len(ranks))

    # Race threat boost: drivers whose FP2 race pace is much better than their grid
    # slot get +0.015 added to their MC pace — they will charge forward in the race
    _rt_flags = (feat["race_threat_flag"].values
                 if "race_threat_flag" in feat.columns
                 else np.zeros(n, dtype=bool))
    _rt_boost = np.where(_rt_flags, 0.015, 0.0)
    if _rt_boost.any():
        _rt_codes = [codes[i] for i in range(n) if _rt_flags[i]]
        print(f"   🚀  Race threat boost (+0.015) → {', '.join(_rt_codes)}")
    scores_v += _rt_boost

    # Grid starting advantage — drivers hold qualifying position for the first
    # GRID_LAPS laps before pace takes over (5/57 ≈ 8.8% of a typical F1 race).
    # Falls back to season avg_grid ranking when next-race quali isn't loaded.
    _qp = feat["quali_pos_next"].copy()
    if (_qp == _qp.max()).all():   # all set to same NaN-sentinel → no quali data
        _qp = pd.Series(feat["avg_grid"].values).rank(method="first", ascending=True)
    _qp = _qp.clip(lower=1).fillna(float(n)).values.astype(float)
    grid_scores_v = 1.03 - 0.06 * np.log(np.maximum(_qp, 1)) / np.log(max(n, 2))
    _grid_order   = np.argsort(_qp)   # index 0 = P1 starter, index n-1 = last
    GRID_LAPS   = 5
    RACE_LAPS   = 57          # representative F1 race length (laps vary 52–78)
    grid_weight = GRID_LAPS / RACE_LAPS   # ≈ 0.0877

    # Tyre strategy per constructor (2026 known team preferences)
    _STRAT_1STOP = {"Mercedes", "Ferrari", "Red Bull", "Aston Martin"}
    is_1stop = np.array([feat.iloc[i]["TeamName"] in _STRAT_1STOP for i in range(n)])
    is_2stop = ~is_1stop

    # Mapa de fiabilidad por código
    rel_map      = dict(zip(reliability["code"], reliability["finish_prob"]))
    mech_risk_map = dict(zip(reliability["code"],
                             reliability.get("mechanical_risk",
                             pd.Series(0.06, index=reliability.index))))

    # ¿Lluvia esperada?
    rain_prob = float(weather.get("rain_prob", 0.0))

    # Per-driver pit penalty: map team stationary time → score multiplier loss.
    # Scale: best crew (2.3s) → 0.010, worst (2.9s) → 0.025.
    _pit_range = _PIT_LOSS_WORST - _PIT_LOSS_BEST  # 0.6
    _pit_penalty = np.array([
        0.010 + 0.015 * (
            PIT_STOP_LOSS.get(feat.iloc[i]["TeamName"], _PIT_LOSS_DEFAULT) - _PIT_LOSS_BEST
        ) / _pit_range
        for i in range(n)
    ])

    # Acumuladores
    wins    = np.zeros(n, dtype=int)
    podiums = np.zeros(n, dtype=int)
    finishes= np.zeros(n, dtype=int)
    pos_all = np.zeros((n, n_sims), dtype=np.float32)

    for sim in range(n_sims):
        # 1. Ruido de ritmo base (±5% gaussiano — calibrado para rango de pace 6%)
        noise_scale = 0.05 + 0.03 * rain_prob   # 0.05 dry → 0.08 fully wet
        sim_scores  = scores_v * (1 + rng.normal(0, noise_scale, n))

        # 1b. Starting grid anchor: blend qualifying grid position into pace scores
        #     Grid is deterministic (no noise); pace noise already applied above.
        sim_scores = (1.0 - grid_weight) * sim_scores + grid_weight * grid_scores_v

        # 2. Safety Car — each SC event compresses the field (applied once per event)
        #    Lambda derived from per-circuit historical SC probability
        n_sc = rng.poisson(sc_lambda)
        for _ in range(n_sc):
            # SC comprime el pelotón → añade ruido extra al 50% trasero
            bottom_half = np.argsort(sim_scores)[:n//2]
            sim_scores[bottom_half] *= (1 + rng.uniform(0, 0.03, len(bottom_half)))

        # 3. Lluvia — shuffle proporcional a rain_prob
        #    At rain_prob=0.3: 18% chance, ~2 drivers shuffled
        #    At rain_prob=1.0: 60% chance, ~7 drivers shuffled
        if rng.random() < rain_prob * 0.6:
            shuffle_size = max(1, round(n * rain_prob / 3))
            shuffle_idx  = rng.choice(n, size=shuffle_size, replace=False)
            sim_scores[shuffle_idx] = rng.permutation(sim_scores[shuffle_idx])

        # 4. Pit stop time loss — team-specific stationary time penalty.
        # Faster crews (Mercedes/McLaren 2.3s) incur smaller score hit than
        # slower ones (Audi 2.9s). Penalty scales linearly across the 0.6s range.
        pit_victim = rng.integers(0, n)
        sim_scores[pit_victim] *= (1.0 - _pit_penalty[pit_victim] * rng.uniform(0.8, 1.2))

        # 5. Vuelta 1: GNN pairwise overtake sampling (active) or generic incident
        if gnn_overtake:
            # Sample each adjacent-grid pair in the top-8 independently.
            # If b passes a, a loses a small score fragment and b gains one.
            # Aggregate effect across 7 pairs approximates the generic 18% model
            # when individual probabilities are ~0.05–0.15 per pair.
            for _gi in range(min(7, n - 1)):
                _idx_a  = _grid_order[_gi]
                _idx_b  = _grid_order[_gi + 1]
                _p_pass = gnn_overtake.get((codes[_idx_a], codes[_idx_b]), 0.05)
                if rng.random() < _p_pass:
                    sim_scores[_idx_a] *= (1.0 - rng.uniform(0.01, 0.03))
                    sim_scores[_idx_b] *= (1.0 + rng.uniform(0.01, 0.02))
        else:
            if rng.random() < 0.18:
                victim = rng.integers(0, min(8, n))   # más probable en top-8
                sim_scores[victim] *= (1 - rng.uniform(0.02, 0.05))

        # 5b. Tyre strategy events
        # Undercut: circuit-specific probability from UNDERCUT_WINDOW.
        # One 1-stop driver running P3-P10 pits 3-5 laps early and jumps a rival.
        # Traffic model: 30% of undercuts exit behind slow traffic and lose
        # 1-2 positions temporarily; pace advantage recovers them over ~5 laps.
        _sorted = np.argsort(-sim_scores)
        _uc = [i for i in _sorted[2:10] if is_1stop[i]]
        if _uc and rng.random() < undercut_prob:
            # Drivers with more fresh softs than average (tyre_inventory_score > 0.5)
            # get +5% selection weight — they have more strategic flexibility to pit early.
            if tyre_inventory_scores:
                _uc_weights = np.array([
                    1.05 if tyre_inventory_scores.get(codes[i], 0.0) > 0.5 else 1.0
                    for i in _uc
                ], dtype=float)
                _uc_weights /= _uc_weights.sum()
                winner_idx = rng.choice(_uc, p=_uc_weights)
            else:
                winner_idx = rng.choice(_uc)
            gain = rng.uniform(0.002, 0.004)
            if rng.random() < 0.30:  # traffic on pit exit
                # net loss after 5-lap recovery = traffic_loss × (5/RACE_LAPS)
                gain -= rng.uniform(0.001, 0.002) * (5.0 / RACE_LAPS)
            sim_scores[winner_idx] += max(0.0, gain)
        # 2-stop fresh tyre pace boost in final 20 laps (≈ +0.007 net per sim)
        sim_scores[is_2stop] *= (1.0 + 0.02 * (20.0 / RACE_LAPS))

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

    # Win% confidence intervals via 100 non-overlapping batches of 100 sims
    # P10/P90 of batch-level win% gives a meaningful uncertainty range:
    #   a driver at 19.7% overall will span roughly ±5pp across batches.
    _BATCH_N    = 100
    _BATCH_SIZE = n_sims // _BATCH_N          # = 100 with default n_sims=10000
    _win_mat    = (pos_all == 1.0).astype(np.float32)   # (n, n_sims)
    _win_batch  = (
        _win_mat[:, : _BATCH_N * _BATCH_SIZE]
        .reshape(n, _BATCH_N, _BATCH_SIZE)
        .mean(axis=2) * 100
    )                                          # shape (n, _BATCH_N)
    p10_win_v = np.percentile(_win_batch, 10, axis=1)   # (n,)
    p90_win_v = np.percentile(_win_batch, 90, axis=1)   # (n,)

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
            "p10_win_pct"   : round(float(p10_win_v[i]), 2),
            "p90_win_pct"   : round(float(p90_win_v[i]), 2),
        })

    mc_df = pd.DataFrame(rows).sort_values("win_mc_pct", ascending=False
                         ).reset_index(drop=True)
    print(f"   ✅  Simulaciones completadas.")
    return mc_df

# ─────────────────────────────────────────────────────────────
#  21. BRIER SCORE — calibración de predicción post-carrera
# ─────────────────────────────────────────────────────────────
def brier_score(pred_df: pd.DataFrame, winner_code: str) -> float:
    """
    Brier score for a single race: mean((p_win - outcome)^2) across all drivers.
      pred_df:      DataFrame with 'code' and 'win_mc_pct' (0-100 scale)
      winner_code:  driver code of actual race winner
    Lower = better; 0.0 = perfect; ~0.045 = random uniform baseline for 22 drivers.
    """
    if pred_df.empty or "win_mc_pct" not in pred_df.columns:
        return float("nan")
    total = sum(
        ((row["win_mc_pct"] / 100.0) - (1.0 if row["code"] == winner_code else 0.0)) ** 2
        for _, row in pred_df.iterrows()
    )
    return round(total / len(pred_df), 4)

# ─────────────────────────────────────────────────────────────
#  22. NORMALIZAR A PROBABILIDAD
# ─────────────────────────────────────────────────────────────
def scores_to_probability(scores: pd.Series, temperature: float = 3.0) -> pd.Series:
    scaled = scores * temperature
    exp_scores = np.exp(scaled - scaled.max())  # subtract max for numerical stability
    if exp_scores.sum() == 0:
        return pd.Series(1 / len(scores), index=scores.index)
    return exp_scores / exp_scores.sum() * 100


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
                              "p10_pos","p90_pos",
                              "p10_win_pct","p90_win_pct"]],
                      on="code", how="left")
    if reliability is not None and not reliability.empty:
        df = df.merge(reliability[["code","finish_prob","real_dnfs","real_races"]],
                      on="code", how="left")

    use_mc = mc_df is not None and not mc_df.empty

    # ── Clima ─────────────────────────────────────────────────
    if weather:
        temp = weather.get("avg_track_temp")
        hum  = weather.get("avg_humidity")
        rain = weather.get("rain_prob", 0.0)
        if rain < 0.10:
            rain_lbl = "☀  Seco esperado"
        elif rain < 0.50:
            rain_lbl = f"⛅  Lluvia posible {rain:.0%}"
        else:
            rain_lbl = f"🌧  Lluvia probable {rain:.0%}"
        _src      = "[nowcast]" if weather.get("nowcast_available") else "[horario]"
        clima_str = (
            rain_lbl
            + (f"  |  Pista ~{temp:.0f}°C" if temp else "")
            + (f"  |  Humedad {hum:.0f}%"  if hum  else "")
            + f"  {_src}"
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

    # ── Compact win / podium summary with CI — todos los pilotos ──────
    if use_mc:
        print(f"\n  📈  PROBABILIDADES — Ganar & Podio  (todos los pilotos)\n")
        _epi_max_code, _epi_max_val = None, 0.0

        for _, r in df.iterrows():
            code    = r["code"]
            win_mc  = r.get("win_mc_pct",    0)
            pod_mc  = r.get("podium_mc_pct", 0)
            p10w    = r.get("p10_win_pct",   0)
            p90w    = r.get("p90_win_pct",   0)
            epi_unc = float(r.get("epistemic_unc", 0.0))

            if epi_unc > _epi_max_val:
                _epi_max_val, _epi_max_code = epi_unc, code

            disagree  = epi_unc > 0.15
            prefix    = "⚠ " if disagree else "  "
            dis_tag   = "  [models disagree]" if disagree else ""

            if win_mc >= 3.0:
                print(f"  {prefix}{code:<5}  {win_mc:>5.1f}% win  "
                      f"(P10: {p10w:.1f}% — P90: {p90w:.1f}%)  │  {pod_mc:>5.1f}% podio"
                      f"{dis_tag}")
            else:
                print(f"  {prefix}{code:<5}  {win_mc:>5.1f}% win  │  {pod_mc:>5.1f}% podio"
                      f"{dis_tag}")

        print()

        # Uncertainty summary lines
        if _epi_max_code:
            print(f"  ⚠  Mayor desacuerdo de modelos   : {_epi_max_code} ({_epi_max_val:.2f})")

        _has_epi = "epistemic_unc" in df.columns
        _has_ale = "aleatoric_unc" in df.columns
        if _has_epi and _has_ale:
            _unc = df[["code", "epistemic_unc", "aleatoric_unc"]].copy().fillna(0.0)
            _epi_top = _unc["epistemic_unc"].max() or 1.0
            _ale_top = _unc["aleatoric_unc"].max() or 1.0
            _unc["_combined"] = (_unc["epistemic_unc"] / _epi_top +
                                  _unc["aleatoric_unc"] / _ale_top)
            _mu = _unc.loc[_unc["_combined"].idxmax()]
            print(f"  🔮  Predicción más incierta       : {_mu['code']} "
                  f"(epistémica: {_mu['epistemic_unc']:.2f}, "
                  f"aleatoria: {_mu['aleatoric_unc']:.1f})")
        print()

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
#  22b. STACKING META-MODEL — per-circuit-type XGB/LGBM weights
# ─────────────────────────────────────────────────────────────
def fit_stacking_meta_model(loo_driver_preds: list) -> "dict | None":
    """
    Train a logistic regression meta-model on LOO per-driver predictions.

    Feature matrix: interaction columns [xgb_p × ctype_i, lgbm_p × ctype_i]
    for each of the 4 circuit types.  fit_intercept=False so that the learned
    coefficients directly represent the relative trust in each base model at
    each circuit type.

    Requires loo_driver_preds to contain 'xgb_win_prob', 'lgbm_win_prob', and
    'circuit_type' fields (added by the updated cross_validate_season()).
    Returns a JSON-serialisable dict of coefficients, or None.
    """
    rows = [r for r in loo_driver_preds
            if "xgb_win_prob" in r and "lgbm_win_prob" in r]
    n_rounds = len({r["round"] for r in rows})

    if n_rounds < _META_MIN_ROUNDS:
        print(f"   ℹ️   Stacking meta-modelo: {n_rounds}/{_META_MIN_ROUNDS} rondas "
              f"— blend 50/50 por ahora")
        return None

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    n_ctypes = len(_META_CTYPES)
    X_meta, y_meta = [], []
    for r in rows:
        xgb_p  = r["xgb_win_prob"]  / 100.0
        lgbm_p = r["lgbm_win_prob"] / 100.0
        ctype  = r.get("circuit_type", "mixed")
        ci     = _META_CTYPES.index(ctype) if ctype in _META_CTYPES else n_ctypes - 1
        feats  = [0.0] * (n_ctypes * 2)
        feats[ci * 2]     = xgb_p
        feats[ci * 2 + 1] = lgbm_p
        X_meta.append(feats)
        y_meta.append(int(r["won"]))

    X_meta = np.array(X_meta, dtype=np.float32)
    y_meta = np.array(y_meta, dtype=np.int32)

    if y_meta.sum() < 1 or y_meta.sum() == len(y_meta):
        return None

    # fit_intercept=True lets the model learn the ~4.5% base win rate via intercept,
    # freeing coefficients to be positive (higher prob → more likely to win).
    clf = LogisticRegression(
        fit_intercept=True, C=10.0, max_iter=1000, random_state=42)
    clf.fit(X_meta, y_meta)
    coef = clf.coef_[0].tolist()

    def _softmax_weights(a: float, b: float) -> tuple:
        """Convert two raw logit coefs to (w_a, w_b) via softmax."""
        ea, eb = np.exp(a - max(a, b)), np.exp(b - max(a, b))  # numerically stable
        s = ea + eb
        return ea / s, eb / s

    print("\n   🔬  Stacking meta-modelo — pesos aprendidos por tipo de circuito:")
    for i, ctype in enumerate(_META_CTYPES):
        w_xgb_raw  = coef[i * 2]
        w_lgbm_raw = coef[i * 2 + 1]
        n_ct = len({r["round"] for r in rows if r.get("circuit_type") == ctype})
        if n_ct > 0:
            frac_xgb, frac_lgbm = _softmax_weights(w_xgb_raw, w_lgbm_raw)
            print(f"      {ctype:<12}: XGB {frac_xgb:.2f}  LGBM {frac_lgbm:.2f}"
                  f"  ({n_ct} rondas)")
        else:
            print(f"      {ctype:<12}: sin datos LOO — usará 50/50")

    return {
        "coef"         : coef,
        "ctypes"       : _META_CTYPES,
        "n_rounds"     : n_rounds,
        "trained_round": max(r["round"] for r in rows),
    }


# ─────────────────────────────────────────────────────────────
#  23. LOO CROSS-VALIDATION — calibración histórica completa
# ─────────────────────────────────────────────────────────────
def cross_validate_season(
    feat      : pd.DataFrame,
    race_df   : pd.DataFrame,
    quali_df  : pd.DataFrame,
    schedule  : pd.DataFrame,
) -> tuple:
    """
    Leave-one-race-out cross-validation across the completed season.

    For each round N:
      - Trains XGBoost on all rounds EXCEPT N  (excluding N prevents data leakage
        in the training labels, though season-average features in feat are built
        from all rounds — a deliberate trade-off to avoid re-fetching all API data)
      - Predicts finishing position for every driver
      - Converts predicted positions to win probabilities via softmax
      - Computes Brier score against the actual round-N winner

    Returns (round_results, driver_preds) where:
      round_results : list of {round, circuit, brier_score, predicted_winner, actual_winner}
      driver_preds  : list of {round, code, win_mc_pct, won} — one entry per driver per race
    Requires len(completed_rounds) >= 4 for meaningful estimates.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("   ⚠  xgboost no instalado — LOO omitido")
        return [], []

    completed_rounds = sorted(race_df["round"].unique())
    if len(completed_rounds) < 4:
        return [], []

    r_feat_cols = [c for c in _RACE_FEAT_COLS if c in feat.columns]
    if not r_feat_cols:
        return [], []

    has_quali = (quali_df is not None and not quali_df.empty
                 and "round" in quali_df.columns)

    # Circuit name + type from schedule
    circuit_map      = {}
    circuit_type_map = {}
    if not schedule.empty:
        for _, row in schedule.iterrows():
            rnd = int(row["round"])
            circuit_map[rnd]      = row.get("name", f"Ronda {rnd}")
            circuit_type_map[rnd] = CIRCUIT_TYPE.get(row.get("circuit", ""), "mixed")

    _has_lgbm_loo = False
    try:
        import lightgbm as _lgb_loo
        _has_lgbm_loo = True
    except ImportError:
        pass

    # avg_grid fallback for inference-phase predicted_quali_pos
    avg_grid_arr = (feat["avg_grid"].fillna(11.5).values.reshape(-1, 1)
                    if "avg_grid" in feat.columns else np.full((len(feat), 1), 11.5))

    results      = []
    driver_preds = []   # per-driver data for calibration curve
    for test_rnd in completed_rounds:
        train_race  = race_df[race_df["round"] != test_rnd]
        train_quali = (quali_df[quali_df["round"] != test_rnd]
                       if has_quali else None)

        if len(train_race["round"].unique()) < 2:
            continue   # need at least 2 training rounds

        _, train_round_rank = _round_weights(train_race)
        rdata = _build_race_Xyw(feat, train_race, train_quali,
                                r_feat_cols, train_round_rank)
        if rdata is None:
            continue
        X, y, w_arr, all_race_cols = rdata

        xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 subsample=0.8, random_state=42, verbosity=0)
        xgb_model.fit(X, y, sample_weight=w_arr)

        # Inference on season feature matrix
        X_pred = feat[r_feat_cols].fillna(0).values
        if "predicted_quali_pos" in all_race_cols:
            X_pred = np.hstack([X_pred, avg_grid_arr])

        xgb_pred_pos  = np.clip(xgb_model.predict(X_pred), 0.5, None)
        xgb_inv_pos   = pd.Series(1.0 / (xgb_pred_pos + 0.1), index=feat.index)
        xgb_win_probs = scores_to_probability(xgb_inv_pos)   # 0-100 scale

        # Also run LightGBM in this LOO fold for stacking signal
        if _has_lgbm_loo:
            try:
                lgbm_loo = _lgb_loo.LGBMRegressor(
                    n_estimators=100, max_depth=3, num_leaves=8,
                    learning_rate=0.1, subsample=0.8, random_state=42, verbose=-1)
                lgbm_loo.fit(X, y, sample_weight=w_arr)
                lgbm_pred_pos  = np.clip(lgbm_loo.predict(X_pred), 0.5, None)
                lgbm_inv_pos   = pd.Series(1.0 / (lgbm_pred_pos + 0.1), index=feat.index)
                lgbm_win_probs = scores_to_probability(lgbm_inv_pos)
            except Exception:
                lgbm_win_probs = xgb_win_probs
        else:
            lgbm_win_probs = xgb_win_probs

        # Blended win probs (50/50 for LOO Brier/calibration consistency)
        win_probs = 0.5 * xgb_win_probs + 0.5 * lgbm_win_probs
        pred_df   = pd.DataFrame({
            "code"      : feat["code"].values,
            "win_mc_pct": win_probs.values,
        })

        # Actual winner for the left-out round
        test_res = race_df[race_df["round"] == test_rnd]
        winners  = test_res[test_res["pos"] == 1]["code"].values
        if len(winners) == 0:
            continue
        actual_winner    = str(winners[0])
        predicted_winner = str(pred_df.loc[pred_df["win_mc_pct"].idxmax(), "code"])
        bs               = brier_score(pred_df, actual_winner)

        _ctype_rnd = circuit_type_map.get(int(test_rnd), "mixed")
        results.append({
            "round"            : int(test_rnd),
            "circuit"          : circuit_map.get(int(test_rnd), f"Ronda {test_rnd}"),
            "circuit_type"     : _ctype_rnd,
            "brier_score"      : bs,
            "predicted_winner" : predicted_winner,
            "actual_winner"    : actual_winner,
        })

        # Accumulate per-driver rows for calibration + stacking meta-model
        _xgb_map  = dict(zip(feat["code"].values, xgb_win_probs.values))
        _lgbm_map = dict(zip(feat["code"].values, lgbm_win_probs.values))
        for _, row in pred_df.iterrows():
            driver_preds.append({
                "round"        : int(test_rnd),
                "code"         : row["code"],
                "win_mc_pct"   : float(row["win_mc_pct"]),
                "xgb_win_prob" : float(_xgb_map.get(row["code"], row["win_mc_pct"])),
                "lgbm_win_prob": float(_lgbm_map.get(row["code"], row["win_mc_pct"])),
                "circuit_type" : _ctype_rnd,
                "won"          : 1 if row["code"] == actual_winner else 0,
            })

    return results, driver_preds


def print_loo_validation(loo_results: list) -> None:
    """Print formatted LOO table with rolling average and trend indicator."""
    if not loo_results:
        return

    scores = [r["brier_score"] for r in loo_results]
    avg_bs = np.mean(scores)

    print("\n📐  VALIDACIÓN LOO — Historial de temporada (Leave-One-Race-Out)")
    header = f"  {'Rd':<4} {'Carrera':<34} {'Brier':<8} {'Pred':<6} {'Real':<6} {'Roll-3'}"
    print(header)
    print("  " + "─" * 68)

    rolling3 = []
    for i, r in enumerate(loo_results):
        roll = float(np.mean(scores[max(0, i - 2): i + 1]))
        rolling3.append(roll)
        hit  = "✓" if r["predicted_winner"] == r["actual_winner"] else " "
        print(f"  R{r['round']:<3} {r['circuit'][:33]:<34} "
              f"{r['brier_score']:.4f}  "
              f"{r['predicted_winner']:<6} {r['actual_winner']:<6}{hit} "
              f"[{roll:.4f}]")

    print("  " + "─" * 68)
    hits = sum(1 for r in loo_results if r["predicted_winner"] == r["actual_winner"])
    print(f"  {'Media LOO':<39} {avg_bs:.4f}   ({hits}/{len(loo_results)} ganadores correctos)")

    # Trend: last 3 vs first 3 (need ≥ 6 rounds)
    if len(scores) >= 6:
        first3 = float(np.mean(scores[:3]))
        last3  = float(np.mean(scores[-3:]))
        delta  = last3 - first3
        if delta < -0.002:
            arrow = "↑ mejorando"
        elif delta > 0.002:
            arrow = "↓ empeorando"
        else:
            arrow = "→ estable"
        print(f"  Tendencia (ú3 vs p3): {first3:.4f} → {last3:.4f}  [{arrow}]")


# ─────────────────────────────────────────────────────────────
#  24. CALIBRACIÓN — curva probabilidad predicha vs real
# ─────────────────────────────────────────────────────────────
_CAL_BUCKETS = [
    ( 0,  5, "0-5%",   2.5),
    ( 5, 10, "5-10%",  7.5),
    (10, 15, "10-15%", 12.5),
    (15, 20, "15-20%", 17.5),
    (20, 101,"20%+",   None),  # midpoint = mean of actual predictions in bucket
]


def calibration_analysis(driver_preds: list) -> dict:
    """
    Build calibration curve from per-driver LOO predictions.

    Each entry in driver_preds: {round, code, win_mc_pct, won}.
    Groups into 5 probability buckets and measures whether predicted
    win rates match actual win rates. Returns a serialisable dict
    suitable for storing in the priors JSON.
    """
    buckets = {}
    for lo, hi, label, midpoint in _CAL_BUCKETS:
        subset   = [p for p in driver_preds if lo <= p["win_mc_pct"] < hi]
        n        = len(subset)
        n_wins   = sum(p["won"] for p in subset)
        act_rate = (n_wins / n * 100) if n > 0 else None
        avg_pred = (sum(p["win_mc_pct"] for p in subset) / n) if n > 0 else None
        mid      = avg_pred if midpoint is None else midpoint   # actual avg for unbounded bucket
        buckets[label] = {
            "lo": lo, "hi": hi,
            "n_predictions": n,
            "n_wins"       : n_wins,
            "actual_rate"  : round(act_rate, 2) if act_rate is not None else None,
            "midpoint"     : round(mid,      1) if mid      is not None else None,
        }

    # Calibration score: MAE (in percentage points) over non-empty buckets
    errors = [
        abs(bd["midpoint"] - bd["actual_rate"])
        for bd in buckets.values()
        if bd["n_predictions"] > 0
           and bd["actual_rate"] is not None
           and bd["midpoint"]   is not None
    ]
    cal_score = round(float(np.mean(errors)), 2) if errors else None

    n_races = len({p["round"] for p in driver_preds})
    return {
        "buckets"          : buckets,
        "calibration_score": cal_score,
        "total_predictions": len(driver_preds),
        "total_wins"       : sum(p["won"] for p in driver_preds),
        "n_races"          : n_races,
    }


def print_calibration(cal: dict) -> None:
    """Print calibration table with bucket stats and overall score."""
    n_races = cal.get("n_races", 0)
    buckets = cal.get("buckets", {})

    print(f"\n📏  CALIBRACIÓN DEL MODELO ({n_races} carreras LOO)")
    if n_races < 5:
        print(f"   ⚠  Solo {n_races} carreras disponibles — resultados ruidosos (recomendado ≥5)")

    print(f"  {'Bucket':<8} {'Predicciones':>13} {'Ganadas':>8} {'Tasa real':>10}  Estado")
    print("  " + "─" * 60)

    for label, bd in buckets.items():
        n   = bd["n_predictions"]
        mid = bd["midpoint"]
        act = bd["actual_rate"]

        if n == 0:
            print(f"  {label:<8} {'—':>13} {'—':>8} {'—':>10}  —")
            continue

        act_str = f"{act:.1f}%" if act is not None else "—"

        if act is None or mid is None:
            status = "—"
        else:
            diff = mid - act   # positive = overconfident (predicted too high)
            if abs(diff) <= 5:
                status = "✓ calibrado"
            elif 5 < diff <= 15:
                status = "~ sobreestimado"
            elif diff > 15:
                status = "↑ muy sobreestimado"
            elif -15 <= diff < -5:
                status = "~ subestimado"
            else:
                status = "↑ muy subestimado"

        print(f"  {label:<8} {n:>13} {bd['n_wins']:>8} {act_str:>10}  {status}")

    print("  " + "─" * 60)
    cs = cal.get("calibration_score")
    if cs is not None:
        quality = "excelente" if cs < 3 else "bueno" if cs < 6 else "mejorable"
        print(f"  Calibration score (MAE): {cs:.2f}pp  [{quality}]  "
              f"(0 = perfecto, <5pp = bueno)")


# ─────────────────────────────────────────────────────────────
#  MEJORA 10 — ACTUALIZACIÓN BAYESIANA ENTRE FINES DE SEMANA
# ─────────────────────────────────────────────────────────────
import json
import os

PRIORS_FILE    = os.environ.get("F1_PRIORS_FILE",    "./f1_2026_bayesian_priors.json")
PROFILES_FILE  = os.environ.get("F1_PROFILES_FILE", "./f1_2026_driver_profiles.json")
SENTIMENT_FILE = os.environ.get("F1_SENTIMENT_FILE", "./f1_2026_sentiment_cache.json")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


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
    ot_difficulty        = OVERTAKING_DIFFICULTY.get(next_circuit, 0.55)
    next_circuit_laps    = CIRCUIT_RACE_LAPS.get(next_circuit, 57)
    circuit_affinity_df  = fetch_circuit_affinity(next_circuit)

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

    # Sprint weekend detection — used for FP session weighting, anomaly logic, SQ fetch
    is_sprint_weekend = next_round in SPRINT_ROUNDS

    print(f"   🏁  Circuito: {next_circuit}")
    print(f"   🗂   Tipo: {next_circuit_type}  |  Dificultad adelantar: {ot_difficulty:.2f}")
    print(f"   ⚖   Peso qualifying dinámico: {round(0.22 + ot_difficulty * 0.16, 3):.1%}")
    if is_sprint_weekend:
        print(f"   🏃  Sprint weekend — R{next_round} {next_circuit}: "
              f"FP1 only, no FP2/FP3 — FP1 long stints proxy race pace")

    # 6d. Datos del próximo GP — clasificación y prácticas ya disponibles
    print(f"\n🎯  Descargando datos pre-carrera del próximo GP (Ronda {next_round})...")
    if of1_available:
        next_quali_df = of1_collect_next_race_qualifying(
                            next_round, of1_sessions, driver_standings)
        next_fp_df    = of1_collect_next_race_fp(next_round, of1_sessions)
        sprint_quali_df = of1_collect_next_sprint_qualifying(next_round, of1_sessions)
        print("\n🔴  Calculando inventario de neumáticos (sesiones completadas)...")
        tyre_inv_df   = compute_tyre_inventory(of1_sessions, next_round)
        print("\n🏎️  Corner telemetry analysis (FastF1)...")
        corner_tel_df = fetch_corner_telemetry(next_round, next_circuit, completed)
        if not next_quali_df.empty:
            print(f"   ✅  Clasificación del próximo GP cargada ({len(next_quali_df)} pilotos)")
        else:
            print("   ⏳  Clasificación del próximo GP aún no disponible")
        if not next_fp_df.empty:
            print(f"   ✅  Prácticas del próximo GP cargadas ({len(next_fp_df)} pilotos)")
        else:
            print("   ⏳  Prácticas del próximo GP aún no disponibles")
        if is_sprint_weekend:
            if not sprint_quali_df.empty:
                print(f"   ✅  Sprint Qualifying cargado ({len(sprint_quali_df)} pilotos)")
            else:
                print("   ⏳  Sprint Qualifying del próximo GP aún no disponible")
    else:
        next_quali_df   = pd.DataFrame(columns=["code","quali_pos_next","quali_time_next"])
        next_fp_df      = pd.DataFrame(columns=["code", "fp_next_delta", "fp2_next_longrun",
                                                "soft_pace_delta", "medium_pace_delta",
                                                "hard_pace_delta", "compound_preference",
                                                "tyre_deg_rate", "deg_rate_soft",
                                                "deg_rate_medium", "deg_rate_hard",
                                                "high_speed_delta", "medium_speed_delta",
                                                "low_speed_delta", "corner_balance",
                                                "race_sim_delta", "race_sim_deg"])
        sprint_quali_df = pd.DataFrame(columns=["code", "sq_pos_next", "sq_time_next"])
        tyre_inv_df     = pd.DataFrame()
        # Corner telemetry uses FastF1 directly — attempt even without OpenF1
        print("\n🏎️  Corner telemetry analysis (FastF1)...")
        corner_tel_df = fetch_corner_telemetry(next_round, next_circuit, completed)

    # 7. Clima y penalizaciones para próxima carrera
    weather  = fetch_weather_for_race(next_round, schedule)
    weather["is_sprint_weekend"] = is_sprint_weekend
    grid_pen = fetch_grid_penalties(next_round, schedule)

    # 8b. Perfiles comportamentales históricos (2023-2025, caché 7 días)
    print("\n🧬  Cargando perfiles comportamentales...")
    _raw_profiles = fetch_driver_behavioral_profiles(
        driver_standings["code"].tolist()
    )
    _prof_metric_cols = ["overtaking_ability", "quali_consistency",
                         "wet_weather_delta", "historical_dnf_rate",
                         "tyre_management_index"]
    behavioral_df = pd.DataFrame([
        {"code": code, **{k: v for k, v in metrics.items()
                          if k in _prof_metric_cols}}
        for code, metrics in _raw_profiles.items()
    ])
    # Print top-5 per behavioral metric
    _metric_display = [
        ("overtaking_ability",    "desc", "Adelantadores netos desde P6-P15"),
        ("quali_consistency",     "asc",  "Consistencia en clasificación (σ vs compañero)"),
        ("wet_weather_delta",     "asc",  "Especialistas en lluvia (neg = mejor en mojado)"),
        ("historical_dnf_rate",   "asc",  "Fiabilidad histórica 2023-2025 (menor DNF%)"),
        ("tyre_management_index", "desc", "Gestión de neumáticos (ganancia últimas 20% vueltas)"),
    ]
    print("\n   📊  TOP 5 POR MÉTRICA COMPORTAMENTAL 2023-2025:")
    for metric, direction, label in _metric_display:
        vals = [(c, d[metric]) for c, d in _raw_profiles.items()
                if d.get(metric) is not None]
        if not vals:
            continue
        top5 = sorted(vals, key=lambda x: x[1], reverse=(direction == "desc"))[:5]
        print(f"   {label}:")
        for rank, (code, val) in enumerate(top5, 1):
            print(f"     {rank}. {code:<5}  {val:+.3f}")
    print()

    # 8c. Press conference sentiment (NLP, cached by round)
    print("\n🎙️  Analizando sentimiento de rueda de prensa...")
    _sentiment_scores = fetch_press_conference_sentiment(next_round, next_circuit, schedule)
    if _sentiment_scores:
        _sent_sorted = sorted(_sentiment_scores.items(), key=lambda x: x[1], reverse=True)
        print("   📊  TOP 5 MÁS CONFIADOS:")
        for _sc, _sv in _sent_sorted[:5]:
            _bar = "█" * max(0, round(abs(_sv) * 10))
            _sign = "+" if _sv >= 0 else ""
            print(f"     {_sc:<5}  {_sign}{_sv:.2f}  {_bar}")
        print("   📊  BOTTOM 5 MENOS CONFIADOS:")
        for _sc, _sv in _sent_sorted[-5:]:
            _bar = "█" * max(0, round(abs(_sv) * 10))
            _sign = "+" if _sv >= 0 else ""
            print(f"     {_sc:<5}  {_sign}{_sv:.2f}  {_bar}")
    sentiment_df = (
        pd.DataFrame([{"code": c, "press_sentiment": v}
                       for c, v in _sentiment_scores.items()])
        if _sentiment_scores
        else pd.DataFrame(columns=["code", "press_sentiment"])
    )

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
        circuit_affinity_df=circuit_affinity_df,
        behavioral_df=behavioral_df,
        sentiment_df=sentiment_df,
        next_circuit_laps=next_circuit_laps,
        next_circuit_type=next_circuit_type,
        sprint_quali_df=sprint_quali_df,
        is_sprint_weekend=is_sprint_weekend,
    )
    # Rain probability available after weather fetch — store for score_manual()
    feat.attrs["rain_prob"] = float(weather.get("rain_prob", 0.0))

    # 8c. Driver-circuit compatibility — top-5 print + save embeddings to profile file
    if "compatibility_score" in feat.columns and feat["compatibility_score"].abs().sum() > 0:
        _circ_emb = feat.attrs.get("circuit_embedding", [])
        if _circ_emb:
            print(f"\n   🎯  Compatibilidad piloto-circuito ({next_circuit}):")
            print(f"      Circuit embedding → Speed={_circ_emb[0]:.2f}, "
                  f"Technical={_circ_emb[1]:.2f}, Endurance={_circ_emb[2]:.2f}")
        _c_top5 = (feat[["code", "compatibility_score"]]
                   .sort_values("compatibility_score", ascending=False)
                   .head(5))
        for _, _r in _c_top5.iterrows():
            _bar = "█" * max(0, int(float(_r["compatibility_score"]) * 20))
            print(f"      {_r['code']:<5}  {float(_r['compatibility_score']):+.3f}  {_bar}")
        # Save driver embeddings under "embeddings" key in profiles file
        _driver_embs = feat.attrs.get("driver_embeddings", {})
        if _driver_embs:
            try:
                import json as _json
                with open(PROFILES_FILE) as _pf:
                    _profiles_data = _json.load(_pf)
                _profiles_data["embeddings"] = _driver_embs
                with open(PROFILES_FILE, "w") as _pf:
                    _json.dump(_profiles_data, _pf, indent=2)
                print(f"      💾  Embeddings guardados → {PROFILES_FILE}")
            except Exception as _e:
                print(f"      ⚠  No se pudieron guardar embeddings: {_e}")

    # 8d. Corner mastery from telemetry — merge into feat + print top-5 table
    if not corner_tel_df.empty and "corner_mastery_score" in corner_tel_df.columns:
        feat = feat.merge(
            corner_tel_df[["code", "corner_mastery_score", "corner_detail"]],
            on="code", how="left")
        feat["corner_mastery_score"] = feat["corner_mastery_score"].fillna(
            feat["corner_mastery_score"].median()
            if feat["corner_mastery_score"].notna().any() else 0.0)
        # Print top-5 with per-corner breakdown
        _cm_top = (corner_tel_df
                   .sort_values("corner_mastery_score", ascending=False)
                   .head(5))
        print(f"\n🏎️  Corner mastery top 5 ({next_circuit}):")
        for _, _cmr in _cm_top.iterrows():
            _detail = _cmr.get("corner_detail", {}) or {}
            _parts  = ", ".join(
                f"{cn}: {ratio:.3f}"
                for cn, ratio in sorted(_detail.items(),
                                        key=lambda x: x[1])[:4]
            )
            print(f"   {str(_cmr['code']):<5}  {_cmr['corner_mastery_score']:.3f}"
                  f"  ({_parts})")
        # Drop corner_detail column — not needed in feat for modelling
        feat = feat.drop(columns=["corner_detail"], errors="ignore")
    else:
        feat["corner_mastery_score"] = 0.0

    # 8f. Tyre set inventory — merge score into feat + print compact table
    print("\n🔴  Inventario de neumáticos (sesiones de práctica completadas):")
    if not tyre_inv_df.empty:
        feat = feat.merge(tyre_inv_df[["code", "tyre_inventory_score"]],
                          on="code", how="left")
        feat["tyre_inventory_score"] = feat["tyre_inventory_score"].fillna(0.0)
        _avg_soft = tyre_inv_df["soft_new_remaining"].mean()
        for _, _tr in tyre_inv_df.sort_values("tyre_inventory_score",
                                               ascending=False).iterrows():
            _flag = "  ⚠ BAJO" if _tr["soft_new_remaining"] < _avg_soft - 1.0 else ""
            print(f"   {str(_tr['code']):<5}: "
                  f"{int(_tr['soft_new_remaining'])} soft new, "
                  f"{int(_tr['medium_new_remaining'])} med new, "
                  f"{int(_tr['hard_new_remaining'])} hard new"
                  f"{_flag}")
        print(f"   (field avg: {_avg_soft:.1f} soft new remaining)")
    else:
        feat["tyre_inventory_score"] = 0.0
        print("   ⏳  Sin datos de práctica completada — inventario neutral (0.0)")

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

    # Determine online-learning mode for this run
    _model_meta     = priors.get("_model_meta", {})
    _model_last_rnd = _model_meta.get("last_trained_round", -1)
    _last_comp_rnd  = max(completed) if completed else 0
    # Warm start when the saved model already reflects this same round (re-run)
    _warm_race_df = (race_df[race_df["round"] == _last_comp_rnd].copy()
                     if (completed
                         and _model_last_rnd == _last_comp_rnd
                         and os.path.exists(XGB_MODEL_FILE))
                     else None)
    _train_label = "warm start" if _warm_race_df is not None else "full retrain"
    print(f"\n💾  Modo modelo: {_train_label} "
          f"(guardado R{_model_last_rnd} → actual R{_last_comp_rnd})")

    # Raw per-model score series — captured for epistemic uncertainty computation
    _xgb_score_raw  = None
    _lgbm_score_raw = None
    gp_result       = None
    gp_std          = None

    if use_xgb:
        print(f"\n🤖  Activando XGBoost + LightGBM + GP — Quali+Race ({len(completed)} carreras ≥ {XGB_MIN_RACES})...")
        xgb_result  = score_xgboost(feat, race_df, quali_df,
                                    warm_start_file=XGB_MODEL_FILE,
                                    warm_race_df=_warm_race_df)
        lgbm_result = score_lightgbm(feat, race_df, quali_df,
                                     warm_start_file=LGBM_MODEL_FILE,
                                     warm_race_df=_warm_race_df)
        gp_result   = score_gaussian_process(feat, race_df, quali_df)

        if xgb_result is not None and lgbm_result is not None:
            xgb_score,  xgb_imp  = xgb_result
            lgbm_score, lgbm_imp = lgbm_result
            _xgb_score_raw  = xgb_score   # capture before blending
            _lgbm_score_raw = lgbm_score
            # ── Stacking meta-model weights (XGB vs LGBM ratio) ──────────────
            _meta    = priors.get("stacking_meta") if priors else None
            _w_xgb, _w_lgbm = 0.5, 0.5
            _meta_label = ""
            if _meta and _meta.get("n_rounds", 0) >= _META_MIN_ROUNDS:
                _mc  = _meta["coef"]
                _mct = _meta.get("ctypes", _META_CTYPES)
                _ci  = (_mct.index(next_circuit_type)
                        if next_circuit_type in _mct else len(_mct) - 1)
                _raw_x, _raw_l = _mc[_ci * 2], _mc[_ci * 2 + 1]
                _ea = np.exp(_raw_x - max(_raw_x, _raw_l))
                _eb = np.exp(_raw_l - max(_raw_x, _raw_l))
                _w_xgb  = float(_ea / (_ea + _eb))
                _w_lgbm = 1.0 - _w_xgb
                _meta_label = f"@ {next_circuit_type}"
                print(f"   🔬  Meta-modelo: XGB peso {_w_xgb:.2f}, LGBM peso {_w_lgbm:.2f}"
                      f" {_meta_label}")
            # ── 3-way blend when GP is available; else 2-way stacking ────────
            if gp_result is not None:
                gp_score, gp_std = gp_result
                _base    = 1.0 - _GP_WEIGHT
                _wf_xgb  = round(_base * _w_xgb,  3)
                _wf_lgbm = round(_base * _w_lgbm, 3)
                feat["xgb_pos"] = (_wf_xgb  * xgb_score
                                 + _wf_lgbm * lgbm_score
                                 + _GP_WEIGHT * gp_score)
                print(f"   🌊  3-way blend: XGB {_wf_xgb:.2f}, LGBM {_wf_lgbm:.2f}, "
                      f"GP {_GP_WEIGHT:.2f}")
                model_used = (f"3-Way XGB({_wf_xgb:.2f})/LGBM({_wf_lgbm:.2f})"
                              f"/GP({_GP_WEIGHT:.2f}) {_meta_label}").strip()
            else:
                feat["xgb_pos"] = _w_xgb * xgb_score + _w_lgbm * lgbm_score
                model_used = (
                    f"Stacking Meta-Modelo XGB({_w_xgb:.2f})/LGBM({_w_lgbm:.2f}) {_meta_label}".strip()
                    if _meta_label else "Quali Model + Race Model Ensemble (XGBoost/LightGBM)")
            all_keys    = set(xgb_imp) | set(lgbm_imp)
            xgb_weights = {k: _w_xgb  * xgb_imp.get(k, 0)
                              + _w_lgbm * lgbm_imp.get(k, 0)
                           for k in all_keys}
            scores     = score_manual(feat, xgb_weights=xgb_weights)
        elif xgb_result is not None:
            xgb_score, xgb_weights = xgb_result
            feat["xgb_pos"] = xgb_score
            scores     = score_manual(feat, xgb_weights=xgb_weights)
            model_used = "XGBoost Quali+Race + Feedback Loop"
        elif lgbm_result is not None:
            lgbm_score, lgbm_imp = lgbm_result
            lgbm_total  = sum(lgbm_imp.values()) or 1
            xgb_weights = {k: v / lgbm_total for k, v in lgbm_imp.items()}
            feat["xgb_pos"] = lgbm_score
            scores     = score_manual(feat, xgb_weights=xgb_weights)
            model_used = "LightGBM Quali+Race + Feedback Loop"
        else:
            scores     = score_manual(feat)
            model_used = "Pesos manuales (modelos fallaron)"
    else:
        remaining = XGB_MIN_RACES - len(completed)
        print(f"\n⚖   Usando modelo de pesos ({len(completed)}/{XGB_MIN_RACES} para ensemble)...")
        # Pre-train both models to extract importances for the feedback loop
        xgb_result  = score_xgboost(feat, race_df, quali_df)
        lgbm_result = score_lightgbm(feat, race_df, quali_df)
        if xgb_result is not None and lgbm_result is not None:
            _, xgb_imp  = xgb_result
            _, lgbm_imp = lgbm_result
            all_keys    = set(xgb_imp) | set(lgbm_imp)
            xgb_weights = {k: 0.5 * xgb_imp.get(k, 0) + 0.5 * lgbm_imp.get(k, 0)
                           for k in all_keys}
            print(f"   💡  Importancias ensemble extraídas ({len(xgb_weights)} features)")
        elif xgb_result is not None:
            _, xgb_weights = xgb_result
            print(f"   💡  Importancias XGBoost extraídas ({len(xgb_weights)} features)")
        scores     = score_manual(feat, xgb_weights=xgb_weights if xgb_weights else None)
        model_used = f"Pesos manuales{' + Feedback Ensemble' if xgb_weights else ''} (faltan {remaining} carreras)"

    # Persist model-round metadata so subsequent runs know whether to warm-start
    if completed and use_xgb and (xgb_result is not None or lgbm_result is not None):
        priors["_model_meta"] = {
            "last_trained_round": max(completed),
            "mode": _train_label,
        }
        save_priors(priors)

    # Epistemic uncertainty — model disagreement (XGB vs LGBM normalized scores)
    if _xgb_score_raw is not None and _lgbm_score_raw is not None:
        _xs_min, _xs_rng = _xgb_score_raw.min(),  (_xgb_score_raw.max()  - _xgb_score_raw.min())
        _ls_min, _ls_rng = _lgbm_score_raw.min(), (_lgbm_score_raw.max() - _lgbm_score_raw.min())
        _xs_norm = (_xgb_score_raw  - _xs_min) / max(_xs_rng, 1e-9)
        _ls_norm = (_lgbm_score_raw - _ls_min) / max(_ls_rng, 1e-9)
        feat["epistemic_unc"] = (_xs_norm - _ls_norm).abs().reindex(feat.index).fillna(0.0)
    else:
        feat["epistemic_unc"] = 0.0

    # GP uncertainty — sparse-data uncertainty from Gaussian Process prediction std
    feat["gp_uncertainty"] = (gp_std.reindex(feat.index).fillna(0.0)
                               if gp_std is not None else 0.0)

    # 9c. Feature importance tracking and comparison
    if xgb_weights:
        _total_w  = sum(xgb_weights.values()) or 1
        _norm_imp = {k: round(v / _total_w, 4) for k, v in xgb_weights.items()}
        _top10    = dict(sorted(_norm_imp.items(), key=lambda x: x[1], reverse=True)[:10])
        _imp_round = max(completed) if completed else 0

        _imp_hist = priors.get("feature_importance_history", [])
        _prev_imp = _imp_hist[-1]["importances"] if _imp_hist else {}

        print("\n   📊  Importancias blend (XGB+LGBM):")
        for fname, fval in _top10.items():
            prev = _prev_imp.get(fname)
            if prev is not None:
                delta = fval - prev
                if delta > 0.001:
                    arrow = f"(↑ from {prev:.3f})"
                elif delta < -0.001:
                    arrow = f"(↓ from {prev:.3f})"
                else:
                    arrow = "(→ stable)"
            else:
                arrow = "(primera vez)" if _prev_imp else ""
            print(f"      {fname:<26} {fval:.3f}  {arrow}")

        top_feat, top_val = next(iter(_top10.items()))
        print(f"   🏆  Top feature esta carrera: {top_feat} ({top_val:.3f})")

        # Append to history, keep last 5 rounds, save immediately
        _imp_hist.append({"round": _imp_round, "importances": _top10})
        priors["feature_importance_history"] = _imp_hist[-5:]
        save_priors(priors)

    # 9d. LSTM momentum blend (active at >= LSTM_MIN_RACES)
    print(f"\n🧠  LSTM momentum model...")
    lstm_result = score_lstm(feat, race_df, completed)
    if lstm_result is not None:
        # Normalise existing scores to [0, 1] before blending
        _s_min, _s_rng = scores.min(), scores.max() - scores.min()
        _scores_norm = (scores - _s_min) / max(float(_s_rng), 1e-9)
        # lstm_result is already [0, 1]
        scores = 0.85 * _scores_norm + 0.15 * lstm_result.reindex(scores.index).fillna(0.5)
        feat["lstm_score"] = lstm_result.reindex(feat.index).fillna(0.0).values
        print(f"   ✅  LSTM blend applied: 0.85 × ensemble + 0.15 × LSTM")
    else:
        feat["lstm_score"] = np.nan

    # 9e. GNN overtaking model (active at >= LSTM_MIN_RACES)
    print(f"\n🕸️  GNN overtaking model...")
    gnn_result = score_gnn_overtaking(
        feat, race_df, completed, next_circuit_type=next_circuit_type
    )
    if gnn_result is not None:
        _gnn_scores, gnn_overtake_matrix = gnn_result
        feat["gnn_score"] = _gnn_scores.reindex(feat.index).fillna(0.0).values
        print(f"   ✅  GNN: {len(gnn_overtake_matrix)} pairwise lap-1 probs wired to MC")
    else:
        feat["gnn_score"]   = np.nan
        gnn_overtake_matrix = None

    # 10. Aplicar priors bayesianos al score
    if priors and completed:
        scores = apply_bayesian_priors(scores, feat, priors)

    # 10b. Convertir a probabilidad
    feat["raw_score"] = scores.values                           # pre-softmax for calibration
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

    # 12b. Pirelli compound selection → circuit_compound_softness
    # Scraped from public sources (inconsistent HTML, not a clean API).
    # Falls back to CIRCUIT_COMPOUND_DEFAULTS_2026 if live source unavailable.
    # Used only as a multiplier on undercut_prob inside MC — low-weight signal.
    print("\n🏎️  Pirelli compound selection...")
    _next_race_name = next_info.get("name", "") if hasattr(next_info, "get") else ""
    _compound_data  = fetch_compound_selection(next_circuit, next_round, _next_race_name)
    _compound_softness = _compound_data["softness"]

    # 13. Simulación Monte Carlo
    # Build per-driver tyre inventory score dict for MC undercut weighting
    _tyre_inv_scores: "dict | None" = None
    if not tyre_inv_df.empty:
        _tyre_inv_scores = dict(zip(tyre_inv_df["code"],
                                    tyre_inv_df["tyre_inventory_score"]))

    mc_df = monte_carlo_simulation(
        feat                  = scored,
        base_scores           = scored["raw_score"],
        reliability           = reliability,
        weather               = weather,
        n_sims                = 10_000,
        circuit_name          = next_circuit,
        gnn_overtake          = gnn_overtake_matrix,
        compound_softness     = _compound_softness,
        tyre_inventory_scores = _tyre_inv_scores,
    )

    # Aleatoric uncertainty — inherent randomness from MC spread relative to win%
    if mc_df is not None and not mc_df.empty:
        _ale = mc_df[["code", "win_mc_pct", "p10_win_pct", "p90_win_pct"]].copy()
        _ale["aleatoric_unc"] = (
            (_ale["p90_win_pct"] - _ale["p10_win_pct"]) /
            _ale["win_mc_pct"].clip(lower=0.1)
        ).round(3)
        scored = scored.merge(_ale[["code", "aleatoric_unc"]], on="code", how="left")
        scored["aleatoric_unc"] = scored["aleatoric_unc"].fillna(0.0)
    else:
        scored["aleatoric_unc"] = 0.0

    # 14. Actualizar priors bayesianos con última carrera completada
    # Guard: only update once per race — prevent repeated dev-run compounding
    if completed:
        last_rnd     = max(completed)
        already_done = priors.get("_meta", {}).get("last_updated_round", -1)
        if already_done != last_rnd:
            print(f"\n🧠  Actualizando priors bayesianos con carrera {last_rnd}...")
            priors = update_bayesian_priors(priors, scored, race_df, last_rnd)
            priors["_meta"] = {"last_updated_round": last_rnd}
            save_priors(priors)
        else:
            print(f"\n🧠  Priors ya actualizados para carrera {last_rnd} — sin cambios")
        print(f"   💾  Priors guardados en: {PRIORS_FILE}")
        # Mostrar pilotos con mayor ajuste
        adjustments = [(c, d["score_multiplier"]) for c, d in priors.items()
                       if isinstance(d, dict) and d.get("races_tracked", 0) >= 1]
        adjustments.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)
        if adjustments[:3]:
            print("   📊  Mayores ajustes acumulados:")
            for code, mult in adjustments[:3]:
                direction = "⬆ subestimado" if mult > 1.0 else "⬇ sobreestimado"
                print(f"      {code}: ×{mult:.3f} ({direction})")

    # 14b. Brier score calibration tracking
    # The CSV on disk still holds predictions for last_rnd at this point —
    # read it NOW before step "13 Guardar CSV" overwrites it with next-race predictions.
    if completed:
        _last = max(completed)
        _brier_done = any(h.get("round") == _last for h in priors.get("brier_history", []))
        if not _brier_done and os.path.exists(OUTPUT_CSV):
            try:
                _old = pd.read_csv(OUTPUT_CSV)
                _old_rnd = int(_old["round_num"].iloc[0]) if "round_num" in _old.columns else -1
                if _old_rnd == _last and "win_mc_pct" in _old.columns:
                    _r_act   = race_df[race_df["round"] == _last]
                    _winners = _r_act[_r_act["pos"] == 1]["code"].values
                    if len(_winners) > 0:
                        _actual = _winners[0]
                        bs      = brier_score(_old, _actual)
                        _pred_w = _old.loc[_old["win_mc_pct"].idxmax(), "code"]
                        _cname  = _old["race_name"].iloc[0] if "race_name" in _old.columns else "?"
                        _hist   = priors.get("brier_history", [])
                        _hist.append({
                            "round"            : _last,
                            "circuit"          : _cname,
                            "brier_score"      : bs,
                            "predicted_winner" : _pred_w,
                            "actual_winner"    : _actual,
                        })
                        priors["brier_history"] = _hist
                        save_priors(priors)
                        print(f"\n📐  Brier Score R{_last} ({_cname}): {bs:.4f}  "
                              f"(predicho: {_pred_w}  |  ganador real: {_actual})")
            except Exception as e:
                print(f"   ⚠  Brier score: {e}")

        # One-time R8 bootstrap — CSV was overwritten before this tracker was added.
        # Approximated win% from the stable pre-race prediction run for Austrian GP.
        _hist = priors.get("brier_history", [])
        if not any(h.get("round") == 8 for h in _hist) and _last >= 8:
            _r8_act = race_df[race_df["round"] == 8]
            _r8_win = _r8_act[_r8_act["pos"] == 1]["code"].values
            if len(_r8_win) > 0:
                _R8_KNOWN = {"RUS": 19.8, "ANT": 13.1, "HAM": 10.7,
                             "LEC": 7.2,  "NOR": 6.0,  "PIA": 5.8}
                _codes = scored["code"].tolist()
                _rem   = (100 - sum(_R8_KNOWN.values())) / max(len(_codes) - len(_R8_KNOWN), 1)
                _r8_df = pd.DataFrame([
                    {"code": c, "win_mc_pct": _R8_KNOWN.get(c, _rem)} for c in _codes
                ])
                bs8 = brier_score(_r8_df, _r8_win[0])
                _hist.append({
                    "round"            : 8,
                    "circuit"          : "Austrian Grand Prix",
                    "brier_score"      : bs8,
                    "predicted_winner" : "RUS",
                    "actual_winner"    : str(_r8_win[0]),
                    "note"             : "bootstrapped — CSV overwritten before tracker added",
                })
                priors["brier_history"] = _hist
                save_priors(priors)
                print(f"\n📐  Brier Score R8 (Austrian Grand Prix) [bootstrapped]: {bs8:.4f}  "
                      f"(predicho: RUS  |  ganador real: {_r8_win[0]})")

    # 14c. LOO cross-validation + calibration curve
    if len(completed) >= 4:
        print(f"\n🔄  Corriendo validación LOO ({len(completed)} carreras)...")
        loo_results, loo_driver_preds = cross_validate_season(
            feat, race_df, quali_df, schedule)
        if loo_results:
            print_loo_validation(loo_results)
            priors["loo_validation"] = loo_results
        if loo_driver_preds:
            cal = calibration_analysis(loo_driver_preds)
            print_calibration(cal)
            priors["calibration"] = cal
            # 14d. Fit/update stacking meta-model from per-model LOO predictions
            _meta_new = fit_stacking_meta_model(loo_driver_preds)
            if _meta_new:
                priors["stacking_meta"] = _meta_new
                print(f"   💾  Meta-modelo guardado "
                      f"({_meta_new['n_rounds']} rondas, R{_meta_new['trained_round']})")
        if loo_results or loo_driver_preds:
            save_priors(priors)

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
        "avg_pitstop", "penalty_count", "sc_gain_avg", "raw_score", "win_pct",
        "win_mc_pct", "p10_win_pct", "p90_win_pct",
        "podium_mc_pct", "finish_mc_pct", "avg_mc_pos",
        "p10_pos", "p90_pos",
        "compound_score", "circuit_score", "soft_delta", "medium_delta", "hard_delta",
        "circuit_type_score", "quali_gap_teammate", "fp2_longrun_delta",
        "momentum_pos", "last_race_pts", "last_race_pos", "constructor_momentum",
        "circuit_affinity",
        "mechanical_risk", "accident_risk", "dominant_failure", "team_mech_rate",
        "sector_balance",
        "quali_pos_next", "quali_time_next", "fp_next_delta", "fp2_next_longrun",
        "soft_pace_delta", "medium_pace_delta", "hard_pace_delta", "compound_preference",
        "tyre_deg_rate", "corner_profile_score",
        "race_sim_delta", "race_sim_deg",
        "streak_score", "post_dnf_bounce", "championship_pressure",
        "epistemic_unc", "aleatoric_unc", "gp_uncertainty",
        "overtaking_ability", "quali_consistency", "wet_weather_delta",
        "historical_dnf_rate", "tyre_management_index",
        "compatibility_score",
        "press_sentiment",
        "anomaly_score", "sandbagging_flag", "struggling_flag", "race_threat_flag",
        "sq_pos_next", "sprint_quali_delta",
        "lstm_score", "gnn_score",
        "tyre_inventory_score",
        "corner_mastery_score",
    ] if c in scored.columns]
    scored[save_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"💾  Resultados guardados en: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
    

#-------run script using these commands:----------
#-------source .venv/bin/activate-----------------
#-------python f1_2026_prediction3.py-------------