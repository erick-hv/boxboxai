"""
F1 2026 — Diagnóstico de Datos OpenF1
======================================
Corre este script ANTES de hacer cambios al predictor.
Te dice exactamente qué datos tienes disponibles en OpenF1
para cada sesión de las rondas 1 y 2 de 2026.

Uso:
    python f1_diagnostico_openf1.py
"""

import requests
import json
import time
from datetime import datetime

BASE = "https://api.openf1.org/v1"
SEASON = 2026

def get(endpoint, params=None):
    try:
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  HTTP {r.status_code} en /{endpoint}")
        return []
    except Exception as e:
        print(f"  Error: {e}")
        return []

def check_endpoint(name, endpoint, params, key_cols=None):
    """Chequea un endpoint y reporta qué columnas tiene y cuántos registros."""
    data = get(endpoint, params)
    time.sleep(0.4)
    if not data:
        print(f"    ❌  {name}: SIN DATOS")
        return False, []
    cols = list(data[0].keys()) if data else []
    n    = len(data)
    # Verificar columnas clave
    if key_cols:
        missing = [c for c in key_cols if c not in cols]
        extra   = [c for c in key_cols if c in cols]
        status  = "✅" if not missing else "⚠ "
        print(f"    {status}  {name}: {n} registros")
        print(f"         Columnas clave presentes : {extra}")
        if missing:
            print(f"         Columnas clave AUSENTES  : {missing}")
    else:
        print(f"    ✅  {name}: {n} registros | Cols: {', '.join(cols[:8])}")
    return True, cols

# ─── 1. SESIONES ────────────────────────────────────────────
print("\n" + "="*60)
print("  📋  DIAGNÓSTICO OPENF1 — F1 2026")
print("="*60)
print(f"\n  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  API:   {BASE}\n")

print("1️⃣  SESIONES DISPONIBLES EN 2026")
print("-"*60)
sessions_raw = get("sessions", {"year": SEASON})
time.sleep(0.4)

if not sessions_raw:
    print("  ❌  NO HAY SESIONES — OpenF1 no tiene datos de 2026")
    exit()

# Organizar sesiones por meeting
meetings = {}
for s in sessions_raw:
    mk = s.get("meeting_key", 0)
    meetings.setdefault(mk, []).append(s)

# Ordenar por fecha
sorted_meetings = sorted(meetings.items(),
                          key=lambda x: x[1][0].get("date_start",""))

# Mostrar solo rondas 1 y 2 (los primeros 2 meetings de carreras)
race_meetings = [(mk, ss) for mk, ss in sorted_meetings
                 if any(s.get("session_name") == "Race" for s in ss)]

print(f"  Total sesiones 2026: {len(sessions_raw)}")
print(f"  Meetings con Race:   {len(race_meetings)}\n")

session_map = {}  # {(round_num, session_code): session_key}
SESSION_CODE_MAP = {
    "Race": "R", "Qualifying": "Q", "Sprint": "S",
    "Sprint Qualifying": "SQ", "Practice 1": "FP1",
    "Practice 2": "FP2", "Practice 3": "FP3",
}

for rnd_idx, (mk, sessions) in enumerate(race_meetings[:3], 1):
    race_sess = next((s for s in sessions if s.get("session_name") == "Race"), sessions[0])
    circuit   = race_sess.get("circuit_short_name", "?")
    country   = race_sess.get("country_name", "?")
    print(f"  🏎  RONDA {rnd_idx}: {circuit} ({country})")
    sessions_sorted = sorted(sessions, key=lambda x: x.get("date_start",""))
    for s in sessions_sorted:
        sk    = s.get("session_key")
        sname = s.get("session_name","?")
        scode = SESSION_CODE_MAP.get(sname, sname)
        date  = s.get("date_start","?")[:10]
        print(f"       {scode:4s} | key={sk} | {date} | {sname}")
        session_map[(rnd_idx, scode)] = sk
    print()

# ─── 2. DATOS DE VUELTA POR SESIÓN ─────────────────────────
print("2️⃣  DISPONIBILIDAD DE DATOS POR SESIÓN")
print("-"*60)

priority_sessions = [
    (1, "R",   ["lap_duration","duration_sector_1","duration_sector_2","duration_sector_3","compound","stint_number","is_pit_out_lap"]),
    (1, "Q",   ["lap_duration","duration_sector_1","duration_sector_2","duration_sector_3"]),
    (1, "FP1", ["lap_duration","duration_sector_1"]),
    (1, "FP2", ["lap_duration","stint_number"]),
    (1, "FP3", ["lap_duration"]),
    (2, "R",   ["lap_duration","duration_sector_1","duration_sector_2","duration_sector_3","compound","stint_number"]),
    (2, "Q",   ["lap_duration","duration_sector_1","duration_sector_2","duration_sector_3"]),
    (2, "FP1", ["lap_duration"]),
    (2, "S",   ["lap_duration"]),   # Sprint
    (2, "SQ",  ["lap_duration"]),   # Sprint Qualifying
]

laps_availability = {}
for (rnd, scode, key_cols) in priority_sessions:
    sk = session_map.get((rnd, scode))
    if not sk:
        print(f"    ⚫  R{rnd} {scode:4s}: session_key no encontrado")
        laps_availability[(rnd, scode)] = False
        continue
    print(f"\n  📍 Ronda {rnd} — {scode} (session_key={sk})")
    ok, cols = check_endpoint(
        f"Laps R{rnd}-{scode}",
        "laps",
        {"session_key": sk},
        key_cols
    )
    laps_availability[(rnd, scode)] = ok

# ─── 3. OTROS ENDPOINTS CRÍTICOS ────────────────────────────
print("\n\n3️⃣  OTROS ENDPOINTS CRÍTICOS (usando R1 Race)")
print("-"*60)

r1_race_key = session_map.get((1, "R"))
r2_race_key = session_map.get((2, "R"))

if r1_race_key:
    print(f"\n  📍 Race Australia (session_key={r1_race_key})")

    check_endpoint("Stints",        "stints",        {"session_key": r1_race_key},
                   ["compound","lap_start","lap_end","stint_number","tyre_age_at_start"])
    check_endpoint("Position",      "position",      {"session_key": r1_race_key},
                   ["position","date"])
    check_endpoint("Pit stops",     "pit",           {"session_key": r1_race_key},
                   ["pit_duration","lap_number"])
    check_endpoint("Race control",  "race_control",  {"session_key": r1_race_key},
                   ["flag","message","date"])
    check_endpoint("Intervals",     "intervals",     {"session_key": r1_race_key},
                   ["gap_to_leader","interval"])
    check_endpoint("Car data (sample)", "car_data",  {"session_key": r1_race_key, "driver_number": 63},
                   ["speed","throttle","brake","n_gear","rpm"])
    check_endpoint("Weather",       "weather",       {"session_key": r1_race_key},
                   ["track_temperature","rainfall","humidity"])

# ─── 4. SAMPLE DATA ─────────────────────────────────────────
print("\n\n4️⃣  MUESTRA DE DATOS REALES (primer lap de Australia Race)")
print("-"*60)

if r1_race_key:
    sample = get("laps", {"session_key": r1_race_key, "driver_number": 63, "lap_number": 1})
    time.sleep(0.4)
    if sample:
        print("\n  Columnas disponibles en /laps:")
        lap = sample[0]
        for k, v in lap.items():
            status = "✅" if v is not None else "⚪ (null)"
            print(f"    {k:<35} {status}  {v}")
    else:
        print("  ❌  Sin datos de laps para Russell R1 L1")

# ─── 5. STINTS SAMPLE ───────────────────────────────────────
print("\n\n5️⃣  MUESTRA DE STINTS (Australia Race)")
print("-"*60)

if r1_race_key:
    stints = get("stints", {"session_key": r1_race_key})
    time.sleep(0.4)
    if stints:
        print(f"\n  {len(stints)} stints totales. Primeros 3:")
        for s in stints[:3]:
            print(f"    {s}")
    else:
        print("  ❌  Sin datos de stints")

# ─── 6. RESUMEN Y RECOMENDACIONES ───────────────────────────
print("\n\n" + "="*60)
print("  📊  RESUMEN Y RECOMENDACIONES")
print("="*60)

available = [(k, v) for k, v in laps_availability.items() if v]
missing   = [(k, v) for k, v in laps_availability.items() if not v]

print(f"\n  ✅  Sesiones CON datos de laps: {len(available)}")
for (rnd, scode), _ in available:
    print(f"       Ronda {rnd} — {scode}")

print(f"\n  ❌  Sesiones SIN datos de laps: {len(missing)}")
for (rnd, scode), _ in missing:
    print(f"       Ronda {rnd} — {scode}")

if len(available) >= 4:
    print("\n  💡  CONCLUSIÓN: OpenF1 tiene suficientes datos para el modelo.")
    print("      El predictor puede usar tiempos de vuelta, sectores, y stints.")
elif len(available) >= 1:
    print("\n  ⚠   CONCLUSIÓN: OpenF1 tiene datos parciales.")
    print("      El modelo puede correr pero con features incompletas.")
else:
    print("\n  ❌  CONCLUSIÓN: OpenF1 NO tiene datos de laps para 2026.")
    print("      El modelo solo puede usar datos de resultados (API Jolpica).")

print("\n" + "="*60)
print("  Copia y pega este output completo para diagnóstico.")
print("="*60 + "\n")