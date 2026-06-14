#!/usr/bin/env python3
"""
BoxBoxAI — 2026 Memory Builder
================================
Run this on your Mac ONCE to load all completed 2026 races
with full FastF1 telemetry into f1_memory_2026.json.

Then commit f1_memory_2026.json to the repo.
Railway will have ground truth for the entire season.

Usage:
    cd "/Users/erick.hernandez/Documents/Erick stuff/F1_Agent"
    python3 build_2026_memory.py

Requirements: pip3 install fastf1 requests pandas numpy
"""

import json, requests, time
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────
SEASON       = 2026
JOLPICA      = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE  = "https://api.openf1.org/v1"
OUTPUT_FILE  = Path(__file__).parent / "f1_memory_2026.json"
CACHE_DIR    = Path(__file__).parent / "f1_cache"
CACHE_DIR.mkdir(exist_ok=True)

# 2026 race calendar — update dates as season progresses
RACE_CALENDAR = [
    (1,  "Australian GP",    "2026-03-15"),
    (2,  "Chinese GP",       "2026-03-22"),
    (3,  "Japanese GP",      "2026-04-06"),
    (4,  "Miami GP",         "2026-05-04"),
    (5,  "Canadian GP",      "2026-05-24"),
    (6,  "Monaco GP",        "2026-06-07"),
    (7,  "Spanish GP",       "2026-06-14"),
    (8,  "Austrian GP",      "2026-06-28"),
    (9,  "British GP",       "2026-07-05"),
    (10, "Belgian GP",       "2026-07-19"),
    (11, "Hungarian GP",     "2026-07-26"),
    (12, "Dutch GP",         "2026-08-23"),
    (13, "Italian GP",       "2026-09-06"),
    (14, "Singapore GP",     "2026-09-20"),
    (15, "Azerbaijan GP",    "2026-09-27"),
    (16, "US GP",            "2026-10-18"),
    (17, "Mexico City GP",   "2026-10-25"),
    (18, "São Paulo GP",     "2026-11-08"),
    (19, "Las Vegas GP",     "2026-11-21"),
    (20, "Qatar GP",         "2026-11-29"),
    (21, "Abu Dhabi GP",     "2026-12-06"),
]

SPRINT_ROUNDS = {2, 4, 8, 16, 18, 20}

TODAY = datetime.now().strftime("%Y-%m-%d")

# ── Helpers ───────────────────────────────────────────────────
def safe_get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(1)
        except Exception as e:
            print(f"   ⚠ {url}: {e}")
            time.sleep(2)
    return None

def get_driver_code(r):
    d = r.get("Driver", {})
    return d.get("code", d.get("familyName","???")[:3].upper())

# ── FastF1 session loader ─────────────────────────────────────
def load_ff1_session(round_num, session_type):
    try:
        import fastf1
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        sess = fastf1.get_session(SEASON, round_num, session_type)
        sess.load(laps=True, telemetry=False, weather=True, messages=True)
        if sess.laps is None or len(sess.laps) == 0:
            return None
        print(f"   ✅ FastF1: {session_type} R{round_num} loaded "
              f"({len(sess.laps)} laps)")
        return sess
    except Exception as e:
        print(f"   ℹ FastF1 {session_type} R{round_num}: {e}")
        return None

# ── Extract rich data from FastF1 session ─────────────────────
def extract_race_data(sess, round_num):
    """Extracts everything useful from a FastF1 race session."""
    import pandas as pd
    import numpy as np
    data = {}

    try:
        laps  = sess.laps
        valid = laps[laps["LapTime"].notna()].copy()
        valid["LapTime_s"] = valid["LapTime"].dt.total_seconds()
        valid = valid[valid["LapTime_s"] > 0]

        # ── Full classification ────────────────────────────
        results = sess.results
        if results is not None and not results.empty:
            code_col = next((c for c in
                ["Abbreviation","Driver"] if c in results.columns), None)
            pos_col  = next((c for c in
                ["ClassifiedPosition","Position"] if c in results.columns), None)
            status_col = next((c for c in
                ["Status","status"] if c in results.columns), None)
            if code_col and pos_col:
                ordered = results.copy()
                ordered[pos_col] = pd.to_numeric(
                    ordered[pos_col], errors="coerce")
                ordered = ordered.sort_values(pos_col)
                data["full_classification"] = [
                    f"P{int(row[pos_col])}:{row[code_col]}"
                    for _, row in ordered.iterrows()
                    if pd.notna(row[pos_col])]
                data["classified_count"] = len(data["full_classification"])

                # DNFs from status
                if status_col:
                    finish_statuses = {
                        "finished","+1 lap","+2 laps","+3 laps",
                        "+4 laps","+5 laps","+6 laps","+7 laps",
                        "1 lap","2 laps",
                    }
                    dnf_rows = ordered[
                        ~ordered[status_col].str.lower().str.strip()
                        .isin(finish_statuses)
                    ]
                    data["dnfs"] = [
                        f"{row[code_col]}({row[status_col]})"
                        for _, row in dnf_rows.iterrows()
                        if pd.notna(row.get(status_col,""))]

        # ── Fastest lap ────────────────────────────────────
        if not valid.empty:
            fl_idx  = valid["LapTime_s"].idxmin()
            fl_row  = valid.loc[fl_idx]
            fl_s    = fl_row["LapTime_s"]
            mins    = int(fl_s // 60)
            secs    = fl_s % 60
            data["fastest_lap"]      = str(fl_row.get("Driver",""))
            data["fastest_lap_time"] = f"{mins}:{secs:06.3f}"
            data["fastest_lap_no"]   = int(fl_row.get("LapNumber",0))

        # ── Top 5 lap times ────────────────────────────────
        best = (valid.groupby("Driver")["LapTime_s"]
                .min().sort_values().head(5))
        ref  = best.iloc[0] if not best.empty else 1
        top5 = []
        for drv, t in best.items():
            mins  = int(t // 60)
            secs  = t % 60
            delta = t - ref
            top5.append({
                "driver": str(drv),
                "time":   f"{mins}:{secs:06.3f}",
                "delta":  f"+{delta:.3f}s" if delta > 0 else "leader",
            })
        data["top5_lap_times"] = top5

        # ── Sector bests ───────────────────────────────────
        sector_bests = {}
        for s_col, s_name in [
            ("Sector1Time","S1"),
            ("Sector2Time","S2"),
            ("Sector3Time","S3"),
        ]:
            if s_col in laps.columns:
                sv = laps[laps[s_col].notna()].copy()
                if not sv.empty:
                    sv[f"{s_col}_s"] = sv[s_col].dt.total_seconds()
                    sv = sv[sv[f"{s_col}_s"] > 0]
                    if not sv.empty:
                        best_idx  = sv[f"{s_col}_s"].idxmin()
                        best_row  = sv.loc[best_idx]
                        t = best_row[f"{s_col}_s"]
                        sector_bests[s_name] = {
                            "driver": str(best_row.get("Driver","")),
                            "time":   f"{t:.3f}s",
                        }
        data["sector_bests"] = sector_bests

        # ── Tyre strategies ────────────────────────────────
        tyre_data = {}
        if "Compound" in laps.columns and "Stint" in laps.columns:
            for (drv, stint), grp in laps.groupby(["Driver","Stint"]):
                compound = str(grp["Compound"].iloc[0]) if not grp.empty else "?"
                n_laps   = len(grp)
                if str(drv) not in tyre_data:
                    tyre_data[str(drv)] = []
                tyre_data[str(drv)].append({
                    "compound": compound, "laps": n_laps})
        data["tyre_stints"] = tyre_data

        # Strategy summary string
        strategies = []
        for drv, stints in list(tyre_data.items())[:10]:
            strat = "→".join(
                f"{s['compound'][:1]}({s['laps']})" for s in stints)
            strategies.append(f"{drv}:{strat}")
        data["tyre_strategy_summary"] = " | ".join(strategies[:8])

        # Pit stop counts
        if "PitInTime" in laps.columns:
            pit_laps = laps[laps["PitInTime"].notna()]
            data["pit_stop_counts"] = {
                str(drv): len(grp)
                for drv, grp in pit_laps.groupby("Driver")}

        # ── Weather ────────────────────────────────────────
        try:
            weather = sess.weather_data
            if weather is not None and not weather.empty:
                data["weather"] = {
                    "air_temp":   round(float(weather["AirTemp"].mean()),1),
                    "track_temp": round(float(weather["TrackTemp"].mean()),1),
                    "rainfall":   bool(weather["Rainfall"].any()),
                    "humidity":   round(float(weather["Humidity"].mean()),1),
                    "wind_speed": round(float(weather["WindSpeed"].mean()),1)
                    if "WindSpeed" in weather.columns else None,
                }
        except Exception:
            pass

        # ── Race control messages ──────────────────────────
        try:
            rc = sess.race_control_messages
            if rc is not None and not rc.empty:
                incidents = []
                sc_periods = []
                penalties  = []
                for _, msg in rc.iterrows():
                    text = str(msg.get("Message",""))
                    cat  = str(msg.get("Category",""))
                    lap  = msg.get("Lap","?")
                    text_up = text.upper()
                    if "SAFETY CAR" in text_up or "VSC" in text_up:
                        sc_type = "VSC" if "VIRTUAL" in text_up else "SC"
                        sc_periods.append(f"L{lap} {sc_type}: {text[:80]}")
                    if "PENALTY" in text_up or "DRIVE THROUGH" in text_up:
                        penalties.append(f"L{lap}: {text[:100]}")
                    if any(kw in text_up for kw in
                           ["RETIRED","ACCIDENT","COLLISION",
                            "RED FLAG","BLACK FLAG"]):
                        incidents.append(f"L{lap} [{cat}]: {text[:100]}")
                data["race_control"]  = incidents[:20]
                data["sc_periods"]    = sc_periods[:10]
                data["penalties"]     = penalties[:10]
                data["sc_count"]      = len([s for s in sc_periods
                                             if "VSC" not in s])
                data["vsc_count"]     = len([s for s in sc_periods
                                             if "VSC" in s])
        except Exception as e:
            print(f"   ⚠ Race control: {e}")

        data["source"] = "fastf1"
        data["loaded_at"] = datetime.now().isoformat()

    except Exception as e:
        print(f"   ⚠ Race data extraction: {e}")

    return data


def extract_quali_data(sess):
    """Extracts qualifying data from FastF1 session."""
    import pandas as pd
    data = {}

    try:
        laps  = sess.laps
        valid = laps[laps["LapTime"].notna()].copy()
        valid["LapTime_s"] = valid["LapTime"].dt.total_seconds()
        valid = valid[valid["LapTime_s"] > 0]

        # Q1, Q2, Q3 results
        q_sessions = {}
        if "Session" in laps.columns:
            for q in ["Q3","Q2","Q1"]:
                q_laps = valid[valid["Session"] == q]
                if q_laps.empty:
                    continue
                best = (q_laps.groupby("Driver")["LapTime_s"]
                        .min().sort_values())
                q_sessions[q] = {
                    str(drv): {
                        "time_s": float(t),
                        "time": f"{int(t//60)}:{t%60:06.3f}",
                    }
                    for drv, t in best.items()
                }
        else:
            # No session column — treat all as Q3
            best = valid.groupby("Driver")["LapTime_s"].min().sort_values()
            q_sessions["Q3"] = {
                str(drv): {
                    "time_s": float(t),
                    "time": f"{int(t//60)}:{t%60:06.3f}",
                }
                for drv, t in best.items()
            }

        data["q_sessions"] = q_sessions

        # Pole
        if "Q3" in q_sessions and q_sessions["Q3"]:
            sorted_q3 = sorted(
                q_sessions["Q3"].items(), key=lambda x: x[1]["time_s"])
            data["pole"]      = sorted_q3[0][0]
            data["pole_time"] = sorted_q3[0][1]["time"]
            data["full_q3"]   = [
                f"P{i+1}:{drv}({info['time']})"
                for i,(drv,info) in enumerate(sorted_q3)]

        # Q2 eliminations (drivers in Q2 but not Q3)
        if "Q2" in q_sessions and "Q3" in q_sessions:
            q2_drivers = set(q_sessions["Q2"].keys())
            q3_drivers = set(q_sessions["Q3"].keys())
            eliminated_q2 = sorted(
                q2_drivers - q3_drivers,
                key=lambda d: q_sessions["Q2"][d]["time_s"])
            data["eliminated_q2"] = eliminated_q2

        # Q1 eliminations
        if "Q1" in q_sessions:
            q1_drivers = set(q_sessions["Q1"].keys())
            advanced    = set(q_sessions.get("Q2",{}).keys()) | \
                          set(q_sessions.get("Q3",{}).keys())
            eliminated_q1 = sorted(
                q1_drivers - advanced,
                key=lambda d: q_sessions["Q1"].get(d,{}).get("time_s",999))
            data["eliminated_q1"] = eliminated_q1

        # Sector bests across all quali
        sector_bests = {}
        for s_col, s_name in [
            ("Sector1Time","S1"),
            ("Sector2Time","S2"),
            ("Sector3Time","S3"),
        ]:
            if s_col in laps.columns:
                sv = laps[laps[s_col].notna()].copy()
                if not sv.empty:
                    sv[f"{s_col}_s"] = sv[s_col].dt.total_seconds()
                    sv = sv[sv[f"{s_col}_s"] > 0]
                    if not sv.empty:
                        best_idx = sv[f"{s_col}_s"].idxmin()
                        br       = sv.loc[best_idx]
                        t        = br[f"{s_col}_s"]
                        sector_bests[s_name] = {
                            "driver": str(br.get("Driver","")),
                            "time":   f"{t:.3f}s",
                        }
        data["sector_bests"] = sector_bests
        data["source"] = "fastf1"

    except Exception as e:
        print(f"   ⚠ Quali extraction: {e}")

    return data


# ── Build rich story ──────────────────────────────────────────
def build_story(episode):
    parts = []
    rname = episode.get("race_name", episode.get("track","?"))
    rnd   = episode.get("round","?")

    parts.append(
        f"R{rnd} {rname}: {episode.get('winner','?')} wins. "
        f"P2: {episode.get('p2','?')}. P3: {episode.get('p3','?')}.")

    if episode.get("fastest_lap"):
        parts.append(
            f"Fastest lap: {episode['fastest_lap']} "
            f"({episode.get('fastest_lap_time','')}).")

    quali = episode.get("qualifying",{})
    if quali.get("pole"):
        parts.append(
            f"Pole: {quali['pole']} ({quali.get('pole_time','')}).")
    if quali.get("eliminated_q2"):
        parts.append(
            f"Q2 eliminations: {', '.join(quali['eliminated_q2'][:5])}.")
    if quali.get("eliminated_q1"):
        parts.append(
            f"Q1 eliminations: {', '.join(quali['eliminated_q1'][:5])}.")

    if episode.get("top5_lap_times"):
        t5 = " | ".join(
            f"{t['driver']} {t['time']} ({t['delta']})"
            for t in episode["top5_lap_times"][:5])
        parts.append(f"Fastest race laps: {t5}.")

    if episode.get("tyre_strategy_summary"):
        parts.append(
            f"Tyre strategies: {episode['tyre_strategy_summary']}.")

    w = episode.get("weather",{})
    if w:
        parts.append(
            f"Conditions: Air {w.get('air_temp','?')}°C, "
            f"Track {w.get('track_temp','?')}°C"
            + (", Rain 🌧" if w.get("rainfall") else ", Dry ☀") + ".")

    if episode.get("dnfs"):
        parts.append(f"DNFs: {', '.join(episode['dnfs'][:6])}.")

    sc = episode.get("sc_periods",[])
    if sc:
        parts.append(f"Safety car: {' | '.join(sc[:3])}.")

    pen = episode.get("penalties",[])
    if pen:
        parts.append(f"Penalties: {' | '.join(pen[:3])}.")

    if episode.get("champ_after"):
        parts.append(f"Championship after: {episode['champ_after']}.")

    sprint = episode.get("sprint",{})
    if sprint.get("sprint_winner"):
        parts.append(
            f"Sprint: {sprint['sprint_winner']} won.")

    return " ".join(parts)


# ── Fetch from Jolpica ────────────────────────────────────────
def fetch_jolpica_race(round_num):
    data = safe_get(f"{JOLPICA}/{SEASON}/{round_num}/results.json")
    time.sleep(0.4)
    if not data:
        return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    if not races or not races[0].get("Results"):
        return None
    race    = races[0]
    results = race["Results"]
    if len(results) < 3:
        return None

    p1 = get_driver_code(results[0])
    p2 = get_driver_code(results[1])
    p3 = get_driver_code(results[2])

    dnfs = [
        f"{get_driver_code(r)}({r.get('status','')})"
        for r in results
        if r.get("status","Finished") not in
           ["Finished","+1 Lap","+2 Laps","+3 Laps",
            "+4 Laps","+5 Laps","+6 Laps","+7 Laps"]
    ]

    fl   = next((get_driver_code(r) for r in results
                 if r.get("FastestLap",{}).get("rank")=="1"), None)
    fl_t = next((r.get("FastestLap",{}).get("Time",{}).get("time","")
                 for r in results
                 if r.get("FastestLap",{}).get("rank")=="1"), "")
    pole = next((get_driver_code(r) for r in results
                 if r.get("grid")=="1"), None)

    fc = [f"P{r.get('position','?')}:{get_driver_code(r)}"
          for r in results if r.get("position")]

    # Standings
    sd = safe_get(
        f"{JOLPICA}/{SEASON}/{round_num}/driverStandings.json")
    time.sleep(0.3)
    champ = ""
    if sd:
        standings = (sd.get("MRData",{})
                      .get("StandingsTable",{})
                      .get("StandingsLists",[{}])[0]
                      .get("DriverStandings",[]))
        champ = " | ".join(
            f"{s['Driver']['code']} {s['points']}pts"
            for s in standings[:5])

    return {
        "winner": p1, "p2": p2, "p3": p3,
        "pole": pole or "",
        "fastest_lap": fl or "",
        "fastest_lap_time": fl_t,
        "dnfs": dnfs,
        "full_classification": fc,
        "champ_after": champ,
        "race_name": race.get("raceName",""),
        "date": race.get("date",""),
        "track": race.get("Circuit",{}).get(
            "Location",{}).get("locality",""),
    }


def fetch_jolpica_quali(round_num):
    data = safe_get(
        f"{JOLPICA}/{SEASON}/{round_num}/qualifying.json")
    time.sleep(0.4)
    if not data:
        return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    if not races:
        return None
    results = races[0].get("QualifyingResults",[])
    if not results:
        return None
    pole = get_driver_code(results[0]) if results else "?"
    pole_time = (results[0].get("Q3") or
                 results[0].get("Q2") or
                 results[0].get("Q1","")) if results else ""
    top10 = [
        f"P{r.get('position','?')}:{get_driver_code(r)}"
        f"({r.get('Q3') or r.get('Q2') or r.get('Q1','')})"
        for r in results[:10]]
    return {"pole": pole, "pole_time": pole_time, "top10": top10}


def fetch_jolpica_sprint(round_num):
    data = safe_get(
        f"{JOLPICA}/{SEASON}/{round_num}/sprint.json")
    time.sleep(0.4)
    if not data:
        return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    if not races:
        return None
    results = races[0].get("SprintResults",[])
    if not results:
        return None
    return {
        "sprint_winner": get_driver_code(results[0]),
        "sprint_p2":     get_driver_code(results[1]) if len(results)>1 else "?",
        "sprint_p3":     get_driver_code(results[2]) if len(results)>2 else "?",
        "sprint_top3":   " | ".join(
            f"P{r.get('position','?')}:{get_driver_code(r)}"
            for r in results[:3]),
    }


# ── Main build loop ───────────────────────────────────────────
def main():
    print("\n🏎  BoxBoxAI 2026 Memory Builder")
    print("="*50)

    # Load existing memory to preserve anything good
    existing = {}
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text())
            print(f"📂 Loaded existing memory: "
                  f"{len(existing.get('episodic',[]))} episodes")
        except Exception:
            pass

    memory = {
        "semantic": existing.get("semantic", {}),
        "episodic": [],
        "meta": {
            "built_at":   datetime.now().isoformat(),
            "season":     SEASON,
            "source":     "FastF1 + Jolpica + OpenF1",
        }
    }

    # Season summary semantic memory
    memory["semantic"]["season/2026"] = {
        "text": (
            "2026 F1 season. New regulations. "
            "Antonelli (Mercedes) leads championship 131pts after 5 races. "
            "Russell 88pts (DNF Montreal power unit). "
            "Hamilton P2 Montreal best Ferrari result. "
            "Verstappen first podium 2026 at Montreal P3. "
            "Sprint weekends: China R2, Miami R4, Austria R8, "
            "USA R16, São Paulo R18, Qatar R20."
        )
    }

    for rnd, name, date_str in RACE_CALENDAR:
        if date_str > TODAY:
            print(f"\n⏭  R{rnd} {name} ({date_str}) — future, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"📥 R{rnd} {name} ({date_str})")
        print("="*50)

        episode = {"round": rnd, "race_name": name, "date": date_str}

        # ── 1. Jolpica race result ─────────────────────────
        print("  Fetching Jolpica race result...")
        jolpica = fetch_jolpica_race(rnd)
        if not jolpica:
            print(f"  ⚠ No race result for R{rnd} — skipping")
            continue

        episode.update(jolpica)
        print(f"  ✅ Jolpica: {jolpica['winner']} wins, "
              f"P2:{jolpica['p2']}, P3:{jolpica['p3']}")

        # ── 2. FastF1 race telemetry ───────────────────────
        print("  Loading FastF1 race telemetry...")
        race_sess = load_ff1_session(rnd, "Race")
        if race_sess:
            race_data = extract_race_data(race_sess, rnd)
            # Override/enrich with FastF1 data
            if race_data.get("full_classification"):
                episode["full_classification"] = \
                    race_data["full_classification"]
            if race_data.get("fastest_lap"):
                episode["fastest_lap"]      = race_data["fastest_lap"]
                episode["fastest_lap_time"] = race_data["fastest_lap_time"]
            episode["top5_lap_times"]       = race_data.get("top5_lap_times",[])
            episode["sector_bests_race"]    = race_data.get("sector_bests",{})
            episode["tyre_stints"]          = race_data.get("tyre_stints",{})
            episode["tyre_strategy_summary"]= race_data.get("tyre_strategy_summary","")
            episode["pit_stop_counts"]      = race_data.get("pit_stop_counts",{})
            episode["weather"]              = race_data.get("weather",{})
            episode["race_control"]         = race_data.get("race_control",[])
            episode["sc_periods"]           = race_data.get("sc_periods",[])
            episode["sc_count"]             = race_data.get("sc_count",0)
            episode["vsc_count"]            = race_data.get("vsc_count",0)
            episode["penalties"]            = race_data.get("penalties",[])
            if not jolpica.get("dnfs") and race_data.get("dnfs"):
                episode["dnfs"]             = race_data["dnfs"]
            episode["telemetry_source"]     = "fastf1"
            print(f"  ✅ FastF1: {len(episode.get('full_classification',[]))} "
                  f"classified, "
                  f"{len(episode.get('tyre_stints',{}))} drivers tyre data, "
                  f"{len(episode.get('race_control',[]))} RC messages")
        else:
            episode["telemetry_source"] = "jolpica_only"
            print("  ℹ FastF1 unavailable — Jolpica only")

        # ── 3. Qualifying ──────────────────────────────────
        print("  Fetching qualifying...")
        quali_jolpica = fetch_jolpica_quali(rnd)
        quali_data    = {
            "pole":      quali_jolpica.get("pole","") if quali_jolpica else "",
            "pole_time": quali_jolpica.get("pole_time","") if quali_jolpica else "",
            "top10":     " | ".join(quali_jolpica.get("top10",[])) if quali_jolpica else "",
        }

        # Enrich qualifying with FastF1
        quali_sess = load_ff1_session(rnd, "Qualifying")
        if quali_sess:
            q_data = extract_quali_data(quali_sess)
            if q_data.get("pole"):
                quali_data["pole"]          = q_data["pole"]
                quali_data["pole_time"]     = q_data["pole_time"]
            if q_data.get("full_q3"):
                quali_data["full_q3"]       = q_data["full_q3"]
            if q_data.get("eliminated_q2"):
                quali_data["eliminated_q2"] = q_data["eliminated_q2"]
            if q_data.get("eliminated_q1"):
                quali_data["eliminated_q1"] = q_data["eliminated_q1"]
            if q_data.get("sector_bests"):
                quali_data["sector_bests"]  = q_data["sector_bests"]
            if q_data.get("q_sessions"):
                quali_data["q_sessions"]    = q_data["q_sessions"]
            print(f"  ✅ Qualifying: pole={quali_data.get('pole','?')} "
                  f"({quali_data.get('pole_time','')})")
        elif quali_jolpica:
            print(f"  ✅ Qualifying (Jolpica): "
                  f"pole={quali_data.get('pole','?')}")
        episode["qualifying"] = quali_data

        # ── 4. Sprint (sprint weekends) ────────────────────
        if rnd in SPRINT_ROUNDS:
            print("  Fetching sprint...")
            sprint = fetch_jolpica_sprint(rnd)
            if sprint:
                episode["sprint"] = sprint
                print(f"  ✅ Sprint: {sprint['sprint_winner']} wins")

        # ── 5. Build rich story ────────────────────────────
        episode["story"] = build_story(episode)

        # ── 6. Store in semantic memory too ───────────────
        memory["semantic"][f"race/r{rnd}"] = {
            "text": episode["story"],
            "tags": [name.lower(), str(rnd), episode.get("winner","").lower()],
        }
        if episode.get("champ_after"):
            memory["semantic"][f"standings/after_r{rnd}"] = {
                "text": f"After R{rnd} {name}: {episode['champ_after']}"
            }
        if quali_data.get("pole"):
            memory["semantic"][f"quali/r{rnd}"] = {
                "text": (
                    f"R{rnd} {name} qualifying: "
                    f"Pole {quali_data['pole']} ({quali_data.get('pole_time','')}). "
                    f"Q2 eliminated: {', '.join(quali_data.get('eliminated_q2',[])[:5])}. "
                    f"Q1 eliminated: {', '.join(quali_data.get('eliminated_q1',[])[:5])}."
                )
            }

        memory["episodic"].append(episode)
        print(f"\n  📋 Story preview:")
        print(f"     {episode['story'][:200]}...")

    # Save
    memory["episodic"].sort(key=lambda x: x.get("round",0))
    OUTPUT_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False))

    n_races  = len(memory["episodic"])
    n_sem    = len(memory["semantic"])
    sources  = set(e.get("telemetry_source","?")
                   for e in memory["episodic"])
    ff1_count = sum(1 for e in memory["episodic"]
                    if e.get("telemetry_source")=="fastf1")

    print(f"\n{'='*50}")
    print(f"✅ Memory built successfully!")
    print(f"   Races: {n_races}")
    print(f"   Semantic entries: {n_sem}")
    print(f"   FastF1 telemetry: {ff1_count}/{n_races} races")
    print(f"   Sources: {sources}")
    print(f"   File: {OUTPUT_FILE}")
    print(f"\n📌 Next step:")
    print(f"   git add f1_memory_2026.json")
    print(f"   git commit -m 'elite 2026 memory - full FastF1 telemetry'")
    print(f"   git push")
    print(f"\n🏎  Your bot now has ground truth for every 2026 race.\n")


if __name__ == "__main__":
    main()
