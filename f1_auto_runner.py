#!/usr/bin/env python3
"""
f1_auto_runner.py — Automated F1 predictor trigger for race weekends.

Run modes:
    python f1_auto_runner.py            # continuous 15-min polling loop
    python f1_auto_runner.py --once     # single check (for cron, preferred)
    python f1_auto_runner.py --status   # print triggered sessions and exit

Session trigger order
  Traditional weekend : FP1 → FP2 → FP3 → Q → [pre-race] → R
  Sprint weekend      : FP1 → SQ  → S   → Q → [pre-race] → R

Each data-session trigger waits 30 min after first detection (buffer for
OpenF1 to finish uploading).  The pre-race trigger fires once when the
clock passes race_start − 2 h.  After the race result appears in Jolpica,
f1_update_priors.py is run first, then the predictor runs for next-round
baseline.

State is persisted in f1_auto_state.json so --once runs via cron are
safe across restarts and never re-trigger the same session.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests

# ── Constants mirrored from f1_2026_predictor.py ─────────────────────────────
SEASON       = 2026
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE  = "https://api.openf1.org/v1"

SPRINT_ROUNDS: set[int] = {2, 4, 5, 9, 12, 16}

# Historical SC/VSC probability per circuit — mirrored from f1_2026_predictor.py
SAFETY_CAR_PROB: dict[str, float] = {
    "Circuit de Monaco"                : 0.85,
    "Baku City Circuit"                : 0.80,
    "Marina Bay Street Circuit"        : 0.75,
    "Albert Park Grand Prix Circuit"   : 0.70,
    "Jeddah Corniche Circuit"          : 0.65,
    "Hungaroring"                      : 0.60,
    "Circuit Park Zandvoort"           : 0.60,
    "Autodromo Enzo e Dino Ferrari"    : 0.55,
    "Circuit Gilles Villeneuve"        : 0.55,
    "Autodromo Jose Carlos Pace"       : 0.55,
    "Circuit of the Americas"          : 0.50,
    "Silverstone Circuit"              : 0.50,
    "Circuit de Spa-Francorchamps"     : 0.50,
    "Las Vegas Strip Circuit"          : 0.50,
    "Suzuka Circuit"                   : 0.45,
    "Red Bull Ring"                    : 0.45,
    "Miami International Autodrome"    : 0.45,
    "Autodromo Hermanos Rodriguez"     : 0.45,
    "Shanghai International Circuit"   : 0.45,
    "Autodromo Nazionale di Monza"     : 0.40,
    "Bahrain International Circuit"    : 0.40,
    "Losail International Circuit"     : 0.40,
    "Circuit de Barcelona-Catalunya"   : 0.40,
    "Yas Marina Circuit"               : 0.35,
}

# OpenF1 meeting_key ↔ Jolpica round mapping (2026 calendar)
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
SESSION_NAME_MAP: dict[str, str] = {
    "Race"              : "R",
    "Qualifying"        : "Q",
    "Sprint"            : "S",
    "Sprint Qualifying" : "SQ",
    "Practice 1"        : "FP1",
    "Practice 2"        : "FP2",
    "Practice 3"        : "FP3",
}

# ── Automation config ─────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
STATE_FILE = HERE / "f1_auto_state.json"
RUNS_DIR   = HERE / "f1_runs"
PREDICTOR  = HERE / "f1_2026_predictor.py"
UPDATER    = HERE / "f1_update_priors.py"
PYTHON     = sys.executable

POLL_INTERVAL_S = 15 * 60   # seconds between checks in continuous mode
DATA_BUFFER_S   = 30 * 60   # seconds after first data detection before running
PRE_RACE_OFFSET = 2 * 3600  # seconds before race start for pre-race trigger
REQ_TIMEOUT     = 12        # HTTP request timeout


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1: HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> dict | list | None:
    """GET with 2 retries; silent 404; returns None on all failures."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
            elif r.status_code == 404:
                return None
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2: State file (atomic write to prevent corruption on SIGINT)
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def _skey(round_num: int, label: str) -> str:
    """Canonical state dict key: 'R9_FP1', 'R9_pre_race', etc."""
    return f"R{round_num}_{label}"


def is_triggered(state: dict, round_num: int, label: str) -> bool:
    return bool(state.get(_skey(round_num, label), {}).get("triggered_at"))


def record_first_seen(state: dict, round_num: int, label: str) -> bool:
    """
    Marks the time data was first detected.  Returns True if this is the
    first time we're recording it (so the caller can print a message).
    """
    k = _skey(round_num, label)
    if k not in state:
        state[k] = {}
    if "first_seen" not in state[k]:
        state[k]["first_seen"] = _now_iso()
        return True
    return False


def buffer_elapsed(state: dict, round_num: int, label: str) -> bool:
    """True if >= DATA_BUFFER_S seconds have passed since first_seen."""
    seen = state.get(_skey(round_num, label), {}).get("first_seen")
    if not seen:
        return False
    delta = (datetime.now(timezone.utc) - datetime.fromisoformat(seen)).total_seconds()
    return delta >= DATA_BUFFER_S


def mark_triggered(state: dict, round_num: int, label: str, log_path: str) -> None:
    k = _skey(round_num, label)
    state.setdefault(k, {})
    state[k]["triggered_at"] = _now_iso()
    state[k]["log"]          = log_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3: Schedule & session helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_schedule() -> list[dict]:
    """Jolpica race list for the season."""
    data = _get(f"{JOLPICA_BASE}/{SEASON}.json", params={"limit": 100})
    if not data:
        return []
    return data.get("MRData", {}).get("RaceTable", {}).get("Races", [])


def find_next_round(races: list[dict]) -> tuple[int | None, dict | None]:
    """
    Walks rounds in order; returns the first round that has no Jolpica
    result yet.  Makes at most N+1 API calls where N = completed rounds.
    """
    for r in races:
        try:
            rnd = int(r.get("round", 0))
        except (ValueError, TypeError):
            continue
        if rnd == 0:
            continue
        data  = _get(f"{JOLPICA_BASE}/{SEASON}/{rnd}/results.json")
        time.sleep(0.4)
        races_ = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races_ or not races_[0].get("Results"):
            return rnd, r
    return None, None


def race_start_utc(race_info: dict) -> datetime | None:
    """Parse race start time from Jolpica schedule entry."""
    date_str = race_info.get("date", "")
    time_str = race_info.get("time", "")
    if not date_str or not time_str:
        return None
    try:
        d = list(map(int, date_str.split("-")))
        t = list(map(int, time_str.rstrip("Z").split(":")))
        return datetime(*d, *t, tzinfo=timezone.utc)
    except Exception:
        return None


def fetch_of1_sessions() -> list[dict]:
    """All OpenF1 sessions for the season (one API call)."""
    data = _get(f"{OPENF1_BASE}/sessions", params={"year": SEASON})
    return data if isinstance(data, list) else []


def session_key_for(of1_sessions: list[dict], round_num: int, code: str) -> int | None:
    """Map Jolpica round + session code → OpenF1 session_key."""
    meeting_key = JOLPICA_ROUND_TO_MEETING_KEY.get(round_num)
    if not meeting_key:
        return None
    for s in of1_sessions:
        if (s.get("meeting_key") == meeting_key
                and SESSION_NAME_MAP.get(s.get("session_name", "")) == code):
            return s.get("session_key")
    return None


def has_laps_data(session_key: int) -> bool:
    """
    Probe OpenF1 /laps — cheapest confirmation that a session has data.
    No `limit` param: OpenF1 currently 404s any /laps request that includes
    one (confirmed on FP1/SQ/Sprint/Q session_keys and a known-complete
    Race session, 2026-07-03), even when the unfiltered query returns real
    data. Unfiltered requests return 200 correctly, so we fetch the full
    response and only check for non-emptiness.
    """
    data = _get(f"{OPENF1_BASE}/laps", params={"session_key": session_key})
    return bool(data)


def has_race_result(round_num: int) -> bool:
    """Check Jolpica for a completed GP result."""
    data  = _get(f"{JOLPICA_BASE}/{SEASON}/{round_num}/results.json")
    time.sleep(0.4)
    races = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return bool(races and races[0].get("Results"))


def has_sprint_result(round_num: int) -> bool:
    """Check Jolpica for a completed Sprint result."""
    data  = _get(f"{JOLPICA_BASE}/{SEASON}/{round_num}/sprint.json")
    time.sleep(0.4)
    races = (data or {}).get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return bool(races and races[0].get("SprintResults"))


def session_plan(round_num: int) -> list[tuple[str, str]]:
    """
    Ordered trigger list: (of1_session_code, state_label).

    Sprint weekends (per SPRINT_ROUNDS) omit FP2/FP3 and add SQ + S.
    The 'R' entry always goes last and uses a Jolpica probe rather than
    OpenF1 laps (race data is more reliably signalled by Jolpica results).
    'pre_race' is handled outside this list (time-based, not data-based).
    """
    if round_num in SPRINT_ROUNDS:
        return [
            ("FP1", "FP1"),
            ("SQ",  "SQ"),   # Sprint Qualifying — one-lap pace, OpenF1 laps probe
            ("S",   "S"),    # Sprint Race       — Jolpica sprint probe
            ("Q",   "Q"),    # GP Qualifying     — OpenF1 laps probe
            ("R",   "R"),    # Grand Prix        — Jolpica results probe + update_priors
        ]
    return [
        ("FP1", "FP1"),
        ("FP2", "FP2"),
        ("FP3", "FP3"),
        ("Q",   "Q"),
        ("R",   "R"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4: Pre-race briefing generator
# ─────────────────────────────────────────────────────────────────────────────

def _csv_float(row: dict, col: str) -> float:
    try:
        return float(row.get(col) or 0)
    except Exception:
        return 0.0


def _csv_int(val) -> int | None:
    try:
        return int(float(val))
    except Exception:
        return None


def _parse_log_meta(log_path: Path) -> dict:
    """
    Extract circuit name, race date, weather line, rain_prob, nowcast flag,
    and any pace-anomaly flag lines from a predictor run log file.
    """
    out: dict = {
        "circuit":      "",
        "date":         "",
        "weather_line": "",
        "rain_prob":    None,   # float 0-1, or None if not logged
        "nowcast":      False,
        "pace_flags":   [],     # formatted strings for sandbagging/race-threat lines
    }
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return out

    for line in lines:
        s = line.strip()
        # "  Circuito        : Red Bull Ring  —  Austria"
        if "Circuito" in s and ":" in s and not out["circuit"]:
            out["circuit"] = s.split(":", 1)[1].split("—")[0].strip().rstrip()
        # "  Fecha           : 2026-07-06"
        if "Fecha" in s and ":" in s and not out["date"]:
            out["date"] = s.split(":", 1)[1].strip()
        # "  Clima           : ☀  Seco esperado  |  Pista ~35°C  |  Humedad 58%"
        if "Clima" in s and ":" in s and not out["weather_line"]:
            out["weather_line"] = s.split(":", 1)[1].strip()
        # "   🎯  Nowcast minutely_15 activo (4 intervalos ×15 min) — rain_prob=12%"
        if "rain_prob=" in s:
            out["nowcast"] = True
            try:
                pct = s.split("rain_prob=")[1].split("%")[0].strip()
                out["rain_prob"] = float(pct) / 100.0
            except Exception:
                pass
        # Pace anomaly flags printed by build_features()
        if any(tag in s for tag in ("SANDBAGGING", "STRUGGLING", "RACE THREAT")):
            out["pace_flags"].append(s)

    return out


def _biggest_risk(ranked: list[dict], rain_prob: float) -> str:
    """One sentence naming the single biggest uncertainty in the prediction."""
    if rain_prob >= 0.40:
        # Find best wet-weather driver among top 5 (most negative wet_weather_delta)
        wet_sorted = sorted(
            ranked[:5],
            key=lambda r: _csv_float(r, "wet_weather_delta"),
        )
        best_wet = wet_sorted[0].get("FullName", wet_sorted[0].get("code", "?"))
        return (
            f"Rain ({rain_prob:.0%} chance) could shuffle the top 3 significantly — "
            f"{best_wet} has the strongest wet-weather pace among contenders."
        )

    # Widest P10–P90 span among top 3 signals most uncertain driver
    widest_driver, widest_span = None, 0
    for row in ranked[:3]:
        p10 = _csv_int(row.get("p10_pos"))
        p90 = _csv_int(row.get("p90_pos"))
        if p10 is not None and p90 is not None and (p90 - p10) > widest_span:
            widest_span, widest_driver = p90 - p10, row

    if widest_driver and widest_span > 7:
        name = widest_driver.get("FullName", widest_driver.get("code", "?"))
        p10  = _csv_int(widest_driver.get("p10_pos"))
        p90  = _csv_int(widest_driver.get("p90_pos"))
        return (
            f"{name}'s P10–P90 interval spans P{p10}–P{p90} ({widest_span} places) — "
            f"the model is uncertain enough that a podium or a points miss are both live."
        )

    # Close field at the top
    top2_gap = _csv_float(ranked[0], "win_mc_pct") - _csv_float(ranked[1], "win_mc_pct")
    if ranked[1:] and top2_gap < 10:
        r2 = ranked[1].get("FullName", ranked[1].get("code", "?"))
        return (
            f"The race is genuinely open at the top — {r2} is within "
            f"{top2_gap:.1f} percentage points of the predicted winner."
        )

    return (
        "Prediction confidence is within normal range — "
        "main risk is an unmodelled strategic call or mechanical issue."
    )


# Maps legacy/API team name variants to 2026 official names.
# Mirrors the normalization in f1_2026_predictor.py so old CSVs
# are corrected at display time without needing a re-run.
_TEAM_NAME_ALIASES: dict[str, str] = {
    "RB F1 Team"           : "Racing Bulls",
    "Visa Cash App RB"     : "Racing Bulls",
    "VCARB"                : "Racing Bulls",
    "AlphaTauri"           : "Racing Bulls",
    "Scuderia AlphaTauri"  : "Racing Bulls",
}

def _normalize_team(name: str) -> str:
    return _TEAM_NAME_ALIASES.get(name, name)


def generate_prerace_briefing(round_num: int | None = None) -> str | None:
    """
    Build a human-readable pre-race briefing from f1_2026_predicciones.csv
    and the most recent predictor log for the round.

    Saves to  f1_runs/R{N}_BRIEFING.txt  and prints to console.
    Returns the output path as a string, or None on failure.

    Can be called directly:
        python f1_auto_runner.py --briefing
    Or triggered automatically after the 2-hour pre-race run.
    """
    import csv as _csv

    csv_path = HERE / "f1_2026_predicciones.csv"
    if not csv_path.exists():
        print("Briefing: f1_2026_predicciones.csv not found — run predictor first.")
        return None

    # ── Load CSV ──────────────────────────────────────────────────────────
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        print("Briefing: CSV is empty.")
        return None

    if round_num is None:
        round_num = _csv_int(rows[0].get("round_num"))
        if round_num is None:
            print("Briefing: could not read round_num from CSV.")
            return None

    # Filter and sort by MC win probability
    rd_rows = [r for r in rows if _csv_int(r.get("round_num")) == round_num]
    if not rd_rows:
        print(f"Briefing: no rows for R{round_num} in CSV.")
        return None

    ranked = sorted(rd_rows, key=lambda r: _csv_float(r, "win_mc_pct"), reverse=True)
    race_name = ranked[0].get("race_name", f"Round {round_num}")

    # ── Find log file ─────────────────────────────────────────────────────
    RUNS_DIR.mkdir(exist_ok=True)
    log_candidates = sorted(
        list(RUNS_DIR.glob(f"R{round_num}_*.log")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Fall back to the austria text file when f1_runs/ has no logs yet
    fallback = HERE / "austria_r8_prediction.txt"
    log_path = log_candidates[0] if log_candidates else (fallback if fallback.exists() else None)
    meta     = _parse_log_meta(log_path) if log_path else {}

    circuit      = meta.get("circuit", "")
    race_date    = meta.get("date", "")
    weather_raw  = meta.get("weather_line", "")
    rain_prob    = meta.get("rain_prob")           # None or float
    nowcast      = meta.get("nowcast", False)
    pace_flags   = meta.get("pace_flags", [])

    sc_prob = SAFETY_CAR_PROB.get(circuit, 0.50)

    # ── Weather description ───────────────────────────────────────────────
    if rain_prob is not None:
        src = "nowcast (live)" if nowcast else "hourly forecast"
        if rain_prob < 0.10:
            w_desc = f"Dry expected — {rain_prob:.0%} rain probability ({src})"
        elif rain_prob < 0.30:
            w_desc = f"Mostly dry — {rain_prob:.0%} chance of rain ({src})"
        elif rain_prob < 0.60:
            w_desc = f"Mixed conditions — {rain_prob:.0%} rain probability ({src})"
        else:
            w_desc = f"Wet conditions expected — {rain_prob:.0%} rain probability ({src})"
        rain_prob_for_risk = rain_prob
    elif weather_raw:
        # Parse pipe-separated "Clima" line into a clean summary.
        # Format: "☀  Seco esperado  |  Pista ~35°C  |  Humedad 58%"
        parts = [p.strip() for p in weather_raw.split("|")]
        # First segment is the condition label (strip emoji)
        cond  = parts[0].replace("☀", "").replace("🌧", "").replace("⛅", "").replace("🌦", "").strip()
        # Translate Spanish condition phrases to English for the briefing only.
        # The predictor logs stay in Spanish — this mapping is applied here alone.
        _ES_TO_EN = {
            "Seco esperado"    : "Dry expected",
            "Lluvia posible"   : "Rain possible",
            "Lluvia probable"  : "Rain likely",
            "Lluvia esperada"  : "Rain expected",
            "Condiciones mixtas": "Mixed conditions",
        }
        cond = _ES_TO_EN.get(cond, cond)
        w_desc      = cond
        rain_prob_for_risk = 0.0
    else:
        w_desc = "Weather data not available in log"
        rain_prob_for_risk = 0.0

    # Extra track conditions from the pipe-separated weather line (temp, humidity)
    track_extra = ""
    if weather_raw:
        parts  = [p.strip() for p in weather_raw.split("|")]
        extras = [p for p in parts[1:] if any(k in p for k in ("°C", "Humedad", "Humidity"))]
        if extras:
            track_extra = "  " + "  ·  ".join(extras)

    # ── SC label ─────────────────────────────────────────────────────────
    if sc_prob >= 0.70:
        sc_note = "very high — multiple SC/VSC periods expected"
    elif sc_prob >= 0.55:
        sc_note = "high — plan for at least one safety car"
    elif sc_prob >= 0.45:
        sc_note = "moderate — safety car is a coin flip"
    else:
        sc_note = "below average — cleaner race likely"

    # ── Epistemic uncertainty threshold ───────────────────────────────────
    unc_vals = [_csv_float(r, "epistemic_unc") for r in ranked]
    unc_mean = sum(unc_vals) / len(unc_vals) if unc_vals else 0
    unc_std  = (sum((v - unc_mean) ** 2 for v in unc_vals) / len(unc_vals)) ** 0.5
    unc_hi   = unc_mean + unc_std   # threshold: top ~16% of field

    # ── Flags from CSV (top 10) ───────────────────────────────────────────
    flag_lines: list[str] = []
    for row in ranked[:10]:
        name   = row.get("FullName", row.get("code", "?"))
        parts  = []
        if str(row.get("sandbagging_flag", "")).lower() == "true":
            parts.append("hiding pace (FP3 suggested better than qualifying showed)")
        if str(row.get("struggling_flag", "")).lower() == "true":
            parts.append("underperforming (FP3 pace worse than qualifying result)")
        if str(row.get("race_threat_flag", "")).lower() == "true":
            parts.append("race threat (FP2 race pace much better than grid slot)")
        if _csv_float(row, "epistemic_unc") > unc_hi:
            parts.append("models disagree on outcome")
        if parts:
            flag_lines.append(f"  {name}: {', '.join(parts)}")

    # Supplement with any flags parsed directly from the log
    for fl in pace_flags:
        # Avoid duplicating content already shown from CSV
        flag_lines.append(f"  {fl}")

    # ── Winner confidence label ───────────────────────────────────────────
    win_pct = _csv_float(ranked[0], "win_mc_pct")
    if win_pct >= 40:
        conf_label = "clear favourite"
    elif win_pct >= 25:
        conf_label = "strong favourite"
    elif win_pct >= 15:
        conf_label = "marginal favourite"
    else:
        conf_label = "slight favourite in an open field"

    # ── Assemble text ─────────────────────────────────────────────────────
    W    = 64
    SEP  = "═" * W
    THIN = "─" * W
    out  = []

    def a(s: str = "") -> None:
        out.append(s)

    a(SEP)
    a(f"  RACE BRIEFING  —  R{round_num}  ·  {race_name}")
    circ_line = circuit if circuit else "(circuit not found in log)"
    if race_date:
        circ_line += f"  |  {race_date}"
    a(f"  {circ_line}")
    a(SEP)
    a()

    a("WEATHER")
    a(f"  {w_desc}")
    if track_extra:
        a(track_extra)
    a()

    a("SAFETY CAR / VSC")
    a(f"  {sc_prob:.0%} probability  —  {sc_note}")
    a()

    a("TOP 10 WIN PROBABILITIES  (10,000 Monte Carlo simulations)")
    a()
    for i, row in enumerate(ranked[:10], 1):
        name = row.get("FullName", row.get("code", "?"))
        team = _normalize_team(row.get("TeamName", ""))
        win  = _csv_float(row, "win_mc_pct")
        pod  = _csv_float(row, "podium_mc_pct")
        exp  = _csv_float(row, "avg_mc_pos")
        p10  = _csv_int(row.get("p10_pos"))
        p90  = _csv_int(row.get("p90_pos"))
        ci   = f"P{p10}–P{p90}" if p10 and p90 else "?"
        a(f"  {i:>2}. {name} ({team})")
        a(f"      Win {win:.1f}%  ·  Podium {pod:.1f}%  ·  Exp finish P{exp:.0f}  ·  Range {ci}")
    a()

    a("PACE FLAGS  (top 10 drivers)")
    if flag_lines:
        for fl in flag_lines:
            a(fl)
    else:
        a("  None detected for top 10")
    a()

    winner   = ranked[0]
    win_name = winner.get("FullName", winner.get("code", "?"))
    win_team = _normalize_team(winner.get("TeamName", ""))
    win_pod  = _csv_float(winner, "podium_mc_pct")
    win_exp  = _csv_float(winner, "avg_mc_pos")
    a("VERDICT")
    a(f"  {win_name} ({win_team}) is the {conf_label}.")
    a(f"  Win: {win_pct:.1f}%  ·  Podium: {win_pod:.1f}%  ·  Expected finish: P{win_exp:.0f}")
    a()

    a("BIGGEST RISK")
    a(f"  {_biggest_risk(ranked, rain_prob_for_risk)}")
    a()

    a(SEP)
    a(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    a(SEP)

    text = "\n".join(out)

    # ── Write and print ───────────────────────────────────────────────────
    out_path = RUNS_DIR / f"R{round_num}_BRIEFING.txt"
    out_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    print(f"\n  Briefing saved: {out_path.relative_to(HERE)}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4b: WhatsApp briefing via CallMeBot
# ─────────────────────────────────────────────────────────────────────────────

# Circuit → country flag emoji for WhatsApp header line
_CIRCUIT_FLAG: dict[str, str] = {
    "Bahrain International Circuit"      : "🇧🇭",
    "Jeddah Corniche Circuit"            : "🇸🇦",
    "Albert Park Grand Prix Circuit"     : "🇦🇺",
    "Suzuka Circuit"                     : "🇯🇵",
    "Shanghai International Circuit"     : "🇨🇳",
    "Miami International Autodrome"      : "🇺🇸",
    "Autodromo Enzo e Dino Ferrari"      : "🇮🇹",
    "Circuit de Monaco"                  : "🇲🇨",
    "Circuit de Barcelona-Catalunya"     : "🇪🇸",
    "Circuit Gilles Villeneuve"          : "🇨🇦",
    "Red Bull Ring"                      : "🇦🇹",
    "Silverstone Circuit"                : "🇬🇧",
    "Hungaroring"                        : "🇭🇺",
    "Circuit de Spa-Francorchamps"       : "🇧🇪",
    "Circuit Park Zandvoort"             : "🇳🇱",
    "Autodromo Nazionale di Monza"       : "🇮🇹",
    "Baku City Circuit"                  : "🇦🇿",
    "Marina Bay Street Circuit"          : "🇸🇬",
    "Circuit of the Americas"            : "🇺🇸",
    "Autodromo Hermanos Rodriguez"       : "🇲🇽",
    "Autodromo Jose Carlos Pace"         : "🇧🇷",
    "Las Vegas Strip Circuit"            : "🇺🇸",
    "Losail International Circuit"       : "🇶🇦",
    "Yas Marina Circuit"                 : "🇦🇪",
}


def _build_whatsapp_message(round_num: int) -> str | None:
    """
    Assemble a compact, emoji-rich WhatsApp message from the same data
    sources as generate_prerace_briefing(). Returns the message string
    or None if the CSV is missing/empty.
    """
    import csv as _csv

    csv_path = HERE / "f1_2026_predicciones.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    rd_rows = [r for r in rows if _csv_int(r.get("round_num")) == round_num]
    if not rd_rows:
        return None
    ranked = sorted(rd_rows, key=lambda r: _csv_float(r, "win_mc_pct"), reverse=True)

    race_name = ranked[0].get("race_name", f"Round {round_num}")

    # Log metadata (same fallback logic as generate_prerace_briefing)
    RUNS_DIR.mkdir(exist_ok=True)
    log_candidates = sorted(
        list(RUNS_DIR.glob(f"R{round_num}_*.log")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    fallback = HERE / "austria_r8_prediction.txt"
    log_path = log_candidates[0] if log_candidates else (fallback if fallback.exists() else None)
    meta = _parse_log_meta(log_path) if log_path else {}

    circuit    = meta.get("circuit", "")
    weather_raw = meta.get("weather_line", "")
    rain_prob  = meta.get("rain_prob")
    nowcast    = meta.get("nowcast", False)

    sc_prob = SAFETY_CAR_PROB.get(circuit, 0.50)
    flag    = _CIRCUIT_FLAG.get(circuit, "🏁")

    # ── Weather block ─────────────────────────────────────────────────────
    # Parse the pipe-separated weather line for condition, temp, humidity.
    # rain_prob (from nowcast) is shown as an explicit % when available.
    _ES_TO_EN_W = {
        "Seco esperado": "Dry expected", "Lluvia posible": "Rain possible",
        "Lluvia probable": "Rain likely", "Lluvia esperada": "Rain expected",
        "Condiciones mixtas": "Mixed conditions",
    }
    temp_str = ""
    hum_str  = ""
    if weather_raw:
        raw_parts = [p.strip() for p in weather_raw.split("|")]
        cond_raw  = raw_parts[0].replace("☀","").replace("🌧","").replace("⛅","").replace("🌦","").strip()
        w_cond    = _ES_TO_EN_W.get(cond_raw, cond_raw)
        for seg in raw_parts[1:]:
            if "°C" in seg:
                temp_str = seg.replace("Pista","").replace("~","").strip()
            if "Humedad" in seg or "Humidity" in seg:
                hum_str = seg.replace("Humedad", "Humidity").strip()
    elif rain_prob is not None:
        if rain_prob < 0.10:   w_cond = "Dry expected"
        elif rain_prob < 0.30: w_cond = "Mostly dry"
        elif rain_prob < 0.60: w_cond = "Mixed conditions"
        else:                   w_cond = "Wet conditions"
    else:
        w_cond = "Weather unavailable"

    rp = rain_prob if rain_prob is not None else 0.0
    low = w_cond.lower()
    if rp >= 0.60:       w_emoji = "🌧️"
    elif rp >= 0.30:     w_emoji = "⛅"
    elif rp >= 0.10:     w_emoji = "🌦️"
    elif "rain" in low:  w_emoji = "🌧️"
    elif "mixed" in low: w_emoji = "⛅"
    else:                w_emoji = "☀️"

    # First line: condition + explicit rain % if nowcast available
    if rain_prob is not None:
        w_line1 = f"{w_emoji} {w_cond} · {rain_prob:.0%} rain chance"
    else:
        w_line1 = f"{w_emoji} {w_cond}"

    # Second line: track temp + humidity (omitted if neither available)
    detail_parts = []
    if temp_str: detail_parts.append(f"Track {temp_str}")
    if hum_str:  detail_parts.append(hum_str)
    w_line2 = ("   " + "  ·  ".join(detail_parts)) if detail_parts else ""

    # ── SC label ──────────────────────────────────────────────────────────
    if sc_prob >= 0.70:   sc_short = "very high"
    elif sc_prob >= 0.55: sc_short = "high"
    elif sc_prob >= 0.45: sc_short = "coin flip"
    else:                 sc_short = "below average"

    # ── Epistemic uncertainty threshold ───────────────────────────────────
    unc_vals = [_csv_float(r, "epistemic_unc") for r in ranked]
    unc_mean = sum(unc_vals) / len(unc_vals) if unc_vals else 0
    unc_std  = (sum((v - unc_mean) ** 2 for v in unc_vals) / len(unc_vals)) ** 0.5
    unc_hi   = unc_mean + unc_std

    def _last(name: str) -> str:
        return name.split()[-1]

    medals = ["🥇", "🥈", "🥉"]
    pred_lines: list[str] = []
    warn_lines: list[str] = []

    for i, row in enumerate(ranked[:5], 1):
        name = row.get("FullName", row.get("code", "?"))
        team = _normalize_team(row.get("TeamName", ""))
        win  = _csv_float(row, "win_mc_pct")
        pod  = _csv_float(row, "podium_mc_pct")
        if i <= 3:
            pred_lines.append(
                f"{medals[i-1]} {_last(name)} ({team}) — {win:.1f}% win · {pod:.1f}% podium"
            )
        else:
            pred_lines.append(f"{i}  {_last(name)} ({team}) — {win:.1f}% win")
        if _csv_float(row, "epistemic_unc") > unc_hi:
            warn_lines.append(f"⚠️ {_last(name)}: models disagree on outcome")

    # Biggest wildcard: widest P10-P90 span in top 3
    for row in ranked[:3]:
        p10 = _csv_float(row, "p10_pos")
        p90 = _csv_float(row, "p90_pos")
        if p10 and p90 and (p90 - p10) > 7:
            warn_lines.append(
                f"⚠️ {_last(row.get('FullName','?'))} P{int(p10)}–P{int(p90)} range — biggest wildcard"
            )
            break

    disclaimer = (
        "⚠️ This is an experimental ML prediction model still under "
        "active development. Actual race results may vary significantly."
    )

    weather_block = [w_line1]
    if w_line2:
        weather_block.append(w_line2)

    parts: list[str] = [
        disclaimer,
        "",
        f"🏁 R{round_num} {race_name} {flag}",
        "",
    ] + weather_block + [
        f"🚦 SC {sc_prob:.0%} — {sc_short}",
        "",
        "🔮 PREDICTION  (10k simulations)",
    ] + pred_lines

    if warn_lines:
        parts += [""] + warn_lines

    parts += ["", f"🏎️ Full breakdown → f1_runs/R{round_num}_BRIEFING.txt"]

    return "\n".join(parts)


def send_whatsapp_briefing(round_num: int) -> None:
    """
    Send a compact pre-race briefing to WhatsApp via the CallMeBot API.

    Reads WHATSAPP_PHONE and CALLMEBOT_API_KEY from environment variables
    (set these as GitHub Actions secrets). Silently skips if either is absent.
    Errors are logged but never raised — caller continues unaffected.
    """
    import os
    phone  = os.environ.get("WHATSAPP_PHONE", "").strip()
    apikey = os.environ.get("CALLMEBOT_API_KEY", "").strip()

    if not phone or not apikey:
        print("  [WhatsApp] WHATSAPP_PHONE or CALLMEBOT_API_KEY not set — skipping.")
        return

    msg = _build_whatsapp_message(round_num)
    if not msg:
        print("  [WhatsApp] Could not build message (CSV missing?) — skipping.")
        return

    url = (
        f"https://api.callmebot.com/whatsapp.php"
        f"?phone={quote(phone)}&text={quote(msg)}&apikey={quote(apikey)}"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            print(f"  [WhatsApp] Briefing sent to {phone[:6]}***  ✓")
        else:
            print(f"  [WhatsApp] API returned {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        print(f"  [WhatsApp] Send failed (non-fatal): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5: Predictor runner (subprocess with tee to log file)
# ─────────────────────────────────────────────────────────────────────────────

def run_predictor(round_num: int, label: str,
                  *, update_priors_first: bool = False) -> str:
    """
    Optionally runs f1_update_priors.py --round N, then f1_2026_predictor.py.
    Full stdout+stderr is saved to a timestamped log file.
    Errors are logged and do NOT raise — caller continues polling.
    Returns the log file path string.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = RUNS_DIR / f"R{round_num}_{label}_{ts}.log"

    banner = (
        f"# F1 Auto Runner — R{round_num} / {label}\n"
        f"# Started : {datetime.now(timezone.utc).isoformat()}\n"
        + (f"# Pre-step: f1_update_priors.py --round {round_num}\n"
           if update_priors_first else "")
        + "#" * 60 + "\n\n"
    )

    print(f"\n{'─'*60}")
    print(f"  TRIGGER  R{round_num} / {label}")
    print(f"  Log    → {log_path.relative_to(HERE)}")
    print(f"{'─'*60}")

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(banner)
        lf.flush()

        # ── Optional prior update ─────────────────────────────────────────
        if update_priors_first:
            print(f"  [1/2] f1_update_priors.py --round {round_num}")
            lf.write(f">>> f1_update_priors.py --round {round_num}\n\n")
            lf.flush()
            try:
                proc = subprocess.run(
                    [PYTHON, str(UPDATER), "--round", str(round_num)],
                    cwd=str(HERE),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                lf.write(proc.stdout or "")
                lf.write(f"\n[exit {proc.returncode}]\n\n")
                lf.flush()
                if proc.returncode != 0:
                    print(f"  WARNING: update_priors exited {proc.returncode} "
                          f"— continuing to predictor")
            except Exception as exc:
                msg = f"ERROR launching update_priors: {exc}\n"
                lf.write(msg)
                print(f"  {msg.strip()}")

        # ── Predictor ─────────────────────────────────────────────────────
        step = "2/2" if update_priors_first else "1/1"
        print(f"  [{step}] f1_2026_predictor.py")
        lf.write(">>> f1_2026_predictor.py\n\n")
        lf.flush()
        try:
            proc = subprocess.run(
                [PYTHON, str(PREDICTOR)],
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            lf.write(proc.stdout or "")
            lf.write(f"\n[exit {proc.returncode}]\n")
            lf.flush()

            # Show last 10 lines on console so the log tail is visible
            tail = (proc.stdout or "").rstrip().splitlines()[-10:]
            if tail:
                print("  ┌─ output tail ───")
                for line in tail:
                    print(f"  │ {line}")
                print(f"  └─ [exit {proc.returncode}]")
        except Exception as exc:
            msg = f"ERROR launching predictor: {exc}\n"
            lf.write(msg)
            print(f"  {msg.strip()}")

    print(f"  Log saved: {log_path.name}\n")
    return str(log_path)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5: Single poll — the core logic
# ─────────────────────────────────────────────────────────────────────────────

def check_once() -> None:
    """
    One complete poll cycle.  Safe to call from cron every 15 minutes.
    All persistent state flows through f1_auto_state.json.
    """
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Polling...")

    state = load_state()

    # ── 1. Fetch schedule and locate next round ───────────────────────────
    races = fetch_schedule()
    if not races:
        print("  ERROR: could not fetch Jolpica schedule.")
        return

    next_rnd, race_info = find_next_round(races)
    if next_rnd is None:
        print("  Season complete — no pending rounds.")
        return

    race_name = race_info.get("raceName", f"Round {next_rnd}")
    is_sprint = next_rnd in SPRINT_ROUNDS
    fmt_label = "sprint" if is_sprint else "traditional"
    print(f"  Watching: R{next_rnd} — {race_name} ({fmt_label} format)")

    # ── 2. Fetch OpenF1 session index once ───────────────────────────────
    of1_sessions = fetch_of1_sessions()

    # ── 3. Session-data triggers ─────────────────────────────────────────
    for of1_code, label in session_plan(next_rnd):
        if is_triggered(state, next_rnd, label):
            continue   # already ran for this session

        # ── Probe: does data exist? ───────────────────────────────────────
        data_available = False

        if of1_code == "R":
            # Grand Prix result: use Jolpica (definitive completion signal)
            data_available = has_race_result(next_rnd)

        elif of1_code == "S":
            # Sprint Race result: Jolpica sprint endpoint
            data_available = has_sprint_result(next_rnd)

        else:
            # FP1 / FP2 / FP3 / Q / SQ: probe OpenF1 /laps
            sk = session_key_for(of1_sessions, next_rnd, of1_code)
            if sk:
                data_available = has_laps_data(sk)

        if not data_available:
            continue

        # ── Buffer: wait 30 min after first detection ─────────────────────
        is_new = record_first_seen(state, next_rnd, label)
        save_state(state)   # checkpoint first_seen immediately
        if is_new:
            print(f"  [{label}] data detected — 30-min buffer started")
            continue

        if not buffer_elapsed(state, next_rnd, label):
            first_seen = state.get(_skey(next_rnd, label), {}).get("first_seen", "")
            elapsed    = int(
                (now - datetime.fromisoformat(first_seen)).total_seconds() / 60
            )
            remaining  = DATA_BUFFER_S // 60 - elapsed
            print(f"  [{label}] buffer: {elapsed}/{DATA_BUFFER_S // 60} min "
                  f"({remaining} min remaining)")
            continue

        # ── Run ───────────────────────────────────────────────────────────
        log = run_predictor(
            next_rnd, label,
            update_priors_first=(of1_code == "R"),
        )
        mark_triggered(state, next_rnd, label, log)
        save_state(state)

    # ── 4. Pre-race time trigger (2 h before race start) ─────────────────
    pre_label = "pre_race"
    if not is_triggered(state, next_rnd, pre_label):
        r_start = race_start_utc(race_info)
        if r_start:
            trigger_at = r_start - timedelta(seconds=PRE_RACE_OFFSET)
            if trigger_at <= now < r_start:
                print(f"  [pre_race] window open — running pre-race final prediction")
                log = run_predictor(next_rnd, pre_label)
                mark_triggered(state, next_rnd, pre_label, log)
                save_state(state)
                # Generate the human-readable briefing after the prediction run
                generate_prerace_briefing(next_rnd)
                # Send compact WhatsApp summary (only for pre-race, not every session)
                send_whatsapp_briefing(next_rnd)
            elif now < trigger_at:
                mins = int((trigger_at - now).total_seconds() / 60)
                print(f"  [pre_race] fires in {mins} min "
                      f"({trigger_at.strftime('%Y-%m-%d %H:%M UTC')})")
            else:
                # Past the pre-race window without triggering (race started)
                # — mark as skipped so we don't try again
                mark_triggered(state, next_rnd, pre_label, "skipped")
                save_state(state)
        else:
            print("  [pre_race] race time not available in schedule — skipped")

    print("  Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6: --status display
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    """Pretty-print f1_auto_state.json — no API calls needed."""
    state = load_state()
    if not state:
        print(f"No state recorded yet ({STATE_FILE} absent or empty).")
        return

    print(f"\nAuto-runner state  ({STATE_FILE})\n")

    # Group entries by round prefix
    by_round: dict[str, list[tuple[str, dict]]] = {}
    for k, v in sorted(state.items()):
        prefix = k.split("_", 1)[0]   # e.g. "R9"
        by_round.setdefault(prefix, []).append((k, v))

    for rnd_prefix, entries in sorted(by_round.items(),
                                       key=lambda x: int(x[0][1:])):
        print(f"  {rnd_prefix}:")
        for k, v in entries:
            label = k.split("_", 1)[1] if "_" in k else k
            if v.get("triggered_at"):
                ts  = v["triggered_at"][:19].replace("T", " ")
                log = Path(v.get("log", "")).name or v.get("log", "")
                icon = "✅"
                print(f"    {icon}  {label:<12}  triggered {ts}  →  {log}")
            elif v.get("first_seen"):
                ts      = v["first_seen"][:19].replace("T", " ")
                elapsed = int(
                    (datetime.now(timezone.utc)
                     - datetime.fromisoformat(v["first_seen"])).total_seconds() / 60
                )
                remaining = max(0, DATA_BUFFER_S // 60 - elapsed)
                icon = "⏳"
                print(f"    {icon}  {label:<12}  data seen {ts}  "
                      f"(buffer: {elapsed}/{DATA_BUFFER_S // 60} min, "
                      f"{remaining} remaining)")
            else:
                print(f"    ❓  {label:<12}  {v}")
    print()


# ─────────────────────────────────────────────────────────────
#  SECTION 7: Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="f1_auto_runner.py",
        description="Automated F1 predictor trigger for race weekends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python f1_auto_runner.py --once      # single check (use with cron)\n"
            "  python f1_auto_runner.py             # continuous 15-min loop\n"
            "  python f1_auto_runner.py --status    # show triggered sessions\n"
            "  python f1_auto_runner.py --briefing  # generate pre-race briefing now\n"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (preferred mode for cron scheduling)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print trigger state for current weekend and exit",
    )
    parser.add_argument(
        "--briefing",
        action="store_true",
        help="Generate pre-race briefing from latest CSV and exit (no API calls)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        metavar="N",
        help="Force briefing for a specific round (use with --briefing)",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.briefing:
        generate_prerace_briefing(round_num=args.round)
        return

    if args.once:
        try:
            check_once()
        except Exception as exc:
            import traceback
            print(f"\nUNHANDLED ERROR: {exc}")
            traceback.print_exc()
        return

    # ── Continuous loop ───────────────────────────────────────────────────
    print(f"F1 Auto Runner — polling every {POLL_INTERVAL_S // 60} min")
    print(f"State : {STATE_FILE}")
    print(f"Logs  : {RUNS_DIR}/")
    print("Press Ctrl+C to stop.\n")
    while True:
        try:
            check_once()
        except Exception as exc:
            import traceback
            print(f"\nERROR (continuing): {exc}")
            traceback.print_exc()
        print(f"Sleeping {POLL_INTERVAL_S // 60} min...")
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
