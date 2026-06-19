"""Data layer for BoxBoxAI: HTTP fetching, memory/state persistence,
FastF1/OpenF1 telemetry enrichment, and predictor I/O."""
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from models import ContextBlock
from news import get_news_context

log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
#  GLOBALS
# ═════════════════════════════════════════════════════════════
SEASON        = 2026
JOLPICA       = "https://api.jolpi.ca/ergast/f1"
MEMORY_FILE   = Path(__file__).parent / "f1_memory_2026.json"
SESSIONS_FILE = Path(__file__).parent / "boxboxai_sessions.json"
PREDICTIONS_FILE       = Path(__file__).parent / "boxboxai_predictions.json"
AUTO_INGEST_FILE       = Path(__file__).parent / "boxboxai_auto_ingest.json"
F1_CACHE_DIR           = Path(__file__).parent / "f1_cache"
PREDICTOR_STATE_FILE   = Path(__file__).parent / "boxboxai_predictor_state.json"
PREDICTOR_CSV          = Path(__file__).parent / "f1_2026_predicciones.csv"
MEMORY_ENRICHMENT_FILE = Path(__file__).parent / "boxboxai_enrichment.json"

_PREDICTOR_CACHE: dict = {}

SPRINT_ROUNDS_2026 = {2, 4, 8, 16, 18, 20}  # Chinese, Miami, Austrian, US, São Paulo, Qatar

# ─── FastF1 availability ──────────────────────────────────────
_FF1_AVAILABLE = False
try:
    import fastf1
    import pandas as pd
    import numpy as np
    F1_CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(F1_CACHE_DIR))
    _FF1_AVAILABLE = True
    log.info("FastF1 available ✅")
except ImportError:
    log.warning("FastF1 not installed — using Jolpica+OpenF1 only")


# ═════════════════════════════════════════════════════════════
#  MEMORY PERSISTENCE
# ═════════════════════════════════════════════════════════════
def load_f1_memory() -> dict:
    """Loads the agent's episodic + semantic memory from file."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    return {
        "semantic": {
            "season/2026_summary": {
                "text": "2026 F1: new aero regs. Antonelli dominant (4 wins). RUS won Australia. ANT leads WDC 131pts, RUS 88pts. RUS DNF Montreal power unit. HAM P2 Montreal. VER first podium Montreal P3.",
            }
        },
        "episodic": [
            {"round":5,"track":"Montreal","winner":"ANT","p2":"HAM","p3":"VER",
             "story":"Antonelli wins after Russell retires (power unit lap 31). Hamilton P2 Ferrari. Verstappen P3.",
             "champ_delta":"ANT +43 over RUS"}
        ]
    }

def save_f1_memory(mem: dict):
    """Saves updated memory to file."""
    try:
        MEMORY_FILE.write_text(json.dumps(mem, indent=2))
        log.info("Memory saved successfully")
    except Exception as e:
        log.error(f"Failed to save memory: {e}")


def load_ingest_state() -> dict:
    if AUTO_INGEST_FILE.exists():
        try:
            return json.loads(AUTO_INGEST_FILE.read_text())
        except Exception:
            pass
    return {}

def save_ingest_state(state: dict):
    try:
        AUTO_INGEST_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.error(f"Failed to save ingest state: {e}")


def load_predictor_state() -> dict:
    if PREDICTOR_STATE_FILE.exists():
        try:
            return json.loads(PREDICTOR_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_predictor_state(state: dict):
    try:
        PREDICTOR_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.error(f"Failed to save predictor state: {e}")


def load_enrichment_state() -> dict:
    if MEMORY_ENRICHMENT_FILE.exists():
        try:
            return json.loads(MEMORY_ENRICHMENT_FILE.read_text())
        except Exception:
            pass
    return {}


def save_enrichment_state(state: dict):
    try:
        MEMORY_ENRICHMENT_FILE.write_text(
            json.dumps(state, indent=2))
    except Exception as e:
        log.error(f"Failed to save enrichment state: {e}")


# ═════════════════════════════════════════════════════════════
#  SESSIONS / USER TRACKING
# ═════════════════════════════════════════════════════════════
def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text())
        except Exception:
            pass
    return {}

def save_sessions(sessions: dict):
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))

def get_user_history(sessions: dict, user_id: str) -> list:
    return sessions.get(user_id, {}).get("history", [])

def update_user_history(sessions: dict, user_id: str,
                         role: str, content: str):
    if user_id not in sessions:
        sessions[user_id] = {
            "history":    [],
            "first_seen": datetime.now().isoformat(),
            "stats":      {
                "total_messages": 0,
                "favorite_topics": {},
                "commands_used":   {},
                "last_active":     None,
            }
        }
    sessions[user_id]["history"].append({"role": role, "content": content})
    sessions[user_id]["last_seen"] = datetime.now().isoformat()
    # Keep last 6 messages (3 exchanges) — enough for conversational context
    # without burning tokens on old history
    sessions[user_id]["history"] = sessions[user_id]["history"][-6:]

    # Track stats for user messages only
    if role == "user":
        stats = sessions[user_id].setdefault("stats", {
            "total_messages": 0, "favorite_topics": {}, "commands_used": {}, "last_active": None
        })
        stats["total_messages"] = stats.get("total_messages", 0) + 1
        stats["last_active"]    = datetime.now().isoformat()

        # Detect topic from message
        topic = _classify_topic(content)
        if topic:
            stats["favorite_topics"][topic] = \
                stats["favorite_topics"].get(topic, 0) + 1


def _classify_topic(text: str) -> str | None:
    """Classifies a message into an F1 topic for stats tracking."""
    t = text.lower()
    if any(w in t for w in ["standings", "championship", "points", "campeonato"]):
        return "championship"
    if any(w in t for w in ["predict", "win", "winner", "ganar", "predicción"]):
        return "predictions"
    if any(w in t for w in ["weather", "rain", "clima", "lluvia"]):
        return "weather"
    if any(w in t for w in ["strategy", "tyre", "pit", "estrategia", "llantas"]):
        return "strategy"
    if any(w in t for w in ["checo", "pérez", "perez", "hamilton", "verstappen",
                             "antonelli", "leclerc", "russell", "norris"]):
        return "drivers"
    if any(w in t for w in ["monaco", "race", "gp", "gran prix", "carrera"]):
        return "races"
    if any(w in t for w in ["debate", "hottake", "wouldyourather"]):
        return "fun"
    return "general"


def track_command(sessions: dict, user_id: str, command: str):
    """Tracks which commands a user uses."""
    if user_id in sessions:
        stats = sessions[user_id].setdefault("stats", {
            "total_messages": 0, "favorite_topics": {}, "commands_used": {}, "last_active": None
        })
        stats["commands_used"][command] = \
            stats["commands_used"].get(command, 0) + 1


# ═════════════════════════════════════════════════════════════
#  PREDICTION ACCURACY TRACKING
# ═════════════════════════════════════════════════════════════
def load_predictions() -> list:
    if PREDICTIONS_FILE.exists():
        try:
            return json.loads(PREDICTIONS_FILE.read_text())
        except Exception:
            pass
    return []

def save_prediction(race: str, predicted: str, actual: str | None = None):
    """Logs a prediction. actual=None until race happens."""
    preds = load_predictions()
    # Check if prediction for this race already exists
    for p in preds:
        if p["race"] == race:
            if actual:
                p["actual"]  = actual
                p["correct"] = predicted.upper() == actual.upper()
            return
    preds.append({
        "race":      race,
        "predicted": predicted,
        "actual":    actual,
        "correct":   None,
        "date":      datetime.now().isoformat(),
    })
    PREDICTIONS_FILE.write_text(json.dumps(preds, indent=2))

def get_prediction_accuracy() -> str:
    """Returns accuracy stats string for injection into system prompt."""
    preds = load_predictions()
    if not preds:
        return ""

    resolved = [p for p in preds if p.get("correct") is not None]
    if not resolved:
        return f"Predictions made this season: {len(preds)} (season ongoing, awaiting results)"

    correct = sum(1 for p in resolved if p["correct"])
    total   = len(resolved)
    pct     = round(correct / total * 100)

    recent = resolved[-3:]
    recent_str = " | ".join(
        f"{'✅' if p['correct'] else '❌'} {p['race']}: predicted {p['predicted']}, actual {p['actual']}"
        for p in recent
    )

    return (f"Prediction accuracy this season: {correct}/{total} correct ({pct}%)\n"
            f"Recent: {recent_str}")


# ═════════════════════════════════════════════════════════════
#  HTTP HELPERS
# ═════════════════════════════════════════════════════════════
def safe_get(url: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_standings() -> tuple[list, list]:
    """Returns (driver_standings, constructor_standings)."""
    d = safe_get(f"{JOLPICA}/{SEASON}/driverStandings.json")
    c = safe_get(f"{JOLPICA}/{SEASON}/constructorStandings.json")

    drivers = []
    if d:
        lists = d.get("MRData",{}).get("StandingsTable",{}).get("StandingsLists",[])
        if lists:
            drivers = lists[0].get("DriverStandings", [])

    constructors = []
    if c:
        lists = c.get("MRData",{}).get("StandingsTable",{}).get("StandingsLists",[])
        if lists:
            constructors = lists[0].get("ConstructorStandings", [])

    return drivers, constructors

def fetch_current_race() -> dict | None:
    """
    Gets the current race weekend.
    Checks today's date against known race dates first (reliable),
    then falls back to API.
    """
    today = datetime.now().date()

    # Hardcoded 2026 race dates — reliable fallback when APIs fail
    RACE_CALENDAR_2026 = [
        (1,  "Australian Grand Prix",     "2026-03-15"),
        (2,  "Chinese Grand Prix",        "2026-03-22"),
        (3,  "Japanese Grand Prix",       "2026-04-06"),
        (4,  "Miami Grand Prix",          "2026-05-04"),
        (5,  "Canadian Grand Prix",       "2026-05-24"),
        (6,  "Monaco Grand Prix",         "2026-06-07"),
        (7,  "Spanish Grand Prix",        "2026-06-14"),
        (8,  "Austrian Grand Prix",       "2026-06-28"),
        (9,  "British Grand Prix",        "2026-07-05"),
        (10, "Belgian Grand Prix",        "2026-07-19"),
        (11, "Hungarian Grand Prix",      "2026-07-26"),
        (12, "Dutch Grand Prix",          "2026-08-23"),
        (13, "Italian Grand Prix",        "2026-09-06"),
        (14, "Singapore Grand Prix",      "2026-09-20"),
        (15, "Azerbaijan Grand Prix",     "2026-09-27"),
        (16, "United States Grand Prix",  "2026-10-18"),
        (17, "Mexico City Grand Prix",    "2026-10-25"),
        (18, "São Paulo Grand Prix",      "2026-11-08"),
        (19, "Las Vegas Grand Prix",      "2026-11-21"),
        (20, "Qatar Grand Prix",          "2026-11-29"),
        (21, "Abu Dhabi Grand Prix",      "2026-12-06"),
    ]

    for rnd, name, date_str in RACE_CALENDAR_2026:
        try:
            race_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            delta = (race_date - today).days
            if -3 <= delta <= 1:  # within race weekend window
                return {
                    "round":    str(rnd),
                    "raceName": name,
                    "date":     date_str,
                    "Circuit":  {"circuitName": name.replace(" Grand Prix",""),
                                 "Location":    {"locality": name.replace(" Grand Prix",""),
                                                 "country": ""}},
                }
        except Exception:
            continue

    # Fall back to API
    try:
        data = safe_get(f"{JOLPICA}/{SEASON}/next.json")
        if data:
            races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
            if races:
                return races[0]
    except Exception:
        pass

    return None


def fetch_next_race() -> dict | None:
    data = safe_get(f"{JOLPICA}/{SEASON}/next.json")
    if not data:
        return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    return races[0] if races else None

def fetch_last_race() -> dict | None:
    data = safe_get(f"{JOLPICA}/{SEASON}/last/results.json")
    if not data:
        return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    return races[0] if races else None


def fetch_race_result(round_num: int, season: int = SEASON) -> dict | None:
    """Fetches full race result from Jolpica."""
    try:
        data = safe_get(
            f"{JOLPICA}/{season}/{round_num}/results.json")
        if not data:
            return None
        races = data.get("MRData",{}).get(
            "RaceTable",{}).get("Races",[])
        if not races:
            return None
        race    = races[0]
        results = race.get("Results",[])
        if not results:
            return None

        def get_driver(r):
            d = r.get("Driver",{})
            return d.get("code", d.get("familyName","?")[:3].upper())

        def get_team(r):
            return r.get("Constructor",{}).get("name","")

        teams = {get_driver(r): get_team(r) for r in results}

        p1 = get_driver(results[0]) if len(results)>0 else "?"
        p2 = get_driver(results[1]) if len(results)>1 else "?"
        p3 = get_driver(results[2]) if len(results)>2 else "?"

        NON_FINISHER_CODES = {"R", "D", "E", "W", "F"}
        dnfs = [
            f"{get_driver(r)}({r.get('status','')})"
            for r in results
            if r.get("positionText","") in NON_FINISHER_CODES
        ]

        fl    = next((get_driver(r) for r in results
                      if r.get("FastestLap",{}).get("rank")=="1"), None)
        fl_t  = next((r.get("FastestLap",{}).get("Time",{}).get("time","")
                      for r in results
                      if r.get("FastestLap",{}).get("rank")=="1"), "")
        pole  = next((get_driver(r) for r in results
                      if r.get("grid")=="1"), None)

        # Live standings after this race
        sd = safe_get(
            f"{JOLPICA}/{season}/{round_num}/driverStandings.json")
        champ_str = ""
        if sd:
            standings = (sd.get("MRData",{})
                          .get("StandingsTable",{})
                          .get("StandingsLists",[{}])[0]
                          .get("DriverStandings",[]))
            if standings:
                champ_str = " | ".join(
                    f"{s.get('Driver',{}).get('code','?')} {s.get('points','?')}pts"
                    for s in standings[:5])

        full_classification = [
            f"P{r.get('position','?')}:{get_driver(r)}"
            for r in results if r.get("position")]

        return {
            "round":              round_num,
            "track":              race.get("Circuit",{}).get(
                                    "Location",{}).get("locality","?"),
            "race_name":          race.get("raceName",""),
            "date":               race.get("date",""),
            "winner":             p1,
            "p2":                 p2,
            "p3":                 p3,
            "pole":               pole or "",
            "fastest_lap":        fl or "",
            "fastest_lap_time":   fl_t,
            "dnfs":               dnfs,
            "full_classification":full_classification,
            "teams":              teams,
            "champ_after":        champ_str,
            "ingested_at":        datetime.now().isoformat(),
        }
    except Exception as e:
        log.error(f"fetch_race_result R{round_num}: {e}")
        return None


def fetch_qualifying_result(round_num: int, season: int = SEASON) -> dict | None:
    """Fetches qualifying results for a given round from Jolpica."""
    try:
        data = safe_get(
            f"{JOLPICA}/{season}/{round_num}/qualifying.json")
        if not data:
            return None

        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None

        race    = races[0]
        results = race.get("QualifyingResults", [])
        if not results:
            return None

        def get_driver(r):
            d = r.get("Driver", {})
            return d.get("code", d.get("familyName", "?"))

        # Build grid top 10
        grid = []
        for r in results[:10]:
            code = get_driver(r)
            q3   = r.get("Q3", r.get("Q2", r.get("Q1", "")))
            grid.append(f"P{r.get('position','?')}: {code} ({q3})")

        pole        = get_driver(results[0]) if results else "?"
        pole_time   = results[0].get("Q3", results[0].get("Q2", "")) if results else ""
        front_row_2 = get_driver(results[1]) if len(results) > 1 else "?"

        return {
            "round":      round_num,
            "race_name":  race.get("raceName", ""),
            "pole":       pole,
            "pole_time":  pole_time,
            "front_row":  f"{pole} / {front_row_2}",
            "grid_top10": grid,
            "date":       race.get("date", ""),
            "ingested_at": datetime.now().isoformat(),
        }

    except Exception as e:
        log.error(f"fetch_qualifying_result R{round_num}: {e}")
        return None


def fetch_sprint_result(round_num: int, season: int = SEASON) -> dict | None:
    """Fetches sprint race results for a given round from Jolpica."""
    try:
        data = safe_get(
            f"{JOLPICA}/{season}/{round_num}/sprint.json")
        if not data:
            return None

        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None

        race    = races[0]
        results = race.get("SprintResults", [])
        if not results:
            return None

        def get_driver(r):
            d = r.get("Driver", {})
            return d.get("code", d.get("familyName", "?"))

        p1 = get_driver(results[0]) if len(results) > 0 else "?"
        p2 = get_driver(results[1]) if len(results) > 1 else "?"
        p3 = get_driver(results[2]) if len(results) > 2 else "?"

        return {
            "round":     round_num,
            "race_name": race.get("raceName", ""),
            "winner":    p1,
            "p2":        p2,
            "p3":        p3,
            "story":     f"Sprint: {p1} wins. P2: {p2}, P3: {p3}.",
            "ingested_at": datetime.now().isoformat(),
        }

    except Exception as e:
        log.error(f"fetch_sprint_result R{round_num}: {e}")
        return None


def fetch_driver_career_stats(driver_code: str) -> str:
    """
    Fetches career stats for a driver from Jolpica.
    Returns formatted string.
    """
    # Map common codes to Jolpica driver IDs
    driver_ids = {
        "HAM": "hamilton", "VER": "max_verstappen", "LEC": "leclerc",
        "RUS": "russell",  "NOR": "norris",         "PIA": "piastri",
        "ALO": "alonso",   "SAI": "sainz",          "GAS": "gasly",
        "PER": "perez",    "STR": "stroll",         "BOT": "bottas",
        "OCO": "ocon",     "HUL": "hulkenberg",     "ALB": "albon",
        "ANT": "antonelli","LAW": "lawson",          "HAD": "hadjar",
        "BEA": "bearman",  "COL": "colapinto",      "BOR": "bortoleto",
        "LIN": "lindblad",
    }

    did = driver_ids.get(driver_code.upper())
    if not did:
        return ""

    try:
        data = safe_get(f"{JOLPICA}/drivers/{did}/results.json",
                        {"limit": 1000})
        if not data:
            return ""

        races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
        if not races:
            return ""

        wins    = sum(1 for r in races if r.get("Results",[{}])[0].get("position") == "1")
        podiums = sum(1 for r in races
                      if int(r.get("Results",[{}])[0].get("position","99")) <= 3)
        poles   = sum(1 for r in races
                      if r.get("Results",[{}])[0].get("grid") == "1")
        total   = len(races)

        return (f"Career stats for {driver_code}: "
                f"{total} races, {wins} wins, {podiums} podiums, {poles} poles")
    except Exception:
        return ""


# ═════════════════════════════════════════════════════════════
#  OPENF1
# ═════════════════════════════════════════════════════════════
def fetch_openf1(endpoint: str, params: dict) -> list:
    """Generic OpenF1 API call."""
    try:
        url = f"https://api.openf1.org/v1/{endpoint}"
        r   = requests.get(url, params=params, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def fetch_live_session() -> dict | None:
    """Checks OpenF1 for any currently active or very recent session.
    Returns session dict or None."""
    try:
        now = datetime.utcnow()
        sessions_data = fetch_openf1("sessions", {"year": now.year})
        if not sessions_data:
            return None

        # Find sessions that started recently (within last 4 hours) or are upcoming
        recent = []
        for s in sessions_data:
            date_start = s.get("date_start", "")
            date_end   = s.get("date_end", "")
            if not date_start:
                continue
            try:
                start = datetime.fromisoformat(date_start.replace("Z",""))
                end   = datetime.fromisoformat(date_end.replace("Z","")) \
                        if date_end else start + timedelta(hours=3)
                # Active if started within 4h and not ended more than 1h ago
                if (now - start).total_seconds() < 4*3600 and \
                   (now - end).total_seconds() < 3600:
                    recent.append(s)
            except Exception:
                continue

        if not recent:
            return None
        # Return most recently started
        return sorted(recent, key=lambda x: x.get("date_start",""))[-1]
    except Exception:
        return None


def get_live_session_context() -> str:
    """Returns live session data if a session is currently active."""
    session = fetch_live_session()
    if not session:
        return ""

    sk        = session.get("session_key")
    name      = session.get("session_name", "")
    circuit   = session.get("circuit_short_name", "")
    date_end  = session.get("date_end", "")

    if not sk:
        return ""

    # Check if session is actually live (not ended)
    try:
        end_dt = datetime.fromisoformat(date_end.replace("Z",""))
        if end_dt < datetime.utcnow():
            return ""  # session ended
    except Exception:
        pass

    # Fetch latest driver positions
    try:
        positions = fetch_openf1("position", {
            "session_key": sk,
            "date":        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if not positions:
            return f"LIVE SESSION: {name} at {circuit} is currently happening."

        # Get latest position for each driver
        latest = {}
        for p in positions:
            drv = str(p.get("driver_number", ""))
            if drv not in latest or p.get("date","") > latest[drv].get("date",""):
                latest[drv] = p

        order = sorted(latest.values(), key=lambda x: x.get("position", 99))[:10]

        OF1_DRV = {
            "1":"ANT","63":"RUS","44":"HAM","16":"LEC","6":"LAW",
            "5":"HAD","4":"NOR","81":"PIA","14":"ALO","18":"STR",
            "10":"GAS","43":"COL","23":"ALB","55":"SAI","30":"LIN",
            "87":"BEA","31":"OCO","27":"HUL","11":"PER","77":"BOT",
            "41":"BOR",
        }

        pos_str = " | ".join(
            f"P{p['position']}:{OF1_DRV.get(str(p.get('driver_number','')),str(p.get('driver_number','')))}"
            for p in order if p.get("position")
        )

        return (f"🔴 LIVE RIGHT NOW: {name} at {circuit}\n"
                f"Current order: {pos_str}")
    except Exception:
        return f"LIVE SESSION: {name} at {circuit} is currently happening."


# ═════════════════════════════════════════════════════════════
#  PRACTICE / SESSION DATA
# ═════════════════════════════════════════════════════════════
def fetch_practice_results(round_num: int, season: int = SEASON) -> str:
    """
    Fetches FP1/FP2/FP3 results from OpenF1 for a given round.
    Uses meeting_key matching for reliability.
    """
    # Step 1: get the meeting_key for this round via meetings endpoint
    try:
        meetings = fetch_openf1("meetings", {"year": season})
        if not meetings:
            return ""

        # Sort meetings by date and pick by round index
        sorted_meetings = sorted(meetings, key=lambda x: x.get("date_start",""))
        if round_num < 1 or round_num > len(sorted_meetings):
            return ""

        meeting = sorted_meetings[round_num - 1]
        meeting_key = meeting.get("meeting_key")
        circuit     = meeting.get("circuit_short_name", "")
        if not meeting_key:
            return ""

    except Exception as e:
        log.warning(f"Meeting lookup failed: {e}")
        return ""

    # Step 2: get all sessions for this meeting
    try:
        sessions_data = fetch_openf1("sessions", {"meeting_key": meeting_key})
        if not sessions_data:
            return ""
    except Exception as e:
        log.warning(f"Sessions lookup failed: {e}")
        return ""

    session_types = ["Practice 1", "Practice 2", "Practice 3"]
    results       = []
    now           = datetime.utcnow()

    for session_name in session_types:
        matching = [s for s in sessions_data
                    if s.get("session_name") == session_name]
        if not matching:
            continue

        session = matching[0]
        sk      = session.get("session_key")
        if not sk:
            continue

        # Check session has ended (with 30min buffer for data to appear)
        end_date = session.get("date_end", "")
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", ""))
                if end_dt > now:
                    # Session is still live — include with note
                    results.append(f"*{session_name}:* 🔴 Currently in progress")
                    continue
            except Exception:
                pass

        try:
            # Get laps and drivers
            laps         = fetch_openf1("laps", {"session_key": sk})
            drivers_data = fetch_openf1("drivers", {"session_key": sk})

            num_to_name = {}
            for d in (drivers_data or []):
                num  = str(d.get("driver_number", ""))
                code = d.get("name_acronym", num)
                if num and code:
                    num_to_name[num] = code

            if not laps:
                results.append(f"*{session_name}:* No lap data yet")
                continue

            # Best lap per driver
            best = {}
            for lap in laps:
                dur = lap.get("lap_duration")
                num = str(lap.get("driver_number", ""))
                if not dur or not num:
                    continue
                try:
                    dur_f = float(dur)
                    if dur_f > 20 and (num not in best or dur_f < best[num]):
                        best[num] = dur_f
                except Exception:
                    pass

            if not best:
                results.append(f"*{session_name}:* No valid lap times yet")
                continue

            sorted_best = sorted(best.items(), key=lambda x: x[1])[:10]
            lines       = [f"*{session_name} — {circuit}:*"]
            for i, (num, lap_time) in enumerate(sorted_best, 1):
                code  = num_to_name.get(num, f"#{num}")
                mins  = int(lap_time // 60)
                secs  = lap_time % 60
                t_str = f"{mins}:{secs:06.3f}"
                gap   = f" (+{lap_time - sorted_best[0][1]:.3f}s)" if i > 1 else " 🔥"
                lines.append(f"P{i}: *{code}* {t_str}{gap}")

            results.append("\n".join(lines))
            time.sleep(0.3)

        except Exception as e:
            log.warning(f"FP data fetch failed for {session_name}: {e}")
            continue

    return "\n\n".join(results) if results else ""


def get_actual_grid_for_prediction(round_num: int | None = None) -> str:
    """
    Fetches the ACTUAL qualifying grid for the upcoming/current race
    so prediction prompts are grounded in reality, not invented.

    Tries OpenF1 (live timing) first, then Jolpica (posted results).
    Returns empty string if no grid is available yet — callers must
    handle this by telling Claude qualifying hasn't happened.
    """
    # Try OpenF1 first — most current
    try:
        sessions_data = fetch_openf1("sessions", {
            "year": SEASON, "session_name": "Qualifying"})
        if sessions_data:
            now_ts = datetime.now().timestamp()
            completed = []
            for s in sessions_data:
                date_end = s.get("date_end", "")
                if not date_end:
                    continue
                try:
                    end_dt = datetime.fromisoformat(
                        date_end.replace("Z", "+00:00"))
                    if end_dt.timestamp() <= now_ts:
                        completed.append(s)
                except Exception:
                    continue
            if completed:
                recent = sorted(completed, key=lambda x: x.get("date_end",""))[-1]
                sk = recent.get("session_key")
                if sk:
                    laps = fetch_openf1("laps", {"session_key": sk})
                    if laps:
                        drivers_raw = fetch_openf1("drivers", {"session_key": sk})
                        num_to_code = {}
                        if drivers_raw:
                            for d in drivers_raw:
                                n = str(d.get("driver_number",""))
                                num_to_code[n] = d.get("name_acronym","?")
                        best: dict = {}
                        for lap in laps:
                            dur = lap.get("lap_duration")
                            num = str(lap.get("driver_number",""))
                            if dur and num:
                                try:
                                    f = float(dur)
                                    if f > 30:
                                        if num not in best or f < best[num]:
                                            best[num] = f
                                except Exception:
                                    pass
                        if best:
                            sorted_d = sorted(best.items(), key=lambda x: x[1])
                            grid = [num_to_code.get(num, f"#{num}")
                                   for num, _ in sorted_d[:10]]
                            return ("ACTUAL QUALIFYING GRID (top 10, from OpenF1): "
                                    + ", ".join(f"P{i+1}:{c}" for i,c in enumerate(grid)))
    except Exception as e:
        log.debug(f"get_actual_grid_for_prediction OpenF1 failed: {e}")

    # Fallback: Jolpica
    try:
        if round_num is None:
            next_race = fetch_next_race()
            round_num = int(next_race.get("round", 0)) if next_race else 0
        if round_num:
            quali = fetch_qualifying_result(round_num, SEASON)
            if quali and quali.get("grid_top10"):
                grid_str = " | ".join(quali["grid_top10"])
                return f"ACTUAL QUALIFYING GRID (top 10, from Jolpica): {grid_str}"
    except Exception as e:
        log.debug(f"get_actual_grid_for_prediction Jolpica failed: {e}")

    return ""


def get_session_context(query: str, live_search_fn=None):
    """
    Fetches real session timing data from OpenF1 first.
    Falls back to live search only if OpenF1 has no data.
    Never returns empty and asks user to paste results.

    live_search_fn: callable — pass live_search_f1 from boxboxai_bot.
    Kept as injection to avoid a circular import (live_search_f1 stays
    in the bot until step 5).
    """
    q = query.lower()

    # Detect session type
    if any(kw in q for kw in ["fp1","practice 1","libre 1","free practice 1"]):
        session_name = "Practice 1"
        session_label = "FP1"
    elif any(kw in q for kw in ["fp2","practice 2","libre 2","free practice 2"]):
        session_name = "Practice 2"
        session_label = "FP2"
    elif any(kw in q for kw in ["fp3","practice 3","libre 3","free practice 3"]):
        session_name = "Practice 3"
        session_label = "FP3"
    elif any(kw in q for kw in ["sprint quali","sprint qualifying","sq"]):
        session_name = "Sprint Qualifying"
        session_label = "Sprint Qualifying"
    elif any(kw in q for kw in ["sprint race","carrera sprint"]):
        session_name = "Sprint"
        session_label = "Sprint Race"
    elif any(kw in q for kw in ["qualifying","quali","clasificación","clasificacion",
                                  "q1","q2","q3","pole"]):
        session_name = "Qualifying"
        session_label = "Qualifying"
    elif any(kw in q for kw in ["sprint"]):
        session_name = "Sprint"
        session_label = "Sprint"
    elif any(kw in q for kw in ["practice","práctica","practica","libre","entreno"]):
        # Generic practice — try FP2 first (most recent)
        session_name = "Practice 2"
        session_label = "FP2"
    else:
        return ""

    # ── Try OpenF1 first — ground truth timing data ───────────
    try:
        sessions_data = fetch_openf1("sessions", {
            "year": SEASON,
            "session_name": session_name,
        })

        if not sessions_data:
            log.info(f"OpenF1: no '{session_name}' sessions returned for {SEASON}")

        if sessions_data:
            now_ts = datetime.now().timestamp()

            # Only consider sessions that have ALREADY ENDED
            completed = []
            for s in sessions_data:
                date_end = s.get("date_end", "")
                if not date_end:
                    continue
                try:
                    end_dt = datetime.fromisoformat(
                        date_end.replace("Z", "+00:00"))
                    if end_dt.timestamp() <= now_ts:
                        completed.append(s)
                except Exception:
                    continue

            if not completed:
                log.info(
                    f"OpenF1: {len(sessions_data)} '{session_name}' "
                    f"sessions found but none completed yet")

            if completed:
                # Most recently completed
                recent = sorted(
                    completed,
                    key=lambda x: x.get("date_end",""))[-1]
                sk = recent.get("session_key")
                circuit = recent.get("circuit_short_name","")
                date    = recent.get("date_start","")[:10]

                log.info(
                    f"OpenF1: using session_key={sk} "
                    f"circuit={circuit} date={date}")

                if sk:
                    # Fetch lap times
                    laps = fetch_openf1("laps", {"session_key": sk})

                    if not laps:
                        log.info(f"OpenF1: session {sk} has no lap data")

                    if laps:
                        # Get driver info
                        drivers_raw = fetch_openf1("drivers",
                                                   {"session_key": sk})
                        num_to_code = {}
                        num_to_team = {}
                        if drivers_raw:
                            for d in drivers_raw:
                                n = str(d.get("driver_number",""))
                                num_to_code[n] = d.get("name_acronym","?")
                                num_to_team[n] = d.get("team_name","")[:12]

                        # Best lap per driver
                        best: dict = {}
                        for lap in laps:
                            dur = lap.get("lap_duration")
                            num = str(lap.get("driver_number",""))
                            if dur and num:
                                try:
                                    f = float(dur)
                                    if f > 30:  # filter outliers
                                        if num not in best or f < best[num]:
                                            best[num] = f
                                except Exception:
                                    pass

                        if best:
                            sorted_d = sorted(
                                best.items(), key=lambda x: x[1])
                            ref = sorted_d[0][1]

                            lines = [
                                f"{session_label} CLASSIFICATION — "
                                f"{circuit} {date}:"]
                            for i,(num,t) in enumerate(sorted_d[:15], 1):
                                code = num_to_code.get(num, f"#{num}")
                                team = num_to_team.get(num,"")
                                mins = int(t//60)
                                secs = t%60
                                if i == 1:
                                    time_str = f"{mins}:{secs:06.3f}"
                                    gap = ""
                                else:
                                    delta = t - ref
                                    time_str = f"{mins}:{secs:06.3f}"
                                    gap = f" +{delta:.3f}s"
                                lines.append(
                                    f"P{i:2d}: {code:4s} ({team})"
                                    f" {time_str}{gap}")

                            log.info(
                                f"OpenF1 {session_label} data: "
                                f"{len(sorted_d)} drivers")
                            try:
                                end_dt = datetime.fromisoformat(
                                    recent.get("date_end","").replace("Z","+00:00"))
                                session_age_h = (
                                    datetime.now(end_dt.tzinfo) - end_dt
                                ).total_seconds() / 3600
                            except Exception:
                                session_age_h = None
                            return ContextBlock(
                                content="\n".join(lines),
                                data_age_hours=session_age_h,
                            )
    except Exception as e:
        log.warning(f"OpenF1 session context failed: {e}")

    # ── Fallback: live search (injected to avoid circular import) ──
    current = fetch_current_race()
    race_name = current.get("raceName","") if current else ""
    search_q  = f"{race_name} {session_label} 2026 results classification"
    if live_search_fn:
        live = live_search_fn(search_q)
        if live:
            return ContextBlock(content=live, completeness="unknown")

    # ── Last resort: news cache ───────────────────────────────
    news = get_news_context(
        f"{race_name} {session_label} 2026 results")
    return ContextBlock(content=news, completeness="unknown") if news else ""


def get_practice_context(query: str, next_race: dict | None = None,
                          mem: dict | None = None) -> str:
    """
    Gets practice session info by searching news sources.
    Much more reliable than OpenF1 lap data parsing.
    """
    q = query.lower()
    is_practice = any(kw in q for kw in [
        "fp1", "fp2", "fp3", "practice", "práctica", "libre",
        "free practice", "entreno", "entrenamiento"
    ])
    if not is_practice:
        return ""

    # Figure out which race weekend we're talking about
    current = fetch_current_race()
    race_name = ""
    if current:
        race_name = current.get("raceName", "")

    # Build search query
    fp_type = ""
    if "fp1" in q or "practice 1" in q or "libre 1" in q:
        fp_type = "FP1 Free Practice 1"
    elif "fp2" in q or "practice 2" in q or "libre 2" in q:
        fp_type = "FP2 Free Practice 2"
    elif "fp3" in q or "practice 3" in q or "libre 3" in q:
        fp_type = "FP3 Free Practice 3"
    else:
        fp_type = "Free Practice"

    search_query = f"{race_name} {fp_type} 2026 results fastest"

    # Search news
    news = get_news_context(search_query)
    if news:
        return f"Practice session news for {race_name} {fp_type}:\n{news}"

    # Also search DuckDuckGo
    try:
        ddg_url = "https://api.duckduckgo.com/"
        r = requests.get(ddg_url, params={
            "q": search_query, "format": "json", "no_html": "1"
        }, timeout=8)
        if r.status_code == 200:
            data = r.json()
            parts = []
            if data.get("AbstractText"):
                parts.append(data["AbstractText"])
            for t in data.get("RelatedTopics", [])[:3]:
                if isinstance(t, dict) and t.get("Text"):
                    parts.append(t["Text"])
            if parts:
                return f"Practice info for {race_name} {fp_type}:\n{' '.join(parts)[:800]}"
    except Exception:
        pass

    return f"Searching for {race_name} {fp_type} results — check /news for latest updates."


# ═════════════════════════════════════════════════════════════
#  FASTF1 / OPENF1 TELEMETRY ENRICHMENT
# ═════════════════════════════════════════════════════════════
def _safe_ff1_load(year: int, round_num: int,
                   session_type: str) -> object | None:
    """Safely loads a FastF1 session. Returns None if unavailable."""
    if not _FF1_AVAILABLE:
        return None
    try:
        import fastf1 as _ff1
        sess = _ff1.get_session(year, round_num, session_type)
        sess.load(laps=True, telemetry=False,
                  weather=True, messages=True)
        if sess.laps is None or len(sess.laps) == 0:
            return None
        return sess
    except Exception as e:
        log.debug(f"FastF1 load failed R{round_num} {session_type}: {e}")
        return None


def _ff1_session_summary(sess) -> dict:
    """
    Extracts rich summary from a FastF1 session object.
    Returns structured dict with everything needed for memory.
    """
    import pandas as pd
    import numpy as np

    summary = {
        "source": "fastf1",
        "top5_times": [],
        "full_order": [],
        "fastest_lap": {"driver": "", "time": "", "lap": 0},
        "sector_bests": {},
        "tyre_stints": {},
        "pit_stops": {},
        "weather": {},
        "incidents": [],
    }

    try:
        laps = sess.laps

        # Full finishing/classification order
        results = sess.results
        if results is not None and not results.empty:
            code_col = next((c for c in
                ["Abbreviation","abbreviation","Driver"] if c in results.columns), None)
            pos_col  = next((c for c in
                ["Position","ClassifiedPosition","position"] if c in results.columns), None)
            if code_col and pos_col:
                ordered = results.sort_values(pos_col)
                summary["full_order"] = ordered[code_col].tolist()[:20]

        # Fastest lap overall
        valid = laps.dropna(subset=["LapTime"])
        valid = valid[valid["LapTime"].notna()]
        if not valid.empty:
            fl_idx  = valid["LapTime"].idxmin()
            fl_row  = valid.loc[fl_idx]
            fl_secs = fl_row["LapTime"].total_seconds()
            mins    = int(fl_secs // 60)
            secs    = fl_secs % 60
            summary["fastest_lap"] = {
                "driver": str(fl_row.get("Driver", "")),
                "time":   f"{mins}:{secs:06.3f}",
                "lap":    int(fl_row.get("LapNumber", 0)),
            }

        # Top 5 fastest laps per driver
        best_per_driver = (
            valid.groupby("Driver")["LapTime"]
            .min()
            .sort_values()
            .head(5)
        )
        ref = best_per_driver.iloc[0].total_seconds() if not best_per_driver.empty else 0
        for drv, lt in best_per_driver.items():
            secs   = lt.total_seconds()
            mins   = int(secs // 60)
            s_part = secs % 60
            delta  = secs - ref
            summary["top5_times"].append({
                "driver": str(drv),
                "time":   f"{mins}:{s_part:06.3f}",
                "delta":  f"+{delta:.3f}" if delta > 0 else "leader",
            })

        # Sector bests
        for s_col, s_name in [
            ("Sector1Time","S1"),
            ("Sector2Time","S2"),
            ("Sector3Time","S3"),
        ]:
            if s_col in laps.columns:
                sv = laps.dropna(subset=[s_col])
                if not sv.empty:
                    best_idx = sv[s_col].idxmin()
                    best_row = sv.loc[best_idx]
                    t = best_row[s_col].total_seconds()
                    summary["sector_bests"][s_name] = {
                        "driver": str(best_row.get("Driver","")),
                        "time":   f"{t:.3f}s",
                    }

        # Tyre stints
        if "Compound" in laps.columns and "Stint" in laps.columns:
            for (drv, stint), grp in laps.groupby(["Driver","Stint"]):
                compound = grp["Compound"].iloc[0] if not grp.empty else "?"
                n_laps   = len(grp)
                if str(drv) not in summary["tyre_stints"]:
                    summary["tyre_stints"][str(drv)] = []
                summary["tyre_stints"][str(drv)].append({
                    "compound": str(compound),
                    "laps":     n_laps,
                })

        # Pit stops (from laps PitInTime/PitOutTime)
        if "PitInTime" in laps.columns:
            pit_laps = laps[laps["PitInTime"].notna()]
            for drv, grp in pit_laps.groupby("Driver"):
                summary["pit_stops"][str(drv)] = len(grp)

        # Weather summary
        try:
            weather = sess.weather_data
            if weather is not None and not weather.empty:
                summary["weather"] = {
                    "air_temp":   round(float(weather["AirTemp"].mean()), 1),
                    "track_temp": round(float(weather["TrackTemp"].mean()), 1),
                    "rainfall":   bool(weather["Rainfall"].any()),
                    "humidity":   round(float(weather["Humidity"].mean()), 1),
                }
        except Exception:
            pass

        # Race control messages (incidents, SC, flags)
        try:
            rc = sess.race_control_messages
            if rc is not None and not rc.empty:
                for _, msg in rc.iterrows():
                    text = str(msg.get("Message",""))
                    cat  = str(msg.get("Category",""))
                    lap  = msg.get("Lap", "?")
                    if any(kw in text.upper() for kw in
                           ["SAFETY CAR","RED FLAG","RETIRED",
                            "PENALTY","INCIDENT","VSC","BLACK"]):
                        summary["incidents"].append(
                            f"L{lap} [{cat}]: {text[:100]}")
        except Exception:
            pass

    except Exception as e:
        log.warning(f"FF1 summary extraction error: {e}")

    return summary


def _openf1_session_summary(round_num: int,
                             session_type: str = "Race") -> dict:
    """
    OpenF1 fallback when FastF1 is unavailable.
    Returns same structure as _ff1_session_summary.
    """
    summary = {
        "source": "openf1",
        "top5_times": [], "full_order": [],
        "fastest_lap": {"driver":"","time":"","lap":0},
        "sector_bests": {}, "tyre_stints": {},
        "pit_stops": {}, "weather": {}, "incidents": [],
    }

    try:
        # Get session key
        sessions_data = fetch_openf1("sessions", {
            "year": SEASON, "session_name": session_type})
        if not sessions_data:
            return summary

        race_sessions = [s for s in sessions_data
                          if s.get("session_name") == session_type]
        if not race_sessions:
            return summary

        # Match by date proximity to this round's actual race weekend,
        # not by list-index (race_sessions[round_num-1] assumes OpenF1's
        # session list has no gaps and starts at round 1 — fragile,
        # and was the same bug pattern fixed in
        # _fetch_session_results_openf1).
        session = None
        try:
            race_info = safe_get(f"{JOLPICA}/{SEASON}/{round_num}.json")
            races = race_info.get("MRData",{}).get("RaceTable",{}).get("Races",[]) \
                if race_info else []
            race_date_str = races[0].get("date","") if races else ""
            if race_date_str:
                race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
                def _date_diff(s):
                    try:
                        sd = datetime.fromisoformat(
                            s.get("date_start","").replace("Z","+00:00"))
                        return abs((sd.replace(tzinfo=None) - race_date).days)
                    except Exception:
                        return 999
                candidates = [s for s in race_sessions if _date_diff(s) <= 4]
                if candidates:
                    session = min(candidates, key=_date_diff)
        except Exception:
            pass

        if session is None:
            # Fallback to old index-based matching if date lookup failed
            race_sessions_sorted = sorted(
                race_sessions, key=lambda x: x.get("date_start",""))
            if round_num > len(race_sessions_sorted):
                return summary
            session = race_sessions_sorted[round_num-1]

        sk = session.get("session_key")
        if not sk:
            return summary

        # Lap times
        laps = fetch_openf1("laps", {"session_key": sk})
        if laps:
            driver_best = {}
            for lap in laps:
                dur = lap.get("lap_duration")
                drv = str(lap.get("driver_number",""))
                if dur and drv:
                    try:
                        dur_f = float(dur)
                        if dur_f > 0:
                            if drv not in driver_best or dur_f < driver_best[drv]:
                                driver_best[drv] = dur_f
                    except Exception:
                        pass

            # Map driver numbers to codes
            driver_map_data = fetch_openf1("drivers",
                                            {"session_key": sk})
            num_to_code = {}
            if driver_map_data:
                for d in driver_map_data:
                    num  = str(d.get("driver_number",""))
                    code = d.get("name_acronym","")
                    if num and code:
                        num_to_code[num] = code

            if driver_best:
                sorted_drivers = sorted(
                    driver_best.items(), key=lambda x: x[1])
                ref = sorted_drivers[0][1]
                for num, t in sorted_drivers[:5]:
                    code  = num_to_code.get(num, num)
                    mins  = int(t // 60)
                    secs  = t % 60
                    delta = t - ref
                    summary["top5_times"].append({
                        "driver": code,
                        "time":   f"{mins}:{secs:06.3f}",
                        "delta":  f"+{delta:.3f}" if delta > 0 else "leader",
                    })
                # Fastest lap
                fl_num, fl_t = sorted_drivers[0]
                fl_code = num_to_code.get(fl_num, fl_num)
                mins    = int(fl_t // 60)
                secs    = fl_t % 60
                summary["fastest_lap"] = {
                    "driver": fl_code,
                    "time":   f"{mins}:{secs:06.3f}",
                    "lap":    0,
                }

        # Stints / tyres
        stints = fetch_openf1("stints", {"session_key": sk})
        if stints:
            for st in stints:
                num      = str(st.get("driver_number",""))
                code     = num_to_code.get(num, num)
                compound = st.get("compound","?")
                lap_s    = st.get("lap_start",0) or 0
                lap_e    = st.get("lap_end",0)   or 0
                n_laps   = max(0, int(lap_e) - int(lap_s))
                if code not in summary["tyre_stints"]:
                    summary["tyre_stints"][code] = []
                summary["tyre_stints"][code].append({
                    "compound": str(compound), "laps": n_laps})

        # Race control
        rc = fetch_openf1("race_control", {"session_key": sk})
        if rc:
            for msg in rc:
                text = str(msg.get("message",""))
                lap  = msg.get("lap_number","?")
                if any(kw in text.upper() for kw in
                       ["SAFETY CAR","RED FLAG","RETIRED",
                        "PENALTY","INCIDENT","VSC"]):
                    summary["incidents"].append(
                        f"L{lap}: {text[:100]}")

    except Exception as e:
        log.warning(f"OpenF1 summary error R{round_num} {session_type}: {e}")

    return summary


def enrich_episode_with_telemetry(episode: dict,
                                   round_num: int) -> dict:
    """
    Enriches a race episode with full telemetry from FastF1/OpenF1.
    Adds: full lap times, sector bests, tyre strategies, weather,
    incidents, pit stop counts, full classification order.
    """
    log.info(f"Enriching R{round_num} with telemetry...")

    # Try FastF1 first
    sess = _safe_ff1_load(SEASON, round_num, "Race")
    if sess:
        telemetry = _ff1_session_summary(sess)
        log.info(f"R{round_num} enriched via FastF1 ✅")
    else:
        telemetry = _openf1_session_summary(round_num, "Race")
        log.info(f"R{round_num} enriched via OpenF1")

    if not telemetry:
        return episode

    # Merge into episode
    # Jolpica's official classification (set during initial ingestion) is authoritative
    # and includes post-race penalty adjustments. Only fill from FastF1's on-track
    # order if no official classification exists yet (e.g. episode created manually).
    if telemetry.get("full_order") and not episode.get("full_classification"):
        episode["full_classification"] = [
            f"P{i+1}:{c}" for i, c in
            enumerate(telemetry["full_order"])]

    if telemetry.get("fastest_lap", {}).get("driver"):
        fl = telemetry["fastest_lap"]
        episode["fastest_lap"]      = fl["driver"]
        episode["fastest_lap_time"] = fl["time"]

    if telemetry.get("top5_times"):
        episode["top5_lap_times"] = telemetry["top5_times"]

    if telemetry.get("sector_bests"):
        episode["sector_bests"] = telemetry["sector_bests"]

    if telemetry.get("tyre_stints"):
        # Build strategy summary
        strategies = []
        for drv, stints in list(telemetry["tyre_stints"].items())[:8]:
            strat = "→".join(
                f"{(s.get('compound') or '')[:1]}({s['laps']})" for s in stints)
            strategies.append(f"{drv}:{strat}")
        episode.setdefault("pitstops", {})["tyre_strategies"] = \
            " | ".join(strategies[:6])
        episode["pitstops"]["stop_counts"] = {
            drv: len(stints) - 1
            for drv, stints in telemetry["tyre_stints"].items()
            if len(stints) > 1
        }

    if telemetry.get("weather"):
        w = telemetry["weather"]
        episode["weather"] = w
        weather_str = (
            f"Air {w.get('air_temp','?')}°C, "
            f"Track {w.get('track_temp','?')}°C"
            + (", 🌧 Rain" if w.get("rainfall") else ", ☀ Dry")
        )
        episode.setdefault("pitstops",{})["weather"] = weather_str

    if telemetry.get("incidents"):
        episode["race_control"] = telemetry["incidents"][:15]

    episode["telemetry_source"] = telemetry.get("source","unknown")
    episode["telemetry_loaded"] = datetime.now().isoformat()

    return episode


def enrich_qualifying_with_telemetry(episode: dict,
                                      round_num: int) -> dict:
    """Enriches qualifying data with full sector times and lap data."""
    log.info(f"Enriching R{round_num} qualifying with telemetry...")

    sess = _safe_ff1_load(SEASON, round_num, "Qualifying")
    if not sess:
        return episode

    try:
        import pandas as pd
        laps  = sess.laps
        valid = laps.dropna(subset=["LapTime"])

        # Best lap per driver in Q3, Q2, Q1
        q_data = {"Q3": {}, "Q2": {}, "Q1": {}}
        if "Session" in laps.columns:
            for q_sess in ["Q3","Q2","Q1"]:
                q_laps = valid[valid["Session"] == q_sess] \
                    if "Session" in valid.columns else valid
                if q_laps.empty:
                    continue
                best = q_laps.groupby("Driver")["LapTime"].min()
                for drv, lt in best.items():
                    t    = lt.total_seconds()
                    mins = int(t // 60)
                    secs = t % 60
                    q_data[q_sess][str(drv)] = f"{mins}:{secs:06.3f}"
        else:
            # No session column — use all laps
            best = valid.groupby("Driver")["LapTime"].min()
            for drv, lt in best.items():
                t    = lt.total_seconds()
                mins = int(t // 60)
                secs = t % 60
                q_data["Q3"][str(drv)] = f"{mins}:{secs:06.3f}"

        # Build rich qualifying summary
        quali = episode.setdefault("qualifying", {})

        # Full Q3 order
        if q_data["Q3"]:
            sorted_q3 = sorted(q_data["Q3"].items(), key=lambda x: x[1])
            quali["full_q3"] = [
                f"P{i+1}:{drv}({t})"
                for i,(drv,t) in enumerate(sorted_q3)]
            if sorted_q3:
                quali["pole"]      = sorted_q3[0][0]
                quali["pole_time"] = sorted_q3[0][1]

        # Q2 elimination (P11-P15)
        if q_data["Q2"]:
            sorted_q2 = sorted(q_data["Q2"].items(), key=lambda x: x[1])
            q2_drivers = [d for d,_ in sorted_q2]
            q3_drivers = set(q_data["Q3"].keys())
            eliminated_q2 = [d for d in q2_drivers if d not in q3_drivers]
            quali["eliminated_q2"] = eliminated_q2[:5]

        # Q1 elimination (P16-P20)
        if q_data["Q1"]:
            sorted_q1 = sorted(q_data["Q1"].items(), key=lambda x: x[1])
            q1_drivers = [d for d,_ in sorted_q1]
            q2_drivers = set(q_data["Q2"].keys()) | set(q_data["Q3"].keys())
            eliminated_q1 = [d for d in q1_drivers if d not in q2_drivers]
            quali["eliminated_q1"] = eliminated_q1[:5]

        # Sector bests across all quali laps
        for s_col, s_name in [
            ("Sector1Time","S1"),
            ("Sector2Time","S2"),
            ("Sector3Time","S3"),
        ]:
            if s_col in laps.columns:
                sv = laps.dropna(subset=[s_col])
                if not sv.empty:
                    best_idx = sv[s_col].idxmin()
                    best_row = sv.loc[best_idx]
                    t = best_row[s_col].total_seconds()
                    quali.setdefault("sector_bests",{})[s_name] = {
                        "driver": str(best_row.get("Driver","")),
                        "time":   f"{t:.3f}s",
                    }

        episode["qualifying"] = quali
        log.info(f"R{round_num} qualifying enriched via FastF1 ✅")

    except Exception as e:
        log.warning(f"Qualifying enrichment error R{round_num}: {e}")

    return episode


def build_rich_story(episode: dict, ask_fn=None) -> str:
    """
    Builds a detailed narrative story from all available data.
    This is what Claude reads to answer questions accurately.

    ask_fn: reserved for step 5 (context_builder.py extraction) when
    build_rich_story may gain an LLM narrative pass. Currently unused —
    the function assembles the story purely from episode data.
    """
    parts = []
    rname = episode.get("race_name", episode.get("track","?"))
    rnd   = episode.get("round","?")

    # Result
    parts.append(
        f"R{rnd} {rname}: {episode.get('winner','?')} wins. "
        f"P2: {episode.get('p2','?')}. P3: {episode.get('p3','?')}.")

    # Fastest lap
    if episode.get("fastest_lap"):
        parts.append(
            f"Fastest lap: {episode['fastest_lap']} "
            f"({episode.get('fastest_lap_time','')}).")

    # Qualifying
    quali = episode.get("qualifying",{})
    if quali.get("pole"):
        parts.append(
            f"Pole: {quali['pole']} ({quali.get('pole_time','')}).")
    if quali.get("eliminated_q2"):
        parts.append(
            f"Q2 eliminations: {', '.join(quali['eliminated_q2'])}.")
    if quali.get("eliminated_q1"):
        parts.append(
            f"Q1 eliminations: {', '.join(quali['eliminated_q1'])}.")

    # Top 5 lap times
    if episode.get("top5_lap_times"):
        times_str = " | ".join(
            f"{t['driver']} {t['time']} ({t['delta']})"
            for t in episode["top5_lap_times"][:5])
        parts.append(f"Fastest laps: {times_str}.")

    # Tyre strategies
    pit = episode.get("pitstops",{})
    if pit.get("tyre_strategies"):
        parts.append(f"Tyre strategies: {pit['tyre_strategies']}.")

    # Weather
    if pit.get("weather"):
        parts.append(f"Conditions: {pit['weather']}.")

    # DNFs
    if episode.get("dnfs"):
        parts.append(f"DNFs: {', '.join(episode['dnfs'][:5])}.")

    # Safety cars from race control
    rc = episode.get("race_control",[])
    sc_msgs = [m for m in rc if "SAFETY CAR" in m.upper() or "VSC" in m.upper()]
    if sc_msgs:
        parts.append(f"Safety car: {' | '.join(sc_msgs[:3])}.")

    # Penalties
    pen_msgs = [m for m in rc if "PENALTY" in m.upper()]
    if pen_msgs:
        parts.append(f"Penalties: {' | '.join(pen_msgs[:3])}.")

    # Championship
    if episode.get("champ_after"):
        parts.append(f"Championship: {episode['champ_after']}.")

    # Sprint
    sprint = episode.get("sprint",{})
    if sprint.get("sprint_winner"):
        parts.append(
            f"Sprint: {sprint['sprint_winner']} won "
            f"({sprint.get('sprint_top3','')}).")

    # Sector bests
    sect = quali.get("sector_bests",{})
    if sect:
        sect_str = " | ".join(
            f"{s}: {d['driver']} {d['time']}"
            for s, d in sect.items())
        parts.append(f"Sector bests (quali): {sect_str}.")

    return " ".join(parts)


# ═════════════════════════════════════════════════════════════
#  PREDICTOR CSV
# ═════════════════════════════════════════════════════════════
def read_predictor_csv() -> list:
    if not PREDICTOR_CSV.exists():
        return []
    try:
        import csv as _csv
        with open(PREDICTOR_CSV, newline="", encoding="utf-8") as f:
            return list(_csv.DictReader(f))
    except Exception:
        return []


def _is_pre_qualifying_csv(rows: list) -> bool:
    """True when the CSV was generated without real qualifying data.

    Real qualifying produces differentiated positions (P1, P2, P3…).
    A pre-qualifying run fills every driver with the same sentinel value
    >= 20 because no session data exists yet.
    """
    if not rows:
        return False
    try:
        quali_vals = []
        for r in rows[:20]:
            v = r.get("quali_pos_next", "")
            if v not in ("", "nan", "NaN", None):
                quali_vals.append(float(v))
        if not quali_vals:
            return True  # no quali data at all
        return len(set(quali_vals)) == 1 and quali_vals[0] >= 20
    except (ValueError, TypeError):
        return False


def format_predictor_for_claude(rows: list) -> str:
    if not rows:
        return ""
    pre_quali = _is_pre_qualifying_csv(rows)

    if pre_quali:
        lines = [
            "=== PREDICTOR OUTPUT (f1_2026_predictor.py v7.0) ===",
            "⚠️ PRE-QUALIFYING PREVIEW — qualifying has NOT happened yet.",
            "Probabilities are Bayesian priors from season form + circuit history ONLY.",
            "Do NOT cite qualifying positions — they do not exist yet.\n",
        ]
    else:
        lines = ["=== PREDICTOR OUTPUT (f1_2026_predictor.py v7.0) ===",
                 "Real Monte Carlo simulation 10,000 runs.\n"]

    def gf(r, k, dec=1):
        try:    return f"{float(r[k]):.{dec}f}"
        except: return "-"
    def g(r, k):
        v = r.get(k, "")
        return v if v not in ("", "nan", "NaN", None) else "-"

    lines.append(f"{'#':<3} {'Driver':<20} {'Team':<16} {'Win%':>6} {'Pod%':>6} {'AvgPos':>7} {'MechRisk':>9} {'Pts':>6}")
    lines.append("-" * 78)
    for i, r in enumerate(rows[:10], 1):
        mech = f"{float(r.get('mechanical_risk',0))*100:.1f}%" if g(r,'mechanical_risk') != "-" else "-"
        pts  = gf(r, 'champ_pts', 0) if g(r, 'champ_pts') != "-" else "-"
        lines.append(f"{i:<3} {g(r,'FullName'):<20} {g(r,'TeamName'):<16} "
                     f"{gf(r,'win_mc_pct'):>6} {gf(r,'podium_mc_pct'):>6} "
                     f"{gf(r,'avg_mc_pos'):>7} {mech:>9} {pts:>6}")

    lines.append("\nKEY FEATURES - Top 5:")
    for r in rows[:5]:
        parts = [f"{g(r,'code')}:"]
        for k, label in [("quali_pos_next","Quali"),("fp_next_delta","FP%"),
                          ("recent_form","Form"),("circuit_score","Circuit"),
                          ("compound_score","Compound"),("mechanical_risk","MechRisk"),
                          ("dominant_failure","Failure")]:
            if k == "quali_pos_next" and pre_quali:
                continue  # sentinel value — omit to prevent confabulation
            v = g(r, k)
            if v != "-":
                try:    parts.append(f"{label}={float(v):.3f}")
                except: parts.append(f"{label}={v}")
        lines.append("  " + "  ".join(parts))

    return "\n".join(lines)


def get_predictor_context(expected_round: int | None = None) -> tuple:
    if not PREDICTOR_CSV.exists():
        return "", []
    try:
        mtime = str(PREDICTOR_CSV.stat().st_mtime)
        if mtime in _PREDICTOR_CACHE:
            block, rows = _PREDICTOR_CACHE[mtime]
        else:
            rows  = read_predictor_csv()
            block = format_predictor_for_claude(rows)
            _PREDICTOR_CACHE.clear()
            _PREDICTOR_CACHE[mtime] = (block, rows)
        if expected_round is not None and rows:
            try:
                csv_round = int(rows[0].get("round_num", ""))
                if csv_round != expected_round:
                    log.warning(
                        f"Stale predictor CSV: CSV is for R{csv_round}, "
                        f"expected R{expected_round} — returning empty"
                    )
                    return "", []
            except (ValueError, TypeError):
                pass  # round_num absent in pre-fix CSV; serve as-is
        return block, rows
    except Exception:
        return "", []


def predictor_winner_summary(rows: list) -> str:
    if not rows:
        return ""
    lines = []
    medals = ["🥇","🥈","🥉"]
    for i, r in enumerate(rows[:3], 1):
        name = r.get("FullName", r.get("code","?"))
        team = r.get("TeamName","")
        try:    win_str = f"{float(r.get('win_mc_pct','?')):.1f}%"
        except: win_str = "?"
        try:    pod_str = f"{float(r.get('podium_mc_pct','?')):.1f}%"
        except: pod_str = "?"
        lines.append(f"{medals[i-1]} *{name}* ({team}) — {win_str} win / {pod_str} podium")
    return "\n".join(lines)
