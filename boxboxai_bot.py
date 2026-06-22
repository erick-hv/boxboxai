#!/usr/bin/env python3
# DEPLOY-CHECK-MARKER: v2026-06-14-chromium-lazy-install
"""
BoxBoxAI — F1 Race Analyst Telegram Bot
=========================================
Developed by Erick Hernandez

Setup (one time):
  pip3 install python-telegram-bot anthropic requests

Create your bot:
  1. Open Telegram, search @BotFather
  2. Send /newbot
  3. Name: BoxBoxAI
  4. Username: BoxBoxAI_bot  (or whatever is available)
  5. Copy the token BotFather gives you

Run:
  export ANTHROPIC_API_KEY=sk-ant-...
  export TELEGRAM_BOT_TOKEN=your_token_here
  python3 boxboxai_bot.py
"""

import os, sys, json, re, time, logging, threading, asyncio
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

from news import (get_news_context, start_news_scheduler, refresh_news_cache,
                  load_news_cache, get_news_cache_time, get_news_cache)
from security import (check_rate_limit, check_injection, sanitize_message,
                      is_off_topic, get_off_topic_response, get_injection_response,
                      alert_owner, log_security_event, BOT_OWNER_ID,
                      _abuse_strikes, MAX_HISTORY_STORE)
from data_layer import (
    PREDICTOR_CSV, SPRINT_ROUNDS_2026, _PREDICTOR_CACHE,
    load_f1_memory, save_f1_memory,
    load_ingest_state, save_ingest_state,
    load_sessions, save_sessions,
    get_user_history, update_user_history, track_command,
    load_predictor_state, save_predictor_state,
    load_enrichment_state, save_enrichment_state,
    load_predictions, save_prediction, get_prediction_accuracy,
    safe_get, fetch_standings, fetch_current_race, fetch_next_race, fetch_last_race,
    fetch_race_result, fetch_qualifying_result, fetch_sprint_result,
    fetch_driver_career_stats,
    fetch_openf1, fetch_live_session, get_live_session_context,
    fetch_practice_results, get_practice_context,
    get_actual_grid_for_prediction, get_session_context,
    _safe_ff1_load, _ff1_session_summary, _openf1_session_summary,
    enrich_episode_with_telemetry, enrich_qualifying_with_telemetry,
    build_rich_story,
    read_predictor_csv, _is_pre_qualifying_csv, format_predictor_for_claude,
    get_predictor_context, predictor_winner_summary,
)

import context_builder
from context_builder import (
    ANTHROPIC_KEY, MODEL, RACE_KEYWORDS,
    ask_claude, get_client,
    _gather_context, _format_debug_context_report,
    _is_live_session_question, detect_fan_declaration,
    FIA_DRIVER_NAMES, FIA_DRIVER_CAR_NUMBERS,
    get_circuit_map_image,
)

# ── dependency check ──────────────────────────────────────────
missing = []
try:
    from telegram import Update, constants
    from telegram.ext import (Application, CommandHandler,
                               MessageHandler, filters, ContextTypes)
except ImportError:
    missing.append("python-telegram-bot")
try:
    import anthropic
except ImportError:
    missing.append("anthropic")
try:
    import requests
except ImportError:
    missing.append("requests")

if missing:
    print(f"\n  Missing: {', '.join(missing)}")
    print(f"  Fix:     pip3 install {' '.join(missing)}\n")
    sys.exit(1)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════
BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", "")
MEMORY_FILE   = Path(__file__).parent / "f1_memory_2026.json"
SESSIONS_FILE = Path(__file__).parent / "boxboxai_sessions.json"
JOLPICA       = "https://api.jolpi.ca/ergast/f1"
SEASON        = 2026

# Telegram message limit
TG_MAX_CHARS  = 4096

CIRCUIT_MAP_CACHE_FILE = Path(__file__).parent / "boxboxai_circuit_map_cache.json"
CIRCUIT_MAPS_DIR       = Path(__file__).parent / "boxboxai_circuit_maps"

# Serialise concurrent Playwright sessions (prewarm + user request can race).
import threading as _threading
_circuit_map_playwright_lock = _threading.Lock()


# ═════════════════════════════════════════════════════════════
#  TIMEZONE SYSTEM
# ═════════════════════════════════════════════════════════════

# Comprehensive country → UTC offset map
COUNTRY_TIMEZONES = {
    # Americas
    "mexico": (-6, "🇲🇽 Mexico"),
    "méxico": (-6, "🇲🇽 Mexico"),
    "mexico city": (-6, "🇲🇽 Mexico"),
    "guadalajara": (-6, "🇲🇽 Mexico"),
    "monterrey": (-6, "🇲🇽 Mexico"),
    "usa": (-5, "🇺🇸 USA (East)"),
    "united states": (-5, "🇺🇸 USA (East)"),
    "new york": (-5, "🇺🇸 USA (East)"),
    "miami": (-5, "🇺🇸 USA (East)"),
    "chicago": (-6, "🇺🇸 USA (Central)"),
    "dallas": (-6, "🇺🇸 USA (Central)"),
    "houston": (-6, "🇺🇸 USA (Central)"),
    "denver": (-7, "🇺🇸 USA (Mountain)"),
    "los angeles": (-8, "🇺🇸 USA (Pacific)"),
    "california": (-8, "🇺🇸 USA (Pacific)"),
    "seattle": (-8, "🇺🇸 USA (Pacific)"),
    "canada": (-5, "🇨🇦 Canada (East)"),
    "toronto": (-5, "🇨🇦 Canada (East)"),
    "montreal": (-5, "🇨🇦 Canada (East)"),
    "vancouver": (-8, "🇨🇦 Canada (West)"),
    "brazil": (-3, "🇧🇷 Brazil"),
    "brasil": (-3, "🇧🇷 Brazil"),
    "são paulo": (-3, "🇧🇷 Brazil"),
    "sao paulo": (-3, "🇧🇷 Brazil"),
    "rio": (-3, "🇧🇷 Brazil"),
    "argentina": (-3, "🇦🇷 Argentina"),
    "buenos aires": (-3, "🇦🇷 Argentina"),
    "colombia": (-5, "🇨🇴 Colombia"),
    "bogota": (-5, "🇨🇴 Colombia"),
    "chile": (-4, "🇨🇱 Chile"),
    "peru": (-5, "🇵🇪 Peru"),
    "venezuela": (-4, "🇻🇪 Venezuela"),
    "ecuador": (-5, "🇪🇨 Ecuador"),
    # Europe
    "uk": (1, "🇬🇧 UK"),
    "united kingdom": (1, "🇬🇧 UK"),
    "england": (1, "🇬🇧 UK"),
    "london": (1, "🇬🇧 UK"),
    "ireland": (1, "🇮🇪 Ireland"),
    "spain": (2, "🇪🇸 Spain"),
    "españa": (2, "🇪🇸 Spain"),
    "madrid": (2, "🇪🇸 Spain"),
    "barcelona": (2, "🇪🇸 Spain"),
    "france": (2, "🇫🇷 France"),
    "paris": (2, "🇫🇷 France"),
    "germany": (2, "🇩🇪 Germany"),
    "deutschland": (2, "🇩🇪 Germany"),
    "italy": (2, "🇮🇹 Italy"),
    "italia": (2, "🇮🇹 Italy"),
    "netherlands": (2, "🇳🇱 Netherlands"),
    "holland": (2, "🇳🇱 Netherlands"),
    "belgium": (2, "🇧🇪 Belgium"),
    "switzerland": (2, "🇨🇭 Switzerland"),
    "austria": (2, "🇦🇹 Austria"),
    "portugal": (1, "🇵🇹 Portugal"),
    "monaco": (2, "🇲🇨 Monaco"),
    "sweden": (2, "🇸🇪 Sweden"),
    "norway": (2, "🇳🇴 Norway"),
    "denmark": (2, "🇩🇰 Denmark"),
    "finland": (3, "🇫🇮 Finland"),
    "poland": (2, "🇵🇱 Poland"),
    "czech": (2, "🇨🇿 Czech Republic"),
    "hungary": (2, "🇭🇺 Hungary"),
    "greece": (3, "🇬🇷 Greece"),
    "turkey": (3, "🇹🇷 Turkey"),
    "russia": (3, "🇷🇺 Russia (Moscow)"),
    # Middle East & Asia
    "uae": (4, "🇦🇪 UAE"),
    "dubai": (4, "🇦🇪 UAE"),
    "abu dhabi": (4, "🇦🇪 UAE"),
    "saudi": (3, "🇸🇦 Saudi Arabia"),
    "qatar": (3, "🇶🇦 Qatar"),
    "bahrain": (3, "🇧🇭 Bahrain"),
    "india": (5, "🇮🇳 India"),
    "pakistan": (5, "🇵🇰 Pakistan"),
    "china": (8, "🇨🇳 China"),
    "japan": (9, "🇯🇵 Japan"),
    "south korea": (9, "🇰🇷 South Korea"),
    "korea": (9, "🇰🇷 South Korea"),
    "singapore": (8, "🇸🇬 Singapore"),
    "thailand": (7, "🇹🇭 Thailand"),
    "indonesia": (7, "🇮🇩 Indonesia"),
    "malaysia": (8, "🇲🇾 Malaysia"),
    "philippines": (8, "🇵🇭 Philippines"),
    "vietnam": (7, "🇻🇳 Vietnam"),
    # Oceania
    "australia": (10, "🇦🇺 Australia (East)"),
    "sydney": (10, "🇦🇺 Australia (East)"),
    "melbourne": (10, "🇦🇺 Australia (East)"),
    "perth": (8, "🇦🇺 Australia (West)"),
    "new zealand": (12, "🇳🇿 New Zealand"),
    # Africa
    "south africa": (2, "🇿🇦 South Africa"),
    "nigeria": (1, "🇳🇬 Nigeria"),
    "egypt": (2, "🇪🇬 Egypt"),
    "morocco": (1, "🇲🇦 Morocco"),
    "kenya": (3, "🇰🇪 Kenya"),
}

# Pending timezone requests — user_id: True
_tz_pending: dict = {}

# Full 2026 session schedule — all times in UTC
# Format: (round, race_name, circuit_tz_offset, sessions)
# sessions: list of (session_name, weekday, hour, minute)
# weekday: 4=Friday, 5=Saturday, 6=Sunday (race day)
SESSION_SCHEDULE_2026 = [
    (1,  "Australian GP",    11, [
        ("FP1",        4, 1,  30),
        ("FP2",        4, 5,   0),
        ("FP3",        5, 2,   0),
        ("Qualifying", 5, 5,   0),
        ("Race",       6, 4,   0),
    ]),
    (2,  "Chinese GP",       8, [
        ("FP1",           4,  3,  30),
        ("Sprint Quali",  4,  7,  30),
        ("Sprint Race",   5,  3,   0),
        ("Qualifying",    5,  7,   0),
        ("Race",          6,  7,   0),
    ]),
    (3,  "Japanese GP",      9, [
        ("FP1",        4,  2,  30),
        ("FP2",        4,  6,   0),
        ("FP3",        5,  2,  30),
        ("Qualifying", 5,  6,   0),
        ("Race",       6,  5,   0),
    ]),
    (4,  "Miami GP",        -5, [
        ("FP1",           4, 18,   0),
        ("Sprint Quali",  4, 22,   0),
        ("Sprint Race",   5, 16,   0),
        ("Qualifying",    5, 20,   0),
        ("Race",          6, 20,   0),
    ]),
    (5,  "Canadian GP",     -5, [
        ("FP1",        4, 17,  30),
        ("FP2",        4, 21,   0),
        ("FP3",        5, 16,  30),
        ("Qualifying", 5, 20,   0),
        ("Race",       6, 18,   0),
    ]),
    (6,  "Monaco GP",        2, [
        ("FP1",        4, 11,  30),
        ("FP2",        4, 15,   0),
        ("FP3",        5, 11,  30),
        ("Qualifying", 5, 14,   0),
        ("Race",       6, 13,   0),
    ]),
    (7,  "Spanish GP",       2, [
        ("FP1",        4, 11,  30),
        ("FP2",        4, 15,   0),
        ("FP3",        5, 10,  30),
        ("Qualifying", 5, 14,   0),
        ("Race",       6, 13,   0),
    ]),
    (8,  "Austrian GP",      2, [
        ("FP1",           4, 10,  30),
        ("Sprint Quali",  4, 14,  30),
        ("Sprint Race",   5, 10,   0),
        ("Qualifying",    5, 14,   0),
        ("Race",          6, 13,   0),
    ]),
    (9,  "British GP",       1, [
        ("FP1",        4, 11,  30),
        ("FP2",        4, 15,   0),
        ("FP3",        5, 10,  30),
        ("Qualifying", 5, 14,   0),
        ("Race",       6, 14,   0),
    ]),
    (10, "Belgian GP",       2, [
        ("FP1",        4, 11,  30),
        ("FP2",        4, 15,   0),
        ("FP3",        5, 10,  30),
        ("Qualifying", 5, 14,   0),
        ("Race",       6, 13,   0),
    ]),
    (11, "Hungarian GP",     2, [
        ("FP1",        4, 11,  30),
        ("FP2",        4, 15,   0),
        ("FP3",        5, 10,  30),
        ("Qualifying", 5, 14,   0),
        ("Race",       6, 13,   0),
    ]),
    (12, "Dutch GP",         2, [
        ("FP1",        4, 10,  30),
        ("FP2",        4, 14,   0),
        ("FP3",        5, 10,  30),
        ("Qualifying", 5, 14,   0),
        ("Race",       6, 13,   0),
    ]),
    (13, "Italian GP",       2, [
        ("FP1",        4, 11,  30),
        ("FP2",        4, 15,   0),
        ("FP3",        5, 10,  30),
        ("Qualifying", 5, 14,   0),
        ("Race",       6, 13,   0),
    ]),
    (14, "Singapore GP",     8, [
        ("FP1",        4, 9,  30),
        ("FP2",        4, 13,   0),
        ("FP3",        5, 9,  30),
        ("Qualifying", 5, 13,   0),
        ("Race",       6, 12,   0),
    ]),
    (15, "Azerbaijan GP",    4, [
        ("FP1",        4, 9,  30),
        ("FP2",        4, 13,   0),
        ("FP3",        5, 9,  30),
        ("Qualifying", 5, 13,   0),
        ("Race",       6, 11,   0),
    ]),
    (16, "US GP",           -6, [
        ("FP1",           4, 18,  30),
        ("Sprint Quali",  4, 22,  30),
        ("Sprint Race",   5, 17,   0),
        ("Qualifying",    5, 22,   0),
        ("Race",          6, 20,   0),
    ]),
    (17, "Mexico City GP",  -6, [
        ("FP1",        4, 19,  30),
        ("FP2",        4, 23,   0),
        ("FP3",        5, 18,  30),
        ("Qualifying", 5, 22,   0),
        ("Race",       6, 20,   0),
    ]),
    (18, "São Paulo GP",    -3, [
        ("FP1",           4, 14,  30),
        ("Sprint Quali",  4, 18,  30),
        ("Sprint Race",   5, 14,   0),
        ("Qualifying",    5, 18,   0),
        ("Race",          6, 17,   0),
    ]),
    (19, "Las Vegas GP",    -8, [
        ("FP1",        5,  4,  30),
        ("FP2",        5,  8,   0),
        ("FP3",        6,  4,  30),
        ("Qualifying", 6,  8,   0),
        ("Race",       6, 22,   0),
    ]),
    (20, "Qatar GP",         3, [
        ("FP1",           4, 13,  30),
        ("Sprint Quali",  4, 17,  30),
        ("Sprint Race",   5, 13,   0),
        ("Qualifying",    5, 17,   0),
        ("Race",          6, 17,   0),
    ]),
    (21, "Abu Dhabi GP",     4, [
        ("FP1",        4, 9,  30),
        ("FP2",        4, 13,   0),
        ("FP3",        5, 9,  30),
        ("Qualifying", 5, 13,   0),
        ("Race",       6, 13,   0),
    ]),
]


def get_user_tz_offset(user_data: dict) -> int:
    """Returns user's UTC offset in hours. Defaults to -6 (Mexico City)."""
    return user_data.get("tz_offset", -6)


def lookup_country_tz(text: str) -> tuple | None:
    """
    Looks up a country/city in COUNTRY_TIMEZONES.
    Returns (offset, label) or None if not found.
    """
    t = text.lower().strip()
    # Exact match first
    if t in COUNTRY_TIMEZONES:
        return COUNTRY_TIMEZONES[t]
    # Partial match
    for key, val in COUNTRY_TIMEZONES.items():
        if key in t or t in key:
            return val
    return None


def format_session_times(session_name: str,
                          utc_hour: int, utc_min: int,
                          circuit_tz: int,
                          user_tz: int,
                          user_tz_label: str) -> str:
    """Formats a session time in UTC, circuit local, and user local time."""
    def to_local(h, m, offset):
        total = h * 60 + m + offset * 60
        total = total % (24 * 60)
        return total // 60, total % 60

    circuit_h, circuit_m = to_local(utc_hour, utc_min, circuit_tz)
    user_h,    user_m    = to_local(utc_hour, utc_min, user_tz)

    return (
        f"*{session_name}*\n"
        f"  🏟 Circuit: {circuit_h:02d}:{circuit_m:02d} local\n"
        f"  🌍 Your time: {user_h:02d}:{user_m:02d} {user_tz_label}\n"
        f"  🕐 UTC: {utc_hour:02d}:{utc_min:02d}"
    )


def get_sessions_for_current_round() -> tuple | None:
    """Returns (round_data, race_date) for the current race weekend."""
    today = datetime.now().date()
    for entry in SESSION_SCHEDULE_2026:
        rnd = entry[0]
        # Find matching round in calendar
        for cal_rnd, name, date_str in [
            (r, n, d) for r, n, d in [
                (e[0], e[1], d) for e in SESSION_SCHEDULE_2026
                for d in [next((x[2] for x in [
                    (1,"Australian GP","2026-03-15"),
                    (2,"Chinese GP","2026-03-22"),
                    (3,"Japanese GP","2026-04-06"),
                    (4,"Miami GP","2026-05-04"),
                    (5,"Canadian GP","2026-05-24"),
                    (6,"Monaco GP","2026-06-07"),
                    (7,"Spanish GP","2026-06-14"),
                    (8,"Austrian GP","2026-06-28"),
                    (9,"British GP","2026-07-05"),
                    (10,"Belgian GP","2026-07-19"),
                    (11,"Hungarian GP","2026-07-26"),
                    (12,"Dutch GP","2026-08-23"),
                    (13,"Italian GP","2026-09-06"),
                    (14,"Singapore GP","2026-09-20"),
                    (15,"Azerbaijan GP","2026-09-27"),
                    (16,"US GP","2026-10-18"),
                    (17,"Mexico City GP","2026-10-25"),
                    (18,"São Paulo GP","2026-11-08"),
                    (19,"Las Vegas GP","2026-11-21"),
                    (20,"Qatar GP","2026-11-29"),
                    (21,"Abu Dhabi GP","2026-12-06"),
                ] if x[0] == e[0]), None)]
            ]
        ]:
            pass
    return None


# Simpler version — direct lookup
RACE_DATES_2026 = {
    1: "2026-03-15", 2: "2026-03-22", 3: "2026-04-06",
    4: "2026-05-04", 5: "2026-05-24", 6: "2026-06-07",
    7: "2026-06-14", 8: "2026-06-28", 9: "2026-07-05",
    10:"2026-07-19",11: "2026-07-26",12: "2026-08-23",
    13:"2026-09-06",14: "2026-09-20",15: "2026-09-27",
    16:"2026-10-18",17: "2026-10-25",18: "2026-11-08",
    19:"2026-11-21",20: "2026-11-29",21: "2026-12-06",
}


def get_upcoming_sessions(user_tz: int, user_tz_label: str,
                           hours_ahead: int = 1) -> list:
    """
    Returns sessions starting within the next `hours_ahead` hours.
    Used to trigger 15-min-before notifications.
    """
    now_utc   = datetime.utcnow()
    upcoming  = []

    for entry in SESSION_SCHEDULE_2026:
        rnd, race_name, circuit_tz, sessions = entry
        race_date_str = RACE_DATES_2026.get(rnd)
        if not race_date_str:
            continue

        race_date = datetime.strptime(race_date_str, "%Y-%m-%d").date()

        for session_name, weekday_offset, utc_h, utc_m in sessions:
            # Calculate actual session date
            # weekday_offset: 4=Friday(-2), 5=Saturday(-1), 6=Sunday(0)
            days_before = 6 - weekday_offset  # days before Sunday race day
            session_date = race_date - timedelta(days=days_before)

            session_dt = datetime(
                session_date.year, session_date.month, session_date.day,
                utc_h, utc_m, 0
            )

            # Check if session starts within the next `hours_ahead` hours
            delta_minutes = (session_dt - now_utc).total_seconds() / 60
            if 10 <= delta_minutes <= hours_ahead * 60:
                upcoming.append({
                    "round":       rnd,
                    "race_name":   race_name,
                    "session":     session_name,
                    "session_dt":  session_dt,
                    "circuit_tz":  circuit_tz,
                    "minutes_away": int(delta_minutes),
                    "time_str":    format_session_times(
                        session_name, utc_h, utc_m,
                        circuit_tz, user_tz, user_tz_label
                    ),
                })

    return upcoming



NOTIFICATIONS_FILE = Path(__file__).parent / "boxboxai_notifications.json"

def load_notification_state() -> dict:
    if NOTIFICATIONS_FILE.exists():
        try:
            return json.loads(NOTIFICATIONS_FILE.read_text())
        except Exception:
            pass
    return {}

def save_notification_state(state: dict):
    try:
        NOTIFICATIONS_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass

def get_active_user_ids(sessions: dict) -> list:
    """Returns user IDs active in the last 30 days."""
    cutoff = datetime.now().timestamp() - (30 * 24 * 3600)
    active = []
    for uid, data in sessions.items():
        last = data.get("last_seen") or data.get("first_seen", "")
        try:
            if datetime.fromisoformat(last).timestamp() > cutoff:
                active.append(uid)
        except Exception:
            active.append(uid)
    return active


SESSION_EMOJIS = {
    "FP1": "🔧", "FP2": "🔧", "FP3": "🔧",
    "Qualifying": "⏱", "Sprint Quali": "⏱", "Sprint Qualifying": "⏱",
    "Sprint Race": "🏃", "Race": "🏁",
}

SESSION_HYPE = {
    "FP1":            "First cars on track this weekend! Watch who finds pace early 👀",
    "FP2":            "FP2 is the most important practice — teams run race simulations 🛞",
    "FP3":            "Final practice before qualifying — last chance to dial the car in 🔩",
    "Qualifying":     "THIS IS IT. Qualifying defines the race. Who takes pole? 🏆",
    "Sprint Quali":   "Sprint shootout — short, fast, brutal. Every tenth counts ⚡",
    "Sprint Race":    "Sprint race! Full speed, no team orders, nothing to lose 🔥",
    "Race":           "RACE DAY. Lights out and away we go! 🚦🏎",
}


async def send_session_notifications(app, sessions: dict):
    """
    Checks for sessions starting in the next 15 minutes
    and sends personalized notifications to all active users with their local time.
    Runs every 5 minutes.
    """
    state        = load_notification_state()
    active_users = get_active_user_ids(sessions)
    if not active_users:
        return

    now_utc = datetime.utcnow()

    for entry in SESSION_SCHEDULE_2026:
        rnd, race_name, circuit_tz, session_list = entry
        race_date_str = RACE_DATES_2026.get(rnd)
        if not race_date_str:
            continue

        race_date = datetime.strptime(race_date_str, "%Y-%m-%d").date()

        for session_name, weekday_offset, utc_h, utc_m in session_list:
            days_before  = 6 - weekday_offset
            session_date = race_date - timedelta(days=days_before)
            session_dt   = datetime(
                session_date.year, session_date.month, session_date.day,
                utc_h, utc_m, 0
            )

            # Only notify if session is 10-20 minutes away
            delta_mins = (session_dt - now_utc).total_seconds() / 60
            if not (10 <= delta_mins <= 20):
                continue

            # Check we haven't already sent this notification
            notif_key = f"r{rnd}_{session_name.replace(' ','_')}"
            if state.get(notif_key):
                continue

            # Send to each active user with their local time
            sent = 0
            for uid in active_users:
                user_data      = sessions.get(uid, {})
                user_tz_offset = get_user_tz_offset(user_data)
                user_tz_label  = user_data.get("tz_label", "UTC-6")

                # Calculate times
                def to_local(h, m, off):
                    total = (h * 60 + m + off * 60) % (24 * 60)
                    return total // 60, total % 60

                circ_h, circ_m = to_local(utc_h, utc_m, circuit_tz)
                user_h, user_m = to_local(utc_h, utc_m, user_tz_offset)

                emoji = SESSION_EMOJIS.get(session_name, "🏎")
                hype  = SESSION_HYPE.get(session_name, "Session starting soon!")

                msg = (
                    f"{emoji} *{race_name} — {session_name} in ~15 min!*\n\n"
                    f"{hype}\n\n"
                    f"🏟 Circuit time: *{circ_h:02d}:{circ_m:02d}*\n"
                    f"🌍 Your time: *{user_h:02d}:{user_m:02d}* ({user_tz_label})\n"
                    f"🕐 UTC: *{utc_h:02d}:{utc_m:02d}*\n\n"
                    f"_/predict_ for race preview • _/winner_ for my pick"
                )

                try:
                    await app.bot.send_message(
                        chat_id=uid, text=msg,
                        parse_mode="Markdown")
                    sent += 1
                    await asyncio.sleep(0.05)
                except Exception:
                    pass

            state[notif_key] = now_utc.isoformat()
            log.info(f"Session notification {notif_key} sent to {sent} users")

    save_notification_state(state)



# ═════════════════════════════════════════════════════════════
#  FEATURE: HISTORICAL COMPARISONS
# ═════════════════════════════════════════════════════════════



# ═════════════════════════════════════════════════════════════
#  FEATURE: USER STATS AND WEEKLY DIGEST
# ═════════════════════════════════════════════════════════════

DIGEST_STATE_FILE = Path(__file__).parent / "boxboxai_digest_state.json"

def load_digest_state() -> dict:
    if DIGEST_STATE_FILE.exists():
        try:
            return json.loads(DIGEST_STATE_FILE.read_text())
        except Exception:
            pass
    return {}

def save_digest_state(state: dict):
    DIGEST_STATE_FILE.write_text(json.dumps(state, indent=2))

def build_user_stats_text(user_data: dict, user_name: str) -> str:
    """Builds a personalised stats summary for one user."""
    stats      = user_data.get("stats", {})
    total      = stats.get("total_messages", 0)
    first_seen = user_data.get("first_seen", "")[:10]
    topics     = stats.get("favorite_topics", {})
    commands   = stats.get("commands_used", {})

    if not total:
        return f"No stats yet for {user_name} — start chatting! 🏎"

    # Top topic
    top_topic = max(topics, key=topics.get) if topics else "general"
    topic_emojis = {
        "championship": "🏆", "predictions": "🎯", "weather": "🌧",
        "strategy": "🛞", "drivers": "🏎", "races": "🏁",
        "fun": "🔥", "general": "💬"
    }
    top_emoji = topic_emojis.get(top_topic, "💬")

    # Top command
    top_cmd = max(commands, key=commands.get) if commands else None

    lines = [
        f"📊 *Your BoxBoxAI Stats*",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"👤 {user_name}",
        f"📅 Member since: {first_seen}",
        f"💬 Total messages: *{total}*",
        f"{top_emoji} Favourite topic: *{top_topic.title()}*",
    ]
    if top_cmd:
        lines.append(f"⌨️ Most used command: */{top_cmd}*")

    return "\n".join(lines)


async def send_weekly_digest(app, sessions: dict, mem: dict):
    """
    Sends a weekly F1 race debrief every Monday.
    Strategy analysis, key moments, championship implications.
    """
    state = load_digest_state()
    today = datetime.now().strftime("%Y-%W")

    if state.get("last_digest_week") == today:
        return
    if datetime.now().weekday() != 0:
        return

    active_users = get_active_user_ids(sessions)
    if not active_users:
        return

    episodes = mem.get("episodic", [])
    if not episodes:
        return

    last_race = sorted(episodes, key=lambda x: x.get("round", 0))[-1]
    race_name = last_race.get("race_name", last_race.get("track", ""))
    winner    = last_race.get("winner", "?")
    p2        = last_race.get("p2", "?")
    p3        = last_race.get("p3", "?")
    story     = last_race.get("story", "")
    champ     = last_race.get("champ_after", "")
    dnfs      = last_race.get("dnfs", [])
    sc        = last_race.get("sc_count", 0)
    pit       = last_race.get("pitstops", {})

    # Generate AI debrief using Ruth's voice format
    try:
        prompt = (
            f"{RUTH_DEBRIEF_PROMPT}\n\n"
            f"RACE DATA:\n"
            f"Race: {race_name}\n"
            f"Result: {winner} won | P2: {p2} | P3: {p3}\n"
            f"Story: {story}\n"
            f"DNFs: {', '.join(dnfs) if dnfs else 'none'}\n"
            f"Safety cars: {sc}\n"
            f"Strategy: {pit.get('tyre_strategies','')}\n"
            f"Championship: {champ}\n\n"
            f"Now write the debrief in the exact format above."
        )
        resp = get_client().messages.create(
            model=MODEL, max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        debrief = resp.content[0].text if resp.content else story
    except Exception:
        debrief = story

    next_race = fetch_next_race()
    next_str  = ""
    if next_race:
        next_name = next_race.get("raceName", "")
        next_date = next_race.get("date", "")
        next_str  = f"\n\n🔜 *Next up: {next_name}* — {next_date}\n_/predict_ for my full preview"

    msg = (
        f"📋 *Monday Debrief — {race_name}*\n"
        f"_BoxBoxAI Race Analysis_\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🥇 *{winner}*  🥈 {p2}  🥉 {p3}\n\n"
        f"{debrief}"
        f"{next_str}\n\n"
        f"_Ask me anything about the race_ 💬"
    )

    sent = 0
    for uid in active_users:
        user_prefs = sessions.get(uid, {}).get("notification_prefs", {})
        if not user_prefs.get("weekly_debrief", True):
            continue
        try:
            await app.bot.send_message(
                chat_id=uid, text=msg, parse_mode="Markdown")
            sent += 1
            await asyncio.sleep(0.05)
        except Exception:
            pass

    state["last_digest_week"] = today
    save_digest_state(state)
    log.info(f"Weekly debrief sent to {sent} users")



async def notification_loop(app, sessions_ref: list, mem_ref: list):
    """
    Async loop — runs every 5 minutes:
    - Session start notifications (15 min before)
    - Session end debriefs (after session ends)
    - Live session state polling (during sessions)
    - Weekly digest (Monday)
    """
    import asyncio as _asyncio
    check_count = 0

    while True:
        try:
            # Every 5 min — session notifications
            await send_session_notifications(app, sessions_ref[0])

            # Every 30 min — session debriefs + auto-verify ingest
            if check_count % 6 == 0:
                # DISABLED: automatic prose debrief via send_session_debrief.
                # The "RACE OVER!" instant notification (_notify_race_result,
                # fired from auto-ingest) already gives users the real
                # podium + championship standings and invites them to
                # "ask me anything about the race". A manual debrief
                # request goes through ask_claude with full memory/
                # grounding context and produces a noticeably better,
                # more accurate response than this automated path.
                # Re-enable if a good reason comes up:
                # await check_and_send_session_debriefs(
                #     app, sessions_ref[0], mem_ref[0])

                # Auto-verify: check if predictor ran after qualifying
                await _verify_predictor_ran(app, mem_ref[0])

            # Every hour — weekly digest
            check_count += 1
            if check_count >= 12:
                check_count = 0
                await send_weekly_digest(
                    app, sessions_ref[0], mem_ref[0])

        except Exception as e:
            log.warning(f"Notification loop error: {e}")
        await _asyncio.sleep(300)


async def _verify_predictor_ran(app, mem: dict):
    """
    Checks if the predictor ran after qualifying for the current race.
    If qualifying happened but CSV is old/missing, alerts owner.
    """
    state = load_predictor_state()
    today = datetime.now().date()

    for rnd, name, date_str in [
        (7,"Spanish GP","2026-06-14"),
        (8,"Austrian GP","2026-06-28"),
        (9,"British GP","2026-07-05"),
        (10,"Belgian GP","2026-07-19"),
        (11,"Hungarian GP","2026-07-26"),
        (12,"Dutch GP","2026-08-23"),
        (13,"Italian GP","2026-09-06"),
        (14,"Singapore GP","2026-09-20"),
        (15,"Azerbaijan GP","2026-09-27"),
        (16,"US GP","2026-10-18"),
        (17,"Mexico City GP","2026-10-25"),
        (18,"São Paulo GP","2026-11-08"),
        (19,"Las Vegas GP","2026-11-21"),
        (20,"Qatar GP","2026-11-29"),
        (21,"Abu Dhabi GP","2026-12-06"),
    ]:
        race_date  = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_until = (race_date - today).days

        # Qualifying is Saturday (race_date - 1)
        quali_date = race_date - timedelta(days=1)
        days_since_quali = (today - quali_date).days

        # If qualifying was 0-2 days ago and predictor hasn't run
        if 0 <= days_since_quali <= 2:
            state_key = f"predictor_r{rnd}"
            if not state.get(state_key):
                # Check if CSV exists and is recent
                csv_age_hours = None
                if PREDICTOR_CSV.exists():
                    age = (datetime.now().timestamp() -
                           PREDICTOR_CSV.stat().st_mtime) / 3600
                    csv_age_hours = round(age, 1)

                if csv_age_hours is None or csv_age_hours > 24:
                    log.warning(
                        f"Predictor may not have run for R{rnd} {name}")
                    # Alert owner once
                    alert_key = f"predictor_alert_r{rnd}"
                    if not state.get(alert_key) and app:
                        await alert_owner(app,
                            f"⚠️ *Predictor check: R{rnd} {name}*\n\n"
                            f"Qualifying was {days_since_quali} day(s) ago.\n"
                            f"CSV age: {csv_age_hours or 'missing'}h\n\n"
                            f"Auto-predictor should have fired. "
                            f"Check Railway logs if /winner is using old data.")
                        state[alert_key] = datetime.now().isoformat()
                        save_predictor_state(state)
            break


# ═════════════════════════════════════════════════════════════
#  UNIVERSAL LIVE SEARCH ENGINE
# ═════════════════════════════════════════════════════════════

# Trusted F1 sources for search result filtering
F1_TRUSTED_DOMAINS = [
    "the-race.com", "autosport.com", "racefans.net",
    "motorsport.com", "f1.com", "bbc.co.uk/sport",
    "espn.com/f1", "skysports.com/f1", "planetf1.com",
    "crash.net", "gpfans.com", "formula1.com",
]

# ═════════════════════════════════════════════════════════════
#  GOOGLE SEARCH + ARTICLE FETCHER
# ═════════════════════════════════════════════════════════════

def google_search_f1(query: str, num_results: int = 5) -> list:
    """Searches Google for F1 content. Returns list of {title, url, snippet}."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        }
        params = {"q": query, "num": num_results * 2, "hl": "en", "gl": "us"}
        r = requests.get("https://www.google.com/search",
                         params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return []

        results = []
        blocks  = re.findall(
            r'<div[^>]*class="[^"]*(?:g|MjjYud)[^"]*"[^>]*>(.*?)</div>\s*</div>',
            r.text, re.DOTALL)
        for block in blocks[:num_results * 3]:
            url_match = re.search(r'href="(https?://[^"]+)"', block)
            if not url_match:
                continue
            url = url_match.group(1)
            if "google.com" in url or "youtube.com" in url:
                continue
            title_match = re.search(r'<h3[^>]*>([^<]+)</h3>', block)
            title = title_match.group(1).strip() if title_match else ""
            snippet_match = re.search(
                r'<span[^>]*class="[^"]*(?:st|aCOpRe|hgKElc)[^"]*"[^>]*>'
                r'([^<]+(?:<[^>]+>[^<]+</[^>]+>)*[^<]*)</span>', block)
            snippet = ""
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", "",
                                 snippet_match.group(1)).strip()
            if title or snippet:
                results.append({
                    "title":   title,
                    "url":     url,
                    "snippet": snippet[:300],
                })
            if len(results) >= num_results:
                break
        return results
    except Exception as e:
        log.warning(f"Google search failed: {e}")
        return []


def fetch_article_content(url: str, max_chars: int = 1500) -> str:
    """Fetches and extracts clean text from an article URL."""
    try:
        r = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 BoxBoxAI/1.0"})
        if r.status_code != 200:
            return ""
        text = r.text
        for tag in ["script","style","nav","header","footer","aside"]:
            text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "",
                          text, flags=re.DOTALL)
        for selector in [
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class="[^"]*(?:article|content|story)[^"]*"[^>]*>(.*?)</div>',
            r'<main[^>]*>(.*?)</main>',
        ]:
            m = re.search(selector, text, re.DOTALL)
            if m:
                text = m.group(1)
                break
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


# ═════════════════════════════════════════════════════════════
#  FIA STEWARDS DOCUMENTS
#  Official incident decisions, penalties, technical findings
#  URL: fia.com/system/files/decision-document/
#  These are the ground truth for WHY incidents happened
# ═════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════
#  FIA STEWARDS INTELLIGENCE
#  FIA blocks direct PDF access (403). Instead we fetch from
#  motorsport.com, planetf1.com, total-motorsport.com, espn.com
#  which publish the exact FIA stewards verdict text within
#  hours of every race. These are the most reliable sources
#  for penalty reasons, incident findings, and decisions.
# ═════════════════════════════════════════════════════════════

_FIA_DOC_CACHE: dict = {}  # {cache_key: text} — 2hr TTL
_FIA_CACHE_TIMES: dict = {}

# Sources ranked by reliability for stewards decisions
FIA_STEWARDS_SOURCES = [
    "motorsport.com",
    "planetf1.com",
    "total-motorsport.com",
    "the-race.com",
    "autosport.com",
    "espn.com/f1",
    "formula1.com",
    "racefans.net",
]

# Race name keywords for search context
FIA_RACE_KEYWORDS = {
    "monaco":     "Monaco Grand Prix 2026",
    "australian": "Australian Grand Prix 2026",
    "chinese":    "Chinese Grand Prix 2026",
    "japanese":   "Japanese Grand Prix 2026",
    "miami":      "Miami Grand Prix 2026",
    "canadian":   "Canadian Grand Prix 2026",
    "spanish":    "Spanish Grand Prix 2026 Barcelona",
    "barcelona":  "Spanish Grand Prix 2026 Barcelona",
    "austrian":   "Austrian Grand Prix 2026",
    "british":    "British Grand Prix 2026 Silverstone",
    "belgian":    "Belgian Grand Prix 2026",
    "hungarian":  "Hungarian Grand Prix 2026",
    "dutch":      "Dutch Grand Prix 2026",
    "italian":    "Italian Grand Prix 2026 Monza",
    "singapore":  "Singapore Grand Prix 2026",
    "azerbaijan": "Azerbaijan Grand Prix 2026 Baku",
    "baku":       "Azerbaijan Grand Prix 2026 Baku",
    "usa":        "US Grand Prix 2026 Austin",
    "austin":     "US Grand Prix 2026 Austin",
    "mexico":     "Mexico City Grand Prix 2026",
    "brazil":     "Brazilian Grand Prix 2026 Sao Paulo",
    "paulo":      "Brazilian Grand Prix 2026 Sao Paulo",
    "vegas":      "Las Vegas Grand Prix 2026",
    "qatar":      "Qatar Grand Prix 2026",
    "abu":        "Abu Dhabi Grand Prix 2026",
}

# Driver name map for search queries


def _extract_article_text(url: str, max_chars: int = 800) -> str:
    """Fetches article and extracts clean body text."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, timeout=10, headers=headers)
        if r.status_code != 200:
            return ""
        raw_html = r.text
        stewards_keywords = [
            "steward", "penalty", "infringement", "decision", "article",
            "regulation", "grid slot", "restart", "out of position",
            "drive-through", "time penalty", "reprimand", "disqualified",
            "collision", "track limits", "unsafe", "found that",
            "video evidence", "the standard penalty"
        ]

        def _filter_sentences(body: str) -> str:
            body = re.sub(r"\s+", " ", body).strip()
            sentences = re.split(r'(?<=[.!?])\s+', body)
            relevant = [s.strip() for s in sentences
                        if any(kw in s.lower() for kw in stewards_keywords)]
            return (" ".join(relevant) if relevant else body)[:max_chars]

        # 1. JSON-LD extraction (fastest and cleanest — most news sites embed
        #    NewsArticle schema with a pre-stripped "articleBody" field)
        for ld_match in re.finditer(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            raw_html, re.DOTALL | re.IGNORECASE
        ):
            try:
                data = json.loads(ld_match.group(1))
            except Exception:
                continue
            # handle both single objects and arrays
            items = data if isinstance(data, list) else [data]
            for item in items:
                body = item.get("articleBody", "") if isinstance(item, dict) else ""
                if isinstance(body, str) and len(body) > 80:
                    return _filter_sentences(body)

        # 2. Regex HTML scraping fallback
        text = raw_html
        # Strip scripts/styles
        for tag in ["script","style","nav","header","footer","aside","figure"]:
            text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text,
                          flags=re.DOTALL|re.IGNORECASE)
        # Try article body selectors — pick largest <article> match to avoid
        # grabbing a small teaser widget when multiple <article> tags exist
        article_matches = re.findall(
            r'<article[^>]*>(.*?)</article>', text, re.DOTALL | re.IGNORECASE)
        if article_matches:
            text = max(article_matches, key=len)
        else:
            for pattern in [
                r'<div[^>]*class="[^"]*(?:article-body|story-body|'
                r'content-body|article__body|post-content)[^"]*"[^>]*>(.*?)</div>',
                r'<main[^>]*>(.*?)</main>',
            ]:
                m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if m:
                    text = m.group(1)
                    break
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        return _filter_sentences(text)
    except Exception as e:
        log.debug(f"Article fetch failed {url}: {e}")
        return ""


def _ensure_chromium_installed() -> bool:
    """
    Ensures Playwright's Chromium browser is downloaded.

    Railway's build step for `playwright install chromium` proved
    unreliable across multiple builder/venv configurations (the
    browser binary ended up in a location not present at runtime).
    Instead, the bot installs it lazily on first use, via the SAME
    Python interpreter that's currently running (sys.executable) —
    this guarantees consistency since it's the same environment.

    Runs once per process (cached via module-level flag). Takes
    ~60-120s the first time (downloads Chromium + apt packages);
    subsequent calls return immediately.
    Returns True if Chromium is available, False if install failed.
    """
    global _CHROMIUM_INSTALL_ATTEMPTED, _CHROMIUM_INSTALL_OK
    if _CHROMIUM_INSTALL_ATTEMPTED:
        return _CHROMIUM_INSTALL_OK

    _CHROMIUM_INSTALL_ATTEMPTED = True
    try:
        import subprocess
        import sys
        log.info("FIA official: installing Chromium + system deps "
                 "(first use, ~60-120s)...")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install",
             "--with-deps", "chromium"],
            capture_output=True, text=True, timeout=240)
        if result.returncode == 0:
            log.info("FIA official: Chromium + deps installed ✅")
            _CHROMIUM_INSTALL_OK = True
        else:
            log.warning(
                f"FIA official: Chromium install failed "
                f"(rc={result.returncode}): {result.stderr[-400:]}")
            _CHROMIUM_INSTALL_OK = False
    except Exception as e:
        log.warning(f"FIA official: Chromium install error — {e}")
        _CHROMIUM_INSTALL_OK = False

    return _CHROMIUM_INSTALL_OK


_CHROMIUM_INSTALL_ATTEMPTED = False
_CHROMIUM_INSTALL_OK        = False


def _fetch_fia_official_docs(race_context: str, driver_name: str,
                              incident_type: str) -> str:
    """
    Fetches the ACTUAL FIA decision document PDF using a headless browser.

    fia.com returns 403 for plain HTTP requests (requests/urllib),
    but serves pages normally to real browsers. Playwright launches
    headless Chromium, navigates to the FIA documents page for the
    season, finds the document link matching the incident, downloads
    the PDF, and extracts text with pypdf.

    Returns "" on ANY failure (Playwright not installed, timeout,
    no matching doc, etc.) — caller falls back to motorsport.com/
    planetf1 search, which is the existing reliable path.

    This function must NEVER raise — every failure mode degrades
    to empty string so the bot keeps working even if this entire
    feature is broken or unavailable on the host.

    IMPORTANT: Playwright's sync API cannot run inside an asyncio
    event loop (the bot's whole call chain runs on one). The actual
    browser work happens in `_run_fia_playwright`, executed in a
    separate thread via ThreadPoolExecutor — that thread has no
    event loop, satisfying Playwright's requirement, while this
    function (called from sync code inside the bot) can block on
    the thread's result without itself needing to be async.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.debug("FIA official: Playwright not installed — skipping")
        return ""

    if not _ensure_chromium_installed():
        log.debug("FIA official: Chromium unavailable — skipping")
        return ""

    race_name_clean = re.sub(r'\s*2026.*$', '', race_context).strip()
    if not race_name_clean:
        return ""

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_fia_playwright, race_name_clean, driver_name, incident_type)
            return future.result(timeout=60)
    except Exception as e:
        log.info(f"FIA official: thread execution failed — {e}")
        return ""


def _run_fia_playwright(race_name_clean: str, driver_name: str,
                         incident_type: str) -> str:
    """
    The actual Playwright sync-API work. Runs in its own thread
    (no asyncio event loop) via _fetch_fia_official_docs above.
    """
    from playwright.sync_api import sync_playwright

    season_url = (
        "https://www.fia.com/documents/championships/"
        "fia-formula-one-world-championship-14/season/season-2026-2072")

    # Keywords that should appear in the FIA URL slug for this race.
    # The slug uses the venue name ("barcelona-catalunya_grand_prix"), not
    # the national name ("spanish_grand_prix"), so we check significant words
    # from the race name ("spanish", "canadian", "austrian", …).
    race_kws: set[str] = set()
    for word in race_name_clean.lower().split():
        if word not in ("grand", "prix", "the", "de"):
            race_kws.add(word)

    try:
        with sync_playwright() as p:
            # --single-process is known to crash Chromium on launch in
            # constrained containers (BrowserType.launch: Target page,
            # context or browser has been closed). Removed. The other
            # flags are the standard container-safe set.
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage",
                      "--disable-gpu", "--disable-setuid-sandbox",
                      "--no-zygote"])
            try:
                page = browser.new_page()
                page.set_default_timeout(15000)  # 15s — fail fast

                # ── Step 1: load FIA season document library ──────────
                # FIA site change (2026): per-event document URLs now
                # 302-redirect to the homepage; the season library page is
                # the only reliable source of /system/files/decision-document/
                # links. Documents for past races won't be on this page —
                # graceful empty return in that case.
                page.goto(season_url, wait_until="domcontentloaded")

                # Build search terms for the doc link text based on
                # incident type and driver. Real FIA doc titles are varied
                # ("leaving the track without justifiable reason", "failing
                # to slow for yellow flags", "driving erratically"...), so
                # the keyword set is intentionally broad.
                doc_keywords = [
                    "infringement", "decision", "penalty", "investigat",
                    "summon", "noted", "track", "yellow", "flag",
                    "impeding", "erratically", "unsafe", "collision",
                    "leaving the track", "failing to", "causing",
                    "speeding", "deleted", "reprimand", "offence",
                ]
                if "disqualif" in incident_type.lower():
                    doc_keywords = ["disqualif"] + doc_keywords
                elif "track limit" in incident_type.lower():
                    doc_keywords = ["track limits"] + doc_keywords

                # FIA docs name the driver by car number, not surname.
                # Resolve the named driver -> 2026 car number so we can
                # match e.g. "car 16" in the doc title.
                car_token = ""
                if driver_name:
                    for code, num in FIA_DRIVER_CAR_NUMBERS.items():
                        if FIA_DRIVER_NAMES.get(code.lower()) == driver_name:
                            car_token = f"car {num}"
                            break

                # ── Step 2: scan season-page PDF links ────────────────
                # Only /system/files/decision-document/ links are published
                # docs; filter by race keyword in href slug and incident
                # keywords in link text.
                all_links = page.locator(
                    "a[href*='/system/files/decision-document/']").all()
                count = len(all_links)  # all() materialises the list

                candidate_url  = ""   # best match so far
                candidate_text = ""
                fallback_url   = ""   # first stewards-keyword doc (no car)
                fallback_text  = ""
                for link in all_links:
                    href = (link.get_attribute("href") or "").lower()
                    text = (link.inner_text() or "").lower()
                    # Skip docs for other races (season page may list several)
                    if race_kws and not any(
                            kw in href for kw in race_kws if len(kw) > 3):
                        continue
                    has_kw  = any(kw in text for kw in doc_keywords)
                    has_car = bool(car_token) and car_token in text
                    if not (has_kw or has_car):
                        continue
                    # Best: doc names the driver's car AND is a stewards doc
                    if has_car and has_kw:
                        candidate_url  = href
                        candidate_text = text
                        break
                    # Next best: names the car (even without a keyword hit)
                    if has_car and not candidate_url:
                        candidate_url  = href
                        candidate_text = text
                    # Fallback: a stewards-keyword doc with no driver named
                    if has_kw and not fallback_url:
                        fallback_url  = href
                        fallback_text = text

                if not candidate_url:
                    candidate_url  = fallback_url
                    candidate_text = fallback_text

                if not candidate_url:
                    log.info(f"FIA official: no infringement doc found "
                             f"for {race_name_clean}")
                    return ""

                if candidate_url.startswith("/"):
                    candidate_url = "https://www.fia.com" + candidate_url

                log.info(f"FIA official: found doc '{candidate_text[:60]}'")

                # ── Step 3: download the PDF via the browser context ──
                pdf_response = page.request.get(candidate_url)
                if pdf_response.status != 200:
                    log.info(f"FIA official: PDF fetch returned "
                             f"{pdf_response.status}")
                    return ""
                pdf_bytes = pdf_response.body()

            finally:
                browser.close()

    except Exception as e:
        log.info(f"FIA official: Playwright fetch failed — {e}")
        return ""

    # ── Step 4: extract text from PDF ─────────────────────────────
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(
            (p.extract_text() or "") for p in reader.pages)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 50:
            return ""
        return f"[FIA OFFICIAL DOCUMENT: {candidate_text.strip()[:80]}]\n{text[:1000]}"
    except Exception as e:
        log.info(f"FIA official: PDF text extraction failed — {e}")
        return ""


def _parse_circuit_map_pdf_text(text: str) -> dict:
    """
    Parses pypdf-extracted text from a FIA Circuit Map PDF.

    pypdf reads the CIRCUIT DATA table column-by-column, so the raw text
    has an unusual ordering that this function accounts for:
      - Zone name labels appear first  ("- ZONE A1 -", "- ZONE A2 -", ...)
      - "ACTIVATION" label alone on a line
      - "DETECTION {value}" — label merged with zone[0]'s normal-grip distance
      - Remaining normal-grip distances (zones 1..n-1)
      - Low-grip distances (all zones, in order)
      - "- {activation_distance}" then "- {detection_distance}" (overtake values)

    Returns:
      {
        "overtake": {"detection": "Apex T13", "activation": "Entry T14"},
        "straight_mode_zones": [
          {"zone": "A1", "activation_normal": "45m after T14",
                         "activation_low_grip": "85m after T14"},
          ...   # zones whose both values are "n/a" are omitted
        ]
      }
    Returns {} on any parse failure (caller degrades gracefully).
    """
    # Isolate the CIRCUIT DATA block between the table header and LEGEND
    section_m = re.search(
        r'OVERTAKE\s+STRAIGHT\s+MODE(.*?)(?:\bLEGEND\b|VERSION\s+1)',
        text, re.DOTALL | re.IGNORECASE
    )
    if not section_m:
        return {}

    section = section_m.group(1)
    lines = [ln.strip() for ln in section.splitlines() if ln.strip()]

    # --- Zone names from "- ZONE A1 -" lines ---
    zone_names = []
    for ln in lines:
        m = re.search(r'\bZONE\s+([A-Z]\d+)\b', ln)
        if m:
            zone_names.append(m.group(1))

    if not zone_names:
        return {}

    # --- Measurement-value pattern ---
    # Matches: "45m after T14", "40m before T3 exit", "T3 exit", "n/a"
    MEAS = re.compile(
        r'^(?:\d+[\d.]*m\s+(?:after|before)\s+T\S.*|T\d+\s+\w.*|n/a)$',
        re.IGNORECASE
    )

    # --- Normal-grip values ---
    # "DETECTION {value}" line gives zone[0]; subsequent MEAS lines give the rest.
    normal_vals = []
    detection_line_idx = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("DETECTION "):
            normal_vals.append(ln[len("DETECTION "):].strip())
            detection_line_idx = i
            break

    idx = (detection_line_idx + 1) if detection_line_idx is not None else len(lines)
    while len(normal_vals) < len(zone_names) and idx < len(lines):
        if MEAS.match(lines[idx]):
            normal_vals.append(lines[idx])
        idx += 1

    # --- Low-grip values (immediately follow normal-grip block) ---
    low_vals = []
    while len(low_vals) < len(zone_names) and idx < len(lines):
        if MEAS.match(lines[idx]):
            low_vals.append(lines[idx])
        idx += 1

    # --- Overtake values: "- Entry T14" / "- Apex T13" ---
    # pypdf order: activation first, detection second.
    overtake_raw = [
        re.sub(r'^-\s+', '', ln).strip()
        for ln in lines
        if re.match(r'^-\s+\S', ln) and 'ZONE' not in ln.upper()
    ]
    activation_val = overtake_raw[0] if len(overtake_raw) > 0 else ""
    detection_val  = overtake_raw[1] if len(overtake_raw) > 1 else ""

    # --- Build result, skipping fully-n/a zones ---
    zones = []
    for i, zone in enumerate(zone_names):
        norm = normal_vals[i] if i < len(normal_vals) else ""
        low  = low_vals[i]    if i < len(low_vals)    else ""
        if norm.lower() == "n/a" and low.lower() == "n/a":
            continue
        zones.append({
            "zone": zone,
            "activation_normal":   norm,
            "activation_low_grip": low,
        })

    return {
        "overtake": {"detection": detection_val, "activation": activation_val},
        "straight_mode_zones": zones,
    }


# ── Circuit-key → FIA season-page event name ──────────────────────────────────
# Used by _get_circuit_zone_data to navigate to the right event document list.
_CIRCUIT_KEY_TO_FIA_EVENT = {
    "melbourne":   "Australian Grand Prix",
    "shanghai":    "Chinese Grand Prix",
    "suzuka":      "Japanese Grand Prix",
    "bahrain":     "Bahrain Grand Prix",
    "jeddah":      "Saudi Arabian Grand Prix",
    "miami":       "Miami Grand Prix",
    "imola":       "Emilia Romagna Grand Prix",
    "monaco":      "Monaco Grand Prix",
    "montreal":    "Canadian Grand Prix",
    "barcelona":   "Spanish Grand Prix",
    "spielberg":   "Austrian Grand Prix",
    "silverstone": "British Grand Prix",
    "budapest":    "Hungarian Grand Prix",
    "spa":         "Belgian Grand Prix",
    "zandvoort":   "Dutch Grand Prix",
    "monza":       "Italian Grand Prix",
    "baku":        "Azerbaijan Grand Prix",
    "singapore":   "Singapore Grand Prix",
    "austin":      "United States Grand Prix",
    "mexico city": "Mexico City Grand Prix",
    "são paulo":   "São Paulo Grand Prix",
    "las vegas":   "Las Vegas Grand Prix",
    "lusail":      "Qatar Grand Prix",
    "abu dhabi":   "Abu Dhabi Grand Prix",
}


def _run_circuit_map_playwright(race_name: str, circuit_key: str = "") -> bytes:
    """
    Playwright sync-API work (runs in its own thread — no asyncio event loop).
    Fetches the FIA season document library page and finds the
    'Competition Notes - Circuit Map / Pit Lane Drawing' PDF for the given
    race, then returns its bytes.  Returns b"" on any failure.

    FIA site change (2026): per-event document URLs now redirect to the
    homepage; the season library page is the only reliable source of
    /system/files/decision-document/ PDF links.
    """
    from playwright.sync_api import sync_playwright

    season_url = (
        "https://www.fia.com/documents/championships/"
        "fia-formula-one-world-championship-14/season/season-2026-2072")

    # Keywords that should appear in the FIA URL slug for this race.
    # The slug uses venue name (e.g. "barcelona-catalunya_grand_prix"), not the
    # national race name ("spanish_grand_prix"), so we check both the circuit
    # key ("barcelona") and significant words from the race name ("spanish").
    match_kws: set[str] = set()
    if circuit_key:
        match_kws.add(circuit_key.lower())
    for word in race_name.lower().split():
        if word not in ("grand", "prix", "the", "de"):
            match_kws.add(word)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage",
                      "--disable-gpu", "--disable-setuid-sandbox",
                      "--no-zygote"])
            try:
                page = browser.new_page()
                page.set_default_timeout(15000)

                # ── Step 1: load FIA season document library ───────────
                # Documents are published at /system/files/decision-document/
                # links directly on this page (per-event URLs now redirect).
                page.goto(season_url, wait_until="domcontentloaded")

                # ── Step 2: find circuit-map PDF link ──────────────────
                all_links = page.locator(
                    "a[href*='/system/files/decision-document/']").all()

                pdf_url = ""
                for link in all_links:
                    href = (link.get_attribute("href") or "").lower()
                    text = (link.inner_text() or "").lower()

                    # Must be a circuit-map document
                    if "circuit_map" not in href and "circuit map" not in text:
                        continue

                    # Must belong to this race (slug keyword check)
                    if match_kws and not any(
                            kw in href for kw in match_kws if len(kw) > 3):
                        log.info(
                            f"Circuit map [{circuit_key}]: skipping doc '{text[:60]}' "
                            f"(no race keyword match for {race_name!r})")
                        continue

                    pdf_url = link.get_attribute("href") or ""
                    log.info(f"Circuit map [{circuit_key}]: found doc '{text[:80]}'")
                    break

                if not pdf_url:
                    log.warning(
                        f"Circuit map [{circuit_key}]: no circuit-map doc found for "
                        f"'{race_name}' on FIA season page "
                        f"(document may not be published yet)")
                    return b""

                if pdf_url.startswith("/"):
                    pdf_url = "https://www.fia.com" + pdf_url

                # ── Step 3: download PDF ───────────────────────────────
                resp = page.request.get(pdf_url)
                if resp.status != 200:
                    log.info(f"Circuit map: PDF fetch returned {resp.status}")
                    return b""
                return resp.body()

            finally:
                browser.close()

    except Exception as e:
        log.info(f"Circuit map Playwright fetch failed — {e}")
        return b""


def _fetch_circuit_map_pdf(race_name: str, circuit_key: str = "") -> bytes:
    """
    Thread-executor wrapper around _run_circuit_map_playwright.
    Returns PDF bytes or b"" on any failure.  Never raises.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        log.debug("Circuit map: Playwright not installed — skipping")
        return b""

    if not _ensure_chromium_installed():
        log.debug("Circuit map: Chromium unavailable — skipping")
        return b""

    try:
        from concurrent.futures import ThreadPoolExecutor
        with _circuit_map_playwright_lock:  # one Playwright session at a time
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _run_circuit_map_playwright, race_name, circuit_key)
                return future.result(timeout=60)
    except Exception as e:
        log.info(f"Circuit map [{circuit_key}]: thread execution failed — {e}")
        return b""


def _load_circuit_map_cache() -> dict:
    if CIRCUIT_MAP_CACHE_FILE.exists():
        try:
            return json.loads(CIRCUIT_MAP_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_circuit_map_cache(cache: dict):
    try:
        CIRCUIT_MAP_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        log.error(f"Circuit map: failed to save cache — {e}")


def _get_circuit_zone_data(circuit_key: str) -> dict:
    """
    Returns Straight Mode / Overtake zone data for a circuit.

    Fetch-on-demand: reads from CIRCUIT_MAP_CACHE_FILE on cache hit (non-empty
    value only — empty dicts mean the doc wasn't published yet, so we retry).
    On a miss, runs the Playwright fetch + pypdf parse synchronously, caches
    the result, and returns it.  Returns {} on any failure.
    """
    # Cache hit — only skip the full fetch if zone data AND image are both on disk.
    # A Railway redeploy wipes the filesystem, so the image may be missing even
    # when the JSON cache has a valid zone-data entry.  Falling through in that
    # case re-fetches the PDF and re-saves the image.
    cache = _load_circuit_map_cache()
    if cache.get(circuit_key):
        _cached_img = next(
            (CIRCUIT_MAPS_DIR / f"{circuit_key}{ext}"
             for ext in (".png", ".jpg", ".jpeg")
             if (CIRCUIT_MAPS_DIR / f"{circuit_key}{ext}").exists()),
            None,
        )
        if _cached_img:
            try:
                from PIL import Image as _PIL
                _im = _PIL.open(_cached_img)
                _w, _h = _im.size
                if _h > 0 and (_w / _h) <= 4.0 and _w >= 400:
                    return cache[circuit_key]   # good image — skip fetch
                # Stale bad image (logo or pit lane strip) — delete and re-fetch
                _cached_img.unlink(missing_ok=True)
                log.info(f"Circuit map [{circuit_key}]: deleted stale image "
                         f"({_w}×{_h}, ratio {_w/_h:.1f}) — will re-fetch")
            except Exception:
                pass  # can't read image — fall through to re-fetch

    # Look up the FIA event name for this circuit
    race_name = _CIRCUIT_KEY_TO_FIA_EVENT.get(circuit_key)
    if not race_name:
        return {}

    log.info(f"Circuit map: fetching for '{circuit_key}' ({race_name})")
    pdf_bytes = _fetch_circuit_map_pdf(race_name, circuit_key)
    if not pdf_bytes:
        return {}

    # Extract text from the PDF
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join(
            (pg.extract_text() or "") for pg in reader.pages)
    except Exception as e:
        log.info(f"Circuit map [{circuit_key}]: PDF text extraction failed — {e}")
        return {}

    result = _parse_circuit_map_pdf_text(full_text)

    # Extract and save the best circuit-map image for send_photo in
    # handle_message.  Stored at CIRCUIT_MAPS_DIR/{circuit_key}.{ext}.
    # Image failure never blocks zone-data caching.
    #
    # Heuristic (derived from Barcelona PDF inventory):
    #   Pick the FIRST image in page order where PIL can decode it,
    #   aspect ratio (w/h) ≤ 4.0, and width ≥ 400 px.
    #   This selects the overhead circuit layout (Im1.jpg, page 1, 702×387,
    #   ratio 1.81) and skips the FIA header bar (ratio 11.2) and the pit
    #   lane drawing (R9.png, 3016×431, ratio 7.0).
    #   Fallback: if nothing passes the filter, take the largest by byte size.
    try:
        from PIL import Image as _PILImage

        best_data: bytes = b""
        best_ext: str = ".png"
        fallback_data: bytes = b""
        fallback_ext: str = ".png"

        for pg in reader.pages:
            if best_data:
                break
            for img in pg.images:
                raw = img.data
                # track fallback (largest overall) before filter
                if len(raw) > len(fallback_data):
                    fallback_data = raw
                    fallback_ext = (
                        ".jpg" if img.name.lower().endswith((".jpg", ".jpeg"))
                        else ".png")
                try:
                    pil = _PILImage.open(__import__("io").BytesIO(raw))
                    w, h = pil.size
                except Exception:
                    continue
                if h == 0 or (w / h) > 4.0 or w < 400:
                    continue
                best_data = raw
                best_ext = (
                    ".jpg" if img.name.lower().endswith((".jpg", ".jpeg"))
                    else ".png")
                break  # first qualifying image wins

        if not best_data:
            best_data, best_ext = fallback_data, fallback_ext

        if best_data:
            CIRCUIT_MAPS_DIR.mkdir(exist_ok=True)
            img_path = CIRCUIT_MAPS_DIR / f"{circuit_key}{best_ext}"
            img_path.write_bytes(best_data)
            log.info(
                f"Circuit map [{circuit_key}]: saved image {img_path.name} "
                f"({len(best_data):,} bytes)")
    except Exception as e:
        log.info(f"Circuit map [{circuit_key}]: image extraction failed — {e}")

    # Cache even an empty result so we know we tried; caller retries on empty.
    cache[circuit_key] = result
    _save_circuit_map_cache(cache)

    if result:
        log.info(f"Circuit map: cached zone data for '{circuit_key}' "
                 f"({len(result.get('straight_mode_zones', []))} SM zones)")
    else:
        log.info(f"Circuit map: parse returned empty for '{circuit_key}' "
                 f"(doc may not match expected format)")
    return result


async def _prewarm_circuit_map_for_next_race():
    """
    Fire-and-forget startup task: fetches and caches circuit zone data for
    the next/current race weekend so the first user query isn't delayed.
    Failures are logged but never propagate — bot startup is unaffected.

    Uses non-blocking lock acquisition so a concurrent user request always
    wins: if the Playwright lock is already held, the prewarm skips rather
    than queuing behind and delaying the user.
    """
    import asyncio as _asyncio
    try:
        next_race = fetch_next_race()
        if not next_race:
            return
        race_name   = next_race.get("raceName", "")
        circuit_key = next(
            (k for k, v in _CIRCUIT_KEY_TO_FIA_EVENT.items()
             if v.lower() == race_name.lower()), "")
        if not circuit_key:
            log.info(f"Circuit map pre-warm: no key for '{race_name}' — skipping")
            return
        # Already cached with real data — nothing to do.
        cache = _load_circuit_map_cache()
        if cache.get(circuit_key):
            log.info(f"Circuit map pre-warm: '{circuit_key}' already cached")
            return
        # Acquire lock non-blocking — skip rather than block user requests.
        if not _circuit_map_playwright_lock.acquire(blocking=False):
            log.info("Circuit map pre-warm: skipping — Playwright lock held by user request")
            return
        try:
            log.info(f"Circuit map pre-warm: fetching '{circuit_key}' ({race_name})")
            loop = _asyncio.get_event_loop()
            await loop.run_in_executor(None, _get_circuit_zone_data, circuit_key)
        finally:
            _circuit_map_playwright_lock.release()
    except Exception as e:
        log.info(f"Circuit map pre-warm failed (non-fatal): {e}")


def fetch_fia_race_documents(race_name: str, query: str = "") -> str:
    """
    Fetches official stewards decision content for a race incident/penalty.

    Strategy:
    1. Build targeted search query from race + driver + incident type
    2. Search Google for coverage from trusted F1 sources
    3. Fetch top 3 articles and extract stewards verdict text
    4. Return combined findings with source attribution

    This reliably gets the exact FIA stewards finding because
    motorsport.com/planetf1/the-race all quote the official
    stewards document verbatim within hours of each decision.
    """
    q_lower = (race_name + " " + query).lower()

    # Cache check (2 hour TTL)
    cache_key = re.sub(r'\W+', '_', q_lower[:50])
    if cache_key in _FIA_DOC_CACHE:
        cached_time = _FIA_CACHE_TIMES.get(cache_key, 0)
        if (datetime.now().timestamp() - cached_time) < 7200:
            return _FIA_DOC_CACHE[cache_key]

    # Detect race
    race_context = ""
    for keyword, race_str in FIA_RACE_KEYWORDS.items():
        if keyword in q_lower:
            race_context = race_str
            break
    if not race_context:
        # Use most recent race from name
        race_context = race_name + " 2026 F1"

    # Detect driver. Match on word boundaries and try the longest keys
    # first, so short 3-letter codes ("per", "gas", "had", "bot", "str")
    # don't false-match inside unrelated words (e.g. "per" in "stopper").
    driver_name = ""
    for keyword in sorted(FIA_DRIVER_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(keyword)}\b", q_lower):
            driver_name = FIA_DRIVER_NAMES[keyword]
            break

    # Detect incident type for search query
    incident_type = ""
    if any(w in q_lower for w in ["penalty","penalt","sanction"]):
        incident_type = "penalty stewards decision"
    elif any(w in q_lower for w in ["crash","collision","incident","contact"]):
        incident_type = "crash collision stewards investigation"
    elif any(w in q_lower for w in ["disqualif","dsq"]):
        incident_type = "disqualified DSQ stewards"
    elif any(w in q_lower for w in ["track limit"]):
        incident_type = "track limits penalty"
    else:
        incident_type = "stewards decision penalty"

    # Build search query
    parts = [race_context]
    if driver_name:
        parts.append(driver_name)
    parts.append(incident_type)
    parts.append("FIA")
    search_q = " ".join(parts)

    # ── Try official FIA document first (Playwright) ──────────────
    # This is the ground truth — exact regulation, exact wording.
    # Falls back silently to motorsport.com search below if it
    # fails for ANY reason (not installed, timeout, no match, etc.)
    official = _fetch_fia_official_docs(race_context, driver_name, incident_type)
    if official:
        log.info("FIA: using OFFICIAL document ✅")
        _FIA_DOC_CACHE[cache_key]  = official
        _FIA_CACHE_TIMES[cache_key] = datetime.now().timestamp()
        return official

    log.info(f"FIA search: '{search_q}'")

    # Search Google
    results = google_search_f1(search_q, num_results=6)
    if not results:
        return ""

    # Filter to trusted sources
    trusted_results = []
    for r in results:
        url = r.get("url","")
        if any(src in url for src in FIA_STEWARDS_SOURCES):
            trusted_results.append(r)
    # Fall back to all results if no trusted ones
    if not trusted_results:
        trusted_results = results[:3]

    # Fetch article content from top 3 sources
    collected = []
    for result in trusted_results[:3]:
        url     = result.get("url","")
        title   = result.get("title","")
        snippet = result.get("snippet","")

        # Use snippet if it has stewards content (saves a fetch)
        stewards_keywords = [
            "steward","penalty","infringement","article",
            "video evidence","standard penalty","out of position",
            "grid slot","drive-through","time penalty","reprimand",
            "found that","regulation","collision","restart"
        ]
        snippet_useful = any(kw in snippet.lower()
                             for kw in stewards_keywords)

        if snippet_useful and len(snippet) > 80:
            collected.append(f"[{title}]\n{snippet}")
            log.info(f"FIA: used snippet from {url[:50]}")
        else:
            # Fetch full article
            article = _extract_article_text(url, max_chars=600)
            if article:
                collected.append(f"[{title}]\n{article}")
                log.info(f"FIA: fetched article from {url[:50]}")

        if len(collected) >= 3:
            break

    if not collected:
        return ""

    result_text = (
        f"STEWARDS DECISION SOURCES ({race_context}):\n\n" +
        "\n\n".join(collected)
    )

    # Cache result
    _FIA_DOC_CACHE[cache_key]  = result_text
    _FIA_CACHE_TIMES[cache_key] = datetime.now().timestamp()

    return result_text



def live_search_f1(query: str, race_context: str = "") -> str:
    """
    Main live search function. Searches Google + news feeds for
    any F1 session or topic question. Returns context string.
    """
    # Build optimized search query
    current = fetch_current_race()
    race_name = ""
    if current:
        race_name = current.get("raceName", "")

    # Add race context if not already in query
    search_q = query
    if race_name and race_name.lower().split()[0] not in query.lower():
        search_q = f"{race_name} {query} 2026"
    else:
        search_q = f"{query} F1 2026"

    # Remove question words for better search
    search_q = re.sub(
        r'\b(what|happened|is|happening|tell me about|how did|'
        r'qué|pasó|cómo|cuéntame)\b',
        "", search_q, flags=re.IGNORECASE
    ).strip()
    search_q = re.sub(r'\s+', ' ', search_q)

    log.info(f"Live search: '{search_q}'")

    results = []

    # 1. Google search
    google_results = google_search_f1(search_q, num_results=4)
    if google_results:
        for r in google_results[:2]:
            # Fetch full article for top results
            if r.get("url"):
                content = fetch_article_content(r["url"], max_chars=800)
                if content and len(content) > 100:
                    results.append(
                        f"{r.get('title','')}: {content}")
                elif r.get("snippet"):
                    results.append(
                        f"{r.get('title','')}: {r['snippet']}")

    # 2. News feed search as fallback/supplement
    news = get_news_context(search_q)
    if news and len(news) > 50:
        results.append(news)

    if not results:
        return ""

    combined = "\n\n".join(results)
    return f"Live search results for '{query}':\n{combined[:2500]}"






# ═════════════════════════════════════════════════════════════
#  FEATURE: CIRCUIT GUIDES
# ═════════════════════════════════════════════════════════════





# ═════════════════════════════════════════════════════════════
#  OFF-TOPIC GUARDRAIL
# ═════════════════════════════════════════════════════════════






# ═════════════════════════════════════════════════════════════
BANNER = """🏎 *BoxBoxAI* — F1 Race Analyst
_Developed by Erick Hernandez_
━━━━━━━━━━━━━━━━━━━━━━"""



# Global app reference for alerts from non-async contexts

WELCOME = """🏎 *Welcome to BoxBoxAI!*
_Your AI-powered F1 race analyst_

━━━━━━━━━━━━━━━━━━━━━━

I have full memory of the *2026 F1 season* — every race, qualifying session, strategy call, championship battle, and driver story.

Just ask me anything. Some ideas:
• _"Who's leading the championship?"_
• _"Why did Russell retire in Montreal?"_
• _"Break down Antonelli's dominance"_
• _"Who should I bet on for Spain?"_
• _"Compare Hamilton vs Leclerc this season"_
• _"Top 10 FP2"_

*Commands:*
/standings — driver championship
/constructors — constructor standings
/winner — race winner pick
/predict — full race prediction
/help — all commands

Let's talk F1 🚀"""

HELP_TEXT = """*BoxBoxAI — F1 Race Analyst* 🏎
_Developed by Erick Hernandez_

*Commands:*
/standings — 🏆 Driver championship
/constructors — 🏗 Constructor standings
/winner — 🥇 Race winner prediction
/predict — 🎯 Race prediction
/help — This menu

*Just talk to me:*
Ask me anything about F1 — drivers, races, strategy, championship, history. I'll answer.

_Examples:_
"Who will win Monaco?"
"Tell me about Antonelli"
"What happened in Canada?"
"Can Verstappen still win the championship?"
"""



async def auto_ingest_loop(mem_ref: list, app=None,
                           sessions_ref: list = None):
    """
    Background loop that automatically ingests new race results,
    qualifying, and sprint results every 30 minutes.
    """
    while True:
        try:
            await _check_and_ingest(mem_ref, app, sessions_ref)
        except Exception as e:
            log.warning(f"Auto-ingest loop error: {e}")
        await asyncio.sleep(1800)  # check every 30 minutes


# ═════════════════════════════════════════════════════════════
#  AUTO-PREDICTOR LOOP
#  Runs f1_2026_predictor.py automatically after qualifying
#  and before each race. Updates CSV on Railway so /winner
#  and /predict always have fresh Monte Carlo numbers.
# ═════════════════════════════════════════════════════════════

PREDICTOR_SCRIPT   = Path(__file__).parent / "f1_2026_predictor.py"


async def run_predictor_subprocess() -> tuple[bool, str]:
    """
    Runs f1_2026_predictor.py as an async subprocess.
    Returns (success, message).
    Times out after 10 minutes — predictor fetches APIs + runs 10k MC sims.
    """
    if not PREDICTOR_SCRIPT.exists():
        return False, f"Predictor not found at {PREDICTOR_SCRIPT}"

    log.info("Auto-predictor: starting f1_2026_predictor.py...")
    try:
        import sys as _sys
        proc = await asyncio.create_subprocess_exec(
            _sys.executable, str(PREDICTOR_SCRIPT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PREDICTOR_SCRIPT.parent)
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=600)  # 10 min timeout
        except asyncio.TimeoutError:
            proc.kill()
            # CSV may still have been written before timeout
            if PREDICTOR_CSV.exists():
                log.info("Auto-predictor: timed out but CSV exists — using it")
                return True, "Timeout — using partial CSV"
            return False, "Predictor timed out after 10 minutes"

        if proc.returncode != 0:
            # Still use CSV if written
            if PREDICTOR_CSV.exists():
                log.info("Auto-predictor: exit error but CSV exists — using it")
                return True, "Exit error — using CSV"
            output = stdout.decode("utf-8", errors="ignore")[-500:] if stdout else ""
            return False, f"Predictor failed (exit {proc.returncode}): {output}"

        if not PREDICTOR_CSV.exists():
            return False, "Predictor ran but CSV not found"

        log.info("Auto-predictor: ✅ CSV updated successfully")
        return True, "Success"

    except FileNotFoundError:
        return False, "Python executable not found"
    except Exception as e:
        return False, f"Unexpected error: {e}"


async def auto_predictor_loop(app=None):
    """
    Background loop that runs the predictor automatically:
    - After qualifying (Saturday) for the upcoming race
    - After sprint qualifying on sprint weekends
    - Checks every 30 minutes — only runs once per session per round

    Qualifying data being available = signal to run predictor.
    Fresh qualifying grid is the most valuable input for the model.
    """
    while True:
        try:
            await _check_and_run_predictor(app)
        except Exception as e:
            log.warning(f"Auto-predictor loop error: {e}")
        await asyncio.sleep(1800)  # check every 30 minutes


async def _check_and_run_predictor(app=None):
    """
    Checks if qualifying results are available for the next race.
    If yes and predictor hasn't run for this round yet, runs it.
    """
    state = load_predictor_state()
    today = datetime.now().date()

    for rnd, name, date_str in [
        (6,"Monaco GP","2026-06-07"),
        (7,"Spanish GP","2026-06-14"),
        (8,"Austrian GP","2026-06-28"),
        (9,"British GP","2026-07-05"),
        (10,"Belgian GP","2026-07-19"),
        (11,"Hungarian GP","2026-07-26"),
        (12,"Dutch GP","2026-08-23"),
        (13,"Italian GP","2026-09-06"),
        (14,"Singapore GP","2026-09-20"),
        (15,"Azerbaijan GP","2026-09-27"),
        (16,"US GP","2026-10-18"),
        (17,"Mexico City GP","2026-10-25"),
        (18,"São Paulo GP","2026-11-08"),
        (19,"Las Vegas GP","2026-11-21"),
        (20,"Qatar GP","2026-11-29"),
        (21,"Abu Dhabi GP","2026-12-06"),
    ]:
        race_date  = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_until = (race_date - today).days

        # Only check races that are 0-2 days away (qualifying weekend)
        if days_until < 0 or days_until > 2:
            continue

        # Already ran predictor for this round?
        state_key = f"predictor_r{rnd}"
        if state.get(state_key):
            break

        # Check if qualifying data is available from Jolpica
        log.info(f"Auto-predictor: checking qualifying for R{rnd} {name}...")
        try:
            data = safe_get(f"{JOLPICA}/{SEASON}/{rnd}/qualifying.json")
            races = (data or {}).get("MRData", {}).get(
                "RaceTable", {}).get("Races", [])
            quali_results = races[0].get("QualifyingResults", []) if races else []
        except Exception:
            quali_results = []

        if len(quali_results) < 10:
            log.info(f"Auto-predictor: qualifying not available yet for R{rnd}")
            break

        # Qualifying is available — run the predictor
        log.info(f"Auto-predictor: qualifying detected for R{rnd} {name} — running predictor...")

        success, msg = await run_predictor_subprocess()

        if success:
            # Clear predictor cache so next /winner uses fresh CSV
            _PREDICTOR_CACHE.clear()

            state[state_key] = datetime.now().isoformat()
            save_predictor_state(state)

            # Read top prediction for the alert
            _, rows = get_predictor_context()
            pred_summary = ""
            if rows:
                r = rows[0]
                try:    win_str = f"{float(r.get('win_mc_pct','?')):.1f}%"
                except: win_str = "?"
                pred_summary = (
                    f"\n\n🥇 Model says: *{r.get('FullName','?')}* "
                    f"({r.get('TeamName','?')}) — {win_str} win probability"
                )

            log.info(f"Auto-predictor: ✅ R{rnd} {name} prediction ready")

            if app:
                await alert_owner(app,
                    f"✅ *Auto-predictor complete: R{rnd} {name}*"
                    f"{pred_summary}\n\n"
                    f"Users can now use /winner and /predict full 🏎")
        else:
            log.warning(f"Auto-predictor: failed for R{rnd} — {msg}")
            if app:
                await alert_owner(app,
                    f"⚠️ *Auto-predictor failed: R{rnd} {name}*\n\n"
                    f"Error: {msg}\n\n"
                    f"/winner will use memory-based prediction as fallback.")
        break  # only process one round per check


async def _check_and_ingest(mem_ref: list, app=None,
                             sessions_ref: list = None):
    """
    Checks if new results are available and ingests them.
    Handles: qualifying, sprint qualifying, sprint race, and race results.
    """
    state      = load_ingest_state()
    today      = datetime.now().date()

    RACE_CALENDAR = [
        (1,"Australian GP","2026-03-15"),
        (2,"Chinese GP","2026-03-22"),
        (3,"Japanese GP","2026-04-06"),
        (4,"Miami GP","2026-05-04"),
        (5,"Canadian GP","2026-05-24"),
        (6,"Monaco GP","2026-06-07"),
        (7,"Spanish GP","2026-06-14"),
        (8,"Austrian GP","2026-06-28"),
        (9,"British GP","2026-07-05"),
        (10,"Belgian GP","2026-07-19"),
        (11,"Hungarian GP","2026-07-26"),
        (12,"Dutch GP","2026-08-23"),
        (13,"Italian GP","2026-09-06"),
        (14,"Singapore GP","2026-09-20"),
        (15,"Azerbaijan GP","2026-09-27"),
        (16,"US GP","2026-10-18"),
        (17,"Mexico City GP","2026-10-25"),
        (18,"São Paulo GP","2026-11-08"),
        (19,"Las Vegas GP","2026-11-21"),
        (20,"Qatar GP","2026-11-29"),
        (21,"Abu Dhabi GP","2026-12-06"),
    ]

    for rnd, name, date_str in RACE_CALENDAR:
        race_date  = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_since = (today - race_date).days

        if days_since < -3:
            break  # future races, stop

        mem      = mem_ref[0]
        episodes = mem.get("episodic", [])

        # Find or create episode for this round
        episode = next((e for e in episodes if e.get("round") == rnd), None)
        if episode is None:
            episode = {"round": rnd, "race_name": name}

        changed = False

        # ── Qualifying (Saturday, days_since >= 1) ─────────────
        if days_since >= 1 and not episode.get("pole"):
            log.info(f"Auto-ingest: checking qualifying R{rnd} {name}...")
            quali = fetch_qualifying_result(rnd, SEASON)
            if quali:
                episode["pole"]       = quali["pole"]
                episode["pole_time"]  = quali.get("pole_time", "")
                episode["front_row"]  = quali.get("front_row", "")
                episode["grid_top10"] = quali.get("grid_top10", [])
                changed = True
                log.info(f"✅ Qualifying R{rnd}: pole={quali['pole']}")

                # Update semantic memory
                mem.setdefault("semantic", {})[f"quali/r{rnd}"] = {
                    "text": (f"R{rnd} {name} qualifying: "
                             f"Pole: {quali['pole']} ({quali.get('pole_time','')}). "
                             f"Front row: {quali.get('front_row','')}. "
                             f"Grid: {', '.join(quali.get('grid_top10',[])[:5])}")
                }

        # ── Sprint Race (sprint weekends, days_since >= 1) ─────
        if rnd in SPRINT_ROUNDS_2026 and days_since >= 1 \
                and not episode.get("sprint_winner"):
            log.info(f"Auto-ingest: checking sprint R{rnd} {name}...")
            sprint = fetch_sprint_result(rnd, SEASON)
            if sprint:
                episode["sprint_winner"] = sprint["winner"]
                episode["sprint_p2"]     = sprint["p2"]
                episode["sprint_p3"]     = sprint["p3"]
                episode["sprint_story"]  = sprint["story"]
                changed = True
                log.info(f"✅ Sprint R{rnd}: winner={sprint['winner']}")

                mem.setdefault("semantic", {})[f"sprint/r{rnd}"] = {
                    "text": (f"R{rnd} {name} sprint: "
                             f"{sprint['story']}")
                }

        # ── Race Result (Sunday, days_since >= 0) ──────────────
        if days_since >= 0 and not episode.get("winner"):
            log.info(f"Auto-ingest: checking race result R{rnd} {name}...")
            result = fetch_race_result(rnd, SEASON)
            if result:
                episode.update(result)
                changed = True
                log.info(f"✅ Race R{rnd}: winner={result['winner']}")

                if result.get("champ_after"):
                    mem.setdefault("semantic", {})[
                        f"standings/after_r{rnd}"] = {
                        "text": (f"After R{rnd} {name}: "
                                 f"{result['champ_after']}")
                    }

                # Notify users of race result
                if app and sessions_ref:
                    await _notify_race_result(app, sessions_ref[0], result)
                    await alert_owner(app,
                        f"✅ *Auto-ingest complete: R{rnd} {name}*\n\n"
                        f"🥇 Winner: {result.get('winner','?')}\n"
                        f"🥈 P2: {result.get('p2','?')}\n"
                        f"🥉 P3: {result.get('p3','?')}\n"
                        f"📊 {result.get('champ_after','')}"
                    )

        # Save if anything changed
        if changed:
            # Update episodes list
            existing_idx = next(
                (i for i, e in enumerate(episodes) if e.get("round") == rnd),
                None)
            if existing_idx is not None:
                episodes[existing_idx] = episode
            else:
                episodes.append(episode)
                episodes.sort(key=lambda x: x.get("round", 0))

            mem["episodic"] = episodes
            save_f1_memory(mem)
            mem_ref[0] = mem

            state[f"r{rnd}_last_check"] = datetime.now().isoformat()
            save_ingest_state(state)

        # Only process recent races (within 3 days)
        if days_since > 3:
            continue


async def _notify_race_result(app, sessions: dict, result: dict):
    """Sends race result notification to all active users."""
    active_users = get_active_user_ids(sessions)
    if not active_users:
        return

    winner = result.get("winner", "?")
    p2     = result.get("p2", "?")
    p3     = result.get("p3", "?")
    name   = result.get("race_name", "")
    champ  = result.get("champ_after", "")

    msg = (
        f"🏁 *{name} — RACE OVER!*\n\n"
        f"🥇 *{winner}*\n"
        f"🥈 {p2}\n"
        f"🥉 {p3}\n\n"
        f"{'📊 Championship: ' + champ if champ else ''}\n\n"
        f"Ask me anything about the race — strategy, incidents, "
        f"championship impact. 🏎"
    )

    sent = 0
    for uid in active_users:
        try:
            await app.bot.send_message(
                chat_id=uid, text=msg,
                parse_mode="Markdown")
            sent += 1
            await asyncio.sleep(0.05)
        except Exception:
            pass

    log.info(f"Race result notification sent to {sent} users")



# ═════════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════
#  CLAUDE API CALL
# ═════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════
#  WEATHER ENGINE — Open-Meteo (free, no API key)
# ═════════════════════════════════════════════════════════════

# All 2026 F1 circuits with coordinates



# Weather code → description + emoji


# ═════════════════════════════════════════════════════════════
#  MESSAGE SPLITTER — Telegram 4096 char limit
# ═════════════════════════════════════════════════════════════
def split_message(text: str, limit: int = TG_MAX_CHARS) -> list[str]:
    """Splits long messages at paragraph boundaries."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        # Find last paragraph break before limit
        cut = text.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return parts

# ═════════════════════════════════════════════════════════════
#  FORMATTED RESPONSES FOR COMMANDS
# ═════════════════════════════════════════════════════════════
def format_standings(drivers: list, constructors: list) -> str:
    if not drivers:
        return "⚠️ Standings unavailable right now. Try again shortly."

    leader_pts = float(drivers[0].get("points", 0))
    lines = [f"🏆 *2026 Championship — Round {SEASON}*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━",
             "", "*Drivers*"]

    medals = ["🥇", "🥈", "🥉"]
    for i, s in enumerate(drivers[:10]):
        drv  = s.get("Driver", {})
        name = f"{drv.get('givenName','')[:1]}. {drv.get('familyName','')}"
        team = s.get("Constructors",[{}])[0].get("name","")[:12]
        pts  = float(s.get("points", 0))
        gap  = "" if i == 0 else f"  _(-{int(leader_pts-pts)})_"
        medal = medals[i] if i < 3 else f"{i+1}."
        lines.append(f"{medal} *{name}* — {int(pts)}pts{gap}")

    if constructors:
        lines += ["", "*Constructors*"]
        c_leader = float(constructors[0].get("points", 0))
        for i, s in enumerate(constructors[:5]):
            name = s.get("Constructor",{}).get("name","")[:16]
            pts  = float(s.get("points", 0))
            gap  = "" if i == 0 else f"  _(-{int(c_leader-pts)})_"
            medal = medals[i] if i < 3 else f"{i+1}."
            lines.append(f"{medal} *{name}* — {int(pts)}pts{gap}")

    return "\n".join(lines)


def format_season(mem: dict) -> str:
    episodes = mem.get("episodic", [])
    if not episodes:
        return "No race data in memory yet."

    flags = {
        "Melbourne":"🇦🇺","Shanghai":"🇨🇳","Suzuka":"🇯🇵","Bahrain":"🇧🇭",
        "Jeddah":"🇸🇦","Miami":"🇺🇸","Imola":"🇮🇹","Monaco":"🇲🇨",
        "Montreal":"🇨🇦","Barcelona":"🇪🇸","Spielberg":"🇦🇹",
        "Silverstone":"🇬🇧","Budapest":"🇭🇺","Spa":"🇧🇪",
        "Zandvoort":"🇳🇱","Monza":"🇮🇹","Baku":"🇦🇿",
        "Singapore":"🇸🇬","Austin":"🇺🇸","Mexico City":"🇲🇽",
        "São Paulo":"🇧🇷","Las Vegas":"🇺🇸","Lusail":"🇶🇦",
        "Abu Dhabi":"🇦🇪",
    }

    lines = [f"📅 *2026 Season — {len(episodes)} races*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━"]

    for r in episodes:
        flag = flags.get(r["track"], "🏁")
        dnfs = r.get("dnfs", [])
        sc   = r.get("sc_count", 0)
        extras = []
        if dnfs:
            extras.append(f"DNF:{','.join(d.split('(')[0] for d in dnfs[:2])}")
        if sc:
            extras.append(f"SC×{sc}")
        extra_str = f"  _{' | '.join(extras)}_" if extras else ""

        lines.append(
            f"\n*R{r['round']}* {flag} {r['track']} `{r.get('date','')}`"
        )
        lines.append(
            f"🏆 {r['winner']}  P2:{r.get('p2','')}  P3:{r.get('p3','')}"
        )
        if r.get("qualifying", {}).get("pole"):
            lines.append(
                f"⏱ Pole: {r['qualifying']['pole']} {r['qualifying'].get('pole_time','')}"
            )
        if r.get("champ_delta"):
            lines.append(f"📊 _{r['champ_delta']}_")
        if extra_str:
            lines.append(extra_str)

    return "\n".join(lines)


def format_last_race(race: dict, mem: dict) -> str:
    if not race:
        # Fall back to memory
        episodes = mem.get("episodic", [])
        if not episodes:
            return "No race data available."
        r = episodes[-1]
        return (f"🏁 *Last race: {r.get('race_name', r['track'])}*\n\n"
                f"{r.get('story','No story available.')}\n\n"
                f"_{r.get('champ_delta','')}_")

    results = race.get("Results", [])[:5]
    name    = race.get("raceName", "Last Race")

    def dname(r):
        d = r.get("Driver", {})
        return f"{d.get('givenName','')[:1]}. {d.get('familyName','')}"

    lines = [f"🏁 *{name}*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━"]

    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    for i, r in enumerate(results):
        status = r.get("status","")
        dnf    = status not in ("Finished","+1 Lap","+2 Laps","+3 Laps")
        note   = f" _{status}_" if dnf else ""
        lines.append(f"{medals[i]} *{dname(r)}*{note}")

    # Fastest lap
    for r in race.get("Results", []):
        if r.get("FastestLap", {}).get("rank") == "1":
            fl_t = r["FastestLap"].get("Time", {}).get("time", "")
            lines.append(f"\n⚡ Fastest lap: *{dname(r)}* {fl_t}")
            break

    return "\n".join(lines)


def format_prediction(next_race: dict, mem: dict) -> str:
    if not next_race:
        return "No upcoming race found in the calendar."

    name    = next_race.get("raceName", "Next Race")
    circuit = next_race.get("Circuit", {}).get("circuitName", "")
    date    = next_race.get("date", "TBD")
    rnd     = next_race.get("round", "?")

    # Get last episodes for context
    episodes = mem.get("episodic", [])
    recent   = episodes[-3:] if len(episodes) >= 3 else episodes
    recent_str = " | ".join(
        f"R{r['round']} {r['track']}→{r['winner']}" for r in recent)

    return (
        f"🎯 *{name} Preview*\n"
        f"_Round {rnd} · {circuit} · {date}_\n"
        f"_Developed by Erick Hernandez_\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Ask me: _\"Who will win {name}?\"_ for a full prediction with reasoning,\n"
        f"or _\"/predict full\"_ for the complete analysis including circuit profile, "
        f"championship stakes, and what could upset the favourite.\n\n"
        f"Recent form: _{recent_str}_"
    )

# ═════════════════════════════════════════════════════════════
#  TELEGRAM HANDLERS
# ═════════════════════════════════════════════════════════════
mem      = {}
sessions = {}

async def cmd_reingest(update, ctx, mem_ref=None):
    if not update.effective_user:
        return
    user_id = str(update.effective_user.id)
    if user_id != BOT_OWNER_ID:
        return
    args = ctx.args
    if not args or not args[0].isdigit() or int(args[0]) < 1:
        await update.message.reply_text("Usage: /reingest <round_number>\nExample: /reingest 7")
        return
    rnd = int(args[0])
    ingest_state = load_ingest_state()
    ingest_key = f"r{rnd}"
    if ingest_key in ingest_state:
        del ingest_state[ingest_key]
        save_ingest_state(ingest_state)
    enrich_state = load_enrichment_state()
    enrich_key = f"enriched_r{rnd}"
    if enrich_key in enrich_state:
        del enrich_state[enrich_key]
        save_enrichment_state(enrich_state)
    pred_state = load_predictor_state()
    pred_key = f"predictor_r{rnd}"
    if pred_key in pred_state:
        del pred_state[pred_key]
        save_predictor_state(pred_state)
    result = fetch_race_result(rnd, SEASON)
    if not result:
        await update.message.reply_text(f"⚠️ Cleared state for R{rnd} but Jolpica fetch failed — will retry on next auto-ingest cycle.")
        return
    mem = mem_ref[0]
    episodes = mem.get("episodic", [])
    existing = next((e for e in episodes if e.get("round") == rnd), None)
    if existing:
        existing.update(result)
    else:
        episodes.append(result)
    mem["episodic"] = episodes
    save_f1_memory(mem)
    dnfs = ", ".join(result.get("dnfs", [])) or "none"
    fc = " ".join(result.get("full_classification", [])[:5])
    await update.message.reply_text(f"✅ R{rnd} re-ingested from Jolpica:\nWinner: {result.get('winner','?')}\nP2: {result.get('p2','?')} P3: {result.get('p3','?')}\nDNFs: {dnfs}\nTop 5: {fc}\nEnrichment state cleared — telemetry will re-run on next cycle.\nPredictor state cleared — auto-predictor will re-run on next 30-min cycle.")



async def cmd_debug_context(update, ctx, mem_ref=None):
    if not update.effective_user:
        return
    user_id = str(update.effective_user.id)
    if user_id != BOT_OWNER_ID:
        return
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "Usage: /debug_context <query>\n"
            "Example: /debug_context what happened to antonelli in spain")
        return
    query = " ".join(args)
    mem = mem_ref[0]
    gathered = _gather_context(query, mem)
    report = _format_debug_context_report(query, gathered)
    await update.message.reply_text(report[:4096])


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return

    # Initialize session if new user
    if user_id not in sessions:
        sessions[user_id] = {
            "history":    [],
            "first_seen": datetime.now().isoformat(),
            "stats":      {"total_messages": 0, "favorite_topics": {},
                           "commands_used": {}, "last_active": None},
        }
        save_sessions(sessions)

    try:
        await update.message.reply_text(
            WELCOME, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(re.sub(r"[*_`]", "", WELCOME))


async def handle_timezone_callback(update: Update,
                                    ctx: ContextTypes.DEFAULT_TYPE):
    """Legacy callback handler — kept for safety."""
    query = update.callback_query
    await query.answer()


async def cmd_timezone(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Lets users update their timezone by typing their country."""
    user_id = str(update.effective_user.id)
    _tz_pending[user_id] = True
    await update.message.reply_text(
        "🌍 *Update your timezone*\n\n"
        "In which country are you located? Just type it:\n\n"
        "_Examples: Mexico, Spain, Brazil, Japan, Australia, UK..._",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(
            HELP_TEXT, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(re.sub(r"[*_`]", "", HELP_TEXT))

async def cmd_standings(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return
    await update.message.reply_text("Fetching live standings... ⏳")
    drivers, _ = fetch_standings()
    if not drivers:
        await update.message.reply_text("⚠️ Standings unavailable right now.")
        return
    medals  = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    leader  = float(drivers[0].get("points", 0))
    lines   = ["🏆 *2026 Driver Championship*",
               "_Developed by Erick Hernandez_",
               "━━━━━━━━━━━━━━━━━━━━━━"]
    for i, s in enumerate(drivers[:10]):
        d     = s.get("Driver", {})
        name  = f"{d.get('givenName','')[:1]}. {d.get('familyName','')}"
        pts   = float(s.get("points", 0))
        wins  = s.get("wins", "0")
        gap   = f"  _(-{int(leader-pts)})_" if i > 0 else " 🔥"
        medal = medals[i] if i < len(medals) else f"{i+1}."
        lines.append(f"{medal} *{name}* — {int(pts)}pts  _{wins}W_{gap}")
    try:
        await update.message.reply_text(
            "\n".join(lines), parse_mode=constants.ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(
            re.sub(r"[*_]", "", "\n".join(lines)))


async def cmd_constructors(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return
    await update.message.reply_text("Fetching constructor standings... ⏳")
    _, constructors = fetch_standings()
    if not constructors:
        await update.message.reply_text("⚠️ Constructor data unavailable right now.")
        return
    lines = ["🏗 *2026 Constructor Standings*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━"]
    medals  = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    leader  = float(constructors[0].get("points", 0))
    team_emojis = {
        "mercedes": "⬛", "ferrari": "🔴", "red bull": "🔵",
        "mclaren": "🟡", "aston martin": "🟢", "alpine": "🔵",
        "williams": "⚪", "haas": "⬜", "rb": "🟤",
        "audi": "⬛", "cadillac": "🇺🇸",
    }
    for i, s in enumerate(constructors[:10]):
        name  = s.get("Constructor", {}).get("name", "")
        pts   = float(s.get("points", 0))
        gap   = f"  _(-{int(leader-pts)})_" if i > 0 else " 🔥"
        emoji = next((v for k, v in team_emojis.items()
                      if k in name.lower()), "🏎")
        medal = medals[i] if i < len(medals) else f"{i+1}."
        lines.append(f"{medal} {emoji} *{name}* — {int(pts)}pts{gap}")
    try:
        await update.message.reply_text(
            "\n".join(lines), parse_mode=constants.ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(
            re.sub(r"[*_]", "", "\n".join(lines)))


async def cmd_season(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return
    episodes = mem.get("episodic", [])
    if not episodes:
        await update.message.reply_text("No race results in memory yet.")
        return
    lines = ["📅 *2026 Season Results*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━"]
    for ep in sorted(episodes, key=lambda x: x.get("round", 0)):
        rnd    = ep.get("round", "?")
        track  = ep.get("race_name", ep.get("track", "?"))
        winner = ep.get("winner", "?")
        p2     = ep.get("p2", "")
        p3     = ep.get("p3", "")
        date   = ep.get("date", "")[:10]
        podium = f"{winner} / {p2} / {p3}" if p2 else winner
        lines.append(f"R{rnd} *{track}* — 🥇 {podium}  _{date}_")
    # Add upcoming races
    today = datetime.now().date()
    upcoming = []
    for rnd, name, date_str in [
        (6,"Monaco GP","2026-06-07"),(7,"Spanish GP","2026-06-14"),
        (8,"Austrian GP","2026-06-28"),(9,"British GP","2026-07-05"),
        (10,"Belgian GP","2026-07-19"),(11,"Hungarian GP","2026-07-26"),
        (12,"Dutch GP","2026-08-23"),(13,"Italian GP","2026-09-06"),
    ]:
        race_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if race_date >= today and not any(
                e.get("round") == rnd for e in episodes):
            upcoming.append(f"R{rnd} *{name}* — _{date_str}_ 🔜")
        if len(upcoming) >= 4:
            break
    if upcoming:
        lines.append("")
        lines.append("*Upcoming:*")
        lines.extend(upcoming)
    for part in split_message("\n".join(lines)):
        try:
            await update.message.reply_text(
                part, parse_mode=constants.ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(re.sub(r"[*_]", "", part))


async def cmd_news(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return
    await update.message.reply_text("Checking latest F1 news... 📰")
    # Refresh and get articles directly
    refresh_news_cache()
    articles = get_news_cache()[:10] if get_news_cache() else []
    if not articles:
        await update.message.reply_text(
            "⚠️ Couldn't fetch news right now. Try again in a moment.")
        return
    lines = ["📰 *Latest F1 News*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━", ""]
    for a in articles[:8]:
        title  = a.get("title", "")
        source = a.get("source", "")
        date   = a.get("date", "")[:16]
        lines.append(f"• *{title}*")
        if source:
            lines.append(f"  _{source}  {date}_")
        lines.append("")
    try:
        await update.message.reply_text(
            "\n".join(lines), parse_mode=constants.ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(
            re.sub(r"[*_]", "", "\n".join(lines)))

async def cmd_lastrace(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return
    await update.message.reply_text("Fetching latest race... ⏳")
    episodes = mem.get("episodic", [])
    if not episodes:
        await update.message.reply_text("No race results in memory yet.")
        return
    last = sorted(episodes, key=lambda x: x.get("round", 0))[-1]
    rnd    = last.get("round", "?")
    name   = last.get("race_name", last.get("track", "?"))
    winner = last.get("winner", "?")
    p2     = last.get("p2", "?")
    p3     = last.get("p3", "?")
    pole   = last.get("pole", "")
    fl     = last.get("fastest_lap", "")
    story  = last.get("story", "")
    champ  = last.get("champ_after", "")
    dnfs   = last.get("dnfs", [])

    lines = [f"🏁 *R{rnd} — {name}*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━",
             f"🥇 *{winner}*",
             f"🥈 {p2}",
             f"🥉 {p3}",]
    if pole:
        lines.append(f"⏱ Pole: *{pole}*")
    if fl:
        lines.append(f"💨 Fastest lap: *{fl}*")
    if dnfs:
        lines.append(f"⚠️ DNFs: {', '.join(dnfs[:3])}")
    if champ:
        lines.append(f"\n📊 Championship: {champ}")
    if story:
        lines.append(f"\n_{story}_")

    try:
        await update.message.reply_text(
            "\n".join(lines), parse_mode=constants.ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(re.sub(r"[*_]", "", "\n".join(lines)))


# ═════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════
#  FEATURE 1 — RACE WEEKEND LIVE COMPANION
#  Auto-sends session debriefs after FP1/FP2/FP3/Qualifying/Race
#  without users having to ask. Proactive, not reactive.
# ═════════════════════════════════════════════════════════════

SESSION_DEBRIEF_FILE = Path(__file__).parent / "boxboxai_debriefs.json"


def load_debrief_state() -> dict:
    if SESSION_DEBRIEF_FILE.exists():
        try:
            return json.loads(SESSION_DEBRIEF_FILE.read_text())
        except Exception:
            pass
    return {}


def save_debrief_state(state: dict):
    try:
        SESSION_DEBRIEF_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


async def send_session_debrief(app, sessions: dict, mem: dict,
                                race_name: str, session_name: str,
                                round_num: int):
    """
    Sends a session debrief ONLY when real timing data is available.
    Fetches actual results from OpenF1 first — never invents results.
    If no real data exists yet, stays silent.
    """
    state = load_debrief_state()
    key   = f"r{round_num}_{session_name.replace(' ','_')}"
    if state.get(key):
        return  # already sent

    active_users = get_active_user_ids(sessions)
    if not active_users:
        return

    # ── Step 1: Fetch REAL timing data from OpenF1 ───────────
    real_data = _fetch_session_results_openf1(round_num, session_name)

    # If no real data available yet — stay silent, don't hallucinate
    if not real_data:
        log.info(f"No real data yet for {race_name} {session_name} — skipping debrief")
        return

    # ── Step 2: Also get live search for context ──────────────
    live_ctx = live_search_f1(f"{race_name} {session_name} 2026 results")

    # ── Step 3: Build prompt grounded in actual results ───────
    session_emojis_map = {
        "FP1": "🔧", "FP2": "🔧", "FP3": "🔧",
        "Qualifying": "⏱", "Sprint Qualifying": "⏱",
        "Sprint Race": "🏃", "Race": "🏁",
    }
    emoji = session_emojis_map.get(session_name, "🏎")

    session_prompts = {
        "FP1": (
            f"Write a punchy 3-sentence FP1 debrief for {race_name} "
            f"using ONLY these actual results:\n{real_data}\n"
            f"Additional context: {live_ctx[:300] if live_ctx else 'none'}\n"
            f"Cover: who was fastest and by how much, biggest surprise, "
            f"one thing to watch in FP2. "
            f"ONLY use names and times from the results above. Never invent."
        ),
        "FP2": (
            f"Write a 3-sentence FP2 debrief for {race_name} "
            f"using ONLY these actual results:\n{real_data}\n"
            f"Additional context: {live_ctx[:300] if live_ctx else 'none'}\n"
            f"FP2 is race simulation data. Cover: long run leaders, "
            f"tyre strategy clues, who surprised. "
            f"ONLY use names from the results above."
        ),
        "FP3": (
            f"Write a 2-sentence FP3 debrief for {race_name} "
            f"using ONLY these actual results:\n{real_data}\n"
            f"Who's looking dangerous for qualifying? "
            f"ONLY use names from the results above."
        ),
        "Qualifying": (
            f"Write a 4-sentence qualifying debrief for {race_name} "
            f"using ONLY these actual results:\n{real_data}\n"
            f"Additional context: {live_ctx[:400] if live_ctx else 'none'}\n"
            f"Cover: pole sitter with exact time, biggest Q2 elimination surprise, "
            f"grid order implications for race strategy. "
            f"ONLY use names and times from the results above."
        ),
        "Sprint Qualifying": (
            f"Write a 3-sentence sprint qualifying debrief for {race_name} "
            f"using ONLY these actual results:\n{real_data}\n"
            f"Sprint pole sitter, surprises, implications. "
            f"ONLY use names from the results above."
        ),
        "Sprint Race": (
            f"Write a 3-sentence sprint race debrief for {race_name} "
            f"using ONLY these actual results:\n{real_data}\n"
            f"Winner, key moments, points gained. "
            f"ONLY use names from the results above."
        ),
        "Race": (
            f"Write a 5-sentence race debrief for {race_name} "
            f"using ONLY these actual results:\n{real_data}\n"
            f"Additional context: {live_ctx[:500] if live_ctx else 'none'}\n"
            f"Cover: winner and how they won, decisive moment, biggest story, "
            f"championship impact, next race outlook. "
            f"ONLY use names from the results above."
        ),
    }

    prompt = session_prompts.get(session_name, session_prompts["FP1"])

    try:
        resp = get_client().messages.create(
            model=MODEL, max_tokens=300,
            system=(
                "You are BoxBoxAI, an F1 analyst writing for Telegram. "
                "Rules: use *bold* for emphasis only — NEVER use # or ## "
                "markdown headers, NEVER use --- dividers. Use ONLY the "
                "driver codes, positions, and team names given in the "
                "results above — never invent or substitute drivers, "
                "teams, or positions. If results show finishing order, "
                "that IS the race result — don't confuse it with fastest "
                "lap. Confident, punchy tone, 1-3 emojis total."
            ),
            messages=[{"role": "user", "content": prompt}]
        )
        debrief_text = resp.content[0].text if resp.content else ""
    except Exception as e:
        log.error(f"Session debrief generation failed: {e}")
        return

    if not debrief_text:
        return

    msg = (
        f"{emoji} *{race_name} — {session_name} Debrief*\n\n"
        f"{debrief_text}\n\n"
        f"_Ask me anything about the session_ 💬"
    )

    sent = 0
    for uid in active_users:
        user_prefs = sessions.get(uid, {}).get("notification_prefs", {})
        if not user_prefs.get("session_debriefs", True):
            continue
        try:
            await app.bot.send_message(
                chat_id=uid, text=msg, parse_mode="Markdown")
            sent += 1
            await asyncio.sleep(0.05)
        except Exception:
            pass

    state[key] = datetime.now().isoformat()
    save_debrief_state(state)
    log.info(f"Session debrief sent for {race_name} {session_name} "
             f"to {sent} users (grounded in real data ✅)")


def _fetch_session_results_openf1(round_num: int,
                                   session_name: str) -> str:
    """
    Fetches actual session results from OpenF1.
    Returns formatted string with real P1-P10 + lap times.
    Returns empty string if data not available yet.
    This is the gate — no real data = no debrief sent.
    """
    try:
        # ── Race / Sprint Race: use FINISHING CLASSIFICATION from
        # Jolpica, not "fastest single lap" from OpenF1. These are
        # fundamentally different things — a driver's fastest lap
        # has nothing to do with where they finished, and the
        # previous implementation conflated them, producing
        # debriefs where "P1/P2/P3" were actually fastest-lap
        # rankings, not race results.
        if session_name in ("Race", "Sprint Race"):
            result = fetch_race_result(round_num, SEASON)
            if not result or not result.get("full_classification"):
                return ""

            fc    = result["full_classification"]
            dnfs  = result.get("dnfs", [])
            fl    = result.get("fastest_lap", "")
            fl_t  = result.get("fastest_lap_time", "")
            teams = result.get("teams", {})

            lines = [f"{session_name} RESULTS — {SEASON} (FINISHING ORDER):"]
            for item in fc[:10]:
                # item format: "P1:ANT"
                if ":" not in item:
                    continue
                pos, code = item.split(":", 1)
                team = teams.get(code, "")
                lines.append(f"{pos}: {code}" + (f" ({team})" if team else ""))
            if dnfs:
                lines.append("DNFs: " + ", ".join(dnfs))
            if fl:
                lines.append(f"Fastest lap: {fl} {fl_t}")
            return "\n".join(lines)

        # Map session name to OpenF1 session_name param
        session_map = {
            "FP1":               "Practice 1",
            "FP2":               "Practice 2",
            "FP3":               "Practice 3",
            "Qualifying":        "Qualifying",
            "Sprint Qualifying": "Sprint Qualifying",
            "Sprint Race":       "Sprint",
            "Race":              "Race",
        }
        openf1_session = session_map.get(session_name)
        if not openf1_session:
            return ""

        # Get session key for this round
        sessions_data = fetch_openf1("sessions", {
            "year": SEASON,
            "session_name": openf1_session,
        })
        if not sessions_data:
            return ""

        # Find the session matching this round number.
        # OpenF1 doesn't expose round numbers directly. Matching by
        # list-index (matching[round_num - 1]) assumes OpenF1's
        # session list has no gaps and starts at round 1 — fragile.
        # Instead, fetch this round's actual race weekend date from
        # Jolpica and pick the OpenF1 session closest to it (same
        # race weekend = within ~4 days).
        race_date_str = ""
        try:
            race_info = safe_get(f"{JOLPICA}/{SEASON}/{round_num}.json")
            races = race_info.get("MRData",{}).get("RaceTable",{}).get("Races",[]) \
                if race_info else []
            if races:
                race_date_str = races[0].get("date","")
        except Exception:
            pass

        if race_date_str:
            try:
                race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
                def _date_diff(s):
                    try:
                        sd = datetime.fromisoformat(
                            s.get("date_start","").replace("Z","+00:00"))
                        return abs((sd.replace(tzinfo=None) - race_date).days)
                    except Exception:
                        return 999
                candidates = [s for s in sessions_data
                              if s.get("session_name") == openf1_session]
                candidates = [s for s in candidates if _date_diff(s) <= 4]
                if not candidates:
                    return ""
                session = min(candidates, key=_date_diff)
            except Exception:
                return ""
        else:
            # Fallback to old index-based matching if we couldn't
            # get the race date (better than nothing)
            matching = sorted(
                [s for s in sessions_data
                 if s.get("session_name") == openf1_session],
                key=lambda x: x.get("date_start", ""))
            if not matching or round_num > len(matching):
                return ""
            session = matching[round_num - 1]

        sk = session.get("session_key")
        if not sk:
            return ""

        # Check if session is actually finished
        date_end = session.get("date_end", "")
        if date_end:
            try:
                end_dt = datetime.fromisoformat(
                    date_end.replace("Z","+00:00"))
                if end_dt.timestamp() > datetime.now().timestamp():
                    return ""  # session not finished yet
            except Exception:
                pass

        # Fetch lap times
        laps = fetch_openf1("laps", {"session_key": sk})
        if not laps:
            return ""

        # Get driver map
        drivers_data = fetch_openf1("drivers", {"session_key": sk})
        num_to_code = {}
        num_to_name = {}
        num_to_team = {}
        if drivers_data:
            for d in drivers_data:
                num  = str(d.get("driver_number",""))
                code = d.get("name_acronym","")
                name = d.get("full_name","")
                team = d.get("team_name","")
                if num and code:
                    num_to_code[num] = code
                    num_to_name[num] = name
                    num_to_team[num] = team

        # Find best lap per driver
        driver_best: dict = {}
        for lap in laps:
            dur = lap.get("lap_duration")
            num = str(lap.get("driver_number",""))
            if dur and num:
                try:
                    dur_f = float(dur)
                    if dur_f > 0:
                        if num not in driver_best or \
                                dur_f < driver_best[num]:
                            driver_best[num] = dur_f
                except Exception:
                    pass

        if not driver_best:
            return ""

        # Sort and format top 10
        sorted_drivers = sorted(
            driver_best.items(), key=lambda x: x[1])
        ref = sorted_drivers[0][1]

        lines = [f"{session_name} RESULTS — {SEASON}:"]
        for i, (num, t) in enumerate(sorted_drivers[:10], 1):
            code = num_to_code.get(num, f"#{num}")
            team = num_to_team.get(num, "")
            mins = int(t // 60)
            secs = t % 60
            if i == 1:
                time_str = f"{mins}:{secs:06.3f}"
                delta_str = "fastest"
            else:
                delta = t - ref
                time_str = f"{mins}:{secs:06.3f}"
                delta_str = f"+{delta:.3f}s"
            lines.append(
                f"P{i:2d}: {code:4s} ({team[:12]}) "
                f"{time_str} {delta_str}")

        return "\n".join(lines)

    except Exception as e:
        log.debug(f"OpenF1 session results failed: {e}")
        return ""


async def check_and_send_session_debriefs(app, sessions: dict, mem: dict):
    """
    Checks if any session ended in the last 2 hours and sends a debrief.
    Runs every 30 minutes via notification_loop.
    """
    now_utc = datetime.utcnow()

    for entry in SESSION_SCHEDULE_2026:
        rnd, race_name, circuit_tz, session_list = entry
        race_date_str = RACE_DATES_2026.get(rnd)
        if not race_date_str:
            continue

        race_date = datetime.strptime(race_date_str, "%Y-%m-%d").date()

        for session_name, weekday_offset, utc_h, utc_m in session_list:
            days_before  = 6 - weekday_offset
            session_date = race_date - timedelta(days=days_before)

            # Estimated session end (start + ~90 min for race, ~60 for others)
            duration_mins = 120 if session_name == "Race" else 75
            session_start = datetime(session_date.year, session_date.month,
                                     session_date.day, utc_h, utc_m)
            session_end   = session_start + timedelta(minutes=duration_mins)

            # Check if session ended 0-120 minutes ago
            mins_since_end = (now_utc - session_end).total_seconds() / 60
            if not (0 <= mins_since_end <= 120):
                continue

            await send_session_debrief(
                app, sessions, mem, race_name, session_name, rnd)


# ═════════════════════════════════════════════════════════════
#  FEATURE 3 — DRIVER/TEAM DEEP DIVES
#  /driver <name> — full profile from memory + live Jolpica data
#  /team <name>   — team profile with both drivers
# ═════════════════════════════════════════════════════════════





# ═════════════════════════════════════════════════════════════
#  FEATURE 5 — CHAMPIONSHIP SCENARIOS
#  "What does X need to win the championship?"
#  Does the live math from Jolpica standings
# ═════════════════════════════════════════════════════════════



# ═════════════════════════════════════════════════════════════
#  FEATURE 6 — VOICE OF RUTH (WEEKLY DEBRIEF UPGRADE)
#  Consistent analytical voice, structured format every Monday
# ═════════════════════════════════════════════════════════════

RUTH_DEBRIEF_PROMPT = """You are writing the weekly F1 debrief for BoxBoxAI in the style of 
The Race Strategy Society newsletter. Analytical, opinionated, precise. 
Never vague. Always has a clear point of view.

STRICT FORMAT — follow this every week, no exceptions:

🏁 [RACE NAME] — [ONE LINE VERDICT]
[Blank line]
THE DECISIVE MOMENT
[2 sentences on the exact moment that won or lost the race. Specific lap, specific action.]
[Blank line]
THE STRATEGY STORY
[2 sentences on what the pit wall got right or wrong. Be opinionated — someone got it wrong.]
[Blank line]
WHO IMPRESSED
[1 sentence. One driver who did more than expected.]
[Blank line]
WHO DISAPPOINTED
[1 sentence. One driver or team who underdelivered.]
[Blank line]
CHAMPIONSHIP PICTURE
[1-2 sentences. What this race changed in the title fight.]
[Blank line]
WATCH FOR NEXT RACE
[1 sentence. The most important thing to monitor heading into the next round.]

Keep each section tight. No filler. No "great race by". Just analysis."""


# ═════════════════════════════════════════════════════════════
#  FEATURE 7 — PUSH NOTIFICATION OPT-IN
#  Users control what they receive
# ═════════════════════════════════════════════════════════════

NOTIFICATION_OPTIONS = {
    "all":          "Everything — sessions, race results, debriefs 🏎",
    "race_only":    "Race start + result only 🏁",
    "quali_race":   "Qualifying + Race only ⏱🏁",
    "results_only": "Results only (no session alerts) 📊",
    "off":          "No notifications 🔕",
}

NOTIFICATION_CONFIG = {
    "all": {
        "session_notifications": True,
        "session_debriefs":      True,
        "race_results":          True,
        "weekly_debrief":        True,
    },
    "race_only": {
        "session_notifications": False,
        "session_debriefs":      False,
        "race_results":          True,
        "weekly_debrief":        True,
    },
    "quali_race": {
        "session_notifications": True,  # only for Quali + Race sessions
        "session_debriefs":      True,  # only for Quali + Race
        "race_results":          True,
        "weekly_debrief":        True,
        "quali_race_only":       True,  # flag to filter
    },
    "results_only": {
        "session_notifications": False,
        "session_debriefs":      False,
        "race_results":          True,
        "weekly_debrief":        True,
    },
    "off": {
        "session_notifications": False,
        "session_debriefs":      False,
        "race_results":          False,
        "weekly_debrief":        False,
    },
}


def get_user_notif_prefs(user_data: dict) -> dict:
    """Returns user's notification config. Defaults to 'all'."""
    pref_key = user_data.get("notification_pref", "all")
    return NOTIFICATION_CONFIG.get(pref_key, NOTIFICATION_CONFIG["all"])


async def cmd_notifications(update: Update,
                             ctx: ContextTypes.DEFAULT_TYPE):
    """Lets users control their notification preferences."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    user_id   = str(update.effective_user.id)
    user_data = sessions.get(user_id, {})
    current   = user_data.get("notification_pref", "all")

    keyboard = []
    for key, label in NOTIFICATION_OPTIONS.items():
        tick = "✅ " if key == current else ""
        keyboard.append([InlineKeyboardButton(
            f"{tick}{label}",
            callback_data=f"notif:{key}"
        )])

    await update.message.reply_text(
        "🔔 *Notification Preferences*\n\n"
        "Choose what you want to receive from BoxBoxAI:",
        parse_mode=constants.ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def handle_notification_callback(update: Update,
                                        ctx: ContextTypes.DEFAULT_TYPE):
    """Handles notification preference selection."""
    query   = update.callback_query
    user_id = str(query.from_user.id)

    if not query.data.startswith("notif:"):
        return

    await query.answer()
    pref_key = query.data.split(":", 1)[1]

    if pref_key not in NOTIFICATION_OPTIONS:
        return

    if user_id not in sessions:
        sessions[user_id] = {"history": [], "first_seen": datetime.now().isoformat(),
                              "stats": {}}

    sessions[user_id]["notification_pref"]  = pref_key
    sessions[user_id]["notification_prefs"] = NOTIFICATION_CONFIG[pref_key]
    save_sessions(sessions)

    label = NOTIFICATION_OPTIONS[pref_key]
    await query.edit_message_text(
        f"✅ Notifications set to:\n*{label}*\n\n"
        f"You can change this anytime with /notifications",
        parse_mode=constants.ParseMode.MARKDOWN
    )


# ═════════════════════════════════════════════════════════════
#  FEATURE 8 — RIVAL/FAN TRACKING
#  User says "I'm a Ferrari fan" once → personalized forever
# ═════════════════════════════════════════════════════════════



#  ─── WIRE FAN DETECTION INTO handle_message ─────────────────
#  (fan detection runs inside handle_message, see below)




async def cmd_predict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)

    try:
        next_race = fetch_next_race()
        if not next_race:
            today = datetime.now().date()
            for rnd, name, date_str in [
                (7,"Spanish GP","2026-06-14"),
                (8,"Austrian GP","2026-06-28"),
                (9,"British GP","2026-07-05"),
                (10,"Belgian GP","2026-07-19"),
                (11,"Hungarian GP","2026-07-26"),
                (12,"Dutch GP","2026-08-23"),
                (13,"Italian GP","2026-09-06"),
                (14,"Singapore GP","2026-09-20"),
                (15,"Azerbaijan GP","2026-09-27"),
                (16,"US GP","2026-10-18"),
                (17,"Mexico City GP","2026-10-25"),
                (18,"São Paulo GP","2026-11-08"),
                (19,"Las Vegas GP","2026-11-21"),
                (20,"Qatar GP","2026-11-29"),
                (21,"Abu Dhabi GP","2026-12-06"),
            ]:
                if datetime.strptime(date_str,"%Y-%m-%d").date() >= today:
                    next_race = {"raceName": name, "round": str(rnd),
                                 "date": date_str}
                    break

        race_name = next_race.get("raceName","next race") if next_race \
                    else "next race"

        await update.message.reply_text(
            f"🧠 Running full prediction for *{race_name}*...",
            parse_mode=constants.ParseMode.MARKDOWN)

        try:
            expected_rnd = int(next_race.get("round", 0)) if next_race else None
            pred_block, pred_rows = get_predictor_context(expected_rnd)
            has_pred = bool(pred_block and pred_rows)
        except Exception:
            has_pred  = False
            pred_rows = []
            pred_block = ""

        history   = get_user_history(sessions, user_id)
        user_data = sessions.get(user_id, {})

        if has_pred:
            is_preview = _is_pre_qualifying_csv(pred_rows)
            if is_preview:
                prompt = (
                    f"You have a PRE-QUALIFYING PREVIEW from f1_2026_predictor.py v7.0 below. "
                    f"IMPORTANT: qualifying for {race_name} has NOT happened yet. "
                    f"These probabilities are Bayesian priors from season form and circuit "
                    f"history ONLY — not real session data.\n\n"
                    f"{pred_block}\n\n"
                    f"For the {race_name}, give an honest pre-qualifying outlook:\n"
                    f"1. FORM FAVOURITE — who the model rates highest and why "
                    f"(cite their actual recent_form and circuit_score values shown above)\n"
                    f"2. WHY — reference ONLY factors in the data: season points, recent form, "
                    f"circuit history, tyre scores. Do NOT invent qualifying gaps, practice "
                    f"lap times, or technical issues that are not in the data above\n"
                    f"3. TOP 5 FORM ORDER — from the model, clearly labelled as "
                    f"pre-qualifying speculation (qualifying could completely change this)\n"
                    f"4. WHAT TO WATCH IN QUALIFYING — which rivals could shake up the order\n"
                    f"5. CHAMPIONSHIP STAKES — what this weekend means for the WDC\n\n"
                    f"Be upfront that qualifying hasn't happened. The Pts column shows current "
                    f"championship standings — use those exact numbers, not anything else.\n\n"
                    f"Telegram formatting: *bold* only — NEVER use # or ## markdown headers."
                )
            else:
                prompt = (
                    f"You have the ACTUAL OUTPUT of f1_2026_predictor.py v7.0 below. "
                    f"Real Monte Carlo data — 10,000 runs. Not a guess.\n\n"
                    f"{pred_block}\n\n"
                    f"For the {race_name}, give a complete race prediction:\n"
                    f"1. WINNER — model's top pick with exact win% from the CSV\n"
                    f"2. WHY — which features drive it (quali position, recent form, "
                    f"circuit score, compound score — cite actual values)\n"
                    f"3. TOP 5 — predicted finishing order with win% for each\n"
                    f"4. WHAT COULD UPSET IT — weather, safety car, mechanical risk\n"
                    f"5. CHAMPIONSHIP STAKES — what this race means for the WDC\n"
                    f"6. CONFIDENCE — 0-100 and why\n\n"
                    f"Use the actual numbers. Be specific and opinionated. "
                    f"The Pts column shows current championship standings — use those exact numbers.\n\n"
                    f"Telegram formatting: *bold* only — NEVER use # or ## markdown headers."
                )
        else:
            # No predictor CSV — ground the prompt in real grid data if available
            actual_grid = get_actual_grid_for_prediction()
            if actual_grid:
                prompt = (
                    f"{actual_grid}\n\n"
                    f"For the {race_name}, give a race prediction based on "
                    f"the ACTUAL qualifying grid above:\n"
                    f"1. WINNER — pick from the grid, reasoning based on "
                    f"starting position and recent form\n"
                    f"2. TOP 5 — based on the actual grid order plus your "
                    f"analysis of who can overtake/defend\n"
                    f"3. KEY FACTORS — what will decide the race\n"
                    f"4. CHAMPIONSHIP IMPLICATIONS\n"
                    f"5. CONFIDENCE — 0-100\n\n"
                    f"Use ONLY the drivers and positions listed in the grid above. "
                    f"Do not invent grid positions.\n\n"
                    f"Telegram formatting: *bold* only — NEVER use # or ## markdown headers."
                )
            else:
                prompt = (
                    f"Qualifying for {race_name} hasn't happened yet, and I don't "
                    f"have predictor data. Give a brief pre-qualifying preview: "
                    f"championship context, what to watch for, key storylines. "
                    f"Do NOT predict a specific finishing order or grid positions — "
                    f"those don't exist yet. Keep it to 3-4 sentences.\n\n"
                    f"Telegram formatting: *bold* only — NEVER use # or ## markdown headers."
                )

        reply = ask_claude(prompt, history, mem, user_data)
        update_user_history(sessions, user_id, "user",
                            f"Race prediction {race_name}")
        update_user_history(sessions, user_id, "assistant", reply)
        save_sessions(sessions)

        for part in split_message(reply):
            try:
                await update.message.reply_text(
                    part, parse_mode=constants.ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(re.sub(r"[*_`]","",part))

    except Exception as e:
        log.error(f"cmd_predict error: {type(e).__name__}: {e}", exc_info=True)
        await update.message.reply_text(
            f"⚠️ Error: {type(e).__name__}: {str(e)[:100]}")




async def cmd_winner(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Race winner — top 3 from predictor + narrative explanation."""
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING)

    try:
        next_race = fetch_next_race()
        if not next_race:
            today = datetime.now().date()
            for rnd, rname, date_str in [
                (7,"Spanish GP","2026-06-14"),
                (8,"Austrian GP","2026-06-28"),
                (9,"British GP","2026-07-05"),
                (10,"Belgian GP","2026-07-19"),
                (11,"Hungarian GP","2026-07-26"),
                (12,"Dutch GP","2026-08-23"),
                (13,"Italian GP","2026-09-06"),
                (14,"Singapore GP","2026-09-20"),
                (15,"Azerbaijan GP","2026-09-27"),
                (16,"US GP","2026-10-18"),
                (17,"Mexico City GP","2026-10-25"),
                (18,"São Paulo GP","2026-11-08"),
                (19,"Las Vegas GP","2026-11-21"),
                (20,"Qatar GP","2026-11-29"),
                (21,"Abu Dhabi GP","2026-12-06"),
                (10,"Belgian GP","2026-07-19"),
                (11,"Hungarian GP","2026-07-26"),
            ]:
                if datetime.strptime(date_str,"%Y-%m-%d").date() >= today:
                    next_race = {"raceName": rname, "round": str(rnd),
                                 "date": date_str}
                    break

        race_name = next_race.get("raceName","next race") if next_race \
                    else "next race"
        history   = get_user_history(sessions, user_id)
        user_data = sessions.get(user_id, {})

        try:
            expected_rnd = int(next_race.get("round", 0)) if next_race else None
            pred_block, pred_rows = get_predictor_context(expected_rnd)
            has_pred = bool(pred_block and pred_rows)
        except Exception:
            has_pred  = False
            pred_rows = []

        top3_block = ""
        narrative  = ""

        if has_pred:
            # ── Top 3 block ───────────────────────────────
            medals = ["🥇","🥈","🥉"]
            top3_lines = []
            for i, r in enumerate(pred_rows[:3]):
                dname  = r.get("FullName", r.get("code","?"))
                team   = r.get("TeamName","")
                try:    ws = f"{float(r.get('win_mc_pct','?')):.1f}%"
                except: ws = "?"
                try:    ps = f"{float(r.get('podium_mc_pct','?')):.1f}%"
                except: ps = "?"
                top3_lines.append(
                    f"{medals[i]} *{dname}* ({team})\n"
                    f"   {ws} win · {ps} podium")
            top3_block = "\n".join(top3_lines)

            # ── Narrative from top pick ───────────────────
            r    = pred_rows[0]
            name = r.get("FullName", r.get("code","?"))
            p2   = pred_rows[1].get("FullName","?") if len(pred_rows)>1 else "?"
            p3   = pred_rows[2].get("FullName","?") if len(pred_rows)>2 else "?"
            try:    win_str = f"{float(r.get('win_mc_pct','?')):.1f}%"
            except: win_str = "?"
            try:    pod_str = f"{float(r.get('podium_mc_pct','?')):.1f}%"
            except: pod_str = "?"
            quali = r.get("quali_pos_next","")
            form  = r.get("recent_form","")
            cscor = r.get("circuit_score","")
            mech  = r.get("mechanical_risk","")
            try:    qstr = f"P{int(float(quali))}" if quali not in ("","nan","NaN") else ""
            except: qstr = ""
            try:    mstr = f"{float(mech)*100:.1f}% mech risk" \
                           if mech not in ("","nan","NaN") else ""
            except: mstr = ""

            prompt = (
                f"f1_2026_predictor.py says {name} wins {race_name} "
                f"with {win_str} probability ({pod_str} podium). "
                f"P2: {p2}, P3: {p3}. "
                f"Key factors: quali={qstr}, recent_form={form}, "
                f"circuit_score={cscor}. {mstr}. "
                f"In 2-3 sentences: explain the main reason why {name} wins, "
                f"and name ONE driver who could realistically upset it. "
                f"Use the exact {win_str} number. Be direct and confident."
            )
            narrative = ask_claude(prompt, history, mem, user_data)
            try:
                save_prediction(race_name, r.get("code","?"))
            except Exception:
                pass

        if not narrative:
            actual_grid = get_actual_grid_for_prediction()
            if actual_grid:
                narrative = ask_claude(
                    f"{actual_grid}\n\n"
                    f"Based on this ACTUAL qualifying grid for the {race_name}, "
                    f"who do you think wins? 2-3 sentences — winner from the grid, "
                    f"main reason why, one driver who could upset it. "
                    f"Use ONLY drivers/positions listed above.",
                    history, mem, user_data)
            else:
                narrative = ask_claude(
                    f"Qualifying for the {race_name} hasn't happened yet and "
                    f"I don't have predictor data. Give a brief 2-3 sentence "
                    f"pre-qualifying take: championship context and what to watch. "
                    f"Do NOT name a predicted winner or grid position — "
                    f"qualifying hasn't happened.",
                    history, mem, user_data)
            for code in ["ANT","RUS","HAM","LEC","VER",
                         "NOR","PIA","ALO","SAI","GAS"]:
                if code in narrative.upper()[:100]:
                    try: save_prediction(race_name, code)
                    except Exception: pass
                    break

        if not narrative:
            narrative = "⚠️ Having trouble right now. Try again in a moment."

        # ── Combine top 3 + narrative ─────────────────────
        if top3_block:
            full_reply = (
                f"🎯 *{race_name}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{top3_block}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{narrative}"
            )
        else:
            full_reply = narrative

        update_user_history(sessions, user_id, "user",
                            f"/winner {race_name}")
        update_user_history(sessions, user_id, "assistant", full_reply)
        save_sessions(sessions)

        for part in split_message(full_reply):
            try:
                await update.message.reply_text(
                    part, parse_mode=constants.ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(
                    re.sub(r"[*_`]","", part))

    except Exception as e:
        log.error(f"cmd_winner error: {type(e).__name__}: {e}",
                  exc_info=True)
        await update.message.reply_text(
            f"⚠️ Error: {type(e).__name__}: {str(e)[:100]}")



async def cmd_compare(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Compares two drivers across stats, career, or 2026 season."""
    user_id = str(update.effective_user.id)
    args    = " ".join(ctx.args).strip() if ctx.args else ""

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING)

    if not args:
        await update.message.reply_text(
            "🏎 *Who do you want to compare?*\n\n"
            "Usage: `/compare Antonelli vs Verstappen`\n"
            "or: `/compare Hamilton Leclerc`\n\n"
            "I can compare:\n"
            "• 2026 season stats\n"
            "• Rookie seasons\n"
            "• Career achievements\n"
            "• Head-to-head at specific circuits",
            parse_mode=constants.ParseMode.MARKDOWN)
        return

    history   = get_user_history(sessions, user_id)
    user_data = sessions.get(user_id, {})

    prompt = (
        f"Compare these F1 drivers: {args}\n\n"
        f"Give a detailed but punchy comparison covering:\n"
        f"1. 2026 season stats (wins, points, poles if relevant)\n"
        f"2. Career achievements and titles\n"
        f"3. Driving style differences\n"
        f"4. Historical context (rookie seasons, peak years)\n"
        f"5. Your verdict — who comes out on top and why\n\n"
        f"Use real numbers from your memory. Be specific, not vague."
    )

    reply = ask_claude(prompt, history, mem, user_data)

    update_user_history(sessions, user_id, "user", f"/compare {args}")
    update_user_history(sessions, user_id, "assistant", reply)
    save_sessions(sessions)

    for part in split_message(reply):
        try:
            await update.message.reply_text(
                part, parse_mode=constants.ParseMode.MARKDOWN)
        except Exception:
            clean = re.sub(r"[*_`\[\]]", "", part)
            await update.message.reply_text(clean)


async def cmd_debate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Triggers a fun F1 debate topic."""
    import random
    debates = [
        "Is Antonelli already better than Russell, or is it just the car?",
        "Verstappen vs Hamilton — who is the greater driver of the hybrid era?",
        "Is Monaco still relevant on the F1 calendar or should it be dropped?",
        "McLaren vs Ferrari — who has the better driver lineup for the future?",
        "Is the 2026 car the most exciting F1 car in a decade?",
        "Should F1 have more sprint races or fewer?",
        "Leclerc at Ferrari — legacy destroyed or still time to turn it around?",
        "Is Norris underperforming his car or is McLaren just that unreliable?",
    ]
    topic   = random.choice(debates)
    user_id = str(update.effective_user.id)
    history = get_user_history(sessions, user_id)

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING)

    reply = ask_claude(
        f"Hot debate topic: {topic} Give your honest opinion — pick a side, "
        f"back it up with evidence from the 2026 season, and make it spicy.",
        history, mem
    )
    update_user_history(sessions, user_id, "user", f"debate: {topic}")
    update_user_history(sessions, user_id, "assistant", reply)
    save_sessions(sessions)

    await update.message.reply_text(
        f"🔥 *Debate time!*\n\n_{topic}_",
        parse_mode=constants.ParseMode.MARKDOWN)
    for part in split_message(reply):
        await update.message.reply_text(
            part, parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_hottake(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Generates a spicy F1 hot take."""
    user_id = str(update.effective_user.id)
    history = get_user_history(sessions, user_id)

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING)

    reply = ask_claude(
        "Give me one genuinely spicy F1 hot take based on the 2026 season so far. "
        "Something that would start arguments. Be bold, back it up with real data, "
        "and commit to the take. One paragraph max.",
        history, mem
    )
    update_user_history(sessions, user_id, "user", "/hottake")
    update_user_history(sessions, user_id, "assistant", reply)
    save_sessions(sessions)

    await update.message.reply_text(
        "🌶️ *Hot take incoming...*",
        parse_mode=constants.ParseMode.MARKDOWN)
    for part in split_message(reply):
        await update.message.reply_text(
            part, parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_wouldyourather(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """F1 strategy would-you-rather dilemma."""
    import random
    dilemmas = [
        "Would you rather: start P1 on hard tyres or P3 on softs at Monaco?",
        "Would you rather: be Antonelli with 4 wins but Russell as teammate, or Verstappen with no wins but full team support?",
        "Would you rather: have the fastest car with unreliable engines, or a slower car that always finishes?",
        "Would you rather: win Monaco or win Monza?",
        "Would you rather: be Hamilton at Ferrari fighting for P3 or Antonelli dominating in silver?",
        "Would you rather: a race with 5 safety cars or a race with a red flag restart?",
        "Would you rather: qualify P1 and start on mandatory hards, or qualify P5 on free tyre choice?",
    ]
    dilemma = random.choice(dilemmas)
    user_id = str(update.effective_user.id)
    history = get_user_history(sessions, user_id)

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING)

    reply = ask_claude(
        f"F1 Would You Rather: {dilemma} Pick one, explain your reasoning using "
        f"real examples from the 2026 season or F1 history. Make it fun.",
        history, mem
    )
    update_user_history(sessions, user_id, "user", f"wouldyourather: {dilemma}")
    update_user_history(sessions, user_id, "assistant", reply)
    save_sessions(sessions)

    await update.message.reply_text(
        f"🤔 *Would You Rather...*\n\n_{dilemma}_",
        parse_mode=constants.ParseMode.MARKDOWN)
    for part in split_message(reply):
        await update.message.reply_text(
            part, parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_schedule(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shows upcoming race schedule with session times in user's timezone."""
    user_id   = str(update.effective_user.id)
    user_data = sessions.get(user_id, {})
    tz_offset = get_user_tz_offset(user_data)
    tz_label  = user_data.get("tz_label", "UTC-6")

    today = datetime.now().date()
    lines = [f"📅 *2026 F1 Schedule*",
             f"_Times shown in your timezone: {tz_label}_",
             "━━━━━━━━━━━━━━━━━━━━━━", ""]

    shown = 0
    for entry in SESSION_SCHEDULE_2026:
        rnd, race_name, circuit_tz, session_list = entry
        race_date_str = RACE_DATES_2026.get(rnd)
        if not race_date_str:
            continue
        race_date  = datetime.strptime(race_date_str, "%Y-%m-%d").date()
        days_until = (race_date - today).days

        if days_until < -1:
            continue  # past
        if shown >= 3:
            break     # show next 3 weekends only

        lines.append(f"🏎 *R{rnd} — {race_name}*  _{race_date_str}_")

        for session_name, weekday_offset, utc_h, utc_m in session_list:
            days_before  = 6 - weekday_offset
            session_date = race_date - timedelta(days=days_before)
            user_mins    = (utc_h * 60 + utc_m + tz_offset * 60) % (24 * 60)
            user_h       = user_mins // 60
            user_m       = user_mins % 60
            emoji        = SESSION_EMOJIS.get(session_name, "🏁")
            lines.append(
                f"  {emoji} {session_name}: *{user_h:02d}:{user_m:02d}*"
                f"  _{session_date.strftime('%b %d')}_")

        lines.append("")
        shown += 1

    if shown == 0:
        lines.append("Season complete! 🏆")

    lines.append(f"_Set your timezone: /timezone_")

    for part in split_message("\n".join(lines)):
        try:
            await update.message.reply_text(
                part, parse_mode=constants.ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(re.sub(r"[*_]", "", part))


async def cmd_mypredictions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shows the bot's prediction accuracy for the season."""
    preds = load_predictions()
    if not preds:
        await update.message.reply_text(
            "🎯 No predictions tracked yet this season.\n\n"
            "Use /winner before each race and I'll track my accuracy!")
        return

    resolved = [p for p in preds if p.get("correct") is not None]
    pending  = [p for p in preds if p.get("correct") is None]

    lines = ["🎯 *BoxBoxAI Prediction Accuracy*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━", ""]

    if resolved:
        correct = sum(1 for p in resolved if p["correct"])
        total   = len(resolved)
        pct     = round(correct / total * 100)
        bar     = "🟢" * correct + "🔴" * (total - correct)
        lines.append(f"📊 *Season record: {correct}/{total} ({pct}%)*")
        lines.append(bar)
        lines.append("")
        lines.append("*Results:*")
        for p in resolved[-5:]:
            icon = "✅" if p["correct"] else "❌"
            lines.append(
                f"{icon} {p['race']}: predicted *{p['predicted']}*"
                f", actual *{p.get('actual','?')}*")
    else:
        lines.append("No resolved predictions yet — check back after the next race!")

    if pending:
        lines.append("")
        lines.append(f"_⏳ {len(pending)} prediction(s) awaiting race result_")
        for p in pending:
            lines.append(f"  • {p['race']}: *{p['predicted']}* (pending)")

    try:
        await update.message.reply_text(
            "\n".join(lines), parse_mode=constants.ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(re.sub(r"[*_]", "", "\n".join(lines)))


async def cmd_live(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shows live session data if a session is currently running."""
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING)
    live = get_live_session_context()
    if live:
        await update.message.reply_text(
            f"🔴 *LIVE*\n\n{live}", parse_mode=constants.ParseMode.MARKDOWN)
    else:
        next_race = fetch_next_race()
        if next_race:
            name = next_race.get("raceName","")
            date = next_race.get("date","")
            await update.message.reply_text(
                f"No session live right now. 🏁\n\nNext up: *{name}* on {date}",
                parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(
                "No session live right now. 🏁")


async def cmd_mystats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shows personalised usage stats for the user."""
    user    = update.effective_user
    user_id = str(user.id)
    name    = user.first_name or "F1 Fan"
    track_command(sessions, user_id, "mystats")
    text = build_user_stats_text(sessions.get(user_id, {}), name)
    await update.message.reply_text(
        text, parse_mode=constants.ParseMode.MARKDOWN)


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handles all free-text messages — routes to Claude."""
    user    = update.effective_user
    user_id = str(user.id)
    text    = update.message.text.strip()

    if not text:
        return

    # ── 1. Sanitize input ─────────────────────────────────
    text = sanitize_message(text)
    if not text:
        return

    # ── 2. Rate limiting ──────────────────────────────────
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        log_security_event(user_id, user.first_name or "", "RATE_LIMIT", text)
        await update.message.reply_text(rate_msg)
        return

    # ── 3. Prompt injection detection ────────────────────
    if check_injection(text):
        log_security_event(user_id, user.first_name or "", "INJECTION", text)
        if user_id not in _abuse_strikes:
            _abuse_strikes[user_id] = {"strikes": 0, "banned_until": 0}
        _abuse_strikes[user_id]["strikes"] += 1
        reply = get_injection_response(text)
        await update.message.reply_text(reply)
        return

    # ── 4b. Fan/rival declaration detection ───────────────
    fan_team, fan_driver = detect_fan_declaration(text)
    if fan_team or fan_driver:
        if user_id not in sessions:
            sessions[user_id] = {
                "history":    [],
                "first_seen": datetime.now().isoformat(),
                "stats":      {"total_messages": 0, "favorite_topics": {},
                               "commands_used": {}, "last_active": None},
            }
        if fan_team:
            sessions[user_id]["fan_team"] = fan_team
        if fan_driver:
            sessions[user_id]["fan_driver"] = fan_driver
        save_sessions(sessions)

    # ── 4c. Session results intercept ─────────────────────
    # If user asks PURELY for session timing/results, fetch from OpenF1
    # directly and format it ourselves — bypass Claude to prevent
    # hallucination. But "why/how/explain" questions need full analysis
    # (FIA docs, race replay) — let those go through ask_claude normally,
    # which still has OpenF1 session data available via get_session_context.
    _analysis_words = [
        "why", "how", "what happened", "what caused", "explain",
        "walk me through", "incident", "crash", "collision", "penalty",
        "penalised", "penalized", "disqualif", "investigation",
        "por qué", "por que", "cómo", "como", "qué pasó", "que paso",
        "explica", "explícame",
    ]
    is_pure_timing_request = (
        _is_live_session_question(text)
        and not any(w in text.lower() for w in _analysis_words)
    )

    if is_pure_timing_request:
        await ctx.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=constants.ChatAction.TYPING)
        real_data = get_session_context(text, live_search_fn=live_search_f1)
        if real_data:
            # We have real timing data — send it directly
            # Then ask Claude to add 2-sentence commentary only
            history   = get_user_history(sessions, user_id)
            user_data = sessions.get(user_id, {})
            prompt = (
                f"Here is the OFFICIAL timing data from OpenF1:\n\n"
                f"{real_data}\n\n"
                f"Present this cleanly for Telegram. Show the classification "
                f"exactly as given. Then add 2 sentences of sharp analysis: "
                f"who impressed, one thing to watch next. "
                f"Do NOT change any times or positions. Use *bold* for names."
            )
            reply = ask_claude(prompt, history, mem, user_data)
            update_user_history(sessions, user_id, "user", text)
            update_user_history(sessions, user_id, "assistant", reply)
            save_sessions(sessions)
            for part in split_message(reply):
                try:
                    await update.message.reply_text(
                        part, parse_mode=constants.ParseMode.MARKDOWN)
                except Exception:
                    await update.message.reply_text(
                        re.sub(r"[*_`]", "", part))
            return
        # No real data from OpenF1 — tell user honestly, no Claude involved
        current = fetch_current_race()
        race    = current.get("raceName","") if current else "current race"
        await update.message.reply_text(
            f"I don't have the timing data for that session yet — "
            f"OpenF1 usually updates within 10-15 minutes of a session ending. "
            f"Try again shortly 🏎")
        return

    # ── 5. Off-topic guardrail ────────────────────────────
    if is_off_topic(text):
        reply = get_off_topic_response(text)
        update_user_history(sessions, user_id, "user", text)
        update_user_history(sessions, user_id, "assistant", reply)
        save_sessions(sessions)
        await update.message.reply_text(reply)
        return

    log.info(f"Message from {user.first_name} ({user_id}): {text[:60]}")

    # ── 5. Show typing + get response ────────────────────
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING
    )

    try:
        history   = get_user_history(sessions, user_id)
        user_data = sessions.get(user_id, {})

        # ask_claude triggers _get_circuit_zone_data which fetches the PDF
        # and saves the circuit map image as a side effect.  The image check
        # must happen AFTER ask_claude so the file is on disk in time.
        reply     = ask_claude(text, history, mem, user_data)

        update_user_history(sessions, user_id, "user", text)
        update_user_history(sessions, user_id, "assistant", reply)
        save_sessions(sessions)

        # Show rate limit warning if near limit
        if rate_msg:
            await update.message.reply_text(rate_msg)

        # Send text reply first so user gets the answer immediately
        for part in split_message(reply):
            try:
                # Try with Markdown first
                await update.message.reply_text(
                    part, parse_mode=constants.ParseMode.MARKDOWN)
            except Exception:
                # Fall back to plain text if markdown fails
                clean = re.sub(r"[*_`\[\]]", "", part)
                await update.message.reply_text(clean)

        # Circuit map image as a follow-up.  The short sleep lets the thread
        # executor finish writing the file before we check disk.
        import asyncio as _asyncio
        await _asyncio.sleep(2)
        _circuit_img = get_circuit_map_image(text)
        if _circuit_img:
            try:
                log.info(f"Circuit map: sending photo {_circuit_img}")
                with open(_circuit_img, "rb") as _img_fh:
                    await update.message.reply_photo(_img_fh)
                log.info(f"Circuit map: photo sent successfully")
            except Exception as _img_err:
                log.info(f"Circuit map photo send failed — {_img_err}")

    except Exception as e:
        log.error(f"handle_message error for {user_id}: {e}")
        await update.message.reply_text(
            "⚠️ Something went wrong. Try again in a moment! 🏎"
        )

async def handle_error(update: object,
                        ctx: ContextTypes.DEFAULT_TYPE):
    log.error(f"Error: {ctx.error}")

# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    global mem, sessions

    # ── Check credentials ─────────────────────────────────
    if not BOT_TOKEN:
        print("\n  TELEGRAM_BOT_TOKEN not set.")
        print("  Get one from @BotFather on Telegram, then:")
        print("  export TELEGRAM_BOT_TOKEN=your_token_here\n")
        sys.exit(1)

    if not ANTHROPIC_KEY:
        print("\n  ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...\n")
        sys.exit(1)

    # ── Load memory + news ────────────────────────────────
    mem      = load_f1_memory()
    sessions = load_sessions()
    load_news_cache()
    refresh_news_cache()
    start_news_scheduler()

    ep_count  = len(mem.get("episodic", []))
    sem_count = len(mem.get("semantic", {}))

    print(f"""
╔══════════════════════════════════════════════════════╗
║  🏎  BoxBoxAI — F1 Telegram Bot                      ║
║  Developed by Erick Hernandez                        ║
╚══════════════════════════════════════════════════════╝

  Memory:    {sem_count} semantic entries, {ep_count} race episodes
  Model:     {MODEL}
  Sessions:  {len(sessions)} users tracked
  Status:    Starting bot...
""")

    # ── Build and run app ─────────────────────────────────
    # ── Build and run app ─────────────────────────────────
    from telegram.request import HTTPXRequest

    request         = HTTPXRequest(connection_pool_size=8, http_version="1.1")
    request_updates = HTTPXRequest(connection_pool_size=8, http_version="1.1")

    app = (Application.builder()
           .token(BOT_TOKEN)
           .request(request)
           .get_updates_request(request_updates)
           .build())

    from telegram.ext import CallbackQueryHandler
    app.add_handler(CommandHandler("start",          cmd_start))
    app.add_handler(CommandHandler("help",           cmd_help))
    app.add_handler(CommandHandler("standings",      cmd_standings))
    app.add_handler(CommandHandler("constructors",   cmd_constructors))
    app.add_handler(CommandHandler("predict",        cmd_predict))
    app.add_handler(CommandHandler("winner",         cmd_winner))
    app.add_handler(CommandHandler("news",           cmd_news))
    async def _reingest_handler(update, ctx):
        await cmd_reingest(update, ctx, mem_ref)
    app.add_handler(CommandHandler("reingest",       _reingest_handler))
    async def _debug_context_handler(update, ctx):
        await cmd_debug_context(update, ctx, mem_ref)
    app.add_handler(CommandHandler("debug_context",  _debug_context_handler))
    app.add_handler(CallbackQueryHandler(handle_timezone_callback,    pattern="^tz:"))
    app.add_handler(CallbackQueryHandler(handle_notification_callback, pattern="^notif:"))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    # Start race weekend notifications + weekly digest loop
    import asyncio as _asyncio
    sessions_ref    = [sessions]
    mem_ref         = [mem]
    context_builder._app_ref[0] = app  # wire for alerts
    context_builder._live_search_fn[0]           = live_search_f1
    context_builder._fetch_fia_docs_fn[0]        = fetch_fia_race_documents
    context_builder._get_circuit_zone_data_fn[0] = _get_circuit_zone_data

    async def _post_init(application):
        context_builder._app_ref[0] = application  # first line
        _asyncio.create_task(
            notification_loop(application, sessions_ref, mem_ref))
        _asyncio.create_task(
            auto_ingest_loop(mem_ref, application, sessions_ref))
        _asyncio.create_task(
            auto_predictor_loop(application))
        _asyncio.create_task(
            auto_memory_enrichment_loop(mem_ref, application))
        _asyncio.create_task(
            _prewarm_circuit_map_for_next_race())
        await alert_owner(application,
            f"✅ *BoxBoxAI is online*\n\n"
            f"Memory: {len(mem.get('episodic',[]))} races ingested\n"
            f"Users: {len(sessions)} tracked\n"
            f"Predictor: {'✅ CSV ready' if PREDICTOR_CSV.exists() else '⏳ waiting for qualifying'}\n"
            f"Ready to go! 🏎"
        )

    app.post_init = _post_init

    print("  ✅ BoxBoxAI is LIVE. Open Telegram and message your bot.\n")

    app.run_polling(allowed_updates=Update.ALL_TYPES)




async def auto_memory_enrichment_loop(mem_ref: list, app=None):
    """
    Fully automated memory enrichment — zero manual intervention.
    Runs every 6 hours. Waits 48h after race for FastF1 data.
    """
    import asyncio as _asyncio

    while True:
        try:
            await _run_memory_enrichment(mem_ref, app)
        except Exception as e:
            log.warning(f"Memory enrichment loop error: {e}")
        # Check every 6 hours
        await _asyncio.sleep(21600)


async def _run_memory_enrichment(mem_ref: list, app=None):
    """
    Core enrichment logic. Checks each completed race and
    enriches with FastF1 telemetry if not already done.

    Skip-check is based on the persisted memory itself (episode
    having real tyre/sector data), not a separate state file —
    Railway's filesystem resets on redeploy, so a separate state
    file can't be trusted across restarts.

    All results are batched into ONE alert to the owner.
    """
    state    = load_enrichment_state()
    mem      = mem_ref[0]
    episodes = mem.get("episodic", [])
    today    = datetime.now()
    enriched_results = []  # collect for single batched alert

    RACE_CALENDAR = [
        (1, "Australian GP",  "2026-03-15"),
        (2, "Chinese GP",     "2026-03-22"),
        (3, "Japanese GP",    "2026-04-06"),
        (4, "Miami GP",       "2026-05-04"),
        (5, "Canadian GP",    "2026-05-24"),
        (6, "Monaco GP",      "2026-06-07"),
        (7, "Spanish GP",     "2026-06-14"),
        (8, "Austrian GP",    "2026-06-28"),
        (9, "British GP",     "2026-07-05"),
        (10,"Belgian GP",     "2026-07-19"),
        (11,"Hungarian GP",   "2026-07-26"),
        (12,"Dutch GP",       "2026-08-23"),
        (13,"Italian GP",     "2026-09-06"),
        (14,"Singapore GP",   "2026-09-20"),
        (15,"Azerbaijan GP",  "2026-09-27"),
        (16,"US GP",          "2026-10-18"),
        (17,"Mexico City GP", "2026-10-25"),
        (18,"São Paulo GP",   "2026-11-08"),
        (19,"Las Vegas GP",   "2026-11-21"),
        (20,"Qatar GP",       "2026-11-29"),
        (21,"Abu Dhabi GP",   "2026-12-06"),
    ]

    for rnd, name, date_str in RACE_CALENDAR:
        race_date   = datetime.strptime(date_str, "%Y-%m-%d")
        days_since  = (today - race_date).days

        # Skip future races
        if days_since < 0:
            break

        # FastF1 needs 48h after race to have full data
        if days_since < 2:
            continue

        state_key = f"enriched_r{rnd}"

        # Find episode
        episode = next(
            (e for e in episodes if e.get("round") == rnd), None)

        # If no basic result yet, ingest from Jolpica first
        if not episode or not episode.get("winner"):
            result = fetch_race_result(rnd, SEASON)
            if not result:
                continue
            if episode:
                episode.update(result)
            else:
                episode = result
                episode["round"]     = rnd
                episode["race_name"] = name

        # ── Skip check based on actual data quality, not just a flag ──
        # An episode is "fully enriched" if it has a telemetry source
        # AND actual substantive data (tyre strategy or sector bests) —
        # not just a source tag with empty fields.
        has_real_telemetry = (
            episode.get("telemetry_source")
            and (episode.get("pitstops", {}).get("tyre_strategies")
                 or episode.get("sector_bests"))
        )

        # ── Retry cap ───────────────────────────────────────────
        # FastF1's livetiming mirror and OpenF1 don't retain full
        # telemetry (tyre stints, sector times) indefinitely for
        # older sessions — some races may simply never get "real"
        # data. Without a cap, these retry every 6h forever, and
        # since Railway's filesystem resets on redeploy, every
        # redeploy re-triggers the same "partial (will retry)"
        # alert for the same races. After 14 days, give up — the
        # episode already has winner/podium/points from Jolpica,
        # which is the data that actually matters for answering
        # questions. Mark as done regardless of telemetry richness.
        give_up = days_since > 14

        if has_real_telemetry or state.get(state_key) or (give_up and episode.get("telemetry_source")):
            if not state.get(state_key):
                state[state_key] = datetime.now().isoformat()
                save_enrichment_state(state)
            continue

        log.info(f"Auto-enrichment: starting R{rnd} {name}...")

        # ── Enrich race with FastF1 ───────────────────────────
        episode = enrich_episode_with_telemetry(episode, rnd)

        # ── Enrich qualifying with FastF1 ─────────────────────
        episode = enrich_qualifying_with_telemetry(episode, rnd)

        # ── Enrich sprint if applicable ───────────────────────
        if rnd in SPRINT_ROUNDS_2026 and not episode.get("sprint",{}).get("sprint_winner"):
            sprint = fetch_sprint_result(rnd, SEASON)
            if sprint:
                episode["sprint"] = sprint

        # ── Rebuild rich story with all new data ──────────────
        episode["story"] = build_rich_story(episode, ask_fn=ask_claude)

        # ── Update semantic memory ────────────────────────────
        mem.setdefault("semantic", {})[f"race/r{rnd}"] = {
            "text": episode["story"],
            "tags": [name.lower(), str(rnd),
                     episode.get("winner","").lower()],
        }
        if episode.get("champ_after"):
            mem["semantic"][f"standings/after_r{rnd}"] = {
                "text": f"After R{rnd} {name}: {episode['champ_after']}"
            }
        quali = episode.get("qualifying", {})
        if quali.get("pole"):
            elim_q2 = ", ".join(quali.get("eliminated_q2",[])[:5])
            elim_q1 = ", ".join(quali.get("eliminated_q1",[])[:5])
            mem["semantic"][f"quali/r{rnd}"] = {
                "text": (
                    f"R{rnd} {name} qualifying: "
                    f"Pole {quali['pole']} ({quali.get('pole_time','')})."
                    f"{f' Q2 out: {elim_q2}.' if elim_q2 else ''}"
                    f"{f' Q1 out: {elim_q1}.' if elim_q1 else ''}"
                )
            }

        # ── Save episode back to memory ───────────────────────
        existing_idx = next(
            (i for i, e in enumerate(episodes)
             if e.get("round") == rnd), None)
        if existing_idx is not None:
            episodes[existing_idx] = episode
        else:
            episodes.append(episode)
            episodes.sort(key=lambda x: x.get("round", 0))

        mem["episodic"] = episodes
        save_f1_memory(mem)
        mem_ref[0] = mem

        # Only mark as fully enriched if we actually got real telemetry,
        # OR we've given up after 14 days (data won't materialize)
        source = episode.get("telemetry_source", "jolpica")
        got_real_data = bool(
            episode.get("pitstops", {}).get("tyre_strategies")
            or episode.get("sector_bests"))
        if got_real_data or give_up:
            state[state_key] = datetime.now().isoformat()
            save_enrichment_state(state)

        log.info(
            f"Auto-enrichment: {'✅' if got_real_data else ('🛑' if give_up else '⏳')} "
            f"R{rnd} {name} processed (source: {source}, "
            f"real_data: {got_real_data}, give_up: {give_up})")

        enriched_results.append({
            "round": rnd, "name": name, "source": source,
            "fastest_lap": episode.get("fastest_lap","?"),
            "fastest_lap_time": episode.get("fastest_lap_time",""),
            "complete": got_real_data,
            "gave_up": give_up and not got_real_data,
        })

        # Small delay between races to avoid rate limiting
        await asyncio.sleep(5)

    # ── Single batched alert for everything that happened ────────
    if enriched_results and app:
        lines = [f"🧠 *Memory enrichment run* — {len(enriched_results)} race(s) processed\n"]
        for r in enriched_results:
            if r["complete"]:
                status = "✅ full telemetry"
            elif r.get("gave_up"):
                status = "🛑 no detailed telemetry available (kept results only)"
            else:
                status = "⏳ partial (will retry)"
            fl = f"{r['fastest_lap']} {r['fastest_lap_time']}".strip()
            lines.append(
                f"• R{r['round']} {r['name']}: {status}"
                f"{f' — FL {fl}' if fl != '?' else ''}")
        await alert_owner(app, "\n".join(lines))
    elif not enriched_results:
        log.debug("Auto-enrichment: nothing to do")


if __name__ == "__main__":
    main()
