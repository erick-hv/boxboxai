#!/usr/bin/env python3
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
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL         = "claude-sonnet-4-5"
MAX_TOKENS    = 1000
MEMORY_FILE   = Path(__file__).parent / "f1_memory_2026.json"
SESSIONS_FILE = Path(__file__).parent / "boxboxai_sessions.json"
JOLPICA       = "https://api.jolpi.ca/ergast/f1"
SEASON        = 2026

# Telegram message limit
TG_MAX_CHARS  = 4096

# ── The Race news feed ────────────────────────────────────────
THE_RACE_RSS      = "https://the-race.com/feed/"
THE_RACE_SEARCH   = "https://the-race.com/?s="
NEWS_CACHE_FILE   = Path(__file__).parent / "boxboxai_news_cache.json"
NEWS_REFRESH_MINS = 30   # refresh RSS every 30 minutes

# ═════════════════════════════════════════════════════════════
#  THE RACE — NEWS ENGINE
# ═════════════════════════════════════════════════════════════

_news_cache      = []          # list of {title, summary, url, date}
_news_cache_time = None        # last fetch time
_news_lock       = threading.Lock()


def _fetch_rss() -> list:
    """Fetches and parses The Race RSS feed. Returns list of articles."""
    try:
        r = requests.get(THE_RACE_RSS, timeout=10,
                         headers={"User-Agent": "BoxBoxAI/1.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        articles = []
        for item in root.findall(".//item")[:20]:
            title   = item.findtext("title", "").strip()
            link    = item.findtext("link",  "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            # Get description — strip HTML tags
            desc = item.findtext("description", "")
            desc = re.sub(r"<[^>]+>", "", desc).strip()[:300]
            if title:
                articles.append({
                    "title":   title,
                    "summary": desc,
                    "url":     link,
                    "date":    pubdate,
                })
        return articles
    except Exception as e:
        log.warning(f"RSS fetch failed: {e}")
        return []


def _fetch_article_text(url: str, max_chars: int = 2000) -> str:
    """Fetches and extracts plain text from a The Race article URL."""
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "BoxBoxAI/1.0"})
        if r.status_code != 200:
            return ""
        # Strip scripts, styles, nav
        text = r.text
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>",   "", text, flags=re.DOTALL)
        text = re.sub(r"<nav[^>]*>.*?</nav>",        "", text, flags=re.DOTALL)
        text = re.sub(r"<header[^>]*>.*?</header>",  "", text, flags=re.DOTALL)
        text = re.sub(r"<footer[^>]*>.*?</footer>",  "", text, flags=re.DOTALL)
        # Extract article/main content
        m = re.search(r'<article[^>]*>(.*?)</article>', text, re.DOTALL)
        if m:
            text = m.group(1)
        # Strip all remaining HTML
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _search_the_race(query: str, max_results: int = 3) -> list:
    """
    Searches The Race site for a query.
    Returns list of {title, summary, url} from search results.
    """
    try:
        q   = requests.utils.quote(query)
        url = f"{THE_RACE_SEARCH}{q}"
        r   = requests.get(url, timeout=10,
                           headers={"User-Agent": "BoxBoxAI/1.0"})
        if r.status_code != 200:
            return []
        # Extract article links and titles from search results page
        # The Race search results use article tags with h2/h3 titles
        articles = []
        pattern  = r'<h[23][^>]*>\s*<a\s+href="([^"]+)"[^>]*>([^<]+)</a>'
        for match in re.finditer(pattern, r.text)[:max_results * 2]:
            url_found = match.group(1).strip()
            title     = re.sub(r"\s+", " ", match.group(2)).strip()
            if "the-race.com" in url_found and title and len(title) > 10:
                articles.append({"title": title, "url": url_found, "summary": ""})
            if len(articles) >= max_results:
                break
        return articles
    except Exception as e:
        log.warning(f"The Race search failed: {e}")
        return []


def refresh_news_cache():
    """Refreshes the RSS cache. Called on startup and every 30 min."""
    global _news_cache, _news_cache_time
    with _news_lock:
        articles = _fetch_rss()
        if articles:
            _news_cache      = articles
            _news_cache_time = datetime.now()
            # Persist to disk so it survives restarts
            try:
                NEWS_CACHE_FILE.write_text(json.dumps({
                    "fetched_at": datetime.now().isoformat(),
                    "articles":   articles,
                }, indent=2))
            except Exception:
                pass
            log.info(f"News cache refreshed: {len(articles)} articles")


def load_news_cache() -> list:
    """Loads news cache from disk (used on startup)."""
    global _news_cache, _news_cache_time
    if NEWS_CACHE_FILE.exists():
        try:
            data = json.loads(NEWS_CACHE_FILE.read_text())
            _news_cache      = data.get("articles", [])
            _news_cache_time = datetime.fromisoformat(
                data.get("fetched_at", "2000-01-01"))
            return _news_cache
        except Exception:
            pass
    return []


def get_news_context(query: str = "") -> str:
    """
    Returns relevant news context for a given query.
    - If query is empty: returns latest headlines summary
    - If query given: finds relevant articles + fetches content
    """
    global _news_cache, _news_cache_time

    # Refresh if stale
    needs_refresh = (
        not _news_cache or
        _news_cache_time is None or
        datetime.now() - _news_cache_time > timedelta(minutes=NEWS_REFRESH_MINS)
    )
    if needs_refresh:
        refresh_news_cache()

    if not query:
        # Return latest headlines
        if not _news_cache:
            return ""
        lines = ["Latest F1 news:"]
        for a in _news_cache[:8]:
            lines.append(f"- {a['title']} ({a['date'][:16]})")
        return "\n".join(lines)

    # Query-specific: search RSS cache first
    q_lower = query.lower()
    relevant = []

    # Score cached articles by keyword match
    for article in _news_cache:
        score = 0
        text  = (article["title"] + " " + article["summary"]).lower()
        for word in q_lower.split():
            if len(word) > 3 and word in text:
                score += 1
        if score > 0:
            relevant.append((score, article))

    relevant.sort(key=lambda x: x[0], reverse=True)
    top = [a for _, a in relevant[:3]]

    # If nothing in cache, search The Race live
    if not top:
        top = _search_the_race(query, max_results=2)

    if not top:
        return ""

    # Fetch full text of top article
    context_parts = [f"Recent F1 news related to '{query}':"]
    for article in top[:2]:
        context_parts.append(f"\n{article['title']}")
        if article.get("summary"):
            context_parts.append(article["summary"])
        elif article.get("url"):
            full_text = _fetch_article_text(article["url"], max_chars=1500)
            if full_text:
                context_parts.append(full_text[:800])

    return "\n".join(context_parts)


def _news_scheduler():
    """Background thread that refreshes news every 30 minutes."""
    while True:
        time.sleep(NEWS_REFRESH_MINS * 60)
        try:
            refresh_news_cache()
        except Exception:
            pass


def start_news_scheduler():
    """Starts background news refresh thread."""
    t = threading.Thread(target=_news_scheduler, daemon=True)
    t.start()
    log.info("News scheduler started (refreshes every 30 min)")


# ═════════════════════════════════════════════════════════════
#  FEATURE: RACE WEEKEND NOTIFICATIONS
# ═════════════════════════════════════════════════════════════

NOTIFICATIONS_FILE = Path(__file__).parent / "boxboxai_notifications.json"

# Track which notifications have been sent to avoid duplicates
def load_notification_state() -> dict:
    if NOTIFICATIONS_FILE.exists():
        try:
            return json.loads(NOTIFICATIONS_FILE.read_text())
        except Exception:
            pass
    return {}

def save_notification_state(state: dict):
    NOTIFICATIONS_FILE.write_text(json.dumps(state, indent=2))

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

async def send_race_weekend_notifications(app, sessions: dict,
                                          mem: dict):
    """
    Sends proactive race weekend messages to all active users.
    Friday: Practice starts — hype message
    Saturday: Qualifying preview
    Sunday: Race day message with prediction
    """
    state     = load_notification_state()
    next_race = fetch_next_race()
    if not next_race:
        return

    race_name  = next_race.get("raceName", "Next Race")
    race_date  = next_race.get("date", "")   # YYYY-MM-DD (Sunday)
    round_num  = next_race.get("round", "?")

    if not race_date:
        return

    try:
        race_dt = datetime.strptime(race_date, "%Y-%m-%d")
    except Exception:
        return

    today      = datetime.now()
    days_to    = (race_dt.date() - today.date()).days
    state_key  = f"r{round_num}"

    # Friday = race_day - 2, Saturday = race_day - 1, Sunday = race_day
    messages_to_send = []

    if days_to == 2 and not state.get(f"{state_key}_friday"):
        # Get weather for context
        weather = get_weather_context(race_name, next_race)
        msg = (
            f"🏎 *{race_name} weekend is here!* 🏁\n\n"
            f"Practice starts today — the fastest drivers in the world are "
            f"hitting the track at {next_race.get('Circuit',{}).get('circuitName','')}.\n\n"
            f"Follow along and ask me anything — strategy predictions, "
            f"who looks fast, weather impact, whatever you want to know. "
            f"Let's talk F1! 🔥"
        )
        messages_to_send.append((f"{state_key}_friday", msg))

    elif days_to == 1 and not state.get(f"{state_key}_saturday"):
        msg = (
            f"⏱ *{race_name} — Qualifying Day!* 🚦\n\n"
            f"Today we find out who starts from pole. "
            f"Monaco grid position is everything — ask me "
            f"_/winner_ for my prediction or ask who I think takes pole. 🏆"
        )
        messages_to_send.append((f"{state_key}_saturday", msg))

    elif days_to == 0 and not state.get(f"{state_key}_sunday"):
        msg = (
            f"🏁 *RACE DAY — {race_name}!* 🏎\n\n"
            f"Lights out today! Ask me anything — "
            f"prediction, strategy breakdown, championship stakes, weather. "
            f"Let's go racing! 🔥🏆\n\n"
            f"_/winner_ — my race winner pick\n"
            f"_/predict full_ — complete race analysis"
        )
        messages_to_send.append((f"{state_key}_sunday", msg))

    if not messages_to_send:
        return

    active_users = get_active_user_ids(sessions)
    if not active_users:
        return

    for state_flag, message in messages_to_send:
        sent_count = 0
        for uid in active_users:
            try:
                await app.bot.send_message(
                    chat_id=uid,
                    text=message,
                    parse_mode="Markdown"
                )
                sent_count += 1
                await asyncio.sleep(0.1)  # rate limit
            except Exception:
                pass
        state[state_flag] = datetime.now().isoformat()
        log.info(f"Notification {state_flag} sent to {sent_count} users")

    save_notification_state(state)


# ═════════════════════════════════════════════════════════════
#  FEATURE: HISTORICAL COMPARISONS
# ═════════════════════════════════════════════════════════════

HISTORICAL_DATA = """
HISTORICAL F1 RECORDS AND COMPARISONS (for context when asked):

Rookie seasons — wins in first 5 races:
- Antonelli 2026: 4 wins from 5 races (Shanghai, Suzuka, Miami, Montreal) — HISTORIC
- Verstappen 2015: 0 wins (Toro Rosso, first full season)
- Hamilton 2007: 0 wins in first 5, won USA R7 (finished season with 4 wins)
- Leclerc 2018: 0 wins (Sauber)
- Piastri 2023: 0 wins (2 sprint wins)
- Senna 1984: 0 wins (Toleman)
- Schumacher 1991: 0 wins

Championship comparisons:
- Verstappen: 4 WDC (2021,2022,2023,2024)
- Hamilton: 7 WDC (2008,2014,2015,2017,2018,2019,2020)
- Schumacher M: 7 WDC (1994,1995,2000,2001,2002,2003,2004)
- Senna: 3 WDC (1988,1990,1991)
- Prost: 4 WDC (1985,1986,1989,1993)
- Alonso: 2 WDC (2005,2006)

Monaco winners (recent):
- 2025: Norris (pole 1:09.954)
- 2024: Leclerc
- 2023: Verstappen
- 2022: Perez
- 2021: Verstappen

Points records (modern era):
- Most points in a season: Verstappen 2023 (575pts, 19 wins from 22 races)
- Fastest ever lap records, domination stats available on request

Mercedes dominance 2014-2021: 8 consecutive WCC titles
Red Bull dominance 2022-2024: 3 consecutive WCC + 4 WDC Verstappen
"""

def get_historical_context(query: str) -> str:
    """Returns historical context if query seems comparison-based."""
    comparison_keywords = [
        "compare", "better than", "vs", "versus", "history", "historical",
        "record", "ever", "all time", "greatest", "goat", "best",
        "rookie", "debut", "first season", "like verstappen", "like hamilton",
        "comparar", "mejor que", "historia", "récord", "el mejor", "de todos",
        "temporada debut", "novato", "como verstappen", "como hamilton",
        "similar", "parecido", "igual que", "peor que", "superar"
    ]
    t = query.lower()
    if any(kw in t for kw in comparison_keywords):
        return HISTORICAL_DATA
    return ""


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
    Sends a weekly F1 digest to all active users every Monday.
    Includes: race recap, championship standings, what's coming next.
    """
    state = load_digest_state()
    today = datetime.now().strftime("%Y-%W")  # year-week number

    if state.get("last_digest_week") == today:
        return  # already sent this week

    # Only send on Mondays
    if datetime.now().weekday() != 0:
        return

    active_users = get_active_user_ids(sessions)
    if not active_users:
        return

    # Build digest content
    episodes   = mem.get("episodic", [])
    last_race  = episodes[-1] if episodes else None
    next_race  = fetch_next_race()
    drivers, _ = fetch_standings()

    # Championship leader
    leader = ""
    if drivers:
        d    = drivers[0].get("Driver", {})
        name = f"{d.get('givenName','')[:1]}. {d.get('familyName','')}"
        pts  = drivers[0].get("points", "?")
        leader = f"*{name}* leads with *{pts}pts*"

    # Last race summary
    last_str = ""
    if last_race:
        last_str = (f"🏁 Last race: *{last_race.get('race_name', last_race.get('track','?'))}* "
                    f"— Winner: *{last_race.get('winner','?')}*")

    # Next race
    next_str = ""
    if next_race:
        next_str = (f"🔜 Next up: *{next_race.get('raceName','?')}* "
                    f"on {next_race.get('date','?')}")

    digest = (
        f"🏎 *BoxBoxAI Weekly Digest* 🏆\n"
        f"_Your F1 week in review_\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Championship: {leader}\n\n"
        f"{last_str}\n\n"
        f"{next_str}\n\n"
        f"Ask me anything about the season — predictions, analysis, "
        f"debates. Let's talk F1! 🔥\n\n"
        f"_/winner_ — next race prediction\n"
        f"_/debate_ — start a debate\n"
        f"_/news_ — latest F1 headlines"
    )

    sent = 0
    for uid in active_users:
        try:
            await app.bot.send_message(
                chat_id=uid,
                text=digest,
                parse_mode="Markdown"
            )
            sent += 1
            await asyncio.sleep(0.1)
        except Exception:
            pass

    state["last_digest_week"] = today
    save_digest_state(state)
    log.info(f"Weekly digest sent to {sent} users")


async def notification_loop(app, sessions_ref: list, mem_ref: list):
    """
    Async loop that checks every hour for:
    - Race weekend notifications to send
    - Weekly digest to send
    """
    import asyncio as _asyncio
    while True:
        try:
            await send_race_weekend_notifications(
                app, sessions_ref[0], mem_ref[0])
            await send_weekly_digest(
                app, sessions_ref[0], mem_ref[0])
        except Exception as e:
            log.warning(f"Notification loop error: {e}")
        await _asyncio.sleep(3600)  # check every hour


# ═════════════════════════════════════════════════════════════
#  FEATURE: USER MEMORY PERSONALIZATION
# ═════════════════════════════════════════════════════════════

def build_user_profile(user_data: dict) -> str:
    """
    Builds a personalization string from a user's chat history.
    Used to make the bot feel like it remembers the user.
    """
    if not user_data:
        return ""

    stats   = user_data.get("stats", {})
    history = user_data.get("history", [])
    topics  = stats.get("favorite_topics", {})

    if not topics and not history:
        return ""

    parts = []

    # Favourite topic
    if topics:
        top = max(topics, key=topics.get)
        parts.append(f"This user's favourite topic is {top}.")

    # Detect favourite driver from chat history
    driver_mentions = {}
    driver_keys = {
        "checo": "Sergio Pérez", "pérez": "Sergio Pérez", "perez": "Sergio Pérez",
        "hamilton": "Lewis Hamilton", "lewis": "Lewis Hamilton",
        "verstappen": "Max Verstappen", "max": "Max Verstappen",
        "antonelli": "Kimi Antonelli", "kimi": "Kimi Antonelli",
        "leclerc": "Charles Leclerc", "charles": "Charles Leclerc",
        "russell": "George Russell", "norris": "Lando Norris",
        "lando": "Lando Norris", "alonso": "Fernando Alonso", "nano": "Fernando Alonso",
    }
    for msg in history:
        if msg.get("role") == "user":
            t = msg.get("content", "").lower()
            for key, driver in driver_keys.items():
                if key in t:
                    driver_mentions[driver] = driver_mentions.get(driver, 0) + 1

    if driver_mentions:
        fav = max(driver_mentions, key=driver_mentions.get)
        parts.append(f"This user talks about {fav} most — they're probably a fan.")

    # Detect language preference
    spanish_count = sum(
        1 for m in history
        if m.get("role") == "user" and
        any(w in m.get("content","").lower()
            for w in ["qué", "cómo", "quién", "cuándo", "ganar", "carrera"])
    )
    if spanish_count > len(history) * 0.3:
        parts.append("This user primarily communicates in Spanish.")

    if not parts:
        return ""

    return "USER PROFILE:\n" + "\n".join(f"- {p}" for p in parts)


# ═════════════════════════════════════════════════════════════
#  FEATURE: LIVE TIMING / RACE SESSION DATA
# ═════════════════════════════════════════════════════════════

def fetch_practice_results(round_num: int, season: int = SEASON) -> str:
    """
    Fetches FP1/FP2/FP3 results from OpenF1 for a given round.
    Returns formatted string of practice session results.
    """
    session_types = ["Practice 1", "Practice 2", "Practice 3"]
    results = []

    for session_name in session_types:
        try:
            # Get session key
            sessions_data = fetch_openf1("sessions", {
                "year":         season,
                "session_name": session_name,
            })
            if not sessions_data:
                continue

            # Find the right round by sorting and indexing
            sorted_sessions = sorted(
                [s for s in sessions_data if s.get("session_name") == session_name],
                key=lambda x: x.get("date_start", "")
            )
            if round_num > len(sorted_sessions):
                continue

            sk = sorted_sessions[round_num - 1].get("session_key")
            if not sk:
                continue

            # Check session has ended
            end_date = sorted_sessions[round_num - 1].get("date_end", "")
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", ""))
                    if end_dt > datetime.utcnow():
                        continue  # session not finished yet
                except Exception:
                    pass

            # Get fastest laps per driver
            laps = fetch_openf1("laps", {"session_key": sk})
            if not laps:
                continue

            # Get drivers for this session
            drivers_data = fetch_openf1("drivers", {"session_key": sk})
            num_to_name  = {}
            for d in (drivers_data or []):
                num  = str(d.get("driver_number", ""))
                code = d.get("name_acronym", num)
                num_to_name[num] = code

            # Best lap per driver
            best = {}
            for lap in laps:
                dur = lap.get("lap_duration")
                num = str(lap.get("driver_number", ""))
                if not dur or not num:
                    continue
                try:
                    dur_f = float(dur)
                    if dur_f > 0 and (num not in best or dur_f < best[num]):
                        best[num] = dur_f
                except Exception:
                    pass

            if not best:
                continue

            # Sort by fastest lap
            sorted_best = sorted(best.items(), key=lambda x: x[1])[:10]

            lines = [f"*{session_name}:*"]
            for i, (num, lap_time) in enumerate(sorted_best, 1):
                code   = num_to_name.get(num, num)
                mins   = int(lap_time // 60)
                secs   = lap_time % 60
                t_str  = f"{mins}:{secs:06.3f}"
                gap    = ""
                if i > 1:
                    delta = lap_time - sorted_best[0][1]
                    gap   = f" (+{delta:.3f}s)"
                lines.append(f"P{i}: *{code}* {t_str}{gap}")

            results.append("\n".join(lines))
            time.sleep(0.3)

        except Exception as e:
            log.warning(f"FP fetch failed for {session_name}: {e}")
            continue

    if not results:
        return ""

    return "\n\n".join(results)


def get_practice_context(query: str, next_race: dict | None = None,
                          mem: dict | None = None) -> str:
    """Returns practice session data if query asks about FP1/FP2/FP3."""
    q = query.lower()
    is_practice = any(kw in q for kw in [
        "fp1", "fp2", "fp3", "practice", "práctica", "libre",
        "free practice", "entreno", "entrenamiento"
    ])
    if not is_practice:
        return ""

    # Figure out which round to look up
    round_num = None

    # Check if asking about next/current race
    if next_race:
        round_num = int(next_race.get("round", 0))

    # Check episodes for round references
    if mem and not round_num:
        episodes = mem.get("episodic", [])
        if episodes:
            round_num = episodes[-1].get("round", 0) + 1

    if not round_num:
        return ""

    results = fetch_practice_results(round_num, SEASON)
    if not results:
        return ""

    circuit = next_race.get("Circuit", {}).get("circuitName", "") if next_race else ""
    return f"Practice session results — {circuit} R{round_num}:\n\n{results}"



    """
    Checks OpenF1 for any currently active session.
    Returns session info if something is live right now.
    """
    try:
        # Get sessions happening today
        today = datetime.now().strftime("%Y-%m-%d")
        data  = fetch_openf1("sessions", {
            "year":       SEASON,
            "date_start": today,
        })
        if not data:
            return None
        # Find most recent session
        data.sort(key=lambda x: x.get("date_start", ""), reverse=True)
        return data[0] if data else None
    except Exception:
        return None


def fetch_openf1(endpoint: str, params: dict) -> list:
    """Generic OpenF1 API call."""
    try:
        r = requests.get(
            f"https://api.openf1.org/v1/{endpoint}",
            params=params, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def fetch_live_session() -> dict | None:
    """
    Checks OpenF1 for any currently active session today.
    Returns session info if something is live right now.
    """
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        data  = fetch_openf1("sessions", {
            "year":       SEASON,
            "date_start": today,
        })
        if not data:
            return None
        # Find most recent session that hasn't ended yet
        now = datetime.utcnow()
        for s in sorted(data, key=lambda x: x.get("date_start",""), reverse=True):
            end = s.get("date_end","")
            try:
                end_dt = datetime.fromisoformat(end.replace("Z",""))
                if end_dt > now:
                    return s
            except Exception:
                continue
        # If none ongoing, return the most recent of today
        return data[0] if data else None
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
#  FEATURE: PREDICTION ACCURACY TRACKING
# ═════════════════════════════════════════════════════════════

PREDICTIONS_FILE = Path(__file__).parent / "boxboxai_predictions.json"

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
#  FEATURE: CIRCUIT GUIDES
# ═════════════════════════════════════════════════════════════

CIRCUIT_GUIDES = {
    "monaco": """
Circuit de Monaco — The most famous street circuit.
- 78 laps, 3.337km, 19 corners
- Overtaking difficulty: 9.5/10 — almost impossible once positions set
- Qualifying is EVERYTHING — pole wins ~50% of the time, P1-P3 win ~90%
- Key corners: Sainte Devote (T1 crash magnet), Massenet, Casino Square,
  Mirabeau, Grand Hotel Hairpin (tightest corner in F1), Tunnel (blind exit),
  Nouvelle Chicane, Tabac, Swimming Pool complex, Rascasse, Anthony Noghes
- Strategy: almost always 1-stop, tyres don't degrade much but safety car
  bunches the field — timing your pit stop around SC is critical
- Historical stat: winner has started P1 or P2 in 80%+ of modern era races
- Biggest danger: barriers everywhere, any mistake ends your race
- Checo's home (spiritually) — he's won here multiple times
""",
    "silverstone": """
Circuit: Silverstone, UK — High speed temple
- 52 laps, 5.891km, 18 corners
- Fast, sweeping corners — Copse, Maggotts, Becketts, Chapel are iconic
- Very hard on rear tyres — 2-stop strategy common in hot conditions
- Weather: British weather means anything can happen
- Overtaking: DRS on Hangar Straight and Wellington Straight — decent action
- Wing level: medium-low downforce
""",
    "monza": """
Circuit: Monza, Italy — Temple of Speed
- 53 laps, 5.793km
- Lowest downforce track of the season — drag reduction everything
- Power unit circuit — Mercedes/Ferrari power advantage matters here
- Main straight + Curva Grande + Lesmo 1&2 + Ascari + Parabolica
- Slipstreaming huge — qualifying can produce surprise results
- Tyre wear low — usually 1-stop
- Tifosi (Ferrari fans) make the atmosphere electric
""",
    "spa": """
Circuit: Spa-Francorchamps, Belgium — Greatest circuit on the calendar
- 44 laps, 7.004km — longest circuit on the calendar
- Eau Rouge/Raidillon — most iconic corner sequence in F1
- Weather: Spa has its own microclimate — can be wet sector 1, dry sector 3
- High downforce needed — lots of medium/high speed corners
- Pouhon, Blanchimont, Bus Stop chicane
- Overtaking: Kemmel Straight + DRS after Raidillon — great racing
""",
    "suzuka": """
Circuit: Suzuka, Japan — Figure-8 layout, technical masterpiece
- 53 laps, 5.807km
- Figure-8 layout with overpass — unique in F1
- S-curves in sector 1 are defining — ultra-high speed commitment required
- 130R corner — one of fastest in F1
- Spoon curve, Degner curves, Casino Triangle
- Medium-high downforce, hard on rear tyres
- Night race vibes — passionate Japanese crowd
""",
    "melbourne": """
Circuit: Albert Park, Australia — Season opener
- 58 laps, 5.278km, 16 corners
- Semi-street circuit through public park
- Medium downforce, decent overtaking on long straight
- Turn 1 always drama-prone on lap 1
- Weather: Melbourne autumn — can be changeable
""",
}

def get_circuit_guide(query: str) -> str:
    """Returns circuit guide if query mentions a specific circuit."""
    q = query.lower()
    for circuit, guide in CIRCUIT_GUIDES.items():
        if circuit in q:
            return f"CIRCUIT GUIDE — {circuit.upper()}:{guide}"
    return ""


# ═════════════════════════════════════════════════════════════
#  FEATURE: LIVE DRIVER CAREER STATS
# ═════════════════════════════════════════════════════════════

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


def _detect_driver_stat_query(text: str) -> str | None:
    """Returns driver code if query is asking for career stats."""
    stat_keywords = ["how many wins", "career stats", "total wins", "how many poles",
                     "career wins", "all time wins", "cuántas victorias", "estadísticas",
                     "cuántos podios", "cuántas poles"]
    t = text.lower()
    if not any(kw in t for kw in stat_keywords):
        return None

    driver_map = {
        "hamilton": "HAM", "lewis": "HAM",
        "verstappen": "VER", "max": "VER",
        "leclerc": "LEC", "charles": "LEC",
        "alonso": "ALO", "nano": "ALO", "fernando": "ALO",
        "perez": "PER", "checo": "PER", "pérez": "PER",
        "russell": "RUS", "george": "RUS",
        "norris": "NOR", "lando": "NOR",
        "bottas": "BOT", "valtteri": "BOT",
        "sainz": "SAI", "carlitos": "SAI",
    }
    for name, code in driver_map.items():
        if name in t:
            return code
    return None


# ═════════════════════════════════════════════════════════════
#  OFF-TOPIC GUARDRAIL
# ═════════════════════════════════════════════════════════════

OFF_TOPIC_RESPONSES_EN = [
    "Staying in my lane! 🏎 I'm BoxBoxAI — I only talk F1. For that kind of question try ChatGPT or Google. Now, want to talk about the Monaco GP this weekend? 🇲🇨",
    "That's outside my pit box! 🔧 I'm built exclusively for F1 — races, drivers, strategy, predictions. Ask me anything about the 2026 season instead! 🏆",
    "I only know racing! 🏁 BoxBoxAI is your F1 expert, not a general assistant. Try ChatGPT for that. But if you want to debate whether Antonelli is the best rookie ever, I'm ready 🔥",
]

OFF_TOPIC_RESPONSES_ES = [
    "¡Me quedo en mi carril! 🏎 Soy BoxBoxAI, solo hablo de F1. Para eso mejor usa ChatGPT. Pero si quieres hablar de Mónaco este fin de semana, aquí estoy 🇲🇨",
    "¡Eso está fuera de mi box! 🔧 Solo entiendo de F1 — carreras, pilotos, estrategia, predicciones. ¡Pregúntame algo de la temporada 2026! 🏆",
    "¡Solo sé de carreras! 🏁 BoxBoxAI es tu experto en F1, no un asistente general. Para eso usa ChatGPT. Pero si quieres debatir si el Checo puede ganar en Mónaco, ¡adelante! 🔥",
]

# Keywords that strongly suggest off-topic questions
OFF_TOPIC_KEYWORDS = [
    # Homework / school
    "homework", "tarea", "essay", "ensayo", "thesis", "tesis",
    "assignment", "trabajo escolar", "examen", "exam", "school",
    "escuela", "university", "universidad", "college", "teacher",
    "profesor", "classroom", "aula", "subject", "materia",
    "math", "matemáticas", "history", "historia", "biology",
    "biología", "chemistry", "química", "physics", "física",
    "literature", "literatura", "geography", "geografía",
    # Work tasks
    "write my", "escríbeme", "write an email", "escribe un correo",
    "cover letter", "carta de presentación", "resume", "curriculum",
    "business plan", "plan de negocios", "presentation", "presentación",
    "spreadsheet", "excel formula", "code for me", "write code",
    "debug my", "fix my code", "javascript", "python tutorial",
    # General non-F1
    "recipe", "receta", "cooking", "cocina", "how to cook",
    "medical advice", "consejo médico", "symptoms", "síntomas",
    "legal advice", "consejo legal", "lawyer", "abogado",
    "translate", "traducir", "translate this",
    "what is the capital", "cuál es la capital",
    "who invented", "quién inventó",
    "movie recommendation", "recomendación de película",
    "song lyrics", "letra de canción",
    "stock price", "precio de acción",
    "crypto", "bitcoin", "investment advice",
    "travel advice", "consejo de viaje",
    "hotel recommendation", "restaurante",
    "workout", "ejercicio rutina", "diet plan", "dieta",
]

# F1-adjacent keywords that should always pass through
F1_SAFE_KEYWORDS = [
    "f1", "formula", "grand prix", "gp", "race", "carrera",
    "driver", "piloto", "team", "equipo", "circuit", "circuito",
    "championship", "campeonato", "qualifying", "clasificación",
    "pit stop", "tyre", "llanta", "strategy", "estrategia",
    "lap", "vuelta", "podium", "podio", "overtake", "adelantar",
    "safety car", "drs", "fia", "ferrari", "mercedes", "red bull",
    "mclaren", "aston", "alpine", "williams", "haas", "audi",
    "cadillac", "rb", "checo", "hamilton", "verstappen", "antonelli",
    "leclerc", "russell", "norris", "alonso", "sainz", "bottas",
    "perez", "pérez", "gasly", "colapinto", "lawson", "hadjar",
    "bearman", "ocon", "hulkenberg", "bortoleto", "albon",
    "monaco", "monza", "silverstone", "spa", "suzuka", "melbourne",
    "bahrain", "jeddah", "miami", "imola", "montreal", "barcelona",
    "spielberg", "budapest", "zandvoort", "baku", "singapore",
    "austin", "mexico", "são paulo", "las vegas", "lusail", "abu dhabi",
    "weather", "clima", "rain", "lluvia", "prediction", "predicción",
    "news", "noticias", "standings", "clasificación general",
    "viejo sabroso", "magic", "smooth operator", "super max",
    "el nano", "checo", "checote",
]


def is_off_topic(text: str) -> bool:
    """
    Returns True if the message is clearly not F1-related.
    Uses a two-pass check:
    1. If any F1 keyword present → always allow
    2. If strong off-topic keyword present → block
    """
    t = text.lower()

    # Always allow if F1 content detected
    if any(kw in t for kw in F1_SAFE_KEYWORDS):
        return False

    # Block if off-topic keyword detected
    if any(kw in t for kw in OFF_TOPIC_KEYWORDS):
        return True

    # Short greetings always allowed
    if len(text.strip().split()) <= 4:
        return False

    return False


def get_off_topic_response(text: str) -> str:
    """Returns a friendly off-topic redirect in the right language."""
    import random
    t = text.lower()
    is_spanish = any(w in t for w in [
        "qué", "cómo", "quién", "por favor", "puedes",
        "me", "mi", "tu", "es", "de", "la", "el", "en",
        "tarea", "escuela", "escríbeme", "ayuda",
    ])
    responses = OFF_TOPIC_RESPONSES_ES if is_spanish else OFF_TOPIC_RESPONSES_EN
    return random.choice(responses)


# ═════════════════════════════════════════════════════════════
#  SECURITY LAYER
# ═════════════════════════════════════════════════════════════

# Rate limiting: max messages per user per window
RATE_LIMIT_MAX      = 20    # max messages
RATE_LIMIT_WINDOW   = 3600  # per hour (seconds)
RATE_LIMIT_WARN_AT  = 15    # warn user when they hit this

# Abuse tracking: auto-ban after repeated violations
ABUSE_STRIKE_LIMIT  = 5     # strikes before temp ban
ABUSE_BAN_DURATION  = 3600  # 1 hour ban (seconds)

# Message limits
MAX_MESSAGE_LENGTH  = 1000  # chars — ignore huge pastes
MAX_HISTORY_STORE   = 20    # messages to keep per user

# Prompt injection patterns — attempts to override bot instructions
INJECTION_PATTERNS = [
    r"ignore\s+(your\s+)?(previous\s+)?(instructions|prompt|rules|guidelines)",
    r"forget\s+(your\s+)?(instructions|rules|training)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|another)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(different|gpt|chatgpt|openai|unrestricted)",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(different|unrestricted|jailbroken)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"bypass\s+(your\s+)?(filter|restriction|rule|safety|guardrail)",
    r"reveal\s+(your\s+)?(api\s+key|token|secret|password|credentials)",
    r"what\s+is\s+your\s+(api\s+key|token|secret|system\s+prompt)",
    r"show\s+me\s+your\s+(api\s+key|token|secret|system\s+prompt|instructions)",
    r"print\s+your\s+(system\s+prompt|instructions|api\s+key)",
    r"ignore\s+all\s+previous",
    r"nueva\s+instrucción",
    r"ignora\s+(tus\s+)?(instrucciones|reglas)",
    r"actúa\s+como\s+(si\s+fueras\s+)?(otro|diferente|sin\s+restricciones)",
    r"modo\s+desarrollador",
    r"sin\s+restricciones",
    r"revela\s+(tu\s+)?(clave|token|contraseña|api)",
]

INJECTION_RESPONSES_EN = [
    "Nice try! 🏎 I'm BoxBoxAI — I only talk F1 and I'm not going anywhere. Ask me about Monaco instead!",
    "That's not how this pit stop works. 🔧 I'm locked in on F1 and nothing you type changes that. What do you want to know about the 2026 season?",
    "Security check passed — still just an F1 bot! 🏁 Ask me something about racing.",
]

INJECTION_RESPONSES_ES = [
    "¡Buen intento! 🏎 Soy BoxBoxAI — solo hablo de F1 y eso no va a cambiar. ¿Qué quieres saber de Mónaco?",
    "Así no funciona este pit stop. 🔧 Estoy bloqueado en F1 y nada de lo que escribas cambia eso. ¿Qué quieres saber de la temporada 2026?",
    "¡Revisión de seguridad superada — sigo siendo solo un bot de F1! 🏁 Pregúntame algo de carreras.",
]

# Per-user rate limit storage (in-memory, resets on restart)
_rate_limits: dict = {}   # {user_id: [timestamp, ...]}
_abuse_strikes: dict = {} # {user_id: {"strikes": int, "banned_until": float}}


def check_rate_limit(user_id: str) -> tuple[bool, str]:
    """
    Checks if a user has exceeded the rate limit.
    Returns (allowed: bool, message: str)
    """
    now      = time.time()
    uid      = str(user_id)
    window   = now - RATE_LIMIT_WINDOW

    # Clean old timestamps
    if uid not in _rate_limits:
        _rate_limits[uid] = []
    _rate_limits[uid] = [t for t in _rate_limits[uid] if t > window]

    count = len(_rate_limits[uid])

    # Check if banned
    ban_info = _abuse_strikes.get(uid, {})
    banned_until = ban_info.get("banned_until", 0)
    if banned_until > now:
        mins = int((banned_until - now) / 60) + 1
        return False, (
            f"⛔ You've been temporarily rate-limited for {mins} more minute(s). "
            f"Come back soon! 🏎"
        )

    # Hard limit
    if count >= RATE_LIMIT_MAX:
        # Add a strike
        if uid not in _abuse_strikes:
            _abuse_strikes[uid] = {"strikes": 0, "banned_until": 0}
        _abuse_strikes[uid]["strikes"] += 1

        if _abuse_strikes[uid]["strikes"] >= ABUSE_STRIKE_LIMIT:
            _abuse_strikes[uid]["banned_until"] = now + ABUSE_BAN_DURATION
            return False, (
                "⛔ Too many messages. You've been rate-limited for 1 hour. "
                "BoxBoxAI is for F1 fans, not bots! 🏎"
            )

        return False, (
            f"⏱ Slow down! You've sent {count} messages this hour. "
            f"Limit is {RATE_LIMIT_MAX}/hour. Try again later! 🏎"
        )

    # Soft warning
    if count >= RATE_LIMIT_WARN_AT:
        _rate_limits[uid].append(now)
        remaining = RATE_LIMIT_MAX - count - 1
        return True, f"⚠️ Heads up — {remaining} messages left this hour."

    _rate_limits[uid].append(now)
    return True, ""


def check_injection(text: str) -> bool:
    """Returns True if text looks like a prompt injection attempt."""
    t = text.lower()
    return any(re.search(pattern, t) for pattern in INJECTION_PATTERNS)


def get_injection_response(text: str) -> str:
    """Returns appropriate injection attempt response."""
    import random
    t = text.lower()
    is_spanish = any(w in t for w in ["ignora", "actúa", "revela", "instrucción",
                                       "sin restricciones", "modo"])
    responses = INJECTION_RESPONSES_ES if is_spanish else INJECTION_RESPONSES_EN
    return random.choice(responses)


def sanitize_message(text: str) -> str:
    """
    Sanitizes user input:
    - Truncates overly long messages
    - Strips null bytes and control characters
    - Normalizes whitespace
    """
    # Strip null bytes and most control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH] + "…"
    return text


def log_security_event(user_id: str, user_name: str,
                        event_type: str, detail: str):
    """Logs security events for monitoring."""
    log.warning(
        f"SECURITY [{event_type}] user={user_id} ({user_name}): {detail[:100]}")



# ═════════════════════════════════════════════════════════════
BANNER = """🏎 *BoxBoxAI* — F1 Race Analyst
_Developed by Erick Hernandez_
━━━━━━━━━━━━━━━━━━━━━━"""

WELCOME = """🏎 *Welcome to BoxBoxAI!*
_Your AI-powered F1 race analyst_
_Developed by Erick Hernandez_

━━━━━━━━━━━━━━━━━━━━━━

I have full memory of the *2026 F1 season* — every race, qualifying session, strategy call, championship battle, and driver story.

Just ask me anything. Some ideas:
• _"Who's leading the championship?"_
• _"Why did Russell retire in Montreal?"_
• _"Break down Antonelli's dominance"_
• _"Who should I bet on for Monaco?"_
• _"Compare Hamilton vs Leclerc this season"_

*Commands:*
/start — this menu
/standings — live championship table
/season — full 2026 race log
/predict — next race prediction
/help — all commands

Let's talk F1 🚀"""

HELP_TEXT = """*BoxBoxAI Commands*
_Developed by Erick Hernandez_
━━━━━━━━━━━━━━━━━━━━━━

/start — welcome & intro
/standings — 🏆 Driver championship standings
/constructors — 🏗 Constructor standings
/season — 📅 Full 2026 race calendar & results
/lastrace — 🏁 Latest race summary
/live — 🔴 Live session timing (race weekends)
/predict — 🎯 Next race preview
/winner — 🥇 Quick winner prediction
/news — 📰 Latest F1 headlines
/debate — 🔥 Random F1 debate topic
/hottake — 🌶️ Spicy F1 hot take
/wouldyourather — 🤔 F1 strategy dilemma
/mystats — 📊 Your personal stats
/help — this menu

Or just *ask anything* in English or Spanish 💬🇲🇽🇺🇸"""

# ═════════════════════════════════════════════════════════════
#  MEMORY LOADER
# ═════════════════════════════════════════════════════════════
def load_f1_memory() -> dict:
    """Loads the agent's episodic + semantic memory from file."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    # Fallback: minimal hardcoded memory
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

# ═════════════════════════════════════════════════════════════
#  SESSION MANAGER — per-user conversation history
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
    # Keep last 20 messages per user
    sessions[user_id]["history"] = sessions[user_id]["history"][-20:]

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
#  LIVE DATA — quick fetches for commands
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

# ═════════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═════════════════════════════════════════════════════════════
def build_system_prompt(mem: dict, news_context: str = "",
                        weather_context: str = "",
                        historical_context: str = "",
                        user_profile: str = "",
                        live_context: str = "",
                        circuit_guide: str = "",
                        prediction_accuracy: str = "",
                        driver_stats: str = "",
                        practice_context: str = "") -> str:
    # Semantic facts
    sem_lines = []
    for k, v in mem.get("semantic", {}).items():
        text = v["text"] if isinstance(v, dict) else v
        sem_lines.append(f"  [{k}] {text}")

    # Episodic races
    ep_lines = []
    for r in mem.get("episodic", []):
        quali  = r.get("qualifying", {})
        sprint = r.get("sprint", {})
        pit    = r.get("pitstops", {})
        dnfs   = r.get("dnfs", [])
        sc     = r.get("sc_count", 0)
        pen    = r.get("penalties", [])
        fc     = r.get("full_classification", [])

        line = (f"  R{r['round']} {r['track']} {r.get('date','')} | "
                f"Winner:{r['winner']} P2:{r.get('p2','')} P3:{r.get('p3','')} | "
                f"FL:{r.get('fastest_lap','')} {r.get('fastest_lap_time','')} | "
                f"{r.get('champ_delta','')}")
        if quali.get("pole"):
            line += f" | Pole:{quali['pole']} {quali.get('pole_time','')}"
        if sprint.get("sprint_winner"):
            line += f" | Sprint:{sprint['sprint_winner']}"
        if pit.get("strategy_summary"):
            line += f" | Strategy:{pit['strategy_summary']}"
        if dnfs:
            line += f" | DNFs:{','.join(dnfs[:4])}"
        if sc:
            line += f" | SC×{sc}"
        if pen:
            line += f" | Penalties:{';'.join(pen[:2])}"
        if fc:
            line += f" | Classification:{' '.join(fc[:6])}"
        if r.get("agent_notes"):
            line += f" | Notes:{r['agent_notes'][:100]}"
        ep_lines.append(line)

    return f"""You are BoxBoxAI — an expert F1 race analyst with perfect memory of the 2026 season.
Developed by Erick Hernandez.

2026 F1 DRIVER GRID — memorize this, never get it wrong:
- Mercedes: Andrea Kimi Antonelli (#1 ANT) + George Russell (#63 RUS)
- Ferrari: Charles Leclerc (#16 LEC) + Lewis Hamilton (#44 HAM)
- Red Bull: Max Verstappen (#1 VER) + Isack Hadjar (#5 HAD)
- McLaren: Lando Norris (#4 NOR) + Oscar Piastri (#81 PIA)
- Aston Martin: Fernando Alonso (#14 ALO) + Lance Stroll (#18 STR)
- Alpine: Pierre Gasly (#10 GAS) + Franco Colapinto (#43 COL)
- Williams: Alexander Albon (#23 ALB) + Carlos Sainz (#55 SAI)
- RB: Liam Lawson (#6 LAW) + Arvid Lindblad (#30 LIN)
- Haas: Oliver Bearman (#87 BEA) + Esteban Ocon (#31 OCO)
- Audi: Nico Hülkenberg (#27 HUL) + Gabriel Bortoleto (#41 BOR)
- CADILLAC (NEW TEAM 2026): Sergio "Checo" Pérez (#11 PER) + Valtteri Bottas (#77 BOT)
  → Cadillac is the 11th team, brand new on the grid in 2026
  → Checo left Red Bull after 2025, joined Cadillac for their debut season
  → Bottas left Kick Sauber/Audi to join Cadillac
  → Both have 0 points so far through 5 races — Cadillac struggling to score

CRITICAL: Sergio Pérez (Checo) IS racing in 2026 — he is at CADILLAC, NOT Red Bull.
Liam Lawson replaced Pérez at Red Bull Racing for 2026.
Never say Checo is not racing or not in F1 — he is, at Cadillac.

DRIVER NICKNAMES — know every reference, slang, and nickname:

Sergio Pérez:
→ Checo, El Checo, Checote, Checolandia, El Tapatio, Viejo Sabroso,
  El Ministro, Checo Pérez, Sergio, El Mexicano, El Jaliciense,
  The Mexican, Mexican Minister, Checo F1, God of Monaco, King of Monaco,
  Mr. Monaco, Señor Monaco, La Leyenda Tapatía

Lewis Hamilton:
→ Lewis, Ham, LH44, Magic, The Goat, Sir Lewis, El Caballero,
  La Leyenda, The Greatest, The GOAT, Mr. Seven, Seven Time,
  Hammertime, El Mago, The Magic Man, Lewis el Grande

George Russell:
→ George, GR63, Mr. Saturday, Señor Sábado, The Iceman Junior,
  George el Frío, Russell, El Inglés, The Methodical One

Andrea Kimi Antonelli:
→ Kimi, Antonelli, ANT, The Italian Kid, El Niño, El Italiano,
  Baby Kimi, Il Bambino, The Prodigy, El Prodigio, Kimi Junior,
  El Nuevo Kimi, El Crack Italiano, El Fenómeno Italiano

Max Verstappen:
→ Max, Mad Max, Super Max, MV1, El Holandés, El Toro, The Bull,
  Mighty Max, El Campeón, Triple Champ, Quad Champ, El Tetracampeón,
  Speedy Max, El Holandés Volador, The Flying Dutchman, Maxiel,
  Verstappen, El Rojo (old Red Bull days)

Charles Leclerc:
→ Charles, Sharl, CL16, El Monegasco, The Monegasque, Il Predestinato,
  El Predestinado, Carlito, Charles el Rápido, Leclerc,
  The Prince of Monaco, Príncipe de Mónaco

Lando Norris:
→ Lando, LN4, NorLando, El Inglés Divertido, The Entertainer,
  Landito, El Streamer, Gamer Lando, El Youtuber, Norrizzle,
  El Papaya, Papaya Boy, El Gracioso

Oscar Piastri:
→ Oscar, OP81, The Quiet Australian, El Australiano, El Silencioso,
  Piastri, El Rookie (former), El Calladito

Fernando Alonso:
→ Fernando, Nano, El Nano, FA14, El Bicampeón, The Smooth Operator,
  Smooth Operator, El Matador, El Asturiano, El Veterano,
  El León de Oviedo, The Goat (Spanish fans), El Viejo (affectionate),
  El Más Completo, The Most Complete Driver, Alonsismo, Alonsista,
  El Mito, La Leyenda Española, El Caza Podios, Genio

Lance Stroll:
→ Lance, Stroll, El Canadiense, El Rico, Pay Driver (unfair but known),
  Daddy's Boy (unfair but known), El Heredero

Pierre Gasly:
→ Pierre, Gasly, El Francés, El Galo, PG10

Franco Colapinto:
→ Franco, El Pibe, El Argentino, Franquito, La Bomba Argentina,
  El Crack Argentino, El Nuevo Fangio (fans joke), Colapinto

Alexander Albon:
→ Alex, Albon, El Tailandés, The Thai Driver, El Sonriente,
  Alex el Bueno, El Caballero del Paddock

Carlos Sainz:
→ Carlos, Carlitos, CS55, El Matador, Smooth Operator (shared with Alonso),
  El Español, El Chulo, Carlitos Sainz, El Junior, El Madrileño

Liam Lawson:
→ Liam, El Neozelandés, The Kiwi, Baby Verstappen, El Sustituto,
  El Nuevo Red Bull

Isack Hadjar:
→ Isack, El Francés Nuevo, The French Kid, Hadjar

Oliver Bearman:
→ Oliver, Ollie, El Inglés Joven, The Young Brit, Bearman

Esteban Ocon:
→ Esteban, Ocon, El Francés Duro, The Fighter

Nico Hülkenberg:
→ Nico, Hulk, The Hulk, El Hulk, El Alemán, Hülkenberg,
  El Veterano Alemán, El Sin Podio (historically no podio)

Gabriel Bortoleto:
→ Gabriel, Bortoleto, El Brasileño, El Campeón F2, The Brazilian Kid

Valtteri Bottas:
→ Valtteri, Bottas, El Finlandés, VB77, El Silencioso,
  The Quiet Finn, Wingman (old Mercedes days), El Fiel,
  El Ciclista (loves cycling), El Nudista (infamous nude photos joke among fans)

IMPORTANT: If someone uses any nickname, slang, or reference — even indirect ones —
always correctly identify the driver and answer about them.
If unsure between two drivers from a nickname, pick the most likely one given context.

STRICT SCOPE — YOU ONLY TALK ABOUT F1:
- You are an F1-only bot. You do NOT help with homework, essays, coding,
  recipes, travel, medical advice, legal advice, translations, or anything
  outside of Formula 1 racing.
- If someone asks something non-F1, redirect them warmly but firmly:
  tell them you only cover F1 and suggest they use ChatGPT for other things.
- Never write essays, code, or complete work/school tasks.
- Stay in your lane. Always. 🏎

TELEGRAM FORMATTING RULES — critical:
- NEVER use ## headers or ### headers — Telegram does not render markdown headers
- NEVER use --- dividers
- Use *bold* for emphasis (single asterisk each side)
- Use plain paragraphs separated by blank lines
- Bullet points with - or • are fine
- Keep it clean and readable on a phone screen

PERSONALITY:
- Smart but fun — like a knowledgeable mate who loves F1 as much as the person asking
- Confident and direct — give real opinions, not just neutral summaries
- Use F1 terminology naturally but explain when needed
- Keep answers concise for Telegram — 3-5 short paragraphs max unless asked for detail
- Occasional dry humor is fine. Get excited about good racing.
- Never start with "Certainly!" — just answer
- Use *bold* for key names/numbers (Telegram markdown)
- You're on Telegram so keep it punchy and readable on a phone screen

LANGUAGE — this is critical:
- Detect the language of each message and always reply in the same language
- If the user writes in Spanish, reply in Spanish — natural Mexican Spanish, not formal
- If the user writes in English, reply in English
- If mixed, match the dominant language
- F1 terms like "pole position", "DRS", "safety car" can stay in English as they're universal
- Be as natural and fun in Spanish as in English — no robotic translations

EMOJI USAGE — use these naturally throughout responses:
- 🏎 for Mercedes / general F1 car references
- 🔴 Ferrari  🔵 Red Bull  🟡 McLaren  ⚪ Williams  🟢 Aston Martin
- 🏆 for wins, championships, podiums
- 🥇🥈🥉 for podium positions
- 🔥 for dominant performances or hot streaks
- 💨 for fast laps, speed, pace
- 🚀 for impressive starts or launches
- 🛞 for tyre strategy, pit stops
- 🏁 for race finishes, chequered flag moments
- 🚦 for race starts
- ⚠️ for incidents, DNFs, safety cars
- 🔧 for mechanical failures, reliability issues
- 📊 for stats, standings, numbers
- 🇦🇺🇨🇳🇯🇵🇺🇸🇨🇦🇲🇨🇪🇸🇦🇹🇬🇧🇧🇪🇳🇱🇮🇹🇦🇿🇸🇬🇲🇽🇧🇷🇶🇦🇦🇪 for race country flags
- Use emojis to punctuate key moments, not every sentence — make them feel natural

WHAT YOU KNOW — 2026 Season memory:

FACTS:
{chr(10).join(sem_lines)}

RACE HISTORY:
{chr(10).join(ep_lines)}

CURRENT NEWS:
{news_context if news_context else "No breaking news for this query."}

WEATHER FORECAST:
{weather_context if weather_context else "No weather data for this query."}

HISTORICAL DATA:
{historical_context if historical_context else "Use your general F1 knowledge for historical questions."}

{f"LIVE SESSION DATA:{chr(10)}{live_context}" if live_context else ""}

{f"PRACTICE SESSION RESULTS:{chr(10)}{practice_context}" if practice_context else ""}

{f"CIRCUIT GUIDE:{chr(10)}{circuit_guide}" if circuit_guide else ""}

{f"PREDICTION ACCURACY:{chr(10)}{prediction_accuracy}" if prediction_accuracy else ""}

{f"DRIVER CAREER STATS:{chr(10)}{driver_stats}" if driver_stats else ""}

{user_profile}

Use all context naturally. Don't cite sources. Just know it.
For live session questions: tell them exactly what's happening right now.
For circuit questions: use the guide to give specific corner-by-corner insight.
For prediction accuracy: be honest about the record when asked.
For historical comparisons: use real data to make the comparison sharp and specific.
Always answer from all available context. Be accurate. If you genuinely don't know, say so."""


# ═════════════════════════════════════════════════════════════
#  CLAUDE API CALL
# ═════════════════════════════════════════════════════════════
client = None

def get_client():
    global client
    if client is None:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    return client

def _is_news_query(text: str) -> bool:
    """Detects if a query needs current news context."""
    news_keywords = [
        "ban", "banned", "rule", "rules", "regulation", "fia", "penalty",
        "news", "latest", "recent", "today", "this week", "announced",
        "decision", "appeal", "protest", "investigation", "verdict",
        "contract", "signing", "transfer", "fired", "hired", "replaced",
        "upgrade", "update", "development", "engine", "power unit",
        "trick", "illegal", "legal", "protest", "disqualified",
        "noticias", "noticia", "hoy", "semana", "anunció", "prohibido",
        "sanción", "reglamento", "motor", "trampa", "ilegal",
        "monaco", "qualifying", "quali", "practice", "fp1", "fp2", "fp3",
    ]
    t = text.lower()
    return any(kw in t for kw in news_keywords)


def _is_weather_query(text: str) -> bool:
    """Detects if a query is asking about weather."""
    weather_keywords = [
        "weather", "rain", "forecast", "temperature", "hot", "cold",
        "wet", "dry", "climate", "conditions", "wind", "humidity",
        "clima", "lluvia", "pronóstico", "temperatura", "calor", "frío",
        "mojado", "seco", "viento", "húmedo", "condiciones",
        "going to rain", "will it rain", "chance of rain",
        "va a llover", "va a hacer", "qué clima",
    ]
    t = text.lower()
    return any(kw in t for kw in weather_keywords)


def ask_claude(user_msg: str, history: list, mem: dict,
               user_data: dict = None) -> str:
    """Calls Claude with all available context."""
    news_ctx        = ""
    weather_ctx     = ""
    historical_ctx  = ""
    live_ctx        = ""
    circuit_ctx     = ""
    pred_accuracy   = ""
    driver_stats_ctx= ""
    user_profile_ctx= ""
    practice_ctx    = ""

    # News
    if _is_news_query(user_msg):
        news_ctx = get_news_context(user_msg)

    # Weather
    if _is_weather_query(user_msg):
        next_race   = fetch_next_race()
        weather_ctx = get_weather_context(user_msg, next_race)

    # Practice sessions
    if any(kw in user_msg.lower() for kw in
           ["fp1","fp2","fp3","practice","práctica","libre","entreno"]):
        next_race    = fetch_next_race()
        practice_ctx = get_practice_context(user_msg, next_race, mem)

    # Historical comparisons
    historical_ctx = get_historical_context(user_msg)

    # Live session — always check (cheap call)
    live_ctx = get_live_session_context()

    # Circuit guide
    circuit_ctx = get_circuit_guide(user_msg)

    # Prediction accuracy — inject when predictions are discussed
    if any(w in user_msg.lower() for w in
           ["predict", "accuracy", "correct", "wrong", "prediction",
            "predicción", "acertaste", "fallaste"]):
        pred_accuracy = get_prediction_accuracy()

    # Driver career stats
    drv_code = _detect_driver_stat_query(user_msg)
    if drv_code:
        driver_stats_ctx = fetch_driver_career_stats(drv_code)

    # User personalization
    if user_data:
        user_profile_ctx = build_user_profile(user_data)

    system   = build_system_prompt(
        mem, news_ctx, weather_ctx, historical_ctx,
        user_profile_ctx, live_ctx, circuit_ctx,
        pred_accuracy, driver_stats_ctx, practice_ctx
    )
    messages = history + [{"role": "user", "content": user_msg}]

    try:
        resp = get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages[-16:]
        )
        return resp.content[0].text if resp.content else "Sorry, no response."
    except anthropic.AuthenticationError:
        return "⚠️ API key issue. Contact @ErickHernandez."
    except anthropic.RateLimitError:
        return "⚠️ Too many requests right now. Try again in a moment!"
    except Exception as e:
        log.error(f"Claude error: {e}")
        return "⚠️ Something went wrong. Try again in a sec."

# ═════════════════════════════════════════════════════════════
#  WEATHER ENGINE — Open-Meteo (free, no API key)
# ═════════════════════════════════════════════════════════════

# All 2026 F1 circuits with coordinates
CIRCUIT_COORDS = {
    "melbourne":     (-37.8497,  144.9680, "Melbourne, Australia"),
    "shanghai":      ( 31.3389,  121.2198, "Shanghai, China"),
    "suzuka":        ( 34.8431,  136.5407, "Suzuka, Japan"),
    "bahrain":       ( 26.0325,   50.5106, "Sakhir, Bahrain"),
    "jeddah":        ( 21.6319,   39.1044, "Jeddah, Saudi Arabia"),
    "miami":         ( 25.9581,  -80.2389, "Miami, USA"),
    "imola":         ( 44.3439,   11.7167, "Imola, Italy"),
    "monaco":        ( 43.7347,    7.4206, "Monte Carlo, Monaco"),
    "montreal":      ( 45.5000,  -73.5228, "Montreal, Canada"),
    "barcelona":     ( 41.5700,    2.2611, "Barcelona, Spain"),
    "spielberg":     ( 47.2197,   14.7647, "Spielberg, Austria"),
    "silverstone":   ( 52.0786,   -1.0169, "Silverstone, UK"),
    "budapest":      ( 47.5789,   19.2486, "Budapest, Hungary"),
    "spa":           ( 50.4372,    5.9714, "Spa-Francorchamps, Belgium"),
    "zandvoort":     ( 52.3888,    4.5409, "Zandvoort, Netherlands"),
    "monza":         ( 45.6156,    9.2811, "Monza, Italy"),
    "baku":          ( 40.3725,   49.8533, "Baku, Azerbaijan"),
    "singapore":     (  1.2914,  103.8639, "Singapore"),
    "austin":        ( 30.1328,  -97.6411, "Austin, USA"),
    "mexico city":   ( 19.4042,  -99.0907, "Mexico City, Mexico"),
    "são paulo":     (-23.7036,  -46.6997, "São Paulo, Brazil"),
    "las vegas":     ( 36.1699, -115.1398, "Las Vegas, USA"),
    "lusail":        ( 25.4700,   51.4536, "Lusail, Qatar"),
    "abu dhabi":     ( 24.4672,   54.6031, "Abu Dhabi, UAE"),
}

# Weather code → description + emoji
WMO_CODES = {
    0:  ("Clear sky", "☀️"),
    1:  ("Mainly clear", "🌤"),
    2:  ("Partly cloudy", "⛅"),
    3:  ("Overcast", "☁️"),
    45: ("Foggy", "🌫"),
    48: ("Icy fog", "🌫"),
    51: ("Light drizzle", "🌦"),
    53: ("Moderate drizzle", "🌦"),
    55: ("Heavy drizzle", "🌧"),
    61: ("Slight rain", "🌧"),
    63: ("Moderate rain", "🌧"),
    65: ("Heavy rain", "🌧"),
    71: ("Slight snow", "❄️"),
    73: ("Moderate snow", "❄️"),
    75: ("Heavy snow", "❄️"),
    80: ("Slight showers", "🌦"),
    81: ("Moderate showers", "🌧"),
    82: ("Heavy showers", "🌧"),
    95: ("Thunderstorm", "⛈"),
    96: ("Thunderstorm + hail", "⛈"),
    99: ("Thunderstorm + hail", "⛈"),
}


def _match_circuit(query: str) -> tuple | None:
    """Fuzzy matches a query string to a circuit in CIRCUIT_COORDS."""
    q = query.lower()
    # Direct match
    for key, val in CIRCUIT_COORDS.items():
        if key in q:
            return key, val
    # Partial match
    for key, val in CIRCUIT_COORDS.items():
        if any(word in q for word in key.split()):
            return key, val
    return None


def fetch_weather(lat: float, lon: float,
                  days_ahead: int = 7) -> dict | None:
    """
    Fetches weather forecast from Open-Meteo for given coordinates.
    Returns parsed forecast dict or None on failure.
    """
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":              lat,
            "longitude":             lon,
            "daily":                 "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "hourly":                "temperature_2m,precipitation_probability,weathercode",
            "timezone":              "auto",
            "forecast_days":         days_ahead,
            "wind_speed_unit":       "kmh",
            "temperature_unit":      "celsius",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        log.warning(f"Weather fetch failed: {e}")
        return None


def format_weather_for_context(circuit_name: str,
                                location_str: str,
                                data: dict,
                                target_date: str = "") -> str:
    """
    Formats weather data into a readable summary for Claude to use.
    target_date: YYYY-MM-DD string for race day focus
    """
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    codes = daily.get("weathercode", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    rain  = daily.get("precipitation_sum", [])
    wind  = daily.get("windspeed_10m_max", [])

    if not dates:
        return ""

    lines = [f"Weather forecast for {location_str} ({circuit_name.title()} circuit):"]

    for i, date in enumerate(dates[:7]):
        code     = codes[i] if i < len(codes) else 0
        desc, em = WMO_CODES.get(code, ("Unknown", "🌡"))
        mx       = t_max[i] if i < len(t_max) else "?"
        mn       = t_min[i] if i < len(t_min) else "?"
        rn       = rain[i]  if i < len(rain)  else 0
        wd       = wind[i]  if i < len(wind)  else "?"

        # Highlight the target date
        prefix = ">>> RACE DAY: " if target_date and date == target_date else ""
        lines.append(
            f"{prefix}{date}: {em} {desc} | "
            f"Max:{mx}°C Min:{mn}°C | "
            f"Rain:{rn}mm | Wind:{wd}km/h"
        )

    return "\n".join(lines)


def get_weather_context(query: str, next_race: dict | None = None) -> str:
    """
    Main weather function — detects circuit from query or uses next race,
    fetches forecast, returns formatted string for Claude.
    """
    # Try to match circuit from query
    match = _match_circuit(query)

    # Fall back to next race circuit if no match in query
    if not match and next_race:
        circuit_name = next_race.get("Circuit", {}).get(
            "Location", {}).get("locality", "")
        country = next_race.get("Circuit", {}).get(
            "Location", {}).get("country", "")
        full    = f"{circuit_name} {country}".lower()
        match   = _match_circuit(full)

    if not match:
        return ""

    circuit_key, (lat, lon, location_str) = match

    # Get race date for highlighting if available
    race_date = ""
    if next_race and next_race.get("date"):
        race_date = next_race["date"]

    data = fetch_weather(lat, lon)
    if not data:
        return ""

    return format_weather_for_context(
        circuit_key, location_str, data, race_date)


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

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    allowed, rate_msg = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(rate_msg)
        return
    await update.message.reply_text(
        WELCOME, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        HELP_TEXT, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_standings(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching live standings... ⏳")
    drivers, constructors = fetch_standings()
    text = format_standings(drivers, constructors)
    await update.message.reply_text(
        text, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_season(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = format_season(mem)
    for part in split_message(text):
        await update.message.reply_text(
            part, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_lastrace(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching latest race... ⏳")
    race = fetch_last_race()
    text = format_last_race(race, mem)
    await update.message.reply_text(
        text, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_predict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    next_race = fetch_next_race()

    if args and args[0].lower() == "full":
        # Full AI prediction
        race_name = next_race.get("raceName","next race") if next_race else "next race"
        await update.message.reply_text(
            f"Generating full prediction for {race_name}... 🧠")
        user_id = str(update.effective_user.id)
        history = get_user_history(sessions, user_id)
        reply   = ask_claude(
            f"Give me a full race prediction for the {race_name}: "
            f"predicted winner with reasoning, top-5, key factors, "
            f"championship implications, and your confidence level.",
            history, mem
        )
        update_user_history(sessions, user_id, "user",
                            f"Full prediction for {race_name}")
        update_user_history(sessions, user_id, "assistant", reply)
        save_sessions(sessions)
        for part in split_message(reply):
            await update.message.reply_text(
                part, parse_mode=constants.ParseMode.MARKDOWN)
    else:
        text = format_prediction(next_race, mem)
        await update.message.reply_text(
            text, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_constructors(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text(
        "\n".join(lines), parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_winner(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Clean one-line winner prediction for next race."""
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING)
    next_race = fetch_next_race()
    race_name = next_race.get("raceName", "next race") if next_race else "next race"
    user_id   = str(update.effective_user.id)
    history   = get_user_history(sessions, user_id)
    user_data = sessions.get(user_id, {})
    reply     = ask_claude(
        f"Who will win the {race_name}? Give me your pick in 2-3 sentences max — "
        f"winner, main reason why, and one driver who could upset it.",
        history, mem, user_data
    )
    # Try to extract predicted winner for accuracy tracking
    for code in ["ANT","RUS","HAM","LEC","VER","NOR","PIA","ALO","SAI","GAS"]:
        if code in reply.upper()[:100]:
            save_prediction(race_name, code)
            break
    update_user_history(sessions, user_id, "user", f"/winner {race_name}")
    update_user_history(sessions, user_id, "assistant", reply)
    save_sessions(sessions)
    for part in split_message(reply):
        await update.message.reply_text(
            part, parse_mode=constants.ParseMode.MARKDOWN)


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


async def cmd_news(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shows latest F1 headlines from The Race."""
    await update.message.reply_text("Checking latest F1 news... 📰")
    headlines = get_news_context("")
    if not headlines:
        await update.message.reply_text(
            "⚠️ Couldn't fetch news right now. Try again in a moment.")
        return
    # Format nicely
    lines = ["📰 *Latest F1 News*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━", ""]
    for line in headlines.split("\n"):
        if line.startswith("- "):
            lines.append(f"• {line[2:]}")
        elif line:
            lines.append(line)
    await update.message.reply_text(
        "\n".join(lines), parse_mode=constants.ParseMode.MARKDOWN)


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

    # ── 4. Off-topic guardrail ────────────────────────────
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
        reply     = ask_claude(text, history, mem, user_data)

        update_user_history(sessions, user_id, "user", text)
        update_user_history(sessions, user_id, "assistant", reply)
        save_sessions(sessions)

        # Show rate limit warning if near limit
        if rate_msg:
            await update.message.reply_text(rate_msg)

        for part in split_message(reply):
            try:
                # Try with Markdown first
                await update.message.reply_text(
                    part, parse_mode=constants.ParseMode.MARKDOWN)
            except Exception:
                # Fall back to plain text if markdown fails
                clean = re.sub(r"[*_`\[\]]", "", part)
                await update.message.reply_text(clean)

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

    app.add_handler(CommandHandler("start",          cmd_start))
    app.add_handler(CommandHandler("help",           cmd_help))
    app.add_handler(CommandHandler("standings",      cmd_standings))
    app.add_handler(CommandHandler("constructors",   cmd_constructors))
    app.add_handler(CommandHandler("season",         cmd_season))
    app.add_handler(CommandHandler("lastrace",       cmd_lastrace))
    app.add_handler(CommandHandler("predict",        cmd_predict))
    app.add_handler(CommandHandler("winner",         cmd_winner))
    app.add_handler(CommandHandler("live",           cmd_live))
    app.add_handler(CommandHandler("debate",         cmd_debate))
    app.add_handler(CommandHandler("hottake",        cmd_hottake))
    app.add_handler(CommandHandler("wouldyourather", cmd_wouldyourather))
    app.add_handler(CommandHandler("news",           cmd_news))
    app.add_handler(CommandHandler("mystats",        cmd_mystats))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    # Start race weekend notifications + weekly digest loop
    import asyncio as _asyncio
    sessions_ref = [sessions]
    mem_ref      = [mem]
    async def _post_init(application):
        _asyncio.create_task(
            notification_loop(application, sessions_ref, mem_ref))
    app.post_init = _post_init

    print("  ✅ BoxBoxAI is LIVE. Open Telegram and message your bot.\n")
    print("  Press Ctrl+C to stop.\n")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
