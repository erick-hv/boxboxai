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
AUTOSPORT_RSS     = "https://www.autosport.com/rss/feed/f1"
RACEFANS_RSS      = "https://www.racefans.net/feed/"
NEWS_FEEDS        = [THE_RACE_RSS, AUTOSPORT_RSS, RACEFANS_RSS]
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
    """
    Fetches and parses F1 news from The Race, Autosport, and RaceFans.
    Returns deduplicated list of articles sorted by date.
    """
    all_articles = []
    seen_titles  = set()

    for feed_url in NEWS_FEEDS:
        try:
            r = requests.get(feed_url, timeout=10,
                             headers={"User-Agent": "BoxBoxAI/1.0"})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.text)
            for item in root.findall(".//item")[:15]:
                title   = item.findtext("title", "").strip()
                link    = item.findtext("link",  "").strip()
                pubdate = item.findtext("pubDate", "").strip()
                desc    = item.findtext("description", "")
                desc    = re.sub(r"<[^>]+>", "", desc).strip()[:300]

                # Deduplicate by title similarity
                title_key = re.sub(r"[^a-z0-9]", "", title.lower())[:40]
                if title and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_articles.append({
                        "title":   title,
                        "summary": desc,
                        "url":     link,
                        "date":    pubdate,
                        "source":  feed_url.split("/")[2].replace("www.",""),
                    })
        except Exception as e:
            log.warning(f"RSS fetch failed for {feed_url}: {e}")
            continue

    # Sort by date descending (most recent first)
    def parse_date(a):
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(a["date"]).timestamp()
        except Exception:
            return 0

    all_articles.sort(key=parse_date, reverse=True)
    return all_articles[:30]


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
    Async loop that:
    - Every 5 min: checks for sessions starting in ~15 min, sends notifications
    - Every hour: sends weekly digest on Mondays
    """
    import asyncio as _asyncio
    check_count = 0
    while True:
        try:
            # Every 5 minutes — session start notifications
            await send_session_notifications(app, sessions_ref[0])

            # Every hour (12 x 5min checks) — weekly digest
            check_count += 1
            if check_count >= 12:
                check_count = 0
                await send_weekly_digest(
                    app, sessions_ref[0], mem_ref[0])
        except Exception as e:
            log.warning(f"Notification loop error: {e}")
        await _asyncio.sleep(300)  # check every 5 minutes


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

def google_search_f1(query: str, num_results: int = 5) -> list:
    """
    Searches Google for F1 content and returns relevant results.
    Filters for trusted F1 domains.
    Returns list of {title, url, snippet}.
    """
    try:
        search_url = "https://www.google.com/search"
        headers    = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        params = {
            "q":    query,
            "num":  num_results * 2,  # request more, filter down
            "hl":   "en",
            "gl":   "us",
        }

        r = requests.get(search_url, params=params,
                         headers=headers, timeout=10)
        if r.status_code != 200:
            return []

        html = r.text
        results = []

        # Extract search result blocks
        # Google uses <div class="g"> for each result
        blocks = re.findall(
            r'<div[^>]*class="[^"]*(?:g|MjjYud)[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html, re.DOTALL)

        for block in blocks[:num_results * 3]:
            # Extract URL
            url_match = re.search(r'href="(https?://[^"]+)"', block)
            if not url_match:
                continue
            url = url_match.group(1)

            # Skip Google internal links
            if "google.com" in url or "youtube.com" in url:
                continue

            # Extract title
            title_match = re.search(r'<h3[^>]*>([^<]+)</h3>', block)
            title = title_match.group(1).strip() if title_match else ""

            # Extract snippet
            snippet_match = re.search(
                r'<span[^>]*class="[^"]*(?:st|aCOpRe|hgKElc)[^"]*"[^>]*>'
                r'([^<]+(?:<[^>]+>[^<]+</[^>]+>)*[^<]*)</span>',
                block)
            snippet = ""
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()

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
            "User-Agent": "Mozilla/5.0 BoxBoxAI/1.0"
        })
        if r.status_code != 200:
            return ""

        text = r.text
        # Remove scripts, styles, nav
        for tag in ["script", "style", "nav", "header", "footer",
                    "aside", "advertisement"]:
            text = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL)

        # Try to get article body
        for selector in [
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class="[^"]*(?:article|content|story|post)[^"]*"[^>]*>(.*?)</div>',
            r'<main[^>]*>(.*?)</main>',
        ]:
            m = re.search(selector, text, re.DOTALL)
            if m:
                text = m.group(1)
                break

        # Strip remaining HTML and clean up
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    except Exception:
        return ""


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


def _is_live_session_question(text: str) -> bool:
    """
    Detects if a question needs live search —
    anything about what's happening in a session,
    results, times, incidents for any session type.
    """
    t = text.lower()

    session_keywords = [
        # Session types
        "fp1", "fp2", "fp3", "free practice",
        "qualifying", "quali", "q1", "q2", "q3",
        "sprint quali", "sprint qualifying", "sprint race", "sprint",
        "race", "carrera",
        # Live/current action words
        "what happened", "what's happening", "what is happening",
        "results", "fastest", "fastest lap", "top time",
        "who is leading", "who's leading", "leaderboard",
        "times", "lap time", "sector", "red flag", "yellow flag",
        "incident", "crash", "dnf", "retired",
        "live", "right now", "currently", "ahora", "en vivo",
        # Spanish equivalents
        "qué pasó", "que paso", "qué está pasando",
        "resultados", "tiempos", "más rápido", "clasificación",
        "vuelta rápida", "bandera roja", "incidente",
        "líder", "primer lugar",
        # Practice specific
        "práctica", "practica", "entreno", "libre",
        # Qualifying specific
        "clasificación", "pole", "eliminado", "knocked out",
        "pole position", "grid",
    ]

    return any(kw in t for kw in session_keywords)



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


def get_session_context(query: str) -> str:
    """
    Universal session handler — covers ALL session types:
    FP1, FP2, FP3, Qualifying, Sprint Qualifying, Sprint Race,
    live or completed, current or past race weekend.
    Uses news search — reliable regardless of API availability.
    """
    q = query.lower()

    # Detect session type
    session_type = ""
    if any(kw in q for kw in ["sprint quali", "sprint qualifying", "sq", "clasificación sprint"]):
        session_type = "Sprint Qualifying"
    elif any(kw in q for kw in ["sprint race", "sprint", "carrera sprint"]):
        session_type = "Sprint Race"
    elif any(kw in q for kw in ["qualifying", "quali", "clasificación", "q1", "q2", "q3", "pole"]):
        session_type = "Qualifying"
    elif any(kw in q for kw in ["fp1", "practice 1", "libre 1", "p1"]):
        session_type = "FP1 Free Practice 1"
    elif any(kw in q for kw in ["fp2", "practice 2", "libre 2", "p2"]):
        session_type = "FP2 Free Practice 2"
    elif any(kw in q for kw in ["fp3", "practice 3", "libre 3", "p3"]):
        session_type = "FP3 Free Practice 3"
    elif any(kw in q for kw in ["practice", "práctica", "libre", "entreno", "entrenamiento"]):
        session_type = "Free Practice"
    else:
        return ""

    # Detect which race — check query for circuit names first
    race_name = ""
    circuit_keywords = {
        "monaco": "Monaco Grand Prix",
        "mónaco": "Monaco Grand Prix",
        "barcelona": "Spanish Grand Prix",
        "spain": "Spanish Grand Prix",
        "españa": "Spanish Grand Prix",
        "austria": "Austrian Grand Prix",
        "spielberg": "Austrian Grand Prix",
        "silverstone": "British Grand Prix",
        "britain": "British Grand Prix",
        "spa": "Belgian Grand Prix",
        "belgium": "Belgian Grand Prix",
        "budapest": "Hungarian Grand Prix",
        "hungary": "Hungarian Grand Prix",
        "zandvoort": "Dutch Grand Prix",
        "monza": "Italian Grand Prix",
        "italy": "Italian Grand Prix",
        "baku": "Azerbaijan Grand Prix",
        "singapore": "Singapore Grand Prix",
        "austin": "United States Grand Prix",
        "mexico": "Mexico City Grand Prix",
        "são paulo": "São Paulo Grand Prix",
        "brazil": "São Paulo Grand Prix",
        "las vegas": "Las Vegas Grand Prix",
        "lusail": "Qatar Grand Prix",
        "qatar": "Qatar Grand Prix",
        "abu dhabi": "Abu Dhabi Grand Prix",
        "montreal": "Canadian Grand Prix",
        "canada": "Canadian Grand Prix",
        "suzuka": "Japanese Grand Prix",
        "japan": "Japanese Grand Prix",
        "shanghai": "Chinese Grand Prix",
        "china": "Chinese Grand Prix",
        "melbourne": "Australian Grand Prix",
        "australia": "Australian Grand Prix",
        "miami": "Miami Grand Prix",
        "imola": "Emilia Romagna Grand Prix",
        "jeddah": "Saudi Arabian Grand Prix",
        "bahrain": "Bahrain Grand Prix",
    }
    for kw, name in circuit_keywords.items():
        if kw in q:
            race_name = name
            break

    # Fall back to current race weekend
    if not race_name:
        current = fetch_current_race()
        if current:
            race_name = current.get("raceName", "")

    search_query = f"{race_name} {session_type} 2026 results"

    # Search news feeds
    news = get_news_context(search_query)
    if news:
        return f"{race_name} — {session_type}:\n{news}"

    # DuckDuckGo fallback
    try:
        r = requests.get("https://api.duckduckgo.com/", params={
            "q": search_query, "format": "json", "no_html": "1"
        }, timeout=8)
        if r.status_code == 200:
            data  = r.json()
            parts = []
            if data.get("AbstractText"):
                parts.append(data["AbstractText"])
            for t in data.get("RelatedTopics", [])[:3]:
                if isinstance(t, dict) and t.get("Text"):
                    parts.append(t["Text"])
            if parts:
                return f"{race_name} — {session_type}:\n{' '.join(parts)[:800]}"
    except Exception:
        pass

    return ""



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
    Checks OpenF1 for any currently active or very recent session.
    Returns session info if something is live or ended within last 2 hours.
    """
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        data  = fetch_openf1("sessions", {
            "year":       SEASON,
            "date_start": today,
        })
        if not data:
            # Try yesterday too for timezone differences
            yesterday = (datetime.utcnow().replace(hour=0,minute=0,second=0)
                        .strftime("%Y-%m-%d"))
            data = fetch_openf1("sessions", {
                "year":       SEASON,
                "date_start": yesterday,
            })
        if not data:
            return None

        now = datetime.utcnow()
        best = None

        for s in sorted(data, key=lambda x: x.get("date_start",""), reverse=True):
            start_str = s.get("date_start", "")
            end_str   = s.get("date_end",   "")
            try:
                start_dt = datetime.fromisoformat(start_str.replace("Z",""))
                end_dt   = datetime.fromisoformat(end_str.replace("Z",""))

                # Currently live
                if start_dt <= now <= end_dt:
                    return s

                # Ended within last 3 hours — still relevant
                hours_ago = (now - end_dt).total_seconds() / 3600
                if 0 <= hours_ago <= 3:
                    if best is None:
                        best = s
            except Exception:
                continue

        return best
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
    "melbourne": """
Albert Park, Australia — Season opener, semi-street circuit
- 58 laps, 5.278km, 16 corners
- Medium downforce, park circuit through Albert Park lake
- Turn 1 and 3 are big overtaking spots, Turn 9/10 chicane crucial
- Safety car common at Turn 1 on lap 1 — always drama on opening lap
- Tyres: medium degradation, usually 1-stop but 2-stop possible in heat
- Weather: Melbourne autumn, can be changeable, occasional rain
- Strategy: undercut works well here — pit early if you have pace
""",
    "shanghai": """
Shanghai International Circuit, China
- 56 laps, 5.451km, 16 corners
- Long back straight with hairpin — best DRS/overtaking of the season
- Sector 1: sweeping high-speed esses, very demanding on front tyres
- Sector 3: long Turn 16 leads onto pit straight — tyre stress
- Tyres: high deg, especially rear. 2-stop common strategy
- Sprint weekend in 2026
- Strategy: the hairpin makes undercuts very effective
- Key battle zone: Turn 14 hairpin — heavy braking, lots of passes
""",
    "suzuka": """
Suzuka Circuit, Japan — Figure-8 layout, technical masterpiece
- 53 laps, 5.807km, 18 corners
- Sector 1: S-curves — ultra-high-speed commitment, terrifying and beautiful
- 130R — one of fastest corners in F1, flat for top cars
- Spoon Curve: double apex, key for sector 3 exit speed
- Degner 1&2: blind exit, crucial for sector time
- Casino Triangle: tricky chicane, easy to run wide
- Overtaking: DRS on main straight, Turn 1 works but difficult
- Tyres: medium-high deg on rears through S-curves
- Weather: Japanese autumn, can be wet — the 2022 rain race happened here
""",
    "bahrain": """
Bahrain International Circuit, Sakhir — Night race
- 57 laps, 5.412km, 15 corners
- Very abrasive surface — one of hardest on tyres all season
- Turns 1-4 complex: key overtaking zone, cars go side by side
- Turn 10 hairpin: best overtaking spot, long run to braking zone
- Turn 14-15 chicane: DRS activation — second big pass opportunity
- Tyres: very high degradation. 2-stop almost mandatory. Rear limited
- Atmosphere: incredible under floodlights in the desert
- Strategy: pit window crucial, undercut is king at Turn 1
""",
    "jeddah": """
Jeddah Corniche Circuit, Saudi Arabia — Night street circuit
- 50 laps, 6.174km — one of longest circuits
- Fastest street circuit in F1. Terrifyingly quick walls everywhere
- 27 corners, most taken flat or near-flat at 200-300 km/h
- Very low downforce setup — all about top speed
- Sector 2: blind chicanes, near-zero margin for error
- Overtaking: mainly Turn 1 and DRS zones — limited due to speed
- Safety car almost certain — walls catch everything
- Tyres: low deg due to smooth surface despite the speed
- Strategy: typically 1-stop, SC window timing is critical
""",
    "miami": """
Miami International Autodrome, USA — Sprint weekend
- 57 laps, 5.412km, 19 corners
- Street-style circuit around Hard Rock Stadium
- Turns 11-16: technical middle sector, very tight, safety car zone
- Main straight DRS: biggest passing zone
- Turn 1 braking: overtaking possible but risky
- Tyres: medium degradation, 1-stop viable but 2-stop faster
- Sprint weekend in 2026
- Atmosphere: American F1 party, big crowd energy
- Strategy: track position very important, undercut works here
""",
    "imola": """
Autodromo Enzo e Dino Ferrari, Imola, Italy
- 63 laps, 4.909km, 19 corners
- Very old school, narrow circuit — extremely limited overtaking
- Tamburello chicane: high-speed approach, site of Senna's 1994 accident
- Variante Alta: crucial chicane for sector 2 time
- Rivazza corners: double left-hander, key for lap time
- Raidillon-style Piratella: fast left-hander, respect required
- Overtaking: virtually impossible outside DRS zone at Turn 2
- Tyres: medium deg, typically 1-stop
- Strategy: qualifying position is almost everything here
""",
    "monaco": """
Circuit de Monaco — The most famous street circuit
- 78 laps, 3.337km, 19 corners
- Overtaking difficulty: 9.5/10 — almost impossible once positions set
- Qualifying is EVERYTHING — pole wins ~50% of the time
- Sainte Devote (T1): crash magnet on lap 1, tight right-hander
- Massenet-Casino: fast left into Casino Square right, commitment needed
- Grand Hotel Hairpin: tightest corner in F1, 15 km/h
- Tunnel: unique in F1, blind exit onto chicane
- Nouvelle Chicane: traditional collision point, marshal's nightmare
- Tabac: fast left, walls either side, no run-off
- Swimming Pool complex: S-section, easy to clip the wall
- Rascasse: final hairpin, notorious for time-wasting in qualifying
- Tyres: low deg due to low speeds, usually 1-stop
- Strategy: SC timing is everything — all pit stops happen under SC
- Checo is the God of Monaco — 2x wins, perfect car placement on walls
""",
    "montreal": """
Circuit Gilles Villeneuve, Canada
- 70 laps, 4.361km, 14 corners
- Semi-street circuit on an island in the St. Lawrence River
- Wall of Champions (T13-14): claims world champions every year
- Casino Hairpin: best overtaking spot, very late braking zone
- Long back straight with chicane: high-speed braking, DRS zone
- Pit lane exit: merges aggressively, causes incidents
- Tyres: medium deg. Mixed 1 and 2-stop strategies common
- Weather: can be anything in Canadian June — wet races common
- Safety car probability: very high due to Wall of Champions
- Strategy: SC timing critical, fuel-saving important here
""",
    "barcelona": """
Circuit de Barcelona-Catalunya, Spain
- 66 laps, 4.655km, 16 corners
- Development circuit — extensively used for testing, teams know it cold
- Turn 1-2: classic overtaking point, heavy braking from high speed
- Turn 3 Renault: long high-speed right-hander, downforce critical
- Turn 9: hardest corner — long, decreasing radius, rear stressed
- DRS zones: main straight and back straight
- Tyres: very high rear degradation. 2-stop almost always faster
- Strategy: tyre management defines the race. Undercut very effective
- Dirty air problem: very hard to follow through high-speed corners
""",
    "spielberg": """
Red Bull Ring, Spielberg, Austria
- 71 laps, 4.318km — shortest lap on calendar
- Very short lap but incredibly fast, set in beautiful mountains
- Turns 3-4: massive uphill braking zone — wheel-to-wheel frequent
- Turn 7-8: final two corners before long straight — DRS key
- Very high downforce circuit despite short lap
- Tyres: high stress on short lap, lots of laps = high total deg
- Austrian Grand Prix atmosphere: Orange army (Max fans) everywhere
- Weather: Styrian mountains bring rain frequently
- Overtaking: Turn 3 and Turn 4 are both viable — good racing
""",
    "silverstone": """
Silverstone, UK — High-speed temple
- 52 laps, 5.891km, 18 corners
- Copse: flat-out at 290+ km/h for top cars. THE commitment corner
- Maggotts-Becketts-Chapel: iconic S-sequence, ultra-high speed
- Hangar Straight + Stowe: biggest overtaking opportunity
- Vale-Club complex: technical final sector
- Wellington Straight: second DRS zone, good passing
- Tyres: very hard on rear tyres through high-speed corners
- British fans: incredible atmosphere, biggest F1 crowd of the year
- Weather: British summer = anything. Rain very common
- Strategy: 2-stop almost mandatory. Tyre deg very high
""",
    "budapest": """
Hungaroring, Budapest, Hungary
- 70 laps, 4.381km, 14 corners
- Monaco without the walls — extremely difficult to overtake
- Very twisty, high-downforce circuit, 0.95 overtaking difficulty
- Turn 1-2 complex: only real overtaking zone, very late braking
- Turn 4: hairpin entry, can catch someone napping
- Turns 11-12: technical complex before long straight
- Tyres: medium deg due to low speeds but very long lap
- Strategy: qualifying position dominates. Overcut possible due to long pit lane
- Atmosphere: boiling hot European summer, always 35°C+
""",
    "spa": """
Circuit de Spa-Francorchamps, Belgium — Greatest circuit on the calendar
- 44 laps, 7.004km — longest circuit on calendar
- Eau Rouge/Raidillon: most iconic sequence in F1. Terrifying uphill flat-out
- Pouhon: double-apex high-speed left, one of best corners in racing
- Blanchimont: 300 km/h left before Bus Stop chicane
- Kemmel Straight: longest DRS zone, massive overtaking opportunity
- Bus Stop chicane: final complex, under/overcut zone
- Weather: Spa has own microclimate — can be wet sector 1, dry sector 3
- Tyres: high deg on long circuit. 2-stop typically
- Atmosphere: forest setting, camping fans, legendary vibe
- History: Belgian GP has produced some of the greatest F1 moments
""",
    "zandvoort": """
Circuit Zandvoort, Netherlands — Seaside banked circuit
- 72 laps, 4.259km, 14 corners
- Banked final corner (Arie Luyendyk): car sticks to wall through banking
- Hugkenheim: banked second-to-last corner, unique sensation for drivers
- Turn 3 chicane: main overtaking zone
- Very narrow, difficult to follow — limited passing opportunities
- Tyres: medium deg. 2-stop typically used
- Orange army (Dutch fans): incredible atmosphere, sea of orange
- Dune setting: sand can get blown onto circuit, grip changes
- Strategy: VSC/SC very common — narrow track = incidents
""",
    "monza": """
Autodromo Nazionale Monza, Italy — Temple of Speed
- 53 laps, 5.793km — fastest average speed of the season
- Lowest downforce setup of the year — teams run near-zero wing
- Rettifilo straight: 350+ km/h before Turn 1 braking
- Curve Grande: fast right taken flat, slight compression
- Lesmo 1&2: right-handers into the forest, tricky
- Ascari chicane: crucial complex, get it wrong and lose lots
- Parabolica (Curva Biassono): long sweeping final corner, 270° arc
- Slipstreaming huge — qualifying can produce amazing battles
- Tyres: low deg despite high speeds — smooth surface
- Strategy: 1-stop almost always. DRS dominant here
- Tifosi (Ferrari fans): incredible passion, orange smoke and flags everywhere
""",
    "baku": """
Baku City Circuit, Azerbaijan — Street circuit chaos
- 51 laps, 6.003km
- Second longest straight in F1 — 2.2km along the Caspian seafront
- Castle section: narrow medieval streets, 7-8 metres wide at tightest
- Turn 8: notorious blind entry, claimed many victims
- Turn 15-16: complex before long straight, crucial for lap time
- Overtaking: lots of it. Long straight + DRS = massive speed differential
- Tyres: low deg on smooth streets, 1-stop typical
- Safety car: virtually guaranteed every year. Chaos track
- Strategy: SC window management critical — timing the VSC/SC pit
- Famous for: bizarre results, random retirements, late drama
""",
    "singapore": """
Marina Bay Street Circuit, Singapore — Night race in the city
- 62 laps, 4.940km
- Hottest race of the year — 35°C with 80%+ humidity
- Very technical, low-speed street circuit
- Anderson Bridge section: historic race track feeling
- Esplanade Drive: wall of fame claimed many front wings
- Turn 18-22: complex chicanes, high SC probability
- Overtaking: extremely difficult. Qualifying crucial
- Tyres: low deg due to low speeds. 1-stop standard
- Strategy: SC almost certain — timing everything
- Atmosphere: stunning night backdrop of Marina Bay skyline
""",
    "austin": """
Circuit of the Americas, Texas, USA
- 56 laps, 5.513km, 20 corners
- Turn 1: uphill, blind apex approach, best mass passing spot
- Sectors 1 and 2: flowing high-speed corners inspired by classic circuits
- Turns 12-13: inspired by Maggotts-Becketts, high-speed esses
- Back straight: DRS zone into Turn 12 hairpin
- Tyres: hard on rear through high-speed complex
- 2-stop usually faster but 1-stop possible
- Sprint weekend some years
- American crowd: huge fan base now, COTA has become legendary venue
""",
    "mexico city": """
Autodromo Hermanos Rodriguez, Mexico City
- 71 laps, 4.304km, 17 corners
- Altitude: 2,285m above sea level — engines produce ~20% less power
- Power units stressed significantly — cooling critical
- Peraltada: famous banked final corner, tight stadium section after
- Foro Sol stadium section: incredible atmosphere, tight and twisty
- Main straight: DRS helps but altitude reduces effect
- Tyres: medium deg. 1-stop usually optimal
- Strategy: unique altitude effect means teams test things here
- Crowd: Mexican fans loudest of the year — Checo home race atmosphere
- Pit lane: very long, overcut strategy interesting
""",
    "são paulo": """
Autodromo Jose Carlos Pace, Interlagos, Brazil
- 71 laps, 4.309km, 15 corners
- One of greatest circuits — anti-clockwise (unusual), very old school
- Senna S: iconic double right-hander at turn 1-2, beautiful and dangerous
- Descida do Lago: blind downhill entry into hairpin
- Curva do Sol: sweeping right in sector 2
- Arquibancadas: amphitheatre feel, fans incredibly close to track
- Weather: tropical, rain almost guaranteed at some point in race week
- Overtaking: Turns 1 and 4 both viable, very exciting racing
- Tyres: high degradation, often 2-stop
- Sprint weekend some years
- Atmosphere: one of the absolute best of the year
""",
    "las vegas": """
Las Vegas Strip Circuit, USA — Night race on famous Strip
- 50 laps, 6.201km — one of longest circuits
- The Strip straight: 1.9km flat-out alongside the casinos
- Very high speed circuit — second fastest average after Monza
- Low downforce setup needed
- Thomas and Mack section: technical stadium complex
- Tyres: very low deg on smooth strip tarmac
- Strategy: typically 1-stop on low-deg surface
- Weather: November desert cold — can be surprisingly cold at night
- Atmosphere: unique spectacle, gambling capital of world as backdrop
- New to calendar: still finding its identity as a racing venue
""",
    "lusail": """
Lusail International Circuit, Qatar — Night race
- 57 laps, 5.380km, 16 corners
- Very fast, flowing circuit — high-speed sweepers throughout
- Turns 1-6: long high-speed complex, committed driving required
- Back section: more technical, slower corners for contrast
- DRS zone: main straight into Turn 1 — decent overtaking
- Tyres: very high deg — blistering issues historically
- 2-stop almost mandatory. Tyre management critical
- Sprint weekend in 2026
- Atmosphere: growing fan base in Middle East
""",
    "abu dhabi": """
Yas Marina Circuit, Abu Dhabi — Season finale, night race
- 58 laps, 5.281km, 16 corners
- Updated 2021: faster, more overtaking-friendly after redesign
- Turn 5-6-7: main technical complex, tight and twisty
- Back straight: DRS zone, reasonable overtaking
- Marina section: night race with hotel, yachts — spectacular backdrop
- Tyres: medium deg, 1-stop typical but 2-stop possible
- Strategy: final race, teams sometimes gamble for championship
- Atmosphere: unique finale feel — end of season emotion
- History: 2021 Hamilton-Verstappen title decider happened here
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


def _detect_language(text: str) -> str:
    """Returns 'es' for Spanish, 'en' for English based on content."""
    spanish_indicators = [
        "qué", "que ", "cómo", "como ", "quién", "quien", "por favor",
        "puedes", "puede", "tienes", "tiene", "eres", "hacer", "haz",
        "dime", "dame", "estás", "esta ", "este ", "ese ", "esa ",
        "ignora", "actúa", "actua", "revela", "instrucción", "instruccion",
        "sin restricciones", "olvida", "nueva ", "nuevo ",
        "tarea", "escuela", "ayuda", "necesito", "quiero", "soy ",
        "del ", "las ", "los ", "una ", "unos ", "para ", "pero ",
    ]
    t = text.lower()
    score = sum(1 for w in spanish_indicators if w in t)
    return "es" if score >= 1 else "en"


def get_off_topic_response(text: str) -> str:
    """Returns off-topic redirect in the same language as the message."""
    import random
    lang = _detect_language(text)
    responses = OFF_TOPIC_RESPONSES_ES if lang == "es" else OFF_TOPIC_RESPONSES_EN
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
    """Returns injection response in the same language as the attempt."""
    import random
    lang = _detect_language(text)
    responses = INJECTION_RESPONSES_ES if lang == "es" else INJECTION_RESPONSES_EN
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

# ── Bot owner — alerts go only here ──────────────────────────
BOT_OWNER_ID = "8892063151"

async def alert_owner(app, message: str):
    """Sends a private alert only to the bot owner (Erick)."""
    try:
        await app.bot.send_message(
            chat_id=BOT_OWNER_ID,
            text=f"🚨 *BoxBoxAI Alert*\n\n{message}",
            parse_mode="Markdown"
        )
    except Exception as e:
        log.error(f"Failed to alert owner: {e}")


# Global app reference for alerts from non-async contexts
_app_ref: list = [None]

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
/timezone — 🌍 Set your timezone for notifications
/standings — 🏆 Driver championship standings
/constructors — 🏗 Constructor standings
/season — 📅 Full 2026 season results
/lastrace — 🏁 Latest race summary
/live — 🔴 Live session timing
/predict — 🎯 Next race preview
/winner — 🥇 Quick winner prediction
/compare — ⚖️ Compare two drivers
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

def save_f1_memory(mem: dict):
    """Saves updated memory to file."""
    try:
        MEMORY_FILE.write_text(json.dumps(mem, indent=2))
        log.info("Memory saved successfully")
    except Exception as e:
        log.error(f"Failed to save memory: {e}")


AUTO_INGEST_FILE = Path(__file__).parent / "boxboxai_auto_ingest.json"

def load_ingest_state() -> dict:
    if AUTO_INGEST_FILE.exists():
        try:
            return json.loads(AUTO_INGEST_FILE.read_text())
        except Exception:
            pass
    return {"last_ingested_round": 0}

def save_ingest_state(state: dict):
    try:
        AUTO_INGEST_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.error(f"Failed to save ingest state: {e}")


def fetch_race_result(round_num: int, season: int = SEASON) -> dict | None:
    """
    Fetches full race result for a given round from Jolpica.
    Returns structured dict ready for memory storage.
    """
    try:
        data = safe_get(
            f"{JOLPICA}/{season}/{round_num}/results.json")
        if not data:
            return None

        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None

        race     = races[0]
        results  = race.get("Results", [])
        if not results:
            return None

        race_name = race.get("raceName", "")
        race_date = race.get("date", "")
        circuit   = race.get("Circuit", {}).get("circuitName", "")

        # Extract top 3
        def get_driver(r):
            d = r.get("Driver", {})
            return d.get("code", d.get("familyName", "?"))

        p1 = get_driver(results[0]) if len(results) > 0 else "?"
        p2 = get_driver(results[1]) if len(results) > 1 else "?"
        p3 = get_driver(results[2]) if len(results) > 2 else "?"

        # Extract DNFs
        dnfs = [
            f"{get_driver(r)} ({r.get('status','')})"
            for r in results
            if r.get("status", "Finished") not in
               ["Finished", "+1 Lap", "+2 Laps", "+3 Laps"]
        ]

        # Fastest lap
        fastest = next(
            (get_driver(r) for r in results
             if r.get("FastestLap", {}).get("rank") == "1"), None)

        # Pole position from grid
        pole = next(
            (get_driver(r) for r in results
             if r.get("grid") == "1"), None)

        # Championship standings after this race
        standings_data = safe_get(
            f"{JOLPICA}/{season}/{round_num}/driverStandings.json")
        champ_str = ""
        if standings_data:
            standings = (standings_data.get("MRData", {})
                        .get("StandingsTable", {})
                        .get("StandingsLists", [{}])[0]
                        .get("DriverStandings", []))
            if standings:
                top3 = standings[:3]
                champ_str = " | ".join(
                    f"{s['Driver']['code']} {s['points']}pts"
                    for s in top3
                )

        story_parts = [f"{p1} wins {race_name}."]
        if p2 and p3:
            story_parts.append(f"P2: {p2}, P3: {p3}.")
        if dnfs:
            story_parts.append(f"DNFs: {', '.join(dnfs[:3])}.")
        if fastest:
            story_parts.append(f"Fastest lap: {fastest}.")

        return {
            "round":       round_num,
            "track":       circuit,
            "race_name":   race_name,
            "date":        race_date,
            "winner":      p1,
            "p2":          p2,
            "p3":          p3,
            "pole":        pole or "",
            "fastest_lap": fastest or "",
            "dnfs":        dnfs,
            "story":       " ".join(story_parts),
            "champ_after": champ_str,
            "ingested_at": datetime.now().isoformat(),
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


# Sprint weekends in 2026
SPRINT_ROUNDS_2026 = {2, 4, 8, 16, 18, 20}  # Chinese, Miami, Austrian, US, São Paulo, Qatar


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
        (14, "Spanish Grand Prix",        "2026-09-13"),
        (15, "Singapore Grand Prix",      "2026-09-20"),
        (16, "Azerbaijan Grand Prix",     "2026-09-27"),
        (17, "United States Grand Prix",  "2026-10-18"),
        (18, "Mexico City Grand Prix",    "2026-10-25"),
        (19, "São Paulo Grand Prix",      "2026-11-08"),
        (20, "Las Vegas Grand Prix",      "2026-11-21"),
        (21, "Qatar Grand Prix",          "2026-11-29"),
        (22, "Abu Dhabi Grand Prix",      "2026-12-06"),
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
                        practice_context: str = "",
                        live_search_context: str = "") -> str:
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

{f"LIVE SESSION SEARCH RESULTS:{chr(10)}{live_search_context}" if live_search_context else ""}

{f"LIVE SESSION DATA:{chr(10)}{live_context}" if live_context else ""}

{f"PRACTICE SESSION RESULTS:{chr(10)}{practice_context}" if practice_context else ""}

{f"CIRCUIT GUIDE:{chr(10)}{circuit_guide}" if circuit_guide else ""}

{f"PREDICTION ACCURACY:{chr(10)}{prediction_accuracy}" if prediction_accuracy else ""}

{f"DRIVER CAREER STATS:{chr(10)}{driver_stats}" if driver_stats else ""}

{user_profile}

Use all context naturally. Don't cite sources. Just know it.
For live or ongoing sessions:
- Never tell users to go to the F1 app or other websites for live timing
- Use LIVE SESSION SEARCH RESULTS above to answer session questions
- Give specific times, positions, incidents from the search results
- If Q1 just finished and Q2 is live, tell them what happened in Q1
- Be specific — lap times, who got knocked out, fastest driver, incidents
- Say "based on latest reports" naturally, not as a disclaimer


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
    live_search_ctx = ""

    # Universal live search — triggers for ANY session question
    if _is_live_session_question(user_msg):
        live_search_ctx = live_search_f1(user_msg)
        log.info(f"Live search triggered for: {user_msg[:50]}")

    # News
    if _is_news_query(user_msg):
        news_ctx = get_news_context(user_msg)

    # Practice also triggers news search
    if any(kw in user_msg.lower() for kw in
           ["fp1","fp2","fp3","practice","práctica","libre","entreno",
            "qualifying","quali","clasificación","sprint"]):
        if not news_ctx:
            news_ctx = get_news_context(user_msg)

    # Weather — use current race weekend not next race
    if _is_weather_query(user_msg):
        current_race = fetch_current_race()
        weather_ctx  = get_weather_context(user_msg, current_race)

    # Universal session handler — FP1/FP2/FP3/Quali/Sprint/live/past
    session_ctx = get_session_context(user_msg)
    if session_ctx:
        practice_ctx = session_ctx
    elif any(kw in user_msg.lower() for kw in
             ["fp1","fp2","fp3","practice","práctica","libre","entreno",
              "qualifying","quali","clasificación","sprint","q1","q2","q3"]):
        if not news_ctx:
            news_ctx = get_news_context(user_msg)

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
        pred_accuracy, driver_stats_ctx, practice_ctx,
        live_search_ctx
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
        asyncio.create_task(alert_owner(
            _app_ref[0],
            "⚠️ *API Authentication Error*\n\n"
            "Your Anthropic API key is invalid or expired.\n"
            "Check console.anthropic.com and update the key in Railway Variables."
        )) if _app_ref[0] else None
        return "⚠️ API key issue. Contact @ErickHernandez."
    except anthropic.RateLimitError:
        asyncio.create_task(alert_owner(
            _app_ref[0],
            "💳 *API Credits Low or Rate Limited*\n\n"
            "You may be running low on API credits or hitting rate limits.\n"
            "Check console.anthropic.com/billing to top up."
        )) if _app_ref[0] else None
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

    # Initialize session if new user
    if user_id not in sessions:
        sessions[user_id] = {
            "history":    [],
            "first_seen": datetime.now().isoformat(),
            "stats":      {"total_messages": 0, "favorite_topics": {},
                           "commands_used": {}, "last_active": None},
        }
        save_sessions(sessions)

    await update.message.reply_text(
        WELCOME, parse_mode=constants.ParseMode.MARKDOWN)

    # Ask for timezone if not set yet
    if "tz_offset" not in sessions.get(user_id, {}):
        _tz_pending[user_id] = True
        await update.message.reply_text(
            "🌍 *One quick thing!*\n\n"
            "In which country are you located? "
            "Just type it and I'll set your timezone automatically "
            "so notifications show in your local time 🕐\n\n"
            "_Examples: Mexico, Spain, Brazil, Japan, USA..._",
            parse_mode=constants.ParseMode.MARKDOWN
        )


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

    # ── 4. Timezone pending — user just typed their country ──
    if _tz_pending.get(user_id):
        tz_result = lookup_country_tz(text)
        if tz_result:
            offset, label = tz_result
            sessions[user_id]["tz_offset"] = offset
            sessions[user_id]["tz_label"]  = label
            save_sessions(sessions)
            _tz_pending.pop(user_id, None)
            # Calculate current UTC time in their timezone as example
            now_utc = datetime.utcnow()
            local_h = (now_utc.hour + offset) % 24
            await update.message.reply_text(
                f"✅ Got it — *{label}*!\n\n"
                f"Your current local time: *{local_h:02d}:{now_utc.minute:02d}*\n\n"
                f"I'll show all session times in your timezone from now on. "
                f"Let's talk F1! 🏎🇲🇽",
                parse_mode=constants.ParseMode.MARKDOWN
            )
            return
        else:
            # Country not recognized — ask again
            await update.message.reply_text(
                f"🤔 I didn't recognize *{text}* as a location.\n\n"
                f"Try typing just the country name, like:\n"
                f"_Mexico, Spain, Brazil, Japan, USA, UK, Australia..._",
                parse_mode=constants.ParseMode.MARKDOWN
            )
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

    from telegram.ext import CallbackQueryHandler
    app.add_handler(CommandHandler("start",          cmd_start))
    app.add_handler(CommandHandler("help",           cmd_help))
    app.add_handler(CommandHandler("timezone",       cmd_timezone))
    app.add_handler(CommandHandler("standings",      cmd_standings))
    app.add_handler(CommandHandler("constructors",   cmd_constructors))
    app.add_handler(CommandHandler("season",         cmd_season))
    app.add_handler(CommandHandler("lastrace",       cmd_lastrace))
    app.add_handler(CommandHandler("live",           cmd_live))
    app.add_handler(CommandHandler("predict",        cmd_predict))
    app.add_handler(CommandHandler("winner",         cmd_winner))
    app.add_handler(CommandHandler("compare",        cmd_compare))
    app.add_handler(CommandHandler("debate",         cmd_debate))
    app.add_handler(CommandHandler("hottake",        cmd_hottake))
    app.add_handler(CommandHandler("wouldyourather", cmd_wouldyourather))
    app.add_handler(CommandHandler("news",           cmd_news))
    app.add_handler(CommandHandler("mystats",        cmd_mystats))
    app.add_handler(CallbackQueryHandler(handle_timezone_callback, pattern="^tz:"))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    # Start race weekend notifications + weekly digest loop
    import asyncio as _asyncio
    sessions_ref    = [sessions]
    mem_ref         = [mem]
    _app_ref[0]     = app  # wire for alerts

    async def _post_init(application):
        _app_ref[0] = application
        _asyncio.create_task(
            notification_loop(application, sessions_ref, mem_ref))
        _asyncio.create_task(
            auto_ingest_loop(mem_ref, application, sessions_ref))
        # Send startup alert to owner
        await alert_owner(application,
            f"✅ *BoxBoxAI is online*\n\n"
            f"Memory: {len(mem.get('episodic',[]))} races ingested\n"
            f"Users: {len(sessions)} tracked\n"
            f"Ready to go! 🏎"
        )
    app.post_init = _post_init

    print("  ✅ BoxBoxAI is LIVE. Open Telegram and message your bot.\n")
    print("  Press Ctrl+C to stop.\n")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
