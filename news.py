"""News subsystem for BoxBoxAI: RSS fetching, caching, and query-based retrieval."""
import json
import logging
import re
import time
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import requests

log = logging.getLogger(__name__)

THE_RACE_RSS      = "https://the-race.com/feed/"
AUTOSPORT_RSS     = "https://www.autosport.com/rss/feed/f1"
RACEFANS_RSS      = "https://www.racefans.net/feed/"
NEWS_FEEDS        = [THE_RACE_RSS, AUTOSPORT_RSS, RACEFANS_RSS]
THE_RACE_SEARCH   = "https://the-race.com/?s="
NEWS_CACHE_FILE   = Path(__file__).parent / "boxboxai_news_cache.json"
NEWS_REFRESH_MINS = 30   # refresh RSS every 30 minutes

_news_cache: list = []   # list of {title, summary, url, date}
_news_cache_time = None  # last fetch time
_news_lock       = threading.Lock()


def get_news_cache_time() -> datetime | None:
    """Returns the last news cache fetch time (live value, not an import-time snapshot)."""
    return _news_cache_time


def get_news_cache() -> list:
    """Returns the live news article list (live value, not an import-time snapshot)."""
    return _news_cache


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
        for match in list(re.finditer(pattern, r.text))[:max_results * 2]:
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
