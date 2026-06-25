#!/usr/bin/env python3
"""
Fetch and render FIA circuit map PDFs for completed 2026 races.

Loads the FIA season document library via Playwright, expands all collapsed
accordion sections, then downloads each circuit map PDF via requests and
renders page 1 as a JPEG.

Saves to boxboxai_circuit_maps/{circuit_key}_v2.jpg
Updates boxboxai_circuit_map_cache.json

Usage:
    python3 fetch_circuit_maps.py
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
import warnings

try:
    import fitz
except ImportError:
    print("ERROR: pymupdf not installed — run: pip install pymupdf")
    sys.exit(1)

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: playwright not installed — run: pip install playwright")
    sys.exit(1)

CIRCUIT_MAPS_DIR    = Path(__file__).parent / "boxboxai_circuit_maps"
CACHE_FILE          = Path(__file__).parent / "boxboxai_circuit_map_cache.json"
CIRCUIT_MAP_VERSION = 2
FIA_BASE            = "https://www.fia.com"
SEASON_URL          = (
    "https://www.fia.com/documents/championships/"
    "fia-formula-one-world-championship-14/season/season-2026-2072"
)

# All completed 2026 races ordered by round.
# race_name words (minus stop words) are used to match the PDF href slug.
# circuit_key is also matched directly against the href.
CIRCUITS = [
    ("melbourne", "Australian Grand Prix"),
    ("shanghai",  "Chinese Grand Prix"),
    ("suzuka",    "Japanese Grand Prix"),
    ("miami",     "Miami Grand Prix"),
    ("monaco",    "Monaco Grand Prix"),
    ("montreal",  "Canadian Grand Prix"),
    ("barcelona", "Spanish Grand Prix"),
    ("spielberg", "Austrian Grand Prix"),
]

STOP_WORDS = {"grand", "prix", "the", "de"}


def _match_keywords(circuit_key: str, race_name: str) -> set[str]:
    kws = {circuit_key.lower()}
    for word in race_name.lower().split():
        if word not in STOP_WORDS:
            kws.add(word)
    return kws


def _load_all_circuit_map_links() -> list[dict]:
    """
    Open the FIA season page via Playwright, click all collapsed accordion
    sections, then return all decision-document PDF links as
    [{"href": str, "text": str}, ...].
    """
    print("Loading FIA season page via Playwright …")
    links_found = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-gpu", "--disable-setuid-sandbox", "--no-zygote"])
        try:
            page = browser.new_page()
            page.set_default_timeout(20000)
            page.goto(SEASON_URL, wait_until="domcontentloaded")

            before = len(page.locator(
                "a[href*='/system/files/decision-document/']").all())
            print(f"  Links visible before expansion: {before}")

            # Click every collapsed race accordion section
            nojs_toggles = page.locator(
                "a[href*='/decision-document-list/nojs/']").all()
            print(f"  Accordion sections to expand: {len(nojs_toggles)}")
            for toggle in nojs_toggles:
                try:
                    toggle.click(timeout=3000)
                except Exception as e:
                    print(f"  toggle click skipped: {e}")
            if nojs_toggles:
                page.wait_for_load_state("networkidle", timeout=25000)

            after_els = page.locator(
                "a[href*='/system/files/decision-document/']").all()
            print(f"  Links visible after expansion: {len(after_els)}")

            for el in after_els:
                href = el.get_attribute("href") or ""
                text = (el.inner_text() or "").strip()
                if href:
                    links_found.append({"href": href, "text": text})

        finally:
            browser.close()

    return links_found


def _find_circuit_map_link(links: list[dict], circuit_key: str,
                           race_name: str) -> str | None:
    """Return the absolute PDF URL for the circuit map, or None."""
    match_kws = _match_keywords(circuit_key, race_name)
    for item in links:
        href = item["href"].lower()
        text = item["text"].lower()

        # Must be a circuit-map document
        if "circuit_map" not in href and "circuit map" not in text:
            continue

        # Must belong to this race
        if not any(kw in href for kw in match_kws if len(kw) > 3):
            continue

        url = item["href"]
        if url.startswith("/"):
            url = FIA_BASE + url
        return url

    return None


def _download_pdf(url: str) -> bytes:
    hdrs = {
        "User-Agent": "Mozilla/5.0",
        "Referer": SEASON_URL,
        "Accept": "application/pdf,*/*",
    }
    # FIA cert uses intermediate not in certifi; verify=False is intentional here only
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        r = requests.get(url, headers=hdrs, timeout=30, verify=False,
                         stream=True, allow_redirects=True)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}")
        return b""
    data = r.content
    if data[:4] != b"%PDF":
        print(f"  Not a PDF (got {data[:8]!r})")
        return b""
    return data


def _render_jpeg(circuit_key: str, pdf_bytes: bytes) -> Path | None:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    # Page 0 is typically a cover/title; page 1 is the overhead circuit map.
    # Fall back to page 0 for single-page PDFs.
    page_idx = 1 if n_pages > 1 else 0
    page = doc[page_idx]
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg", jpg_quality=85)
    CIRCUIT_MAPS_DIR.mkdir(exist_ok=True)
    out = CIRCUIT_MAPS_DIR / f"{circuit_key}_v{CIRCUIT_MAP_VERSION}.jpg"
    out.write_bytes(img_bytes)
    return out


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, CACHE_FILE)


def main():
    all_links = _load_all_circuit_map_links()
    if not all_links:
        print("ERROR: no links found — aborting")
        sys.exit(1)

    # Filter to circuit map candidates for a quick preview
    cm_links = [l for l in all_links
                if "circuit_map" in l["href"].lower()
                or "circuit map" in l["text"].lower()]
    print(f"\nCircuit-map links found across all sections: {len(cm_links)}")
    for l in cm_links:
        print(f"  {l['href'][-90:]}")

    cache   = _load_cache()
    results = {}

    print()
    for circuit_key, race_name in CIRCUITS:
        print(f"{'─'*60}")
        print(f"{race_name} ({circuit_key})")

        pdf_url = _find_circuit_map_link(all_links, circuit_key, race_name)
        if not pdf_url:
            print(f"  NOT FOUND — no circuit-map link matched")
            results[circuit_key] = "NOT_FOUND"
            continue

        print(f"  URL: …{pdf_url[-80:]}")
        pdf_bytes = _download_pdf(pdf_url)
        if not pdf_bytes:
            results[circuit_key] = "FETCH_FAILED"
            continue

        print(f"  Downloaded {len(pdf_bytes):,} bytes")
        try:
            img_path = _render_jpeg(circuit_key, pdf_bytes)
            size = img_path.stat().st_size
            if size < 10_000:
                print(f"  RENDER_FAILED — image too small ({size} bytes)")
                results[circuit_key] = "RENDER_FAILED"
            else:
                print(f"  ✅ Rendered → {img_path.name} ({size:,} bytes)")
                results[circuit_key] = "RENDERED"
                if circuit_key not in cache:
                    cache[circuit_key] = {}
        except Exception as e:
            print(f"  RENDER_FAILED — {e}")
            results[circuit_key] = "RENDER_FAILED"

        time.sleep(0.5)

    _save_cache(cache)

    print(f"\n{'='*60}")
    print("RESULTS:")
    for key, status in results.items():
        icon = "✅" if status == "RENDERED" else "❌"
        print(f"  {icon} {key}: {status}")

    print(f"\nFiles in {CIRCUIT_MAPS_DIR}:")
    if CIRCUIT_MAPS_DIR.exists():
        for f in sorted(CIRCUIT_MAPS_DIR.iterdir()):
            print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    else:
        print("  (directory does not exist)")


if __name__ == "__main__":
    main()
