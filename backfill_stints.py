#!/usr/bin/env python3
"""
Backfill tyre stint data from OpenF1 /v1/stints for all 6 completed 2026 races.
Adds tyre_strategies (dict) and stint_source to each episodic memory entry.

Usage:
  python3 backfill_stints.py           # writes to f1_memory_2026.json
  python3 backfill_stints.py --dry-run # prints summary only, no file write
"""

import argparse
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

MEMORY_FILE = Path("f1_memory_2026.json")
OPENF1_BASE = "https://api.openf1.org/v1"
RATE_DELAY  = 0.4  # seconds between API calls

SESSION_KEYS = {
    1: ("Australian Grand Prix", 11234),
    2: ("Chinese Grand Prix",    11245),
    3: ("Japanese Grand Prix",   11253),
    4: ("Miami Grand Prix",      11280),
    5: ("Canadian Grand Prix",   11291),
    6: ("Monaco Grand Prix",     11299),
}

COMPOUND_ABBREV = {
    "SOFT":         "S",
    "MEDIUM":       "M",
    "HARD":         "H",
    "INTERMEDIATE": "I",
    "WET":          "W",
}


def fetch(url: str) -> list:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def build_tyre_strategies(stints: list, driver_map: dict) -> dict:
    """
    Returns {driver_code: "M(1-16), H(17-38), M(39-58)"} for all drivers.
    Handles None compound (DNF mid-stint) by appending [DNF] to prior stint.
    """
    by_driver: dict[int, list] = defaultdict(list)
    for s in stints:
        if s["lap_end"] >= s["lap_start"]:  # filter formation-lap artifacts
            by_driver[s["driver_number"]].append(s)

    strategies = {}
    for driver_num, driver_stints in sorted(by_driver.items()):
        driver_stints.sort(key=lambda x: x["stint_number"])
        code = driver_map.get(driver_num, f"#{driver_num}")

        all_none = all(s.get("compound") is None for s in driver_stints)
        if all_none:
            strategies[code] = "? [DNF — no tyre data]"
            continue

        parts = []
        dnf_flagged = False
        for s in driver_stints:
            compound = s.get("compound")
            if compound is None:
                # Final partial stint with no compound recorded — mark DNF on previous
                dnf_flagged = True
                continue
            abbrev = COMPOUND_ABBREV.get(compound, compound[0] if compound else "?")
            parts.append(f"{abbrev}({s['lap_start']}-{s['lap_end']})")

        summary = ", ".join(parts)
        if dnf_flagged:
            summary += " [DNF]"
        strategies[code] = summary

    return strategies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary only — do not write to memory file")
    args = parser.parse_args()

    memory = json.loads(MEMORY_FILE.read_text())
    episodes = memory.get("episodic", [])

    # Index episodes by round for quick lookup
    ep_by_round = {ep["round"]: ep for ep in episodes}

    all_results = {}  # round -> (race_name, strategies)

    for rnd, (race_name, sk) in SESSION_KEYS.items():
        print(f"\nR{rnd} {race_name} (session_key={sk})")

        # Fetch driver number -> code mapping
        print(f"  Fetching drivers...", end=" ", flush=True)
        drivers_raw = fetch(f"{OPENF1_BASE}/drivers?session_key={sk}")
        driver_map = {d["driver_number"]: d["name_acronym"] for d in drivers_raw}
        print(f"{len(driver_map)} drivers")
        time.sleep(RATE_DELAY)

        # Fetch stints
        print(f"  Fetching stints...", end=" ", flush=True)
        stints = fetch(f"{OPENF1_BASE}/stints?session_key={sk}")
        print(f"{len(stints)} stint records")
        time.sleep(RATE_DELAY)

        # Build per-driver strategy strings
        strategies = build_tyre_strategies(stints, driver_map)
        all_results[rnd] = (race_name, strategies)

        # Print per-driver summary
        for code, summary in sorted(strategies.items()):
            print(f"    {code}: {summary}")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN — no file written.")
        return

    # Apply to memory and write atomically
    for rnd, (race_name, strategies) in all_results.items():
        if rnd not in ep_by_round:
            print(f"WARNING: R{rnd} not found in episodic memory — skipping")
            continue
        ep_by_round[rnd]["tyre_strategies"] = strategies
        ep_by_round[rnd]["stint_source"] = "openf1"

    tmp = MEMORY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(memory, indent=2, ensure_ascii=False))
    os.replace(tmp, MEMORY_FILE)

    print(f"\n{'='*60}")
    print(f"Written to {MEMORY_FILE} (atomic replace).")
    print(f"Updated {len(all_results)} episodes with tyre_strategies + stint_source.")


if __name__ == "__main__":
    main()
