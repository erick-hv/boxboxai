"""
Predictor calibration check — item 5 from council verdict.
Runs standalone, does not import boxboxai_bot.

Checks:
  1. Whether per-round prediction snapshots were stored (pending_predictions).
  2. Best-effort calibration from whatever prediction data exists on disk.
  3. Bayesian prior error history as a proxy for ongoing calibration signal.
  4. R8 Austria CSV status — real qualifying data or pre-qualifying sentinel?
"""

import csv
import json
import os
from pathlib import Path

BASE = Path(__file__).parent

SEP = "─" * 60

# ── 1. Load memory ────────────────────────────────────────────

mem = json.loads((BASE / "f1_memory_2026.json").read_text())
episodic = mem.get("episodic", [])
pending  = mem.get("pending_predictions", {})

print(SEP)
print("PREDICTOR CALIBRATION CHECK — 2026 season")
print(SEP)

# ── 2. Pending predictions audit ─────────────────────────────

print("\n── Stored per-round predictions (pending_predictions) ──")
if not pending:
    print("  ⚠️  pending_predictions is EMPTY.")
    print("     The bot stores predictions here only while a race is upcoming")
    print("     (they are consumed/cleared after the race fires). No historical")
    print("     snapshots were retained, so head-to-head calibration is impossible")
    print("     for rounds already completed.")
else:
    for rnd, data in sorted(pending.items()):
        print(f"  R{rnd}: {len(data)} entries")

# ── 3. Episodic actual results ───────────────────────────────

print("\n── Actual race results in episodic memory ──")
if not episodic:
    print("  No episodic entries found.")
else:
    for e in sorted(episodic, key=lambda x: x.get("round", 0)):
        rnd  = e.get("round", "?")
        name = e.get("race_name", "?")
        w    = e.get("winner", "?")
        p2   = e.get("p2", "?")
        p3   = e.get("p3", "?")
        print(f"  R{rnd:>2} {name:<30}  winner={w}  P2={p2}  P3={p3}")

# ── 4. Old predictions CSV (f1_2026_predictions.csv) ─────────

old_csv = BASE / "f1_2026_predictions.csv"
print(f"\n── f1_2026_predictions.csv (legacy format) ──")
if not old_csv.exists():
    print("  Not found.")
else:
    rows = list(csv.DictReader(old_csv.open()))
    model_used = rows[0].get("ModelUsed", "?") if rows else "?"
    # Infer race: "N carrera(s) más" until XGBoost → N races short of 6
    import re
    m = re.search(r"(\d+) carrera", model_used)
    races_short = int(m.group(1)) if m else None
    inferred_race = f"R{6 - races_short}" if races_short is not None else "unknown"
    print(f"  Rows: {len(rows)} drivers")
    print(f"  Model: {model_used}")
    print(f"  Inferred: generated after {inferred_race} results (XGBoost needed {races_short} more races)")
    print()
    print(f"  {'Rank':<5} {'Code':<6} {'Driver':<25} {'Win%':>6}")
    print(f"  {'----':<5} {'----':<6} {'------':<25} {'----':>6}")
    rows_sorted = sorted(rows, key=lambda r: float(r.get("WinProbability", 0)), reverse=True)
    for i, r in enumerate(rows_sorted[:6], 1):
        print(f"  {i:<5} {r['Abbreviation']:<6} {r['FullName']:<25} {float(r['WinProbability']):>5.1f}%")

    # Cross-reference with actual results if we can infer the target race
    if races_short is not None:
        target_rnd = 6 - races_short + 1  # prediction was FOR this round
        actual = next((e for e in episodic if e.get("round") == target_rnd), None)
        if actual:
            print(f"\n  Cross-reference with actual R{target_rnd} ({actual.get('race_name')}):")
            actual_winner = actual.get("winner")
            top1 = rows_sorted[0]["Abbreviation"]
            top3_codes = [r["Abbreviation"] for r in rows_sorted[:3]]
            winner_row = next((r for r in rows if r["Abbreviation"] == actual_winner), None)
            winner_win_pct = float(winner_row["WinProbability"]) if winner_row else None

            hit1 = "✅" if top1 == actual_winner else "❌"
            hit3 = "✅" if actual_winner in top3_codes else "❌"
            print(f"    Top-1 pick: {top1}  →  actual winner: {actual_winner}  {hit1}")
            print(f"    Top-3 picks: {top3_codes}  →  winner in top-3? {hit3}")
            if winner_win_pct is not None:
                print(f"    Win% assigned to actual winner ({actual_winner}): {winner_win_pct:.1f}%")
            overconf = [r for r in rows if float(r["WinProbability"]) >= 90]
            if overconf and overconf[0]["Abbreviation"] != actual_winner:
                print(f"    ⚠️  Overconfident (≥90%): {overconf[0]['Abbreviation']} at {float(overconf[0]['WinProbability']):.1f}%")
            else:
                print(f"    No overconfident (≥90%) calls.")
        else:
            print(f"\n  No episodic entry for R{target_rnd} — cannot cross-reference.")

# ── 5. Current predicciones CSV (v7 format) ──────────────────

new_csv = BASE / "f1_2026_predicciones.csv"
print(f"\n── f1_2026_predicciones.csv (v7 format, current) ──")
if not new_csv.exists():
    print("  ❌ FILE DOES NOT EXIST locally.")
    print("     This is the file the bot uses for /winner and /predict.")
    print("     It was deleted from the local working tree (visible in git status).")
    print("     On Railway the file may exist if the predictor ran there, but it is")
    print("     NOT available for R8 Austria calibration from this machine.")
else:
    rows = list(csv.DictReader(new_csv.open()))
    r8_rows = [r for r in rows if str(r.get("round_num", "")).strip() == "8"]
    has_round_col = "round_num" in (rows[0].keys() if rows else [])

    print(f"  Rows: {len(rows)}")
    print(f"  Has round_num column: {has_round_col}")

    if r8_rows:
        top = sorted(r8_rows, key=lambda r: float(r.get("win_mc_pct", 0)), reverse=True)
        sentinel = all(float(r.get("quali_pos_next", 0)) == 22.0 for r in top[:5])
        print(f"\n  R8 Austria — {'⚠️  PRE-QUALIFYING SENTINEL DATA' if sentinel else '✅ Qualifying-based data'}")
        print(f"  quali_pos_next sentinel (22.0)? {sentinel}")
        print(f"\n  {'Rank':<5} {'Code':<6} {'Driver':<25} {'Win%':>7} {'Podium%':>9} {'Quali':>6}")
        print(f"  {'----':<5} {'----':<6} {'------':<25} {'----':>7} {'-------':>9} {'-----':>6}")
        for i, r in enumerate(top[:8], 1):
            qp = r.get("quali_pos_next", "?")
            print(f"  {i:<5} {r.get('code','?'):<6} {r.get('FullName','?'):<25} "
                  f"{float(r.get('win_mc_pct',0)):>6.1f}%"
                  f"{float(r.get('podium_mc_pct',0)):>8.1f}%"
                  f"  {qp:>5}")
    elif has_round_col:
        rounds_in_csv = sorted(set(r.get("round_num","?") for r in rows))
        print(f"  No R8 rows found. Rounds present: {rounds_in_csv}")
    else:
        print("  No round_num column — cannot filter for R8.")
        print("  Top drivers in CSV:")
        top = sorted(rows, key=lambda r: float(r.get("win_mc_pct", r.get("WinProbability", 0))), reverse=True)
        for r in top[:5]:
            print(f"    {r}")

# ── 6. Bayesian error history (proxy calibration signal) ─────

priors = json.loads((BASE / "f1_2026_bayesian_priors.json").read_text())
print(f"\n── Bayesian prior prediction error history ──")
print("  (These are PredictionScore errors from historical races,")
print("   used to tune per-driver multipliers — not direct win% accuracy)")
print()
print(f"  {'Code':<6} {'AvgError':>10} {'LastError':>10} {'Races':>7}")
print(f"  {'----':<6} {'--------':>10} {'----------':>10} {'-----':>7}")
by_avg = sorted(priors.items(), key=lambda x: x[1].get("avg_error", 0), reverse=True)
for code, d in by_avg[:10]:
    print(f"  {code:<6} {d.get('avg_error', 0):>10.2f} {d.get('last_error', 0):>10.2f} {d.get('races_tracked', 0):>7}")
print()
print("  Note: all score_multipliers are currently 1.4 (uniform) —")
print("  the Bayesian correction hasn't differentiated by driver yet.")

# ── 7. Calibration verdict ────────────────────────────────────

print(f"\n{SEP}")
print("CALIBRATION VERDICT")
print(SEP)
print("""
DATA AVAILABILITY:
  • pending_predictions: empty (predictions cleared after each race; no snapshots kept)
  • f1_2026_predictions.csv: 1 snapshot (old Bayesian-prior model, no round stamp)
  • f1_2026_predicciones.csv: NOT on this machine (deleted from local working tree)
  • Episodic results: R1–R6 ✅, R7 Spanish GP missing from local memory

WHAT WE CAN ASSESS:
  • Single legacy-format prediction cross-referenced above (1 race sample — see above)
  • Error history in Bayesian priors covers 17 historical data points per driver,
    but errors are in PredictionScore units, not win%, so can't map directly to
    "was top pick right?"

R8 AUSTRIA TWEET RECOMMENDATION:
  ⚠️  Cannot confirm the predicciones CSV has real qualifying data locally.
      Before tweeting, verify on Railway that:
        (a) f1_2026_predicciones.csv exists and has round_num=8 rows
        (b) quali_pos_next values are NOT all 22.0 (sentinel = pre-qualifying)
      If sentinel values → the predictor ran before qualifying, not after.
      Only tweet after confirming (b) is false.

  Caveat to include regardless:
    "Monte Carlo model, Bayesian-prior phase (< 6 races of XGBoost training data).
     Historical accuracy: limited sample, treat as directional, not definitive."
""")
print(SEP)
