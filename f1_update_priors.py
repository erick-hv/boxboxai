#!/usr/bin/env python3
"""
f1_update_priors.py — Standalone Bayesian prior updater.

Run this after each race, BEFORE running f1_2026_predictor.py again
(the predictor overwrites f1_2026_predicciones.csv with next-race predictions).

Usage:
    python f1_update_priors.py --round 9
    python f1_update_priors.py --round 9 --force   # re-apply even if already done

What it does:
  1. Checks _meta.last_updated_round dedup guard — skips if already done
  2. Fetches actual race result from Jolpica for --round
  3. Loads f1_2026_predicciones.csv and validates it is for --round
  4. Computes prediction error per driver (actual_pos - predicted_rank)
  5. Updates score_multipliers in f1_2026_bayesian_priors.json
  6. Prints a full summary table sorted by finishing position
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

# ─── Constants (mirror f1_2026_predictor.py) ────────────────────────────────
SEASON        = 2026
JOLPICA_BASE  = "https://api.jolpi.ca/ergast/f1"
REQ_TIMEOUT   = 15
REQ_DELAY     = 0.4

OUTPUT_CSV    = "f1_2026_predicciones.csv"
PRIORS_FILE   = os.environ.get("F1_PRIORS_FILE", "./f1_2026_bayesian_priors.json")

LEARNING_RATE = 0.15   # how fast priors shift (0 = never, 1 = only last result)
MAX_PRIOR     = 1.40   # ceiling on score_multiplier
MIN_PRIOR     = 0.60   # floor on score_multiplier


# ─── API helpers ─────────────────────────────────────────────────────────────

def api_get(url: str) -> dict | None:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    print(f"   ⚠  JSON inválido en {url}: {e}")
                    return None
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"   ⏳  Rate limit — esperando {wait}s...")
                time.sleep(wait)
            elif r.status_code == 404:
                return None
            else:
                print(f"   ⚠  HTTP {r.status_code} en {url}")
                time.sleep(1)
        except requests.exceptions.ConnectionError:
            print("   ❌  Sin conexión — verifica tu internet.")
            time.sleep(2)
        except Exception as e:
            print(f"   ⚠  Error: {e}")
            time.sleep(1)
    return None


# ─── Priors I/O ──────────────────────────────────────────────────────────────

def load_priors() -> dict:
    try:
        with open(PRIORS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_priors(priors: dict):
    try:
        with open(PRIORS_FILE, "w") as f:
            json.dump(priors, f, indent=2)
    except Exception as e:
        print(f"   ⚠  No se pudieron guardar priors: {e}")


# ─── Data loading ─────────────────────────────────────────────────────────────

def fetch_race_result(round_num: int) -> pd.DataFrame:
    """Fetch actual race finishing order from Jolpica for a given round."""
    print(f"🏁  Descargando resultado de Ronda {round_num} desde Jolpica...")
    data  = api_get(f"{JOLPICA_BASE}/{SEASON}/{round_num}/results.json")
    if not data:
        return pd.DataFrame()
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        print(f"   ⚠  Sin datos de carrera para Ronda {round_num} — ¿ya se disputó?")
        return pd.DataFrame()

    race      = races[0]
    race_name = race.get("raceName", f"Ronda {round_num}")
    print(f"   ✅  {race_name}")

    rows = []
    for res in race.get("Results", []):
        drv    = res.get("Driver", {})
        code   = drv.get("code", drv.get("driverId", "???").upper()[:3])
        try:
            pos = int(res.get("position", 99))
        except (ValueError, TypeError):
            pos = 99
        status = res.get("status", "")
        dnf    = 0 if status in ("Finished", "+1 Lap", "+2 Laps") else 1
        name   = f"{drv.get('givenName','').strip()} {drv.get('familyName','').strip()}".strip()
        rows.append({
            "code"      : code,
            "full_name" : name,
            "actual_pos": pos,
            "status"    : status,
            "dnf"       : dnf,
        })

    df = pd.DataFrame(rows)
    df["race_name"] = race_name
    return df


def load_predictions(round_num: int) -> pd.DataFrame:
    """Load f1_2026_predicciones.csv and validate it is for round_num."""
    if not os.path.exists(OUTPUT_CSV):
        print(f"   ⚠  CSV no encontrado: {OUTPUT_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(OUTPUT_CSV)
    if df.empty:
        return df

    csv_round = int(df["round_num"].iloc[0]) if "round_num" in df.columns else -1
    if csv_round != round_num:
        print(f"   ⚠  {OUTPUT_CSV} contiene predicciones para Ronda {csv_round}, no Ronda {round_num}.")
        print(f"      Ejecuta este script ANTES de correr f1_2026_predictor.py,")
        print(f"      que sobreescribe el CSV con las predicciones de la siguiente carrera.")
        return pd.DataFrame()

    return df


# ─── Bayesian update (mirrors update_bayesian_priors() in main predictor) ────

def compute_updates(pred_df: pd.DataFrame,
                    actual_df: pd.DataFrame,
                    priors: dict) -> list[dict]:
    """
    For each driver:
      pred_rank  = 1-indexed rank by win_mc_pct descending
      error      = actual_pos - pred_rank
                     > 0  → finished worse than predicted (overestimated)
                     < 0  → finished better than predicted (underestimated)
      adjustment = -(error / n) * LEARNING_RATE
      new_prior  = clip(old_prior * (1 + adjustment), MIN_PRIOR, MAX_PRIOR)
    """
    n = len(pred_df)

    sorted_pred = pred_df.sort_values("win_mc_pct", ascending=False).reset_index(drop=True)
    pred_rank   = {row["code"]: i + 1 for i, (_, row) in enumerate(sorted_pred.iterrows())}

    actual_map  = dict(zip(actual_df["code"], actual_df["actual_pos"]))
    dnf_map     = dict(zip(actual_df["code"], actual_df["dnf"]))
    status_map  = dict(zip(actual_df["code"], actual_df["status"]))
    name_map    = dict(zip(actual_df["code"], actual_df["full_name"]))

    updates = []
    for _, row in sorted_pred.iterrows():
        code   = row["code"]
        p_rank = pred_rank.get(code, n)
        a_pos  = actual_map.get(code)

        if a_pos is None:
            continue
        try:
            a_pos = float(a_pos)
        except (TypeError, ValueError):
            continue

        error         = a_pos - p_rank
        norm_error    = error / n
        old_prior     = float(priors.get(code, {}).get("score_multiplier", 1.0))
        adjustment    = -norm_error * LEARNING_RATE
        new_prior     = float(np.clip(old_prior * (1 + adjustment), MIN_PRIOR, MAX_PRIOR))

        history = list(priors.get(code, {}).get("error_history", []))
        history = (history + [round(error, 2)])[-10:]

        updates.append({
            "code"          : code,
            "full_name"     : name_map.get(code, row.get("FullName", code)),
            "pred_rank"     : p_rank,
            "actual_pos"    : int(a_pos),
            "error"         : round(error, 2),
            "old_prior"     : round(old_prior, 4),
            "new_prior"     : round(new_prior, 4),
            "delta"         : round(new_prior - old_prior, 4),
            "dnf"           : dnf_map.get(code, 0),
            "status"        : status_map.get(code, ""),
            "error_history" : history,
            "races_tracked" : priors.get(code, {}).get("races_tracked", 0) + 1,
        })

    return updates


def apply_updates(priors: dict, updates: list[dict]) -> dict:
    for u in updates:
        code = u["code"]
        priors[code] = {
            "score_multiplier" : u["new_prior"],
            "last_error"       : u["error"],
            "avg_error"        : round(float(np.mean(u["error_history"])), 2),
            "error_history"    : u["error_history"],
            "races_tracked"    : u["races_tracked"],
        }
    return priors


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(round_num: int, race_name: str, updates: list[dict]):
    by_actual = sorted(updates, key=lambda u: u["actual_pos"])

    w = 74
    print(f"\n{'─' * w}")
    print(f"  📊  PRIORS ACTUALIZADOS — Ronda {round_num}: {race_name}")
    print(f"{'─' * w}")
    hdr = (f"  {'CODE':<5} {'PILOTO':<22} {'PRED':>5} {'REAL':>5} "
           f"{'ERR':>6}  {'OLD×':>7} {'NEW×':>7} {'Δ':>7}  FLAG")
    print(hdr)
    print(f"  {'─' * (w - 2)}")

    for u in by_actual:
        arrow   = "⬆" if u["delta"] > 0 else ("⬇" if u["delta"] < 0 else "─")
        flag    = f"{arrow}  ❌DNF" if u["dnf"] else arrow
        name_s  = u["full_name"][:21]
        print(
            f"  {u['code']:<5} {name_s:<22} {u['pred_rank']:>5} {u['actual_pos']:>5} "
            f"{u['error']:>+6.1f}  {u['old_prior']:>7.4f} {u['new_prior']:>7.4f} "
            f"{u['delta']:>+7.4f}  {flag}"
        )

    print(f"{'─' * w}")

    movers = sorted(updates, key=lambda u: abs(u["delta"]), reverse=True)[:5]
    print(f"\n  Mayores cambios:")
    for u in movers:
        direction = "⬆ subestimado" if u["delta"] > 0 else "⬇ sobreestimado"
        races = u["races_tracked"]
        print(f"    {u['code']:<5}  {direction:<18}  ×{u['new_prior']:.4f}"
              f"  (err {u['error']:+.1f} pos, {races} carrera{'s' if races != 1 else ''} rastreada{'s' if races != 1 else ''})")

    print()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Update f1_2026_bayesian_priors.json from actual race result."
    )
    parser.add_argument("--round", type=int, required=True,
                        help="Race round number to process (e.g. --round 9)")
    parser.add_argument("--force", action="store_true",
                        help="Re-apply even if this round is already in last_updated_round")
    args      = parser.parse_args()
    round_num = args.round

    print(f"\n🧠  F1 Prior Updater — Ronda {round_num}")
    print(f"    Priors: {PRIORS_FILE}")
    print(f"    CSV:    {OUTPUT_CSV}\n")

    # ── Dedup guard ───────────────────────────────────────────────────────────
    priors       = load_priors()
    already_done = priors.get("_meta", {}).get("last_updated_round", -1)
    if already_done == round_num and not args.force:
        print(f"✅  Priors ya actualizados para Ronda {round_num} — sin cambios.")
        print(f"    Usa --force para re-aplicar de todos modos.")
        return

    # ── Fetch actual race result ───────────────────────────────────────────────
    actual_df = fetch_race_result(round_num)
    if actual_df.empty:
        print(f"❌  No se pudo obtener resultado de Ronda {round_num}. Abortando.")
        return
    race_name = str(actual_df["race_name"].iloc[0])

    # ── Load prediction CSV ───────────────────────────────────────────────────
    pred_df = load_predictions(round_num)
    if pred_df.empty:
        print(f"❌  Sin predicciones válidas para Ronda {round_num}. Abortando.")
        return

    # ── Compute updates ───────────────────────────────────────────────────────
    print(f"🔄  Calculando ajustes para Ronda {round_num} ({race_name})...")
    updates = compute_updates(pred_df, actual_df, priors)
    if not updates:
        print("   ⚠  Sin actualizaciones — verifica que los códigos de piloto coinciden.")
        return

    # ── Apply and save ────────────────────────────────────────────────────────
    priors = apply_updates(priors, updates)
    priors["_meta"] = {"last_updated_round": round_num}
    save_priors(priors)
    print(f"   💾  Priors guardados en: {PRIORS_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(round_num, race_name, updates)


if __name__ == "__main__":
    main()
