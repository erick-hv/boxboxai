#!/usr/bin/env python3
"""
Standalone, read-only prediction explainer for the F1 2026 predictor.

Run manually, on demand — never called by f1_auto_runner.py or the GitHub
Actions pipeline:

    python shap_explain.py --round 9 --driver ANT
    python shap_explain.py --round 9 --driver ANT HAM RUS
    python shap_explain.py --round 9 --top 3

Two layers of explanation:

  Layer 1 — exact additive breakdown of score_manual()'s formula, the
            function that actually determines win_pct (see
            f1_2026_predictor.py:4295-4493). Reconstructed from columns
            already persisted in f1_2026_predicciones.csv. A handful of
            score_manual/_RACE_FEAT_COLS inputs (quali_pos, fl_rate,
            lap_std, constructor_pts) are NOT persisted to that CSV and
            are reported as unavailable rather than guessed.

  Layer 2 — SHAP TreeExplainer values from the saved XGBoost/LightGBM
            race models, loaded read-only from
            f1_2026_xgb_race_model.json / f1_2026_lgbm_race_model.txt.
            These describe what the tree sub-models think drives
            finishing position. They feed 60% of a feedback-loop weight
            blend into score_manual (the rest is season-history
            features only — "next race" features like quali_pos_next
            keep their manual weight untouched). They are NOT the same
            thing as Layer 1 and are labeled as a diagnostic.

  GP is excluded from both layers: the 3-way XGB/LGBM/GP blend
  (f1_2026_predictor.py:6954-6966) is computed into feat["xgb_pos"]
  but that column is never read again anywhere in the pipeline — it
  does not reach score_manual, the LSTM blend, apply_bayesian_priors,
  or monte_carlo_simulation. GP currently contributes 0% to win_pct,
  so there is nothing to attribute.

This script only reads f1_2026_predicciones.csv, f1_2026_bayesian_priors.json,
and the two saved model files. It never calls .fit(), save_model(), or
save_priors(), and never writes to any existing pipeline output file.
--save writes ONLY to a path you name explicitly.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRED_CSV        = "f1_2026_predicciones.csv"
PRIORS_JSON     = "f1_2026_bayesian_priors.json"
XGB_MODEL_FILE  = "f1_2026_xgb_race_model.json"
LGBM_MODEL_FILE = "f1_2026_lgbm_race_model.txt"

# Mirrors f1_2026_predictor.py:4507-4514 — race model input columns, in
# training order, plus predicted_quali_pos appended. Kept in manual sync;
# if that list changes in the predictor, update this one too.
_RACE_FEAT_COLS = [
    "champ_pts", "avg_finish", "avg_grid", "fl_rate", "lap_std",
    "constructor_pts", "recent_form", "momentum_pos", "sprint_pts",
    "fp_avg_delta", "avg_sector_delta", "tyre_deg_slope", "lap1_gain",
    "teammate_delta", "avg_pitstop", "sc_gain_avg", "dnf_rate",
    "penalty_count", "overtaking_ability", "quali_consistency",
    "tyre_management_index",
]
_ALL_RACE_COLS = _RACE_FEAT_COLS + ["predicted_quali_pos"]

# Columns _RACE_FEAT_COLS/score_manual need that are NOT persisted to the
# output CSV (only exist in the ephemeral in-memory `feat` DataFrame).
_NOT_IN_CSV = {"fl_rate", "lap_std", "constructor_pts", "quali_pos"}


# ─────────────────────────────────────────────────────────────
#  Layer 1 — score_manual() additive breakdown
#  Duplicated (not imported) from f1_2026_predictor.py:score_manual to
#  avoid importing that module (its import-time fastf1.Cache.enable_cache()
#  call is a side effect this script deliberately avoids). Mirrors
#  f1_2026_predictor.py:4295-4493 — re-sync manually if that formula changes.
# ─────────────────────────────────────────────────────────────
def _score_manual_terms(feat: pd.DataFrame, overtaking_difficulty: float = 0.55,
                         rain_prob: float = 0.0):
    """
    Returns (terms: dict[col] -> pd.Series contribution, weights: dict,
    has_next_quali: bool, missing: list[(col, nominal_weight)]).
    Mirrors score_manual()'s weight regime exactly for columns available
    in the CSV; columns in _NOT_IN_CSV are reported, not guessed.
    """
    n = len(feat)

    def rank_asc(col):
        return (n + 1 - feat[col].rank(ascending=True, method="min")) / n

    def rank_desc(col):
        return feat[col].rank(ascending=False, method="min").rsub(n + 1) / n

    quali_w_dynamic = round(0.22 + overtaking_difficulty * 0.16, 3)
    delta_w  = quali_w_dynamic - 0.28
    finish_w = max(0.04, 0.10 - delta_w * 0.5)
    recent_w = max(0.04, 0.10 - delta_w * 0.5)

    has_next_quali = (
        "quali_pos_next" in feat.columns and
        feat["quali_pos_next"].notna().sum() >= 10 and
        feat["quali_pos_next"].nunique() >= 5
    )

    w = {
        "quali_pos_next":     0.55 if has_next_quali else 0.00,
        "fp_next_delta":      0.08 if has_next_quali else 0.00,
        "fp2_next_longrun":   0.05 if has_next_quali else 0.00,
        "sector_balance":     0.03 if has_next_quali else 0.00,
        "soft_pace_delta":    0.02 if has_next_quali else 0.00,
        "medium_pace_delta":  0.02 if has_next_quali else 0.00,
        "tyre_deg_rate":      0.03 if has_next_quali else 0.00,
        "corner_profile_score": 0.02 if has_next_quali else 0.00,
        "race_sim_delta":     0.03 if has_next_quali else 0.00,
        "quali_pos":          0.05 if has_next_quali else quali_w_dynamic,
        "quali_gap_teammate": 0.04 if has_next_quali else 0.06,
        "avg_grid":           0.03 if has_next_quali else 0.06,
        "champ_pts":          0.10 if has_next_quali else 0.11,
        "avg_finish":         0.04 if has_next_quali else finish_w,
        "constructor_pts":         0.05 if has_next_quali else 0.08,
        "constructor_momentum":    0.05 if has_next_quali else 0.05,
        "recent_form":        0.05 if has_next_quali else recent_w,
        "momentum_pos":       0.02 if has_next_quali else 0.03,
        "fl_rate":            0.02 if has_next_quali else 0.04,
        "lap_std":            0.02 if has_next_quali else 0.04,
        "sprint_pts":         0.02 if has_next_quali else 0.03,
        "fp_avg_delta":       0.01,
        "fp2_longrun_delta":  0.01,
        "tyre_deg_slope":     0.02 if has_next_quali else 0.03,
        "compound_score":     0.01 if has_next_quali else 0.03,
        "circuit_score":      0.02 if has_next_quali else 0.03,
        "circuit_type_score": 0.01 if has_next_quali else 0.03,
        "circuit_affinity":   0.04 if has_next_quali else 0.06,
        "lap1_gain":          0.01 if has_next_quali else 0.02,
        "teammate_delta":     0.01,
        "avg_pitstop":        0.01 if has_next_quali else 0.02,
        "sc_gain_avg":        0.01,
        "avg_sector_delta":   0.01 if has_next_quali else 0.02,
        "streak_score":            0.03,
        "post_dnf_bounce":         0.02,
        "championship_pressure":   0.02,
        "overtaking_ability":      0.03,
        "quali_consistency":       0.02,
        "wet_weather_delta":       round(0.05 * rain_prob, 4),
        "historical_dnf_rate":     0.01,
        "tyre_management_index":   0.02,
        "compatibility_score":     0.04 if has_next_quali else 0.03,
        "press_sentiment":         0.03 if has_next_quali else 0.00,
        "sprint_quali_delta":      0.02 if (has_next_quali and
                                             "sprint_quali_delta" in feat.columns and
                                             feat["sprint_quali_delta"].abs().sum() > 0)
                                        else 0.00,
        "tyre_inventory_score":    0.02 if has_next_quali else 0.00,
        "corner_mastery_score":    0.03 if (has_next_quali and
                                             "corner_mastery_score" in feat.columns and
                                             feat["corner_mastery_score"].abs().sum() > 0.01)
                                        else 0.00,
    }

    if has_next_quali:
        _next_keys = {"quali_pos_next", "fp_next_delta", "fp2_next_longrun",
                      "sector_balance", "soft_pace_delta", "medium_pace_delta",
                      "tyre_deg_rate", "corner_profile_score", "race_sim_delta",
                      "tyre_inventory_score", "corner_mastery_score"}
        season_sum = sum(v for k, v in w.items() if k not in _next_keys)
        season_target = 1.0 - sum(w[k] for k in _next_keys if k in w)
        if season_sum > 0:
            sf = season_target / season_sum
            for k in list(w.keys()):
                if k not in _next_keys:
                    w[k] *= sf

    total_w = sum(w.values()) or 1.0
    w = {k: v / total_w for k, v in w.items()}

    spec = [
        ("quali_pos_next", "asc"), ("fp_next_delta", "asc"), ("fp2_next_longrun", "asc"),
        ("sector_balance", "asc"), ("soft_pace_delta", "asc"), ("medium_pace_delta", "asc"),
        ("tyre_deg_rate", "asc"), ("corner_profile_score", "asc"), ("race_sim_delta", "asc"),
        ("quali_pos", "asc"), ("quali_gap_teammate", "desc"), ("avg_grid", "asc"),
        ("champ_pts", "desc"), ("avg_finish", "asc"), ("constructor_pts", "desc"),
        ("constructor_momentum", "desc"), ("recent_form", "desc"), ("momentum_pos", "asc"),
        ("fl_rate", "desc"), ("lap_std", "asc"), ("sprint_pts", "desc"),
        ("fp_avg_delta", "asc"), ("fp2_longrun_delta", "asc"), ("tyre_deg_slope", "asc"),
        ("compound_score", "desc"), ("circuit_score", "desc"), ("circuit_type_score", "desc"),
        ("circuit_affinity", "asc"), ("lap1_gain", "desc"), ("teammate_delta", "desc"),
        ("avg_pitstop", "asc"), ("sc_gain_avg", "desc"), ("avg_sector_delta", "asc"),
        ("streak_score", "desc"), ("post_dnf_bounce", "desc"), ("championship_pressure", "desc"),
        ("overtaking_ability", "desc"), ("quali_consistency", "asc"), ("wet_weather_delta", "asc"),
        ("historical_dnf_rate", "asc"), ("tyre_management_index", "desc"),
        ("compatibility_score", "desc"), ("press_sentiment", "desc"),
        ("sprint_quali_delta", "desc"), ("tyre_inventory_score", "desc"),
        ("corner_mastery_score", "desc"),
    ]

    terms, missing = {}, []
    for col, direction in spec:
        weight = w.get(col, 0.0)
        if col in _NOT_IN_CSV:
            if weight > 1e-9:
                missing.append((col, weight))
            continue
        if col in feat.columns and feat[col].notna().any():
            fn = rank_asc if direction == "asc" else rank_desc
            terms[col] = fn(col) * weight
        # else: column truly absent this round (e.g. no sprint weekend) — weight is 0 anyway

    # DNF / penalty subtractive terms (score_manual:4488-4492)
    dnf   = feat.get("dnf_rate", pd.Series(0.0, index=feat.index)).fillna(0) * 0.10
    hdnf  = feat.get("historical_dnf_rate", pd.Series(0.0, index=feat.index)).fillna(0) * 0.03
    pen_raw = feat.get("penalty_count", pd.Series(0.0, index=feat.index)).fillna(0)
    pen_max = feat.get("penalty_count", pd.Series(1.0, index=feat.index)).max() + 1
    pen   = (pen_raw / pen_max) * 0.05
    terms["dnf_rate (penalty)"]            = -dnf
    terms["historical_dnf_rate (penalty)"] = -hdnf
    terms["penalty_count (penalty)"]       = -pen

    return terms, w, has_next_quali, missing


def _load_feedback_snapshot(priors_path: str):
    """Most recent feature_importance_history entry (approximate feedback-
    loop weights). Returns (round, top10 dict) or (None, {})."""
    try:
        priors = json.load(open(priors_path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, {}
    hist = priors.get("feature_importance_history", [])
    if not hist:
        return None, {}
    latest = hist[-1]
    return latest.get("round"), latest.get("importances", {})


# ─────────────────────────────────────────────────────────────
#  Layer 2 — TreeExplainer over the saved XGB/LGBM race models
# ─────────────────────────────────────────────────────────────
def _build_x_pred_row(row: pd.Series, feat_round: pd.DataFrame):
    """Builds one X_pred row in _ALL_RACE_COLS order from CSV columns,
    with faithful fallbacks for the few inputs not persisted to the CSV."""
    values, imputed = [], []
    for col in _RACE_FEAT_COLS:
        if col in _NOT_IN_CSV:
            values.append(0.0)
            imputed.append(col)
        else:
            values.append(float(row.get(col, 0.0)))

    # predicted_quali_pos: real fallback used at f1_2026_predictor.py:4672-4678
    # -- use quali_pos_next when meaningful, else avg_grid (midfield proxy).
    qpn = row.get("quali_pos_next", 0.0)
    has_next_quali = (
        "quali_pos_next" in feat_round.columns and
        feat_round["quali_pos_next"].notna().sum() >= 10 and
        feat_round["quali_pos_next"].nunique() >= 5
    )
    predicted_quali_pos = float(qpn) if (has_next_quali and qpn > 0) else float(row.get("avg_grid", 11.5))
    values.append(predicted_quali_pos)
    return np.array(values, dtype=float), imputed


def _tree_shap_for_driver(model_file: str, loader, row, feat_round, label):
    try:
        import shap
    except ImportError:
        print(f"   ⚠  shap not installed — skipping {label} layer 2 (pip install shap)")
        return None
    if not Path(model_file).exists():
        print(f"   ⚠  {model_file} not found — skipping {label}")
        return None

    booster = loader(model_file)
    x_row, imputed = _build_x_pred_row(row, feat_round)
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(x_row.reshape(1, -1))
    sv = np.array(sv).reshape(-1)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).reshape(-1)[0])

    pairs = sorted(zip(_ALL_RACE_COLS, sv), key=lambda t: abs(t[1]), reverse=True)
    return {
        "label": label, "pairs": pairs, "base_value": base_value,
        "predicted": base_value + float(sv.sum()), "imputed": imputed,
    }


# ─────────────────────────────────────────────────────────────
#  Report assembly
# ─────────────────────────────────────────────────────────────
def explain_driver(feat_round: pd.DataFrame, code: str, top_n: int, out_lines: list):
    if code not in feat_round["code"].values:
        out_lines.append(f"\n⚠  Driver '{code}' not found in this round's predictions — skipped.")
        return
    row = feat_round[feat_round["code"] == code].iloc[0]
    rank = int(feat_round.sort_values("win_pct", ascending=False)
               .reset_index(drop=True)
               .query("code == @code").index[0]) + 1

    out_lines.append("\n" + "=" * 72)
    out_lines.append(f"{code} — {row.get('FullName', '?')} ({row.get('TeamName', '?')})"
                      f" — R{int(row['round_num'])} {row.get('race_name', '')}")
    out_lines.append(f"win_pct: {row['win_pct']:.2f}%   raw_score: {row['raw_score']:.4f}"
                      f"   (P{rank} of {len(feat_round)})")
    out_lines.append("=" * 72)

    # ── Layer 1 ──────────────────────────────────────────────────────
    terms, weights, has_next_quali, missing = _score_manual_terms(feat_round)
    out_lines.append(f"\nLayer 1 — score_manual additive breakdown "
                      f"(exact for columns available in {PRED_CSV}; "
                      f"regime: {'post-qualifying' if has_next_quali else 'pre-qualifying'})")

    row_terms = [(c, s.loc[row.name]) for c, s in terms.items()
                 if abs(s.loc[row.name]) > 1e-9]
    row_terms.sort(key=lambda t: t[1], reverse=True)
    shown_sum = sum(v for _, v in row_terms)

    for col, val in row_terms[:top_n]:
        out_lines.append(f"  {col:<28} {val:+.4f}")
    if len(row_terms) > top_n:
        out_lines.append(f"  ... ({len(row_terms) - top_n} more terms below threshold)")

    if missing:
        out_lines.append("\n  Not available (not persisted to " + PRED_CSV + "):")
        for col, wt in missing:
            out_lines.append(f"    {col:<28} nominal weight ≈ {wt:.3f}  — omitted, not guessed")

    out_lines.append(f"\n  Σ shown terms = {shown_sum:+.4f}   csv raw_score = {row['raw_score']:.4f}"
                      f"   Δ = {row['raw_score'] - shown_sum:+.4f}")
    out_lines.append("  (Δ comes from: the missing columns above, the feedback-loop weight blend"
                      " shown next, the 15% LSTM momentum blend, and the Bayesian per-driver"
                      " prior multiplier — all applied after score_manual().)")

    # ── Feedback-loop snapshot (approximate) ──────────────────────────
    fb_round, fb_imp = _load_feedback_snapshot(PRIORS_JSON)
    if fb_imp:
        out_lines.append(f"\n  Feedback-loop snapshot (approximate — most recent recorded "
                          f"XGB+LGBM blended importances, as of round {fb_round}; these get "
                          f"blended 60/40 into the season-history weights above on each live run,"
                          f" but the exact per-run values aren't persisted, only this top-10):")
        for k, v in sorted(fb_imp.items(), key=lambda kv: -kv[1])[:10]:
            out_lines.append(f"    {k:<28} {v:.4f}")

    # ── Layer 2 ──────────────────────────────────────────────────────
    out_lines.append(f"\nLayer 2 — sub-model diagnostic (SHAP TreeExplainer, saved models, "
                      f"units = predicted finishing position; negative = pushes prediction "
                      f"toward P1). This is what feeds the feedback-loop blend above — it is "
                      f"NOT independent attribution of win_pct.")

    import xgboost as xgb
    import lightgbm as lgb

    xgb_out = _tree_shap_for_driver(
        XGB_MODEL_FILE, lambda f: (lambda b: (b.load_model(f), b)[1])(xgb.Booster()),
        row, feat_round, "XGBoost race model")
    lgbm_out = _tree_shap_for_driver(
        LGBM_MODEL_FILE, lambda f: lgb.Booster(model_file=f),
        row, feat_round, "LightGBM race model")

    for result in (xgb_out, lgbm_out):
        if result is None:
            continue
        out_lines.append(f"\n  {result['label']} "
                          f"(base_value={result['base_value']:.2f}, "
                          f"predicted_pos={result['predicted']:.2f}):")
        for col, val in result["pairs"][:6]:
            flag = "  [imputed — not in CSV, treat as unreliable]" if col in result["imputed"] else ""
            out_lines.append(f"    {col:<24} {val:+.3f}{flag}")

    out_lines.append(f"\n  GP component: excluded. feat['xgb_pos'] (0.59 XGB / 0.13 LGBM / 0.20 GP"
                      f" blend, f1_2026_predictor.py:6954-6966) is computed but never read again —"
                      f" it does not reach score_manual, the LSTM blend, or monte_carlo_simulation."
                      f" GP currently contributes 0% to win_pct.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True, help="round_num to explain")
    ap.add_argument("--driver", nargs="*", default=None, help="driver code(s), e.g. ANT HAM")
    ap.add_argument("--top", type=int, default=None,
                     help="explain the top N drivers by win_pct instead of --driver")
    ap.add_argument("--terms", type=int, default=12, help="max Layer-1 terms to print per driver")
    ap.add_argument("--save", type=str, default=None, help="write the report to this file (optional)")
    args = ap.parse_args()

    if not Path(PRED_CSV).exists():
        sys.exit(f"ERROR: {PRED_CSV} not found — run the predictor at least once first.")

    df = pd.read_csv(PRED_CSV)
    feat_round = df[df["round_num"] == args.round].reset_index(drop=True)
    if feat_round.empty:
        sys.exit(f"ERROR: no rows for round_num={args.round} in {PRED_CSV}.")

    if args.driver:
        codes = args.driver
    elif args.top:
        codes = feat_round.sort_values("win_pct", ascending=False)["code"].head(args.top).tolist()
    else:
        codes = [feat_round.sort_values("win_pct", ascending=False)["code"].iloc[0]]

    out_lines = [
        f"shap_explain.py — read-only, on-demand report (not part of the automated pipeline)",
        f"Source: {PRED_CSV} (round {args.round}), {PRIORS_JSON}, {XGB_MODEL_FILE}, {LGBM_MODEL_FILE}",
    ]
    for code in codes:
        explain_driver(feat_round, code, args.terms, out_lines)

    report = "\n".join(out_lines)
    print(report)

    if args.save:
        Path(args.save).write_text(report + "\n")
        print(f"\n💾  Saved to {args.save}")


if __name__ == "__main__":
    main()
