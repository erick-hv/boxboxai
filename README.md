# F1 2026 ML Race Predictor

**A best-in-class standalone machine learning predictor for Formula 1 race outcomes.**

---

## What it does

The predictor fetches live data from three public APIs, runs a three-model machine learning ensemble, simulates 10,000 race scenarios, and outputs per-driver win probabilities, podium probabilities, and confidence intervals — all from a single command.

- **Live data** from Jolpica (race results, standings, qualifying), OpenF1 (lap times, tyre stints, pit stops, weather), and Open-Meteo (rain nowcast)
- **Three-model ensemble**: XGBoost (two-stage quali + race), LightGBM, and Gaussian Process Regression — blended by a stacking meta-model that learns optimal per-circuit-type weights from leave-one-out validation
- **10,000 Monte Carlo simulations** per run — explicit safety car probability, undercut windows, DNF risk by component, and weather perturbations
- **Full grid output**: win%, podium%, expected finishing position, P10/P90 confidence intervals, and uncertainty estimates for all 22 drivers

---

## Feature set

### Data sources

| Source | What it provides |
|--------|-----------------|
| **Jolpica** (`api.jolpi.ca/ergast/f1`) | Race results, standings, qualifying positions, pit stop data, lap times |
| **OpenF1** (`api.openf1.org/v1`) | Sector times, tyre stint data, pit stop timing, race control messages, weather feed |
| **Open-Meteo** | Rain probability nowcast for race location and weekend |
| **FIA behavioral data** | Historical overtaking events, qualifying consistency, tyre management rates (2023–2025) |

### Machine learning models

| Model | Role |
|-------|------|
| **XGBoost (quali stage)** | Predicts qualifying position from season-to-date features when actual quali hasn't happened yet |
| **XGBoost (race stage)** | Regresses race finishing position from 21 engineered features; online learning via warm start |
| **LightGBM** | Parallel race position regressor with gradient-boosted trees; same feature set as XGBoost |
| **Gaussian Process Regression** | Explicitly models prediction uncertainty as a function of data density; produces per-driver σ (epistemic uncertainty) alongside position estimate |
| **Stacking meta-model** | Logistic regression that learns optimal XGB/LGBM blend weights per circuit type (high-speed / street / technical / mixed) from LOO validation history |
| **Bayesian priors** | Per-driver score multipliers derived from prediction error history; applied at all stages and updated after every race |

### Features (21 race model features)

**Season history**
- `champ_pts` — current championship points
- `avg_finish` — season average finishing position
- `avg_grid` — season average grid position
- `fl_rate` — fastest lap rate
- `lap_std` — lap-time consistency (σ within races)
- `recent_form` — exponentially-weighted results (last 5 races)
- `momentum_pos` — exp-weighted recent starting position
- `sprint_pts` — sprint race points accumulated

**Circuit and tyre**
- `fp_avg_delta` — practice pace delta vs. field mean
- `avg_sector_delta` — sector time delta vs. teammate
- `tyre_deg_slope` — tyre degradation rate (s/lap linear fit)
- `avg_pitstop` — average pit stop time this season

**Race-specific**
- `lap1_gain` — average positions gained on lap 1
- `teammate_delta` — race pace delta vs. teammate
- `sc_gain_avg` — historical position gain under safety car
- `dnf_rate` — season DNF rate (mechanical + accident)
- `penalty_count` — penalty incidents this season

**Behavioral profiles (2023–2025 historical)**
- `overtaking_ability` — net positions gained from P6–P15 starts
- `quali_consistency` — qualifying gap σ vs. teammate across seasons
- `tyre_management_index` — pace gain in final 20% of stints vs. field

**Driver-circuit compatibility**
- `compatibility_score` — cosine similarity of 3D driver embedding (Aggression, Consistency, Endurance) and 3D circuit embedding (Speed, Technical, Endurance)

### Next-race features (when qualifying has happened)

When pre-race qualifying data is available, a second feature tier is unlocked and qualifying gets a 55% dominant weight:

- `quali_pos_next` — actual qualifying position (P1–P22)
- `fp_next_delta` — FP pace delta vs. field at this circuit
- `fp2_next_longrun` — FP2 long-run pace delta
- `sector_balance` — σ across S1/S2/S3 (lower = more balanced)
- `soft_pace_delta`, `medium_pace_delta` — compound-specific pace
- `tyre_deg_rate` — observed tyre deg at this circuit in practice
- `corner_profile_score` — corner-speed profile match for this circuit type
- `race_sim_delta` — FP1 race simulation pace delta

### Outputs

| Column | Description |
|--------|-------------|
| `win_pct` | Model ensemble win probability (%) |
| `win_mc_pct` | Monte Carlo win probability (%) |
| `podium_mc_pct` | Monte Carlo podium probability (%) |
| `avg_mc_pos` | Expected finishing position |
| `p10_pos` / `p90_pos` | P10–P90 confidence interval on finishing position |
| `epistemic_unc` | Model disagreement uncertainty (XGB vs LGBM vs GP) |
| `aleatoric_unc` | Monte Carlo spread (inherent race randomness) |
| `gp_uncertainty` | Gaussian Process σ (data-density uncertainty) |
| `compatibility_score` | Driver-circuit style match (–1 to 1) |
| `mechanical_risk` | Component-level DNF probability |
| `accident_risk` | Incident-based DNF probability |
| `circuit_affinity` | Historical driver performance at this circuit |
| `wet_weather_delta` | Driver pace delta in wet vs. dry (negative = faster in wet) |

### Validation and calibration

- **Leave-one-out (LOO) cross-validation** across all completed rounds — MAE in positions and Brier score per round
- **Calibration curve**: win probability buckets vs. empirical win rate
- **Feature importance tracking**: XGBoost and LightGBM importances logged per round; fed back into manual weight system
- **Bayesian self-correction**: prediction error per driver (predicted rank vs. actual rank) updates a per-driver correction multiplier stored in priors and applied to subsequent predictions
- **Model disagreement flag**: drivers where XGB and LGBM disagree by > 2 positions are flagged `[models disagree]` in output

---

## Files

| File | Description |
|------|-------------|
| `f1_2026_predictor.py` | Main predictor — data fetch, feature engineering, ensemble models, Monte Carlo, output |
| `f1_update_priors.py` | Standalone post-race prior updater; run after each race before the next prediction |
| `f1_2026_bayesian_priors.json` | Bayesian priors (per-driver multipliers), LOO validation history, stacking meta-model coefficients |
| `f1_2026_driver_profiles.json` | Historical behavioral profiles (2023–2025) and 3D driver style embeddings |
| `f1_2026_predicciones.csv` | Latest prediction output (one row per driver) |
| `f1_2026_xgb_race_model.json` | Saved XGBoost race model (warm-started each round) |
| `f1_2026_lgbm_race_model.txt` | Saved LightGBM race model (warm-started each round) |
| `requirements.txt` | Python dependencies |
| `f1_cache/` | Cached API responses (FastF1 SQLite cache) |

---

## How to run

```bash
pip install -r requirements.txt

# Run the predictor (writes f1_2026_predicciones.csv)
python f1_2026_predictor.py

# Update Bayesian priors after a race result is official
python f1_update_priors.py --round N
```

Run `f1_2026_predictor.py` **before qualifying** for an early race simulation driven by season-to-date history and practice pace. Run it again **after qualifying** to unlock the 55% qualifying-weighted prediction with full next-race feature set.

---

## Race weekend workflow

| When | What to do | What you get |
|------|-----------|--------------|
| **Thursday / FP1** | `python f1_2026_predictor.py` | Early race simulation from FP1 delta, corner speed profile, historical affinity |
| **Friday / FP2** | `python f1_2026_predictor.py` | Tyre deg slope, long-run pace, compound-specific deltas added |
| **Saturday post-quali** | `python f1_2026_predictor.py` | Full prediction: 55% qualifying weight + all race features. Stacking meta-model picks XGB/LGBM weights for this circuit type |
| **Sunday –2h** | `python f1_2026_predictor.py` | Weather nowcast upgrade if rain probability has changed overnight |
| **Sunday post-race** | `python f1_update_priors.py --round N` then `python f1_2026_predictor.py` | Priors updated, prediction record logged, next-round early prediction ready |

---

## 2026 season status

**Round 8 — Austrian GP (Silverstone Circuit, next)**
- R8 result: Russell win predicted ✓ — Brier score **0.0315** (27% better than random baseline)
- Championship: ANT leads 171 pts · HAM 125 pts · RUS 131 pts (after R8)
- Model: 3-way ensemble (XGB + LGBM + GP), online warm-start learning, 8 races of Bayesian priors
- Stacking meta-model: 8 LOO rounds — XGB/LGBM weights learned per circuit type

---

## Technical architecture

The predictor runs as a single Python script with no server or daemon. On each invocation it fetches fresh data from Jolpica and OpenF1 over HTTP (responses are cached to `f1_cache/` by FastF1 and request-level TTLs), builds a 21-column feature matrix (one row per driver), and passes it through three models. XGBoost and LightGBM are trained incrementally via warm start — each new race adds 20 trees weighted 3× to emphasise recency, while the full training set provides the base. The Gaussian Process uses an RBF + WhiteKernel composite kernel with `normalize_y=True`; its σ output explicitly measures how sparse the feature space is around each driver — useful signal in an 8-race season. A logistic regression stacking meta-model trained on LOO validation history learns per-circuit-type blending weights (high-speed, street, technical, mixed) for the XGB/LGBM pair; GP always gets a fixed 20% share. The blended score feeds a Monte Carlo engine that runs 10,000 race simulations with safety car Poisson draws, undercut probability windows, and component-level DNF risk by team, producing full probability distributions rather than point estimates. Bayesian priors — per-driver correction multipliers derived from rolling prediction error — are applied before scoring and updated after every race result. Driver behavioral profiles derived from 2023–2025 race data add five historical features (overtaking ability, qualifying consistency, wet-weather delta, DNF rate, tyre management index) and three-dimensional style embeddings whose cosine similarity with circuit embeddings produces a driver-circuit compatibility score.
