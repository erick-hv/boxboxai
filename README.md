# BoxBoxAI

A Formula 1 analysis Telegram bot that gives grounded, data-driven answers about the 2026 season — no hallucinated lap times, no invented grid orders.

---

## What it does

BoxBoxAI is a conversational F1 analyst you interact with over Telegram. Ask it anything about the 2026 season and it answers from real data: race results, qualifying sheets, FIA stewards decisions, tyre strategy, circuit characteristics, championship projections, and ML-based race predictions. The design priority throughout is accuracy over confidence — if the data isn't there, it says so plainly rather than improvising.

Concrete things it handles well:

- **Race debrief** — winner, podium, DNFs, fastest lap, tyre strategies, full classification
- **Qualifying** — pole positions, sector times, grid order
- **FIA stewards decisions** — fetched live from fia.com (via Playwright, since their site blocks `requests`), parsed and surfaced for penalty/incident questions
- **Driver deep-dive** — career stats, recent form, teammate comparison, championship trajectory
- **Circuit guides** — sector characteristics, DRS zones, overtaking difficulty, tyre degradation tendencies, 2026 Straight Mode activation points
- **Race predictions** — XGBoost + Monte Carlo model, updated after qualifying on race weekend
- **Championship scenarios** — points math for "what does Norris need to close the gap"
- **Live session data** — OpenF1 lap times and sector splits during race weekends

---

## Architecture

Three Python processes sharing one memory store:

```
boxboxai_bot.py     ← production Telegram bot (~8,700 lines)
f1_agent.py         ← developer CLI/REPL for the same functions
f1_2026_predictor.py ← ML predictor, spawned as a subprocess
         │
         └── f1_memory_2026.json   (single source of truth)
```

### Context injection, not tool use

There are no Anthropic tools defined. Every analysis flows through `ask_claude()` → `_gather_context()` → `build_system_prompt()`. The system prompt is assembled from conditionally-injected context blocks — each block only included when its trigger fires:

```
NEXT_RACE      triggered by next-race / prediction / weather queries
NEWS           triggered by news/practice/qualifying keywords
LIVE_SEARCH    triggered by live session questions when OpenF1 has no data
FIA_DOCS       triggered by penalty/retirement/incident/stewards keywords
WEATHER        triggered by weather keywords
LIVE_SESSION   triggered by OpenF1 live session feed
SESSION_DATA   triggered by FP1/FP2/FP3/qualifying/sprint keywords
CIRCUIT        triggered by circuit name or strategy keywords
DRIVER_STATS   triggered by career stats questions
DRIVER_PROFILE triggered by driver deep-dive questions
RACE_REPLAY    triggered by replay/lap-by-lap questions
CHAMPIONSHIP_SCENARIOS triggered by standings/gap/title keywords
FAN_PROFILE    triggered by fan tracking context
HISTORY        triggered by historical comparison keywords
PREDICTION_RECORD triggered by accuracy/prediction keywords
USER           always injected (short personalization block)
```

Each block has an explicit character limit matched to its real content size. These limits were audited after discovering silent truncation bugs that were cutting 52–85% of content — every block in `build_system_prompt()` now logs its post-truncation size at `DEBUG` level.

### Memory model (`f1_memory_2026.json`)

| Key | Contents |
|-----|----------|
| `semantic` | Keyed knowledge facts (`circuit/monaco`, `driver/norris`, `team/mclaren`, etc.) with `tags` and `relevance_keys` |
| `episodic` | One rich entry per completed race — winner, podium, qualifying, pitstops, tyre strategies, DNFs, race control messages, championship standings snapshot, narrative `story`. `telemetry_source` is `"fastf1"` once enriched, else `"jolpica"` |
| `pending_predictions` | Predictor output stored after qualifying, surfaced by `/predict` on race day |

### Background loops (launched post-init, each ~6h)

| Loop | What it does |
|------|-------------|
| `auto_ingest_loop` | Detects new results, fetches telemetry, generates post-race briefing, runs prediction self-correction |
| `auto_predictor_loop` | Saturday night: on qualifying availability, spawns predictor subprocess and stores `pending_predictions` |
| `auto_memory_enrichment_loop` | Waits ~48h post-race for FastF1, enriches episodes with tyre strategies / sector bests / stint data. Retries capped at 14 days to avoid endless re-enrichment |
| `notification_loop` | Session reminders (timezone-aware), Sunday weekly digest |

---

## Data sources

| Source | What it provides | Notes |
|--------|-----------------|-------|
| **Jolpica-F1** (`api.jolpi.ca/ergast/f1`) | Canonical results, standings, qualifying, pitstops | Ergast-compatible API; primary results source |
| **OpenF1** (`api.openf1.org/v1`) | Lap times, stints, pit stops, weather, race control messages | Used for live session data during race weekends |
| **FastF1** | Telemetry, sector times, tyre stint data | Cached in `./f1_cache/`. Data lags ~48h post-race; enrichment loop accounts for this |
| **FIA.com** | Stewards decisions, technical directives, circuit maps | Scraped via Playwright/Chromium — fia.com returns 403 to `requests`. Playwright sync API runs in a thread to avoid asyncio conflict |
| **Motorsport.com** | Fallback news and article context | Extracted via JSON-LD structured data embedded in article pages |
| **RSS feeds** | News headlines from F1 outlets | 30-minute refresh cache (`boxboxai_news_cache.json`) |
| **Pirelli** | Tyre compound data | No stable public API endpoint found; not currently automated |

The system degrades cleanly at each layer: works without FastF1 (Jolpica only), without a predictor CSV (grid-based analysis), without qualifying data (pre-quali preview mode).

---

## The predictor (`f1_2026_predictor.py`)

Version 7.0 — "Bayesian + Momentum + Component Risk"

### Two-phase model

**Races < 6:** Bayesian priors from `f1_2026_bayesian_priors.json` — per-driver score multipliers derived from pre-season expectations, blended with rolling prediction-error history. XGBoost isn't trained yet (insufficient data).

**Races ≥ 6:** XGBoost regression trained on ~24 engineered features per driver:

- Qualifying position (real or sentinel 22.0 pre-qualifying)
- Recent form (exponentially weighted last 5 results)
- Circuit score (historical driver performance at this venue type)
- Compound/tyre degradation slope
- Sector delta vs. teammate
- Mechanical / DNF risk score
- Momentum (points trend over last 3 rounds)
- Teammate comparison

A **Leave-One-Out cross-validation** loop measures MAE in positions. XGBoost feature importances feed back into the manual weight system as a "feedback loop" — the learned importances update how the weights are blended.

### Monte Carlo

10,000 simulations per run produce `win_pct`, `podium_pct`, and average finishing position per driver. Output: `f1_2026_predicciones.csv` (one row per driver, with `round_num` stamp for staleness detection).

### Pre-qualifying mode

When qualifying hasn't happened, `quali_pos_next` is set to sentinel value `22.0` for all drivers. `_is_pre_qualifying_csv()` detects this and `format_predictor_for_claude()` suppresses the qualifying position column, replacing it with a "PRE-QUALIFYING PREVIEW" header so Claude doesn't treat grid-agnostic probabilities as post-qualifying predictions.

### Stale-CSV protection

The CSV is stamped with `round_num`. `get_predictor_context(expected_round=N)` returns empty if the CSV round doesn't match — prevents a Spanish GP prediction from leaking into Austrian GP analysis.

---

## Key engineering decisions

### Anti-hallucination grounding

The single biggest risk with an LLM F1 bot is invented positions and times. Three mechanisms address this:

**CLASSIFICATION FACT injection** — when a query mentions a driver and a race, the driver's exact finishing position (or DNF with cause) is extracted from `full_classification` in episodic memory and injected as a hard fact with `"Use this exact position when answering."` This prevents Claude from inventing mid-field positions that weren't in the compact race summary.

**RACE CONTROL FACT injection** — in-race penalties (e.g., 5-second time penalties served during a pit stop) don't change the final classification and can't be inferred from results. The race control message feed is searched for the driver's surname and injected separately. Crucially: if race control messages exist for a race but none mention the queried driver, that *absence* is also injected as informative negative evidence.

**"No stewards doc = mechanical, not missing data"** — the system prompt instructs Claude that absence of FIA stewards documents for a DNF is itself informative. Stewards investigate on-track incidents; mechanical failures don't generate documents. Claude is told to say "no stewards investigation was opened, which points to a mechanical issue" rather than "I don't have that information."

### Driver-code word-boundary matching

Mapping text to driver codes (`"hamilton"` → `HAM`) is a false-positive minefield. Common English words like `"per"`, `"had"`, `"gas"`, `"ham"` were all in early versions of the map, producing absurd matches (`"performance"` → PER, `"I had a good lap"` → HAD). The current implementation uses word-boundary regex matching and explicitly removes common-word codes from the map. Tested with regressions like `"is barcelona a 1 or 2 stopper"` (must not match PER), `"fill up the gas tank"` (must not match GAS).

### Context block size audit

An audit found six blocks where the `[:N]` limit in `ctx_blocks.append()` silently cut between 52% and 85% of real content. The RACE REPLAY block (typically ~597 chars) was routed through the USER block's 150-char limit. HISTORY was capped at 200 when real content runs to 1,200. Each block now has a limit matched to its actual content ceiling, and `build_system_prompt()` logs `log.debug(f"ctx_block LABEL: N chars")` after every append for future auditing.

### Spa/Spanish substring collision

`"spanish"` contains `"spa"` — if circuit lookup uses simple substring matching, `"tell me about the Spanish GP"` resolves to Spa-Francorchamps. The `_resolve_circuit_key()` function uses word-boundary matching (`\bspa\b`) to prevent this while keeping `"spa francorchamps"` and `"belgium gp at spa"` working correctly.

---

## Setup and running

### Environment variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # Required — console.anthropic.com
export TELEGRAM_BOT_TOKEN=123456:ABC...  # Required — @BotFather on Telegram
export F1_PREDICTOR_PATH=...             # Optional — overrides default predictor script path
```

### Install

```bash
pip install -r requirements.txt

# Playwright needs its browser binaries (done once)
playwright install chromium
```

Playwright/Chromium is **lazy-installed** in the Railway container on first use — the container doesn't have it at build time, so `get_circuit_guide()` and `fetch_fia_race_documents()` trigger installation on the first call that needs it.

### Run

```bash
# Production Telegram bot
python3 boxboxai_bot.py

# Developer CLI/REPL (same memory, no Telegram)
python3 f1_agent.py

# ML predictor standalone (writes f1_2026_predicciones.csv)
python3 f1_2026_predictor.py

# One-time memory bootstrap (loads completed 2026 races into f1_memory_2026.json)
python3 build_2026_memory.py

# OpenF1 column-availability diagnostic (run before changing predictor data fields)
python3 f1_diagnostico_openf1.py
```

### Deployment

Railway via `Procfile`:

```
worker: python3 boxboxai_bot.py
```

The deploy-version sanity marker at the top of `boxboxai_bot.py` (`# DEPLOY-CHECK-MARKER`) lets you verify Railway picked up the latest push.

### Runtime state files

These are generated at runtime and gitignored:

| File | Contents |
|------|----------|
| `f1_memory_2026.json` | Shared memory store (semantic + episodic + predictions) |
| `boxboxai_sessions.json` | User profiles, timezone, preferences, per-user chat history (last 6 messages) |
| `boxboxai_news_cache.json` | RSS cache, 30-min refresh |
| `boxboxai_enrichment.json` | FastF1 enrichment state |
| `boxboxai_predictor_state.json` | Auto-predictor run state |
| `f1_2026_predicciones.csv` | Latest predictor output |
| `f1_cache/` | FastF1 SQLite HTTP cache |

---

## Testing

```bash
python3 -m pytest test_boxboxai_core.py -v
```

86 tests covering:

- **Trigger functions** — `_needs_fia_docs`, `_is_live_session_question`, `_is_weather_query` and their false-positive cases
- **Driver resolution** — `resolve_driver_code`, `_resolve_multiple_driver_codes`, word-boundary edge cases
- **Predictor staleness** — `get_predictor_context` round-stamp matching, missing CSV, old CSV without `round_num` column
- **Pre-qualifying detection** — sentinel value detection, block formatting with/without qualifying positions
- **Tyre strategy routing** — regression against the key-name mismatches that were silently dropping tyre data
- **Circuit guide** — Spa/Spanish collision, zone data injection, graceful degradation when Playwright fails
- **FIA circuit map parsing** — `_parse_circuit_map_pdf_text` against real pypdf-extracted text from the 2026 Barcelona circuit map PDF
- **Context truncation limits** — every context block verified at its real size boundary
- **`/reingest` race condition** — regression for the `mem_ref[0]` vs stale local copy bug
- **`/debug_context` formatting** — `_format_debug_context_report` char-count accuracy, preview capping, empty-block omission

No mocking of external APIs in the core test suite. Tests that touch `cmd_reingest` and `cmd_debug_context` use `AsyncMock` with patches on specific bot functions rather than full integration.

---

## Bot commands

| Command | Who | What |
|---------|-----|-------|
| `/start` | All users | Welcome message with feature overview |
| `/help` | All users | Command reference |
| `/standings` | All users | Current driver championship standings |
| `/constructors` | All users | Constructor championship standings |
| `/predict` | All users | Race prediction for the next/current round |
| `/winner` | All users | Quick win probability summary |
| `/reingest <round>` | Owner only | Force re-fetch a race result from Jolpica (clears ingest/enrichment/predictor state for that round) |
| `/debug_context <query>` | Owner only | Dry-run the context pipeline for a query — shows which blocks would be injected, their character counts, and a 100-char preview of each, without making a Claude API call |

Free-text messages route through `handle_message()` → `ask_claude()` for full conversational analysis.

### CLI agent commands (`f1_agent.py`)

The REPL exposes the same memory and predictor without the Telegram layer:

`/memory`, `/season`, `/standings`, `/preview`, `/add_race`, `/ingest`, `/correct`, `/predict`

---

## Known limitations

**Monolith.** `boxboxai_bot.py` is a single ~8,700-line file. It works and the logic is well-organized into sections, but it is not structured as a package with modules. This was a deliberate choice to keep deployment simple (one file to push) rather than an oversight.

**FastF1 lag.** Telemetry data isn't available until ~48 hours after a race. Enrichment runs automatically but answers about specific sector times or stint lengths will be generic immediately post-race.

**Calibration tracking is limited.** Prediction accuracy is tracked (the `PREDICTION RECORD` block and self-correction loop), but the calibration methodology is informal — win percentage predictions aren't formally compared against empirical frequencies.

**No Pirelli API.** Tyre compound data for race weekends (allocation per driver, compound characteristics) would improve predictions meaningfully. No stable public Pirelli endpoint exists.

**Race weekend timing.** The session reminder and auto-ingest calendar is hardcoded for the 2026 F1 season. It will need updating for 2027.

**English and Mexican Spanish only.** The bot detects user language and responds in kind, but only these two are tested.

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Bot framework | `python-telegram-bot` 22.7 |
| LLM | Anthropic API — `claude-sonnet-4-5` |
| ML predictor | XGBoost, scikit-learn, NumPy, pandas |
| F1 telemetry | FastF1 |
| FIA document scraping | Playwright (Chromium) + pypdf |
| HTTP | requests |
| Deployment | Railway (single worker dyno) |
| Testing | pytest |

---

## Why this exists

This started as a personal project to combine a long-standing F1 interest with hands-on work on production AI systems — specifically, exploring what it takes to make an LLM give *accurate* answers in a domain where hallucination is immediately obvious (you either got the race result right or you didn't). The bot is in daily use during the 2026 season and has been a useful forcing function for finding real failure modes in LLM-grounded systems: truncation bugs that silently drop data, false-positive trigger functions that inject irrelevant context, and the gap between "Claude sounds confident" and "Claude has the actual data."

Development has been AI-assisted throughout using Claude Code, which is fitting given the subject matter.
