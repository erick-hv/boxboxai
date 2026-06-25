# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BoxBoxAI is an F1 race-analyst Telegram bot for the 2026 season. It pairs Anthropic Claude (the narrative/analysis engine) with a custom ML race-winner predictor and three F1 data APIs. The same memory store, predictor, and Claude-integration patterns are shared across a production Telegram bot and a local CLI agent.

## Running

```bash
# Required environment variables (both processes)
export ANTHROPIC_API_KEY=sk-ant-...      # console.anthropic.com
export TELEGRAM_BOT_TOKEN=123456:ABC...  # @BotFather (bot only)
export F1_PREDICTOR_PATH=...             # optional override for predictor script path

# Production Telegram bot — runs app.run_polling() forever (Procfile: worker)
python3 boxboxai_bot.py

# Local interactive CLI agent (REPL; prompts for API key if unset)
python3 f1_agent.py

# ML predictor standalone — writes f1_2026_predicciones.csv
python3 f1_2026_predictor.py

# One-time memory bootstrap — loads completed 2026 races + telemetry into f1_memory_2026.json
python3 build_2026_memory.py

# OpenF1 column-availability diagnostic (run before changing predictor data fields)
python3 f1_diagnostico_openf1.py
```

There is no build, lint, or test suite. Deployment is on Railway via the `Procfile` (`worker: python3 boxboxai_bot.py`); Chromium for Playwright is lazy-installed on first use (containers lack it at build time). The `# DEPLOY-CHECK-MARKER` comment at the top of `boxboxai_bot.py` is a deploy-version sanity marker.

Install deps with `pip3 install -r requirements.txt`. `claude-sonnet-4-5` is the model used by both the bot and CLI agent.

## Architecture

### Three processes, one shared state

- **`boxboxai_bot.py`** (~8k lines) — the production bot. `main()` loads memory, registers Telegram handlers, and launches four async background loops. Free-text messages route through `handle_message()` → `ask_claude()`.
- **`f1_agent.py`** (~2.8k lines) — a developer CLI/REPL over the same memory and predictor. Useful for manually running `/predict`, `/ingest`, `/verify`, `/correct`, etc. without the Telegram layer.
- **`f1_2026_predictor.py`** (~3.2k lines) — the ML predictor, run as a subprocess by both the bot and agent (300s timeout) and standalone.

All three read/write **`f1_memory_2026.json`** (the single source of truth) and consume **`f1_2026_predicciones.csv`** (predictor output).

### Claude integration uses context injection, NOT tool use

There are no Anthropic tools defined. All analysis flows through `build_system_prompt()`, which assembles: compact episodic memory (last ~8 races), semantic facts, and dynamic context blocks (live session, news, weather, FIA documents, practice, circuit guide, driver stats) that are injected **only when available**. The system prompt is deliberately token-optimized. When adding a new data capability, the pattern is: fetch the data → inject it as a context block in the system prompt → let Claude reason over it. Do not reach for tool use unless deliberately changing this architecture.

### Memory model (`f1_memory_2026.json`)

- `semantic` — keyed knowledge facts (`circuit/monaco`, `driver/norris`, `team/mclaren`, `predictor/model`) with `tags`/`relevance_keys`; injected into the system prompt.
- `episodic` — one rich entry per completed race (winner, podium, qualifying, pitstops/tyre strategies, DNFs, championship standings, and a narrative `story`). `telemetry_source` is `"fastf1"` once enriched, else `"jolpica"`.
- `pending_predictions` — predictor output stored after qualifying, surfaced by `/predict` on race day.

### Data sources (graceful degradation)

1. **Jolpica-F1** (`https://api.jolpi.ca/ergast/f1`) — canonical results, standings, qualifying, pitstops. Ergast-compatible.
2. **OpenF1** (`https://api.openf1.org/v1`) — laps, stints, pit stops, weather, race-control messages.
3. **FastF1** — telemetry/sector data, cached in `./f1_cache/` (the SQLite HTTP cache + per-season dirs). Optional; FastF1 data lags the race by ~48h, which the enrichment loop accounts for.

The system degrades cleanly: works without FastF1 (Jolpica only), without a predictor CSV (grid-based analysis), and without qualifying (pre-quali preview). Plus FIA stewards' documents scraped from fia.com via **Playwright/Chromium** (fia.com blocks plain `requests`; Playwright sync API must run in a separate thread to avoid the asyncio conflict — see git history).

### Bot background loops (each ~every 6h, launched post-init)

- **`auto_ingest_loop`** — detects new race results, fetches telemetry, auto-generates a post-race briefing, runs prediction-vs-reality self-correction.
- **`auto_predictor_loop`** — Saturday-night: on qualifying availability, spawns the predictor subprocess and stores `pending_predictions`.
- **`auto_memory_enrichment_loop`** — waits ~48h post-race for FastF1, enriches episodic memory with tyre strategies / sector bests / stint data. **Retries are capped at 14 days** then gives up (avoids the endless re-enrichment loop — see commit 57b18de); emits a single batched owner alert per run.
- **`notification_loop`** — session reminders (timezone-converted) and a Sunday weekly digest, driven by `boxboxai_sessions.json`.

State/cache files (gitignored or runtime-generated): `boxboxai_sessions.json` (user profiles, timezone, prefs, per-user chat history — last ~16 msgs to Claude), `boxboxai_news_cache.json` (RSS, 30-min refresh), `boxboxai_enrichment.json`, `boxboxai_predictor_state.json` / `f1_scheduler_state.json`.

### Predictor model (`f1_2026_predictor.py`, "v7.0 hybrid")

- **Races < 6:** Bayesian priors from `f1_2026_bayesian_priors.json` (per-driver score multipliers + rolling prediction-error history) — XGBoost isn't trained yet.
- **Races ≥ 6:** XGBoost regression over ~24 engineered features (qualifying position, recent form, circuit/compound scores, tyre-deg slope, sector deltas, mechanical/DNF risk, momentum, teammate comparison).
- **Monte Carlo** (~10k sims/driver) produces `win_pct` / `podium_pct` / avg position.
- Output: `f1_2026_predicciones.csv` (one row per driver). `f1_2026_predictions.csv` is an older redundant format; prefer `predicciones`.

## Conventions

- Output files use Spanish names (`predicciones`) alongside English — match the existing file when referencing.
- API calls are throttled (~0.4s between predictor requests) to respect rate limits; preserve these delays.
- The bot alerts the owner via `alert_owner()` on API failures (`AuthenticationError`, `RateLimitError`) and batches enrichment/ingestion alerts into single messages to avoid spam.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Bayesian Priors (Railway persistence)

`f1_2026_bayesian_priors.json` is committed to git and deployed from there.
Updates the predictor writes during a Railway run are lost on next redeploy.
After each race weekend: run `python3 f1_2026_predictor.py` locally to get updated priors,
then `git add f1_2026_bayesian_priors.json && git commit -m "chore: update Bayesian priors RNN"`.
To use a Railway persistent volume instead, set env var `F1_PRIORS_FILE=/data/f1_2026_bayesian_priors.json`.
