# Graph Report - .  (2026-06-15)

## Corpus Check
- 22 files · ~74,190 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 522 nodes · 1179 edges · 35 communities (31 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Background Loops & Auto-Ingestion|Background Loops & Auto-Ingestion]]
- [[_COMMUNITY_Telegram Bot Commands|Telegram Bot Commands]]
- [[_COMMUNITY_ML Feature Engineering|ML Feature Engineering]]
- [[_COMMUNITY_Predictor Core & OpenF1 Fetch|Predictor Core & OpenF1 Fetch]]
- [[_COMMUNITY_Jolpica API Layer|Jolpica API Layer]]
- [[_COMMUNITY_Claude Query Engine|Claude Query Engine]]
- [[_COMMUNITY_Race Context Fetchers|Race Context Fetchers]]
- [[_COMMUNITY_Bot Formatting & Output|Bot Formatting & Output]]
- [[_COMMUNITY_CLI Agent & Utilities|CLI Agent & Utilities]]
- [[_COMMUNITY_Memory & State Persistence|Memory & State Persistence]]
- [[_COMMUNITY_Agent CLI Commands|Agent CLI Commands]]
- [[_COMMUNITY_Notification System|Notification System]]
- [[_COMMUNITY_Data Fetch Layer|Data Fetch Layer]]
- [[_COMMUNITY_Race Preview & Standings|Race Preview & Standings]]
- [[_COMMUNITY_Bayesian & Monte Carlo Sim|Bayesian & Monte Carlo Sim]]
- [[_COMMUNITY_OpenF1 Data Pipeline|OpenF1 Data Pipeline]]
- [[_COMMUNITY_News & RSS Feed|News & RSS Feed]]
- [[_COMMUNITY_Championship & Rate Limiting|Championship & Rate Limiting]]
- [[_COMMUNITY_Predictor Scheduler State|Predictor Scheduler State]]
- [[_COMMUNITY_System Prompt & Memory Retrieval|System Prompt & Memory Retrieval]]
- [[_COMMUNITY_Weather Context|Weather Context]]
- [[_COMMUNITY_User Stats Tracking|User Stats Tracking]]
- [[_COMMUNITY_Safety & Language Detection|Safety & Language Detection]]
- [[_COMMUNITY_FIA Docs & Web Search|FIA Docs & Web Search]]
- [[_COMMUNITY_Episode Verification Pipeline|Episode Verification Pipeline]]
- [[_COMMUNITY_Post-Race Briefing|Post-Race Briefing]]
- [[_COMMUNITY_OpenF1 Diagnostics|OpenF1 Diagnostics]]
- [[_COMMUNITY_Playwright FIA Scraper|Playwright FIA Scraper]]
- [[_COMMUNITY_Session Time Formatting|Session Time Formatting]]
- [[_COMMUNITY_DNF Component Analysis|DNF Component Analysis]]
- [[_COMMUNITY_Lap 1 Position Data|Lap 1 Position Data]]
- [[_COMMUNITY_Teammate Delta|Teammate Delta]]
- [[_COMMUNITY_Circuit Type Profile|Circuit Type Profile]]
- [[_COMMUNITY_Race Weather Fetch|Race Weather Fetch]]
- [[_COMMUNITY_Bayesian Prior Persistence|Bayesian Prior Persistence]]

## God Nodes (most connected - your core abstractions)
1. `DataFrame` - 55 edges
2. `main()` - 51 edges
3. `ask_claude()` - 40 edges
4. `handle_message()` - 27 edges
5. `main()` - 24 edges
6. `DEFAULT_TYPE` - 23 edges
7. `Update` - 22 edges
8. `api_get()` - 19 edges
9. `auto_ingest()` - 17 edges
10. `race_to_episode()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `fetch_telemetry_snapshot()` --semantically_similar_to--> `enrich_episode_with_telemetry()`  [INFERRED] [semantically similar]
  f1_agent.py → boxboxai_bot.py
- `run_predictor()` --semantically_similar_to--> `run_predictor_subprocess()`  [INFERRED] [semantically similar]
  f1_agent.py → boxboxai_bot.py
- `_scheduler_loop()` --semantically_similar_to--> `auto_predictor_loop()`  [INFERRED] [semantically similar]
  f1_agent.py → boxboxai_bot.py
- `auto_ingest()` --implements--> `Graceful Degradation (Jolpica→OpenF1→FastF1)`  [INFERRED]
  f1_agent.py → CLAUDE.md
- `load_memory()` --semantically_similar_to--> `load_f1_memory()`  [INFERRED] [semantically similar]
  f1_agent.py → boxboxai_bot.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three Processes, One Shared Memory Store** — boxboxai_bot_load_f1_memory, f1_agent_load_memory, build_2026_memory_main, f1_memory_2026_json [EXTRACTED 1.00]
- **Predictor Subprocess Invocation Pattern** — boxboxai_bot_run_predictor_subprocess, f1_agent_run_predictor, f1_2026_predictor_main [EXTRACTED 1.00]
- **Context Injection Pipeline (news+session+FIA+live)** — boxboxai_bot_get_news_context, boxboxai_bot_get_session_context, boxboxai_bot_fetch_fia_race_documents, boxboxai_bot_live_search_f1, boxboxai_bot_build_system_prompt [EXTRACTED 1.00]

## Communities (35 total, 4 thin omitted)

### Community 0 - "Background Loops & Auto-Ingestion"
Cohesion: 0.07
Nodes (36): alert_owner(), auto_ingest_loop(), auto_predictor_loop(), build_rich_story(), _check_and_ingest(), _check_and_run_predictor(), enrich_episode_with_telemetry(), enrich_qualifying_with_telemetry() (+28 more)

### Community 1 - "Telegram Bot Commands"
Cohesion: 0.12
Nodes (36): cmd_compare(), cmd_debate(), cmd_help(), cmd_hottake(), cmd_live(), cmd_notifications(), cmd_predict(), cmd_schedule() (+28 more)

### Community 2 - "ML Feature Engineering"
Cohesion: 0.11
Nodes (31): DataFrame, calc_dnf_rate(), collect_circuit_sector_score(), collect_compound_strategy(), collect_lap1_gain(), collect_lap_consistency(), collect_practice_pace(), collect_quali_gap_teammate() (+23 more)

### Community 3 - "Predictor Core & OpenF1 Fetch"
Cohesion: 0.12
Nodes (30): build_features(), build_reliability(), main(), of1_check_available(), of1_collect_lap_consistency(), of1_collect_next_race_fp(), of1_collect_next_race_qualifying(), of1_collect_pitstop_performance() (+22 more)

### Community 4 - "Jolpica API Layer"
Cohesion: 0.12
Nodes (28): api_get(), fetch_api_qualifying(), fetch_api_race_results(), fetch_api_sprint_results(), fetch_constructor_standings(), fetch_driver_standings(), fetch_grid_penalties(), fetch_pitstop_performance() (+20 more)

### Community 5 - "Claude Query Engine"
Cohesion: 0.07
Nodes (28): ask_claude(), build_fan_context(), build_user_profile(), _detect_driver_stat_query(), get_historical_context(), get_race_replay_context(), _is_championship_scenario(), _is_driver_deep_dive() (+20 more)

### Community 6 - "Race Context Fetchers"
Cohesion: 0.08
Nodes (28): check_injection(), detect_fan_declaration(), fetch_article_content(), fetch_current_race(), get_circuit_guide(), get_next_race_context(), get_practice_context(), get_session_context() (+20 more)

### Community 7 - "Bot Formatting & Output"
Cohesion: 0.09
Nodes (21): _classify_topic(), cmd_mypredictions(), format_predictor_for_claude(), get_prediction_accuracy(), get_predictor_context(), get_sessions_for_current_round(), get_user_notif_prefs(), handle_error() (+13 more)

### Community 8 - "CLI Agent & Utilities"
Cohesion: 0.11
Nodes (26): blu(), _classify_dnf_cause(), fetch_pitstops(), fetch_sprint(), format_predictor_results(), format_telemetry_for_memory(), get_driver_code(), get_driver_name() (+18 more)

### Community 9 - "Memory & State Persistence"
Cohesion: 0.12
Nodes (25): auto_memory_enrichment_loop(), load_f1_memory(), load_news_cache(), load_sessions(), main(), Loads news cache from disk (used on startup)., Starts background news refresh thread., Loads the agent's episodic + semantic memory from file. (+17 more)

### Community 10 - "Agent CLI Commands"
Cohesion: 0.23
Nodes (25): auto_ingest(), bold(), cmd_add_race(), cmd_correct(), cmd_ingest(), cmd_memory(), cmd_predict(), cmd_season() (+17 more)

### Community 11 - "Notification System"
Cohesion: 0.09
Nodes (23): check_and_send_session_debriefs(), get_active_user_ids(), get_client(), get_user_tz_offset(), load_debrief_state(), load_digest_state(), load_notification_state(), notification_loop() (+15 more)

### Community 12 - "Data Fetch Layer"
Cohesion: 0.12
Nodes (20): fetch_driver_career_stats(), fetch_last_race(), fetch_live_session(), fetch_openf1(), fetch_practice_results(), fetch_qualifying_result(), fetch_race_result(), _fetch_session_results_openf1() (+12 more)

### Community 13 - "Race Preview & Standings"
Cohesion: 0.14
Nodes (17): build_race_preview(), cmd_preview(), fetch_all_races(), fetch_constructor_standings(), fetch_driver_standings(), fetch_lap_times_fastest(), fetch_latest_race(), fetch_qualifying() (+9 more)

### Community 14 - "Bayesian & Monte Carlo Sim"
Cohesion: 0.13
Nodes (15): apply_bayesian_priors(), monte_carlo_simulation(), print_report(), Modelo de pesos manuales — 2026 specific.     Qualifying es el predictor dominan, Entrena XGBoost con los datos disponibles usando Leave-One-Out CV., Corre N simulaciones de la carrera inyectando variaciones aleatorias en:       1, Imprime el reporte formateado en consola — dos tablas limpias., Después de cada carrera, compara la predicción del modelo con el resultado real (+7 more)

### Community 15 - "OpenF1 Data Pipeline"
Cohesion: 0.16
Nodes (16): fetch_openf1(), fetch_openf1_session_key(), fetch_race_results_openf1(), fetch_telemetry_snapshot(), of1_fetch_drivers(), of1_fetch_race_control(), of1_fetch_session_results(), of1_get_session_key() (+8 more)

### Community 16 - "News & RSS Feed"
Cohesion: 0.17
Nodes (13): cmd_news(), _fetch_article_text(), _fetch_rss(), get_news_context(), _news_scheduler(), Fetches and extracts plain text from a The Race article URL., Searches The Race site for a query.     Returns list of {title, summary, url} fr, Refreshes the RSS cache. Called on startup and every 30 min. (+5 more)

### Community 17 - "Championship & Rate Limiting"
Cohesion: 0.20
Nodes (11): build_championship_scenarios(), build_driver_profile(), check_rate_limit(), cmd_constructors(), cmd_lastrace(), cmd_standings(), fetch_standings(), Checks if a user has exceeded the rate limit.     Returns (allowed: bool, messag (+3 more)

### Community 18 - "Predictor Scheduler State"
Cohesion: 0.28
Nodes (9): _check_and_run_predictor(), fetch_next_race_info(), _load_scheduler_state(), Reads f1_2026_predicciones.csv and returns list of driver rows., Checks if qualifying for the next race is available.     If yes, and predictor h, read_predictor_csv(), _save_scheduler_state(), _scheduler_state_file() (+1 more)

### Community 19 - "System Prompt & Memory Retrieval"
Cohesion: 0.32
Nodes (8): build_system_prompt(), Token-optimized system prompt builder.     Core: ~400 tokens always. Context: in, Context Injection Architecture (no tool use), build_system_prompt(), Returns only the memory chunks most relevant to this query., retrieve_relevant_memory(), score_relevance(), Token-Optimized System Prompt (400 base tokens)

### Community 20 - "Weather Context"
Cohesion: 0.25
Nodes (8): fetch_weather(), format_weather_for_context(), get_weather_context(), _match_circuit(), Matches a query string to a circuit in CIRCUIT_COORDS using     word-boundary ma, Fetches weather forecast from Open-Meteo for given coordinates.     Returns pars, Formats weather data into a readable summary for Claude to use.     target_date:, Main weather function — detects circuit from query or uses next race,     fetche

### Community 21 - "User Stats Tracking"
Cohesion: 0.33
Nodes (6): build_user_stats_text(), cmd_mystats(), Tracks which commands a user uses., Shows personalised usage stats for the user., Builds a personalised stats summary for one user., track_command()

### Community 22 - "Safety & Language Detection"
Cohesion: 0.33
Nodes (6): _detect_language(), get_injection_response(), get_off_topic_response(), Returns 'es' for Spanish, 'en' for English based on content., Returns off-topic redirect in the same language as the message., Returns injection response in the same language as the attempt.

### Community 23 - "FIA Docs & Web Search"
Cohesion: 0.33
Nodes (6): _extract_article_text(), fetch_fia_race_documents(), google_search_f1(), Searches Google for F1 content. Returns list of {title, url, snippet}., Fetches article and extracts clean body text., Fetches official stewards decision content for a race incident/penalty.      Str

### Community 24 - "Episode Verification Pipeline"
Cohesion: 0.33
Nodes (6): Full verification pipeline for one episode:     1. Web search for race result, Searches the web for verified race results using DuckDuckGo     (no API key need, Asks Claude to compare the API-sourced episode against web search     results an, verify_and_patch_episode(), verify_episode_with_claude(), web_search_race_result()

### Community 25 - "Post-Race Briefing"
Cohesion: 0.40
Nodes (4): generate_post_race_briefing(), After ingesting a new race, Claude auto-writes a post-race debrief:     - What h, Generates and stores a post-race briefing after ingestion., run_post_race_briefing()

### Community 26 - "OpenF1 Diagnostics"
Cohesion: 0.50
Nodes (4): check_endpoint(), get(), F1 2026 — Diagnóstico de Datos OpenF1 ====================================== Cor, Chequea un endpoint y reporta qué columnas tiene y cuántos registros.

### Community 27 - "Playwright FIA Scraper"
Cohesion: 0.50
Nodes (4): _ensure_chromium_installed(), _fetch_fia_official_docs(), Ensures Playwright's Chromium browser is downloaded.      Railway's build step f, Fetches the ACTUAL FIA decision document PDF using a headless browser.      fia.

### Community 28 - "Session Time Formatting"
Cohesion: 0.50
Nodes (4): format_session_times(), get_upcoming_sessions(), Formats a session time in UTC, circuit local, and user local time., Returns sessions starting within the next `hours_ahead` hours.     Used to trigg

### Community 29 - "DNF Component Analysis"
Cohesion: 0.50
Nodes (4): build_component_retirement_risk(), classify_dnf_component(), Clasifica el status de DNF en una categoría de componente., Analiza el historial de abandonos por COMPONENTE para cada piloto y equipo.

### Community 30 - "Lap 1 Position Data"
Cohesion: 0.50
Nodes (4): of1_collect_lap1_gain(), of1_get_position(), Descarga datos de posición vuelta por vuelta desde OpenF1., Ganancia en vuelta 1 via OpenF1 /position.

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Predictor Core & OpenF1 Fetch` to `Background Loops & Auto-Ingestion`, `Circuit Type Profile`, `ML Feature Engineering`, `Race Weather Fetch`, `Jolpica API Layer`, `Bayesian Prior Persistence`, `Agent CLI Commands`, `Bayesian & Monte Carlo Sim`, `Predictor Scheduler State`, `Lap 1 Position Data`, `Teammate Delta`?**
  _High betweenness centrality (0.364) - this node is a cross-community bridge._
- **Why does `run_predictor_subprocess()` connect `Background Loops & Auto-Ingestion` to `Agent CLI Commands`, `Predictor Core & OpenF1 Fetch`, `Bot Formatting & Output`?**
  _High betweenness centrality (0.293) - this node is a cross-community bridge._
- **Why does `run_predictor()` connect `Agent CLI Commands` to `Background Loops & Auto-Ingestion`, `Predictor Core & OpenF1 Fetch`, `CLI Agent & Utilities`, `Race Preview & Standings`, `Predictor Scheduler State`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `handle_message()` (e.g. with `fetch_driver_career_stats()` and `get_circuit_guide()`) actually correct?**
  _`handle_message()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Fetches and parses F1 news from The Race, Autosport, and RaceFans.     Returns d`, `Fetches and extracts plain text from a The Race article URL.`, `Searches The Race site for a query.     Returns list of {title, summary, url} fr` to the rest of the system?**
  _208 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Background Loops & Auto-Ingestion` be split into smaller, more focused modules?**
  _Cohesion score 0.06984126984126984 - nodes in this community are weakly interconnected._
- **Should `Telegram Bot Commands` be split into smaller, more focused modules?**
  _Cohesion score 0.12222222222222222 - nodes in this community are weakly interconnected._