#!/usr/bin/env python3
"""
F1 Race Analyst Agent v4.0
===========================
One command to run:   python3 f1_agent_v3.py
One install:          pip3 install anthropic requests

What's new in v4:
  - Telemetry snapshots in episodic memory: fastest lap, sector bests,
    tyre compounds, stint lengths, pit stop times from OpenF1
  - Post-race self-briefing: after every ingestion Claude auto-writes
    a race debrief and evaluates the previous prediction
  - Auto-predictor scheduler: background thread watches for qualifying
    results Saturday night and runs predictor automatically so Sunday
    prediction is ready without any manual trigger
"""

import json, os, sys, time, textwrap, re, subprocess, csv, threading
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── dependency check ──────────────────────────────────────────
missing = []
try:
    import anthropic
except ImportError:
    missing.append("anthropic")
try:
    import requests
except ImportError:
    missing.append("requests")

if missing:
    print(f"\n  Missing: {', '.join(missing)}")
    print(f"  Fix:     pip3 install {' '.join(missing)}\n")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════
SEASON       = 2026
MEMORY_FILE  = Path(__file__).parent / "f1_memory_2026.json"
JOLPICA      = "https://api.jolpi.ca/ergast/f1"
MODEL        = "claude-sonnet-4-5"
MAX_TOKENS   = 1500
TOP_K_MEMORY = 6      # how many memory chunks to inject per query

# ── Predictor integration ─────────────────────────────────────
# Agent looks for your predictor script next to itself by default.
# Override by setting env var: export F1_PREDICTOR_PATH=/path/to/f1_2026_predictor.py
PREDICTOR_SCRIPT  = Path(os.environ.get(
    "F1_PREDICTOR_PATH",
    str(Path(__file__).parent / "f1_2026_predictor.py")
))
PREDICTOR_CSV     = Path(__file__).parent / "f1_2026_predicciones.csv"
PREDICTOR_TIMEOUT = 300   # seconds — predictor can take a while fetching APIs

# ── Scheduler config ──────────────────────────────────────────
SCHEDULER_CHECK_INTERVAL = 1800   # check every 30 min for new quali/race
BRIEFING_FILE = Path(__file__).parent / "f1_briefings.json"

# ── OpenF1 API ────────────────────────────────────────────────
OPENF1_BASE = "https://api.openf1.org/v1"

# ═════════════════════════════════════════════════════════════
#  COLORS (makes the terminal readable)
# ═════════════════════════════════════════════════════════════
R  = "\033[91m"   # red
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

def red(s):  return f"{R}{s}{RESET}"
def grn(s):  return f"{G}{s}{RESET}"
def yel(s):  return f"{Y}{s}{RESET}"
def blu(s):  return f"{B}{s}{RESET}"
def cyn(s):  return f"{C}{s}{RESET}"
def bold(s): return f"{BOLD}{s}{RESET}"
def dim(s):  return f"{DIM}{s}{RESET}"

# ═════════════════════════════════════════════════════════════
#  DEFAULT MEMORY — seeded with real 2025 data
# ═════════════════════════════════════════════════════════════
DEFAULT_MEMORY = {
    "semantic": {
        "circuit/monaco": {
            "text": "Circuit de Monaco. 3.337km street circuit, 78 laps. OVERTAKING_DIFFICULTY=0.95. Key sector S3. Norris 2025 pole record 1:09.954. Lowest overtaking rate on calendar. Qualifying position is maximally important. No SC in 2025 race (first since 2021).",
            "tags": ["street", "monaco", "high-qualifying-weight"],
            "relevance_keys": ["monaco", "street", "qualifying", "overtaking"]
        },
        "circuit/spain": {
            "text": "Circuit de Barcelona-Catalunya. OVERTAKING_DIFFICULTY=0.65. Mixed circuit. Key sector S1. DRS on main straight. Good overtaking opportunities vs Monaco but still hard.",
            "tags": ["mixed", "spain", "barcelona"],
            "relevance_keys": ["spain", "barcelona", "catalunya", "mixed"]
        },
        "circuit/monza": {
            "text": "Autodromo Nazionale di Monza. OVERTAKING_DIFFICULTY=0.30. High-speed. Key sector S1. Easiest circuit to overtake. Slipstream dominant. Low-downforce setup.",
            "tags": ["high_speed", "monza", "italy"],
            "relevance_keys": ["monza", "italy", "high speed", "slipstream"]
        },
        "circuit/silverstone": {
            "text": "Silverstone Circuit. OVERTAKING_DIFFICULTY=0.45. High-speed. Key sector S1 (Copse, Maggots, Becketts). British GP. High-speed corners suit aerodynamically efficient cars.",
            "tags": ["high_speed", "silverstone", "britain"],
            "relevance_keys": ["silverstone", "britain", "british", "copse"]
        },
        "driver/norris": {
            "text": "Lando Norris. McLaren #4. British. 2025 WDC winner (by 2pts over Verstappen at Abu Dhabi). Monaco specialist — pole record + win 2025. Won AUS R1, BHR R4, SAU R5, MON R8. Consistent tyre manager. Strong recent_form in second half of season. compound_score excellent.",
            "tags": ["mclaren", "champion", "norris"],
            "relevance_keys": ["norris", "lando", "mclaren", "champion", "wdc"]
        },
        "driver/piastri": {
            "text": "Oscar Piastri. McLaren #81. Australian. Led 2025 WDC for 15 of 24 rounds. Won CHN R2, JPN R3, MIA R6, ESP R7. Finished 3rd in 2025 WDC. Dominated first half of season. Champ lead eroded from Monaco onwards.",
            "tags": ["mclaren", "piastri"],
            "relevance_keys": ["piastri", "oscar", "mclaren"]
        },
        "driver/verstappen": {
            "text": "Max Verstappen. Red Bull #1. Dutch. 4x WDC 2021-24. RB21 not matching MCL39 pace in 2025. Finished 2nd in 2025 WDC (2pts behind Norris). Still wins on power circuits. Good qualifying. Strategic risk-taker — pitted lap 77/78 at Monaco.",
            "tags": ["redbull", "verstappen", "4x-wdc"],
            "relevance_keys": ["verstappen", "max", "red bull", "redbull"]
        },
        "driver/leclerc": {
            "text": "Charles Leclerc. Ferrari #16. Monegasque. Monaco specialist — P2 Monaco 2025 home race. Would have won 2021 Monaco from pole (car failure). High circuit_type_score for street circuits. Emotional connection to Monaco.",
            "tags": ["ferrari", "leclerc", "street-specialist"],
            "relevance_keys": ["leclerc", "charles", "ferrari", "monaco", "street"]
        },
        "driver/hamilton": {
            "text": "Lewis Hamilton. Ferrari #44. British. Moved from Mercedes after 12 years for 2025. Transition year — learning Ferrari culture and setup philosophy. P5 Monaco 2025.",
            "tags": ["ferrari", "hamilton", "transition"],
            "relevance_keys": ["hamilton", "lewis", "ferrari", "mercedes"]
        },
        "driver/russell": {
            "text": "George Russell. Mercedes #63. British. Monaco 2025: Q2 mechanical issue → P14 start. Bottled behind Williams. Corner cut on Albon earned penalty. Called mandatory 2-stop rule 'flawed'.",
            "tags": ["mercedes", "russell"],
            "relevance_keys": ["russell", "george", "mercedes"]
        },
        "team/mclaren": {
            "text": "McLaren MCL39. Mercedes PU. Back-to-back 2024+2025 constructors champions. Norris+Piastri strongest driver pairing in F1. Fastest avg pitstop. Best compound_score. Dominant in 2025 — won 12 of 24 races.",
            "tags": ["top-team", "mclaren", "constructors-champion"],
            "relevance_keys": ["mclaren", "mcl39", "orange", "constructors"]
        },
        "team/redbull": {
            "text": "Red Bull RB21. Honda RBPT PU. Verstappen leads. Good in qualifying. Race pace not matching McLaren 2025. Strategic gambles (late pitstops). Perez dropped — Lawson joined mid-season.",
            "tags": ["top-team", "redbull"],
            "relevance_keys": ["red bull", "redbull", "rb21", "verstappen", "lawson"]
        },
        "team/ferrari": {
            "text": "Ferrari SF-25. Ferrari PU. Leclerc+Hamilton. Strong on street circuits and technical tracks. Hamilton adaptation to Ferrari culture ongoing throughout 2025.",
            "tags": ["top-team", "ferrari"],
            "relevance_keys": ["ferrari", "sf-25", "scuderia", "leclerc", "hamilton"]
        },
        "team/mercedes": {
            "text": "Mercedes W16. Mercedes PU. Russell+Antonelli. Dominant 2026. ANT leads WDC 131pts. RUS 88pts after DNF Montreal lap 31 power unit failure while leading.",
            "tags": ["mercedes"],
            "relevance_keys": ["mercedes", "w16", "russell", "antonelli"]
        },
        "predictor/model": {
            "text": "f1_2026_predictor.py v7.0. Hybrid: manual weights (<6 races) -> XGBoost (>=6 races). 24 features. MC 10000 sims. Dynamic qualifying weight: 0.22 + (OVERTAKING_DIFFICULTY x 0.16). Bayesian priors persist.",
            "tags": ["predictor", "model"],
            "relevance_keys": ["predictor", "model", "xgboost", "monte carlo", "prediction", "weights"]
        },
        "predictor/qualifying_weight": {
            "text": "Dynamic qualifying weight: 0.22 + (overtaking_difficulty x 0.16). Monaco=0.95 -> 37.2%. Monza=0.30 -> 27%. Spain=0.65 -> 32.4%. When quali data available: quali_pos_next weight = 45%.",
            "tags": ["predictor", "qualifying", "weights"],
            "relevance_keys": ["qualifying weight", "overtaking difficulty", "quali", "weight", "feature importance"]
        },
        "predictor/monte_carlo": {
            "text": "MC simulation: pace noise, SC probability Poisson(1.2), pit variance, lap-1 incident 18%, DNF reliability model. 10000 sims. Outputs: win%, podium%, finish%, avg_pos, P10/P90.",
            "tags": ["predictor", "monte-carlo"],
            "relevance_keys": ["monte carlo", "simulation", "probability", "dnf", "reliability", "safety car"]
        },
        "season/2026_summary": {
            "text": "2026 F1: new aero regs. 5 races done. ANT 4 wins (Shanghai/Suzuka/Miami/Montreal). RUS won Australia. Standings: ANT 131, RUS 88, LEC 75, HAM 72, VER 43, NOR 58, PIA 48. RUS DNF Montreal power unit. HAM P2 Montreal best Ferrari result. VER first podium Montreal. NOR DNF Montreal. McLaren struggling.",
            "tags": ["season", "2026", "summary"],
            "relevance_keys": ["season", "2026", "championship", "standings", "wdc", "wcc", "title"]
        },
    },

    "episodic": [
        {
            "round": 1, "track": "Melbourne", "circuit": "Albert Park Grand Prix Circuit",
            "country": "Australia", "date": "2026-03-15", "race_name": "Australian Grand Prix",
            "winner": "RUS", "p2": "ANT", "p3": "HAM",
            "fastest_lap": "ANT", "fastest_lap_time": "",
            "dnfs": [],
            "story": "Russell wins season opener from pole. Antonelli P2 on F1 debut (0.293s off in quali). Hamilton P3 Ferrari. Mercedes 1-2 signals W16 is fastest car.",
            "qualifying": {"pole": "RUS", "pole_time": "1:18.518", "top10": "P1:RUS(1:18.518) P2:ANT(1:18.811) P3:HAM P4:LEC P5:VER", "q3_runners": ["RUS","ANT","HAM","LEC","VER"]},
            "sprint": {},
            "pitstops": {"strategy_summary": "Mostly 1-stop", "fastest_stop_driver": "", "fastest_stop_time": "", "one_stoppers": 11, "two_stoppers": 6, "three_plus_stoppers": 0},
            "telemetry": {},
            "champ_after": "RUS 25 ANT 18 HAM 15",
            "champ_delta": "RUS leads after R1",
            "prediction": None, "prediction_correct": None,
            "agent_notes": "ANT debut. RUS 0.293s ahead in quali but ANT fastest lap in race.",
            "tags": ["rus-wins", "melbourne", "season-opener", "mercedes-12"]
        },
        {
            "round": 2, "track": "Shanghai", "circuit": "Shanghai International Circuit",
            "country": "China", "date": "2026-03-22", "race_name": "Chinese Grand Prix",
            "winner": "ANT", "p2": "RUS", "p3": "VER",
            "fastest_lap": "ANT", "fastest_lap_time": "1:35.275",
            "dnfs": [],
            "story": "Sprint weekend. RUS wins sprint. ANT pole (1:32.064, 0.222s over RUS) and wins race with FL 1:35.275. Mercedes 1-2. ANT takes championship lead.",
            "qualifying": {"pole": "ANT", "pole_time": "1:32.064", "top10": "P1:ANT(1:32.064) P2:RUS(1:32.286) P3:VER P4:LEC P5:NOR", "q3_runners": ["ANT","RUS","VER","LEC","NOR"]},
            "sprint": {"sprint_winner": "RUS", "sprint_p2": "ANT", "sprint_p3": "VER", "sprint_top3": "P1:RUS P2:ANT P3:VER"},
            "pitstops": {"strategy_summary": "Mostly 1-stop", "fastest_stop_driver": "", "fastest_stop_time": "", "one_stoppers": 15, "two_stoppers": 5, "three_plus_stoppers": 0},
            "telemetry": {},
            "champ_after": "ANT 43 RUS 43 VER 30",
            "champ_delta": "ANT leads on countback",
            "prediction": None, "prediction_correct": None,
            "agent_notes": "Pattern: ANT faster in race trim, RUS beats ANT in sprints.",
            "tags": ["ant-wins", "shanghai", "sprint-weekend", "mercedes-12"]
        },
        {
            "round": 3, "track": "Suzuka", "circuit": "Suzuka Circuit",
            "country": "Japan", "date": "2026-04-06", "race_name": "Japanese Grand Prix",
            "winner": "ANT", "p2": "RUS", "p3": "LEC",
            "fastest_lap": "ANT", "fastest_lap_time": "1:32.432",
            "dnfs": [],
            "story": "ANT wins Suzuka from pole (1:28.778, 0.298s over RUS). Second consecutive win. RUS P2. Leclerc P3. ANT fastest lap 1:32.432.",
            "qualifying": {"pole": "ANT", "pole_time": "1:28.778", "top10": "P1:ANT(1:28.778) P2:RUS(1:29.076) P3:LEC P4:VER P5:HAM", "q3_runners": ["ANT","RUS","LEC","VER","HAM"]},
            "sprint": {},
            "pitstops": {"strategy_summary": "Almost all 1-stop (19 of 20)", "fastest_stop_driver": "", "fastest_stop_time": "", "one_stoppers": 19, "two_stoppers": 1, "three_plus_stoppers": 0},
            "telemetry": {},
            "champ_after": "ANT 68 RUS 61 LEC 45",
            "champ_delta": "ANT +7 over RUS",
            "prediction": None, "prediction_correct": None,
            "agent_notes": "ANT 3 fastest laps from 3 races. Consistent raw pace.",
            "tags": ["ant-wins", "suzuka", "high-speed", "mercedes-12"]
        },
        {
            "round": 4, "track": "Miami", "circuit": "Miami International Autodrome",
            "country": "USA", "date": "2026-05-04", "race_name": "Miami Grand Prix",
            "winner": "ANT", "p2": "RUS", "p3": "HAM",
            "fastest_lap": "ANT", "fastest_lap_time": "",
            "dnfs": [],
            "story": "Sprint weekend. ANT wins from pole — third consecutive race win. RUS P2. Hamilton P3 Ferrari. RUS wins sprint again. ANT leads by 14pts.",
            "qualifying": {"pole": "ANT", "pole_time": "", "top10": "P1:ANT P2:RUS P3:HAM P4:LEC P5:VER", "q3_runners": ["ANT","RUS","HAM","LEC","VER"]},
            "sprint": {"sprint_winner": "RUS", "sprint_p2": "ANT", "sprint_p3": "LEC", "sprint_top3": "P1:RUS P2:ANT P3:LEC"},
            "pitstops": {"strategy_summary": "Mixed 1 and 2-stop", "fastest_stop_driver": "", "fastest_stop_time": "", "one_stoppers": 13, "two_stoppers": 7, "three_plus_stoppers": 0},
            "telemetry": {},
            "champ_after": "ANT 93 RUS 79 HAM 60",
            "champ_delta": "ANT +14 over RUS",
            "prediction": None, "prediction_correct": None,
            "agent_notes": "RUS pattern: wins sprints, loses races. ANT fastest lap 4th race in a row.",
            "tags": ["ant-wins", "miami", "sprint-weekend"]
        },
        {
            "round": 5, "track": "Montreal", "circuit": "Circuit Gilles Villeneuve",
            "country": "Canada", "date": "2026-05-24", "race_name": "Canadian Grand Prix",
            "winner": "ANT", "p2": "HAM", "p3": "VER",
            "fastest_lap": "ANT", "fastest_lap_time": "1:14.210",
            "dnfs": ["RUS(power unit lap 31)", "NOR(retirement)", "ALO(retirement)", "ALB(collision with PIA hairpin)", "PER(retirement)", "LIN(DNS lap 0)"],
            "story": "RUS takes pole (1:12.578) and wins sprint. Mixed wet/inters start. NOR leads briefly. Gripping ANT-RUS battle. RUS DNF lap 31 POWER UNIT FAILURE while leading — NOT a points finish, retired from race. VSC deployed, field pits. ANT inherits lead, controls to flag. Hamilton P2 best Ferrari result of season, holding off Verstappen. VER P3 first podium of 2026. LEC P4. Hadjar P5 (2 penalties). Colapinto P6 best career result. Lawson P7. Piastri P11 (McLaren poor weekend). Albon DNF hit by Piastri hairpin. Lindblad DNS.",
            "qualifying": {"pole": "RUS", "pole_time": "1:12.578", "top10": "P1:RUS(1:12.578) P2:ANT P3:NOR P4:HAM P5:VER P6:LEC P7:HAD P8:COL P9:LAW P10:GAS", "q3_runners": ["RUS","ANT","NOR","HAM","VER","LEC","HAD","COL","LAW","GAS"]},
            "sprint": {"sprint_winner": "RUS", "sprint_p2": "ANT", "sprint_p3": "VER", "sprint_top3": "P1:RUS P2:ANT P3:VER"},
            "pitstops": {"strategy_summary": "10 one-stop, 7 two-stop, 2 three-plus. VSC triggered mass pit window lap 31.", "fastest_stop_driver": "", "fastest_stop_time": "", "one_stoppers": 10, "two_stoppers": 7, "three_plus_stoppers": 2},
            "telemetry": {},
            "champ_after": "ANT 131 RUS 88 LEC 75 HAM 72 NOR 58 PIA 48 VER 43",
            "champ_delta": "ANT +43 over RUS. Gap doubled from +14 to +43 due to RUS DNF.",
            "prediction": None, "prediction_correct": None,
            "agent_notes": "CRITICAL: RUS DNF power unit while leading — scored ZERO points. Russell finished ZERO points this race, did NOT finish. HAM best Ferrari result of 2026. VER first podium 2026. NOR DNF McLaren struggling all season. ANT 4 consecutive wins.",
            "tags": ["ant-wins", "montreal", "russell-dnf-power-unit", "hamilton-p2-ferrari", "verstappen-first-podium-2026", "sprint-weekend", "wet-start", "vsc"]
        },
    ],

    "predictions_log": [],
    "confidence_adjustments": {},
    "meta": {
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "version": "3.0",
        "season": SEASON
    }
}

# ═════════════════════════════════════════════════════════════
#  MEMORY LOAD / SAVE
# ═════════════════════════════════════════════════════════════
def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text())
            # merge any new semantic keys from DEFAULT not in saved file
            for k, v in DEFAULT_MEMORY["semantic"].items():
                if k not in data.get("semantic", {}):
                    data.setdefault("semantic", {})[k] = v
            return data
        except Exception:
            pass
    return DEFAULT_MEMORY.copy()

def save_memory(mem: dict):
    mem["meta"]["last_updated"] = datetime.now().isoformat()
    MEMORY_FILE.write_text(json.dumps(mem, indent=2, ensure_ascii=False))

# ═════════════════════════════════════════════════════════════
#  SMART RETRIEVAL — keyword-based relevance scoring
#  (simulates vector search without needing embeddings)
# ═════════════════════════════════════════════════════════════
def score_relevance(query: str, text: str, keys: list) -> float:
    q = query.lower()
    score = 0.0
    for k in keys:
        if k.lower() in q:
            score += 2.0
    words = re.findall(r'\w+', q)
    text_lower = text.lower()
    for w in words:
        if len(w) > 3 and w in text_lower:
            score += 0.5
    return score

def retrieve_relevant_memory(query: str, mem: dict, top_k: int = TOP_K_MEMORY) -> str:
    """Returns only the memory chunks most relevant to this query."""
    chunks = []

    # Score semantic entries
    for key, val in mem["semantic"].items():
        text = val["text"] if isinstance(val, dict) else val
        keys = val.get("relevance_keys", []) if isinstance(val, dict) else []
        s = score_relevance(query, text, keys)
        if s > 0:
            chunks.append((s, f"[SEMANTIC:{key}] {text}"))

    # Score episodic entries
    q_lower = query.lower()
    for ep in mem["episodic"]:
        s = 0.0
        track = ep["track"].lower()
        if track in q_lower:
            s += 3.0
        if ep["winner"].lower() in q_lower:
            s += 2.0
        # Also match on qualifying pole setter
        quali = ep.get("qualifying", {})
        if quali.get("pole","").lower() in q_lower:
            s += 1.5
        # Sprint winner
        sprint = ep.get("sprint", {})
        if sprint.get("sprint_winner","").lower() in q_lower:
            s += 1.5
        for tag in ep.get("tags", []):
            if tag.replace("-"," ") in q_lower:
                s += 1.0
        if str(ep["round"]) in q_lower:
            s += 1.0
        if "championship" in q_lower or "standings" in q_lower or "season" in q_lower:
            s += 0.8
        if "qualifying" in q_lower or "quali" in q_lower or "pole" in q_lower:
            s += 1.0 if quali else 0.0
        if "sprint" in q_lower:
            s += 1.0 if sprint else 0.0
        if "pit" in q_lower or "strategy" in q_lower or "stop" in q_lower:
            s += 1.0 if ep.get("pitstops") else 0.0
        if "safety car" in q_lower or " sc " in q_lower or "vsc" in q_lower:
            s += 2.0 if ep.get("sc_count", 0) > 0 else 0.0
        if "penalty" in q_lower or "penalt" in q_lower:
            s += 2.0 if ep.get("penalties") else 0.0
        if "dns" in q_lower or "did not start" in q_lower:
            s += 2.0 if ep.get("dns") else 0.0
        if "red flag" in q_lower:
            s += 2.0 if ep.get("red_flags") else 0.0
        if "classification" in q_lower or "full result" in q_lower:
            s += 1.5 if ep.get("full_classification") else 0.0
        if s > 0:
            # Build a rich memory text from all available fields
            pit   = ep.get("pitstops", {})
            notes = f" | Notes: {ep['agent_notes']}" if ep.get("agent_notes") else ""
            pred  = f" | Prediction: {ep['prediction']} (correct: {ep['prediction_correct']})" if ep.get("prediction") else ""
            quali_txt = (f" | Pole: {quali.get('pole','')} {quali.get('pole_time','')} | Q top10: {quali.get('top10','')}"
                         if quali else "")
            sprint_txt = (f" | Sprint: {sprint.get('sprint_top3','')}"
                          if sprint else "")
            pit_txt = (f" | Strategy: {pit.get('strategy_summary','')} | Fastest stop: {pit.get('fastest_stop_driver','')} {pit.get('fastest_stop_time','')}s"
                       if pit and pit.get("strategy_summary") else "")
            dnf_txt = f" | DNFs: {', '.join(ep.get('dnfs',[])[:4])}" if ep.get("dnfs") else ""
            sc_txt = (f" | SC:{ep.get('sc_count',0)}x VSC:{ep.get('vsc_count',0)}x"
                      f" periods:{' '.join(s.get('type','')+'L'+str(s.get('lap_in','?'))+'-'+str(s.get('lap_out','?')) for s in ep.get('sc_periods',[])[:2])}"
                      if ep.get("sc_count", 0) or ep.get("vsc_count", 0) else "")
            pen_txt = (f" | Penalties: {'; '.join(ep.get('penalties',[])[:2])}"
                       if ep.get("penalties") else "")
            dns_txt = (f" | DNS: {', '.join(ep.get('dns',[]))}"
                       if ep.get("dns") else "")
            rf_txt  = (f" | RedFlag laps:{ep.get('red_flags','')}"
                       if ep.get("red_flags") else "")
            fc_txt  = (f" | FullResult: {' '.join(ep.get('full_classification',[])[:5])}"
                       if ep.get("full_classification") else "")
            telem_txt = ""
            if telem.get("fastest_lap_overall"):
                fl_drv = telem.get("fastest_lap_driver","?")
                telem_txt = (
                    f" | FL:{telem['fastest_lap_overall']}s({fl_drv})"
                )
                # Add top strategies if available
                if telem.get("tyre_stints"):
                    top_stints = list(telem["tyre_stints"].items())[:3]
                    strat = " ".join(
                        f"{d}:" + "→".join(
                            s["compound"][:1] for s in stints)
                        for d, stints in top_stints
                    )
                    telem_txt += f" | Tyres:{strat}"
            chunks.append((s, (
                f"[EPISODIC:R{ep['round']}-{ep['track']}] "
                f"Date:{ep['date']} Winner:{ep['winner']} P2:{ep['p2']} P3:{ep['p3']} "
                f"FL:{ep.get('fastest_lap','')} {ep.get('fastest_lap_time','')}"
                f"{telem_txt}{quali_txt}{sprint_txt}{pit_txt}"
                f"{sc_txt}{pen_txt}{dns_txt}{rf_txt}{fc_txt}"
                f"{dnf_txt}"
                f" Story:{ep['story']}"
                f" Championship:{ep['champ_after']} ({ep['champ_delta']})"
                f"{pred}{notes}"
            )))

    # Always include confidence adjustments if they exist
    if mem.get("confidence_adjustments"):
        conf_text = "Current confidence adjustments from self-correction: " + json.dumps(mem["confidence_adjustments"])
        chunks.append((1.0, f"[WORKING:confidence] {conf_text}"))

    # Sort by relevance, take top_k
    chunks.sort(key=lambda x: x[0], reverse=True)
    top = chunks[:top_k]

    if not top:
        # Fallback: return recent episodic + season summary
        recent = mem["episodic"][-3:]
        fallback = [f"[EPISODIC:R{r['round']}-{r['track']}] Winner:{r['winner']} {r['champ_delta']}"
                    for r in recent]
        fallback.append(f"[SEMANTIC:season/2025_summary] {mem['semantic'].get('season/2025_summary', {}).get('text', '')}")
        return "\n".join(fallback)

    return "\n".join(f"  {text}" for _, text in top)

# ═════════════════════════════════════════════════════════════
#  FEATURE 1 — TELEMETRY SNAPSHOT FROM OPENF1
# ═════════════════════════════════════════════════════════════

def fetch_openf1(endpoint: str, params: dict) -> list:
    """Generic OpenF1 API call — returns list or empty."""
    try:
        r = requests.get(f"{OPENF1_BASE}/{endpoint}",
                         params=params, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def fetch_openf1_session_key(year: int, round_num: int,
                              session_type: str = "Race") -> int | None:
    """Gets the OpenF1 session_key for a given round + session type."""
    sessions = fetch_openf1("sessions", {
        "year": year,
        "session_name": session_type,
        "circuit_key": round_num   # approximate — filter by meeting below
    })
    # Fallback: fetch all sessions for year and match by meeting order
    if not sessions:
        sessions = fetch_openf1("sessions", {"year": year,
                                              "session_name": session_type})
    if not sessions:
        return None
    # Sort by date and find the nth race (round_num)
    race_sessions = sorted(
        [s for s in sessions if s.get("session_name") == session_type],
        key=lambda x: x.get("date_start", "")
    )
    if round_num <= len(race_sessions):
        return race_sessions[round_num - 1].get("session_key")
    return None

def fetch_telemetry_snapshot(round_num: int, season: int = SEASON) -> dict:
    """
    Fetches key telemetry for a race from OpenF1:
    - Fastest lap per driver (lap times)
    - Tyre compounds used + stint lengths
    - Pit stop durations
    Returns a compact snapshot dict for episodic memory storage.
    """
    snapshot = {
        "source": "openf1",
        "fetched_at": datetime.now().isoformat(),
        "fastest_laps": {},      # {driver_code: lap_time_seconds}
        "sector_bests": {},      # {driver_code: {s1, s2, s3}}
        "tyre_stints": {},       # {driver_code: [{compound, laps}]}
        "pit_stops": {},         # {driver_code: [duration_seconds]}
        "fastest_lap_overall": None,
        "fastest_lap_driver":  None,
    }

    # Get session key
    sk = fetch_openf1_session_key(season, round_num, "Race")
    if not sk:
        return snapshot

    # ── Lap times ────────────────────────────────────────────
    laps_raw = fetch_openf1("laps", {"session_key": sk})
    time.sleep(0.3)

    driver_map = {}   # number → code (filled from laps data)
    if laps_raw:
        # Build best lap per driver
        driver_best = {}
        for lap in laps_raw:
            dur = lap.get("lap_duration")
            drv = str(lap.get("driver_number", ""))
            if not dur or not drv:
                continue
            try:
                dur_f = float(dur)
                if dur_f <= 0:
                    continue
                if drv not in driver_best or dur_f < driver_best[drv]:
                    driver_best[drv] = dur_f
                # Sector bests
                for sx, skey in [("s1","duration_sector_1"),
                                  ("s2","duration_sector_2"),
                                  ("s3","duration_sector_3")]:
                    sv = lap.get(skey)
                    if sv:
                        try:
                            sv_f = float(sv)
                            if sv_f > 0:
                                if drv not in snapshot["sector_bests"]:
                                    snapshot["sector_bests"][drv] = {}
                                cur = snapshot["sector_bests"][drv].get(sx, 999)
                                if sv_f < cur:
                                    snapshot["sector_bests"][drv][sx] = round(sv_f, 3)
                        except Exception:
                            pass
            except Exception:
                continue

        # Format as round seconds
        snapshot["fastest_laps"] = {
            k: round(v, 3) for k, v in driver_best.items()
        }
        if driver_best:
            best_drv = min(driver_best, key=driver_best.get)
            snapshot["fastest_lap_driver"]  = best_drv
            snapshot["fastest_lap_overall"] = round(driver_best[best_drv], 3)

    # ── Stints / tyres ────────────────────────────────────────
    stints_raw = fetch_openf1("stints", {"session_key": sk})
    time.sleep(0.3)

    if stints_raw:
        for st in stints_raw:
            drv      = str(st.get("driver_number", ""))
            compound = st.get("compound", "UNKNOWN")
            lap_s    = st.get("lap_start", 0) or 0
            lap_e    = st.get("lap_end",   0) or 0
            laps     = max(0, int(lap_e) - int(lap_s))
            if drv:
                if drv not in snapshot["tyre_stints"]:
                    snapshot["tyre_stints"][drv] = []
                snapshot["tyre_stints"][drv].append({
                    "compound": compound,
                    "laps": laps,
                    "lap_start": lap_s,
                })

    # ── Pit stops ─────────────────────────────────────────────
    pit_raw = fetch_openf1("pit", {"session_key": sk})
    time.sleep(0.3)

    if pit_raw:
        for pit in pit_raw:
            drv = str(pit.get("driver_number", ""))
            dur = pit.get("pit_duration")
            if drv and dur:
                try:
                    dur_f = float(dur)
                    if 1.5 < dur_f < 60:
                        if drv not in snapshot["pit_stops"]:
                            snapshot["pit_stops"][drv] = []
                        snapshot["pit_stops"][drv].append(round(dur_f, 2))
                except Exception:
                    pass

    return snapshot

def format_telemetry_for_memory(snap: dict, driver_codes: dict) -> str:
    """
    Converts a telemetry snapshot into a compact readable string
    for injection into the agent's memory context.
    driver_codes maps driver_number (str) → code (e.g. "ANT")
    """
    if not snap or not snap.get("fastest_laps"):
        return ""

    lines = ["Telemetry snapshot:"]

    # Overall fastest lap
    if snap.get("fastest_lap_overall"):
        fl_drv = driver_codes.get(
            snap["fastest_lap_driver"], snap["fastest_lap_driver"])
        lines.append(
            f"  Fastest lap: {fl_drv} {snap['fastest_lap_overall']}s")

    # Top 5 fastest laps
    if snap["fastest_laps"]:
        sorted_laps = sorted(snap["fastest_laps"].items(),
                             key=lambda x: x[1])[:5]
        lap_parts = []
        for drv_num, t in sorted_laps:
            code = driver_codes.get(drv_num, drv_num)
            lap_parts.append(f"{code}:{t}s")
        lines.append(f"  Top5 laps: {' | '.join(lap_parts)}")

    # Tyre strategies (top drivers)
    if snap["tyre_stints"]:
        strat_parts = []
        for drv_num, stints in list(snap["tyre_stints"].items())[:8]:
            code = driver_codes.get(drv_num, drv_num)
            s    = "→".join(
                f"{st['compound'][:1]}({st['laps']})" for st in stints)
            strat_parts.append(f"{code}:{s}")
        lines.append(f"  Strategies: {' | '.join(strat_parts)}")

    # Fastest pit stop
    if snap["pit_stops"]:
        best_pit = None
        best_time = 999.0
        for drv_num, stops in snap["pit_stops"].items():
            m = min(stops)
            if m < best_time:
                best_time = m
                best_pit = drv_num
        if best_pit:
            code = driver_codes.get(best_pit, best_pit)
            lines.append(f"  Fastest pit: {code} {best_time}s")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
#  FEATURE 2 — POST-RACE SELF-BRIEFING
# ═════════════════════════════════════════════════════════════

def load_briefings() -> list:
    if BRIEFING_FILE.exists():
        try:
            return json.loads(BRIEFING_FILE.read_text())
        except Exception:
            pass
    return []

def save_briefing(briefing: dict):
    briefings = load_briefings()
    briefings.append(briefing)
    BRIEFING_FILE.write_text(json.dumps(briefings, indent=2))

def generate_post_race_briefing(episode: dict, mem: dict,
                                 client) -> str:
    """
    After ingesting a new race, Claude auto-writes a post-race debrief:
    - What happened and why
    - How it compared to the prediction (if one was made)
    - What it means for the championship
    - What to watch for next race
    Saves to briefings file and stores summary in episode.
    """
    # Build prediction assessment
    pred_text = ""
    if episode.get("prediction"):
        correct = episode.get("prediction_correct")
        pred_text = (
            f"The agent predicted {episode['prediction']} to win. "
            f"That was {'CORRECT' if correct else 'WRONG — actual winner was '+episode['winner']}. "
        )

    # Recent championship context
    last3 = mem["episodic"][-3:] if len(mem["episodic"]) >= 3 else mem["episodic"]
    champ_context = " | ".join(
        f"R{r['round']} {r['track']}: {r['winner']}" for r in last3)

    quali = episode.get("qualifying", {})
    pit   = episode.get("pitstops", {})
    telem = episode.get("telemetry", {})

    briefing_prompt = f"""Write a concise post-race debrief for {episode['race_name']} (R{episode['round']}).

Race facts:
- Winner: {episode['winner']} (from pole: {quali.get('pole','unknown')})
- Podium: {episode['winner']}, {episode['p2']}, {episode['p3']}
- Fastest lap: {episode.get('fastest_lap','')} {episode.get('fastest_lap_time','')}
- DNFs: {', '.join(episode.get('dnfs',[])[:4]) or 'none'}
- Strategy: {pit.get('strategy_summary','N/A')}
- Qualifying top3: {quali.get('top10','N/A')[:60]}
- Championship after: {episode.get('champ_after','N/A')}
{pred_text}
Recent form: {champ_context}

Write 3 short paragraphs:
1. What happened and the key story of the race
2. What drove the result (strategy, pace, incidents)
3. Championship implications and what to watch for next race

Keep it punchy — this is a quick debrief Erick reads between races. No headers, just paragraphs."""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": briefing_prompt}]
        )
        text = resp.content[0].text if resp.content else ""
        return text
    except Exception as e:
        return f"Auto-briefing failed: {e}"

def run_post_race_briefing(episode: dict, mem: dict, client):
    """Generates and stores a post-race briefing after ingestion."""
    print(f"\n  {cyn('Generating post-race briefing...')}", end=" ", flush=True)
    briefing_text = generate_post_race_briefing(episode, mem, client)

    # Store in episode
    episode["briefing"] = briefing_text
    episode["briefing_generated_at"] = datetime.now().isoformat()

    # Save to briefings file
    save_briefing({
        "round": episode["round"],
        "race":  episode.get("race_name", episode["track"]),
        "date":  episode["date"],
        "text":  briefing_text,
        "generated_at": datetime.now().isoformat()
    })

    print(grn("done"))
    print(f"\n  {bold('📋 Post-race briefing:')}")
    # Print wrapped nicely
    for line in briefing_text.strip().split("\n"):
        if line.strip():
            print(f"  {line}")
    print()


# ═════════════════════════════════════════════════════════════
#  FEATURE 3 — AUTO-PREDICTOR SCHEDULER
# ═════════════════════════════════════════════════════════════

_scheduler_running = False
_scheduler_thread  = None

def _scheduler_state_file() -> Path:
    return Path(__file__).parent / "f1_scheduler_state.json"

def _load_scheduler_state() -> dict:
    f = _scheduler_state_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {"last_quali_run": None, "last_race_check": None}

def _save_scheduler_state(state: dict):
    _scheduler_state_file().write_text(json.dumps(state, indent=2))

def _check_and_run_predictor(mem: dict, client,
                               print_fn=None) -> bool:
    """
    Checks if qualifying for the next race is available.
    If yes, and predictor hasn't run for this round yet, runs it.
    Returns True if predictor was triggered.
    """
    state = _load_scheduler_state()

    next_race = fetch_next_race_info()
    if not next_race:
        return False

    round_num = int(next_race.get("round", 0))
    race_date = next_race.get("date", "")

    # Don't run predictor more than once per quali session
    state_key = f"quali_run_r{round_num}"
    if state.get(state_key):
        return False

    # Check if qualifying results are available for this round
    quali = fetch_qualifying(SEASON, round_num)
    if not quali or len(quali) < 10:
        return False   # quali not done yet

    # Qualifying is available — run the predictor
    if print_fn:
        print_fn(f"\n  {cyn('🤖 Scheduler: Qualifying detected for R{round_num} — auto-running predictor...')}")
    else:
        print(f"\n  {cyn('🤖 Scheduler: Qualifying detected for R'+str(round_num)+' — auto-running predictor...')}")

    success, msg = run_predictor(show_output=False)

    if success:
        rows = read_predictor_csv(PREDICTOR_CSV)
        if rows:
            # Store the auto-prediction result in the next episode slot
            # (as a pending prediction — not yet an episode)
            mem.setdefault("pending_predictions", {})[str(round_num)] = {
                "generated_at": datetime.now().isoformat(),
                "quali_available": True,
                "top3_prediction": [
                    rows[0].get("code","?"),
                    rows[1].get("code","?") if len(rows)>1 else "?",
                    rows[2].get("code","?") if len(rows)>2 else "?",
                ],
                "winner_win_pct": rows[0].get("win_mc_pct","?"),
                "winner_predicted": rows[0].get("code","?"),
            }
            save_memory(mem)

        state[state_key] = datetime.now().isoformat()
        _save_scheduler_state(state)

        if print_fn:
            pred = rows[0] if rows else {}
            print_fn(
                f"  {grn('✅ Auto-prediction ready for R'+str(round_num))}: "
                f"{pred.get('FullName','?')} {pred.get('win_mc_pct','?')}% win probability"
            )
        return True

    return False

def _scheduler_loop(mem_ref: list, client_ref: list):
    """
    Background thread that wakes every 30 minutes and:
    1. Checks if qualifying results are available
    2. If yes, runs the predictor automatically
    3. Checks if a new race result was posted
    4. If yes, triggers ingestion + briefing
    """
    global _scheduler_running
    while _scheduler_running:
        try:
            mem    = mem_ref[0]
            client = client_ref[0]

            # Check for new race results
            all_races = fetch_all_races()
            if all_races:
                existing = {ep["round"] for ep in mem.get("episodic", [])}
                today    = datetime.now().strftime("%Y-%m-%d")
                new = [r for r in all_races
                       if int(r.get("round",0)) not in existing
                       and r.get("Results")
                       and len(r.get("Results",[])) >= 3
                       and r.get("date","9999") <= today]
                if new:
                    # New race found — ingest silently
                    from io import StringIO
                    standings    = fetch_driver_standings()
                    constructors = fetch_constructor_standings()
                    for race in sorted(new, key=lambda x: int(x.get("round",0))):
                        ep = race_to_episode(race, SEASON)
                        if not ep:
                            continue
                        if standings:
                            top3  = standings[:3]
                            parts = []
                            for s in top3:
                                d = s.get("Driver",{})
                                c = d.get("code", d.get("familyName","?")[:3].upper())
                                parts.append(f"{c} {s.get('points','?')}pts")
                            ep["champ_after"]  = " | ".join(parts)
                            ep["champ_delta"]  = f"{top3[0].get('Driver',{}).get('code','?')} leads"
                        if constructors:
                            cc = constructors[0].get("Constructor",{}).get("name","")
                            ep["constructor_leader"] = f"{cc} {constructors[0].get('points','')}pts"

                        # Fetch telemetry
                        rnd = int(race.get("round",0))
                        telem = fetch_telemetry_snapshot(rnd, SEASON)
                        if telem.get("fastest_laps"):
                            ep["telemetry"] = telem

                        mem["episodic"].append(ep)
                        mem["episodic"].sort(key=lambda x: x["round"])

                        # Self-correction
                        run_self_correction(mem, client)

                        # Auto-briefing
                        run_post_race_briefing(ep, mem, client)

                    save_memory(mem)

            # Check if predictor should run (qualifying available)
            _check_and_run_predictor(mem, client)

        except Exception:
            pass   # scheduler must never crash the main thread

        # Sleep in small increments so we can stop cleanly
        for _ in range(SCHEDULER_CHECK_INTERVAL // 10):
            if not _scheduler_running:
                break
            time.sleep(10)

def start_scheduler(mem: dict, client):
    """Starts the background scheduler thread."""
    global _scheduler_running, _scheduler_thread
    _scheduler_running = True
    mem_ref    = [mem]
    client_ref = [client]
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(mem_ref, client_ref),
        daemon=True   # dies when main process exits
    )
    _scheduler_thread.start()
    print(f"  {dim('Background scheduler started (checks every 30min for new quali/race)')}")

def stop_scheduler():
    global _scheduler_running
    _scheduler_running = False

# ═════════════════════════════════════════════════════════════
#  VERIFICATION LAYER — web cross-check after every ingest
# ═════════════════════════════════════════════════════════════

def web_search_race_result(race_name: str, season: int) -> str:
    """
    Searches the web for verified race results using DuckDuckGo
    (no API key needed). Returns raw search snippet text.
    """
    try:
        query   = f"{season} {race_name} F1 race result winner podium DNF"
        url     = "https://api.duckduckgo.com/"
        params  = {"q": query, "format": "json", "no_html": "1",
                   "skip_disambig": "1"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return ""
        data = r.json()
        # Collect text from AbstractText + RelatedTopics
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
        return " ".join(parts)[:2000]
    except Exception:
        return ""


def verify_episode_with_claude(episode: dict, web_text: str,
                                client) -> dict:
    """
    Asks Claude to compare the API-sourced episode against web search
    results and flag/fix any discrepancies.

    Returns the corrected episode + a verification report.
    """
    if not web_text or len(web_text) < 50:
        episode["verification"] = {
            "status": "skipped",
            "reason": "No web data available",
            "corrections": []
        }
        return episode

    prompt = f"""You are verifying F1 race data accuracy. Compare the API data against web sources and identify any errors.

API DATA (from Jolpica — may contain errors):
- Race: {episode.get('race_name')} R{episode.get('round')} {episode.get('date')}
- Winner: {episode.get('winner')}
- P2: {episode.get('p2')}
- P3: {episode.get('p3')}
- Fastest lap: {episode.get('fastest_lap')} {episode.get('fastest_lap_time','')}
- DNFs listed: {', '.join(episode.get('dnfs', [])) or 'none'}
- Pole: {episode.get('qualifying', {}).get('pole','')} {episode.get('qualifying', {}).get('pole_time','')}

WEB SOURCE TEXT:
{web_text}

Your job:
1. Is the winner correct? Y/N
2. Is P2 correct? Y/N
3. Is P3 correct? Y/N
4. Are the DNFs accurate? Any drivers listed as finishers who actually retired?
5. Any other major errors?

Respond ONLY as JSON, nothing else:
{{
  "winner_correct": true/false,
  "winner_verified": "CODE",
  "p2_correct": true/false,
  "p2_verified": "CODE",
  "p3_correct": true/false,
  "p3_verified": "CODE",
  "dnfs_accurate": true/false,
  "missing_dnfs": ["CODE1", "CODE2"],
  "false_dnfs": ["CODE1"],
  "pole_correct": true/false,
  "pole_verified": "CODE",
  "corrections_made": ["description of each fix"],
  "confidence": "high/medium/low",
  "notes": "brief summary"
}}"""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip() if resp.content else "{}"

        # Strip any markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        verification = json.loads(raw)

        corrections = []

        # Apply winner fix
        if not verification.get("winner_correct") and verification.get("winner_verified"):
            old = episode["winner"]
            episode["winner"] = verification["winner_verified"]
            corrections.append(f"Winner: {old} → {episode['winner']}")

        # Apply P2 fix
        if not verification.get("p2_correct") and verification.get("p2_verified"):
            old = episode["p2"]
            episode["p2"] = verification["p2_verified"]
            corrections.append(f"P2: {old} → {episode['p2']}")

        # Apply P3 fix
        if not verification.get("p3_correct") and verification.get("p3_verified"):
            old = episode["p3"]
            episode["p3"] = verification["p3_verified"]
            corrections.append(f"P3: {old} → {episode['p3']}")

        # Apply missing DNFs
        missing = verification.get("missing_dnfs", [])
        for code in missing:
            entry = f"{code}(confirmed DNF)"
            if not any(code in d for d in episode.get("dnfs", [])):
                episode.setdefault("dnfs", []).append(entry)
                corrections.append(f"Added DNF: {code}")

        # Remove false DNFs
        false_dnfs = verification.get("false_dnfs", [])
        if false_dnfs:
            before = episode.get("dnfs", [])
            episode["dnfs"] = [
                d for d in before
                if not any(f in d for f in false_dnfs)
            ]
            for f in false_dnfs:
                corrections.append(f"Removed false DNF: {f}")

        # Apply pole fix
        if not verification.get("pole_correct") and verification.get("pole_verified"):
            old = episode.get("qualifying", {}).get("pole", "")
            episode.setdefault("qualifying", {})["pole"] = verification["pole_verified"]
            corrections.append(f"Pole: {old} → {verification['pole_verified']}")

        episode["verification"] = {
            "status":       "verified",
            "confidence":   verification.get("confidence", "medium"),
            "corrections":  corrections,
            "notes":        verification.get("notes", ""),
            "verified_at":  datetime.now().isoformat(),
        }

        return episode

    except (json.JSONDecodeError, Exception) as e:
        episode["verification"] = {
            "status":      "failed",
            "reason":      str(e),
            "corrections": [],
            "raw_response": raw[:200] if "raw" in dir() else ""
        }
        return episode


def verify_and_patch_episode(episode: dict, client) -> tuple[dict, list]:
    """
    Full verification pipeline for one episode:
    1. Web search for race result
    2. Claude cross-checks API data vs web
    3. Patches any errors in the episode
    Returns (patched_episode, list_of_corrections)
    """
    race_name = episode.get("race_name", episode.get("track", ""))
    season    = episode.get("date", "")[:4] or str(SEASON)

    web_text = web_search_race_result(race_name, season)
    episode  = verify_episode_with_claude(episode, web_text, client)

    corrections = episode.get("verification", {}).get("corrections", [])
    return episode, corrections


def safe_get(url: str, timeout: int = 10, params: dict = None) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def fetch_latest_race(season: int = SEASON) -> dict | None:
    data = safe_get(f"{JOLPICA}/{season}/last/results.json")
    if not data:
        return None
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0] if races else None

def fetch_all_races(season: int = SEASON) -> list:
    """
    Fetches every completed race result for the season.
    Gets the full calendar first, skips future races by date,
    then fetches each past round individually.
    """
    schedule_data = safe_get(f"{JOLPICA}/{season}.json", params={"limit": 30})
    if not schedule_data:
        return []
    all_rounds = schedule_data.get("MRData", {}).get(
        "RaceTable", {}).get("Races", [])
    if not all_rounds:
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    completed = []
    for race_info in all_rounds:
        race_date = race_info.get("date", "9999-99-99")
        if race_date > today:
            continue   # skip future races — no API call needed
        rnd  = race_info.get("round", "0")
        data = safe_get(f"{JOLPICA}/{season}/{rnd}/results.json")
        if not data:
            time.sleep(0.3)
            continue
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races and races[0].get("Results"):
            completed.append(races[0])
        time.sleep(0.3)

    return completed

def fetch_next_race_info(season: int = SEASON) -> dict | None:
    data = safe_get(f"{JOLPICA}/{season}/next.json")
    if not data:
        return None
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0] if races else None

def fetch_driver_standings(season: int = SEASON) -> list:
    data = safe_get(f"{JOLPICA}/{season}/driverStandings.json")
    if not data:
        return []
    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return []
    return lists[0].get("DriverStandings", [])

def fetch_constructor_standings(season: int = SEASON) -> list:
    data = safe_get(f"{JOLPICA}/{season}/constructorStandings.json")
    if not data:
        return []
    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return []
    return lists[0].get("ConstructorStandings", [])

def fetch_qualifying(season: int, round_num: int) -> list:
    data = safe_get(f"{JOLPICA}/{season}/{round_num}/qualifying.json")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    return races[0].get("QualifyingResults", [])

def fetch_sprint(season: int, round_num: int) -> list:
    """Fetches sprint race results if available for this round."""
    data = safe_get(f"{JOLPICA}/{season}/{round_num}/sprint.json")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    return races[0].get("SprintResults", [])

def fetch_pitstops(season: int, round_num: int) -> list:
    """Fetches pit stop data for a round."""
    data = safe_get(f"{JOLPICA}/{season}/{round_num}/pitstops.json")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    return races[0].get("PitStops", [])

def fetch_lap_times_fastest(season: int, round_num: int) -> list:
    """Fetches fastest lap data."""
    data = safe_get(f"{JOLPICA}/{season}/{round_num}/fastest/1/results.json")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    return races[0].get("Results", [])

# ─────────────────────────────────────────────────────────────
#  OPENF1 PRIMARY SOURCE — complete race weekend status
# ─────────────────────────────────────────────────────────────

# Canonical driver number → 3-letter code for 2026 grid
OF1_DRIVER_MAP = {
    "1":  "ANT", "63": "RUS", "44": "HAM", "16": "LEC",
    "12": "ANT", "3":  "RIC", "6":  "LAW", "30": "LIN",
    "81": "PIA", "4":  "NOR", "14": "ALO", "18": "STR",
    "27": "HUL", "5":  "HAD", "43": "COL", "87": "BEA",
    "10": "GAS", "31": "OCO", "23": "ALB", "55": "SAI",
    "77": "BOT", "11": "PER", "41": "BOR", "2":  "SAR",
}

# Status strings from OpenF1 that mean "classified finisher"
OF1_CLASSIFIED = {"finished", "lapped", "+1 lap", "+2 laps", "+3 laps",
                   "+4 laps", "+5 laps", "+6 laps", "+7 laps",
                   "1 lap", "2 laps", "3 laps"}

# Race control message categories we care about
RC_RETIREMENT_KEYWORDS = ["retired", "retirement", "out of the race",
                           "stopped on track", "did not finish"]
RC_PENALTY_KEYWORDS    = ["time penalty", "drive-through", "stop and go",
                           "grid penalty", "disqualified", "dsq",
                           "black flag", "reprimand", "penalty"]
RC_SC_KEYWORDS         = ["safety car", "virtual safety car", "vsc",
                           "deployed", "end of safety car"]
RC_DNF_CAUSES = {
    "power unit": ["power unit", "engine", "electrical", "turbo",
                   "hybrid", "mgu", "ers", "battery"],
    "gearbox":    ["gearbox", "transmission", "driveshaft"],
    "hydraulics": ["hydraulics", "hydraulic"],
    "suspension": ["suspension", "upright", "steering", "wheel rim"],
    "accident":   ["accident", "collision", "contact", "crash",
                   "spin", "barrier", "wall", "damage"],
    "brakes":     ["brakes", "brake failure", "brake issue"],
    "tyres":      ["tyre", "puncture", "blowout", "delamination"],
    "overheating":["overheating", "cooling", "water pressure",
                   "oil pressure", "fire"],
    "dns":        ["did not start", "dns", "withdrew"],
}


def _classify_dnf_cause(text: str) -> str:
    """Returns the most specific DNF cause from a text string."""
    t = text.lower()
    for cause, keywords in RC_DNF_CAUSES.items():
        if any(kw in t for kw in keywords):
            return cause
    return "retirement"


def of1_get_session_key(season: int, round_num: int,
                         session_type: str = "Race") -> int | None:
    """Gets the OpenF1 session_key for a given round + session type."""
    data = fetch_openf1("sessions", {"year": season,
                                      "session_name": session_type})
    if not data:
        return None
    sessions = sorted(
        [s for s in data if s.get("session_name") == session_type],
        key=lambda x: x.get("date_start", "")
    )
    if 0 < round_num <= len(sessions):
        return sessions[round_num - 1].get("session_key")
    return None


def of1_fetch_session_results(session_key: int) -> list:
    """OpenF1 /session_result — official final classifications."""
    return fetch_openf1("session_result", {"session_key": session_key})


def of1_fetch_race_control(session_key: int) -> list:
    """OpenF1 /race_control — all RC messages for the session."""
    return fetch_openf1("race_control", {"session_key": session_key})


def of1_fetch_drivers(session_key: int) -> dict:
    """OpenF1 /drivers — driver number → name/code mapping for this session."""
    data = fetch_openf1("drivers", {"session_key": session_key})
    mapping = {}
    for d in (data or []):
        num  = str(d.get("driver_number", ""))
        code = d.get("name_acronym", "")
        if num and code:
            mapping[num] = code
    return mapping


def of1_parse_full_results(of1_results: list,
                            race_control: list,
                            driver_map: dict) -> dict:
    """
    Comprehensive parser for OpenF1 session results + race control.

    Captures:
    - Full classified order P1-P20 with lap-down status
    - All retirements with specific cause (power unit, accident, etc)
    - DNS (did not start) separately from DNF
    - Post-race penalties (time penalties, DSQ) that changed positions
    - Safety car / VSC periods with lap numbers
    - Pit lane penalties (drive-through, stop-go)
    - Fastest lap holder
    - Grid penalties detected from RC messages
    - Weather/flag events (red flag, safety car)

    Returns a rich dict used directly by race_to_episode.
    """
    # Merge session driver map with global map
    full_map = {**OF1_DRIVER_MAP, **driver_map}

    def code(num):
        return full_map.get(str(num), str(num))

    # ── 1. Parse session results ───────────────────────────
    classified  = []   # (pos, code, laps_down, points)
    dnfs        = {}   # code → cause string
    dns_list    = []   # codes that did not start

    for entry in (of1_results or []):
        drv_num  = str(entry.get("driver_number", ""))
        drv_code = code(drv_num)
        pos      = entry.get("position")
        status   = str(entry.get("status", "")).strip().lower()
        pts      = entry.get("points", 0)

        # DNS
        if any(k in status for k in ["dns", "did not start", "withdrew"]):
            dns_list.append(drv_code)
            continue

        # Classified finisher — includes lapped cars
        is_classified = (
            pos and str(pos).isdigit() and int(pos) <= 22
        ) or any(k in status for k in OF1_CLASSIFIED)

        if is_classified and pos:
            laps_down = ""
            for k in ["+1 lap", "+2 laps", "+3 laps", "+4 laps",
                      "+5 laps", "+6 laps", "1 lap", "2 laps"]:
                if k in status:
                    laps_down = k.replace("+","")
                    break
            classified.append({
                "pos":       int(pos),
                "code":      drv_code,
                "laps_down": laps_down,
                "points":    pts,
            })
        elif any(k in status for k in
                 ["dnf", "ret", "accident", "mechanical",
                  "power", "collision", "out"]):
            dnfs[drv_code] = _classify_dnf_cause(status)

    classified.sort(key=lambda x: x["pos"])

    # ── 2. Parse race control messages ────────────────────
    penalties     = []   # {driver, type, detail, lap}
    sc_periods    = []   # {type, lap_in, lap_out}
    rc_dnf_causes = {}   # code → enriched cause
    fastest_lap   = {}   # {code, lap, time}
    red_flags     = []   # lap numbers
    grid_penalties= []   # {driver, places, reason}

    current_sc    = None

    for msg in sorted(race_control or [],
                      key=lambda x: x.get("date", "")):
        text    = str(msg.get("message", "")).strip()
        text_lo = text.lower()
        drv_num = str(msg.get("driver_number", "") or "")
        lap     = msg.get("lap_number") or msg.get("lap") or "?"
        cat     = str(msg.get("category", "")).lower()
        flag    = str(msg.get("flag", "")).lower()
        drv_c   = code(drv_num) if drv_num else ""

        # Retirements — enrich cause
        if any(kw in text_lo for kw in RC_RETIREMENT_KEYWORDS):
            if drv_c:
                cause = _classify_dnf_cause(text_lo)
                rc_dnf_causes[drv_c] = cause
                if drv_c not in dnfs:
                    dnfs[drv_c] = cause

        # Penalties
        if any(kw in text_lo for kw in RC_PENALTY_KEYWORDS):
            p_type = "unknown penalty"
            for pt in ["time penalty", "drive-through", "stop and go",
                       "grid penalty", "disqualified", "reprimand",
                       "black flag"]:
                if pt in text_lo:
                    p_type = pt
                    break
            penalties.append({
                "driver": drv_c or "?",
                "type":   p_type,
                "detail": text[:120],
                "lap":    lap,
            })

        # Safety car / VSC
        if "safety car" in text_lo or "vsc" in text_lo or \
           "virtual safety car" in text_lo:
            sc_type = "VSC" if ("virtual" in text_lo or "vsc" in text_lo) \
                             else "SC"
            if "deployed" in text_lo or "in this lap" in text_lo:
                current_sc = {"type": sc_type, "lap_in": lap, "lap_out": "?"}
            elif "end" in text_lo or "withdrawn" in text_lo:
                if current_sc:
                    current_sc["lap_out"] = lap
                    sc_periods.append(current_sc)
                    current_sc = None
                else:
                    sc_periods.append({"type": sc_type,
                                       "lap_in": "?", "lap_out": lap})

        # Red flag
        if "red flag" in text_lo or flag == "red":
            red_flags.append(lap)

        # Fastest lap
        if "fastest lap" in text_lo and drv_c:
            t_match = re.search(r'\d+:\d+\.\d+', text)
            fastest_lap = {
                "code": drv_c,
                "lap":  lap,
                "time": t_match.group(0) if t_match else "",
            }

        # Grid penalties (pre-race RC messages)
        if "grid penalty" in text_lo and drv_c:
            p_match = re.search(r'(\d+)\s*(?:place|position)', text_lo)
            places  = int(p_match.group(1)) if p_match else 0
            reason  = "engine" if "engine" in text_lo else \
                      "gearbox" if "gearbox" in text_lo else "other"
            grid_penalties.append({
                "driver": drv_c, "places": places, "reason": reason
            })

    # Close any open SC period
    if current_sc:
        current_sc["lap_out"] = "end"
        sc_periods.append(current_sc)

    # ── 3. Merge DNF causes from RC into classified check ─
    # If a driver appears in both classified and dnfs, RC wins
    classified_codes = {e["code"] for e in classified}
    for drv_c, cause in rc_dnf_causes.items():
        if drv_c not in classified_codes:
            dnfs[drv_c] = cause

    # ── 4. Build clean DNF list ────────────────────────────
    dnf_list = [f"{c}({cause})" for c, cause in dnfs.items()]
    dns_list_fmt = [f"{c}(DNS)" for c in dns_list]

    # ── 5. Full classification P1-P20 ─────────────────────
    full_classification = []
    for e in classified:
        entry = f"P{e['pos']}:{e['code']}"
        if e["laps_down"]:
            entry += f"({e['laps_down']})"
        full_classification.append(entry)

    # ── 6. Penalty summary ────────────────────────────────
    penalty_summary = []
    for p in penalties:
        if p["driver"] and p["driver"] != "?":
            penalty_summary.append(
                f"{p['driver']}: {p['type']} (lap {p['lap']})")

    # ── 7. SC summary ─────────────────────────────────────
    sc_summary = []
    for sc in sc_periods:
        sc_summary.append(
            f"{sc['type']} laps {sc['lap_in']}–{sc['lap_out']}")

    if not classified or len(classified) < 3:
        return {}

    return {
        # Core results
        "winner":              classified[0]["code"],
        "p2":                  classified[1]["code"] if len(classified) > 1 else "",
        "p3":                  classified[2]["code"] if len(classified) > 2 else "",

        # Full classification
        "full_classification": full_classification,   # ["P1:ANT", "P2:HAM", ...]
        "classified_count":    len(classified),

        # DNFs / DNS
        "dnfs":                dnf_list,
        "dns":                 dns_list_fmt,
        "dnf_count":           len(dnfs),

        # Penalties
        "penalties":           penalty_summary,
        "grid_penalties":      grid_penalties,
        "post_race_penalties": [p for p in penalties
                                if p["type"] in ("time penalty","disqualified")],

        # Safety car
        "sc_periods":          sc_periods,
        "sc_count":            len([s for s in sc_periods if s["type"] == "SC"]),
        "vsc_count":           len([s for s in sc_periods if s["type"] == "VSC"]),

        # Flags
        "red_flags":           red_flags,

        # Fastest lap
        "fastest_lap_rc":      fastest_lap,

        # Data quality
        "data_source_detail":  "openf1_session_result+race_control",
    }


def fetch_race_results_openf1(round_num: int,
                               season: int = SEASON) -> dict:
    """
    PRIMARY data source. Fetches from OpenF1 with full status parsing.
    Returns comprehensive dict or empty dict if unavailable.
    """
    sk = of1_get_session_key(season, round_num, "Race")
    if not sk:
        return {}

    driver_map   = of1_fetch_drivers(sk)
    time.sleep(0.2)
    of1_results  = of1_fetch_session_results(sk)
    time.sleep(0.2)
    race_control = of1_fetch_race_control(sk)
    time.sleep(0.2)

    return of1_parse_full_results(of1_results, race_control, driver_map)



def get_driver_code(r: dict) -> str:
    drv = r.get("Driver", {})
    return drv.get("code") or drv.get("familyName", "???")[:3].upper()

def get_driver_name(r: dict) -> str:
    drv = r.get("Driver", {})
    return f"{drv.get('givenName','')} {drv.get('familyName','')}".strip()

def parse_qualifying(quali_results: list) -> dict:
    """Parses qualifying into a clean summary dict."""
    if not quali_results:
        return {}

    pole = quali_results[0]
    pole_code = get_driver_code(pole)
    pole_time = pole.get("Q3") or pole.get("Q2") or pole.get("Q1") or "N/A"

    top10 = []
    for q in quali_results[:10]:
        code = get_driver_code(q)
        best = q.get("Q3") or q.get("Q2") or q.get("Q1") or "N/A"
        top10.append(f"P{q.get('position','?')}:{code}({best})")

    return {
        "pole": pole_code,
        "pole_time": pole_time,
        "top10": " | ".join(top10),
        "q3_runners": [get_driver_code(q) for q in quali_results[:10]],
    }

def parse_sprint(sprint_results: list) -> dict:
    """Parses sprint race results."""
    if not sprint_results:
        return {}
    top3 = sprint_results[:3]
    return {
        "sprint_winner": get_driver_code(top3[0]) if top3 else "",
        "sprint_p2":     get_driver_code(top3[1]) if len(top3) > 1 else "",
        "sprint_p3":     get_driver_code(top3[2]) if len(top3) > 2 else "",
        "sprint_top3":   " | ".join(
            f"P{r.get('position','?')}:{get_driver_code(r)}"
            for r in top3
        ),
        "sprint_pts": {
            get_driver_code(r): r.get("points","0")
            for r in sprint_results[:8]
        }
    }

def parse_pitstops(pitstop_data: list) -> dict:
    """Summarises pit stop data — fastest stop, avg per driver, strategies."""
    if not pitstop_data:
        return {}

    stop_counts = {}
    stop_times  = {}
    for ps in pitstop_data:
        did  = ps.get("driverId","")
        dur  = ps.get("duration","")
        try:
            dur_f = float(dur)
        except Exception:
            continue
        stop_counts[did] = stop_counts.get(did, 0) + 1
        stop_times.setdefault(did, []).append(dur_f)

    # Fastest individual stop
    fastest_stop = None
    fastest_time = 999.0
    for did, times in stop_times.items():
        m = min(times)
        if m < fastest_time:
            fastest_time = m
            fastest_stop = did

    # Most stops (likely safety car opportunist or strategy play)
    most_stops = max(stop_counts, key=stop_counts.get) if stop_counts else ""

    # One-stoppers vs two-stoppers
    one_stop = [d for d, c in stop_counts.items() if c == 1]
    two_stop = [d for d, c in stop_counts.items() if c == 2]
    three_plus = [d for d, c in stop_counts.items() if c >= 3]

    return {
        "fastest_pitstop_driver": fastest_stop,
        "fastest_pitstop_time":   round(fastest_time, 3) if fastest_stop else None,
        "one_stoppers":  len(one_stop),
        "two_stoppers":  len(two_stop),
        "three_plus_stoppers": len(three_plus),
        "most_stops_driver": most_stops,
        "most_stops_count":  stop_counts.get(most_stops, 0),
        "strategy_summary": (
            f"{len(one_stop)} drivers 1-stop, "
            f"{len(two_stop)} drivers 2-stop"
            + (f", {len(three_plus)} drivers 3+" if three_plus else "")
        )
    }

def race_to_episode(race: dict, season: int = SEASON) -> dict:
    """
    Converts a race into a full episodic memory entry.
    DATA PRIORITY:
      1. OpenF1 session_result + race_control  → podium, DNFs (most accurate)
      2. Jolpica results                        → fallback + points data
      3. OpenF1 laps/stints/pit                → telemetry
      4. Jolpica qualifying/sprint/pitstops    → session context
    """
    results = race.get("Results", [])
    if len(results) < 3:
        return None

    round_num = int(race.get("round", 0))
    circuit   = race.get("Circuit", {}).get("circuitName", "Unknown")
    country   = race.get("Circuit", {}).get("Location", {}).get("country", "")
    track     = race.get("Circuit", {}).get("Location", {}).get("locality", country)

    # ── SOURCE 1: OpenF1 (primary for positions + DNFs) ───────
    of1_data = fetch_race_results_openf1(round_num, season)
    time.sleep(0.3)

    if of1_data and of1_data.get("winner"):
        # OpenF1 has reliable data — use it as primary
        p1_code  = of1_data["winner"]
        p2_code  = of1_data["p2"]
        p3_code  = of1_data["p3"]
        dnf_list = of1_data["dnfs"]
        data_source = "openf1"

        # Rich fields only available from OpenF1
        full_classification = of1_data.get("full_classification", [])
        dns_drivers         = of1_data.get("dns", [])
        penalties           = of1_data.get("penalties", [])
        grid_penalties      = of1_data.get("grid_penalties", [])
        post_race_penalties = of1_data.get("post_race_penalties", [])
        sc_periods          = of1_data.get("sc_periods", [])
        sc_count            = of1_data.get("sc_count", 0)
        vsc_count           = of1_data.get("vsc_count", 0)
        red_flags           = of1_data.get("red_flags", [])
        fl_rc               = of1_data.get("fastest_lap_rc", {})
        if fl_rc.get("code") and not fl_code:
            fl_code = fl_rc.get("code", "")
            fl_time = fl_rc.get("time", "")
    else:
        # ── SOURCE 2: Jolpica fallback ─────────────────────────
        p1_code = get_driver_code(results[0])
        p2_code = get_driver_code(results[1])
        p3_code = get_driver_code(results[2])

        CLASSIFIED_STATUSES = {
            "finished", "+1 lap", "+2 laps", "+3 laps",
            "+4 laps", "+5 laps", "+6 laps", "+7 laps"
        }
        dnf_list = []
        for r in results:
            status = r.get("status", "").strip()
            if status.lower() not in CLASSIFIED_STATUSES:
                code  = get_driver_code(r)
                cause = _classify_dnf_cause(status)
                dnf_list.append(f"{code}({cause})")

        full_classification = [
            f"P{r.get('position','?')}:{get_driver_code(r)}"
            for r in results if r.get("position")
        ]
        dns_drivers         = []
        penalties           = []
        grid_penalties      = []
        post_race_penalties = []
        sc_periods          = []
        sc_count            = 0
        vsc_count           = 0
        red_flags           = []
        data_source         = "jolpica_fallback"

    # ── Always from Jolpica: fastest lap + points ──────────
    fl_code, fl_time = "", ""
    for r in results:
        if r.get("FastestLap", {}).get("rank") == "1":
            fl_code = get_driver_code(r)
            fl_time = r.get("FastestLap", {}).get("Time", {}).get("time", "")
            break

    # Points scored (Jolpica only source for this)
    points_map = {
        get_driver_code(r): r.get("points", "0")
        for r in results
    }

    # Grid vs finish (position changes)
    position_changes = []
    for r in results[:12]:
        try:
            grid   = int(r.get("grid", 0))
            finish = int(r.get("position", 0))
            if grid > 0 and finish > 0 and abs(grid - finish) >= 3:
                change = grid - finish
                direction = "▲" if change > 0 else "▼"
                position_changes.append(
                    f"{get_driver_code(r)}{direction}{abs(change)}"
                )
        except Exception:
            pass

    # ── Supplementary fetches ──────────────────────────────
    time.sleep(0.3)
    quali_raw   = fetch_qualifying(season, round_num)
    time.sleep(0.3)
    sprint_raw  = fetch_sprint(season, round_num)
    time.sleep(0.3)
    pitstop_raw = fetch_pitstops(season, round_num)

    # ── Telemetry from OpenF1 ──────────────────────────────
    telem = fetch_telemetry_snapshot(round_num, season)
    time.sleep(0.3)

    quali_data   = parse_qualifying(quali_raw)
    sprint_data  = parse_sprint(sprint_raw)
    pitstop_data = parse_pitstops(pitstop_raw)

    # ── Build the story ────────────────────────────────────
    story_parts = [
        f"{get_driver_name(results[0])} wins {race.get('raceName','?')}.",
        f"P2: {get_driver_name(results[1])}.",
        f"P3: {get_driver_name(results[2])}.",
    ]
    if fl_code:
        story_parts.append(f"Fastest lap: {fl_code} ({fl_time}).")
    if dnf_list:
        story_parts.append(f"DNFs: {', '.join(dnf_list[:5])}.")
    if dns_drivers:
        story_parts.append(f"DNS: {', '.join(dns_drivers)}.")
    if penalties:
        story_parts.append(f"Penalties: {'; '.join(penalties[:3])}.")
    if sc_count:
        sc_str = " ".join(
            f"{s['type']} L{s['lap_in']}-{s['lap_out']}"
            for s in sc_periods[:3])
        story_parts.append(f"Safety car: {sc_str}.")
    if red_flags:
        story_parts.append(f"Red flag: lap(s) {', '.join(str(l) for l in red_flags)}.")
    if position_changes:
        story_parts.append(f"Big movers: {', '.join(position_changes[:4])}.")
    if quali_data:
        story_parts.append(
            f"Pole: {quali_data['pole']} ({quali_data['pole_time']}).")
    if sprint_data:
        story_parts.append(
            f"Sprint: {sprint_data['sprint_winner']} won "
            f"({sprint_data.get('sprint_top3','')}).")
    if pitstop_data:
        story_parts.append(f"Strategy: {pitstop_data['strategy_summary']}.")

    # ── Tags ───────────────────────────────────────────────
    tags = [
        f"{p1_code.lower()}-wins",
        track.lower().replace(" ", "-"),
    ]
    if sprint_data:
        tags.append("sprint-weekend")
    if dnf_list:
        tags.append("dnf")
    if pitstop_data.get("three_plus_stoppers", 0) > 2:
        tags.append("strategy-chaos")

    return {
        # ── Core race result ───────────────────────────────
        "round":        round_num,
        "track":        track,
        "circuit":      circuit,
        "country":      country,
        "date":         race.get("date", ""),
        "race_name":    race.get("raceName", ""),
        "winner":       p1_code,
        "p2":           p2_code,
        "p3":           p3_code,
        "fastest_lap":  fl_code,
        "fastest_lap_time": fl_time,
        "dnfs":         dnf_list,
        "points_top5":  {k: v for k, v in list(points_map.items())[:5]},
        "position_changes": position_changes,

        # ── Qualifying ─────────────────────────────────────
        "qualifying": {
            "pole":       quali_data.get("pole", ""),
            "pole_time":  quali_data.get("pole_time", ""),
            "top10":      quali_data.get("top10", ""),
            "q3_runners": quali_data.get("q3_runners", []),
        },

        # ── Sprint (empty dict if no sprint) ───────────────
        "sprint": sprint_data,

        # ── Pit stops / strategy ───────────────────────────
        "pitstops": {
            "strategy_summary":      pitstop_data.get("strategy_summary", ""),
            "fastest_stop_driver":   pitstop_data.get("fastest_pitstop_driver", ""),
            "fastest_stop_time":     pitstop_data.get("fastest_pitstop_time", ""),
            "one_stoppers":          pitstop_data.get("one_stoppers", 0),
            "two_stoppers":          pitstop_data.get("two_stoppers", 0),
            "three_plus_stoppers":   pitstop_data.get("three_plus_stoppers", 0),
        },

        # ── Full classification P1-P20 ─────────────────────
        "full_classification": full_classification,

        # ── Incidents ──────────────────────────────────────
        "dns":                 dns_drivers,
        "penalties":           penalties,
        "grid_penalties":      grid_penalties,
        "post_race_penalties": post_race_penalties,

        # ── Safety car & flags ─────────────────────────────
        "sc_periods":  sc_periods,
        "sc_count":    sc_count,
        "vsc_count":   vsc_count,
        "red_flags":   red_flags,

        # ── Memory / agent fields ──────────────────────────
        "story":             " ".join(story_parts),
        "champ_after":       "",
        "champ_delta":       "",
        "data_source":       data_source,
        "prediction":        None,
        "prediction_correct": None,
        "agent_notes":       "",
        "tags":              tags,

        # ── Telemetry snapshot (OpenF1) ────────────────────
        "telemetry": telem if telem.get("fastest_laps") else {},
    }

def auto_ingest(mem: dict, client=None) -> tuple[bool, str]:
    """
    On startup: checks every completed race this season and ingests
    any that are missing from episodic memory — full weekend data.
    """
    print(dim("  Checking for new race data..."), end=" ", flush=True)

    all_races = fetch_all_races()
    if not all_races:
        # Fallback to /last if bulk fetch fails
        latest = fetch_latest_race()
        all_races = [latest] if latest else []

    if not all_races:
        print(dim("API unavailable"))
        return False, "Jolpica API unavailable"

    existing_rounds = {ep["round"] for ep in mem["episodic"]}
    today = datetime.now().strftime("%Y-%m-%d")
    new_races = [
        r for r in all_races
        if int(r.get("round", 0)) not in existing_rounds
        and r.get("Results")            # must have actual results
        and len(r.get("Results", [])) >= 3   # at least 3 finishers
        and r.get("date", "9999") <= today   # must be in the past
    ]

    if not new_races:
        latest_name = all_races[-1].get("raceName", "?") if all_races else "?"
        print(grn(f"up to date (latest: {latest_name})"))
        return False, f"Already have all {len(existing_rounds)} races"

    # Fetch standings once for all new episodes
    standings    = fetch_driver_standings()
    constructors = fetch_constructor_standings()

    def standings_summary(standings: list) -> tuple[str, str]:
        if not standings:
            return "", ""
        top3 = standings[:3]
        parts = []
        for s in top3:
            drv  = s.get("Driver", {})
            code = drv.get("code", drv.get("familyName","???")[:3].upper())
            pts  = s.get("points", "?")
            parts.append(f"{code} {pts}pts")
        leader = top3[0].get("Driver", {}).get("code", "?")
        return " | ".join(parts), f"{leader} leads"

    ingested = []
    for race in sorted(new_races, key=lambda x: int(x.get("round", 0))):
        rnd  = int(race.get("round", 0))
        name = race.get("raceName", f"R{rnd}")
        print(f"\n  {dim('Ingesting '+name+'...')}", end=" ", flush=True)

        episode = race_to_episode(race, SEASON)
        if not episode:
            print(yel("skipped (no results)"))
            continue

        src_label = grn("OpenF1✓") if episode.get("data_source") == "openf1" else yel("Jolpica fallback")
        print(f"[{src_label}]", end=" ", flush=True)

        # ── Verification: cross-check API data against web ───
        if client:
            print(f"\n  {dim('Verifying against web...')}", end=" ", flush=True)
            episode, corrections = verify_and_patch_episode(episode, client)
            conf = episode.get("verification", {}).get("confidence", "?")
            if corrections:
                print(yel(f"⚠ {len(corrections)} correction(s) applied [{conf}]"))
                for c in corrections:
                    print(f"    {yel('→')} {c}")
            else:
                print(grn(f"clean [{conf}]"))
        else:
            episode["verification"] = {"status": "skipped", "corrections": []}

        champ_after, champ_delta = standings_summary(standings)
        episode["champ_after"] = champ_after
        episode["champ_delta"] = champ_delta

        # Constructor leader
        if constructors:
            cc = constructors[0].get("Constructor", {}).get("name", "")
            cc_pts = constructors[0].get("points", "")
            episode["constructor_leader"] = f"{cc} {cc_pts}pts"

        mem["episodic"].append(episode)
        ingested.append(f"R{rnd} {name}")
        print(grn("✓"))

        # Auto-generate post-race briefing if client available
        if client:
            run_post_race_briefing(episode, mem, client)

        time.sleep(0.4)   # be kind to the API

    if ingested:
        mem["episodic"].sort(key=lambda x: x["round"])
        save_memory(mem)
        summary = ", ".join(ingested)
        print(grn(f"\n  Ingested: {summary}"))
        return True, f"Ingested {len(ingested)} race(s): {summary}"

    return False, "Nothing new to ingest"

# ═════════════════════════════════════════════════════════════
#  SELF-CORRECTION — compares prediction vs reality
# ═════════════════════════════════════════════════════════════
def run_self_correction(mem: dict, client) -> None:
    """
    For any episode where a prediction exists but hasn't been evaluated,
    compares prediction vs result and updates confidence_adjustments.
    """
    adjustments = mem.setdefault("confidence_adjustments", {})

    for ep in mem["episodic"]:
        if ep.get("prediction") and ep.get("prediction_correct") is None:
            predicted = ep["prediction"].lower()
            actual    = ep["winner"].lower()
            correct   = predicted in actual or actual in predicted

            ep["prediction_correct"] = correct
            driver_key = ep["winner"]

            if correct:
                adjustments[driver_key] = round(
                    adjustments.get(driver_key, 1.0) * 1.05, 3)
                ep["agent_notes"] += f" [Self-correction: prediction correct, +5% confidence for {driver_key}]"
            else:
                # Boost the driver we missed
                adjustments[driver_key] = round(
                    adjustments.get(driver_key, 1.0) * 1.08, 3)
                # Penalize the one we over-predicted
                over = ep["prediction"].split()[0].upper()
                adjustments[over] = round(
                    adjustments.get(over, 1.0) * 0.96, 3)
                ep["agent_notes"] += (
                    f" [Self-correction: predicted {ep['prediction']}, actual {ep['winner']}. "
                    f"+8% conf {driver_key}, -4% conf {over}]"
                )

    save_memory(mem)

# ═════════════════════════════════════════════════════════════
#  FORWARD REASONING — race preview builder
# ═════════════════════════════════════════════════════════════
def build_race_preview(mem: dict, client) -> str:
    """Fetches live data and builds a preview for the next race."""
    print(dim("  Fetching next race data..."))

    next_race = fetch_next_race_info()
    standings = fetch_driver_standings()

    if not next_race:
        return "Could not fetch next race info from API."

    race_name   = next_race.get("raceName", "Next Race")
    circuit     = next_race.get("Circuit", {}).get("circuitName", "Unknown")
    country     = next_race.get("Circuit", {}).get("Location", {}).get("country", "")
    date        = next_race.get("date", "TBD")
    round_num   = next_race.get("round", "?")

    # Try to get qualifying results if available
    quali = fetch_qualifying(SEASON, round_num)
    quali_text = ""
    if quali:
        quali_text = "\nQUALIFYING RESULTS (live from API):\n"
        for q in quali[:10]:
            drv  = q.get("Driver", {})
            code = drv.get("code", drv.get("familyName","???")[:3].upper())
            pos  = q.get("position", "?")
            q3   = q.get("Q3") or q.get("Q2") or q.get("Q1") or "N/A"
            quali_text += f"  P{pos}: {code} ({q3})\n"

    standings_text = ""
    if standings:
        standings_text = "\nCURRENT CHAMPIONSHIP (live from API):\n"
        for s in standings[:8]:
            drv  = s.get("Driver", {})
            code = drv.get("code", drv.get("familyName","???")[:3].upper())
            pts  = s.get("points", "0")
            pos  = s.get("position", "?")
            standings_text += f"  P{pos}: {code} {pts}pts\n"

    # Get circuit profile from semantic memory
    circuit_key = f"circuit/{circuit.lower().replace(' ','-').split('-')[0]}"
    circuit_mem = ""
    for k, v in mem["semantic"].items():
        if "circuit" in k:
            text = v["text"] if isinstance(v, dict) else v
            if any(word.lower() in text.lower()
                   for word in circuit.split() if len(word) > 4):
                circuit_mem = text
                break

    # Recent form from last 3 episodes
    recent = mem["episodic"][-3:]
    recent_text = "\nRECENT FORM (last 3 races from episodic memory):\n"
    for r in recent:
        recent_text += f"  R{r['round']} {r['track']}: {r['winner']} won. {r['champ_delta']}\n"

    context = (
        f"NEXT RACE: {race_name} (Round {round_num})\n"
        f"Circuit: {circuit}, {country}\n"
        f"Date: {date}\n"
        f"Circuit profile: {circuit_mem or 'Not in semantic memory'}\n"
        f"{standings_text}"
        f"{quali_text}"
        f"{recent_text}"
    )

    return context

# ═════════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═════════════════════════════════════════════════════════════
SYSTEM_BASE = """You are an expert F1 race analyst with a perfect memory of the 2026 season.

RESPONSE STYLE — this is critical:
- Write like a knowledgeable friend explaining F1, not a robot citing sources
- Use plain conversational English — no markdown headers with ##, no [SEMANTIC:] tags, no bullet-point overload
- Short paragraphs that flow naturally. Bold key names/numbers sparingly
- For quick questions: 3-5 sentences is enough. Only go long if asked for deep analysis
- Never start with "Certainly!" or "Great question!" — just answer directly
- Numbers and stats are great, but weave them into sentences naturally

YOUR MEMORY (use this to answer accurately — don't cite it, just know it):
{retrieved_memory}

LIVE DATA:
{live_data}

CONFIDENCE ADJUSTMENTS FROM PAST PREDICTIONS:
{confidence}

You're talking to Erick — an engineer-PM who built the f1_2026_predictor.py himself. He knows F1 deeply. Don't over-explain basics, talk to him like a peer."""


def build_system_prompt(mem: dict, query: str, live_data: str = "") -> str:
    retrieved = retrieve_relevant_memory(query, mem)
    confidence = json.dumps(mem.get("confidence_adjustments", {}), indent=2) or "None yet"
    return SYSTEM_BASE.format(
        retrieved_memory=retrieved,
        live_data=live_data or "Not fetched this turn",
        confidence=confidence
    )

# ═════════════════════════════════════════════════════════════
#  COMMANDS
# ═════════════════════════════════════════════════════════════
def cmd_memory(mem: dict):
    print(f"\n{bold('━'*58)}")
    print(f"  {bold('🧠 MEMORY LAYERS')}")
    print(f"{'━'*58}")

    sem = mem["semantic"]
    ep  = mem["episodic"]

    print(f"\n  {cyn('SEMANTIC')} {dim(f'({len(sem)} entries — stable world knowledge)')}")
    for k in list(sem.keys())[:6]:
        v = sem[k]
        text = v["text"] if isinstance(v, dict) else v
        print(f"  {dim('['+ k +']')}")
        print(f"    {text[:85]}{'...' if len(text)>85 else ''}")
    if len(sem) > 6:
        print(f"  {dim(f'  ... and {len(sem)-6} more entries')}")

    print(f"\n  {yel('EPISODIC')} {dim(f'({len(ep)} race analyses — what happened & why)')}")
    for r in ep:
        star = red(" ◄ latest") if r["round"] == max(e["round"] for e in ep) else ""
        pred = f" | pred:{r['prediction']}({'✓' if r['prediction_correct'] else '✗'})" if r.get("prediction") else ""
        print(f"  R{r['round']:2} {r['track']:14} {bold(r['winner'])}{pred}{star}")

    conf = mem.get("confidence_adjustments", {})
    print(f"\n  {blu('WORKING')} {dim('(this session — rebuilt each query from semantic+episodic)')}")
    print(f"    Smart retrieval: top-{TOP_K_MEMORY} relevant chunks per query")
    print(f"    Self-correction confidence adjustments: {len(conf)} drivers tracked")

    print(f"\n  {G}PROCEDURAL{RESET} {dim('(always-on system prompt — baked into every query)')}")
    print(f"    Memory citation rules, self-correction logic, predictor bridge")
    print(f"    Analysis structure: context → what happened → why → implications\n")


def cmd_season(mem: dict):
    ep = mem["episodic"]
    flags = {
        "Australia":"🇦🇺","China":"🇨🇳","Japan":"🇯🇵","Bahrain":"🇧🇭",
        "Saudi Arabia":"🇸🇦","Miami":"🇺🇸","Spain":"🇪🇸","Monaco":"🇲🇨",
        "Canada":"🇨🇦","Austria":"🇦🇹","Britain":"🇬🇧","Hungary":"🇭🇺",
        "Belgium":"🇧🇪","Netherlands":"🇳🇱","Italy":"🇮🇹","Azerbaijan":"🇦🇿",
        "Singapore":"🇸🇬","Texas":"🇺🇸","Mexico":"🇲🇽","Brazil":"🇧🇷",
        "Las Vegas":"🇺🇸","Qatar":"🇶🇦","Abu Dhabi":"🇦🇪","Baku":"🇦🇿",
        "Imola":"🇮🇹","Jeddah":"🇸🇦",
    }
    print(f"\n{bold('━'*62)}")
    print(f"  {bold(f'📅 {SEASON} SEASON LOG — {len(ep)} races in episodic memory')}")
    print(f"{'━'*62}")
    for r in ep:
        flag  = flags.get(r["track"], "🏁")
        tags  = " ".join(f"{dim('['+t+']')}" for t in r.get("tags",[])[:3])
        quali = r.get("qualifying", {})
        sprint= r.get("sprint", {})
        pit   = r.get("pitstops", {})
        pred  = ""
        if r.get("prediction"):
            correct = grn("✓") if r.get("prediction_correct") else red("✗")
            pred = f" {dim('| pred:')} {r['prediction']} {correct}"

        print(f"\n  R{r['round']:2} {flag} {bold(r['track']):<16} {dim(r['date'])}")
        print(f"      🏆 {bold(r['winner'])}  P2:{r['p2']}  P3:{r['p3']}"
              + (f"  FL:{dim(r.get('fastest_lap',''))} {dim(r.get('fastest_lap_time',''))}" if r.get("fastest_lap") else ""))

        if quali:
            print(f"      {cyn('Q:')} Pole {bold(quali.get('pole','?'))} {dim(quali.get('pole_time',''))}"
                  + (f"  | {dim(quali.get('top10',''))}" if quali.get("top10") else ""))

        if sprint and sprint.get("sprint_winner"):
            print(f"      {yel('S:')} Sprint won by {bold(sprint['sprint_winner'])}"
                  f"  {dim(sprint.get('sprint_top3',''))}")

        if pit and pit.get("strategy_summary"):
            fastest = (f"  | Fastest stop: {pit.get('fastest_stop_driver','')} "
                       f"{pit.get('fastest_stop_time','')}s"
                       if pit.get("fastest_stop_time") else "")
            print(f"      {G}P:{RESET} {dim(pit['strategy_summary'])}{dim(fastest)}")

        if r.get("sc_count", 0) or r.get("vsc_count", 0):
            sc_parts = []
            if r.get("sc_count"):  sc_parts.append(f"SC×{r['sc_count']}")
            if r.get("vsc_count"): sc_parts.append(f"VSC×{r['vsc_count']}")
            print(f"      {yel('SC:')} {dim(' '.join(sc_parts))}")

        if r.get("red_flags"):
            print(f"      {red('🚩 Red flag:')} {dim('lap(s) '+str(r['red_flags']))}")

        if r.get("penalties"):
            print(f"      {yel('⚠ Penalties:')} {dim('; '.join(r['penalties'][:2]))}")

        if r.get("dns"):
            print(f"      {dim('DNS:')} {dim(', '.join(r['dns']))}")

        if r.get("full_classification"):
            top5 = " ".join(r["full_classification"][:5])
            print(f"      {dim('Top5:')} {dim(top5)}")

        if r.get("dnfs"):
            print(f"      {red('DNF:')} {dim(', '.join(r['dnfs'][:4]))}")

        print(f"      {yel(r.get('champ_delta',''))}")
        print(f"      {tags}{pred}")
    print()


def cmd_standings(mem: dict):
    print(f"\n{bold('━'*56)}")
    print(f"  {bold('🏆 CHAMPIONSHIP — fetching live from Jolpica...')}")
    print(f"{'━'*56}")

    standings = fetch_driver_standings()
    if standings:
        print(f"  {grn('Live data from Jolpica API:')}\n")
        print(f"  {'P':<3} {'Driver':<20} {'Team':<16} {'Pts':<6} Gap")
        print(f"  {'─'*3} {'─'*20} {'─'*16} {'─'*6} {'─'*8}")
        leader_pts = float(standings[0].get("points", 0))
        for s in standings[:10]:
            p    = s.get("position", "?")
            drv  = s.get("Driver", {})
            name = f"{drv.get('givenName','')} {drv.get('familyName','')}".strip()[:19]
            cons = s.get("Constructors", [{}])[0].get("name", "?")[:15]
            pts  = float(s.get("points", 0))
            gap  = "leader" if pts == leader_pts else f"-{int(leader_pts - pts)}"
            bar  = "█" * int(pts / leader_pts * 18)
            print(f"  {p:<3} {name:<20} {cons:<16} {int(pts):<6} {dim(gap):<10} {yel(bar)}")
    else:
        print(f"  {yel('API unavailable — showing last known standings:')}")
        last_ep = mem["episodic"][-1] if mem["episodic"] else {}
        print(f"  After R{last_ep.get('round','?')} {last_ep.get('track','?')}: {last_ep.get('champ_after','N/A')}")
    print()


def cmd_preview(mem: dict, client) -> str:
    """Build a race preview with live data for the agent to analyze."""
    print(f"\n  {cyn('Building race preview with live data...')}")
    return build_race_preview(mem, client)


def cmd_add_race(mem: dict):
    print(f"\n  {bold('📝 ADD RACE TO EPISODIC MEMORY')}")
    print(f"  {dim('Saves to f1_memory_v2.json permanently')}\n")
    try:
        rnd    = int(input("  Round number: "))
        track  = input("  Track/city name: ").strip()
        date   = input("  Date (YYYY-MM-DD): ").strip()
        winner = input("  Winner code (e.g. NOR): ").strip().upper()
        p2     = input("  P2 code: ").strip().upper()
        p3     = input("  P3 code: ").strip().upper()
        story  = input("  What happened (one line): ").strip()
        delta  = input("  Championship delta (e.g. 'NOR +7 over PIA'): ").strip()
        pred   = input("  What did you predict? (leave blank if none): ").strip() or None
        tags   = [t.strip() for t in input("  Tags (comma-separated): ").split(",") if t.strip()]

        ep = {
            "round": rnd, "track": track, "date": date,
            "winner": winner, "p2": p2, "p3": p3, "fastest_lap": "",
            "story": story, "champ_after": "", "champ_delta": delta,
            "prediction": pred, "prediction_correct": None,
            "agent_notes": "", "tags": tags
        }
        mem["episodic"].append(ep)
        mem["episodic"].sort(key=lambda x: x["round"])
        save_memory(mem)
        total = len(mem["episodic"])
        print(f"\n  {grn('✅ R'+str(rnd)+' '+track+' saved to episodic memory ('+str(total)+' total)')}\n")
    except (KeyboardInterrupt, ValueError, EOFError):
        print(f"\n  {yel('Cancelled.')}\n")


def cmd_ingest(mem: dict, client=None):
    """Manual trigger for auto-ingestion."""
    print(f"\n  {cyn('Auto-ingesting from Jolpica API...')}")
    updated, msg = auto_ingest(mem, client)
    if updated:
        print(f"  {grn('✅ ' + msg)}")
        print(f"  Running self-correction on new episode...")
    else:
        print(f"  {yel(msg)}\n")


def cmd_correct(mem: dict, client):
    """Manual trigger for self-correction."""
    run_self_correction(mem, client)
    conf = mem.get("confidence_adjustments", {})
    if conf:
        print(f"\n  {bold('Self-correction confidence adjustments:')}")
        for driver, mult in sorted(conf.items(), key=lambda x: abs(x[1]-1.0), reverse=True):
            direction = grn("▲ underestimated") if mult > 1.0 else red("▼ overestimated")
            print(f"  {driver:<6} ×{mult:.3f}  {direction}")
        print()
    else:
        print(f"  {yel('No predictions logged yet to evaluate.')}\n")


# ═════════════════════════════════════════════════════════════
#  PREDICTOR INTEGRATION — runs your actual Python script
# ═════════════════════════════════════════════════════════════

def read_predictor_csv(csv_path: Path) -> list[dict]:
    """Reads f1_2026_predicciones.csv and returns list of driver rows."""
    if not csv_path.exists():
        return []
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    return rows


def format_predictor_results(rows: list[dict]) -> str:
    """Formats CSV rows into a clean block for Claude to reason about."""
    if not rows:
        return "No predictor output available."

    lines = ["PREDICTOR OUTPUT (from f1_2026_predictor.py — real numbers):"]
    lines.append(f"{'─'*64}")

    # Header
    cols = ["code", "FullName", "TeamName",
            "win_mc_pct", "podium_mc_pct", "finish_mc_pct",
            "avg_mc_pos", "p10_pos", "p90_pos",
            "champ_pts", "recent_form", "avg_finish",
            "tyre_deg_slope", "compound_score", "circuit_score",
            "quali_pos_next", "fp_next_delta",
            "mechanical_risk", "dominant_failure"]

    lines.append(f"  {'#':<3} {'Driver':<18} {'Team':<14} "
                 f"{'Win%':>6} {'Pod%':>6} {'Fin%':>6} "
                 f"{'AvgPos':>7} {'P10':>4} {'P90':>4} "
                 f"{'QualiNext':>10} {'MechRisk':>9}")
    lines.append(f"  {'─'*3} {'─'*18} {'─'*14} "
                 f"{'─'*6} {'─'*6} {'─'*6} "
                 f"{'─'*7} {'─'*4} {'─'*4} "
                 f"{'─'*10} {'─'*9}")

    for i, r in enumerate(rows[:15], 1):
        def g(k, default="—"):
            v = r.get(k, default)
            return v if v not in (None, "", "nan", "NaN") else default

        def gf(k, dec=1):
            try:
                return f"{float(r[k]):.{dec}f}"
            except Exception:
                return "—"

        quali = f"P{int(float(r['quali_pos_next']))}" if r.get("quali_pos_next") not in (None,"","nan","NaN") else "—"
        mech  = f"{float(r.get('mechanical_risk',0))*100:.1f}%" if r.get("mechanical_risk") not in (None,"","nan","NaN") else "—"

        lines.append(
            f"  {i:<3} {g('FullName'):<18} {g('TeamName'):<14} "
            f"{gf('win_mc_pct'):>6} {gf('podium_mc_pct'):>6} {gf('finish_mc_pct'):>6} "
            f"{gf('avg_mc_pos'):>7} {gf('p10_pos',0):>4} {gf('p90_pos',0):>4} "
            f"{quali:>10} {mech:>9}"
        )

    # Add key feature detail for top 5
    lines.append(f"\n  KEY FEATURES — Top 5 drivers:")
    for r in rows[:5]:
        def g(k): return r.get(k, "—")
        def gf(k, dec=2):
            try: return f"{float(r[k]):.{dec}f}"
            except: return "—"

        lines.append(
            f"  {g('code')}: "
            f"recent_form={gf('recent_form',0)}  "
            f"tyre_deg={gf('tyre_deg_slope',4)}  "
            f"compound_score={gf('compound_score')}  "
            f"circuit_score={gf('circuit_score')}  "
            f"fp_next={gf('fp_next_delta')}%  "
            f"failure_risk={g('dominant_failure')}"
        )

    # Model info if available
    lines.append(f"\n  {'─'*64}")
    try:
        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        mtime = "unknown"
    lines.append(f"  CSV generated: {mtime}")

    return "\n".join(lines)


def run_predictor(show_output: bool = True) -> tuple[bool, str]:
    """
    Actually runs f1_2026_predictor.py as a subprocess.
    Returns (success: bool, message: str).
    The predictor writes f1_2026_predicciones.csv which we then read.
    """
    if not PREDICTOR_SCRIPT.exists():
        return False, (
            f"Predictor not found at: {PREDICTOR_SCRIPT}\n"
            f"  Fix: put f1_2026_predictor.py next to this file, or:\n"
            f"  export F1_PREDICTOR_PATH=/full/path/to/f1_2026_predictor.py"
        )

    print(f"\n  {cyn('Running f1_2026_predictor.py...')}")
    print(f"  {dim('(this fetches live F1 API data + runs 10,000 Monte Carlo sims)')}")
    print(f"  {dim('Takes 1-5 minutes. Output streams below:')}")
    print(f"  {'─'*56}")

    try:
        # Run with the same Python executable, inherit working directory
        python_exe = sys.executable
        proc = subprocess.Popen(
            [python_exe, str(PREDICTOR_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PREDICTOR_SCRIPT.parent),
            bufsize=1
        )

        output_lines = []
        if show_output and proc.stdout:
            for line in proc.stdout:
                line_stripped = line.rstrip()
                if line_stripped:
                    print(f"  {dim(line_stripped)}")
                    output_lines.append(line_stripped)
        else:
            out, _ = proc.communicate(timeout=PREDICTOR_TIMEOUT)
            output_lines = out.splitlines() if out else []

        proc.wait(timeout=PREDICTOR_TIMEOUT)

        print(f"  {'─'*56}")

        if proc.returncode != 0:
            # Check if CSV was still written (partial success)
            if PREDICTOR_CSV.exists():
                print(f"  {yel('⚠ Predictor exited with errors but CSV was written — using partial results')}")
                return True, "Partial run — CSV available"
            return False, f"Predictor failed (exit code {proc.returncode})"

        if not PREDICTOR_CSV.exists():
            return False, f"Predictor ran but CSV not found at: {PREDICTOR_CSV}"

        print(f"  {grn('✅ Predictor finished — CSV loaded')}")
        return True, "Success"

    except subprocess.TimeoutExpired:
        proc.kill()
        # Still try to use CSV if it was written before timeout
        if PREDICTOR_CSV.exists():
            print(f"  {yel('⚠ Predictor timed out but CSV exists — using it')}")
            return True, "Timeout — using existing CSV"
        return False, f"Predictor timed out after {PREDICTOR_TIMEOUT}s"

    except FileNotFoundError:
        return False, (
            f"Python not found at: {python_exe}\n"
            f"  The predictor needs the same Python you're using to run this agent."
        )

    except Exception as e:
        return False, f"Unexpected error running predictor: {e}"


def cmd_predict(mem: dict, client, hist: list):
    """
    The real /predict:
      1. Runs f1_2026_predictor.py (your actual code)
      2. Reads the CSV output (real win%, Monte Carlo, features)
      3. Fetches live standings + next race info from Jolpica
      4. Builds a rich context block
      5. Passes everything to Claude to interpret
    Claude's job: analyze WHAT YOUR MODEL SAID and WHY, add narrative context
    from episodic+semantic memory, flag what could upset the model's numbers.
    """
    print(f"\n  {bold(red('━'*56))}")
    print(f"  {bold('🎯 PREDICT MODE — Running your predictor')}")
    print(f"  {bold(red('━'*56))}")

    # Step 1 — run the predictor
    success, msg = run_predictor(show_output=True)

    predictor_block = ""
    if success:
        rows = read_predictor_csv(PREDICTOR_CSV)
        if rows:
            predictor_block = format_predictor_results(rows)
            print(f"\n  {grn('Predictor output loaded: '+str(len(rows))+' drivers')}")
        else:
            predictor_block = "Predictor ran but CSV was empty."
            print(f"  {yel('CSV was empty.')}")
    else:
        predictor_block = (
            f"Predictor could not run: {msg}\n\n"
            f"FALLBACK: Analyze using memory + live API data only.\n"
            f"Tell the user to check that f1_2026_predictor.py is in the same\n"
            f"folder as this agent, and all its dependencies are installed."
        )
        print(f"\n  {yel('Predictor unavailable: '+msg)}")
        print(f"  {dim('Falling back to memory + live API data...')}")

    # Step 2 — fetch live Jolpica data
    print(f"\n  {dim('Fetching live race data from Jolpica...')}", end=" ", flush=True)
    live_block = build_race_preview(mem, client)
    print(grn("done"))

    # Step 3 — build the full context
    full_context = f"{predictor_block}\n\n{live_block}"

    # Step 4 — craft the analysis prompt
    query = (
        "You are now analyzing the ACTUAL OUTPUT of f1_2026_predictor.py v7.0. "
        "This is real data, not a guess.\n\n"
        "Do the following:\n"
        "1. PREDICTOR VERDICT: State the model's top-5 clearly with exact win% numbers from the CSV\n"
        "2. WHY THE MODEL SAYS THIS: Explain what features are driving the top pick "
        "(quali_pos_next, recent_form, circuit_score, compound_score, etc.) "
        "— cite the actual values from the CSV\n"
        "3. MEMORY VALIDATION: Cross-check with episodic memory — "
        "does this prediction match historical patterns for this driver/circuit? "
        "What do past episodes tell us?\n"
        "4. WHAT COULD UPSET IT: What factors could override the model "
        "(weather, safety car, mechanical risk, strategy chaos like Monaco 2-stop)?\n"
        "5. CHAMPIONSHIP STAKES: What does this race mean for the WDC battle?\n"
        "6. YOUR CONFIDENCE: On a 0-100 scale, how much do you trust this prediction and why?\n\n"
        "Be specific. Use the actual numbers. This is a serious analysis."
    )

    hist.append({
        "role": "user",
        "content": f"PREDICTOR DATA + LIVE RACE DATA:\n{full_context}\n\nANALYSIS REQUEST:\n{query}"
    })

    return query, full_context


def print_help():
    print(f"""
  {bold('┌─────────────────────────────────────────────────────────┐')}
  {bold('│  F1 ANALYST AGENT v3.0  —  Commands                     │')}
  {bold('└─────────────────────────────────────────────────────────┘')}

  {cyn('/predict')}     ★ Runs f1_2026_predictor.py → reads CSV → Claude
              {dim('analyzes YOUR model output with real win%, MC sims,')}
              {dim('feature values, then adds narrative from memory')}

  {cyn('/briefing')}    Show the last auto-generated post-race debrief
  {cyn('/verify')}      Re-verify latest race data against web sources
  {cyn('/memory')}      Show all 4 memory layers + confidence adjustments
  {cyn('/season')}      Full season log: race + qualifying + sprint + pitstops
  {cyn('/standings')}   Live championship standings (from Jolpica API)
  {cyn('/ingest')}      Pull any new races + telemetry + verify + auto-briefing
  {cyn('/reingest')}    Wipe + re-ingest full season with complete weekend data
  {cyn('/correct')}     Run self-correction (prediction vs reality check)
  {cyn('/add')}         Manually add a race to episodic memory
  {cyn('/clear')}       Clear chat history (memory stays)
  {cyn('/help')}        This menu
  {cyn('/quit')}        Exit

  {dim('Every new race is automatically verified against web sources.')}

  {dim('Or just ask anything — smart retrieval finds relevant memory.')}
  {dim('Examples:')}
  {dim('  "Why did Norris win Monaco?"')}
  {dim('  "What does tyre_deg_slope tell us about Verstappen?"')}
  {dim('  "Compare Piastri vs Norris across the season"')}
  {dim('  "Explain the compound_score feature in the predictor"')}
""")

# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    # ── banner ────────────────────────────────────────────────
    print(f"""
{red('╔══════════════════════════════════════════════════════════════╗')}
{red('║')}  {bold('🏎  F1 Race Analyst Agent v3.0')}                               {red('║')}
{red('║')}  {dim('Runs your predictor · Reads the CSV · Claude interprets it')} {red('║')}
{red('╚══════════════════════════════════════════════════════════════╝')}""")

    # ── API key ───────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print(f"\n  {yel('No ANTHROPIC_API_KEY found.')}")
        print(f"  Set it permanently:  {cyn('echo export ANTHROPIC_API_KEY=sk-ant-... >> ~/.zshrc && source ~/.zshrc')}")
        print(f"  Or paste it now:")
        try:
            api_key = input(f"  {bold('API key › ')}").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if not api_key:
            print(f"  {red('No key provided. Exiting.')}"); sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── load memory ───────────────────────────────────────────
    mem  = load_memory()
    hist = []

    first_run = not MEMORY_FILE.exists()
    save_memory(mem)

    # ── auto-ingest on startup ────────────────────────────────
    print()
    updated, msg = auto_ingest(mem, client)
    if updated:
        run_self_correction(mem, client)

    # ── start background scheduler ────────────────────────────
    start_scheduler(mem, client)

    # ── status ────────────────────────────────────────────────
    last_ep = mem["episodic"][-1] if mem["episodic"] else {}
    predictor_status = grn("found ✓") if PREDICTOR_SCRIPT.exists() else red("NOT FOUND — /predict will fail")

    # Check for pending auto-prediction
    pending = mem.get("pending_predictions", {})
    next_race = fetch_next_race_info()
    next_round = str(next_race.get("round","")) if next_race else ""
    auto_pred = pending.get(next_round)

    print(f"\n  {grn('✅ Ready')}")
    print(f"  {dim('Semantic:')}   {len(mem['semantic'])} entries")
    print(f"  {dim('Episodic:')}   {len(mem['episodic'])} race analyses  {dim('(latest: R'+str(last_ep.get('round','?'))+' '+last_ep.get('track','?')+')')}")
    print(f"  {dim('Predictor:')}  {predictor_status}")
    print(f"  {dim('Path:')}       {dim(str(PREDICTOR_SCRIPT))}")
    print(f"  {dim('Self-corrections:')} {len(mem.get('confidence_adjustments',{}))} drivers tracked")
    print(f"  {dim('Scheduler:')}  {grn('running')} (checks every 30min)")
    if auto_pred:
        print(f"  {grn('🤖 Auto-prediction ready:')} {auto_pred.get('winner_predicted','?')} — {auto_pred.get('winner_win_pct','?')}% win. Type /predict to see full analysis.")
    print(f"\n  {dim('Type /help for commands or just ask anything.')}\n")

    # ── main loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(f"  {bold('You ›')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {grn('👋 Memory saved. See you next race.')}\n")
            stop_scheduler()
            save_memory(mem)
            break

        if not user_input:
            continue

        # ── commands ──────────────────────────────────────────
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print(f"\n  {grn('👋 Memory saved. See you next race.')}\n")
            stop_scheduler()
            save_memory(mem); break

        elif cmd == "/help":
            print_help(); continue
        elif cmd == "/memory":
            cmd_memory(mem); continue
        elif cmd == "/season":
            cmd_season(mem); continue
        elif cmd == "/standings":
            cmd_standings(mem); continue
        elif cmd == "/ingest":
            cmd_ingest(mem, client); continue
        elif cmd == "/reingest":
            print(f"\n  {yel('Clearing episodic memory and re-ingesting full season...')}")
            confirm = input(f"  {bold('Type YES to confirm: ')}").strip()
            if confirm == "YES":
                mem["episodic"] = []
                save_memory(mem)
                updated, msg = auto_ingest(mem, client)
                if updated:
                    run_self_correction(mem, client)
                n = len(mem["episodic"])
                print(f"  {grn('Done. '+str(n)+' races ingested.')}\n")
            else:
                print(f"  {yel('Cancelled.')}\n")
            continue
        elif cmd == "/briefing":
            # Show last N briefings
            briefings = load_briefings()
            if not briefings:
                print(f"\n  {yel('No briefings yet. Run /ingest after a race to generate one.')}\n")
            else:
                last = briefings[-1]
                print(f"\n  {bold('📋 Last briefing: '+last.get('race','?')+' ('+last.get('date','')+')')}")
                print(f"  {dim('Generated: '+last.get('generated_at','?')[:16])}\n")
                for line in last.get("text","").strip().split("\n"):
                    if line.strip():
                        print(f"  {line}")
                print()
            continue
        elif cmd == "/verify":
            # Re-verify the most recent episode against web sources
            if not mem["episodic"]:
                print(f"\n  {yel('No episodes to verify.')}\n")
            else:
                last = mem["episodic"][-1]
                rname = last.get("race_name", last.get("track","?"))
                print(f"\n  {cyn('Re-verifying '+rname+' against web...')}")
                patched, corrections = verify_and_patch_episode(last, client)
                mem["episodic"][-1] = patched
                if corrections:
                    print(f"  {yel('Corrections applied:')}")
                    for c in corrections:
                        print(f"    {yel('→')} {c}")
                else:
                    conf = patched.get("verification",{}).get("confidence","?")
                    print(f"  {grn('Data verified clean ['+conf+']')}")
                save_memory(mem)
                print()
            continue
        elif cmd == "/add":
            cmd_add_race(mem); continue
        elif cmd == "/clear":
            hist = []
            print(f"  {grn('✓ Chat history cleared (memory kept).')}\n"); continue
        elif cmd == "/predict":
            query, live = cmd_predict(mem, client, hist)
            user_input = query
            # hist already has the message appended inside cmd_predict
            # so we skip the normal hist.append below
            system = build_system_prompt(mem, user_input, live)
            trimmed_hist = hist[-10:]
            print(f"\n  {red(bold('Agent ›'))} ", end="", flush=True)
            full_text = ""
            try:
                with client.messages.stream(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=trimmed_hist
                ) as stream:
                    for chunk in stream.text_stream:
                        print(chunk, end="", flush=True)
                        full_text += chunk
                print("\n")
                hist.append({"role": "assistant", "content": full_text})
                # Store what the model predicted for self-correction later
                last_ep = mem["episodic"][-1] if mem["episodic"] else None
                if last_ep:
                    for driver in ["NOR","PIA","VER","LEC","HAM","RUS","SAI","ANT","LAW","ALO"]:
                        if driver in full_text.upper():
                            if not last_ep.get("prediction"):
                                last_ep["prediction"] = driver
                            break
                save_memory(mem)
            except anthropic.AuthenticationError:
                print(f"\n  {red('⚠ Invalid API key.')}\n")
            except anthropic.APIConnectionError:
                print(f"\n  {red('⚠ No internet connection.')}\n")
            except anthropic.RateLimitError:
                print(f"\n  {yel('⚠ Rate limit — wait a moment and try again.')}\n")
            except Exception as e:
                print(f"\n  {red(f'⚠ Error: {e}')}\n")
            continue

        # ── Normal AI call (all non-/predict queries) ─────────
        live = ""
        system = build_system_prompt(mem, user_input, live)
        if user_input not in [m["content"] for m in hist if m["role"]=="user"]:
            hist.append({"role": "user", "content": user_input})

        # Keep history to last 10 turns to avoid token bloat
        trimmed_hist = hist[-10:]

        print(f"\n  {red(bold('Agent ›'))} ", end="", flush=True)
        full_text = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=trimmed_hist
            ) as stream:
                for chunk in stream.text_stream:
                    print(chunk, end="", flush=True)
                    full_text += chunk
            print("\n")

            hist.append({"role": "assistant", "content": full_text})

            # Log prediction to episodic if agent made one
            if any(word in full_text.lower() for word in
                   ["predict", "winner will be", "i expect", "likely winner", "my pick"]):
                last_ep = mem["episodic"][-1] if mem["episodic"] else None
                if last_ep and not last_ep.get("prediction"):
                    # Try to extract who was predicted
                    for driver in ["NOR","PIA","VER","LEC","HAM","RUS","SAI","ANT"]:
                        if driver in full_text.upper():
                            last_ep["prediction"] = driver
                            break

            save_memory(mem)

        except anthropic.AuthenticationError:
            print(f"\n  {red('⚠ Invalid API key.')}")
            print(f"  {dim('Run: export ANTHROPIC_API_KEY=sk-ant-...')}\n")
        except anthropic.APIConnectionError:
            print(f"\n  {red('⚠ No internet connection.')}\n")
        except anthropic.RateLimitError:
            print(f"\n  {yel('⚠ Rate limit hit — wait a moment and try again.')}\n")
        except Exception as e:
            print(f"\n  {red(f'⚠ Error: {e}')}\n")


if __name__ == "__main__":
    main()
