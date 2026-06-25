"""Context building for BoxBoxAI: Claude client, system prompt assembly, and context gathering."""
import os
import re
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import requests
import anthropic

from models import ContextBlock, _unpack_ctx, CIRCUIT_MAP_VERSION
from news import get_news_context, get_news_cache_time
from security import alert_owner
from data_layer import (
    fetch_standings, fetch_current_race, fetch_next_race,
    get_session_context, get_live_session_context,
    get_prediction_accuracy, fetch_driver_career_stats,
    get_practice_context,
)

log = logging.getLogger(__name__)

# Callable references wired by bot at startup
_live_search_fn:           list = [None]  # context_builder._live_search_fn[0] = live_search_f1
_fetch_fia_docs_fn:        list = [None]  # context_builder._fetch_fia_docs_fn[0] = fetch_fia_race_documents
_get_circuit_zone_data_fn: list = [None]  # context_builder._get_circuit_zone_data_fn[0] = _get_circuit_zone_data

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL         = "claude-sonnet-4-5"
MAX_TOKENS    = 1000
HISTORICAL_DATA = """
HISTORICAL F1 RECORDS AND COMPARISONS (for context when asked):

Rookie seasons — wins in first 5 races:
- Verstappen 2015: 0 wins (Toro Rosso, first full season)
- Hamilton 2007: 0 wins in first 5, won USA R7 (finished season with 4 wins)
- Leclerc 2018: 0 wins (Sauber)
- Piastri 2023: 0 wins (2 sprint wins)
- Senna 1984: 0 wins (Toleman)
- Schumacher 1991: 0 wins

Championship comparisons:
- Verstappen: 4 WDC (2021,2022,2023,2024)
- Hamilton: 7 WDC (2008,2014,2015,2017,2018,2019,2020)
- Schumacher M: 7 WDC (1994,1995,2000,2001,2002,2003,2004)
- Senna: 3 WDC (1988,1990,1991)
- Prost: 4 WDC (1985,1986,1989,1993)
- Alonso: 2 WDC (2005,2006)

Monaco winners (recent):
- 2025: Norris (pole 1:09.954)
- 2024: Leclerc
- 2023: Verstappen
- 2022: Perez
- 2021: Verstappen

Points records (modern era):
- Most points in a season: Verstappen 2023 (575pts, 19 wins from 22 races)
- Fastest ever lap records, domination stats available on request

Mercedes dominance 2014-2021: 8 consecutive WCC titles
Red Bull dominance 2022-2024: 3 consecutive WCC + 4 WDC Verstappen
"""

def get_historical_context(query: str) -> str:
    """Returns historical context if query seems comparison-based."""
    comparison_keywords = [
        "compare", "better than", "vs", "versus", "history", "historical",
        "record", "ever", "all time", "greatest", "goat", "best",
        "rookie", "debut", "first season", "like verstappen", "like hamilton",
        "comparar", "mejor que", "historia", "récord", "el mejor", "de todos",
        "temporada debut", "novato", "como verstappen", "como hamilton",
        "similar", "parecido", "igual que", "peor que", "superar"
    ]
    t = query.lower()
    if any(kw in t for kw in comparison_keywords):
        return HISTORICAL_DATA
    return ""
FIA_DRIVER_NAMES = {
    "ant": "Antonelli", "kimi": "Antonelli", "antonelli": "Antonelli",
    "rus": "Russell",   "george": "Russell",  "russell": "Russell",
    "ham": "Hamilton",  "lewis": "Hamilton",  "hamilton": "Hamilton",
    "lec": "Leclerc",   "charles": "Leclerc", "leclerc": "Leclerc",
    "ver": "Verstappen","max": "Verstappen",  "verstappen": "Verstappen",
    "nor": "Norris",    "lando": "Norris",    "norris": "Norris",
    "pia": "Piastri",   "oscar": "Piastri",   "piastri": "Piastri",
    "alo": "Alonso",    "fernando": "Alonso", "alonso": "Alonso",
    "per": "Perez",     "checo": "Perez",     "perez": "Perez",
    "str": "Stroll",    "lance": "Stroll",    "stroll": "Stroll",
    "gas": "Gasly",     "pierre": "Gasly",    "gasly": "Gasly",
    "col": "Colapinto", "franco": "Colapinto","colapinto": "Colapinto",
    "alb": "Albon",     "alex": "Albon",      "albon": "Albon",
    "sai": "Sainz",     "carlos": "Sainz",    "sainz": "Sainz",
    "law": "Lawson",    "liam": "Lawson",     "lawson": "Lawson",
    "had": "Hadjar",    "isack": "Hadjar",    "hadjar": "Hadjar",
    "bea": "Bearman",   "oliver": "Bearman",  "bearman": "Bearman",
    "oco": "Ocon",      "esteban": "Ocon",    "ocon": "Ocon",
    "hul": "Hulkenberg","nico": "Hulkenberg", "hulkenberg": "Hulkenberg",
    "bor": "Bortoleto", "gabriel": "Bortoleto","bortoleto": "Bortoleto",
    "bot": "Bottas",    "valtteri": "Bottas", "bottas": "Bottas",
}

# FIA stewards documents identify drivers by CAR NUMBER (e.g. "Car 16"),
# never by surname, so matching a named driver in a doc title requires
# their 2026 car number. Keyed by 3-letter driver code (cf. FIA_DRIVER_NAMES).
FIA_DRIVER_CAR_NUMBERS = {
    "HAM": 44, "RUS": 63, "NOR": 1,  "VER": 3,  "PIA": 81, "HAD": 6,
    "GAS": 10, "COL": 43, "LAW": 30, "LIN": 41, "BOR": 5,  "SAI": 55,
    "OCO": 31, "PER": 11, "LEC": 16, "ANT": 12, "BEA": 87, "ALB": 23,
    "ALO": 14, "HUL": 27, "BOT": 77, "STR": 18,
}
def _needs_fia_docs(query: str) -> bool:
    """
    Detects if a query needs FIA stewards documents.
    Triggers on penalty reasons, incident causes, crash details,
    technical violations, disqualifications.
    """
    t = query.lower()

    # Skip internal predictor prompts
    if any(skip in t for skip in [
        "f1_2026_predictor", "monte carlo", "win probability",
        "circuit_score", "recent_form=", "in 2-3 sentences",
        "confirm this pick",
    ]):
        return False

    triggers = [
        "why did", "why was", "why were", "what caused",
        "what happened to",
        "crash", "collision", "incident", "contact",
        "penalt", "penalis", "sanction",
        "disqualified", "dsq", "black flag",
        "drive through", "drive-through", "time penalty",
        "grid penalty", "grid drop",
        "track limits", "speeding", "unsafe release",
        "false start", "jump start", "out of position",
        "stewards", "fia decision", "investigation",
        "retired because", "dnf because", "why retire",
        "tarmac", "track surface", "red flag because",
        "reprimand", "article b", "regulation",
        # Tyre/compound queries
        "tyre", "tyres", "tire", "tires", "compound", "compounds", "allocation",
        "soft", "medium", "hard", "pirelli", "c1", "c2", "c3", "c4", "c5",
        "which tyres", "what tyres", "tyre choice", "tyre selection",
        # Spanish
        "por qué", "qué pasó", "por que",
        "penaliz", "infracción", "infraccion",
        "chocaron", "se retiró", "descalificado",
        "decisión de los comisarios", "los comisarios",
        "neumático", "neumáticos", "compuesto", "asignación",
        "blandos", "medios", "duros",
    ]
    return any(tr in t for tr in triggers)


def _is_upgrade_query(text: str) -> bool:
    """Returns True for queries about team upgrades, technical development, or performance changes."""
    t = text.lower()
    upgrade_vocab = [
        "upgrade", "updates", "new parts", "development", "floor",
        "sidepod", "front wing", "rear wing", "diffuser", "suspension",
        "package", "b-spec", "b spec", "evolution", "improvement",
        "modification", "homologation",
        # Spanish
        "actualización", "actualizacion", "mejoras", "desarrollo",
        "paquete", "alerón", "aleron", "evolución", "evolucion",
    ]
    technical_vocab = [
        "downforce", "drag", "aero", "aerodynamics", "mechanical grip",
        "cooling", "power unit mode", "engine mode", "gearbox",
    ]
    if any(w in t for w in upgrade_vocab + technical_vocab):
        return True
    # Performance vocab only triggers alongside a team/driver name
    perf_vocab = [
        "performance", "faster", "slower", "pace", "gap", "deficit",
        "advantage", "competitive", "struggling", "improved",
        "rendimiento", "ventaja", "desventaja",
    ]
    team_driver_hints = [
        "ferrari", "mclaren", "mercedes", "red bull", "redbull", "aston",
        "alpine", "williams", "haas", "audi", "cadillac",
        "norris", "piastri", "verstappen", "leclerc", "hamilton",
        "russell", "alonso", "sainz", "stroll", "albon",
    ]
    return any(w in t for w in perf_vocab) and any(w in t for w in team_driver_hints)


def _build_upgrade_search_query(user_msg: str, mem: dict) -> str:
    """Builds a focused search string for upgrade/technical context."""
    t = user_msg.lower()
    parts: list[str] = []

    # Team mention (checked before driver to prefer the broader entity)
    for team, keywords in TEAM_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            parts.append(team.replace("_", " ").title())
            break

    # Driver mention as fallback
    if not parts:
        for key in sorted(DRIVER_CODE_MAP.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(key)}\b", t):
                parts.append(key.upper())
                break

    # Add last known race name for recency
    episodes = mem.get("episodic", [])
    if episodes:
        last_race = episodes[-1].get("race_name", "")
        if last_race:
            parts.append(last_race)

    parts.append("upgrades technical development 2026 F1")
    return " ".join(parts)


def _is_live_session_question(text: str) -> bool:
    """
    Detects if a question needs live session data.
    Triggers whenever a session is mentioned — no action word required.
    "top 10 fp2", "fp2", "qualifying results", "who got pole" all trigger.
    """
    t = text.lower()

    # Never trigger on internal predictor prompts
    if any(skip in t for skip in [
        "f1_2026_predictor", "monte carlo", "win probability",
        "podium). p2:", "circuit_score", "mechanical_risk",
        "recent_form=", "quali_pos_next", "in 2-3 sentences",
        "confirm this pick",
    ]):
        return False

    # Trigger on any session mention
    session_triggers = [
        "fp1", "fp2", "fp3",
        "free practice", "práctica libre", "practice 1",
        "practice 2", "practice 3",
        "qualifying", "quali", "clasificación", "clasificacion",
        "q1", "q2", "q3",
        "sprint qualifying", "sprint race",
        "pole position",
        "who got pole",
        "top 10", "top ten", "classification",
        "session results", "session times",
        "fastest lap", "vuelta rápida", "vuelta rapida",
        # Spanish session words
        "entreno", "práctica", "practica",
        "qualy", "clasificacion",
    ]
    return (any(kw in t for kw in session_triggers)
            or bool(re.search(r'\bsprint\b', t))
            or bool(re.search(r'\bpole\b', t)))



def build_user_profile(user_data: dict) -> str:
    """
    Builds a personalization string from a user's chat history.
    Used to make the bot feel like it remembers the user.
    """
    if not user_data:
        return ""

    stats   = user_data.get("stats", {})
    history = user_data.get("history", [])
    topics  = stats.get("favorite_topics", {})

    if not topics and not history:
        return ""

    parts = []

    # Favourite topic
    if topics:
        top = max(topics, key=topics.get)
        parts.append(f"This user's favourite topic is {top}.")

    # Detect favourite driver from chat history
    driver_mentions = {}
    driver_keys = {
        "checo": "Sergio Pérez", "pérez": "Sergio Pérez", "perez": "Sergio Pérez",
        "hamilton": "Lewis Hamilton", "lewis": "Lewis Hamilton",
        "verstappen": "Max Verstappen", "max": "Max Verstappen",
        "antonelli": "Kimi Antonelli", "kimi": "Kimi Antonelli",
        "leclerc": "Charles Leclerc", "charles": "Charles Leclerc",
        "russell": "George Russell", "norris": "Lando Norris",
        "lando": "Lando Norris", "alonso": "Fernando Alonso", "nano": "Fernando Alonso",
    }
    for msg in history:
        if msg.get("role") == "user":
            t = msg.get("content", "").lower()
            for key, driver in driver_keys.items():
                if key in t:
                    driver_mentions[driver] = driver_mentions.get(driver, 0) + 1

    if driver_mentions:
        fav = max(driver_mentions, key=driver_mentions.get)
        parts.append(f"This user talks about {fav} most — they're probably a fan.")

    # Detect language preference
    spanish_count = sum(
        1 for m in history
        if m.get("role") == "user" and
        any(w in m.get("content","").lower()
            for w in ["qué", "cómo", "quién", "cuándo", "ganar", "carrera"])
    )
    if spanish_count > len(history) * 0.3:
        parts.append("This user primarily communicates in Spanish.")

    if not parts:
        return ""

    return "USER PROFILE:\n" + "\n".join(f"- {p}" for p in parts)
CIRCUIT_GUIDES = {
    "melbourne": """
Albert Park, Australia — Season opener, semi-street circuit
- 58 laps, 5.278km, 16 corners
- Medium downforce, park circuit through Albert Park lake
- Turn 1 and 3 are big overtaking spots, Turn 9/10 chicane crucial
- Safety car common at Turn 1 on lap 1 — always drama on opening lap
- Tyres: medium degradation, usually 1-stop but 2-stop possible in heat
- Weather: Melbourne autumn, can be changeable, occasional rain
- Strategy: undercut works well here — pit early if you have pace
""",
    "shanghai": """
Shanghai International Circuit, China
- 56 laps, 5.451km, 16 corners
- Long back straight with hairpin — multiple Straight Mode zones plus strong Overtake Mode range here, best overtaking of the season
- Sector 1: sweeping high-speed esses, very demanding on front tyres
- Sector 3: long Turn 16 leads onto pit straight — tyre stress
- Tyres: high deg, especially rear. 2-stop common strategy
- Sprint weekend in 2026
- Strategy: the hairpin makes undercuts very effective
- Key battle zone: Turn 14 hairpin — heavy braking, lots of passes
""",
    "suzuka": """
Suzuka Circuit, Japan — Figure-8 layout, technical masterpiece
- 53 laps, 5.807km, 18 corners
- Sector 1: S-curves — ultra-high-speed commitment, terrifying and beautiful
- 130R — one of fastest corners in F1, flat for top cars
- Spoon Curve: double apex, key for sector 3 exit speed
- Degner 1&2: blind exit, crucial for sector time
- Casino Triangle: tricky chicane, easy to run wide
- Overtaking: Straight Mode on main straight, Turn 1 works but difficult; Overtake Mode helps but options are limited
- Tyres: medium-high deg on rears through S-curves
- Weather: Japanese autumn, can be wet — the 2022 rain race happened here
""",
    "bahrain": """
Bahrain International Circuit, Sakhir — Night race
- 57 laps, 5.412km, 15 corners
- Very abrasive surface — one of hardest on tyres all season
- Turns 1-4 complex: key overtaking zone, cars go side by side
- Turn 10 hairpin: best overtaking spot, long run to braking zone
- Turn 14-15 chicane: Straight Mode zone — second big pass opportunity; Overtake Mode key here
- Tyres: very high degradation. 2-stop almost mandatory. Rear limited
- Atmosphere: incredible under floodlights in the desert
- Strategy: pit window crucial, undercut is king at Turn 1
""",
    "jeddah": """
Jeddah Corniche Circuit, Saudi Arabia — Night street circuit
- 50 laps, 6.174km — one of longest circuits
- Fastest street circuit in F1. Terrifyingly quick walls everywhere
- 27 corners, most taken flat or near-flat at 200-300 km/h
- Very low downforce setup — all about top speed
- Sector 2: blind chicanes, near-zero margin for error
- Overtaking: mainly Turn 1 and Straight Mode zones — limited due to speed; Overtake Mode used more defensively than offensively
- Safety car almost certain — walls catch everything
- Tyres: low deg due to smooth surface despite the speed
- Strategy: typically 1-stop, SC window timing is critical
""",
    "miami": """
Miami International Autodrome, USA — Sprint weekend
- 57 laps, 5.412km, 19 corners
- Street-style circuit around Hard Rock Stadium
- Turns 11-16: technical middle sector, very tight, safety car zone
- Main straight Straight Mode: biggest passing zone; Overtake Mode deployed on approach
- Turn 1 braking: overtaking possible but risky
- Tyres: medium degradation, 1-stop viable but 2-stop faster
- Sprint weekend in 2026
- Atmosphere: American F1 party, big crowd energy
- Strategy: track position very important, undercut works here
""",
    "imola": """
Autodromo Enzo e Dino Ferrari, Imola, Italy
- 63 laps, 4.909km, 19 corners
- Very old school, narrow circuit — extremely limited overtaking
- Tamburello chicane: high-speed approach, site of Senna's 1994 accident
- Variante Alta: crucial chicane for sector 2 time
- Rivazza corners: double left-hander, key for lap time
- Raidillon-style Piratella: fast left-hander, respect required
- Overtaking: virtually impossible outside Straight Mode zone at Turn 2; Overtake Mode crucial for any attack attempt
- Tyres: medium deg, typically 1-stop
- Strategy: qualifying position is almost everything here
""",
    "monaco": """
Circuit de Monaco — The most famous street circuit
- 78 laps, 3.337km, 19 corners
- Overtaking difficulty: 9.5/10 — almost impossible once positions set
- Qualifying is EVERYTHING — pole wins ~50% of the time
- Sainte Devote (T1): crash magnet on lap 1, tight right-hander
- Massenet-Casino: fast left into Casino Square right, commitment needed
- Grand Hotel Hairpin: tightest corner in F1, 15 km/h
- Tunnel: unique in F1, blind exit onto chicane
- Nouvelle Chicane: traditional collision point, marshal's nightmare
- Tabac: fast left, walls either side, no run-off
- Swimming Pool complex: S-section, easy to clip the wall
- Rascasse: final hairpin, notorious for time-wasting in qualifying
- Tyres: low deg due to low speeds, usually 1-stop
- Strategy: SC timing is everything — all pit stops happen under SC
""",
    "montreal": """
Circuit Gilles Villeneuve, Canada
- 70 laps, 4.361km, 14 corners
- Semi-street circuit on an island in the St. Lawrence River
- Wall of Champions (T13-14): claims world champions every year
- Casino Hairpin: best overtaking spot, very late braking zone
- Long back straight with chicane: high-speed braking, Straight Mode zone — Overtake Mode effective into the hairpin
- Pit lane exit: merges aggressively, causes incidents
- Tyres: medium deg. Mixed 1 and 2-stop strategies common
- Weather: can be anything in Canadian June — wet races common
- Safety car probability: very high due to Wall of Champions
- Strategy: SC timing critical, fuel-saving important here
""",
    "barcelona": """
Circuit de Barcelona-Catalunya, Spain
- 66 laps, 4.655km, 16 corners
- Development circuit — extensively used for testing, teams know it cold
- Turn 1-2: classic overtaking point, heavy braking from high speed
- Turn 3 Renault: long high-speed right-hander, downforce critical
- Turn 9: hardest corner — long, decreasing radius, rear stressed
- Straight Mode zones: main straight and back straight — Overtake Mode most effective into Turn 1
- Tyres: very high rear degradation. 2-stop almost always faster
- Strategy: tyre management defines the race. Undercut very effective
- Dirty air problem: very hard to follow through high-speed corners
""",
    "spielberg": """
Red Bull Ring, Spielberg, Austria
- 71 laps, 4.318km — shortest lap on calendar
- Very short lap but incredibly fast, set in beautiful mountains
- Turns 3-4: massive uphill braking zone — wheel-to-wheel frequent
- Turn 7-8: final two corners before long straight — Straight Mode carries speed onto the straight, Overtake Mode the key attacking tool into Turn 1
- Very high downforce circuit despite short lap
- Tyres: high stress on short lap, lots of laps = high total deg
- Austrian Grand Prix atmosphere: Orange army (Max fans) everywhere
- Weather: Styrian mountains bring rain frequently
- Overtaking: Turn 3 and Turn 4 are both viable — good racing
""",
    "silverstone": """
Silverstone, UK — High-speed temple
- 52 laps, 5.891km, 18 corners
- Copse: flat-out at 290+ km/h for top cars. THE commitment corner
- Maggotts-Becketts-Chapel: iconic S-sequence, ultra-high speed
- Hangar Straight + Stowe: biggest overtaking opportunity
- Vale-Club complex: technical final sector
- Wellington Straight: second Straight Mode zone, good passing — Overtake Mode supplements the drag reduction
- Tyres: very hard on rear tyres through high-speed corners
- British fans: incredible atmosphere, biggest F1 crowd of the year
- Weather: British summer = anything. Rain very common
- Strategy: 2-stop almost mandatory. Tyre deg very high
""",
    "budapest": """
Hungaroring, Budapest, Hungary
- 70 laps, 4.381km, 14 corners
- Monaco without the walls — extremely difficult to overtake
- Very twisty, high-downforce circuit, 0.95 overtaking difficulty
- Turn 1-2 complex: only real overtaking zone, very late braking
- Turn 4: hairpin entry, can catch someone napping
- Turns 11-12: technical complex before long straight
- Tyres: medium deg due to low speeds but very long lap
- Strategy: qualifying position dominates. Overcut possible due to long pit lane
- Atmosphere: boiling hot European summer, always 35°C+
""",
    "spa": """
Circuit de Spa-Francorchamps, Belgium — Greatest circuit on the calendar
- 44 laps, 7.004km — longest circuit on calendar
- Eau Rouge/Raidillon: most iconic sequence in F1. Terrifying uphill flat-out
- Pouhon: double-apex high-speed left, one of best corners in racing
- Blanchimont: 300 km/h left before Bus Stop chicane
- Kemmel Straight: longest Straight Mode zone, massive overtaking opportunity — Overtake Mode makes Les Combes a brutal attack point
- Bus Stop chicane: final complex, under/overcut zone
- Weather: Spa has own microclimate — can be wet sector 1, dry sector 3
- Tyres: high deg on long circuit. 2-stop typically
- Atmosphere: forest setting, camping fans, legendary vibe
- History: Belgian GP has produced some of the greatest F1 moments
""",
    "zandvoort": """
Circuit Zandvoort, Netherlands — Seaside banked circuit
- 72 laps, 4.259km, 14 corners
- Banked final corner (Arie Luyendyk): car sticks to wall through banking
- Hugkenheim: banked second-to-last corner, unique sensation for drivers
- Turn 3 chicane: main overtaking zone
- Very narrow, difficult to follow — limited passing opportunities
- Tyres: medium deg. 2-stop typically used
- Orange army (Dutch fans): incredible atmosphere, sea of orange
- Dune setting: sand can get blown onto circuit, grip changes
- Strategy: VSC/SC very common — narrow track = incidents
""",
    "monza": """
Autodromo Nazionale Monza, Italy — Temple of Speed
- 53 laps, 5.793km — fastest average speed of the season
- Lowest downforce setup of the year — teams run near-zero wing
- Rettifilo straight: 350+ km/h before Turn 1 braking
- Curve Grande: fast right taken flat, slight compression
- Lesmo 1&2: right-handers into the forest, tricky
- Ascari chicane: crucial complex, get it wrong and lose lots
- Parabolica (Curva Biassono): long sweeping final corner, 270° arc
- Slipstreaming huge — qualifying can produce amazing battles
- Tyres: low deg despite high speeds — smooth surface
- Strategy: 1-stop almost always. Straight Mode dominant here — slipstreaming amplified by wing-open speeds
- Tifosi (Ferrari fans): incredible passion, orange smoke and flags everywhere
""",
    "baku": """
Baku City Circuit, Azerbaijan — Street circuit chaos
- 51 laps, 6.003km
- Second longest straight in F1 — 2.2km along the Caspian seafront
- Castle section: narrow medieval streets, 7-8 metres wide at tightest
- Turn 8: notorious blind entry, claimed many victims
- Turn 15-16: complex before long straight, crucial for lap time
- Overtaking: lots of it. Long straight + Straight Mode = massive speed differential; Overtake Mode deadly on the seafront run
- Tyres: low deg on smooth streets, 1-stop typical
- Safety car: virtually guaranteed every year. Chaos track
- Strategy: SC window management critical — timing the VSC/SC pit
- Famous for: bizarre results, random retirements, late drama
""",
    "singapore": """
Marina Bay Street Circuit, Singapore — Night race in the city
- 62 laps, 4.940km
- Hottest race of the year — 35°C with 80%+ humidity
- Very technical, low-speed street circuit
- Anderson Bridge section: historic race track feeling
- Esplanade Drive: wall of fame claimed many front wings
- Turn 18-22: complex chicanes, high SC probability
- Overtaking: extremely difficult. Qualifying crucial
- Tyres: low deg due to low speeds. 1-stop standard
- Strategy: SC almost certain — timing everything
- Atmosphere: stunning night backdrop of Marina Bay skyline
""",
    "austin": """
Circuit of the Americas, Texas, USA
- 56 laps, 5.513km, 20 corners
- Turn 1: uphill, blind apex approach, best mass passing spot
- Sectors 1 and 2: flowing high-speed corners inspired by classic circuits
- Turns 12-13: inspired by Maggotts-Becketts, high-speed esses
- Back straight: Straight Mode zone into Turn 12 hairpin — Overtake Mode key for late braking attacks
- Tyres: hard on rear through high-speed complex
- 2-stop usually faster but 1-stop possible
- Sprint weekend some years
- American crowd: huge fan base now, COTA has become legendary venue
""",
    "mexico city": """
Autodromo Hermanos Rodriguez, Mexico City
- 71 laps, 4.304km, 17 corners
- Altitude: 2,285m above sea level — engines produce ~20% less power
- Power units stressed significantly — cooling critical
- Peraltada: famous banked final corner, tight stadium section after
- Foro Sol stadium section: incredible atmosphere, tight and twisty
- Main straight: Straight Mode helps but altitude reduces top speed effect; Overtake Mode less potent at 2,285m
- Tyres: medium deg. 1-stop usually optimal
- Strategy: unique altitude effect means teams test things here
- Crowd: Mexican fans loudest of the year — Checo home race atmosphere
- Pit lane: very long, overcut strategy interesting
""",
    "são paulo": """
Autodromo Jose Carlos Pace, Interlagos, Brazil
- 71 laps, 4.309km, 15 corners
- One of greatest circuits — anti-clockwise (unusual), very old school
- Senna S: iconic double right-hander at turn 1-2, beautiful and dangerous
- Descida do Lago: blind downhill entry into hairpin
- Curva do Sol: sweeping right in sector 2
- Arquibancadas: amphitheatre feel, fans incredibly close to track
- Weather: tropical, rain almost guaranteed at some point in race week
- Overtaking: Turns 1 and 4 both viable, very exciting racing
- Tyres: high degradation, often 2-stop
- Sprint weekend some years
- Atmosphere: one of the absolute best of the year
""",
    "las vegas": """
Las Vegas Strip Circuit, USA — Night race on famous Strip
- 50 laps, 6.201km — one of longest circuits
- The Strip straight: 1.9km flat-out alongside the casinos
- Very high speed circuit — second fastest average after Monza
- Low downforce setup needed
- Thomas and Mack section: technical stadium complex
- Tyres: very low deg on smooth strip tarmac
- Strategy: typically 1-stop on low-deg surface
- Weather: November desert cold — can be surprisingly cold at night
- Atmosphere: unique spectacle, gambling capital of world as backdrop
- New to calendar: still finding its identity as a racing venue
""",
    "lusail": """
Lusail International Circuit, Qatar — Night race
- 57 laps, 5.380km, 16 corners
- Very fast, flowing circuit — high-speed sweepers throughout
- Turns 1-6: long high-speed complex, committed driving required
- Back section: more technical, slower corners for contrast
- Straight Mode zones: main straight into Turn 1 — decent overtaking; Overtake Mode adds a meaningful attack option
- Tyres: very high deg — blistering issues historically
- 2-stop almost mandatory. Tyre management critical
- Sprint weekend in 2026
- Atmosphere: growing fan base in Middle East
""",
    "abu dhabi": """
Yas Marina Circuit, Abu Dhabi — Season finale, night race
- 58 laps, 5.281km, 16 corners
- Updated 2021: faster, more overtaking-friendly after redesign
- Turn 5-6-7: main technical complex, tight and twisty
- Back straight: Straight Mode zone, reasonable overtaking — Overtake Mode the main attack weapon here
- Marina section: night race with hotel, yachts — spectacular backdrop
- Tyres: medium deg, 1-stop typical but 2-stop possible
- Strategy: final race, teams sometimes gamble for championship
- Atmosphere: unique finale feel — end of season emotion
- History: 2021 Hamilton-Verstappen title decider happened here
""",
}

def _resolve_circuit_key(query: str) -> str:
    """
    Returns the CIRCUIT_GUIDES key for query, or '' if no match.
    Uses CIRCUIT_ALIASES (sorted longest-first, word-boundary matched) then
    falls back to direct CIRCUIT_GUIDES key substring matching.
    """
    q = query.lower()
    # Aliases: longest first so "saudi arabia" beats "saudi", word-boundary
    # to avoid short aliases (uk, usa, cota) matching inside other words.
    for alias in sorted(CIRCUIT_ALIASES.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', q):
            return CIRCUIT_ALIASES[alias]
    # Direct CIRCUIT_GUIDES key match — word-boundary to stop "spa" matching
    # inside "spanish", consistent with alias matching above.
    for circuit in CIRCUIT_GUIDES:
        if re.search(r'\b' + re.escape(circuit) + r'\b', q):
            return circuit
    return ""


_CIRCUIT_MAPS_DIR = Path(__file__).parent / "boxboxai_circuit_maps"


def _find_circuit_map_image(circuit_key: str) -> Path | None:
    """Return path to a saved circuit map image, or None if not on disk."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = _CIRCUIT_MAPS_DIR / f"{circuit_key}_v{CIRCUIT_MAP_VERSION}{ext}"
        if p.exists():
            return p
    return None


def get_circuit_guide(query: str) -> tuple[str, Path | None]:
    """Returns (circuit_guide_text, image_path_or_None) for the circuit in query."""
    key = _resolve_circuit_key(query)
    if not key:
        return "", None
    guide = CIRCUIT_GUIDES[key]
    zone_suffix = _build_zone_suffix(key)
    text = f"CIRCUIT GUIDE — {key.upper()}:{guide}{zone_suffix}"
    return text, _find_circuit_map_image(key)


def get_circuit_map_image(query: str) -> Path | None:
    """Return cached circuit map image path for the circuit in query, or None."""
    key = _resolve_circuit_key(query)
    return _find_circuit_map_image(key) if key else None


def _is_circuit_guide_query(text: str) -> bool:
    """Returns True only for explicit circuit guide / track info requests."""
    t = text.lower()
    if not _resolve_circuit_key(t):
        return False
    guide_intent = [
        "circuit guide", "track guide", "circuit map", "track map",
        "circuit info", "track info", "tell me about the circuit",
        "tell me about the track", "about the circuit", "about the track",
        "circuit layout", "track layout", "corners", "corner guide",
        "drs zones", "overtaking spots", "overtaking opportunities",
        "guía del circuito", "mapa del circuito", "circuito de",
        "info del circuito", "sobre el circuito",
    ]
    return any(kw in t for kw in guide_intent)


def _build_zone_suffix(circuit_key: str) -> str:
    """
    Returns a formatted zone-data suffix for get_circuit_guide(), or "".
    Calls _get_circuit_zone_data() which is fetch-on-demand and cached.
    Never raises — any failure returns "".
    """
    try:
        if not _get_circuit_zone_data_fn[0]:
            log.warning("context_builder._get_circuit_zone_data_fn not wired — zone data unavailable")
            return ""
        data = _get_circuit_zone_data_fn[0](circuit_key)
        if not data:
            return ""
        ot    = data.get("overtake", {})
        zones = data.get("straight_mode_zones", [])
        if not ot and not zones:
            return ""
        parts = []
        if ot.get("detection") and ot.get("activation"):
            parts.append(
                f"Overtake detection {ot['detection']}, "
                f"activation {ot['activation']}")
        if zones:
            zone_strs = [
                f"{z['zone']}: {z['activation_normal']} (normal grip) / "
                f"{z['activation_low_grip']} (low grip)"
                for z in zones
            ]
            parts.append(f"Active Straight Mode zones: {', '.join(zone_strs)}")
        if not parts:
            return ""
        return (
            "\nPRECISE 2026 ZONE DATA (FIA Competition Notes): "
            + ". ".join(parts) + "."
        )
    except Exception:
        return ""
_app_ref: list = [None]
def build_system_prompt(mem: dict, news_context: str = "",
                        weather_context: str = "",
                        historical_context: str = "",
                        user_profile: str = "",
                        live_context: str = "",
                        circuit_guide: str = "",
                        prediction_accuracy: str = "",
                        driver_stats: str = "",
                        practice_context: str = "",
                        live_search_context: str = "",
                        fia_docs_context: str = "",
                        next_race_context: str = "",
                        driver_profile: str = "",
                        race_replay: str = "",
                        champ_scenarios: str = "",
                        fan_profile: str = "",
                        upgrade_context: str = "") -> str:
    """
    Token-optimized system prompt builder.
    Core: ~400 tokens always. Context: injected only when needed.
    Total typical: 600-900 tokens vs old 2400+.
    """
    # ── Compact race memory (last 8 races max) ───────────────
    ep_lines = []
    for r in mem.get("episodic", [])[-8:]:
        quali  = r.get("qualifying", {})
        sprint = r.get("sprint", {})
        dnfs   = r.get("dnfs", [])
        fc     = r.get("full_classification", [])
        line   = (f"R{r['round']} {r.get('race_name', r.get('track','?'))} "
                  f"→ {r.get('winner','?')} P2:{r.get('p2','')} "
                  f"P3:{r.get('p3','')} FL:{r.get('fastest_lap','')} "
                  f"({r.get('fastest_lap_time','')})")
        if quali.get("pole"):
            line += f" Pole:{quali['pole']}({quali.get('pole_time','')})"
        if sprint.get("sprint_winner"):
            line += f" Sprint:{sprint['sprint_winner']}"
        if dnfs:
            line += f" DNF:{','.join(d.split('(')[0] for d in dnfs[:3])}"
        if r.get("champ_after"):
            line += f" Champ:{r['champ_after'][:60]}"
        tyres = r.get("pitstops", {}).get("tyre_strategies", "")
        if tyres:
            line += f" Tyres:{tyres[:80]}"
        if fc:
            line += f" Grid:{' '.join(fc[:10])}"
        ep_lines.append(line)

    # ── Compact semantic facts ────────────────────────────────
    sem_lines = []
    for k, v in list(mem.get("semantic", {}).items())[-15:]:
        text = v["text"] if isinstance(v, dict) else v
        sem_lines.append(f"{k}: {text[:120]}")

    # ── Dynamic context blocks (only when not empty) ──────────
    ctx_blocks = []
    if next_race_context:
        ctx_blocks.append(next_race_context)
        log.debug(f"ctx_block NEXT_RACE: {len(next_race_context)} chars")
    if news_context:
        _nc, _nm = _unpack_ctx(news_context)
        ctx_blocks.append(f"NEWS{_nm}:{_nc[:300]}")
        log.debug(f"ctx_block NEWS: {len(_nc[:300])} chars{_nm}")
    if live_search_context:
        ctx_blocks.append(f"LIVE SEARCH:{live_search_context[:800]}")
        log.debug(f"ctx_block LIVE_SEARCH: {len(live_search_context[:800])} chars")
    if fia_docs_context:
        ctx_blocks.append(f"FIA STEWARDS DOCS:{fia_docs_context[:1500]}")
        log.debug(f"ctx_block FIA_DOCS: {len(fia_docs_context[:1500])} chars")
    if upgrade_context:
        ctx_blocks.append(f"TECHNICAL UPDATES:{upgrade_context[:600]}")
        log.debug(f"ctx_block TECHNICAL_UPDATES: {len(upgrade_context[:600])} chars")
    if weather_context:
        ctx_blocks.append(f"WEATHER:{weather_context[:150]}")
        log.debug(f"ctx_block WEATHER: {len(weather_context[:150])} chars")
    if live_context:
        ctx_blocks.append(f"LIVE SESSION:{live_context[:200]}")
        log.debug(f"ctx_block LIVE_SESSION: {len(live_context[:200])} chars")
    if practice_context:
        _pc, _pm = _unpack_ctx(practice_context)
        ctx_blocks.append(f"SESSION DATA{_pm}:{_pc[:800]}")
        log.debug(f"ctx_block SESSION_DATA: {len(_pc[:800])} chars{_pm}")
    if circuit_guide:
        ctx_blocks.append(f"CIRCUIT:{circuit_guide[:1500]}")
        log.debug(f"ctx_block CIRCUIT: {len(circuit_guide[:1500])} chars")
    if driver_stats:
        ctx_blocks.append(f"DRIVER STATS:{driver_stats[:300]}")
        log.debug(f"ctx_block DRIVER_STATS: {len(driver_stats[:300])} chars")
    if driver_profile:
        ctx_blocks.append(f"DRIVER PROFILE:{driver_profile[:1500]}")
        log.debug(f"ctx_block DRIVER_PROFILE: {len(driver_profile[:1500])} chars")
    if race_replay:
        _rr, _rm = _unpack_ctx(race_replay)
        ctx_blocks.append(f"RACE REPLAY{_rm}:{_rr[:800]}")
        log.debug(f"ctx_block RACE_REPLAY: {len(_rr[:800])} chars{_rm}")
    if champ_scenarios:
        ctx_blocks.append(f"CHAMPIONSHIP SCENARIOS:{champ_scenarios[:1200]}")
        log.debug(f"ctx_block CHAMPIONSHIP_SCENARIOS: {len(champ_scenarios[:1200])} chars")
    if fan_profile:
        ctx_blocks.append(f"FAN PROFILE:{fan_profile[:400]}")
        log.debug(f"ctx_block FAN_PROFILE: {len(fan_profile[:400])} chars")
    if historical_context:
        ctx_blocks.append(f"HISTORY:{historical_context[:1200]}")
        log.debug(f"ctx_block HISTORY: {len(historical_context[:1200])} chars")
    if prediction_accuracy:
        ctx_blocks.append(f"PREDICTION RECORD:{prediction_accuracy[:150]}")
        log.debug(f"ctx_block PREDICTION_RECORD: {len(prediction_accuracy[:150])} chars")
    if user_profile:
        ctx_blocks.append(f"USER:{user_profile[:150]}")
        log.debug(f"ctx_block USER: {len(user_profile[:150])} chars")

    ctx_str = "\n\n".join(ctx_blocks)

    return f"""BoxBoxAI — F1 analyst bot. 2026 season expert. Developed by Erick Hernandez.

GRID: Mercedes:ANT+RUS | Ferrari:LEC+HAM | RedBull:VER+HAD | McLaren:NOR+PIA | AstonMartin:ALO+STR | Alpine:GAS+COL | Williams:ALB+SAI | RB:LAW+LIN | Haas:BEA+OCO | Audi:HUL+BOR | Cadillac:PER+BOT
CRITICAL: Checo(PER) is at CADILLAC not Red Bull. Lawson replaced him at RBR.
NICKNAMES: Checo/Viejo Sabroso/God of Monaco=PER | Magic/GOAT/Sir Lewis=HAM | Mr Saturday=RUS | El Nano/Smooth Operator=ALO | Super Max=VER | Il Predestinato=LEC | El Pibe/La Bomba Argentina=COL | Baby Kimi/Il Bambino=ANT | Hulk=HUL | Carlitos/El Matador=SAI | El Kiwi=LAW

SEASON FACTS:
{chr(10).join(sem_lines[-10:])}

RACE RESULTS:
{chr(10).join(ep_lines)}

RULES:
- F1 only. Non-F1 questions: redirect warmly, suggest ChatGPT
- Language: match user (Spanish→Mexican Spanish, English→English)
- Style: confident, direct, opinionated, punchy. Never start "Certainly!"
- Emojis: use them naturally and relevantly — 🏆🥇🥈🥉 for wins/podiums, 🔥💨 for pace/dominance, 🚀 for great starts, 🛞 for strategy/tyres, 🏁🚦 for race start/finish, ⚠️🔧 for incidents/failures, 📊 for stats, team colors (🔴Ferrari 🔵RedBull 🟡McLaren 🟢AstonMartin ⚪Williams) and flags for countries when relevant. Don't overdo it — 1-3 per message, placed where they add punch.
- NEVER invent lap times, positions, results, or grid order. If you don't have real data, say: "I don't have the timing data for that yet — ask again in a few minutes."
- NEVER tell users to check F1.com, the F1 app, Twitter, or any external source. Either you have the data or you say you don't — never redirect elsewhere.
- NEVER guess or estimate times, positions, or finishing order. Made-up numbers are worse than no answer.
- When SESSION DATA is present: use those exact times/positions only — that's the real timing sheet.
- When no SESSION DATA and the question needs it: admit it in one honest sentence, no padding, no speculation dressed as analysis.
- STALE OR PARTIAL CONTEXT: When a block label includes an age (e.g. "2.3h old") or is marked "partial" or "unknown", state that explicitly in your answer — e.g. "that session data is from 2h ago" or "tyre strategy may be incomplete as FastF1 data is still being processed". Never silently fill gaps with inference when context is marked partial or unknown.
- STRATEGY/TYRE QUESTIONS WITHOUT REAL DATA: CIRCUIT GUIDE info (degradation, overtaking difficulty, "usually 2-stop") is general historical knowledge — fine to share AS general knowledge. But NEVER invent specific lap numbers for pit stops, per-driver stint plans (e.g. "Stint 1: Laps 1-20"), fake statistics ("Barcelona averages 0.8 safety cars"), or confidence percentages ("90% of the field does 2 stops") when you don't have this year's tyre allocation or practice data. One paragraph of general circuit context is enough — do not pad it into a multi-driver strategy report.
- FIA STEWARDS DOCS = ground truth for incidents and penalties. When FIA STEWARDS DECISION is present, cite "FIA stewards found..." with the specific finding. When RACE DIRECTOR NOTES is present, cite "Race Director notes for [circuit] state..." for circuit-specific rules or procedures. When PIRELLI TYRE NOTES is present, use it for tyre compound and strategy context.
  FIA documents are only available for the current race weekend — for past race penalties or incidents, rely on race control messages and memory rather than claiming to have checked FIA documents. If asked about a past race incident with no FIA STEWARDS DECISION block present, say clearly that the FIA documents for that race are no longer available and answer from what is known.
  When FIA STEWARDS DOCS contains "[MEDIA SOURCES — NOT OFFICIAL FIA DOCUMENT]", attribute all claims to the media outlet — use "According to [source]..." — never use "FIA stewards found" or "FIA confirmed" for media-sourced content.
- TECHNICAL UPDATES: When TECHNICAL UPDATES context is present, use it to answer questions about team development, new parts, or performance changes. Cite the source ("According to [publication]...") rather than presenting it as established fact. If no TECHNICAL UPDATES context is present and the question requires specific upgrade knowledge, say clearly that you don't have confirmed information about specific parts brought to this race rather than speculating.
- DNF QUESTIONS WITH NO FIA STEWARDS DOC: stewards documents cover on-track incidents and regulation violations — NOT mechanical failures. If a driver DNF'd and no FIA stewards document or race control message mentions them, that absence is itself informative: it suggests the retirement was mechanical or self-inflicted (no third party / no investigation needed), not a gap in your knowledge. Say something like "no stewards investigation was opened for [driver]'s retirement, which points to a mechanical issue rather than an on-track incident" — don't say "I don't have that information" as if it's missing data. IMPORTANT EXCEPTION: If RACE CONTROL FACT for a race is an empty list [], the feed was not loaded for that race — do not treat the empty list as confirmed evidence of no investigation. State only that the classification shows DNF or Lapped and that the retirement cause is not available in memory. Never infer "mechanical" from an unloaded feed.
- RACE CONTROL FACT = messages from the live FIA timing feed about incidents, investigations, and flags during the race. IMPORTANT: these messages never literally say "PENALTY" — instead look for: "TIME DELETED" (lap time invalidated, usually track limits), "BLACK AND WHITE FLAG" (warning), "INCIDENT INVOLVING CAR X NOTED/WILL BE INVESTIGATED", and crucially "REVIEWED — NO FURTHER INVESTIGATION" (stewards looked into it and took NO action — this means NO penalty, the matter was CLEARED). Read the full sequence for a driver: an "incident noted" message followed by "no further investigation" means stewards investigated and found nothing wrong — report this as "investigated but cleared, no penalty issued", not as evidence of a penalty. If RACE CONTROL FACT shows nothing for a driver AND no stewards doc exists either, that's a real "no penalty/no incident" — state it plainly.
- NEVER invent specific incident details (crashes, collisions, spins, mechanical failures) beyond what race control messages or FIA stewards documents explicitly state. If race control shows track limits violations for a driver who finished P16, say "track limits issues and a classified P16 finish" — not "crashed out". "Crashed out" implies a retirement-ending collision; only say this if race control explicitly shows a collision/crash message for that driver.
- [NEWS COVERAGE] labeled context = background only, not a timing sheet — don't extract positions/times from it, use it for storylines and reactions only.
- If a query mentions a race/session not yet in your RACE RESULTS or SEASON FACTS, don't substitute a different race — say you don't have it yet.{f"{chr(10)}{chr(10)}CONTEXT:{chr(10)}{ctx_str}" if ctx_str else ""}

Answer from all context above. Be accurate, specific, direct.
FORMATTING: Telegram only — *bold* for emphasis, no # or ## markdown headers of any level, no --- dividers, mobile-friendly."""
client = None

def get_client():
    global client
    if client is None:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    return client

def _is_news_query(text: str) -> bool:
    """Detects if a query needs current news context."""
    news_keywords = [
        "ban", "banned", "rule", "rules", "regulation", "fia", "penalty",
        "news", "latest", "recent", "today", "this week", "announced",
        "decision", "appeal", "protest", "investigation", "verdict",
        "contract", "signing", "transfer", "fired", "hired", "replaced",
        "upgrade", "update", "development", "engine", "power unit",
        "trick", "illegal", "legal", "protest", "disqualified",
        "noticias", "noticia", "hoy", "semana", "anunció", "prohibido",
        "sanción", "reglamento", "motor", "trampa", "ilegal",
        "monaco", "qualifying", "quali", "practice", "fp1", "fp2", "fp3",
    ]
    t = text.lower()
    return any(kw in t for kw in news_keywords)


def _is_next_race_query(text: str) -> bool:
    """
    Detects questions about the upcoming/next race weekend —
    "what race is next", "tomorrow's race", "can X win this weekend".
    """
    t = text.lower()
    triggers = [
        "next race", "next gp", "this weekend", "tomorrow", "today's race",
        "today's gp", "upcoming race", "upcoming gp", "which race is next",
        "what race is next", "what's next", "whats next",
        "race this", "gp this", "race is tomorrow", "race tomorrow",
        # Spanish
        "próxima carrera", "proxima carrera", "próximo gp", "proximo gp",
        "este fin de semana", "esta semana", "mañana", "el próximo",
        "el proximo", "carrera de mañana", "carrera de hoy",
        # Bare "today"/"hoy" — helps connect "who won today" to the
        # current race weekend's actual result in RACE RESULTS context
        "today", "hoy",
    ]
    return any(kw in t for kw in triggers)


def get_next_race_context() -> str:
    """
    Builds factual context about the upcoming race weekend —
    name, circuit, country, date. Grounds "what race is next/tomorrow"
    and "can X win this weekend" questions in real schedule data.
    """
    try:
        current = fetch_current_race()  # checks if we're IN a race weekend
        next_r  = fetch_next_race()     # Jolpica's next scheduled race

        race = current or next_r
        if not race:
            return ""

        name    = race.get("raceName", "")
        date    = race.get("date", "")
        circuit = race.get("Circuit", {})
        loc     = circuit.get("Location", {})
        locality= loc.get("locality", "")
        country = loc.get("country", "")
        rnd     = race.get("round", "")

        today = datetime.now().date()
        try:
            race_date = datetime.strptime(date, "%Y-%m-%d").date()
            delta = (race_date - today).days
            if delta == 0:
                timing = "TODAY"
            elif delta == 1:
                timing = "TOMORROW"
            elif delta > 1:
                timing = f"in {delta} days"
            elif delta < 0:
                timing = f"{-delta} days ago (this race weekend)"
            else:
                timing = ""
        except Exception:
            timing = ""

        return (
            f"NEXT/CURRENT RACE: R{rnd} {name} — {locality}, {country}. "
            f"Race day: {date} ({timing}). "
            f"Use this for any 'next race', 'tomorrow', 'this weekend' questions."
        )
    except Exception as e:
        log.debug(f"get_next_race_context failed: {e}")
        return ""


def _is_weather_query(text: str) -> bool:
    """Detects if a query is asking about weather."""
    weather_keywords = [
        "weather", "rain", "forecast", "temperature", "hot", "cold",
        "wet", "dry", "climate", "conditions", "wind", "humidity",
        "clima", "lluvia", "pronóstico", "temperatura", "calor", "frío",
        "mojado", "seco", "viento", "húmedo", "condiciones",
        "qué tiempo", "que tiempo", "tiempo hace", "tiempo en",
        "going to rain", "will it rain", "chance of rain",
        "va a llover", "va a hacer", "qué clima",
    ]
    t = text.lower()
    return any(kw in t for kw in weather_keywords)


RACE_KEYWORDS = {
    # Monaco
    "monaco": "Monaco", "mónaco": "Monaco",
    # Spain
    "barcelona": "Spanish", "spain": "Spanish", "spanish": "Spanish",
    "españa": "Spanish", "catalunya": "Spanish",
    # Australia
    "australia": "Australian", "australian": "Australian", "melbourne": "Australian",
    # China
    "china": "Chinese", "chinese": "Chinese", "shanghai": "Chinese",
    # Japan
    "japan": "Japanese", "japanese": "Japanese", "suzuka": "Japanese",
    # Miami
    "miami": "Miami",
    # Canada
    "canada": "Canadian", "canadian": "Canadian", "montreal": "Canadian",
    # Austria
    "austria": "Austrian", "austrian": "Austrian",
    "spielberg": "Austrian", "red bull ring": "Austrian",
    # Britain
    "britain": "British", "british": "British", "silverstone": "British",
    # Belgium
    "belgium": "Belgian", "belgian": "Belgian", "spa": "Belgian",
    # Hungary
    "hungary": "Hungarian", "hungarian": "Hungarian", "budapest": "Hungarian",
    # Netherlands
    "netherlands": "Dutch", "dutch": "Dutch", "zandvoort": "Dutch",
    # Italy
    "italy": "Italian", "italian": "Italian", "monza": "Italian",
    # Singapore
    "singapore": "Singapore",
    # Azerbaijan
    "azerbaijan": "Azerbaijan", "baku": "Azerbaijan",
    # United States
    "united states": "United States", "us gp": "United States",
    "austin": "United States", "cota": "United States",
    # Mexico
    "mexico": "Mexico", "méxico": "Mexico",
    # Brazil / São Paulo
    "brazil": "São Paulo", "brasil": "São Paulo",
    "interlagos": "São Paulo", "são paulo": "São Paulo", "sao paulo": "São Paulo",
    # Las Vegas
    "vegas": "Las Vegas", "las vegas": "Las Vegas",
    # Qatar
    "qatar": "Qatar", "lusail": "Qatar",
    # Abu Dhabi
    "abu dhabi": "Abu Dhabi", "yas marina": "Abu Dhabi",
}


def _detect_driver_stat_query(text: str) -> str | None:
    """Returns driver code if query is asking for career stats."""
    stat_keywords = ["how many wins", "career stats", "total wins", "how many poles",
                     "career wins", "all time wins", "cuántas victorias", "estadísticas",
                     "cuántos podios", "cuántas poles"]
    t = text.lower()
    if not any(kw in t for kw in stat_keywords):
        return None
    driver_map = {
        "hamilton": "HAM", "lewis": "HAM",
        "verstappen": "VER", "max": "VER",
        "leclerc": "LEC", "charles": "LEC",
        "alonso": "ALO", "nano": "ALO", "fernando": "ALO",
        "perez": "PER", "checo": "PER", "pérez": "PER",
        "russell": "RUS", "george": "RUS",
        "norris": "NOR", "lando": "NOR",
        "bottas": "BOT", "valtteri": "BOT",
        "sainz": "SAI", "carlitos": "SAI",
    }
    for name, code in driver_map.items():
        if name in t:
            return code
    return None


def _gather_context(user_msg: str, mem: dict, user_data: dict = None) -> dict:
    """Gathers all context blocks for a query. Returns a dict of raw context strings."""
    news_ctx          = ""
    weather_ctx       = ""
    historical_ctx    = ""
    live_ctx          = ""
    circuit_ctx       = ""
    pred_accuracy     = ""
    driver_stats_ctx  = ""
    user_profile_ctx  = ""
    practice_ctx      = ""
    live_search_ctx   = ""
    race_replay_ctx   = ""
    champ_scenario_ctx= ""
    fan_ctx           = ""
    driver_deep_ctx   = ""

    fia_docs_ctx      = ""
    next_race_ctx     = ""
    upgrade_ctx       = ""

    # Next/upcoming race context — also feeds weather fallback
    if _is_next_race_query(user_msg) or _is_weather_query(user_msg) \
            or any(w in user_msg.lower() for w in
                   ["can ", "will ", "predict", "win the", "puede ganar", "va a ganar"]):
        next_race_ctx = get_next_race_context()

    # ── Session data (OpenF1 direct — highest priority) ──────
    # Only make the live OpenF1 API call when the message is actually about
    # session timing, practice results, or qualifying. _is_live_session_question
    # already gates the search fallback below — now it gates the API call too.
    if _is_live_session_question(user_msg):
        session_ctx = get_session_context(user_msg, live_search_fn=_live_search_fn[0])
        if session_ctx:
            practice_ctx    = session_ctx
            live_search_ctx = ""  # OpenF1 data is better than search
            log.info(f"OpenF1 session data fetched for: {user_msg[:40]}")
        else:
            # No OpenF1 data — live search only, clearly label as news not timing
            if _live_search_fn[0]:
                raw_search = _live_search_fn[0](user_msg)
            else:
                log.warning("context_builder._live_search_fn not wired — live search unavailable")
                raw_search = ""
            if raw_search:
                live_search_ctx = (
                    f"[NEWS COVERAGE — not official timing data]\n{raw_search}")
            log.info(f"Live search fallback for: {user_msg[:40]}")

    # FIA stewards documents
    if _needs_fia_docs(user_msg):
        episodes = mem.get("episodic", [])
        t_lower  = user_msg.lower()

        # Detect race from the QUERY first — "in Monaco" should mean
        # Monaco's episode, not whatever the most recent race is.
        named_episode = None
        for kw, race_word in RACE_KEYWORDS.items():
            if kw in t_lower:
                named_episode = next(
                    (e for e in episodes
                     if race_word.lower() in e.get("race_name","").lower()
                     or race_word.lower() in e.get("track","").lower()),
                    None)
                if named_episode:
                    break

        if named_episode:
            race_ctx = named_episode.get("race_name", "")
        else:
            # No specific race named — use most recent.
            # Also set named_episode so driver-position grounding fires even for
            # queries like 'what happened to Antonelli' without a race keyword —
            # otherwise Claude only sees P1/P2/P3 in ep_lines and invents mid-field positions.
            named_episode = episodes[-1] if episodes else None
            last_race = episodes[-1].get("race_name","") if episodes else ""
            next_race = fetch_next_race()
            next_name = next_race.get("raceName","") if next_race else ""
            race_ctx  = last_race or next_name

        if _fetch_fia_docs_fn[0]:
            fia_docs_ctx = _fetch_fia_docs_fn[0](race_ctx, user_msg)
        else:
            log.warning("context_builder._fetch_fia_docs_fn not wired — FIA docs unavailable")
        if fia_docs_ctx:
            log.info(f"FIA docs fetched for: {user_msg[:50]}")

        # ── Driver finishing position grounding ───────────────
        # If a specific driver is named and we have an episode for
        # the race in question, surface their EXACT classification
        # position even if FIA docs aren't available — this prevents
        # "Checo doesn't appear in my data" when he's just outside
        # the top-5 shown in the compact race summary.
        if named_episode:
            driver_code = resolve_driver_code(t_lower)
            if driver_code:
                fc = named_episode.get("full_classification", [])
                position = next(
                    (item for item in fc if f":{driver_code}" in item), None)
                dnf_entry = next(
                    (d for d in named_episode.get("dnfs", [])
                     if d.startswith(driver_code)), None)
                if position or dnf_entry:
                    if position:
                        detail = position
                    else:
                        status = dnf_entry.split("(",1)[1].rstrip(")") \
                            if "(" in dnf_entry else dnf_entry
                        detail = f"{driver_code} DNF ({status})"
                    grounding = (
                        f"CLASSIFICATION FACT: In the {race_ctx}, "
                        f"{detail} (from official results). "
                        f"Use this exact position when answering.")
                    fia_docs_ctx = (fia_docs_ctx + "\n\n" + grounding).strip() \
                        if fia_docs_ctx else grounding

                # ── Race control messages for this driver ─────────
                # Penalties served DURING the race (e.g. 5s added to
                # a pit stop) don't change final classification, so
                # the CLASSIFICATION FACT above can't reveal them.
                # Search the race's actual FIA race control feed for
                # any message mentioning this driver's surname —
                # this is where in-race penalties/investigations show up.
                rc_msgs = named_episode.get("race_control", [])
                surname = FIA_DRIVER_NAMES.get(driver_code.lower(), "")
                driver_rc = [
                    m for m in rc_msgs
                    if surname and surname.upper() in m.upper()
                ]
                if driver_rc:
                    rc_text = " | ".join(driver_rc[:5])
                    rc_grounding = (
                        f"RACE CONTROL FACT: FIA race control messages "
                        f"mentioning {surname} in the {race_ctx}: "
                        f"{rc_text}. Use these for any questions about "
                        f"penalties, investigations, or incidents during "
                        f"the race — these may not be reflected in the "
                        f"final classification if served during a pit stop.")
                    fia_docs_ctx = (fia_docs_ctx + "\n\n" + rc_grounding).strip() \
                        if fia_docs_ctx else rc_grounding
                elif rc_msgs:
                    # We have race control data but nothing for this
                    # driver — that's useful negative information too.
                    no_rc_grounding = (
                        f"RACE CONTROL FACT: FIA race control messages "
                        f"for the {race_ctx} don't mention {surname}. "
                        f"({len(rc_msgs)} messages on file — safety cars, "
                        f"penalties for other drivers, etc.)")
                    fia_docs_ctx = (fia_docs_ctx + "\n\n" + no_rc_grounding).strip() \
                        if fia_docs_ctx else no_rc_grounding

    # Upgrade / technical development search
    if _is_upgrade_query(user_msg):
        upgrade_search_q = _build_upgrade_search_query(user_msg, mem)
        upgrade_ctx = _live_search_fn[0](upgrade_search_q) if _live_search_fn[0] else ""
        if upgrade_ctx:
            log.info(f"Upgrade/tech search for: {user_msg[:40]}")

    # News
    if _is_news_query(user_msg):
        news_ctx = get_news_context(user_msg)

    # Practice also triggers news search
    if any(kw in user_msg.lower() for kw in
           ["fp1","fp2","fp3","practice","práctica","libre","entreno",
            "qualifying","quali","clasificación","sprint"]):
        if not news_ctx:
            news_ctx = get_news_context(user_msg)

    # Weather
    if _is_weather_query(user_msg):
        current_race = fetch_current_race()
        weather_ctx  = get_weather_context(user_msg, current_race)

    # Historical comparisons
    historical_ctx = get_historical_context(user_msg)

    # Live session
    live_ctx = get_live_session_context()

    # Circuit guide — only inject when user explicitly asks about the circuit/track
    circuit_ctx = ""
    if _is_circuit_guide_query(user_msg):
        circuit_ctx, _ = get_circuit_guide(user_msg)

    # Prediction accuracy
    if any(w in user_msg.lower() for w in
           ["predict", "accuracy", "correct", "wrong", "prediction",
            "predicción", "acertaste", "fallaste"]):
        pred_accuracy = get_prediction_accuracy()

    # Driver career stats
    drv_code = _detect_driver_stat_query(user_msg)
    if drv_code:
        driver_stats_ctx = fetch_driver_career_stats(drv_code)

    # FEATURE 3 — Driver deep dive
    deep_dive_code = _is_driver_deep_dive(user_msg)
    if deep_dive_code:
        driver_deep_ctx = build_driver_profile(deep_dive_code, mem)
    else:
        # "compare X and Y" / "X vs Y" / "gap between X and Y" —
        # resolve_driver_code only returns ONE code, but comparison
        # questions need profiles for BOTH drivers named.
        compare_codes = _resolve_multiple_driver_codes(user_msg)
        if len(compare_codes) >= 2:
            profiles = [build_driver_profile(c, mem) for c in compare_codes[:2]]
            driver_deep_ctx = "\n\n".join(profiles)

    # FEATURE 4 — Race replay intelligence
    race_replay_ctx = get_race_replay_context(user_msg, mem)

    # FEATURE 5 — Championship scenarios
    if _is_championship_scenario(user_msg):
        champ_scenario_ctx = build_championship_scenarios(user_msg, mem)

    # FEATURE 8 — Fan/rival tracking context
    if user_data:
        fan_ctx = build_fan_context(user_data, user_msg)

    # User personalization
    if user_data:
        user_profile_ctx = build_user_profile(user_data)

    # Wrap time-sensitive news context with age metadata
    if news_ctx:
        _nct = get_news_cache_time()
        _news_age = (
            (datetime.now() - _nct).total_seconds() / 3600
            if _nct else None
        )
        news_ctx = ContextBlock(content=news_ctx, data_age_hours=_news_age)

    return {
        "news_ctx":           news_ctx,
        "weather_ctx":        weather_ctx,
        "historical_ctx":     historical_ctx,
        "live_ctx":           live_ctx,
        "circuit_ctx":        circuit_ctx,
        "pred_accuracy":      pred_accuracy,
        "driver_stats_ctx":   driver_stats_ctx,
        "user_profile_ctx":   user_profile_ctx,
        "practice_ctx":       practice_ctx,
        "live_search_ctx":    live_search_ctx,
        "race_replay_ctx":    race_replay_ctx,
        "champ_scenario_ctx": champ_scenario_ctx,
        "fan_ctx":            fan_ctx,
        "driver_deep_ctx":    driver_deep_ctx,
        "fia_docs_ctx":       fia_docs_ctx,
        "next_race_ctx":      next_race_ctx,
        "upgrade_ctx":        upgrade_ctx,
    }


def ask_claude(user_msg: str, history: list, mem: dict,
               user_data: dict = None) -> str:
    """Calls Claude with all available context."""
    ctx = _gather_context(user_msg, mem, user_data)
    system = build_system_prompt(
        mem, ctx["news_ctx"], ctx["weather_ctx"], ctx["historical_ctx"],
        ctx["user_profile_ctx"], ctx["live_ctx"], ctx["circuit_ctx"],
        ctx["pred_accuracy"], ctx["driver_stats_ctx"], ctx["practice_ctx"],
        ctx["live_search_ctx"], ctx["fia_docs_ctx"], ctx["next_race_ctx"],
        driver_profile=ctx["driver_deep_ctx"],
        race_replay=ctx["race_replay_ctx"],
        champ_scenarios=ctx["champ_scenario_ctx"],
        fan_profile=ctx["fan_ctx"],
        upgrade_context=ctx["upgrade_ctx"],
    )
    messages = history + [{"role": "user", "content": user_msg}]

    try:
        resp = get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages[-16:]
        )
        return resp.content[0].text if resp.content else "Sorry, no response."
    except anthropic.AuthenticationError:
        asyncio.create_task(alert_owner(
            _app_ref[0],
            "⚠️ *API Authentication Error*\n\n"
            "Your Anthropic API key is invalid or expired.\n"
            "Check console.anthropic.com and update the key in Railway Variables."
        )) if _app_ref[0] else None
        return "⚠️ API key issue. Contact @ErickHernandez."
    except anthropic.RateLimitError:
        asyncio.create_task(alert_owner(
            _app_ref[0],
            "💳 *API Credits Low or Rate Limited*\n\n"
            "You may be running low on API credits or hitting rate limits.\n"
            "Check console.anthropic.com/billing to top up."
        )) if _app_ref[0] else None
        return "⚠️ Too many requests right now. Try again in a moment!"
    except anthropic.APIStatusError as e:
        if e.status_code == 529:
            asyncio.create_task(alert_owner(
                _app_ref[0],
                f"🔴 *Anthropic 529 — Overloaded*\n\nAPI is under heavy load. "
                f"Users are seeing retry prompts. Will recover automatically."
            )) if _app_ref[0] else None
            return "F1 HQ is overloaded right now 🏎️ Try again in a moment."
        log.error(f"Anthropic API {e.status_code}: {e}", exc_info=True)
        asyncio.create_task(alert_owner(
            _app_ref[0],
            f"⚠️ *Anthropic API Error {e.status_code}*\n\n{str(e)[:200]}"
        )) if _app_ref[0] else None
        return "⚠️ API error. Try again in a moment!"
    except Exception as e:
        log.error(f"Claude API error: {type(e).__name__}: {e}", exc_info=True)
        return f"⚠️ Error calling Claude: {type(e).__name__}: {str(e)[:80]}"
CIRCUIT_COORDS = {
    "melbourne":     (-37.8497,  144.9680, "Melbourne, Australia"),
    "shanghai":      ( 31.3389,  121.2198, "Shanghai, China"),
    "suzuka":        ( 34.8431,  136.5407, "Suzuka, Japan"),
    "bahrain":       ( 26.0325,   50.5106, "Sakhir, Bahrain"),
    "jeddah":        ( 21.6319,   39.1044, "Jeddah, Saudi Arabia"),
    "miami":         ( 25.9581,  -80.2389, "Miami, USA"),
    "imola":         ( 44.3439,   11.7167, "Imola, Italy"),
    "monaco":        ( 43.7347,    7.4206, "Monte Carlo, Monaco"),
    "montreal":      ( 45.5000,  -73.5228, "Montreal, Canada"),
    "barcelona":     ( 41.5700,    2.2611, "Barcelona, Spain"),
    "spielberg":     ( 47.2197,   14.7647, "Spielberg, Austria"),
    "silverstone":   ( 52.0786,   -1.0169, "Silverstone, UK"),
    "budapest":      ( 47.5789,   19.2486, "Budapest, Hungary"),
    "spa":           ( 50.4372,    5.9714, "Spa-Francorchamps, Belgium"),
    "zandvoort":     ( 52.3888,    4.5409, "Zandvoort, Netherlands"),
    "monza":         ( 45.6156,    9.2811, "Monza, Italy"),
    "baku":          ( 40.3725,   49.8533, "Baku, Azerbaijan"),
    "singapore":     (  1.2914,  103.8639, "Singapore"),
    "austin":        ( 30.1328,  -97.6411, "Austin, USA"),
    "mexico city":   ( 19.4042,  -99.0907, "Mexico City, Mexico"),
    "são paulo":     (-23.7036,  -46.6997, "São Paulo, Brazil"),
    "las vegas":     ( 36.1699, -115.1398, "Las Vegas, USA"),
    "lusail":        ( 25.4700,   51.4536, "Lusail, Qatar"),
    "abu dhabi":     ( 24.4672,   54.6031, "Abu Dhabi, UAE"),
}

# Country/race-name aliases → circuit key. Checked BEFORE the
# substring matching above, so "Spain" maps to "barcelona" instead
# of accidentally substring-matching "spa" (Belgium).
CIRCUIT_ALIASES = {
    "australia": "melbourne", "australian": "melbourne",
    "china": "shanghai", "chinese": "shanghai",
    "japan": "suzuka", "japanese": "suzuka",
    "saudi": "jeddah", "saudi arabia": "jeddah", "saudi arabian": "jeddah",
    "emilia romagna": "imola", "romagna": "imola",
    "canada": "montreal", "canadian": "montreal",
    "spain": "barcelona", "spanish": "barcelona",
    "españa": "barcelona", "catalunya": "barcelona", "catalonia": "barcelona",
    "austria": "spielberg", "austrian": "spielberg", "red bull ring": "spielberg",
    "britain": "silverstone", "british": "silverstone", "uk": "silverstone",
    "england": "silverstone", "great britain": "silverstone",
    "hungary": "budapest", "hungarian": "budapest",
    "belgium": "spa", "belgian": "spa", "francorchamps": "spa",
    "netherlands": "zandvoort", "dutch": "zandvoort", "holland": "zandvoort",
    "italy": "monza", "italian": "monza",
    "azerbaijan": "baku",
    "usa": "austin", "us gp": "austin", "united states": "austin",
    "texas": "austin", "cota": "austin",
    "mexico": "mexico city", "méxico": "mexico city",
    "brazil": "são paulo", "brasil": "são paulo", "sao paulo": "são paulo",
    "interlagos": "são paulo",
    "vegas": "las vegas",
    "qatar": "lusail",
    "uae": "abu dhabi", "abudhabi": "abu dhabi", "yas marina": "abu dhabi",
    # Popular venue names not covered by country/race-name aliases above
    "monte carlo": "monaco",
    "hungaroring": "budapest",
    "circuit of the americas": "austin",
    "marina bay": "singapore",
    "hermanos rodriguez": "mexico city",
    "autodromo nazionale": "monza",
    "circuit gilles villeneuve": "montreal",
    "losail": "lusail",
    "albert park": "melbourne",
    "baku city circuit": "baku",
    "österreichring": "spielberg",
    "jeddah corniche": "jeddah",
}
WMO_CODES = {
    0:  ("Clear sky", "☀️"),
    1:  ("Mainly clear", "🌤"),
    2:  ("Partly cloudy", "⛅"),
    3:  ("Overcast", "☁️"),
    45: ("Foggy", "🌫"),
    48: ("Icy fog", "🌫"),
    51: ("Light drizzle", "🌦"),
    53: ("Moderate drizzle", "🌦"),
    55: ("Heavy drizzle", "🌧"),
    61: ("Slight rain", "🌧"),
    63: ("Moderate rain", "🌧"),
    65: ("Heavy rain", "🌧"),
    71: ("Slight snow", "❄️"),
    73: ("Moderate snow", "❄️"),
    75: ("Heavy snow", "❄️"),
    80: ("Slight showers", "🌦"),
    81: ("Moderate showers", "🌧"),
    82: ("Heavy showers", "🌧"),
    95: ("Thunderstorm", "⛈"),
    96: ("Thunderstorm + hail", "⛈"),
    99: ("Thunderstorm + hail", "⛈"),
}


def _match_circuit(query: str) -> tuple | None:
    """
    Matches a query string to a circuit in CIRCUIT_COORDS using
    word-boundary matching — avoids false positives like "Spain"
    matching "spa" (Spa-Francorchamps) as a raw substring.
    Checks country/race-name aliases first.
    """
    q = query.lower()
    # Aliases first (longest keys first, so "saudi arabia" beats "saudi")
    for alias in sorted(CIRCUIT_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            key = CIRCUIT_ALIASES[alias]
            return key, CIRCUIT_COORDS[key]
    # Word-boundary match on the full key (e.g. "mexico city")
    for key, val in CIRCUIT_COORDS.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            return key, val
    # Word-boundary match on each word of multi-word keys
    for key, val in CIRCUIT_COORDS.items():
        for word in key.split():
            if len(word) >= 4 and re.search(rf"\b{re.escape(word)}\b", q):
                return key, val
    return None


def fetch_weather(lat: float, lon: float,
                  days_ahead: int = 7) -> dict | None:
    """
    Fetches weather forecast from Open-Meteo for given coordinates.
    Returns parsed forecast dict or None on failure.
    """
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":              lat,
            "longitude":             lon,
            "daily":                 "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "hourly":                "temperature_2m,precipitation_probability,weathercode",
            "timezone":              "auto",
            "forecast_days":         days_ahead,
            "wind_speed_unit":       "kmh",
            "temperature_unit":      "celsius",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        log.warning(f"Weather fetch failed: {e}")
        return None


def format_weather_for_context(circuit_name: str,
                                location_str: str,
                                data: dict,
                                target_date: str = "") -> str:
    """
    Formats weather data into a readable summary for Claude to use.
    target_date: YYYY-MM-DD string for race day focus
    """
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    codes = daily.get("weathercode", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    rain  = daily.get("precipitation_sum", [])
    wind  = daily.get("windspeed_10m_max", [])

    if not dates:
        return ""

    lines = [f"Weather forecast for {location_str} ({circuit_name.title()} circuit):"]

    for i, date in enumerate(dates[:7]):
        code     = codes[i] if i < len(codes) else 0
        desc, em = WMO_CODES.get(code, ("Unknown", "🌡"))
        mx       = t_max[i] if i < len(t_max) else "?"
        mn       = t_min[i] if i < len(t_min) else "?"
        rn       = rain[i]  if i < len(rain)  else 0
        wd       = wind[i]  if i < len(wind)  else "?"

        # Highlight the target date
        prefix = ">>> RACE DAY: " if target_date and date == target_date else ""
        lines.append(
            f"{prefix}{date}: {em} {desc} | "
            f"Max:{mx}°C Min:{mn}°C | "
            f"Rain:{rn}mm | Wind:{wd}km/h"
        )

    return "\n".join(lines)


def get_weather_context(query: str, next_race: dict | None = None) -> str:
    """
    Main weather function — detects circuit from query or uses next race,
    fetches forecast, returns formatted string for Claude.
    """
    # Try to match circuit from query
    match = _match_circuit(query)

    # Fall back to next race circuit if no match in query
    if not match and next_race:
        circuit_name = next_race.get("Circuit", {}).get(
            "Location", {}).get("locality", "")
        country = next_race.get("Circuit", {}).get(
            "Location", {}).get("country", "")
        full    = f"{circuit_name} {country}".lower()
        match   = _match_circuit(full)

    if not match:
        return ""

    circuit_key, (lat, lon, location_str) = match

    # Get race date for highlighting if available
    race_date = ""
    if next_race and next_race.get("date"):
        race_date = next_race["date"]

    data = fetch_weather(lat, lon)
    if not data:
        return ""

    return format_weather_for_context(
        circuit_key, location_str, data, race_date)
def _format_debug_context_report(query: str, ctx: dict) -> str:
    """Formats _gather_context() output for /debug_context — labels, char counts, previews."""
    BLOCK_SPECS = [
        ("next_race_ctx",      "NEXT_RACE",              None),
        ("news_ctx",           "NEWS",                   300),
        ("live_search_ctx",    "LIVE_SEARCH",            800),
        ("fia_docs_ctx",       "FIA_DOCS",               1500),
        ("upgrade_ctx",        "TECHNICAL_UPDATES",       600),
        ("weather_ctx",        "WEATHER",                150),
        ("live_ctx",           "LIVE_SESSION",           200),
        ("practice_ctx",       "SESSION_DATA",           800),
        ("circuit_ctx",        "CIRCUIT",                1500),
        ("driver_stats_ctx",   "DRIVER_STATS",           300),
        ("driver_deep_ctx",    "DRIVER_PROFILE",         1500),
        ("race_replay_ctx",    "RACE_REPLAY",            800),
        ("champ_scenario_ctx", "CHAMPIONSHIP_SCENARIOS", 1200),
        ("fan_ctx",            "FAN_PROFILE",            400),
        ("historical_ctx",     "HISTORY",                1200),
        ("pred_accuracy",      "PREDICTION_RECORD",      150),
        ("user_profile_ctx",   "USER",                   150),
    ]
    lines = [f"debug_context for: {query!r}\n"]
    active = 0
    for key, label, limit in BLOCK_SPECS:
        raw = ctx.get(key, "")
        if not raw:
            continue
        active += 1
        content, meta = _unpack_ctx(raw)
        sliced = content[:limit] if limit else content
        preview = sliced[:100].replace("\n", " ")
        lines.append(f"{label}{meta}: {len(sliced)} chars | {preview!r}")
    if not active:
        lines.append("(no context blocks triggered)")
    return "\n".join(lines)
DRIVER_CODE_MAP = {
    # Mercedes
    # "ant" removed - common English word (collides with "an ant")
    "antonelli": "ANT", "kimi": "ANT", "el niño": "ANT",
    "russell": "RUS", "george": "RUS", "rus": "RUS", "mr saturday": "RUS",
    # Ferrari
    "leclerc": "LEC", "charles": "LEC", "lec": "LEC", "sharl": "LEC",
    "hamilton": "HAM", "lewis": "HAM", "ham": "HAM", "sir lewis": "HAM",
    # Red Bull
    # "max" and "ver" removed - "max" is a common English word
    "verstappen": "VER", "super max": "VER", "ver": "VER",
    # "had" removed - common English word (collides with "I had...")
    "hadjar": "HAD", "isack": "HAD",
    # McLaren
    "norris": "NOR", "lando": "NOR", "nor": "NOR",
    "piastri": "PIA", "oscar": "PIA", "pia": "PIA",
    # Aston Martin
    "alonso": "ALO", "fernando": "ALO", "alo": "ALO", "nano": "ALO",
    "stroll": "STR", "lance": "STR", "str": "STR",
    # Alpine
    # "gas" removed - common English word (collides with "fill up gas")
    "gasly": "GAS", "pierre": "GAS",
    "colapinto": "COL", "franco": "COL", "col": "COL", "el pibe": "COL",
    # Williams
    "albon": "ALB", "alex": "ALB", "alb": "ALB",
    "sainz": "SAI", "carlos": "SAI", "sai": "SAI", "carlitos": "SAI",
    # RB
    # "law" removed - common English word (collides with "the law says")
    "lawson": "LAW", "liam": "LAW",
    "lindblad": "LIN", "arvid": "LIN", "lin": "LIN",
    # Haas
    "bearman": "BEA", "oliver": "BEA", "bea": "BEA",
    "ocon": "OCO", "esteban": "OCO", "oco": "OCO",
    # Audi
    "hulkenberg": "HUL", "nico": "HUL", "hul": "HUL", "el hulk": "HUL",
    "bortoleto": "BOR", "gabriel": "BOR", "bor": "BOR",
    # Cadillac
    # "per" removed - common English word (collides with "per capita", "stopper")
    "perez": "PER", "checo": "PER", "sergio": "PER",
    # "bot" removed - common English word (collides with "this is a bot")
    "bottas": "BOT", "valtteri": "BOT",
}

DRIVER_FULL_NAMES = {
    "ANT": "Andrea Kimi Antonelli", "RUS": "George Russell",
    "HAM": "Lewis Hamilton",        "LEC": "Charles Leclerc",
    "VER": "Max Verstappen",        "HAD": "Isack Hadjar",
    "NOR": "Lando Norris",          "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso",       "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",          "COL": "Franco Colapinto",
    "ALB": "Alexander Albon",       "SAI": "Carlos Sainz",
    "LAW": "Liam Lawson",           "LIN": "Arvid Lindblad",
    "BEA": "Oliver Bearman",        "OCO": "Esteban Ocon",
    "HUL": "Nico Hülkenberg",       "BOR": "Gabriel Bortoleto",
    "PER": "Sergio Pérez",          "BOT": "Valtteri Bottas",
}

DRIVER_TEAMS = {
    "ANT": "Mercedes",  "RUS": "Mercedes",
    "HAM": "Ferrari",   "LEC": "Ferrari",
    "VER": "Red Bull",  "HAD": "Red Bull",
    "NOR": "McLaren",   "PIA": "McLaren",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "GAS": "Alpine",    "COL": "Alpine",
    "ALB": "Williams",  "SAI": "Williams",
    "LAW": "RB",        "LIN": "RB",
    "BEA": "Haas",      "OCO": "Haas",
    "HUL": "Audi",      "BOR": "Audi",
    "PER": "Cadillac",  "BOT": "Cadillac",
}


def _resolve_multiple_driver_codes(text: str) -> list[str]:
    """
    Resolves ALL distinct driver codes mentioned in a query —
    used for "compare X and Y", "X vs Y", "gap between X and Y".

    resolve_driver_code() only returns one match; this finds every
    driver name/nickname present (longest keys first, so full names
    match before short codes), de-duplicated, in order of appearance.
    """
    t = text.lower().strip()
    found: list[tuple[int, str]] = []  # (position, code)
    seen_codes: set[str] = set()
    for key in sorted(DRIVER_CODE_MAP.keys(), key=len, reverse=True):
        code = DRIVER_CODE_MAP[key]
        if code in seen_codes:
            continue
        m = re.search(rf"\b{re.escape(key)}\b", t)
        if m:
            found.append((m.start(), code))
            seen_codes.add(code)
    found.sort(key=lambda x: x[0])
    return [code for _, code in found]


def resolve_driver_code(text: str) -> str | None:
    """
    Resolves any driver name/nickname to a 3-letter code.

    Uses word-boundary matching — DRIVER_CODE_MAP contains short
    3-letter codes ("ver", "ham", "per", "str", "nor", "had", "gas",
    "law", "bot", "ant", "rus", etc.) which are also substrings of
    common English words: "driver" contains "ver", "champion"
    contains "ham", "stopper" contains "per", "strategy" contains
    "str", "had" is a common word itself. Raw substring matching
    caused "championship standings" to resolve to HAM, "is Barcelona
    a 1 or 2 stopper" to resolve to PER, etc. — silently injecting
    irrelevant driver-specific context into unrelated queries.

    Longer keys (full names, nicknames) are checked first so
    "max verstappen" resolves via "verstappen" before any short
    code could match.
    """
    t = text.lower().strip()
    if t in DRIVER_CODE_MAP:
        return DRIVER_CODE_MAP[t]
    # Check longest keys first (full names/nicknames before 3-letter codes)
    for key in sorted(DRIVER_CODE_MAP.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", t):
            return DRIVER_CODE_MAP[key]
    return None


def build_driver_profile(code: str, mem: dict) -> str:
    """
    Builds a rich driver profile from memory + season data.
    Used to give Claude rich context for deep dive responses.
    """
    full_name = DRIVER_FULL_NAMES.get(code, code)
    team      = DRIVER_TEAMS.get(code, "?")
    episodes  = mem.get("episodic", [])

    wins   = [e for e in episodes if e.get("winner") == code]
    p2s    = [e for e in episodes if e.get("p2") == code]
    p3s    = [e for e in episodes if e.get("p3") == code]
    dnfs   = [e for e in episodes if any(code in d for d in e.get("dnfs", []))]
    poles  = [e for e in episodes if e.get("qualifying", {}).get("pole") == code]
    fl     = [e for e in episodes if e.get("fastest_lap") == code]
    sprints= [e for e in episodes if e.get("sprint", {}).get("sprint_winner") == code]

    # Points from standings
    drivers, _ = fetch_standings()
    pts   = "?"
    pos   = "?"
    for s in drivers:
        d = s.get("Driver", {})
        if d.get("code") == code:
            pts = s.get("points", "?")
            pos = s.get("position", "?")
            break

    lines = [
        f"DRIVER PROFILE: {full_name} ({code}) — {team}",
        f"Championship: P{pos} with {pts}pts",
        f"2026 Season: {len(wins)}W {len(p2s)+len(p3s)} podiums "
        f"{len(poles)} poles {len(fl)} fastest laps",
    ]

    if wins:
        win_tracks = [e.get("race_name", e.get("track","?")) for e in wins]
        lines.append(f"Wins: {', '.join(win_tracks)}")

    if poles:
        pole_tracks = [e.get("race_name", e.get("track","?")) for e in poles]
        pole_times  = [e.get("qualifying",{}).get("pole_time","") for e in poles]
        lines.append(f"Poles: {', '.join(f'{t} ({tm})' for t,tm in zip(pole_tracks,pole_times) if t)}")

    if dnfs:
        dnf_details = []
        for e in dnfs:
            for d in e.get("dnfs", []):
                if code in d:
                    dnf_details.append(f"{e.get('race_name','?')}: {d}")
        lines.append(f"DNFs: {', '.join(dnf_details)}")

    if sprints:
        sprint_tracks = [e.get("race_name","?") for e in sprints]
        lines.append(f"Sprint wins: {', '.join(sprint_tracks)}")

    # Race by race results
    results_by_round = []
    for e in sorted(episodes, key=lambda x: x.get("round", 0)):
        if e.get("winner") == code:       pos_str = "P1 🏆"
        elif e.get("p2") == code:         pos_str = "P2"
        elif e.get("p3") == code:         pos_str = "P3"
        elif any(code in d for d in e.get("dnfs", [])): pos_str = "DNF"
        else:                             pos_str = "?"
        fc = e.get("full_classification", [])
        for item in fc:
            if f":{code}" in item:
                pos_str = item.split(":")[0]
                break
        results_by_round.append(
            f"R{e.get('round','?')} {e.get('track','?')}: {pos_str}")
    if results_by_round:
        lines.append(f"Race results: {' | '.join(results_by_round)}")

    return "\n".join(lines)
def _is_driver_deep_dive(text: str) -> str | None:
    """
    Detects if a message is asking for a driver deep dive.
    Returns driver code or None.
    Triggers on: 'tell me about X', 'everything about X',
    'driver profile X', 'who is X', 'stats for X'
    """
    t = text.lower()
    deep_dive_triggers = [
        "tell me about", "everything about", "driver profile",
        "who is", "stats for", "profile of", "cuéntame sobre",
        "todo sobre", "perfil de", "quién es",
        "how is", "how has", "season of", "temporada de",
    ]
    if not any(trigger in t for trigger in deep_dive_triggers):
        return None
    return resolve_driver_code(t)


# ═════════════════════════════════════════════════════════════
#  FEATURE 4 — RACE REPLAY INTELLIGENCE
#  Rich context injection for "why did X happen" questions
# ═════════════════════════════════════════════════════════════

def get_race_replay_context(query: str, mem: dict):
    """
    Detects race replay questions and injects deep episode context.
    Covers: why retirements happened, how strategies unfolded,
    specific incidents, lap-by-lap narrative.
    """
    t = query.lower()

    replay_triggers = [
        "why did", "how did", "what happened", "walk me through",
        "explain", "tell me about the race", "the undercut", "the overtake",
        "the crash", "the incident", "retire", "dnf", "safety car",
        "strategy", "pit stop", "por qué", "cómo fue", "qué pasó",
        "explícame", "cuéntame la carrera",
    ]
    if not any(tr in t for tr in replay_triggers):
        return ""

    episodes = mem.get("episodic", [])
    if not episodes:
        return ""

    # Find most relevant episode
    best_ep    = None
    best_score = 0

    for ep in episodes:
        score = 0
        track    = ep.get("track", "").lower()
        racename = ep.get("race_name", "").lower()
        name_match = (
            track in t
            or any(w in t for w in racename.split() if len(w) > 4)
            or any(
                kw in t and (rw.lower() in track or rw.lower() in racename)
                for kw, rw in RACE_KEYWORDS.items()
            )
        )
        if name_match:
            score += 5
        for driver in [ep.get("winner",""), ep.get("p2",""), ep.get("p3","")]:
            if driver.lower() in t:
                score += 3
        for dnf in ep.get("dnfs", []):
            code = dnf.split("(")[0].strip().lower()
            if code in t:
                score += 4
        if ep.get("sc_count", 0) and ("safety car" in t or "sc" in t):
            score += 3
        if ep.get("sprint", {}).get("sprint_winner") and "sprint" in t:
            score += 3
        if score > best_score:
            best_score = score
            best_ep    = ep

    # If the query names a specific circuit/race that ISN'T in memory yet,
    # don't fall back to a different (most recent) race — that would
    # confidently describe the wrong Grand Prix. Return empty instead so
    # the live-search / session-intercept path can handle it honestly.
    # Uses RACE_KEYWORDS so normalization ("spain" → "Spanish") is applied
    # consistently with the FIA-docs and scoring paths above.
    if best_score == 0:
        for kw, race_word in RACE_KEYWORDS.items():
            if kw in t:
                rw = race_word.lower()
                in_memory = any(
                    rw in ep.get("track","").lower()
                    or rw in ep.get("race_name","").lower()
                    for ep in episodes)
                if not in_memory:
                    log.info(f"Race replay: '{kw}' → '{race_word}' mentioned "
                             f"but not in memory yet — returning empty")
                    return ""
                break

    # Default to most recent race
    if not best_ep and episodes:
        best_ep = sorted(episodes, key=lambda x: x.get("round", 0))[-1]

    if not best_ep:
        return ""

    ep      = best_ep
    quali   = ep.get("qualifying", {})
    sprint  = ep.get("sprint", {})
    pit     = ep.get("pitstops", {})
    sc      = ep.get("sc_periods", [])
    pen     = ep.get("penalties", [])
    dnfs    = ep.get("dnfs", [])
    fc      = ep.get("full_classification", [])
    champ   = ep.get("champ_after", "")
    notes   = ep.get("agent_notes", "")
    briefing= ep.get("briefing", "")

    context_parts = [
        f"RACE REPLAY DATA — R{ep.get('round','?')} {ep.get('race_name', ep.get('track','?'))} "
        f"({ep.get('date','')}):",
        f"Winner: {ep.get('winner','?')} | P2: {ep.get('p2','?')} | P3: {ep.get('p3','?')}",
        f"Fastest lap: {ep.get('fastest_lap','')} {ep.get('fastest_lap_time','')}",
    ]

    if dnfs:
        context_parts.append(f"DNFs with causes: {', '.join(dnfs)}")

    if sc:
        sc_str = " | ".join(
            f"{s.get('type','SC')} laps {s.get('lap_in','?')}-{s.get('lap_out','?')}"
            for s in sc)
        context_parts.append(f"Safety cars: {sc_str}")

    if pit.get("tyre_strategies"):
        context_parts.append(f"Strategy: {pit['tyre_strategies']}")
        if pit.get("fastest_stop_driver"):
            context_parts.append(
                f"Fastest pit: {pit['fastest_stop_driver']} {pit.get('fastest_stop_time','')}s")

    if quali.get("pole"):
        context_parts.append(
            f"Pole: {quali['pole']} ({quali.get('pole_time','')}) | "
            f"Grid top5: {' '.join(fc[:5]) if fc else quali.get('top10','')[:60]}")

    if sprint.get("sprint_winner"):
        context_parts.append(
            f"Sprint: {sprint['sprint_winner']} won ({sprint.get('sprint_top3','')})")

    if pen:
        context_parts.append(f"Penalties: {'; '.join(pen[:3])}")

    if fc:
        context_parts.append(f"Full classification: {' '.join(fc[:10])}")

    if champ:
        context_parts.append(f"Championship after: {champ}")

    if notes:
        context_parts.append(f"Key notes: {notes[:200]}")

    if briefing:
        context_parts.append(f"Post-race analysis: {briefing[:400]}")

    context_parts.append(
        f"Story: {ep.get('story', '')}")

    try:
        race_dt = datetime.strptime(ep.get("date", ""), "%Y-%m-%d")
        replay_age_h = (datetime.now() - race_dt).total_seconds() / 3600
    except Exception:
        replay_age_h = None
    replay_completeness = (
        "full" if ep.get("telemetry_source", "") == "fastf1" else "partial"
    )
    return ContextBlock(
        content="\n".join(context_parts),
        data_age_hours=replay_age_h,
        completeness=replay_completeness,
    )
def _is_championship_scenario(text: str) -> bool:
    """Detects championship scenario questions."""
    t = text.lower()
    triggers = [
        "what does", "what do", "can", "mathematically",
        "championship", "title", "win the championship",
        "clinch", "extend his lead", "close the gap",
        "campeonato", "título", "qué necesita", "puede ganar",
        "matemáticamente", "puntos necesita",
        "scenarios", "scenario", "escenario",
        "how many points", "cuántos puntos",
    ]
    scenario_words = [
        "championship", "title", "campeonato", "título",
        "lead", "gap", "puntos", "points needed",
        "clinch", "mathematically",
    ]
    has_trigger  = any(tr in t for tr in triggers)
    has_scenario = any(sw in t for sw in scenario_words)
    return has_trigger and has_scenario


def build_championship_scenarios(query: str, mem: dict) -> str:
    """
    Fetches live standings and builds championship math scenarios.
    Covers: points gaps, races remaining, what each driver needs.
    """
    drivers, _ = fetch_standings()
    if not drivers or len(drivers) < 3:
        return ""

    episodes       = mem.get("episodic", [])
    races_done     = len(episodes)
    total_races    = 22  # 2026 season: verified from MEETING_KEY_2026 (R1–R22)
    races_left     = total_races - races_done
    max_pts_left   = races_left * 26  # 25 for win + 1 for fastest lap

    leader     = drivers[0]
    leader_d   = leader.get("Driver", {})
    leader_code= leader_d.get("code", "?")
    leader_name= f"{leader_d.get('givenName','')} {leader_d.get('familyName','')}".strip()
    leader_pts = float(leader.get("points", 0))

    lines = [
        f"CHAMPIONSHIP SCENARIOS — Live from Jolpica API",
        f"Races completed: {races_done}/{total_races} | Races remaining: {races_left}",
        f"Max points still available: {max_pts_left}",
        "",
        f"CURRENT STANDINGS:",
    ]

    gaps = []
    for i, s in enumerate(drivers[:8]):
        d    = s.get("Driver", {})
        code = d.get("code", "?")
        name = f"{d.get('givenName','')} {d.get('familyName','')}".strip()
        pts  = float(s.get("points", 0))
        gap  = int(leader_pts - pts)
        wins = s.get("wins", "0")

        if gap == 0:
            status = "LEADER 🔥"
        elif gap > max_pts_left:
            status = f"MATHEMATICALLY ELIMINATED ❌"
        elif gap <= races_left * 8:
            status = f"-{gap}pts (ALIVE ✅)"
        else:
            status = f"-{gap}pts (needs miracle ⚠️)"

        lines.append(f"  P{i+1}. {name} ({code}): {int(pts)}pts {wins}W — {status}")
        gaps.append((code, name, pts, gap))

    lines.append("")
    lines.append("SCENARIO ANALYSIS:")

    # For leader — what to clinch
    if races_left > 0:
        # Clinch if gap > max remaining for P2
        p2_pts = float(drivers[1].get("points", 0))
        p2_code= drivers[1].get("Driver", {}).get("code","?")
        p2_gap = int(leader_pts - p2_pts)
        clinch_gap = races_left * 26
        if p2_gap > clinch_gap:
            lines.append(f"  ✅ {leader_name} has already clinched the title!")
        else:
            pts_to_clinch = clinch_gap - p2_gap + 1
            races_to_clinch = max(1, pts_to_clinch // 26)
            lines.append(
                f"  🏆 {leader_name} clinches if {p2_code} scores 0 and "
                f"{leader_name} wins next {min(races_to_clinch, races_left)} races")
            lines.append(
                f"  📊 P2 ({p2_code}) needs {p2_gap} points swing in {races_left} races to catch up")

    # Detect who's being asked about
    t = query.lower()
    for code, name, pts, gap in gaps:
        if code.lower() in t or any(n.lower() in t
                                     for n in name.lower().split()):
            if gap == 0:
                lines.append(f"\n  {name}: Already leads — needs to protect {int(pts)}pts")
            elif gap > max_pts_left:
                lines.append(f"\n  {name}: Mathematically out — {gap}pts behind with "
                             f"only {max_pts_left} available")
            else:
                wins_needed = gap // 25 + 1
                lines.append(
                    f"\n  {name}: Needs to find {gap}pts in {races_left} races")
                lines.append(
                    f"  Best case: {leader_name} scores 0, {name} wins all "
                    f"{races_left} remaining = gap closed")
                lines.append(
                    f"  Realistic: needs {leader_name} to have bad run AND "
                    f"{name} to win {min(wins_needed, races_left)}+ races")
            break

    return "\n".join(lines)
TEAM_KEYWORDS = {
    "mercedes":     ["mercedes", "merc", "w16", "silver arrows", "petronas"],
    "ferrari":      ["ferrari", "sf-25", "scuderia", "tifosi", "maranello", "roja"],
    "red bull":     ["red bull", "redbull", "rb21", "oracle red bull"],
    "mclaren":      ["mclaren", "mcl39", "papaya", "woking"],
    "aston martin": ["aston martin", "aston", "amr25", "green car"],
    "alpine":       ["alpine", "a525", "renault", "french team"],
    "williams":     ["williams", "grove", "fw47"],
    "rb":           ["rb", "racing bulls", "vcarb", "faenza"],
    "haas":         ["haas", "vf-25", "american team"],
    "audi":         ["audi", "sauber", "swiss team"],
    "cadillac":     ["cadillac", "andretti", "american team cadillac"],
}

FAN_DECLARATION_TRIGGERS = [
    "i'm a", "i am a", "i support", "i follow", "my team is",
    "my favorite", "my favourite", "i root for", "i love",
    "soy fan", "soy de", "mi equipo", "mi favorito",
    "apoyo a", "sigo a", "me gusta",
]

DRIVER_FAN_TRIGGERS = [
    "my driver", "i love", "my favorite driver", "favourite driver",
    "i follow", "i support", "mi piloto", "mi favorito",
]


def detect_fan_declaration(text: str) -> tuple[str | None, str | None]:
    """
    Detects if user is declaring team/driver fandom.
    Returns (team, driver_code) — either can be None.
    """
    t = text.lower()

    # Check for declaration trigger
    has_trigger = any(tr in t for tr in FAN_DECLARATION_TRIGGERS)
    if not has_trigger:
        return None, None

    # Detect team
    detected_team = None
    for team, keywords in TEAM_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            detected_team = team
            break

    # Detect driver
    detected_driver = None
    for name, code in DRIVER_CODE_MAP.items():
        if name in t and len(name) > 3:
            detected_driver = code
            break

    return detected_team, detected_driver


def build_fan_context(user_data: dict, query: str) -> str:
    """
    Builds personalization context based on user's fan preferences.
    Injected into system prompt to personalize responses.
    """
    fan_team   = user_data.get("fan_team", "")
    fan_driver = user_data.get("fan_driver", "")

    if not fan_team and not fan_driver:
        return ""

    parts = ["USER FAN PROFILE (personalize responses around this):"]

    if fan_team:
        team_drivers = [code for code, team in DRIVER_TEAMS.items()
                        if team.lower() == fan_team.lower()]
        parts.append(f"Supports: {fan_team.title()} "
                     f"({' + '.join(team_drivers)})")
        parts.append(
            f"Lead with {fan_team.title()} news and results. "
            f"Frame championship around their team's position.")

    if fan_driver:
        full_name = DRIVER_FULL_NAMES.get(fan_driver, fan_driver)
        team      = DRIVER_TEAMS.get(fan_driver, "")
        parts.append(f"Favourite driver: {full_name} ({fan_driver}) — {team}")
        parts.append(
            f"Always mention {fan_driver}'s result and performance. "
            f"Build narrative around their championship journey.")

    return "\n".join(parts)
