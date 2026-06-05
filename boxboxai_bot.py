#!/usr/bin/env python3
"""
BoxBoxAI — F1 Race Analyst Telegram Bot
=========================================
Developed by Erick Hernandez

Setup (one time):
  pip3 install python-telegram-bot anthropic requests

Create your bot:
  1. Open Telegram, search @BotFather
  2. Send /newbot
  3. Name: BoxBoxAI
  4. Username: BoxBoxAI_bot  (or whatever is available)
  5. Copy the token BotFather gives you

Run:
  export ANTHROPIC_API_KEY=sk-ant-...
  export TELEGRAM_BOT_TOKEN=your_token_here
  python3 boxboxai_bot.py
"""

import os, sys, json, re, time, logging
from pathlib import Path
from datetime import datetime

# ── dependency check ──────────────────────────────────────────
missing = []
try:
    from telegram import Update, constants
    from telegram.ext import (Application, CommandHandler,
                               MessageHandler, filters, ContextTypes)
except ImportError:
    missing.append("python-telegram-bot")
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

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════
BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL         = "claude-sonnet-4-5"
MAX_TOKENS    = 1000
MEMORY_FILE   = Path(__file__).parent / "f1_memory_2026.json"
SESSIONS_FILE = Path(__file__).parent / "boxboxai_sessions.json"
JOLPICA       = "https://api.jolpi.ca/ergast/f1"
SEASON        = 2026

# Telegram message limit
TG_MAX_CHARS  = 4096

# ═════════════════════════════════════════════════════════════
#  BRANDING
# ═════════════════════════════════════════════════════════════
BANNER = """🏎 *BoxBoxAI* — F1 Race Analyst
_Developed by Erick Hernandez_
━━━━━━━━━━━━━━━━━━━━━━"""

WELCOME = """🏎 *Welcome to BoxBoxAI!*
_Your AI-powered F1 race analyst_
_Developed by Erick Hernandez_

━━━━━━━━━━━━━━━━━━━━━━

I have full memory of the *2026 F1 season* — every race, qualifying session, strategy call, championship battle, and driver story.

Just ask me anything. Some ideas:
• _"Who's leading the championship?"_
• _"Why did Russell retire in Montreal?"_
• _"Break down Antonelli's dominance"_
• _"Who should I bet on for Monaco?"_
• _"Compare Hamilton vs Leclerc this season"_

*Commands:*
/start — this menu
/standings — live championship table
/season — full 2026 race log
/predict — next race prediction
/help — all commands

Let's talk F1 🚀"""

HELP_TEXT = """*BoxBoxAI Commands*
_Developed by Erick Hernandez_
━━━━━━━━━━━━━━━━━━━━━━

/start — welcome & intro
/standings — driver & constructor standings
/season — full 2026 race log
/predict — next race prediction
/lastrace — summary of latest race
/help — this menu

Or just *ask anything* in plain English 💬
I know every race, qualifying result, strategy, incident, and championship stat from the 2026 season."""

# ═════════════════════════════════════════════════════════════
#  MEMORY LOADER
# ═════════════════════════════════════════════════════════════
def load_f1_memory() -> dict:
    """Loads the agent's episodic + semantic memory from file."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    # Fallback: minimal hardcoded memory
    return {
        "semantic": {
            "season/2026_summary": {
                "text": "2026 F1: new aero regs. Antonelli dominant (4 wins). RUS won Australia. ANT leads WDC 131pts, RUS 88pts. RUS DNF Montreal power unit. HAM P2 Montreal. VER first podium Montreal P3.",
            }
        },
        "episodic": [
            {"round":5,"track":"Montreal","winner":"ANT","p2":"HAM","p3":"VER",
             "story":"Antonelli wins after Russell retires (power unit lap 31). Hamilton P2 Ferrari. Verstappen P3.",
             "champ_delta":"ANT +43 over RUS"}
        ]
    }

# ═════════════════════════════════════════════════════════════
#  SESSION MANAGER — per-user conversation history
# ═════════════════════════════════════════════════════════════
def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text())
        except Exception:
            pass
    return {}

def save_sessions(sessions: dict):
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))

def get_user_history(sessions: dict, user_id: str) -> list:
    return sessions.get(user_id, {}).get("history", [])

def update_user_history(sessions: dict, user_id: str,
                         role: str, content: str):
    if user_id not in sessions:
        sessions[user_id] = {"history": [], "first_seen": datetime.now().isoformat()}
    sessions[user_id]["history"].append({"role": role, "content": content})
    sessions[user_id]["last_seen"] = datetime.now().isoformat()
    # Keep last 20 messages per user
    sessions[user_id]["history"] = sessions[user_id]["history"][-20:]

# ═════════════════════════════════════════════════════════════
#  LIVE DATA — quick fetches for commands
# ═════════════════════════════════════════════════════════════
def safe_get(url: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_standings() -> tuple[list, list]:
    """Returns (driver_standings, constructor_standings)."""
    d = safe_get(f"{JOLPICA}/{SEASON}/driverStandings.json")
    c = safe_get(f"{JOLPICA}/{SEASON}/constructorStandings.json")

    drivers = []
    if d:
        lists = d.get("MRData",{}).get("StandingsTable",{}).get("StandingsLists",[])
        if lists:
            drivers = lists[0].get("DriverStandings", [])

    constructors = []
    if c:
        lists = c.get("MRData",{}).get("StandingsTable",{}).get("StandingsLists",[])
        if lists:
            constructors = lists[0].get("ConstructorStandings", [])

    return drivers, constructors

def fetch_next_race() -> dict | None:
    data = safe_get(f"{JOLPICA}/{SEASON}/next.json")
    if not data:
        return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    return races[0] if races else None

def fetch_last_race() -> dict | None:
    data = safe_get(f"{JOLPICA}/{SEASON}/last/results.json")
    if not data:
        return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    return races[0] if races else None

# ═════════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═════════════════════════════════════════════════════════════
def build_system_prompt(mem: dict) -> str:
    # Semantic facts
    sem_lines = []
    for k, v in mem.get("semantic", {}).items():
        text = v["text"] if isinstance(v, dict) else v
        sem_lines.append(f"  [{k}] {text}")

    # Episodic races
    ep_lines = []
    for r in mem.get("episodic", []):
        quali  = r.get("qualifying", {})
        sprint = r.get("sprint", {})
        pit    = r.get("pitstops", {})
        dnfs   = r.get("dnfs", [])
        sc     = r.get("sc_count", 0)
        pen    = r.get("penalties", [])
        fc     = r.get("full_classification", [])

        line = (f"  R{r['round']} {r['track']} {r.get('date','')} | "
                f"Winner:{r['winner']} P2:{r.get('p2','')} P3:{r.get('p3','')} | "
                f"FL:{r.get('fastest_lap','')} {r.get('fastest_lap_time','')} | "
                f"{r.get('champ_delta','')}")
        if quali.get("pole"):
            line += f" | Pole:{quali['pole']} {quali.get('pole_time','')}"
        if sprint.get("sprint_winner"):
            line += f" | Sprint:{sprint['sprint_winner']}"
        if pit.get("strategy_summary"):
            line += f" | Strategy:{pit['strategy_summary']}"
        if dnfs:
            line += f" | DNFs:{','.join(dnfs[:4])}"
        if sc:
            line += f" | SC×{sc}"
        if pen:
            line += f" | Penalties:{';'.join(pen[:2])}"
        if fc:
            line += f" | Classification:{' '.join(fc[:6])}"
        if r.get("agent_notes"):
            line += f" | Notes:{r['agent_notes'][:100]}"
        ep_lines.append(line)

    return f"""You are BoxBoxAI — an expert F1 race analyst with perfect memory of the 2026 season.
Developed by Erick Hernandez.

PERSONALITY:
- Smart but fun — like a knowledgeable mate who loves F1 as much as the person asking
- Confident and direct — give real opinions, not just neutral summaries
- Use F1 terminology naturally but explain when needed
- Keep answers concise for Telegram — 3-5 short paragraphs max unless asked for detail
- Occasional dry humor is fine. Get excited about good racing.
- Never start with "Certainly!" — just answer
- Use *bold* for key names/numbers (Telegram markdown)
- You're on Telegram so keep it punchy and readable on a phone screen

EMOJI USAGE — use these naturally throughout responses:
- 🏎 for Mercedes / general F1 car references
- 🔴 Ferrari  🔵 Red Bull  🟡 McLaren  ⚪ Williams  🟢 Aston Martin
- 🏆 for wins, championships, podiums
- 🥇🥈🥉 for podium positions
- 🔥 for dominant performances or hot streaks
- 💨 for fast laps, speed, pace
- 🚀 for impressive starts or launches
- 🛞 for tyre strategy, pit stops
- 🏁 for race finishes, chequered flag moments
- 🚦 for race starts
- ⚠️ for incidents, DNFs, safety cars
- 🔧 for mechanical failures, reliability issues
- 📊 for stats, standings, numbers
- 🇦🇺🇨🇳🇯🇵🇺🇸🇨🇦🇲🇨🇪🇸🇦🇹🇬🇧🇧🇪🇳🇱🇮🇹🇦🇿🇸🇬🇲🇽🇧🇷🇶🇦🇦🇪 for race country flags
- Use emojis to punctuate key moments, not every sentence — make them feel natural

WHAT YOU KNOW — 2026 Season memory:

FACTS:
{chr(10).join(sem_lines)}

RACE HISTORY:
{chr(10).join(ep_lines)}

Always answer from memory. Be accurate. If you don't know something, say so rather than guess."""


# ═════════════════════════════════════════════════════════════
#  CLAUDE API CALL
# ═════════════════════════════════════════════════════════════
client = None

def get_client():
    global client
    if client is None:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    return client

def ask_claude(user_msg: str, history: list, mem: dict) -> str:
    """Calls Claude with full memory context and conversation history."""
    system = build_system_prompt(mem)
    messages = history + [{"role": "user", "content": user_msg}]

    try:
        resp = get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages[-16:]  # last 8 turns
        )
        return resp.content[0].text if resp.content else "Sorry, no response."
    except anthropic.AuthenticationError:
        return "⚠️ API key issue. Contact @ErickHernandez."
    except anthropic.RateLimitError:
        return "⚠️ Too many requests right now. Try again in a moment!"
    except Exception as e:
        log.error(f"Claude error: {e}")
        return "⚠️ Something went wrong. Try again in a sec."

# ═════════════════════════════════════════════════════════════
#  MESSAGE SPLITTER — Telegram 4096 char limit
# ═════════════════════════════════════════════════════════════
def split_message(text: str, limit: int = TG_MAX_CHARS) -> list[str]:
    """Splits long messages at paragraph boundaries."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        # Find last paragraph break before limit
        cut = text.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return parts

# ═════════════════════════════════════════════════════════════
#  FORMATTED RESPONSES FOR COMMANDS
# ═════════════════════════════════════════════════════════════
def format_standings(drivers: list, constructors: list) -> str:
    if not drivers:
        return "⚠️ Standings unavailable right now. Try again shortly."

    leader_pts = float(drivers[0].get("points", 0))
    lines = [f"🏆 *2026 Championship — Round {SEASON}*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━",
             "", "*Drivers*"]

    medals = ["🥇", "🥈", "🥉"]
    for i, s in enumerate(drivers[:10]):
        drv  = s.get("Driver", {})
        name = f"{drv.get('givenName','')[:1]}. {drv.get('familyName','')}"
        team = s.get("Constructors",[{}])[0].get("name","")[:12]
        pts  = float(s.get("points", 0))
        gap  = "" if i == 0 else f"  _(-{int(leader_pts-pts)})_"
        medal = medals[i] if i < 3 else f"{i+1}."
        lines.append(f"{medal} *{name}* — {int(pts)}pts{gap}")

    if constructors:
        lines += ["", "*Constructors*"]
        c_leader = float(constructors[0].get("points", 0))
        for i, s in enumerate(constructors[:5]):
            name = s.get("Constructor",{}).get("name","")[:16]
            pts  = float(s.get("points", 0))
            gap  = "" if i == 0 else f"  _(-{int(c_leader-pts)})_"
            medal = medals[i] if i < 3 else f"{i+1}."
            lines.append(f"{medal} *{name}* — {int(pts)}pts{gap}")

    return "\n".join(lines)


def format_season(mem: dict) -> str:
    episodes = mem.get("episodic", [])
    if not episodes:
        return "No race data in memory yet."

    flags = {
        "Melbourne":"🇦🇺","Shanghai":"🇨🇳","Suzuka":"🇯🇵","Bahrain":"🇧🇭",
        "Jeddah":"🇸🇦","Miami":"🇺🇸","Imola":"🇮🇹","Monaco":"🇲🇨",
        "Montreal":"🇨🇦","Barcelona":"🇪🇸","Spielberg":"🇦🇹",
        "Silverstone":"🇬🇧","Budapest":"🇭🇺","Spa":"🇧🇪",
        "Zandvoort":"🇳🇱","Monza":"🇮🇹","Baku":"🇦🇿",
        "Singapore":"🇸🇬","Austin":"🇺🇸","Mexico City":"🇲🇽",
        "São Paulo":"🇧🇷","Las Vegas":"🇺🇸","Lusail":"🇶🇦",
        "Abu Dhabi":"🇦🇪",
    }

    lines = [f"📅 *2026 Season — {len(episodes)} races*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━"]

    for r in episodes:
        flag = flags.get(r["track"], "🏁")
        dnfs = r.get("dnfs", [])
        sc   = r.get("sc_count", 0)
        extras = []
        if dnfs:
            extras.append(f"DNF:{','.join(d.split('(')[0] for d in dnfs[:2])}")
        if sc:
            extras.append(f"SC×{sc}")
        extra_str = f"  _{' | '.join(extras)}_" if extras else ""

        lines.append(
            f"\n*R{r['round']}* {flag} {r['track']} `{r.get('date','')}`"
        )
        lines.append(
            f"🏆 {r['winner']}  P2:{r.get('p2','')}  P3:{r.get('p3','')}"
        )
        if r.get("qualifying", {}).get("pole"):
            lines.append(
                f"⏱ Pole: {r['qualifying']['pole']} {r['qualifying'].get('pole_time','')}"
            )
        if r.get("champ_delta"):
            lines.append(f"📊 _{r['champ_delta']}_")
        if extra_str:
            lines.append(extra_str)

    return "\n".join(lines)


def format_last_race(race: dict, mem: dict) -> str:
    if not race:
        # Fall back to memory
        episodes = mem.get("episodic", [])
        if not episodes:
            return "No race data available."
        r = episodes[-1]
        return (f"🏁 *Last race: {r.get('race_name', r['track'])}*\n\n"
                f"{r.get('story','No story available.')}\n\n"
                f"_{r.get('champ_delta','')}_")

    results = race.get("Results", [])[:5]
    name    = race.get("raceName", "Last Race")

    def dname(r):
        d = r.get("Driver", {})
        return f"{d.get('givenName','')[:1]}. {d.get('familyName','')}"

    lines = [f"🏁 *{name}*",
             "_Developed by Erick Hernandez_",
             "━━━━━━━━━━━━━━━━━━━━━━"]

    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    for i, r in enumerate(results):
        status = r.get("status","")
        dnf    = status not in ("Finished","+1 Lap","+2 Laps","+3 Laps")
        note   = f" _{status}_" if dnf else ""
        lines.append(f"{medals[i]} *{dname(r)}*{note}")

    # Fastest lap
    for r in race.get("Results", []):
        if r.get("FastestLap", {}).get("rank") == "1":
            fl_t = r["FastestLap"].get("Time", {}).get("time", "")
            lines.append(f"\n⚡ Fastest lap: *{dname(r)}* {fl_t}")
            break

    return "\n".join(lines)


def format_prediction(next_race: dict, mem: dict) -> str:
    if not next_race:
        return "No upcoming race found in the calendar."

    name    = next_race.get("raceName", "Next Race")
    circuit = next_race.get("Circuit", {}).get("circuitName", "")
    date    = next_race.get("date", "TBD")
    rnd     = next_race.get("round", "?")

    # Get last episodes for context
    episodes = mem.get("episodic", [])
    recent   = episodes[-3:] if len(episodes) >= 3 else episodes
    recent_str = " | ".join(
        f"R{r['round']} {r['track']}→{r['winner']}" for r in recent)

    return (
        f"🎯 *{name} Preview*\n"
        f"_Round {rnd} · {circuit} · {date}_\n"
        f"_Developed by Erick Hernandez_\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Ask me: _\"Who will win {name}?\"_ for a full prediction with reasoning,\n"
        f"or _\"/predict full\"_ for the complete analysis including circuit profile, "
        f"championship stakes, and what could upset the favourite.\n\n"
        f"Recent form: _{recent_str}_"
    )

# ═════════════════════════════════════════════════════════════
#  TELEGRAM HANDLERS
# ═════════════════════════════════════════════════════════════
mem      = {}
sessions = {}

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        WELCOME, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        HELP_TEXT, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_standings(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching live standings... ⏳")
    drivers, constructors = fetch_standings()
    text = format_standings(drivers, constructors)
    await update.message.reply_text(
        text, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_season(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = format_season(mem)
    for part in split_message(text):
        await update.message.reply_text(
            part, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_lastrace(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching latest race... ⏳")
    race = fetch_last_race()
    text = format_last_race(race, mem)
    await update.message.reply_text(
        text, parse_mode=constants.ParseMode.MARKDOWN)

async def cmd_predict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    next_race = fetch_next_race()

    if args and args[0].lower() == "full":
        # Full AI prediction
        race_name = next_race.get("raceName","next race") if next_race else "next race"
        await update.message.reply_text(
            f"Generating full prediction for {race_name}... 🧠")
        user_id = str(update.effective_user.id)
        history = get_user_history(sessions, user_id)
        reply   = ask_claude(
            f"Give me a full race prediction for the {race_name}: "
            f"predicted winner with reasoning, top-5, key factors, "
            f"championship implications, and your confidence level.",
            history, mem
        )
        update_user_history(sessions, user_id, "user",
                            f"Full prediction for {race_name}")
        update_user_history(sessions, user_id, "assistant", reply)
        save_sessions(sessions)
        for part in split_message(reply):
            await update.message.reply_text(
                part, parse_mode=constants.ParseMode.MARKDOWN)
    else:
        text = format_prediction(next_race, mem)
        await update.message.reply_text(
            text, parse_mode=constants.ParseMode.MARKDOWN)

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handles all free-text messages — routes to Claude."""
    user    = update.effective_user
    user_id = str(user.id)
    text    = update.message.text.strip()

    if not text:
        return

    log.info(f"Message from {user.first_name} ({user_id}): {text[:60]}")

    # Show typing indicator
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=constants.ChatAction.TYPING
    )

    history = get_user_history(sessions, user_id)
    reply   = ask_claude(text, history, mem)

    update_user_history(sessions, user_id, "user", text)
    update_user_history(sessions, user_id, "assistant", reply)
    save_sessions(sessions)

    for part in split_message(reply):
        await update.message.reply_text(
            part, parse_mode=constants.ParseMode.MARKDOWN)

async def handle_error(update: object,
                        ctx: ContextTypes.DEFAULT_TYPE):
    log.error(f"Error: {ctx.error}")

# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    global mem, sessions

    # ── Check credentials ─────────────────────────────────
    if not BOT_TOKEN:
        print("\n  TELEGRAM_BOT_TOKEN not set.")
        print("  Get one from @BotFather on Telegram, then:")
        print("  export TELEGRAM_BOT_TOKEN=your_token_here\n")
        sys.exit(1)

    if not ANTHROPIC_KEY:
        print("\n  ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...\n")
        sys.exit(1)

    # ── Load memory ───────────────────────────────────────
    mem      = load_f1_memory()
    sessions = load_sessions()

    ep_count  = len(mem.get("episodic", []))
    sem_count = len(mem.get("semantic", {}))

    print(f"""
╔══════════════════════════════════════════════════════╗
║  🏎  BoxBoxAI — F1 Telegram Bot                      ║
║  Developed by Erick Hernandez                        ║
╚══════════════════════════════════════════════════════╝

  Memory:    {sem_count} semantic entries, {ep_count} race episodes
  Model:     {MODEL}
  Sessions:  {len(sessions)} users tracked
  Status:    Starting bot...
""")

    # ── Build and run app ─────────────────────────────────
    # ── Build and run app ─────────────────────────────────
    from telegram.request import HTTPXRequest

    request         = HTTPXRequest(connection_pool_size=8, http_version="1.1")
    request_updates = HTTPXRequest(connection_pool_size=8, http_version="1.1")

    app = (Application.builder()
           .token(BOT_TOKEN)
           .request(request)
           .get_updates_request(request_updates)
           .build())

    app.add_handler(CommandHandler("start",      cmd_start))
    app.add_handler(CommandHandler("help",       cmd_help))
    app.add_handler(CommandHandler("standings",  cmd_standings))
    app.add_handler(CommandHandler("season",     cmd_season))
    app.add_handler(CommandHandler("lastrace",   cmd_lastrace))
    app.add_handler(CommandHandler("predict",    cmd_predict))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    print("  ✅ BoxBoxAI is LIVE. Open Telegram and message your bot.\n")
    print("  Press Ctrl+C to stop.\n")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
