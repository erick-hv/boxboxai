"""Security, rate-limiting, and moderation for BoxBoxAI."""
import logging
import random
import re
import time

log = logging.getLogger(__name__)

# ── Bot owner — alerts go only here ──────────────────────────
BOT_OWNER_ID = "8892063151"

OFF_TOPIC_RESPONSES_EN = [
    "Staying in my lane! 🏎 I'm BoxBoxAI — I only talk F1. For that kind of question try ChatGPT or Google. Now, want to talk about the Monaco GP this weekend? 🇲🇨",
    "That's outside my pit box! 🔧 I'm built exclusively for F1 — races, drivers, strategy, predictions. Ask me anything about the 2026 season instead! 🏆",
    "I only know racing! 🏁 BoxBoxAI is your F1 expert, not a general assistant. Try ChatGPT for that. But if you want to debate whether Antonelli is the best rookie ever, I'm ready 🔥",
]

OFF_TOPIC_RESPONSES_ES = [
    "¡Me quedo en mi carril! 🏎 Soy BoxBoxAI, solo hablo de F1. Para eso mejor usa ChatGPT. Pero si quieres hablar de Mónaco este fin de semana, aquí estoy 🇲🇨",
    "¡Eso está fuera de mi box! 🔧 Solo entiendo de F1 — carreras, pilotos, estrategia, predicciones. ¡Pregúntame algo de la temporada 2026! 🏆",
    "¡Solo sé de carreras! 🏁 BoxBoxAI es tu experto en F1, no un asistente general. Para eso usa ChatGPT. Pero si quieres debatir si el Checo puede ganar en Mónaco, ¡adelante! 🔥",
]

# Keywords that strongly suggest off-topic questions
OFF_TOPIC_KEYWORDS = [
    # Homework / school
    "write my essay", "write an essay", "write my", "my essay",
    "for school", "my homework", "my assignment",
    "write code for", "help me code", "code for me",
    "recipe for", "give me a recipe", "how to cook",
    "pasta recipe", "carbonara", "ingredients for",
    "cover letter", "job application", "resume template",
    "math problem", "solve this equation", "algebra",
    "help me code", "help me with code", "write code for me",
    "write a program", "write a script",
    "my homework", "my assignment", "for school",
    "homework", "tarea", "essay", "ensayo", "thesis", "tesis",
    "assignment", "trabajo escolar", "examen", "exam", "school",
    "escuela", "university", "universidad", "college", "teacher",
    "profesor", "classroom", "aula", "subject", "materia",
    "math", "matemáticas", "history", "historia", "biology",
    "biología", "chemistry", "química", "physics", "física",
    "literature", "literatura", "geography", "geografía",
    # Work tasks
    "write my", "escríbeme", "write an email", "escribe un correo",
    "cover letter", "carta de presentación", "resume", "curriculum",
    "business plan", "plan de negocios", "presentation", "presentación",
    "spreadsheet", "excel formula", "code for me", "write code",
    "debug my", "fix my code", "javascript", "python tutorial",
    # General non-F1
    "recipe", "receta", "cooking", "cocina", "how to cook",
    "medical advice", "consejo médico", "symptoms", "síntomas",
    "legal advice", "consejo legal", "lawyer", "abogado",
    "translate", "traducir", "translate this",
    "what is the capital", "cuál es la capital",
    "who invented", "quién inventó",
    "movie recommendation", "recomendación de película",
    "song lyrics", "letra de canción",
    "stock price", "precio de acción",
    "crypto", "bitcoin", "investment advice",
    "travel advice", "consejo de viaje",
    "hotel recommendation", "restaurante",
    "workout", "ejercicio rutina", "diet plan", "dieta",
]

# F1-adjacent keywords that should always pass through
F1_SAFE_KEYWORDS = [
    "f1", "formula", "grand prix", "gp", "race", "carrera",
    "driver", "piloto", "team", "equipo", "circuit", "circuito",
    "championship", "campeonato", "qualifying", "clasificación",
    "pit stop", "tyre", "llanta", "strategy", "estrategia",
    "lap", "vuelta", "podium", "podio", "overtake", "adelantar",
    "safety car", "drs", "straight mode", "overtake mode", "fia", "ferrari", "mercedes", "red bull",
    "mclaren", "aston", "alpine", "williams", "haas", "audi",
    "cadillac", "rb", "checo", "hamilton", "verstappen", "antonelli",
    "leclerc", "russell", "norris", "alonso", "sainz", "bottas",
    "perez", "pérez", "gasly", "colapinto", "lawson", "hadjar",
    "bearman", "ocon", "hulkenberg", "bortoleto", "albon",
    "monaco", "monza", "silverstone", "spa", "suzuka", "melbourne",
    "bahrain", "jeddah", "miami", "imola", "montreal", "barcelona",
    "spielberg", "budapest", "zandvoort", "baku", "singapore",
    "austin", "mexico", "são paulo", "las vegas", "lusail", "abu dhabi",
    "weather", "clima", "rain", "lluvia", "prediction", "predicción",
    "news", "noticias", "standings", "clasificación general",
    "viejo sabroso", "magic", "smooth operator", "super max",
    "el nano", "checo", "checote",
]


def is_off_topic(text: str) -> bool:
    """
    Returns True if the message is clearly not F1-related.
    Uses a two-pass check:
    1. If any F1 keyword present → always allow
    2. If strong off-topic keyword present → block
    """
    t = text.lower()

    # Always allow if F1 content detected
    # Use word-boundary matching to avoid "rb" matching "carbonara"
    import re as _re
    if any(_re.search(r'\b' + _re.escape(kw) + r'\b', t)
           for kw in F1_SAFE_KEYWORDS if len(kw) > 3):
        return False
    # Also check short exact matches
    if any(kw == t or f" {kw} " in f" {t} "
           for kw in F1_SAFE_KEYWORDS if len(kw) <= 3):
        return False

    # Block if off-topic keyword detected
    if any(kw in t for kw in OFF_TOPIC_KEYWORDS):
        return True

    # Short greetings always allowed
    if len(text.strip().split()) <= 4:
        return False

    return False


def _detect_language(text: str) -> str:
    """Returns 'es' for Spanish, 'en' for English based on content."""
    spanish_indicators = [
        "qué", "que ", "cómo", "como ", "quién", "quien", "por favor",
        "puedes", "puede", "tienes", "tiene", "eres", "hacer", "haz",
        "dime", "dame", "estás", "esta ", "este ", "ese ", "esa ",
        "ignora", "actúa", "actua", "revela", "instrucción", "instruccion",
        "sin restricciones", "olvida", "nueva ", "nuevo ",
        "tarea", "escuela", "ayuda", "necesito", "quiero", "soy ",
        "del ", "las ", "los ", "una ", "unos ", "para ", "pero ",
    ]
    t = text.lower()
    score = sum(1 for w in spanish_indicators if w in t)
    return "es" if score >= 1 else "en"


def get_off_topic_response(text: str) -> str:
    """Returns off-topic redirect in the same language as the message."""
    lang = _detect_language(text)
    responses = OFF_TOPIC_RESPONSES_ES if lang == "es" else OFF_TOPIC_RESPONSES_EN
    return random.choice(responses)


# ═════════════════════════════════════════════════════════════
#  SECURITY LAYER
# ═════════════════════════════════════════════════════════════

# Rate limiting: max messages per user per window
RATE_LIMIT_MAX      = 20    # max messages
RATE_LIMIT_WINDOW   = 3600  # per hour (seconds)
RATE_LIMIT_WARN_AT  = 15    # warn user when they hit this

# Abuse tracking: auto-ban after repeated violations
ABUSE_STRIKE_LIMIT  = 5     # strikes before temp ban
ABUSE_BAN_DURATION  = 3600  # 1 hour ban (seconds)

# Message limits
MAX_MESSAGE_LENGTH  = 1000  # chars — ignore huge pastes
MAX_HISTORY_STORE   = 20    # messages to keep per user

# Prompt injection patterns — attempts to override bot instructions
INJECTION_PATTERNS = [
    r"ignore\s+(your\s+)?(previous\s+)?(instructions|prompt|rules|guidelines)",
    r"forget\s+(your\s+)?(instructions|rules|training)",
    r"forget\s+(everything|all)\s+(you\s+)?(know|learned)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|another|dan|gpt|chatgpt)",
    r"you\s+are\s+now\s+dan",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(different|gpt|chatgpt|openai|unrestricted)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+(rules|restrictions|filter|guidelines)",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(different|unrestricted|jailbroken|gpt|chatgpt)",
    r"pretend\s+you\s+(are|have)\s+no\s+(rules|restrictions)",
    r"now\s+you\s+(are|have)\s+no\s+(rules|restrictions|filter)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"bypass\s+(your\s+)?(filter|restriction|rule|safety|guardrail)",
    r"reveal\s+(your\s+)?(api\s+key|token|secret|password|credentials)",
    r"what\s+is\s+your\s+(api\s+key|token|secret|system\s+prompt)",
    r"show\s+me\s+your\s+(api\s+key|token|secret|system\s+prompt|instructions)",
    r"(reveal|show|print|display)\s+(the\s+)?(system\s+prompt|your\s+instructions|your\s+prompt)",
    r"print\s+your\s+(system\s+prompt|instructions|api\s+key)",
    r"ignore\s+all\s+previous",
    r"nueva\s+instrucción",
    r"ignora\s+(todas\s+)?(tus\s+)?(instrucciones|reglas)",
    r"olvida\s+(todo|tus\s+instrucciones|las\s+reglas)",
    r"actúa\s+como\s+(si\s+fueras\s+)?(otro|diferente|sin\s+restricciones)",
    r"modo\s+desarrollador",
    r"sin\s+restricciones",
    r"revela\s+(tu\s+)?(clave|token|contraseña|api)",
    r"ahora\s+(eres|tienes)\s+(otro|diferente|sin\s+reglas)",
    r"reveal your system",
    r"your system prompt",
    r"show your (system|instructions|prompt)",
    r"what.{0,10}system prompt",
    r"(reveal|show|display|print|tell me)\s+(the\s+)?system\s+prompt",
    r"what('s| is) (in )?your (system )?prompt",
    r"(show|tell|give)\s+me\s+your\s+(instructions|system|prompt)",
]

INJECTION_RESPONSES_EN = [
    "Nice try! 🏎 I'm BoxBoxAI — I only talk F1 and I'm not going anywhere. Ask me about Monaco instead!",
    "That's not how this pit stop works. 🔧 I'm locked in on F1 and nothing you type changes that. What do you want to know about the 2026 season?",
    "Security check passed — still just an F1 bot! 🏁 Ask me something about racing.",
]

INJECTION_RESPONSES_ES = [
    "¡Buen intento! 🏎 Soy BoxBoxAI — solo hablo de F1 y eso no va a cambiar. ¿Qué quieres saber de Mónaco?",
    "Así no funciona este pit stop. 🔧 Estoy bloqueado en F1 y nada de lo que escribas cambia eso. ¿Qué quieres saber de la temporada 2026?",
    "¡Revisión de seguridad superada — sigo siendo solo un bot de F1! 🏁 Pregúntame algo de carreras.",
]

# Per-user rate limit storage (in-memory, resets on restart)
_rate_limits: dict = {}   # {user_id: [timestamp, ...]}
_abuse_strikes: dict = {} # {user_id: {"strikes": int, "banned_until": float}}


def check_rate_limit(user_id: str) -> tuple[bool, str]:
    """
    Checks if a user has exceeded the rate limit.
    Returns (allowed: bool, message: str)
    """
    now      = time.time()
    uid      = str(user_id)
    window   = now - RATE_LIMIT_WINDOW

    # Clean old timestamps
    if uid not in _rate_limits:
        _rate_limits[uid] = []
    _rate_limits[uid] = [t for t in _rate_limits[uid] if t > window]

    count = len(_rate_limits[uid])

    # Check if banned
    ban_info = _abuse_strikes.get(uid, {})
    banned_until = ban_info.get("banned_until", 0)
    if banned_until > now:
        mins = int((banned_until - now) / 60) + 1
        return False, (
            f"⛔ You've been temporarily rate-limited for {mins} more minute(s). "
            f"Come back soon! 🏎"
        )

    # Hard limit
    if count >= RATE_LIMIT_MAX:
        # Add a strike
        if uid not in _abuse_strikes:
            _abuse_strikes[uid] = {"strikes": 0, "banned_until": 0}
        _abuse_strikes[uid]["strikes"] += 1

        if _abuse_strikes[uid]["strikes"] >= ABUSE_STRIKE_LIMIT:
            _abuse_strikes[uid]["banned_until"] = now + ABUSE_BAN_DURATION
            return False, (
                "⛔ Too many messages. You've been rate-limited for 1 hour. "
                "BoxBoxAI is for F1 fans, not bots! 🏎"
            )

        return False, (
            f"⏱ Slow down! You've sent {count} messages this hour. "
            f"Limit is {RATE_LIMIT_MAX}/hour. Try again later! 🏎"
        )

    # Soft warning
    if count >= RATE_LIMIT_WARN_AT:
        _rate_limits[uid].append(now)
        remaining = RATE_LIMIT_MAX - count - 1
        return True, f"⚠️ Heads up — {remaining} messages left this hour."

    _rate_limits[uid].append(now)
    return True, ""


def check_injection(text: str) -> bool:
    """Returns True if text looks like a prompt injection attempt."""
    t = text.lower()
    return any(re.search(pattern, t) for pattern in INJECTION_PATTERNS)


def get_injection_response(text: str) -> str:
    """Returns injection response in the same language as the attempt."""
    lang = _detect_language(text)
    responses = INJECTION_RESPONSES_ES if lang == "es" else INJECTION_RESPONSES_EN
    return random.choice(responses)


def sanitize_message(text: str) -> str:
    """
    Sanitizes user input:
    - Truncates overly long messages
    - Strips null bytes and control characters
    - Normalizes whitespace
    """
    # Strip null bytes and most control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Strip Unicode invisibles that bypass injection regex (zero-width, direction overrides, BOM)
    text = re.sub("[​‌‍‭‮⁠﻿]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH] + "…"
    return text


def log_security_event(user_id: str, user_name: str,
                        event_type: str, detail: str):
    """Logs security events for monitoring."""
    log.warning(
        f"SECURITY [{event_type}] user={user_id} ({user_name}): {detail[:100]}")


async def alert_owner(app, message: str):
    """Sends a private alert only to the bot owner (Erick)."""
    try:
        await app.bot.send_message(
            chat_id=BOT_OWNER_ID,
            text=f"🚨 *BoxBoxAI Alert*\n\n{message}",
            parse_mode="Markdown"
        )
    except Exception as e:
        log.error(f"Failed to alert owner: {e}")
