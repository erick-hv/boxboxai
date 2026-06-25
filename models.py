"""Shared data models for BoxBoxAI context assembly."""
from dataclasses import dataclass

# Increment when the circuit map extraction method changes so stale on-disk
# images from older versions are automatically ignored and re-fetched.
CIRCUIT_MAP_VERSION = 2

RACE_CALENDAR_2026 = [
    {"round":  1, "name": "Australian GP",  "circuit": "Melbourne",          "date": "2026-03-15", "meeting_key": 1279},
    {"round":  2, "name": "Chinese GP",     "circuit": "Shanghai",           "date": "2026-03-22", "meeting_key": 1280},
    {"round":  3, "name": "Japanese GP",    "circuit": "Suzuka",             "date": "2026-04-06", "meeting_key": 1281},
    {"round":  4, "name": "Miami GP",       "circuit": "Miami",              "date": "2026-05-04", "meeting_key": 1284},
    {"round":  5, "name": "Canadian GP",    "circuit": "Montreal",           "date": "2026-05-24", "meeting_key": 1285},
    {"round":  6, "name": "Monaco GP",      "circuit": "Monte Carlo",        "date": "2026-06-07", "meeting_key": 1286},
    {"round":  7, "name": "Spanish GP",     "circuit": "Catalunya",          "date": "2026-06-14", "meeting_key": 1287},
    {"round":  8, "name": "Austrian GP",    "circuit": "Spielberg",          "date": "2026-06-28", "meeting_key": 1288},
    {"round":  9, "name": "British GP",     "circuit": "Silverstone",        "date": "2026-07-05", "meeting_key": 1289},
    {"round": 10, "name": "Belgian GP",     "circuit": "Spa-Francorchamps",  "date": "2026-07-19", "meeting_key": 1290},
    {"round": 11, "name": "Hungarian GP",   "circuit": "Hungaroring",        "date": "2026-07-26", "meeting_key": 1291},
    {"round": 12, "name": "Dutch GP",       "circuit": "Zandvoort",          "date": "2026-08-23", "meeting_key": 1292},
    {"round": 13, "name": "Italian GP",     "circuit": "Monza",              "date": "2026-09-06", "meeting_key": 1293},
    {"round": 14, "name": "Madring GP",     "circuit": "Madrid",             "date": "2026-09-13", "meeting_key": 1294},
    {"round": 15, "name": "Singapore GP",   "circuit": "Marina Bay",         "date": "2026-09-20", "meeting_key": 1296},
    {"round": 16, "name": "Azerbaijan GP",  "circuit": "Baku",               "date": "2026-09-27", "meeting_key": 1295},
    {"round": 17, "name": "US GP",          "circuit": "Austin",             "date": "2026-10-18", "meeting_key": 1297},
    {"round": 18, "name": "Mexico City GP", "circuit": "Mexico City",        "date": "2026-10-25", "meeting_key": 1298},
    {"round": 19, "name": "São Paulo GP",   "circuit": "Interlagos",         "date": "2026-11-08", "meeting_key": 1299},
    {"round": 20, "name": "Las Vegas GP",   "circuit": "Las Vegas",          "date": "2026-11-21", "meeting_key": 1300},
    {"round": 21, "name": "Qatar GP",       "circuit": "Lusail",             "date": "2026-11-29", "meeting_key": 1301},
    {"round": 22, "name": "Abu Dhabi GP",   "circuit": "Yas Marina",         "date": "2026-12-06", "meeting_key": 1302},
]


@dataclass
class ContextBlock:
    """Wraps a context string with optional freshness metadata."""
    content: str
    data_age_hours: float | None = None
    completeness: str = "full"   # "full" | "partial" | "unknown"


def _unpack_ctx(val) -> tuple[str, str]:
    """Returns (content, meta_suffix) from a plain str or ContextBlock.

    meta_suffix is empty for plain strings and for full/no-age ContextBlocks,
    so callers can always do f"LABEL{meta}:{content[:N]}" safely.
    """
    if isinstance(val, ContextBlock):
        parts = []
        if val.data_age_hours is not None:
            parts.append(f"{val.data_age_hours:.1f}h old")
        if val.completeness != "full":
            parts.append(val.completeness)
        meta = f" ({', '.join(parts)})" if parts else ""
        return val.content, meta
    return val, ""
