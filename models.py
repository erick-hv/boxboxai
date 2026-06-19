"""Shared data models for BoxBoxAI context assembly."""
from dataclasses import dataclass


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
