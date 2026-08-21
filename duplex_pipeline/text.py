from __future__ import annotations

import unicodedata


def is_effective_char(char: str) -> bool:
    if char.isspace():
        return False
    return not unicodedata.category(char).startswith(("P", "S", "Z", "C"))


def effective_char_count(text: str) -> int:
    """Count semantic characters while excluding punctuation and whitespace."""
    return sum(1 for char in str(text) if is_effective_char(char))
