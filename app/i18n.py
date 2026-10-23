"""Landing-page locale catalogs (JSON files under ``app/locales/``).

English is the fallback: if a key is missing in French, readers still see
the English string instead of a blank.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LOCALES = Path(__file__).resolve().parent / "locales"
FALLBACK = "en"
_LANG = re.compile(r"^[a-z]{2}$")


def _load(language: str) -> dict:
    """Read one locale JSON file from disk.

    Args:
        language: Two-letter code (e.g. ``en``, ``fr``).

    Returns:
        Parsed JSON object.
    """
    path = LOCALES / f"{language}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base`` (dicts recurse; leaves replace).

    Args:
        base: Fallback catalog (usually English).
        override: Preferred language catalog.

    Returns:
        A new merged dict (inputs are not mutated).
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge(base[key], value)
        else:
            merged[key] = value
    return merged


def catalog(language: str | None) -> dict:
    """Return the merged catalog for a language.

    Args:
        language: Two-letter code, or ``None`` / invalid → English only.

    Returns:
        Nested dict of UI strings for the landing page (and later tools).
    """
    base = _load(FALLBACK)
    if not language or not _LANG.fullmatch(language):
        return base
    path = LOCALES / f"{language}.json"
    if not path.exists() or language == FALLBACK:
        return base
    return _merge(base, _load(language))


def preferred(
    accept_language: str | None,
    explicit: str | None = None,
) -> str:
    """Pick a locale from ``?lang=`` or the ``Accept-Language`` header.

    Args:
        accept_language: Raw ``Accept-Language`` header, if any.
        explicit: Optional ``lang`` query param (wins when valid).

    Returns:
        A two-letter language code we actually ship, else ``en``.
    """
    if (
        explicit
        and _LANG.fullmatch(explicit)
        and (LOCALES / f"{explicit}.json").exists()
    ):
        return explicit
    first = (accept_language or "").split(",")[0].split(";")[0]
    language = first.split("-")[0].strip().lower()
    if _LANG.fullmatch(language) and (LOCALES / f"{language}.json").exists():
        return language
    return FALLBACK
