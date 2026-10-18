"""Locale catalogs (SPEC 3): every text a person or the model reads.

English is complete and is the fallback for every missing key; it is also the
only language of the model-facing prompts.
"""

import functools
import json
import pathlib
import re

import jinja2

LOCALES = pathlib.Path(__file__).resolve().parent.parent / "locales"
FALLBACK = "en"
LANGUAGE = re.compile(r"[a-z]{2}")
TEMPLATES = jinja2.Environment(  # The localized HTML pages.
    loader=jinja2.FileSystemLoader(pathlib.Path(__file__).resolve().parent),
    autoescape=True,
)


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        both_dicts = isinstance(value, dict) and isinstance(base.get(key), dict)
        merged[key] = _merge(base[key], value) if both_dicts else value
    return merged


def _load(language: str) -> dict:
    return json.loads((LOCALES / f"{language}.json").read_text())


@functools.cache
def catalog(language: str | None) -> dict:
    """Returns a language's catalog, completed by the English one.

    Args:
        language: An ISO 639-1 code; anything else gives English.

    Returns:
        The merged catalog.
    """
    base = _load(FALLBACK)
    if not language or not LANGUAGE.fullmatch(language):
        return base
    if not (LOCALES / f"{language}.json").exists():
        return base
    return _merge(base, _load(language))


def prompts() -> dict:
    """Returns the model-facing prompt texts."""
    return catalog(FALLBACK)["prompts"]


def preferred(accept_language: str | None) -> str:
    """Returns the first language of an Accept-Language header we have."""
    first = (accept_language or "").split(",")[0].split(";")[0]
    language = first.split("-")[0].strip().lower()
    known = LANGUAGE.fullmatch(language) and (LOCALES / f"{language}.json")
    return language if known and known.exists() else FALLBACK
