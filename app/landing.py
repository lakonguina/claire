"""Public landing page: the promise, and how to connect Claude."""

import functools
import pathlib

import fastapi
from fastapi import responses

from app import i18n
from app import oauth
from app import text
from app import vocab

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
SCREEN_SUFFIXES = (".webp", ".png", ".jpg")
CONNECTOR_NAME = "Lingua Franca"
# Where each step happens in Claude; a step without a link happens on its own.
LINKS = {
    vocab.SetupStep.INSTRUCTION: "https://claude.ai/new#settings/account",
    vocab.SetupStep.CONNECTOR: (
        "https://claude.ai/new?modal=add-custom-connector"
        "#customize/connectors/yours"
    ),
}

router = fastapi.APIRouter(include_in_schema=False)


@functools.cache
def _replies(language: str) -> dict:
    """Renders the demo replies, counted like show_reply counts them."""
    replies = i18n.catalog(language)["landing"]["demo"]["replies"]
    return {
        side: {
            "html": text.PARSER.render(replies[side]),
            "words": text.count_words(replies[side]),
        }
        for side in vocab.DemoSide
    }


def _screen(folder: str, name: str) -> str | None:
    """Returns the URL of assets/<folder>/<name>.<image suffix>, if any."""
    for suffix in SCREEN_SUFFIXES:
        path = ASSETS / folder / f"{name}{suffix}"
        if path.exists():
            return f"/assets/{folder}/{path.name}"
    return None


STEPS = [
    {"id": step, "link": LINKS.get(step), "screen": _screen("setup", step)}
    for step in vocab.SetupStep
]
# Screenshots of the demo replies; without them, the catalog's texts show.
DEMO_SCREENS = {side: _screen("demo", side) for side in vocab.DemoSide}


@router.get("/", response_class=responses.HTMLResponse)
async def page(request: fastapi.Request) -> str:
    """Renders the landing page in the browser's language."""
    language = i18n.preferred(request.headers.get("accept-language"))
    return i18n.TEMPLATES.get_template("landing.html").render(
        language=language,
        texts=i18n.catalog(language)["landing"],
        instruction=i18n.catalog(i18n.FALLBACK)["tools"]["server"],
        connector_name=CONNECTOR_NAME,
        mcp_url=oauth.RESOURCE,
        steps=STEPS,
        demo_screens=DEMO_SCREENS,
        replies=_replies(language),
        SetupStep=vocab.SetupStep,
        DemoSide=vocab.DemoSide,
    )
