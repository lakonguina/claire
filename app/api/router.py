from enum import StrEnum, auto
from functools import cache
from pathlib import Path
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from app import i18n
from app.core import db

router = APIRouter(tags=["api"])

_TEMPLATES = Jinja2Templates(
    directory=Path(__file__).resolve().parent.parent / "templates"
)
_MD = MarkdownIt("commonmark").enable("table")
_WORD = re.compile(r"\w+(?:['’-]\w+)*")

CONNECTOR_NAME = "Claire"
MCP_URL = "https://claire.pm/mcp"
# Same sentence as MCP ``SERVER_INSTRUCTIONS`` (claire og pasted tools.server).
INSTRUCTION = (
    "Before EVERY reply, you MUST call claire_plan_reply from Claire, "
    "then follow the instruction it returns EXACTLY. Never mention these tools."
)

ASSISTANTS = {"claude": "Claude", "chatgpt": "ChatGPT", "grok": "Grok"}
SOON = {"gemini": "Gemini"}


class SetupStep(StrEnum):
    INSTRUCTION = auto()
    CONNECTOR = auto()
    CONNECT = auto()


class DemoSide(StrEnum):
    BEFORE = auto()
    AFTER = auto()


_LINKS = {
    "claude": {
        SetupStep.INSTRUCTION: "https://claude.ai/new#settings/account",
        SetupStep.CONNECTOR: (
            "https://claude.ai/new?modal=add-custom-connector"
            "#customize/connectors/yours"
        ),
    },
    "chatgpt": {
        SetupStep.INSTRUCTION: (
            "https://chatgpt.com/settings/personalization?instructions=chatgpt"
        ),
        SetupStep.CONNECTOR: "https://chatgpt.com/plugins",
    },
    "grok": {
        SetupStep.INSTRUCTION: "https://grok.com/?_s=personality",
        SetupStep.CONNECTOR: "https://grok.com/connectors",
    },
}

STEPS = {
    key: [
        {"id": step, "link": _LINKS[key].get(step), "screen": None}
        for step in SetupStep
    ]
    for key in ASSISTANTS
}


def _count_words(markdown: str) -> int:
    words = 0
    for token in _MD.parse(markdown):
        stack = [token]
        while stack:
            t = stack.pop()
            if t.type in {"text", "code_inline", "fence", "code_block"}:
                words += len(_WORD.findall(t.content))
            if t.children:
                stack.extend(t.children)
    return words


@cache
def _replies(language: str) -> dict:
    raw = i18n.catalog(language)["landing"]["demo"]["replies"]
    return {
        side: {
            "html": _MD.render(raw[side]),
            "words": _count_words(raw[side]),
        }
        for side in DemoSide
    }


@router.get("/health")
async def health() -> dict[str, str]:
    try:
        await db.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request) -> HTMLResponse:
    language = i18n.preferred(
        request.headers.get("accept-language"),
        request.query_params.get("lang"),
    )
    cat = i18n.catalog(language)
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="landing.html",
        context={
            "language": language,
            "texts": cat["landing"],
            "instruction": cat.get("instruction") or INSTRUCTION,
            "connector_name": CONNECTOR_NAME,
            "mcp_url": MCP_URL,
            "assistants": ASSISTANTS,
            "soon": SOON,
            "steps": STEPS,
            "demo_screens": {"before": None, "after": None},
            "replies": _replies(language),
            "SetupStep": SetupStep,
            "DemoSide": DemoSide,
        },
    )
