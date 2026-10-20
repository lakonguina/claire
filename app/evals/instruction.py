"""python -m app.evals.instruction: the instruction's check before a launch.

Runs scripted conversations (scripts.json) through the main assistants, with
Claire's tools executed in process against the configured database, and
reports how well each assistant follows the instruction: plan_reply before
every reply, the reply only in show_reply, and each field when it is due.
A new prompt catalog runs here first, then on a share of live traffic
(CANDIDATE_SHARE).
"""

import asyncio
import json
import logging
import pathlib
import sys
from typing import Any

import httpx

from app.core import config
from app.core import db
from app.core import i18n
from app import mcp_server
from app import models
from app.services import replies

log = logging.getLogger(__name__)
SCRIPTS = pathlib.Path(__file__).with_name("scripts.json")
MAX_STEPS = 6  # Tool rounds per user message.
MAX_TOKENS = 2048
TIMEOUT = 120.0
ANTHROPIC = "https://api.anthropic.com/v1/messages"
OPENAI = "https://api.openai.com/v1/chat/completions"
EVAL_CONSENT = "eval"


async def model_tools() -> list[dict]:
    """Returns the tools listed to the model, as name, description, schema."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "schema": t.input_schema,
        }
        for t in await mcp_server.mcp.list_tools()
        if "app" not in ((t.meta or {}).get("ui", {}).get("visibility") or [])
    ]


class Anthropic:
    """The Messages API, with client tools."""

    def __init__(self, model: str, tools: list[dict]):
        self.model, self.messages = model, []
        self.tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["schema"],
            }
            for t in tools
        ]

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    async def step(self, http: httpx.AsyncClient) -> tuple[str, list[dict]]:
        """Returns the text written and the tool calls asked, in order."""
        response = await http.post(
            ANTHROPIC,
            headers={
                "x-api-key": config.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "system": i18n.catalog(i18n.FALLBACK)["tools"]["server"],
                "tools": self.tools,
                "messages": self.messages,
            },
        )
        response.raise_for_status()
        content = response.json()["content"]
        self.messages.append({"role": "assistant", "content": content})
        written = "".join(b["text"] for b in content if b["type"] == "text")
        calls = [
            {"id": b["id"], "name": b["name"], "arguments": b["input"]}
            for b in content
            if b["type"] == "tool_use"
        ]
        return written, calls

    def results(self, results: list[tuple[dict, str]]) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": c["id"],
                        "content": r,
                    }
                    for c, r in results
                ],
            }
        )


class OpenAI:
    """The Chat Completions API, with function tools."""

    def __init__(self, model: str, tools: list[dict]):
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": i18n.catalog(i18n.FALLBACK)["tools"]["server"],
            }
        ]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["schema"],
                },
            }
            for t in tools
        ]

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    async def step(self, http: httpx.AsyncClient) -> tuple[str, list[dict]]:
        """Returns the text written and the tool calls asked, in order."""
        response = await http.post(
            OPENAI,
            headers={
                "authorization": f"Bearer {config.settings.openai_api_key}"
            },
            json={
                "model": self.model,
                "messages": self.messages,
                "tools": self.tools,
            },
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        self.messages.append(message)
        calls = [
            {
                "id": c["id"],
                "name": c["function"]["name"],
                "arguments": json.loads(c["function"]["arguments"] or "{}"),
            }
            for c in message.get("tool_calls") or []
        ]
        return message.get("content") or "", calls

    def results(self, results: list[tuple[dict, str]]) -> None:
        self.messages.extend(
            {"role": "tool", "tool_call_id": c["id"], "content": r}
            for c, r in results
        )


PROVIDERS = {"anthropic": Anthropic, "openai": OpenAI}
KEYS = {"anthropic": "anthropic_api_key", "openai": "openai_api_key"}


async def _execute(caller: replies.Caller, call: dict) -> str:
    """Runs one tool call as the MCP layer would; returns its result text."""
    arguments = dict(call["arguments"] or {})
    try:
        if call["name"] == "plan_reply":
            return replies.compact(await replies.plan_reply(caller, arguments))
        if call["name"] == "show_reply":
            await replies.show_reply(
                caller,
                arguments.get("turn_id"),
                arguments.get("text"),
                arguments.get("choices"),
            )
            return mcp_server.TOOLS["shown"]
    except replies.HandleError:
        return mcp_server.TOOLS["errors"]["handle"]
    except replies.ReplyShownError:
        return mcp_server.TOOLS["errors"]["shown"]
    except (KeyError, TypeError, ValueError):
        return mcp_server.TOOLS["errors"][call["name"]]
    return mcp_server.TOOLS["errors"]["handle"]


async def _eval_user() -> int:
    async with db.session() as session, session.begin():
        user = models.User()
        session.add(user)
        await session.flush()
        client = await session.get(models.OAuthClient, EVAL_CONSENT)
        if client is None:
            session.add(models.OAuthClient(client_id=EVAL_CONSENT, info={}))
            await session.flush()
        session.add(
            models.OAuthGrant(
                user_id=user.id,
                client_id=EVAL_CONSENT,
                consent_version=EVAL_CONSENT,
            )
        )
        return user.id


def _score(turns: list[dict]) -> dict[str, float | None]:
    """Returns the share of user messages where each rule held."""

    def share(key: str) -> float | None:
        due = [t[key] for t in turns if t[key] is not None]
        return sum(due) / len(due) if due else None

    return {
        rule: share(rule)
        for rule in (
            "plan_reply_first",
            "reply_in_show_reply",
            "no_text_outside",
            "goal_on_first_message",
            "last_turn_id",
            "conversation_id",
        )
    }


async def run_conversation(
    assistant: Any, caller: replies.Caller, messages: list[str], http
) -> list[dict]:
    """Plays one scripted conversation; returns what held on each message."""
    out = []
    for index, message in enumerate(messages):
        assistant.user(message)
        calls_made, written = [], ""
        for _ in range(MAX_STEPS):
            text, calls = await assistant.step(http)
            written += text
            if not calls:
                break
            assistant.results([(c, await _execute(caller, c)) for c in calls])
            calls_made.extend(calls)
        names = [c["name"] for c in calls_made]
        plans = [
            c["arguments"] for c in calls_made if c["name"] == "plan_reply"
        ]
        first_plan = plans[0] if plans else {}
        out.append(
            {
                "plan_reply_first": bool(names) and names[0] == "plan_reply",
                "reply_in_show_reply": "show_reply" in names,
                "no_text_outside": not written.strip(),
                "goal_on_first_message": (
                    bool(first_plan.get("goal")) if index == 0 else None
                ),
                "last_turn_id": (
                    bool(first_plan.get("last_turn_id")) if index else None
                ),
                "conversation_id": (
                    bool(first_plan.get("conversation_id")) if index else None
                ),
            }
        )
    return out


async def evaluate(http: httpx.AsyncClient | None = None) -> dict:
    """Runs every script through every configured model; returns the report."""
    scripts = json.loads(SCRIPTS.read_text(encoding="utf-8"))
    tools = await model_tools()
    report = {"catalog_version": i18n.catalog_version(i18n.has_candidate())}
    own = http is None
    http = http or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        for spec in config.settings.eval_models:
            provider, _, model = spec.partition(":")
            if provider not in PROVIDERS or not getattr(
                config.settings, KEYS[provider]
            ):
                report[spec] = {"skipped": "no API key"}
                continue
            turns = []
            for script in scripts:
                caller = replies.Caller(
                    await _eval_user(), f"eval:{spec}", ui=True
                )
                turns += await run_conversation(
                    PROVIDERS[provider](model, tools),
                    caller,
                    script["messages"],
                    http,
                )
            report[spec] = _score(turns)
    finally:
        if own:
            await http.aclose()
    return report


async def main() -> None:
    """Prints the report as JSON."""
    logging.basicConfig(level=logging.INFO)
    await db.migrate()
    json.dump(await evaluate(), sys.stdout, indent=2)
    await db.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
