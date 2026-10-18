"""System tests through the MCP endpoint and the HTTP routes (SPEC 12)."""

import logging
import math
import random

from alembic import autogenerate
from alembic import migration
import sqlalchemy as sa

from app import config
from app import db
from app import i18n
from app import models
from app import nightly
from app import params as params_lib
from app import service
from app import vocab
from app.model import bandits
from app.model import prefs
from tests import conftest

MARKER = "SECRET_REPLY_MARKER_93f1"
PLAN = {
    "request_type": "explanation",
    "topic": "how to deploy the server",
    "language": "en",
    "needed_words": 200,
}
AUTH = {"authorization": "Bearer admin-secret"}
FORM = {"content-type": "application/x-www-form-urlencoded"}
WIDGET_SENTENCE = i18n.prompts()["widget_mode"]


def turn_number(plan: dict) -> int:
    return int(plan["turn_id"].removeprefix(service.TURN_PREFIX))


async def scalar(database, statement):
    async with database() as session:
        return await session.scalar(statement)


async def rows(database, statement) -> list:
    async with database() as session:
        return list(await session.scalars(statement))


async def dump(database) -> str:
    """Returns every row of every table, as one string."""
    async with database() as session:
        return "\n".join(
            [
                str((await session.execute(sa.select(table))).all())
                for table in models.Base.metadata.sorted_tables
            ]
        )


async def snapshot(database) -> dict:
    """Returns the derived state, comparable across rebuilds."""
    out = {}
    async with database() as session:
        for model in nightly.DERIVED:
            result = await session.execute(sa.select(model.__table__))
            out[model.__tablename__] = sorted(map(str, result.all()))
    return out


async def pref_count(
    database, user_id: int, scope: str, dimension: str, value: str
) -> float:
    pc = models.PrefCount
    count = await scalar(
        database,
        sa.select(pc.count).where(
            pc.user_id == user_id,
            pc.scope == scope,
            pc.dimension == dimension,
            pc.value == value,
        ),
    )
    return count or 0.0


def with_settings(monkeypatch, **changes) -> None:
    monkeypatch.setattr(
        config, "settings", config.settings.model_copy(update=changes)
    )


async def plan_and_close(client, **extra) -> tuple[dict, dict]:
    """Plans a turn, shows a reply, and closes it with a next plan."""
    plan = (await client.call("plan_reply", **PLAN))["data"]
    await client.call("show_reply", turn_id=plan["turn_id"], text="word " * 30)
    nxt = (
        await client.call(
            "plan_reply",
            **PLAN,
            conversation_id=plan["conversation_id"],
            **extra,
        )
    )["data"]
    return plan, nxt


async def test_full_turn_sequence_and_no_trace_of_text(
    make_client, database, caplog
):
    caplog.set_level(logging.DEBUG)
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    assert plan["instruction"].endswith(WIDGET_SENTENCE)
    shown = await client.call(
        "show_reply",
        turn_id=plan["turn_id"],
        text=f"# Answer\n\n{MARKER} is three words long.",
    )
    assert shown["content"] == [{"type": "text", "text": "ok"}]
    content = shown["structuredContent"]
    assert content["profile_url"].endswith(service.token_hash(client.token))
    b = content["buttons"][0]
    await client.call(
        "record_click",
        turn_id=plan["turn_id"],
        button=b["id"],
        position=b["position"],
    )
    nxt = (
        await client.call(
            "plan_reply",
            **PLAN,
            conversation_id=plan["conversation_id"],
            last_reply_outcome="too_long",
            user_quote="Shorter.",
        )
    )["data"]
    assert nxt["conversation_id"] == plan["conversation_id"]
    turn_id = turn_number(plan)
    words = await scalar(
        database,
        sa.select(models.Reply.reply_words_exact).where(
            models.Reply.turn_id == turn_id
        ),
    )
    assert words == 6
    for model in (models.TurnInference, models.TopicEmbedding):
        assert await scalar(
            database, sa.select(model).where(model.turn_id == turn_id)
        )
    server_logs = "\n".join(
        r.getMessage() for r in caplog.records if r.name != "httpx"
    )
    assert MARKER not in await dump(database)
    assert MARKER not in server_logs
    assert client.token not in server_logs
    assert "/mcp/<redacted>" in server_logs


async def test_show_reply_errors_hide_the_text(make_client, caplog):
    client = await make_client()
    for args in (
        {"text": MARKER},
        {"turn_id": "t_999", "text": MARKER},
        {"turn_id": "x", "text": [MARKER]},
    ):
        result = await client.call("show_reply", **args)
        assert result["isError"] and MARKER not in str(result)
    assert MARKER not in caplog.text


async def test_mode_selection_in_both_protocol_eras(make_client, database):
    for version in (conftest.LEGACY, conftest.MODERN):
        for apps, mode in (
            (True, vocab.ReplyMode.WIDGET),
            (False, vocab.ReplyMode.FALLBACK),
        ):
            client = await make_client(apps=apps, version=version)
            plan = (await client.call("plan_reply", **PLAN))["data"]
            assert plan["instruction"].endswith(WIDGET_SENTENCE) == apps
            async with database() as session:
                turn = await session.get(models.Turn, turn_number(plan))
                conv = await session.get(
                    models.Conversation, turn.conversation_id
                )
            assert (turn.reply_mode, conv.host) == (mode, "claude-ai")


async def test_foreign_handles_are_rejected(make_client):
    alice, bob = await make_client(), await make_client()
    plan = (await alice.call("plan_reply", **PLAN))["data"]
    turn_id = plan["turn_id"]
    assert (await bob.call("show_reply", turn_id=turn_id, text="hi"))["isError"]
    assert (
        await bob.call(
            "record_click", turn_id=turn_id, button="shorter", position=0
        )
    )["isError"]
    assert (await bob.call("report_context", turn_id=turn_id))["isError"]
    assert (
        await bob.call(
            "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
        )
    )["isError"]


async def test_unknown_token_is_404(http):
    assert (await http.post("/mcp/not-a-token", json={})).status_code == 404


async def test_repair_without_quote_is_no_signal(make_client, database):
    plan, _ = await plan_and_close(
        await make_client(), last_reply_outcome="too_long"
    )
    observed = await scalar(
        database,
        sa.select(models.Outcome.observed).where(
            models.Outcome.turn_id == turn_number(plan)
        ),
    )
    assert observed == vocab.Outcome.NO_SIGNAL


async def test_arms_share_and_behaviour(make_client, monkeypatch):
    with_settings(
        monkeypatch,
        control_share=0.1,
        concise_share=0.1,
    )
    client = await make_client()
    plans = [
        (await client.call("plan_reply", **PLAN))["data"] for _ in range(300)
    ]
    held_back = [p for p in plans if p["target_words"] is None]
    assert 0.12 <= len(held_back) / len(plans) <= 0.3
    arms = i18n.prompts()["arms"]
    expected = {f"{arms[a]} {WIDGET_SENTENCE}" for a in arms}
    assert {p["instruction"] for p in held_back} == expected
    reply = await client.call(
        "show_reply", turn_id=held_back[0]["turn_id"], text="x"
    )
    assert reply["structuredContent"]["buttons"] == []


async def test_click_drops_the_assistant_verdict(make_client, database):
    client = await make_client()
    probs = {}
    for clicked in (False, True):  # The verdict is too_short both times.
        plan = (await client.call("plan_reply", **PLAN))["data"]
        reply = await client.call(
            "show_reply", turn_id=plan["turn_id"], text="one two"
        )
        if clicked:
            b = next(
                b
                for b in reply["structuredContent"]["buttons"]
                if b["id"] != vocab.Button.MORE_DETAIL
            )
            await client.call(
                "record_click",
                turn_id=plan["turn_id"],
                button=b["id"],
                position=b["position"],
            )
        await client.call(
            "plan_reply",
            **PLAN,
            conversation_id=plan["conversation_id"],
            last_reply_outcome="too_short",
            user_quote="tell me more",
        )
        probs[clicked] = await scalar(
            database,
            sa.select(models.TurnInference.reaction_probs).where(
                models.TurnInference.turn_id == turn_number(plan)
            ),
        )
    assert probs[False][vocab.Outcome.TOO_SHORT] > 0.5
    assert probs[True][vocab.Outcome.TOO_SHORT] < 0.2


async def test_format_issue_moves_the_preference(make_client, database):
    client = await make_client()
    await plan_and_close(
        client,
        last_reply_outcome="wrong_format",
        user_quote="give me a list",
        format_issue="want_list",
    )
    count = await pref_count(
        database,
        await client.user_id(),
        PLAN["request_type"],
        vocab.Dimension.FORMAT,
        vocab.PrefValue.BULLETS,
    )
    assert count >= prefs.EXPLICIT


async def test_preference_button_moves_the_preference(make_client, database):
    client = await make_client()
    b = None
    while b is None:  # Until a button with a preference signal is shown.
        plan = (await client.call("plan_reply", **PLAN))["data"]
        reply = await client.call(
            "show_reply", turn_id=plan["turn_id"], text="word " * 30
        )
        b = next(
            (
                b
                for b in reply["structuredContent"]["buttons"]
                if bandits.EFFECTS[b["id"]].preference
            ),
            None,
        )
    await client.call(
        "record_click",
        turn_id=plan["turn_id"],
        button=b["id"],
        position=b["position"],
    )
    await client.call(
        "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
    )
    dimension, value = bandits.EFFECTS[b["id"]].preference
    count = await pref_count(
        database, await client.user_id(), prefs.USER_SCOPE, dimension, value
    )
    assert count >= prefs.EXPLICIT


async def test_buttons_and_labels_come_in_the_turns_language(make_client):
    client = await make_client()
    plan = await client.call("plan_reply", **{**PLAN, "language": "fr"})
    content = (
        await client.call(
            "show_reply", turn_id=plan["data"]["turn_id"], text="bonjour"
        )
    )["structuredContent"]
    french = i18n.catalog("fr")
    assert content["labels"] == french["widget"]
    for b in content["buttons"]:
        assert b["label"] == french["buttons"][b["id"]]["label"]


async def test_ambiguous_requests_can_get_a_question(make_client):
    client = await make_client()
    instructions = set()
    for _ in range(30):
        plan = await client.call("plan_reply", **PLAN, ambiguous=True)
        instructions.add(
            plan["data"]["instruction"].removesuffix(" " + WIDGET_SENTENCE)
        )
    assert i18n.prompts()["clarify"] in instructions
    assert len(instructions) > 1


async def test_mobile_report_moves_the_length_cell(make_client, database):
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    await client.call(
        "report_context",
        turn_id=plan["turn_id"],
        platform=vocab.Platform.MOBILE,
        locale="fr-FR",
    )
    nxt = (
        await client.call(
            "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
        )
    )["data"]
    style = await scalar(
        database,
        sa.select(models.Turn.style).where(models.Turn.id == turn_number(nxt)),
    )
    assert style["cell"] == "explanation@mobile"


async def test_profile_page_lists_and_applies_corrections(
    http, make_client, database
):
    client = await make_client()
    await plan_and_close(client)
    key = service.token_hash(client.token)
    page = await http.get(f"/me/{key}")
    title = i18n.catalog(i18n.FALLBACK)["profile"]["title"]
    assert page.status_code == 200 and title in page.text
    corrected = await http.post(
        f"/me/{key}",
        content="scope=*&dimension=format&value=table",
        headers=FORM,
    )
    assert corrected.status_code == 303
    count = await pref_count(
        database,
        await client.user_id(),
        prefs.USER_SCOPE,
        vocab.Dimension.FORMAT,
        vocab.PrefValue.TABLE,
    )
    assert count >= prefs.EDIT
    length = await http.post(
        f"/me/{key}",
        content="scope=explanation&dimension=length&value=short",
        headers=FORM,
    )
    assert length.status_code == 303
    bad = await http.post(
        f"/me/{key}",
        content="scope=*&dimension=format&value=nonsense",
        headers=FORM,
    )
    assert bad.status_code == 422
    assert (await http.get("/me/not-a-key")).status_code == 404


async def test_replay_rebuilds_the_online_state(http, make_client, database):
    client = await make_client()
    for _ in range(3):
        plan = (await client.call("plan_reply", **PLAN))["data"]
        for outcome in ("too_long", "moved_on", "too_short"):
            await client.call(
                "show_reply", turn_id=plan["turn_id"], text="word " * 40
            )
            plan = (
                await client.call(
                    "plan_reply",
                    **PLAN,
                    conversation_id=plan["conversation_id"],
                    last_reply_outcome=outcome,
                    user_quote="ok then",
                )
            )["data"]
    await http.post(
        f"/me/{service.token_hash(client.token)}",
        content="scope=*&dimension=tone&value=direct",
        headers=FORM,
    )
    online = await snapshot(database)
    async with database() as session, session.begin():
        p = await params_lib.load_params(session)
        assert await nightly.replay(session, p) == 9
    assert await snapshot(database) == online


async def test_nightly_job_runs(make_client, database):
    await plan_and_close(await make_client())
    async with database() as session, session.begin():
        assert await nightly.run(session) == 1
    mu0 = await scalar(
        database,
        sa.select(models.Hyperparam.value).where(
            models.Hyperparam.name == "mu0"
        ),
    )
    assert mu0 is not None


async def test_policy_swap_through_configuration(
    make_client, database, monkeypatch
):
    with_settings(monkeypatch, policy="fixed-v1")
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    assert plan["target_words"] == 160  # 0.8 x 200.
    versions = await rows(database, sa.select(models.Turn.model_version))
    assert versions == ["fixed-v1"]


async def test_metrics_and_gold_labels(http, make_client):
    plan, _ = await plan_and_close(await make_client())
    assert (await http.get("/metrics")).status_code == 401
    labelled = await http.post(
        "/gold_label",
        json={"turn_id": plan["turn_id"], "outcome": "moved_on"},
        headers=AUTH,
    )
    assert labelled.status_code == 200
    body = (await http.get("/metrics", headers=AUTH)).json()
    assert body["tool_coverage"]["turns"] == 2
    assert body["by_group"][vocab.Arm.TREATMENT]["tool_words_per_turn"] > 0


async def test_dashboard(http, make_client):
    plan, _ = await plan_and_close(await make_client())
    assert "Lingua Franca Admin" in (await http.get("/admin")).text
    assert (await http.get("/admin/api/overview")).status_code == 401
    overview = (await http.get("/admin/api/overview", headers=AUTH)).json()
    assert overview["counts"]["turns"] == 2
    conversation = plan["conversation_id"].removeprefix(
        service.CONVERSATION_PREFIX
    )
    turns = (
        await http.get(
            "/admin/api/turns",
            params={"conversation": conversation},
            headers=AUTH,
        )
    ).json()
    assert [t["turn_index"] for t in turns] == [2, 1]
    detail = (
        await http.get(f"/admin/api/turns/{turn_number(plan)}", headers=AUTH)
    ).json()
    assert detail["next_turn"] and detail["inference"]["reaction_probs"]
    users = (await http.get("/admin/api/users", headers=AUTH)).json()
    user = (
        await http.get(f"/admin/api/users/{users[0]["id"]}", headers=AUTH)
    ).json()
    assert user["lengths"] and user["preferences"]
    model = (await http.get("/admin/api/model", headers=AUTH)).json()
    assert vocab.Source.ASSISTANT in model["confusions"]
    replayed = (await http.post("/admin/api/nightly", headers=AUTH)).json()
    assert replayed == {"replayed_turns": 1}


async def test_tools_listed_to_the_model(make_client):
    client = await make_client()
    tools = {
        t["name"]: t
        for t in (await client.rpc("tools/list"))["result"]["tools"]
    }
    assert set(tools) == {
        "plan_reply",
        "show_reply",
        "record_click",
        "report_context",
    }
    for app_only in ("record_click", "report_context"):
        assert tools[app_only]["_meta"]["ui"]["visibility"] == ["app"]


async def test_schema_matches_the_models(database):
    """The migrations produce exactly the models' schema (no drift)."""
    del database

    def diff(connection):
        context = migration.MigrationContext.configure(
            connection, opts={"compare_type": True}
        )
        return autogenerate.compare_metadata(context, models.Base.metadata)

    async with db.engine.connect() as connection:
        assert await connection.run_sync(diff) == []


def test_nightly_learns_behaviour_classes_and_host_priors():
    """Two kinds of assistants, told apart by speed alone, per host."""
    rng = random.Random(4)
    p = params_lib.Params()
    turns, conversations = [], []
    for conversation_id in range(80):
        fast = conversation_id % 2 == 0
        speed = math.log(90 if fast else 35)
        conversations.append(
            {
                "id": conversation_id,
                "host": "fast-host" if fast else "slow-host",
                "kappa_mean": 0.0,
                "family_probs": {},
            }
        )
        for _ in range(3):
            words = rng.randint(80, 200)
            writing = words / math.exp(speed + rng.gauss(0, 0.05))
            turns.append(
                {
                    "user_id": 1,
                    "conversation_id": conversation_id,
                    "cell": "explanation",
                    "needed_words": 150,
                    "length_mean": math.log(words),
                    "duration": writing + p.overhead,
                }
            )
    nightly.refit_families(turns, conversations, p, {})
    assert len(p.families.params) == 2
    fast_class = max(
        p.families.params, key=lambda f: p.families.params[f]["speed_mu"]
    )
    prior = p.families.host_prior["fast-host"]
    assert max(prior, key=prior.get) == fast_class
