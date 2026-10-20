"""System tests through the MCP endpoint and the HTTP routes."""

import json
import logging
import math
import random

from alembic import autogenerate
from alembic import migration
import httpx
import pandas as pd
import sqlalchemy as sa

from app.core import config
from app.core import db
from app.core import i18n
from app.core import vocab
from app.evals import instruction
from app.jobs import nightly
from app import mcp_server
from app.model import family as family_lib
from app.model import prefs
from app.model import reaction
from app.model import stats
from app.model import theta
from app import models
from app.routers import oauth
from app.services import metrics
from app.services import params as params_lib
from app.services import replies
from app.services import topics
from tests import conftest

MARKER = "SECRET_REPLY_MARKER_93f1"
PLAN = {
    "topic": "how to deploy the server",
    "language": "en",
    "needed_words": 200,
}
AUTH = {"authorization": "Bearer admin-secret"}
WIDGET_SENTENCE = i18n.prompts()["widget_mode"]


def turn_number(plan: dict) -> int:
    return int(plan["turn_id"].removeprefix(replies.TURN_PREFIX))


async def scalar(database, statement):
    async with database() as session:
        return await session.scalar(statement)


async def rows(database, statement) -> list:
    async with database() as session:
        return list(await session.scalars(statement))


async def dump(database) -> str:
    """Returns every row of every table as stored, ciphertexts included."""
    async with database() as session:
        return "\n".join(
            [
                str(
                    (
                        await session.execute(
                            sa.select(
                                sa.func.to_jsonb(
                                    sa.literal_column(f'"{table.name}"')
                                )
                            ).select_from(table)
                        )
                    ).all()
                )
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
    assert WIDGET_SENTENCE in plan["instruction"]
    shown = await client.call(
        "show_reply",
        turn_id=plan["turn_id"],
        text=f"# Answer\n\n{MARKER} is three words long.",
    )
    assert shown["content"] == [
        {"type": "text", "text": mcp_server.TOOLS["shown"]}
    ]
    await client.call(
        "record_event", turn_id=plan["turn_id"], kind=vocab.WidgetEvent.COPY
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
    assert client.access not in server_logs
    assert "POST /mcp 200" in server_logs


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


async def test_one_show_reply_per_turn(make_client, database):
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    await client.call("show_reply", turn_id=plan["turn_id"], text="one two")
    again = await client.call(
        "show_reply", turn_id=plan["turn_id"], text="one two three"
    )
    assert again["isError"]
    assert again["content"][0]["text"] == mcp_server.TOOLS["errors"]["shown"]
    words = await scalar(
        database,
        sa.select(sa.func.array_agg(models.Reply.reply_words_exact)).where(
            models.Reply.turn_id == turn_number(plan)
        ),
    )
    assert words == [2]


async def test_mode_selection_in_both_protocol_eras(make_client, database):
    for version in (conftest.LEGACY, conftest.MODERN):
        for apps, mode in (
            (True, vocab.ReplyMode.WIDGET),
            (False, vocab.ReplyMode.FALLBACK),
        ):
            client = await make_client(apps=apps, version=version)
            plan = (await client.call("plan_reply", **PLAN))["data"]
            assert (WIDGET_SENTENCE in plan["instruction"]) == apps
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
            "record_event", turn_id=turn_id, kind=vocab.WidgetEvent.COPY
        )
    )["isError"]
    assert (await bob.call("report_context", turn_id=turn_id))["isError"]
    assert (
        await bob.call(
            "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
        )
    )["isError"]


async def test_a_conversation_stays_in_its_host(http, make_client, database):
    web = await make_client(apps=True)
    plan = (await web.call("plan_reply", **PLAN))["data"]
    code = conftest.McpClient(http, web.access, apps=False, host="claude-code")
    carried = (
        await code.call(
            "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
        )
    )["data"]
    assert carried["conversation_id"] != plan["conversation_id"]
    resumed = (
        await web.call(
            "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
        )
    )["data"]
    assert resumed["conversation_id"] == plan["conversation_id"]
    async with database() as session:
        turns = [
            await session.get(models.Turn, turn_number(p))
            for p in (plan, carried, resumed)
        ]
        conv = await session.get(models.Conversation, turns[1].conversation_id)
    assert (conv.host, conv.ui_supported) == ("claude-code", False)
    assert [t.turn_index for t in turns] == [1, 1, 2]


async def test_connector_urls_are_gone(http):
    assert (await http.post("/mcp/not-a-token", json={})).status_code == 404


async def test_the_landing_page_gives_what_to_paste(http):
    page = await http.get("/", headers={"accept-language": "fr-FR"})
    assert page.status_code == 200
    assert i18n.catalog("fr")["landing"]["title_b"] in page.text
    assert mcp_server.TOOLS["server"] in page.text
    assert oauth.RESOURCE in page.text
    css = await http.get("/static/site/landing.css")
    assert css.status_code == 200


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
        holdout_share=0.1,
        concise_share=0.1,
    )
    client = await make_client()
    plans = [
        (await client.call("plan_reply", **PLAN))["data"] for _ in range(300)
    ]
    arms = i18n.prompts()["arms"]
    expected = {f"{arms[a]} {WIDGET_SENTENCE}" for a in arms}

    def arm_part(plan: dict) -> str:  # Without what is asked after it.
        return plan["instruction"].split(WIDGET_SENTENCE)[0] + WIDGET_SENTENCE

    held_back = [p for p in plans if arm_part(p) in expected]
    assert 0.12 <= len(held_back) / len(plans) <= 0.3
    assert {arm_part(p) for p in held_back} == expected
    reply = await client.call(
        "show_reply", turn_id=held_back[0]["turn_id"], text="x"
    )
    assert reply["structuredContent"]["choices"] == []


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
        topics.ANY_DOMAIN,
        vocab.Dimension.FORMAT,
        vocab.PrefValue.BULLETS,
    )
    assert count >= prefs.EXPLICIT


async def test_labels_come_in_the_turns_language(make_client):
    client = await make_client()
    plan = await client.call("plan_reply", **{**PLAN, "language": "fr"})
    content = (
        await client.call(
            "show_reply", turn_id=plan["data"]["turn_id"], text="bonjour"
        )
    )["structuredContent"]
    assert content["labels"] == i18n.catalog("fr")["widget"]


async def test_ambiguous_requests_can_get_a_question(make_client):
    client = await make_client()
    instructions = set()
    for _ in range(30):
        plan = await client.call("plan_reply", **PLAN, ambiguity=0.9)
        instructions.add(
            plan["data"]["instruction"].removesuffix(" " + WIDGET_SENTENCE)
        )
    clarify = i18n.prompts()["clarify_widget"]
    assert any(i.startswith(clarify) for i in instructions)
    assert len(instructions) > 1
    for _ in range(10):  # Below the threshold, never asked first.
        plan = await client.call("plan_reply", **PLAN, ambiguity=0.1)
        assert (
            i18n.prompts()["clarify_widget"] not in plan["data"]["instruction"]
        )


async def test_mobile_report_becomes_a_length_feature(make_client, database):
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
    assert style["mobile"]


async def test_replay_rebuilds_the_online_state(make_client, database):
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
    online = await snapshot(database)
    async with database() as session, session.begin():
        p = await params_lib.load_params(session)
        assert await nightly.replay(session, p) == 9
    assert await snapshot(database) == online


async def test_nightly_job_runs(make_client, database):
    await plan_and_close(await make_client())
    async with database() as session, session.begin():
        assert await nightly.run(session) == 1
    typical = await scalar(
        database,
        sa.select(models.Hyperparam.value).where(
            models.Hyperparam.name == f"{params_lib.PRIOR_MEAN}{theta.BIAS}"
        ),
    )
    assert typical is not None


async def test_policy_swap_through_configuration(
    make_client, database, monkeypatch
):
    with_settings(monkeypatch, policy="fixed-v1")
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    assert "160 words" in plan["instruction"]  # 0.8 x 200.
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
    assert "Claire Admin" in (await http.get("/admin")).text
    french = i18n.catalog("fr")["dashboard"]["tabs"]["model"]
    page = await http.get("/admin", headers={"accept-language": "fr-FR,en"})
    assert french in page.text and 'lang="fr"' in page.text
    assert (await http.get("/admin/api/overview")).status_code == 401
    overview = (await http.get("/admin/api/overview", headers=AUTH)).json()
    assert overview["counts"]["turns"] == 2
    conversation = plan["conversation_id"].removeprefix(
        replies.CONVERSATION_PREFIX
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
        "record_event",
        "report_context",
    }
    for app_only in ("record_event", "report_context"):
        assert tools[app_only]["_meta"]["ui"]["visibility"] == ["app"]
    # ChatGPT requires the three hints on every tool.
    for tool in tools.values():
        assert tool["annotations"] == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
        }


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
                    "theta_mean": math.log(0.6),
                    "needed_words": 150,
                    "length_mean": math.log(words),
                    "duration": writing + p.overhead,
                }
            )
    nightly.refit_families(pd.DataFrame(turns), pd.DataFrame(conversations), p)
    assert len(p.families.params) == 2
    fast_class = max(
        p.families.params, key=lambda f: p.families.params[f]["speed_mu"]
    )
    prior = p.families.host_prior["fast-host"]
    assert max(prior, key=prior.get) == fast_class


async def test_every_widget_address_serves_the_current_widget(make_client):
    client = await make_client()
    tools = (await client.rpc("tools/list"))["result"]["tools"]
    show = next(t for t in tools if t["name"] == "show_reply")
    assert show["_meta"]["ui"]["resourceUri"] == mcp_server.UI_URI
    for uri in (mcp_server.UI_URI, "ui://claire/reply.html"):
        read = await client.rpc("resources/read", {"uri": uri})
        assert read["result"]["contents"][0]["text"] == mcp_server.WIDGET_HTML


async def raw_text(database, model) -> str:
    """Returns a text column as stored: the ciphertext, in hex."""
    column = model.__table__.c.text
    async with database() as session:
        return str(
            (
                await session.execute(sa.select(sa.func.encode(column, "hex")))
            ).all()
        )


async def test_texts_are_stored_encrypted(make_client, database, caplog):
    caplog.set_level(logging.DEBUG)
    client = await make_client()
    plan = (
        await client.call(
            "plan_reply",
            **{**PLAN, "topic": f"{MARKER} topic"},
            user_message=f"{MARKER} please",
            goal=f"{MARKER} goal",
        )
    )["data"]
    await client.call(
        "show_reply", turn_id=plan["turn_id"], text=f"{MARKER} reply"
    )
    turn_id = turn_number(plan)
    texts = [
        await scalar(
            database, sa.select(model.text).where(model.turn_id == turn_id)
        )
        for model in (models.Message, models.Reply, models.Goal)
    ]
    assert texts == [f"{MARKER} please", f"{MARKER} reply", f"{MARKER} goal"]
    async with database() as session:
        turn = await session.get(models.Turn, turn_id)
    assert turn.topic == f"{MARKER} topic"
    assert "user_message" not in turn.raw_payload
    stored = await dump(database)
    assert MARKER not in stored
    assert MARKER.encode().hex() not in stored
    assert MARKER not in caplog.text


async def test_long_messages_keep_their_length_only(make_client, database):
    client = await make_client()
    plan = (
        await client.call("plan_reply", **PLAN, user_message="word " * 151)
    )["data"]
    message = await scalar(
        database,
        sa.select(models.Message).where(
            models.Message.turn_id == turn_number(plan)
        ),
    )
    assert (message.words, message.text) == (151, None)


async def test_plan_reply_errors_hide_the_message(make_client, caplog):
    client = await make_client()
    result = await client.call(
        "plan_reply", **{**PLAN, "needed_words": "many"}, user_message=MARKER
    )
    assert result["isError"] and MARKER not in str(result)
    assert MARKER not in caplog.text


async def test_sampled_conversations_ask_for_the_next_message(
    make_client, monkeypatch
):
    asked = i18n.prompts()["send_message"]
    for share, expected in ((1.0, True), (0.0, False)):
        with_settings(monkeypatch, message_sample=share)
        plan = (await (await make_client()).call("plan_reply", **PLAN))["data"]
        assert (asked in plan["instruction"]) == expected


async def test_decisions_are_logged_with_their_propensities(
    make_client, database
):
    client = await make_client()
    plan = (
        await client.call("plan_reply", **PLAN, messages_since_last_call=0)
    )["data"]
    async with database() as session:
        turn = await session.get(models.Turn, turn_number(plan))
        conv = await session.get(models.Conversation, turn.conversation_id)
    assert turn.instruction == plan["instruction"]
    assert turn.catalog_version == i18n.catalog_version()
    assert turn.messages_since_last_call == 0
    assert 0 < turn.propensity <= 1
    assert set(turn.style["propensities"]["form"]) == set(vocab.Dimension)
    assert all(0 < p <= 1 for p in turn.style["propensities"]["form"].values())
    assert 0 < turn.style["propensities"]["expertise"] <= 1
    assert conv.arm_propensity == 1 - sum(config.settings.arm_shares.values())
    assert (conv.host_version, conv.capabilities) == ("1", conftest.APPS_CAPS)


async def test_goals_and_the_ground_truth_question(
    http, make_client, database, monkeypatch
):
    with_settings(monkeypatch, ground_truth_share=1.0)
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN, goal="Deploy the server"))[
        "data"
    ]
    await client.call("show_reply", turn_id=plan["turn_id"], text="word " * 30)
    nxt = (
        await client.call(
            "plan_reply",
            **PLAN,
            conversation_id=plan["conversation_id"],
            last_turn_id=plan["turn_id"],
            goal_reached=True,
        )
    )["data"]
    shown = await client.call("show_reply", turn_id=nxt["turn_id"], text="ok")
    assert shown["structuredContent"]["survey"]
    answer = await client.call(
        "record_event",
        turn_id=nxt["turn_id"],
        kind=vocab.WidgetEvent.SATISFACTION,
        value=vocab.Satisfaction.YES,
    )
    assert not answer.get("isError")
    goal = await scalar(database, sa.select(models.Goal.text))
    assert goal == "Deploy the server"
    body = (await http.get("/metrics", headers=AUTH)).json()
    treated = body["by_group"][vocab.Arm.TREATMENT]
    assert treated["words_per_goal_reached"] > 0
    assert body["widget"]["satisfaction"] == {vocab.Satisfaction.YES: 1}
    assert body["fields"]["all"]["goal_on_first_turn"] == 1.0


async def test_an_edited_message_closes_the_turn_it_follows(
    make_client, database
):
    client = await make_client()
    first = (await client.call("plan_reply", **PLAN))["data"]
    conversation = {"conversation_id": first["conversation_id"]}
    second = (
        await client.call(
            "plan_reply", **PLAN, **conversation, last_turn_id=first["turn_id"]
        )
    )["data"]
    # The person edits their second message: the assistant's previous turn
    # is the first one again.
    edited = (
        await client.call(
            "plan_reply", **PLAN, **conversation, last_turn_id=first["turn_id"]
        )
    )["data"]
    async with database() as session:
        turns = {
            t.id: t
            for t in await session.scalars(
                sa.select(models.Turn).order_by(models.Turn.id)
            )
        }
    first_id = turn_number(first)
    assert turns[turn_number(second)].follows == first_id
    edit = turns[turn_number(edited)]
    assert (edit.follows, edit.branched, edit.turn_index) == (first_id, True, 3)
    closed = await rows(database, sa.select(models.TurnInference.turn_id))
    assert closed == [first_id]  # Closed once, by the first call after it.
    online = await snapshot(database)
    async with database() as session, session.begin():
        p = await params_lib.load_params(session)
        assert await nightly.replay(session, p) == 1
    assert await snapshot(database) == online


def ask_first(monkeypatch) -> None:
    """Makes every bandit pick its first arm: asking first, raising a point."""
    monkeypatch.setattr(
        replies, "_bandit", lambda counts, kind, arms: (list(arms)[0], 1.0)
    )


async def test_choices_only_for_clarifying_questions(make_client, monkeypatch):
    ask_first(monkeypatch)
    client = await make_client()
    options = ["The web app", "The API", "word " * 20, 3]
    for flag, expected in (
        ({"ambiguity": 0.9}, ["The web app", "The API"]),
        ({}, []),
    ):
        plan = (await client.call("plan_reply", **PLAN, **flag))["data"]
        shown = await client.call(
            "show_reply",
            turn_id=plan["turn_id"],
            text="Which one?",
            choices=options,
        )
        assert (
            shown["isError"]
            or shown["structuredContent"]["choices"] == expected
        )


async def test_widget_events_are_checked(make_client, database):
    client = await make_client()
    turn_id = (await client.call("plan_reply", **PLAN))["data"]["turn_id"]
    for kind, value, valid in (
        (vocab.WidgetEvent.VISIBLE, 12.5, True),
        (vocab.WidgetEvent.VISIBLE, "long", False),
        (vocab.WidgetEvent.CHOICE, 1, True),
        (vocab.WidgetEvent.CHOICE, replies.FREE_CHOICE, True),
        (vocab.WidgetEvent.CHOICE, "my own words", False),
        (vocab.WidgetEvent.SATISFACTION, "maybe", False),
        ("clicked_everything", None, False),
    ):
        result = await client.call(
            "record_event", turn_id=turn_id, kind=kind, value=value
        )
        assert bool(result.get("isError")) != valid, (kind, value)
    values = await rows(database, sa.select(models.WidgetEvent.value))
    assert values == [{"value": 12.5}, {"value": 1}, {"value": "free"}]


async def test_a_data_request_erases_the_person(http, make_client, database):
    client = await make_client()
    plan, _ = await plan_and_close(client)
    other, _ = await plan_and_close(await make_client())
    # Not theirs: refused, and no code.
    refused = await client.call(
        "show_reply", turn_id=other["turn_id"], text="x"
    )
    assert refused["content"][0]["text"] == mcp_server.TOOLS["errors"]["handle"]
    nxt = (
        await client.call(
            "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
        )
    )["data"]
    code = (await client.call("show_reply", turn_id=nxt["turn_id"], text="x"))[
        "structuredContent"
    ]["privacy"]["code"]
    user_id = await client.user_id()
    forged = code[:-1] + ("0" if code[-1] != "0" else "1")
    for value, status in ((forged, 404), (code, 200)):
        response = await http.post(
            "/admin/api/privacy/erase", json={"code": value}, headers=AUTH
        )
        assert response.status_code == status
    for model in (models.Turn, models.Conversation, models.ToolCall):
        left = await rows(
            database, sa.select(model.id).where(model.user_id == user_id)
        )
        assert not left
    assert await rows(database, sa.select(models.Turn.user_id).distinct()) != []
    request = await scalar(database, sa.select(models.DeletionRequest))
    assert request.user_id == user_id and request.done_at
    disconnected = await http.post(
        oauth.MCP_PATH,
        json={},
        headers={"authorization": f"Bearer {client.access}"},
    )
    assert disconnected.status_code == 401


async def test_retention_purges_old_texts(make_client, database, monkeypatch):
    client = await make_client()
    plan = (
        await client.call("plan_reply", **PLAN, user_message="hello there")
    )["data"]
    await client.call("show_reply", turn_id=plan["turn_id"], text="hi")
    with_settings(monkeypatch, retention_days=0)
    async with database() as session, session.begin():
        assert await nightly.purge(session) == 2
    for model in (models.Reply, models.Message):
        assert await scalar(database, sa.select(model.text)) is None
    assert (
        await scalar(database, sa.select(models.Reply.reply_words_exact)) == 1
    )
    assert (await scalar(database, sa.select(models.Purge))).rows == 2


async def test_consent_privacy_and_visits(http, database):
    client_id = await conftest.register(http)
    consent = await conftest.consent_url(http, client_id, "x" * 43)
    page = await http.get(consent, headers={"referer": "https://claude.ai/"})
    assert i18n.catalog("en")["consent"]["storage"] in page.text
    policy = await http.get(
        oauth.PRIVACY_PATH, headers={"accept-language": "fr"}
    )
    assert policy.status_code == 200
    assert config.settings.privacy_email in policy.text
    await http.get("/?utm_source=newsletter&utm_campaign=launch")
    visits = await rows(
        database, sa.select(models.Visit).order_by(models.Visit.id)
    )
    assert [(v.page, v.referrer, v.utm) for v in visits] == [
        ("consent", "https://claude.ai/", {}),
        (
            "landing",
            None,
            {"utm_source": "newsletter", "utm_campaign": "launch"},
        ),
    ]


async def test_a_rephrased_request_counts_as_a_miss(make_client, database):
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    await client.call("show_reply", turn_id=plan["turn_id"], text="word " * 30)
    await client.call(  # The same question again, and no verdict.
        "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
    )
    inference = await scalar(
        database,
        sa.select(models.TurnInference).where(
            models.TurnInference.turn_id == turn_number(plan)
        ),
    )
    assert inference.next_topic_distance < params_lib.Params().rephrase_distance
    prior = stats.normalize(list(params_lib.REACTION_PRIOR))
    misunderstood = reaction.OUTCOMES.index(vocab.Outcome.MISUNDERSTOOD)
    probs = inference.reaction_probs
    assert probs[vocab.Outcome.MISUNDERSTOOD] > prior[misunderstood]


async def test_a_raised_point_names_reference_works(make_client, monkeypatch):
    monkeypatch.setattr(
        replies,
        "_bandit",
        lambda counts, kind, arms: (vocab.Expertise.RAISE, 1.0),
    )
    plan = (await (await make_client()).call("plan_reply", **PLAN))["data"]
    assert i18n.prompts()["expertise"] in plan["instruction"]


async def test_an_ablated_field_is_never_asked(make_client, monkeypatch):
    with_settings(
        monkeypatch, ablation_share=1.0, ablation_fields=["ambiguity"]
    )
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN, ambiguity=1.0))["data"]
    omit = i18n.prompts()["omit_field"].format(field="ambiguity")
    assert omit in plan["instruction"]
    assert i18n.prompts()["clarify_widget"] not in plan["instruction"]


async def test_a_candidate_catalog_is_launched_on_a_share(
    make_client, database, monkeypatch, tmp_path
):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"prompts": {"no_preamble": "Start with the answer."}})
    )
    monkeypatch.setattr(i18n, "CANDIDATE", candidate)
    i18n.prompts.cache_clear()
    i18n.catalog_version.cache_clear()
    with_settings(monkeypatch, candidate_share=1.0)
    try:
        plan = (await (await make_client()).call("plan_reply", **PLAN))["data"]
        assert "Start with the answer." in plan["instruction"]
        version = await scalar(database, sa.select(models.Turn.catalog_version))
        assert version == i18n.catalog_version(True) != i18n.catalog_version()
    finally:
        i18n.prompts.cache_clear()
        i18n.catalog_version.cache_clear()


async def test_a_class_that_ignores_a_request_stops_getting_it(
    make_client, database, monkeypatch
):
    with_settings(monkeypatch, message_sample=1.0)
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    for _ in range(int(nightly.MIN_WEIGHT)):  # Asked each time, never sent.
        plan = (
            await client.call(
                "plan_reply", **PLAN, conversation_id=plan["conversation_id"]
            )
        )["data"]
    async with database() as session, session.begin():
        p = await params_lib.load_params(session)
        await nightly.refit_compliance(session, p)
        await nightly.save_params(session, p)
    params_lib.cache.invalidate()
    family = f"{family_lib.POPULATION}:{vocab.Requested.USER_MESSAGE}"
    assert family in p.noncompliant
    plan = (await (await make_client()).call("plan_reply", **PLAN))["data"]
    assert i18n.prompts()["send_message"] not in plan["instruction"]


async def test_the_instruction_check_scores_an_assistant(database, monkeypatch):
    del database
    with_settings(
        monkeypatch,
        eval_models=["anthropic:claude-sonnet-5-5", "openai:gpt-test"],
        anthropic_api_key="key",
    )
    monkeypatch.setattr(
        instruction, "SCRIPTS", instruction.SCRIPTS.with_name("scripts.json")
    )
    state = {}

    def reply(request: httpx.Request) -> httpx.Response:
        """A compliant assistant: plan, then the reply in show_reply."""
        body = json.loads(request.content)
        last = body["messages"][-1]
        if isinstance(last["content"], str):  # A new user message.
            state["turn"] = None
            args = {**PLAN, "goal": "Learn"}
            if state.get("conversation"):
                args |= {
                    "conversation_id": state["conversation"],
                    "last_turn_id": state["last"],
                }
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "p",
                            "name": "plan_reply",
                            "input": args,
                        }
                    ]
                },
            )
        result = last["content"][0]["content"]
        if result.startswith("{"):
            plan = json.loads(result)
            state.update(
                conversation=plan["conversation_id"], last=plan["turn_id"]
            )
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "s",
                            "name": "show_reply",
                            "input": {"turn_id": plan["turn_id"], "text": "ok"},
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"content": []})

    transport = httpx.MockTransport(reply)
    async with httpx.AsyncClient(transport=transport) as http:
        state.clear()
        monkeypatch.setattr(
            instruction,
            "run_conversation",
            _fresh_state(instruction.run_conversation, state),
        )
        report = await instruction.evaluate(http)
    scores = report["anthropic:claude-sonnet-5-5"]
    assert scores["plan_reply_first"] == 1.0
    assert scores["reply_in_show_reply"] == 1.0
    assert scores["no_text_outside"] == 1.0
    assert scores["goal_on_first_message"] == 1.0
    assert scores["last_turn_id"] == 1.0
    assert report["openai:gpt-test"] == {"skipped": "no API key"}


def _fresh_state(run, state):
    """Starts each scripted conversation without the previous one's ids."""

    async def wrapped(*args, **kwargs):
        state.clear()
        return await run(*args, **kwargs)

    return wrapped


async def test_metrics_cover_every_diagnostic(http, make_client):
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN, ambiguity=0.7))["data"]
    await client.call("show_reply", turn_id=plan["turn_id"], text="word " * 30)
    await client.call(
        "plan_reply",
        **PLAN,
        conversation_id=plan["conversation_id"],
        goal_reached=True,
    )
    body = (await http.get("/metrics", headers=AUTH)).json()
    for key in (
        "fields",
        "field_ablation",
        "catalogs",
        "ambiguity",
        "funnel",
        "widget",
    ):
        assert key in body, key
    assert set(body["fields"]["all"]) == set(metrics.FIELDS)
    treated = body["by_group"][vocab.Arm.TREATMENT]
    assert treated["turns_over_token_budget"] is not None
    assert body["ambiguity"]["curve"]
