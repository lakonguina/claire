"""System tests through the MCP endpoint and the HTTP routes (SPEC 12)."""

import dataclasses
import logging

from app import config
from app import db
from app import i18n
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


async def fetch(pool, sql: str, *args) -> list[dict]:
    async with pool.connection() as conn:
        return await db.fetch(conn, sql, *args)


async def fetchrow(pool, sql: str, *args) -> dict | None:
    async with pool.connection() as conn:
        return await db.fetchrow(conn, sql, *args)


async def dump(pool) -> str:
    """Returns every row of every table, as one string."""
    tables = await fetch(
        pool, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )
    return "\n".join(
        [
            str(await fetch(pool, f"SELECT * FROM {t["tablename"]}"))
            for t in tables
        ]
    )


async def pref_count(
    pool, user_id: int, scope: str, dimension: str, value: str
) -> float:
    found = await fetchrow(
        pool,
        "SELECT count FROM pref_counts WHERE user_id = %s AND "
        "scope = %s AND dimension = %s AND value = %s",
        user_id,
        scope,
        dimension,
        value,
    )
    return found["count"] if found else 0.0


def with_settings(monkeypatch, **changes) -> None:
    monkeypatch.setattr(
        config, "settings", dataclasses.replace(config.settings, **changes)
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
    make_client, pool, caplog
):
    caplog.set_level(logging.DEBUG)
    client = await make_client()
    plan = (
        await client.call(
            "plan_reply", **PLAN, assistant_model="claude-opus-5-5"
        )
    )["data"]
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
    reply = await fetchrow(
        pool,
        "SELECT reply_words_exact FROM replies WHERE turn_id = %s",
        turn_id,
    )
    assert reply["reply_words_exact"] == 6
    for table in ("turn_inference", "topic_embeddings"):
        assert await fetchrow(
            pool, f"SELECT 1 FROM {table} WHERE turn_id = %s", turn_id
        )
    server_logs = "\n".join(
        r.getMessage() for r in caplog.records if r.name != "httpx"
    )
    assert MARKER not in await dump(pool)
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


async def test_mode_selection_in_both_protocol_eras(make_client, pool):
    for version in (conftest.LEGACY, conftest.MODERN):
        for apps, mode in (
            (True, vocab.ReplyMode.WIDGET),
            (False, vocab.ReplyMode.FALLBACK),
        ):
            client = await make_client(apps=apps, version=version)
            plan = (await client.call("plan_reply", **PLAN))["data"]
            assert plan["instruction"].endswith(WIDGET_SENTENCE) == apps
            found = await fetchrow(
                pool,
                "SELECT t.reply_mode, c.host FROM turns t JOIN "
                "conversations c ON c.id = t.conversation_id WHERE t.id = %s",
                turn_number(plan),
            )
            assert (found["reply_mode"], found["host"]) == (mode, "claude-ai")


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


async def test_repair_without_quote_is_no_signal(make_client, pool):
    plan, _ = await plan_and_close(
        await make_client(), last_reply_outcome="too_long"
    )
    found = await fetchrow(
        pool,
        "SELECT observed FROM outcomes WHERE turn_id = %s",
        turn_number(plan),
    )
    assert found["observed"] == vocab.Outcome.NO_SIGNAL


async def test_arms_share_and_behaviour(make_client, monkeypatch):
    with_settings(
        monkeypatch, arm_shares={vocab.Arm.CONTROL: 0.1, vocab.Arm.CONCISE: 0.1}
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


async def test_click_drops_the_assistant_verdict(make_client, pool):
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
        probs[clicked] = (
            await fetchrow(
                pool,
                "SELECT reaction_probs FROM turn_inference WHERE turn_id = %s",
                turn_number(plan),
            )
        )["reaction_probs"]
    assert probs[False][vocab.Outcome.TOO_SHORT] > 0.5
    assert probs[True][vocab.Outcome.TOO_SHORT] < 0.2


async def test_format_issue_moves_the_preference(make_client, pool):
    client = await make_client()
    await plan_and_close(
        client,
        last_reply_outcome="wrong_format",
        user_quote="give me a list",
        format_issue="want_list",
    )
    count = await pref_count(
        pool,
        await client.user_id(),
        PLAN["request_type"],
        vocab.Dimension.FORMAT,
        vocab.PrefValue.BULLETS,
    )
    assert count >= prefs.EXPLICIT


async def test_preference_button_moves_the_preference(make_client, pool):
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
        pool, await client.user_id(), prefs.USER_SCOPE, dimension, value
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
    instructions = {
        (await client.call("plan_reply", **PLAN, ambiguous=True))["data"][
            "instruction"
        ].removesuffix(" " + WIDGET_SENTENCE)
        for _ in range(30)
    }
    assert i18n.prompts()["clarify"] in instructions
    assert len(instructions) > 1


async def test_mobile_report_moves_the_length_cell(make_client, pool):
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
    found = await fetchrow(
        pool, "SELECT style FROM turns WHERE id = %s", turn_number(nxt)
    )
    assert found["style"]["cell"] == "explanation@mobile"


async def test_profile_page_lists_and_applies_corrections(
    http, make_client, pool
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
        pool,
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


async def test_replay_rebuilds_the_online_state(http, make_client, pool):
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

    async def snapshot() -> dict:
        return {
            t: sorted(map(str, await fetch(pool, f"SELECT * FROM {t}")))
            for t in nightly.DERIVED
        }

    online = await snapshot()
    async with pool.connection() as conn:
        p = await params_lib.load_params(conn)
        assert await nightly.replay(conn, p) == 9
    assert await snapshot() == online


async def test_nightly_job_runs(make_client, pool):
    await plan_and_close(await make_client())
    async with pool.connection() as conn:
        assert await nightly.run(conn) == 1
        assert await db.fetchrow(
            conn, "SELECT value FROM hyperparams WHERE name = 'mu0'"
        )


async def test_policy_swap_through_configuration(
    make_client, pool, monkeypatch
):
    with_settings(monkeypatch, policy="fixed-v1")
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    assert plan["target_words"] == 160  # 0.8 x 200.
    found = await fetchrow(pool, "SELECT model_version FROM turns")
    assert found["model_version"] == "fixed-v1"


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
    turns = (
        await http.get(
            "/admin/api/turns",
            params={
                "conversation": plan["conversation_id"].removeprefix(
                    service.CONVERSATION_PREFIX
                )
            },
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


async def test_arm_check_matches_the_vocabulary(pool):
    found = await fetchrow(
        pool,
        "SELECT pg_get_constraintdef(oid) AS check FROM pg_constraint "
        "WHERE conname = 'conversations_grp_check'",
    )
    assert all(f"'{arm}'" in found["check"] for arm in vocab.Arm)
