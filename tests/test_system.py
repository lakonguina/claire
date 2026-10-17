import dataclasses
import logging

from app import nightly, service
from app.config import settings
from app.db import fetch, fetchrow
from app.params import load_params
from tests.conftest import LEGACY, MODERN

MARKER = "SECRET_REPLY_MARKER_93f1"
PLAN = {"request_type": "explanation", "topic": "how to deploy the server", "needed_words": 200}


async def dump(db) -> str:
    """Every row of every table, as one string."""
    async with db.connection() as conn:
        tables = [r["tablename"] for r in await fetch(conn, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")]
        return "\n".join([str(await fetch(conn, f"SELECT * FROM {t}")) for t in tables])


async def test_full_turn_sequence_and_no_trace_of_text(make_client, db, caplog):
    caplog.set_level(logging.DEBUG)
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN, assistant_model="claude-opus-5-5"))["data"]
    assert plan["instruction"].endswith("Write the reply only in show_reply.")
    shown = await client.call("show_reply", turn_id=plan["turn_id"], text=f"# Answer\n\n{MARKER} is three words long.")
    assert shown["content"] == [{"type": "text", "text": "ok"}]
    buttons = shown["structuredContent"]["buttons"]
    if buttons:  # treatment
        b = buttons[0]
        assert (await client.call("record_click", turn_id=plan["turn_id"], button=b["id"], position=b["position"]))
    nxt = (await client.call("plan_reply", **PLAN, conversation_id=plan["conversation_id"],
                             last_reply_outcome="too_long", user_quote="Shorter."))["data"]
    assert nxt["conversation_id"] == plan["conversation_id"]

    async with db.connection() as conn:
        turn_id = int(plan["turn_id"][2:])
        reply = await fetchrow(conn, "SELECT reply_words_exact FROM replies WHERE turn_id = %s", turn_id)
        assert reply["reply_words_exact"] == 6
        assert await fetchrow(conn, "SELECT 1 FROM turn_inference WHERE turn_id = %s", turn_id)
        assert await fetchrow(conn, "SELECT 1 FROM theta_posteriors")
    server_logs = "\n".join(r.getMessage() for r in caplog.records if r.name != "httpx")  # httpx is the test client
    assert MARKER not in await dump(db)
    assert MARKER not in server_logs
    assert client.url.split("/")[-1] not in server_logs  # the token never reaches the logs
    assert "/mcp/<redacted>" in server_logs


async def test_show_reply_errors_hide_the_text(make_client, db, caplog):
    client = await make_client()
    for args in ({"text": MARKER}, {"turn_id": "t_999", "text": MARKER}, {"turn_id": "x", "text": [MARKER]}):
        result = await client.call("show_reply", **args)
        assert result["isError"] and MARKER not in str(result)
    assert MARKER not in caplog.text


async def test_mode_selection_in_both_protocol_eras(make_client, db):
    for version in (LEGACY, MODERN):
        for apps, suffix, mode in ((True, "only in show_reply.", "widget"), (False, "call show_feedback.", "fallback")):
            client = await make_client(apps=apps, version=version)
            plan = (await client.call("plan_reply", **PLAN))["data"]
            assert plan["instruction"].endswith(suffix), (version, apps)
            async with db.connection() as conn:
                row = await fetchrow(conn, """SELECT t.reply_mode, c.host FROM turns t JOIN conversations c
                                              ON c.id = t.conversation_id WHERE t.id = %s""", int(plan["turn_id"][2:]))
            assert (row["reply_mode"], row["host"]) == (mode, "claude-ai")


async def test_foreign_handles_are_rejected(make_client):
    alice, bob = await make_client(), await make_client()
    plan = (await alice.call("plan_reply", **PLAN))["data"]
    assert (await bob.call("show_reply", turn_id=plan["turn_id"], text="hi"))["isError"]
    assert (await bob.call("show_feedback", turn_id=plan["turn_id"]))["isError"]
    assert (await bob.call("plan_reply", **PLAN, conversation_id=plan["conversation_id"]))["isError"]


async def test_unknown_token_is_404(http):
    r = await http.post("/mcp/not-a-token", json={})
    assert r.status_code == 404


async def test_repair_without_quote_is_no_signal(make_client, db):
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    await client.call("plan_reply", **PLAN, conversation_id=plan["conversation_id"], last_reply_outcome="too_long")
    async with db.connection() as conn:
        row = await fetchrow(conn, "SELECT observed FROM outcomes WHERE turn_id = %s", int(plan["turn_id"][2:]))
    assert row["observed"] == "no_signal"


async def test_control_group_share_and_behaviour(make_client, db):
    client = await make_client()
    plans = [(await client.call("plan_reply", **PLAN))["data"] for _ in range(300)]
    control = [p for p in plans if p["target_words"] is None]
    assert 0.05 <= len(control) / len(plans) <= 0.16
    for p in control[:5]:
        assert p["instruction"] == "Reply normally. Write the reply only in show_reply."
        assert (await client.call("show_reply", turn_id=p["turn_id"], text="x"))["structuredContent"]["buttons"] == []


async def test_click_drops_the_assistant_verdict(make_client, db, monkeypatch):
    monkeypatch.setattr(service, "settings", dataclasses.replace(settings, control_share=0.0))
    client = await make_client()
    probs = {}
    for clicked in (False, True):  # the assistant says too_short both times; only the second turn has a click
        plan = (await client.call("plan_reply", **PLAN))["data"]
        buttons = (await client.call("show_reply", turn_id=plan["turn_id"], text="one two"))["structuredContent"]["buttons"]
        if clicked:
            b = next(b for b in buttons if b["id"] != "more_detail")  # a click that does not mean too_short
            await client.call("record_click", turn_id=plan["turn_id"], button=b["id"], position=b["position"])
        await client.call("plan_reply", **PLAN, conversation_id=plan["conversation_id"],
                          last_reply_outcome="too_short", user_quote="tell me more")
        async with db.connection() as conn:
            probs[clicked] = (await fetchrow(conn, "SELECT reaction_probs FROM turn_inference WHERE turn_id = %s",
                                             int(plan["turn_id"][2:])))["reaction_probs"]
    assert probs[False]["too_short"] > 0.5
    assert probs[True]["too_short"] < 0.2


async def test_replay_rebuilds_the_online_state(make_client, db):
    client = await make_client()
    for _ in range(3):
        plan = (await client.call("plan_reply", **PLAN))["data"]
        for outcome in ("too_long", "moved_on", "too_short"):
            await client.call("show_reply", turn_id=plan["turn_id"], text="word " * 40)
            plan = (await client.call("plan_reply", **PLAN, conversation_id=plan["conversation_id"],
                                      last_reply_outcome=outcome, user_quote="ok then"))["data"]
    tables = ["turn_inference", "conversation_state", "theta_posteriors", "alpha_posteriors", "user_reading", "beta_counts"]

    async def snapshot():
        async with db.connection() as conn:
            return {t: sorted(map(str, await fetch(conn, f"SELECT * FROM {t}"))) for t in tables}

    online = await snapshot()
    async with db.connection() as conn:
        assert await nightly.replay(conn, await load_params(conn)) == 9
    assert await snapshot() == online


async def test_nightly_job_runs(make_client, db):
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    await client.call("plan_reply", **PLAN, conversation_id=plan["conversation_id"])
    async with db.connection() as conn:
        assert await nightly.run(conn) == 1
        assert await fetchrow(conn, "SELECT value FROM hyperparams WHERE name = 'mu0'")


async def test_policy_swap_through_configuration(make_client, db, monkeypatch):
    monkeypatch.setattr(service, "settings", dataclasses.replace(settings, control_share=0.0, policy="fixed-v1"))
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    assert plan["target_words"] == 160  # 0.8 x 200
    async with db.connection() as conn:
        assert (await fetchrow(conn, "SELECT model_version FROM turns"))["model_version"] == "fixed-v1"


async def test_admin_endpoints(http, make_client):
    client = await make_client()
    plan = (await client.call("plan_reply", **PLAN))["data"]
    await client.call("plan_reply", **PLAN, conversation_id=plan["conversation_id"])
    auth = {"authorization": "Bearer admin-secret"}
    assert (await http.get("/metrics")).status_code == 401
    assert (await http.post("/gold_label", json={"turn_id": plan["turn_id"], "outcome": "moved_on"},
                            headers=auth)).status_code == 200
    assert "moved_on" in (await http.get("/debug", headers=auth)).text
    body = (await http.get("/metrics", headers=auth)).json()
    assert body["tool_coverage"]["turns"] == 2
