"""OAuth connectors without accounts, end to end."""

from urllib import parse

from app.core import i18n
from app.routers import oauth
from tests import conftest

PLAN = {
    "topic": "oauth",
    "language": "en",
    "needed_words": 50,
}


async def test_the_consent_page_speaks_the_browser_language(http):
    client_id = await conftest.register(http)
    consent = await conftest.consent_url(http, client_id, "x" * 43)
    page = await http.get(consent, headers={"accept-language": "fr-FR"})
    title = i18n.catalog("fr")["consent"]["title"].format(app="Claude")
    assert page.status_code == 200 and title in page.text


async def test_mcp_without_a_token_starts_the_oauth_flow(http):
    response = await http.post("/mcp", json={})
    assert response.status_code == 401
    assert oauth.RESOURCE_METADATA in response.headers["www-authenticate"]
    path = parse.urlparse(oauth.RESOURCE_METADATA).path
    resource = (await http.get(path)).json()
    assert resource["resource"] == oauth.RESOURCE
    server = (await http.get("/.well-known/oauth-authorization-server")).json()
    assert server["registration_endpoint"].endswith("/register")
    bad = await http.post(
        "/mcp", json={}, headers={"authorization": "Bearer x"}
    )
    assert 'error="invalid_token"' in bad.headers["www-authenticate"]


async def test_an_app_connects_without_an_account(http):
    status, tokens = await conftest.redeem(
        http, *await conftest.authorize(http)
    )
    assert status == 200, tokens
    app = conftest.McpClient(http, tokens["access_token"], True)
    plan = await app.call("plan_reply", **PLAN)
    assert not plan.get("isError"), plan
    other = await conftest.redeem(http, *await conftest.authorize(http))
    other_user = await oauth.user_for_access_token(other[1]["access_token"])
    assert other_user != await app.user_id()  # Each connector, a profile.


async def test_a_replayed_code_disconnects_the_app(http):
    client_id, code, verifier = await conftest.authorize(http)
    _, tokens = await conftest.redeem(http, client_id, code, verifier)
    replayed, _ = await conftest.redeem(http, client_id, code, verifier)
    assert replayed == 400
    assert await oauth.user_for_access_token(tokens["access_token"]) is None


async def test_refresh_rotates_and_a_replay_disconnects(http):
    client_id, code, verifier = await conftest.authorize(http)
    _, first = await conftest.redeem(http, client_id, code, verifier)
    refresh = {
        "grant_type": "refresh_token",
        "refresh_token": first["refresh_token"],
    }
    status, second = await conftest.token(http, client_id, **refresh)
    assert status == 200
    user = await oauth.user_for_access_token(first["access_token"])
    assert await oauth.user_for_access_token(second["access_token"]) == user
    replayed, _ = await conftest.token(http, client_id, **refresh)
    assert replayed == 400
    assert await oauth.user_for_access_token(second["access_token"]) is None


async def test_revoking_disconnects_the_app(http):
    client_id, code, verifier = await conftest.authorize(http)
    _, tokens = await conftest.redeem(http, client_id, code, verifier)
    revoked = await http.post(
        "/revoke",
        content=parse.urlencode(
            {
                "client_id": client_id,
                "client_secret": "",  # The SDK requires the field.
                "token": tokens["refresh_token"],
            }
        ),
        headers=conftest.FORM,
    )
    assert revoked.status_code == 200, revoked.text
    assert await oauth.user_for_access_token(tokens["access_token"]) is None


async def test_refusing_returns_access_denied(http):
    client_id = await conftest.register(http)
    consent = await conftest.consent_url(http, client_id, "x" * 43)
    denied = await http.post(
        oauth.CONSENT_PATH,
        content=parse.urlencode(
            {"request": conftest.query(consent)["request"]}
        ),
        headers=conftest.FORM,
    )
    assert (
        conftest.query(denied.headers["location"])["error"] == "access_denied"
    )
    assert (await http.get(consent)).status_code == 400  # Used once.
