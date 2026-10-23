"""OAuth 2.1 for the MCP connector — no accounts, anonymous consent.

An AI host adds ``https://…/mcp``. The MCP SDK runs metadata, dynamic
client registration, PKCE and tokens against the Postgres storage below.
The consent page creates a new :class:`~app.models.Person`; the grant is
their only identity, and the host keeps the tokens.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import secrets
import urllib.parse
from pathlib import Path

import fastapi
import sqlalchemy as sa
from fastapi import responses
from fastapi.templating import Jinja2Templates
from mcp.server.auth import provider as auth
from mcp.server.auth import routes as auth_routes
from mcp.server.auth import settings as auth_settings
from mcp.shared import auth as shared_auth
from pydantic import AnyHttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from app import i18n
from app import models
from app import store
from app.core import config
from app.core import db
from app.core import geo
from app.core import vocab
from app.mcp.request_info import collect_http_request_info

LIFETIMES = {
    vocab.Credential.REQUEST: datetime.timedelta(minutes=10),
    vocab.Credential.CODE: datetime.timedelta(minutes=5),
    vocab.Credential.ACCESS: datetime.timedelta(hours=1),
    vocab.Credential.REFRESH: datetime.timedelta(days=90),
}
MCP_PATH = "/mcp"
CONSENT_PATH = "/oauth/consent"
PRIVACY_PATH = "/privacy"
ALLOW = "allow"
PERSON_COOKIE = "claire_person"
# Browsers cap cookie lifetime at 400 days.
PERSON_COOKIE_AGE = 60 * 60 * 24 * 400
_SECRET_BYTES = 32

_TEMPLATES = Jinja2Templates(
    directory=Path(__file__).resolve().parents[1] / "templates"
)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _public() -> str:
    return config.settings.public_url.rstrip("/")


def _new_secret() -> str:
    """Return a random unguessable token."""
    return secrets.token_urlsafe(_SECRET_BYTES)


def _person_cookie(person_id: int) -> str:
    """Signed person id, so a browser that connects again keeps its person."""
    signature = hmac.new(
        config.settings.data_key.encode(),
        str(person_id).encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{person_id}.{signature}"


def _cookie_person_id(value: str | None) -> int | None:
    """Return the person id of a valid :func:`_person_cookie`, else ``None``."""
    head = (value or "").partition(".")[0]
    if not head.isdigit():
        return None
    if not hmac.compare_digest(value or "", _person_cookie(int(head))):
        return None
    return int(head)


def _token_hash(token: str) -> str:
    """Return the stored hash of a token (cleartext is never stored)."""
    return hashlib.sha256(token.encode()).hexdigest()


SETTINGS = auth_settings.AuthSettings(
    issuer_url=AnyHttpUrl(_public()),
    resource_server_url=AnyHttpUrl(_public() + MCP_PATH),
    # Checked in :meth:`Provider.authorize` instead.
    validate_token_resource=False,
    client_registration_options=auth_settings.ClientRegistrationOptions(
        enabled=True
    ),
    revocation_options=auth_settings.RevocationOptions(enabled=True),
)
RESOURCE = str(SETTINGS.resource_server_url).rstrip("/")
RESOURCE_METADATA = str(auth_routes.build_resource_metadata_url(RESOURCE))


def _issue(
    session: AsyncSession,
    kind: vocab.Credential,
    client_id: str,
    grant_id: int | None,
    data: dict,
) -> str:
    """Store a new credential's hash; return the cleartext credential."""
    token = _new_secret()
    session.add(
        models.OAuthCredential(
            token_hash=_token_hash(token),
            kind=kind,
            client_id=client_id,
            grant_id=grant_id,
            data=data,
            expires_at=_now() + LIFETIMES[kind],
        )
    )
    return token


async def _live(
    session: AsyncSession,
    token: str,
    kind: vocab.Credential,
    client_id: str | None = None,
    *,
    spend: bool = False,
) -> tuple[models.OAuthCredential, models.OAuthGrant | None] | None:
    """Return a valid credential and its grant, or ``None``.

    A code or refresh token presented again after use was stolen or
    replayed: its whole grant is revoked (RFC 9700 §4.14).

    Args:
        session: Session inside a transaction.
        token: Credential as presented.
        kind: Kind it must be.
        client_id: Client it must belong to, if known.
        spend: Whether to mark it used (codes, refresh tokens, requests).

    Returns:
        The credential and its grant (``None`` for a pending request).
    """
    credential_t, grant_t = models.OAuthCredential, models.OAuthGrant
    statement = (
        sa.select(credential_t, grant_t)
        .outerjoin(grant_t)
        .where(
            credential_t.token_hash == _token_hash(token),
            credential_t.kind == kind,
            credential_t.expires_at > _now(),
        )
        .with_for_update(of=credential_t)
    )
    if client_id is not None:
        statement = statement.where(credential_t.client_id == client_id)
    row = (await session.execute(statement)).first()
    if row is None or (row.OAuthGrant and row.OAuthGrant.revoked_at):
        return None
    credential, grant = row
    if credential.used_at:
        if grant:
            grant.revoked_at = _now()
        return None
    if spend:
        credential.used_at = _now()
    return credential, grant


async def _gone(
    session: AsyncSession, request: str, texts: dict
) -> fastapi.HTTPException:
    """Say why a consent request is unusable: already answered, or expired."""
    answered = await session.scalar(
        sa.select(models.OAuthCredential.token_hash).where(
            models.OAuthCredential.token_hash == _token_hash(request),
            models.OAuthCredential.kind == vocab.Credential.REQUEST,
            models.OAuthCredential.used_at.is_not(None),
        )
    )
    return fastapi.HTTPException(
        status_code=400, detail=texts["used" if answered else "expired"]
    )


def _tokens(
    session: AsyncSession,
    client_id: str,
    grant: models.OAuthGrant,
    data: dict,
) -> shared_auth.OAuthToken:
    """Issue an access and a refresh token for a grant."""
    access = vocab.Credential.ACCESS
    return shared_auth.OAuthToken(
        access_token=_issue(session, access, client_id, grant.id, data),
        expires_in=int(LIFETIMES[access].total_seconds()),
        refresh_token=_issue(
            session, vocab.Credential.REFRESH, client_id, grant.id, data
        ),
        scope=" ".join(data["scopes"]) or None,
    )


def _token_data(scopes: list[str], resource: str | None) -> dict:
    return {"scopes": scopes, "resource": resource}


class Provider(
    auth.OAuthAuthorizationServerProvider[
        auth.AuthorizationCode, auth.RefreshToken, auth.AccessToken
    ]
):
    """MCP SDK authorization-server storage backed by Postgres."""

    async def get_client(
        self, client_id: str
    ) -> shared_auth.OAuthClientInformationFull | None:
        async with db.Session() as session:
            client = await session.get(models.OAuthClient, client_id)
        return (
            shared_auth.OAuthClientInformationFull.model_validate(client.info)
            if client
            else None
        )

    async def register_client(
        self, client_info: shared_auth.OAuthClientInformationFull
    ) -> None:
        async with db.Session() as session, session.begin():
            session.add(
                models.OAuthClient(
                    client_id=client_info.client_id,
                    info=client_info.model_dump(mode="json", exclude_none=True),
                )
            )

    async def authorize(
        self,
        client: shared_auth.OAuthClientInformationFull,
        params: auth.AuthorizationParams,
    ) -> str:
        """Hold the request until the person consents on our page."""
        if params.resource and params.resource.rstrip("/") != RESOURCE:
            raise auth.AuthorizeError("invalid_target")
        async with db.Session() as session, session.begin():
            request = _issue(
                session,
                vocab.Credential.REQUEST,
                client.client_id,
                None,
                params.model_dump(mode="json"),
            )
        query = urllib.parse.urlencode({"request": request})
        return f"{_public()}{CONSENT_PATH}?{query}"

    async def load_authorization_code(
        self,
        client: shared_auth.OAuthClientInformationFull,
        authorization_code: str,
    ) -> auth.AuthorizationCode | None:
        async with db.Session() as session, session.begin():
            found = await _live(
                session,
                authorization_code,
                vocab.Credential.CODE,
                client.client_id,
            )
        if found is None:
            return None
        credential, grant = found
        assert grant is not None
        return auth.AuthorizationCode(
            code=authorization_code,
            client_id=credential.client_id,
            expires_at=credential.expires_at.timestamp(),
            subject=str(grant.person_id),
            **credential.data,
        )

    async def exchange_authorization_code(
        self,
        client: shared_auth.OAuthClientInformationFull,
        authorization_code: auth.AuthorizationCode,
    ) -> shared_auth.OAuthToken:
        async with db.Session() as session, session.begin():
            found = await _live(
                session,
                authorization_code.code,
                vocab.Credential.CODE,
                client.client_id,
                spend=True,
            )
            if found is None:
                raise auth.TokenError("invalid_grant")
            return _tokens(
                session,
                client.client_id,
                found[1],
                _token_data(
                    authorization_code.scopes, authorization_code.resource
                ),
            )

    async def load_refresh_token(
        self, client: shared_auth.OAuthClientInformationFull, refresh_token: str
    ) -> auth.RefreshToken | None:
        async with db.Session() as session, session.begin():
            found = await _live(
                session,
                refresh_token,
                vocab.Credential.REFRESH,
                client.client_id,
            )
        if found is None:
            return None
        credential, grant = found
        assert grant is not None
        return auth.RefreshToken(
            token=refresh_token,
            client_id=credential.client_id,
            expires_at=int(credential.expires_at.timestamp()),
            subject=str(grant.person_id),
            **credential.data,
        )

    async def exchange_refresh_token(
        self,
        client: shared_auth.OAuthClientInformationFull,
        refresh_token: auth.RefreshToken,
        scopes: list[str],
    ) -> shared_auth.OAuthToken:
        """Rotate the pair: the old refresh token can never be used again."""
        async with db.Session() as session, session.begin():
            found = await _live(
                session,
                refresh_token.token,
                vocab.Credential.REFRESH,
                client.client_id,
                spend=True,
            )
            if found is None:
                raise auth.TokenError("invalid_grant")
            grant = found[1]
            assert grant is not None
            await store.append_event(
                session,
                person_id=grant.person_id,
                event_type=vocab.EventType.TOKEN_REFRESH,
                meta={"client_id": client.client_id, "via": "refresh"},
            )
            person = await session.get(models.Person, grant.person_id)
            if person is not None:
                person.last_seen_at = _now()
            return _tokens(
                session,
                client.client_id,
                grant,
                _token_data(scopes, refresh_token.resource),
            )

    async def load_access_token(self, token: str) -> auth.AccessToken | None:
        async with db.Session() as session, session.begin():
            found = await _live(session, token, vocab.Credential.ACCESS)
        if found is None:
            return None
        credential, grant = found
        assert grant is not None
        return auth.AccessToken(
            token=token,
            client_id=credential.client_id,
            expires_at=int(credential.expires_at.timestamp()),
            subject=str(grant.person_id),
            **credential.data,
        )

    async def revoke_token(
        self, token: auth.AccessToken | auth.RefreshToken
    ) -> None:
        """Disconnect the app: every credential of the grant dies."""
        async with db.Session() as session, session.begin():
            credential = await session.get(
                models.OAuthCredential, _token_hash(token.token)
            )
            if credential and credential.grant_id:
                grant = await session.get(
                    models.OAuthGrant, credential.grant_id
                )
                if grant is not None:
                    grant.revoked_at = grant.revoked_at or _now()


provider = Provider()


def routes() -> list:
    """Return OAuth metadata, registration, authorize and token routes."""
    issuer = SETTINGS.issuer_url
    return [
        *auth_routes.create_auth_routes(
            provider,
            issuer,
            client_registration_options=SETTINGS.client_registration_options,
            revocation_options=SETTINGS.revocation_options,
        ),
        *auth_routes.create_protected_resource_routes(
            SETTINGS.resource_server_url,
            [issuer],
            resource_name="Claire",
        ),
    ]


router = fastapi.APIRouter(include_in_schema=False)


def _texts(request: fastapi.Request) -> tuple[str, dict]:
    language = i18n.preferred(request.headers.get("accept-language"))
    return language, i18n.catalog(language)["consent"]


def consent_version(language: str) -> str:
    """Return a short version tag of the consent wording the person saw."""
    texts = i18n.catalog(language)
    digest = hashlib.sha256(
        json.dumps(
            {"consent": texts.get("consent", {}), "privacy": texts.get("privacy", {})},
            sort_keys=True,
        ).encode()
    )
    return f"{language}:{digest.hexdigest()[:12]}"


@router.get(PRIVACY_PATH, response_class=responses.HTMLResponse)
async def privacy_page(request: fastapi.Request) -> responses.HTMLResponse:
    """Short data & privacy notice (linked from consent)."""
    language = i18n.preferred(request.headers.get("accept-language"))
    texts = i18n.catalog(language)["privacy"]
    html = _TEMPLATES.get_template("privacy.html").render(
        language=language,
        texts=texts,
    )
    return responses.HTMLResponse(html)


@router.get(CONSENT_PATH, response_class=responses.HTMLResponse)
async def consent_page(
    request: fastapi.Request,
    pending: str = fastapi.Query(alias="request"),
) -> responses.HTMLResponse:
    """Ask the person to connect the host app; say what is stored."""
    language, texts = _texts(request)
    async with db.Session() as session, session.begin():
        found = await _live(session, pending, vocab.Credential.REQUEST)
        if found is None:
            raise await _gone(session, pending, texts)
    client = await provider.get_client(found[0].client_id)
    app_name = (
        (client.client_name if client else None)
        or urllib.parse.urlparse(str(found[0].data["redirect_uri"])).hostname
        or "AI"
    )
    html = _TEMPLATES.get_template("consent.html").render(
        language=language,
        texts=texts,
        app=app_name,
        request=pending,
        allow=ALLOW,
        policy=PRIVACY_PATH,
    )
    return responses.HTMLResponse(html)


@router.post(CONSENT_PATH)
async def consent(request: fastapi.Request) -> responses.RedirectResponse:
    """Send the host its code for this browser's person, or refuse.

    The person comes from the browser's cookie, or is created anonymous.
    """
    form = dict(urllib.parse.parse_qsl((await request.body()).decode()))
    language, texts = _texts(request)
    info = collect_http_request_info(request)
    if form.get("action") == ALLOW and form.get("accept") != "1":
        raise fastapi.HTTPException(
            status_code=400, detail=texts["need_accept"]
        )
    async with db.Session() as session, session.begin():
        found = await _live(
            session,
            form.get("request", ""),
            vocab.Credential.REQUEST,
            spend=True,
        )
        if found is None:
            raise await _gone(session, form.get("request", ""), texts)
        pending = found[0]
        params = auth.AuthorizationParams.model_validate(pending.data)
        back = str(params.redirect_uri)
        if form.get("action") != ALLOW:
            location = auth.construct_redirect_uri(
                back, error="access_denied", state=params.state
            )
            return responses.RedirectResponse(location, status_code=303)
        known = _cookie_person_id(request.cookies.get(PERSON_COOKIE))
        person = await session.get(models.Person, known) if known else None
        if person is None:
            person = await store.create_person(session)
        store.touch_person(person, info)
        person.country = geo.country(info.ip)
        grant = models.OAuthGrant(
            person_id=person.id,
            client_id=pending.client_id,
            consent_version=consent_version(language),
        )
        session.add(grant)
        await session.flush()
        await store.append_event(
            session,
            person_id=person.id,
            event_type=vocab.EventType.TOKEN_REFRESH,
            request=info,
            meta={
                "client_id": pending.client_id,
                "via": "consent",
                "consent_version": grant.consent_version,
            },
        )
        code = _issue(
            session,
            vocab.Credential.CODE,
            pending.client_id,
            grant.id,
            {
                "scopes": params.scopes or [],
                "code_challenge": params.code_challenge,
                "redirect_uri": back,
                "redirect_uri_provided_explicitly": (
                    params.redirect_uri_provided_explicitly
                ),
                "resource": params.resource,
            },
        )
    location = auth.construct_redirect_uri(back, code=code, state=params.state)
    response = responses.RedirectResponse(location, status_code=303)
    response.set_cookie(
        PERSON_COOKIE,
        _person_cookie(person.id),
        max_age=PERSON_COOKIE_AGE,
        httponly=True,
        secure=_public().startswith("https://"),
        samesite="lax",
    )
    return response
