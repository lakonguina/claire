"""OAuth 2.1 connectors without accounts (SPEC 4.7).

An app adds https://<host>/mcp, the same URL for everyone. The MCP SDK's
handlers run the protocol (metadata, dynamic client registration, PKCE,
tokens) on the storage below. The consent page creates a new anonymous
user: the grant is the person's only identity, and the app keeps its tokens.
"""

import datetime
import urllib.parse

import fastapi
from fastapi import responses
from mcp.server.auth import provider as auth
from mcp.server.auth import routes as auth_routes
from mcp.server.auth import settings as auth_settings
from mcp.shared import auth as shared_auth
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config
from app import db
from app import i18n
from app import models
from app import service
from app import vocab

LIFETIMES = {
    vocab.Credential.REQUEST: datetime.timedelta(minutes=10),
    vocab.Credential.CODE: datetime.timedelta(minutes=5),
    vocab.Credential.ACCESS: datetime.timedelta(hours=1),
    vocab.Credential.REFRESH: datetime.timedelta(days=90),
}
MCP_PATH = "/mcp"
CONSENT_PATH = "/oauth/consent"
ALLOW = "allow"
Session = sa_asyncio.AsyncSession


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _public() -> str:
    return config.settings.public_url.rstrip("/")


SETTINGS = auth_settings.AuthSettings(
    issuer_url=_public(),
    resource_server_url=_public() + MCP_PATH,
    validate_token_resource=False,  # authorize() refuses other resources.
    client_registration_options=auth_settings.ClientRegistrationOptions(
        enabled=True
    ),
    revocation_options=auth_settings.RevocationOptions(enabled=True),
)
RESOURCE = str(SETTINGS.resource_server_url)
RESOURCE_METADATA = str(auth_routes.build_resource_metadata_url(RESOURCE))


def _issue(
    session: Session,
    kind: vocab.Credential,
    client_id: str,
    grant_id: int | None,
    data: dict,
) -> str:
    """Stores a new credential's hash; returns the credential."""
    token = service.new_secret()
    session.add(
        models.OAuthCredential(
            token_hash=service.token_hash(token),
            kind=kind,
            client_id=client_id,
            grant_id=grant_id,
            data=data,
            expires_at=_now() + LIFETIMES[kind],
        )
    )
    return token


async def _live(
    session: Session,
    token: str,
    kind: vocab.Credential,
    client_id: str | None = None,
    spend: bool = False,
) -> tuple[models.OAuthCredential, models.OAuthGrant | None] | None:
    """Returns a valid credential and its grant, or None.

    A code or refresh token presented again after its use was stolen or
    replayed: its whole grant is revoked (RFC 9700 4.14).

    Args:
        session: A session inside a transaction.
        token: The credential as presented.
        kind: The kind it must be.
        client_id: The client it must belong to, if known.
        spend: Whether to mark it used: codes, refresh tokens, requests.

    Returns:
        The credential and its grant (None for a pending request).
    """
    c, g = models.OAuthCredential, models.OAuthGrant
    statement = (
        sa.select(c, g)
        .outerjoin(g)
        .where(
            c.token_hash == service.token_hash(token),
            c.kind == kind,
            c.expires_at > _now(),
        )
        .with_for_update(of=c)
    )
    if client_id is not None:
        statement = statement.where(c.client_id == client_id)
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


def _tokens(
    session: Session, client_id: str, grant: models.OAuthGrant, data: dict
) -> shared_auth.OAuthToken:
    """Issues an access and a refresh token for a grant."""
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
    """The SDK's authorization server storage, in Postgres."""

    async def get_client(
        self, client_id: str
    ) -> shared_auth.OAuthClientInformationFull | None:
        async with db.session() as session:
            client = await session.get(models.OAuthClient, client_id)
        return (
            shared_auth.OAuthClientInformationFull.model_validate(client.info)
            if client
            else None
        )

    async def register_client(
        self, client_info: shared_auth.OAuthClientInformationFull
    ) -> None:
        async with db.session() as session, session.begin():
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
        """Holds the request until the person consents on our page."""
        if params.resource and params.resource.rstrip("/") != RESOURCE:
            raise auth.AuthorizeError("invalid_target")
        async with db.session() as session, session.begin():
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
        async with db.session() as session, session.begin():
            found = await _live(
                session,
                authorization_code,
                vocab.Credential.CODE,
                client.client_id,
            )
        if found is None:
            return None
        credential, grant = found
        return auth.AuthorizationCode(
            code=authorization_code,
            client_id=credential.client_id,
            expires_at=credential.expires_at.timestamp(),
            subject=str(grant.user_id),
            **credential.data,
        )

    async def exchange_authorization_code(
        self,
        client: shared_auth.OAuthClientInformationFull,
        authorization_code: auth.AuthorizationCode,
    ) -> shared_auth.OAuthToken:
        async with db.session() as session, session.begin():
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
        async with db.session() as session, session.begin():
            found = await _live(
                session,
                refresh_token,
                vocab.Credential.REFRESH,
                client.client_id,
            )
        if found is None:
            return None
        credential, grant = found
        return auth.RefreshToken(
            token=refresh_token,
            client_id=credential.client_id,
            expires_at=int(credential.expires_at.timestamp()),
            subject=str(grant.user_id),
            **credential.data,
        )

    async def exchange_refresh_token(
        self,
        client: shared_auth.OAuthClientInformationFull,
        refresh_token: auth.RefreshToken,
        scopes: list[str],
    ) -> shared_auth.OAuthToken:
        """Rotates the pair: the old refresh token can never be used again."""
        async with db.session() as session, session.begin():
            found = await _live(
                session,
                refresh_token.token,
                vocab.Credential.REFRESH,
                client.client_id,
                spend=True,
            )
            if found is None:
                raise auth.TokenError("invalid_grant")
            return _tokens(
                session,
                client.client_id,
                found[1],
                _token_data(scopes, refresh_token.resource),
            )

    async def load_access_token(self, token: str) -> auth.AccessToken | None:
        async with db.session() as session, session.begin():
            found = await _live(session, token, vocab.Credential.ACCESS)
        if found is None:
            return None
        credential, grant = found
        return auth.AccessToken(
            token=token,
            client_id=credential.client_id,
            expires_at=int(credential.expires_at.timestamp()),
            subject=str(grant.user_id),
            **credential.data,
        )

    async def revoke_token(
        self, token: auth.AccessToken | auth.RefreshToken
    ) -> None:
        """Disconnects the app: every credential of the grant dies."""
        async with db.session() as session, session.begin():
            credential = await session.get(
                models.OAuthCredential, service.token_hash(token.token)
            )
            if credential and credential.grant_id:
                grant = await session.get(
                    models.OAuthGrant, credential.grant_id
                )
                grant.revoked_at = grant.revoked_at or _now()


provider = Provider()


def routes() -> list:
    """Returns the OAuth metadata, registration, authorize and token routes."""
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
            resource_name="Lingua Franca",
        ),
    ]


async def user_for_access_token(token: str) -> int | None:
    """Returns the user an access token acts for, or None."""
    access = await provider.load_access_token(token)
    return int(access.subject) if access else None


def challenge(invalid: bool) -> str:
    """Returns the WWW-Authenticate header that starts the OAuth flow."""
    error = 'error="invalid_token", ' if invalid else ""
    return f'Bearer {error}resource_metadata="{RESOURCE_METADATA}"'


router = fastapi.APIRouter(include_in_schema=False)


def _texts(request: fastapi.Request) -> tuple[str, dict]:
    language = i18n.preferred(request.headers.get("accept-language"))
    return language, i18n.catalog(language)["consent"]


@router.get(CONSENT_PATH, response_class=responses.HTMLResponse)
async def consent_page(
    request: fastapi.Request,
    pending: str = fastapi.Query(alias="request"),
) -> str:
    """Asks the person to connect the app; nothing to fill in."""
    language, texts = _texts(request)
    async with db.session() as session, session.begin():
        found = await _live(session, pending, vocab.Credential.REQUEST)
    if found is None:
        raise fastapi.HTTPException(status_code=400, detail=texts["expired"])
    client = await provider.get_client(found[0].client_id)
    app_name = (
        client.client_name
        or urllib.parse.urlparse(str(found[0].data["redirect_uri"])).hostname
    )
    return i18n.TEMPLATES.get_template("consent.html").render(
        language=language,
        texts=texts,
        app=app_name,
        request=pending,
        allow=ALLOW,
    )


@router.post(CONSENT_PATH)
async def consent(request: fastapi.Request) -> responses.RedirectResponse:
    """Creates the anonymous user and sends the app its code, or a refusal."""
    form = dict(urllib.parse.parse_qsl((await request.body()).decode()))
    async with db.session() as session, session.begin():
        found = await _live(
            session,
            form.get("request", ""),
            vocab.Credential.REQUEST,
            spend=True,
        )
        if found is None:
            raise fastapi.HTTPException(
                status_code=400, detail=_texts(request)[1]["expired"]
            )
        pending = found[0]
        params = auth.AuthorizationParams.model_validate(pending.data)
        back = str(params.redirect_uri)
        if form.get("action") != ALLOW:
            location = auth.construct_redirect_uri(
                back, error="access_denied", state=params.state
            )
            return responses.RedirectResponse(location, status_code=303)
        user = service.new_user()
        session.add(user)
        await session.flush()
        grant = models.OAuthGrant(user_id=user.id, client_id=pending.client_id)
        session.add(grant)
        await session.flush()
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
    return responses.RedirectResponse(location, status_code=303)
