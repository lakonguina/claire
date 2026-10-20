"""The database schema as typed SQLAlchemy models.

Events are append-only: the application never updates them, and deletes them
only for retention and data requests; the owner may correct them by hand.
Derived tables are rebuilt nightly by replay.

Columns holding a vocabulary that evolves (platforms, widget events) are
plain text: history keeps retired values, and the enums validate new writes.
Stable vocabularies (arms, reply modes, sources) are CHECK-constrained.
Every text a person or a reply wrote is encrypted by the application
(core/crypto.py): EncryptedText columns.
"""

# pylint cannot infer that orm.Mapped is generic under SQLAlchemy 2.
# pylint: disable=unsubscriptable-object

import datetime
import enum
import json
from typing import Annotated, Any

import numpy as np
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects import postgresql

from app.core import crypto
from app.core import vocab

EMBEDDING_DIMENSIONS = 384
NAMING = {
    "ix": "%(table_name)s_%(column_0_N_name)s_idx",
    "uq": "%(table_name)s_%(column_0_N_name)s_key",
    "ck": "%(table_name)s_%(constraint_name)s_check",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "pk": "%(table_name)s_pkey",
}


def _checked(enum_class: type[enum.StrEnum]) -> sa.Enum:
    """A text column constrained to an enum's values."""
    return sa.Enum(
        enum_class,
        name=enum_class.__name__.lower(),
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [m.value for m in members],
    )


Serial = Annotated[
    int, orm.mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
]
Key = Annotated[int, orm.mapped_column(sa.BigInteger, primary_key=True)]
TextKey = Annotated[str, orm.mapped_column(primary_key=True)]
Created = Annotated[
    datetime.datetime, orm.mapped_column(server_default=sa.text("now()"))
]
UserId = Annotated[
    int, orm.mapped_column(sa.BigInteger, sa.ForeignKey("users.id"))
]
TurnId = Annotated[
    int, orm.mapped_column(sa.BigInteger, sa.ForeignKey("turns.id"), index=True)
]
Grid = list[float]
Json = dict[str, Any]
Embedding = Annotated[
    Any, orm.mapped_column(pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSIONS))
]
EmptyJson = Annotated[
    Json, orm.mapped_column(server_default=sa.text("'{}'::jsonb"))
]


class FloatArray(sa.TypeDecorator):
    """A float8[] read as a numpy array, written from any sequence."""

    impl = postgresql.ARRAY(sa.Float)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else [float(v) for v in value]

    def process_result_value(self, value, dialect):
        return None if value is None else np.asarray(value, dtype=float)

    def compare_values(self, x, y):
        # The ARRAY default `x == y` broadcasts numpy arrays elementwise.
        if x is None or y is None:
            return x is y
        return np.array_equal(x, y)


class EncryptedText(sa.TypeDecorator):
    """A text stored as its AES-GCM ciphertext, read back as the text."""

    impl = postgresql.BYTEA
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else crypto.encrypt(value)

    def process_result_value(self, value, dialect):
        return None if value is None else crypto.decrypt(value)


class EncryptedJson(sa.TypeDecorator):
    """A JSON document stored as the AES-GCM ciphertext of its text."""

    impl = postgresql.BYTEA
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else crypto.encrypt(json.dumps(value))

    def process_result_value(self, value, dialect):
        return None if value is None else json.loads(crypto.decrypt(value))


class Base(orm.DeclarativeBase):
    """Declarative base, with Postgres-style constraint names."""

    metadata = sa.MetaData(naming_convention=NAMING)
    type_annotation_map = {
        str: sa.Text,
        Json: postgresql.JSONB,
        Grid: FloatArray,
        list[str]: postgresql.ARRAY(sa.Text),
        datetime.datetime: sa.DateTime(timezone=True),
    }


class _TurnRow(Base):
    """A row about one turn."""

    __abstract__ = True
    id: orm.Mapped[Serial]
    turn_id: orm.Mapped[TurnId]


class _TurnEvent(_TurnRow):
    """A timed event about one turn."""

    __abstract__ = True
    created_at: orm.Mapped[Created]


def _events(model: str) -> orm.Relationship:
    """A turn's events, read-only and loaded explicitly, in time order."""
    return orm.relationship(
        model, order_by=f"{model}.id", viewonly=True, lazy="raise"
    )


# Events.


class User(Base):
    """A person, anonymous: known only by their connector.

    An app reaches the user through an OAuth grant.
    """

    __tablename__ = "users"
    id: orm.Mapped[Serial]
    created_at: orm.Mapped[Created]


class OAuthClient(Base):
    """An AI app registered by OAuth dynamic client registration."""

    __tablename__ = "oauth_clients"
    client_id: orm.Mapped[TextKey]
    info: orm.Mapped[Json]  # The RFC 7591 client information, as sent.
    created_at: orm.Mapped[Created]


class OAuthGrant(Base):
    """One connector: an app authorized to act as a user, until revoked."""

    __tablename__ = "oauth_grants"
    id: orm.Mapped[Serial]
    user_id: orm.Mapped[UserId]
    client_id: orm.Mapped[str] = orm.mapped_column(
        sa.ForeignKey("oauth_clients.client_id")
    )
    created_at: orm.Mapped[Created]
    revoked_at: orm.Mapped[datetime.datetime | None]
    # The version of the consent wording the person accepted.
    consent_version: orm.Mapped[str]


class OAuthCredential(Base):
    """A pending request, code, access or refresh token, by its hash.

    A grant's credentials all die with it: a code or refresh token used
    twice revokes the grant (RFC 9700 4.14).
    """

    __tablename__ = "oauth_credentials"
    token_hash: orm.Mapped[TextKey]
    kind: orm.Mapped[vocab.Credential] = orm.mapped_column(
        _checked(vocab.Credential)
    )
    client_id: orm.Mapped[str] = orm.mapped_column(
        sa.ForeignKey("oauth_clients.client_id")
    )
    grant_id: orm.Mapped[int | None] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("oauth_grants.id"), index=True
    )
    data: orm.Mapped[Json]  # The SDK model fields not held in columns.
    expires_at: orm.Mapped[datetime.datetime]
    used_at: orm.Mapped[datetime.datetime | None]


class Conversation(Base):
    """One chat, with its comparison arm, host and mode."""

    __tablename__ = "conversations"
    id: orm.Mapped[Serial]
    user_id: orm.Mapped[UserId]
    grp: orm.Mapped[vocab.Arm] = orm.mapped_column(_checked(vocab.Arm))
    arm_propensity: orm.Mapped[float]
    host: orm.Mapped[str]
    host_version: orm.Mapped[str | None]
    capabilities: orm.Mapped[EmptyJson]  # As the client declared them.
    ui_supported: orm.Mapped[bool]
    protocol_version: orm.Mapped[str | None]
    # Whether its user messages are collected.
    sampled: orm.Mapped[bool]
    # The tool field never asked in it, to measure what the field is worth.
    ablated_field: orm.Mapped[str | None]
    ablation_propensity: orm.Mapped[float]
    # The prompt catalog its instructions come from.
    catalog: orm.Mapped[vocab.Catalog] = orm.mapped_column(
        _checked(vocab.Catalog)
    )
    catalog_propensity: orm.Mapped[float]
    created_at: orm.Mapped[Created]


class Turn(Base):
    """One plan_reply call: the request and the decision taken."""

    __tablename__ = "turns"
    __table_args__ = (sa.Index(None, "conversation_id", "turn_index"),)
    id: orm.Mapped[Serial]
    user_id: orm.Mapped[UserId]
    conversation_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("conversations.id")
    )
    turn_index: orm.Mapped[int]
    created_at: orm.Mapped[Created]
    reply_mode: orm.Mapped[vocab.ReplyMode] = orm.mapped_column(
        _checked(vocab.ReplyMode)
    )
    model_version: orm.Mapped[str]
    schema_version: orm.Mapped[int]
    catalog_version: orm.Mapped[str]  # Of the prompts the instruction uses.
    # Every argument as sent, encrypted: it holds the person's words.
    raw_payload: orm.Mapped[Json] = orm.mapped_column(EncryptedJson)
    instruction: orm.Mapped[str]  # Exactly as returned to the assistant.
    topic: orm.Mapped[str] = orm.mapped_column(EncryptedText)
    needed_words: orm.Mapped[int]
    language: orm.Mapped[str]
    user_message_words: orm.Mapped[int | None]
    ambiguity: orm.Mapped[float | None]
    # Replies the assistant gave since its previous call, without calling.
    messages_since_last_call: orm.Mapped[int | None]
    # The turn this call closed: the assistant's previous turn. branched when
    # it was not the latest one: the person edited or regenerated a message.
    follows: orm.Mapped[int | None] = orm.mapped_column(
        sa.BigInteger, index=True
    )
    branched: orm.Mapped[bool]
    target_words: orm.Mapped[int | None]
    requested_words: orm.Mapped[int | None]
    propensity: orm.Mapped[float | None]
    theta_sample: orm.Mapped[float | None]
    b_sample: orm.Mapped[float | None]
    post_mean: orm.Mapped[float | None]
    post_sd: orm.Mapped[float | None]
    style: orm.Mapped[EmptyJson]
    transport_session: orm.Mapped[str | None]

    replies: orm.Mapped[list["Reply"]] = _events("Reply")
    outcomes: orm.Mapped[list["Outcome"]] = _events("Outcome")
    widget_events: orm.Mapped[list["WidgetEvent"]] = _events("WidgetEvent")
    gold_labels: orm.Mapped[list["GoldLabel"]] = _events("GoldLabel")
    contexts: orm.Mapped[list["TurnContext"]] = _events("TurnContext")
    embeddings: orm.Mapped[list["TopicEmbedding"]] = _events("TopicEmbedding")
    shadow_decisions: orm.Mapped[list["ShadowDecision"]] = _events(
        "ShadowDecision"
    )


class Reply(_TurnEvent):
    """One show_reply call: the exact word count, and the text if consented."""

    __tablename__ = "replies"
    reply_words_exact: orm.Mapped[int | None]
    text: orm.Mapped[str | None] = orm.mapped_column(EncryptedText)
    choices: orm.Mapped[int]  # Options shown with a clarifying question.


class Message(_TurnEvent):
    """A user message the assistant copied, when the instruction asked.

    Longer messages than the setting are kept as their length only.
    """

    __tablename__ = "messages"
    words: orm.Mapped[int]
    text: orm.Mapped[str | None] = orm.mapped_column(EncryptedText)


class Goal(_TurnEvent):
    """A goal the person started on a turn, as the assistant stated it."""

    __tablename__ = "goals"
    conversation_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("conversations.id"), index=True
    )
    text: orm.Mapped[str] = orm.mapped_column(EncryptedText)


class Outcome(_TurnEvent):
    """One observation of how the user reacted to a turn."""

    __tablename__ = "outcomes"
    source: orm.Mapped[vocab.Source] = orm.mapped_column(_checked(vocab.Source))
    observed: orm.Mapped[str]
    user_quote: orm.Mapped[str | None] = orm.mapped_column(EncryptedText)
    last_reply_words: orm.Mapped[int | None]
    reading_seconds: orm.Mapped[float | None]
    format_issue: orm.Mapped[str | None]
    goal_reached: orm.Mapped[bool | None]
    model_version: orm.Mapped[str]
    schema_version: orm.Mapped[int]


class GoldLabel(_TurnEvent):
    """The owner's true outcome of a turn."""

    __tablename__ = "gold_labels"
    outcome: orm.Mapped[str]


class ShadowDecision(_TurnRow):
    """What a candidate policy would have decided."""

    __tablename__ = "shadow_decisions"
    model_version: orm.Mapped[str]
    target_words: orm.Mapped[int | None]
    propensity: orm.Mapped[float | None]
    details: orm.Mapped[EmptyJson]


class ToolCall(Base):
    """One tool call, for coverage monitoring."""

    __tablename__ = "tool_calls"
    id: orm.Mapped[Serial]
    user_id: orm.Mapped[UserId]
    tool: orm.Mapped[str]
    created_at: orm.Mapped[Created]


class TurnContext(_TurnEvent):
    """The host context the widget reported."""

    __tablename__ = "turn_context"
    platform: orm.Mapped[str | None]
    locale: orm.Mapped[str | None]
    time_zone: orm.Mapped[str | None]


class WidgetEvent(_TurnEvent):
    """What the widget saw: time on screen, a copy, an answered question."""

    __tablename__ = "widget_events"
    kind: orm.Mapped[str]
    value: orm.Mapped[EmptyJson]


class Visit(Base):
    """A page view of the landing or consent page: the acquisition funnel.

    No address or identifier: the page, where the visit came from, and its
    UTM parameters.
    """

    __tablename__ = "visits"
    id: orm.Mapped[Serial]
    page: orm.Mapped[str]
    referrer: orm.Mapped[str | None]
    utm: orm.Mapped[EmptyJson]
    created_at: orm.Mapped[Created]


class DeletionRequest(Base):
    """A person's data request, received by email, and when it was done."""

    __tablename__ = "deletion_requests"
    id: orm.Mapped[Serial]
    user_id: orm.Mapped[UserId]
    created_at: orm.Mapped[Created]
    done_at: orm.Mapped[datetime.datetime | None]


class Purge(Base):
    """One retention purge: texts older than `before` were deleted."""

    __tablename__ = "purges"
    id: orm.Mapped[Serial]
    before: orm.Mapped[datetime.datetime]
    rows: orm.Mapped[int]
    created_at: orm.Mapped[Created]


class TopicEmbedding(_TurnRow):
    """The embedding of a turn's topic."""

    __tablename__ = "topic_embeddings"
    model: orm.Mapped[str]
    embedding: orm.Mapped[Embedding]


# Derived state.


class TurnInference(Base):
    """The fused length, reaction and reading of a closed turn."""

    __tablename__ = "turn_inference"
    turn_id: orm.Mapped[Key]
    length_mean: orm.Mapped[float]
    length_var: orm.Mapped[float]
    theta_mean: orm.Mapped[float]  # The predicted preference, before it.
    reaction_probs: orm.Mapped[Json]
    reading_probs: orm.Mapped[Json]
    # Cosine distance from this topic to the next request's: a rephrase
    # when small.
    next_topic_distance: orm.Mapped[float | None]


class ConversationState(Base):
    """The assistant's behaviour class, compliance and speed."""

    __tablename__ = "conversation_state"
    conversation_id: orm.Mapped[Key]
    family_probs: orm.Mapped[Json]
    kappa_mean: orm.Mapped[float]
    kappa_var: orm.Mapped[float]
    speed_mean: orm.Mapped[float]
    speed_var: orm.Mapped[float]


class LengthPosterior(Base):
    """A user's Gaussian belief over the weights of the length features."""

    __tablename__ = "length_posteriors"
    user_id: orm.Mapped[Key]
    features: orm.Mapped[list[str]]
    mean: orm.Mapped[Grid]
    cov: orm.Mapped[Grid]  # Row-major, len(features) squared.


class UserReading(Base):
    """A user's reading speed, behaviour and reaction prior."""

    __tablename__ = "user_reading"
    user_id: orm.Mapped[Key]
    rho_mean: orm.Mapped[float]
    rho_var: orm.Mapped[float]
    mixture: orm.Mapped[list[float]] = orm.mapped_column(postgresql.JSONB)
    reaction_prior: orm.Mapped[list[float]] = orm.mapped_column(
        postgresql.JSONB
    )


class PrefCount(Base):
    """Dirichlet counts; user 0 is the population, scope "*" the user."""

    __tablename__ = "pref_counts"
    user_id: orm.Mapped[Key]
    scope: orm.Mapped[TextKey]
    dimension: orm.Mapped[TextKey]
    value: orm.Mapped[TextKey]
    count: orm.Mapped[float]


class TopicCluster(Base):
    """A learned topic domain."""

    __tablename__ = "topic_clusters"
    id: orm.Mapped[Key]
    centroid: orm.Mapped[Embedding]
    medoid_topic: orm.Mapped[str] = orm.mapped_column(EncryptedText)
    size: orm.Mapped[int]
    updated_at: orm.Mapped[Created]


# Learned global parameters (refit nightly).


class FamilyParam(Base):
    """Learned behaviour of a behaviour class."""

    __tablename__ = "family_params"
    model_family: orm.Mapped[TextKey]
    b: orm.Mapped[float]
    w: orm.Mapped[float]
    kappa: orm.Mapped[float]
    speed_mu: orm.Mapped[float]
    speed_sd: orm.Mapped[float]
    updated_at: orm.Mapped[Created]


class SensorNoise(Base):
    """Learned noise of a length sensor."""

    __tablename__ = "sensor_noise"
    sensor: orm.Mapped[TextKey]
    model_family: orm.Mapped[TextKey]
    sigma: orm.Mapped[float]
    updated_at: orm.Mapped[Created]


class AnnotatorConfusion(Base):
    """Learned P(observed | true outcome) of an annotator."""

    __tablename__ = "annotator_confusion"
    annotator: orm.Mapped[TextKey]
    stratum: orm.Mapped[TextKey]
    true_outcome: orm.Mapped[TextKey]
    observed: orm.Mapped[TextKey]
    prob: orm.Mapped[float]
    updated_at: orm.Mapped[Created]


class HostFamilyPrior(Base):
    """Learned P(behaviour class | host)."""

    __tablename__ = "host_family_prior"
    host: orm.Mapped[TextKey]
    model_family: orm.Mapped[TextKey]
    prob: orm.Mapped[float]
    updated_at: orm.Mapped[Created]


class Hyperparam(Base):
    """A learned scalar hyperparameter."""

    __tablename__ = "hyperparams"
    name: orm.Mapped[TextKey]
    value: orm.Mapped[float]
    updated_at: orm.Mapped[Created]


DERIVED = (
    TurnInference,
    ConversationState,
    LengthPosterior,
    UserReading,
    PrefCount,
)

LEARNED = (
    FamilyParam,
    SensorNoise,
    AnnotatorConfusion,
    HostFamilyPrior,
    Hyperparam,
)
