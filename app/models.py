"""The database schema as typed SQLAlchemy models (SPEC 9).

Events are append-only (a trigger rejects UPDATE, DELETE and TRUNCATE; see the
initial migration). Derived tables are rebuilt nightly by replay.

Columns holding a vocabulary that evolves (buttons, request types, platforms)
are plain text: history keeps retired values, and the enums validate new
writes. Stable vocabularies (arms, reply modes, sources) are CHECK-constrained.
"""

# pylint cannot infer that orm.Mapped is generic under SQLAlchemy 2.
# pylint: disable=unsubscriptable-object

import datetime
import enum
from typing import Annotated, Any

import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects import postgresql

from app import vocab

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


class Base(orm.DeclarativeBase):
    """Declarative base, with Postgres-style constraint names."""

    metadata = sa.MetaData(naming_convention=NAMING)
    type_annotation_map = {
        str: sa.Text,
        Json: postgresql.JSONB,
        Grid: postgresql.ARRAY(sa.Float),
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
    """A person, known only by the hash of their connector token."""

    __tablename__ = "users"
    id: orm.Mapped[Serial]
    token_hash: orm.Mapped[str] = orm.mapped_column(unique=True)
    created_at: orm.Mapped[Created]


class Conversation(Base):
    """One chat, with its comparison arm, host and mode."""

    __tablename__ = "conversations"
    id: orm.Mapped[Serial]
    user_id: orm.Mapped[UserId]
    grp: orm.Mapped[vocab.Arm] = orm.mapped_column(_checked(vocab.Arm))
    host: orm.Mapped[str]
    ui_supported: orm.Mapped[bool]
    # No longer asked (SPEC 6.3); kept for the rows that have it.
    assistant_model: orm.Mapped[str | None]
    protocol_version: orm.Mapped[str | None]
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
    raw_payload: orm.Mapped[Json]
    request_type: orm.Mapped[str]
    topic: orm.Mapped[str]
    needed_words: orm.Mapped[int]
    language: orm.Mapped[str]
    user_message_words: orm.Mapped[int | None]
    # No longer sent (SPEC 4.1 and 6.3); kept for the rows that have them.
    pasted_content: orm.Mapped[str | None]
    user_specified: orm.Mapped[list[str] | None] = orm.mapped_column(
        postgresql.ARRAY(sa.Text)
    )
    assistant_model: orm.Mapped[str | None]
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
    buttons_shown: orm.Mapped[list["ButtonShown"]] = _events("ButtonShown")
    clicks: orm.Mapped[list["Click"]] = _events("Click")
    gold_labels: orm.Mapped[list["GoldLabel"]] = _events("GoldLabel")
    contexts: orm.Mapped[list["TurnContext"]] = _events("TurnContext")
    embeddings: orm.Mapped[list["TopicEmbedding"]] = _events("TopicEmbedding")
    shadow_decisions: orm.Mapped[list["ShadowDecision"]] = _events(
        "ShadowDecision"
    )


class Reply(_TurnEvent):
    """One show_reply call: the exact word count, never the text."""

    __tablename__ = "replies"
    reply_words_exact: orm.Mapped[int | None]


class Outcome(_TurnEvent):
    """One observation of how the user reacted to a turn."""

    __tablename__ = "outcomes"
    source: orm.Mapped[vocab.Source] = orm.mapped_column(_checked(vocab.Source))
    observed: orm.Mapped[str]
    user_quote: orm.Mapped[str | None]
    last_reply_words: orm.Mapped[int | None]
    reading_seconds: orm.Mapped[float | None]
    format_issue: orm.Mapped[str | None]
    model_version: orm.Mapped[str]
    schema_version: orm.Mapped[int]


class ButtonShown(_TurnRow):
    """A button placed under a reply."""

    __tablename__ = "buttons_shown"
    button: orm.Mapped[str]
    position: orm.Mapped[int]
    shuffled: orm.Mapped[bool]
    thompson_draw: orm.Mapped[float | None]


class Click(_TurnEvent):
    """A click on a shown button."""

    __tablename__ = "clicks"
    button: orm.Mapped[str]
    position: orm.Mapped[int]


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


class ProfileEdit(Base):
    """A correction made on the profile page."""

    __tablename__ = "profile_edits"
    id: orm.Mapped[Serial]
    user_id: orm.Mapped[UserId]
    scope: orm.Mapped[str]
    dimension: orm.Mapped[str]
    value: orm.Mapped[str]
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
    reaction_probs: orm.Mapped[Json]
    reading_probs: orm.Mapped[Json]


class ConversationState(Base):
    """The assistant's behaviour class, compliance and speed."""

    __tablename__ = "conversation_state"
    conversation_id: orm.Mapped[Key]
    family_probs: orm.Mapped[Json]
    kappa_mean: orm.Mapped[float]
    kappa_var: orm.Mapped[float]
    speed_mean: orm.Mapped[float]
    speed_var: orm.Mapped[float]


class ThetaPosterior(Base):
    """The grid posterior of a length cell ("<type>" or "<type>@mobile")."""

    __tablename__ = "theta_posteriors"
    user_id: orm.Mapped[Key]
    request_type: orm.Mapped[TextKey]
    grid: orm.Mapped[Grid]


class AlphaPosterior(Base):
    """The grid posterior of a user's overall length ratio."""

    __tablename__ = "alpha_posteriors"
    user_id: orm.Mapped[Key]
    grid: orm.Mapped[Grid]


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
    medoid_topic: orm.Mapped[str]
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
    ThetaPosterior,
    AlphaPosterior,
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
