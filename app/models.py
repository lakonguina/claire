"""The database schema as typed SQLAlchemy models (SPEC 9).

Events are append-only (a trigger rejects UPDATE, DELETE and TRUNCATE; see the
initial migration). Derived tables are rebuilt nightly by replay.

Columns holding a vocabulary that evolves (buttons, request types, platforms)
are plain text: history keeps retired values, and the enums validate new
writes. Stable vocabularies (arms, reply modes, sources) are CHECK-constrained.
"""

import datetime
import enum
from typing import Any

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


class Base(orm.DeclarativeBase):
    """Declarative base, with Postgres-style constraint names."""

    metadata = sa.MetaData(naming_convention=NAMING)
    type_annotation_map = {
        str: sa.Text,
        dict[str, Any]: postgresql.JSONB,
        list[float]: postgresql.ARRAY(sa.Float),
        datetime.datetime: sa.DateTime(timezone=True),
    }


Id = orm.Mapped[int]
Now = sa.text("now()")


def _id() -> orm.MappedColumn:
    return orm.mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)


def _created() -> orm.MappedColumn:
    return orm.mapped_column(server_default=Now)


def _turn() -> orm.MappedColumn:
    return orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("turns.id"), index=True
    )


# Events.


class User(Base):
    """A person, known only by the hash of their connector token."""

    __tablename__ = "users"
    id: Id = _id()
    token_hash: orm.Mapped[str] = orm.mapped_column(unique=True)
    created_at: orm.Mapped[datetime.datetime] = _created()


class Conversation(Base):
    """One chat, with its comparison arm, host and mode."""

    __tablename__ = "conversations"
    id: Id = _id()
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("users.id")
    )
    grp: orm.Mapped[vocab.Arm] = orm.mapped_column(_checked(vocab.Arm))
    host: orm.Mapped[str]
    ui_supported: orm.Mapped[bool]
    # No longer asked (SPEC 6.3); kept for the rows that have it.
    assistant_model: orm.Mapped[str | None]
    protocol_version: orm.Mapped[str | None]
    created_at: orm.Mapped[datetime.datetime] = _created()


class Turn(Base):
    """One plan_reply call: the request and the decision taken."""

    __tablename__ = "turns"
    __table_args__ = (sa.Index(None, "conversation_id", "turn_index"),)
    id: Id = _id()
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("users.id")
    )
    conversation_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("conversations.id")
    )
    turn_index: orm.Mapped[int]
    created_at: orm.Mapped[datetime.datetime] = _created()
    reply_mode: orm.Mapped[vocab.ReplyMode] = orm.mapped_column(
        _checked(vocab.ReplyMode)
    )
    model_version: orm.Mapped[str]
    schema_version: orm.Mapped[int]
    raw_payload: orm.Mapped[dict[str, Any]]
    request_type: orm.Mapped[str]
    topic: orm.Mapped[str]
    needed_words: orm.Mapped[int]
    language: orm.Mapped[str]
    user_message_words: orm.Mapped[int | None]
    # No longer sent (SPEC 4.1); kept for the rows that have them.
    pasted_content: orm.Mapped[str | None]
    user_specified: orm.Mapped[list[str] | None] = orm.mapped_column(
        postgresql.ARRAY(sa.Text)
    )
    # No longer asked (SPEC 6.3); kept for the rows that have it.
    assistant_model: orm.Mapped[str | None]
    target_words: orm.Mapped[int | None]
    requested_words: orm.Mapped[int | None]
    propensity: orm.Mapped[float | None]
    theta_sample: orm.Mapped[float | None]
    b_sample: orm.Mapped[float | None]
    post_mean: orm.Mapped[float | None]
    post_sd: orm.Mapped[float | None]
    style: orm.Mapped[dict[str, Any]] = orm.mapped_column(
        server_default=sa.text("'{}'::jsonb")
    )
    transport_session: orm.Mapped[str | None]


class Reply(Base):
    """One show_reply call: the exact word count, never the text."""

    __tablename__ = "replies"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    reply_words_exact: orm.Mapped[int | None]
    created_at: orm.Mapped[datetime.datetime] = _created()


class Outcome(Base):
    """One observation of how the user reacted to a turn."""

    __tablename__ = "outcomes"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    source: orm.Mapped[vocab.Source] = orm.mapped_column(_checked(vocab.Source))
    observed: orm.Mapped[str]
    user_quote: orm.Mapped[str | None]
    last_reply_words: orm.Mapped[int | None]
    reading_seconds: orm.Mapped[float | None]
    format_issue: orm.Mapped[str | None]
    model_version: orm.Mapped[str]
    schema_version: orm.Mapped[int]
    created_at: orm.Mapped[datetime.datetime] = _created()


class ButtonShown(Base):
    """A button placed under a reply."""

    __tablename__ = "buttons_shown"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    button: orm.Mapped[str]
    position: orm.Mapped[int]
    shuffled: orm.Mapped[bool]
    thompson_draw: orm.Mapped[float | None]


class Click(Base):
    """A click on a shown button."""

    __tablename__ = "clicks"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    button: orm.Mapped[str]
    position: orm.Mapped[int]
    created_at: orm.Mapped[datetime.datetime] = _created()


class GoldLabel(Base):
    """The owner's true outcome of a turn."""

    __tablename__ = "gold_labels"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    outcome: orm.Mapped[str]
    created_at: orm.Mapped[datetime.datetime] = _created()


class ShadowDecision(Base):
    """What a candidate policy would have decided."""

    __tablename__ = "shadow_decisions"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    model_version: orm.Mapped[str]
    target_words: orm.Mapped[int | None]
    propensity: orm.Mapped[float | None]
    details: orm.Mapped[dict[str, Any]] = orm.mapped_column(
        server_default=sa.text("'{}'::jsonb")
    )


class ToolCall(Base):
    """One tool call, for coverage monitoring."""

    __tablename__ = "tool_calls"
    id: Id = _id()
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("users.id")
    )
    tool: orm.Mapped[str]
    created_at: orm.Mapped[datetime.datetime] = _created()


class TurnContext(Base):
    """The host context the widget reported."""

    __tablename__ = "turn_context"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    platform: orm.Mapped[str | None]
    locale: orm.Mapped[str | None]
    created_at: orm.Mapped[datetime.datetime] = _created()


class ProfileEdit(Base):
    """A correction made on the profile page."""

    __tablename__ = "profile_edits"
    id: Id = _id()
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, sa.ForeignKey("users.id")
    )
    scope: orm.Mapped[str]
    dimension: orm.Mapped[str]
    value: orm.Mapped[str]
    created_at: orm.Mapped[datetime.datetime] = _created()


class TopicEmbedding(Base):
    """The embedding of a turn's topic."""

    __tablename__ = "topic_embeddings"
    id: Id = _id()
    turn_id: orm.Mapped[int] = _turn()
    model: orm.Mapped[str]
    embedding: orm.Mapped[Any] = orm.mapped_column(
        pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSIONS)
    )


EVENTS = (
    User,
    Conversation,
    Turn,
    Reply,
    Outcome,
    ButtonShown,
    Click,
    GoldLabel,
    ShadowDecision,
    ToolCall,
    TurnContext,
    ProfileEdit,
    TopicEmbedding,
)

# Derived state.


class TurnInference(Base):
    """The fused length, reaction and reading of a closed turn."""

    __tablename__ = "turn_inference"
    turn_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, primary_key=True
    )
    length_mean: orm.Mapped[float]
    length_var: orm.Mapped[float]
    reaction_probs: orm.Mapped[dict[str, Any]]
    reading_probs: orm.Mapped[dict[str, Any]]


class ConversationState(Base):
    """The assistant's family, compliance and speed."""

    __tablename__ = "conversation_state"
    conversation_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, primary_key=True
    )
    family_probs: orm.Mapped[dict[str, Any]]
    kappa_mean: orm.Mapped[float]
    kappa_var: orm.Mapped[float]
    speed_mean: orm.Mapped[float]
    speed_var: orm.Mapped[float]


class ThetaPosterior(Base):
    """The grid posterior of a length cell."""

    __tablename__ = "theta_posteriors"
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, primary_key=True
    )
    # The length cell: a request type, or "<type>@mobile" (SPEC 6.5).
    request_type: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    grid: orm.Mapped[list[float]]


class AlphaPosterior(Base):
    """The grid posterior of a user's overall length ratio."""

    __tablename__ = "alpha_posteriors"
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, primary_key=True
    )
    grid: orm.Mapped[list[float]]


class UserReading(Base):
    """A user's reading speed, behaviour and reaction prior."""

    __tablename__ = "user_reading"
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, primary_key=True
    )
    rho_mean: orm.Mapped[float]
    rho_var: orm.Mapped[float]
    mixture: orm.Mapped[list[float]] = orm.mapped_column(postgresql.JSONB)
    reaction_prior: orm.Mapped[list[float]] = orm.mapped_column(
        postgresql.JSONB
    )


class PrefCount(Base):
    """Dirichlet counts of a preference, button or clarification."""

    __tablename__ = "pref_counts"
    # User 0 is the population; scope "*" is the user level (SPEC 7).
    user_id: orm.Mapped[int] = orm.mapped_column(
        sa.BigInteger, primary_key=True
    )
    scope: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    dimension: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    value: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    count: orm.Mapped[float]


class TopicCluster(Base):
    """A learned topic domain."""

    __tablename__ = "topic_clusters"
    id: orm.Mapped[int] = orm.mapped_column(sa.BigInteger, primary_key=True)
    centroid: orm.Mapped[Any] = orm.mapped_column(
        pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSIONS)
    )
    medoid_topic: orm.Mapped[str]
    size: orm.Mapped[int]
    updated_at: orm.Mapped[datetime.datetime] = _created()


DERIVED = (
    TurnInference,
    ConversationState,
    ThetaPosterior,
    AlphaPosterior,
    UserReading,
    PrefCount,
)

# Learned global parameters (refit nightly).


class FamilyParam(Base):
    """Learned behaviour of a model family."""

    __tablename__ = "family_params"
    model_family: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    b: orm.Mapped[float]
    w: orm.Mapped[float]
    kappa: orm.Mapped[float]
    speed_mu: orm.Mapped[float]
    speed_sd: orm.Mapped[float]
    updated_at: orm.Mapped[datetime.datetime] = _created()


class SensorNoise(Base):
    """Learned noise of a length sensor."""

    __tablename__ = "sensor_noise"
    sensor: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    model_family: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    sigma: orm.Mapped[float]
    updated_at: orm.Mapped[datetime.datetime] = _created()


class AnnotatorConfusion(Base):
    """Learned P(observed | true outcome) of an annotator."""

    __tablename__ = "annotator_confusion"
    annotator: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    stratum: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    true_outcome: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    observed: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    prob: orm.Mapped[float]
    updated_at: orm.Mapped[datetime.datetime] = _created()


class HostFamilyPrior(Base):
    """Learned P(family | host)."""

    __tablename__ = "host_family_prior"
    host: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    model_family: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    prob: orm.Mapped[float]
    updated_at: orm.Mapped[datetime.datetime] = _created()


class Hyperparam(Base):
    """A learned scalar hyperparameter."""

    __tablename__ = "hyperparams"
    name: orm.Mapped[str] = orm.mapped_column(primary_key=True)
    value: orm.Mapped[float]
    updated_at: orm.Mapped[datetime.datetime] = _created()


LEARNED = (
    FamilyParam,
    SensorNoise,
    AnnotatorConfusion,
    HostFamilyPrior,
    Hyperparam,
)
