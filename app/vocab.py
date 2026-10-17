"""Closed vocabularies: the single source of truth for every identifier."""

import enum


class RequestType(enum.StrEnum):
    FACT = enum.auto()
    EXPLANATION = enum.auto()
    ADVICE = enum.auto()
    DECISION = enum.auto()
    CODE = enum.auto()
    WRITING = enum.auto()
    PLAN = enum.auto()


class Outcome(enum.StrEnum):
    MOVED_ON = enum.auto()
    TOO_LONG = enum.auto()
    TOO_SHORT = enum.auto()
    MISSING_INFO = enum.auto()
    MISUNDERSTOOD = enum.auto()
    WRONG_FORMAT = enum.auto()
    NO_SIGNAL = enum.auto()


class FormatIssue(enum.StrEnum):
    TOO_TECHNICAL = enum.auto()
    TOO_SIMPLE = enum.auto()
    WANT_LIST = enum.auto()
    WANT_PROSE = enum.auto()
    WANT_EXAMPLE = enum.auto()
    WRONG_TONE = enum.auto()


class Dimension(enum.StrEnum):
    FORMAT = enum.auto()
    LEVEL = enum.auto()
    EXAMPLES = enum.auto()
    TONE = enum.auto()
    HEADINGS = enum.auto()


class PrefValue(enum.StrEnum):
    AUTO = enum.auto()
    BULLETS = enum.auto()
    PROSE = enum.auto()
    TABLE = enum.auto()
    SIMPLE = enum.auto()
    TECHNICAL = enum.auto()
    DIRECT = enum.auto()
    WARM = enum.auto()
    YES = enum.auto()
    NO = enum.auto()


class Button(enum.StrEnum):
    SHORTER = enum.auto()
    MORE_DETAIL = enum.auto()
    ONE_LINE = enum.auto()
    EXAMPLE = enum.auto()
    SIMPLER = enum.auto()
    AS_LIST = enum.auto()
    AS_PROSE = enum.auto()
    NOT_THIS = enum.auto()


class Arm(enum.StrEnum):
    TREATMENT = enum.auto()
    CONTROL = enum.auto()
    CONCISE = enum.auto()


class ReplyMode(enum.StrEnum):
    WIDGET = enum.auto()
    FALLBACK = enum.auto()


class Platform(enum.StrEnum):
    WEB = enum.auto()
    DESKTOP = enum.auto()
    MOBILE = enum.auto()


class LengthChoice(enum.StrEnum):
    SHORT = enum.auto()
    MEDIUM = enum.auto()
    DETAILED = enum.auto()


DIMENSION_VALUES = {
    Dimension.FORMAT: (
        PrefValue.AUTO,
        PrefValue.BULLETS,
        PrefValue.PROSE,
        PrefValue.TABLE,
    ),
    Dimension.LEVEL: (PrefValue.AUTO, PrefValue.SIMPLE, PrefValue.TECHNICAL),
    Dimension.EXAMPLES: (PrefValue.AUTO, PrefValue.YES),
    Dimension.TONE: (PrefValue.AUTO, PrefValue.DIRECT, PrefValue.WARM),
    Dimension.HEADINGS: (PrefValue.AUTO, PrefValue.YES, PrefValue.NO),
}
# Dimensions learned per topic domain rather than per request type.
DOMAIN_SCOPED = frozenset({Dimension.LEVEL})
# Format issue -> the (dimension, value) it asks for; None: anything else.
FORMAT_ISSUES = {
    FormatIssue.TOO_TECHNICAL: (Dimension.LEVEL, PrefValue.SIMPLE),
    FormatIssue.TOO_SIMPLE: (Dimension.LEVEL, PrefValue.TECHNICAL),
    FormatIssue.WANT_LIST: (Dimension.FORMAT, PrefValue.BULLETS),
    FormatIssue.WANT_PROSE: (Dimension.FORMAT, PrefValue.PROSE),
    FormatIssue.WANT_EXAMPLE: (Dimension.EXAMPLES, PrefValue.YES),
    FormatIssue.WRONG_TONE: (Dimension.TONE, None),
}


class RandomizationUnit(enum.StrEnum):
    CONVERSATION = enum.auto()
    USER = enum.auto()


class Clarify(enum.StrEnum):
    """The two arms of the clarification bandit (SPEC 4.1)."""

    ASK = enum.auto()
    ANSWER = enum.auto()


class ArmKind(enum.StrEnum):
    """Two-value bandits stored as preference dimensions "<kind>:<arm>"."""

    BUTTON = enum.auto()
    CLARIFY = enum.auto()


# The profile-page dimension that edits a length cell, not a preference.
LENGTH = "length"


class Source(enum.StrEnum):
    """Who observed a turn's outcome: the annotators of SPEC 6.4."""

    ASSISTANT = enum.auto()
    BUTTON = enum.auto()
    READING = enum.auto()


class ModelFamily(enum.StrEnum):
    """Assistant model families (SPEC 6.3); the last word is the marker."""

    CLAUDE_OPUS = "claude-opus"
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_HAIKU = "claude-haiku"
    GPT = "gpt"
    OTHER = "other"

    @property
    def marker(self) -> str:
        """The word that names this family in a model's self-report."""
        return self.value.rsplit("-", 1)[-1]
