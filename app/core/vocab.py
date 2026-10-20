"""Closed vocabularies: the single source of truth for every identifier."""

import enum


class Outcome(enum.StrEnum):
    MOVED_ON = enum.auto()
    TOO_LONG = enum.auto()
    TOO_SHORT = enum.auto()
    MISSING_INFO = enum.auto()
    MISUNDERSTOOD = enum.auto()
    WRONG_FORMAT = enum.auto()
    WRONG_FACT = enum.auto()
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
    """The values of the form dimensions (DIMENSION_VALUES)."""

    AUTO = enum.auto()
    BULLETS = enum.auto()
    PROSE = enum.auto()
    TABLE = enum.auto()
    PROSE_BULLETS = enum.auto()  # Prose, with one short list where it helps.
    PROSE_TABLE = enum.auto()  # Prose, with one table where it fits.
    SIMPLE = enum.auto()
    TECHNICAL = enum.auto()
    DIRECT = enum.auto()
    WARM = enum.auto()
    YES = enum.auto()
    NO = enum.auto()


class Arm(enum.StrEnum):
    TREATMENT = enum.auto()
    HOLDOUT = enum.auto()  # The assistant's normal behaviour: the baseline.
    CONCISE = enum.auto()


class ReplyMode(enum.StrEnum):
    WIDGET = enum.auto()
    FALLBACK = enum.auto()


class Platform(enum.StrEnum):
    WEB = enum.auto()
    DESKTOP = enum.auto()
    MOBILE = enum.auto()


DIMENSION_VALUES = {
    Dimension.FORMAT: (
        PrefValue.AUTO,
        PrefValue.BULLETS,
        PrefValue.PROSE,
        PrefValue.TABLE,
        PrefValue.PROSE_BULLETS,
        PrefValue.PROSE_TABLE,
    ),
    Dimension.LEVEL: (PrefValue.AUTO, PrefValue.SIMPLE, PrefValue.TECHNICAL),
    Dimension.EXAMPLES: (PrefValue.AUTO, PrefValue.YES),
    Dimension.TONE: (PrefValue.AUTO, PrefValue.DIRECT, PrefValue.WARM),
    Dimension.HEADINGS: (PrefValue.AUTO, PrefValue.YES, PrefValue.NO),
}
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
    # Per conversation while there are few active people, then per person.
    AUTO = enum.auto()


class Clarify(enum.StrEnum):
    """The two arms of the clarification bandit."""

    ASK = enum.auto()
    ANSWER = enum.auto()


class Expertise(enum.StrEnum):
    """The two arms of the expertise bandit: raise a blocking point or not."""

    RAISE = enum.auto()
    SKIP = enum.auto()


class ArmKind(enum.StrEnum):
    """Two-value bandits stored as preference dimensions "<kind>:<arm>"."""

    CLARIFY = enum.auto()
    EXPERTISE = enum.auto()


class Source(enum.StrEnum):
    """Who observed a turn's outcome: the annotators of the reaction model."""

    ASSISTANT = enum.auto()
    READING = enum.auto()
    REPHRASE = enum.auto()  # The next request asks the same thing again.
    # The person's answer to "did you get what you needed?": nightly only,
    # it calibrates the other annotators.
    SATISFACTION = enum.auto()


class Catalog(enum.StrEnum):
    """The prompt catalogs: the current one, and a candidate being launched."""

    CURRENT = enum.auto()
    CANDIDATE = enum.auto()


class Requested(enum.StrEnum):
    """What an instruction may ask the assistant for, beyond the reply."""

    USER_MESSAGE = enum.auto()
    CHOICES = enum.auto()


class WidgetEvent(enum.StrEnum):
    """What the widget reports about a shown reply."""

    VISIBLE = enum.auto()  # Seconds the reply was on screen.
    COPY = enum.auto()  # "Copy reply" clicked.
    CHOICE = enum.auto()  # An option of the choice question picked.
    SATISFACTION = enum.auto()  # The answer to "did you get what you needed?".
    PRIVACY_LINK = enum.auto()  # The data request link opened.
    RENDER_ERROR = enum.auto()
    CANCELLED = enum.auto()  # The host cancelled the tool call.


class Satisfaction(enum.StrEnum):
    """The answers to the ground-truth question."""

    YES = enum.auto()
    PARTLY = enum.auto()
    NO = enum.auto()


class Credential(enum.StrEnum):
    """The OAuth credentials the server issues."""

    REQUEST = enum.auto()  # A pending authorization, until the consent.
    CODE = enum.auto()
    ACCESS = enum.auto()
    REFRESH = enum.auto()


class Assistant(enum.StrEnum):
    """The AIs the landing page shows how to connect."""

    CLAUDE = enum.auto()
    CHATGPT = enum.auto()
    GROK = enum.auto()


class SetupStep(enum.StrEnum):
    """The steps of connecting an assistant, in order (landing page)."""

    INSTRUCTION = enum.auto()
    CONNECTOR = enum.auto()
    CONNECT = enum.auto()


class DemoSide(enum.StrEnum):
    """The two replies the landing page compares."""

    BEFORE = enum.auto()
    AFTER = enum.auto()
