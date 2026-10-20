"""Word count and sentences of a Markdown reply."""

import re

import markdown_it

PARSER = markdown_it.MarkdownIt("commonmark").enable("table")
WORD = re.compile(r"\w+(?:['’-]\w+)*")
# Tokens whose content is text the reader reads: inline text and code.
TEXT_TOKENS = frozenset({"text", "code_inline", "fence", "code_block"})


def _contents(tokens: list) -> list[str]:
    out = []
    for token in tokens:
        if token.type in TEXT_TOKENS:
            out.append(token.content)
        out.extend(_contents(token.children or []))
    return out


def count_words(markdown: str) -> int:
    """Counts the words a reader sees in rendered Markdown."""
    return sum(len(WORD.findall(c)) for c in _contents(PARSER.parse(markdown)))


SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
MIN_SENTENCE_WORDS = 4


def _inline_text(token) -> str:
    return "".join(
        child.content
        for child in token.children or []
        if child.type in ("text", "code_inline")
    )


def sentences(markdown: str) -> list[str]:
    """Returns the sentences a reader reads, without code blocks.

    Each paragraph, list item or table cell is split at sentence ends;
    fragments shorter than a few words assert nothing and are dropped.
    """
    out = []
    for token in PARSER.parse(markdown):
        if token.type != "inline":
            continue
        for sentence in SENTENCE_END.split(_inline_text(token)):
            sentence = " ".join(sentence.split())
            if len(WORD.findall(sentence)) >= MIN_SENTENCE_WORDS:
                out.append(sentence)
    return out
