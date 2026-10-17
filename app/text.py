"""Exact word count of a Markdown reply (SPEC 4.2)."""

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
