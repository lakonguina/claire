"""Detect reply forms from Markdown (for logging what was actually shown)."""

from __future__ import annotations

from markdown_it import MarkdownIt

from app.core import vocab

_PARSER = MarkdownIt("commonmark").enable("table")

# markdown-it "open" / block token types → Claire forms
_TOKEN_FORM = {
    "fence": vocab.Form.CODE,
    "code_block": vocab.Form.CODE,
    "bullet_list_open": vocab.Form.BULLETS,
    "ordered_list_open": vocab.Form.NUMBERED_LIST,
    "table_open": vocab.Form.TABLE,
}


def detect_forms(markdown: str) -> list[vocab.Form]:
    """Return the forms present in a Markdown string, in document order.

    Checklist items (``- [ ]``) map to :attr:`~app.core.vocab.Form.CHECKLIST`
    instead of :attr:`~Form.BULLETS`. Plain paragraphs yield :attr:`~Form.TEXT`.

    Args:
        markdown: Reply text as written by the host.

    Returns:
        Deduplicated forms in first-seen order (never empty: at least text).
    """
    seen: list[vocab.Form] = []
    has_text = False
    tokens = _PARSER.parse(markdown or "")

    def add(form: vocab.Form) -> None:
        if form not in seen:
            seen.append(form)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type in _TOKEN_FORM:
            form = _TOKEN_FORM[token.type]
            if token.type == "bullet_list_open" and _checklist_at(tokens, i):
                form = vocab.Form.CHECKLIST
            add(form)
        elif token.type == "inline" and (token.content or "").strip():
            has_text = True
        i += 1

    if has_text or not seen:
        if vocab.Form.TEXT not in seen:
            seen.insert(0, vocab.Form.TEXT)
    return seen


def _checklist_at(tokens: list, list_open_index: int) -> bool:
    """True when this bullet list is a task list (``- [ ]`` / ``- [x]``)."""
    depth = 0
    for token in tokens[list_open_index:]:
        if token.type == "bullet_list_open":
            depth += 1
        elif token.type == "bullet_list_close":
            depth -= 1
            if depth == 0:
                break
        elif depth == 1 and token.type == "inline":
            content = (token.content or "").lstrip()
            if not (
                content.startswith("[ ]")
                or content.startswith("[x]")
                or content.startswith("[X]")
            ):
                return False
    return True
