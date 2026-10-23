"""Encrypt private text before it hits the database.

A database dump or backup must not reveal message contents. We use
AES-256-GCM (industry standard authenticated encryption) from the
``cryptography`` library. Only a process that holds ``DATA_KEY`` can read.
"""

from __future__ import annotations

import base64
import functools
import os

from cryptography.hazmat.primitives.ciphers import aead

from app.core import config

# Random bytes prepended to every ciphertext so the same text never looks
# identical twice on disk (required for GCM safety).
NONCE_BYTES = 12


class MissingKeyError(Exception):
    """Raised when ``DATA_KEY`` is missing or is not 32 bytes of base64."""


@functools.cache
def _key() -> bytes:
    """Decode and validate ``DATA_KEY`` once per process.

    Returns:
        The raw 32-byte AES key.

    Raises:
        MissingKeyError: Key absent, malformed, or wrong length.
    """
    try:
        key = base64.b64decode(config.settings.data_key, validate=True)
    except ValueError as exc:
        raise MissingKeyError from exc
    if len(key) != 32:
        raise MissingKeyError
    return key


def check_key() -> None:
    """Fail at startup if encryption cannot work.

    Call this in the app lifespan so a missing key is obvious immediately,
    not on the first real user message.
    """
    _key()


def encrypt(text: str) -> bytes:
    """Seal UTF-8 text; returns ``nonce || ciphertext``.

    Args:
        text: Cleartext to protect (e.g. JSON of a tool payload).

    Returns:
        Bytes safe to store in ``events.payload_enc``.
    """
    nonce = os.urandom(NONCE_BYTES)
    return nonce + aead.AESGCM(_key()).encrypt(nonce, text.encode(), None)


def decrypt(blob: bytes) -> str:
    """Open a blob produced by :func:`encrypt`.

    Args:
        blob: ``nonce || ciphertext`` from the database.

    Returns:
        The original UTF-8 string.
    """
    nonce, sealed = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    return aead.AESGCM(_key()).decrypt(nonce, sealed, None).decode()
