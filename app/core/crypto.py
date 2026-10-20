"""Encryption of stored text and the codes of data requests.

Text is encrypted by the application (AES-256-GCM), so a database dump or a
backup reveals none of it; only processes that hold DATA_KEY can read it.
"""

import base64
import functools
import hashlib
import hmac
import os

from cryptography.hazmat.primitives.ciphers import aead

from app.core import config

NONCE_BYTES = 12
CODE_CHARS = 16
CODE_SEPARATOR = "-"


class MissingKeyError(Exception):
    """DATA_KEY is missing or is not 32 bytes of base64."""


@functools.cache
def _key() -> bytes:
    try:
        key = base64.b64decode(config.settings.data_key, validate=True)
    except ValueError as exc:
        raise MissingKeyError from exc
    if len(key) != 32:
        raise MissingKeyError
    return key


def check_key() -> None:
    """Fails at start, rather than at the first stored text, without a key."""
    _key()


def encrypt(text: str) -> bytes:
    """Returns the nonce followed by the AES-GCM ciphertext of a text."""
    nonce = os.urandom(NONCE_BYTES)
    return nonce + aead.AESGCM(_key()).encrypt(nonce, text.encode(), None)


def decrypt(blob: bytes) -> str:
    """Returns the text an encrypt() output holds."""
    nonce, sealed = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    return aead.AESGCM(_key()).decrypt(nonce, sealed, None).decode()


def _mac(user_id: int) -> str:
    digest = hmac.new(_key(), f"privacy:{user_id}".encode(), hashlib.sha256)
    return digest.hexdigest()[:CODE_CHARS]


def privacy_code(user_id: int) -> str:
    """Returns the code that ties a data request to a person.

    The internal number names no one; the MAC proves the widget minted it.
    """
    return f"{user_id}{CODE_SEPARATOR}{_mac(user_id)}"


def user_of_code(code: str) -> int | None:
    """Returns the person a privacy code belongs to, or None if forged."""
    number, _, mac = code.strip().partition(CODE_SEPARATOR)
    if not number.isdigit() or not hmac.compare_digest(mac, _mac(int(number))):
        return None
    return int(number)
