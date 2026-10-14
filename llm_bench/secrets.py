from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from llm_bench.settings import Settings, settings as default_settings

_PREFIX = "enc:v1:"


def _fernet(cfg: Settings) -> Fernet:
    # Derive a stable Fernet key from the configured secret.
    digest = hashlib.sha256(cfg.secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(value: str, cfg: Settings | None = None) -> str:
    cfg = cfg or default_settings
    if not value or value.startswith(_PREFIX):
        return value
    token = _fernet(cfg).encrypt(value.encode("utf-8")).decode("ascii")
    return f"{_PREFIX}{token}"


def decrypt_secret(value: str, cfg: Settings | None = None) -> str:
    cfg = cfg or default_settings
    if not value or not value.startswith(_PREFIX):
        return value
    token = value[len(_PREFIX) :]
    try:
        return _fernet(cfg).decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError(
            "Failed to decrypt API key. Check LLM_BENCH_SECRET_KEY matches the key used at save time."
        ) from exc


def mask_secret(value: str) -> str:
    if not value:
        return ""
    plain = value[len(_PREFIX) :] if value.startswith(_PREFIX) else value
    if len(plain) <= 4:
        return "*" * len(plain)
    return ("*" * max(0, len(plain) - 4)) + plain[-4:]
