"""Secure storage for API secrets (§8 Security).

Keys are encrypted at rest: OS keyring when available, else a local
keyring-compatible encrypted file store. Secrets are never logged and never
returned to the UI — the API surface is write/check/delete only.
"""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .db import DEFAULT_DB_PATH

FALLBACK_DIR = os.environ.get(
    "MEMICOIN_SECRETS_DIR",
    str(Path(DEFAULT_DB_PATH).parent) if DEFAULT_DB_PATH != ":memory:" else os.getcwd(),
)
FALLBACK_FILE = os.path.join(FALLBACK_DIR, "secrets.encrypted")
KEY_FILE = os.path.join(FALLBACK_DIR, "secrets.key")

_SERVICE = "memicoin_trader"


def _ensure_dir() -> None:
    os.makedirs(FALLBACK_DIR, exist_ok=True)


def _load_fernet() -> Fernet:
    _ensure_dir()
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
    else:
        os.chmod(KEY_FILE, 0o600)
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return Fernet(key)


class SecretStore:
    """OS keyring when usable; otherwise Fernet-encrypted local file."""

    def __init__(self):
        self._backend: str | None = None

    def _try_keyring(self) -> bool:
        if self._backend is not None:
            return self._backend == "keyring"
        try:
            import keyring
            keyring.get_keyring()
            self._backend = "keyring"
            return True
        except Exception:
            self._backend = "file"
            return False

    def set(self, key: str, value: str) -> None:
        if self._try_keyring():
            try:
                import keyring
                keyring.set_password(_SERVICE, key, value)
                return
            except Exception:
                self._backend = "file"
        f = _load_fernet()
        data = self._read_all(f)
        data[key] = value
        with open(FALLBACK_FILE, "wb") as fh:
            fh.write(f.encrypt(json_dumps(data)))

    def get(self, key: str) -> str | None:
        if self._try_keyring():
            try:
                import keyring
                v = keyring.get_password(_SERVICE, key)
                if v is not None:
                    return v
            except Exception:
                self._backend = "file"
        try:
            f = _load_fernet()
            data = self._read_all(f)
            return data.get(key)
        except (FileNotFoundError, InvalidToken):
            return None

    def delete(self, key: str) -> None:
        if self._try_keyring():
            try:
                import keyring
                keyring.delete_password(_SERVICE, key)
                return
            except Exception:
                self._backend = "file"
        try:
            f = _load_fernet()
            data = self._read_all(f)
            data.pop(key, None)
            with open(FALLBACK_FILE, "wb") as fh:
                fh.write(f.encrypt(json_dumps(data)))
        except FileNotFoundError:
            pass

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def backend_name(self) -> str:
        self._try_keyring()
        return self._backend or "file"

    def _read_all(self, f: Fernet) -> dict:
        try:
            with open(FALLBACK_FILE, "rb") as fh:
                return json_loads(f.decrypt(fh.read()))
        except FileNotFoundError:
            return {}


def json_dumps(d: dict) -> bytes:
    import json
    return json.dumps(d).encode()


def json_loads(b: bytes) -> dict:
    import json
    return json.loads(b.decode())
