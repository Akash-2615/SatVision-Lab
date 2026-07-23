"""
Simple local admin gate for lab exercise (no DB).

Default password: admin123  (override via ADMIN_PASSWORD env or data/admin_config.json)
Tokens are random secrets kept in memory for the process lifetime.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Dict, Optional, Set

from fastapi import Header, HTTPException

from model_service import DATA_ROOT, ensure_data_dirs

ADMIN_CONFIG = DATA_ROOT / "admin_config.json"
DEFAULT_PASSWORD = "admin123"

# active session tokens
_SESSIONS: Set[str] = set()


def _load_password() -> str:
    env = os.environ.get("ADMIN_PASSWORD")
    if env:
        return env
    ensure_data_dirs()
    if ADMIN_CONFIG.exists():
        try:
            data = json.loads(ADMIN_CONFIG.read_text(encoding="utf-8"))
            if data.get("password"):
                return str(data["password"])
        except (json.JSONDecodeError, OSError):
            pass
    # write default config once so admins know where to change it
    if not ADMIN_CONFIG.exists():
        ADMIN_CONFIG.write_text(
            json.dumps(
                {
                    "password": DEFAULT_PASSWORD,
                    "note": "Change this password. Or set ADMIN_PASSWORD env var.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return DEFAULT_PASSWORD


def verify_password(password: str) -> bool:
    expected = _load_password()
    return hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8"))


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS.add(token)
    return token


def revoke_session(token: Optional[str]) -> None:
    if token:
        _SESSIONS.discard(token)


def is_admin(token: Optional[str]) -> bool:
    return bool(token) and token in _SESSIONS


def require_admin(authorization: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency — expects Authorization: Bearer <token>."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not is_admin(token):
        raise HTTPException(
            status_code=401,
            detail="Admin authentication required. Unlock Training as admin first.",
        )
    return token  # type: ignore[return-value]


def password_fingerprint() -> str:
    """Non-reversible hint that a password is configured (not the password)."""
    return hashlib.sha256(_load_password().encode()).hexdigest()[:8]
