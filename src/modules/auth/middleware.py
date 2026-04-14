"""
API key authentication.

Clients must send their key in one of two ways:
  - Header:  Authorization: Bearer <key>
  - Header:  X-API-Key: <key>

The client_id derived from the key is used for rate limiting, event scoping,
and audit logging. It is a short hash of the key — never the key itself.
"""
import hashlib

import structlog
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import get_settings
from src.core.exceptions import AuthenticationError

log = structlog.get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _derive_client_id(api_key: str) -> str:
    """Stable, short client ID derived from the key. Used for scoping, never for auth."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


async def get_current_client_id(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
    api_key_header: str | None = Security(_api_key_header),
) -> str:
    """
    FastAPI dependency — extracts and validates the API key.
    Returns a stable client_id string derived from the key.

    Usage:
        async def endpoint(client_id: str = Depends(get_current_client_id)):
    """
    settings = get_settings()

    # No auth configured → open mode (development)
    if not settings.valid_api_keys:
        return "dev"

    raw_key: str | None = None

    if bearer is not None:
        raw_key = bearer.credentials
    elif api_key_header is not None:
        raw_key = api_key_header

    if not raw_key or raw_key not in settings.valid_api_keys:
        log.warning(
            "auth_failed",
            ip=request.client.host if request.client else "unknown",
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    client_id = _derive_client_id(raw_key)
    structlog.contextvars.bind_contextvars(client_id=client_id)
    return client_id
