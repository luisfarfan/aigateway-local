"""
Application exception hierarchy.

Rules:
  - All application errors inherit from GatewayError.
  - HTTP mapping (status codes) lives in the API layer, not here.
  - Domain and infrastructure code raise these; the API layer catches and converts them.
"""
from uuid import UUID


class GatewayError(Exception):
    """Base for all application-level errors."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ─── Job errors ───────────────────────────────────────────────────────────────

class JobNotFoundError(GatewayError):
    def __init__(self, job_id: UUID) -> None:
        super().__init__(f"Job '{job_id}' not found.")
        self.job_id = job_id


class JobAlreadyExistsError(GatewayError):
    """Raised when a job with the same idempotency_key already exists."""
    def __init__(self, idempotency_key: str, existing_job_id: UUID) -> None:
        super().__init__(
            f"Job with idempotency_key='{idempotency_key}' already exists: {existing_job_id}"
        )
        self.idempotency_key = idempotency_key
        self.existing_job_id = existing_job_id


class InvalidJobStatusTransitionError(GatewayError):
    def __init__(self, job_id: UUID, from_status: str, to_status: str) -> None:
        super().__init__(
            f"Invalid status transition for job '{job_id}': {from_status} → {to_status}"
        )
        self.job_id = job_id
        self.from_status = from_status
        self.to_status = to_status


class JobCancellationError(GatewayError):
    """Raised when a job cannot be cancelled (already terminal)."""
    def __init__(self, job_id: UUID, current_status: str) -> None:
        super().__init__(
            f"Job '{job_id}' cannot be cancelled — current status: {current_status}"
        )
        self.job_id = job_id
        self.current_status = current_status


# ─── Provider errors ──────────────────────────────────────────────────────────

class ProviderNotFoundError(GatewayError):
    def __init__(self, provider_id: str, available: list[str] | None = None) -> None:
        hint = f" Available: {available}" if available else ""
        super().__init__(f"Provider '{provider_id}' not registered.{hint}")
        self.provider_id = provider_id


class ProviderNotSupportedError(GatewayError):
    def __init__(self, provider_id: str, job_type: str, model: str | None) -> None:
        super().__init__(
            f"Provider '{provider_id}' does not support job_type='{job_type}' "
            f"with model='{model}'."
        )
        self.provider_id = provider_id
        self.job_type = job_type
        self.model = model


class ProviderExecutionError(GatewayError):
    """Raised by a provider when execution fails in an unrecoverable way."""
    def __init__(self, provider_id: str, message: str) -> None:
        super().__init__(f"Provider '{provider_id}' execution error: {message}")
        self.provider_id = provider_id


# ─── Storage errors ───────────────────────────────────────────────────────────

class StorageError(GatewayError):
    """Raised when an object storage operation fails."""


class ArtifactNotFoundError(StorageError):
    def __init__(self, key: str) -> None:
        super().__init__(f"Artifact not found in storage: '{key}'")
        self.key = key


# ─── Auth errors ──────────────────────────────────────────────────────────────

class AuthenticationError(GatewayError):
    """Raised when an API request carries an invalid or missing API key."""
    def __init__(self) -> None:
        super().__init__("Invalid or missing API key.")


class RateLimitError(GatewayError):
    """Raised when a client exceeds the rate limit."""
    def __init__(self, limit: int) -> None:
        super().__init__(f"Rate limit exceeded: {limit} requests/minute.")
        self.limit = limit
