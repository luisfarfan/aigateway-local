"""
Provider registry — central catalog of all registered AI engine adapters.

Providers self-register at worker startup via registry.register().
Workers and dispatchers use registry.resolve() to get the correct adapter for a job.
"""
from src.core.domain import JobType
from src.modules.providers.base import BaseProvider, ProviderCapability


class ProviderRegistry:
    """
    Maps provider_id strings to BaseProvider adapter instances.

    Usage:
        registry = ProviderRegistry()
        registry.register(StubProvider())
        registry.register(DiffusersProvider())

        provider = registry.resolve("diffusers", JobType.IMAGE_GENERATION, "sdxl")
        result = await provider.execute(context)
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """
        Register a provider adapter. Raises if provider_id is already registered.
        Called once per provider at worker startup.
        """
        pid = provider.provider_id
        if pid in self._providers:
            raise ValueError(
                f"Provider '{pid}' is already registered. "
                "Each provider_id must be unique."
            )
        self._providers[pid] = provider

    def resolve(
        self,
        provider_id: str,
        job_type: JobType,
        model: str | None = None,
    ) -> BaseProvider:
        """
        Return the adapter for the given (provider_id, job_type, model) combination.
        Raises ValueError if the provider is not registered or doesn't support the request.
        """
        provider = self._providers.get(provider_id)
        if provider is None:
            available = list(self._providers.keys())
            raise ValueError(
                f"Provider '{provider_id}' not registered. Available: {available}"
            )
        if not provider.supports(job_type, model):
            raise ValueError(
                f"Provider '{provider_id}' does not support "
                f"job_type='{job_type}' with model='{model}'."
            )
        return provider

    def list_capabilities(self) -> list[ProviderCapability]:
        """Returns capability declarations for all registered providers."""
        return [p.capability for p in self._providers.values()]

    def list_provider_ids(self) -> list[str]:
        return list(self._providers.keys())

    def get(self, provider_id: str) -> BaseProvider | None:
        """Returns the provider or None if not registered. Non-raising lookup."""
        return self._providers.get(provider_id)

    def __contains__(self, provider_id: str) -> bool:
        return provider_id in self._providers
