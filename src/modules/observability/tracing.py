"""
Trazas OpenTelemetry con la convención GenAI, exportadas a Langfuse.

Se instrumenta con OTel y no con el SDK de Langfuse a propósito: los atributos
quedan en un vocabulario estándar (`gen_ai.*`), así que mañana el mismo span
puede ir a Phoenix, a Tempo o a los tres a la vez cambiando un endpoint, sin
tocar una línea del gateway.

Langfuse acepta OTLP sobre HTTP con autenticación Basic —el par de claves del
proyecto en base64—, así que no hace falta ningún exportador propietario.

Si no hay endpoint configurado, el proveedor de trazas queda igualmente activo
pero sin exportar: el código que abre spans no necesita saber si alguien
escucha, y no arranca una dependencia de red que nadie pidió.

Un detalle que costó una depuración: OpenTelemetry **no permite reemplazar** un
`TracerProvider` ya instalado — lo rechaza con un warning y sigue con el
anterior. Y algunas librerías instalan el suyo al importarse (CrewAI lo hace).
Si eso pasa, nuestros spans se crean contra un proveedor que nadie exporta y
todo parece funcionar salvo que a Langfuse no llega nada. Por eso acá, si ya hay
un proveedor del SDK instalado, se le **añade** el exportador en vez de intentar
reemplazarlo.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, StatusCode

log = structlog.get_logger(__name__)

TRACER_NAME = "aigateway.llm"

_configured = False


def configure_tracing(
    *,
    service_name: str = "aigateway-local",
    endpoint: str = "",
    public_key: str = "",
    secret_key: str = "",
    environment: str = "development",
) -> None:
    """Instala el proveedor de trazas. Idempotente.

    Un fallo montando el exportador **no** puede tumbar el arranque: se degrada
    a trazas locales sin exportar. Perder observabilidad es malo; perder el
    servicio por perder observabilidad es peor.
    """
    global _configured
    if _configured:
        return

    resource = Resource.create(
        {"service.name": service_name, "deployment.environment": environment}
    )

    # Si otra librería ya instaló un proveedor del SDK, se reutiliza: intentar
    # reemplazarlo no funciona y deja los spans yéndose a un sitio sin salida.
    existing = trace.get_tracer_provider()
    reused = isinstance(existing, TracerProvider)
    provider = existing if reused else TracerProvider(resource=resource)
    if reused:
        log.info("tracing.reusing_existing_provider")

    if endpoint and public_key and secret_key:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=endpoint,
                        headers={"Authorization": f"Basic {token}"},
                    )
                )
            )
            log.info("tracing.exporter_ready", endpoint=endpoint)
        except Exception as exc:  # noqa: BLE001
            log.error("tracing.exporter_failed", endpoint=endpoint, error=str(exc))
    else:
        log.info("tracing.local_only", reason="falta endpoint o claves de Langfuse")

    if not reused:
        trace.set_tracer_provider(provider)
    _configured = True


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


@contextmanager
def llm_span(
    name: str,
    *,
    system: str,
    model: str,
    project: str,
    route: str,
):
    """Span de una petición lógica, con los atributos de entrada ya puestos.

    Los de salida (tokens, costo, modelo que respondió) se agregan al cerrar con
    `record_result`, porque hasta entonces no se conocen — y el modelo que
    responde puede no ser el que se pidió, si hubo fallback.
    """
    with tracer().start_as_current_span(name) as span:
        span.set_attribute("gen_ai.system", system)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", route)
        span.set_attribute("proxima.project", project)
        span.set_attribute("proxima.route", route)
        yield span


def record_result(
    span: Span,
    *,
    response_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float | None,
    cost_equivalent_usd: float | None,
    priced: bool,
    cache: str,
    attempts: int = 1,
    searched: bool = False,
) -> None:
    """Atributos de salida. `response_model` puede diferir del pedido."""
    span.set_attribute("gen_ai.response.model", response_model)
    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
    span.set_attribute("proxima.cache", cache)
    span.set_attribute("proxima.attempts", attempts)
    span.set_attribute("proxima.searched", searched)
    # `priced` viaja explícito para que un costo ausente se distinga de un costo
    # cero: sin esa marca, un modelo sin precio parece gratis.
    span.set_attribute("proxima.priced", priced)
    if cost_usd is not None:
        span.set_attribute("proxima.cost_usd", cost_usd)
    if cost_equivalent_usd is not None:
        span.set_attribute("proxima.cost_usd_equivalent", cost_equivalent_usd)


def record_error(span: Span, *, kind: str, message: str, retryable: bool) -> None:
    span.set_status(StatusCode.ERROR, message)
    span.set_attribute("error.type", kind)
    span.set_attribute("proxima.retryable", retryable)


def record_attempt(span: Span, *, number: int, outcome: str, error: str | None) -> None:
    """Un intento del guard, como evento dentro del span.

    Van como eventos y no como spans hijos porque son parte de la misma petición
    lógica: quien mira la traza quiere ver "salió al tercer intento", no tres
    llamadas sueltas sin relación aparente.
    """
    attributes: dict[str, Any] = {"attempt": number, "outcome": outcome}
    if error:
        attributes["error"] = error[:200]
    span.add_event("guard.attempt", attributes)
