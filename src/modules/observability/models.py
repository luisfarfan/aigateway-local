"""
Histórico propio de llamadas a modelos.

Langfuse es la interfaz para mirar trazas; esto es la fuente para reportes,
evaluaciones y cualquier consulta que quiera cruzarse con las tablas del
gateway. Se guarda acá además de exportar porque CLIProxyAPI retiene sus
estadísticas **60 segundos en memoria** y Langfuse es un sistema aparte que
puede caerse, cambiarse o purgarse sin avisar.

Dos tablas y no una, porque son dos preguntas distintas:

  * `llm_requests` — una fila por petición **lógica**. Es la unidad de
    facturación y la que responde "cuánto gastó este proyecto".
  * `llm_attempts` — una fila por llamada **física**. Es la unidad de
    diagnóstico: reparaciones del guard, reintentos, fallbacks entre modelos.

Con una sola tabla habría que elegir entre contar el costo dos veces o perder
el detalle de por qué una petición necesitó tres llamadas.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class LLMRequest(SQLModel, table=True):
    """Una petición lógica, sin importar cuántas llamadas hicieron falta."""

    __tablename__ = "llm_requests"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Quién y para qué. `project` es la dimensión de toda la contabilidad.
    project: str = Field(index=True)
    route: str = Field(index=True)  # chat | websearch | structured | image
    client_id: str | None = Field(default=None, index=True)
    # Presente sólo si vino por el plano de jobs; el plano síncrono no tiene job.
    job_id: UUID | None = Field(default=None, index=True)

    # Qué se pidió y qué respondió. Difieren cuando hubo fallback (F4).
    requested_model: str = Field(index=True)
    response_model: str | None = Field(default=None, index=True)
    family: str | None = Field(default=None, index=True)

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)

    # Dos costos, no uno: `cost_usd` es lo cobrado (0 con suscripción OAuth) y
    # `cost_equivalent_usd` el precio de lista. `priced=False` significa que no
    # había tarifa — distinto de haber costado cero.
    cost_usd: float | None = Field(default=None)
    cost_equivalent_usd: float | None = Field(default=None)
    priced: bool = Field(default=False)

    cache: str = Field(default="disabled", index=True)  # hit | miss | disabled | bypass
    attempts: int = Field(default=1)
    searched: bool = Field(default=False)

    outcome: str = Field(index=True)  # ok | invalid_output | upstream_error | timeout
    error_kind: str | None = Field(default=None, index=True)
    error_message: str | None = Field(default=None)

    duration_s: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=_utcnow, index=True)

    # Traza correspondiente en Langfuse. Une el histórico con la vista de trazas
    # sin duplicar los prompts acá.
    trace_id: str | None = Field(default=None, index=True)

    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class LLMAttempt(SQLModel, table=True):
    """Una llamada física. Varias por petición cuando hubo reparación o fallback."""

    __tablename__ = "llm_attempts"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    request_id: UUID = Field(foreign_key="llm_requests.id", index=True)

    number: int = Field(default=1)
    model: str = Field(index=True)
    # ok | not_json | schema_invalid | upstream_error | timeout
    outcome: str = Field(index=True)

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    duration_s: float = Field(default=0.0)
    error: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_utcnow, index=True)
