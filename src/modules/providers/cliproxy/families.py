"""
Familia de proveedor a partir de un id de modelo.

Por qué existe: la ruta correcta para una misma capacidad depende del proveedor
de arriba, no del modelo. El websearch de Google va por la superficie nativa, el
de OpenAI por `/v1/responses`, y el de Anthropic por un tool block distinto. Sin
un único sitio que resuelva "de quién es este modelo", esa decisión se duplica en
cada adaptador y deriva.

Cómo lo resuelve: `owned_by` de `GET /v1/models` es la fuente autoritativa —
el mismo `claude-sonnet-4-6` puede llegar por `anthropic` o por `antigravity`, y
son caminos distintos con capacidades distintas. Los prefijos del id sólo se usan
como respaldo cuando el catálogo no está disponible, porque adivinar por prefijo
es exactamente lo que falla cuando un proveedor renombra un modelo.
"""

from __future__ import annotations

from enum import StrEnum


class Family(StrEnum):
    """Familia de proveedor. Determina qué traducción aplica."""

    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    ANTIGRAVITY = "antigravity"
    XAI = "xai"
    # Modelos que corren en esta máquina. Familia propia porque no comparten ni
    # tarifa ni cuota con nada cloud, y mezclarlos ensuciaría cache y costos.
    LOCAL = "local"
    UNKNOWN = "unknown"


# `owned_by` tal como lo devuelve CLIProxyAPI en GET /v1/models.
_OWNED_BY_TO_FAMILY: dict[str, Family] = {
    "google": Family.GOOGLE,
    "gemini": Family.GOOGLE,
    "openai": Family.OPENAI,
    "codex": Family.OPENAI,
    "anthropic": Family.ANTHROPIC,
    "claude": Family.ANTHROPIC,
    "antigravity": Family.ANTIGRAVITY,
    "xai": Family.XAI,
    "ollama": Family.LOCAL,
}

# Respaldo por prefijo, sólo si no hay catálogo. Ordenado del prefijo más
# específico al más genérico: `gpt-oss` es un modelo open-weights servido por
# antigravity, no un modelo de OpenAI, y tiene que ganarle a `gpt-`.
_PREFIX_TO_FAMILY: tuple[tuple[str, Family], ...] = (
    ("gpt-oss", Family.ANTIGRAVITY),
    ("gemini", Family.GOOGLE),
    ("claude", Family.ANTHROPIC),
    ("gpt-", Family.OPENAI),
    ("codex", Family.OPENAI),
    ("o1", Family.OPENAI),
    ("o3", Family.OPENAI),
    ("grok", Family.XAI),
)


def family_from_owned_by(owned_by: str | None) -> Family:
    """Familia según el `owned_by` del catálogo. `UNKNOWN` si no se reconoce."""
    if not owned_by:
        return Family.UNKNOWN
    return _OWNED_BY_TO_FAMILY.get(owned_by.strip().lower(), Family.UNKNOWN)


def family_from_model_id(model: str) -> Family:
    """Respaldo por prefijo. Usar sólo si el catálogo no resolvió la familia.

    Un id con prefijo de organización (`lucho/gemini-2.5-flash`) se resuelve por
    la parte de después de la barra: el prefijo es el proyecto, no el proveedor.
    """
    candidate = model.strip().lower()
    if "/" in candidate:
        candidate = candidate.rsplit("/", 1)[-1]
    for prefix, fam in _PREFIX_TO_FAMILY:
        if candidate.startswith(prefix):
            return fam
    return Family.UNKNOWN


def is_gemini_model(model: str) -> bool:
    """True si el id es de la familia Gemini, sin importar quién lo sirva.

    Antigravity sirve modelos Gemini y modelos que no lo son bajo el mismo
    `owned_by`, así que para elegir la traducción de imagen o de websearch hay
    que mirar también el id, no sólo la familia.
    """
    candidate = model.strip().lower()
    if "/" in candidate:
        candidate = candidate.rsplit("/", 1)[-1]
    return candidate.startswith("gemini")
