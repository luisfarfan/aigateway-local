"""
Quién hace la llamada. Sin esto la contabilidad de costos no existe.

El gateway sirve a varios sistemas con la misma clave, así que el único dato que
separa el gasto es la cabecera `X-Proxima-Project`. Cuando era opcional, el
97.8% del tráfico caía en un balde `default` y la pregunta "quién gastó más"
tenía una sola respuesta inútil: "default". Medido sobre 3.386 peticiones reales.

Por eso ahora es obligatoria y con forma fija. Dos decisiones que parecen
severas y no lo son:

  * **`default` se rechaza explícitamente.** Es el nombre al que caía todo antes;
    aceptarlo dejaría la puerta abierta a reproducir el mismo agujero con un
    valor que *parece* declarado.
  * **La forma se valida.** `Tienda`, `tienda ` y `tienda-fotos` serían tres
    proyectos distintos en el reporte. Un typo no debe fragmentar la
    contabilidad en silencio.
"""

from __future__ import annotations

import re

HEADER = "X-Proxima-Project"

# kebab-case: empieza con minúscula, sigue con minúsculas, dígitos o guiones.
# 3-40 caracteres. Sale de los nombres que ya se usaban (`prueba-async`,
# `demo-costos`, `desde-github`), así que no obliga a renombrar nada.
PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,39}$")

# Reservado: era el balde donde caía todo lo no declarado.
RESERVED = frozenset({"default"})

_EJEMPLO = "ej. `tienda-fotos`, `rag-catalogo`"


class InvalidProject(ValueError):
    """La cabecera falta o no cumple la forma. Es un 400: lo arregla quien llama."""


def validate(raw: str | None) -> str:
    """Devuelve el proyecto normalizado, o levanta `InvalidProject`.

    Se recortan los espacios antes de validar porque un `"tienda "` colado por
    una config mal copiada es un error de dedo, no una intención distinta — y
    fallar ahí, con el motivo, enseña más que aceptarlo y partir el reporte.
    """
    if raw is None or not raw.strip():
        raise InvalidProject(
            f"Falta la cabecera {HEADER}. Es obligatoria: identifica qué sistema "
            f"hace la llamada, y sin ella el costo de todos se mezcla en un solo "
            f"balde. Usa kebab-case de 3 a 40 caracteres ({_EJEMPLO})."
        )

    project = raw.strip()

    if project in RESERVED:
        raise InvalidProject(
            f"{project!r} está reservado: era el balde donde caía todo lo no "
            f"declarado. Usa el nombre real del sistema que llama ({_EJEMPLO})."
        )

    if not PATTERN.match(project):
        raise InvalidProject(
            f"{project!r} no cumple la forma esperada. Debe ser kebab-case de 3 a "
            f"40 caracteres: empezar con minúscula y seguir con minúsculas, "
            f"dígitos o guiones ({_EJEMPLO})."
        )

    return project
