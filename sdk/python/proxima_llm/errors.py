"""
El error que devuelve el gateway, con lo necesario para decidir qué hacer.

El gateway ya clasifica cada fallo y dice explícitamente si reintentar sirve.
El SDK transporta esa decisión en vez de obligar a cada consumidor a
reinterpretar códigos HTTP —que es donde antes se equivocaban: un 503 por
"ninguna credencial cubre este modelo" invita a reintentar para siempre algo que
no se va a arreglar solo.
"""

from __future__ import annotations


class ProximaError(Exception):
    """Fallo devuelto por el gateway.

    `retryable` viene del gateway, no se deduce del status: sólo él sabe si ya
    agotó la cadena de modelos o si el problema es transitorio.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "unknown",
        status: int = 0,
        retryable: bool = False,
        attempts: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status
        self.retryable = retryable
        # Presente cuando falló la salida estructurada: qué pasó en cada intento.
        self.attempts = attempts or []

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"
