"""
Cliente de Proxima Gateway.

Se habla HTTP con el gateway, que es quien sabe de proveedores, traducciones,
fallback, cache y costos. Este paquete no reimplementa nada de eso: si algo se
puede arreglar del lado del gateway, no debería requerir publicar una versión
nueva del SDK.

Dos superficies, `Gateway` (async) y `SyncGateway`, porque los proyectos que lo
van a consumir usan una cada uno y forzar a ninguno a reescribir su stack sería
un costo inventado.

    from proxima_llm import SyncGateway

    gw = SyncGateway("http://gateway:8000", project="tienda")
    print(gw.chat("¿Cuál es la capital de Perú?").text)
"""

from proxima_llm.client import Gateway, SyncGateway
from proxima_llm.errors import ProximaError
from proxima_llm.tiers import CHEAP, FAST, SMART
from proxima_llm.types import Completion, Embeddings, Image, Source

__all__ = [
    "CHEAP",
    "FAST",
    "SMART",
    "Completion",
    "Embeddings",
    "Gateway",
    "Image",
    "ProximaError",
    "Source",
    "SyncGateway",
]
__version__ = "0.1.0"
