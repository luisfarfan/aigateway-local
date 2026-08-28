"""
Circuit breaker por modelo, con estado en Redis.

Para qué: un modelo que viene devolviendo 429 o cuyo token está revocado va a
seguir haciéndolo un rato. Sin breaker, cada petición vuelve a descubrirlo,
paga la latencia del intento fallido y recién entonces cae al siguiente de la
cadena. Con breaker, tras unos pocos fallos el modelo se saltea directamente.

En Redis y no en memoria porque la API y los workers son procesos separados: un
breaker por proceso obliga a cada uno a quemarse por su cuenta y multiplica los
fallos por la cantidad de procesos. Se usa el mismo patrón de contador con TTL
que ya emplea `ModalityScheduler`.

Todo el módulo es best-effort: si Redis no está, `is_open` devuelve False y el
tráfico pasa. Un breaker caído tiene que degradar hacia dejar pasar, nunca hacia
bloquear — bloquear por no poder consultar el estado sería inventarse una caída.
"""

from __future__ import annotations

import structlog

from src.core.redis import get_redis
from src.modules.routing.config import BreakerPolicy

log = structlog.get_logger(__name__)

_FAILURES_KEY = "breaker:fails:{model}"
_OPEN_KEY = "breaker:open:{model}"


class CircuitBreaker:
    def __init__(self, policy: BreakerPolicy) -> None:
        self._policy = policy

    async def is_open(self, model: str) -> bool:
        """True si al modelo hay que saltearlo ahora mismo."""
        try:
            return bool(await get_redis().get(_OPEN_KEY.format(model=model)))
        except Exception as exc:  # noqa: BLE001 — sin Redis, se deja pasar
            log.debug("breaker.unavailable", model=model, error=str(exc))
            return False

    async def record_failure(self, model: str) -> bool:
        """Cuenta un fallo. Devuelve True si con este se abrió el circuito.

        El contador vive lo que dura la ventana, así que fallos espaciados nunca
        se acumulan hasta el umbral: lo que se busca detectar es una racha, no un
        total histórico.
        """
        failures_key = _FAILURES_KEY.format(model=model)
        try:
            redis = get_redis()
            failures = await redis.incr(failures_key)
            if failures == 1:
                await redis.expire(failures_key, self._policy.window_s)

            if failures >= self._policy.failure_threshold:
                await redis.set(_OPEN_KEY.format(model=model), "1", ex=self._policy.open_for_s)
                await redis.delete(failures_key)
                log.warning(
                    "breaker.opened",
                    model=model,
                    failures=failures,
                    open_for_s=self._policy.open_for_s,
                )
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            log.debug("breaker.unavailable", model=model, error=str(exc))
            return False

    async def record_success(self, model: str) -> None:
        """Un acierto borra la racha.

        No cierra un circuito ya abierto: mientras está abierto no se manda
        tráfico, así que un éxito en ese estado no debería existir. Lo cierra el
        vencimiento del TTL, que es lo que le da al upstream tiempo de recuperarse.
        """
        try:
            await get_redis().delete(_FAILURES_KEY.format(model=model))
        except Exception as exc:  # noqa: BLE001
            log.debug("breaker.unavailable", model=model, error=str(exc))

    async def open(self, model: str, seconds: int, *, reason: str) -> None:
        """Abre el circuito a mano. Lo usa el watchdog.

        Que el watchdog escriba sobre el mismo estado que el breaker, en vez de
        llevar una tabla de salud aparte, deja al routing consultando una sola
        cosa. Dos fuentes de verdad sobre si un modelo sirve terminan
        contradiciéndose.
        """
        try:
            await get_redis().set(_OPEN_KEY.format(model=model), "1", ex=seconds)
            log.warning("breaker.opened", model=model, reason=reason, open_for_s=seconds)
        except Exception as exc:  # noqa: BLE001
            log.debug("breaker.unavailable", model=model, error=str(exc))

    async def close(self, model: str) -> None:
        """Cierra el circuito y borra la racha. Lo usa el watchdog al ver que un
        modelo volvió."""
        try:
            redis = get_redis()
            await redis.delete(_OPEN_KEY.format(model=model))
            await redis.delete(_FAILURES_KEY.format(model=model))
        except Exception as exc:  # noqa: BLE001
            log.debug("breaker.unavailable", model=model, error=str(exc))

    async def state(self, model: str) -> dict[str, object]:
        """Estado legible, para diagnóstico y para el endpoint de salud."""
        try:
            redis = get_redis()
            failures = await redis.get(_FAILURES_KEY.format(model=model))
            is_open = bool(await redis.get(_OPEN_KEY.format(model=model)))
        except Exception as exc:  # noqa: BLE001
            return {"model": model, "available": False, "error": str(exc)}
        return {
            "model": model,
            "available": True,
            "open": is_open,
            "failures": int(failures or 0),
            "threshold": self._policy.failure_threshold,
        }
