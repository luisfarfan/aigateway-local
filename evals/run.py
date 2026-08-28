"""
Corre el dataset de evals contra el gateway y compara modelos.

Para qué: hoy la única forma de saber si un modelo sirve para una tarea es
probarlo en producción y esperar a que alguien note las respuestas malas. Esto
lo convierte en un número — tasa de acierto, reparaciones gastadas, latencia y
costo, por modelo — que se puede mirar antes de cambiar la cadena de
`routing.yaml`, y que se puede volver a correr después para ver si algo empeoró.

Uso:
    python evals/run.py                          # todos los modelos de la ruta structured
    python evals/run.py --models gemini-3-flash ollama/qwen2.5:7b
    python evals/run.py --repeat 3               # promedia varias corridas

Requiere el gateway levantado. Cada caso es una llamada real: cuesta cuota.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sdk" / "python"))

from proxima_llm import Gateway, ProximaError  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "structured.yaml"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    reason: str = ""
    repairs: int = 0
    duration_s: float = 0.0
    tokens: int = 0


@dataclass
class ModelReport:
    model: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def repairs(self) -> int:
        return sum(r.repairs for r in self.results)

    @property
    def median_duration(self) -> float:
        return statistics.median([r.duration_s for r in self.results]) if self.results else 0.0

    @property
    def tokens(self) -> int:
        return sum(r.tokens for r in self.results)


def check(expectations: dict[str, Any], parsed: dict[str, Any]) -> str:
    """`""` si cumple todo; si no, la primera diferencia.

    Las afirmaciones se escriben en el dataset con nombres planos
    (`producto_marca`) para que un caso nuevo no requiera tocar este archivo.
    Que valide contra el schema no alcanza: un JSON correcto puede tener el
    contenido equivocado, que es justo lo que interesa medir.
    """
    for key, expected in expectations.items():
        if key == "secciones_len":
            got = len(parsed.get("secciones") or [])
        elif key == "secciones_tipos":
            got = [s.get("tipo") for s in parsed.get("secciones") or []]
        elif key == "producto_marca":
            got = (parsed.get("producto") or {}).get("marca")
        elif key == "precio_nulo":
            got = parsed.get("precio") is None
        else:
            got = parsed.get(key)

        if got != expected:
            return f"{key}: esperaba {expected!r}, obtuvo {got!r}"
    return ""


async def run_case(gw: Gateway, case: dict[str, Any], model: str) -> CaseResult:
    started = time.monotonic()
    try:
        # Sin cache: un acierto mediría la memoria del gateway, no el modelo.
        completion = await gw.structured(
            case["prompt"],
            schema=case["schema"],
            name=case["id"],
            model=model,
            no_cache=True,
        )
    except ProximaError as exc:
        return CaseResult(
            case["id"],
            passed=False,
            reason=f"{exc.kind}: {exc.message[:120]}",
            repairs=len(exc.attempts),
            duration_s=time.monotonic() - started,
        )

    elapsed = time.monotonic() - started
    parsed = completion.parsed or {}
    reason = check(case.get("expect") or {}, parsed)
    repairs = int((completion.raw.get("proxima") or {}).get("repairs", 0))

    return CaseResult(
        case["id"],
        passed=not reason,
        reason=reason,
        repairs=repairs,
        duration_s=elapsed,
        tokens=completion.total_tokens,
    )


async def evaluate(
    gateway_url: str, models: list[str], cases: list[dict[str, Any]], repeat: int
) -> list[ModelReport]:
    reports: list[ModelReport] = []
    async with Gateway(gateway_url, project="evals") as gw:
        for model in models:
            report = ModelReport(model=model)
            for _ in range(repeat):
                for case in cases:
                    result = await run_case(gw, case, model)
                    report.results.append(result)
                    mark = "ok  " if result.passed else "FALLA"
                    print(
                        f"  {mark} {case['id']:22} {result.duration_s:5.1f}s"
                        + (f"  {result.reason}" if result.reason else "")
                    )
            reports.append(report)
            print()
    return reports


def print_summary(reports: list[ModelReport]) -> None:
    print("=" * 78)
    print(f"{'modelo':28} {'acierto':>9} {'reparac.':>9} {'mediana':>9} {'tokens':>9}")
    print("-" * 78)
    for report in sorted(reports, key=lambda r: r.rate, reverse=True):
        print(
            f"{report.model:28} {report.passed}/{len(report.results):<7} "
            f"{report.rate * 100:5.0f}% {report.repairs:>8} "
            f"{report.median_duration:8.1f}s {report.tokens:>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dataset", default=str(DATASET))
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.dataset).read_text())
    cases = data.get("cases") or []

    models = args.models
    if not models:
        routing = yaml.safe_load((REPO_ROOT / "config" / "routing.yaml").read_text())
        models = (routing.get("routes") or {}).get("structured") or []
    if not models:
        print("No hay modelos que evaluar")
        return 2

    print(f"{len(cases)} casos × {args.repeat} corrida(s) × {len(models)} modelo(s)\n")
    reports = asyncio.run(evaluate(args.gateway, models, cases, args.repeat))
    print_summary(reports)

    # Sale distinto de cero si algún modelo falla todo: sirve para CI.
    return 0 if any(r.rate > 0 for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
