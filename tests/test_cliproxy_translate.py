"""
La traducción, verificada contra las respuestas reales grabadas en F0.

Cada test de parseo alimenta el fixture tal cual salió del proveedor, así que si
upstream cambia una forma, esto falla antes que el gateway en runtime. Los tests
de construcción de request fijan la decisión de ruta: cuál superficie se usa para
cada familia y por qué.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.modules.providers.cliproxy.families import (
    Family,
    family_from_model_id,
    family_from_owned_by,
)
from src.modules.providers.cliproxy.translate import (
    chat_request,
    image_request,
    parse_chat,
    parse_image,
    parse_websearch,
    to_openai_chat_completion,
    websearch_request,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cliproxy"


def _rehydrate(value: Any) -> Any:
    """Reconstruye los blobs que el grabador sustituyó por su huella.

    El grabador guarda `{prefix, length, sha256}` para no versionar 874 KB de
    base64. El parser, en cambio, sólo ve strings en producción, y aflojarlo para
    aceptar dicts sería adaptar el código al andamio del test. Se devuelve un
    string del largo original con el prefijo intacto: lo que el parser mira.
    """
    if isinstance(value, dict):
        if value.get("__blob__"):
            prefix = value["prefix"]
            return prefix + "A" * max(0, value["length"] - len(prefix))
        return {k: _rehydrate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rehydrate(v) for v in value]
    return value


def fixture_body(name: str) -> Any:
    path = FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"fixture {name} no grabado — correr scripts/record_fixtures.py")
    return _rehydrate(json.loads(path.read_text())["response"]["body"])


# ─── Familias ─────────────────────────────────────────────────────────────────


def test_owned_by_manda_sobre_el_prefijo_del_id():
    """El mismo id de modelo puede llegar por dos caminos distintos, y no son
    equivalentes: el Claude de antigravity sale por Vertex y no soporta lo mismo
    que el de Anthropic. Por eso `owned_by` es la fuente autoritativa."""
    assert family_from_owned_by("antigravity") is Family.ANTIGRAVITY
    assert family_from_owned_by("anthropic") is Family.ANTHROPIC
    assert family_from_model_id("claude-sonnet-4-6") is Family.ANTHROPIC


def test_gpt_oss_no_es_openai():
    """`gpt-oss-120b` es open-weights servido por antigravity. Un respaldo por
    prefijo ingenuo lo mandaría a `/v1/responses`, que no existe para él."""
    assert family_from_model_id("gpt-oss-120b-medium") is Family.ANTIGRAVITY
    assert family_from_model_id("gpt-5.4-mini") is Family.OPENAI


def test_prefijo_de_organizacion_no_confunde_la_familia():
    """CLIProxyAPI expone ids con el proyecto delante (`lucho/gemini-2.5-flash`).
    El prefijo es el proyecto, no el proveedor."""
    assert family_from_model_id("lucho/gemini-2.5-flash") is Family.GOOGLE


# ─── Ruteo de websearch ───────────────────────────────────────────────────────


def test_websearch_google_va_por_la_superficie_nativa():
    """Medido en F0: el tool block sobre /v1/chat/completions se descarta. La
    ruta nativa es la única que hace grounding en vanilla."""
    request = websearch_request(
        model="gemini-3-flash",
        family=Family.ANTIGRAVITY,
        messages=[{"role": "user", "content": "hola"}],
    )
    assert request.path == "/v1beta/models/gemini-3-flash:generateContent"
    assert request.body["tools"] == [{"googleSearch": {}}]


def test_websearch_openai_va_por_responses():
    request = websearch_request(
        model="gpt-5.4-mini",
        family=Family.OPENAI,
        messages=[{"role": "user", "content": "hola"}],
    )
    assert request.path == "/v1/responses"
    assert request.body["tools"] == [{"type": "web_search_preview"}]
    # Un turno único viaja como string plano, no como array.
    assert request.body["input"] == "hola"


def test_websearch_openai_separa_el_system_como_instructions():
    """`/v1/responses` no tiene rol `system`: va en su propio campo."""
    request = websearch_request(
        model="gpt-5.4-mini",
        family=Family.OPENAI,
        messages=[
            {"role": "system", "content": "Sé breve."},
            {"role": "user", "content": "hola"},
        ],
    )
    assert request.body["instructions"] == "Sé breve."
    assert request.body["input"] == "hola"


def test_websearch_google_traduce_roles_y_system():
    """Gemini sólo conoce `user` y `model`, y el system va aparte."""
    request = websearch_request(
        model="gemini-3-flash",
        family=Family.GOOGLE,
        messages=[
            {"role": "system", "content": "Sé breve."},
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "qué tal"},
        ],
    )
    assert [c["role"] for c in request.body["contents"]] == ["user", "model"]
    assert request.body["systemInstruction"]["parts"][0]["text"] == "Sé breve."


def test_antigravity_no_gemini_no_tiene_ruta_de_websearch():
    """Antigravity sirve modelos que no son Gemini bajo el mismo `owned_by`.
    Fallar acá es mejor que mandar `googleSearch` a un modelo que no lo entiende."""
    with pytest.raises(ValueError, match="websearch"):
        websearch_request(
            model="gpt-oss-120b-medium",
            family=Family.ANTIGRAVITY,
            messages=[{"role": "user", "content": "hola"}],
        )


# ─── Parseo contra fixtures reales ────────────────────────────────────────────


def test_parse_chat_lee_texto_y_uso():
    result = parse_chat(fixture_body("chat_plain"), model="gpt-5.4-mini")
    assert "PONG" in result.text
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens


def test_parse_websearch_openai_detecta_que_busco():
    result = parse_websearch(
        fixture_body("websearch_codex_responses"), model="gpt-5.4-mini", family=Family.OPENAI
    )
    assert result.searched is True
    assert result.text.strip()


def test_parse_websearch_google_extrae_las_fuentes():
    """El bug de `sources` vacío no se arregla: desaparece al cambiar de
    superficie, porque `groundingChunks` sólo existe en la nativa."""
    result = parse_websearch(
        fixture_body("websearch_gemini_native"),
        model="gemini-3-flash",
        family=Family.ANTIGRAVITY,
    )
    assert result.searched is True
    assert result.sources, "la ruta nativa tiene que devolver fuentes"
    for source in result.sources:
        assert source.uri
        assert source.title


def test_el_path_openai_compat_no_devuelve_fuentes():
    """Contraste con el test anterior: mismo modelo, superficie equivocada,
    cero fuentes. Es la justificación del módulo entero."""
    result = parse_chat(
        fixture_body("websearch_gemini_openai_compat_UNSUPPORTED"), model="gemini-3-flash"
    )
    assert result.sources == []


def test_parse_image_gemini_sale_del_mensaje_de_chat():
    result = parse_image(fixture_body("image_gemini_chat"), model="gemini-3.1-flash-image")
    assert len(result.images) == 1
    assert result.images[0].startswith("data:image/")


def test_image_request_elige_la_via_segun_el_modelo():
    assert image_request(model="gemini-3.1-flash-image", prompt="x").path == (
        "/v1/chat/completions"
    )
    assert image_request(model="gpt-image-2", prompt="x").path == "/v1/images/generations"


# ─── Contrato de salida ───────────────────────────────────────────────────────


def test_salida_openai_lleva_las_fuentes_fuera_del_message():
    """Un cliente OpenAI ignora la clave extra; el que la conoce no tiene que
    reparsear el texto para recuperar las fuentes."""
    result = parse_websearch(
        fixture_body("websearch_gemini_native"),
        model="gemini-3-flash",
        family=Family.ANTIGRAVITY,
    )
    payload = to_openai_chat_completion(result)

    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["usage"]["total_tokens"] >= 0
    assert payload["proxima"]["searched"] is True
    assert payload["proxima"]["sources"][0]["uri"]


def test_chat_request_no_pide_streaming():
    """El plano síncrono responde de una; el streaming es otra conversación."""
    request = chat_request(model="m", messages=[{"role": "user", "content": "x"}])
    assert request.body["stream"] is False


def test_codex_extrae_las_fuentes_de_las_anotaciones():
    """Encontrado probando el SDK contra el gateway expuesto: `sources` salía
    siempre vacío por la ruta de Codex — el mismo bug que la auditoría le
    señalaba a otro proyecto, y que yo había arreglado sólo para Gemini.

    Las fuentes de la Responses API viven en `annotations`, no en un campo de
    grounding."""
    payload = {
        "model": "gpt-5.4-mini",
        "output": [
            {"type": "web_search_call"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "El precio es X.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://coindesk.com/x",
                                "title": "CoinDesk",
                            },
                            # Repetida: no debe aparecer dos veces.
                            {
                                "type": "url_citation",
                                "url": "https://coindesk.com/x",
                                "title": "CoinDesk",
                            },
                            {"type": "file_citation", "file_id": "f-1"},
                        ],
                    }
                ],
            },
        ],
    }
    result = parse_websearch(payload, model="gpt-5.4-mini", family=Family.OPENAI)

    assert result.searched is True
    assert [s.uri for s in result.sources] == ["https://coindesk.com/x"]


def test_codex_sin_anotaciones_sigue_marcando_que_busco():
    """`searched` y `sources` son señales distintas: el modelo a veces cita en
    prosa sin emitir la anotación. Deducir una de la otra daría un falso
    negativo de búsqueda."""
    payload = {
        "output": [
            {"type": "web_search_call"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "algo", "annotations": []}],
            },
        ]
    }
    result = parse_websearch(payload, model="gpt-5.4-mini", family=Family.OPENAI)

    assert result.searched is True
    assert result.sources == []


# ─── Function calling ─────────────────────────────────────────────────────────


def test_las_herramientas_de_funcion_se_reenvian_tal_cual():
    """Bug encontrado con el gateway ya en la red: `tools` no se reenviaba, así
    que el modelo respondía "no tengo acceso a esa herramienta" y cualquier
    agente se quedaba sin su siguiente paso. Se pasan verbatim: su forma ya es
    la de OpenAI, reescribirlas sólo podría estropearlas."""
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    request = chat_request(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        tools=tools,
        tool_choice="auto",
    )
    assert request.body["tools"] == tools
    assert request.body["tool_choice"] == "auto"


def test_sin_tools_no_se_ensucia_el_cuerpo():
    """Mandar `tools: []` no es lo mismo que no mandarlo: algunos proveedores
    tratan la lista vacía como 'usa herramientas' y se confunden."""
    request = chat_request(model="m", messages=[], tools=None)
    assert "tools" not in request.body
    assert "tool_choice" not in request.body


def test_parse_chat_recupera_los_tool_calls():
    payload = {
        "model": "gpt-5.4-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Lima"}'},
                        }
                    ],
                }
            }
        ],
    }
    result = parse_chat(payload, model="gpt-5.4-mini")
    assert result.tool_calls[0]["function"]["name"] == "get_weather"


def test_el_finish_reason_avisa_que_hay_que_ejecutar_herramientas():
    """No es decorativo: un cliente agéntico decide si ejecuta o termina
    mirando este campo. Con `stop` fijo, el agente da la conversación por
    cerrada y nunca llama a la herramienta."""
    from src.modules.providers.cliproxy.translate import LLMResult

    con_tools = to_openai_chat_completion(
        LLMResult(text="", model="m", tool_calls=[{"id": "c1", "type": "function"}])
    )
    sin_tools = to_openai_chat_completion(LLMResult(text="hola", model="m"))

    assert con_tools["choices"][0]["finish_reason"] == "tool_calls"
    assert con_tools["choices"][0]["message"]["tool_calls"][0]["id"] == "c1"
    assert sin_tools["choices"][0]["finish_reason"] == "stop"
