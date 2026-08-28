"""
El guard de salida estructurada, sin red.

Los casos no son hipotéticos: cada uno reproduce un modo de fallo observado
detrás de CLIProxyAPI — el bloque de código markdown que envuelve el JSON, la
prosa alrededor del objeto, el valor de enum inventado, el schema descartado por
completo.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.modules.providers.cliproxy.errors import CliproxyRetryableError
from src.modules.structured.guard import (
    InvalidStructuredOutput,
    call_with_guard,
    extract_json,
)
from src.modules.structured.schema import (
    describe_type,
    schema_instruction,
    to_strict_json_schema,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "categoria": {"enum": ["science_tech", "deportes"], "type": "string"},
        "notas": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["titulo", "categoria"],
}


class ScriptedChat:
    """Devuelve respuestas en orden. Registra la conversación de cada intento."""

    def __init__(self, *responses: str | Exception) -> None:
        self._responses = list(responses)
        self.conversations: list[list[dict[str, Any]]] = []
        self.response_formats: list[Any] = []

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        self.conversations.append(messages)
        self.response_formats.append(response_format)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt, 10, 5


async def run(chat: ScriptedChat, **kwargs: Any):
    return await call_with_guard(
        chat,
        messages=[{"role": "user", "content": "clasifica esto"}],
        schema=SCHEMA,
        name="Clasificacion",
        model="gemini-3-flash",
        **kwargs,
    )


# ─── Schema estricto ──────────────────────────────────────────────────────────


def test_estricto_cierra_los_objetos_y_exige_todo():
    strict = to_strict_json_schema(SCHEMA)
    assert strict["additionalProperties"] is False
    # Un opcional se queda en `required` con su unión nullable: la convención de
    # OpenAI es obligatorio-pero-nullable, no ausente.
    assert set(strict["required"]) == {"titulo", "categoria", "notas"}


def test_estricto_no_muta_la_entrada():
    original = {"type": "object", "properties": {"a": {"type": "string"}}}
    to_strict_json_schema(original)
    assert "additionalProperties" not in original


def test_oneOf_se_convierte_en_anyOf_y_muere_el_discriminator():
    """El modo estricto rechaza `oneOf` de plano y no conoce `discriminator`.
    Pydantic emite ambos para una unión discriminada."""
    schema = {
        "type": "object",
        "properties": {
            "bloque": {
                "oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}],
                "discriminator": {"propertyName": "tipo"},
            }
        },
        "$defs": {
            "A": {"type": "object", "properties": {"tipo": {"const": "a"}}},
            "B": {"type": "object", "properties": {"tipo": {"const": "b"}}},
        },
    }
    strict = to_strict_json_schema(schema)
    bloque = strict["properties"]["bloque"]

    assert "oneOf" not in bloque
    assert "discriminator" not in bloque
    assert len(bloque["anyOf"]) == 2
    # Y los nodos anidados también quedan estrictos.
    assert strict["$defs"]["A"]["additionalProperties"] is False


# ─── El contrato en el prompt ─────────────────────────────────────────────────


def test_los_enums_se_describen_con_sus_valores():
    """Decirle "enum" a un modelo le hace contestar 'Technology' donde el
    contrato pedía 'science_tech'. Los valores son la razón de nombrar el tipo."""
    rendered = describe_type(SCHEMA["properties"]["categoria"], SCHEMA)
    assert '"science_tech"' in rendered
    assert '"deportes"' in rendered
    assert "enum" not in rendered


def test_los_refs_se_resuelven_un_salto_para_alcanzar_el_enum():
    """Pydantic pone los enums detrás de un `$ref` a `$defs`."""
    schema = {
        "type": "object",
        "properties": {"cat": {"$ref": "#/$defs/Cat"}},
        "$defs": {"Cat": {"enum": ["x", "y"], "type": "string"}},
    }
    assert '"x"' in describe_type(schema["properties"]["cat"], schema)


def test_un_ref_roto_no_rompe_la_descripcion():
    """Describir un campo a medias nunca debe tumbar una petición."""
    assert describe_type({"$ref": "#/$defs/NoExiste"}, {"$defs": {}}) == "NoExiste"


def test_la_instruccion_lleva_el_schema_entero():
    """Gemini y Claude nunca ven `response_format`. La conversación es el único
    sitio donde les llega el contrato."""
    text = schema_instruction(to_strict_json_schema(SCHEMA), name="Clasificacion")
    assert "science_tech" in text
    assert '"type":"object"' in text.replace(" ", "")


# ─── Extracción ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Claro, aquí tienes:\n{"a": 1}\nEspero que sirva.',
    ],
)
def test_el_json_se_recupera_venga_como_venga(raw: str):
    """La capa de prosa existe porque un modelo que ignoró `response_format`
    suele devolver el objeto correcto envuelto. Tirarlo gastaría un intento."""
    assert extract_json(raw) == '{"a": 1}'


def test_sin_json_devuelve_none():
    assert extract_json("no tengo idea") is None
    assert extract_json("") is None


# ─── El bucle ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_camino_feliz_un_solo_intento():
    chat = ScriptedChat('{"titulo": "t", "categoria": "deportes"}')
    result = await run(chat)

    assert result.value["categoria"] == "deportes"
    assert result.repairs == 0
    assert result.prompt_tokens == 10
    # El contrato viaja por los dos canales a la vez.
    assert result.attempts[0].outcome == "ok"
    assert chat.response_formats[0]["json_schema"]["strict"] is True
    assert any(m["role"] == "system" and "JSON" in m["content"] for m in chat.conversations[0])


@pytest.mark.asyncio
async def test_repara_un_enum_inventado():
    """El fallo real: el modelo contesta 'Technology' donde el contrato dice
    'science_tech'."""
    chat = ScriptedChat(
        '{"titulo": "t", "categoria": "Technology"}',
        '{"titulo": "t", "categoria": "science_tech"}',
    )
    result = await run(chat)

    assert result.value["categoria"] == "science_tech"
    assert result.repairs == 1
    assert result.attempts[0].outcome == "schema_invalid"
    # La reparación cita el error y le muestra al modelo lo que escribió.
    repair = chat.conversations[1]
    assert any("Technology" in m["content"] for m in repair)
    assert any("categoria" in m["content"] for m in repair if m["role"] == "user")


@pytest.mark.asyncio
async def test_el_tercer_intento_tira_la_conversacion():
    """Si el contexto fue lo que descarriló al modelo, insistir sobre él lo
    vuelve a descarrilar."""
    chat = ScriptedChat(
        "no pienso responder json",
        "sigo sin hacerlo",
        '{"titulo": "t", "categoria": "deportes"}',
    )
    result = await run(chat)

    assert result.repairs == 2
    tercera = chat.conversations[2]
    assert not any(m["content"] == "clasifica esto" for m in tercera)


@pytest.mark.asyncio
async def test_agotados_los_intentos_levanta_con_el_detalle():
    """Nunca un objeto a medias ni defaults inventados: un fallo silencioso acá
    son datos corruptos aguas abajo."""
    chat = ScriptedChat("nada", "nada", "nada")
    with pytest.raises(InvalidStructuredOutput) as exc:
        await run(chat)

    assert len(exc.value.attempts) == 3
    assert all(a.outcome == "not_json" for a in exc.value.attempts)


@pytest.mark.asyncio
async def test_un_fallo_de_upstream_no_gasta_reparaciones():
    """Reformular el prompt no arregla un 429. Se corta de inmediato."""
    chat = ScriptedChat(CliproxyRetryableError("429 cuota"))
    with pytest.raises(InvalidStructuredOutput) as exc:
        await run(chat)

    assert len(exc.value.attempts) == 1
    assert exc.value.attempts[0].outcome == "upstream_error"


@pytest.mark.asyncio
async def test_se_valida_contra_el_schema_original_no_contra_el_estricto():
    """Lo estricto es una exigencia del transporte de OpenAI, no el contrato de
    quien llama. Omitir un campo opcional es válido y no debe reintentarse."""
    chat = ScriptedChat('{"titulo": "t", "categoria": "deportes"}')  # sin `notas`
    result = await run(chat)
    assert result.repairs == 0


def test_un_const_sin_type_recibe_el_suyo():
    """Encontrado por los evals contra un modelo real: OpenAI rechaza con HTTP
    400 cualquier nodo sin `type`, y Pydantic emite el discriminante de una
    unión como `{"const": "txt_blk"}` pelado. El rechazo es del transporte, así
    que el bucle de reparación del guard no podía salvarlo: la petición nunca
    llegaba al modelo."""
    strict = to_strict_json_schema(
        {
            "type": "object",
            "properties": {"tipo": {"const": "txt_blk"}, "n": {"const": 3}},
        }
    )
    assert strict["properties"]["tipo"]["type"] == "string"
    assert strict["properties"]["n"]["type"] == "integer"


def test_un_enum_homogeneo_sin_type_recibe_el_suyo():
    strict = to_strict_json_schema({"type": "object", "properties": {"c": {"enum": ["a", "b"]}}})
    assert strict["properties"]["c"]["type"] == "string"


def test_un_enum_de_tipos_mezclados_se_deja_como_esta():
    """Inventar un tipo ahí sería adivinar."""
    strict = to_strict_json_schema({"type": "object", "properties": {"c": {"enum": ["a", 1]}}})
    assert "type" not in strict["properties"]["c"]


def test_un_type_existente_no_se_pisa():
    strict = to_strict_json_schema(
        {"type": "object", "properties": {"c": {"enum": ["a"], "type": "string"}}}
    )
    assert strict["properties"]["c"]["type"] == "string"
