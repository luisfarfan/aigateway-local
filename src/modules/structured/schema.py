"""
Transformaciones de JSON Schema para que un modelo devuelva la forma pedida.

Dos problemas distintos, y por eso dos familias de funciones:

1. **Los modelos de OpenAI rechazan el schema crudo de Pydantic.** El modo
   estricto (`response_format.json_schema` con `strict: true`) exige que todo
   nodo objeto lleve `additionalProperties: false` y liste *todas* sus
   propiedades en `required`; además prohíbe `oneOf` y no conoce
   `discriminator`. Pydantic no emite nada de eso, así que la petición vuelve
   con HTTP 400. Eso lo arregla `to_strict_json_schema`.

2. **Gemini y Claude descartan `response_format` entero.** No es que lo
   incumplan: nunca lo ven. Responden prosa libre con nombres de campo
   inventados. El único sitio que les queda para recibir el contrato es la
   conversación misma, y ahí importa un detalle que parece cosmético y no lo es:
   hay que renderizar **los valores reales de cada enum**. A un modelo al que se
   le dice "enum" le contesta `'Technology'` donde el contrato pedía
   `science_tech`. Eso lo arma `schema_instruction`.
"""

from __future__ import annotations

import copy
import json
from typing import Any

_COMBINATORS = ("anyOf", "oneOf", "allOf", "prefixItems")


def to_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copia del schema con las restricciones del modo estricto aplicadas.

    Nunca muta la entrada. Los campos opcionales se quedan en `required` con su
    unión nullable intacta: la convención de OpenAI es *obligatorio pero
    nullable*, no ausente.
    """
    result = copy.deepcopy(schema)
    _strictify(result)
    return result


def _strictify(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _strictify(item)
        return
    if not isinstance(node, dict):
        return

    # El modo estricto acepta `anyOf` pero rechaza `oneOf` de plano, y no conoce
    # `discriminator`. Pydantic emite ambos para una unión discriminada. Se
    # renombra antes de recorrer los combinadores para que el `anyOf` resultante
    # también se visite. La unión sigue siendo inequívoca: cada variante ya
    # lleva su `const` en la propiedad discriminante.
    if "oneOf" in node:
        node["anyOf"] = node.pop("oneOf")
    node.pop("discriminator", None)

    for key in ("$defs", "definitions", "properties"):
        section = node.get(key)
        if isinstance(section, dict):
            for sub in section.values():
                _strictify(sub)

    for key in _COMBINATORS:
        section = node.get(key)
        if isinstance(section, list):
            for sub in section:
                _strictify(sub)

    items = node.get("items")
    if isinstance(items, dict | list):
        _strictify(items)

    _ensure_type(node)

    if _is_object_schema(node):
        node["additionalProperties"] = False
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties)


# El modo estricto de OpenAI exige que **todo** nodo lleve `type`, incluidos los
# que se describen sólo por su valor. Un `{"const": "txt_blk"}` —lo que emite
# Pydantic para el discriminante de una unión— se rechaza con
# "schema must have a 'type' key", y el rechazo es un HTTP 400: ni siquiera
# llega al modelo, así que el bucle de reparación del guard no puede salvarlo.
_TYPE_BY_PYTHON: tuple[tuple[type, str], ...] = (
    # bool antes que int: en Python `bool` es subclase de `int`.
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
)


def _json_type_of(value: Any) -> str | None:
    if value is None:
        return "null"
    for python_type, json_type in _TYPE_BY_PYTHON:
        if isinstance(value, python_type):
            return json_type
    return None


def _ensure_type(node: dict[str, Any]) -> None:
    """Deduce `type` de un `const` o un `enum` cuando falta.

    Se deduce en vez de exigirlo porque quien escribe el schema no tiene por qué
    saber de esta peculiaridad: un `const` ya dice cuál es su tipo. Un `enum` con
    valores de tipos mezclados se deja como está — inventar un tipo ahí sería
    adivinar.
    """
    if "type" in node:
        return

    if "const" in node:
        if (json_type := _json_type_of(node["const"])) is not None:
            node["type"] = json_type
        return

    values = node.get("enum")
    if isinstance(values, list) and values:
        types = {_json_type_of(v) for v in values}
        if len(types) == 1 and None not in types:
            node["type"] = types.pop()


def _is_object_schema(node: dict[str, Any]) -> bool:
    node_type = node.get("type")
    if node_type == "object":
        return True
    if isinstance(node_type, list) and "object" in node_type:
        return True
    return "properties" in node


# ─── El contrato en la conversación ───────────────────────────────────────────


def schema_instruction(schema: dict[str, Any], *, name: str) -> str:
    """Mensaje `system` que enuncia el contrato en el primer intento.

    Se manda además de `response_format`, a propósito: un proveedor que sí lo
    respeta no se ve afectado — iba a producir esos mismos campos igual.
    """
    return (
        f"Responde SOLO con JSON crudo para '{name}': sin prosa, sin explicación, "
        "sin bloques de código markdown. Claves de primer nivel, exactamente "
        f"estas y ninguna otra: {describe_fields(schema)}.\n"
        "A continuación va el JSON Schema completo que tu respuesta DEBE cumplir. "
        "Respétalo en TODOS los niveles, no sólo el primero: los objetos "
        "anidados, los arrays y las uniones discriminadas tienen cada uno sus "
        "propias claves obligatorias, y una variante de unión se elige por su "
        "propiedad discriminante, cuyo valor debe ser una de las constantes "
        "listadas.\n"
        f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    )


def repair_instruction(error_message: str, schema: dict[str, Any], *, name: str) -> str:
    """Mensaje del reintento de reparación.

    Cita el error exacto y vuelve a enunciar los campos esperados: a un modelo
    que ignoró el schema no se le repite la misma petición opaca, se le da una
    corrección estructural concreta.
    """
    return (
        "Tu respuesta anterior no se pudo interpretar con la estructura pedida.\n"
        f"Error: {error_message}\n"
        f"Devuelve SOLO un objeto JSON para '{name}' con exactamente estos campos "
        f"(sin campos extra, sin prosa, sin bloques de código): {describe_fields(schema)}."
    )


def describe_fields(schema: dict[str, Any]) -> str:
    """Los campos de primer nivel como `nombre (tipo), …`, para el prompt."""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "(los campos del schema)"
    return ", ".join(
        f"{field} ({describe_type(prop, schema)})" for field, prop in properties.items()
    )


def describe_type(prop: Any, root: dict[str, Any] | None = None) -> str:
    """Etiqueta compacta de tipo para una propiedad.

    Los enums se renderizan con sus **valores permitidos**, no con la palabra
    "enum", y los `$ref` se resuelven contra `root` para poder alcanzarlos. Los
    valores son la razón entera de nombrar el tipo: sin ellos el modelo inventa
    los suyos.
    """
    if not isinstance(prop, dict):
        return "any"

    resolved = _resolve_ref(prop, root)

    enum_values = resolved.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return "uno de " + "|".join(json.dumps(v, ensure_ascii=False) for v in enum_values)

    node_type = resolved.get("type")
    if isinstance(node_type, str):
        return node_type
    if isinstance(node_type, list):
        return "|".join(str(item) for item in node_type)

    any_of = resolved.get("anyOf")
    if isinstance(any_of, list):
        return "|".join(describe_type(sub, root) for sub in any_of)

    ref = prop.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]
    return "any"


def _resolve_ref(prop: dict[str, Any], root: dict[str, Any] | None) -> dict[str, Any]:
    """Sigue un `$ref` local un salto dentro de `root`.

    Un salto, no un resolutor completo: los alias de enum que emite Pydantic
    están a esa profundidad. Un `$ref` ausente o externo degrada al nodo
    original en vez de levantar — describir un campo a medias nunca debe romper
    una petición.
    """
    ref = prop.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/") or root is None:
        return prop

    node: Any = root
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(node, dict) or part not in node:
            return prop
        node = node[part]
    return node if isinstance(node, dict) else prop
