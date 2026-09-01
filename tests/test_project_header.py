"""
`X-Proxima-Project` obligatoria.

Por qué se fuerza, con el número que lo motivó: sobre 3.386 peticiones reales,
el **97,8 % cayó en un balde `default`** porque la cabecera era opcional. Con
eso, "quién gastó más" tenía una sola respuesta y era inútil. Un default
silencioso es peor que un error: parece que el dato existe.
"""

from __future__ import annotations

import pytest

from src.modules.observability.project import (
    HEADER,
    InvalidProject,
    validate,
)


@pytest.mark.parametrize(
    "nombre",
    ["rag", "tienda-fotos", "demo-costos", "desde-github", "prueba-async", "a1b", "x" * 40],
)
def test_acepta_los_nombres_que_ya_se_usaban(nombre: str):
    """La convención sale de los proyectos reales del histórico. Si rechazara
    alguno, obligaría a renombrar sistemas que ya funcionan."""
    assert validate(nombre) == nombre


@pytest.mark.parametrize(
    ("valor", "porque"),
    [
        (None, "falta del todo"),
        ("", "vacía"),
        ("   ", "sólo espacios"),
        ("default", "reservado: era el balde de lo no declarado"),
        ("ab", "menos de 3 caracteres"),
        ("x" * 41, "más de 40"),
        ("Tienda", "mayúscula"),
        ("tienda fotos", "espacio"),
        ("tienda_fotos", "guion bajo"),
        ("-tienda", "empieza con guion"),
        ("1tienda", "empieza con dígito"),
        ("tiendá", "acento"),
    ],
)
def test_rechaza_lo_que_fragmentaria_la_contabilidad(valor, porque: str):
    """`Tienda`, `tienda ` y `tienda-fotos` serían tres proyectos distintos en el
    reporte. Un typo no puede partir el gasto en silencio."""
    with pytest.raises(InvalidProject):
        validate(valor)


def test_default_se_rechaza_aunque_venga_declarado():
    """Aceptarlo dejaría reproducir el mismo agujero con un valor que *parece*
    declarado — y el reporte volvería a tener un balde que no dice nada."""
    with pytest.raises(InvalidProject, match="reservado"):
        validate("default")


def test_los_espacios_de_los_bordes_se_recortan():
    """Un `"tienda "` colado por una config mal copiada es un error de dedo, no
    otro proyecto."""
    assert validate("  tienda-fotos  ") == "tienda-fotos"


def test_el_error_dice_como_arreglarlo():
    """Quien recibe el 400 tiene que poder corregirlo sin abrir el código: el
    mensaje lleva el nombre de la cabecera, la forma y un ejemplo."""
    try:
        validate(None)
    except InvalidProject as exc:
        mensaje = str(exc)
    assert HEADER in mensaje
    assert "kebab-case" in mensaje
    assert "tienda-fotos" in mensaje
