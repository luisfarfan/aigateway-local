"""renombra las columnas de metadatos al nombre de los modelos

Los modelos declaran `extra_data`, pero las bases creadas antes de que se
renombrara el campo tienen `artifacts.extra_metadata` y
`worker_runtimes.runtime_metadata`. El desajuste no es cosmético: el INSERT de
un artefacto falla con `UndefinedColumnError`, así que **el plano de jobs no
puede guardar NINGÚN artefacto** — ni imagen, ni audio, ni video. El job corre,
llama al proveedor, sube el archivo a MinIO, y revienta al registrar la fila.

Se aplicó a mano en el entorno local el 2026-09-01; esto lo hace reproducible
para cualquier otro despliegue.

Es CONDICIONAL a propósito. Esta revisión tiene que ser un no-op en dos casos
que van a existir a la vez:

  * bases nuevas, creadas por la baseline, que ya nacen con `extra_data`;
  * bases viejas ya reparadas a mano, como la local.

Un `alter_column` incondicional fallaría en ambas y dejaría el `upgrade head`
roto justo donde debería no hacer nada.

Revision ID: 29d01702ba9c
Revises: 7b8bb7dc02fa
Create Date: 2026-09-01 16:48:50.659547
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "29d01702ba9c"
down_revision: str | None = "7b8bb7dc02fa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (tabla, nombre viejo, nombre que declaran los modelos)
RENOMBRES = [
    ("artifacts", "extra_metadata", "extra_data"),
    ("worker_runtimes", "runtime_metadata", "extra_data"),
]


def _columnas(tabla: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if tabla not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns(tabla)}


def _renombrar(tabla: str, desde: str, hacia: str) -> None:
    """Renombra sólo si hace falta. RENAME conserva los datos; un drop+add no.

    Que preserve los datos importa: son los metadatos de artefactos ya
    generados, y perderlos por una migración sería peor que el bug que arregla.
    """
    columnas = _columnas(tabla)
    if desde in columnas and hacia not in columnas:
        op.alter_column(tabla, desde, new_column_name=hacia)


def upgrade() -> None:
    for tabla, viejo, nuevo in RENOMBRES:
        _renombrar(tabla, viejo, nuevo)


def downgrade() -> None:
    for tabla, viejo, nuevo in RENOMBRES:
        _renombrar(tabla, nuevo, viejo)
