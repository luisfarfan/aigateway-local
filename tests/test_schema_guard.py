"""
Que producción no cree tablas sola, y que no arranque con el esquema viejo.

Las dos vías —`create_all` desde los modelos y `alembic upgrade`— nunca deben
estar activas a la vez. Tenerlas fue lo que produjo el desajuste que rompió el
plano de jobs: `create_all` crea lo que falta pero **no altera lo que ya
existe**, así que un campo renombrado en los modelos se queda con el nombre
viejo en la base, en silencio, hasta que un INSERT falla en caliente.
"""

from __future__ import annotations

import pytest

from src.core.config import Settings


def test_produccion_no_crea_tablas_sola():
    """Un default peligroso que hay que acordarse de desactivar termina activo
    el día que se despliega con prisa."""
    assert Settings(environment="production").db_auto_create_tables is False


def test_desarrollo_las_sigue_creando():
    """La comodidad de dev no se pierde: levantar sin correr migraciones a mano
    es lo que hace usable el entorno local."""
    assert Settings(environment="development").db_auto_create_tables is True


def test_se_puede_forzar_en_produccion_de_forma_explicita(monkeypatch):
    """Para una base efímera —pruebas de carga, por ejemplo—. Explícito, así es
    una decisión y no un olvido."""
    monkeypatch.setenv("DB_AUTO_CREATE_TABLES", "true")
    assert Settings(environment="production").db_auto_create_tables is True


@pytest.mark.asyncio
async def test_no_arranca_si_la_base_no_esta_migrada(monkeypatch):
    """Arrancar contra un esquema viejo no da un error limpio: da fallos
    parciales y tardíos —un INSERT que revienta cuando ya se llamó al proveedor
    y se subió el archivo— y eso se diagnostica mucho peor que no arrancar.

    El mensaje tiene que decir las DOS salidas: migrar, o marcar la base si ya
    estaba correcta pero nunca se selló.
    """
    from src.core import database

    class ConexionFalsa:
        async def execute(self, *_a, **_kw):
            class R:
                @staticmethod
                def scalar_one_or_none():
                    return "una-revision-vieja"

            return R()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    class EngineFalso:
        @staticmethod
        def begin():
            return ConexionFalsa()

    # Se reemplaza el engine entero: `begin` es de sólo lectura en AsyncEngine.
    monkeypatch.setattr(database, "engine", EngineFalso())

    with pytest.raises(database.SchemaOutOfDate) as exc:
        await database.verify_schema_is_current()

    mensaje = str(exc.value)
    assert "alembic upgrade head" in mensaje
    assert "alembic stamp head" in mensaje
    assert "una-revision-vieja" in mensaje


@pytest.mark.asyncio
async def test_un_fallo_del_chequeo_no_deja_el_servicio_en_el_piso(monkeypatch):
    """Si no se puede leer la configuración de Alembic, se avisa y se sigue. Un
    bug en el chequeo no puede ser peor que el problema que vigila."""
    from src.core import database

    def explota(*_a, **_kw):
        raise RuntimeError("alembic.ini ilegible")

    monkeypatch.setattr("alembic.script.ScriptDirectory.from_config", explota)
    await database.verify_schema_is_current()  # no levanta
