"""
tiempos.py
==========
Registro del tiempo y la memoria de cada celda de un notebook, y del costo de cada modelo.

En la primera celda del notebook:

    from tfm import tiempos
    tiempos.activar("G01")

y en la última:

    tiempos.reporte()

El registro usa los eventos de IPython (pre_run_cell y post_run_cell), así que funciona igual en VS Code,
Jupyter y papermill. Por celda guarda la sección vigente (el último título de Markdown visto), la primera
línea del código, los segundos, la memoria del proceso (RSS) y la memoria libre del equipo; avisa si la
memoria libre baja de 1,5 GB, que es cuando Windows empieza a paginar y el kernel puede caer.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import config

try:
    import psutil
except ImportError:                       # psutil viene con ipykernel; sin el solo se mide el tiempo
    psutil = None

_ESTADO: dict = {"activo": False, "filas": [], "seccion": "", "inicio": None, "grupo": ""}
_COSTO: list[dict] = []
AVISO_MEMORIA_GB = 1.5


def _memoria() -> tuple[float, float]:
    if psutil is None:
        return float("nan"), float("nan")
    return psutil.Process().memory_info().rss / 2**30, psutil.virtual_memory().available / 2**30


def seccion(nombre: str) -> None:
    """Fija la sección vigente (los notebooks la llaman en cada título)."""
    _ESTADO["seccion"] = nombre


def _antes(info) -> None:
    _ESTADO["inicio"] = time.perf_counter()
    lineas = (getattr(info, "raw_cell", "") or "").strip().splitlines()
    # Los notebooks generados empiezan cada celda con "# Sección 6.3. ..." (un comentario)
    if lineas and lineas[0].startswith("# Sección "):
        _ESTADO["seccion"] = lineas[0][len("# Sección "):].strip()
        lineas = lineas[1:]
    _ESTADO["codigo"] = lineas[:1]


def _despues(resultado) -> None:
    if _ESTADO["inicio"] is None:
        return
    seg = time.perf_counter() - _ESTADO["inicio"]
    rss, libre = _memoria()
    _ESTADO["filas"].append({"n": len(_ESTADO["filas"]) + 1, "seccion": _ESTADO["seccion"],
                             "primera_linea": (_ESTADO["codigo"] or [""])[0][:90], "segundos": round(seg, 2),
                             "memoria_proceso_gb": round(rss, 2), "memoria_libre_gb": round(libre, 2),
                             "error": bool(getattr(resultado, "error_in_exec", None)),
                             "fin": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    if libre == libre and libre < AVISO_MEMORIA_GB and _ESTADO.get("avisado") != _ESTADO["seccion"]:
        _ESTADO["avisado"] = _ESTADO["seccion"]          # un aviso por sección, no uno por celda
        print(f"AVISO: quedan {libre:.1f} GB de memoria libre. Cierre otros kernels o programas antes de seguir.")
    _ESTADO["inicio"] = None


def activar(grupo: str = "") -> None:
    """Registra los eventos de IPython (una sola vez por kernel)."""
    from IPython import get_ipython
    ip = get_ipython()
    _ESTADO["grupo"] = grupo
    if ip is None or _ESTADO["activo"]:
        return
    ip.events.register("pre_run_cell", _antes)
    ip.events.register("post_run_cell", _despues)
    _ESTADO["activo"] = True


def tabla() -> pd.DataFrame:
    return pd.DataFrame(_ESTADO["filas"])


def reporte(guardar: bool = True, mas_lentas: int = 15) -> pd.DataFrame:
    """Tiempo por sección, celdas más lentas y memoria mínima libre; guarda el detalle en CSV."""
    from IPython.display import Markdown, display
    t = tabla()
    if t.empty:
        print("Sin registro de tiempos (tiempos.activar() no se llamo al inicio).")
        return t
    por_seccion = t.groupby("seccion", sort=False).agg(celdas=("n", "size"), segundos=("segundos", "sum"),
                                                       memoria_proceso_max_gb=("memoria_proceso_gb", "max"),
                                                       memoria_libre_min_gb=("memoria_libre_gb", "min"))
    por_seccion["minutos"] = (por_seccion["segundos"] / 60).round(2)
    por_seccion["porcentaje"] = (100 * por_seccion["segundos"] / por_seccion["segundos"].sum()).round(1)
    display(Markdown(f"**Tiempo total: {t['segundos'].sum() / 60:.1f} min** en {len(t)} celdas; memoria libre mínima "
                     f"{t['memoria_libre_gb'].min():.1f} GB; memoria máxima del proceso {t['memoria_proceso_gb'].max():.1f} GB."))
    display(por_seccion)
    display(Markdown(f"**Las {mas_lentas} celdas más lentas**"))
    display(t.sort_values("segundos", ascending=False).head(mas_lentas))
    if guardar:
        config.DIR_TIEMPOS.mkdir(parents=True, exist_ok=True)
        sello = datetime.now().strftime("%Y%m%d_%H%M")
        t.to_csv(config.DIR_TIEMPOS / f"tfm_{_ESTADO['grupo'] or 'notebook'}_{sello}_celdas.csv", index=False)
        por_seccion.to_csv(config.DIR_TIEMPOS / f"tfm_{_ESTADO['grupo'] or 'notebook'}_{sello}_secciones.csv")
        if _COSTO:
            costo().to_csv(config.DIR_TIEMPOS / f"tfm_{_ESTADO['grupo'] or 'notebook'}_{sello}_costo_modelos.csv", index=False)
    return por_seccion


# --- Costo de los modelos -------------------------------------------------------------------------

@contextmanager
def medir(modelo: str, etapa: str, unidad: str = "", **extra):
    """Mide el tiempo y el aumento de memoria de una etapa de un modelo (ajuste, búsqueda, predicción...).

        with tiempos.medir("SARIMAX E2-MOS", "ajuste", unidad=n0):
            r = sarima.ajustar(...)
    """
    rss0, _ = _memoria()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rss1, _ = _memoria()
        _COSTO.append({"modelo": modelo, "etapa": etapa, "unidad": unidad, "segundos": round(time.perf_counter() - t0, 3),
                       "incremento_memoria_gb": round(rss1 - rss0, 3), **extra})


def registrar_costo(modelo: str, etapa: str, segundos: float, unidad: str = "", **extra) -> None:
    """Registra un tiempo medido por otro medio (por ejemplo, el de una búsqueda de scikit-learn)."""
    _COSTO.append({"modelo": modelo, "etapa": etapa, "unidad": unidad, "segundos": round(float(segundos), 3), **extra})


def costo() -> pd.DataFrame:
    return pd.DataFrame(_COSTO)
