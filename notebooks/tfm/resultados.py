"""
resultados.py
=============
Consolidacón de resultados en TFT_Latex/resultados.json, del que se alimenta resultados.tex (y por
tanto el documento del TFM). Estructura jerarquica:

    meta                                    corte, particiones, semilla, fecha de generación
    localidades.<G01>.datos.<...>           extracción, corrección horaria, imputación, días válidos
    localidades.<G01>.objetivos.<...>       estadísticos de Yf y PR por unidad
    localidades.<G01>.modelos.<id>.<...>    objetivo, horizonte, escenario, orden, ecuación, métricas
    comparación.<...>                       mejores modelos y pruebas de Diebold-Mariano

En LaTeX se lee por ruta, p. ej. \\res{localidades.G01.objetivos.inv_667af33d.yf_diaria_media}.
Las claves solo usan letras, números y guion bajo (seguras para jsonparse).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import config

_RE_CLAVE = re.compile(r"[^A-Za-z0-9_]")


def _clave(k: Any) -> str:
    return _RE_CLAVE.sub("_", str(k))


def _limpio(v: Any) -> Any:
    """Convierte a tipos JSON (numpy, pandas, NaN -> None) y redondea a 6 cifras significativas."""
    if isinstance(v, dict):
        return {_clave(k): _limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_limpio(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else float(f"{f:.6g}")
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, pd.Timedelta):
        return str(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def cargar() -> dict:
    if config.ARCHIVO_RESULTADOS.exists():
        try:
            return json.loads(config.ARCHIVO_RESULTADOS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def guardar(datos: dict) -> None:
    datos.setdefault("meta", {}).pop("version", None)
    datos["meta"].update({
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corte_utc": config.CORTE_UTC.isoformat(), "particiones": config.PARTICIONES, "semilla": config.SEMILLA,
    })
    config.ARCHIVO_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    config.ARCHIVO_RESULTADOS.write_text(json.dumps(_limpio(datos), ensure_ascii=False, indent=2), encoding="utf-8")


class _Cerrojo:
    """Cerrojo entre procesos con un archivo .lock (creacón exclusiva): permite ejecutar los cuadernos de
    varias localidades a la vez sin que una escritura de resultados.json pise a otra."""

    def __init__(self, ruta, espera_max: float = 120.0):
        self.ruta = ruta.with_name(ruta.name + ".lock")
        self.espera_max = espera_max

    def __enter__(self):
        import os
        import time
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - t0 > self.espera_max:      # cerrojo huerfano de un proceso caido
                    try:
                        os.remove(self.ruta)
                    except OSError:
                        pass
                    t0 = time.time()
                time.sleep(0.2)

    def __exit__(self, *exc):
        import os
        os.close(self.fd)
        try:
            os.remove(self.ruta)
        except OSError:
            pass


def registrar(ruta: str, valor: Any) -> None:
    """Escribe valor en la ruta 'a.b.c' (crea los niveles que falten) y guarda, con cerrojo entre procesos."""
    with _Cerrojo(config.ARCHIVO_RESULTADOS):
        datos = cargar()
        nodo = datos
        partes = [_clave(p) for p in ruta.split(".")]
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = _limpio(valor)
        guardar(datos)


def leer(ruta: str, defecto: Any = None) -> Any:
    nodo = cargar()
    for p in ruta.split("."):
        if not isinstance(nodo, dict) or _clave(p) not in nodo:
            return defecto
        nodo = nodo[_clave(p)]
    return nodo
