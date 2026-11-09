"""
particiones.py
==============
Partición temporal de los datos, sin barajar: entrenamiento, validación, prueba y en vivo, según las
fechas locales (Colombia) de config.PARTICIONES.

    entrenamiento   2024-01-01 a 2026-05-31   ajuste de modelos y de todo lo que aprende de los datos
    validación      2026-06-01 a 2026-07-31   elección de hiperparámetros, órdenes y variables
    prueba          2026-08-01 a 2026-09-30   evaluación final, una sola vez por modelo
    vivo            desde 2026-10-01          pronósticos reales frente a lo que se mida en octubre

Convención: en una serie con índice UTC de intervalos etiquetados al final, el intervalo pertenece al
día local de (t - 1 s); así la hora que termina a las 00:00 del 1 de junio es todavía del 31 de mayo.
Una serie diaria con índice de fechas locales sin zona se asigna por su fecha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

ORDEN = ("entrenamiento", "validacion", "prueba", "vivo")


def dias_locales(indice) -> pd.DatetimeIndex:
    """Dia local de cada elemento del índice (ver la convención del módulo)."""
    idx = pd.DatetimeIndex(indice)
    if idx.tz is not None:
        return (idx - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).tz_localize(None).normalize()
    return idx.normalize()


def etiquetar(indice, particiones: dict | None = None) -> pd.Series:
    """Nombre de la partición de cada elemento (None si cae fuera de todas)."""
    p = particiones or config.PARTICIONES
    dias = dias_locales(indice)
    etiqueta = np.full(len(dias), None, dtype=object)
    for nombre in ORDEN:
        if nombre not in p:
            continue
        ini, fin = (pd.Timestamp(x) for x in p[nombre])
        etiqueta[np.asarray((dias >= ini) & (dias <= fin))] = nombre
    return pd.Series(etiqueta, index=indice, name="particion")


def dividir(df: pd.DataFrame | pd.Series, particiones: dict | None = None) -> dict[str, pd.DataFrame | pd.Series]:
    """{partición: subconjunto} en orden temporal."""
    e = etiquetar(df.index, particiones).to_numpy()
    return {n: df[e == n] for n in ORDEN}


def resumen(df: pd.DataFrame | pd.Series, particiones: dict | None = None) -> pd.DataFrame:
    """Filas, primer y último día local de cada partición."""
    filas = []
    for n, sub in dividir(df, particiones).items():
        d = dias_locales(sub.index)
        filas.append({"particion": n, "filas": len(sub), "desde": d.min() if len(d) else pd.NaT,
                      "hasta": d.max() if len(d) else pd.NaT, "dias": d.nunique()})
    return pd.DataFrame(filas)


def inicio_validacion(particiones: dict | None = None) -> pd.Timestamp:
    """Primer día local de la validación (frontera que ningún ajuste puede cruzar)."""
    p = particiones or config.PARTICIONES
    return pd.Timestamp(p["validacion"][0])
