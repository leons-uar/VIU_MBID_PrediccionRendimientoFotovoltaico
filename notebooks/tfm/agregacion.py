"""
agregacion.py
=============
Agregación temporal según la NATURALEZA de cada variable, después de imputar en el muestreo nativo.

Reglas (y por que):
- media: magnitudes instantáneas o medias de periodo (potencia, irradiancia, temperatura, tensión).
  La media de las muestras del intervalo estima la media del periodo; multiplicada por la duración da
  la energía (kWh) o la irradiación (kWh/m2).
- suma: acumulados por periodo (precipitación). Sumar conserva el total; promediar lo subestimaría.
- máximo: máximos del periodo (rachas de viento).
- último: contadores (energía acumulada) y estados; la energía del periodo se obtiene después como
  diferencia entre el último valor de un intervalo y el del anterior (ver objetivos.py).

Etiquetado: cada intervalo se etiqueta en su FINAL, cerrado por la derecha: el valor de las 10:00
resume (09:00, 10:00]. Es la convención de la radiación de Open-Meteo y evita desplazar una hora la
radiación frente a la producción. Las series etiquetadas al inicio (NASA POWER) se desplazan antes.

Cobertura: un valor agregado solo se acepta si el intervalo tiene al menos COBERTURA_MINIMA_INTERVALO
de sus muestras esperadas; si no, queda NaN (no se agrega sobre casi nada).
"""

from __future__ import annotations

import pandas as pd

from . import config

REGLA_POR_NATURALEZA = {
    "potencia": "media", "irradiancia": "media", "continua": "media", "contador": "ultimo",
    "estado": "ultimo", "acumulado": "suma", "maximo": "maximo", "texto": None,
}


def agregar(df: pd.DataFrame, reglas: dict[str, str], destino: str, origen: str,
            imputado: pd.DataFrame | None = None, marca: str = "fin",
            cobertura_minima: float = config.COBERTURA_MINIMA_INTERVALO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega una serie regular de paso `origen` a `destino` (p. ej. '5min' -> '1h').

    imputado: máscara booleana (True = valor imputado) para informar la fracción imputada de cada intervalo.
    marca: 'fin' (por defecto) o 'inicio' si la serie de origen etiqueta el inicio del periodo.
    Devuelve (agregada, calidad) donde calidad tiene la cobertura y la fracción imputada por columna.
    """
    d = df.copy()
    if marca == "inicio":
        d.index = d.index + pd.Timedelta(origen)
    esperadas = pd.Timedelta(destino) / pd.Timedelta(origen)
    remuestreo = d.resample(destino, closed="right", label="right")
    salida, calidad = {}, {}
    for col, regla in reglas.items():
        if regla is None or col not in d.columns:
            continue
        r = remuestreo[col]
        valor = {"media": r.mean, "suma": r.sum, "maximo": r.max, "ultimo": r.last}[regla]()
        if regla == "suma":
            valor = valor.where(r.count() > 0)
        cobertura = r.count() / esperadas
        valor = valor.where(cobertura >= cobertura_minima)
        salida[col] = valor
        calidad[f"{col}__cobertura"] = cobertura
        if imputado is not None and col in imputado.columns:
            m = imputado[col].astype(float)
            if marca == "inicio":
                m.index = m.index + pd.Timedelta(origen)
            calidad[f"{col}__imputado"] = m.resample(destino, closed="right", label="right").mean()
    return pd.DataFrame(salida), pd.DataFrame(calidad)


def mensual(diario: pd.DataFrame, reglas: dict[str, str], minimo_dias: int = 20) -> pd.DataFrame:
    g = diario.groupby(diario.index.to_period("M"))
    salida = {}
    for col, regla in reglas.items():
        if regla is None or col not in diario.columns:
            continue
        r = g[col]
        valor = {"media": r.mean, "suma": r.sum, "maximo": r.max, "ultimo": r.last}[regla]()
        salida[col] = valor.where(r.count() >= minimo_dias)
    return pd.DataFrame(salida)
