"""
metricas.py
===========
Métricas de verificación de pronósticos, siguiendo a Yang et al. (2020), "Verification of deterministic
solar forecasts":

    MAE   = media |y - y^|                    error típico, en las unidades del objetivo
    RMSE  = raiz(media (y - y^)^2)            penaliza más los errores grandes
    MBE   = media (y^ - y)                    sesgo: positivo si el modelo sobrestima
    nMAE, nRMSE = MAE, RMSE / media(y) x 100  relativos, para comparar objetivos y localidades
    R2    = 1 - SSE / SST
    skill = 1 - RMSE_modelo / RMSE_referencia (persistencia inteligente, climatología o CLIPER)

No se usa el MAPE: con valores cercanos a cero (amanecer, días muy nublados) explota.

Regla del TFM: las métricas se calculan SOLO sobre valores observados del objetivo (nunca imputados),
con la máscara `observado`.

diebold_mariano() compara la precisión de dos pronósticos sobre el mismo conjunto (Diebold y Mariano,
1995), con la corrección para muestras pequeñas y horizontes de h pasos de Harvey, Leybourne y Newbold
(1997). diebold_mariano_diario() hace la misma prueba con la pérdida sumada por día, y conjunto_confianza()
reúne los modelos que no se distinguen del mejor (Hansen, Lunde y Nason, 2011), lo que controla las
comparaciones múltiples entre muchos modelos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def _alinear(y, yhat, observado=None) -> tuple[np.ndarray, np.ndarray]:
    y = pd.Series(y)
    yhat = pd.Series(yhat, index=y.index) if not isinstance(yhat, pd.Series) else yhat.reindex(y.index)
    ok = y.notna() & yhat.notna()
    if observado is not None:
        ok &= pd.Series(observado, index=y.index).fillna(False).astype(bool)
    return y[ok].to_numpy(dtype=float), yhat[ok].to_numpy(dtype=float)


def evaluar(y, yhat, observado=None, rmse_referencia: float | None = None) -> dict:
    """Metricas de un pronóstico. rmse_referencia: RMSE de la referencia en las mismas muestras."""
    a, b = _alinear(y, yhat, observado)
    n = len(a)
    if n == 0:
        return {"n": 0}
    e = b - a
    media = a.mean()
    sst = ((a - media) ** 2).sum()
    mae, rmse = float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))
    salida = {"n": n, "mae": mae, "rmse": rmse, "mbe": float(e.mean()),
              "nmae_pct": 100 * mae / media if media else np.nan, "nrmse_pct": 100 * rmse / media if media else np.nan,
              "r2": float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan}
    if rmse_referencia:
        salida["skill"] = 1 - rmse / rmse_referencia
    return salida


def diebold_mariano(y, yhat1, yhat2, h: int = 1, potencia: int = 2, observado=None) -> dict:
    """Prueba de igual precisión predictiva entre dos pronósticos de la misma serie.

    d_t = |e1_t|^p - |e2_t|^p. H0: E[d] = 0. Estadístico DM con la varianza de largo plazo de d (h - 1
    autocovarianzas) y corrección de Harvey, Leybourne y Newbold; p-valor bilateral con t de n - 1 grados.
    Un estadístico negativo indica que el pronóstico 1 es más preciso."""
    y = pd.Series(y)
    y1 = pd.Series(yhat1).reindex(y.index)
    y2 = pd.Series(yhat2).reindex(y.index)
    ok = y.notna() & y1.notna() & y2.notna()
    if observado is not None:
        ok &= pd.Series(observado, index=y.index).fillna(False).astype(bool)
    e1, e2 = (y1 - y)[ok].to_numpy(), (y2 - y)[ok].to_numpy()
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    n = len(d)
    if n < 10:
        return {"n": n, "dm": np.nan, "p_valor": np.nan}
    dm_media = d.mean()
    dc = d - dm_media
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    var_larga = gamma[0] + 2 * sum(gamma[1:])
    if var_larga <= 0:
        return {"n": n, "dm": np.nan, "p_valor": np.nan}
    dm = dm_media / np.sqrt(var_larga / n)
    correccion = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correccion
    p = 2 * stats.t.sf(np.abs(dm_hln), df=n - 1)
    return {"n": n, "dm": float(dm_hln), "p_valor": float(p), "diferencia_media": float(dm_media)}


def perdidas_diarias(y, pronosticos: dict, observado=None) -> pd.DataFrame:
    """Error cuadrático de cada pronóstico sumado por día local, solo en las horas en que todos tienen valor
    (y en las observadas). Filas: días; columnas: modelos."""
    y = pd.Series(y)
    p = pd.DataFrame({k: pd.Series(v).reindex(y.index) for k, v in pronosticos.items()})
    ok = y.notna() & p.notna().all(axis=1)
    if observado is not None:
        ok &= pd.Series(observado, index=y.index).fillna(False).astype(bool)
    e2 = p[ok].sub(y[ok], axis=0) ** 2
    dia = (e2.index - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).normalize()
    return e2.groupby(dia).sum()


def diebold_mariano_diario(y, yhat1, yhat2, observado=None) -> dict:
    """Diebold-Mariano sobre la diferencia de pérdidas sumada por día (h = 1). Los errores de las horas de un
    mismo día vienen del mismo origen y están correlacionados; al sumarlos por día la serie de diferencias
    queda casi sin autocorrelación y el número de observaciones es el de días independientes."""
    L = perdidas_diarias(y, {"a": yhat1, "b": yhat2}, observado)
    n = len(L)
    if n < 10:
        return {"n": n, "dm": np.nan, "p_valor": np.nan}
    d = (L["a"] - L["b"]).to_numpy()
    var = d.var(ddof=0)
    if var <= 0:
        return {"n": n, "dm": np.nan, "p_valor": np.nan}
    dm = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    return {"n": n, "dm": float(dm), "p_valor": float(2 * stats.t.sf(abs(dm), df=n - 1)), "diferencia_media": float(d.mean())}


def holm(p: pd.Series) -> pd.Series:
    """p-valores ajustados por Holm (1979) para la misma prueba repetida en varias unidades: el k-esimo menor se
    multiplica por (m - k + 1), se fuerza a no decrecer y se acota en 1. Los NaN quedan como NaN."""
    p = pd.Series(p, dtype=float)
    validos = p.dropna().sort_values()
    m = len(validos)
    ajustados = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * validos.to_numpy()))
    return pd.Series(ajustados, index=validos.index).reindex(p.index)


def conjunto_confianza(perdidas: pd.DataFrame, alfa: float = 0.10, bloque: int = 7, repeticiones: int = 1000,
                       semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Model Confidence Set de Hansen, Lunde y Nason (2011): el conjunto de modelos que no se distinguen del mejor
    con una confianza de 1 - alfa, por eliminación sucesiva con el estadístico máximo y bootstrap estacionario
    por bloques de días. pérdidas: filas = días, columnas = modelos (perdidas_diarias). Devuelve el p-valor del
    MCS de cada modelo y si queda incluido."""
    from arch.bootstrap import MCS
    L = perdidas.dropna()
    if len(L) < 20 or L.shape[1] < 2:
        return pd.DataFrame({"p_mcs": np.nan, "incluido": True}, index=perdidas.columns)
    m = MCS(L, size=alfa, reps=repeticiones, block_size=bloque, method="max", bootstrap="stationary", seed=semilla)
    m.compute()
    p = m.pvalues["Pvalue"].reindex(L.columns)
    return pd.DataFrame({"p_mcs": p, "incluido": [c in m.included for c in L.columns]})
