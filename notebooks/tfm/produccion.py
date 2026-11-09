"""
produccion.py
=============
Los dos modelos que se emiten cada día en la operación, descritos por su orden y sus coeficientes:

    * el SARIMAX E2-MOS horario del día siguiente (H01DH), con el pronóstico IFS como exógena;
    * el SARIMAX diario de 1 a 7 días, que con la climatología por el PR reciente forma el pronóstico a 30 días
      (H07DD y H30DD).

ajustar_horario y ajustar_diario los ajustan con todo lo anterior a un corte, con el orden que eligió el cuaderno de la
localidad. Para pronosticar no se reajusta nada: con_coeficientes construye el modelo con la serie hasta el origen y
lo filtra con los coeficientes del ajuste (el filtro de Kalman actualiza el estado con las observaciones nuevas).
Lo usa la evaluación en vivo (07_vivo).
"""

from __future__ import annotations

import ast
import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config
from .modelos import horizonte_30d as h30
from .modelos import sarima

FORMATO = 1                        # formato del archivo JSON de un modelo guardado


def describir(resultado, **meta) -> dict:
    """Orden, coeficientes y metadatos de un SARIMAX ajustado: lo que necesita con_coeficientes para pronosticar."""
    m = resultado.model
    return {"formato": FORMATO, "orden": list(m.order), "estacional": list(m.seasonal_order), "tendencia": m.trend,
            "exog": list(m.exog_names or []), "parametros": {k: float(v) for k, v in resultado.params.items()},
            "aic": float(resultado.aic), "observaciones": int(resultado.nobs),
            "entrenado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **meta}


def con_coeficientes(guardado: dict, y: pd.Series, exog: pd.DataFrame | None = None):
    """Modelo con la serie hasta hoy y los coeficientes guardados, sin reajustar (filtro de Kalman)."""
    mod = SARIMAX(y, exog=exog, order=tuple(guardado["orden"]), seasonal_order=tuple(guardado["estacional"]),
                  trend=guardado["tendencia"], enforce_stationarity=False, enforce_invertibility=False)
    params = pd.Series(guardado["parametros"]).reindex(mod.param_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.filter(params.to_numpy(), conserve_memory=sarima.MEMORIA_PRONOSTICO)


# --- Ajuste con un corte ---------------------------------------------------------------------------
# Los dos modelos que se emiten cada día, ajustados con todo lo anterior al corte y con el orden que eligió el cuaderno
# de la localidad (lo usa la evaluación en vivo, 07_vivo)
ORDEN_H2_DEFECTO = ((1, 0, 0), (1, 0, 1, 24))
ORDEN_D_DEFECTO = (1, 0, 0)


def orden_h2(dec: dict) -> tuple:
    """Orden del SARIMAX horario elegido en el cuaderno de la localidad ("(1, 0, 0) x (1, 0, 1, 24)")."""
    for k, v in dec.get("modelos", {}).items():
        if k.startswith("sarimax_h2_") and isinstance(v, dict) and "orden" in v:
            a, b = v["orden"].split(" x ")
            return ast.literal_eval(a), ast.literal_eval(b)
    return ORDEN_H2_DEFECTO


def orden_diario(dec: dict) -> tuple:
    """Orden del SARIMAX diario elegido en el cuaderno. En resultados.json las claves están saneadas
    ("SARIMAX_1__0__2__E2_MOS" por "SARIMAX(1, 0, 2) E2-MOS"); se aceptan las dos formas."""
    for k, v in dec.get("modelos", {}).items():
        if k.startswith("h3_") and isinstance(v, dict):
            for nombre in v.get("yf", {}):
                m = re.match(r"^SARIMAX_(\d+)__(\d+)__(\d+)_", nombre) or re.match(r"^SARIMAX\((\d+), (\d+), (\d+)\)", nombre)
                if m:
                    return tuple(int(x) for x in m.groups())
    return ORDEN_D_DEFECTO


def ajustar_horario(u, corte_utc: pd.Timestamp) -> dict | None:
    """SARIMAX E2-MOS horario del día siguiente ajustado con las horas anteriores a `corte_utc`, descrito como lo
    guarda guardar(); None si el ajuste queda explosivo."""
    from . import pipeline
    y = u.horario["yf"].where(u.horario["en_regimen"]).asfreq("1h")
    y = y.loc[y.index < corte_utc]
    x = pipeline.yr_exogena(u, "ifs025_pronostico_d1", y.index)
    o_h, e_h = orden_h2(u.decisiones)
    resultado = sarima.ajustar(y, o_h, e_h, exog=x)
    return describir(resultado, corte_utc=str(corte_utc)) if sarima.estable(resultado) else None


def ajustar_diario(u, corte: pd.Timestamp) -> dict | None:
    """SARIMAX diario de 1 a 7 días ajustado con los días anteriores a `corte` (fecha local), con los percentiles
    de su error por antelación en los últimos 60 días (bandas del pronóstico a 30 días). Si el orden del cuaderno
    queda explosivo se prueba el orden por defecto; si también lo es, None."""
    from . import pipeline
    y_d = u.diario["yf"].where(u.diario["en_regimen"]).asfreq("D")
    y_d = y_d.loc[y_d.index < pd.Timestamp(corte)]
    leads = pipeline.yr_diario_por_antelacion(u)
    x_d = leads[1].reindex(y_d.index).ffill().fillna(0).to_frame("yr")

    def exog_lead(origen, ventana):
        return pd.DataFrame({"yr": [leads[h].get(d, np.nan) for h, d in enumerate(ventana, start=1)]},
                            index=ventana).ffill().fillna(0)

    o_d = orden_diario(u.decisiones)
    resultado = sarima.ajustar(y_d, o_d, exog=x_d)
    if not sarima.estable(resultado) and tuple(o_d) != ORDEN_D_DEFECTO:
        resultado = sarima.ajustar(y_d, ORDEN_D_DEFECTO, exog=x_d)
    if not sarima.estable(resultado):
        return None
    origenes = [d - pd.Timedelta(days=1) for d in y_d.index[-60:]]
    pr_d = sarima.pronostico_por_origen(resultado, y_d, origenes, 7, x_d, exog_futuro=exog_lead)
    errores = h30.cuantiles_error(pr_d, y_d)
    return describir(resultado, hasta=str(y_d.index.max().date()), corte=str(pd.Timestamp(corte).date()),
                     errores_h=errores.round(5).to_dict(orient="index"))


# --- Series en formato de tabla ----------------------------------------------------------------------

def serie_web(prediccion: pd.Series, inferior: pd.Series | None = None, superior: pd.Series | None = None,
              real: pd.Series | None = None) -> list[dict]:
    """[{t, predicción, inferior, superior, real}] con t en ISO 8601 hora de Colombia (formato de Recharts)."""
    idx = prediccion.index
    filas = []
    for t in idx:
        tt = pd.Timestamp(t)
        tt = tt.tz_localize(config.ZONA) if tt.tz is None else tt.tz_convert(config.ZONA)
        f = {"t": tt.isoformat(), "prediccion": _num(prediccion.get(t))}
        for k, s in (("inferior", inferior), ("superior", superior), ("real", real)):
            if s is not None:
                f[k] = _num(s.get(t))
        filas.append(f)
    return filas


def _num(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 5)


def pronostico_30_dias(guardado_d: dict | None, y_d: pd.Series, exog_d: pd.DataFrame | None, yr_futuro_d: dict[int, float],
                       yr_d: pd.Series, clim: pd.DataFrame, kwp: float, errores: pd.DataFrame | None,
                       origen: pd.Timestamp, **opciones) -> pd.DataFrame:
    """Yf diario de D+1 a D+30 (modelo diario de 1 a 7 días si hay pronóstico meteorológico; valor central
    climatológico por PR reciente después). opciones: centro, anomalía, beta, clim_fase y bandas de
    horizonte_30d.pronosticar (los que eligió el cuaderno de la localidad)."""
    h3 = None
    if guardado_d is not None and exog_d is not None:
        res = con_coeficientes(guardado_d, y_d.loc[:origen], exog_d.loc[:origen])
        dias = [origen + pd.Timedelta(days=h) for h in range(1, 8)]
        x_fut = pd.DataFrame({"yr": [yr_futuro_d.get(h, np.nan) for h in range(1, 8)]}, index=dias)
        validos = x_fut["yr"].notna().to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = res.get_forecast(7, exog=x_fut.fillna(0.0)).predicted_mean.to_numpy()
        h3 = pd.DataFrame({"h": range(1, 8), "yhat": np.where(validos, np.clip(pr, 0, None), np.nan)})
    pr30 = h30.pr_reciente(y_d, yr_d, origen)
    return h30.pronosticar(origen, clim, pr30, kwp, h3, yr_futuro_d, errores, **opciones)


