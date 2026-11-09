"""
estructura.py
=============
Estructura de las series antes de modelar (Hyndman y Athanasopoulos, 2021): descomposición (clásica y STL), estacionalidad de 24 h frente a la anual
(periodograma, partición de la varianza y climatología), y pruebas de estacionariedad ADF y KPSS.

Las pruebas necesitan series sin huecos: se usa el tramo continuo más largo (sin rellenar nada), y se
informa su longitud. La descomposición de los objetivos usa en cambio la serie completa del entrenamiento
(serie_completa_diagnostico), con los huecos rellenados solo para ese diagnóstico y su fracción informada.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller, kpss

from . import config


def tramo_continuo_mas_largo(s: pd.Series, rellenar_max: int = 0) -> pd.Series:
    """El tramo más largo de la serie sin NaN (para pruebas que no admiten huecos).

    rellenar_max: antes de buscar el tramo se interpolan, solo para este diagnóstico, los huecos de hasta
    ese número de pasos. Sin ello, unas pocas horas aisladas (fallas, huecos de transmisión) parten la
    serie en tramos tan cortos que las pruebas pierden potencia."""
    if rellenar_max:
        from .imputacion import _interpolar_bloques_cortos
        s, _ = _interpolar_bloques_cortos(s, rellenar_max, "time")
    ok = s.notna().to_numpy()
    if not ok.any():
        return s.iloc[:0]
    grupo = np.cumsum(~ok)
    largo = pd.Series(ok).groupby(grupo).sum()
    g = largo.idxmax()
    return s[(grupo == g) & ok]


def serie_completa_diagnostico(s: pd.Series, referencia: pd.Series | None = None, corto: int = 3,
                               minimo: float = 0.2, k_max: float = 1.2) -> tuple[pd.Series, pd.Series]:
    """La serie completa, sin huecos, para la descomposición y las pruebas que no admiten faltantes.

    Solo para estos diagnósticos (los modelos no la usan): los huecos de hasta `corto` pasos se interpolan
    en el tiempo y los más largos, si hay referencia (Yr de cielo despejado, kWh/kWp), con el índice de
    cielo despejado k = s / referencia: k se calcula solo con sol alto (referencia > mínimo), se acota en
    [0, k_max] y se interpola en el tiempo, y el valor es k por la referencia (0 de noche), lo que conserva
    la campana del día. Sin referencia, los huecos largos se interpolan en el tiempo. Los extremos sin dato
    se recortan. Devuelve la serie y la máscara de lo rellenado, para informar su fracción y medir las
    varianzas solo en lo observado."""
    if not s.notna().any():
        return s.iloc[:0], pd.Series(False, index=s.index[:0])
    s = s.loc[s.first_valid_index():s.last_valid_index()]
    from .imputacion import _interpolar_bloques_cortos
    salida, _ = _interpolar_bloques_cortos(s, corto, "time")
    lineal = salida.interpolate(method="time", limit_area="inside")
    if referencia is not None:
        r = referencia.reindex(salida.index).to_numpy(dtype=float)
        con_sol = r > minimo
        k = pd.Series(np.where(con_sol, salida.to_numpy(dtype=float) / np.where(con_sol, r, 1.0), np.nan), index=salida.index)
        k_int = k.clip(0, k_max).interpolate(method="time", limit_area="inside").to_numpy()
        hueco = salida.isna().to_numpy()
        fisico = np.where(np.isfinite(r), np.nan_to_num(k_int) * np.clip(r, 0, None), np.nan)
        usar = hueco & np.isfinite(k_int) & np.isfinite(fisico)
        salida = lineal.copy()
        salida[usar] = fisico[usar]
    else:
        salida = lineal
    return salida, s.isna() & salida.notna()


def estacionariedad(s: pd.Series, nombre: str = "", rellenar_max: int = 0) -> dict:
    """ADF (H0: raíz unitaria, no estacionaria) y KPSS (H0: estacionaria alrededor de una constante).

    Conclusión conjunta: ADF rechaza y KPSS no rechaza -> estacionaria; ADF no rechaza y KPSS rechaza
    -> no estacionaria (diferenciar); ambos rechazan -> estacionaria con cambios de nivel o
    heterocedasticidad; ninguno rechaza -> datos insuficientes para decidir."""
    t = tramo_continuo_mas_largo(s, rellenar_max)
    if len(t) < 30 or np.nanstd(t.to_numpy()) == 0:      # tramo corto o constante: no hay prueba posible
        return {"serie": nombre, "n": len(t)}
    adf = adfuller(t.to_numpy(), autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # KPSS avisa cuando el p-valor sale de su tabla
        kp = kpss(t.to_numpy(), regression="c", nlags="auto")
    rechaza_adf, rechaza_kpss = adf[1] < 0.05, kp[1] < 0.05
    conclusion = {(True, False): "estacionaria", (False, True): "no estacionaria",
                  (True, True): "estacionaria con cambios de nivel o varianza", (False, False): "indeterminada"}
    return {"serie": nombre, "n": len(t), "desde": t.index.min(), "hasta": t.index.max(),
            "adf_estadistico": adf[0], "adf_p": adf[1], "kpss_estadistico": kp[0], "kpss_p": kp[1],
            "conclusion": conclusion[(rechaza_adf, rechaza_kpss)]}


def periodograma(s: pd.Series, paso_horas: float = 1.0, picos: int = 8) -> pd.DataFrame:
    """Periodos (en horas y días) con más potencia espectral y su fracción de la varianza total.
    La serie se centra y sus huecos se rellenan con la media horaria (solo para este diagnóstico)."""
    x = s.copy()
    h = x.index.tz_convert(config.ZONA).hour if x.index.tz is not None else x.index.hour
    x = x.fillna(x.groupby(h).transform("mean")).fillna(x.mean())
    f, p = periodogram(x.to_numpy() - x.mean(), fs=1.0 / paso_horas, scaling="spectrum")
    f, p = f[1:], p[1:]
    orden = np.argsort(p)[::-1][:picos]
    return pd.DataFrame({"periodo_h": 1 / f[orden], "periodo_dias": 1 / f[orden] / 24,
                         "fraccion_varianza": p[orden] / p.sum()}).reset_index(drop=True)


def particion_varianza(s: pd.Series) -> pd.Series:
    """Que parte de la varianza horaria explica la hora del día (ciclo de 24 h), cuanto agrega el mes
    (ciclo anual sobre el diario) y cuanto queda para el tiempo meteorológico (residuo)."""
    x = s.dropna()
    loc = x.index.tz_convert(config.ZONA)
    var = x.var()
    hora = x.groupby(loc.hour).transform("mean")
    mes_hora = x.groupby([loc.month, loc.hour]).transform("mean")
    return pd.Series({"ciclo_diario": 1 - ((x - hora) ** 2).mean() / var,
                      "ciclo_anual_adicional": (((x - hora) ** 2).mean() - ((x - mes_hora) ** 2).mean()) / var,
                      "residuo_meteorologico": ((x - mes_hora) ** 2).mean() / var})


def climatologia_mensual(ghi_horaria: pd.Series) -> pd.DataFrame:
    """Irradiacion diaria media por mes (kWh/m2/dia) y su variabilidad entre años."""
    loc = ghi_horaria.copy()
    loc.index = (loc.index - pd.Timedelta(seconds=1)).tz_convert(config.ZONA)
    diaria = loc.resample("D").sum(min_count=20) / 1000.0
    tabla = diaria.groupby([diaria.index.year, diaria.index.month]).mean().unstack(0)
    return pd.DataFrame({"media": tabla.mean(axis=1), "desviacion_entre_anios": tabla.std(axis=1),
                         "minimo": tabla.min(axis=1), "maximo": tabla.max(axis=1), "anios": tabla.notna().sum(axis=1)})
