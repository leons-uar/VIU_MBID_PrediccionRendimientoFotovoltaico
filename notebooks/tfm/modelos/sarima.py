"""
sarima.py
=========
Familia SARIMA con statsmodels (clase SARIMAX), en notación de Box y Jenkins:

    phi_p(B) Phi_P(B^s) (1 - B)^d (1 - B^s)^D y_t = c + theta_q(B) Theta_Q(B^s) eps_t,   eps_t ~ RB(0, sigma^2)

    B          operador de rezago: B y_t = y_{t-1}, B^s y_t = y_{t-s}
    phi_p(B)   = 1 - phi_1 B - ... - phi_p B^p             parte autorregresiva (AR) no estacional
    Phi_P(B^s) = 1 - Phi_1 B^s - ... - Phi_P B^{Ps}         parte autorregresiva estacional
    theta_q(B) = 1 + theta_1 B + ... + theta_q B^q          parte de medias móviles (MA) no estacional
    Theta_Q(B^s) = 1 + Theta_1 B^s + ... + Theta_Q B^{Qs}   parte de medias móviles estacional
    d, D       diferencias no estacionales y estacionales;  s = 24 h en la serie horaria

Se sigue la notación de Box y Jenkins (phi AR, theta MA, eps_t); la tabla de equivalencias con otras
notaciones esta en el glosario.

SARIMAX = SARIMA + regresión lineal sobre variables exógenas x_t (errores con estructura SARIMA):
    y_t = beta' x_t + u_t,   con u_t ~ SARIMA(p, d, q)(P, D, Q)_s

El pronóstico del día siguiente se emite desde cada origen (final del día D - 1) con los parámetros
estimados en el entrenamiento, filtrando la serie completa (los datos posteriores al origen no se usan:
la predicción es dinamica desde el origen). Los huecos se tratan como faltantes en el filtro de Kalman.
"""

from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace import kalman_filter as kf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Memoria: con s = 24 el espacio de estados tiene 25 a 49 estados y guardar sus covarianzas para cada hora
# ocupa cientos de MB. Para pronosticar solo hacen falta el estado predicho y su covarianza; para los
# residuos, solo los errores de pronóstico. Lo demás no se guarda.
MEMORIA_PRONOSTICO = (kf.MEMORY_NO_FILTERED_COV | kf.MEMORY_NO_SMOOTHING | kf.MEMORY_NO_GAIN
                      | kf.MEMORY_NO_STD_FORECAST | kf.MEMORY_NO_FORECAST_COV)
MEMORIA_RESIDUOS = MEMORIA_PRONOSTICO | kf.MEMORY_NO_PREDICTED_COV


def ajustar(y: pd.Series, orden=(1, 0, 0), estacional=(0, 0, 0, 0), exog: pd.DataFrame | None = None,
            tendencia: str | None = "c", maxiter: int = 200):
    """Ajusta por máxima verosimilitud. tendencia 'c' (constante) solo si no hay diferenciación."""
    d, D = orden[1], estacional[1]
    tend = tendencia if (d == 0 and D == 0) else None
    modelo = SARIMAX(y, exog=exog, order=orden, seasonal_order=estacional, trend=tend,
                     enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = modelo.fit(disp=False, maxiter=maxiter, method="lbfgs", low_memory=True)
        if not r.mle_retvals.get("converged", True) and maxiter < 500:
            # Sin convergencia se reintenta desde el punto alcanzado con más iteraciones
            r = modelo.fit(start_params=r.params, disp=False, maxiter=500, method="lbfgs", low_memory=True)
        return r


def estable(resultado, tolerancia: float = 1e-3) -> bool:
    """True si los parámetros son finitos y ninguna raíz autorregresiva (incluidas las estacionales) cae
    dentro del círculo unidad más allá de la tolerancia. Con enforce_stationarity=False, un reajuste con pocos
    datos puede quedar explosivo y dar pronósticos que crecen sin límite; la tolerancia admite el
    autorregresivo estacional cercano a 1, que equivale a una diferencia estacional."""
    if not np.all(np.isfinite(np.asarray(resultado.params, dtype=float))):
        return False
    raices = np.atleast_1d(np.asarray(resultado.arroots))
    return bool(np.all(np.abs(raices) >= 1 - tolerancia)) if raices.size else True


def buscar_orden(y: pd.Series, p=(0, 1, 2), d=(0,), q=(0, 1, 2), P=(0,), D=(0,), Q=(0,), s: int = 0,
                 exog: pd.DataFrame | None = None, criterio: str = "aic") -> pd.DataFrame:
    """Busqueda en rejilla por criterio de información (Hyndman y Athanasopoulos, 2021, cap. 9). Devuelve todos
    los ajustes ordenados, con AIC, BIC, p-valor de Ljung-Box de los residuos y tiempo de ajuste."""
    filas = []
    for pp, dd, qq, PP, DD, QQ in itertools.product(p, d, q, P, D, Q):
        if pp == qq == PP == QQ == 0:
            continue
        estacional = (PP, DD, QQ, s) if s else (0, 0, 0, 0)
        t0 = time.time()
        try:
            r = ajustar(y, (pp, dd, qq), estacional, exog)
            lb = diagnostico(r, rezagos=(s or 24,))["ljung_box_p"].iloc[0]
            filas.append({"orden": (pp, dd, qq), "estacional": estacional, "aic": r.aic, "bic": r.bic,
                          "ljung_box_p": lb, "segundos": time.time() - t0, "convergio": r.mle_retvals.get("converged", True)})
        except Exception as e:                        # noqa: BLE001 (un orden que no converge se registra)
            filas.append({"orden": (pp, dd, qq), "estacional": estacional, "error": str(e)[:120]})
    return pd.DataFrame(filas).sort_values(criterio).reset_index(drop=True)


def residuos(resultado) -> pd.Series:
    """Residuos (errores de pronóstico a un paso) del modelo ajustado, sin el arranque del filtro.

    El ajuste se hace en modo de baja memoria (sin residuos guardados), así que se vuelve a filtrar con los
    parámetros. Se descartan las primeras `loglikelihood_burn` observaciones, cuyo error de pronóstico
    proviene de la inicialización difusa y no del modelo (con D = 1 son las primeras 24 h y pueden ser
    enormes o no finitos), y los residuos no finitos de las horas sin dato."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtrado = resultado.model.filter(resultado.params, conserve_memory=MEMORIA_RESIDUOS)
    res = pd.Series(np.asarray(filtrado.resid).ravel(), index=resultado.model.data.row_labels)
    res = res.iloc[int(getattr(resultado, "loglikelihood_burn", 0) or 0):]
    return res[np.isfinite(res)]


def coeficientes(resultado) -> pd.DataFrame:
    """Coeficientes estimados con su error estándar, estadístico z, p-valor e intervalo de confianza del
    95 % (lo mismo que summary().tables[1], pero como tabla de datos exportable)."""
    ic = resultado.conf_int()
    return pd.DataFrame({"coeficiente": resultado.params, "error_estandar": resultado.bse, "z": resultado.zvalues,
                         "p_valor": resultado.pvalues, "ic95_inf": ic.iloc[:, 0], "ic95_sup": ic.iloc[:, 1]})


def tabla_autocorrelacion(s: pd.Series, rezagos: int = 72, ljung_box: bool = True) -> pd.DataFrame:
    """ACF y PACF por rezago con su banda de confianza del 95 % (la que dibuja plot_acf: +-1,96/sqrt(n)
    para la PACF y la de Bartlett para la ACF) y, si se pide, la prueba de Ljung-Box acumulada."""
    from statsmodels.tsa.stattools import acf, pacf
    x = pd.Series(s).dropna().to_numpy()
    rezagos = min(rezagos, len(x) // 2 - 1)            # la PACF solo admite rezagos hasta la mitad de la muestra
    if rezagos < 1 or len(x) <= rezagos + 1:
        return pd.DataFrame(columns=["rezago", "acf", "acf_ic95_inf", "acf_ic95_sup", "pacf", "pacf_ic95_inf", "pacf_ic95_sup"])
    r, ic_r = acf(x, nlags=rezagos, alpha=0.05, result_object=False)
    p, ic_p = pacf(x, nlags=rezagos, alpha=0.05, method="ywm")
    t = pd.DataFrame({"rezago": np.arange(rezagos + 1), "acf": r, "acf_ic95_inf": ic_r[:, 0] - r,
                      "acf_ic95_sup": ic_r[:, 1] - r, "pacf": p, "pacf_ic95_inf": ic_p[:, 0] - p,
                      "pacf_ic95_sup": ic_p[:, 1] - p})
    t["acf_significativa"] = (t["acf"] < t["acf_ic95_inf"]) | (t["acf"] > t["acf_ic95_sup"])
    t["pacf_significativa"] = (t["pacf"] < t["pacf_ic95_inf"]) | (t["pacf"] > t["pacf_ic95_sup"])
    if ljung_box:
        lb = acorr_ljungbox(x, lags=list(range(1, rezagos + 1)), return_df=True)
        t["ljung_box_q"] = np.r_[np.nan, lb["lb_stat"].to_numpy()]
        t["ljung_box_p"] = np.r_[np.nan, lb["lb_pvalue"].to_numpy()]
    return t.set_index("rezago")


def diagnostico(resultado, rezagos=(24, 48)) -> pd.DataFrame:
    """Prueba de Ljung-Box de los residuos (H0: ruido blanco) en los rezagos indicados. El ajuste se hace
    en modo de baja memoria (sin residuos guardados), así que se vuelve a filtrar con los parámetros."""
    res = residuos(resultado)
    if len(res) <= max(rezagos) or not np.isfinite(res.var()) or res.var() == 0:
        return pd.DataFrame({"rezago": list(rezagos), "ljung_box_estadistico": np.nan, "ljung_box_p": np.nan})
    lb = acorr_ljungbox(res, lags=list(rezagos), return_df=True)
    return pd.DataFrame({"rezago": list(rezagos), "ljung_box_estadistico": lb["lb_stat"].to_numpy(),
                         "ljung_box_p": lb["lb_pvalue"].to_numpy()})


def _coma(texto: str) -> str:
    """Coma decimal dentro de una formula ({,} evita el espacio que LaTeX pone tras la coma en modo matematico)."""
    return texto.replace(".", "{,}")


def _cientifico(v: float) -> str:
    texto = f"{v:.4g}"
    if "e" in texto:
        m, e = texto.split("e")
        texto = f"{m} \\times 10^{{{int(e)}}}"
    return _coma(texto)


SIMBOLOS_EXOGENAS = {"yr": "Y_{r,t}"}


def ecuacion(resultado, y_t: str = "Y_{f,t}") -> str:
    """Ecuacion ajustada en LaTeX (coeficientes redondeados, coma decimal), en notación de Box y Jenkins. La
    constante se omite si es nula al redondear."""
    p = resultado.params
    partes_ar, partes_ma, partes_sar, partes_sma = [], [], [], []
    for k, v in p.items():
        if k.startswith("ar.S.L"):
            partes_sar.append(_coma(f"{-v:+.3f}") + f"B^{{{k.split('L')[-1]}}}")
        elif k.startswith("ar.L"):
            partes_ar.append(_coma(f"{-v:+.3f}") + f"B^{{{k.split('L')[-1]}}}")
        elif k.startswith("ma.S.L"):
            partes_sma.append(_coma(f"{v:+.3f}") + f"B^{{{k.split('L')[-1]}}}")
        elif k.startswith("ma.L"):
            partes_ma.append(_coma(f"{v:+.3f}") + f"B^{{{k.split('L')[-1]}}}")
    o, so = resultado.model.order, resultado.model.seasonal_order
    izq = "".join(f"(1{''.join(x)})" for x in (partes_ar, partes_sar) if x)
    dif = ("(1-B)" if o[1] == 1 else f"(1-B)^{o[1]}" if o[1] > 1 else "") + \
          (f"(1-B^{{{so[3]}}})" if so[1] == 1 else "")
    der = "".join(f"(1{''.join(x)})" for x in (partes_ma, partes_sma) if x)
    exog = [k for k in p.index if not k.startswith(("ar.", "ma.", "sigma2", "intercept", "drift"))]
    # Las exógenas conocidas con su símbolo del documento (yr es la productividad de referencia Y_r, no el objetivo);
    # las demás, con su nombre de columna
    simbolo = {k: SIMBOLOS_EXOGENAS.get(k, f"\\mathrm{{{k.replace('_', chr(92) + '_')}}}_t") for k in exog}
    # beta' x con el signo de cada coeficiente: Y - (b1 x1 + b2 x2) = Y - b1 x1 - b2 x2
    reg = "".join(_coma(f" {'-' if p[k] >= 0 else '+'} {abs(p[k]):.3f}") + f"\\,{simbolo[k]}" for k in exog)
    cte = _coma(f"{p['intercept']:.4f}") + " + " if "intercept" in p.index and abs(p["intercept"]) >= 5e-5 else ""
    lado_izq = f"{izq}{dif}{y_t}" if not exog else f"{izq}{dif}\\left({y_t}{reg}\\right)"
    return f"{lado_izq} = {cte}{der}\\varepsilon_t,\\quad \\hat\\sigma^2 = {_cientifico(p.get('sigma2', np.nan))}"


def pronostico_por_origen(resultado, y_completo: pd.Series, origenes: list[pd.Timestamp], pasos: int,
                          exog_completo: pd.DataFrame | None = None, exog_futuro=None) -> pd.DataFrame:
    """Pronosticos dinamicos de `pasos` pasos desde cada origen, con los parámetros del entrenamiento.

    Sin `exog_futuro`, el modelo se reconstruye sobre la serie completa (entrenamiento + validación +
    prueba) solo para filtrar; los parámetros no se reestiman. Desde cada origen la predicción es
    dinamica: usa los datos hasta el origen y sus propias predicciones después.

    Con `exog_futuro` (DataFrame con el índice de la serie, o función (origen, indice_ventana) ->
    DataFrame), la historia hasta el origen usa `exog_completo` y la ventana pronosticada usa
    `exog_futuro`. Sirve para entrenar con exógenas observadas y pronosticar con las pronosticadas
    ("perfect prognosis") o para usar un pronóstico distinto según la antelación (H3). No hace falta
    volver a filtrar: en una regresión con errores SARIMA, y^ = beta' x + u^, y el pronóstico del error
    u^ solo depende de la historia, así que basta sumar beta' (x_futuro - x_completo) en la ventana.

    Devuelve columnas origen, t, h (paso) y yhat."""
    m = resultado.model
    d, D = m.order[1], m.seasonal_order[1]
    tend = "c" if ("intercept" in resultado.params.index and d == 0 and D == 0) else None
    posicion = pd.Series(np.arange(len(y_completo)), index=y_completo.index)
    beta = resultado.params[list(exog_completo.columns)] if exog_completo is not None else None
    modelo = SARIMAX(y_completo, exog=exog_completo, order=m.order, seasonal_order=m.seasonal_order, trend=tend,
                     enforce_stationarity=False, enforce_invertibility=False)
    filtrado = modelo.filter(resultado.params, conserve_memory=MEMORIA_PRONOSTICO)
    filas = []
    for o in origenes:
        if o not in posicion.index:
            continue
        i0 = int(posicion[o]) + 1
        i1 = min(i0 + pasos - 1, len(y_completo) - 1)
        if i0 > i1:
            continue
        pred = np.asarray(filtrado.get_prediction(start=i0, end=i1, dynamic=True).predicted_mean, dtype=float)
        ventana = y_completo.index[i0:i1 + 1]
        if exog_futuro is not None:
            x_fut = exog_futuro(o, ventana) if callable(exog_futuro) else exog_futuro.reindex(ventana)
            ajuste = (x_fut[beta.index].to_numpy(dtype=float) - exog_completo.loc[ventana, beta.index].to_numpy(dtype=float)) @ beta.to_numpy()
            pred = pred + ajuste
        filas.append(pd.DataFrame({"origen": o, "t": ventana, "h": np.arange(1, i1 - i0 + 2), "yhat": pred}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["origen", "t", "h", "yhat"])
