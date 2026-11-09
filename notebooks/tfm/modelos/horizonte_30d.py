"""
horizonte_30d.py
================
Pronóstico diario del rendimiento a 30 días desde un día de emisión D (origen: final del día D, hora de
Colombia), combinando dos fuentes de información según la antelación h:

    h = 1..7   el mejor modelo del horizonte H3 (SARIMAX diario con el Yr del pronóstico IFS archivado
               con h días de antelación): hasta una semana el pronóstico meteorológico aporta información.
    h = 8..30  climatología: Yr esperado del día del año (ERA5 2015-2024 en el plano del inversor,
               ventana de +-15 días) multiplicado por el PR de los últimos 30 días observados. Más allá
               de una semana el pronóstico numérico ya no supera a la climatología en irradiancia.

    Yf_hat(D + h) = modelo H3                              si h <= 7
                  = PR_30(D) * Yr_clim(día del año de D + h)   si h > 7
    PR_hat(D + h) = Yf_hat / Yr pronosticado (h <= 7) o PR_30(D) (h > 7)
    E_hat(D + h)  = Yf_hat * P0 (kWh)

La climatología se calibra a la fuente observada con que se mide el PR (calibrar_climatologia): si el PR se
calcula con el satélite MSG y la climatología con ERA5, la diferencia de nivel entre ambas fuentes pasaria
entera al pronóstico.

Bandas del 80 %: para h <= 7, los percentiles 10 y 90 del error del modelo en la validación para esa
antelación; para h > 7, PR_30 por los percentiles 10 y 90 del Yr climatológico del día (la variabilidad
del tiempo entre años).

Nada usa datos posteriores al origen: el PR_30 se calcula con los días hasta D y la climatología con
2015-2024. En operación el pronóstico se reemite cada día con los mismos coeficientes; los coeficientes del
modelo de 1 a 7 días se reestiman cada config.REENTRENAR_CADA_X_DIAS días.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONTE = 30
LIMITE_MODELO = 7


def climatologia_diaria(yr_diario: pd.Series, ventana: int = 15) -> pd.DataFrame:
    """Yr climatológico por día del año (1..366): media y percentiles 10, 50 y 90 de todos los días de
    todos los años que caen a +-ventana días del día del año (con vuelta de año)."""
    s = yr_diario.dropna()
    doy = pd.DatetimeIndex(s.index).dayofyear.to_numpy()
    filas = {}
    for d in range(1, 367):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 366 - dist)
        v = s.to_numpy()[dist <= ventana]
        filas[d] = {"media": v.mean(), "p10": np.percentile(v, 10), "p50": np.percentile(v, 50),
                    "p90": np.percentile(v, 90), "dias": len(v)}
    return pd.DataFrame(filas).T.rename_axis("dia_del_ano")


def calibrar_climatologia(clim: pd.DataFrame, yr_observado: pd.Series, yr_fuente_clima: pd.Series,
                          hasta: pd.Timestamp | str | None = None, minimo: int = 30) -> tuple[pd.DataFrame, float]:
    """Lleva la climatología a la escala de la fuente observada con que se calcula el PR.

    El PR reciente es Yf / Yr con la fuente observada elegida (p. ej. el satélite MSG), pero la climatología
    sale de ERA5. Si las dos fuentes difieren en nivel, PR_30 * Yr_clim hereda esa diferencia como sesgo. El
    factor es el cociente de sumas del Yr de ambas fuentes en los días comunes hasta `hasta` (sin datos
    posteriores); con menos de `minimo` días comunes no se calibra (factor 1)."""
    a, b = yr_observado.loc[:hasta], yr_fuente_clima.reindex(yr_observado.index).loc[:hasta]
    ok = a.notna() & b.notna() & (b > 0)
    factor = float(a[ok].sum() / b[ok].sum()) if ok.sum() >= minimo else 1.0
    c = clim.copy()
    c[["media", "p10", "p50", "p90"]] = c[["media", "p10", "p50", "p90"]] * factor
    return c, factor


def pr_reciente(yf_diario: pd.Series, yr_diario: pd.Series, fin: pd.Timestamp, dias: int = 30, minimo: int = 10) -> float:
    """PR de los últimos `dias` días hasta `fin` (incluido) como cociente de sumas, con al menos `minimo`
    días con Yf e Yr observados; si no hay suficientes, se amplía la ventana hasta 90 días."""
    for d in (dias, 60, 90):
        ini = fin - pd.Timedelta(days=d - 1)
        a, b = yf_diario.loc[ini:fin], yr_diario.loc[ini:fin]
        ok = a.notna() & b.notna() & (b > 0)
        if ok.sum() >= minimo:
            return float(a[ok].sum() / b[ok].sum())
    return np.nan


def cuantiles_error(pred_h3: pd.DataFrame, y: pd.Series, q=(0.1, 0.9)) -> pd.DataFrame:
    """Percentiles del error (observado - pronóstico) por antelación h, a partir de pronósticos por origen
    (columnas t, h, yhat)."""
    e = pred_h3.assign(obs=y.reindex(pred_h3["t"]).to_numpy())
    e["error"] = e["obs"] - e["yhat"]
    return e.dropna(subset=["error"]).groupby("h")["error"].quantile(list(q)).unstack().rename(columns={q[0]: "q_inf", q[1]: "q_sup"})


CENTROS = ("media", "mediana", "anomalia", "fase_oni")      # valores centrales candidatos de 8 a 30 días, de simple a complejo


def pendiente_anomalia(yr_largo: pd.Series, ventana: int = 30, horizonte: int = HORIZONTE) -> pd.Series:
    """Cuanto persiste la anomalía reciente de la irradiación. Con la serie larga de ERA5 (independiente de la
    validación y la prueba): A_t = suma de los últimos `ventana` días / su climatología - 1 y
    a_{t+h} = Yr_{t+h} / climatología - 1; la pendiente beta_h = cov(a_{t+h}, A_t) / var(A_t) es el pronóstico de
    mínimos cuadrados de la anomalía a h días dada la anomalía reciente."""
    s = yr_largo.dropna()
    clim = climatologia_diaria(s)["media"]
    c = pd.Series(clim.reindex(np.minimum(pd.DatetimeIndex(s.index).dayofyear, 366)).to_numpy(), index=s.index)
    s, c = s.asfreq("D"), c.asfreq("D")
    a = s / c - 1
    minimo = int(0.8 * ventana)
    A = s.rolling(ventana, min_periods=minimo).sum() / c.where(s.notna()).rolling(ventana, min_periods=minimo).sum() - 1
    beta = {}
    for h in range(1, horizonte + 1):
        par = pd.concat([A, a.shift(-h)], axis=1).dropna()
        beta[h] = float(np.cov(par.iloc[:, 1], par.iloc[:, 0])[0, 1] / par.iloc[:, 0].var()) if len(par) > 100 else 0.0
    return pd.Series(beta, name="beta").rename_axis("h")


def anomalia_reciente(yr_observado: pd.Series, clim: pd.DataFrame, fin: pd.Timestamp, dias: int = 30, minimo: int = 20) -> float:
    """Anomalia de la irradiación de los últimos `dias` días hasta `fin` respecto de la climatología (cociente de sumas)."""
    ini = fin - pd.Timedelta(days=dias - 1)
    s = yr_observado.loc[ini:fin].dropna()
    if len(s) < minimo:
        return 0.0
    c = clim["media"].reindex(np.minimum(pd.DatetimeIndex(s.index).dayofyear, 366)).to_numpy()
    return float(s.sum() / np.nansum(c) - 1)


def climatologia_por_fase(yr_largo: pd.Series, fase_por_mes: pd.Series, ventana: int = 15) -> dict[str, pd.DataFrame]:
    """Climatologia diaria con solo los días de los meses de cada fase del ONI (El Niño, neutro, La Niña)."""
    s = yr_largo.dropna()
    fase = pd.Series(pd.DatetimeIndex(s.index).to_period("M"), index=s.index).map(fase_por_mes)
    return {f: climatologia_diaria(s[(fase == f).to_numpy()], ventana) for f in fase.dropna().unique() if (fase == f).sum() >= 365}


def escenarios_analogos(origen: pd.Timestamp, yr_largo: pd.Series, pr_30: float, horizonte: int = HORIZONTE) -> pd.DataFrame:
    """Trayectorias posibles de D+1 a D+horizonte: la secuencia de Yr de las mismas fechas en cada año de la serie
    larga, por el PR reciente. Muestran la variabilidad de un día a otro que el valor central promedia; solo
    sirven para visualizar, no entran en las métricas."""
    origen = pd.Timestamp(origen).normalize()
    s = yr_largo.dropna()
    s.index = pd.DatetimeIndex(s.index).normalize()
    dias = [origen + pd.Timedelta(days=h) for h in range(1, horizonte + 1)]
    salida = {}
    for anio in sorted(set(s.index.year)):
        fechas = []
        for d in dias:
            try:
                fechas.append(d.replace(year=anio))
            except ValueError:                                     # 29 de febrero en un año no bisiesto
                fechas.append(d.replace(year=anio, day=28))
        v = s.reindex(pd.DatetimeIndex(fechas)).to_numpy()
        if np.isfinite(v).mean() >= 0.9:
            salida[str(anio)] = pr_30 * v
    return pd.DataFrame(salida, index=pd.DatetimeIndex(dias))


def pronosticar(origen: pd.Timestamp, clim: pd.DataFrame, pr_30: float, kwp: float,
                h3: pd.DataFrame | None = None, yr_pronostico: dict[int, float] | None = None,
                errores_h3: pd.DataFrame | None = None, horizonte: int = HORIZONTE, centro: str = "media",
                anomalia: float = 0.0, beta: pd.Series | None = None, clim_fase: pd.DataFrame | None = None,
                bandas: tuple[float, float] | None = None, errores_pr_h3: pd.DataFrame | None = None,
                bandas_pr: tuple[float, float] | None = None, pr_clim: pd.Series | None = None) -> pd.DataFrame:
    """Pronostico de D+1..D+horizonte desde el origen D (fecha local).

    h3: pronóstico del modelo de 1 a 7 días para este origen (columnas h, yhat); yr_pronostico: {h: Yr
    diario pronosticado}, para el PR esperado; errores_h3: salida de cuantiles_error().
    Desde el día 8, el valor central es `centro` (CENTROS): la media climatológica, la mediana, la media ajustada por
    la anomalía reciente amortiguada (anomalía x beta_h) o la media de los años con la misma fase del ONI (clim_fase).
    Se guardan los cuatro para poder compararlos. bandas: cocientes (observado / pronóstico) de los percentiles 10 y
    90 del error empírico de 8 a 30 días; sin ellas, la banda es la variabilidad climatológica del Yr.
    El PR, el segundo objetivo, lleva lo mismo que Yf: su valor central (Yf del modelo / Yr pronosticado de 1 a 7
    días; el PR reciente de 8 a 30), su intervalo del 80 % con el error de validación (errores_pr_h3 de 1 a 7 días,
    bandas_pr de 8 a 30) y su climatología (pr_clim: PR medio histórico por día del año)."""
    origen = pd.Timestamp(origen).normalize()
    filas = []
    modelo = {} if h3 is None else dict(zip(h3["h"].astype(int), h3["yhat"]))
    con_bandas = bandas is not None and all(b is not None and np.isfinite(float(b)) for b in bandas)
    con_bandas_pr = bandas_pr is not None and all(b is not None and np.isfinite(float(b)) for b in bandas_pr)
    for h in range(1, horizonte + 1):
        dia = origen + pd.Timedelta(days=h)
        doy = min(dia.dayofyear, 366)
        c = clim.loc[doy]
        yf_clim = pr_30 * c["media"]
        cand = {"media": yf_clim, "mediana": pr_30 * c["p50"],
                "anomalia": yf_clim * (1 + (float(beta.get(h, 0.0)) if beta is not None else 0.0) * anomalia),
                "fase_oni": pr_30 * clim_fase.loc[doy, "media"] if clim_fase is not None and doy in clim_fase.index else yf_clim}
        extra = {f"yf_{k}": v for k, v in cand.items()}
        if h <= LIMITE_MODELO and h in modelo and np.isfinite(modelo[h]):
            yf = max(0.0, float(modelo[h]))
            inf, sup = (errores_h3.loc[h, ["q_inf", "q_sup"]] if errores_h3 is not None and h in errores_h3.index else (np.nan, np.nan))
            yr_h = (yr_pronostico or {}).get(h, np.nan)
            pr = yf / yr_h if yr_h and np.isfinite(yr_h) and yr_h > 0 else pr_30
            pinf, psup = (errores_pr_h3.loc[h, ["q_inf", "q_sup"]] if errores_pr_h3 is not None and h in errores_pr_h3.index
                          else (np.nan, np.nan))
            filas.append({"dia": dia, "h": h, "metodo": "modelo", "yf": yf, "yf_p10": max(0.0, yf + inf), "yf_p90": yf + sup,
                          "pr": pr, "pr_p10": max(0.0, pr + pinf), "pr_p90": pr + psup, "yf_climatologia": yf_clim, **extra})
        else:
            yf = cand.get(centro, yf_clim)
            p10, p90 = (yf * bandas[0], yf * bandas[1]) if con_bandas else (pr_30 * c["p10"], pr_30 * c["p90"])
            pr10, pr90 = (pr_30 * bandas_pr[0], pr_30 * bandas_pr[1]) if con_bandas_pr else (np.nan, np.nan)
            filas.append({"dia": dia, "h": h, "metodo": "climatologia", "yf": yf, "yf_p10": p10, "yf_p90": p90, "pr": pr_30,
                          "pr_p10": pr10, "pr_p90": pr90, "yf_climatologia": yf_clim, **extra})
    t = pd.DataFrame(filas).set_index("dia")
    t["pr_climatologia"] = ([float(pr_clim.get(min(d.dayofyear, 366), np.nan)) for d in t.index] if pr_clim is not None
                            else np.nan)
    for k in ("yf", "yf_p10", "yf_p90", "yf_climatologia"):
        t[k.replace("yf", "energia_kwh")] = t[k] * kwp
    t["origen"] = origen
    return t


def pr_climatologico(pr_diario: pd.Series, hasta, ventana: int = 31) -> pd.Series:
    """PR medio histórico por día del año (1..366) con los días anteriores a `hasta`: media circular de `ventana`
    días. Con menos de dos años de datos muchos días quedan sin valor, y la línea solo se dibuja donde lo hay."""
    s = pr_diario[(pr_diario.index < pd.Timestamp(hasta)) & pr_diario.notna()]
    por_dia = s.groupby(s.index.dayofyear).mean().reindex(range(1, 367))
    ext = pd.concat([por_dia.iloc[-ventana:], por_dia, por_dia.iloc[:ventana]])
    suave = ext.rolling(ventana, center=True, min_periods=max(5, ventana // 4)).mean().iloc[ventana:-ventana]
    suave.index = range(1, 367)
    return suave


def bandas_empiricas(pronosticos: pd.DataFrame, y: pd.Series, columna: str = "yf", q=(0.1, 0.9)) -> tuple[float, float]:
    """Percentiles del cociente observado / pronóstico de 8 a 30 días (todas las antelaciones juntas), para una banda
    calibrada con el error real del valor central y no con la variabilidad climatológica."""
    p = pronosticos[pronosticos["h"] > LIMITE_MODELO]
    r = y.reindex(p.index).to_numpy() / p[columna].to_numpy()
    r = r[np.isfinite(r)]
    return (float(np.quantile(r, q[0])), float(np.quantile(r, q[1]))) if len(r) >= 50 else (np.nan, np.nan)


def perdidas_por_origen(pronosticos: pd.DataFrame, y: pd.Series, columnas) -> pd.DataFrame:
    """Error cuadrático medio de 8 a 30 días de cada pronóstico, un valor por origen (para comparar los valores
    centrales con el conjunto de modelos equivalentes al mejor)."""
    p = pronosticos[pronosticos["h"] > LIMITE_MODELO].assign(obs=lambda d: y.reindex(d.index).to_numpy()).dropna(subset=["obs"])
    return pd.DataFrame({c: ((p[c] - p["obs"]) ** 2).groupby(p["origen"]).mean() for c in columnas})


def evaluar(pronosticos: pd.DataFrame, y: pd.Series, columna: str = "yf") -> pd.DataFrame:
    """MAE, RMSE, sesgo y nRMSE por antelación de pronósticos apilados (columnas h y `columna`, índice =
    día objetivo), solo en días observados."""
    from .. import metricas
    p = pronosticos.assign(obs=y.reindex(pronosticos.index).to_numpy()).dropna(subset=["obs", columna])
    filas = [{"h": h, **metricas.evaluar(g["obs"], g[columna])} for h, g in p.groupby("h")]
    return pd.DataFrame(filas).set_index("h")


def total_mensual(pronosticos: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Error del total del mes: para cada origen que es el último día de un mes, suma del pronóstico y de lo
    observado en los días del mes siguiente con dato observado."""
    filas = []
    for o, g in pronosticos.groupby("origen"):
        if (o + pd.Timedelta(days=1)).day != 1:
            continue
        mes = (o + pd.Timedelta(days=1)).to_period("M")
        g = g[g.index.to_period("M") == mes].assign(obs=lambda d: y.reindex(d.index).to_numpy())
        ok = g["obs"].notna()
        if ok.sum() < 10:
            continue
        filas.append({"mes": str(mes), "dias_observados": int(ok.sum()), "observado": g.loc[ok, "obs"].sum(),
                      "pronostico": g.loc[ok, "yf"].sum(), "climatologia": g.loc[ok, "yf_climatologia"].sum()})
    t = pd.DataFrame(filas)
    if not t.empty:
        t["error_pct"] = 100 * (t["pronostico"] - t["observado"]) / t["observado"]
        t["error_climatologia_pct"] = 100 * (t["climatologia"] - t["observado"]) / t["observado"]
    return t
