"""
seleccion.py
============
Selección de variables con criterios de filtro, incrustados y de importancia (Guyon y Elisseeff, 2003) y
Learning. Todo lo que aprende de los datos se ajusta SOLO con el entrenamiento; la importancia por
permutación se mide en validación.

1. experimento_fuentes_irradiancia()  que fuente de irradiancia explica mejor la producción medida
2. correlaciones()        Pearson (lineal) y Spearman (monótona) de cada candidata con el objetivo
3. correlacion_cruzada()  correlación con rezagos, para ver desfases entre variable y objetivo
4. informacion_mutua()    dependencia no lineal (estimador de k vecinos), semilla 37
5. vif()                  multicolinealidad entre candidatas (VIF > 10: muy redundante)
6. lasso()                LassoCV con validación temporal (TimeSeriesSplit); coeficientes estandarizados
7. permutacion()          importancia por permutación de un bosque aleatorio, medida en validación
8. ranking()              consolida los rangos de los métodos anteriores
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, objetivos, particiones


# --------------------------------------------------------------------------------------------------
# 1. Fuente de irradiancia
# --------------------------------------------------------------------------------------------------
def experimento_fuentes_irradiancia(yf: pd.Series, fuentes: dict[str, pd.DataFrame], lugar, inclinacion: float,
                                    azimut: float, ajuste: np.ndarray, evaluacion: np.ndarray,
                                    caras: list | None = None) -> tuple[pd.DataFrame, dict]:
    """Compara fuentes de irradiancia por su capacidad de explicar el Yf horario medido.

    Para cada fuente se calcula Yr (POA con pvlib / 1 kW/m2) y se ajusta el modelo físico más simple,
    Yf = PR_f x Yr, con un solo parámetro PR_f por mínimos cuadrados en las horas de `ajuste`. Ese
    parámetro absorbe el sesgo de la fuente; lo que queda (nRMSE, R2) mide si la fuente sigue la
    variabilidad real. Se evalúa en las horas de `evaluacion` comunes a todas las fuentes, en escala
    horaria y diaria (días con al menos el 90 % de sus horas de sol en la muestra común).

    Una fuente con menos del 90 % de disponibilidad en las horas de evaluación (p. ej. NASA POWER, que
    publica con meses de retraso) no recorta la muestra común: se evalúa en sus propias horas y se marca
    con muestra = 'propia'.

    yf: Yf horario observado (sin imputar, en régimen estable); fuentes: {nombre: DataFrame ghi, dni, dhi}.
    Devuelve (tabla, {nombre: Yf estimado})."""
    yr = {}
    for nombre, f in fuentes.items():
        poa, _ = objetivos.poa_inversor(f["ghi"], f["dni"], f["dhi"], lugar, inclinacion, azimut, caras=caras)
        yr[nombre] = (poa / 1000.0 / config.G_REF).reindex(yf.index)
    sol = pd.concat(yr, axis=1).max(axis=1) > 0.05
    base = yf.notna() & sol
    disponible = {k: float(r[base & evaluacion].notna().mean()) if (base & evaluacion).any() else 0.0 for k, r in yr.items()}
    incluidas = [k for k, v in disponible.items() if v >= 0.9]
    comun = base & pd.concat({k: yr[k] for k in incluidas}, axis=1).notna().all(axis=1) if incluidas else base
    dias = particiones.dias_locales(yf.index)
    horas_sol_dia = pd.Series(sol.to_numpy(), index=dias).groupby(level=0).sum()
    filas, estimado = [], {}
    for nombre, r in yr.items():
        muestra = comun if nombre in incluidas else base & r.notna()
        m_aj, m_ev = muestra & ajuste, muestra & evaluacion
        if m_aj.sum() < 100 or m_ev.sum() < 24:
            continue
        pr = float((yf[m_aj] * r[m_aj]).sum() / (r[m_aj] ** 2).sum())
        est = pr * r
        estimado[nombre] = est
        e = (est - yf)[m_ev]
        media = yf[m_ev].mean()
        d_obs = pd.Series(yf[m_ev].to_numpy(), index=dias[m_ev.to_numpy()]).groupby(level=0).agg(["sum", "size"])
        d_est = pd.Series(est[m_ev].to_numpy(), index=dias[m_ev.to_numpy()]).groupby(level=0).sum()
        completos = d_obs["size"] >= config.UMBRAL_DIA_VALIDO * horas_sol_dia.reindex(d_obs.index)
        ed = (d_est - d_obs["sum"])[completos]
        filas.append({
            "fuente": nombre, "muestra": "comun" if nombre in incluidas else "propia",
            "disponibilidad_eval": disponible[nombre], "horas_ajuste": int(m_aj.sum()), "horas_evaluacion": int(m_ev.sum()),
            "pr_ajustado": pr, "nrmse_h_pct": 100 * np.sqrt((e ** 2).mean()) / media,
            "nmae_h_pct": 100 * e.abs().mean() / media, "r2_h": 1 - (e ** 2).sum() / ((yf[m_ev] - media) ** 2).sum(),
            "dias_evaluacion": int(completos.sum()),
            "nrmse_d_pct": 100 * np.sqrt((ed ** 2).mean()) / d_obs["sum"][completos].mean() if completos.any() else np.nan,
            "r_d": float(np.corrcoef(d_est[completos], d_obs["sum"][completos])[0, 1]) if completos.sum() > 2 else np.nan,
        })
    if not filas:
        return pd.DataFrame(), estimado
    return pd.DataFrame(filas).sort_values("nrmse_h_pct").reset_index(drop=True), estimado


# --------------------------------------------------------------------------------------------------
# 2-8. Relevancia y redundancia de candidatas
# --------------------------------------------------------------------------------------------------
def correlaciones(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Pearson y Spearman de cada columna con y (pares completos)."""
    filas = []
    for c in X.columns:
        a, b = X[c].align(y, join="inner")
        ok = a.notna() & b.notna()
        if ok.sum() < 30 or a[ok].std() == 0:
            continue
        filas.append({"variable": c, "n": int(ok.sum()), "pearson": a[ok].corr(b[ok]),
                      "spearman": a[ok].corr(b[ok], method="spearman")})
    t = pd.DataFrame(filas)
    return t.assign(abs_spearman=t["spearman"].abs()).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def correlacion_cruzada(y: pd.Series, x: pd.Series, max_rezago: int = 48) -> pd.Series:
    """corr(y_t, x_{t-k}) para k = -max..max (k > 0: x adelanta a y)."""
    return pd.Series({k: y.corr(x.shift(k)) for k in range(-max_rezago, max_rezago + 1)}, name=x.name)


def _completos(X: pd.DataFrame, y: pd.Series, max_filas: int | None = None, semilla: int = config.SEMILLA):
    d = pd.concat([X, y.rename("__y")], axis=1)
    # Una variable sin ningún dato en la muestra (por ejemplo, una fuente que no cubre las pocas semanas de una unidad
    # reciente) dejaría sin filas completas a todas las demás: se deja fuera
    d = d[[c for c in d.columns if c == "__y" or d[c].notna().any()]]
    completas = d.dropna()
    if completas.empty:
        # Si aun así no queda ninguna fila completa, se quitan las variables con menos datos hasta que las haya
        for c in d.drop(columns="__y").notna().mean().sort_values().index:
            d = d.drop(columns=c)
            completas = d.dropna()
            if len(completas):
                break
    d = completas
    if max_filas and len(d) > max_filas:
        d = d.sample(max_filas, random_state=semilla).sort_index()
    return d.drop(columns="__y"), d["__y"]


def informacion_mutua(X: pd.DataFrame, y: pd.Series, max_filas: int = 20000, semilla: int = config.SEMILLA) -> pd.Series:
    Xc, yc = _completos(X, y, max_filas, semilla)
    mi = mutual_info_regression(Xc.to_numpy(), yc.to_numpy(), n_neighbors=3, random_state=semilla)
    return pd.Series(mi, index=Xc.columns, name="informacion_mutua").sort_values(ascending=False)


def vif(X: pd.DataFrame, max_filas: int = 20000, semilla: int = config.SEMILLA) -> pd.Series:
    """Factor de inflación de la varianza de cada candidata respecto a las demás: VIF_j = 1 / (1 - R2_j)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = X.dropna()
    if len(Xc) > max_filas:
        Xc = Xc.sample(max_filas, random_state=semilla)
    Xc = Xc.loc[:, Xc.std() > 0]
    Z = ((Xc - Xc.mean()) / Xc.std()).to_numpy()
    Z = np.column_stack([np.ones(len(Z)), Z])
    return pd.Series({c: variance_inflation_factor(Z, i + 1) for i, c in enumerate(Xc.columns)},
                     name="vif").sort_values(ascending=False)


def lasso(X: pd.DataFrame, y: pd.Series, pliegues: int = 5, semilla: int = config.SEMILLA) -> tuple[pd.Series, float]:
    """LassoCV estandarizado con validación cruzada temporal. Devuelve (coeficientes, alpha elegido)."""
    Xc, yc = _completos(X, y)
    modelo = make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(pliegues), random_state=semilla,
                                                     max_iter=20000, n_jobs=config.N_JOBS))
    modelo.fit(Xc.to_numpy(), yc.to_numpy())
    las = modelo[-1]
    return pd.Series(las.coef_, index=Xc.columns, name="coef_lasso").sort_values(key=np.abs, ascending=False), float(las.alpha_)


def permutacion(X_ent: pd.DataFrame, y_ent: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                repeticiones: int = 10, semilla: int = config.SEMILLA, max_filas_val: int = 5000) -> pd.DataFrame:
    """Importancia por permutación (aumento del MAE en validación al barajar cada variable) de un bosque
    aleatorio ajustado con el entrenamiento.

    El bosque usa config.N_JOBS procesos y la permutación uno solo: anidar n_jobs=-1 en ambos abria un
    proceso por nucleo, cada uno con su copia del bosque y de los datos, y agotaba la memoria."""
    Xe, ye = _completos(X_ent, y_ent)
    Xv, yv = _completos(X_val[Xe.columns], y_val, max_filas_val, semilla)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=semilla, n_jobs=config.N_JOBS)
    rf.fit(Xe.to_numpy(), ye.to_numpy())
    r = permutation_importance(rf, Xv.to_numpy(), yv.to_numpy(), scoring="neg_mean_absolute_error",
                               n_repeats=repeticiones, random_state=semilla, n_jobs=1)
    return pd.DataFrame({"variable": Xe.columns, "importancia": r.importances_mean, "desviacion": r.importances_std}) \
        .sort_values("importancia", ascending=False).reset_index(drop=True)


def ranking(**criterios: pd.Series) -> pd.DataFrame:
    """Rango medio de cada variable en varios criterios (mayor valor = más relevante en todos)."""
    t = pd.concat({k: v for k, v in criterios.items()}, axis=1)
    rangos = t.rank(ascending=False)
    t["rango_medio"] = rangos.mean(axis=1)
    return t.sort_values("rango_medio")


def seleccionar(orden: list[str], X: pd.DataFrame, umbral_corr: float = 0.90, maximo: int = 15,
                obligatorias: tuple[str, ...] = ()) -> list[str]:
    """Recorre las variables de más a menos relevante y agrega cada una solo si su correlación absoluta
    con todas las ya elegidas es menor que `umbral_corr` (evita redundancia, como el VIF). Las
    obligatorias (p. ej. el rezago de 24 h del objetivo) entran primero."""
    elegidas: list[str] = [v for v in obligatorias if v in X.columns]
    c = X.corr().abs()
    for v in orden:
        if len(elegidas) >= maximo:
            break
        if v in elegidas or v not in X.columns:
            continue
        if all(c.loc[v, e] < umbral_corr for e in elegidas):
            elegidas.append(v)
    return elegidas
