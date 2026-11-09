"""
ml.py
=====
Modelos de aprendizaje automático para el pronóstico del rendimiento, con un proceso estándar de ajuste
(Bergstra y Bengio, 2012):

    1. modelos base con hiperparámetros por defecto
    2. búsqueda aleatoria (RandomizedSearchCV) con validación cruzada temporal (TimeSeriesSplit)
    3. búsqueda en rejilla (GridSearchCV) alrededor del mejor punto de la aleatoria
    4. tabla resumen y comparación en la validación

Modelos: KNN, árbol de decisón (DT), bosque aleatorio (RF), gradient boosting histogramico (HGB), SVR,
LightGBM (Ke et al., 2017) y un ensamble por votación (ENS) de los mejores.

Sin fuga de información: la imputación (mediana) y el escalado (solo KNN y SVR) van DENTRO del
Pipeline, así que en cada pliegue se ajustan con su parte de entrenamiento. La semilla es 37.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .. import config

S = config.SEMILLA


def _pipe(modelo, escalar: bool, imputar: bool = True) -> Pipeline:
    pasos = []
    if imputar:
        pasos.append(("imputar", SimpleImputer(strategy="median")))
    if escalar:
        pasos.append(("escalar", StandardScaler()))
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)


# nombre: (constructor del pipeline, distribuciones de la búsqueda aleatoria)
MODELOS = {
    "KNN": (lambda: _pipe(KNeighborsRegressor(), True),
            {"modelo__n_neighbors": randint(3, 60), "modelo__weights": ["uniform", "distance"], "modelo__p": [1, 2]}),
    "DT": (lambda: _pipe(DecisionTreeRegressor(random_state=S), False),
           {"modelo__max_depth": randint(3, 20), "modelo__min_samples_leaf": randint(1, 60),
            "modelo__min_samples_split": randint(2, 40)}),
    "RF": (lambda: _pipe(RandomForestRegressor(random_state=S, n_jobs=1), False),
           {"modelo__n_estimators": randint(100, 400), "modelo__max_depth": [None, 8, 12, 16, 24],
            "modelo__min_samples_leaf": randint(1, 30), "modelo__max_features": [0.3, 0.5, 0.7, 1.0]}),
    "HGB": (lambda: _pipe(HistGradientBoostingRegressor(random_state=S), False, imputar=False),
            {"modelo__learning_rate": loguniform(0.02, 0.3), "modelo__max_iter": randint(100, 600),
             "modelo__max_leaf_nodes": randint(8, 64), "modelo__min_samples_leaf": randint(5, 80),
             "modelo__l2_regularization": loguniform(1e-4, 1.0)}),
    "SVR": (lambda: _pipe(SVR(kernel="rbf"), True),
            {"modelo__C": loguniform(0.1, 100), "modelo__gamma": loguniform(1e-3, 1.0), "modelo__epsilon": loguniform(1e-3, 0.1)}),
    "LGBM": (lambda: _pipe(LGBMRegressor(random_state=S, verbose=-1, n_jobs=1), False, imputar=False),
             {"modelo__n_estimators": randint(100, 800), "modelo__learning_rate": loguniform(0.01, 0.2),
              "modelo__num_leaves": randint(8, 64), "modelo__min_child_samples": randint(5, 80),
              "modelo__subsample": uniform(0.6, 0.4), "modelo__subsample_freq": [1],
              "modelo__colsample_bytree": uniform(0.5, 0.5), "modelo__reg_lambda": loguniform(1e-3, 10)}),
}


# Hiperparámetros que se refinan en la rejilla (los demás quedan en el mejor valor de la búsqueda
# aleatoria): así la rejilla tiene 9 combinaciones y no miles.
REFINAR = {"KNN": ("modelo__n_neighbors",), "DT": ("modelo__max_depth", "modelo__min_samples_leaf"),
           "RF": ("modelo__n_estimators", "modelo__min_samples_leaf"), "HGB": ("modelo__learning_rate", "modelo__max_leaf_nodes"),
           "SVR": ("modelo__C", "modelo__gamma"), "LGBM": ("modelo__learning_rate", "modelo__num_leaves")}


def _vecindad(valor, distribucion):
    """Tres puntos alrededor del mejor valor de la búsqueda aleatoria, para la rejilla."""
    if isinstance(valor, (bool, str)) or valor is None:
        return [valor]
    if isinstance(distribucion, list):
        return [valor]
    if isinstance(valor, (int, np.integer)):
        paso = max(1, int(round(abs(valor) * 0.25)))
        return sorted({max(1, int(valor) - paso), int(valor), int(valor) + paso})
    return sorted({float(valor) * 0.7, float(valor), float(valor) * 1.3})


def ajustar(nombre: str, X: pd.DataFrame, y: pd.Series, n_iter: int = 15, pliegues: int = 4,
            rejilla: bool = True) -> dict:
    """Busqueda aleatoria y luego en rejilla (MAE, validación cruzada temporal) de un modelo.

    Devuelve {'base', 'aleatoria', 'rejilla', 'mejor', 'mae_cv_*', 'parametros', 'segundos'}."""
    constructor, distribuciones = MODELOS[nombre]
    cv = TimeSeriesSplit(pliegues)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = constructor().fit(X, y)
        aleatoria = RandomizedSearchCV(constructor(), distribuciones, n_iter=n_iter, cv=cv,
                                       scoring="neg_mean_absolute_error", random_state=S, n_jobs=config.N_JOBS, refit=True)
        aleatoria.fit(X, y)
        salida = {"base": base, "aleatoria": aleatoria, "mejor": aleatoria.best_estimator_,
                  "mae_cv_aleatoria": -aleatoria.best_score_, "parametros": aleatoria.best_params_}
        if rejilla:
            malla = {k: (_vecindad(v, distribuciones[k]) if k in REFINAR.get(nombre, ()) else [v])
                     for k, v in aleatoria.best_params_.items()}
            fina = GridSearchCV(constructor(), malla, cv=cv, scoring="neg_mean_absolute_error", n_jobs=config.N_JOBS, refit=True)
            fina.fit(X, y)
            if -fina.best_score_ <= salida["mae_cv_aleatoria"]:
                salida.update({"rejilla": fina, "mejor": fina.best_estimator_, "mae_cv_rejilla": -fina.best_score_,
                               "parametros": fina.best_params_})
            else:
                salida.update({"rejilla": fina, "mae_cv_rejilla": -fina.best_score_})
    salida["segundos"] = time.time() - t0
    return salida


def ensamble(mejores: dict[str, Pipeline]) -> VotingRegressor:
    """Ensamble por votación (promedio) de los modelos ya ajustados que se le pasen (se reentrena)."""
    return VotingRegressor([(k, v) for k, v in mejores.items()])
