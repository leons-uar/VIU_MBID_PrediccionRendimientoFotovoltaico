"""
redes.py
========
Redes neuronales por etapas (Keras 3 con PyTorch en CPU), de lo básico a lo avanzado:

    K1 MLP       perceptrón multicapa sobre las mismas variables tabulares que el aprendizaje automático
    K2 LSTM      red recurrente que pronostica las 24 horas del día siguiente a partir de la secuencia
                 horaria de ese día (pronósticos meteorológicos, geometría solar y rezagos de 24 y 168 h).
                 Es bidireccional: todas las entradas del día objetivo se conocen al emitir el pronóstico,
                 así que leer la secuencia en ambos sentidos no es fuga de información.

Escalado ajustado solo con el entrenamiento. Parada temprana con el último 15 % (en el tiempo) del
entrenamiento, no con la validación, que queda para comparar modelos. Semilla 37.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd

from .. import config, particiones


def configurar() -> None:
    import keras
    keras.utils.set_random_seed(config.SEMILLA)


def mlp(n_entradas: int, capas=(64, 32), abandono: float = 0.2):
    import keras
    from keras import layers
    entrada = keras.Input(shape=(n_entradas,))
    x = entrada
    for u in capas:
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(abandono)(x)
    salida = layers.Dense(1, activation="relu")(x)          # el rendimiento no es negativo
    modelo = keras.Model(entrada, salida, name="K1_MLP")
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return modelo


def lstm_dia(n_entradas: int, unidades: int = 32, bidireccional: bool = True, abandono: float = 0.2):
    import keras
    from keras import layers
    entrada = keras.Input(shape=(24, n_entradas))
    capa = layers.LSTM(unidades, return_sequences=True)
    x = layers.Bidirectional(capa)(entrada) if bidireccional else capa(entrada)
    x = layers.Dropout(abandono)(x)
    x = layers.TimeDistributed(layers.Dense(16, activation="relu"))(x)
    salida = layers.TimeDistributed(layers.Dense(1, activation="relu"))(x)
    modelo = keras.Model(entrada, salida, name="K2_LSTM")
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return modelo


def entrenar(modelo, X: np.ndarray, y: np.ndarray, fraccion_parada: float = 0.15, epocas: int = 300,
             paciencia: int = 25, lote: int = 32):
    """Entrena con parada temprana sobre el último tramo temporal del entrenamiento (sin barajar el corte)."""
    import keras
    corte = int(len(X) * (1 - fraccion_parada))
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)
    return modelo.fit(X[:corte], y[:corte], validation_data=(X[corte:], y[corte:]), epochs=epocas,
                      batch_size=lote, callbacks=[parada], verbose=0, shuffle=True)


def secuencias_diarias(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Convierte series horarias (índice UTC, etiqueta al final) en tensores por día local:
    X -> (días, 24, variables), y -> (días, 24). Solo días con las 24 horas en el índice.
    Devuelve también el índice de días y una matriz (días, 24) con las marcas de tiempo horarias."""
    dias = particiones.dias_locales(X.index)
    hora = np.asarray(((X.index - pd.Timedelta(seconds=1)).tz_convert(config.ZONA)).hour)
    completos = pd.Series(1, index=dias).groupby(level=0).size()
    completos = completos[completos == 24].index
    sel = np.isin(dias, completos)
    Xs, ys, hs, ts = X[sel], y.reindex(X.index)[sel], hora[sel], X.index[sel]
    d = dias[sel]
    orden = np.lexsort((hs, d.asi8))
    n = len(completos)
    Xa = Xs.to_numpy(dtype=float)[orden].reshape(n, 24, X.shape[1])
    ya = ys.to_numpy(dtype=float)[orden].reshape(n, 24)
    ta = np.asarray(ts)[orden].reshape(n, 24)
    return Xa, ya, pd.DatetimeIndex(sorted(completos)), ta
