"""
escalado.py
===========
Escalado estadístico ajustado SOLO con el entrenamiento y aplicado igual a validación, prueba y vivo.

Si la media y la desviación (o el mínimo y el máximo) se calcularan con todos los datos, el modelo
"veria" la escala del futuro: es una fuga de información que mejora artificialmente las métricas. Por
eso Escalador recuerda hasta que día se ajustó y verificar_sin_fuga() falla si ese día alcanza la
validación.

Lo usan los modelos sensibles a la escala (KNN, SVR, MLP y redes); los árboles y SARIMA no lo
necesitan. La normalizacón física (Yf por kWp, k* por cielo despejado) no es un escalado estadístico:
usa constantes y no filtra información.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from . import particiones

_TIPOS = {"estandar": StandardScaler, "minmax": MinMaxScaler, "robusto": RobustScaler}


class FugaDeInformacion(RuntimeError):
    """El escalado se ajustó con datos de validación, prueba o en vivo."""


class Escalador:
    """Envoltorio de un escalador de scikit-learn que conserva índice y columnas y controla la fuga.

    Los NaN se ignoran al ajustar y se conservan al transformar."""

    def __init__(self, tipo: str = "estandar"):
        if tipo not in _TIPOS:
            raise ValueError(f"tipo de escalado desconocido: {tipo} (opciones: {', '.join(_TIPOS)})")
        self.tipo = tipo
        self._sk = _TIPOS[tipo]()
        self.columnas: list[str] = []
        self.ultimo_dia_ajuste: pd.Timestamp | None = None

    def ajustar(self, X: pd.DataFrame) -> "Escalador":
        self.columnas = list(X.columns)
        self.ultimo_dia_ajuste = particiones.dias_locales(X.index).max()
        self._sk.fit(X.to_numpy(dtype=float))
        return self

    def transformar(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._sk.transform(X[self.columnas].to_numpy(dtype=float)), index=X.index,
                            columns=self.columnas)

    def invertir(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._sk.inverse_transform(X[self.columnas].to_numpy(dtype=float)), index=X.index,
                            columns=self.columnas)

    def verificar_sin_fuga(self, particiones_: dict | None = None) -> None:
        """Lanza Fugadeinformacón si el ajuste incluyo algún día desde el inicio de la validación."""
        frontera = particiones.inicio_validacion(particiones_)
        if self.ultimo_dia_ajuste is None:
            raise RuntimeError("el escalador no esta ajustado")
        if self.ultimo_dia_ajuste >= frontera:
            raise FugaDeInformacion(f"escalado ajustado hasta {self.ultimo_dia_ajuste:%Y-%m-%d}, "
                                    f"pero la validacion empieza el {frontera:%Y-%m-%d}")


