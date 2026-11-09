"""
referencias.py
==============
Modelos de referencia contra los que se mide todo modelo (Yang et al., 2020; Yang, 2019). Un modelo
solo aporta si supera a estas referencias, que no aprenden nada complejo:

    persistencia simple        y^(t) = y(t - h)
    persistencia inteligente   y^(t) = k*(t - h) x Yr_cs(t)   (persiste la atenuación, no el valor)
    climatología               y^(t) = media del entrenamiento a esa hora del día (o media diaria)
    CLIPER                     k*^(t) = rho_h k*(t - h) + (1 - rho_h) media(k*), con rho_h la
                               autocorrelación de k* al plazo h en el entrenamiento; y^ = k*^ x Yr_cs

Para H2 (día siguiente horario) h = 24 h; para H3 a h días, la serie diaria con rezago h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def _hora_local(indice: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((indice - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).hour)


def horarias_h2(horario: pd.DataFrame, entrenamiento: np.ndarray, col_y: str = "yf") -> pd.DataFrame:
    """Pronosticos de referencia del día siguiente horario (rezago 24 h) para toda la serie."""
    y, k, cs = horario[col_y], horario["kstar"], horario["yr_cs"]
    hora = _hora_local(horario.index)
    clim_h = y[entrenamiento].groupby(hora[entrenamiento]).mean()
    k_tr = k[entrenamiento]
    rho = k_tr.autocorr(24) if k_tr.notna().sum() > 48 else 0.0
    rho = float(np.clip(rho, 0, 1)) if pd.notna(rho) else 0.0
    k_media = float(k_tr.mean())
    k24 = k.shift(24)
    salida = pd.DataFrame({
        "persistencia": y.shift(24),
        "persistencia_inteligente": k24 * cs,
        "climatologia": pd.Series(clim_h.reindex(hora).to_numpy(), index=y.index),
        "cliper": (rho * k24.fillna(k_media) + (1 - rho) * k_media) * cs,
    }, index=y.index)
    salida.attrs["rho_24"] = rho
    return salida.where(cs > 0, 0.0)            # de noche, 0 (físico)


def diarias_h3(diario: pd.DataFrame, entrenamiento: np.ndarray, h: int, col_y: str = "yf") -> pd.DataFrame:
    """Referencias para el valor diario a h días. Índice diario de fechas locales (con huecos)."""
    d = diario.asfreq("D")
    tr = pd.Series(entrenamiento, index=diario.index).reindex(d.index).fillna(False).to_numpy()
    y = d[col_y]
    salida = pd.DataFrame({"persistencia": y.shift(h)}, index=d.index)
    if col_y == "yf":
        k = d["kstar"]
        rho = k[tr].autocorr(h)
        rho = float(np.clip(rho, 0, 1)) if pd.notna(rho) else 0.0
        k_media = float(k[tr].mean())
        salida["persistencia_inteligente"] = k.shift(h) * d["yr_cs"]
        salida["climatologia"] = k_media * d["yr_cs"]
        salida["cliper"] = (rho * k.shift(h).fillna(k_media) + (1 - rho) * k_media) * d["yr_cs"]
    else:
        media = float(y[tr].mean())
        rho = y[tr].autocorr(h)
        rho = float(np.clip(rho, 0, 1)) if pd.notna(rho) else 0.0
        salida["climatologia"] = media
        salida["cliper"] = rho * y.shift(h).fillna(media) + (1 - rho) * media
    salida.attrs["rho"] = rho
    return salida.reindex(diario.index)


