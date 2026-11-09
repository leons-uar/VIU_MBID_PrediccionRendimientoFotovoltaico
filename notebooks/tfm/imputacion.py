"""
imputacion.py
=============
Regularización e imputación de series crudas EN SU MUESTREO NATIVO (antes de agregar). Nada de esto
se escribe en la BD: es parte del análisis reproducible.

1. regularizar(): lleva la serie a una rejilla regular (p. ej. 5 min). SolaX registra con segundos
   variables (10:04:07, 10:09:06...), así que cada instante se ajusta al múltiplo más cercano del paso;
   los instantes que faltan en la rejilla quedan como NaN (así los huecos se ven y se cuentan).
2. clasificar_huecos(): cada hueco (bloque de NaN consecutivos) se mide y se clasifica: corto (<= 1 h),
   medio (1 a 6 h) o largo (> 6 h); diurno si contiene algún instante con el sol sobre el horizonte.
3. imputar(): según la naturaleza de la variable:
   - "potencia" e "irradiancia": de noche valen 0 (dato físico, no estadístico); de día, solo los
     huecos completos de hasta 1 h se interpolan en el tiempo; los más largos quedan NaN.
   - "contador" (energía acumulada): interpolación lineal de huecos cortos, sin cruzar reinicios.
   - "continua" (temperatura, tensión...): interpolación temporal de huecos cortos, de día o de noche.
   - "estado" (códigos): se arrastra el último valor solo en huecos cortos.
   Devuelve la serie imputada, una máscara con el motivo de cada relleno y un reporte.

Por que interpolación para huecos cortos (y no media, mediana, arrastre o spline): se decide con
experimento_imputacion(), que borra tramos diurnos OBSERVADOS de 15 min a 3 h y los reconstruye con cada
método. En la potencia de los inversores de G01 la interpolación lineal y la del índice de cielo
despejado empatan hasta 1 h (error medio absoluto de 11 a 14 % de la media); el arrastre (17 a 24 %), el
spline cúbico (17 a 31 %, oscila en los bordes) y la media horaria (26 a 28 %, rompe el ciclo diario y
la autocorrelación) son peores. Por encima de 1 h el error crece y la interpolación lineal se sesga hacia
abajo; el índice de cielo despejado se degrada menos, porque solo interpola la atenuación por nubes y
toma la forma de campana del modelo de cielo despejado.
- Los huecos de más de 1 h NO se inventan: el día afectado se excluye del cálculo de objetivos y métricas.
- KNN o una regresión necesitarían covariables del mismo instante y arriesgan fuga de información.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

CORTO, MEDIO = pd.Timedelta("1h"), pd.Timedelta("6h")

# Naturaleza de las columnas de telemetría (SolaX) por expresón regular sobre el nombre.
_TEL = [
    (re.compile(r"^(daily|today|total)(yield|acoutput|acinput|importenergy|exportenergy)(m2)?$"), "contador"),
    (re.compile(r"^(importenergy|exportenergy)$"), "contador"),
    (re.compile(r"^devicestatus$"), "estado"),
    (re.compile(r"^plantlocaltime$"), "texto"),
    # Solo las potencias de GENERACIÓN valen 0 de noche (la reactiva, el factor de potencia o la
    # potencia de red pueden no ser nulos de noche: van como "continua").
    (re.compile(r"^(acpower\d|totalactivepower|mppt\d+power|pv\d+power|mppttotalinputpower)$"), "potencia"),
]


def naturaleza_telemetria(columna: str) -> str:
    for patron, tipo in _TEL:
        if patron.match(columna):
            return tipo
    return "continua"


_UNIDADES_TEL = [
    (re.compile(r"(yield|output|input|energy)(m2)?$"), "kWh"),
    (re.compile(r"power"), "W"),
    (re.compile(r"voltage"), "V"),
    (re.compile(r"current"), "A"),
    (re.compile(r"temperature"), "°C"),
    (re.compile(r"frequency"), "Hz"),
]


def unidad_telemetria(columna: str) -> str:
    """Unidad de una columna de telemetría SolaX según su nombre (la API no la entrega)."""
    for patron, unidad in _UNIDADES_TEL:
        if patron.search(columna):
            return unidad
    return ""


def regularizar(df: pd.DataFrame, paso: str, columnas: list[str] | None = None,
                inicio: pd.Timestamp | None = None, fin: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rejilla regular de `paso` (p. ej. '5min'): ajuste al múltiplo más cercano y NaN donde no hay dato."""
    cols = columnas or [c for c in df.columns if c != "datatime"]
    d = df[["datatime", *cols]].copy()
    d["datatime"] = d["datatime"].dt.round(paso)
    numericas = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]
    otras = [c for c in cols if c not in numericas]
    agg = {**{c: "mean" for c in numericas}, **{c: "last" for c in otras}}
    d = d.groupby("datatime").agg(agg)
    rejilla = pd.date_range(inicio if inicio is not None else d.index.min(),
                            fin if fin is not None else d.index.max(), freq=paso, name="datatime")
    return d.reindex(rejilla)


def bloques_nan(serie: pd.Series) -> pd.DataFrame:
    """Bloques consecutivos de NaN: inicio, fin, n (pasos) y posiciones."""
    falta = serie.isna().to_numpy()
    if not falta.any():
        return pd.DataFrame(columns=["inicio", "fin", "n", "i0", "i1"])
    cambios = np.diff(np.concatenate([[0], falta.astype(int), [0]]))
    i0, i1 = np.where(cambios == 1)[0], np.where(cambios == -1)[0] - 1
    idx = serie.index
    return pd.DataFrame({"inicio": idx[i0], "fin": idx[i1], "n": i1 - i0 + 1, "i0": i0, "i1": i1})


def clasificar_huecos(serie: pd.Series, paso: str, de_dia: pd.Series | None = None) -> pd.DataFrame:
    """Huecos de una serie regular con su duración, clase (corto/medio/largo) y si son diurnos."""
    b = bloques_nan(serie)
    if b.empty:
        return b.assign(duracion=pd.Series(dtype="timedelta64[ns]"), clase=pd.Series(dtype=str), diurno=pd.Series(dtype=bool))
    p = pd.Timedelta(paso)
    b["duracion"] = b["n"] * p
    b["clase"] = np.select([b["duracion"] <= CORTO, b["duracion"] <= MEDIO], ["corto", "medio"], "largo")
    if de_dia is not None:
        dia = de_dia.reindex(serie.index).fillna(False).to_numpy()
        b["diurno"] = [bool(dia[a:z + 1].any()) for a, z in zip(b["i0"], b["i1"])]
    else:
        b["diurno"] = True
    return b


def _interpolar_bloques_cortos(s: pd.Series, max_pasos: int, metodo: str = "time") -> tuple[pd.Series, pd.Series]:
    """Interpola SOLO los bloques de NaN completos de hasta max_pasos (pandas con limit rellenaria a medias)."""
    b = bloques_nan(s)
    if b.empty:
        return s, pd.Series(False, index=s.index)
    interp = s.interpolate(method=metodo, limit_area="inside")
    rellenable = np.zeros(len(s), dtype=bool)
    for a, z, n in zip(b["i0"], b["i1"], b["n"]):
        if n <= max_pasos and a > 0 and z < len(s) - 1:
            rellenable[a:z + 1] = True
    salida = s.copy()
    salida[rellenable] = interp[rellenable]
    return salida, pd.Series(rellenable & salida.notna().to_numpy(), index=s.index)


def _interpolar_indice_cielo(s: pd.Series, referencia: pd.Series, max_pasos: int,
                             minimo: float = 20.0) -> tuple[pd.Series, pd.Series]:
    """Interpola el índice de cielo despejado k = s / referencia y lo devuelve a la escala original.

    La referencia (GHI o POA de cielo despejado, W/m2) aporta la forma de campana del día; lo que se
    interpola es solo la atenuación por nubes. Con el sol muy bajo (referencia <= mínimo) k es inestable
    y se usa la interpolación lineal."""
    lineal, inter = _interpolar_bloques_cortos(s, max_pasos, "time")
    if not inter.any():
        return lineal, inter
    r = referencia.reindex(s.index).to_numpy(dtype=float)
    valido = r > minimo
    k = pd.Series(np.where(valido, s.to_numpy(dtype=float) / np.where(valido, r, 1.0), np.nan), index=s.index)
    k_int = k.interpolate(method="time", limit_area="inside").to_numpy()
    salida = lineal.copy()
    usar = inter.to_numpy() & valido & np.isfinite(k_int)
    salida[usar] = k_int[usar] * r[usar]
    return salida, inter


METODOS_EXPERIMENTO = ("lineal", "cielo_despejado", "spline_cubico", "arrastre", "media_horaria")


def experimento_imputacion(serie: pd.Series, de_dia: pd.Series, paso: str, referencia: pd.Series | None = None,
                           duraciones=("15min", "30min", "1h", "2h", "3h"), n: int = 300, contexto: str = "6h",
                           semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Huecos artificiales en tramos diurnos OBSERVADOS: se borran, se reconstruyen con cada método y se
    mide el error contra el valor real. Es la evidencia para elegir la técnica y el largo máximo.

    Para cada duración se eligen al azar (semilla 37) hasta n posiciones donde el hueco y sus dos anclas
    (la muestra anterior y la posterior) están observados y son diurnos. Cada método trabaja sobre un
    tramo de +-`contexto` alrededor del hueco. Métodos:
      lineal           interpolación temporal lineal entre las anclas
      cielo_despejado  interpolación del índice de cielo despejado (requiere `referencia`)
      spline_cubico    spline cúbico con los puntos observados del tramo
      arrastre         último valor observado (ffill)
      media_horaria    media de la serie a esa hora del día (climatología)
    """
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(semilla)
    p = pd.Timedelta(paso)
    s = serie.to_numpy(dtype=float)
    obs_dia = ~np.isnan(s) & de_dia.reindex(serie.index).fillna(False).to_numpy()
    ctx = int(pd.Timedelta(contexto) / p)
    local = serie.index.tz_convert(config.ZONA)
    minuto = np.asarray(local.hour * 60 + local.minute)
    climatologia = pd.Series(s).groupby(minuto).mean()
    ref = referencia.reindex(serie.index) if referencia is not None else None
    metodos = [m for m in METODOS_EXPERIMENTO if m != "cielo_despejado" or ref is not None]
    filas = []
    for dur in duraciones:
        m = int(pd.Timedelta(dur) / p)
        # j es el ancla anterior: obs_dia[j : j + m + 2] completo (ancla, hueco y ancla)
        conv = np.convolve(obs_dia.astype(int), np.ones(m + 2, dtype=int), mode="valid")
        inicios = np.where(conv == m + 2)[0] + 1
        inicios = inicios[(inicios - ctx >= 0) & (inicios + m + ctx <= len(s))]
        if len(inicios) == 0:
            continue
        elegidos = rng.choice(inicios, size=min(n, len(inicios)), replace=False)
        err = {k: [] for k in metodos}
        verdad_total = []
        for i in elegidos:
            a, z = i - ctx, i + m + ctx
            seg = pd.Series(s[a:z].copy(), index=serie.index[a:z])
            verdad = seg.iloc[ctx:ctx + m].to_numpy().copy()
            seg.iloc[ctx:ctx + m] = np.nan
            hueco = slice(ctx, ctx + m)
            est = {"lineal": seg.interpolate(method="time", limit_area="inside").to_numpy()[hueco],
                   "arrastre": seg.ffill().to_numpy()[hueco],
                   "media_horaria": climatologia.reindex(minuto[a:z][hueco]).to_numpy()}
            x = np.flatnonzero(seg.notna().to_numpy())
            est["spline_cubico"] = CubicSpline(x, seg.to_numpy()[x])(np.arange(ctx, ctx + m))
            if ref is not None:
                est["cielo_despejado"] = _interpolar_indice_cielo(seg, ref.iloc[a:z], m)[0].to_numpy()[hueco]
            for k in metodos:
                err[k].append(est[k] - verdad)
            verdad_total.append(verdad)
        media = float(np.mean(np.concatenate(verdad_total)))
        for k in metodos:
            e = np.concatenate(err[k])
            filas.append({"duracion": dur, "metodo": k, "huecos": len(elegidos), "muestras": int(e.size),
                          "mae": float(np.nanmean(np.abs(e))), "rmse": float(np.sqrt(np.nanmean(e ** 2))),
                          "mbe": float(np.nanmean(e)), "rmae_pct": 100 * float(np.nanmean(np.abs(e))) / media})
    return pd.DataFrame(filas)


def imputar(reg: pd.DataFrame, naturalezas: dict[str, str], paso: str, de_dia: pd.Series,
            maximo: pd.Timedelta = config.IMPUTAR_HUECO_MAXIMO, referencia: pd.Series | dict | None = None,
            metodo_diurno: str = "lineal") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Imputa una serie regularizada. Devuelve (imputada, motivo, reporte).

    motivo: '' observado, 'noche' (0 físico), 'interpolado' (hueco corto), 'faltante' (sin imputar).
    metodo_diurno: "lineal" o "cielo_despejado" para potencia e irradiancia (referencia: una serie de cielo
    despejado común o un dict {columna: serie}); se elige con experimento_imputacion().
    """
    max_pasos = int(maximo / pd.Timedelta(paso))
    dia = de_dia.reindex(reg.index).fillna(False)
    salida, motivo, filas = reg.copy(), pd.DataFrame("", index=reg.index, columns=reg.columns), []
    for col in reg.columns:
        tipo = naturalezas.get(col, "continua")
        s = reg[col]
        faltante0 = s.isna()
        if tipo == "texto" or not pd.api.types.is_numeric_dtype(s):
            motivo.loc[faltante0, col] = "faltante"
            filas.append({"variable": col, "naturaleza": tipo, "total": len(s), "faltantes": int(faltante0.sum()),
                          "noche": 0, "interpolado": 0, "sin_imputar": int(faltante0.sum())})
            continue
        n_noche = 0
        if tipo in ("potencia", "irradiancia"):
            noche = faltante0 & ~dia
            s = s.where(~noche, 0.0)
            motivo.loc[noche, col] = "noche"
            n_noche = int(noche.sum())
        if tipo == "estado":
            s2 = s.ffill(limit=max_pasos)
            inter = s2.notna() & s.isna()
            b = bloques_nan(s)
            for a, z, n in zip(b["i0"], b["i1"], b["n"]):
                if n > max_pasos:
                    s2.iloc[a:z + 1] = np.nan
                    inter.iloc[a:z + 1] = False
            s = s2
        elif tipo == "contador":
            s, inter = _interpolar_bloques_cortos(s, max_pasos, "time")
            dif = s.diff()
            reinicio = (dif < 0) & inter                 # no interpolar a través de un reinicio
            s[reinicio] = np.nan
            inter = inter & ~reinicio
        elif tipo == "acumulado":
            inter = pd.Series(False, index=s.index)      # acumulados (lluvia): no se inventan
        elif tipo in ("potencia", "irradiancia") and metodo_diurno == "cielo_despejado" and \
                (ref_col := referencia.get(col) if isinstance(referencia, dict) else referencia) is not None:
            s, inter = _interpolar_indice_cielo(s, ref_col, max_pasos)
        else:
            s, inter = _interpolar_bloques_cortos(s, max_pasos, "time")
        motivo.loc[inter.to_numpy(), col] = "interpolado"
        sigue = s.isna()
        motivo.loc[sigue, col] = "faltante"
        salida[col] = s
        filas.append({"variable": col, "naturaleza": tipo, "total": len(s), "faltantes": int(faltante0.sum()),
                      "noche": n_noche, "interpolado": int(inter.sum()), "sin_imputar": int(sigue.sum())})
    rep = pd.DataFrame(filas)
    if not rep.empty:
        rep["pct_faltante"] = 100 * rep["faltantes"] / rep["total"]
        rep["pct_interpolado"] = 100 * rep["interpolado"] / rep["total"]
        rep["pct_sin_imputar"] = 100 * rep["sin_imputar"] / rep["total"]
    return salida, motivo, rep


def dias_validos(observado: pd.Series, de_dia: pd.Series, umbral: float = config.UMBRAL_DIA_VALIDO) -> pd.Series:
    """Fraccion de instantes diurnos OBSERVADOS (no imputados) por día local y si supera el umbral."""
    o = observado.reindex(de_dia.index).fillna(False) & de_dia
    local = de_dia.index.tz_convert(config.ZONA).date
    frac = o.groupby(local).sum() / de_dia.groupby(local).sum().replace(0, np.nan)
    return pd.DataFrame({"fraccion_observada": frac, "valido": frac >= umbral})
