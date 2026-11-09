"""
auditoria_fv.py
===============
Sin piranómetro, el nivel absoluto del rendimiento depende de dos datos que no se miden en sitio: el nivel de
la irradiancia de la fuente elegida y la potencia pico P0 de cada inversor. Este módulo los audita con
reglas fijadas antes de ver los resultados.

Nivel de la irradiancia
    Cada fuente observada se compara con NSRDB (satélite GOES, la fuente mejor validada en América y la que
    mejor explicó la producción de 2025) en los días de 2025 con datos completos de ambas:
        f = suma(H_NSRDB) / suma(H_fuente)
    La fuente se lleva al nivel de NSRDB multiplicando GHI, DNI y DHI por f antes de la transposición. Es un
    factor de escala: no cambia la variabilidad de la fuente ni el error relativo de los modelos de Yf, solo
    el nivel de Yr y del PR. Controles: el cociente de cada fuente con el cielo despejado (Ineichen) en el
    10 % de días más despejados según NSRDB, y el factor por temporada (seca, de diciembre a marzo, y resto).

Potencia pico
    1. Prueba física, que no depende de ninguna fuente: el percentil 99 de la potencia DC de 5 min (estado
       102, régimen estable) no puede superar al P0 del generador; por encima de 1,05 x P0 el valor declarado
       es imposible.
    2. Identificación en horas despejadas y sin recorte (de 9 a 15 h, kt >= 0,80 con la fuente calibrada,
       G_POA >= 600 W/m2, potencia AC máxima de la hora por debajo del 95 % de la nominal, sin falla):
           kWp = P_DC / (G_POA / 1000 x (1 + gamma x (T_celda - 25)) x eta_DC)
       con T_celda = T_aire + G_POA / 800 x (NOCT - 20) y eta_DC = 0,97. Gamma y NOCT son los del datasheet del
       módulo de la planta; sin datasheet, -0,0029 /°C (mediana de los módulos con datasheet de la flota) y 45 °C.
       Se informa la mediana de los valores diarios y su intervalo del 95 % por bootstrap de días.
    Regla: se usa el kWp identificado si la prueba física falla (o, si es menor, el percentil 99 de la potencia
    DC, que es una cota inferior del P0) o si r = identificado / declarado > 1,15. Con r < 0,70 se conserva el
    declarado y la unidad se marca como de capacidad efectiva limitada (limitación de inyección, cadenas
    desconectadas o P0 sobreestimado) solo si la prueba física coincide (percentil 99 por debajo de 0,70 x P0);
    si no coincide, la identificación se marca como no concluyente. En los demás casos se conserva el
    declarado. La corrección vive en el análisis: la base de datos guarda el valor declarado.
    Si el kWp viene del instalador (paneles conectados x potencia del módulo), es el que se usa siempre: la
    identificación queda como control de la telemetría (cociente r) y explica las diferencias (limitación, strings
    desconectados o una irradiancia de nivel distinto).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, solar

REFERENCIA_NIVEL = "nsrdb_goes"
OBSERVADAS = ("satelite_msg", "ifs_analisis", "era5", "nasa_power")
PERIODO_NIVEL = ("2025-01-01", "2025-12-31")
MESES_SECOS = (12, 1, 2, 3)

GAMMA = -0.0029                   # sin datasheet: mediana de los módulos de la flota con datasheet
NOCT = 45.0
ETA_DC = 0.97
KT_MINIMO = 0.80
POA_MINIMA = 600.0
RECORTE = 0.95
LIMITE_FISICO = 1.05
R_SUBESTIMADO = 1.15
R_LIMITADO = 0.70
HORAS_MINIMAS, DIAS_MINIMOS = 30, 10


def _dia_local(indice: pd.DatetimeIndex) -> pd.Index:
    """Dia local de una serie horaria etiquetada al final del intervalo."""
    return (indice - pd.Timedelta("30min")).tz_convert(config.ZONA).normalize()


def irradiacion_diaria(ghi: pd.Series, horas_minimas: int = 22) -> pd.Series:
    """Irradiacion diaria (kWh/m2) a partir de medias horarias de GHI (W/m2), solo con días casi completos."""
    s = ghi.astype(float)
    dia = _dia_local(s.index)
    n = s.notna().groupby(dia).sum()
    return (s.groupby(dia).sum(min_count=1) / 1000).where(n >= horas_minimas).dropna()


def factores_nivel(fuentes: dict[str, pd.DataFrame], lugar, periodo: tuple[str, str] = PERIODO_NIVEL) -> pd.DataFrame:
    """Factor de nivel de cada fuente frente a NSRDB y sus controles (una fila por fuente)."""
    if REFERENCIA_NIVEL not in fuentes or fuentes[REFERENCIA_NIVEL] is None:
        return pd.DataFrame()
    ini, fin = (pd.Timestamp(p).tz_localize(config.ZONA) for p in periodo)
    diarios = {k: irradiacion_diaria(v["ghi"]) for k, v in fuentes.items() if v is not None}
    diarios = {k: d[(d.index >= ini) & (d.index <= fin)] for k, d in diarios.items()}
    ref = diarios[REFERENCIA_NIVEL]
    idx = ref.index
    horas = pd.date_range(idx.min().tz_convert("UTC") + pd.Timedelta("1h"), (idx.max() + pd.Timedelta("1D")).tz_convert("UTC"),
                          freq="1h")                                   # medias horarias etiquetadas al final
    cs = solar.cielo_despejado(horas, lugar, pd.Timedelta("30min"))["ghi_cs"]
    cs_d = irradiacion_diaria(cs).reindex(idx)
    claros = (ref / cs_d) >= (ref / cs_d).quantile(0.90)
    filas = {}
    for k, d in diarios.items():
        # Días con irradiación de día (el satélite MSG no tiene datos diurnos al oeste de unos 75,2° O)
        comun = d.index.intersection(idx)[(d.reindex(d.index.intersection(idx)) > 0.5).to_numpy()]
        if len(comun) < 30:
            continue
        a, b = ref.loc[comun], d.loc[comun]
        seca = comun.month.isin(MESES_SECOS)
        cl = comun[claros.reindex(comun).fillna(False).to_numpy()]
        filas[k] = {"factor": float(a.sum() / b.sum()), "dias": int(len(comun)),
                    "factor_seca": float(a[seca].sum() / b[seca].sum()) if seca.sum() >= 20 else np.nan,
                    "factor_resto": float(a[~seca].sum() / b[~seca].sum()) if (~seca).sum() >= 20 else np.nan,
                    "cociente_cielo_despejado": float(b.loc[cl].sum() / cs_d.loc[cl].sum()) if len(cl) else np.nan}
    return pd.DataFrame(filas).T.rename_axis("fuente")


def calibrar(fuente: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Lleva GHI, DNI y DHI al nivel de la referencia (factor de escala)."""
    salida = fuente.copy()
    for c in ("ghi", "dni", "dhi"):
        if c in salida.columns:
            salida[c] = salida[c] * factor
    return salida


def capacidad(p_dc: pd.Series, p_ac_max: pd.Series, falla: pd.Series, g_poa: pd.Series, ghi: pd.Series,
              ghi_cs: pd.Series, t_aire: pd.Series, en_regimen: np.ndarray, pdc_5min: pd.Series | None,
              nominal_kw: float, kwp_declarado: float, semilla: int = config.SEMILLA, gamma_pct: float | None = None,
              noct: float | None = None, origen: str | None = None) -> dict:
    """Prueba física, kWp identificado y decisón sobre el P0 que usa el análisis (series horarias en W y W/m2,
    etiquetadas al final; pdc_5min: potencia DC de 5 min con estado 102 dentro del régimen estable)."""
    salida = {"kwp_declarado": float(kwp_declarado), "nominal_kw": float(nominal_kw)}
    idx = p_dc.index
    p_ac_max, falla, g_poa, ghi, ghi_cs, t_aire = (s.reindex(idx) for s in (p_ac_max, falla, g_poa, ghi, ghi_cs, t_aire))
    p99 =float(np.nanquantile(pdc_5min.dropna(), 0.99) / 1000) if pdc_5min is not None and pdc_5min.notna().sum() > 1000 else np.nan
    salida["p99_dc_kw"] = p99
    salida["p99_dc_sobre_kwp"] = p99 / kwp_declarado if np.isfinite(p99) and kwp_declarado else np.nan
    hora = (p_dc.index - pd.Timedelta("30min")).tz_convert(config.ZONA).hour
    kt = ghi / ghi_cs.where(ghi_cs > 50)
    sel = (pd.Series(en_regimen, index=p_dc.index) & (hora >= 9) & (hora <= 14) & (kt >= KT_MINIMO) & (g_poa >= POA_MINIMA)
           & (p_ac_max < RECORTE * nominal_kw * 1000) & (falla.fillna(1) == 0) & (p_dc > 0) & t_aire.notna())
    gamma = gamma_pct / 100 if gamma_pct is not None and np.isfinite(gamma_pct) else GAMMA
    noct = noct if noct is not None and np.isfinite(noct) else NOCT
    salida.update({"gamma": gamma, "noct": noct})
    t_celda = t_aire + g_poa / 800 * (noct - 20)
    kwp_h = (p_dc / 1000) / (g_poa / 1000 * (1 + gamma * (t_celda - 25)) * ETA_DC)
    kwp_h = kwp_h[sel]
    dias = kwp_h.groupby(_dia_local(kwp_h.index)).median()
    salida.update({"horas_despejadas": int(len(kwp_h)), "dias_despejados": int(len(dias))})
    if len(kwp_h) >= HORAS_MINIMAS and len(dias) >= DIAS_MINIMOS:
        rng = np.random.default_rng(semilla)
        boot = np.median(rng.choice(dias.to_numpy(), size=(1000, len(dias)), replace=True), axis=1)
        salida.update({"kwp_identificado": float(dias.median()), "ic95_inf": float(np.percentile(boot, 2.5)),
                       "ic95_sup": float(np.percentile(boot, 97.5))})
        salida["r"] = salida["kwp_identificado"] / kwp_declarado
    else:
        salida.update({"kwp_identificado": np.nan, "ic95_inf": np.nan, "ic95_sup": np.nan, "r": np.nan})
    imposible = np.isfinite(salida["p99_dc_sobre_kwp"]) and salida["p99_dc_sobre_kwp"] > LIMITE_FISICO
    r = salida["r"]
    if imposible:
        # El identificado no puede quedar por debajo de lo que el generador ya entregó
        kwp = max(salida["kwp_identificado"], p99) if np.isfinite(salida["kwp_identificado"]) else p99
        decision = "identificado: el declarado es fisicamente imposible"
    elif np.isfinite(r) and r > R_SUBESTIMADO:
        kwp, decision = salida["kwp_identificado"], "identificado: el declarado esta subestimado"
    elif np.isfinite(r) and r < R_LIMITADO:
        # Solo se clasifica como limitada si la prueba física coincide: cuando los picos de potencia DC llegan cerca
        # del declarado, una identificación baja indica horas mal clasificadas como despejadas o limitación de
        # inyección en esas horas, no un generador más pequeño
        limitada = np.isfinite(salida["p99_dc_sobre_kwp"]) and salida["p99_dc_sobre_kwp"] < R_LIMITADO
        kwp = kwp_declarado
        decision = "declarado: capacidad efectiva limitada" if limitada else "declarado: identificacion no concluyente"
    elif np.isfinite(r):
        kwp, decision = kwp_declarado, "declarado: coherente con la telemetria"
    else:
        kwp, decision = kwp_declarado, "declarado: sin horas despejadas suficientes"
    if origen == "instalador":
        kwp, decision = kwp_declarado, "instalador; telemetria: " + decision.split(": ", 1)[1]
    salida.update({"kwp_analisis": float(kwp), "decision": decision})
    return salida
