"""
objetivos.py
============
Variables objetivo del TFM (IEC 61724-1) a partir de la telemetría ya imputada y agregada:

    E_AC   energía AC del periodo (kWh)                      = media(P_AC) x duración / 1000
    Yf     rendimiento específico o productividad final        = E_AC / P0                  (kWh/kWp)
    Yr     productividad de referencia                         = H_POA / G_ref              (h = kWh/m2 / (1 kW/m2))
    PR     índice de rendimiento (Performance Ratio)           = Yf / Yr = sum(E_AC) / (P0 * sum(H_POA) / G_ref)
    Yr_cs  productividad de referencia de cielo despejado      = H_POA_cs / G_ref
    k*     índice de cielo despejado del sistema               = Yf / Yr_cs

El PR horario solo se calcula con G_POA >= 200 W/m2 (con poca luz el cociente es inestable); el PR
diario y mensual se calculan como cociente de SUMAS, nunca como promedio de cocientes.
"""

from __future__ import annotations

import pandas as pd

from . import config, solar

COLUMNAS_POTENCIA_AC = ("acpower1", "acpower2", "acpower3", "totalactivepower")


def potencia_ac(df: pd.DataFrame) -> pd.Series:
    """P_AC (W) = acpower1 + acpower2 + acpower3 (totalactivepower viene vacío en estos inversores)."""
    fases = [c for c in ("acpower1", "acpower2", "acpower3") if c in df.columns]
    if fases:
        return df[fases].sum(axis=1, min_count=1).rename("p_ac")
    return df["totalactivepower"].rename("p_ac")


def energia_por_contador(totalyield_ultimo: pd.Series, potencia_nominal_kw: float, horas: float = 1.0) -> pd.Series:
    """Energia del periodo como diferencia del contador acumulado (kWh). Se descartan reinicios
    (diferencia negativa) y saltos imposibles (más que la potencia nominal durante todo el periodo)."""
    d = totalyield_ultimo.diff()
    imposible = (d < 0) | (d > 1.2 * potencia_nominal_kw * horas)
    return d.where(~imposible).rename("e_contador")


def objetivos_horarios(e_kwh: pd.Series, kwp: float, poa: pd.Series, poa_cs: pd.Series) -> pd.DataFrame:
    """Yf, Yr, PR, Yr_cs y k* horarios. e_kwh: energía AC de la hora; poa y poa_cs: medias horarias (W/m2)."""
    yf = e_kwh / kwp
    yr = poa / 1000.0 / config.G_REF               # W/m2 medio durante 1 h -> kWh/m2 -> h
    yr_cs = poa_cs / 1000.0 / config.G_REF
    pr = (yf / yr).where(poa >= config.UMBRAL_G_POA_PR)
    kstar = (yf / yr_cs).where(yr_cs > 0.05)
    return pd.DataFrame({"e_kwh": e_kwh, "yf": yf, "yr": yr, "pr": pr, "yr_cs": yr_cs, "kstar": kstar,
                         "g_poa": poa, "g_poa_cs": poa_cs})


def objetivos_diarios(horario: pd.DataFrame, umbral: float = config.UMBRAL_DIA_VALIDO) -> pd.DataFrame:
    """Sumas diarias (día local). PR diario = sum(Yf) / sum(Yr) en las horas con ambos valores.

    Un día es válido si al menos `umbral` (90 %) de sus horas de sol tienen Yf e Yr. Las horas de sol
    se definen con el cielo despejado (Yr_cs > 0,01), que nunca falta; las horas nocturnas no cuentan,
    porque valen 0 y harian pasar por completo un día al que le falta toda la tarde."""
    loc = horario.copy()
    loc.index = (loc.index - pd.Timedelta(seconds=1)).tz_convert(config.ZONA)
    dia = loc.index.date
    sol = loc["yr_cs"] > 0.01
    ambos = loc["yf"].notna() & loc["yr"].notna()
    horas_sol = sol.groupby(dia).sum()
    horas_sol_ok = (sol & ambos).groupby(dia).sum()
    g = loc[ambos].groupby(loc[ambos].index.date)
    d = pd.DataFrame({
        "e_kwh": g["e_kwh"].sum(), "yf": g["yf"].sum(), "yr": g["yr"].sum(), "yr_cs": g["yr_cs"].sum(),
        "horas": g.size(),
    })
    d["horas_sol"] = horas_sol.reindex(d.index)
    d["fraccion_sol"] = (horas_sol_ok / horas_sol).reindex(d.index)
    d["pr"] = (d["yf"] / d["yr"]).where(d["yr"] > 0)
    d["kstar"] = (d["yf"] / d["yr_cs"]).where(d["yr_cs"] > 0)
    d = d[d["fraccion_sol"] >= umbral]
    d.index = pd.to_datetime(d.index)
    d.index.name = "dia"
    return d


def objetivos_planta(horarios: dict[str, pd.DataFrame], kwp: dict[str, float]) -> pd.DataFrame:
    """Planta = suma de inversores. Solo horas en que TODOS los inversores tienen energía (si falta uno,
    la suma subestimaría la planta). Yf_planta = sum(E) / sum(P0); Yr es la misma POA promedio ponderada."""
    e = pd.concat({k: h["e_kwh"] for k, h in horarios.items()}, axis=1)
    completo = e.notna().all(axis=1)
    p0 = sum(kwp[k] for k in horarios)
    yr = sum(horarios[k]["yr"] * kwp[k] for k in horarios) / p0
    yr_cs = sum(horarios[k]["yr_cs"] * kwp[k] for k in horarios) / p0
    poa = yr * 1000.0 * config.G_REF
    poa_cs = yr_cs * 1000.0 * config.G_REF
    return objetivos_horarios(e.sum(axis=1).where(completo), p0, poa, poa_cs)


def caras_de(inv) -> list[tuple[float, float, float]]:
    """Caras del generador de un inversor [(inclinación, azimut, fracción de paneles)]: las de
    telemetry.plantas_fv_caras si existen (columna `caras` de datos.constantes); si no, la orientación de
    telemetry.devices_fv."""
    caras = inv.get("caras") if hasattr(inv, "get") else None
    if isinstance(caras, (list, tuple)) and len(caras):
        return [tuple(map(float, c)) for c in caras]
    return [(float(inv["inclinacion"]), float(inv["azimut"]), 1.0)]


def poa_inversor(ghi: pd.Series, dni: pd.Series, dhi: pd.Series, lugar, inclinacion: float, azimut: float,
                 paso: str = "1h", caras: list[tuple[float, float, float]] | None = None) -> tuple[pd.Series, pd.Series]:
    """POA medida (de la fuente de irradiancia) y POA de cielo despejado para un inversor (medias horarias).
    Con `caras`, media ponderada por paneles de las caras del generador."""
    medio = pd.Timedelta(paso) / 2
    poa = solar.irradiancia_plano(ghi, dni, dhi, lugar, inclinacion, azimut, desplazamiento=medio, caras=caras)["poa_global"]
    cs = solar.cielo_despejado(ghi.index, lugar, desplazamiento=medio)
    poa_cs = solar.irradiancia_plano(cs["ghi_cs"], cs["dni_cs"], cs["dhi_cs"], lugar, inclinacion, azimut,
                                     desplazamiento=medio, caras=caras)["poa_global"]
    return poa.rename("g_poa"), poa_cs.rename("g_poa_cs")
