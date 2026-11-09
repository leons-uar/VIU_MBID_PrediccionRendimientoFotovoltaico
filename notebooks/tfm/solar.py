"""
solar.py
========
Geometría solar y radiación con pvlib: posición del sol, cielo despejado (Ineichen con turbidez de
Linke climatológica), transposición al plano de los módulos (Perez) y temperatura de módulo.

Convenciones:
- Instantes en UTC. Para valores medios de un periodo etiquetado al FINAL (como la radiación de
  Open-Meteo), la geometría se calcula en el punto medio del periodo (t - paso/2).
- Azimut de pvlib/brujula: 0 = norte, 90 = este, 180 = sur, 270 = oeste (el mismo de
  telemetry.devices_fv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def ubicacion(latitud: float, longitud: float, elevacion: float = 0.0) -> pvlib.location.Location:
    return pvlib.location.Location(latitud, longitud, tz="UTC", altitude=elevacion)


def posicion_solar(indice: pd.DatetimeIndex, lugar: pvlib.location.Location,
                   desplazamiento: pd.Timedelta | None = None) -> pd.DataFrame:
    """Cenit, elevación y azimut solar. desplazamiento: se resta al índice (p. ej. paso/2 para medias)."""
    t = indice - desplazamiento if desplazamiento is not None else indice
    pos = lugar.get_solarposition(t)
    pos.index = indice
    return pos[["apparent_zenith", "zenith", "apparent_elevation", "azimuth"]]


def es_de_dia(indice: pd.DatetimeIndex, lugar: pvlib.location.Location, elevacion_minima: float = 0.0,
              desplazamiento: pd.Timedelta | None = None) -> pd.Series:
    """True cuando el sol esta por encima de elevacion_minima grados."""
    pos = posicion_solar(indice, lugar, desplazamiento)
    return pos["apparent_elevation"] > elevacion_minima


def cielo_despejado(indice: pd.DatetimeIndex, lugar: pvlib.location.Location,
                    desplazamiento: pd.Timedelta | None = None) -> pd.DataFrame:
    """GHI, DNI y DHI de cielo despejado (W/m2), modelo Ineichen-Perez con turbidez de Linke mensual."""
    t = indice - desplazamiento if desplazamiento is not None else indice
    cs = lugar.get_clearsky(t, model="ineichen")
    cs.index = indice
    return cs.rename(columns={"ghi": "ghi_cs", "dni": "dni_cs", "dhi": "dhi_cs"})


def irradiancia_plano(ghi: pd.Series, dni: pd.Series, dhi: pd.Series, lugar: pvlib.location.Location,
                      inclinacion: float, azimut: float, desplazamiento: pd.Timedelta | None = None,
                      albedo: float = 0.2, caras: list[tuple[float, float, float]] | None = None) -> pd.DataFrame:
    """Irradiancia en el plano de los módulos (POA, W/m2) con el modelo de cielo difuso de Perez.

    caras: [(inclinación, azimut, fracción de paneles), ...] para generadores repartidos en varias caras
    (techos a dos aguas o de cadera); la POA es la media ponderada por paneles y se ignoran inclinación y
    azimut. Con una sola cara (o sin caras) es el cálculo de siempre."""
    if caras and len(caras) > 1:
        partes = [f * irradiancia_plano(ghi, dni, dhi, lugar, i, a, desplazamiento, albedo) for i, a, f in caras]
        return sum(partes[1:], partes[0]) / sum(f for _, _, f in caras)
    if caras:
        inclinacion, azimut = caras[0][0], caras[0][1]
    indice = ghi.index
    pos = posicion_solar(indice, lugar, desplazamiento)
    t = indice - desplazamiento if desplazamiento is not None else indice
    dni_extra = pd.Series(pvlib.irradiance.get_extra_radiation(t).to_numpy(), index=indice)
    masa_aire = pd.Series(pvlib.atmosphere.get_relative_airmass(pos["apparent_zenith"].to_numpy()), index=indice)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=inclinacion, surface_azimuth=azimut,
        solar_zenith=pos["apparent_zenith"], solar_azimuth=pos["azimuth"],
        dni=dni.clip(lower=0), ghi=ghi.clip(lower=0), dhi=dhi.clip(lower=0),
        dni_extra=dni_extra, airmass=masa_aire, albedo=albedo, model="perez")
    salida = poa[["poa_global", "poa_direct", "poa_diffuse"]].fillna(0.0).clip(lower=0)
    # De noche (sol bajo el horizonte) la POA es 0 por definición.
    dia = (pos["apparent_elevation"] > 0).to_numpy()
    salida.loc[~dia] = 0.0
    # De día, sin dato de irradiancia no hay POA: queda faltante. Rellenarla con 0 haría Yr = 0 y un
    # PR infinito en los días en que la fuente fallo.
    faltante = (ghi.isna() | dni.isna() | dhi.isna()).to_numpy() & dia
    salida.loc[faltante] = np.nan
    return salida


def temperatura_modulo(poa_global: pd.Series, temp_aire: pd.Series, viento: pd.Series | None = None) -> pd.Series:
    """Temperatura de celda (grados C) con el modelo de Faiman (u0 = 25, u1 = 6.84). viento en m/s."""
    v = viento if viento is not None else pd.Series(1.0, index=poa_global.index)
    return pd.Series(pvlib.temperature.faiman(poa_global, temp_aire, v), index=poa_global.index)


def separar_dni_dhi(ghi: pd.Series, lugar: pvlib.location.Location,
                    desplazamiento: pd.Timedelta | None = None) -> pd.DataFrame:
    """Si una fuente solo trae GHI, estima DNI y DHI con el modelo de Erbs."""
    pos = posicion_solar(ghi.index, lugar, desplazamiento)
    t = ghi.index - desplazamiento if desplazamiento is not None else ghi.index
    r = pvlib.irradiance.erbs(ghi.clip(lower=0), pos["zenith"], t)
    return pd.DataFrame({"dni": np.asarray(r["dni"]), "dhi": np.asarray(r["dhi"])}, index=ghi.index)
