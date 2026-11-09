"""
caracteristicas.py
==================
Ingeniería de características para los modelos, construidas SIN mirar el futuro.

Convención temporal (la misma de todo el TFM): cada valor horario resume el intervalo (t - 1 h, t] y se
etiqueta al final, en UTC. Las fuentes se llevan a esa convención según la naturaleza de cada variable
(columna tipovalor de meteo.variables):

    media        (radiación de Open-Meteo: media de la hora anterior)      se usa tal cual
    acumulado    (precipitación: suma de la hora anterior)                 se usa tal cual
    máximo       (rachas: máximo de la hora anterior)                      se usa tal cual
    instantaneo  (temperatura, nubosidad, humedad, viento: valor en t)     media de los extremos t - 1 h y t

Escenarios de exógenas:
    E1  observada ("perfect prognosis"): el valor real de la hora objetivo (análisis, reanálisis, satélite)
    E2  pronóstico archivado: el valor que pronóstico el modelo numérico con la antelación del horizonte
        (Open-Meteo Previous Runs, previousday = 1 para el día siguiente, = h para h días)

Rezagos del objetivo: solo los disponibles al emitir el pronóstico. Para H2 (día siguiente horario,
emitido al final del día D - 1) el rezago mínimo es 24 h; para H3 a h días, h días.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calidad, config, imputacion, solar

# Rangos físicos plausibles de variables no radiativas (fuera de ellos: faltante)
RANGOS = {
    "temperature_2m": (-5, 50), "dew_point_2m": (-10, 40), "relative_humidity_2m": (0, 100),
    "cloud_cover": (0, 100), "cloud_cover_low": (0, 100), "cloud_cover_mid": (0, 100), "cloud_cover_high": (0, 100),
    "precipitation": (0, 300), "wind_speed_10m": (0, 60), "wind_gusts_10m": (0, 90), "cape": (0, 8000),
    "aerosol_optical_depth": (0, 5), "dust": (0, 5000), "pressure_msl": (900, 1100), "surface_pressure": (800, 1100),
}


def meteo_horaria(crudo: pd.DataFrame, tipos: dict[str, str], lugar, paso: str = "1h", marca: str = "fin",
                  filtro: dict | None = None, imputar: bool = True, metodo_diurno: str = "lineal") -> pd.DataFrame:
    """Serie meteorológica cruda -> medias horarias etiquetadas al final, con control de calidad.

    tipos: {variable: tipovalor}. marca: 'fin' (Open-Meteo), 'inicio' (NASA POWER) o 'instante' (NSRDB).
    filtro: p. ej. {'previousday': 1} o {'modelo': 'era5'} para tablas con varias claves.
    Pasos: filtro de claves, rejilla regular, límites físicos (BSRN para irradiancia, RANGOS para el
    resto), imputación de huecos de hasta 1 h según la naturaleza (con la técnica elegida en el
    experimento de imputación) y agregación a 1 h."""
    df = crudo
    for k, v in (filtro or {}).items():
        df = df[df[k] == v]
    variables = [v for v in tipos if v in df.columns]
    reg = imputacion.regularizar(df, paso, variables)
    centro = {"fin": pd.Timedelta(paso) / 2, "inicio": -pd.Timedelta(paso) / 2, "instante": None}[marca]
    reg = reg.mask(calidad.limites_irradiancia(reg, lugar, centro))
    for v in variables:
        if v in RANGOS:
            lo, hi = RANGOS[v]
            reg[v] = reg[v].clip(lower=lo, upper=hi) if v.startswith("cloud_cover") else reg[v].where(reg[v].between(lo, hi))
    if imputar:
        dia = solar.es_de_dia(reg.index, lugar, desplazamiento=centro)
        nat = {v: ("irradiancia" if calidad._TIPO_IRRADIANCIA.get(v) else
                   "acumulado" if tipos[v] in ("acumulado", "maximo") else "continua") for v in variables}
        ref = None
        if metodo_diurno == "cielo_despejado":
            cs = solar.cielo_despejado(reg.index, lugar, centro)
            ref = {v: cs["dni_cs"] if calidad._TIPO_IRRADIANCIA.get(v) == "dni" else cs["ghi_cs"]
                   for v in variables if nat[v] == "irradiancia"}
        reg, _, _ = imputacion.imputar(reg, nat, paso, dia, referencia=ref, metodo_diurno=metodo_diurno)
    if marca == "inicio":
        reg.index = reg.index + pd.Timedelta(paso)
    if pd.Timedelta(paso) < pd.Timedelta("1h"):
        reglas = {v: {"acumulado": "suma", "maximo": "maximo"}.get(tipos[v], "media") for v in variables}
        reg, _ = agregacion_horaria(reg, reglas, paso)
    else:
        inst = [v for v in variables if tipos[v] == "instantaneo"]
        reg[inst] = (reg[inst] + reg[inst].shift(1)) / 2      # media de los extremos del intervalo
    return reg


def agregacion_horaria(reg: pd.DataFrame, reglas: dict[str, str], paso: str):
    from . import agregacion
    return agregacion.agregar(reg, reglas, "1h", paso)


# Fuentes de irradiancia: (serie en la cache, paso nativo, marca, filtro, (GHI, DNI, DHI))
_OM = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation")
FUENTES_IRRADIANCIA = {
    "satelite_msg": ("om_satellite__eumetsat_lsa_saf_msg", "15min", "fin", None, _OM),
    "ifs_analisis": ("om_archive__ecmwf_ifs", "1h", "fin", None, _OM),
    "era5": ("om_archive__era5", "1h", "fin", None, _OM),
    "nsrdb_goes": ("nsrdb_goes", "10min", "instante", None, ("ghi", "dni", "dhi")),
    "nasa_power": ("nasa_power", "1h", "inicio", None, ("allsky_sfc_sw_dwn", "allsky_sfc_sw_dni", "allsky_sfc_sw_diff")),
    "ifs025_pronostico_d1": ("om_previous_runs__ecmwf_ifs025", "1h", "fin", {"previousday": 1}, _OM),
    "gfs_pronostico_d1": ("om_previous_runs__gfs_seamless", "1h", "fin", {"previousday": 1}, _OM),
}
FUENTES_PRONOSTICO = ("ifs025_pronostico_d1", "gfs_pronostico_d1")
# Nombres completos para figuras y tablas
NOMBRE_FUENTE = {
    "satelite_msg": "Satélite MSG (EUMETSAT)",
    "ifs_analisis": "Análisis IFS (ECMWF)",
    "era5": "Reanálisis ERA5 (ECMWF)",
    "nsrdb_goes": "Satélite GOES (NSRDB)",
    "nasa_power": "NASA POWER",
    "ifs025_pronostico_d1": "Pronóstico IFS 0,25° a un día",
    "gfs_pronostico_d1": "Pronóstico GFS a un día",
}


def irradiancia_fuente(crudos: dict[str, pd.DataFrame], nombre: str, lugar, filtro: dict | None = None,
                       metodo_diurno: str = "lineal") -> pd.DataFrame | None:
    """GHI, DNI y DHI horarios (W/m2, etiqueta al final) de una fuente de FUENTES_IRRADIANCIA. Si la
    fuente no trae DNI y DHI, se separan con el modelo de Erbs. filtro reemplaza al de la tabla (p. ej.
    {'previousday': 3} para el pronóstico a 3 días)."""
    serie, paso, marca, filtro0, (cg, cn, cd) = FUENTES_IRRADIANCIA[nombre]
    if serie not in crudos or cg not in crudos[serie].columns:
        return None
    crudo = crudos[serie]
    cols = [c for c in (cg, cn, cd) if c in crudo.columns]
    h = meteo_horaria(crudo, {c: "media" for c in cols}, lugar, paso, marca, filtro or filtro0, metodo_diurno=metodo_diurno)
    salida = pd.DataFrame({"ghi": h[cg]})
    if cn in h.columns and cd in h.columns:
        salida["dni"], salida["dhi"] = h[cn], h[cd]
    else:
        sep = solar.separar_dni_dhi(salida["ghi"], lugar, pd.Timedelta("30min"))
        salida["dni"], salida["dhi"] = sep["dni"], sep["dhi"]
    return salida


def exogenas_meteo(crudos: dict[str, pd.DataFrame], tipos_catalogo: pd.DataFrame, lugar, escenario: str = "E1",
                   previousday: int = 1) -> pd.DataFrame:
    """Candidatas meteorológicas no radiativas (temperatura, humedad, nubosidad, lluvia, viento...).

    E1: análisis IFS de la hora objetivo y aerosoles CAMS. E2: pronóstico IFS 0,25 archivado con
    `previousday` días de antelación; los aerosoles, sin pronóstico archivado, entran como persistencia
    (valor 24 h antes). tipos_catalogo: meteo.variables (columnas fuente, variable, tipovalor)."""
    radiativas = set(_OM) | {"direct_radiation", "terrestrial_radiation", "global_tilted_irradiance"}

    def _de(serie, fuente, filtro=None, prefijo=""):
        if serie not in crudos:
            return None
        tipos = tipos_catalogo[tipos_catalogo.fuente == fuente].set_index("variable")["tipovalor"].to_dict()
        tipos = {v: t for v, t in tipos.items() if v in crudos[serie].columns and v not in radiativas}
        h = meteo_horaria(crudos[serie], tipos, lugar, "1h", "fin", filtro)
        return h.add_prefix(prefijo)

    if escenario == "E1":
        partes = [_de("om_archive__ecmwf_ifs", "om_archive"), _de("om_air_quality__cams_global", "om_air_quality")]
    else:
        aq = _de("om_air_quality__cams_global", "om_air_quality")
        partes = [_de("om_previous_runs__ecmwf_ifs025", "om_previous_runs", {"previousday": previousday}),
                  aq.shift(24).add_suffix("_rez24") if aq is not None else None]
    return pd.concat([p for p in partes if p is not None], axis=1)


def ciclicas(indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Hora del día y día del año como senos y cosenos (hora local del centro del intervalo)."""
    c = (indice - pd.Timedelta("30min")).tz_convert(config.ZONA)
    h = c.hour + c.minute / 60
    doy = c.dayofyear
    return pd.DataFrame({"hora_sin": np.sin(2 * np.pi * h / 24), "hora_cos": np.cos(2 * np.pi * h / 24),
                         "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25)},
                        index=indice)


def geometria(indice: pd.DatetimeIndex, lugar) -> pd.DataFrame:
    """Elevacion y azimut solar en el centro del intervalo, y coseno del cenit (0 de noche)."""
    pos = solar.posicion_solar(indice, lugar, pd.Timedelta("30min"))
    return pd.DataFrame({"elevacion_solar": pos["apparent_elevation"], "azimut_solar": pos["azimuth"],
                         "cos_cenit": np.cos(np.radians(pos["zenith"])).clip(lower=0)}, index=indice)


def rezagos(s: pd.Series, pasos: list[int], nombre: str | None = None) -> pd.DataFrame:
    """s desplazada `p` pasos (p > 0 mira al pasado)."""
    n = nombre or s.name
    return pd.DataFrame({f"{n}_rez{p}": s.shift(p) for p in pasos}, index=s.index)


def dias_secos(precipitacion_diaria: pd.Series, umbral_mm: float = 1.0) -> pd.Series:
    """Dias consecutivos sin lluvia (< umbral) hasta cada día: indicador de acumulacón de suciedad.
    La lluvia de al menos `umbral_mm` reinicia la cuenta (limpieza natural, Kimber et al., 2006)."""
    seco = (precipitacion_diaria < umbral_mm).astype(int)
    grupo = (seco == 0).cumsum()
    return seco.groupby(grupo).cumsum().rename("dias_secos")


def temperatura_modulo(poa: pd.Series, t_aire: pd.Series, viento: pd.Series | None = None) -> pd.Series:
    """Temperatura de celda estimada (Faiman) a partir de la POA, la temperatura y el viento."""
    v = viento.reindex(poa.index) if viento is not None else None
    return solar.temperatura_modulo(poa, t_aire.reindex(poa.index), v).rename("t_modulo")


def diario_a_horario(diario: pd.Series, indice: pd.DatetimeIndex, desfase_dias: int = 0) -> pd.Series:
    """Asigna a cada hora el valor diario del día local (menos `desfase_dias`): p. ej. el Yf del día
    anterior completo para todas las horas del día objetivo."""
    from . import particiones
    d = diario.copy()
    d.index = pd.DatetimeIndex(d.index).normalize() + pd.Timedelta(days=desfase_dias)
    dias = particiones.dias_locales(indice)
    return pd.Series(d.reindex(dias).to_numpy(), index=indice, name=f"{diario.name}_dia_rez{desfase_dias}")


def matriz_h2(horario: pd.DataFrame, diario: pd.DataFrame, exogenas: pd.DataFrame | None, lugar,
              col_y: str = "yf", rezagos_h=(24, 48, 168)) -> tuple[pd.DataFrame, pd.Series]:
    """Matriz de características para H2 (día siguiente horario, emitido al final del día D - 1).

    Rezagos horarios del objetivo >= 24 h, k* de la misma hora del día anterior, Yf y k* del día anterior
    completo, Yr de cielo despejado (conocido de antemano), geometría solar, variables ciclicas y las
    exógenas del escenario (ya alineadas a la hora objetivo)."""
    y = horario[col_y]
    partes = [rezagos(y, list(rezagos_h), col_y),
              rezagos(horario["kstar"], [24], "kstar"),
              diario_a_horario(diario["yf"], horario.index, 1),
              diario_a_horario(diario["kstar"], horario.index, 1),
              horario[["yr_cs"]], geometria(horario.index, lugar), ciclicas(horario.index)]
    if exogenas is not None:
        partes.append(exogenas.reindex(horario.index))
    return pd.concat(partes, axis=1), y


# --- Matriz de todas las variables de todas las fuentes (correlación de todas contra todas) ---------
# Fuente cruda -> (prefijo corto, paso nativo, marca de tiempo, filtro). Las climatologías y las ejecuciones
# completas no entran (otra escala de tiempo o varias ejecuciones por hora).
FUENTES_MATRIZ = {
    "om_satellite__eumetsat_lsa_saf_msg": ("msg", "15min", "fin", None),
    "om_archive__ecmwf_ifs": ("ifs", "1h", "fin", None),
    "om_archive__era5": ("era5", "1h", "fin", None),
    "om_previous_runs__ecmwf_ifs025": ("ifs025_d1", "1h", "fin", {"previousday": 1}),
    "om_previous_runs__gfs_seamless": ("gfs_d1", "1h", "fin", {"previousday": 1}),
    "om_historical_forecast__gfs_seamless": ("gfs_0_6h", "1h", "fin", None),
    "om_historical_forecast__icon_seamless": ("icon_0_6h", "1h", "fin", None),
    "om_air_quality__cams_global": ("cams", "1h", "fin", None),
    "nasa_power": ("nasa", "1h", "inicio", None),
    "nsrdb_goes": ("nsrdb", "10min", "instante", None),
}
_NO_VARIABLES = {"datatime", "modelo", "previousday", "run", "grupo", "cargatime", "cloud_type", "wind_direction_10m",
                 "wd10m", "solar_zenith_angle"}


def matriz_todas_las_fuentes(crudos: dict[str, pd.DataFrame], extra: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Todas las variables numéricas de todas las fuentes meteorológicas en una rejilla horaria común
    (etiqueta al final de la hora, UTC), con nombre fuente.variable. Solo para la exploracón (correlación
    de todas contra todas y PCA): no aplica imputación; los valores imposibles de irradiancia de la BSRN
    se tratan como faltantes con calidad.limites_irradiancia en las fuentes que la tienen. extra: tablas
    horarias ya preparadas que se agregan con su prefijo (p. ej. {"ideam": tabla, "inversor": tabla})."""
    columnas = {}
    for clave, (pref, paso, marca, filtro) in FUENTES_MATRIZ.items():
        df = crudos.get(clave)
        if df is None or df.empty:
            continue
        for k, v in (filtro or {}).items():
            df = df[df[k] == v]
        vars_ = [c for c in df.columns if c not in _NO_VARIABLES and pd.api.types.is_numeric_dtype(df[c])]
        s = df.set_index("datatime")[vars_].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.where(s > -900)                                  # NASA POWER marca los faltantes con -999
        if paso != "1h":
            s = s.resample("1h", label="right", closed="right").mean()
        elif marca == "inicio":
            s.index = s.index + pd.Timedelta("1h")
        for c in s.columns:
            columnas[f"{pref}.{c}"] = s[c]
    for pref, tabla in (extra or {}).items():
        for c in tabla.columns:
            columnas[f"{pref}.{c}"] = tabla[c]
    m = pd.concat(columnas, axis=1)
    return m.loc[:, m.notna().sum() > 24 * 30]                 # fuera variables con menos de un mes de datos


def pares_redundantes(c: pd.DataFrame, umbral: float = 0.99) -> pd.DataFrame:
    """Pares de variables con |r| >= umbral: entregan prácticamente la misma información."""
    a = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack()
    a = a[a.abs() >= umbral]
    return a.rename("correlacion").reset_index().rename(columns={"level_0": "variable_a", "level_1": "variable_b"}) \
        .sort_values("correlacion", key=np.abs, ascending=False).reset_index(drop=True)
