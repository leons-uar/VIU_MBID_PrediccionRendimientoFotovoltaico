"""
caracterizacion.py
==================
Descripción de las series crudas: cuántos registros tienen, en que rango, con que paso nativo,
cuántos nulos, de que tipo y fuente es cada variable y en que unidad se mide. Es la etapa de
"seleccion y comprensón de los datos" del KDD, antes de tocar nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, datos, imputacion


def paso_modal(tiempos: pd.Series) -> pd.Timedelta | None:
    """Paso más frecuente entre instantes consecutivos (redondeado al segundo)."""
    t = pd.Series(pd.to_datetime(tiempos).drop_duplicates().sort_values().to_numpy())
    if len(t) < 3:
        return None
    dif = t.diff().dropna().dt.round("s")
    return dif.mode().iloc[0] if not dif.empty else None


def resumen_series(crudos: dict[str, pd.DataFrame], series: list[datos.Serie]) -> pd.DataFrame:
    """Una fila por serie: origen, filtros, registros, rango, paso modal y % de nulos medio."""
    por_nombre = {s.nombre: s for s in series}
    filas = []
    for nombre, df in crudos.items():
        s = por_nombre.get(nombre)
        variables = [c for c in df.columns if c in (s.variables if s else df.columns)]
        claves = [c for c in ("modelo", "previousday", "run") if c in df.columns]
        tiempos = df["datatime"] if not claves else df.loc[df.groupby(claves, dropna=False).ngroup() == 0, "datatime"]
        filas.append({
            "serie": nombre, "origen": s.origen if s else "", "filtros": s.filtros if s else {},
            "registros": len(df), "variables": len(variables),
            "inicio_utc": df["datatime"].min(), "fin_utc": df["datatime"].max(),
            "paso_nativo": paso_modal(tiempos),
            "pct_nulos_medio": float(df[variables].isna().mean().mean() * 100) if variables else np.nan,
        })
    return pd.DataFrame(filas)


def resumen_variables(df: pd.DataFrame, origen: str, variables: list[str]) -> pd.DataFrame:
    """Estadisticos por variable con unidad y naturaleza (de meteo.variables cuando aplica)."""
    cat = datos.catalogo()
    esquema, tabla = origen.split(".", 1)
    meta = cat[(cat.esquema == esquema) & (cat.tabla == tabla)].set_index("columna")
    filas = []
    for v in variables:
        s = df[v]
        numerica = pd.api.types.is_numeric_dtype(s)
        if esquema == "telemetry":      # la API de SolaX no entrega unidades: se deducen del nombre
            unidad, naturaleza = imputacion.unidad_telemetria(v), imputacion.naturaleza_telemetria(v)
        else:
            unidad = meta.get("unidad", pd.Series(dtype=str)).get(v, "")
            naturaleza = meta.get("tipovalor", pd.Series(dtype=str)).get(v, "")
        filas.append({
            "variable": v, "tipo": str(s.dtype), "unidad": unidad, "naturaleza": naturaleza,
            "registros": int(s.notna().sum()), "nulos": int(s.isna().sum()), "pct_nulos": float(s.isna().mean() * 100),
            "minimo": s.min() if numerica else None, "media": s.mean() if numerica else None,
            "mediana": s.median() if numerica else None, "maximo": s.max() if numerica else None,
            "desviacion": s.std() if numerica else None,
        })
    return pd.DataFrame(filas)


def cobertura_diaria(regular: pd.Series, de_dia: pd.Series | None = None) -> pd.Series:
    """Fraccion de instantes con dato por día local (solo diurnos si se pasa de_dia)."""
    presente = regular.notna()
    if de_dia is not None:
        presente = presente[de_dia.reindex(presente.index).fillna(False)]
    dias = presente.index.tz_convert(config.ZONA).date
    return presente.groupby(dias).mean().rename("cobertura")


def comparar_fuentes(series: dict[str, pd.Series], mascara: pd.Series | None = None) -> pd.DataFrame:
    """Error entre pares de fuentes (misma variable y rejilla): sesgo (MBE), RMSE y correlación, en
    unidades y en % de la media de la fuente a. máscara: solo se comparan los instantes en True (p. ej.
    horas de sol, para que las noches en 0 no inflen la correlación)."""
    nombres = list(series)
    filas = []
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            x, y = series[a].align(series[b], join="inner")
            ok = x.notna() & y.notna()
            if mascara is not None:
                ok &= mascara.reindex(x.index).fillna(False).astype(bool)
            if ok.sum() < 10:
                continue
            e = y[ok] - x[ok]
            media = float(x[ok].mean())
            rmse = float(np.sqrt((e ** 2).mean()))
            rel = (lambda v: 100 * v / media) if media else (lambda v: np.nan)    # fuente a sin irradiancia
            filas.append({"fuente_a": a, "fuente_b": b, "n": int(ok.sum()), "mbe_b_menos_a": e.mean(),
                          "rmse": rmse, "rmbe_pct": rel(e.mean()), "rrmse_pct": rel(rmse),
                          "correlacion": float(x[ok].corr(y[ok]))})
    return pd.DataFrame(filas)


# --- Ficha de las fuentes meteorológicas ----------------------------------------------------------
# Resolución y retraso según la documentación de cada proveedor (Open-Meteo: páginas de cada API;
# NASA POWER: Data Sources; NREL: NSRDB GOES v4; IDEAM: datos.gov.co), consultada en octubre de 2026.
# El retraso "medido" se calcula con los datos de la base en ficha_fuentes().
FICHA_FUENTES = {
    "om_satellite__eumetsat_lsa_saf_msg": {"fuente": "Satélite MSG (EUMETSAT LSA SAF)", "tipo": "observación por satélite",
        "resolucion": "0,05° (~5 km) en el nadir; más gruesa en el borde del disco", "paso": "15 min",
        "licencia": "CC BY 4.0", "pronostico": False},
    "om_archive__ecmwf_ifs": {"fuente": "Análisis ECMWF IFS", "tipo": "análisis de modelo numérico",
        "resolucion": "9 km", "paso": "1 h", "licencia": "CC BY 4.0", "pronostico": False},
    "om_archive__era5": {"fuente": "Reanálisis ERA5", "tipo": "reanálisis", "resolucion": "0,25° (~25 km)",
        "paso": "1 h", "licencia": "CC BY 4.0", "pronostico": False},
    "nsrdb_goes": {"fuente": "NSRDB GOES v4 (NREL)", "tipo": "observación por satélite", "resolucion": "2 km",
        "paso": "10 min", "licencia": "pública", "pronostico": False},
    "nasa_power": {"fuente": "NASA POWER", "tipo": "satélite (CERES) y reanálisis (MERRA-2)",
        "resolucion": "radiación 1° (~111 km); meteorología 0,5° × 0,625°", "paso": "1 h",
        "licencia": "pública", "pronostico": False},
    "om_previous_runs__ecmwf_ifs025": {"fuente": "Pronóstico ECMWF IFS 0,25° archivado", "tipo": "pronóstico numérico",
        "resolucion": "0,25° (~28 km)", "paso": "1 h", "licencia": "CC BY 4.0", "pronostico": True},
    "om_previous_runs__gfs_seamless": {"fuente": "Pronóstico NCEP GFS archivado", "tipo": "pronóstico numérico",
        "resolucion": "0,25° (~28 km) en Colombia", "paso": "1 h", "licencia": "CC BY 4.0", "pronostico": True},
    "om_air_quality__cams_global": {"fuente": "CAMS global (aerosoles)", "tipo": "análisis y pronóstico de composición",
        "resolucion": "0,4° (~45 km)", "paso": "1 h", "licencia": "CC BY 4.0", "pronostico": True},
    "ideam": {"fuente": "Estaciones automáticas IDEAM", "tipo": "observación en tierra (sin radiación)",
        "resolucion": "puntual (estación)", "paso": "2 a 60 min", "licencia": "CC BY-SA 4.0, no validados",
        "pronostico": False},
}
# Columna con la variable principal de cada fuente para medir hasta cuando hay dato
_COLUMNA_PRINCIPAL = {"om_satellite__eumetsat_lsa_saf_msg": "shortwave_radiation", "om_archive__ecmwf_ifs": "shortwave_radiation",
                      "om_archive__era5": "shortwave_radiation", "nsrdb_goes": "ghi", "nasa_power": "allsky_sfc_sw_dwn",
                      "om_previous_runs__ecmwf_ifs025": "shortwave_radiation", "om_previous_runs__gfs_seamless": "shortwave_radiation",
                      "om_air_quality__cams_global": "aerosol_optical_depth", "ideam": "valorobservado"}


def _distancia_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def retraso_publicacion(grupo: str, ahora: pd.Timestamp | None = None) -> dict[str, tuple[pd.Timestamp, float]]:
    """Ultimo dato válido de cada fuente en la base de datos y su retraso en días respecto a `ahora` (por
    defecto, el momento de la consulta). Es el retraso con que la fuente llega en operación, que no depende
    de la ventana de la extracción del cuaderno. Las columnas de radiación se miden aparte porque algunas
    fuentes (NASA POWER) publican la meteorología antes que la radiación."""
    g = datos._validar_grupo(grupo)
    if ahora is None:
        # Con los datos publicados, "ahora" es el fin de lo observado (así el resultado no cambia con el día en que se ejecuta)
        ahora = (pd.Timestamp(config.fin_observado(), tz=config.ZONA).tz_convert("UTC") if config.FUENTE_DATOS != "bd"
                 else pd.Timestamp.now(tz="UTC"))
    salida = {}
    for clave, col in _COLUMNA_PRINCIPAL.items():
        if clave == "ideam":
            continue
        tabla, _, modelo = clave.partition("__")
        try:
            ultimo = datos.ultimo_valido(g, tabla, modelo or None, col)
        except Exception:  # noqa: BLE001 (tabla sin la columna o sin acceso: la fuente queda sin dato)
            continue
        if pd.notna(ultimo):
            salida[clave] = (ultimo, round((ahora - ultimo) / pd.Timedelta("1D"), 1))
    return salida


def ficha_fuentes(crudos: dict[str, pd.DataFrame], punto, celdas: pd.DataFrame, fecha_referencia: pd.Timestamp,
                  retraso_max_dias: float = 2.0, publicacion: dict[str, tuple[pd.Timestamp, float]] | None = None) -> pd.DataFrame:
    """Viabilidad operativa de cada fuente: resolución, distancia del punto de la localidad a la celda que
    devolvió la API, último dato disponible, retraso respecto a `fecha_referencia` y si puede usarse en
    producción (retraso de hasta `retraso_max_dias`; los pronósticos siempre están disponibles antes del día).
    Con `publicacion` (salida de retraso_publicacion) el último dato y el retraso son los de la base de datos
    en el momento de la ejecución, no los de la ventana extraida."""
    filas = []
    for clave, ficha in FICHA_FUENTES.items():
        if clave not in crudos:
            continue
        df = crudos[clave]
        col = _COLUMNA_PRINCIPAL.get(clave)
        if "previousday" in df.columns:
            df = df[df["previousday"] == df["previousday"].min()]
        valido = df.loc[df[col].notna(), "datatime"] if col in df.columns else df["datatime"]
        if ficha["pronostico"]:
            valido = valido[valido <= fecha_referencia]
        ultimo = valido.max() if len(valido) else pd.NaT
        tabla, _, modelo = clave.partition("__")
        c = celdas[(celdas["fuente"] == tabla) & ((celdas["modelo"] == modelo) | (modelo == ""))] if len(celdas) else celdas
        dist = _distancia_km(punto.latitude, punto.longitude, c.iloc[0].latitude, c.iloc[0].longitude) if len(c) else np.nan
        retraso = (fecha_referencia - ultimo) / pd.Timedelta("1D") if pd.notna(ultimo) else np.nan
        if publicacion and clave in publicacion and not ficha["pronostico"]:
            ultimo, retraso = publicacion[clave]
        filas.append({"serie": clave, **{k: v for k, v in ficha.items() if k != "pronostico"},
                      "distancia_celda_km": round(dist, 1) if dist == dist else np.nan,
                      "ultimo_dato_utc": ultimo, "retraso_dias": round(retraso, 1) if retraso == retraso else np.nan,
                      "apta_produccion": bool(ficha["pronostico"] or (retraso == retraso and retraso <= retraso_max_dias))})
    return pd.DataFrame(filas).set_index("serie")
