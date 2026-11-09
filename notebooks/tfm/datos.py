"""
datos.py
========
Lectura de los datos crudos, tal como los entregaron las API: la telemetría de cada inversor SolaX con su muestreo
nativo (5 min), los productos meteorológicos de cada localidad (15 min el satélite, horarios los reanálisis y los
pronósticos, 30 min NSRDB) y las estaciones del IDEAM. Aquí no se agrega, corrige ni imputa nada: eso ocurre después,
en módulos separados y documentados (calidad, imputación, agregación).

Hay dos fuentes con la misma interfaz (config.FUENTE_DATOS):
* "csv", la de los cuadernos: los archivos de datos/ que acompanan al repositorio. datos/MANIFIESTO.sha256 guarda la
  huella de cada archivo y cada lectura la comprueba, así que un archivo alterado o incompleto detiene el cuaderno.
* "bd": la base PostgreSQL de la que salen esos archivos. Solo la usan la exportación de los CSV y la operación diaria.

En los CSV, los identificadores son los códigos de la tesis (P01, P01I0102): no hay uuid, seriales ni nombres.

    series = datos.series_localidad("G01")
    crudos = datos.leer(series, "2024-01-01", "2026-11-07")
    const = datos.constantes("G01")
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config, consultas

_RE_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_RE_IDENT = re.compile(r"^[a-z0-9_]+$")
_RE_GRUPO = re.compile(r"^G\d{2}$")

# Claves (no variables) de cada tabla meteorológica, además de grupo y datatime.
CLAVES_METEO = {
    "om_archive": ["modelo"], "om_satellite": ["modelo"], "om_historical_forecast": ["modelo"],
    "om_air_quality": ["modelo"], "om_previous_runs": ["modelo", "previousday"],
    "om_single_runs": ["modelo", "run"], "nasa_power": [], "nsrdb_goes": [],
}
# Modelos descargados por producto.
MODELOS_METEO = {
    "om_archive": ["era5", "ecmwf_ifs"], "om_satellite": ["eumetsat_lsa_saf_msg"],
    "om_historical_forecast": ["gfs_seamless", "icon_seamless"], "om_air_quality": ["cams_global"],
    "om_previous_runs": ["ecmwf_ifs025", "gfs_seamless"], "om_single_runs": ["ecmwf_ifs"],
    "nasa_power": [None], "nsrdb_goes": [None],
}
# Columnas de telemetría que no son medidas (no se piden como variables).
NO_VARIABLES = {"datatime", "deviceuuid", "cargatime", "grupo", "modelo", "previousday", "run"}
# Telemetría que usa el análisis: potencia AC por fase, energía, temperatura, potencia y tensión de las entradas DC,
# estado, potencia de red (sensor) y la hora local que informa la planta (para corregir el reloj)
COLUMNAS_TELEMETRIA = ("plantlocaltime", "devicestatus", "acpower1", "acpower2", "acpower3", "totalyield", "dailyyield",
                       "invertertemperature", "mppttotalinputpower", "gridpower",
                       *(f"mppt{k}{m}" for k in range(1, 13) for m in ("voltage", "power")),
                       *(f"pv{k}{m}" for k in range(1, 13) for m in ("voltage", "power")))

DIR_PARAMETROS = config.DIR_DATOS / "parametros"
MANIFIESTO = config.DIR_DATOS / "MANIFIESTO.sha256"


@dataclass(frozen=True)
class Serie:
    """Especificacion de una serie: de donde sale, que variables y con que filtros."""
    nombre: str                                   # etiqueta única en la localidad
    origen: str                                   # esquema.tabla, p. ej. "meteo.om_satellite"
    variables: tuple[str, ...]                    # columnas de medida
    filtros: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    grupo: str | None = None                      # localidad a la que pertenece

    @property
    def tabla(self) -> str:
        return self.origen.split(".", 1)[1]


def _csv() -> bool:
    return config.FUENTE_DATOS != "bd"


# --- Archivos de datos/ (fuente "csv") --------------------------------------------------------------

@lru_cache(maxsize=1)
def _huellas() -> dict[str, str]:
    if not MANIFIESTO.exists():
        raise FileNotFoundError(f"No estan los datos: falta {MANIFIESTO} (ver README, seccion Datos)")
    filas = (linea.split("  ", 1) for linea in MANIFIESTO.read_text(encoding="utf-8").splitlines() if linea.strip())
    return {ruta.strip(): huella for huella, ruta in filas}


_VERIFICADOS: set[str] = set()


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def archivo(relativo: str) -> Path:
    """Ruta de un archivo de datos/, comprobada contra su huella SHA256 (una vez por proceso)."""
    ruta = config.DIR_DATOS / relativo
    if relativo not in _VERIFICADOS:
        esperada = _huellas().get(relativo)
        if esperada is None:
            raise FileNotFoundError(f"{relativo} no esta en datos/MANIFIESTO.sha256")
        if not ruta.exists():
            raise FileNotFoundError(f"Falta el archivo de datos {ruta}")
        if sha256(ruta) != esperada:
            raise ValueError(f"{relativo}: el contenido no coincide con su huella en datos/MANIFIESTO.sha256")
        _VERIFICADOS.add(relativo)
    return ruta


@lru_cache(maxsize=None)
def parametro(nombre: str) -> pd.DataFrame:
    """Tabla de datos/parametros/ (constantes, catálogos, nomenclatura)."""
    return pd.read_csv(archivo(f"parametros/{nombre}.csv"), dtype={"plantuuid": str, "deviceuuid": str})


def _de_localidad(nombre: str, grupo: str) -> pd.DataFrame:
    t = parametro(nombre)
    return t[t["localidad"] == grupo].drop(columns="localidad").reset_index(drop=True)


@lru_cache(maxsize=6)
def _serie_csv(grupo: str, nombre: str) -> pd.DataFrame:
    """Todas las filas publicadas de una serie (se filtran por fecha en leer_serie)."""
    s = parametro("series")
    fila = s[(s["localidad"] == grupo) & (s["serie"] == nombre)]
    if fila.empty:
        raise KeyError(f"{grupo}: no hay datos publicados de la serie {nombre}")
    texto = {c: str for c in json.loads(fila["texto"].iloc[0])}     # códigos con ceros a la izquierda, modelos, unidades
    partes = [pd.read_csv(archivo(a), engine="pyarrow", dtype=texto or None) for a in json.loads(fila["archivos"].iloc[0])]
    return _a_utc(pd.concat(partes, ignore_index=True))


# --- Base de datos (fuente "bd") ------------------------------------------------------------------

def _sql_a_pandas(sql: str, perfil: str = "ANALISIS") -> pd.DataFrame:
    import sys
    import connectorx as cx
    sys.path.insert(0, str(config.RAIZ / "src"))
    from bd_consultas import _uri  # (credenciales desde .env, nunca se imprimen)
    return cx.read_sql(_uri(perfil), sql, return_type="pandas")


# --- Utilidades -----------------------------------------------------------------------------------

def _instante(valor) -> str:
    """Fecha o instante a ISO UTC. Una fecha sin hora se interpreta en hora de Colombia."""
    t = pd.Timestamp(valor)
    if t.tzinfo is None:
        t = t.tz_localize(config.ZONA)
    return t.tz_convert("UTC").isoformat()


def _a_utc(df: pd.DataFrame, columnas=("datatime", "run")) -> pd.DataFrame:
    for c in columnas:
        if c in df.columns:
            # En los CSV conviven instantes con y sin fracción de segundo: ISO 8601 los lee todos
            texto = not pd.api.types.is_datetime64_any_dtype(df[c])
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601" if texto else None).dt.as_unit("ns")
    return df


@lru_cache(maxsize=1)
def _columnas() -> pd.DataFrame:
    return parametro("columnas") if _csv() else _sql_a_pandas(consultas.COLUMNAS)


def columnas_de(origen: str) -> list[str]:
    esquema, tabla = origen.split(".", 1)
    c = _columnas()
    return c.loc[(c.esquema == esquema) & (c.tabla == tabla)].sort_values("orden")["columna"].tolist()


@lru_cache(maxsize=1)
def variables_meteo() -> pd.DataFrame:
    """Diccionario de variables meteorológicas (unidad, tipo de valor, marca de tiempo, periodo)."""
    return parametro("variables_meteo") if _csv() else _sql_a_pandas(consultas.VARIABLES_METEO)


def catalogo() -> pd.DataFrame:
    """Todas las columnas de telemetría y meteorología con su tipo y, si es meteorológica, su unidad y naturaleza."""
    c = _columnas().copy()
    v = variables_meteo().rename(columns={"fuente": "tabla", "variable": "columna"})
    v["esquema"] = "meteo"
    return c.merge(v, on=["esquema", "tabla", "columna"], how="left")


def oni() -> pd.DataFrame:
    """Indice ONI mensual de NOAA (El Niño / La Niña): anio, mes, oni."""
    return parametro("oni") if _csv() else _sql_a_pandas(consultas.ONI)


def plantas_grupos() -> pd.DataFrame:
    """Localidad (grupo meteorológico) de cada planta."""
    if _csv():
        return parametro("plantas_grupos")
    return _sql_a_pandas("SELECT plantuuid::text AS plantuuid, grupo FROM meteo.plantas_grupos")


# --- Constantes de una localidad ------------------------------------------------------------------

def _validar_grupo(grupo: str) -> str:
    if not _RE_GRUPO.match(grupo or ""):
        raise ValueError(f"grupo no valido: {grupo!r} (formato G01)")
    return grupo


def _caras(texto) -> list[tuple[float, float, float]] | None:
    if not isinstance(texto, str) or not texto.strip():
        return None
    return [tuple(map(float, c)) for c in json.loads(texto)]


def constantes(grupo: str) -> dict[str, pd.DataFrame]:
    """Constantes de la localidad: punto (coordenadas y elevación), celdas de cada fuente, plantas, dispositivos con su
    configuración FV, etapas de capacidad y eventos declarados por el instalador."""
    g = _validar_grupo(grupo)
    if _csv():
        disp = _de_localidad("dispositivos", g)
        disp["caras"] = disp["caras"].map(_caras)
        plantas = _a_utc(_de_localidad("plantas", g), ("createtime",))
        etapas = _de_localidad("etapas", g)
        eventos = _de_localidad("eventos", g)
        for t, cols in ((etapas, ("desde",)), (eventos, ("desde", "hasta", "conformidad_or"))):
            for c in cols:
                t[c] = pd.to_datetime(t[c])
        return {"grupo": _de_localidad("grupos", g), "celdas": _de_localidad("celdas", g), "plantas": plantas,
                "dispositivos": disp, "etapas": etapas, "eventos": eventos}
    return _constantes_bd(g)


def _constantes_bd(g: str) -> dict[str, pd.DataFrame]:
    punto = _sql_a_pandas(consultas.GRUPO.format(grupo=g))
    celdas = _sql_a_pandas(consultas.CELDAS_GRUPO.format(grupo=g))
    plantas = _a_utc(_sql_a_pandas(consultas.PLANTAS_GRUPO.format(grupo=g)), ("createtime",))
    disp = _sql_a_pandas(consultas.DISPOSITIVOS_GRUPO.format(grupo=g))
    # Caras del generador de cada planta (telemetry.plantas_fv_caras): [(inclinación, azimut, fracción)]
    caras = _sql_a_pandas(consultas.CARAS_GRUPO.format(grupo=g))
    por_planta = {u: [(float(r.inclinacion), float(r.azimut), float(r.fraccion)) for r in grp.itertuples()]
                  for u, grp in caras.groupby("plantuuid")}
    disp["caras"] = disp["plantuuid"].map(por_planta)
    plantas = plantas.merge(_sql_a_pandas(consultas.PLANTAS_FV_GRUPO.format(grupo=g)), on="plantuuid", how="left")
    # Elevación del punto: la que devuelve Open-Meteo (modelo digital de 90 m) para el satélite o IFS.
    elev = celdas.loc[celdas.fuente.isin(["om_satellite", "om_archive"]), "elevation"].dropna()
    punto["elevacion"] = float(elev.iloc[0]) if len(elev) else 0.0
    etapas = _sql_a_pandas(consultas.ETAPAS_GRUPO.format(grupo=g))
    eventos = _sql_a_pandas(consultas.EVENTOS_GRUPO.format(grupo=g))
    # Coordenadas de las plantas a 0,01 grados (unos 1,1 km): las mismas que se publican
    for c in ("latitude", "longitude"):
        plantas[c] = plantas[c].astype(float).round(2)
    # Coeficiente de temperatura y NOCT del módulo de cada planta, para la auditoría del P0
    disp = disp.merge(plantas[["plantuuid", "gamma_pmax_pct_c", "noct_c", "bifacialidad_pct"]], on="plantuuid", how="left")
    return {"grupo": punto, "celdas": celdas, "plantas": plantas, "dispositivos": disp, "etapas": etapas, "eventos": eventos}


# --- Series de una localidad ----------------------------------------------------------------------

def _columnas_con_datos(tabla: str, deviceuuid: str) -> list[str]:
    candidatas = [c for c in columnas_de(f"telemetry.{tabla}") if c not in NO_VARIABLES]
    conteos = ", ".join(f"count({c}) AS {c}" for c in candidatas)
    fila = _sql_a_pandas(consultas.CONTEO_COLUMNAS_DISPOSITIVO.format(conteos=conteos, tabla=tabla,
                                                                       deviceuuid=deviceuuid)).iloc[0]
    return [c for c in candidatas if int(fila[c]) > 0]


def series_localidad(grupo: str, meteo: bool = True, ideam: bool = True) -> list[Serie]:
    """Todas las series de una localidad: cada inversor y medidor SolaX y cada producto o modelo meteorológico."""
    g = _validar_grupo(grupo)
    if _csv():
        s = parametro("series")
        s = s[(s["localidad"] == g) & ~s["serie"].str.endswith("_climatologia")]   # la climatología se pide aparte
        es_ideam = s["origen"] == "meteo.ideam_observaciones"
        es_meteo = s["origen"].str.startswith("meteo.") & ~es_ideam
        s = s[~(es_meteo & (not meteo)) & ~(es_ideam & (not ideam))]
        return [Serie(r.serie, r.origen, tuple(json.loads(r.variables)), json.loads(r.filtros), g) for r in s.itertuples()]
    disp = constantes(g)["dispositivos"]
    series: list[Serie] = []
    for _, d in disp.iterrows():
        tabla = {1: "inverter_history", 3: "meter_history"}.get(int(d.devicetype))
        if tabla is None:
            continue
        cols = _columnas_con_datos(tabla, d.deviceuuid)
        if tabla == "inverter_history":
            cols = [c for c in cols if c in COLUMNAS_TELEMETRIA]
        if cols:
            pref = "inv" if tabla.startswith("inverter") else "med"
            series.append(Serie(f"{pref}_{d.deviceuuid[:8]}", f"telemetry.{tabla}", tuple(cols),
                                {"deviceuuid": d.deviceuuid}, g))
    if meteo:
        vm = variables_meteo()
        for tabla, modelos in MODELOS_METEO.items():
            variables = tuple(vm.loc[vm.fuente == tabla, "variable"])
            for m in modelos:
                nombre = tabla if m is None else f"{tabla}__{m}"
                series.append(Serie(nombre, f"meteo.{tabla}", variables, {"modelo": m} if m else {}, g))
    if ideam:
        series.append(Serie("ideam", "meteo.ideam_observaciones", ("valorobservado",), {}, g))
    return series


# --- Lectura --------------------------------------------------------------------------------------

def sql_de(serie: Serie, inicio, fin) -> str:
    """SQL exacto (validado) que se envia a la base para una serie (fuente "bd")."""
    desde, hasta = _instante(inicio), _instante(fin)
    validas = set(columnas_de(serie.origen))
    for v in serie.variables:
        if not _RE_IDENT.match(v) or (v not in validas and serie.tabla != "ideam_observaciones"):
            raise ValueError(f"variable no valida para {serie.origen}: {v!r}")
    if serie.origen.startswith("telemetry."):
        uuid = str(serie.filtros.get("deviceuuid", "")).lower()
        if not _RE_UUID.match(uuid):
            raise ValueError(f"deviceuuid no valido: {uuid!r}")
        return consultas.SERIE_DISPOSITIVO.format(columnas=", ".join(serie.variables), tabla=serie.tabla,
                                                  deviceuuid=uuid, inicio=desde, fin=hasta)
    g = _validar_grupo(serie.grupo or "")
    if serie.tabla == "ideam_observaciones":
        return consultas.IDEAM_GRUPO.format(grupo=g, inicio=desde, fin=hasta)
    claves = CLAVES_METEO[serie.tabla]
    filtro = ""
    modelo = serie.filtros.get("modelo")
    if modelo is not None:
        if not _RE_IDENT.match(modelo):
            raise ValueError(f"modelo no valido: {modelo!r}")
        filtro += f"AND modelo = '{modelo}' "
    if "previousday" in serie.filtros:
        filtro += f"AND previousday = {int(serie.filtros['previousday'])} "
    orden = ", ".join(["datatime", *[c for c in claves if c != "modelo"]])
    return consultas.SERIE_METEO.format(claves="".join(f"{c}, " for c in claves), columnas=", ".join(serie.variables),
                                        tabla=serie.tabla, grupo=g, filtro=filtro, inicio=desde, fin=hasta, orden=orden)


def _ruta_cache(serie: Serie):
    return config.DIR_CACHE / (serie.grupo or "global") / f"{serie.nombre}.parquet"


def filas_en_bd(sql: str) -> int:
    """Numero de filas que devuelve hoy la consulta, sin traerlas (el conteo lo hace la base)."""
    return int(_sql_a_pandas(f"SELECT count(*) AS n FROM ({sql}) AS q").iloc[0, 0])


def leer_serie(serie: Serie, inicio, fin, refrescar: bool = False) -> pd.DataFrame:
    """Serie cruda entre inicio (incluido) y fin (excluido), en el mismo orden que la consulta a la base. Índice
    0..n; datatime en UTC (no único si hay claves, como el día de emisión de un pronóstico)."""
    if _csv():
        df = _serie_csv(serie.grupo, serie.nombre)
        desde, hasta = pd.Timestamp(_instante(inicio)), pd.Timestamp(_instante(fin))
        df = df[(df["datatime"] >= desde) & (df["datatime"] < hasta)]
        claves = [] if not serie.origen.startswith("meteo.") or serie.tabla == "ideam_observaciones" else \
            [c for c in CLAVES_METEO[serie.tabla] if c != "modelo"]
        return df.sort_values(["datatime", *claves], kind="stable").reset_index(drop=True)
    return _leer_serie_bd(serie, inicio, fin, refrescar)


def _leer_serie_bd(serie: Serie, inicio, fin, refrescar: bool = False) -> pd.DataFrame:
    """Lectura de la base con cache Parquet. La cache se usa solo si la consulta es la misma y la base devuelve hoy
    el mismo número de filas: los procesos de recuperacón cargan datos atrasados después de la extracción."""
    sql = sql_de(serie, inicio, fin)
    ruta = _ruta_cache(serie)
    meta_ruta = ruta.with_suffix(".json")
    if not refrescar and ruta.exists() and meta_ruta.exists():
        meta = json.loads(meta_ruta.read_text(encoding="utf-8"))
        if meta.get("sql") == sql and meta.get("filas") == filas_en_bd(sql):
            return pd.read_parquet(ruta)
    df = _a_utc(_sql_a_pandas(sql))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    meta = {"serie": serie.nombre, "origen": serie.origen, "filtros": serie.filtros, "inicio": _instante(inicio),
            "fin": _instante(fin), "filas": len(df), "sql": sql, "sha256": sha256(ruta),
            "extraido_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta_ruta.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return df


def leer(series: list[Serie], inicio, fin, refrescar: bool = False) -> dict[str, pd.DataFrame]:
    """Lee varias series: {nombre: DataFrame crudo}."""
    return {s.nombre: leer_serie(s, inicio, fin, refrescar) for s in series}


def meteo_hora(crudos: dict[str, pd.DataFrame], serie: str, variable: str, previousday: int | None = None) -> pd.Series | None:
    """Una variable horaria de una serie meteorológica cruda, indexada por su instante UTC (sin procesar:
    para comparaciones rápidas; el modelado usa caracteristicas.meteo_horaria con control de calidad)."""
    df = crudos.get(serie)
    if df is None or variable not in df.columns:
        return None
    if "previousday" in df.columns:
        df = df[df["previousday"] == (previousday if previousday is not None else df["previousday"].min())]
    s = df.set_index("datatime")[variable].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index().rename(variable)


def ultimo_valido(grupo: str, tabla: str, modelo: str | None, columna: str) -> pd.Timestamp:
    """Ultimo instante con un valor válido (>= 0) de una columna meteorológica en los datos de la localidad."""
    g = _validar_grupo(grupo)
    if _csv():
        nombre = tabla if not modelo else f"{tabla}__{modelo}"
        try:
            d = _serie_csv(g, nombre)
        except KeyError:
            return pd.NaT
        if columna not in d.columns:
            return pd.NaT
        validos = d.loc[d[columna] >= 0, "datatime"]
        return validos.max() if len(validos) else pd.NaT
    filtro = f" and modelo = '{modelo}'" if modelo else ""
    d = _sql_a_pandas(f"select max(datatime) as ultimo from meteo.{tabla} where grupo = '{g}'{filtro} and {columna} >= 0")
    return pd.to_datetime(d["ultimo"].iloc[0], utc=True) if len(d) and pd.notna(d["ultimo"].iloc[0]) else pd.NaT


def energia_diaria_contador(deviceuuids: list[str], desde, hasta, grupo: str | None = None) -> pd.Series:
    """Energia diaria (kWh) por fecha local de [desde, hasta), con el contador acumulado de cada inversor (totalyield):
    la diferencia entre el máximo de un día y el del día anterior, sumada entre inversores. No depende de que lleguen
    todas las lecturas de cinco minutos ni de la zona horaria del reloj del inversor; un día queda vacío si algún
    inversor no tiene lecturas ese día o el anterior, o si la diferencia es negativa (reinicio del contador)."""
    d0 = (pd.Timestamp(desde) - pd.Timedelta(days=1)).date()
    dias = pd.date_range(d0, pd.Timestamp(hasta) - pd.Timedelta(days=1), freq="D")
    if _csv():
        ini = pd.Timestamp(d0, tz=config.ZONA).tz_convert("UTC")
        fin = pd.Timestamp(pd.Timestamp(hasta).date(), tz=config.ZONA).tz_convert("UTC")
        s = parametro("series")
        partes = []
        for u in deviceuuids:
            g = grupo or s.loc[s["serie"] == f"inv_{u[:8]}", "localidad"].iloc[0]
            d = _serie_csv(g, f"inv_{u[:8]}")[["datatime", "totalyield"]]
            d = d[(d["datatime"] >= ini) & (d["datatime"] < fin)]
            partes.append(pd.DataFrame({"deviceuuid": u, "dia": d["datatime"].dt.tz_convert(config.ZONA).dt.tz_localize(None).dt.normalize(),
                                        "total": d["totalyield"]}))
        t = pd.concat(partes).groupby(["deviceuuid", "dia"], as_index=False)["total"].max() if partes else pd.DataFrame()
    else:
        uu = ", ".join(f"'{u}'" for u in deviceuuids)
        sql = (f"SELECT deviceuuid::text AS deviceuuid, (datatime AT TIME ZONE '{config.ZONA}')::date AS dia, max(totalyield) AS total "
               f"FROM telemetry.inverter_history WHERE deviceuuid IN ({uu}) "
               f"AND datatime >= ('{d0}'::date + interval '5 hours') AND datatime < ('{pd.Timestamp(hasta).date()}'::date + interval '5 hours') "
               f"GROUP BY 1, 2")
        t = _sql_a_pandas(sql)
    if t.empty:
        return pd.Series(np.nan, index=dias[1:], name="e_contador_kwh")
    t["dia"] = pd.to_datetime(t["dia"])
    tot = t.pivot(index="dia", columns="deviceuuid", values="total").reindex(columns=list(deviceuuids)).reindex(dias)
    e = tot.diff()
    e = e.where(e >= 0)
    return e.sum(axis=1, min_count=len(deviceuuids)).iloc[1:].rename("e_contador_kwh")


def manifiesto_cache(grupo: str) -> pd.DataFrame:
    """Lo que se leyo de cada serie de una localidad (serie, origen, rango, filas y huella)."""
    if _csv():
        s = parametro("series")
        return s[s["localidad"] == grupo].drop(columns="localidad").reset_index(drop=True)
    filas = [json.loads(m.read_text(encoding="utf-8")) for m in sorted((config.DIR_CACHE / grupo).glob("*.json"))]
    return pd.DataFrame(filas)
