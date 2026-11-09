"""
ideam.py
========
Observaciones de las estaciones automáticas del IDEAM (datos.gov.co) cercanas a cada localidad:
de filas nativas (estación, sensor, valor cada 2 a 10 minutos) a series horarias comparables con el
análisis IFS y ERA5 de Open-Meteo.

    1. Cada sensor se asigna a una variable común (temperature_2m, relative_humidity_2m, precipitation,
       surface_pressure, wind_speed_10m). Una estación puede tener dos sensores de la misma variable
       (convencional y GPRS): se usa el que tiene más datos.
    2. Control de calidad con límites físicos (los valores imposibles pasan a faltantes).
    3. Agregación horaria etiquetada al final de la hora: media para las variables continuas y suma para
       la precipitación, solo si la hora tiene al menos el 80 % de las lecturas esperadas según el paso
       nativo del sensor.
    4. Se elige la estación más cercana con datos de cada variable.

Corrección horaria: el proceso de extracción (src/28_ideam.py) calcula datatime = fechaobservacón + 5 h,
suponiendo que fechaobservacón esta en hora de Colombia. La temperatura muestra que no es así: con ese
datatime el mínimo diario cae hacia las 11:00 y el máximo hacia las 19:00 locales, y la correlación con
la temperatura del análisis IFS es máxima con un desfase de 5 h. fechaobservacón ya esta en UTC, así que
el instante real es datatime - 5 h. Se corrige aquí, en el análisis (la base de datos guarda el dato
crudo), y se valida con la correlación cruzada (correlacion_cruzada_horaria).

Los datos del IDEAM en datos.gov.co no están validados por el instituto y llegan con uno a dos días de
retraso; no hay pronóstico, así que solo pueden entrar en los modelos como información del pasado.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

# variable común: (palabras clave en la descripción del sensor, límites físicos, agregación horaria)
VARIABLES = {
    "temperature_2m": (("TEMPERATURA DEL AIRE",), (5.0, 45.0), "media"),
    "relative_humidity_2m": (("HUMEDAD DEL AIRE",), (5.0, 100.0), "media"),
    "precipitation": (("PRECIPITACION",), (0.0, 50.0), "suma"),        # por lectura de 2-10 min
    "surface_pressure": (("PRESION ATMOSFERICA",), (900.0, 1100.0), "media"),
    "wind_speed_10m": (("VELOCIDAD DEL VIENTO",), (0.0, 60.0), "media"),
}
COBERTURA_MINIMA = 0.80
LLUVIA_MAXIMA_MM_H = 150.0                          # intensidad máxima plausible (record horario de Colombia ~ 120-150 mm)
CORRECCION_HORARIA = pd.Timedelta(hours=-5)        # instante real = datatime - 5 h (ver arriba)


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).upper()


def variable_de(descripcion: str) -> str | None:
    d = _sin_tildes(descripcion)
    if "SUELO" in d:
        return None
    for v, (claves, _, _) in VARIABLES.items():
        if any(k in d for k in claves):
            return v
    return None


def catalogo(crudo: pd.DataFrame) -> pd.DataFrame:
    """Sensores disponibles por estación: variable común, unidad, distancia, lecturas y paso nativo."""
    d = crudo.assign(variable=crudo["descripcionsensor"].map(variable_de))
    g = d.groupby(["codigoestacion", "codigosensor", "descripcionsensor", "unidadmedida", "variable"], dropna=False)
    t = g.agg(distancia_km=("distanciakm", "first"), lecturas=("valorobservado", "size"),
              inicio=("datatime", "min"), fin=("datatime", "max")).reset_index()
    # Paso nativo: mediana de los intervalos positivos (las lecturas repetidas en el mismo instante dan 0)
    pasos = d.sort_values("datatime").groupby("codigosensor")["datatime"].apply(
        lambda s: s.diff()[lambda x: x > pd.Timedelta(0)].median())
    t["paso_nativo"] = t["codigosensor"].map(pasos)
    return t.sort_values(["variable", "distancia_km", "lecturas"], ascending=[True, True, False])


def horaria(crudo: pd.DataFrame, correccion: pd.Timedelta = CORRECCION_HORARIA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Series horarias por variable común (estación más cercana con datos) y tabla de lo usado.

    Devuelve (tabla horaria con una columna por variable, resumen por variable con estación, sensor,
    distancia, horas válidas e imposibles descartados)."""
    crudo = crudo.assign(datatime=crudo["datatime"] + correccion)
    cat = catalogo(crudo)
    series, resumen = {}, []
    for var, (_, (lo, hi), regla) in VARIABLES.items():
        candidatos = cat[cat["variable"] == var]
        for est, g in candidatos.groupby("codigoestacion", sort=False):
            sensor = g.sort_values("lecturas", ascending=False).iloc[0]
            s = crudo.loc[crudo["codigosensor"] == sensor["codigosensor"]].set_index("datatime")["valorobservado"].astype(float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if var == "wind_speed_10m" and "m/s" in _sin_tildes(sensor["unidadmedida"]).lower():
                s = s * 3.6                                       # m/s -> km/h, como Open-Meteo
            paso = sensor["paso_nativo"]
            if pd.isna(paso) or paso <= pd.Timedelta(0):
                paso = pd.Timedelta("10min")
            if regla == "suma":
                # La lluvia de una lectura no puede superar la intensidad máxima plausible (150 mm/h) durante el
                # paso del sensor: 5 mm en 2 minutos, 25 mm en 10 minutos
                hi = min(hi, LLUVIA_MAXIMA_MM_H * paso / pd.Timedelta("1h"))
            imposibles = int(((s < lo) | (s > hi)).sum())
            s = s.where((s >= lo) & (s <= hi))
            esperadas = max(1, int(round(pd.Timedelta("1h") / paso)))
            r = s.resample("1h", label="right", closed="right")
            h = r.sum(min_count=1) if regla == "suma" else r.mean()
            h = h.where(r.count() >= COBERTURA_MINIMA * esperadas)
            if regla == "suma":
                h = h.where(h <= LLUVIA_MAXIMA_MM_H)              # una hora con más de 150 mm es un error del sensor
            if h.notna().sum() < 24 * 30:                         # menos de un mes de horas válidas: siguiente estación
                continue
            series[var] = h.rename(var)
            resumen.append({"variable": var, "estacion": est, "sensor": sensor["descripcionsensor"],
                            "distancia_km": float(sensor["distancia_km"]), "paso_nativo": str(paso),
                            "horas_validas": int(h.notna().sum()), "imposibles_descartados": imposibles,
                            "inicio": h.first_valid_index(), "fin": h.last_valid_index()})
            break
    tabla = pd.concat(series, axis=1) if series else pd.DataFrame()
    return tabla, pd.DataFrame(resumen)


def lluvia_diaria(precipitacion_horaria: pd.Series, zona: str, minimo_horas: int = 20) -> pd.Series:
    """Lluvia diaria (día local) con al menos `minimo_horas` horas válidas."""
    p = precipitacion_horaria.copy()
    dias = (p.index - pd.Timedelta("1h")).tz_convert(zona).normalize().tz_localize(None)   # hora etiquetada al final
    s = p.groupby(dias).sum(min_count=1)
    n = p.notna().groupby(dias).sum()
    return s.where(n >= minimo_horas).rename("precipitacion_ideam")


def acuerdo_lluvia(a: pd.Series, b: pd.Series, umbral_mm: float = 1.0) -> dict:
    """Coincidencia de días con lluvia (>= umbral) entre dos fuentes diarias: aciertos, probabilidad de
    detección (POD) y falsas alarmas (FAR) de b tomando a como referencia."""
    x, y = a.align(b, join="inner")
    ok = x.notna() & y.notna()
    la, lb = (x[ok] >= umbral_mm), (y[ok] >= umbral_mm)
    tp, fn, fp = int((la & lb).sum()), int((la & ~lb).sum()), int((~la & lb).sum())
    return {"dias": int(ok.sum()), "dias_lluvia_ref": int(la.sum()), "dias_lluvia_b": int(lb.sum()),
            "acierto_pct": float(100 * (la == lb).mean()) if ok.any() else np.nan,
            "pod": tp / (tp + fn) if tp + fn else np.nan, "far": fp / (tp + fp) if tp + fp else np.nan,
            "correlacion_diaria": float(x[ok].corr(y[ok])) if ok.sum() > 2 else np.nan}


def alineacion(crudo: pd.DataFrame, referencias: dict[str, pd.Series], correcciones=(-4, -5, -6)) -> pd.DataFrame:
    """Experimento de alineación: para cada corrección horaria candidata y cada variable, rezago de la correlación
    máxima entre la estación y el modelo. La presión es la variable que decide: su marea semidiurna tiene una fase
    regular que los modelos reproducen bien, mientras que la fase del ciclo diario de la temperatura puede diferir
    entre la estación y el modelo por razones físicas (inercia del sensor, brisa, suelo). referencias: {variable:
    serie horaria instantanea del modelo}."""
    filas = []
    for h in correcciones:
        tabla = horaria(crudo, pd.Timedelta(hours=h))[0]
        for var, ref in referencias.items():
            if var not in tabla.columns or ref is None:
                continue
            cc = correlacion_cruzada_horaria(tabla[var], ref)
            filas.append({"correccion_h": h, "variable": var, "rezago_max_h": int(cc.idxmax()),
                          "correlacion_max": float(cc.max()), "correlacion_rezago_0": float(cc.get(0, np.nan))})
    return pd.DataFrame(filas)


def correccion_por_alineacion(tabla: pd.DataFrame, defecto: int = -5) -> int:
    """Correccion con la que la presión (o, sin presión, la temperatura) queda con la correlación máxima en el
    rezago 0."""
    for var in ("surface_pressure", "temperature_2m"):
        ok = tabla[(tabla["variable"] == var) & (tabla["rezago_max_h"] == 0)]
        if len(ok):
            return int(ok.sort_values("correlacion_max", ascending=False).iloc[0]["correccion_h"])
    return defecto


def correlacion_cruzada_horaria(a: pd.Series, b: pd.Series, max_rezago: int = 8) -> pd.Series:
    """corr(a_t, b_{t-k}) para k = -max..max horas: el máximo en k = 0 indica que las dos series están
    alineadas en el tiempo."""
    return pd.Series({k: a.corr(b.shift(k, freq="h").reindex(a.index)) for k in range(-max_rezago, max_rezago + 1)},
                     name="correlacion")
