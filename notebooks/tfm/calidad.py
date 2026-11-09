"""
calidad.py
==========
Controles de calidad de la telemetría cruda ANTES de imputar o agregar. Nada se corrige en la BD:
cada corrección queda en el análisis, documentada y reproducible.

1. corregir_hora(): en algunos periodos SolaX registro las lecturas con la zona horaria de China
   (UTC+8) en lugar de la de Colombia. Se detecta FILA A FILA con la diferencia entre plantLocalTime
   (reloj del inversor) y dataTime (UTC):
       -5 h  -> correcto (Colombia)
       -4 h  -> plantas configuradas como "Indiana (East)": dataTime es correcto
       +8 h  -> dataTime adelantado 11 h respecto al día solar: hora real = dataTime + 13 h
       otro  -> zona del inversor desconocida (p. ej. -12 h en P24I0101 en octubre de 2026): la fila se descarta
                y se cuenta, porque no hay una corrección validada; queda como hueco.
   Validación: tras corregir, el centroide de la producción cae a las 12:03 hora local (antes, 23:03).
2. limites_fisicos(): potencia negativa o mayor que 1,15 veces la nominal, temperatura fuera de rango.
3. regimenes_capacidad(): detecta cambios de capacidad efectiva (ampliaciones o limitaciones) con el
   percentil 95 de la potencia máxima diaria relativa a la nominal, mes a mes, y afina por días el primer
   día de cada régimen dentro de su primer mes (inicio_diario_regimen).
4. outliers_iqr(): valores atípicos por hora del día con el criterio IQR (k = 1,5 y 3), como en Minería
   de Datos; se marcan, no se borran (en energía solar un valor alto puede ser real: realce por nubes).
5. limites_irradiancia(): límites físicamente posibles de la BSRN (Long y Shi, 2008) para GHI, DHI y DNI.
   Un valor fuera de ellos es un error de la fuente (no un extremo real) y se trata como faltante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from . import config, solar

DESFASE_CHINA_H = 8
CORRECCION_CHINA = pd.Timedelta(hours=13)
DESFASES_CORRECTOS_H = (-5, -4)                  # Colombia e "Indiana (East)": dataTime correcto


def corregir_hora(crudo: pd.DataFrame) -> pd.DataFrame:
    """Anade 'desfase_h' y corrige 'datatime' en las filas registradas con la zona horaria de China.

    Conserva el instante original en 'datatime_api'. Si tras corregir dos filas caen en el mismo
    instante, se conserva la que no necesito corrección (dato de origen sin ambiguedad). Las filas con un
    desfase sin corrección validada se descartan; su número queda en d.attrs["filas_desfase_desconocido"]."""
    d = crudo.copy()
    d["datatime_api"] = d["datatime"]
    d.attrs["filas_desfase_desconocido"] = 0
    if "plantlocaltime" not in d.columns:
        d["desfase_h"] = np.nan
        return d
    local = pd.to_datetime(d["plantlocaltime"])
    utc_ingenuo = d["datatime"].dt.tz_convert("UTC").dt.tz_localize(None)
    d["desfase_h"] = ((local - utc_ingenuo).dt.total_seconds() / 3600).round()
    desconocido = d["desfase_h"].notna() & ~d["desfase_h"].isin([*DESFASES_CORRECTOS_H, DESFASE_CHINA_H])
    n_desconocido = int(desconocido.sum())
    d = d.loc[~desconocido].copy()
    china = d["desfase_h"] == DESFASE_CHINA_H
    d.loc[china, "datatime"] = d.loc[china, "datatime"] + CORRECCION_CHINA
    d["corregida"] = china
    d = d.sort_values(["datatime", "corregida"]).drop_duplicates("datatime", keep="first")
    d = d.sort_values("datatime").reset_index(drop=True)
    d.attrs["filas_desfase_desconocido"] = n_desconocido
    return d


def resumen_correccion_hora(d: pd.DataFrame) -> pd.DataFrame:
    """Filas y rango temporal por valor de desfase (para el informe)."""
    return (d.groupby("desfase_h", dropna=False)
             .agg(filas=("datatime", "size"), desde=("datatime", "min"), hasta=("datatime", "max"))
             .reset_index())


def limites_fisicos(df: pd.DataFrame, potencia_nominal_w: float) -> pd.DataFrame:
    """Marca (True = fuera de rango) valores físicamente imposibles. No los borra."""
    m = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith("acpower") or c in ("mppttotalinputpower", "totalactivepower"):
            m[c] = (df[c] < -50) | (df[c] > 1.15 * potencia_nominal_w * (1 if c != "mppttotalinputpower" else 1.5))
        elif c == "invertertemperature":
            m[c] = (df[c] < -5) | (df[c] > 95)
        elif c.startswith("acvoltage"):
            # Red de baja tensión en Colombia: 127 V fase-neutro (220 V entre fases)
            m[c] = (df[c] < 0) | (df[c] > 300)
        elif c.startswith("accurrent"):
            # Corriente por fase con la potencia nominal a 120 V, con un margen del 50 %
            m[c] = (df[c] < 0) | (df[c] > 1.5 * potencia_nominal_w / (3 * 120))
    return m


_TIPO_IRRADIANCIA = {
    "shortwave_radiation": "ghi", "ghi": "ghi", "allsky_sfc_sw_dwn": "ghi",
    "diffuse_radiation": "dhi", "dhi": "dhi", "allsky_sfc_sw_diff": "dhi",
    "direct_normal_irradiance": "dni", "dni": "dni", "allsky_sfc_sw_dni": "dni",
    "direct_radiation": "bh",
}


def limites_irradiancia(df: pd.DataFrame, lugar, desplazamiento: pd.Timedelta | None = None) -> pd.DataFrame:
    """True donde la irradiancia esta fuera de los límites físicamente posibles de la BSRN:

        GHI: -4 < GHI < 1,5 S0 cos(z)^1,2 + 100      DHI: -4 < DHI < 0,95 S0 cos(z)^1,2 + 50
        DNI: -4 < DNI < S0                            directa horizontal: < S0 cos(z) + 50

    S0 es la irradiancia extraterrestre del día y z el ángulo cenital (en el centro del intervalo si los
    valores son medias de periodo: desplazamiento = paso / 2)."""
    pos = solar.posicion_solar(df.index, lugar, desplazamiento)
    t = df.index - desplazamiento if desplazamiento is not None else df.index
    s0 = pd.Series(pvlib.irradiance.get_extra_radiation(t).to_numpy(), index=df.index)
    mu0 = np.cos(np.radians(pos["zenith"])).clip(lower=0)
    tope = {"ghi": 1.5 * s0 * mu0 ** 1.2 + 100, "dhi": 0.95 * s0 * mu0 ** 1.2 + 50, "dni": s0, "bh": s0 * mu0 + 50}
    m = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in df.columns:
        tipo = _TIPO_IRRADIANCIA.get(c)
        if tipo:
            m[c] = (df[c] < -4) | (df[c] > tope[tipo])
    return m


def regimenes_capacidad(p_ac: pd.Series, potencia_nominal_w: float, umbral: float = 0.20,
                        meses_minimos: int = 2) -> pd.DataFrame:
    """Percentil 95 mensual de la potencia máxima diaria / nominal y segmentos estables.

    Un cambio sostenido (>= meses_minimos) de más de `umbral` (fracción, relativa a la mediana del
    segmento en curso) abre un segmento nuevo: ampliación de paneles, limitación de inyección o falla
    prolongada. El umbral es relativo y no absoluto: pasar de 0,72 a 0,92 de la nominal (+28 %) es un
    cambio de capacidad aunque la diferencia absoluta sea de solo 0,20. En los meses estables de la flota
    la variación por el tiempo meteorológico no pasa del 16 %.
    """
    local = p_ac.copy()
    local.index = local.index.tz_convert(config.ZONA)
    pmax_dia = local.resample("D").max()
    mensual = pmax_dia.resample("MS").quantile(0.95) / potencia_nominal_w
    mensual = mensual.dropna()
    segmento, actual, valores = [], 0, []
    for i, (mes, r) in enumerate(mensual.items()):
        med = np.median(valores) if valores else np.nan
        if valores and abs(r - med) / med > umbral:
            siguientes = mensual.iloc[i:i + meses_minimos]
            if len(siguientes) >= meses_minimos and (abs(siguientes - med) / med > umbral).all():
                actual += 1
                valores = []
        valores.append(r)
        segmento.append(actual)
    salida = pd.DataFrame({"p95_pmax_relativa": mensual, "regimen": segmento})
    # Primer día de cada régimen: el mes lo decide la regla mensual; el día, la envolvente semanal de la potencia
    # máxima diaria (niveles: su mediana en el régimen anterior y en el nuevo sin su primer mes)
    pmax_rel = pmax_dia / potencia_nominal_w
    env = envolvente_semanal(pmax_rel)
    mes_de = env.index.normalize() - pd.to_timedelta(env.index.day - 1, unit="D")
    inicio = {}
    for k, g in salida.groupby("regimen"):
        mes0 = g.index.min()
        if k == 0:
            inicio[k] = mes0
            continue
        nivel_ant = env[np.isin(mes_de, salida.index[salida["regimen"] == k - 1])].median()
        nivel_nuevo = env[np.isin(mes_de, g.index[1:] if len(g) > 1 else g.index)].median()
        inicio[k] = inicio_diario_regimen(pmax_rel, mes0, nivel_ant, nivel_nuevo)
    salida["inicio"] = salida["regimen"].map(inicio)
    return salida


def envolvente_semanal(pmax_rel: pd.Series, ventana: int = 7, atras: bool = True) -> pd.Series:
    """Maximo de la potencia máxima diaria en `ventana` días (los anteriores al día si atras, los siguientes si no).
    Casi toda semana tiene un día despejado, así que la envolvente sigue la capacidad y no las nubes."""
    if atras:
        return pmax_rel.rolling(ventana, min_periods=1).max()
    return pmax_rel[::-1].rolling(ventana, min_periods=1).max()[::-1]


def inicio_diario_regimen(pmax_rel: pd.Series, mes: pd.Timestamp, nivel_ant: float, nivel_nuevo: float,
                          ventana: int = 7, dias_despues: int = 30) -> pd.Timestamp:
    """Primer día de un régimen de capacidad dentro de su primer mes.

    La regla mensual (percentil 95 de la potencia máxima diaria) decide en que mes empieza el régimen, pero el
    cambio suele caer a mitad de mes: si el régimen se tomara desde el día 1, entrarian en el hasta unas
    cuatro semanas del anterior (por ejemplo, la ampliación de P04 entro hacia el 20 de septiembre de 2025).
    El día se ubica con la envolvente semanal de la potencia máxima diaria relativa a la nominal (pmax_rel, días
    locales): el máximo de los 7 días anteriores si la capacidad sube (salta el primer día despejado del nivel
    nuevo) y de los 7 siguientes si baja (cae el primer día del nivel nuevo). Es el punto de cambio de mínima
    desviación absoluta: el día que separa mejor los días cercanos al nivel anterior de los cercanos al nuevo,
    desde el primer día del mes hasta `dias_despues` días después de su fin. Solo se busca dentro del primer mes,
    de modo que el inicio nunca se adelanta al mes que fijo la regla mensual."""
    if not (np.isfinite(nivel_ant) and np.isfinite(nivel_nuevo)) or nivel_ant == nivel_nuevo:
        return mes
    fin_mes = mes + pd.offsets.MonthEnd(0)
    sube = nivel_nuevo > nivel_ant
    tramo = pmax_rel.loc[mes - pd.Timedelta(days=ventana - 1):fin_mes + pd.Timedelta(days=dias_despues + ventana - 1)]
    env = envolvente_semanal(tramo, ventana, atras=sube).loc[mes:fin_mes + pd.Timedelta(days=dias_despues)].dropna()
    if env.empty:
        return mes
    # Costo de empezar el régimen en el día i: desviación de los días previos al nivel anterior y de los demás al nuevo.
    # Salvo una constante, es la suma previa de s = |env - anterior| - |env - nuevo| (negativa cerca del anterior)
    s = (env - nivel_ant).abs().to_numpy() - (env - nivel_nuevo).abs().to_numpy()
    previa = np.concatenate([[0.0], np.cumsum(s)])
    candidatos = [i for i, d in enumerate(env.index) if d <= fin_mes]
    if not candidatos:
        return mes
    return env.index[min(candidatos, key=lambda i: previa[i])]


def inicio_por_etapa(p_ac: pd.Series, potencia_nominal_w: float, desde, fraccion_anterior: float) -> pd.Timestamp | None:
    """Primer día en servicio de una etapa de capacidad registrada (telemetry.plantas_fv_etapas), afinado por días.

    La tabla registra el mes de una ampliación y los paneles antes y después. Cuando la ampliación cae en el primer
    mes del régimen vigente (P04: la limitación termino el 1 de septiembre de 2025 y los paneles nuevos entraron
    hacia el día 20), la regla mensual no ve ningún salto y el régimen empezaria con días de la etapa anterior. El
    día se busca con inicio_diario_regimen dentro del mes de la etapa: el nivel nuevo es la mediana de la envolvente
    semanal de la potencia máxima diaria en los meses siguientes, y el anterior, ese nivel por la fracción de
    paneles (antes / después). None si no hay datos después del mes de la etapa."""
    local = p_ac.copy()
    local.index = local.index.tz_convert(config.ZONA)
    pmax_rel = local.resample("D").max() / potencia_nominal_w
    t = pd.Timestamp(desde)
    mes = (t.tz_localize(config.ZONA) if t.tzinfo is None else t.tz_convert(config.ZONA)).normalize().replace(day=1)
    despues = envolvente_semanal(pmax_rel).loc[mes + pd.offsets.MonthBegin(1):].dropna()
    if despues.empty or not np.isfinite(fraccion_anterior):
        return None
    nivel = float(despues.median())
    return inicio_diario_regimen(pmax_rel, mes, fraccion_anterior * nivel, nivel)


def con_etapa(periodo: tuple | None, inicio_etapa) -> tuple | None:
    """El periodo (inicio, mes final) sin los días anteriores a la entrada en servicio de una etapa registrada
    (inicio_por_etapa). None si la etapa empieza después del fin del periodo."""
    if periodo is None or inicio_etapa is None or pd.isna(inicio_etapa):
        return periodo
    ini = max(_fecha_local(periodo[0]), _fecha_local(inicio_etapa).normalize())
    fin_mes = _fecha_local(periodo[1]).normalize() + pd.offsets.MonthEnd(0)
    return None if ini > fin_mes else (ini, periodo[1])


def ultimo_regimen_estable(regimenes: pd.DataFrame, meses_minimos: int = 4) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Inicio (día) y fin (mes) del último régimen con al menos `meses_minimos` meses.

    El inicio es el primer día del régimen afinado por días (columna inicio de regimenes_capacidad) o, sin esa
    columna, el primer día de su primer mes. Historia corta: si la unidad tiene menos de `meses_minimos` meses
    en total y no se detecta ningún salto (un solo régimen), todo el registro se toma como un régimen. Con dos
    o más regímenes y ninguno largo, no hay régimen estable."""
    if regimenes.empty:
        return None
    for reg in sorted(regimenes["regimen"].unique(), reverse=True):
        meses = regimenes.index[regimenes["regimen"] == reg]
        if len(meses) >= meses_minimos:
            ini = regimenes.loc[meses, "inicio"].iloc[0] if "inicio" in regimenes else meses.min()
            return (meses.min() if pd.isna(ini) else ini), meses.max()
    if regimenes["regimen"].nunique() == 1:
        return regimenes.index.min(), regimenes.index.max()
    return None


POA_MINIMA_DIA = 50.0              # W/m2 en el plano: un sistema sano produce alrededor del 4 % de su kWp
POA_MINIMA_HORA = 200.0            # W/m2: del orden del 16 % del kWp; las primeras y últimas horas del día quedan fuera
FRACCION_SIN_PRODUCCION = 0.01     # energía horaria menor que el 1 % del kWp: el sistema no produjo


def sin_produccion(oh: pd.DataFrame, kwp: float, poa_minima_dia: float = POA_MINIMA_DIA,
                   poa_minima_hora: float = POA_MINIMA_HORA, fraccion: float = FRACCION_SIN_PRODUCCION) -> pd.Series:
    """Horas en que el sistema no produjo aunque había sol y el inversor no informaba falla: indisponibilidad no
    declarada (inversor apagado o en espera, planta desconectada, o limitada a cero sin consumo en el predio). Ninguna
    variable meteorológica la explica, igual que una falla.

    Dos casos, siempre sin estado de falla y con energía menor que `fraccion` del kWp (se usa el kWp y no la potencia
    nominal del inversor, porque en una planta con el inversor sobredimensionado el 1 % de la nominal no es poco):
    * Día sin servicio: todas sus horas con al menos `poa_minima_dia` W/m2 en el plano (al menos 4) están así, mediodía
      incluido; quedan marcadas todas sus horas de sol.
    * Hora suelta: una hora con al menos `poa_minima_hora` W/m2. El umbral es más alto porque en las primeras y últimas
      horas del día un cero puede ser sistemático (el inversor aun no arranca o una sombra lo tapa todas las tardes):
      eso es comportamiento normal que el modelo debe aprender, no una indisponibilidad.
    oh: objetivos horarios (e_kwh, g_poa, yr_cs y, si existe, frac_falla), índice UTC con la etiqueta al final de la hora."""
    e = oh["e_kwh"]
    g = oh["g_poa"].to_numpy(dtype=float)
    hay = e.notna().to_numpy()
    sin_falla = (oh["frac_falla"].fillna(0) == 0).to_numpy() if "frac_falla" in oh else np.ones(len(oh), dtype=bool)
    nula = hay & sin_falla & (e.to_numpy(dtype=float) < fraccion * kwp)
    sol_dia = hay & (g >= poa_minima_dia)
    dia = pd.DatetimeIndex((oh.index - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).tz_localize(None).normalize())
    cuenta = pd.DataFrame({"sol": sol_dia, "nula": nula & sol_dia}, index=dia).groupby(level=0).sum()
    muertos = cuenta.index[(cuenta["sol"] >= 4) & (cuenta["nula"] == cuenta["sol"])]
    de_dia = (oh["yr_cs"] > 0.01).to_numpy() if "yr_cs" in oh else np.ones(len(oh), dtype=bool)
    suelta = nula & (g >= poa_minima_hora)
    return pd.Series(suelta | (np.isin(dia, muertos) & de_dia), index=oh.index, name="sin_produccion")


def _fecha_local(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert(config.ZONA).tz_localize(None) if t.tz is not None else t


def mascara_regimen(indice: pd.DatetimeIndex, periodo: tuple | None) -> np.ndarray:
    """True para los instantes o días dentro del periodo (mes inicial, mes final) de un régimen.

    Con un índice UTC de intervalos etiquetados al final se usa el día local del intervalo (t - 1 s);
    con un índice de fechas locales sin zona, la fecha misma."""
    if periodo is None:
        return np.zeros(len(indice), dtype=bool)
    ini = _fecha_local(periodo[0]).normalize()
    fin = _fecha_local(periodo[1]).normalize() + pd.offsets.MonthEnd(0)
    idx = pd.DatetimeIndex(indice)
    dias = (idx - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).tz_localize(None).normalize() \
        if idx.tz is not None else idx.normalize()
    return np.asarray((dias >= ini) & (dias <= fin))


def outliers_iqr(serie: pd.Series, k: float = 1.5, por_hora: bool = True) -> pd.Series:
    """True donde el valor cae fuera de [Q1 - k IQR, Q3 + k IQR] (por hora local del día si por_hora)."""
    s = serie.dropna()
    if por_hora:
        h = s.index.tz_convert(config.ZONA).hour
        q1 = s.groupby(h).transform(lambda x: x.quantile(0.25))
        q3 = s.groupby(h).transform(lambda x: x.quantile(0.75))
    else:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    fuera = (s < q1 - k * iqr) | (s > q3 + k * iqr)
    return fuera.reindex(serie.index).fillna(False)
