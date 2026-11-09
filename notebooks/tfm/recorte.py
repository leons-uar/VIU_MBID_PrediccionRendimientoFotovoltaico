"""
recorte.py
==========
Limitación de inyección (inyección cero) y régimen vigente de cada planta (experimento E28).

Con el control de exportación activo, el inversor maestro mide con su sensor de red (la pinza amperimétrica o el
medidor conectado a el; SolaX lo publica como gridpower de ese inversor) la potencia que la planta intercambia con
la red. Si la producción fotovoltaica supera al consumo del predio, el inversor la reduce hasta que la red queda
en cero: la producción sigue al consumo y no al recurso solar. Esas horas no sirven para aprender ni para evaluar
un modelo meteorológico del rendimiento, igual que las horas en falla.

Marcas (siempre con la hora ya corregida por calidad.corregir_hora):

1. Sensor de red. Una lectura de 5 min con producción (potencia AC de la planta mayor que el 5 % de su
   percentil 99 de 9 a 15 h) esta "en cero" si |gridpower| <= 2 % de ese percentil y no es un cero exacto. Un
   inversor cuyo gridpower es cero exacto en casi todas las lecturas no tiene sensor en la plataforma.
   * Día limitado: la mediana móvil de 7 días de la fracción diaria en cero (8 a 15 h) es de 0,5 o más. Define los
     periodos (el régimen B empieza después del último día limitado).
   * Hora recortada: al menos la mitad de las lecturas de la hora en cero, en cualquier día. Con el control activo, el
     inversor solo recorta cuando la producción supera al consumo, así que fuera de los días limitados quedan horas
     sueltas recortadas (limitación parcial: un 1,2 % de las horas de sol del régimen B de la flota, hasta el 9 % en P20).
   La marca de la planta vale para todos sus inversores, porque el sensor mide el intercambio de toda la planta.

2. Punto de operación, para las plantas sin sensor y como confirmación. Para reducir la potencia, el inversor se
   aleja del punto de máxima potencia hacia la tensión de circuito abierto: con la red en cero, la potencia AC
   mediana cae al 11 a 35 % de su percentil 99 y la tensión MPPT sube de un 10 a un 15 %. Por hora se calcula el
   cociente entre la tensión de operación (media de las entradas MPPT ponderada por su potencia) y su
   referencia (mediana con potencia de al menos el 80 % del percentil 99), y la potencia relativa. La regla
   (tensión alta y potencia baja en una hora despejada) se calibra contra la marca del sensor solo con datos de
   entrenamiento, y se informan su precisión y su exhaustividad (calibrar_tension).

Régimen vigente (régimen B): empieza después del último día limitado, siempre que los últimos días antes del corte
no esten limitados; si lo están, la planta queda "limitada al corte" y fuera del régimen B. Como la mediana
centrada se retrasa hasta tres días en la transición, el primer día es el primero posterior al último limitado cuya
propia fracción en cero (con al menos 20 lecturas con producción) es menor que 0,25; los días intermedios, aun
limitados o estrangulados hasta casi no producir, quedan fuera (E42). Se combina con el último régimen de capacidad
(calidad.regimenes_capacidad) en periodo_b.

Lo declarado por el instalador (telemetry.plantas_eventos) manda sobre lo estimado: una planta que sigue en inyección
cero queda limitada al corte aunque el sensor no lo vea (P20 y P23: el consumo del predio absorbe casi toda la
producción), una fecha de fin confirmada fija el inicio de B (P10) y B no empieza antes de que termine una
reparación (P04, reemplazo de 20 paneles). En las plantas sin sensor, la regla de tensión deja de marcar horas
después de la conformidad del operador de red si el instalador confirma que ya no hay límite de exportación (P01).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calidad, config, datos
from . import objetivos as obj

FRACCION_RED_CERO = 0.02          # |gridpower| <= 2 % del p99 de la potencia AC de la planta
FRACCION_PRODUCCION = 0.05        # lecturas con producción: potencia AC > 5 % del p99
UMBRAL_DIA_LIMITADO = 0.5         # mediana móvil de 7 días de la fracción diaria en cero
FRACCION_DIA_LIBRE = 0.25         # primer día del régimen B: su propia fracción en cero, menor que la mitad del umbral
UMBRAL_HORA_RECORTADA = 0.5       # fracción de lecturas en cero de la hora (en cualquier día)
VENTANA_DIAS = 7
HORAS_DIA = (8, 15)               # horas locales para la fracción diaria (incluidas)
LECTURAS_MINIMAS_DIA = 20
DIAS_LIBRES_MINIMOS = 28          # el régimen B necesita al menos 4 semanas sin limitación
DIAS_COLA_CORTE = 7               # si alguno de los últimos 7 días antes del corte esta limitado: limitada al corte
CERO_EXACTO_SIN_SENSOR = 0.9      # fracción de ceros exactos de gridpower a partir de la cual no hay sensor
# Regla de tensión para plantas sin sensor, calibrada con calibrar_tension contra la marca del sensor en el
# entrenamiento de las 21 plantas con sensor (cuaderno 01_calidad_regimenes_flota, E28): F1 de 0,70 en la calibración
# y de 0,61 fuera de ella (validación y prueba). Comparar la potencia con la envolvente de la misma hora y mes dio
# F1 de 0,62 y 0,52: en las plantas limitadas la envolvente también queda deprimida por el recorte.
PARAMETROS_TENSION = {"delta": 0.06, "kappa": 0.8, "claridad_minima": 0.5}
# Horas en que se aplica la regla (hora local de inicio del intervalo, la de 9 a la de 14, es decir, de 9:00 a 15:00),
# las mismas con que se calibra. Fuera de ellas el módulo frío de la mañana y el sol bajo dan la misma firma sin
# recorte: frente al sensor, la precisión de la regla es de 0,13 a las 6 h, 0,35 a las 7 h y 0,53 a las 8 h, y de
# 0,51 y 0,35 a las 15 y 16 h, frente a 0,65 a 0,73 de 9 a 14 h (todas las plantas con sensor hasta el corte). Aplicarla
# todo el día marcaba como recortadas las mañanas despejadas de la planta sin sensor y dejaba incompletos sus días.
HORAS_TENSION = (9, 15)

_LECTURAS: dict[tuple, tuple] = {}   # marcas de cada planta ya calculadas en este proceso (sus inversores las comparten)


def tension_operacion(d: pd.DataFrame) -> pd.Series:
    """Tension de operación (V) del generador: media de las entradas MPPT ponderada por su potencia (o de las
    entradas pv si el inversor no informa las MPPT). Sin potencia en ninguna entrada, NaN."""
    for prefijo in ("mppt", "pv"):
        pares = [(f"{prefijo}{k}voltage", f"{prefijo}{k}power") for k in range(1, 33)
                 if f"{prefijo}{k}voltage" in d.columns and f"{prefijo}{k}power" in d.columns]
        if not pares:
            continue
        v = d[[a for a, _ in pares]].to_numpy(dtype=float)
        p = d[[b for _, b in pares]].to_numpy(dtype=float)
        p = np.where((p > 0) & (v > 0), p, 0.0)
        suma = p.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vm = np.where(suma > 0, (np.nan_to_num(v) * p).sum(axis=1) / suma, np.nan)
        if np.isfinite(vm).any():
            return pd.Series(vm, index=d.index, name="v_op")
    return pd.Series(np.nan, index=d.index, name="v_op")


def lecturas_planta(grupo: str, plantuuid: str, inicio, fin, refrescar: bool = False) -> dict:
    """Lecturas de 5 min de los inversores de una planta, con la hora corregida.

    Devuelve {"ac": DataFrame (W) por inversor, "v": DataFrame (V) por inversor, "red": Series (W) del inversor
    con sensor o None, "con_sensor": código del inversor con sensor o None, "estado": DataFrame del devicestatus,
    "desfase_desconocido": filas descartadas}. El índice es UTC, redondeado a 5 min."""
    from . import nomenclatura as nom
    disp = datos.constantes(grupo)["dispositivos"]
    invs = disp[(disp["plantuuid"].astype(str) == str(plantuuid)) & (disp["devicetype"] == 1)]
    nombres = {f"inv_{u[:8]}": str(u) for u in invs["deviceuuid"].astype(str)}
    series = [s for s in datos.series_localidad(grupo, meteo=False, ideam=False) if s.nombre in nombres]
    crudos = datos.leer(series, inicio, fin, refrescar=refrescar)
    ac, v, red, estado, descartadas = {}, {}, {}, {}, 0
    for nombre, crudo in crudos.items():
        if crudo is None or crudo.empty:
            continue
        codigo = nom.inversor(nombres[nombre])
        d = calidad.corregir_hora(crudo)
        descartadas += d.attrs.get("filas_desfase_desconocido", 0)
        d = d.set_index(d["datatime"].dt.floor("5min"))
        d = d[~d.index.duplicated(keep="first")]
        ac[codigo] = obj.potencia_ac(d)
        v[codigo] = tension_operacion(d)
        if "devicestatus" in d.columns:
            estado[codigo] = d["devicestatus"]
        if "gridpower" in d.columns:
            red[codigo] = d["gridpower"]
    # El inversor con sensor: el de menos ceros exactos en gridpower (si alguno tiene menos del 90 %)
    con_sensor, mejor = None, CERO_EXACTO_SIN_SENSOR
    for codigo, s in red.items():
        s = s.dropna()
        if len(s) and (s == 0).mean() < mejor:
            con_sensor, mejor = codigo, float((s == 0).mean())
    return {"ac": pd.DataFrame(ac).sort_index(), "v": pd.DataFrame(v).sort_index(),
            "red": red[con_sensor].sort_index() if con_sensor else None, "con_sensor": con_sensor,
            "estado": pd.DataFrame(estado).sort_index(), "desfase_desconocido": descartadas}


def _hora_local(indice: pd.DatetimeIndex) -> pd.Index:
    return indice.tz_convert(config.ZONA).hour


def _dia_local(indice: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dia local de cada lectura de 5 min (la lectura etiqueta el final de su intervalo)."""
    return (indice - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).tz_localize(None).normalize()


def marcas_sensor(lect: dict) -> pd.DataFrame:
    """Por lectura de 5 min: producción de la planta (W), si produce y si la red esta en cero (1/0; NaN sin
    producción o sin sensor)."""
    ac = lect["ac"]
    if ac.empty:
        return pd.DataFrame(columns=["ac", "produce", "en_cero"])
    ac_planta = ac.sum(axis=1, min_count=1)
    horas = _hora_local(ac_planta.index)
    p99 = ac_planta[(horas >= 9) & (horas <= 14)].quantile(0.99)
    produce = ac_planta > FRACCION_PRODUCCION * p99
    salida = pd.DataFrame({"ac": ac_planta, "produce": produce})
    if lect["red"] is None:
        salida["en_cero"] = np.nan
        return salida
    red = lect["red"].reindex(salida.index)
    en_cero = ((red != 0) & (red.abs() <= FRACCION_RED_CERO * p99)).astype(float)
    salida["en_cero"] = en_cero.where(produce & red.notna())
    salida.attrs["p99_w"] = float(p99)
    return salida


def diario(marcas: pd.DataFrame) -> pd.DataFrame:
    """Fraccion diaria de lecturas en cero (8 a 15 h, días con al menos 20 lecturas con producción), su mediana
    móvil centrada de 7 días y el día limitado."""
    m = marcas.dropna(subset=["en_cero"])
    h = _hora_local(m.index)
    m = m[(h >= HORAS_DIA[0]) & (h <= HORAS_DIA[1])]
    g = m.groupby(_dia_local(m.index))["en_cero"]
    d = pd.DataFrame({"lecturas": g.size(), "fraccion_cero": g.mean()})
    d = d[d["lecturas"] >= LECTURAS_MINIMAS_DIA]
    if d.empty:
        return d.assign(mediana_7d=pd.Series(dtype=float), limitado=pd.Series(dtype=bool))
    completo = d["fraccion_cero"].asfreq("D")
    mediana = completo.rolling(VENTANA_DIAS, center=True, min_periods=3).median()
    d["mediana_7d"] = mediana.reindex(d.index)
    d["limitado"] = d["mediana_7d"] >= UMBRAL_DIA_LIMITADO
    return d


def horario(marcas: pd.DataFrame, dias: pd.DataFrame) -> pd.DataFrame:
    """Por hora (UTC, etiqueta al final como en agregacion.agregar): fracción de lecturas en cero, si el día esta
    limitado y si la hora esta recortada (al menos la mitad de sus lecturas en cero, en cualquier día: la limitación
    parcial fuera de los días limitados también recorta)."""
    r = marcas["en_cero"].resample("1h", closed="right", label="right")
    h = pd.DataFrame({"frac_cero": r.mean(), "lecturas_cero": r.count()})
    limitado = dias["limitado"] if "limitado" in dias else pd.Series(dtype=bool)
    dia_h = _dia_local(h.index)
    h["dia_limitado"] = pd.Series(limitado.reindex(dia_h).to_numpy(), index=h.index).fillna(False).astype(bool)
    h["recortada"] = (h["frac_cero"] >= UMBRAL_HORA_RECORTADA).fillna(False)
    return h


def tension_horaria(lect: dict) -> pd.DataFrame:
    """Por hora e inversor: cociente de tensión (tensión de operación / referencia) y potencia relativa al p99.
    Columnas con MultiIndex (variable, inversor)."""
    ac, v = lect["ac"], lect["v"]
    filas = {}
    for codigo in ac.columns:
        p = ac[codigo]
        horas = _hora_local(p.index)
        p99 = p[(horas >= 9) & (horas <= 14)].quantile(0.99)
        if not np.isfinite(p99) or p99 <= 0 or codigo not in v:
            continue
        vv = v[codigo].reindex(p.index)
        ref = vv[(p >= 0.8 * p99) & (horas >= 10) & (horas <= 14)].median()
        if not np.isfinite(ref) or ref <= 0:
            continue
        cociente = (vv / ref).where(p > FRACCION_PRODUCCION * p99)
        relativa = (p / p99).where(p > FRACCION_PRODUCCION * p99)
        rv = cociente.resample("1h", closed="right", label="right").median()
        rp = relativa.resample("1h", closed="right", label="right").median()
        filas[("v_rel", codigo)] = rv
        filas[("p_rel", codigo)] = rp
    return pd.DataFrame(filas)


def en_ventana_tension(indice: pd.DatetimeIndex, horas: tuple[int, int] = HORAS_TENSION) -> np.ndarray:
    """True en las horas (índice horario con etiqueta al final, como el de tension_horaria) cuyo intervalo empieza
    entre las horas locales horas[0] y horas[1] (sin incluirla)."""
    inicio = (pd.DatetimeIndex(indice) - pd.Timedelta("30min")).tz_convert(config.ZONA).hour
    return np.asarray((inicio >= horas[0]) & (inicio < horas[1]))


def regla_tension(v_rel: pd.Series, p_rel: pd.Series, claridad: pd.Series, delta: float = 0.08,
                  kappa: float = 0.5, claridad_minima: float = 0.6, horas: tuple[int, int] | None = HORAS_TENSION) -> pd.Series:
    """Hora recortada según el punto de operación: tensión al menos (1 + delta) veces su referencia, potencia por
    debajo de kappa veces su p99, hora despejada (G_POA / G_POA de cielo despejado >= claridad_minima) y dentro de
    la ventana horaria en que la regla es fiable (HORAS_TENSION; fuera de ella nunca marca). horas=None no aplica la
    ventana: solo para una muestra que ya viene restringida a ella (la calibración)."""
    idx = v_rel.index.union(p_rel.index)
    c = claridad.reindex(idx)
    marca = (v_rel.reindex(idx) >= 1 + delta) & (p_rel.reindex(idx) < kappa) & (c >= claridad_minima)
    if horas is not None:
        marca = marca & en_ventana_tension(idx, horas)
    return marca.rename("recortada_tension")


def calibrar_tension(v_rel: pd.Series, p_rel: pd.Series, claridad: pd.Series, referencia: pd.Series,
                     deltas=(0.04, 0.06, 0.08, 0.10, 0.12), kappas=(0.4, 0.5, 0.6, 0.7, 0.8),
                     claridades=(0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Precision, exhaustividad y F1 de la regla de tensión frente a la marca del sensor (referencia, booleana)
    en las horas con ambas marcas, para cada combinación de parámetros. Se ordena por F1. La muestra debe venir
    restringida a la ventana de la regla (en_ventana_tension): aquí la ventana no se vuelve a aplicar."""
    filas = []
    base = referencia.dropna().astype(bool)
    for d_ in deltas:
        for k_ in kappas:
            for c_ in claridades:
                pred = regla_tension(v_rel, p_rel, claridad, d_, k_, c_, horas=None).reindex(base.index)
                validas = pred.notna()
                y, yhat = base[validas], pred[validas].astype(bool)
                vp = int((y & yhat).sum())
                precision = vp / max(int(yhat.sum()), 1)
                exhaustividad = vp / max(int(y.sum()), 1)
                f1 = 2 * precision * exhaustividad / (precision + exhaustividad) if precision + exhaustividad else 0.0
                filas.append({"delta": d_, "kappa": k_, "claridad_minima": c_, "horas": int(validas.sum()),
                              "recortadas_sensor": int(y.sum()), "recortadas_regla": int(yhat.sum()),
                              "precision": precision, "exhaustividad": exhaustividad, "f1": f1})
    return pd.DataFrame(filas).sort_values("f1", ascending=False).reset_index(drop=True)


def claridad_grupo(grupo: str, inicio, fin, refrescar: bool = False) -> pd.Series:
    """Indice de claridad horario del grupo, GHI / GHI de cielo despejado (Ineichen), con la fuente de detección de
    la auditoría del P0: el satélite MSG si cubre más del 90 % de las horas y, si no (al oeste de unos 75,2° O),
    el análisis IFS. Sirve para aplicar la regla de tensión solo en horas despejadas."""
    from . import caracteristicas as caract, solar
    const = datos.constantes(grupo)
    punto = const["grupo"].iloc[0]
    lugar = solar.ubicacion(punto.latitude, punto.longitude, punto.elevacion)
    series = {"satelite_msg": "om_satellite__eumetsat_lsa_saf_msg", "ifs_analisis": "om_archive__ecmwf_ifs"}
    disponibles = [s for s in datos.series_localidad(grupo, meteo=True, ideam=False) if s.nombre in series.values()]
    crudos = datos.leer(disponibles, inicio, fin, refrescar=refrescar)
    for nombre in ("satelite_msg", "ifs_analisis"):
        irr = caract.irradiancia_fuente(crudos, nombre, lugar)
        if irr is None or irr.empty:
            continue
        cs = solar.cielo_despejado(irr.index, lugar, pd.Timedelta("30min"))["ghi_cs"]
        de_dia = cs > 50
        if nombre == "satelite_msg" and irr.loc[de_dia, "ghi"].notna().mean() <= 0.9:
            continue
        c = (irr["ghi"] / cs).where(de_dia).clip(0, 1.5).rename("claridad")
        c.attrs["fuente"] = nombre
        return c
    return pd.Series(dtype=float, name="claridad")


def marcas_planta(grupo: str, plantuuid: str, codigo: str, inicio, fin, refrescar: bool = False,
                  claridad: pd.Series | None = None) -> pd.DataFrame:
    """Marcas horarias de un inversor: fracción en cero de la red, tensión y potencia relativas, y si la hora esta
    recortada (por el sensor de la planta o, si la planta no tiene sensor, por la regla de tensión en las horas
    despejadas según `claridad` = G_POA / G_POA de cielo despejado). attrs["dias"] tiene la marca diaria del
    sensor (None sin sensor: entonces no se recorta el régimen, solo se excluyen horas) y attrs["sensor"] el
    inversor con sensor."""
    clave = (grupo, str(plantuuid), str(inicio), str(fin))
    if clave not in _LECTURAS:
        lect = lecturas_planta(grupo, plantuuid, inicio, fin, refrescar)
        m = marcas_sensor(lect)
        d = diario(m) if lect["con_sensor"] else None
        h = horario(m, d if d is not None else pd.DataFrame())
        _LECTURAS[clave] = (lect["con_sensor"], d, h, tension_horaria(lect))
    sensor, d, h, th = _LECTURAS[clave]
    indice = h.index.union(th.index) if not th.empty else h.index
    salida = pd.DataFrame(index=indice)
    salida["frac_cero"] = h["frac_cero"].reindex(indice)
    tiene_tension = ("v_rel", codigo) in th.columns
    if tiene_tension:
        salida["v_rel"] = th[("v_rel", codigo)].reindex(indice)
        salida["p_rel"] = th[("p_rel", codigo)].reindex(indice)
    if sensor is not None:
        salida["recortada"] = h["recortada"].reindex(indice).fillna(False).astype(bool)
    elif tiene_tension and claridad is not None:
        regla = regla_tension(salida["v_rel"], salida["p_rel"], claridad, **PARAMETROS_TENSION)
        salida["recortada"] = regla.reindex(indice).fillna(False).astype(bool)
    else:
        salida["recortada"] = False
    salida.attrs["dias"] = d
    salida.attrs["sensor"] = sensor
    return salida


def instalador_planta(eventos: pd.DataFrame | None, plantuuid: str) -> dict:
    """Lo que el instalador declaro de una planta: limitación vigente, fin confirmado de la limitación, conformidad del
    operador de red (cota inferior del fin) y reparaciones con fecha (telemetry.plantas_eventos)."""
    salida = {"limitada_vigente": False, "fin_limitacion": None, "conformidad_or": None, "reparaciones": []}
    if eventos is None or len(eventos) == 0:
        return salida
    e = eventos[eventos["plantuuid"].astype(str) == str(plantuuid)]
    ctrl, rep = e[e["tipo"] == "control_exportacion"], e[(e["tipo"] == "reparacion") & e["hasta"].notna()]

    def dia(v):
        return None if v is None or pd.isna(v) else pd.Timestamp(v).tz_localize(None).normalize()
    salida["limitada_vigente"] = bool(ctrl["vigente"].fillna(False).astype(bool).any()) if len(ctrl) else False
    if len(ctrl):
        salida["fin_limitacion"] = dia(ctrl["hasta"].max())
        salida["conformidad_or"] = dia(ctrl["conformidad_or"].max())
    salida["reparaciones"] = [(dia(r.desde), dia(r.hasta)) for r in rep.itertuples()]
    return salida


def sin_marcas_tras_conformidad(recortada: pd.Series, instalador: dict | None, sensor) -> pd.Series:
    """Planta sin sensor de red: después de la conformidad del operador de red, si el instalador no declara una
    limitación vigente, la regla de tensión ya no se interpreta como limitación (sus marcas eran falsos positivos)."""
    if sensor is not None or not instalador or instalador["limitada_vigente"] or instalador["conformidad_or"] is None:
        return recortada
    desde = pd.Timestamp(instalador["conformidad_or"], tz=config.ZONA).tz_convert("UTC")
    return recortada & np.asarray(recortada.index < desde)


def regimen_vigente(dias: pd.DataFrame, corte, dias_libres_minimos: int = DIAS_LIBRES_MINIMOS,
                    cola: int = DIAS_COLA_CORTE, instalador: dict | None = None) -> dict:
    """Estado al corte con lo estimado de la telemetría (_regimen_telemetria) corregido con lo que declaro el
    instalador. La clave "origen" dice de donde sale el estado, y "fin_reparacion" el primer día después de la última
    reparación terminada antes del corte (B no puede empezar antes)."""
    r = _regimen_telemetria(dias, corte, dias_libres_minimos, cola)
    r["origen"] = "telemetria"
    if not instalador:
        return r
    corte_d = (pd.Timestamp(corte).tz_localize(None) if pd.Timestamp(corte).tzinfo else pd.Timestamp(corte)).normalize()
    fines = [h + pd.Timedelta(days=1) for _, h in instalador["reparaciones"] if h is not None and h < corte_d]
    if fines:
        r["fin_reparacion"] = max(fines)
    if instalador["limitada_vigente"]:
        r.update({"estado": "limitada al corte", "inicio_b": None, "dias_b": 0, "origen": "instalador"})
        return r
    fin = instalador["fin_limitacion"]
    if fin is not None and fin < corte_d:
        inicio = fin + pd.Timedelta(days=1)
        # Si la telemetría muestra limitación después de la fecha declarada, se conserva la más tardía
        if r.get("inicio_b") is None or pd.Timestamp(r["inicio_b"]) <= inicio:
            dias_b = int((corte_d - inicio).days)
            r.update({"estado": "liberada" if dias_b >= dias_libres_minimos else "liberada hace poco", "inicio_b": inicio,
                      "ultimo_dia_limitado": fin, "dias_b": dias_b, "origen": "instalador"})
    return r


def _regimen_telemetria(dias: pd.DataFrame, corte, dias_libres_minimos: int = DIAS_LIBRES_MINIMOS,
                        cola: int = DIAS_COLA_CORTE) -> dict:
    """Estado de la planta al corte (exclusivo, fecha local) y, si procede, el primer día del régimen B.

    estado: "nunca limitada", "liberada" (inicio_b = primer día posterior al último día limitado con su propia
    fracción en cero menor que FRACCION_DIA_LIBRE), "limitada al corte" (alguno de los últimos `cola` días antes del
    corte limitado) o "sin sensor" (sin marcas)."""
    corte = pd.Timestamp(corte).tz_localize(None).normalize() if pd.Timestamp(corte).tzinfo else pd.Timestamp(corte).normalize()
    if dias is None or dias.empty or "limitado" not in dias:
        return {"estado": "sin sensor", "inicio_b": None, "ultimo_dia_limitado": None, "dias_b": None}
    d = dias.loc[dias.index < corte]
    if d.empty:
        return {"estado": "sin datos", "inicio_b": None, "ultimo_dia_limitado": None, "dias_b": None}
    limitados = d.index[d["limitado"].fillna(False).astype(bool)]
    if len(limitados) == 0:
        # Sin limitación no hay restriccón: el régimen B es el último régimen de capacidad completo
        return {"estado": "nunca limitada", "inicio_b": None, "ultimo_dia_limitado": None,
                "dias_b": int((corte - d.index.min()).days)}
    ultimo = limitados.max()
    if ultimo >= corte - pd.Timedelta(days=cola):
        return {"estado": "limitada al corte", "inicio_b": None, "ultimo_dia_limitado": ultimo, "dias_b": 0}
    # E42: la mediana centrada se retrasa hasta tres días en la transición, y un día con muy pocas lecturas con
    # producción (estrangulado por un consumo mínimo, como un domingo) no tiene fracción propia. El régimen B empieza
    # el primer día posterior que muestra por si mismo inyección libre; los días intermedios quedan fuera
    despues = d.loc[d.index > ultimo]
    libres = despues.index[(despues["fraccion_cero"] < FRACCION_DIA_LIBRE).to_numpy()]
    inicio = libres.min() if len(libres) else ultimo + pd.Timedelta(days=1)
    dias_b = int((corte - inicio).days)
    estado = "liberada" if dias_b >= dias_libres_minimos else "liberada hace poco"
    return {"estado": estado, "inicio_b": inicio, "ultimo_dia_limitado": ultimo, "dias_b": dias_b}


def periodo_b(periodo_capacidad: tuple | None, regimen: dict) -> tuple | None:
    """Periodo del régimen B para calidad.mascara_regimen: desde el más tardío entre el inicio del último régimen
    de capacidad, el primer día sin limitación y el día siguiente a la última reparación, hasta el fin del régimen de
    capacidad. None si la planta esta limitada al corte o no tiene régimen de capacidad estable."""
    if periodo_capacidad is None or regimen.get("estado") in ("limitada al corte", "sin datos"):
        return None
    # Fechas locales sin zona (el régimen de capacidad viene por meses con zona; el de la limitación, por días)
    inicio = calidad._fecha_local(periodo_capacidad[0])
    for clave in ("inicio_b", "fin_reparacion"):
        if regimen.get(clave) is not None:
            inicio = max(inicio, calidad._fecha_local(regimen[clave]))
    fin_mes = calidad._fecha_local(periodo_capacidad[1]).normalize() + pd.offsets.MonthEnd(0)
    if inicio > fin_mes:
        return None                    # el régimen de capacidad termino antes del fin de la limitación
    return inicio, periodo_capacidad[1]
