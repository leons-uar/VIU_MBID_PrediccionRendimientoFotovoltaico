"""
pipeline.py
===========
Las partes 1 a 5 del notebook de localidad (extracción cruda, corrección horaria, regularización,
imputación, agregación y objetivos) para un inversor, con las decisiones ya tomadas por los experimentos
(fuente de irradiancia, técnica de imputación, periodo de régimen estable). Lo usan los procesos de
producción (src/40_pronostico_diario.py y src/41_reentrenamiento.py), que no pueden ejecutar el notebook.

Nada de lo calculado se escribe en la base de datos: la BD solo guarda los datos crudos de las APIs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import agregacion as agg
from . import auditoria_fv as aud, calidad, caracteristicas as caract, config, datos, imputacion as imp, nomenclatura as nom
from . import objetivos as obj
from . import recorte as recorte_
from . import solar

COLS_TEL = ["acpower1", "acpower2", "acpower3", "totalyield", "dailyyield", "invertertemperature",
            "mppttotalinputpower", "devicestatus"]
SERIES_METEO = ("om_archive__ecmwf_ifs", "om_archive__era5", "om_satellite__eumetsat_lsa_saf_msg",
                "om_previous_runs__ecmwf_ifs025", "om_single_runs__ecmwf_ifs")


@dataclass
class Unidad:
    grupo: str
    codigo: str
    inversor: pd.Series
    lugar: object
    horario: pd.DataFrame          # objetivos horarios (yf, yr, pr, kstar, en_regimen, frac_imputada, frac_falla, recortada...)
    diario: pd.DataFrame
    crudos: dict
    decisiones: dict = dataclasses.field(default_factory=dict)
    regimen: dict = dataclasses.field(default_factory=dict)   # estado de la limitación al corte y periodo del régimen B


def decisiones(grupo: str) -> dict:
    """Decisiones de los experimentos de la localidad, leidas de resultados.json: fuente de irradiancia y su
    factor de nivel frente a NSRDB, técnica de imputación, potencia pico que usa el análisis (auditoría), valor
    central y bandas del pronóstico a 30 días."""
    from . import resultados
    r = resultados.cargar().get("localidades", {}).get(grupo, {})
    c = r.get("conclusiones", {})
    datos_ = r.get("datos", {})
    principal = c.get("unidad_principal")
    h4 = r.get("modelos", {}).get(f"h4_30d_{principal}", {})
    # Sin unidad principal (grupo sin unidades modelables) no hay conclusiones, pero el cuaderno si eligió la fuente y
    # la imputación: se leen de sus experimentos antes de caer en los valores de omisón
    fuente = c.get("fuente_irradiancia") or datos_.get("experimento_fuentes", {}).get("elegida") or "ifs_analisis"
    imputacion = c.get("imputacion") or datos_.get("experimento_imputacion", {}).get("metodo_elegido") or "lineal"
    return {"fuente_irradiancia": fuente, "imputacion": imputacion,
            "unidad_principal": principal, "modelos": r.get("modelos", {}), "modelo_final": r.get("modelo_final", {}),
            "nivel": {k: v.get("factor") for k, v in datos_.get("nivel_fuentes", {}).items() if v and v.get("factor")},
            "kwp": {k: v.get("kwp_analisis") for k, v in datos_.get("auditoria_p0", {}).items() if v and v.get("kwp_analisis")},
            # Si el cuaderno no pudo calibrar las bandas, resultados.json las guarda como null: pasan a NaN
            "centro_30": h4.get("centro", "media"),
            "bandas_30": tuple(np.nan if b is None else float(b) for b in (h4.get("bandas_8_30") or (np.nan, np.nan))),
            "beta_anomalia": pd.Series({int(k): v for k, v in (h4.get("beta_anomalia") or {}).items()}, dtype=float),
            "modelables": [u for u, i in r.get("unidades_modelables", {}).get("unidades", {}).items()
                           if i.get("modelable") and i.get("tipo") == "inversor"]}


def unidades_operacion(grupos: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(grupo, código, deviceuuid) de los inversores modelados en los cuadernos: los que operan cada día."""
    from . import resultados
    loc = resultados.cargar().get("localidades", {})
    tabla = nom.tabla().set_index("codigo")
    salida = []
    for g in sorted(grupos or loc):
        for u in decisiones(g)["modelables"]:
            if u in tabla.index:
                salida.append((g, u, str(tabla.loc[u, "deviceuuid"])))
    return salida


def irradiancia(crudos: dict, nombre: str, lugar, dec: dict, filtro: dict | None = None, metodo_diurno: str = "lineal") -> pd.DataFrame | None:
    """irradiancia_fuente() con las fuentes observadas al nivel de NSRDB (factor de la auditoría de nivel), como
    en los cuadernos. Los pronósticos no se calibran."""
    f = caract.irradiancia_fuente(crudos, nombre, lugar, filtro=filtro, metodo_diurno=metodo_diurno)
    if f is not None and nombre in aud.OBSERVADAS and dec.get("nivel", {}).get(nombre):
        f = aud.calibrar(f, float(dec["nivel"][nombre]))
    return f


def leer_crudos(grupo: str, deviceuuid: str, inicio: str, fin: str, refrescar: bool = True) -> dict[str, pd.DataFrame]:
    clave = f"inv_{deviceuuid[:8]}"
    series = [s for s in datos.series_localidad(grupo, meteo=True, ideam=False) if s.nombre in (clave, *SERIES_METEO)]
    return datos.leer(series, inicio, fin, refrescar=refrescar)


def procesar(grupo: str, deviceuuid: str, inicio: str, fin: str, refrescar: bool = True,
             crudos: dict | None = None, corte: str | None = None, aplicar_recorte: bool = True) -> Unidad:
    """Objetivos horarios y diarios de un inversor, como en las partes 2 a 5 del notebook.

    corte: fecha local (exclusiva) con la que se deciden el régimen de capacidad y el de la limitación de
    inyección (por defecto, `fin`). aplicar_recorte: False solo para comparar con el histórico sin depurar
    (régimen A) o calibrar la regla de tensión; en la operación y en los cuadernos siempre es True."""
    dec = decisiones(grupo)
    const = datos.constantes(grupo)
    punto = const["grupo"].iloc[0]
    lugar = solar.ubicacion(punto.latitude, punto.longitude, punto.elevacion)
    inv = const["dispositivos"].set_index("deviceuuid").loc[deviceuuid]
    inv = inv.copy()
    inv["deviceuuid"] = deviceuuid
    # Potencia pico que fija la auditoría de los cuadernos (la base de datos guarda la declarada)
    inv["kwpdc_declarado"] = inv["kwpdc"]
    inv["kwpdc"] = float(dec["kwp"].get(nom.inversor(deviceuuid), inv["kwpdc"]))
    crudos = crudos or leer_crudos(grupo, deviceuuid, inicio, fin, refrescar)
    clave = f"inv_{deviceuuid[:8]}"

    d = calidad.corregir_hora(crudos[clave])                                   # 2.1 zona horaria
    r = imp.regularizar(d, "5min", [c for c in COLS_TEL if c in d.columns])     # 2.4 rejilla de 5 min
    de_dia = solar.es_de_dia(r.index, lugar)
    cs = solar.cielo_despejado(r.index, lugar)
    ref = solar.irradiancia_plano(cs["ghi_cs"], cs["dni_cs"], cs["dhi_cs"], lugar, inv.inclinacion, inv.azimut, caras=obj.caras_de(inv))["poa_global"]
    i, motivo, _ = imp.imputar(r, {c: imp.naturaleza_telemetria(c) for c in r.columns}, "5min", de_dia,
                               referencia=ref, metodo_diurno=dec["imputacion"])           # 3 imputación
    est = i.get("devicestatus")
    base = pd.DataFrame({"p_ac": obj.potencia_ac(i), "totalyield": i.get("totalyield"),
                         "invertertemperature": i.get("invertertemperature"),
                         "falla": est.round().isin([103, 104]).astype(float).where(est.notna()) if est is not None else np.nan})
    horario, cal = agg.agregar(base, {"p_ac": "media", "totalyield": "ultimo", "invertertemperature": "media", "falla": "media"},
                               "1h", "5min", imputado=pd.DataFrame({"p_ac": motivo["acpower1"] == "interpolado"}))   # 4

    irr = irradiancia(crudos, dec["fuente_irradiancia"], lugar, dec, metodo_diurno=dec["imputacion"])   # 5 (al nivel de NSRDB)
    e = horario["p_ac"] / 1000.0
    idx = e.index.intersection(irr.index)
    poa, poa_cs = obj.poa_inversor(irr.loc[idx, "ghi"], irr.loc[idx, "dni"], irr.loc[idx, "dhi"], lugar, inv.inclinacion, inv.azimut, caras=obj.caras_de(inv))
    oh = obj.objetivos_horarios(e.loc[idx], inv.kwpdc, poa, poa_cs)
    oh["frac_imputada"] = cal["p_ac__imputado"].reindex(idx)
    oh["frac_falla"] = horario["falla"].reindex(idx)
    oh.loc[(oh["frac_falla"].fillna(0) > 0).to_numpy(), ["e_kwh", "yf", "pr", "kstar"]] = np.nan
    # Sin producción con sol y sin estado de falla (indisponibilidad no declarada): se excluye como la falla
    oh["sin_produccion"] = calidad.sin_produccion(oh, inv.kwpdc)
    oh.loc[oh["sin_produccion"].to_numpy(), ["e_kwh", "yf", "pr", "kstar"]] = np.nan

    # E28: limitación de inyección. Las horas recortadas se excluyen como las horas en falla
    corte = corte or fin
    corte_utc = pd.Timestamp(corte, tz=config.ZONA).tz_convert("UTC")
    marcas = recorte_.marcas_planta(grupo, str(inv["plantuuid"]), nom.inversor(deviceuuid), inicio, fin, refrescar,
                                    claridad=(oh["g_poa"] / oh["g_poa_cs"]).where(oh["g_poa_cs"] > 50))
    instalador = recorte_.instalador_planta(const.get("eventos"), str(inv["plantuuid"]))
    marcas["recortada"] = recorte_.sin_marcas_tras_conformidad(marcas["recortada"], instalador, marcas.attrs.get("sensor"))
    oh["recortada"] = marcas["recortada"].reindex(idx).fillna(False).astype(bool)
    if aplicar_recorte:
        oh.loc[oh["recortada"].to_numpy(), ["e_kwh", "yf", "pr", "kstar"]] = np.nan

    p5 = obj.potencia_ac(d.set_index("datatime"))
    antes_del_corte = p5.loc[p5.index < corte_utc]                              # el régimen se decide sin ver el corte
    regimen = recorte_.regimen_vigente(marcas.attrs.get("dias"), corte, instalador=instalador)
    if aplicar_recorte and regimen.get("inicio_b") is not None and regimen.get("estado") != "limitada al corte":
        # Tras el fin de la limitación, los regímenes de capacidad se buscan solo con los datos posteriores
        antes_del_corte = antes_del_corte.loc[antes_del_corte.index >= pd.Timestamp(regimen["inicio_b"], tz=config.ZONA).tz_convert("UTC")]
    periodo_cap = calidad.ultimo_regimen_estable(calidad.regimenes_capacidad(antes_del_corte, inv.ratedpower * 1000))   # 2.3
    periodo = recorte_.periodo_b(periodo_cap, regimen) if aplicar_recorte else periodo_cap
    # Una ampliación registrada (telemetry.plantas_fv_etapas) no deja entrar días de la etapa anterior; su día de
    # entrada en servicio se afina con la potencia máxima diaria, como en los cuadernos (calidad.inicio_por_etapa)
    etapas = const.get("etapas")
    if etapas is not None and len(etapas):
        e = etapas[etapas["plantuuid"] == str(inv["plantuuid"])].sort_values("desde")
        e = e[pd.to_datetime(e["desde"]) < pd.Timestamp(corte)]
        if len(e) >= 2:
            ini_e = calidad.inicio_por_etapa(p5.loc[p5.index < corte_utc], inv.ratedpower * 1000, e["desde"].iloc[-1],
                                             float(e["paneles"].iloc[-2]) / float(e["paneles"].iloc[-1]))
            periodo = calidad.con_etapa(periodo, ini_e)
            regimen["inicio_etapa"] = ini_e
    regimen.update({"periodo_capacidad": periodo_cap, "periodo": periodo, "sensor": marcas.attrs.get("sensor")})
    if periodo is None:
        motivo = "limitada al corte" if regimen.get("estado") == "limitada al corte" else "sin regimen de capacidad estable de al menos 4 meses"
        raise ValueError(f"{nom.inversor(deviceuuid)}: {motivo}")
    oh["en_regimen"] = calidad.mascara_regimen(oh.index, periodo)
    od = obj.objetivos_diarios(oh)
    od["en_regimen"] = calidad.mascara_regimen(od.index, periodo)
    return Unidad(grupo, nom.inversor(deviceuuid), inv, lugar, oh, od, crudos, dec, regimen)


def yr_exogena(u: Unidad, fuente: str, indice: pd.DatetimeIndex, filtro: dict | None = None) -> pd.DataFrame:
    """Yr horario de una fuente en el plano del inversor, con los huecos diurnos cubiertos por el análisis
    IFS (como exog_yr en el notebook): la regresión del SARIMAX no admite exógenas faltantes."""
    def yr_de(nombre, filtro_=None):
        f = irradiancia(u.crudos, nombre, u.lugar, u.decisiones, filtro=filtro_)
        if f is None:
            return pd.Series(np.nan, index=indice)
        f = f.reindex(indice)
        poa, _ = obj.poa_inversor(f["ghi"], f["dni"], f["dhi"], u.lugar, u.inversor.inclinacion, u.inversor.azimut, caras=obj.caras_de(u.inversor))
        return poa / 1000.0
    yr = yr_de(fuente, filtro)
    if yr.isna().any() and fuente != "ifs_analisis":
        yr = yr.fillna(yr_de("ifs_analisis"))
    return yr.interpolate(limit=3).fillna(0.0).rename("yr").to_frame()


def era5_climatologia(grupo: str) -> pd.DataFrame:
    s = next(x for x in datos.series_localidad(grupo, meteo=True, ideam=False) if x.nombre == "om_archive__era5")
    return datos.leer_serie(dataclasses.replace(s, nombre="om_archive__era5_climatologia"), *config.CLIMATOLOGIA_ERA5)


def yr_diario(u: Unidad, irr: pd.DataFrame | None, minimo_horas: int = 20) -> pd.Series:
    """Yr diario (día local, kWh/m2) en el plano del inversor a partir de GHI, DNI y DHI horarios."""
    if irr is None or irr.empty:
        return pd.Series(dtype=float)
    poa, _ = obj.poa_inversor(irr["ghi"], irr["dni"], irr["dhi"], u.lugar, u.inversor.inclinacion, u.inversor.azimut, caras=obj.caras_de(u.inversor))
    s = (poa / 1000.0).copy()
    s.index = (s.index - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).tz_localize(None)
    return s.resample("D").sum(min_count=minimo_horas)


def yr_diario_por_antelacion(u: Unidad) -> dict[int, pd.Series]:
    """Yr diario pronosticado con h = 1..7 días de antelación (Open-Meteo Previous Runs, previousday = h)."""
    return {h: yr_diario(u, caract.irradiancia_fuente(u.crudos, "ifs025_pronostico_d1", u.lugar, filtro={"previousday": h}))
            for h in range(1, 8)}


