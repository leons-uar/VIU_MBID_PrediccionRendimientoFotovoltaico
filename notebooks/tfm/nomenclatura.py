"""
nomenclatura.py
===============
Códigos legibles para plantas e inversores en el análisis y en el documento: P02 es la planta 2 y
P02I0103 el inversor 1 de los 3 que tiene la planta 2.

    Planta:   P + número de la planta (01..NN), por fecha de alta en SolaxCloud (createtime); la más
              antigua es P01. Empate: uuid.
    Inversor: P.. + I + número del inversor dentro de la planta (01..) + total de inversores de la
              planta (dos digitos). Orden dentro de la planta: primer dato en la telemetría; empate: uuid.

Los códigos se calculan en el análisis a partir del catálogo de la base de datos; no se guardan en ella
ni modifican los procesos de extracción. Son estables mientras no se den de alta plantas con fecha
anterior a las existentes: una planta nueva recibe el número siguiente. En los datos publicados (datos/parametros)
los identificadores ya son estos códigos; la tabla que los relaciona con los uuid de la base no se publica.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import config

ARCHIVO = config.RAIZ / "autor" / "datos_bd" / "nomenclatura.csv"     # solo con la base de datos (no se publica)


def construir(catalogo: pd.DataFrame) -> pd.DataFrame:
    """Asigna los códigos a partir de un catálogo con columnas plantuuid, createtime, deviceuuid y
    primer_dato (una fila por inversor; las plantas sin inversores tienen deviceuuid nulo)."""
    c = catalogo.copy()
    plantas = c[["plantuuid", "createtime"]].drop_duplicates("plantuuid").sort_values(["createtime", "plantuuid"])
    ancho = max(2, len(str(len(plantas))))
    codigo_planta = {u: f"P{i:0{ancho}d}" for i, u in enumerate(plantas["plantuuid"], start=1)}
    c["planta"] = c["plantuuid"].map(codigo_planta)
    inv = c.dropna(subset=["deviceuuid"]).copy()
    inv["primer_dato"] = pd.to_datetime(inv["primer_dato"], utc=True)
    inv = inv.sort_values(["planta", "primer_dato", "deviceuuid"], na_position="last")
    inv["numero"] = inv.groupby("planta").cumcount() + 1
    inv["total"] = inv.groupby("planta")["deviceuuid"].transform("size")
    inv["codigo"] = [f"{p}I{n:02d}{t:02d}" for p, n, t in zip(inv["planta"], inv["numero"], inv["total"])]
    return inv[["codigo", "planta", "numero", "total", "deviceuuid", "plantuuid", "primer_dato"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def tabla(refrescar: bool = False) -> pd.DataFrame:
    """Tabla de nomenclatura: la publicada en datos/parametros o, con la base de datos, calculada desde su catálogo."""
    from . import consultas, datos
    if config.FUENTE_DATOS != "bd":
        t = datos.parametro("nomenclatura").copy()
        t["primer_dato"] = pd.to_datetime(t["primer_dato"], utc=True, format="ISO8601")
        return t
    t = construir(datos._sql_a_pandas(consultas.NOMENCLATURA))
    ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(ARCHIVO, index=False)
    return t


def inversor(deviceuuid: str) -> str:
    """Codigo del inversor (P02I0103) a partir de su uuid."""
    t = tabla()
    fila = t.loc[t["deviceuuid"] == str(deviceuuid), "codigo"]
    if fila.empty:
        raise KeyError(f"inversor sin codigo: {deviceuuid}")
    return fila.iloc[0]


def planta(plantuuid: str) -> str:
    """Codigo de la planta (P02) a partir de su uuid."""
    t = tabla()
    fila = t.loc[t["plantuuid"] == str(plantuuid), "planta"]
    if fila.empty:
        from . import consultas, datos
        if config.FUENTE_DATOS != "bd":
            raise KeyError(f"planta sin codigo: {plantuuid}")
        cat = datos._sql_a_pandas(consultas.NOMENCLATURA)
        cat = cat[["plantuuid", "createtime"]].drop_duplicates().sort_values(["createtime", "plantuuid"]).reset_index(drop=True)
        return f"P{int(cat.index[cat['plantuuid'] == str(plantuuid)][0]) + 1:02d}"
    return fila.iloc[0]


def es_planta(codigo: str) -> bool:
    return "I" not in codigo
