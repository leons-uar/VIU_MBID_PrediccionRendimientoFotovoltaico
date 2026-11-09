"""
consultas.py
============
Plantillas SQL con nombre para leer la BD `solar` en crudo. Son SQL explicito y legible: el mismo
texto lo puede usar una API interna (por ejemplo, la que consumiria una web en Next.js), de modo
que notebooks y web lean los datos exactamente igual.

Reglas:
- Solo SELECT de lectura y nunca SELECT *: siempre se nombran las columnas.
- ConnectorX no admite parámetros enlazados, así que los valores se insertan con format() DESPUÉS
  de validarlos en datos.py (identificadores contra information_schema, UUID con expresón regular,
  fechas normalizadas a ISO en UTC).
- No se agrega ni se imputa nada en SQL: la BD entrega el dato tal como se guardó.
"""

# Columnas de una tabla o vista (para validar nombres y para el catálogo).
COLUMNAS = """
SELECT table_schema AS esquema, table_name AS tabla, column_name AS columna, data_type AS tipo, ordinal_position AS orden
FROM information_schema.columns
WHERE table_schema IN ('telemetry', 'meteo')
ORDER BY table_schema, table_name, ordinal_position
"""

# Diccionario de variables meteorológicas (unidad, tipo de valor, marca de tiempo, periodo).
VARIABLES_METEO = """
SELECT fuente, variable, unidad, tipovalor, marcatiempo, periodominutos, equivalente, descripcion
FROM meteo.variables
ORDER BY fuente, variable
"""

# Plantas de una localidad (grupo meteorológico) con sus constantes. Coordenadas redondeadas.
PLANTAS_GRUPO = """
SELECT p.plantuuid::text AS plantuuid, pg.grupo, pg.distanciakm, p.pvcapacity, p.batterycapacity,
       p.planttimezone, p.createtime, p.latitude, p.longitude, p.businesstype
FROM telemetry.plants p
JOIN meteo.plantas_grupos pg USING (plantuuid)
WHERE pg.grupo = '{grupo}'
ORDER BY p.pvcapacity DESC NULLS LAST
"""

# Dispositivos de una localidad con su configuración fotovoltaica (estimada o del instalador).
DISPOSITIVOS_GRUPO = """
SELECT d.deviceuuid::text AS deviceuuid, d.plantuuid::text AS plantuuid, d.devicetype, d.devicemodel,
       d.ratedpower, d.onlinestatus, f.kwpdc, f.inclinacion, f.azimut, f.origen, f.excluir, f.motivoexclusion
FROM telemetry.devices d
JOIN meteo.plantas_grupos pg USING (plantuuid)
LEFT JOIN telemetry.devices_fv f USING (deviceuuid)
WHERE pg.grupo = '{grupo}'
ORDER BY d.devicetype, d.deviceuuid
"""

# Caras del generador de las plantas de una localidad (SQL 19)
CARAS_GRUPO = """
SELECT c.plantuuid::text AS plantuuid, c.cara, c.azimut, c.inclinacion, c.paneles, c.fraccion, c.techo
FROM telemetry.plantas_fv_caras c
JOIN meteo.plantas_grupos pg USING (plantuuid)
WHERE pg.grupo = '{grupo}'
ORDER BY c.plantuuid, c.cara
"""

# Etapas de capacidad registradas de las plantas de una localidad (ampliaciones: paneles desde una fecha)
ETAPAS_GRUPO = """
SELECT e.plantuuid::text AS plantuuid, e.desde, e.paneles
FROM telemetry.plantas_fv_etapas e
JOIN meteo.plantas_grupos pg USING (plantuuid)
WHERE pg.grupo = '{grupo}'
ORDER BY e.plantuuid, e.desde
"""

# Atributos constantes de las plantas de una localidad (SQL 19 y 23), con los datos del módulo (datasheet)
PLANTAS_FV_GRUPO = """
SELECT f.plantuuid::text AS plantuuid, f.departamento, f.ciudad, f.region_natural, f.altitud_dem_m, f.altitud_ge_m,
       f.tipo, f.paneles, f.caras_panel, f.material_celdas, f.modulo, f.wp_panel, f.kwp_dc, f.inversor_externo,
       m.gamma_pmax_pct_c, m.noct_c, m.bifacialidad_pct
FROM telemetry.plantas_fv f
JOIN meteo.plantas_grupos pg USING (plantuuid)
LEFT JOIN telemetry.modulos_fv m ON m.modelo = f.modulo
WHERE pg.grupo = '{grupo}'
"""

# Eventos que declaro el instalador (limitación de inyección, reparaciones) de las plantas de una localidad (SQL 23)
EVENTOS_GRUPO = """
SELECT e.plantuuid::text AS plantuuid, e.tipo, e.desde, e.hasta, e.limite_w, e.conformidad_or, e.vigente, e.detalle
FROM telemetry.plantas_eventos e
JOIN meteo.plantas_grupos pg USING (plantuuid)
WHERE pg.grupo = '{grupo}'
ORDER BY e.plantuuid, e.tipo, e.desde
"""

# Todas las plantas y sus inversores, con la fecha de alta de la planta y el primer dato de cada
# inversor, para la nomenclatura P01I0102 (tfm/nomenclatura.py).
NOMENCLATURA = """
SELECT p.plantuuid::text AS plantuuid, p.createtime, d.deviceuuid::text AS deviceuuid,
       (SELECT min(h.datatime) FROM telemetry.inverter_history h WHERE h.deviceuuid = d.deviceuuid) AS primer_dato
FROM telemetry.plants p
LEFT JOIN telemetry.devices d ON d.plantuuid = p.plantuuid AND d.devicetype = 1
ORDER BY p.createtime, p.plantuuid
"""

# Punto meteorológico del grupo y celdas devueltas por cada API (con elevación).
GRUPO = """
SELECT grupo, zona, latitude, longitude, diametrokm, prioridad, nplantas, ninversores
FROM meteo.grupos WHERE grupo = '{grupo}'
"""
CELDAS_GRUPO = """
SELECT fuente, modelo, latitude, longitude, elevation
FROM meteo.celdas WHERE grupo = '{grupo}' ORDER BY fuente, modelo
"""

# Cuántos valores no nulos tiene cada columna de un dispositivo (para no traer columnas vacías).
# {conteos} = "count(col1) AS col1, count(col2) AS col2, ..."
CONTEO_COLUMNAS_DISPOSITIVO = """
SELECT {conteos} FROM telemetry.{tabla} WHERE deviceuuid = '{deviceuuid}'
"""

# Serie cruda de un dispositivo SolaX (inversor o medidor).
SERIE_DISPOSITIVO = """
SELECT datatime, {columnas}
FROM telemetry.{tabla}
WHERE deviceuuid = '{deviceuuid}' AND datatime >= '{inicio}' AND datatime < '{fin}'
ORDER BY datatime
"""

# Serie cruda de un producto meteorológico de un grupo. {claves} incluye modelo, previousday o run
# según la tabla; {filtro} añade condiciones de modelo/previousday/run ya validadas.
SERIE_METEO = """
SELECT datatime, {claves}{columnas}
FROM meteo.{tabla}
WHERE grupo = '{grupo}' {filtro} AND datatime >= '{inicio}' AND datatime < '{fin}'
ORDER BY {orden}
"""

# Observaciones IDEAM de las estaciones asociadas a un grupo (fila nativa).
IDEAM_GRUPO = """
SELECT o.datatime, o.codigoestacion, o.codigosensor, s.descripcionsensor, s.unidadmedida, s.dataset,
       o.valorobservado, ge.distanciakm
FROM meteo.ideam_observaciones o
JOIN meteo.ideam_sensores s USING (codigosensor)
JOIN meteo.ideam_grupos_estaciones ge ON ge.codigoestacion = o.codigoestacion AND ge.grupo = '{grupo}'
WHERE o.datatime >= '{inicio}' AND o.datatime < '{fin}'
ORDER BY o.datatime
"""

# Índice ONI (El Niño / La Niña), si ya se cargo con src/29_noaa_oni.py.
ONI = """
SELECT anio, mes, oni FROM meteo.enso_oni ORDER BY anio, mes
"""
