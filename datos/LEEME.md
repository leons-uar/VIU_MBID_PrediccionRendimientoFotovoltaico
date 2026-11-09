# Datos del trabajo

Datos crudos que leen los cuadernos, exportados sin cambios de la base de datos del proyecto (PostgreSQL con
TimescaleDB, cuya estructura está en `sql/`). Cubren desde el 2024-01-01 hasta el 2026-11-06 inclusive, hora de
Colombia; los pronósticos archivados llegan ocho días más allá, porque los emitidos hasta ese día ya existían. Ningún
valor está corregido, imputado ni agregado: todo eso ocurre en los cuadernos.

Todos los archivos son CSV con coma como separador y punto decimal, en UTF-8; los de series van comprimidos con gzip
(`.csv.gz`), y pandas los lee directamente. `MANIFIESTO.sha256` guarda la huella SHA256 de cada archivo, y los
cuadernos se detienen si alguna no coincide.

## Identificadores

Las plantas se identifican como P01, P02... por su fecha de alta en SolaxCloud, y los inversores como P02I0103: el
inversor 1 de los 3 de la planta 2. No hay nombres, direcciones, seriales ni identificadores de la plataforma, y las
coordenadas de las plantas están redondeadas a 0,01°.

## Marcas de tiempo

`datatime` es el instante UTC tal como lo guardó cada fuente. En la telemetría, `plantlocaltime` es la hora del reloj
del inversor: en algunos periodos SolaX la registró con la zona de China (UTC+8) en lugar de la de Colombia (UTC-5),
y el cuaderno de cada localidad lo detecta y corrige fila a fila con la diferencia entre las dos columnas.

## `telemetria/<inversor>.csv.gz`

Lecturas de cada inversor SolaX cada unos 5 minutos (API de histórico de SolaxCloud).

| Columna | Unidad | Contenido |
|---|---|---|
| `datatime` | UTC | Instante de la lectura según la API |
| `plantlocaltime` | hora local del inversor | Hora del reloj del inversor |
| `devicestatus` | código | Estado según SolaX: 100 espera, 102 funcionamiento normal, 103 y 104 falla |
| `acpower1`, `acpower2`, `acpower3` | W | Potencia activa por fase; su suma es la potencia alterna |
| `totalyield` | kWh | Contador de energía alterna total |
| `dailyyield` | kWh | Contador de energía del día, que se reinicia a medianoche |
| `invertertemperature` | °C | Temperatura interna del inversor |
| `mppttotalinputpower` | W | Potencia continua de entrada, suma de los MPPT |
| `gridpower` | W | Intercambio con la red medido por el sensor del inversor maestro, positivo al exportar (solo en el maestro de las plantas con sensor) |
| `mpptNvoltage`, `mpptNpower` | V, W | Tensión y potencia de cada seguidor del punto de máxima potencia (N = 1 a 12, según el modelo) |
| `pvNvoltage`, `pvNpower` | V, W | Tensión y potencia de cada entrada de paneles (N = 1 a 12, según el modelo) |

Una celda vacía es un dato que el inversor no envió.

## `meteo/<localidad>/<serie>.csv.gz`

Una serie por producto meteorológico y localidad (G01 a G14), en el punto de la localidad.

| Serie | Fuente | Columnas propias |
|---|---|---|
| `om_satellite__eumetsat_lsa_saf_msg` | Satélite MSG (EUMETSAT LSA SAF) por Open-Meteo | `modelo` |
| `om_archive__ecmwf_ifs` | Análisis ECMWF IFS por Open-Meteo | `modelo` |
| `om_archive__era5` | Reanálisis ERA5 por Open-Meteo | `modelo` |
| `om_archive__era5_climatologia` | ERA5 de 2015 a 2024, para la climatología del pronóstico a 30 días | `modelo` |
| `om_previous_runs__ecmwf_ifs025`, `om_previous_runs__gfs_seamless` | Pronósticos archivados IFS 0,25° y GFS, de 1 a 7 días de antelación | `modelo`, `previousday` (días de antelación) |
| `om_single_runs__ecmwf_ifs` | Ejecuciones completas del IFS (00 y 12 UTC) | `modelo`, `run` (inicio de la ejecución) |
| `om_historical_forecast__gfs_seamless`, `om_historical_forecast__icon_seamless` | Pronóstico histórico GFS e ICON | `modelo` |
| `om_air_quality__cams_global` | Aerosoles CAMS | `modelo` |
| `nasa_power` | NASA POWER, horario | |
| `nsrdb_goes` | NSRDB GOES (NREL), cada 10 minutos, hasta 2025 | |
| `ideam` | Estaciones automáticas del IDEAM a menos de 15 km, en formato largo | `codigoestacion`, `codigosensor`, `descripcionsensor`, `unidadmedida`, `dataset`, `valorobservado`, `distanciakm` |

El resto de columnas son variables meteorológicas con el nombre de su proveedor (por ejemplo `shortwave_radiation`,
la irradiancia global horizontal en W/m²). `parametros/variables_meteo.csv` da la unidad, el tipo de valor (media,
instantáneo o acumulado), la marca de tiempo y la equivalencia entre fuentes de cada una.

## `parametros/`

| Archivo | Contenido |
|---|---|
| `plantas.csv` | Una fila por planta: localidad, coordenadas, altitud, región, tipo de instalación, paneles, módulo, potencia por panel, potencia pico y datos del módulo |
| `dispositivos.csv` | Una fila por inversor: modelo y potencia nominal, potencia pico conectada, inclinación y azimut de cada cara (`caras`, lista de inclinación, azimut y fracción de paneles), coeficiente de temperatura, NOCT y bifacialidad del módulo |
| `modulos_fv.csv` | Catálogo de módulos transcrito de sus fichas técnicas (`datasheets/`) |
| `eventos.csv` | Eventos que declaró el instalador: control de exportación con su fecha de conformidad del operador de red, si sigue activo, reparaciones y strings desconectados |
| `etapas.csv` | Etapas de capacidad de las plantas (paneles desde una fecha) |
| `nomenclatura.csv` | Código de cada inversor, su planta, su número y la fecha de su primer dato |
| `grupos.csv`, `plantas_grupos.csv` | Localidades (grupos meteorológicos) y la localidad de cada planta |
| `celdas.csv` | Coordenadas y elevación de la celda que devolvió cada fuente en cada localidad |
| `series.csv` | Lista de las series: archivos, variables, filtros, rango, filas y huella |
| `columnas.csv` | Columnas de cada tabla de la base de datos, con su tipo |
| `variables_meteo.csv` | Catálogo de las variables meteorológicas |
| `oni.csv` | Índice ONI mensual de NOAA CPC (fase de El Niño y La Niña) |

## Licencias

| Datos | Licencia |
|---|---|
| Telemetría y parámetros de las plantas | CC BY 4.0 |
| Open-Meteo (ERA5, ECMWF IFS, GFS, ICON, EUMETSAT, CAMS) | CC BY 4.0, con atribución a Open-Meteo y a cada proveedor |
| NASA POWER | Datos públicos de NASA, con su cita |
| NSRDB (NREL) | Datos abiertos (CC BY 3.0 US en el registro de datos abiertos de AWS), con atribución a DOE/NREL |
| IDEAM | CC BY-SA 4.0 (datos abiertos de Colombia, no validados por el IDEAM) |
| NOAA CPC (índice ONI) | Dominio público |