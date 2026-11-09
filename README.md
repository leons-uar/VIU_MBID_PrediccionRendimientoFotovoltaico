# Predicción del rendimiento de instalaciones fotovoltaicas distribuidas en Colombia

Código, datos y cuadernos ejecutados del Trabajo Fin de Máster del Máster Universitario en Big Data y Ciencia de
Datos de la Universidad Internacional de Valencia (VIU). El trabajo pronostica el rendimiento específico ($Y_f$) y el
índice de rendimiento ($PR$) de una flota de inversores fotovoltaicos en Colombia con modelos de series temporales y
de aprendizaje automático, alimentados con datos meteorológicos abiertos.

## Cómo ver los resultados sin ejecutar nada

Los cuadernos ejecutados están en `notebooks/ejecutados/` y GitHub los muestra con sus tablas y figuras:

| Cuaderno | Qué hace |
|---|---|
| `01_calidad_regimenes_flota` | Calidad de la telemetría de toda la flota (hora, sensor de red, limitación de inyección, régimen vigente de cada planta y disponibilidad) |
| `02_imputacion` | Experimento de imputación con huecos simulados |
| `03_localidad_G01` a `03_localidad_G14` | Proceso KDD completo de cada localidad, con calidad, imputación, objetivos, fuente de irradiancia, selección de variables, modelos, prueba y pronóstico a 30 días. `ejecutados/unidades/` tiene las demás unidades de cada localidad |
| `04_global` | Modelo global entrenado con todas las unidades (panel) y arranque en frío |
| `05_bloques` | Validación por bloques mensuales con origen móvil |
| `06_comparacion` | Agregación de la flota, que compara localidades, familias de modelos y fuentes |
| `07_vivo` | Pronóstico emitido en el corte frente a lo observado después |

Las métricas consolidadas de todos los cuadernos están en `resultados/resultados.json`.

## Cómo reproducirlo

Requisitos: Python 3.11.

```
python -m venv .venv
.venv\Scripts\activate            (Linux o macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
python tools/ejecutar_flota.py
```

Los cuadernos se desarrollaron para el análisis interactivo de cada localidad. Para reproducir el trabajo completo,
`tools/ejecutar_flota.py` ejecuta la flota de forma parametrizada con papermill, de modo que comprueba los datos a usar y
ejecuta todos los cuadernos en orden con los parámetros de cada localidad y unidad. Los ejecutados quedan en
`notebooks/ejecutados/` y las figuras, tablas y métricas en `resultados/`.

**Duración aproximada:** unas 29 horas en un portátil de 8 núcleos de procesamiento. El cuaderno de una localidad tarda entre 4 y 60 minutos, y una localidad con todas sus unidades, como G01, casi 4 horas.

```
python tools/ejecutar_flota.py G01
```

Cada cuaderno también se puede abrir y ejecutar en Jupyter desde la carpeta `notebooks/`. `03_localidad` es una
plantilla con los parámetros `GRUPO` (G01 a G14) y `UNIDAD` (vacía para la unidad principal).

Todo lo aleatorio usa la semilla 37. El modelado usa los datos hasta el 2026-10-31 (corte), y lo observado llega
hasta el 2026-11-06.

## Datos

`datos/` contiene los datos crudos, tal como los entregaron las API, en CSV comprimido.

- `telemetria/`, con las lecturas de 5 minutos de cada inversor.
- `meteo/<localidad>/`, con los productos meteorológicos de cada localidad.
- `parametros/`, con las constantes de las plantas e inversores, nomenclatura, catálogos y la lista de series.

Ningún valor está corregido, porque la corrección horaria, la imputación y la agregación ocurren dentro de los cuadernos.
`datos/LEEME.md` describe cada archivo y cada columna.
`datos/MANIFIESTO.sha256` guarda la huella de cada archivo, y los cuadernos se detienen si alguna no coincide.

Las plantas y los inversores se identifican con códigos (P01, P01I0102), sin nombres, seriales ni identificadores de
la plataforma. Las coordenadas de las plantas están redondeadas a 0,01°. Los datos de las plantas se usan y publican
con autorización.

Licencias:

| Datos | Licencia |
|---|---|
| Telemetría y parámetros de las plantas | CC BY 4.0 |
| Open-Meteo (ERA5, ECMWF IFS, GFS, ICON, EUMETSAT, CAMS) | CC BY 4.0, con atribución a Open-Meteo y a cada proveedor |
| NASA POWER | Datos públicos de NASA, con su cita |
| NSRDB (NREL) | Datos abiertos (CC BY 3.0 US en el registro de datos abiertos de AWS), con atribución a DOE/NREL |
| IDEAM | CC BY-SA 4.0 (datos abiertos de Colombia, no validados por el IDEAM) |
| NOAA CPC (índice ONI) | Dominio público |

`datasheets/` tiene las fichas técnicas de los módulos y del inversor, publicadas por sus fabricantes, y su
transcripción (`DataSheets.txt`), de la que sale `datos/parametros/modulos_fv.csv`.

## Estructura

```
notebooks/      cuadernos (01 a 07), sus copias ejecutadas y el paquete tfm con las funciones que comparten
tools/          ejecutar_flota.py, que ejecuta todos los cuadernos en orden con papermill
datos/          datos crudos en CSV, parámetros y manifiesto SHA256
datasheets/     fichas técnicas de módulos e inversores
resultados/     resultados.json (al ejecutar se crean también figuras, tablas e intermedios)
sql/            estructura de la base de datos de la que salen los CSV
```

El paquete `tfm` está junto a los cuadernos porque las 14 localidades siguen el mismo proceso, y una sola
implementación evita que diverjan.

## Licencia del código

MIT (ver `LICENSE`).