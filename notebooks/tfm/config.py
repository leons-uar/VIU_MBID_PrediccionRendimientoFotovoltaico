"""
config.py
=========
Configuración única del análisis del TFM: rutas, zona horaria, fecha de corte, particiones,
semilla y umbrales. Todos los notebooks y módulos leen de aquí, de modo que un cambio de
criterio queda en un solo lugar y es trazable.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# --- Rutas -------------------------------------------------------------------------------------
# Todo vive dentro del repositorio: los datos crudos que se leen (datos/) y lo que producen los cuadernos
# (resultados/). Las carpetas de salida se crean al escribir, así que basta con clonar y ejecutar.
RAIZ = Path(__file__).resolve().parents[2]                       # raíz del repositorio (notebooks/tfm -> raíz)
DIR_DATOS = RAIZ / "datos"                                        # datos crudos en CSV (telemetría, meteorología, parámetros)
# De donde se leen los datos: "csv" (lo publicado en datos/) o "bd" (la base PostgreSQL del autor, solo para
# construir los CSV y para la operación diaria)
FUENTE_DATOS = os.environ.get("SX_FUENTE_DATOS", "csv").lower()
DIR_CACHE = RAIZ / "autor" / "cache_bd"                           # cache de las lecturas de la base (solo con "bd")
DIR_RESULTADOS = RAIZ / "resultados"                              # métricas, figuras, tablas e intermedios
DIR_INTERMEDIOS = DIR_RESULTADOS / "intermedios"                  # series procesadas, paneles y predicciones (Parquet)
DIR_MODELOS = RAIZ / "modelos_entrenados"                         # modelos guardados para la operación
DIR_FIGURAS = DIR_RESULTADOS / "figuras"                          # PDF y PNG de 300 ppp
DIR_TABLAS = DIR_RESULTADOS / "tablas"
ARCHIVO_RESULTADOS = DIR_RESULTADOS / "resultados.json"           # métricas consolidadas de todos los cuadernos

# --- Tiempo ------------------------------------------------------------------------------------
ZONA = "America/Bogota"                                           # UTC-5, sin horario de verano

# Corte del modelado (fecha local, exclusiva). Todo lo que el análisis aprende o decide (entrenamiento, validación,
# prueba, regímenes, auditoría del P0, limitación de inyección) usa solo datos anteriores: equivale a entrenar a
# primera hora del día del corte. Se cambia sin tocar el código con la variable de entorno SX_FIN_MODELADO
# (p. ej. "2026-12-01"); la operación reentrena con el día 1 de cada mes (src/41_reentrenamiento.py --mensual).
FIN_MODELADO = os.environ.get("SX_FIN_MODELADO", "2026-11-01")
INICIO_DATOS = "2024-01-01"
MESES_PRUEBA = 2
MESES_VALIDACION = 2


def particiones_relativas(fin_modelado: str | None = None) -> dict[str, tuple[str, str]]:
    """Particiones por fecha local (Colombia), ambos extremos incluidos, relativas al corte: prueba = los
    MESES_PRUEBA meses completos anteriores al corte, validación = los MESES_VALIDACION anteriores a la prueba,
    entrenamiento = todo lo anterior (dentro del régimen vigente de cada unidad). "vivo" son los días desde el
    corte: pronósticos emitidos con los modelos del corte frente a lo que se observe después."""
    fin = pd.Timestamp(fin_modelado or FIN_MODELADO)
    ini_prueba = fin - pd.DateOffset(months=MESES_PRUEBA)
    ini_val = ini_prueba - pd.DateOffset(months=MESES_VALIDACION)
    dia = pd.Timedelta(days=1)
    f = lambda t: pd.Timestamp(t).strftime("%Y-%m-%d")  # noqa: E731
    return {"entrenamiento": (INICIO_DATOS, f(ini_val - dia)), "validacion": (f(ini_val), f(ini_prueba - dia)),
            "prueba": (f(ini_prueba), f(fin - dia)), "vivo": (f(fin), f(fin + pd.DateOffset(months=2) - dia))}


PARTICIONES = particiones_relativas()
# Último instante de los datos del modelado (23:59:59 hora de Colombia del día anterior al corte)
CORTE_UTC = (pd.Timestamp(FIN_MODELADO, tz=ZONA) - pd.Timedelta(seconds=1)).tz_convert("UTC")
# Fin (exclusivo) de las observaciones posteriores al corte que se comparan con los pronósticos emitidos en el
# corte. Nunca se usan para ajustar ni para decidir. "" = mañana (incluye hoy); variable de entorno SX_FIN_OBSERVADO.
# Por defecto, el día siguiente al último día completo de los datos publicados (2026-11-06); "hoy" = mañana, para la
# operación diaria con la base de datos.
FIN_OBSERVADO = os.environ.get("SX_FIN_OBSERVADO", "2026-11-07")


def fin_observado() -> str:
    """FIN_OBSERVADO efectivo (fecha local exclusiva)."""
    if FIN_OBSERVADO in ("", "hoy"):
        return (pd.Timestamp.now(tz=ZONA).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return FIN_OBSERVADO


# Fin (exclusivo) de la extracción de los cuadernos: incluye lo observado después del corte, que solo entra en la
# evaluación en vivo (partición "vivo"); las decisiones se restringen a FIN_MODELADO.
FIN_ANALISIS = fin_observado()
# La flota del documento son las plantas con datos antes del fin de septiembre de 2026; las posteriores (P24 a P26)
# solo entran en la operación
ALTA_MAXIMA_FLOTA = "2026-09-30"
# Nombres visibles de los horizontes (documento, tablas, figuras, cuadernos y web). El código conserva las claves
# H2, H3 y H4 (archivos de modelos, resultados.json). H01DH: un día con resolución horaria; H07DD: 7 días con
# resolución diaria; H30DD: 30 días con resolución diaria (y el total del mes).
ETIQUETA_HORIZONTE = {"H2": "H01DH", "H3": "H07DD", "H4": "H30DD"}
# Climatología de ERA5: diez años completos (fin exclusivo)
CLIMATOLOGIA_ERA5 = ("2015-01-01", "2025-01-01")

# --- Reproducibilidad --------------------------------------------------------------------------
SEMILLA = 37

# --- Recursos ----------------------------------------------------------------------------------
# Procesos en paralelo de scikit-learn. Con n_jobs=-1 anidado (un bosque dentro de una búsqueda) cada
# proceso copia el modelo y los datos, y en un equipo de 14 GB agota la memoria: se limita a 6.
N_JOBS = max(1, min(6, (os.cpu_count() or 2) - 2))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(N_JOBS))

# --- Operación ---------------------------------------------------------------------------------
# Cada cuántos días se reestiman los coeficientes de los modelos en producción (1, 7, 15 o 30).
# La tarea programada corre a diario y solo reentrena cuando se cumple este plazo.
REENTRENAR_CADA_X_DIAS = int(os.environ.get("REENTRENAR_CADA_X_DIAS", "30"))


def fijar_semilla(semilla: int = SEMILLA) -> int:
    """Fija la semilla en random, numpy, torch y keras (los que esten instalados)."""
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch
        torch.manual_seed(semilla)
    except ImportError:
        pass
    os.environ.setdefault("KERAS_BACKEND", "torch")     # Keras 3 sobre PyTorch (CPU)
    try:
        import keras
        keras.utils.set_random_seed(semilla)
    except ImportError:
        pass
    return semilla


# --- Constantes físicas y umbrales -------------------------------------------------------------
G_REF = 1.0                        # kW/m2: irradiancia de referencia STC (IEC 61724-1)
UMBRAL_G_POA_PR = 200.0            # W/m2: el PR horario solo se calcula por encima de este valor
UMBRAL_DIA_VALIDO = 0.90           # fracción mínima de muestras diurnas presentes para usar un día
COBERTURA_MINIMA_INTERVALO = 0.80  # fracción mínima de muestras para aceptar un valor agregado
CENIT_MAXIMO_EVALUACION = 85.0     # grados: las métricas solo se calculan con el sol por encima
IMPUTAR_HUECO_MAXIMO = pd.Timedelta("1h")   # huecos diurnos hasta 1 h: interpolación temporal
# Hora observada (la única que cuenta en las métricas y en el entrenamiento del aprendizaje automático): al menos 11
# de sus 12 lecturas de cinco minutos medidas. Una lectura interpolada cambia la media horaria en una doceava parte de
# su propio error, despreciable frente al del pronóstico (experimento de imputación, sección C del cuaderno). Exigir
# las 12 dejaba fuera casi todo octubre de 2026, cuando la plataforma empezó a perder lecturas sueltas (en P01I0102,
# 43 de 288 horas de sol de octubre frente a 269 con este criterio).
FRACCION_IMPUTADA_MAXIMA = 1 / 12


def hora_observada(frac_imputada) -> "pd.Series":
    """True en las horas con como máximo FRACCION_IMPUTADA_MAXIMA de lecturas imputadas (sin dato: no observada)."""
    return pd.Series(frac_imputada).fillna(1) <= FRACCION_IMPUTADA_MAXIMA + 1e-9

# --- Alcance de la fase 1 ----------------------------------------------------------------------
GRUPOS_FASE1 = ("G01", "G02")

# --- Estilo de gráficos (coherente con el naranja de la plantilla VIU) -------------------------
# Paleta categórica en orden fijo (la serie n siempre recibe el color n). Validada para daltonismo
# (separacón CVD >= 11 en pares adyacentes), croma mínimo y contraste >= 3:1 sobre fondo blanco.
COLOR_PRINCIPAL = "#E65113"      # naranja VIU: la serie principal (lo observado o el mejor modelo)
COLOR_SECUNDARIO = "#2F6DB5"     # azul: la segunda serie o el pronóstico
COLOR_TENUE = "#A9C1E0"          # azul claro: elementos de contexto (referencias, barras no destacadas)
COLOR_NEUTRO = "#6B6B6B"         # gris: referencias como CLIPER o la climatología
PALETA = ["#E65113", "#2F6DB5", "#2E8B57", "#8E44AD", "#B8860B", "#16A085", "#C0392B", "#D35FA0"]
MARCADORES = ["o", "^", "s", "D", "v", "P", "X", "*"]
LINEAS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2))]
MAPA_CALOR = "turbo"             # azul -> cian -> verde -> amarillo -> rojo (como Energies 12(22):4402)
# Tamaño único del texto de las figuras, medido en el documento impreso: ejes, marcas, leyendas, títulos y anotaciones.
# Cada figura conserva su ancho y su alto; al guardarla, el texto se escala según la fracción del ancho de texto del
# documento (15 cm) con que se incluye (graficos.guardar, argumento `ancho`).
TEXTO_IMPRESO_PT = 8.0
ANCHO_TEXTO_CM = 15.0
# Las leyendas de los ejes, que van encima de cada gráfico, usan una fracción de ese tamaño: con el mismo tamaño que
# los ejes ocupaban demasiado y desordenaban la figura
ESCALA_LEYENDA = 0.625
# Imágenes dentro de los cuadernos ejecutados: resolución y número de colores. Bastan para leerlas en pantalla y dejan
# cada cuaderno en un peso que GitHub muestra; las figuras del documento se guardan aparte, a 300 ppp
DPI_CUADERNO = 80
COLORES_CUADERNO = 256
DIR_DATOS_FIGURAS = DIR_RESULTADOS / "datos_figuras"   # CSV con los valores de cada figura
DIR_TIEMPOS = DIR_RESULTADOS / "tiempos"


def crear_directorios() -> None:
    for d in (DIR_RESULTADOS, DIR_INTERMEDIOS, DIR_FIGURAS, DIR_TABLAS, DIR_DATOS_FIGURAS, DIR_TIEMPOS):
        d.mkdir(parents=True, exist_ok=True)


# Color por naturaleza de la variable (graficos.color_de): (palabras clave del nombre, color). Siete tonos
# validados para daltonismo (protan, deutan, tritan) sobre fondo blanco; cada variable va en su propio panel
# con su nombre en el eje, así que el color nunca es la única forma de identificarla. La lluvia comparte el
# azul de la nubosidad (misma naturaleza: agua) y los aerosoles van en gris.
COLOR_VARIABLE = [
    (("nubosidad", "cloud", "nube", "lluvia", "precip", "rain"), "#2F6DB5"),   # nubes y lluvia: azul
    (("humedad", "humidity", "rocío", "dew"), "#2BB3D6"),                     # humedad: cian
    (("temperatura", "temperature", "t_modulo", "°c"), "#2E8B57"),            # temperatura: verde
    (("viento", "wind"), "#A070D0"),                                          # viento: violeta
    (("aod", "polvo", "dust", "aerosol"), "#6B6B6B"),                         # aerosoles: gris
    (("pr ", "pr diario", "pr esperado", "pr horario", "índice de rendimiento", "pr"), "#9B1D20"),   # PR: rojo oscuro
    (("yr", "irradian", "ghi", "poa", "g_poa", "kwh/m"), "#D1A000"),          # irradiancia y Yr: amarillo ambar
    (("yf", "rendimiento", "energía", "energia", "kwh", "p_ac", "potencia"), "#E65113"),           # Yf y energía: naranja VIU
]
COLOR_PRONOSTICO = "#2F6DB5"     # fuentes de pronóstico frente a las observadas (naranja) en el experimento de fuentes
