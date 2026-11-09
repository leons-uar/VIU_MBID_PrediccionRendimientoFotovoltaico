"""
ejecutar_flota.py
=================
Ejecuta todos los cuadernos en orden, con papermill, desde los datos de datos/:

1. 03_localidad para cada localidad (G01 a G14) con su unidad principal. Esa ejecución registra en
   resultados/resultados.json que otras unidades del grupo se pueden modelar.
2. Una ejecución de 03_localidad por cada una de esas otras unidades (inversores y plantas de varios inversores),
   cada una en su propio proceso para liberar la memoria.
3. Los cuadernos de la flota:
   - 01_calidad_regimenes_flota, cuya ablación usa las decisiones que acaban de registrar las localidades;
   - 02_imputacion;
   - 04_global;
   - 05_bloques, en dos procesos en paralelo y después el resumen;
   - 06_comparacion;
   - 07_vivo.

Los cuadernos ejecutados quedan en notebooks/ejecutados/ (los de cada unidad, en ejecutados/unidades/). Las figuras,
tablas y métricas quedan en resultados/. Un error en una unidad queda en el registro y la secuencia sigue.

Uso, desde la raíz del repositorio:
    python tools/ejecutar_flota.py                todo (unas 29 horas en un portátil de 8 núcleos)
    python tools/ejecutar_flota.py G01            solo esa localidad y sus unidades, sin los cuadernos de la flota
    python tools/ejecutar_flota.py --solo-flota   solo los cuadernos de la flota (paso 3)

Kernel de Jupyter: el de la variable de entorno SX_KERNEL, o "python3".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CUADERNOS = Path(__file__).resolve().parents[1] / "notebooks"
RAIZ = CUADERNOS.parent
RES = RAIZ / "resultados" / "resultados.json"
EJEC = CUADERNOS / "ejecutados"
LOGS = RAIZ / "resultados" / "tiempos" / "ejecuciones"
KERNEL = os.environ.get("SX_KERNEL", "python3")
GRUPOS = [f"G{i:02d}" for i in range(1, 15)]


def log(m: str) -> None:
    linea = f"{datetime.now():%Y-%m-%d %H:%M:%S} {m}"
    print(linea, flush=True)
    with (LOGS / "flota.log").open("a", encoding="utf-8") as f:
        f.write(linea + "\n")


def orden(nombre: str, entrada: str, salida: Path, parametros: dict[str, str]) -> list[str]:
    args = [sys.executable, "-m", "papermill", entrada, str(salida), "-k", KERNEL, "--log-output"]
    for k, v in parametros.items():
        args += ["-p", k, v]
    return args


def papermill(nombre: str, entrada: str, salida: Path, parametros: dict[str, str] | None = None) -> int:
    t0 = time.time()
    log(f"inicio {nombre}")
    with (LOGS / f"papermill_{nombre}.log").open("w", encoding="utf-8") as f:
        # cwd = notebooks/: así `import tfm` encuentra el paquete que esta junto a los cuadernos
        r = subprocess.run(orden(nombre, entrada, salida, parametros or {}), cwd=CUADERNOS, stdout=f, stderr=subprocess.STDOUT)
    log(f"fin {nombre}: codigo {r.returncode}, {(time.time() - t0) / 60:.1f} min")
    return r.returncode


def bloques_en_paralelo() -> list[str]:
    """Validacion por bloques: cada proceso ajusta la mitad de las unidades; después, el resumen."""
    fallas, procesos = [], []
    for k in (1, 2):
        nombre = f"bloques_{k}"
        log(f"inicio {nombre}")
        f = (LOGS / f"papermill_{nombre}.log").open("w", encoding="utf-8")
        args = orden(nombre, "05_bloques.ipynb", EJEC / f"05_bloques_parte{k}.ipynb", {"PARTE": f"{k}/2"})
        procesos.append((nombre, subprocess.Popen(args, cwd=CUADERNOS, stdout=f, stderr=subprocess.STDOUT), f, time.time()))
    for nombre, pr, f, t0 in procesos:
        codigo = pr.wait()
        f.close()
        log(f"fin {nombre}: codigo {codigo}, {(time.time() - t0) / 60:.1f} min")
        if codigo != 0:
            fallas.append(nombre)
    if papermill("bloques", "05_bloques.ipynb", EJEC / "05_bloques.ipynb", {"SOLO_RESUMEN": "True"}) != 0:
        fallas.append("bloques")
    return fallas


def unidades_modelables(grupo: str) -> tuple[str | None, list[str]]:
    d = json.loads(RES.read_text(encoding="utf-8"))
    u = d.get("localidades", {}).get(grupo, {}).get("unidades_modelables", {})
    principal = u.get("principal")
    otras = [n for n, v in u.get("unidades", {}).items() if v.get("modelable") and n != principal]
    return principal, otras


def main() -> None:
    args = [a.upper() for a in sys.argv[1:] if not a.startswith("--")]
    solo_flota = "--solo-flota" in sys.argv
    grupos = [] if solo_flota else (args or GRUPOS)
    flota = solo_flota or not args
    for d in (LOGS, EJEC / "unidades", RES.parent / "respaldos"):
        d.mkdir(parents=True, exist_ok=True)
    if not (RAIZ / "datos" / "MANIFIESTO.sha256").exists():
        raise SystemExit("Faltan los datos: datos/MANIFIESTO.sha256 no existe (ver README, seccion Datos).")

    # Respaldo de resultados.json y limpieza de las claves que se van a regenerar, con el mismo cerrojo entre procesos
    # que usan los cuadernos
    sys.path.insert(0, str(CUADERNOS))
    from tfm.resultados import _Cerrojo
    if not RES.exists():
        RES.write_text("{}", encoding="utf-8")
    sello = datetime.now().strftime("%Y%m%d_%H%M")
    with _Cerrojo(RES):
        shutil.copy2(RES, RES.parent / "respaldos" / f"resultados_{sello}.json")
        d = json.loads(RES.read_text(encoding="utf-8"))
        for g in grupos:
            d.get("localidades", {}).pop(g, None)
        if flota:
            for clave in ("comparacion", "global", "bloques", "vivo"):
                d.pop(clave, None)
        RES.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    fallas = []
    for g in grupos:
        if papermill(g, "03_localidad.ipynb", EJEC / f"03_localidad_{g}.ipynb", {"GRUPO": g}) != 0:
            fallas.append(g)
            continue
        principal, otras = unidades_modelables(g)
        log(f"{g}: principal {principal}; otras unidades modelables: {', '.join(otras) or 'ninguna'}")
        for n in otras:
            if papermill(f"{g}_{n}", "03_localidad.ipynb", EJEC / "unidades" / f"03_localidad_{g}_{n}.ipynb",
                         {"GRUPO": g, "UNIDAD": n}) != 0:
                fallas.append(f"{g}_{n}")
    if flota:
        for nombre, cuaderno in (("calidad_flota", "01_calidad_regimenes_flota.ipynb"), ("imputacion", "02_imputacion.ipynb"),
                                 ("global", "04_global.ipynb")):
            if papermill(nombre, cuaderno, EJEC / cuaderno) != 0:
                fallas.append(nombre)
        fallas += bloques_en_paralelo()
        for nombre, cuaderno in (("comparacion", "06_comparacion.ipynb"), ("vivo", "07_vivo.ipynb")):
            if papermill(nombre, cuaderno, EJEC / cuaderno) != 0:
                fallas.append(nombre)
    log("terminado" + (f"; con errores en: {', '.join(fallas)}" if fallas else " sin errores"))
    if fallas:
        sys.exit(1)


if __name__ == "__main__":
    main()
