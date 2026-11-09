"""
graficos.py
===========
Gráficos con un estilo uniforme y exportación directa a la carpeta de figuras del documento LaTeX
(PDF para el documento y PNG para revisar). Cada figura puede guardar, además, la tabla con sus valores
en SX/resultados/datos_figuras/<nombre>.csv, para analizarla con herramientas externas.

Convenciones:
    * Ejes de tiempo en hora de Colombia (UTC-5), con fecha completa (AAAA-MM-DD HH:MM, AAAA-MM-DD o
      AAAA-MM según el rango), etiquetas verticales y letra pequeña.
    * Colores categoricos en orden fijo (config.PALETA, validada para daltonismo); variables de la misma
      naturaleza con un solo tono y distinta intensidad. Las series superpuestas llevan además marcador y
      tipo de línea propios, para que dos series que coinciden se distingan.
"""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.colors import ListedColormap

from . import config

ETIQUETA_TIEMPO = "Hora de Colombia (UTC−5)"
ETIQUETA_FECHA = "Fecha (Colombia, UTC−5)"


def _png_cuaderno(fig: plt.Figure) -> str | None:
    """Imagen de una figura en la salida del cuaderno: a config.DPI_CUADERNO y con una paleta de
    config.COLORES_CUADERNO colores, sin tramado. Se ve igual en pantalla y pesa una cuarta parte del PNG de color
    completo."""
    import base64
    import io
    from PIL import Image
    if not fig.axes and not fig.texts:
        return None
    crudo = io.BytesIO()
    fig.canvas.print_figure(crudo, format="png", dpi=config.DPI_CUADERNO, bbox_inches="tight",
                            facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    imagen = Image.open(crudo).convert("RGB").quantize(config.COLORES_CUADERNO, dither=Image.Dither.NONE)
    salida = io.BytesIO()
    imagen.save(salida, "PNG", optimize=True)
    return base64.b64encode(salida.getvalue()).decode("ascii")


def _salida_compacta() -> None:
    # Dentro de un cuaderno, las figuras se muestran con _png_cuaderno en lugar del PNG de matplotlib
    try:
        from IPython import get_ipython
    except ImportError:
        return
    ip = get_ipython()
    if ip is None:
        return
    # El backend inline registra sus propios formatos al activarse, con la primera figura: se activa antes para que
    # no reemplace a este
    plt.close(plt.figure())
    ip.display_formatter.formatters["image/png"].for_type(plt.Figure, _png_cuaderno)


def estilo() -> None:
    sns.set_theme(style="whitegrid", palette=config.PALETA)
    _salida_compacta()
    plt.rcParams.update({
        "figure.dpi": config.DPI_CUADERNO, "savefig.dpi": 200, "axes.titleweight": "bold", "figure.titleweight": "bold",
        "figure.figsize": (11, 4),
        "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8,
        # Sin esto matplotlib rotula las fechas en UTC aunque la serie este en hora local
        "timezone": config.ZONA,
        "axes.prop_cycle": cycler(color=config.PALETA) + cycler(linestyle=config.LINEAS),
        "grid.linewidth": 0.5, "grid.alpha": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3,
        "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.hspace": 0.04,
    })


# --- Notación matemática en ejes, títulos y leyendas -------------------------------------------------
# Se aplica al guardar cada figura: los símbolos escritos como texto plano en los cuadernos ("Yf", "PR",
# "k*") pasan a notación matemática de matplotlib (mathtext, sin instalar LaTeX).
SIMBOLOS = [
    (r"corr\(T_IDEAM,t ; T_IFS,t[−-]k\)", r"$\rho(T_{\mathrm{IDEAM},t},\,T_{\mathrm{IFS},t-k})$"),
    (r"corr\(Yf_t, Yr_\{t-k\}\)", r"$\rho(Y_{f,t},\,Y_{r,t-k})$"),
    (r"\bYr_cs\b", r"$Y_{r,\mathrm{cs}}$"),
    (r"\bYf\b", r"$Y_f$"),
    (r"\bYr\b", r"$Y_r$"),
    (r"\bPR\b", r"$PR$"),
    (r"\bk\*", r"$k^*$"),
    (r"\bP_AC\b", r"$P_{\mathrm{AC}}$"),
    (r"\bG_POA\b", r"$G_{\mathrm{POA}}$"),
    (r"\bP0\b", r"$P_0$"),
    (r"\bR²", r"$R^2$"),
]


def coma_decimal(texto: str) -> str:
    """Coma decimal en las cifras de un texto, como en el documento (26.7 -> 26,7). No toca lo que esta entre $
    (notación matemática), ni fechas, horas, códigos o números en notación científica."""
    import re
    if not texto or "." not in texto:
        return texto
    partes = texto.split("$")
    for i in range(0, len(partes), 2):           # los índices pares quedan fuera de las expresiones $...$
        # Los números de sección ("seccion 12.1") no son cifras: se dejan con punto
        trozos = re.split(r"((?:[Ss]ecci[oó]n(?:es)?|[Aa]partados?)\s+\d+(?:\.\d+)+)", partes[i])
        partes[i] = "".join(tr if k % 2 else re.sub(r"(?<![\w.,])(\d+)\.(\d+)(?!\w|\.\d)", r"\1,\2", tr)
                            for k, tr in enumerate(trozos))
    return "$".join(partes)


def texto_es(texto: str) -> str:
    """Coma decimal en un texto Markdown de los cuadernos (coma_decimal), sin tocar los fragmentos de código entre
    comillas invertidas ni la notación matemática entre $."""
    if not texto or "." not in texto:
        return texto
    partes = str(texto).split("`")
    for i in range(0, len(partes), 2):           # los índices pares quedan fuera del código
        partes[i] = coma_decimal(partes[i])
    return "`".join(partes)


class FormatoComa(ScalarFormatter):
    """El formateador numérico por defecto de matplotlib, con coma decimal en las marcas de los ejes."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", "{,}" if self.get_useMathText() else ",")

    def get_offset(self):
        return super().get_offset().replace(".", "{,}" if self.get_useMathText() else ",")


def matematico(texto: str, codigos: bool = True) -> str:
    import re
    if not texto:
        return texto
    texto = coma_decimal(texto)
    if "$" in texto:
        return texto
    for patron, reemplazo in SIMBOLOS:
        texto = re.sub(patron, lambda _m, r=reemplazo: r, texto)
    # Códigos (P01I0102, P04, G02) en mecanográfica con serifa: en la fuente sin serifa la I parece una l.
    # En los títulos no se aplica, para que todo el título quede en la misma fuente.
    if codigos:
        texto = re.sub(r"(?<![\w$\\{])(P\d{2}I\d{4}|P\d{2}|G\d{2})(?![\w}])", lambda m: r"$\mathtt{" + m.group(1) + "}$", texto)
    return texto


def matematizar(fig: plt.Figure) -> None:
    """Aplica la notación matemática a todos los textos de la figura."""
    for ax in fig.axes:
        # Se cambia solo el texto, sin tocar el tamaño ni la posición que fijo cada gráfico
        for txt in (ax.title, ax._left_title, ax._right_title):
            txt.set_text(matematico(txt.get_text(), codigos=False))
        for txt in (ax.xaxis.label, ax.yaxis.label):
            txt.set_text(matematico(txt.get_text()))
        leyenda = ax.get_legend()
        if leyenda is not None:
            for txt in leyenda.get_texts():
                txt.set_text(matematico(txt.get_text()))
        # Etiquetas de categorías (códigos de unidades en el eje); los ejes numéricos solo cambian a coma decimal
        for eje in (ax.xaxis, ax.yaxis):
            if isinstance(eje.get_major_locator(), FixedLocator):
                textos = [e.get_text() for e in eje.get_majorticklabels()]
                nuevos = [matematico(s) for s in textos]
                if nuevos != textos:
                    eje.set_ticklabels(nuevos)
            elif type(eje.get_major_formatter()) is ScalarFormatter:
                eje.get_major_formatter().__class__ = FormatoComa      # conserva su estado (desplazamiento, potencias)
        # Rótulos sueltos (medianas, valores sobre las barras): solo la coma decimal
        for txt in ax.texts:
            txt.set_text(coma_decimal(txt.get_text()))
    for leyenda in fig.legends:
        for txt in leyenda.get_texts():
            txt.set_text(matematico(txt.get_text()))
        leyenda.get_title().set_text(matematico(leyenda.get_title().get_text(), codigos=False))
    if fig._suptitle is not None:
        fig._suptitle.set_text(matematico(fig._suptitle.get_text(), codigos=False))


def ampliar_leyenda(ax: plt.Axes, tamano: float) -> None:
    """Agranda el texto de la leyenda sin moverla. Si la leyenda estaba en la posición automática ("best"),
    se fija en la esquina que ocupaba con el tamaño original, para que al crecer no salte a otro lugar."""
    leyenda = ax.get_legend()
    if leyenda is None:
        return
    loc, anclaje = leyenda._loc, leyenda._bbox_to_anchor
    if loc == 0 and anclaje is None:
        ax.figure.canvas.draw()
        (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(leyenda.get_window_extent())
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        vertical = "upper" if cy > 2 / 3 else "lower" if cy < 1 / 3 else "center"
        horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        loc = "center" if vertical == horizontal == "center" else f"{vertical} {horizontal}"
    ax.legend(leyenda.legend_handles, [t.get_text() for t in leyenda.get_texts()], loc=loc, bbox_to_anchor=anclaje,
              fontsize=tamano, ncols=leyenda._ncols, frameon=leyenda.get_frame_on())


def color_de(nombre: str) -> str:
    """Color según la naturaleza de la variable (config.COLOR_VARIABLE), por palabras clave del nombre."""
    import re
    n = nombre.lower()
    # Las claves cortas (yf, pr, aod...) solo cuentan como palabra: "pr" no debe coincidir con "presión"
    coincide = lambda c: re.search(rf"(?<![a-záéíóúñ]){re.escape(c)}(?![a-záéíóúñ])", n) if len(c) <= 3 else c in n
    for claves, color in config.COLOR_VARIABLE:
        if any(coincide(c) for c in claves):
            return color
    return config.COLOR_NEUTRO


def estilos_series(n: int) -> list[dict]:
    """Color, marcador y tipo de línea para n series superpuestas (en orden fijo, nunca ciclico)."""
    if n > len(config.PALETA):
        raise ValueError(f"{n} series superan los {len(config.PALETA)} colores categoricos: agrupe o use paneles")
    return [{"color": config.PALETA[i], "marker": config.MARCADORES[i], "linestyle": config.LINEAS[i]} for i in range(n)]


def tonos(color: str, n: int, claro: float = 0.75) -> list:
    """n intensidades de un mismo tono (de claro a oscuro), para variables de la misma naturaleza."""
    base = np.array(plt.matplotlib.colors.to_rgb(color))
    return [tuple(base + (1 - base) * claro * (1 - i / max(1, n - 1))) for i in range(n)]


# --- Fechas en los ejes ---------------------------------------------------------------------------

def dias_locales(indice) -> pd.DatetimeIndex:
    """Fechas diarias sin zona como medianoche de Colombia: matplotlib dibuja en UTC las fechas sin zona y las
    corre 5 h a la izquierda de su marca."""
    i = pd.DatetimeIndex(indice)
    return i.tz_localize(config.ZONA) if i.tz is None else i.tz_convert(config.ZONA)


def _es_eje_fecha(eje) -> bool:
    f = eje.get_major_formatter()
    return isinstance(f, (mdates.AutoDateFormatter, mdates.ConciseDateFormatter, mdates.DateFormatter))


# Pasos en días de las marcas de fecha: el menor que deja las etiquetas sin tocarse
_PASOS_DIAS = (1, 2, 3, 4, 7, 14, 21, 28)


def formatear_fechas(fig: plt.Figure, tamano: float = 7, tam_unico: float | None = None) -> bool:
    """Fecha completa en los ejes de tiempo: AAAA-MM-DD HH:MM si el rango es de pocos días, AAAA-MM-DD si
    es de semanas o meses y AAAA-MM si es de varios meses; vertical en X, horizontal en Y. Con `tam_unico` (el
    tamaño del texto de la figura) se ponen como mucho las marcas que caben en el largo del eje sin tocarse.
    Devuelve True si la figura tenía algún eje de fechas."""
    hubo = False
    for ax in fig.axes:
        hermanos = [a for a in ax.get_shared_x_axes().get_siblings(ax) if a.figure is fig]
        inferior = len(hermanos) <= 1 or ax.get_position().y0 <= min(a.get_position().y0 for a in hermanos) + 1e-6
        for eje, x in ((ax.xaxis, True), (ax.yaxis, False)):
            if not _es_eje_fecha(eje):
                continue
            hubo = True
            a, b = (ax.get_xlim() if x else ax.get_ylim())
            dias = abs(b - a)
            caja = ax.get_position()
            largo = (caja.width * fig.get_figwidth() if x else caja.height * fig.get_figheight()) * 72
            n_max = 30 if tam_unico is None else max(4, min(30, int(largo / (1.7 * tam_unico))))
            if dias <= 10:
                fmt, loc = "%Y-%m-%d %H:%M", mdates.AutoDateLocator(minticks=min(10, n_max), maxticks=n_max if x else min(10, n_max),
                                                                   tz=config.ZONA)
            elif dias / _PASOS_DIAS[-1] <= n_max:
                # Marcas a intervalos iguales (un día, dos días, semana, dos semanas...). Un DayLocator con intervalo
                # o con días del mes se reinicia cada mes y deja la cuadrícula irregular
                paso = next(p for p in _PASOS_DIAS if dias / p <= n_max)
                desde = pd.Timestamp(mdates.num2date(min(a, b), tz=config.ZONA)).ceil("D")
                marcas = pd.date_range(desde, pd.Timestamp(mdates.num2date(max(a, b), tz=config.ZONA)), freq=f"{paso}D")
                fmt, loc = "%Y-%m-%d", FixedLocator(mdates.date2num(marcas.to_pydatetime()))
            else:
                meses = dias / 30.44
                fmt, loc = "%Y-%m", mdates.MonthLocator(interval=max(1, int(np.ceil(meses / n_max))), tz=config.ZONA)
            eje.set_major_locator(loc)
            eje.set_major_formatter(mdates.DateFormatter(fmt, tz=config.ZONA))
            for t in eje.get_ticklabels():
                t.set_rotation(90 if x else 0)
                t.set_horizontalalignment("center" if x else "right")
                t.set_fontsize(tamano)
            if x and not inferior:                 # ejes compartidos: la fecha solo en el panel de abajo
                eje.set_tick_params(labelbottom=False)
                ax.set_xlabel("")
                continue
            etiqueta = ax.get_xlabel() if x else ax.get_ylabel()
            if not etiqueta or etiqueta.startswith("Hora de Colombia") or etiqueta in ("datatime", "t", "fecha", "Fecha"):
                (ax.set_xlabel if x else ax.set_ylabel)(ETIQUETA_TIEMPO if fmt.endswith("%M") else ETIQUETA_FECHA, fontsize=8)
    return hubo


# --- Guardar la figura y sus datos ----------------------------------------------------------------

# Palabras que no quedan solas al final de una línea: se pegan a la siguiente ("a CLIPER", "del mejor")
_CORTAS = {"a", "al", "de", "del", "el", "la", "las", "los", "en", "y", "o", "u", "con", "por", "sin", "se"}


def partir(texto: str, ancho: int = 12, salto: str = "\n") -> str:
    """Parte un rótulo largo en líneas de unas `ancho` letras por palabras completas, sin dejar una palabra corta
    sola al final de una línea (para rótulos de ejes, de leyendas y encabezados de tablas)."""
    grupos, pendiente = [], ""
    for p in str(texto).split():
        if grupos and not pendiente and p.startswith("(") and p.endswith(")") and len(p) <= 7:
            grupos[-1] += f" {p}"              # unidades cortas, como (%) o (kWh), van con la palabra anterior
            continue
        if p.lower() in _CORTAS:
            pendiente = f"{pendiente} {p}".strip()
            continue
        grupos.append(f"{pendiente} {p}".strip())
        pendiente = ""
    if pendiente:
        grupos = grupos[:-1] + [f"{grupos[-1]} {pendiente}"] if grupos else [pendiente]
    lineas, actual = [], ""
    for g in grupos:
        if actual and len(actual) + 1 + len(g) > ancho:
            lineas.append(actual)
            actual = g
        else:
            actual = f"{actual} {g}".strip()
    return salto.join([*lineas, actual])


def tamano_texto(fig: plt.Figure, ancho: float = 1.0) -> float:
    """Tamano (pt) que, con la figura reducida a `ancho` x el ancho de texto del documento, se imprime a
    config.TEXTO_IMPRESO_PT."""
    escala = ancho * config.ANCHO_TEXTO_CM / 2.54 / fig.get_figwidth()
    return config.TEXTO_IMPRESO_PT / escala


def _partible(texto: str) -> bool:
    # No se parte un texto con notación matemática que tenga espacios dentro de los $...$
    trozos = texto.split("$")
    return not any(" " in m for m in trozos[1::2])


def unificar_texto(fig: plt.Figure, ancho: float = 1.0, leyenda_max: int = 46, escala_ejes: float = 1.0,
                   partir_titulos: bool = True) -> float:
    """Un solo tamaño para todo el texto de la figura (ver tamano_texto). Los títulos que no caben en el ancho de su
    eje (o de la figura) y los rótulos de leyenda de más de `leyenda_max` letras se parten en líneas. Con
    `escala_ejes` las marcas y los rótulos de los ejes quedan a esa fracción del tamaño único, y con
    `partir_titulos=False` los títulos de los ejes se dejan en una línea. Devuelve el tamaño aplicado."""
    import matplotlib.text as mtext
    tam = tamano_texto(fig, ancho)
    letra = 0.55 * tam                                   # ancho medio de una letra (pt)
    ancho_fig = fig.get_figwidth() * 72
    for ax in fig.axes:
        ax.tick_params(labelsize=tam)          # también las marcas que se creen al dibujar
        for hijo in ax.child_axes:             # ejes secundarios (energía junto al rendimiento, por ejemplo)
            hijo.tick_params(labelsize=tam)
        caja = ax.get_position()
        for titulo in (ax.title, ax._left_title, ax._right_title):
            s = titulo.get_text()
            maximo = int(caja.width * ancho_fig / letra)
            if partir_titulos and s and len(s) > maximo and chr(10) not in s and _partible(s):
                titulo.set_text(partir(s, maximo))
        # Rótulos de los ejes más largos que su eje (paneles bajos o estrechos): se parten para no invadir el vecino
        for rotulo, largo in ((ax.xaxis.label, caja.width * ancho_fig),
                              (ax.yaxis.label, caja.height * fig.get_figheight() * 72)):
            s = rotulo.get_text()
            maximo = max(8, int(0.9 * largo / (letra * escala_ejes)))
            if s and len(s) > maximo and chr(10) not in s and _partible(s):
                rotulo.set_text(partir(s, maximo))
        ley = ax.get_legend()
        if ley is not None:
            for t in ley.get_texts():
                s = t.get_text()
                if len(s) > leyenda_max and chr(10) not in s and _partible(s):
                    t.set_text(partir(s, leyenda_max))
    # Título general y título de una leyenda de figura (hace de título general cuando la leyenda va arriba)
    for sup in [getattr(fig, "_suptitle", None)] + [ley.get_title() for ley in fig.legends]:
        if sup is None or not sup.get_text():
            continue
        s = sup.get_text()
        maximo = int(ancho_fig / letra)
        if len(s) > maximo and chr(10) not in s and _partible(s):
            sup.set_text(partir(s, maximo))
    for t in fig.findobj(mtext.Text):
        if t.get_text():
            t.set_fontsize(tam)
    if escala_ejes != 1.0:
        for ax in fig.axes:
            ax.tick_params(labelsize=tam * escala_ejes)
            ax.xaxis.label.set_fontsize(tam * escala_ejes)
            ax.yaxis.label.set_fontsize(tam * escala_ejes)
    for ax in fig.axes:
        _marcas_sin_choque(fig, ax, tam * escala_ejes)
    return tam


def _marcas_sin_choque(fig: plt.Figure, ax: plt.Axes, tam: float) -> None:
    """Las marcas apiladas (eje Y, o eje X con rótulos verticales) tan juntas que el tamaño único las montaria unas
    sobre otras, como en un mapa de correlación de 150 variables, se reducen lo justo para que no se toquen."""
    caja = ax.get_position()
    for eje, largo in ((ax.yaxis, caja.height * fig.get_figheight() * 72),
                       (ax.xaxis, caja.width * fig.get_figwidth() * 72)):
        rotulos = [t for t in eje.get_ticklabels() if t.get_text()]
        if len(rotulos) < 2:
            continue
        if eje is ax.xaxis and abs(rotulos[0].get_rotation() % 180 - 90) > 30:
            continue                               # rótulos horizontales: los parte o espacia quien hace la figura
        limite = 0.9 * largo / len(rotulos)
        if tam > limite:
            eje.set_tick_params(labelsize=max(limite, 1.0))


def _columnas_leyenda(textos: list[str], tam: float, ancho_pt: float) -> int:
    """Mayor número de columnas con el que la leyenda cabe en `ancho_pt` (la leyenda se llena por columnas)."""
    letra = 0.5 * tam
    anchos = [(2.0 + 0.8) * tam + max(len(linea) for linea in s.split(chr(10))) * letra for s in textos]
    ncols = len(textos)
    while ncols > 1:
        filas = -(-len(textos) // ncols)
        columnas = [max(anchos[j * filas:(j + 1) * filas] or [0.0]) for j in range(ncols)]
        if sum(columnas) + 1.2 * tam * (ncols - 1) <= ancho_pt:
            break
        ncols -= 1
    # Filas parejas: con las mismas filas, el menor número de columnas (6 entradas en 2 filas van de 3 en 3, no 5 y 1)
    filas = -(-len(textos) // ncols)
    return -(-len(textos) // filas)


def leyendas_arriba(fig: plt.Figure, tam: float) -> None:
    """Cada leyenda de un eje pasa encima de su eje, fuera del área de datos, centrada y con las columnas que quepan en
    el ancho del eje, para que nunca tape una serie; el título del eje sube lo que ocupa la leyenda. Los mapas de calor
    no cambian: su escala es la barra de color, a la derecha. Si algo falla, la leyenda se queda como estaba."""
    ancho_fig = fig.get_figwidth() * 72
    movidas = []
    for ax in fig.axes:
        ley = ax.get_legend()
        if ley is None:
            continue
        try:
            textos = [t.get_text() for t in ley.get_texts()]
            if not textos:
                continue
            titulo_ley = ley.get_title().get_text()
            t_ley = config.ESCALA_LEYENDA * tam
            # Un eje que ocupa casi todo el ancho deja a su leyenda usar el de la figura; en paneles lado a lado, solo
            # el de su eje, para no invadir al vecino
            ancho_eje = ax.get_position().width * ancho_fig
            ncols = _columnas_leyenda(textos, t_ley, 0.95 * ancho_fig if ancho_eje > 0.6 * ancho_fig else ancho_eje)
            nueva = ax.legend(ley.legend_handles, textos, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=ncols,
                              frameon=False, fontsize=t_ley, borderaxespad=0.2, columnspacing=1.5, handlelength=2.2,
                              markerscale=ley.markerscale, title=titulo_ley or None, title_fontsize=t_ley)
            movidas.append((ax, nueva))
        except Exception:
            continue
    if not movidas:
        return
    try:
        fig.canvas.draw()
        for ax, ley in movidas:
            alto = ley.get_window_extent().height * 72 / fig.dpi
            ax._set_title_offset_trans(plt.rcParams["axes.titlepad"] + alto + 0.15 * tam)
    except Exception:
        pass


def guardar(fig: plt.Figure, nombre: str, latex: bool = True, datos: pd.DataFrame | pd.Series | None = None,
            ancho: float = 1.0, escala_ejes: float = 1.0, escala_fechas: float = 1.0, leyenda_dentro: bool = False,
            partir_titulos: bool = True, escala_textos: float = 1.0) -> None:
    """Formatea las fechas, unifica el tamaño del texto, lleva las leyendas encima de sus ejes, pone los títulos en
    negrita y guarda la figura en resultados/figuras/<nombre>.pdf y .png y, si se dan, sus valores en
    resultados/datos_figuras/<nombre>.csv. `ancho`: fracción del ancho de texto del documento con que se incluye la
    figura. `escala_ejes`: fracción del tamaño único para las marcas y los rótulos de los ejes; `escala_fechas`, solo
    para las marcas de fecha. `leyenda_dentro`: cada leyenda se queda donde la puso quien hizo la figura (dentro del
    eje) en lugar de subir encima. `partir_titulos=False`: los títulos de los ejes no se parten en líneas."""
    matematizar(fig)
    disposicion = fig.get_layout_engine()
    hay_fechas = formatear_fechas(fig, tam_unico=tamano_texto(fig, ancho))
    tam = unificar_texto(fig, ancho, escala_ejes=escala_ejes, partir_titulos=partir_titulos)
    if escala_fechas != 1.0:
        for ax in fig.axes:
            for eje in (ax.xaxis, ax.yaxis):
                if _es_eje_fecha(eje):
                    eje.set_tick_params(labelsize=tam * escala_fechas)
    if escala_textos != 1.0:
        # Rótulos sueltos dentro de los ejes (medianas, valores sobre las barras)
        for ax in fig.axes:
            for t in ax.texts:
                t.set_fontsize(tam * escala_textos)
    for ax in fig.axes:
        for titulo in (ax.title, ax._left_title, ax._right_title):
            titulo.set_fontweight("bold")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontweight("bold")
    for ley in fig.legends:
        ley.get_title().set_fontweight("bold")
    if leyenda_dentro:
        for ax in fig.axes:
            ley = ax.get_legend()
            if ley is not None:
                for t in [*ley.get_texts(), ley.get_title()]:
                    t.set_fontsize(config.ESCALA_LEYENDA * tam)
    else:
        leyendas_arriba(fig, tam)
    if hay_fechas and disposicion is None:
        # Las etiquetas verticales de fecha ocupan más alto que las originales: se recalcula el reparto
        # para que no se monten sobre el título del panel de abajo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
    if datos is not None:
        guardar_datos(datos, nombre)
    if not latex:
        return
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.png", bbox_inches="tight", dpi=300)   # el documento usa el PNG; el PDF queda de respaldo


def guardar_datos(datos: pd.DataFrame | pd.Series, nombre: str) -> None:
    config.DIR_DATOS_FIGURAS.mkdir(parents=True, exist_ok=True)
    d = datos.to_frame() if isinstance(datos, pd.Series) else datos
    d = d.copy()
    if isinstance(d.index, pd.DatetimeIndex) and d.index.tz is not None:
        d.index = d.index.tz_convert(config.ZONA)       # el CSV queda en hora de Colombia, como la figura
    d.to_csv(config.DIR_DATOS_FIGURAS / f"{nombre}.csv", encoding="utf-8")


def mostrar(datos: pd.DataFrame | pd.Series, todas: bool = False, n: int = 10, decimales: int = 4) -> None:
    """Muestra los valores de una figura: las primeras n filas o todas."""
    from IPython.display import display
    d = datos.to_frame() if isinstance(datos, pd.Series) else datos
    if isinstance(d.index, pd.DatetimeIndex) and d.index.tz is not None:
        d = d.copy()
        d.index = d.index.tz_convert(config.ZONA)
    if not todas:
        print(f"{len(d)} filas x {d.shape[1]} columnas; primeras {min(n, len(d))}:")
    v = (d if todas else d.head(n)).copy()
    num = v.select_dtypes("number").columns                 # redondear solo las columnas numéricas
    v[num] = v[num].round(decimales)
    display(v)


def _local(indice: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return indice.tz_convert(config.ZONA) if indice.tz is not None else indice


# --- Series de tiempo -----------------------------------------------------------------------------

def serie_con_huecos(s: pd.Series, huecos: pd.DataFrame | None = None, titulo: str = "", unidad: str = "",
                     desde=None, hasta=None) -> plt.Figure:
    """Serie regular con los huecos sombreados (rojo: diurnos, gris: nocturnos)."""
    x = s.copy()
    if desde is not None or hasta is not None:
        x = x.loc[desde:hasta]
    fig, ax = plt.subplots()
    ax.plot(_local(x.index), x.to_numpy(), lw=0.6, color=config.COLOR_PRINCIPAL)
    if huecos is not None and not huecos.empty:
        h = huecos[(huecos["fin"] >= x.index.min()) & (huecos["inicio"] <= x.index.max())]
        for _, r in h.iterrows():
            ax.axvspan(_local(pd.DatetimeIndex([r["inicio"]]))[0], _local(pd.DatetimeIndex([r["fin"]]))[0],
                       color="red" if r.get("diurno", True) else "grey", alpha=0.25, lw=0)
    ax.set_title(titulo)
    ax.set_ylabel(unidad)
    return fig


def serie_completa(s: pd.Series, de_dia: pd.Series | None = None, titulo: str = "", unidad: str = "",
                   paso_grafico: str = "1h") -> plt.Figure:
    """Toda la serie, del primer al último dato, con una franja inferior que marca cada hora sin dato
    (solo las horas de sol: de noche el inversor no reporta y eso es normal). La serie se dibuja a paso horario (media) para que
    el gráfico sea legible con meses de datos de 5 minutos."""
    x = s.resample(paso_grafico, label="right", closed="right").mean() if paso_grafico else s
    sol = de_dia.reindex(s.index, method="nearest").fillna(False).astype(bool) if de_dia is not None else pd.Series(True, index=s.index)
    # Porcentaje diario de los instantes de sol sin dato (una hora suelta no se vería en una franja de meses)
    dia = (s.index - pd.Timedelta(seconds=1)).tz_convert(config.ZONA).normalize() if s.index.tz is not None else s.index.normalize()
    falta_dia = (100 * (s.isna() & sol).groupby(dia).sum() / sol.groupby(dia).sum().replace(0, np.nan)).rename("pct_sol_sin_dato")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True, gridspec_kw={"height_ratios": [4, 1.3], "hspace": 0.06})
    ax.plot(_local(x.index), x.to_numpy(), lw=0.4, color=config.COLOR_PRINCIPAL, rasterized=True)
    ax.set_title(titulo)
    ax.set_ylabel(unidad)
    xd = falta_dia.index if falta_dia.index.tz is not None else falta_dia.index.tz_localize(config.ZONA)
    ax2.bar(xd, falta_dia.to_numpy(), width=1.0, color="#C0392B", lw=0)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel("% de sol\nsin dato", fontsize=7)
    return fig


def calendario_cobertura(cobertura: pd.Series, titulo: str = "") -> plt.Figure:
    """Mapa de calor día del mes x mes con el porcentaje de horas de sol esperadas que tienen dato.

    El denominador son las horas de sol del día (cielo despejado > 0); un 100 % significa que todas
    tienen dato."""
    c = cobertura.copy() * 100.0
    c.index = pd.to_datetime(c.index)
    tabla = pd.DataFrame({"mes": c.index.to_period("M").astype(str), "dia": c.index.day, "v": c.to_numpy()}) \
        .pivot(index="mes", columns="dia", values="v")
    fig, ax = plt.subplots(figsize=(12, max(3, 0.28 * len(tabla))))
    sns.heatmap(tabla, cmap="RdYlGn", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "% de horas de sol esperadas con dato"})
    ax.set_title(titulo)
    ax.set_xlabel("Día del mes")
    ax.set_ylabel("Mes (AAAA-MM)")
    return fig


def tabla_dia_hora(s: pd.Series, horas: tuple[int, int] | None = None) -> pd.DataFrame:
    """Matriz hora local (filas) x día local (columnas) con la media horaria de la serie."""
    loc = s.copy()
    loc.index = _local(loc.index)
    h = loc.groupby([loc.index.normalize(), loc.index.hour]).mean().unstack(0)
    dias = pd.date_range(h.columns.min(), h.columns.max(), freq="D", tz=h.columns.tz)
    h = h.reindex(index=range(24), columns=dias)
    if horas is not None:
        h = h.loc[horas[0]:horas[1]]
    h.index.name = "hora_local"
    return h


def mapa_dia_hora(s: pd.Series, titulo: str = "", unidad: str = "", horas: tuple[int, int] | None = (5, 19),
                  noche: pd.Series | None = None, vmax: float | None = None, cmap: str | None = None,
                  ax: plt.Axes | None = None, cax: plt.Axes | None = None) -> plt.Figure:
    """Mapa de calor día (eje X) x hora local (eje Y), como la Figura 5 de Energies 12(22):4402.

    Blanco: falta el dato en una hora de sol. Gris claro: noche sin dato (el inversor se apaga de noche y
    no reporta, lo que es normal). `horas` = (5, 19) muestra de 05:00 a 19:00, que cubre el amanecer más
    temprano y el ocaso más tardío del año a 11 grados N; horas=None muestra las 24 h."""
    m = tabla_dia_hora(s, horas)
    es_noche = None
    if noche is not None:
        es_noche = tabla_dia_hora(noche.astype(float), horas).reindex_like(m).fillna(1.0) > 0.5
    fig = ax.figure if ax is not None else plt.figure(figsize=(12, 3.8 if horas else 5))
    ax = ax or fig.add_subplot(111)
    cols = m.columns if m.columns.tz is not None else m.columns.tz_localize(config.ZONA)
    x = mdates.date2num(cols.tz_convert(config.ZONA).to_pydatetime())     # medianoche local, como instante
    x = np.r_[x, x[-1] + 1] if len(x) else x
    y = np.r_[m.index.to_numpy(), m.index[-1] + 1]
    ax.set_facecolor("white")
    ax.xaxis_date(tz=config.ZONA)      # antes de dibujar: si va después, reinicia los límites
    if es_noche is not None:
        gris = np.where(es_noche.to_numpy() & m.isna().to_numpy(), 1.0, np.nan)
        ax.pcolormesh(x, y, np.ma.masked_invalid(gris), cmap=ListedColormap(["#D9D9D9"]), vmin=0, vmax=1, shading="flat")
    mapa = plt.get_cmap(cmap or config.MAPA_CALOR).copy()
    mapa.set_bad(alpha=0.0)
    q = ax.pcolormesh(x, y, np.ma.masked_invalid(m.to_numpy()), cmap=mapa, vmin=0,
                      vmax=vmax if vmax is not None else np.nanpercentile(m.to_numpy(), 99.5), shading="flat",
                      rasterized=True)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[-1], y[0])
    ax.set_yticks(m.index.to_numpy()[:: 2 if horas is None else 1] + 0.5)
    ax.set_yticklabels([f"{int(h):02d}:00" for h in m.index.to_numpy()[:: 2 if horas is None else 1]], fontsize=7)
    ax.set_ylabel("Hora local (inicio)")
    ax.set_title(titulo)
    if cax is not None:
        fig.colorbar(q, cax=cax, label=unidad)
    else:
        fig.colorbar(q, ax=ax, label=unidad, pad=0.01, fraction=0.03)
    return fig


def panel_diario_y_mapa(s: pd.Series, diario: pd.Series, titulo: str = "", unidad: str = "",
                        unidad_diaria: str = "", horas: tuple[int, int] | None = (5, 19),
                        noche: pd.Series | None = None, vmax: float | None = None,
                        en_regimen: pd.Series | None = None) -> plt.Figure:
    """Dos filas: arriba la magnitud diaria (línea; los días sin dato cortan la línea) y abajo el mapa de
    calor día x hora de la misma serie, con el eje de tiempo compartido. `en_regimen` (booleana diaria) sombrea
    en las dos filas los días fuera del régimen de capacidad estable: la línea no los dibuja y el mapa si, así
    que el sombreado explica la diferencia."""
    fig = plt.figure(figsize=(12, 7.2 if horas is None else 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 2.4 if horas is None else 2.2], width_ratios=[1, 0.018])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    fig.add_subplot(gs[0, 1]).set_visible(False)
    cax = fig.add_subplot(gs[1, 1])
    d = diario.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(d.index))
    d.index = idx.tz_localize(config.ZONA) if idx.tz is None else idx.tz_convert(config.ZONA)
    d = d.asfreq("D")
    ax1.plot(mdates.date2num(d.index.to_pydatetime()) + 0.5, d.to_numpy(), lw=1.0, color=color_de(unidad_diaria or titulo))
    ax1.set_ylabel(unidad_diaria, fontsize=8)
    fig.suptitle(titulo, fontweight="bold")      # sobre la figura: deja libre el espacio encima del panel para la leyenda
    plt.setp(ax1.get_xticklabels(), visible=False)
    mapa_dia_hora(s, "", unidad, horas, noche, vmax, ax=ax2, cax=cax)
    if en_regimen is not None:
        r = en_regimen.copy()
        ir = pd.DatetimeIndex(pd.to_datetime(r.index))
        r.index = ir.tz_localize(config.ZONA) if ir.tz is None else ir.tz_convert(config.ZONA)
        # Un día sin datos no esta fuera del régimen: hereda el estado del día anterior (el régimen es un tramo continuo)
        fuera = ~r.asfreq("D").astype("boolean").ffill().fillna(False).astype(bool)
        bloques = (fuera != fuera.shift()).cumsum()[fuera]
        for k, (_, b) in enumerate(bloques.groupby(bloques)):
            x0 = mdates.date2num(b.index[0].to_pydatetime())
            x1 = mdates.date2num(b.index[-1].to_pydatetime()) + 1
            for a in (ax1, ax2):
                a.axvspan(x0, x1, facecolor="none", edgecolor=config.COLOR_NEUTRO, hatch="///", lw=0, alpha=0.5,
                          label="fuera del régimen de capacidad estable" if (a is ax1 and k == 0) else None, zorder=0.5)
        if len(bloques):
            # La leyenda va encima del panel, a la derecha, fuera de la serie
            ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, borderaxespad=0.2)
    return fig


def objetivo_y_exogenas(paneles: dict[str, tuple[pd.Series | pd.DataFrame, str]], titulo: str = "",
                        alto_panel: float = 2.8) -> plt.Figure:
    """Paneles apilados con el eje de tiempo compartido: el objetivo arriba y cada exógena debajo, cada
    variable con su propio eje (nunca dos escalas en el mismo eje). paneles = {nombre: (serie o tabla, unidad)};
    si se da una tabla, sus columnas son variables de la misma naturaleza y van en tonos del mismo color."""
    n = len(paneles)
    fig, axes = plt.subplots(n, 1, figsize=(12, alto_panel * n + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, (nombre, (datos, unidad))) in enumerate(zip(axes, paneles.items())):
        color = color_de(nombre)
        if isinstance(datos, pd.DataFrame):
            for (col, serie), tono, ls in zip(datos.items(), tonos(color, datos.shape[1], 0.55)[::-1], config.LINEAS):
                ax.plot(_local(serie.index), serie.to_numpy(), lw=1.2, color=tono, linestyle=ls, label=col)
            ax.legend(frameon=False)                 # guardar la lleva encima del panel
        else:
            ax.plot(_local(datos.index), datos.to_numpy(), lw=1.2, color=color)
        ax.set_ylabel(f"{nombre}\n({unidad})" if unidad else nombre, fontsize=8)
    axes[0].set_title(titulo)
    return fig


def dispersion_coloreada(x: pd.Series, y: pd.Series, c: pd.Series, xlabel: str, ylabel: str, clabel: str,
                         titulo: str = "") -> plt.Figure:
    d = pd.concat([x.rename("x"), y.rename("y"), c.rename("c")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    q = ax.scatter(d["x"], d["y"], c=d["c"], s=2.5, alpha=0.6, cmap="viridis", rasterized=True)
    fig.colorbar(q, ax=ax, label=clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo or f"r = {d['x'].corr(d['y']):.3f} (n = {len(d)})")
    return fig


def boxplot_por(s: pd.Series, por: str = "hora", titulo: str = "", unidad: str = "") -> plt.Figure:
    """Boxplot por hora local del día o por mes."""
    loc = s.dropna()
    idx = _local(loc.index)
    grupo = idx.hour if por == "hora" else idx.month
    fig, ax = plt.subplots()
    sns.boxplot(x=grupo, y=loc.to_numpy(), ax=ax, color=config.COLOR_PRINCIPAL, fliersize=1.5, linewidth=0.8)
    ax.set_xlabel("Hora local" if por == "hora" else "Mes")
    ax.set_ylabel(unidad)
    ax.set_title(titulo)
    return fig


def original_vs_imputada(original: pd.Series, imputada: pd.Series, motivo: pd.Series, titulo: str = "",
                         unidad: str = "", desde=None, hasta=None) -> plt.Figure:
    """Serie original (con huecos) y serie imputada superpuestas; los puntos imputados se resaltan."""
    o, i, m = (x.loc[desde:hasta] for x in (original, imputada, motivo))
    fig, ax = plt.subplots()
    ax.plot(_local(i.index), i.to_numpy(), lw=1.6, color=config.COLOR_SECUNDARIO, label="imputada", alpha=0.6, linestyle="--")
    ax.plot(_local(o.index), o.to_numpy(), lw=0.9, color=config.COLOR_PRINCIPAL, label="original")
    for etiqueta, color, marca in (("interpolado", "#2E8B57", "o"), ("noche", "grey", "s")):
        sel = m == etiqueta
        if sel.any():
            ax.scatter(_local(i.index[sel]), i[sel], s=5, color=color, marker=marca, label=etiqueta, zorder=3)
    ax.legend(loc="upper right")
    ax.set_title(titulo)
    ax.set_ylabel(unidad)
    return fig


def antes_despues_agregacion(nativa: pd.Series, agregada: pd.Series, titulo: str = "", unidad: str = "",
                             desde=None, hasta=None) -> plt.Figure:
    """Serie en su muestreo nativo y agregada (escalones al final de cada periodo)."""
    n, a = nativa.loc[desde:hasta], agregada.loc[desde:hasta]
    fig, ax = plt.subplots()
    ax.plot(_local(n.index), n.to_numpy(), lw=0.6, color=config.COLOR_PRINCIPAL, label="nativa")
    ax.step(_local(a.index), a.to_numpy(), where="pre", lw=1.4, color=config.COLOR_SECUNDARIO, linestyle="--",
            label="agregada (etiqueta al final)")
    ax.legend(loc="upper right")
    ax.set_title(titulo)
    ax.set_ylabel(unidad)
    return fig


def lineas(tabla: pd.DataFrame, titulo: str = "", xlabel: str = "", ylabel: str = "", ax: plt.Axes | None = None,
           anotar_coincidentes: bool = True) -> plt.Figure:
    """Una línea por columna con marcador y tipo de línea propios (para pocas series y pocos puntos, por
    ejemplo el error por duración del hueco o por antelación). Si dos series coinciden en todos sus puntos
    se anota en la leyenda, para que no parezca que falta una."""
    fig = ax.figure if ax is not None else plt.figure(figsize=(9, 4))
    ax = ax or fig.add_subplot(111)
    estilos = estilos_series(tabla.shape[1])
    x = np.arange(len(tabla)) if not np.issubdtype(np.asarray(tabla.index).dtype, np.number) else tabla.index
    for (col, serie), e, k in zip(tabla.items(), estilos, range(tabla.shape[1])):
        etiqueta = str(col)
        if anotar_coincidentes:
            iguales = [str(c) for c in tabla.columns[:k] if np.allclose(tabla[c].to_numpy(), serie.to_numpy(), equal_nan=True, rtol=0.005)]
            if iguales:
                etiqueta += f" (coincide con {', '.join(iguales)})"
        ax.plot(x, serie.to_numpy(), color=e["color"], marker=e["marker"], linestyle=e["linestyle"],
                markersize=5 if k % 2 else 3, markerfacecolor="none" if k % 2 else e["color"],
                markeredgewidth=1.0, lw=1.6, alpha=0.95, label=etiqueta, zorder=3 + k)
    if not np.issubdtype(np.asarray(tabla.index).dtype, np.number):
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in tabla.index], fontsize=8)
    elif np.issubdtype(np.asarray(tabla.index).dtype, np.integer):
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))          # rezagos, días, profundidades: enteros
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # La posición se decide con el tamaño de 7 puntos y se conserva al agrandar el texto a 9,5
    ax.legend(fontsize=7)
    ampliar_leyenda(ax, 9.5)
    return fig


def mapa_correlacion(df: pd.DataFrame, metodo: str = "pearson", titulo: str = "", anotar: bool | None = None) -> plt.Figure:
    c = df.corr(method=metodo)
    n = len(c)
    lado = min(22, 1 + 0.32 * n) if n > 18 else 1 + 0.55 * n
    fig, ax = plt.subplots(figsize=(lado, lado * 0.9))
    sns.heatmap(c, annot=(n <= 18) if anotar is None else anotar, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                ax=ax, square=True, cbar_kws={"shrink": 0.7, "label": f"correlación de {metodo}"},
                xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=6 if n > 30 else 8)
    ax.set_title(titulo or f"Correlación de {metodo}")
    return fig


def dispersion(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, titulo: str = "") -> plt.Figure:
    a, b = x.align(y, join="inner")
    ok = a.notna() & b.notna()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(a[ok], b[ok], s=1.5, alpha=0.3, color=config.COLOR_PRINCIPAL, rasterized=True)
    lim = [0, np.nanmax([a[ok].max(), b[ok].max()])]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo or f"r = {a[ok].corr(b[ok]):.3f}")
    return fig


def diagnostico_residuos(res: pd.Series, titulo: str = "", rezagos: int = 48, mascara: pd.Series | None = None) -> plt.Figure:
    """Diagnostico de residuos en cuatro paneles (como plot_diagnostics de statsmodels): residuos en el
    tiempo, histograma con la normal ajustada, gráfico Q-Q normal y ACF de los residuos.

    máscara: instantes que entran en la serie, el histograma y el Q-Q (por ejemplo, las horas de sol: de
    noche el residuo es 0 y deformaria la distribución). La ACF usa siempre la serie completa, porque
    necesita los rezagos consecutivos."""
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf
    completa = res.dropna()
    r = completa[mascara.reindex(completa.index).fillna(False).astype(bool)] if mascara is not None else completa
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    ax[0, 0].plot(_local(r.index) if isinstance(r.index, pd.DatetimeIndex) else r.index, r.to_numpy(), lw=0.4,
                  color=config.COLOR_PRINCIPAL, rasterized=True)
    ax[0, 0].set_title("Residuos estandarizados" if abs(r.std() - 1) < 0.2 else "Residuos")
    ax[0, 1].hist(r.to_numpy(), bins=60, density=True, color=config.COLOR_TENUE, edgecolor="white")
    x = np.linspace(r.min(), r.max(), 200)
    ax[0, 1].plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=config.COLOR_SECUNDARIO, label="normal")
    ax[0, 1].legend(); ax[0, 1].set_title("Histograma y normal ajustada")
    stats.probplot(r.to_numpy(), dist="norm", plot=ax[1, 0])
    ax[1, 0].get_lines()[0].set(markersize=1, color=config.COLOR_PRINCIPAL)
    ax[1, 0].get_lines()[1].set(color="black")
    ax[1, 0].set_title("Q-Q normal")
    ax[1, 0].set_xlabel("Cuantiles teóricos de la normal")
    ax[1, 0].set_ylabel("Cuantiles de los residuos")
    plot_acf(completa.to_numpy(), lags=rezagos, ax=ax[1, 1], title=f"ACF de los residuos ({rezagos} rezagos, serie completa)")
    ax[1, 1].set_xlabel("Rezago (h)")
    fig.suptitle(titulo)
    return fig


# --- Pronóstico diario a 30 días ------------------------------------------------------------------

def pronostico_30d(t: pd.DataFrame, observado: pd.Series | None = None, titulo: str = "", kwp: float | None = None,
                   mostrar_pr: bool = True, escenarios: pd.DataFrame | None = None,
                   pr_observado: pd.Series | None = None) -> plt.Figure:
    """Pronostico diario de D+1 a D+30: Yf como línea con su intervalo del 80 %, lo observado como línea con
    marcadores (cortada donde falta un día), la climatología como referencia y, en un panel aparte (mostrar_pr=False
    lo quita), el PR con los mismos elementos: valor central, intervalo del 80 %, climatología y observado.
    El eje derecho del panel de Yf da la energía (kWh = Yf x kWp): es la misma magnitud en otra unidad.
    escenarios: trayectorias de años análogos (una columna por año), en líneas finas: muestran la variación de un
    día a otro que el valor central promedia."""
    # Los dos paneles con el mismo alto; las leyendas van encima de cada panel (guardar), fuera de los datos
    filas = 2 if mostrar_pr else 1
    fig, axes = plt.subplots(filas, 1, figsize=(12, 2.0 + 3.35 * filas), sharex=True)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    x = pd.DatetimeIndex(t.index).tz_localize(config.ZONA) if pd.DatetimeIndex(t.index).tz is None else t.index
    h = {}
    for metodo, color, marca, etiqueta, banda in (
            ("modelo", config.COLOR_PRINCIPAL, "o", "SARIMAX diario (1 a 7 días)", "intervalo del 80 % (1 a 7 días)"),
            ("climatologia", config.COLOR_SECUNDARIO, "^", "climatología × PR reciente (8 a 30 días)",
             "intervalo del 80 % (8 a 30 días)")):
        m = (t["metodo"] == metodo).to_numpy()
        if not m.any():
            continue
        h["ip_" + metodo] = ax.fill_between(x[m], t["yf_p10"].to_numpy()[m], t["yf_p90"].to_numpy()[m], color=color, alpha=0.18,
                                            lw=0, label=banda)
        h[metodo], = ax.plot(x[m], t["yf"].to_numpy()[m], color=color, marker=marca, ms=3, lw=2.2, ls="-", label=etiqueta)
    h["clim"], = ax.plot(x, t["yf_climatologia"].to_numpy(), color=config.COLOR_NEUTRO, lw=1.4, ls="--", label="climatología")
    techo_esc = 0.0
    if escenarios is not None and not escenarios.empty:
        e = escenarios.reindex(pd.DatetimeIndex(t.index).normalize())
        xe = pd.DatetimeIndex(e.index).tz_localize(config.ZONA) if pd.DatetimeIndex(e.index).tz is None else e.index
        for k, col in enumerate(e.columns):
            linea, = ax.plot(xe, e[col].to_numpy(), color=config.COLOR_TENUE, lw=0.7, alpha=0.8, zorder=1,
                             label=f"años análogos ({e.columns[0]} a {e.columns[-1]})" if k == 0 else "_esc")
            if k == 0:
                h["esc"] = linea
        techo_esc = float(np.nanpercentile(e.to_numpy(), 95))
    if observado is not None:
        o = observado.reindex(t.index)
        h["obs"], = ax.plot(x, o.to_numpy(), color="black", lw=1.2, ls="-", marker="o", ms=3, zorder=5, label="observado")
    corte = t.index[t["metodo"] == "modelo"]
    if len(corte):
        ax.axvline(x[len(corte) - 1] + pd.Timedelta("12h"), color="grey", lw=0.8, ls=":")
    ax.set_ylabel("Yf (kWh/kWp/día)")
    ax.set_title(titulo)
    techo = np.nanmax([np.nanmax(t["yf_p90"]), np.nanmax(observado.reindex(t.index)) if observado is not None else 0, techo_esc])
    techo = techo if np.isfinite(techo) and techo > 0 else 1.0
    ax.set_ylim(0, 1.08 * techo)
    orden = [k for k in ("modelo", "ip_modelo", "climatologia", "ip_climatologia", "clim", "esc", "obs") if k in h]
    ax.legend([h[k] for k in orden], [h[k].get_label() for k in orden])
    if kwp:
        sec = ax.secondary_yaxis("right", functions=(lambda v: v * kwp, lambda v: v / kwp))
        sec.set_ylabel("Energía (kWh/día, $P_0$ = " + f"{kwp:.1f}".replace(".", ",") + " kWp)")
    if mostrar_pr:
        a = axes[1]
        hp = {}
        for metodo, color, marca in (("modelo", config.COLOR_PRINCIPAL, "o"), ("climatologia", config.COLOR_SECUNDARIO, "^")):
            m = (t["metodo"] == metodo).to_numpy()
            if not m.any():
                continue
            if {"pr_p10", "pr_p90"} <= set(t.columns) and np.isfinite(t["pr_p10"].to_numpy()[m]).any():
                hp["ip"] = a.fill_between(x[m], t["pr_p10"].to_numpy()[m], t["pr_p90"].to_numpy()[m], color=color, alpha=0.18,
                                          lw=0, label="intervalo del 80 %")
            hp[metodo], = a.plot(x[m], t["pr"].to_numpy()[m], color=color, marker=marca, ms=3, lw=1.8,
                                 label="PR pronosticado" if metodo == "modelo" else "PR reciente (8 a 30 días)")
        if "pr_climatologia" in t.columns and np.isfinite(t["pr_climatologia"].to_numpy(dtype=float)).any():
            hp["clim"], = a.plot(x, t["pr_climatologia"].to_numpy(), color=config.COLOR_NEUTRO, lw=1.4, ls="--",
                                 label="PR histórico de la fecha")
        vals = [t["pr"].to_numpy(dtype=float)]
        if pr_observado is not None:
            po = pr_observado.reindex(t.index)
            hp["obs"], = a.plot(x, po.to_numpy(), color="black", lw=1.2, ls="-", marker="o", ms=3, zorder=5, label="observado")
            vals.append(po.to_numpy(dtype=float))
        for c in ("pr_p10", "pr_p90"):
            if c in t.columns:
                vals.append(t[c].to_numpy(dtype=float))
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        a.set_ylabel("PR diario")
        if len(v):
            a.set_ylim(max(0, v.min() - 0.05), min(1.3, v.max() + 0.05))
        orden_pr = [k for k in ("modelo", "climatologia", "ip", "clim", "obs") if k in hp]
        a.legend([hp[k] for k in orden_pr], [hp[k].get_label() for k in orden_pr])
    return fig


def mes_en_curso(observado: pd.Series, pron7: pd.DataFrame, pron30: pd.DataFrame, emitido: pd.Timestamp, titulo: str = "",
                 kwp: float | None = None, mostrar_pr: bool = True, pr_observado: pd.Series | None = None,
                 pr_esperado: pd.Series | None = None, estimado: pd.Series | None = None,
                 dias: pd.DatetimeIndex | None = None) -> plt.Figure:
    """Mes calendario con el pronóstico emitido el día `emitido` frente a lo que ocurrió después.
    1. Hasta tres barras por día, lado a lado: lo observado (o estimado, rayado, en los días sin datos), el
       pronóstico de 1 a 7 días (modelo con el pronóstico meteorológico) y el pronóstico a 30 días (valor central
       climatológico por el PR reciente), cada pronóstico con su intervalo del 80 %. Los días posteriores a la
       emisión que ya ocurrieron muestran lo observado junto a lo pronosticado, con datos que el modelo no vio.
       El eje derecho expresa la misma magnitud en kWh.
    2. Opcional (mostrar_pr): PR diario observado y esperado.
    pron7 y pron30: tablas por día con las columnas yf, yf_p10 y yf_p90."""
    filas = 2 if mostrar_pr else 1
    fig, axes = plt.subplots(filas, 1, figsize=(12, 3.4 + 2.2 * (filas - 1)), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.2][:filas]}, squeeze=False)
    axes = axes[:, 0]
    dias = pd.DatetimeIndex(dias if dias is not None else observado.index)
    xn = mdates.date2num(dias.tz_localize(config.ZONA).to_pydatetime())
    obs = observado.reindex(dias)
    est = (estimado.reindex(dias) if estimado is not None else pd.Series(np.nan, index=dias)).where(obs.isna())
    p7, p30 = pron7.reindex(dias), pron30.reindex(dias)
    # Posición de cada barra: centrada si es la única del día, lado a lado si hay varias (observado o estimado,
    # 1 a 7 días, 30 días)
    ancho = 0.3
    base = obs.fillna(est)
    series = np.vstack([base.to_numpy(), p7["yf"].to_numpy(), p30["yf"].to_numpy()])
    presentes = np.isfinite(series)
    orden = np.cumsum(presentes, axis=0) - 1
    desplaz = (orden - (presentes.sum(axis=0) - 1) / 2) * ancho
    pos = [xn + desplaz[k] for k in range(3)]
    ax = axes[0]
    h = {"obs": ax.bar(pos[0], obs.to_numpy(), width=ancho, color=config.COLOR_PRINCIPAL, label="observado")}
    if est.notna().any():
        h["est"] = ax.bar(pos[0], est.to_numpy(), width=ancho, color="white", edgecolor=config.COLOR_PRINCIPAL, hatch="////", lw=0.8,
                          label="estimado (sin telemetría)")
    for k, (clave, t, color, etiqueta) in enumerate((("p7", p7, config.COLOR_SECUNDARIO, "pronóstico de 1 a 7 días"),
                                                      ("p30", p30, config.COLOR_TENUE, "pronóstico a 30 días")), start=1):
        if t["yf"].notna().any():
            h[clave] = ax.bar(pos[k], t["yf"].to_numpy(), width=ancho, color=color, label=etiqueta)
            err = np.vstack([(t["yf"] - t["yf_p10"]).clip(lower=0).to_numpy(), (t["yf_p90"] - t["yf"]).clip(lower=0).to_numpy()])
            h["ip"] = ax.errorbar(pos[k], t["yf"].to_numpy(), yerr=err, fmt="none", ecolor=config.COLOR_NEUTRO, elinewidth=1.0,
                                  capsize=1.5, label="intervalo del 80 %")
    x_e = mdates.date2num(pd.Timestamp(emitido).tz_localize(config.ZONA).to_pydatetime()) + 0.5
    h["emi"] = ax.axvline(x_e, color=config.COLOR_NEUTRO, ls="--", lw=1.0, label=f"emisión ({pd.Timestamp(emitido):%Y-%m-%d})")
    ax.set_xlim(xn[0] - 0.7, xn[-1] + 0.7)
    # Techo del eje con lo que haya (una serie vacía o sin datos no cuenta, sin avisos de numpy)
    valores = np.concatenate([np.asarray(s, dtype=float).ravel() for s in (p7["yf_p90"], p30["yf_p90"], obs, est)])
    valores = valores[np.isfinite(valores)]
    techo = float(valores.max()) if valores.size else np.nan
    techo = techo if np.isfinite(techo) and techo > 0 else 1.0      # mes sin ningún dato (unidad sin reportes)
    ax.set_ylim(0, 1.08 * techo)
    ax.set_ylabel("Yf (kWh/kWp/día)"); ax.set_title(titulo)
    orden = [k for k in ("obs", "est", "p7", "p30", "ip", "emi") if k in h]
    ax.legend([h[k] for k in orden], [h[k].get_label() for k in orden], fontsize=8, ncols=3, loc="upper left")
    if kwp:
        sec = ax.secondary_yaxis("right", functions=(lambda v: v * kwp, lambda v: v / kwp))
        sec.set_ylabel("kWh/día")
    if mostrar_pr:
        ax = axes[1]
        if pr_observado is not None:
            ax.plot(xn, pr_observado.reindex(dias).where(obs.notna()).to_numpy(), color=color_de("pr"), marker="o", ms=2.5,
                    lw=1.6, label="observado")
        if pr_esperado is not None:
            ax.plot(xn, pr_esperado.reindex(dias).to_numpy(), color=color_de("pr"), marker="s", ms=2.5, lw=1.6, ls="--", alpha=0.7,
                    label="esperado")
        ax.axvline(x_e, color=config.COLOR_NEUTRO, ls="--", lw=1.0)
        ax.set_ylabel("PR diario"); ax.legend(fontsize=8)
    axes[-1].xaxis_date(tz=config.ZONA)
    return fig
