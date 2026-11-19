"""Einheitlicher Stil für alle Grafiken (Folien-tauglich, farbenblind-geprüfte Palette).

Palette: Referenz-Palette der dataviz-Richtlinie (helle Variante).
- Kategorial in fester Reihenfolge (nie zyklisch); Farbe folgt der Entität, nicht dem Rang.
- Sequenziell: ein Farbton (Blau), hell → dunkel.
- Text immer in Text-Farben, nie in Serienfarben. Gitter: feine, durchgezogene Haarlinien.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#b9b8b0"          # „Rest“/ausgeblendet
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

FONTS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": FONTS, "font.size": 11,
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.titlesize": 14, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "legend.frameon": False, "legend.fontsize": 10,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def plain_log_axis(axis) -> None:
    """Log-Achse mit normalen Zahlen (10, 20, 50, 100 …) statt 10² / 3×10²."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    axis.set_minor_formatter(NullFormatter())


def color_map(keys: list[str]) -> dict[str, str]:
    """Feste Zuordnung Entität → Farbe (sortiert, damit sie zwischen Grafiken gleich bleibt)."""
    keys = sorted(keys)
    if len(keys) > len(SERIES):
        raise ValueError("Mehr als 8 Kategorien – zusammenfassen oder aufteilen")
    return {k: SERIES[i] for i, k in enumerate(keys)}


def finish(fig, ax, title: str, subtitle: str | None, source: str | None, path: Path) -> Path:
    """Titel links oben, Untertitel darunter in Sekundärfarbe, Quelle unten links; speichert PNG (200 dpi)."""
    ax.set_title(title, color=TEXT, pad=26 if subtitle else 12)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10, color=TEXT_2)
    if source:
        fig.text(0.01, 0.01, source, ha="left", va="bottom", fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04 if source else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
