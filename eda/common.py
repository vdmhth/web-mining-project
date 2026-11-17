"""Shared constants, text helpers and chart style for the EDA scripts."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eda"
FIG = OUT / "figures"
TABLES = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

LABELS = ["E", "S", "C", "I"]
GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
GAIN_AMAZON = {"E": 1.0, "S": 0.01, "C": 0.1, "I": 0.0}

#text rules
NUM_RE = r"\d+(?:\.\d+)?"
UNIT_RE = (r"\d+(?:\.\d+)?\s*(?:gb|tb|mb|oz|ounce|lb|lbs|pound|inch|in|\"|ft|feet|mm|cm|m|ml|l|liter|w|watt|v|volt"
           r"|mah|pack|pk|pcs|count|ct|qt|gal|gallon|xl|xxl|x)\b")
NEG_RE = r"\b(?:without|w/o|no|not|non)[\s-]+([a-z0-9]+)"
HTML_RE = r"<[a-zA-Z/][^>]{0,60}>"
COLORS = {"black", "white", "red", "blue", "green", "yellow", "pink", "purple", "gray", "grey", "silver",
          "gold", "brown", "orange", "beige", "clear", "navy", "teal"}


def tokens(s):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower()) if isinstance(s, str) else []


def missing(s):
    return s.isna() | (s.fillna("").str.strip() == "")


def colors_of(s):
    return {("gray" if c == "grey" else c) for c in tokens(s) if c in COLORS}


def title_negates(title, term):
    """True when the (lower-case) title itself excludes the term, e.g. "no nails", "without hose", "bpa-free"."""
    t = re.escape(term)
    return re.search(rf"\b(?:without|w/o|no|non|free of)[\s-]+(?:\w+[\s-]+)?{t}|{t}[\s-]*free\b", title) is not None


#chart style
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABEL_COLOR = {"E": "#104281", "S": "#256abf", "C": "#5598e7", "I": "#86b6ef"}
LABEL_TEXT = {"E": "#ffffff", "S": "#ffffff", "C": INK, "I": INK}
SERIES = ["#2a78d6", "#eb6834"]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor=SURFACE)
    plt.close(fig)
