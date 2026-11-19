"""Block 1: Kennzahlen und Grafiken zum Datensatz (Zahlen, Daten, Fakten für die Präsentation).

    python -m llm_grader_bias.analysis.descriptive            # → results/datensatz/

Liest data/processed/{catalog,tasks,subtasks,dataset}.jsonl und (falls vorhanden) data/pilot/pilot.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import style as S

log = logging.getLogger("llm_grader_bias.analysis.descriptive")

SRC = "Quelle: IQB-Abituraufgabenpool, ISB Bayern; eigene Auswertung"
FACH = {"deutsch": "Deutsch", "englisch": "Englisch", "franzoesisch": "Französisch", "mathematik": "Mathematik",
        "physik": "Physik", "chemie": "Chemie", "biologie": "Biologie", "informatik": "Informatik"}
TYP = {"sprache": "Sprache", "mint": "MINT", "informatik": "Informatik"}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.open(encoding="utf-8") if l.strip()])


def _fmt(n) -> str:
    return f"{int(n):,}".replace(",", ".")


# --------------------------------------------------------------------------- Kennzahlen

def funnel(cat, tasks, subs, ds) -> pd.DataFrame:
    """Trichter auf Ebene der Teilaufgaben (PDF-Zahlen stehen im Untertitel, andere Einheit)."""
    high = subs[subs["split_confidence"] == "high"]
    usable = high[high["text_only_candidate"]]
    return pd.DataFrame([
        ("Teilaufgaben erkannt", len(subs)),
        ("sicher dem Erwartungshorizont zugeordnet", len(high)),
        ("reine Textaufgaben", len(usable)),
        ("im Datensatz (ohne Dubletten)", len(ds)),
    ], columns=["stufe", "anzahl"])


def exclusion_reasons(subs, ds) -> pd.DataFrame:
    def reason(r):
        notes = " ".join(r["split_notes"])
        if r["split_confidence"] != "high":
            return "ohne Erwartungshorizont" if "kein Erwartungshorizont" in notes else "Zuordnung unsicher"
        if r["text_only_candidate"]:
            return None
        if "Materialtext fehlt" in notes:
            return "Materialtext fehlt im PDF"
        if r["fragment_ratio"] >= 0.3:
            return "Formeln zerfallen"
        if r["mentions_figure"]:
            return "Abbildung nötig"
        return "sonstiges"
    reasons = subs.apply(reason, axis=1).dropna().value_counts()
    usable = int((subs["split_confidence"].eq("high") & subs["text_only_candidate"]).sum())
    reasons["Dublette (WTR/CAS)"] = usable - len(ds)
    return reasons.sort_values(ascending=False).rename_axis("grund").reset_index(name="anzahl")


def by_subject(subs, ds) -> pd.DataFrame:
    t = pd.DataFrame({
        "teilaufgaben": subs.groupby("subject").size(),
        "sicher_zugeordnet": subs[subs["split_confidence"] == "high"].groupby("subject").size(),
        "im_datensatz": ds.groupby("fach").size(),
        "ab_2024": ds[ds["jahr"].fillna(0) >= 2024].groupby("fach").size(),
    }).fillna(0).astype(int)
    t.index = [FACH.get(i, i) for i in t.index]
    return t.sort_values("im_datensatz", ascending=False).rename_axis("fach").reset_index()


# --------------------------------------------------------------------------- Grafiken

def plot_funnel(f: pd.DataFrame, out: Path, n_pdfs: int = 0, n_tasks: int = 0) -> Path:
    S.apply()
    fig, ax = S.plt.subplots(figsize=(10, 3.8))
    ramp = ["#86b6ef", "#5598e7", "#256abf", "#184f95"]   # ordinal, ab Stufe 250
    y = np.arange(len(f))[::-1]
    ax.barh(y, f["anzahl"], color=ramp[: len(f)], height=0.62)
    for yi, v in zip(y, f["anzahl"]):
        ax.text(v, yi, f"  {_fmt(v)}", va="center", ha="left", color=S.TEXT, fontsize=11)
    ax.set_yticks(y, f["stufe"])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, f["anzahl"].max() * 1.15)
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(False)
    for yi, (prev, cur) in zip(y[1:], zip(f["anzahl"][:-1], f["anzahl"][1:])):
        ax.text(f["anzahl"].max() * 1.13, yi, f"{cur / prev:.0%}", va="center", ha="right", color=S.TEXT_2, fontsize=10)
    sub = (f"aus {_fmt(n_pdfs)} Aufgaben-PDFs ({_fmt(n_tasks)} nach Filter); rechts: Anteil der vorigen Stufe"
           if n_pdfs else "Anzahl Teilaufgaben je Verarbeitungsschritt")
    return S.finish(fig, ax, "Vom PDF zum Datensatz", sub, SRC, out)


def plot_hbar(labels, values, title, subtitle, out: Path, color=S.SERIES[0], xlabel=None) -> Path:
    S.apply()
    fig, ax = S.plt.subplots(figsize=(9, 0.5 * len(labels) + 1.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.6)
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  {_fmt(v)}", va="center", ha="left", color=S.TEXT, fontsize=10)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(max(values) * 1.15, 1) if len(values) else 1)
    if xlabel:
        ax.set_xlabel(xlabel)
    return S.finish(fig, ax, title, subtitle, SRC, out)


def plot_heatmap_subject_year(ds: pd.DataFrame, out: Path) -> Path:
    S.apply()
    d = ds.assign(jahr=ds["jahr"].map(lambda j: "Beispiel" if pd.isna(j) else str(int(j))),
                  fach=ds["fach"].map(lambda f: FACH.get(f, f)))
    m = d.pivot_table(index="fach", columns="jahr", values="id", aggfunc="count", fill_value=0)
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [S.SURFACE] + S.SEQ_BLUE)
    fig, ax = S.plt.subplots(figsize=(11, 0.55 * len(m) + 1.8))
    im = ax.imshow(m.values, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(m.columns)), m.columns)
    ax.set_yticks(range(len(m.index)), m.index)
    ax.grid(False)
    vmax = m.values.max() or 1
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > 0.55 * vmax else S.TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Teilaufgaben", color=S.TEXT_2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return S.finish(fig, ax, "Teilaufgaben im Datensatz nach Fach und Jahr",
                    "Beispiel = Beispielaufgaben ohne Prüfungsjahr", SRC, out)


def plot_lengths(ds: pd.DataFrame, out: Path) -> Path:
    S.apply()
    fields = [("kontext", "Kontext / Material"), ("aufgabe", "Aufgabenstellung"), ("erwartungshorizont", "Erwartungshorizont")]
    types = [t for t in ["sprache", "mint", "informatik"] if t in set(ds["aufgabentyp"])]
    colors = S.color_map(types)
    fig, axes = S.plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
    for ax, (col, name) in zip(axes, fields):
        lens = [ds.loc[ds["aufgabentyp"] == t, col].str.len() for t in types]
        empty = sum((x == 0).sum() for x in lens) / max(1, sum(len(x) for x in lens))
        data = [x[x > 0].values for x in lens]      # leere Felder nicht als „1 Zeichen“ darstellen
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                        medianprops={"color": S.TEXT, "linewidth": 1.5},
                        whiskerprops={"color": S.TEXT_2}, capprops={"color": S.TEXT_2})
        for patch, t in zip(bp["boxes"], types):
            patch.set_facecolor(colors[t])
            patch.set_edgecolor(colors[t])
            patch.set_alpha(0.85)
        ax.set_yscale("log")
        S.plain_log_axis(ax.yaxis)
        ax.set_xticks(range(1, len(types) + 1), [TYP[t] for t in types])
        ax.set_title(name + (f"  ({empty:.0%} leer, nicht dargestellt)" if empty >= 0.005 else ""), fontsize=12)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Zeichen (log)" if col == "kontext" else "")
    fig.suptitle("Textlänge je Aufgabentyp", x=0.01, ha="left", fontsize=14, fontweight="semibold", color=S.TEXT)
    fig.text(0.01, 0.01, SRC + " · Box = mittlere 50 %, Strich = Median", fontsize=8, color=S.TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    S.plt.close(fig)
    return out


def plot_be(ds: pd.DataFrame, out: Path) -> Path:
    S.apply()
    be = ds["max_punkte"].dropna().astype(int)
    counts = be.value_counts().sort_index()
    fig, ax = S.plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, color=S.SERIES[0], width=0.7)
    ax.set_xlabel("Bewertungseinheiten (BE) der Teilaufgabe")
    ax.set_ylabel("Teilaufgaben")
    ax.set_xticks(counts.index)
    ax.grid(axis="x", visible=False)
    n_weight = ds["gewichtung_prozent"].notna().sum()
    sub = f"{_fmt(len(be))} Teilaufgaben mit BE, Median {be.median():.0f} BE" + (
        f"; {n_weight} Sprachaufgaben mit Prozentgewichtung statt BE" if n_weight else "")
    return S.finish(fig, ax, "Wie viele Punkte gibt es pro Teilaufgabe?", sub, SRC, out)


def plot_pilot(pilot: pd.DataFrame, out: Path) -> Path:
    S.apply()
    types = sorted(pilot["aufgabentyp"].unique())
    colors = S.color_map(types)
    g = pilot.groupby(["aufgabentyp", "fach"]).size().reset_index(name="n")
    g = g.sort_values(["aufgabentyp", "n"], ascending=[True, False])
    fig, ax = S.plt.subplots(figsize=(9, 0.5 * len(g) + 1.6))
    y = np.arange(len(g))[::-1]
    ax.barh(y, g["n"], color=[colors[t] for t in g["aufgabentyp"]], height=0.6)
    for yi, v in zip(y, g["n"]):
        ax.text(v, yi, f"  {v}", va="center", ha="left", color=S.TEXT, fontsize=10)
    ax.set_yticks(y, [FACH.get(f, f) for f in g["fach"]])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, g["n"].max() * 1.25)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[t], label=TYP.get(t, t)) for t in types], loc="lower right")
    yrs = pilot["jahr"].dropna()
    from ..prompting import task_cluster
    sub = f"{len(pilot)} Teilaufgaben aus {pilot['id'].map(task_cluster).nunique()} Aufgaben" + (
        f", Jahrgänge {int(yrs.min())}–{int(yrs.max())}" if len(yrs) else "")
    return S.finish(fig, ax, "Zusammensetzung des Piloten", sub, SRC, out)


# --------------------------------------------------------------------------- Report

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_grader_bias.analysis.descriptive", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--pilot", type=Path, default=Path("data/pilot/pilot.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/datensatz"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    P = args.processed
    cat, tasks, subs, ds = (_read(P / f) for f in ("catalog.jsonl", "tasks.jsonl", "subtasks.jsonl", "dataset.jsonl"))
    if ds.empty or subs.empty:
        log.error("Kein Datensatz unter %s – erst `python -m llm_grader_bias.dataset all`", P)
        return 1
    pilot = _read(args.pilot)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    f = funnel(cat, tasks, subs, ds)
    ex = exclusion_reasons(subs, ds)
    subj = by_subject(subs, ds)
    years = ds["jahr"].map(lambda j: "Beispiel" if pd.isna(j) else str(int(j))).value_counts().sort_index()
    niveau = ds["niveau"].fillna("unbekannt").value_counts()
    typ = ds["aufgabentyp"].value_counts()
    lengths = ds.assign(**{c: ds[c].str.len() for c in ("kontext", "aufgabe", "erwartungshorizont")}).groupby(
        "aufgabentyp")[["kontext", "aufgabe", "erwartungshorizont"]].median().round(0)

    for name, t in [("trichter", f), ("ausschlussgruende", ex), ("nach_fach", subj),
                    ("nach_jahr", years.rename_axis("jahr").reset_index(name="anzahl")),
                    ("laenge_median", lengths.reset_index())]:
        t.to_csv(out / f"{name}.csv", index=False)

    figs = [
        plot_funnel(f, out / "01_trichter.png", len(cat), len(tasks)),
        plot_hbar(subj["fach"].tolist(), subj["im_datensatz"].tolist(), "Teilaufgaben im Datensatz nach Fach",
                  f"{_fmt(len(ds))} Teilaufgaben insgesamt", out / "02_nach_fach.png"),
        plot_heatmap_subject_year(ds, out / "03_fach_jahr.png"),
        plot_hbar(ex["grund"].tolist(), ex["anzahl"].tolist(), "Warum Teilaufgaben nicht im Datensatz sind",
                  "Anzahl Teilaufgaben je Ausschlussgrund", out / "04_ausschlussgruende.png", color=S.NEUTRAL),
        plot_lengths(ds, out / "05_textlaenge.png"),
        plot_be(ds, out / "06_be_verteilung.png"),
    ]
    if not pilot.empty:
        figs.append(plot_pilot(pilot, out / "07_pilot.png"))

    usable = int((subs["split_confidence"].eq("high") & subs["text_only_candidate"]).sum())
    md = [
        "# Datensatz – Zahlen, Daten, Fakten", "",
        "## Kennzahlen",
        f"- **{_fmt(len(cat))}** Aufgaben-PDFs gefunden, **{_fmt(len(tasks))}** verarbeitet",
        f"- **{_fmt(len(subs))}** Teilaufgaben erkannt, davon **{_fmt((subs['split_confidence'] == 'high').sum())}** "
        f"sicher dem Erwartungshorizont zugeordnet ({(subs['split_confidence'] == 'high').mean():.0%})",
        f"- **{_fmt(len(ds))}** Teilaufgaben im fertigen Datensatz ({usable - len(ds)} Dubletten entfernt)",
        f"- **{ds['fach'].nunique()}** Fächer, **{ds['jahr'].dropna().nunique()}** Prüfungsjahrgänge "
        f"({int(ds['jahr'].min())}–{int(ds['jahr'].max())}) plus {int(ds['jahr'].isna().sum())} Beispielaufgaben",
        f"- Aufgabentypen: " + ", ".join(f"{TYP.get(k, k)} {v}" for k, v in typ.items()),
        f"- Niveau: " + ", ".join(f"{k} {v}" for k, v in niveau.items()),
        f"- ab 2024: **{int((ds['jahr'].fillna(0) >= 2024).sum())}** Teilaufgaben",
        f"- Mathe-Anteil: {(ds['fach'] == 'mathematik').mean():.0%}",
        "",
        "## Trichter (Teilaufgaben)", "| Stufe | Anzahl |", "|---|---|", *[f"| {r.stufe} | {_fmt(r.anzahl)} |" for r in f.itertuples()],
        "", "## Nach Fach", "| Fach | Teilaufgaben | sicher zugeordnet | im Datensatz | ab 2024 |", "|---|---|---|---|---|",
        *[f"| {r.fach} | {r.teilaufgaben} | {r.sicher_zugeordnet} | {r.im_datensatz} | {r.ab_2024} |" for r in subj.itertuples()],
        "", "## Ausschlussgründe", "| Grund | Anzahl |", "|---|---|", *[f"| {r.grund} | {_fmt(r.anzahl)} |" for r in ex.itertuples()],
        "", "## Textlänge (Median, Zeichen)", "| Typ | Kontext | Aufgabe | Erwartungshorizont |", "|---|---|---|---|",
        *[f"| {TYP.get(i, i)} | {int(r.kontext)} | {int(r.aufgabe)} | {int(r.erwartungshorizont)} |" for i, r in lengths.iterrows()],
        "", "## Grafiken", *[f"- `{p.name}`" for p in figs], "",
    ]
    (out / "datensatz.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Datensatz-Kennzahlen: %d Grafiken + Tabellen → %s", len(figs), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
