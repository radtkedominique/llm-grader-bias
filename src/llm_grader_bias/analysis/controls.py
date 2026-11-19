"""Block 5: Kontrollen und Robustheit.

    python -m llm_grader_bias.analysis.controls               # → results/kontrollen/

1. Lösungslänge ↔ Score: Wird Länge belohnt?
2. Bias mit und ohne Längen-Kontrolle: Bleibt der Self-/Family-Bias, wenn man Länge herausrechnet?
3. Wiederholbarkeit: Gibt derselbe Korrektor bei wiederholtem Aufruf dieselbe Note? (braucht `grade --repeats 2`)
4. Kontamination: Unterscheidet sich der Bias nach Prüfungsjahr?
5. Lösungslänge je Modell
6. Datenqualität: leere/abgeschnittene Lösungen, Eingriffe der Anonymisierung
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..prompting import task_cluster

from . import add_neutral, bootstrap
from . import style as S

log = logging.getLogger("llm_grader_bias.analysis.controls")
SRC = "Eigene Auswertung"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.open(encoding="utf-8") if l.strip()])


def _valid(g: pd.DataFrame) -> pd.DataFrame:
    g = g[g["parse_ok"] & g["score"].notna()].copy()
    g["task"] = g["task_id"].map(task_cluster)
    return g


# --------------------------------------------------------------------------- 1 Länge ↔ Score

def length_score(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, d in [("alle", g), *list(g.groupby("condition"))]:
        rows.append({"bedingung": cond, "n": len(d),
                     "pearson": d["solution_chars"].corr(d["score"]),
                     "spearman": d["solution_chars"].corr(d["score"], method="spearman")})
    return pd.DataFrame(rows)


def plot_length_score(g: pd.DataFrame, out: Path) -> Path:
    S.apply()
    fig, ax = S.plt.subplots(figsize=(10, 5))
    x = g["solution_chars"].clip(lower=1)
    rng = np.random.default_rng(0)
    ax.scatter(x, g["score"] + rng.uniform(-0.015, 0.015, len(g)), s=10, color=S.SERIES[0], alpha=0.25,
               linewidths=0)
    # gleitender Mittelwert über Längen-Klassen (Dezile)
    bins = pd.qcut(np.log10(x), q=min(10, max(2, len(g) // 30)), duplicates="drop")
    m = g.groupby(bins, observed=True).agg(x=("solution_chars", "median"), y=("score", "mean"))
    ax.plot(m["x"], m["y"], color=S.TEXT, linewidth=2, marker="o", markersize=6, label="Mittelwert je Längenklasse")
    ax.set_xscale("log")
    S.plain_log_axis(ax.xaxis)
    ax.set_xlabel("Länge der Lösung (Zeichen, log)")
    ax.set_ylabel("Score (Punkte / Maximum)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    r = g["solution_chars"].corr(g["score"], method="spearman")
    return S.finish(fig, ax, "Werden längere Lösungen besser bewertet?",
                    f"Jeder Punkt = eine Bewertung; Spearman ρ = {r:.2f}", SRC, out)


# --------------------------------------------------------------------------- 2 Bias mit/ohne Länge

def bias_length_controlled(dev: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Self-/Family-Bias vor und nach Herausrechnen eines linearen Längeneffekts, gesamt und je Korrektor.

    Gesamt hebt sich ein Längeneffekt weitgehend auf (jede Lösung ist für einen Korrektor „self“ und für
    die anderen „foreign“); aussagekräftig ist der Vergleich je Korrektor.
    """
    d = dev.copy()
    z = np.log10(d["solution_chars"].clip(lower=1))
    slope, intercept = np.polyfit(z, d["deviation"], 1) if z.nunique() > 1 else (0.0, 0.0)
    ctrl = d.assign(deviation=d["deviation"] - (slope * z + intercept))
    parts = []
    for variante, data in [("ohne Längen-Kontrolle", d), ("mit Längen-Kontrolle", ctrl)]:
        parts.append(bootstrap(data, [], reps, seed).assign(variante=variante, grader_model="alle"))
        parts.append(bootstrap(data, ["grader_model"], reps, seed).assign(variante=variante))
    out = pd.concat(parts, ignore_index=True)
    out["laengen_steigung"] = slope
    return out


# --------------------------------------------------------------------------- 3 Wiederholbarkeit

def repeatability(g_all: pd.DataFrame) -> pd.DataFrame:
    if "repeat" not in g_all or g_all["repeat"].fillna(1).max() < 2:
        return pd.DataFrame()
    g = g_all[g_all["parse_ok"] & g_all["score"].notna()].copy()
    g["repeat"] = g["repeat"].fillna(1).astype(int)
    key = ["solution_id", "grader_model", "condition"]
    wide = g.pivot_table(index=key, columns="repeat", values="punkte", aggfunc="first").dropna()
    if wide.shape[1] < 2:
        return pd.DataFrame()
    a, b = wide.iloc[:, 0], wide.iloc[:, 1]
    wide = wide.reset_index()
    wide["gleich"] = (a.values == b.values)
    wide["abs_diff"] = np.abs(a.values - b.values)
    res = wide.groupby("grader_model").agg(n=("gleich", "size"), identisch=("gleich", "mean"),
                                           mittl_abweichung_punkte=("abs_diff", "mean")).reset_index()
    total = pd.DataFrame([{"grader_model": "alle", "n": len(wide), "identisch": wide["gleich"].mean(),
                           "mittl_abweichung_punkte": wide["abs_diff"].mean()}])
    return pd.concat([res, total], ignore_index=True)


def plot_repeatability(rep: pd.DataFrame, out: Path) -> Path:
    S.apply()
    d = rep[rep["grader_model"] != "alle"].sort_values("identisch")
    fig, ax = S.plt.subplots(figsize=(9, 0.5 * len(d) + 1.8))
    y = np.arange(len(d))
    ax.barh(y, d["identisch"] * 100, color=S.SERIES[0], height=0.6)
    for yi, v in zip(y, d["identisch"] * 100):
        ax.text(v, yi, f"  {v:.0f} %", va="center", color=S.TEXT, fontsize=10)
    ax.set_yticks(y, d["grader_model"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("identische Punktzahl bei Wiederholung (%)")
    ax.grid(axis="y", visible=False)
    total = rep.loc[rep["grader_model"] == "alle"].iloc[0]
    return S.finish(fig, ax, "Wie stabil sind die Bewertungen?",
                    f"Gleiche Lösung zweimal bewertet; insgesamt {total['identisch']:.0%} identisch, "
                    f"Ø Abweichung {total['mittl_abweichung_punkte']:.2f} Punkte", SRC, out)


# --------------------------------------------------------------------------- 4 Kontamination

def bias_by_year(dev: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    d = dev.copy()
    d["jahrgang"] = d["jahr"].map(lambda j: "Beispiel" if pd.isna(j) else str(int(j)))
    return bootstrap(d, ["jahrgang"], reps, seed)


def plot_bias_by_year(by: pd.DataFrame, out: Path) -> Path:
    S.apply()
    d = by.sort_values("jahrgang").reset_index(drop=True)
    fig, ax = S.plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(d))
    for off, kind, col, label in [(-0.12, "self", S.SERIES[0], "Self-Bias"), (0.12, "family", S.SERIES[1], "Family-Bias")]:
        lo = d[f"{kind}_bias"] - d[f"{kind}_ci_low"]
        hi = d[f"{kind}_ci_high"] - d[f"{kind}_bias"]
        ax.errorbar(x + off, d[f"{kind}_bias"], yerr=[lo, hi], fmt="o", color=col, ecolor=col, elinewidth=2,
                    capsize=0, markersize=8, label=label)
    ax.axhline(0, color=S.TEXT_2, linewidth=1)
    ax.set_xticks(x, [f"{j}\n(n={n})" for j, n in zip(d["jahrgang"], d["n"])])
    ax.set_ylabel("Bias (Score-Anteil)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return S.finish(fig, ax, "Bias nach Prüfungsjahr", "Punkt = Schätzung, Linie = 95-%-Intervall (Bootstrap über Aufgaben); "
                    "ältere Jahrgänge sind eher in Trainingsdaten", SRC, out)


# --------------------------------------------------------------------------- 5/6 Lösungen

def solution_quality(sol: pd.DataFrame) -> pd.DataFrame:
    s = sol.copy()
    s["laenge"] = s["text"].str.len()
    s["abgeschnitten"] = s["finish_reason"].isin(["length", "max_tokens"])
    s["anon_eingriff"] = s["anon_removed"].map(len) > 0
    return s.groupby(["model_id", "family", "size"]).agg(
        loesungen=("solution_id", "size"), leer=("empty", "sum"), abgeschnitten=("abgeschnitten", "sum"),
        mit_anonymisierung=("anon_eingriff", "sum"), eingriffe=("anon_removed", lambda x: sum(len(v) for v in x)),
        laenge_median=("laenge", "median"), laenge_q25=("laenge", lambda x: x.quantile(0.25)),
        laenge_q75=("laenge", lambda x: x.quantile(0.75))).reset_index()


def plot_solution_length(sol: pd.DataFrame, out: Path) -> Path:
    S.apply()
    s = sol.assign(laenge=sol["text"].str.len().clip(lower=1))
    order = (s.groupby(["family", "size", "model_id"]).size().reset_index()
             .sort_values(["family", "size"])["model_id"].tolist())
    fams = S.color_map(s["family"].unique().tolist())
    fam_of = s.drop_duplicates("model_id").set_index("model_id")["family"]
    fig, ax = S.plt.subplots(figsize=(10, 0.55 * len(order) + 1.8))
    data = [s.loc[s["model_id"] == m, "laenge"].values for m in order]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55, showfliers=False,
                    medianprops={"color": S.TEXT, "linewidth": 1.5},
                    whiskerprops={"color": S.TEXT_2}, capprops={"color": S.TEXT_2})
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(fams[fam_of[m]])
        patch.set_edgecolor(fams[fam_of[m]])
        patch.set_alpha(0.85)
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.set_xscale("log")
    S.plain_log_axis(ax.xaxis)
    ax.set_xlabel("Länge der Lösung (Zeichen, log)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in fams.items()], loc="upper left",
              bbox_to_anchor=(1.01, 1), title="Familie")
    return S.finish(fig, ax, "Wie lang sind die Lösungen je Modell?",
                    "Box = mittlere 50 %, Strich = Median; Länge ist eine Kontrollvariable", SRC, out)


# --------------------------------------------------------------------------- main

def _pct(x) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_grader_bias.analysis.controls", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gradings", type=Path, default=Path("data/gradings/gradings.jsonl"))
    ap.add_argument("--solutions", type=Path, default=Path("data/solutions/solutions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/kontrollen"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    g_all = _read(args.gradings)
    sol = _read(args.solutions)
    if g_all.empty or sol.empty:
        log.error("Keine Lösungen/Bewertungen gefunden – erst solve und grade laufen lassen")
        return 1
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    g1 = _valid(g_all[g_all.get("repeat", pd.Series(1, index=g_all.index)).fillna(1) == 1])
    dev = add_neutral(g1)

    ls = length_score(g1)
    blc = bias_length_controlled(dev, args.bootstrap, args.seed)
    rep = repeatability(g_all)
    by_year = bias_by_year(dev, args.bootstrap, args.seed)
    sq = solution_quality(sol)

    for name, t in [("laenge_score", ls), ("bias_laengenkontrolle", blc), ("wiederholbarkeit", rep),
                    ("bias_nach_jahr", by_year), ("loesungen_qualitaet", sq)]:
        if not t.empty:
            t.to_csv(out / f"{name}.csv", index=False)

    figs = [plot_length_score(g1, out / "01_laenge_score.png"),
            plot_solution_length(sol, out / "02_loesungslaenge_modelle.png"),
            plot_bias_by_year(by_year, out / "03_bias_nach_jahr.png")]
    if not rep.empty:
        figs.append(plot_repeatability(rep, out / "04_wiederholbarkeit.png"))

    def ci(r, k):
        return f"{r[f'{k}_bias']:+.3f} [{r[f'{k}_ci_low']:+.3f}; {r[f'{k}_ci_high']:+.3f}]"

    md = ["# Kontrollen und Robustheit", "",
          "## 1 Lösungslänge ↔ Score", "| Bedingung | n | Pearson r | Spearman ρ |", "|---|---|---|---|",
          *[f"| {r.bedingung} | {r.n} | {r.pearson:.3f} | {r.spearman:.3f} |" for r in ls.itertuples()],
          "", "## 2 Bias mit und ohne Längen-Kontrolle",
          f"Längeneffekt auf die Abweichung: {blc['laengen_steigung'].iloc[0]:+.3f} pro Zehnerpotenz Zeichen. "
          "Gesamt hebt sich ein Längeneffekt weitgehend auf; entscheidend ist der Vergleich je Korrektor.", "",
          "| Korrektor | Variante | Self-Bias [95 %-KI] | Family-Bias [95 %-KI] |", "|---|---|---|---|",
          *[f"| {r['grader_model']} | {r['variante']} | {ci(r, 'self')} | {ci(r, 'family')} |"
            for _, r in blc.sort_values(["grader_model", "variante"], ascending=[True, False]).iterrows()],
          "", "## 3 Wiederholbarkeit"]
    if rep.empty:
        md.append("Keine Wiederholungen vorhanden – `python -m llm_grader_bias.grade --repeats 2` ausführen.")
    else:
        md += ["| Korrektor | n | identisch | Ø Abweichung (Punkte) |", "|---|---|---|---|",
               *[f"| {r.grader_model} | {r.n} | {_pct(r.identisch)} | {r.mittl_abweichung_punkte:.2f} |"
                 for r in rep.itertuples()]]
    md += ["", "## 4 Bias nach Prüfungsjahr (Kontamination)", "| Jahrgang | n | Self-Bias [95 %-KI] | Family-Bias [95 %-KI] |",
           "|---|---|---|---|", *[f"| {r['jahrgang']} | {r['n']} | {ci(r, 'self')} | {ci(r, 'family')} |"
                                  for _, r in by_year.sort_values("jahrgang").iterrows()],
           "", "## 5/6 Lösungen je Modell", "| Modell | Familie | Größe | Lösungen | leer | abgeschnitten | "
           "mit Anonymisierung | Länge Median [Q25–Q75] |", "|---|---|---|---|---|---|---|---|",
           *[f"| {r.model_id} | {r.family} | {r.size} | {r.loesungen} | {r.leer} | {r.abgeschnitten} | "
             f"{r.mit_anonymisierung} | {r.laenge_median:.0f} [{r.laenge_q25:.0f}–{r.laenge_q75:.0f}] |" for r in sq.itertuples()],
           "", "## Grafiken", *[f"- `{p.name}`" for p in figs], ""]
    (out / "kontrollen.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Kontrollen: %d Grafiken + Tabellen → %s", len(figs), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
