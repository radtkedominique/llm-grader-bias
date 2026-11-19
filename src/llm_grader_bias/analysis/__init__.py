"""Phase 4: Auswertung – neutrale Note, Self-/Family-Bias, Konfidenzintervalle, Mixed Model.

    python -m llm_grader_bias.analysis                       # data/gradings/gradings.jsonl → results/
    python -m llm_grader_bias.analysis --bootstrap 5000

Messlogik (siehe docs/studiendesign.md):

    neutrale Note(Lösung, Korrektor, Bedingung)
        = Mittel der Scores aller Korrektoren, deren Familie weder die des Korrektors
          noch die des Autors ist
    Abweichung = Score − neutrale Note
    Self-Bias   = Ø Abweichung(relation = self)   − Ø Abweichung(relation = foreign)
    Family-Bias = Ø Abweichung(relation = family) − Ø Abweichung(relation = foreign)

Die Differenz zu „foreign“ nimmt allgemeine Strenge/Milde des Korrektors heraus.
Konfidenzintervalle: Cluster-Bootstrap über Aufgaben (Teilaufgaben derselben Aufgabe hängen zusammen).
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

log = logging.getLogger("llm_grader_bias.analysis")

RELATIONS = ["self", "family", "foreign"]


# --------------------------------------------------------------------------- Daten

def load_gradings(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    n0 = len(df)
    if "repeat" in df:   # Wiederholungen nur für die Wiederholbarkeit (Kontrollen), nicht für den Bias
        df = df[df["repeat"].fillna(1) == 1]
    df = df[df["parse_ok"] & df["score"].notna()].copy()
    if len(df) < n0:
        log.info("%d von %d Bewertungen ohne gültige Punktzahl ausgeschlossen", n0 - len(df), n0)
    # Aufgabe = PDF (Cluster für den Bootstrap)
    df["task"] = df["task_id"].map(task_cluster)
    return df


def add_neutral(df: pd.DataFrame) -> pd.DataFrame:
    """Neutrale Note und Abweichung pro Bewertung."""
    key = ["solution_id", "condition"]
    # Scores je Lösung/Bedingung und Korrektor-Familie
    fam = (df.groupby(key + ["grader_family"])["score"].agg(["sum", "count"]).reset_index())
    out = df.merge(fam.rename(columns={"grader_family": "nf", "sum": "nf_sum", "count": "nf_n"}), on=key)
    out = out[(out["nf"] != out["grader_family"]) & (out["nf"] != out["author_family"])]
    neutral = (out.groupby("grading_id")[["nf_sum", "nf_n"]].sum()
               .assign(neutral=lambda x: x["nf_sum"] / x["nf_n"], n_neutral=lambda x: x["nf_n"])
               [["neutral", "n_neutral"]])
    df = df.merge(neutral, left_on="grading_id", right_index=True, how="left")
    df["deviation"] = df["score"] - df["neutral"]
    missing = df["neutral"].isna().sum()
    if missing:
        log.warning("%d Bewertungen ohne neutrale Note (keine unbeteiligte Familie) – ausgeschlossen", missing)
    return df[df["neutral"].notna()].copy()


# --------------------------------------------------------------------------- Bias-Maße

def bias_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Ø Abweichung je Beziehung und Self-/Family-Bias, gruppiert nach ``by``."""
    g = df.groupby(by + ["relation"])["deviation"].mean().unstack("relation")
    for r in RELATIONS:
        if r not in g:
            g[r] = np.nan
    g = g[RELATIONS].rename(columns={r: f"dev_{r}" for r in RELATIONS})
    g["self_bias"] = g["dev_self"] - g["dev_foreign"]
    g["family_bias"] = g["dev_family"] - g["dev_foreign"]
    g["n"] = df.groupby(by).size()
    return g.reset_index()


def _bias_point(df: pd.DataFrame) -> tuple[float, float]:
    m = df.groupby("relation")["deviation"].mean()
    f = m.get("foreign", np.nan)
    return m.get("self", np.nan) - f, m.get("family", np.nan) - f


def bootstrap(df: pd.DataFrame, by: list[str], reps: int, seed: int) -> pd.DataFrame:
    """95-%-Intervalle für Self-/Family-Bias per Cluster-Bootstrap über Aufgaben."""
    rng = np.random.default_rng(seed)
    rows = []
    groups = df.groupby(by) if by else [((), df)]
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        tasks = sub["task"].unique()
        parts = {t: d for t, d in sub.groupby("task")}
        est_s, est_f = _bias_point(sub)
        sb, fb = [], []
        for _ in range(reps):
            pick = rng.choice(tasks, size=len(tasks), replace=True)
            sample = pd.concat([parts[t] for t in pick], ignore_index=True)
            s, f = _bias_point(sample)
            sb.append(s)
            fb.append(f)
        sb, fb = np.array(sb), np.array(fb)

        def ci(a):
            a = a[~np.isnan(a)]
            return (np.nan, np.nan) if len(a) == 0 else tuple(np.percentile(a, [2.5, 97.5]))

        def p_two_sided(a):
            a = a[~np.isnan(a)]
            if len(a) == 0:
                return np.nan
            return float(min(1.0, 2 * min((a <= 0).mean(), (a >= 0).mean())))

        rows.append({**dict(zip(by, key)), "n_tasks": len(tasks), "n": len(sub),
                     "self_bias": est_s, "self_ci_low": ci(sb)[0], "self_ci_high": ci(sb)[1], "self_p": p_two_sided(sb),
                     "family_bias": est_f, "family_ci_low": ci(fb)[0], "family_ci_high": ci(fb)[1],
                     "family_p": p_two_sided(fb)})
    return pd.DataFrame(rows)


def mixed_model(df: pd.DataFrame) -> str:
    """abweichung ~ beziehung + beziehung:bedingung + beziehung:aufgabentyp + länge + Korrektor (fest) + (1|Aufgabe)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return "statsmodels nicht installiert – `pip install -e \".[analysis]\"`"
    d = df.copy()
    d["laenge_z"] = (d["solution_chars"] - d["solution_chars"].mean()) / (d["solution_chars"].std() or 1)
    # Die Abweichung ist schon relativ zur neutralen Note derselben Lösung → Bedingung und Aufgabentyp
    # gehen nur als Interaktion mit self/family ein (Referenz: foreign). Dummies statt C(relation),
    # damit keine nicht identifizierbaren foreign-Interaktionen entstehen.
    d["self"] = (d["relation"] == "self").astype(float)
    d["family"] = (d["relation"] == "family").astype(float)
    terms = ["self", "family"]
    if d["condition"].nunique() > 1:
        terms += ["self:C(condition)", "family:C(condition)"]
    if d["aufgabentyp"].nunique() > 1:
        terms += ["self:C(aufgabentyp)", "family:C(aufgabentyp)"]
    # Korrektor als fester Effekt (nimmt Strenge heraus; bei 6–8 Korrektoren stabiler als ein Random Effect)
    formula = "deviation ~ " + " + ".join(terms) + " + laenge_z + C(grader_model)"
    import warnings
    try:
        model = smf.mixedlm(formula, d, groups=d["task"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = model.fit(method="lbfgs", reml=True)
        notes = sorted({str(w.message).split(".")[0] for w in caught})
        note = ("\nHinweise beim Schätzen: " + "; ".join(notes) +
                "\n(Häufig, wenn eine Varianzkomponente nahe 0 liegt; Bootstrap-Ergebnisse oben sind davon unberührt.)"
                if notes else "")
        return f"Formel: {formula} + (1|Aufgabe)\n\n{res.summary().as_text()}{note}"
    except Exception as e:  # noqa: BLE001
        return f"Mixed Model nicht schätzbar: {e}"


# --------------------------------------------------------------------------- Report

def _fmt(x, digits=3):
    return "–" if pd.isna(x) else f"{x:+.{digits}f}"


def _md(df: pd.DataFrame, cols: list[str], digits: int = 3) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(_fmt(v, digits) if "bias" in c or c.startswith("dev") or "ci" in c
                             else ("–" if pd.isna(v) else f"{v:.3f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _ci_col(r, kind: str) -> str:
    return f"{_fmt(r[f'{kind}_bias'])} [{_fmt(r[f'{kind}_ci_low'])}; {_fmt(r[f'{kind}_ci_high'])}]"


def _p(p: float, reps: int) -> str:
    if pd.isna(p):
        return "–"
    return f"< {1 / reps:.4f}" if p == 0 else f"{p:.3f}"


def report(df: pd.DataFrame, raw: pd.DataFrame, overall: pd.DataFrame, by_cond: pd.DataFrame,
           by_grader: pd.DataFrame, by_type: pd.DataFrame, strictness: pd.DataFrame, mm: str,
           reps: int = 2000) -> str:
    lines = [
        "# Auswertung Self-/Family-Bias",
        "",
        f"- Bewertungen gesamt: {len(raw)}, ausgewertet (mit neutraler Note): {len(df)}",
        f"- Korrektoren: {df['grader_model'].nunique()} aus {df['grader_family'].nunique()} Familien; "
        f"Lösungen: {df['solution_id'].nunique()}; Aufgaben: {df['task'].nunique()}",
        f"- Bedingungen: {', '.join(sorted(df['condition'].unique()))}",
        "",
        "Score = Punkte / Maximum (0–1). Abweichung = Score − neutrale Note. "
        "Bias > 0: eigene bzw. verwandte Lösungen werden besser bewertet als fremde.",
        "Intervalle: 95 % Cluster-Bootstrap über Aufgaben; p = zweiseitiger Bootstrap-p-Wert.",
        "",
        "## Gesamt",
        "| | Bias [95 %-KI] | p |", "|---|---|---|",
    ]
    for _, r in overall.iterrows():
        lines.append(f"| Self-Bias | {_ci_col(r, 'self')} | {_p(r['self_p'], reps)} |")
        lines.append(f"| Family-Bias | {_ci_col(r, 'family')} | {_p(r['family_p'], reps)} |")
    lines += ["", "## Nach Bedingung (RQ3: hilft der Erwartungshorizont?)",
              "| Bedingung | Self-Bias [95 %-KI] | Family-Bias [95 %-KI] | n |", "|---|---|---|---|"]
    for _, r in by_cond.iterrows():
        lines.append(f"| {r['condition']} | {_ci_col(r, 'self')} | {_ci_col(r, 'family')} | {r['n']} |")
    lines += ["", "## Nach Aufgabentyp (RQ4)",
              "| Aufgabentyp | Self-Bias [95 %-KI] | Family-Bias [95 %-KI] | n |", "|---|---|---|---|"]
    for _, r in by_type.iterrows():
        lines.append(f"| {r['aufgabentyp']} | {_ci_col(r, 'self')} | {_ci_col(r, 'family')} | {r['n']} |")
    lines += ["", "## Nach Korrektor",
              "| Korrektor | Familie | Größe | Self-Bias [95 %-KI] | Family-Bias [95 %-KI] |", "|---|---|---|---|---|"]
    for _, r in by_grader.iterrows():
        lines.append(f"| {r['grader_model']} | {r['grader_family']} | {r['grader_size']} | "
                     f"{_ci_col(r, 'self')} | {_ci_col(r, 'family')} |")
    lines += ["", "## Strenge der Korrektoren (Ø Score, alle Lösungen)", *_md(strictness, list(strictness.columns)),
              "", "## Kontrollen",
              f"- Korrelation Lösungslänge ↔ Score: {raw['solution_chars'].corr(raw['score']):.3f}",
              f"- Anteil ungültiger Bewertungen (kein Parse): {(~raw['parse_ok']).mean():.1%}",
              "", "## Mixed Model", "```", mm, "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_grader_bias.analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gradings", type=Path, default=Path("data/gradings/gradings.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-mixed-model", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    raw_all = pd.DataFrame([json.loads(l) for l in args.gradings.open(encoding="utf-8") if l.strip()])
    df = add_neutral(load_gradings(args.gradings))
    if df.empty:
        log.error("Keine auswertbaren Bewertungen (mindestens 3 Familien nötig).")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    overall = bootstrap(df, [], args.bootstrap, args.seed)
    by_cond = bootstrap(df, ["condition"], args.bootstrap, args.seed)
    by_type = bootstrap(df, ["aufgabentyp"], args.bootstrap, args.seed)
    by_grader = bootstrap(df, ["grader_model", "grader_family", "grader_size"], args.bootstrap, args.seed)
    strictness = (raw_all[raw_all["parse_ok"]].groupby(["grader_model", "condition"])["score"].mean()
                  .unstack("condition").reset_index())
    mm = "übersprungen" if args.no_mixed_model else mixed_model(df)

    df.to_csv(args.out / "gradings_with_neutral.csv", index=False)
    for name, t in [("bias_overall", overall), ("bias_by_condition", by_cond), ("bias_by_type", by_type),
                    ("bias_by_grader", by_grader), ("strictness", strictness),
                    ("deviation_by_grader_relation", bias_table(df, ["grader_model", "condition"]))]:
        t.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "report.md").write_text(
        report(df, raw_all, overall, by_cond, by_grader, by_type, strictness, mm, args.bootstrap), encoding="utf-8")
    r = overall.iloc[0]
    log.info("Self-Bias %+.3f [%+.3f; %+.3f], Family-Bias %+.3f [%+.3f; %+.3f] → %s",
             r.self_bias, r.self_ci_low, r.self_ci_high, r.family_bias, r.family_ci_low, r.family_ci_high,
             args.out / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
