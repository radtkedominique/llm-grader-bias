"""Geschichtete, reproduzierbare Auswahl der Pilot-Teilaufgaben.

    python -m llm_grader_bias.pilot                      # Standard: configs/pilot.yaml
    python -m llm_grader_bias.pilot --n-per-type 5       # kleinerer Probelauf

Ablauf:
1. Filter: Jahr (``min_year``, Ausnahmen für Fächer ohne Jahr), Kontextlänge.
2. Pro Aufgabentyp eine Quote; innerhalb des Typs reihum über die Fächer (``balance_subjects``).
3. Höchstens ``max_per_task`` Teilaufgaben aus derselben Aufgabe.
4. Fester Seed → gleiche Auswahl bei jedem Lauf.

Reicht der Vorrat für eine Quote nicht, wird das gemeldet und mit dem Vorhandenen weitergemacht.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

log = logging.getLogger("llm_grader_bias.pilot")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def task_of(row: dict) -> str:
    from .prompting import task_cluster
    return task_cluster(row["id"])


def eligible(row: dict, cfg: dict) -> bool:
    year = row.get("jahr")
    if year is None:
        if row["fach"] not in cfg.get("allow_without_year", []):
            return False
    elif cfg.get("min_year") and year < cfg["min_year"]:
        return False
    limit = cfg.get("max_context_chars") or 0
    if limit and len(row.get("kontext", "")) > limit:
        return False
    return True


def select(rows: list[dict], cfg: dict) -> tuple[list[dict], list[str]]:
    rng = random.Random(cfg.get("seed", 42))
    warnings: list[str] = []
    pool = [r for r in rows if eligible(r, cfg)]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["id"]):   # stabile Reihenfolge vor dem Mischen
        by_type[r["aufgabentyp"]].append(r)

    max_per_task = cfg.get("max_per_task") or 10**9
    chosen: list[dict] = []
    for typ, quota in cfg["quotas"].items():
        candidates = by_type.get(typ, [])
        rng.shuffle(candidates)
        per_task: Counter[str] = Counter()
        picked: list[dict] = []

        if cfg.get("balance_subjects", True):
            by_subject: dict[str, list[dict]] = defaultdict(list)
            for r in candidates:
                by_subject[r["fach"]].append(r)
            subjects = sorted(by_subject)
            rng.shuffle(subjects)
            # reihum je Fach eine Teilaufgabe, bis Quote voll oder alles aufgebraucht
            while len(picked) < quota and any(by_subject.values()):
                for s in subjects:
                    while by_subject[s]:
                        r = by_subject[s].pop()
                        if per_task[task_of(r)] < max_per_task:
                            picked.append(r)
                            per_task[task_of(r)] += 1
                            break
                    if len(picked) >= quota:
                        break
        else:
            for r in candidates:
                if len(picked) >= quota:
                    break
                if per_task[task_of(r)] < max_per_task:
                    picked.append(r)
                    per_task[task_of(r)] += 1

        if len(picked) < quota:
            warnings.append(f"{typ}: nur {len(picked)} von {quota} möglich (Vorrat {len(candidates)})")
        chosen.extend(picked)
    chosen.sort(key=lambda r: (r["aufgabentyp"], r["fach"], r["id"]))
    return chosen, warnings


def summary_md(chosen: list[dict], cfg: dict, warnings: list[str]) -> str:
    by_type = Counter(r["aufgabentyp"] for r in chosen)
    by_subject = Counter(r["fach"] for r in chosen)
    by_year = Counter("Beispiel" if r["jahr"] is None else str(r["jahr"]) for r in chosen)
    lines = [
        "# Pilot-Auswahl",
        "",
        f"- Teilaufgaben: **{len(chosen)}** aus {len({task_of(r) for r in chosen})} Prüfungsaufgaben",
        f"- Seed {cfg.get('seed')}, ab Jahr {cfg.get('min_year')}, ohne Jahr erlaubt: {cfg.get('allow_without_year')}",
        f"- Max. {cfg.get('max_per_task')} Teilaufgaben pro Aufgabe",
        "",
        "| Aufgabentyp | n |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted(by_type.items())],
        "",
        "| Fach | n |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted(by_subject.items())],
        "",
        "| Jahr | n |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted(by_year.items())],
    ]
    if warnings:
        lines += ["", "## Hinweise", *[f"- {w}" for w in warnings]]
    lines += ["", "## Teilaufgaben", "", "| id | Fach | Jahr | Punkte |", "|---|---|---|---|"]
    for r in chosen:
        pts = r["max_punkte"] if r["max_punkte"] is not None else f"{r['gewichtung_prozent']} %"
        lines.append(f"| `{r['id']}` | {r['fach']} | {r['jahr'] or '–'} | {pts} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_grader_bias.pilot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    ap.add_argument("--n-per-type", type=int, default=None, help="Quote je Aufgabentyp überschreiben")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.n_per_type:
        cfg["quotas"] = {k: args.n_per_type for k in cfg["quotas"]}
    if args.seed is not None:
        cfg["seed"] = args.seed

    rows = load_jsonl(Path(cfg["dataset"]))
    chosen, warnings = select(rows, cfg)
    for w in warnings:
        log.warning(w)

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_suffix(".md").write_text(summary_md(chosen, cfg, warnings), encoding="utf-8")
    log.info("Pilot: %d Teilaufgaben → %s (Übersicht: %s)", len(chosen), out, out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
