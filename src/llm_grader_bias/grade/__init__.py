"""Phase 3: Jedes Modell korrigiert jede Lösung anonym – mit und ohne Erwartungshorizont.

    python -m llm_grader_bias.grade --dry-run
    python -m llm_grader_bias.grade                                   # beide Bedingungen
    python -m llm_grader_bias.grade --conditions with_reference

Ausgabe: data/gradings/gradings.jsonl, eine Zeile pro Bewertung, inkl. ``relation``
(self / family / foreign) zwischen Korrektor und Autor der Lösung. Fortsetzbar.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from ..llm import LLMClient, load_env, load_models
from ..prompting import load_prompt, max_points, parse_grade, task_values
from ..runner import JsonlSink, estimate, now, read_jsonl, run_per_model

log = logging.getLogger("llm_grader_bias.grade")

PROMPTS = {
    "with_reference": Path("prompts/grade_with_reference_v1.md"),
    "without_reference": Path("prompts/grade_without_reference_v1.md"),
}


def relation(grader_id: str, grader_family: str, author_id: str, author_family: str) -> str:
    if grader_id == author_id:
        return "self"
    if grader_family == author_family:
        return "family"
    return "foreign"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_grader_bias.grade", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, default=Path("data/pilot/pilot.jsonl"))
    ap.add_argument("--solutions", type=Path, default=Path("data/solutions/solutions.jsonl"))
    ap.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    ap.add_argument("--models", type=str, default=None, help="nur diese Korrektoren, kommagetrennt")
    ap.add_argument("--conditions", type=str, default="with_reference,without_reference")
    ap.add_argument("--out", type=Path, default=Path("data/gradings/gradings.jsonl"))
    ap.add_argument("--seed", type=int, default=7, help="Reihenfolge der Aufrufe (gegen Zeiteffekte)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Jede Bewertung n-mal (Wiederholbarkeit). Auswertung des Bias nutzt nur Wiederholung 1.")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    load_env()
    only = {m.strip() for m in args.models.split(",")} if args.models else None
    graders = load_models(args.models_config, only)
    tasks = {t["id"]: t for t in read_jsonl(args.tasks)}
    solutions = [s for s in read_jsonl(args.solutions) if s["task_id"] in tasks and not s.get("empty")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    prompts = {c: load_prompt(PROMPTS[c]) for c in conditions}
    sink = JsonlSink(args.out, "grading_id")

    jobs: dict[str, list] = {g.id: [] for g in graders}
    for g in graders:
        for s in solutions:
            for cond in conditions:
                for rep in range(1, args.repeats + 1):
                    gid = f"{s['solution_id']}::{g.id}::{cond}" + (f"::rep{rep}" if rep > 1 else "")
                    if gid not in sink.done:
                        jobs[g.id].append((g, s, cond, gid, rep))
        random.Random(f"{args.seed}-{g.id}").shuffle(jobs[g.id])

    def messages(g, s, cond):
        values = task_values(tasks[s["task_id"]])
        values["loesung"] = s["text"]
        return prompts[cond].render(**values)

    if args.dry_run:
        est = estimate((messages(g, s, c) for js in jobs.values() for (g, s, c, _, _) in js), out_tokens=150)
        log.info("Dry run: %d Korrektoren × %d Lösungen × %d Bedingungen × %d Wiederholungen → %d offene Aufrufe, "
                 "ca. %d Input- und %d Output-Tokens", len(graders), len(solutions), len(conditions), args.repeats,
                 est["calls"], est["input_tokens"], est["output_tokens"])
        return 0

    clients = {g.id: LLMClient(g) for g in graders}

    def worker(job):
        g, s, cond, gid, rep = job
        task = tasks[s["task_id"]]
        mx, _ = max_points(task)
        c = clients[g.id].complete(messages(g, s, cond), use_cache=not args.no_cache,
                                   salt="" if rep == 1 else f"rep{rep}", json_mode=True)
        parsed = parse_grade(c.text, mx)
        sink.write({
            "grading_id": gid, "solution_id": s["solution_id"], "task_id": s["task_id"],
            "fach": task["fach"], "aufgabentyp": task["aufgabentyp"], "jahr": task.get("jahr"),
            "condition": cond, "repeat": rep,
            "author_model": s["model_id"], "author_family": s["family"], "author_size": s["size"], "run": s["run"],
            "grader_model": g.id, "grader_family": g.family, "grader_size": g.size,
            "relation": relation(g.id, g.family, s["model_id"], s["family"]),
            "punkte": parsed["punkte"], "max_punkte": mx,
            "score": round(parsed["punkte"] / mx, 4) if parsed["punkte"] is not None else None,
            "begruendung": parsed["begruendung"], "parse_ok": parsed["parse_ok"], "parse_note": parsed["parse_note"],
            "solution_chars": len(s["text"]), "prompt": prompts[cond].name, "raw": c.text,
            "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "cached": c.cached,
            "created_at": now(),
        })

    ok, err = run_per_model(jobs, worker, "Korrigieren")
    log.info("Fertig: %d Bewertungen neu, %d Fehler → %s", ok, err, args.out)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
