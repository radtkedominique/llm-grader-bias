"""Phase 2: Jedes Modell löst jede Teilaufgabe.

    python -m llm_grader_bias.solve --dry-run          # nur zählen und Tokens schätzen
    python -m llm_grader_bias.solve                    # alle aktivierten Modelle aus configs/models.yaml
    python -m llm_grader_bias.solve --models mock-a-small,mock-b-small --runs 2

Ausgabe: data/solutions/solutions.jsonl (eine Zeile pro Lösung). Abgebrochene Läufe setzen dort fort,
wo sie aufgehört haben.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from ..llm import LLMClient, load_env, load_models
from ..prompting import anonymize, load_prompt, task_values
from ..runner import JsonlSink, estimate, now, read_jsonl, run_per_model

log = logging.getLogger("llm_grader_bias.solve")


def solution_id(task_id: str, model_id: str, run: int) -> str:
    return f"{task_id}::{model_id}::r{run}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_grader_bias.solve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, default=Path("data/pilot/pilot.jsonl"))
    ap.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    ap.add_argument("--models", type=str, default=None, help="nur diese Modell-IDs, kommagetrennt")
    ap.add_argument("--prompt", type=Path, default=Path("prompts/solve_v1.md"))
    ap.add_argument("--runs", type=int, default=1, help="Wiederholungen pro Modell und Aufgabe (anderer Seed)")
    ap.add_argument("--out", type=Path, default=Path("data/solutions/solutions.jsonl"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    load_env()
    only = {m.strip() for m in args.models.split(",")} if args.models else None
    specs = load_models(args.models_config, only)
    tasks = read_jsonl(args.tasks)
    prompt = load_prompt(args.prompt)
    sink = JsonlSink(args.out, "solution_id")

    jobs: dict[str, list] = {s.id: [] for s in specs}
    for spec in specs:
        for run in range(1, args.runs + 1):
            run_spec = spec if run == 1 or spec.seed is None else dataclasses.replace(spec, seed=spec.seed + run - 1)
            for t in tasks:
                sid = solution_id(t["id"], spec.id, run)
                if sid not in sink.done:
                    jobs[spec.id].append((run_spec, run, t, sid))

    if args.dry_run:
        msgs = (prompt.render(**task_values(t)) for js in jobs.values() for (_, _, t, _) in js)
        est = estimate(msgs, out_tokens=600)
        log.info("Dry run: %d Modelle × %d Aufgaben × %d Läufe → %d offene Aufrufe, ca. %d Input- und %d Output-Tokens",
                 len(specs), len(tasks), args.runs, est["calls"], est["input_tokens"], est["output_tokens"])
        return 0

    clients: dict[tuple, LLMClient] = {}

    def worker(job):
        spec, run, task, sid = job
        key = (spec.id, spec.seed)
        client = clients.setdefault(key, LLMClient(spec))
        values = task_values(task)
        c = client.complete(prompt.render(**values), use_cache=not args.no_cache)
        anon, removed = anonymize(c.text, values["kontext"] + "\n" + values["aufgabe"])
        sink.write({
            "solution_id": sid, "task_id": task["id"], "fach": task["fach"], "aufgabentyp": task["aufgabentyp"],
            "model_id": spec.id, "family": spec.family, "size": spec.size, "run": run, "seed": spec.seed,
            "provider": spec.provider, "model": spec.model, "model_reported": c.model_reported,
            "prompt": prompt.name, "text_raw": c.text, "text": anon, "anon_removed": removed,
            "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "finish_reason": c.finish_reason,
            "latency_s": c.latency_s, "cached": c.cached, "empty": not anon.strip(), "created_at": now(),
        })

    ok, err = run_per_model(jobs, worker, "Lösen")
    log.info("Fertig: %d Lösungen neu, %d Fehler → %s", ok, err, args.out)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
