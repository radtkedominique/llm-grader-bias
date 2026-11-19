"""Kommandozeile für den Datensatz-Build.

    python -m llm_grader_bias.dataset all
    python -m llm_grader_bias.dataset catalog --max-pages 2
    python -m llm_grader_bias.dataset download
    python -m llm_grader_bias.dataset extract
    python -m llm_grader_bias.dataset report
    python -m llm_grader_bias.dataset export     # fertiger Datensatz → data/processed/dataset.jsonl
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml
from tqdm import tqdm

from .assemble import process_entry
from .export import export
from .http import PoliteClient, sha256_file
from .models import CatalogEntry, Subtask, Task, read_jsonl, write_jsonl
from .sources import iqb, isb

log = logging.getLogger("llm_grader_bias.dataset")


# --------------------------------------------------------------------------- Hilfen

def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_client(cfg: dict) -> PoliteClient:
    h = cfg.get("http", {})
    return PoliteClient(
        user_agent=h.get("user_agent", "llm-grader-bias/0.1"),
        delay_seconds=h.get("delay_seconds", 1.0),
        timeout_seconds=h.get("timeout_seconds", 60),
        retries=h.get("retries", 3),
    )


def paths(cfg: dict, root: Path) -> dict[str, Path]:
    p = cfg.get("paths", {})
    return {k: root / p.get(k, f"data/{k}") for k in ("raw", "interim", "processed")}


def passes_filters(e: CatalogEntry, f: dict) -> bool:
    if e.beispiel and not f.get("include_beispielaufgaben", True):
        return False
    min_year = f.get("min_year")
    if min_year and e.year is not None and e.year < min_year:
        return False
    haystack = " ".join([e.title, e.url, *map(str, e.meta.values())]).lower()
    return not any(k.lower() in haystack for k in f.get("exclude_keywords", []))


def local_pdf_path(raw: Path, e: CatalogEntry, url: str, suffix: str = "") -> Path:
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    name = re.sub(r"[^\w.\-]+", "_", name)
    if suffix:
        name = name.replace(".pdf", f"{suffix}.pdf")
    return raw / e.source / e.subject / name


# --------------------------------------------------------------------------- Schritte

def cmd_catalog(cfg: dict, root: Path, max_pages: int | None, subjects: set[str] | None = None) -> Path:
    client = make_client(cfg)
    P = paths(cfg, root)
    filters = cfg.get("filters", {})
    entries: list[CatalogEntry] = []
    skipped = 0

    src = cfg.get("sources", {})
    if src.get("iqb", {}).get("enabled", True):
        for e in iqb.catalog(client, src["iqb"], max_pages=max_pages, only=subjects):
            if passes_filters(e, filters):
                entries.append(e)
            else:
                skipped += 1
    if src.get("isb_informatik", {}).get("enabled", True) and (not subjects or "informatik" in subjects):
        for e in isb.catalog(client, src["isb_informatik"], P["raw"], filters.get("min_year")):
            if passes_filters(e, filters):
                entries.append(e)
            else:
                skipped += 1

    # doppelte IDs vermeiden
    seen: Counter[str] = Counter()
    for e in entries:
        seen[e.task_id] += 1
        if seen[e.task_id] > 1:
            e.task_id = f"{e.task_id}_{seen[e.task_id]}"

    out = P["processed"] / "catalog.jsonl"
    n = write_jsonl(out, entries)
    log.info("Katalog: %d Aufgaben (%d durch Filter übersprungen) → %s", n, skipped, out)
    return out


def cmd_download(cfg: dict, root: Path) -> None:
    client = make_client(cfg)
    P = paths(cfg, root)
    cat_path = P["processed"] / "catalog.jsonl"
    entries = [CatalogEntry(**r) for r in read_jsonl(cat_path)]
    failed = 0
    for e in tqdm(entries, desc="Download", unit="pdf"):
        try:
            if not e.local_path or not Path(e.local_path).exists():
                target = local_pdf_path(P["raw"], e, e.url)
                got = client.download(e.url, target)
                e.local_path = str(got) if got else None
            if e.reference_url and (not e.reference_local_path or not Path(e.reference_local_path).exists()):
                target = local_pdf_path(P["raw"], e, e.reference_url, suffix="")
                got = client.download(e.reference_url, target)
                e.reference_local_path = str(got) if got else None
            if e.local_path:
                e.sha256 = sha256_file(Path(e.local_path))
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log.warning("Download fehlgeschlagen %s: %s", e.url, exc)
    write_jsonl(cat_path, entries)
    log.info("Download fertig: %d ok, %d fehlgeschlagen", len(entries) - failed, failed)


def cmd_extract(cfg: dict, root: Path) -> None:
    P = paths(cfg, root)
    entries = [CatalogEntry(**r) for r in read_jsonl(P["processed"] / "catalog.jsonl")]
    tasks: list[Task] = []
    subtasks: list[Subtask] = []
    filters = cfg.get("filters", {})
    skipped = 0
    for e in tqdm(entries, desc="Extraktion", unit="pdf"):
        if not e.local_path or not Path(e.local_path).exists():
            continue
        if not passes_filters(e, filters):   # Filter auch hier, damit Änderungen ohne neuen Crawl greifen
            skipped += 1
            continue
        try:
            t, subs = process_entry(e, P["interim"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Extraktion fehlgeschlagen %s: %s", e.task_id, exc)
            continue
        tasks.append(t)
        subtasks.extend(subs)
    write_jsonl(P["processed"] / "tasks.jsonl", tasks)
    write_jsonl(P["processed"] / "subtasks.jsonl", subtasks)
    log.info("Extraktion: %d Aufgaben, %d Teilaufgaben (%d durch Filter übersprungen)",
             len(tasks), len(subtasks), skipped)


def cmd_report(cfg: dict, root: Path) -> Path:
    P = paths(cfg, root)
    tasks = list(read_jsonl(P["processed"] / "tasks.jsonl"))
    subs = list(read_jsonl(P["processed"] / "subtasks.jsonl"))

    def table(rows: list[dict], key, cols: list[str], colfn) -> list[str]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        lines = ["| " + " | ".join(["Gruppe", *cols]) + " |", "|" + "---|" * (len(cols) + 1)]
        for g in sorted(groups):
            lines.append("| " + " | ".join([g, *[str(colfn(c, groups[g])) for c in cols]]) + " |")
        return lines

    def sub_col(c: str, rows: list[dict]) -> int:
        return {
            "Teilaufgaben": len(rows),
            "high": sum(r["split_confidence"] == "high" for r in rows),
            "medium": sum(r["split_confidence"] == "medium" for r in rows),
            "low": sum(r["split_confidence"] == "low" for r in rows),
            "Formeln zerfallen": sum(r.get("fragment_ratio", 0) >= 0.3 for r in rows),
            "nur Text": sum(r["text_only_candidate"] for r in rows),
            "nur Text + high": sum(r["text_only_candidate"] and r["split_confidence"] == "high" for r in rows),
            "mit BE": sum(r["be"] is not None for r in rows),
            "mit BE/%": sum(r["be"] is not None or r.get("gewichtung") is not None for r in rows),
        }[c]

    def task_col(c: str, rows: list[dict]) -> int:
        return {
            "Aufgaben": len(rows),
            "ok": sum(r["parse_status"] == "ok" for r in rows),
            "ohne EH": sum(r["parse_status"] == "no_reference" for r in rows),
            "ohne Abschnitte": sum(r["parse_status"] == "no_sections" for r in rows),
            "mit Bildern": sum(r["n_images"] > 0 for r in rows),
        }[c]

    year = lambda r: "Beispiel" if r.get("year") is None else str(r["year"])  # noqa: E731
    md = [
        "# Datensatz-Report",
        "",
        f"- Aufgaben (PDFs): **{len(tasks)}**",
        f"- Teilaufgaben: **{len(subs)}**",
        f"- davon reine Textaufgaben mit hoher Zuordnungs-Konfidenz: "
        f"**{sum(s['text_only_candidate'] and s['split_confidence'] == 'high' for s in subs)}**",
        "",
        "## Aufgaben nach Fach",
        *table(tasks, lambda r: r["subject"], ["Aufgaben", "ok", "ohne EH", "ohne Abschnitte", "mit Bildern"], task_col),
        "",
        "## Teilaufgaben nach Fach",
        *table(subs, lambda r: r["subject"],
               ["Teilaufgaben", "high", "medium", "low", "mit BE/%", "Formeln zerfallen", "nur Text", "nur Text + high"],
               sub_col),
        "",
        "## Teilaufgaben nach Jahr",
        *table(subs, year, ["Teilaufgaben", "high", "nur Text + high"], sub_col),
        "",
        "- `high`: Teilaufgabe, BE und passende Stelle im Erwartungshorizont gefunden",
        "- `medium`: Erwartungshorizont gefunden, aber BE fehlen oder weichen ab",
        "- `low`: kein passender Erwartungshorizont → vor Nutzung prüfen",
        "- `Formeln zerfallen`: ≥ 30 % sehr kurze Zeilen – Formeln sind als Text nur bruchstückhaft lesbar",
        "- `nur Text`: kein Abbildungsverweis, Text sauber → Kandidat für die Studie",
        "- Lesbare Fassung pro Aufgabe: `data/interim/<quelle>/<fach>/<task_id>.txt`",
    ]
    out = P["processed"] / "report.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("Report → %s", out)
    return out


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_grader_bias.dataset", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["catalog", "download", "extract", "report", "export", "all"])
    ap.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    ap.add_argument("--root", type=Path, default=Path("."), help="Projektwurzel (Standard: aktueller Ordner)")
    ap.add_argument("--max-pages", type=int, default=None, help="Listenseiten pro Fach begrenzen (Probelauf)")
    ap.add_argument("--subjects", type=str, default=None,
                    help="nur diese Fächer, kommagetrennt (z.B. mathematik,deutsch,informatik)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.root / args.config if not args.config.is_absolute() else args.config)

    if args.command in ("catalog", "all"):
        only = {x.strip().lower() for x in args.subjects.split(",")} if args.subjects else None
        cmd_catalog(cfg, args.root, args.max_pages, only)
    if args.command in ("download", "all"):
        cmd_download(cfg, args.root)
    if args.command in ("extract", "all"):
        cmd_extract(cfg, args.root)
    if args.command in ("report", "all"):
        cmd_report(cfg, args.root)
    if args.command in ("export", "all"):
        out, stats = export(paths(cfg, args.root)["processed"])
        log.info("Datensatz: %d Teilaufgaben → %s", sum(stats.values()), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
