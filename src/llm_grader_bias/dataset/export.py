"""Nutzbare Teilaufgaben → ein fertiger Datensatz (data/processed/dataset.jsonl).

Auswahl: split_confidence == "high" und text_only_candidate == True.
Dubletten (z. B. dieselbe Teilaufgabe in der WTR- und der CAS-Fassung) werden entfernt.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from .models import read_jsonl, write_jsonl

AUFGABENTYP = {
    "deutsch": "sprache", "englisch": "sprache", "franzoesisch": "sprache",
    "mathematik": "mint", "physik": "mint", "chemie": "mint", "biologie": "mint",
    "informatik": "informatik",
}
FIELDS = """
id               eindeutige ID der Teilaufgabe
fach             deutsch, englisch, franzoesisch, mathematik, physik, chemie, biologie, informatik
aufgabentyp      sprache | mint | informatik  (für RQ4)
jahr             Prüfungsjahr, null = Beispielaufgabe
niveau           grundlegend | erhoeht
quelle           iqb | isb
kontext          Einleitung und Material (Texte, Daten), die zum Lösen nötig sind
aufgabe          eigentliche Aufgabenstellung der Teilaufgabe
erwartungshorizont  offizielle Musterlösung zu dieser Teilaufgabe
max_punkte       Bewertungseinheiten (BE); null bei Sprachaufgaben mit Prozentgewichtung
gewichtung_prozent  Gewichtung in Prozent (Sprachen), sonst null
quelle_url       Original-PDF
"""


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def export(processed: Path) -> tuple[Path, Counter]:
    rows = []
    seen: set[str] = set()
    dupes = 0
    tasks = {t["task_id"]: t for t in read_jsonl(processed / "tasks.jsonl")}
    for s in read_jsonl(processed / "subtasks.jsonl"):
        if s["split_confidence"] != "high" or not s["text_only_candidate"]:
            continue
        key = hashlib.sha1((_norm(s["prompt_text"]) + "|" + _norm(s["reference_text"])).encode()).hexdigest()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        t = tasks.get(s["task_id"], {})
        rows.append({
            "id": s["subtask_id"],
            "fach": s["subject"],
            "aufgabentyp": AUFGABENTYP.get(s["subject"], "sonstige"),
            "jahr": s["year"],
            "niveau": s["niveau"],
            "quelle": t.get("source"),
            "kontext": s["context_text"],
            "aufgabe": s["prompt_text"],
            "erwartungshorizont": s["reference_text"],
            "max_punkte": s["be"],
            "gewichtung_prozent": s.get("gewichtung"),
            "quelle_url": t.get("url"),
        })
    rows.sort(key=lambda r: (r["aufgabentyp"], r["fach"], r["jahr"] or 0, r["id"]))
    out = processed / "dataset.jsonl"
    write_jsonl(out, rows)

    stats = Counter(r["fach"] for r in rows)
    years = Counter("Beispiel" if r["jahr"] is None else str(r["jahr"]) for r in rows)
    card = [
        "# Datensatz `dataset.jsonl`",
        "",
        f"- Teilaufgaben: **{len(rows)}** (Dubletten entfernt: {dupes})",
        "- Auswahl: Zuordnung Aufgabe ↔ Erwartungshorizont sicher (`high`), kein Abbildungsverweis, "
        "Formeln lesbar, Materialtext vorhanden",
        "- Nur für Forschungszwecke. Nicht weiterveröffentlichen (Urheberrecht IQB / ISB Bayern).",
        "",
        "## Nach Fach",
        "| Fach | Teilaufgaben |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(stats.items())],
        "",
        "## Nach Jahr",
        "| Jahr | Teilaufgaben |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(years.items())],
        "",
        "## Felder",
        "```",
        FIELDS.strip(),
        "```",
    ]
    (processed / "dataset.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    return out, stats
