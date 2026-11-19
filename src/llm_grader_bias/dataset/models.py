"""Datenstrukturen für Katalog, Aufgaben und Teilaufgaben."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class CatalogEntry:
    """Ein Aufgaben-PDF, wie es in der Liste der Quelle steht."""

    task_id: str
    source: str                      # "iqb" | "isb"
    kind: str                        # "iqb_pool" | "isb_illupa" | "isb_abitur"
    subject: str                     # z.B. "mathematik"
    url: str                         # PDF mit der Aufgabe (bei IQB inkl. Erwartungshorizont)
    title: str = ""
    year: int | None = None
    beispiel: bool = False           # IQB-Beispielaufgabe ohne Prüfungsjahr
    niveau: str | None = None        # "grundlegend" | "erhoeht"
    reference_url: str | None = None # separates PDF mit Lösung (z.B. ISB IlluPA)
    reference_expected: bool = True  # gibt es laut Quelle einen Erwartungshorizont?
    meta: dict = field(default_factory=dict)
    local_path: str | None = None
    reference_local_path: str | None = None
    sha256: str | None = None


@dataclass
class Task:
    """Eine Aufgabe nach der Extraktion (ein PDF bzw. PDF-Paar)."""

    task_id: str
    source: str
    kind: str
    subject: str
    year: int | None
    beispiel: bool
    niveau: str | None
    title: str
    meta: dict
    url: str
    n_pages: int
    n_images: int                    # Bilder im Inhaltsbereich der Aufgabenseiten (ohne Logos)
    mentions_figure: bool            # Text verweist auf Abb./Grafik/Material
    aufgabe_text: str
    reference_text: str              # Erwartungshorizont / Lösungsvorschlag
    other_sections: dict             # Standardbezug, Bewertungshinweise, ...
    parse_status: str                # "ok" | "no_reference" | "no_sections" | "error"
    parse_notes: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)   # Layout-Statistik (Ränder, Marken, Formel-Spans)


@dataclass
class Subtask:
    """Eine Teilaufgabe – die Einheit, die gelöst und korrigiert wird."""

    subtask_id: str                  # "<task_id>/<aufgabe>-<teil>", z.B. ".../1-a" oder ".../I-2-b"
    task_id: str
    subject: str
    year: int | None
    niveau: str | None
    aufgabe: str                     # "1", "2" bzw. "I-1" (Bayern: römischer Block + Nummer)
    teil: str                        # "a", "b", ... oder "" wenn nicht teilbar
    context_text: str                # Einleitung/Material vor der Teilaufgabe
    prompt_text: str                 # eigentliche Aufgabenstellung
    reference_text: str              # passender Teil des Erwartungshorizonts
    be: int | None                   # Bewertungseinheiten
    gewichtung: int | None           # Gewichtung in Prozent (Sprachen haben statt BE Prozentangaben)
    mentions_figure: bool
    fragment_ratio: float            # Anteil sehr kurzer Zeilen (hoch = Formeln zerfallen)
    text_only_candidate: bool        # keine Abbildung, Text sauber extrahiert (heuristisch)
    split_confidence: str            # "high" | "medium" | "low"
    split_notes: list[str] = field(default_factory=list)


def write_jsonl(path: Path, rows: Iterable) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            data = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
