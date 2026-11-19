"""ISB Bayern – Informatik (Coding-Aufgaben in Java/Python).

Zwei Arten von Material:
- **IlluPA** (Illustrierende Prüfungsaufgaben ab 2026): Aufgaben-PDF + separates PDF mit
  Lösungsvorschlägen → mit Erwartungshorizont.
- **Original-Abiturprüfungen**: nur Aufgaben, kein offizieller Erwartungshorizont → nur für die
  Bedingung „ohne Erwartungshorizont“ nutzbar.

Die Dateinamen der Abiturprüfungen wechseln über die Jahre, deshalb werden mehrere Muster probiert.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from ..http import PoliteClient
from ..models import CatalogEntry

log = logging.getLogger(__name__)


def catalog(
    client: PoliteClient, cfg: dict, raw_dir: Path, min_year: int | None = None
) -> Iterator[CatalogEntry]:
    """Beim Suchen der Abitur-PDFs wird die gefundene Datei direkt im Cache abgelegt."""
    for item in cfg.get("illupa", []):
        yield CatalogEntry(
            task_id=f"isb_informatik_{item['id']}",
            source="isb",
            kind="isb_illupa",
            subject="informatik",
            url=item["aufgabe_url"],
            reference_url=item.get("loesung_url"),
            title=f"Illustrierende Prüfungsaufgaben Informatik ({item['niveau']})",
            year=None,
            beispiel=True,
            niveau=item.get("niveau"),
            reference_expected=bool(item.get("loesung_url")),
            meta={"Bundesland": "Bayern", "Stand": "September 2023"},
        )

    abitur = cfg.get("abitur")
    if not abitur:
        return
    base = abitur["base_url"]
    for year in abitur.get("years", []):
        if min_year and year < min_year:
            continue
        found = None
        target = raw_dir / "isb" / "informatik" / f"abitur_{year}.pdf"
        for pattern in abitur.get("filename_patterns", []):
            url = base + pattern.format(year=year)
            if client.download(url, target, allow_404=True):
                found = url
                break
        if not found:
            log.info("[isb] Abitur Informatik %d: keine Datei gefunden", year)
            continue
        log.info("[isb] Abitur Informatik %d: %s", year, found)
        yield CatalogEntry(
            task_id=f"isb_informatik_abitur_{year}",
            source="isb",
            kind="isb_abitur",
            subject="informatik",
            url=found,
            title=f"Abiturprüfung Informatik Bayern {year}",
            year=year,
            niveau=None,
            reference_expected=False,
            meta={"Bundesland": "Bayern"},
            local_path=str(target),
        )
