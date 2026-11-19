"""IQB – Gemeinsamer Abituraufgabenpool der Länder.

Aufbau der Website (Stand Nov 2026):
- Übersicht: https://www.iqb.hu-berlin.de/de/schule/aufgaben/sekii/
- Pro Fach eine Liste, 12 Aufgaben pro Seite, weitere Seiten über ``?page=N``
  (der Button „Mehr laden“ lädt dieselben Seiten nach).
- Jede Aufgabe ist ein Block mit Überschrift, Download-Link
  ``/media/exercise_files/Abituraufgaben_<Fach>/<Code>_Aufgabe.pdf`` und
  „Informationen zur Aufgabe“ (Label/Wert-Paare: Anforderungsniveau, Sachgebiet, Prüfungsjahr …).
- Das PDF enthält Aufgabe UND Erwartungshorizont (Abschnitte „1 Aufgabe“, „2 Erwartungshorizont“ …).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterator
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from ..http import PoliteClient
from ..models import CatalogEntry

log = logging.getLogger(__name__)

EXERCISE_PATH = "/media/exercise_files/"

# Label-Anfänge aus „Informationen zur Aufgabe“ (über alle Fächer gesammelt)
KNOWN_LABELS = (
    "Anforderungsniveau",
    "Sachgebiet",
    "Prüfungsteil",
    "Aufgabengruppe",
    "Digitales Hilfsmittel",
    "Prüfungsjahr",
    "Aufgabenart",
    "Aufgabentyp",
    "Gattung",
    "Lektüre",
    "Inhaltsbereich",
    "Inhaltsbereiche",
    "Kompetenzbereich",
    "Themenbereich",
    "Teilbereich",
    "Fach",
)

SUBJECT_LINK_RE = re.compile(r"abiturpr(?:%c3%bc|ü|ue)fungsaufgaben-([^/]+)/?$", re.I)
YEAR_RE = re.compile(r"\b(20[1-3]\d)\b")
STEM_YEAR_RE = re.compile(r"^(20[1-3]\d)")  # Dateiname beginnt mit dem Jahr, z.B. 2025MerhoehtB…


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", unquote(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def discover_subjects(client: PoliteClient, overview_url: str) -> dict[str, str]:
    """Findet die Fachseiten auf der Übersichtsseite. Leeres Dict, wenn nichts erkannt wird."""
    try:
        html = client.get_text(overview_url)
    except Exception as exc:  # noqa: BLE001
        log.warning("IQB-Übersicht nicht ladbar (%s) – nutze Fächerliste aus der Config", exc)
        return {}
    soup = BeautifulSoup(html, "html.parser")
    subjects: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(overview_url, a["href"])
        m = SUBJECT_LINK_RE.search(urlparse(href).path)
        if m:
            name = _slug(m.group(1)).replace("_", "")
            name = {"franzosisch": "franzoesisch"}.get(name, name)
            subjects.setdefault(name, href if href.endswith("/") else href + "/")
    return subjects


def _block_for_link(link: Tag) -> Tag:
    """Kleinster Vorfahre, der eine Überschrift enthält und nur diese eine Aufgabe umfasst."""
    best: Tag = link
    node: Tag | None = link
    while node is not None and node.parent is not None and node.name != "body":
        parent = node.parent
        links = {a["href"] for a in parent.find_all("a", href=True) if EXERCISE_PATH in a["href"]}
        if len(links) > 1:
            break
        node = parent
        if node.find(["h2", "h3", "h4"]):
            best = node
    return best


def _parse_meta(block: Tag) -> dict[str, str]:
    meta: dict[str, str] = {}

    # 1) Definition lists
    for dt in block.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            meta[dt.get_text(" ", strip=True).rstrip(":")] = dd.get_text(" ", strip=True)
    if meta:
        return meta

    # 2) Textfolge: Label, Wert, Label, Wert …
    strings = [s.strip() for s in block.stripped_strings if s.strip()]
    i = 0
    while i < len(strings):
        s = strings[i]
        if ":" in s and s.split(":", 1)[0].strip().startswith(KNOWN_LABELS):
            label, value = s.split(":", 1)
            if value.strip():
                meta[label.strip()] = value.strip()
                i += 1
                continue
            if i + 1 < len(strings):
                meta[label.strip()] = strings[i + 1]
                i += 2
                continue
        if s.rstrip(":").startswith(KNOWN_LABELS) and i + 1 < len(strings):
            nxt = strings[i + 1]
            if not nxt.rstrip(":").startswith(KNOWN_LABELS):
                meta[s.rstrip(":").strip()] = nxt
                i += 2
                continue
        i += 1
    return meta


def _meta_value(meta: dict[str, str], prefix: str) -> str | None:
    for k, v in meta.items():
        if k.startswith(prefix):
            return v
    return None


def _niveau(text: str) -> str | None:
    t = text.lower()
    if "erhöht" in t or "erhoeht" in t:
        return "erhoeht"
    if "grundlegend" in t:
        return "grundlegend"
    return None


def parse_listing(html: str, page_url: str, subject: str) -> list[CatalogEntry]:
    """Parst eine Listenseite und gibt alle Aufgaben darauf zurück."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if EXERCISE_PATH not in href or not href.lower().endswith(".pdf"):
            continue
        url = urljoin(page_url, href)
        if url in seen:
            continue
        seen.add(url)

        block = _block_for_link(link)
        heading = block.find(["h2", "h3", "h4"])
        title = heading.get_text(" ", strip=True) if heading else ""
        meta = _parse_meta(block)

        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        stem = re.sub(r"_Aufgabe$", "", filename.rsplit(".", 1)[0])

        jahr_text = _meta_value(meta, "Prüfungsjahr") or ""
        beispiel = "beispiel" in (jahr_text + " " + stem + " " + title).lower()
        year = None
        m = YEAR_RE.search(jahr_text) or STEM_YEAR_RE.match(stem)
        if m and not beispiel:
            year = int(m.group(1))

        niveau = _niveau(_meta_value(meta, "Anforderungsniveau") or "") or _niveau(stem) or _niveau(title)

        entries.append(
            CatalogEntry(
                task_id=f"iqb_{subject}_{_slug(stem)}",
                source="iqb",
                kind="iqb_pool",
                subject=subject,
                url=url,
                title=title,
                year=year,
                beispiel=beispiel,
                niveau=niveau,
                reference_expected=True,
                meta=meta,
            )
        )
    return entries


def crawl_subject(
    client: PoliteClient, subject: str, url: str, max_pages: int
) -> Iterator[CatalogEntry]:
    """Geht ``?page=1..N`` durch, bis eine Seite keine neuen Aufgaben mehr bringt."""
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}?page={page}"
        try:
            html = client.get_text(page_url)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] Seite %d nicht ladbar: %s – breche ab", subject, page, exc)
            break
        new = [e for e in parse_listing(html, page_url, subject) if e.url not in seen]
        if not new:
            log.info("[%s] keine neuen Aufgaben auf Seite %d – fertig", subject, page)
            break
        for e in new:
            seen.add(e.url)
            yield e
        log.info("[%s] Seite %d: %d Aufgaben (gesamt %d)", subject, page, len(new), len(seen))


def catalog(
    client: PoliteClient, cfg: dict, max_pages: int | None = None, only: set[str] | None = None
) -> Iterator[CatalogEntry]:
    subjects = discover_subjects(client, cfg["overview_url"]) or {}
    for name, url in (cfg.get("subjects") or {}).items():
        subjects.setdefault(name, url)
    if only:
        subjects = {k: v for k, v in subjects.items() if k in only}
    limit = max_pages or cfg.get("max_pages_per_subject", 200)
    for name, url in sorted(subjects.items()):
        yield from crawl_subject(client, name, url, limit)
