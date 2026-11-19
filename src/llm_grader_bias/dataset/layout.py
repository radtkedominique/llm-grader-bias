"""Layout-basierte Zerlegung von Abitur-PDFs in Abschnitte, Aufgaben und Teilaufgaben.

Warum über Koordinaten statt nur über Text?
- Formeln zerfallen beim Extrahieren in einzelne Zeichen („a“, „e“, „x“). Echte Marker für
  Teilaufgaben stehen immer in derselben linken Spalte, Formelfragmente weiter rechts.
- Die BE stehen in einer eigenen Spalte (IQB/IlluPA rechts, Bayern-Abitur links), auf Höhe
  des Markers der Teilaufgabe.
- Kopfzeilen nennen teils einen anderen Abschnitt als den Seiteninhalt → Kopf-/Fußband wird
  verworfen, Abschnitte werden nur über die echten Überschriften im Text erkannt.

Unterstützte Layouts (Stand Nov 2026):
- IQB-Pool:      „1“ + „Aufgabe“ als Überschrift, Marker „1 a …“, „ b …“, BE rechts
- ISB IlluPA:    „Aufgabe I“, Marker „1 …“, „a …“ bzw. „1a …“, BE rechts
- ISB Abitur:    „I.“, Marker „1. …“, „a) …“, BE links
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

AUFGABE = "aufgabe"
REFERENCE = "erwartungshorizont"

SECTION_NAMES: list[tuple[str, re.Pattern]] = [
    ("kurzbeschreibung", re.compile(r"^kurzbeschreibung$", re.I)),
    (AUFGABE, re.compile(r"^(aufgabe|aufgabenstellung)$", re.I)),
    ("material", re.compile(r"^materiali?e?n?$", re.I)),
    (REFERENCE, re.compile(r"^(erwartungshorizont|l(ö|oe)sungsvorschl(ä|ae)ge?)$", re.I)),
    ("standardbezug", re.compile(r"^standardbezug$", re.I)),
    ("bewertungshinweise", re.compile(r"^bewertungshinweise$", re.I)),
]

COMBINED_RE = re.compile(r"^(\d{1,2})\.?\s*([a-h])\)?(?:\s+(.*))?$")          # „1 a …“, „1a …“, „1. a) …“
LETTER_RE = re.compile(r"^([a-h])\)?(?:\s+(?![=<>∈⋅·])(.*))?$")  # „a …“, „a)“, „b“, „a ( )“ (EH mit Formel)
NUMBER_RE = re.compile(r"^(\d{1,2})\.?\s+([^\d\s,.;:=].*)$")                  # „1 Um eine …“, „1. Zu …“
# „Aufgabe II“ (IlluPA) oder „II.“ (Abitur Bayern); ein nacktes „II“ ist meist eine Achsen-/Graphbeschriftung
ROMAN_RE = re.compile(r"^(?:Aufgabe\s+(I|II|III|IV|V|VI)|(I|II|III|IV|V|VI)\.)$")
INT_RE = re.compile(r"^\d{1,3}$")
NOISE_RE = re.compile(r"^\(?Fortsetzung( auf der)? nächste[rn]? Seite\)?$|^Seite \d+ \(von \d+\)$", re.I)

FIGURE_RE = re.compile(
    r"\b(Abb\.|Abbildung(en)?|Grafik(en)?|Diagramm(e|en)?|Klassendiagramm|Graphen?|Skizze|Skizzieren|"
    r"Zeichnen|Foto(s|grafie)?|Karikatur|Schaubild|Bild\s*\d|Figure|Fig\.)",
    re.I,
)

HEADER_BAND = 0.045   # oberste 4,5 % der Seite = Kopfzeile
FOOTER_BAND = 0.93    # ab 93 % = Fußzeile / Seitenzahl
Y_TOL = 4.0           # gleiche Zeilenhöhe
COL_TOL = 4.0         # gleiche Spalte


@dataclass
class Line:
    idx: int
    page: int
    x0: float
    y0: float
    text: str

    @property
    def pos(self) -> tuple[int, float]:
        return (self.page, self.y0)


@dataclass
class Part:
    letter: str
    start: tuple[int, float]
    marker_line: int
    lines: list[str] = field(default_factory=list)
    xs: list[float | None] = field(default_factory=list)   # x-Position je Zeile (für Code-Einrückung)
    intro_before: list[str] = field(default_factory=list)  # Zwischentext vor dieser Teilaufgabe
    be: int | None = None
    weight: int | None = None          # Gewichtung in Prozent (Sprachen)
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class Group:
    top: str            # römische Nummer (Bayern) oder ""
    nr: int
    start: tuple[int, float]
    context: list[str] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    total_be: int | None = None

    @property
    def key(self) -> str:
        return f"{self.top}-{self.nr}" if self.top else str(self.nr)


@dataclass
class ParsedPdf:
    n_pages: int
    width: float
    height: float
    lines: list[Line]
    page_texts: list[str]
    images_per_page: list[int]
    sections: dict[str, list[Line]] = field(default_factory=dict)
    section_found: bool = False


# --------------------------------------------------------------------------- Einlesen

def read_pdf(path: Path) -> ParsedPdf:
    doc = pymupdf.open(path)
    lines: list[Line] = []
    page_texts: list[str] = []
    image_xrefs: list[set[int]] = []
    width = height = 0.0
    for pno, page in enumerate(doc):
        width, height = page.rect.width, page.rect.height
        image_xrefs.append({img[0] for img in page.get_images(full=True)})
        page_lines: list[str] = []
        # rawdict liefert jedes Zeichen mit Position: linker Rand = erstes sichtbares Zeichen
        # (die Zeilen-BBox beginnt sonst bei führenden Leerzeichen)
        for block in page.get_text("rawdict")["blocks"]:
            for l in block.get("lines", []):
                for s in l["spans"]:
                    s["text"] = "".join(c["c"] for c in s.get("chars", []))
                first = next((c for s in l["spans"] for c in s.get("chars", []) if not c["c"].isspace()), None)
                text = "".join(s["text"] for s in l["spans"]).replace("­", "").strip()
                if not text:
                    continue
                x0 = first["bbox"][0] if first else l["bbox"][0]
                y0 = l["bbox"][1]
                if y0 < HEADER_BAND * height or y0 > FOOTER_BAND * height:
                    continue
                if NOISE_RE.match(text):
                    continue
                lines.append(Line(len(lines), pno, x0, y0, text))
                page_lines.append(text)
        page_texts.append("\n".join(page_lines))
    doc.close()
    # Logos/Kopfgrafiken kommen auf fast jeder Seite vor → nicht als Abbildung zählen
    counts = Counter(x for xs in image_xrefs for x in xs)
    repeated = {x for x, c in counts.items() if len(image_xrefs) > 1 and c >= max(2, len(image_xrefs) // 2)}
    images = [len(xs - repeated) for xs in image_xrefs]
    parsed = ParsedPdf(len(page_texts), width, height, lines, page_texts, images)
    _find_sections(parsed)
    return parsed


def _section_name(text: str) -> str | None:
    for canon, pat in SECTION_NAMES:
        if pat.match(text.strip()):
            return canon
    return None


def _find_sections(p: ParsedPdf) -> None:
    """IQB-Überschriften: Zahl und Name als getrennte Zeilen auf gleicher Höhe, oder „2 Name“."""
    starts: list[tuple[int, str]] = []  # (Zeilenindex, Name)
    for i, l in enumerate(p.lines):
        m = re.match(r"^(\d)\s+(.+)$", l.text)
        if m and (name := _section_name(m.group(2))):
            starts.append((i, name))
            continue
        if re.fullmatch(r"\d", l.text):
            for other in p.lines[max(0, i - 3): i + 4]:
                if other is not l and other.page == l.page and abs(other.y0 - l.y0) <= Y_TOL and other.x0 > l.x0:
                    name = _section_name(other.text)
                    if name:
                        starts.append((i, name))
                        break
    # Überschriften nach Position sortieren
    starts.sort(key=lambda s: p.lines[s[0]].pos)
    if not any(name == AUFGABE for _, name in starts):
        p.sections = {AUFGABE: list(p.lines)}
        return
    p.section_found = True
    bounds = [(p.lines[i].pos, name, i) for i, name in starts]
    heading_idx = set()
    for i, _ in starts:
        heading_idx.add(i)
        # Namenszeile der Überschrift ebenfalls ausblenden
        for other in p.lines[max(0, i - 3): i + 4]:
            if other.page == p.lines[i].page and abs(other.y0 - p.lines[i].y0) <= Y_TOL:
                heading_idx.add(other.idx)
    for l in p.lines:
        if l.idx in heading_idx:
            continue
        current = None
        for pos, name, _ in bounds:
            if pos <= (l.page, l.y0 + 0.5):
                current = name
            else:
                break
        if current:
            p.sections.setdefault(current, []).append(l)


# --------------------------------------------------------------------------- Spalten

def _cluster_min(values: list[float]) -> float | None:
    """Kleinster x-Wert, der mindestens zweimal (±COL_TOL) vorkommt; sonst der kleinste."""
    if not values:
        return None
    vs = sorted(values)
    for v in vs:
        if sum(abs(v - w) <= COL_TOL for w in vs) >= 2:
            return v
    return vs[0]


def _be_column(lines: list[Line], width: float) -> tuple[float, float]:
    """x-Bereich der BE-Spalte, über die Beschriftung „BE“; sonst rechter Rand."""
    xs = [l.x0 for l in lines if l.text.strip() in ("BE", "Nr. BE")]
    if xs:
        x = Counter(round(v) for v in xs).most_common(1)[0][0]
        return (x - 15, x + 25)
    afb = [l.x0 for l in lines if l.text.strip() in ("BE/AFB", "AFB")]
    if afb:
        return (min(afb) - 20, width)      # Spalten I, II, III rechts neben/unter dem Label
    return (0.8 * width, width)


def _is_be(l: Line, be_col: tuple[float, float]) -> bool:
    return bool(INT_RE.match(l.text)) and be_col[0] <= l.x0 <= be_col[1]


# --------------------------------------------------------------------------- Zerlegen

def _merge_lone_numbers(body: list[Line], letter_col: float) -> list[Line]:
    """„2“ links + „Die Abbildung …“ rechts auf gleicher Höhe → eine Zeile „2 Die Abbildung …“."""
    out: list[Line] = []
    used: set[int] = set()
    by_page: dict[int, list[Line]] = {}
    for l in body:
        by_page.setdefault(l.page, []).append(l)
    for l in body:
        if l.idx in used:
            continue
        if re.fullmatch(r"\d{1,2}\.?", l.text) and l.x0 < letter_col - 2:
            right = [o for o in by_page[l.page]
                     if o.idx != l.idx and o.idx not in used and abs(o.y0 - l.y0) <= Y_TOL and o.x0 > l.x0]
            if right:
                nxt = min(right, key=lambda o: o.x0)
                used.add(nxt.idx)
                out.append(Line(l.idx, l.page, l.x0, l.y0, f"{l.text} {nxt.text}"))
                continue
        out.append(l)
    return out


TEILAUFGABE_RE = re.compile(r"^Teilaufgabe\s+(\d{1,2})\b\.?\s*(.*)$")
WEIGHT_RE = re.compile(r"^\(?(?:ca\.\s*)?(\d{1,3})\s*%\)?$")
SUMME_RE = re.compile(r"^Summe\d?$")


def _mode_x(values: list[float]) -> float | None:
    """Häufigste Spalte (auf 2 pt gerundet); bei Gleichstand die linkeste."""
    if not values:
        return None
    c = Counter(round(v / 2) * 2 for v in values)
    best = max(c.values())
    return min(x for x, n in c.items() if n == best)


def _letter_events(ordered: list[Line], letter_col: float, number_col: float | None, width: float):
    groups: list[Group] = []
    marker_ids: set[int] = set()
    group: Group | None = None
    top = ""
    expected = "a"

    def new_group(nr: int, start: tuple[int, float]) -> Group:
        g = Group(top=top, nr=nr, start=start)
        groups.append(g)
        return g

    for l in ordered:
        t = l.text.strip()
        m_roman = ROMAN_RE.match(t)
        if m_roman and l.x0 > 0.25 * width:
            top = m_roman.group(1) or m_roman.group(2)
            group, expected = None, "a"
            marker_ids.add(l.idx)
            continue

        m = COMBINED_RE.match(t)
        if m and l.x0 <= letter_col + COL_TOL:
            nr, letter, rest = int(m.group(1)), m.group(2), m.group(3) or ""
            if letter == "a":
                group = new_group(nr, l.pos)
                expected = "a"
            if group is not None and letter == expected:
                group.parts.append(Part(letter, l.pos, l.idx, [rest] if rest else []))
                expected = chr(ord(letter) + 1)
                marker_ids.add(l.idx)
                continue

        m = NUMBER_RE.match(t)
        if m and number_col is not None and abs(l.x0 - number_col) <= COL_TOL and l.x0 < letter_col - 2:
            nr = int(m.group(1))
            if nr == (group.nr + 1 if group else 1) or group is None or not group.parts:
                group = new_group(nr, l.pos)
                group.context.append(m.group(2))
                expected = "a"
                marker_ids.add(l.idx)
                continue

        m = LETTER_RE.match(t)
        if m and abs(l.x0 - letter_col) <= COL_TOL:
            letter, rest = m.group(1), m.group(2) or ""
            if letter == "a" and (group is None or group.parts):
                group = new_group((group.nr + 1) if group else 1, l.pos)
                expected = "a"
            if group is not None and letter == expected:
                group.parts.append(Part(letter, l.pos, l.idx, [rest] if rest else []))
                expected = chr(ord(letter) + 1)
                marker_ids.add(l.idx)
                continue
    return groups, marker_ids


def _numbered_events(ordered: list[Line]):
    """Teilaufgaben nur mit Nummern („1 Stellen Sie …“, „1. Analysieren …“, „Teilaufgabe 1“)."""
    cands = [l for l in ordered if NUMBER_RE.match(l.text) or TEILAUFGABE_RE.match(l.text)]
    col = _mode_x([l.x0 for l in cands])
    if col is None:
        return [], set(), None
    group = Group(top="", nr=1, start=(0, -1.0))
    marker_ids: set[int] = set()
    expected = 1
    for l in ordered:
        m = TEILAUFGABE_RE.match(l.text) or (NUMBER_RE.match(l.text) if abs(l.x0 - col) <= COL_TOL else None)
        if not m:
            continue
        nr = int(m.group(1))
        if nr != expected:
            continue
        rest = m.group(2) or ""
        group.parts.append(Part(str(nr), l.pos, l.idx, [rest] if rest else []))
        marker_ids.add(l.idx)
        expected += 1
    return ([group] if group.parts else []), marker_ids, col


def parse_groups(lines: list[Line], width: float) -> list[Group]:
    """Zerlegt die Zeilen eines Abschnitts in Aufgaben (Groups) und Teilaufgaben (Parts).

    Zwei Modi:
    - Buchstaben (Mathe, Physik, Informatik): „1 a …“, „b …“, „a) …“
    - Nummern (Bio, Chemie, Deutsch, Englisch …): „1 …“, „1. …“, „Teilaufgabe 1“
    """
    if not lines:
        return []
    be_col = _be_column(lines, width)
    afb = any(l.text.strip() in ("BE/AFB", "AFB") for l in lines)   # BE nach Anforderungsbereich verteilt
    weights = [l for l in lines if WEIGHT_RE.match(l.text.strip()) and l.x0 > 0.6 * width]
    weight_ids = {l.idx for l in weights}
    labels = {"BE", "Nr.", "Nr. BE", "BE/AFB", "AFB", "I", "II", "III"}
    be_lines = [l for l in lines if _is_be(l, be_col)]
    be_ids = {l.idx for l in be_lines}
    body = [l for l in lines if l.idx not in be_ids and l.idx not in weight_ids
            and not (l.text.strip() in labels and l.x0 > 0.6 * width) and l.text.strip() not in ("BE", "Nr.")]
    summe_rows = [(l.page, l.y0) for l in lines if SUMME_RE.match(l.text.strip())]

    letter_xs = [l.x0 for l in body if LETTER_RE.match(l.text)]
    combined_xs = [l.x0 for l in body if COMBINED_RE.match(l.text)]
    letter_col = _cluster_min(letter_xs) if letter_xs else _cluster_min(combined_xs)

    groups: list[Group] = []
    marker_ids: set[int] = set()
    if letter_col is not None:
        body_l = _merge_lone_numbers(body, letter_col)
        number_col = _cluster_min(combined_xs + [l.x0 for l in body_l if NUMBER_RE.match(l.text) and l.x0 < letter_col - 2])
        ordered = sorted(body_l, key=lambda l: (l.page, round(l.y0 / Y_TOL), l.x0))
        groups, marker_ids = _letter_events(ordered, letter_col, number_col, width)
        if any(g.parts for g in groups):
            body = body_l
    if not any(g.parts for g in groups):
        ordered = sorted(body, key=lambda l: (l.page, round(l.y0 / Y_TOL), l.x0))
        groups, marker_ids, letter_col = _numbered_events(ordered)
    if not any(g.parts for g in groups):
        return []

    # ---- Text den Regionen zuordnen (Zeilen behalten ihre Lesereihenfolge)
    regions: list[tuple[tuple[int, float], Group, Part | None]] = []
    for g in groups:
        regions.append((g.start, g, None))
        for p in g.parts:
            regions.append((p.start, g, p))
    regions.sort(key=lambda r: r[0])

    def region_for(pos: tuple[int, float]):
        found = None
        for start, g, p in regions:
            if start <= (pos[0], pos[1] + 0.5):
                found = (g, p)
            else:
                break
        return found

    inter: dict[int, list[str]] = {}   # id(part) → Zwischentext, der zur nächsten Teilaufgabe gehört
    for l in body:
        if l.idx in marker_ids:
            continue
        r = region_for(l.pos)
        if r is None:
            continue
        g, p = r
        if p is None:
            g.context.append(l.text)
        elif abs(l.x0 - letter_col) <= COL_TOL and l.pos > p.start and not p.letter.isdigit():
            # Zeile in der Marker-Spalte, aber kein Marker → Einleitung für die folgenden Teilaufgaben
            inter.setdefault(id(p), []).append(l.text)
        elif id(p) in inter:
            inter[id(p)].append(l.text)
        else:
            p.lines.append(l.text)
            p.xs.append(l.x0)

    for g in groups:
        for i, p in enumerate(g.parts):
            if id(p) in inter:
                if i + 1 < len(g.parts):
                    g.parts[i + 1].intro_before = inter[id(p)]
                else:
                    p.xs = [None] * (len(p.lines) - len(p.xs)) + p.xs + [None] * len(inter[id(p)])
                    p.lines.extend(inter[id(p)])

    for g in groups:
        for p in g.parts:
            _restore_code_indent(p)

    all_parts = [(g, p) for g in groups for p in g.parts]

    def at_marker(line: Line):
        return next((gp for gp in all_parts
                     if gp[1].start[0] == line.page and abs(gp[1].start[1] - line.y0) <= Y_TOL + 2), None)

    # ---- Gewichtung in Prozent (Deutsch, Englisch …)
    for w in weights:
        hit = at_marker(w) or region_for(w.pos)
        if hit and hit[1] is not None and hit[1].weight is None:
            hit[1].weight = int(WEIGHT_RE.match(w.text.strip()).group(1))

    # ---- BE
    for b in be_lines:
        value = int(b.text)
        if afb:
            # EH mit Spalten I/II/III: alle Zahlen einer Teilaufgabe addieren, Summenzeile auslassen
            if any(pg == b.page and abs(y - b.y0) <= Y_TOL for pg, y in summe_rows):
                continue
            r = region_for(b.pos)
            if r and r[1] is not None:
                r[1].be = (r[1].be or 0) + value
            continue
        hit = at_marker(b)
        if hit and hit[1].be is None:
            hit[1].be = value
            continue
        r = region_for(b.pos)
        if r is None:
            continue
        g, p = r
        if p is None:
            continue
        if p.be is None:
            p.be = value
            p.notes.append("BE nicht auf Markerhöhe")
        elif p is g.parts[-1]:
            g.total_be = value
    return [g for g in groups if g.parts]


CODE_RE = re.compile(
    r"^\s*(def |class |return\b|if\b|else\b|elif\b|for\b|while\b|public |private |protected |"
    r"@\w+|import |\}|.*\{\s*$|.*;\s*$)"
)
INDENT_PT = 22.0  # Standard-Einrückungsschritt, falls nur eine Ebene vorkommt


def _restore_code_indent(p: Part) -> None:
    """Stellt die Einrückung von Code-Zeilen aus den x-Positionen wieder her (Python braucht sie)."""
    # Zeilen ohne Position (Text der Markerzeile) am Anfang auffüllen
    xs = [None] * (len(p.lines) - len(p.xs)) + list(p.xs)
    code_idx = [i for i, t in enumerate(p.lines) if CODE_RE.match(t)]
    if len(code_idx) < 2:
        return
    # Basis = linkester Code-Block: kleinste x-Position der Code-Zeilen
    code_xs = sorted({round(xs[i], 1) for i in code_idx if xs[i] is not None})
    if not code_xs:
        return
    base = code_xs[0]
    gaps = [b - a for a, b in zip(code_xs, code_xs[1:]) if b - a >= 3]
    step = min(gaps) if gaps else INDENT_PT   # Einrückungsschritt aus dem Satz ablesen (ISB: ca. 22 pt)
    out = []
    for t, x in zip(p.lines, xs):
        if x is not None and x - base >= step * 0.6:
            level = round((x - base) / step)
            out.append("    " * level + t)
        else:
            out.append(t)
    p.lines = out


def mentions_figure(text: str) -> bool:
    return bool(FIGURE_RE.search(text))


def fragment_ratio(text: str) -> float:
    """Anteil sehr kurzer Zeilen – hoch bei zerfallenen Formeln."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(len(l.strip()) <= 3 for l in lines) / len(lines)
