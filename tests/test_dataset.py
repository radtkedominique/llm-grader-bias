"""Tests mit synthetischen Seiten/PDFs, die die echten Layouts von IQB und ISB nachbilden."""

from pathlib import Path

import pymupdf

from llm_grader_bias.dataset import layout as L
from llm_grader_bias.dataset.assemble import _pair
from llm_grader_bias.dataset.models import CatalogEntry
from llm_grader_bias.dataset.sources.iqb import parse_listing

# --------------------------------------------------------------------------- Katalog

LISTING_HTML = """
<html><body>
<div class="results">
  <div class="exercise">
    <h2>2025 - Analysis - erhöht - Bearbeitung mit Hilfsmitteln - MMS(CAS) - 01</h2>
    <a href="/media/exercise_files/Abituraufgaben_Mathematik/2025MerhoehtBAnalysisMMS1_Aufgabe.pdf"
       title="download-button">Aufgabe</a>
    <h3>Informationen zur Aufgabe</h3>
    <div><span>Sachgebiet</span><span>Analysis</span></div>
    <div><span>Anforderungsniveau</span><span>erhöht</span></div>
    <div><span>Prüfungsjahr/Beispielaufgaben</span><span>2025</span></div>
  </div>
  <div class="exercise">
    <h2>Beispielaufgabe - Analytische Geometrie - grundlegend - WTR - 01</h2>
    <a href="/media/exercise_files/Abituraufgaben_Mathematik/BeispielaufgabenMgrundlegendBAGLAA2WTR_Aufgabe.pdf">Aufgabe</a>
    <h3>Informationen zur Aufgabe</h3>
    <dl><dt>Anforderungsniveau</dt><dd>grundlegend</dd>
        <dt>Prüfungsjahr/Beispielaufgaben</dt><dd>Beispielaufgaben</dd></dl>
  </div>
</div>
<button>Mehr laden</button>
</body></html>
"""


def test_parse_listing():
    page = "https://www.iqb.hu-berlin.de/de/schule/aufgaben/sekii/abiturpruefungsaufgaben-mathematik/"
    a, b = parse_listing(LISTING_HTML, page, "mathematik")
    assert a.year == 2025 and not a.beispiel and a.niveau == "erhoeht"
    assert a.meta["Sachgebiet"] == "Analysis"
    assert a.url.startswith("https://www.iqb.hu-berlin.de/media/exercise_files/")
    assert a.task_id == "iqb_mathematik_2025merhoehtbanalysismms1"
    assert b.beispiel and b.year is None and b.niveau == "grundlegend"


# --------------------------------------------------------------------------- PDFs bauen

def make_pdf(path: Path, pages: list[list[tuple[float, float, str]]]) -> Path:
    """Jede Seite: Liste von (x, y, Text) – y ist die Grundlinie."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page(width=595, height=842)
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=10)
    doc.save(path)
    return path


def iqb_pages() -> list[list[tuple[float, float, str]]]:
    """IQB-Layout: Überschrift „1“ + „Aufgabe“ getrennt, Marker 71/83, BE bei 513, Kopfzeile oben."""
    p1 = [
        (71, 120, "Kurzbeschreibung"),
        (71, 300, "1"), (99, 300, "Aufgabe"),
        (508, 330, "BE"),
        (71, 350, "Gegeben ist die Funktion f."),
        (71, 380, "1"), (83, 380, "a Berechnen Sie die Nullstelle."), (513, 380, "2"),
        (80.2, 400, " b Bestimmen Sie die Extrempunkte von f und"),   # führendes Leerzeichen wie im Original
        (99, 414, "begründen Sie die Art."), (513, 400, "7"),
        (83, 440, "Der Graph von f und die x-Achse schließen eine Fläche ein."),
        (83, 470, "c Berechnen Sie den Inhalt dieser Fläche."), (513, 470, "3"),
        (99, 500, "e"),                         # Formelfragment, KEIN Marker (falsche Spalte)
        (71, 560, "2"), (99, 560, "Die Abbildung zeigt einen Wassertank."),
        (83, 600, "a Geben Sie das Volumen an."), (513, 600, "4"),
        (83, 630, "b Skizzieren Sie den Graphen."), (513, 630, "4"),
        (510, 660, "20"),
    ]
    p2 = [
        (489, 30, "2  Erwartungshorizont"),     # irreführende Kopfzeile → muss ignoriert werden
        (71, 120, "2"), (99, 120, "Erwartungshorizont"),
        (508, 150, "BE"),
        (71, 180, "1"), (83, 180, "a x = 2"), (513, 180, "2"),
        (83, 210, "b Hochpunkt (1|2), Tiefpunkt (0|0)."), (513, 210, "7"),
        (83, 240, "c A = 4,5"), (513, 240, "3"),
    ]
    p3 = [
        (489, 30, "3  Standardbezug"),          # Kopfzeile nennt den falschen Abschnitt
        (71, 100, "2"), (83, 100, "a V = 10"), (513, 100, "4"),
        (83, 130, "b Graph steigt streng monoton."), (513, 130, "4"),
        (510, 160, "20"),
        (71, 220, "3"), (99, 220, "Standardbezug"),
        (71, 250, "Teilaufgabe BE K1 K2"),
        (71, 270, "1 a 2 I"),
    ]
    return [p1, p2, p3]


def test_iqb_layout(tmp_path: Path):
    pdf = L.read_pdf(make_pdf(tmp_path / "iqb.pdf", iqb_pages()))
    assert pdf.section_found
    a = L.parse_groups(pdf.sections[L.AUFGABE], pdf.width)
    r = L.parse_groups(pdf.sections[L.REFERENCE], pdf.width)
    assert [(g.key, [p.letter for p in g.parts]) for g in a] == [("1", ["a", "b", "c"]), ("2", ["a", "b"])]
    assert [(g.key, [p.letter for p in g.parts]) for g in r] == [("1", ["a", "b", "c"]), ("2", ["a", "b"])]
    assert [p.be for g in a for p in g.parts] == [2, 7, 3, 4, 4]
    assert a[-1].total_be == 20
    # Zwischentext gehört zur folgenden Teilaufgabe c, nicht zu b
    b, c = a[0].parts[1], a[0].parts[2]
    assert "schließen eine Fläche" not in b.text
    assert any("schließen eine Fläche" in t for t in c.intro_before)
    assert "begründen Sie die Art" in b.text
    assert "Wassertank" in " ".join(a[1].context)
    # EH der Aufgabe 2 steht unter der falschen Kopfzeile „Standardbezug“ – trotzdem gefunden
    assert r[1].parts[1].text.startswith("Graph steigt")


def test_pair_iqb(tmp_path: Path):
    pdf = L.read_pdf(make_pdf(tmp_path / "iqb.pdf", iqb_pages()))
    a = L.parse_groups(pdf.sections[L.AUFGABE], pdf.width)
    r = L.parse_groups(pdf.sections[L.REFERENCE], pdf.width)
    e = CatalogEntry(task_id="t", source="iqb", kind="iqb_pool", subject="mathematik", url="u")
    subs = _pair(e, a, r, "x", "y")
    assert [s.subtask_id for s in subs] == ["t/1-a", "t/1-b", "t/1-c", "t/2-a", "t/2-b"]
    assert all(s.split_confidence == "high" for s in subs)
    assert subs[2].reference_text == "A = 4,5"
    assert "schließen eine Fläche" in subs[2].context_text
    assert "BE-Summe bestätigt" in subs[0].split_notes


def test_bayern_abitur_layout(tmp_path: Path):
    """Bayern-Abitur: „I.“ zentriert, „1.“ bei 65, „a)“ bei 82, BE LINKS bei 35."""
    page = [
        (34, 100, "BE"),
        (294, 70, "I."),
        (65, 130, "1. Für ein Skigebiet soll eine Software entwickelt werden."),
        (82, 147, "Jeder Skipass hat eine eindeutige Nummer."),
        (35, 180, "10"), (82, 180, "a) Erstellen Sie ein Klassendiagramm."),
        (39, 220, "5"), (82, 220, "b) Implementieren Sie die Klasse PISTE."),
        (99, 237, "Die Methoden zum Geben dürfen vorausgesetzt werden."),
        (65, 280, "2. Die Lift-Objekte werden in einem Binärbaum verwaltet."),
        (39, 310, "4"), (82, 310, "a) Zeichnen Sie den Baum."),
        (294, 400, "II."),
        (65, 430, "1. Ein Prozessor arbeitet mit Registern."),
        (39, 460, "3"), (82, 460, "a) Beschreiben Sie den Befehlszyklus."),
    ]
    pdf = L.read_pdf(make_pdf(tmp_path / "by.pdf", [page]))
    groups = L.parse_groups(pdf.sections[L.AUFGABE], pdf.width)
    assert [(g.key, [p.letter for p in g.parts]) for g in groups] == [
        ("I-1", ["a", "b"]), ("I-2", ["a"]), ("II-1", ["a"])
    ]
    assert [p.be for g in groups for p in g.parts] == [10, 5, 4, 3]
    assert "Skigebiet" in " ".join(groups[0].context)
    assert "vorausgesetzt" in groups[0].parts[1].text


def test_fragment_ratio():
    assert L.fragment_ratio("Berechnen Sie\ndie Nullstelle.") == 0
    assert L.fragment_ratio("a\nx\n=\n2\nBerechnen Sie") == 0.8


def test_code_indent_restored(tmp_path: Path):
    page = [
        (536, 60, "BE"),
        (273, 40, "Aufgabe I"),
        (62, 80, "1a Mögliche Lösung in Python:"), (536, 80, "4"),
        (85, 100, "def istVorhanden(self, nr: int) -> bool:"),
        (91, 114, "if self.nr == nr:"),
        (97, 128, "return True"),
        (91, 142, "return False"),
        (64, 170, "b Text ohne Code."), (536, 170, "2"),
    ]
    pdf = L.read_pdf(make_pdf(tmp_path / "code.pdf", [page]))
    g = L.parse_groups(pdf.sections[L.AUFGABE], pdf.width)[0]
    assert g.parts[0].text.splitlines()[1:] == [
        "def istVorhanden(self, nr: int) -> bool:",
        "    if self.nr == nr:",
        "        return True",
        "    return False",
    ]
    assert g.parts[1].text == "Text ohne Code."
