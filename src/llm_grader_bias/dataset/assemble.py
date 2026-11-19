"""Katalogeintrag + PDF(s) → Task und Subtasks."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from . import layout as L
from .models import CatalogEntry, Subtask, Task

FRAGMENT_LIMIT = 0.3   # ab diesem Anteil sehr kurzer Zeilen gilt der Text als zerfallen (Formeln)


def _text(lines: list[L.Line]) -> str:
    return "\n".join(l.text for l in lines).strip()


def process_entry(e: CatalogEntry, interim: Path) -> tuple[Task, list[Subtask]]:
    notes: list[str] = []
    pdf = L.read_pdf(Path(e.local_path))
    a_lines = pdf.sections.get(L.AUFGABE, [])
    r_lines = pdf.sections.get(L.REFERENCE, [])
    r_width = pdf.width
    raw_pages = list(pdf.page_texts)

    if e.reference_local_path:
        ref = L.read_pdf(Path(e.reference_local_path))
        r_lines = ref.sections.get(L.REFERENCE) or ref.sections.get(L.AUFGABE, [])
        r_width = ref.width
        raw_pages += ["=== Lösungs-PDF ==="] + ref.page_texts
        notes.append("Erwartungshorizont aus separatem PDF")
    if not pdf.section_found:
        notes.append("keine Abschnittsüberschriften – ganzes PDF als Aufgabe")

    # Material (Texte, Daten) steht bei Sprachen hinter der Aufgabenstellung, bei Naturwissenschaften
    # in einem eigenen Abschnitt. Es wird als Kontext an jede Teilaufgabe gehängt.
    a_lines, material_lines = _split_material(a_lines)
    material_lines += pdf.sections.get("material", [])
    material_text = _material_text(material_lines)

    aufgabe_text, reference_text = _text(a_lines), _text(r_lines)
    out_dir = interim / e.source / e.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{e.task_id}.raw.txt").write_text("\n\n=== Seite ===\n\n".join(raw_pages), encoding="utf-8")

    a_groups = L.parse_groups(a_lines, pdf.width)
    r_groups = L.parse_groups(r_lines, r_width)

    pages = {l.page for l in a_lines}
    n_images = sum(pdf.images_per_page[p] for p in pages)
    status = "no_sections" if not a_lines else ("no_reference" if not r_lines else "ok")

    task = Task(
        task_id=e.task_id, source=e.source, kind=e.kind, subject=e.subject, year=e.year,
        beispiel=e.beispiel, niveau=e.niveau, title=e.title, meta=e.meta, url=e.url,
        n_pages=pdf.n_pages, n_images=n_images, mentions_figure=L.mentions_figure(aufgabe_text),
        aufgabe_text=aufgabe_text, reference_text=reference_text,
        other_sections={**{k: _text(v) for k, v in pdf.sections.items() if k not in (L.AUFGABE, L.REFERENCE)},
                        "material": material_text},
        parse_status=status, parse_notes=notes,
        stats={"groups_aufgabe": len(a_groups), "groups_referenz": len(r_groups),
               "parts_aufgabe": sum(len(g.parts) for g in a_groups),
               "parts_referenz": sum(len(g.parts) for g in r_groups)},
    )

    subtasks = _pair(e, a_groups, r_groups, aufgabe_text, reference_text, material_text)
    _write_readable(out_dir / f"{e.task_id}.txt", subtasks)
    return task, subtasks


MATERIAL_RE = re.compile(r"^(Material(ien)?(\s+\d+(\.\d+)?)?|Textgrundlage|Text\s+\d)$")


def _split_material(lines: list[L.Line]) -> tuple[list[L.Line], list[L.Line]]:
    """Trennt „Material“ (und alles danach) von der Aufgabenstellung."""
    for i, l in enumerate(lines):
        if i > 0 and MATERIAL_RE.match(l.text.strip()):
            return lines[:i], lines[i:]
    return lines, []


def _material_text(lines: list[L.Line]) -> str:
    if not lines:
        return ""
    margin = min(l.x0 for l in lines)
    # Zeilennummern links neben dem Text („5“, „10“ …) entfernen
    kept = [l for l in lines if not (re.fullmatch(r"\d{1,3}", l.text) and l.x0 < margin + 2 and
                                     any(o.x0 > l.x0 + 15 for o in lines[:50]))]
    return _text(kept)


MATERIAL_REF_RE = re.compile(r"\(\s*M\s?\d|Material|\bM\s?\d\b|Abb", re.I)


def _pair(e: CatalogEntry, a_groups: list[L.Group], r_groups: list[L.Group],
          aufgabe_text: str, reference_text: str, material_text: str = "") -> list[Subtask]:
    material_has_figure = L.mentions_figure(material_text)
    # Urheberrechtlich geschützte Texte (v. a. Englisch) stehen nicht im PDF, nur Anfang … Ende der Ausschnitte
    material_missing = "Anfangs- und Endpunkte" in material_text
    def mk(aufgabe: str, teil: str, context: str, prompt: str, ref: str, be, conf: str, notes: list[str],
           weight=None):
        full_context = context
        if material_text:
            full_context = (context + "\n\n" if context else "") + "=== Material ===\n" + material_text
        # Abbildung zählt, wenn Aufgabentext/Einleitung darauf verweisen – oder wenn die Teilaufgabe auf
        # Material verweist und im Material Abbildungen vorkommen
        fig = L.mentions_figure(context + "\n" + prompt)
        if not fig and material_has_figure and MATERIAL_REF_RE.search(prompt):
            fig = True
            notes = notes + ["verweist auf Material mit Abbildungen"]
        frag = round(L.fragment_ratio(context + "\n" + prompt), 3)
        if material_missing:
            notes = notes + ["Materialtext fehlt im PDF (nur Anfang … Ende der Ausschnitte)"]
        return Subtask(
            subtask_id=f"{e.task_id}/{aufgabe}-{teil or 'gesamt'}", task_id=e.task_id, subject=e.subject,
            year=e.year, niveau=e.niveau, aufgabe=aufgabe, teil=teil, context_text=full_context,
            prompt_text=prompt, reference_text=ref, be=be, gewichtung=weight, mentions_figure=fig,
            fragment_ratio=frag,
            text_only_candidate=bool(prompt) and not fig and frag < FRAGMENT_LIMIT and not material_missing,
            split_confidence=conf, split_notes=notes,
        )

    if not a_groups:
        if not aufgabe_text:
            return []
        conf = "medium" if reference_text else "low"
        return [mk("1", "", "", aufgabe_text, reference_text, None, conf,
                   ["keine Teilaufgaben erkannt – ganze Aufgabe als Einheit"])]

    ref_index = {(g.key, p.letter): p for g in r_groups for p in g.parts}
    out: list[Subtask] = []
    running: dict[str, int] = {}   # BE-Summe pro Block seit der letzten angegebenen Gesamtsumme
    for g in a_groups:
        context = list(g.context)
        for p in g.parts:
            context += p.intro_before
            notes = list(p.notes)
            rp = ref_index.get((g.key, p.letter))
            be = p.be
            if be is None and rp is not None and rp.be is not None:
                be = rp.be
                notes.append("BE aus Erwartungshorizont")
            elif be is not None and rp is not None and rp.be is not None and rp.be != be:
                notes.append(f"BE weicht ab: Aufgabe {be}, EH {rp.be}")
            running[g.top] = running.get(g.top, 0) + (be or 0)

            if not reference_text:
                conf = "low"
                notes.append("kein Erwartungshorizont")
            elif rp is None or not rp.text:
                conf = "low"
                notes.append("keine passende Stelle im Erwartungshorizont")
            elif (be is None and p.weight is None) or any(n.startswith("BE weicht") for n in notes):
                conf = "medium"
            else:
                conf = "high"
            out.append(mk(g.key, p.letter, "\n".join(context).strip(), p.text,
                          rp.text if rp else "", be, conf, notes, weight=p.weight))
        if g.total_be is not None:
            # Gesamtsumme gilt für alle Aufgaben des Blocks seit der letzten Summe
            note = ("BE-Summe bestätigt" if running.get(g.top) == g.total_be
                    else f"BE-Summe {running.get(g.top)} ≠ angegebene Summe {g.total_be}")
            for st in out:
                if st.aufgabe.split("-")[0] == (g.top or st.aufgabe.split("-")[0]) and not any(
                        n.startswith("BE-Summe") for n in st.split_notes):
                    st.split_notes.append(note)
            running[g.top] = 0

    seen: Counter[str] = Counter()
    for st in out:
        seen[st.subtask_id] += 1
        if seen[st.subtask_id] > 1:
            st.subtask_id += f"_{seen[st.subtask_id]}"
    return out


def _write_readable(path: Path, subtasks: list[Subtask]) -> None:
    """Lesbare Fassung zum Durchsehen von Hand."""
    parts = []
    for s in subtasks:
        parts.append(
            f"################ {s.aufgabe}{s.teil}  | BE {s.be} | {s.split_confidence} | "
            f"Fragmente {s.fragment_ratio} | Abb. {s.mentions_figure}\n"
            f"{'; '.join(s.split_notes)}\n"
            f"--- Kontext\n{s.context_text}\n--- Aufgabe\n{s.prompt_text}\n--- Erwartungshorizont\n{s.reference_text}\n"
        )
    path.write_text("\n".join(parts), encoding="utf-8")
