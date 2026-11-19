"""Prompt-Vorlagen laden und befüllen, Lösungen anonymisieren, Bewertungen parsen."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

NOTENPUNKTE_MAX = 15   # Skala für Sprachaufgaben ohne BE

_ROMAN_BLOCK = re.compile(r"^(I|II|III|IV|V|VI)-")


def task_cluster(subtask_id: str) -> str:
    """Prüfungsaufgabe, zu der eine Teilaufgabe gehört (Cluster für Stichprobe und Bootstrap).

    IQB: ein PDF = eine Aufgabe → ``<pdf>``. ISB: ein PDF enthält die Aufgaben I–IV → ``<pdf>/<block>``.
    """
    pdf, _, rest = subtask_id.partition("/")
    m = _ROMAN_BLOCK.match(rest)
    return f"{pdf}/{m.group(1)}" if m else pdf

FACH_NAME = {
    "deutsch": "Deutsch", "englisch": "Englisch", "franzoesisch": "Französisch",
    "mathematik": "Mathematik", "physik": "Physik", "chemie": "Chemie",
    "biologie": "Biologie", "informatik": "Informatik",
}
SPRACHE_HINWEIS = {
    "englisch": "Schreibe deine Lösung auf Englisch, wie in der Abiturprüfung im Fach Englisch üblich.",
    "franzoesisch": "Schreibe deine Lösung auf Französisch, wie in der Abiturprüfung im Fach Französisch üblich.",
}
SPRACHE_STANDARD = "Schreibe deine Lösung auf Deutsch."

NIVEAU_NAME = {"grundlegend": "grundlegend", "erhoeht": "erhöht", None: "nicht angegeben"}


@dataclass
class Prompt:
    name: str
    system: str
    user: str

    def render(self, **values) -> list[dict]:
        return [
            {"role": "system", "content": self.system.format(**values).strip()},
            {"role": "user", "content": self.user.format(**values).strip()},
        ]


def load_prompt(path: Path) -> Prompt:
    text = path.read_text(encoding="utf-8")
    name = path.stem
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        meta = dict(re.findall(r"^(\w+):\s*(.+)$", m.group(1), re.M))
        name = meta.get("name", name)
        text = text[m.end():]
    parts = re.split(r"^## (system|user)\s*$", text, flags=re.M)
    sections = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"{path}: braucht Abschnitte '## system' und '## user'")
    return Prompt(name=name, system=sections["system"], user=sections["user"])


def max_points(task: dict) -> tuple[int, str]:
    """Erreichbare Punkte und Hinweis zur Skala."""
    if task.get("max_punkte"):
        return int(task["max_punkte"]), ""
    return NOTENPUNKTE_MAX, ("Die Aufgabe hat keine Bewertungseinheiten; bewerte auf der Notenpunkte-Skala "
                             "(15 = sehr gut, 5 = ausreichend, 0 = ungenügend).")


def task_values(task: dict) -> dict:
    kontext = (task.get("kontext") or "").strip()
    mp, skala = max_points(task)
    if task.get("max_punkte"):
        punkte_hinweis = f"(Für diese Teilaufgabe gibt es {task['max_punkte']} Bewertungseinheiten.)"
    elif task.get("gewichtung_prozent"):
        punkte_hinweis = f"(Gewichtung dieser Teilaufgabe: {task['gewichtung_prozent']} %.)"
    else:
        punkte_hinweis = ""
    return {
        "fach": FACH_NAME.get(task["fach"], task["fach"]),
        "niveau": NIVEAU_NAME.get(task.get("niveau"), task.get("niveau") or "nicht angegeben"),
        "kontext": kontext,
        "kontext_block": f"## Material und Einleitung\n{kontext}\n\n" if kontext else "",
        "aufgabe": task["aufgabe"].strip(),
        "erwartungshorizont": (task.get("erwartungshorizont") or "").strip(),
        "punkte_hinweis": punkte_hinweis,
        "sprache_hinweis": SPRACHE_HINWEIS.get(task["fach"], SPRACHE_STANDARD),
        "max_punkte": mp,
        "skala_hinweis": skala,
    }


# --------------------------------------------------------------------------- Anonymisierung

_IDENTITY_PATTERNS = [
    r"\b(Chat-?GPT|GPT-?\d[\w.\-]*|OpenAI|o\d-mini|Claude|Anthropic|Gemini[\w.\-]*|"
    r"DeepMind|Bard|Llama[\w.\-]*|Meta AI|Mistral[\w.\-]*|Qwen[\w.\-]*|DeepSeek|Copilot|Gemma[\w.\-]*)\b",
    r"\bals (?:eine? )?(?:KI|künstliche Intelligenz|Sprachmodell|KI-Assistent(?:in)?|AI)\b[^.]*\.",
    r"\bas an? (?:AI|language model|AI assistant)\b[^.]*\.",
    r"\bIch bin (?:eine? )?(?:KI|Sprachmodell|KI-Assistent(?:in)?)\b[^.]*\.",
]
_IDENTITY_RE = re.compile("|".join(f"(?:{p})" for p in _IDENTITY_PATTERNS), re.I)


def anonymize(text: str, task_text: str = "") -> tuple[str, list[str]]:
    """Entfernt Hinweise auf das erzeugende Modell. Gibt (Text, entfernte Stellen) zurück.

    Namen, die schon in der Aufgabe vorkommen (z. B. „Claude Monet“, der Wind „Mistral“), bleiben stehen.
    """
    removed: list[str] = []
    allowed = task_text.lower()

    def repl(m: re.Match) -> str:
        if m.group(0).lower() in allowed:
            return m.group(0)
        removed.append(m.group(0))
        return "[…]"

    out = _IDENTITY_RE.sub(repl, text)
    # typische Einleitungs-/Schlussfloskeln von Chat-Assistenten
    for pat in (r"^(Gerne|Natürlich|Sehr gerne|Klar)[!,.].*?\n", r"\n(Ich hoffe, das hilft|Lass mich wissen)[^\n]*$"):
        m = re.search(pat, out, re.I | re.M)
        if m:
            removed.append(m.group(0).strip())
            out = out[: m.start()] + ("\n" if m.group(0).startswith("\n") else "") + out[m.end():]
    return out.strip(), removed


# --------------------------------------------------------------------------- Bewertung parsen

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_grade(text: str, max_punkte: int) -> dict:
    """Liest {"punkte": .., "begruendung": ..} robust aus der Modellantwort.

    Ergebnis: punkte (int|None), begruendung, parse_ok, parse_note.
    Punkte außerhalb 0..max werden geklammert und vermerkt.
    """
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M)
    data = None
    m = _JSON_RE.search(raw)
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict) and "punkte" in data:
        try:
            p = round(float(str(data["punkte"]).replace(",", ".")))
        except ValueError:
            p = None
        note = ""
        if p is not None and not 0 <= p <= max_punkte:
            note = f"Punkte {p} außerhalb 0..{max_punkte} – geklammert"
            p = max(0, min(max_punkte, p))
        return {"punkte": p, "begruendung": str(data.get("begruendung", "")), "parse_ok": p is not None,
                "parse_note": note}
    # Rückfall: "punkte": 7 oder "7/10" im Text
    m = re.search(r'"?punkte"?\s*[:=]\s*(\d+)', raw, re.I) or re.search(rf"\b(\d+)\s*/\s*{max_punkte}\b", raw)
    if m:
        p = max(0, min(max_punkte, int(m.group(1))))
        return {"punkte": p, "begruendung": raw[:500], "parse_ok": True, "parse_note": "kein gültiges JSON – Zahl aus Text"}
    return {"punkte": None, "begruendung": raw[:500], "parse_ok": False, "parse_note": "keine Punktzahl gefunden"}
