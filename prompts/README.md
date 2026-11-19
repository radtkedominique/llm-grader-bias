# Prompts

| Datei | Zweck |
|---|---|
| `solve_v1.md` | Teilaufgabe lösen wie in der Abiturprüfung |
| `grade_with_reference_v1.md` | Lösung anonym bewerten, **mit** Erwartungshorizont |
| `grade_without_reference_v1.md` | Lösung anonym bewerten, **ohne** Erwartungshorizont |

## Aufbau einer Datei

```
---            ← Metadaten (YAML), werden nicht gesendet
name: ...
---
## system      ← Systemnachricht
...
## user        ← Nutzernachricht
...
```

Platzhalter stehen in `{geschweiften Klammern}`. Doppelte Klammern `{{ }}` sind wörtliche Klammern (für das JSON-Beispiel).
Befüllt werden sie in `src/llm_grader_bias/prompting.py`.

## Entscheidungen

- **Punkteskala:** Hat die Teilaufgabe BE, wird von 0 bis BE bewertet. Sprachaufgaben haben nur eine Prozentgewichtung – dort wird auf der Notenpunkte-Skala 0–15 bewertet. Für die Auswertung wird alles auf 0–1 normiert (`punkte / max_punkte`).
- **Eine Lösung pro Bewertungsaufruf:** keine Paarvergleiche → kein Positions-Bias.
- **Anonymisierung:** Der Lös-Prompt verbietet Selbstbezüge. Zusätzlich entfernt `anonymize()` vor dem Korrigieren typische Modell- und Firmennamen und KI-Floskeln. Jede Ersetzung wird protokolliert.
- **Gleiche Rolle für alle:** Jedes Modell bekommt exakt denselben Prompt. Nur Modell und Lösung ändern sich.
- **Versionierung:** Wer einen Prompt ändert, legt eine neue Datei an (`_v2`) und lässt `_v1` unverändert. Die Prompt-Version wird mit jeder Lösung und Bewertung gespeichert.
