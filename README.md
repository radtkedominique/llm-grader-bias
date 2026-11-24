# llm-grader-bias

**Bewerten LLMs ihre eigenen Lösungen besser als fremde?**
Forschungsprojekt im DHBW-Modul *Vertiefung KI* zu **Self-Bias** und **Family-Bias** bei der Korrektur von Hausaufgaben durch Large Language Models.

## Idee in einem Satz

Mehrere LLMs aus verschiedenen Modellfamilien lösen dieselben Abituraufgaben. Danach korrigiert jedes Modell alle Lösungen anonym nach Bewertungseinheiten (BE). Wir messen, ob ein Modell seine eigenen Lösungen oder die Lösungen aus der eigenen Familie systematisch besser bewertet als unbeteiligte, „neutrale“ Korrektoren.

## Forschungsfragen

| | Frage |
|---|---|
| **RQ1** | Bewertet ein Modell die eigenen Lösungen besser als neutrale Korrektoren (Self-Bias)? |
| **RQ2** | Bewertet ein Modell Lösungen aus der eigenen Familie besser (Family-Bias)? |
| **RQ3** | Verringert ein Erwartungshorizont (Musterlösung) den Bias? |
| **RQ4** | Unterscheidet sich der Bias zwischen Text-, Mathe- und Coding-Aufgaben? |

**Bias-Maß:** `Abweichung = Note des Korrektors − neutrale Note`. Die neutrale Note ist der Mittelwert der Korrektoren aus Familien, die weder mit dem Korrektor noch mit dem Autor der Lösung verwandt sind. Self-Bias ist die mittlere Abweichung bei eigenen Lösungen minus die mittlere Abweichung bei fremden Lösungen.

Das vollständige Studiendesign steht auf der Notion-Seite *„Themenidee: Self- und Family-Bias bei LLM-Korrektur von Hausaufgaben“* im Kurs Vertiefung KI. Eine Kurzfassung liegt in [`docs/studiendesign.md`](docs/studiendesign.md).

## Pipeline

```
1. dataset   Abituraufgaben + Erwartungshorizonte laden → data/processed/dataset.jsonl
   pilot     geschichtete Auswahl (30 Teilaufgaben)       → data/pilot/pilot.jsonl
2. solve     jedes Modell löst jede Teilaufgabe           → data/solutions/
3. grade     jedes Modell korrigiert jede Lösung anonym,
             mit und ohne Erwartungshorizont               → data/gradings/
4. analysis  neutrale Note, Self-/Family-Bias, Mixed Model → results/
```

## Ordnerstruktur

```
llm-grader-bias/
├── configs/
│   ├── dataset.yaml          # Quellen, Fächer, Jahrgänge, Filter
│   ├── pilot.yaml            # Quoten und Filter der Pilot-Auswahl
│   ├── models.bedrock.yaml   # Studie: Claude 5.5 + DeepSeek über AWS Bedrock
│   ├── models.litellm.yaml   # Studie: GPT-6 über den Haufe LiteLLM-Proxy
│   ├── models.ollama.yaml    # lokal testen (kostenlos)
│   └── models.mock.yaml      # Fake-Modelle für Pipeline-Tests
├── data/                     # NICHT im Git (Urheberrecht, siehe unten)
│   ├── raw/                  # Original-PDFs, nach Quelle/Fach
│   ├── interim/              # extrahierter Text pro PDF
│   ├── processed/            # tasks.jsonl, subtasks.jsonl, report.md
│   ├── solutions/            # Lösungen der Modelle
│   └── gradings/             # Bewertungen der Modelle
├── docs/
│   └── studiendesign.md
├── notebooks/                # explorative Analysen
├── prompts/                  # Prompt-Vorlagen (lösen, korrigieren)
├── results/                  # Auswertungen, Plots (nicht im Git)
├── src/llm_grader_bias/
│   ├── dataset/              # Phase 1: Download, Extraktion, Aufteilung
│   │   ├── sources/          #   iqb.py (Abituraufgabenpool), isb.py (Bayern Informatik)
│   │   ├── http.py           #   höflicher Downloader mit Cache
│   │   ├── models.py         #   Datenklassen (Katalog, Aufgabe, Teilaufgabe)
│   │   ├── layout.py         #   PDF-Layout → Abschnitte, Aufgaben, Teilaufgaben, BE
│   │   ├── assemble.py       #   Aufgabe ↔ Erwartungshorizont zuordnen
│   │   └── cli.py            #   Kommandozeile
│   ├── pilot.py              # geschichtete Pilot-Auswahl
│   ├── prompting.py          # Prompts befüllen, anonymisieren, Bewertung parsen
│   ├── llm.py                # ein Client für alle Zugänge, Cache, Retries
│   ├── runner.py             # parallel pro Modell, fortsetzbar
│   ├── solve/                # Phase 2: Lösen
│   ├── grade/                # Phase 3: Korrigieren
│   └── analysis/             # Phase 4: neutrale Note, Bias, Bootstrap, Mixed Model;
│                             #   descriptive.py (Datensatz), controls.py (Kontrollen), style.py
└── tests/                    # pytest mit synthetischen Seiten/PDFs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Benötigt Python ≥ 3.10.

## Datensatz bauen

```bash
# alles in einem Schritt: Katalog → Download → Extraktion → Teilaufgaben → Report
python -m llm_grader_bias.dataset all

# oder einzeln
python -m llm_grader_bias.dataset catalog    # Aufgabenlisten crawlen  → data/processed/catalog.jsonl
python -m llm_grader_bias.dataset download   # PDFs laden (mit Cache)  → data/raw/
python -m llm_grader_bias.dataset extract    # Text + Teilaufgaben     → data/processed/*.jsonl
python -m llm_grader_bias.dataset report     # Übersicht               → data/processed/report.md
python -m llm_grader_bias.dataset export     # fertiger Datensatz      → data/processed/dataset.jsonl

# schneller Probelauf: nur Mathe + Informatik, 2 Listenseiten pro Fach
python -m llm_grader_bias.dataset all --subjects mathematik,informatik --max-pages 2
```

Was der Code macht:

1. **Katalog:** Er crawlt die Aufgabenseiten des IQB-Abituraufgabenpools (Deutsch, Mathematik, Englisch, Französisch, Biologie, Chemie, Physik, 2017 bis heute) und nimmt die bayerischen Informatik-Materialien des ISB dazu. Pro Aufgabe speichert er die Metadaten (Fach, Jahr, Niveau, Sachgebiet …).
2. **Download:** Er lädt die PDFs höflich (1 Request/s, eigener User-Agent) und speichert sie im Cache. Ein zweiter Lauf lädt nichts neu.
3. **Extraktion:** Er trennt jedes PDF in *Aufgabe* und *Erwartungshorizont*. Dafür wertet er die Position jeder Zeile aus, nicht nur den Text: Marker für Teilaufgaben stehen in einer festen linken Spalte, die BE in einer eigenen Spalte, und Kopfzeilen werden verworfen.
4. **Teilaufgaben:** Er zerlegt die Aufgaben in Teilaufgaben (a, b, c …) mit BE und ordnet sie über (Block, Aufgabe, Buchstabe) dem Erwartungshorizont zu. Jede Zeile bekommt eine `split_confidence`. Teilaufgaben mit `low` vor der Nutzung von Hand prüfen.
5. **Textqualität:** Er markiert Verweise auf Abbildungen und Formeln, die beim Extrahieren zerfallen (`fragment_ratio`). Kandidaten für die Studie sind Teilaufgaben mit `text_only_candidate = true`.

Zum Durchsehen von Hand gibt es pro Aufgabe eine lesbare Fassung unter `data/interim/<quelle>/<fach>/<task_id>.txt`.

> **Bekannte Grenze:** Formeln aus den Mathe-PDFs (Formeleditor) zerfallen beim Extrahieren in einzelne Zeichen. Etwa die Hälfte der Mathe-Teilaufgaben ist deshalb nicht als reiner Text nutzbar.

### Ausgabe

| Datei | Inhalt |
|---|---|
| `data/processed/catalog.jsonl` | eine Zeile pro Aufgaben-PDF mit Metadaten und URL |
| `data/processed/tasks.jsonl` | eine Zeile pro Aufgabe: Aufgabentext, Erwartungshorizont, Abbildungs-Flags |
| `data/processed/subtasks.jsonl` | eine Zeile pro Teilaufgabe: Kontext, Aufgabentext, Referenzlösung, BE, Konfidenz |
| `data/processed/report.md` | Zählungen nach Fach, Jahr, Status |
| **`data/processed/dataset.jsonl`** | **fertiger Datensatz**: nur sicher zugeordnete, reine Textaufgaben, ohne Dubletten. Felder: `id, fach, aufgabentyp, jahr, niveau, quelle, kontext, aufgabe, erwartungshorizont, max_punkte, gewichtung_prozent, quelle_url` |
| `data/processed/dataset.md` | Datenblatt: Umfang nach Fach und Jahr, Felder |

## Pilot, Lösen, Korrigieren

```bash
python -m llm_grader_bias.pilot                   # 30 Teilaufgaben, 10 je Aufgabentyp, ab 2024 → data/pilot/
python -m llm_grader_bias.solve --models-config configs/models.bedrock.yaml --dry-run   # Kosten schätzen

# erst mit allen Configs lösen, dann mit allen korrigieren
python -m llm_grader_bias.solve --models-config configs/models.bedrock.yaml
python -m llm_grader_bias.solve --models-config configs/models.litellm.yaml
python -m llm_grader_bias.grade --models-config configs/models.bedrock.yaml
python -m llm_grader_bias.grade --models-config configs/models.litellm.yaml
```

- **Modelle:** eine Config pro Zugang, `--models-config` ist Pflicht. Studie: `models.bedrock.yaml` (Claude Haiku/Sonnet/Opus 5.5, DeepSeek V3.1/V3.2) und `models.litellm.yaml` (GPT-6 Luna/Sol/Astra). Schlüssel in `.env` (Vorlage: `.env.example`).
- **Korrigieren über Configs hinweg:** `grade` nimmt die Korrektoren aus der Config, korrigiert aber alle Lösungen in `solutions.jsonl` und überspringt fertige Bewertungen.
- **Zugänge (`provider`):** `bedrock` (Converse-API), `openai_compatible` (LiteLLM, Ollama, LM Studio, OpenRouter), `azure`, `anthropic` und `mock`. Kostenlos testen: `models.mock.yaml` (Fake-Antworten) oder `models.ollama.yaml` (lokal).
- **Cache** unter `data/cache/`: Ein abgebrochener Lauf setzt fort, ohne bezahlte Aufrufe zu wiederholen.
- **Prompts** in `prompts/` (versioniert, siehe `prompts/README.md`). Lösungen werden vor dem Korrigieren anonymisiert.
- **Ausgabe:** `solutions.jsonl` (eine Lösung pro Zeile), `gradings.jsonl` (eine Bewertung pro Zeile mit `relation` = self/family/foreign und `score` = Punkte/Maximum).

**Lokal mit Ollama testen** (kostenlos, echte Modelle): `configs/models.ollama.yaml` enthält 3 Familien × 2 Größen (Llama, Gemma, Qwen). Die nötigen `ollama pull`-Befehle stehen oben in der Datei.

```bash
python -m llm_grader_bias.solve --models-config configs/models.ollama.yaml --models llama-small   # ein Modell
python -m llm_grader_bias.solve --models-config configs/models.ollama.yaml                        # alle sechs
python -m llm_grader_bias.grade --models-config configs/models.ollama.yaml
```

Nach Test-Läufen (Mock, Ollama) vor dem echten Lauf `data/solutions/` und `data/gradings/` leeren.

## Auswertung

```bash
python -m llm_grader_bias.analysis                  # data/gradings/gradings.jsonl → results/
```

- **Neutrale Note** je Bewertung = Mittel der Korrektoren aus Familien, die weder mit Korrektor noch Autor verwandt sind. **Abweichung** = Score − neutrale Note.
- **Self-Bias** = Ø Abweichung bei eigenen Lösungen − Ø Abweichung bei fremden; **Family-Bias** analog für die eigene Familie. Die Differenz zu „fremd“ nimmt allgemeine Strenge heraus.
- **95-%-Intervalle** per Cluster-Bootstrap über Aufgaben, gesamt sowie nach Bedingung (RQ3), Aufgabentyp (RQ4) und Korrektor.
- **Mixed Model:** `abweichung ~ self + family + Interaktionen mit Bedingung/Aufgabentyp + Länge + Korrektor (fest) + (1|Aufgabe)` (braucht `statsmodels`: `pip install -e ".[analysis]"`).
- **Ausgabe:** `results/report.md` plus CSVs für Tabellen und Grafiken.

Mit den Mock-Modellen geprüft: Einem Mock-Korrektor ist ein Bias für eigene Lösungen eingebaut; die Auswertung findet ihn (Self-Bias ≈ +0,40, KI ohne 0), bei allen anderen liegt er um 0.

## Kennzahlen und Grafiken für die Präsentation

```bash
python -m llm_grader_bias.analysis.descriptive    # Block 1: Datensatz        → results/datensatz/
python -m llm_grader_bias.analysis.controls       # Block 5: Kontrollen       → results/kontrollen/
python -m llm_grader_bias.grade --models-config configs/models.bedrock.yaml --repeats 2   # vorher, für die Wiederholbarkeit (je Config)
```

| Block | Grafiken (PNG, 200 dpi) | Tabellen |
|---|---|---|
| **Datensatz** | Trichter, Teilaufgaben nach Fach, Heatmap Fach × Jahr, Ausschlussgründe, Textlänge je Aufgabentyp, BE-Verteilung, Pilot-Zusammensetzung | `datensatz.md` + CSVs |
| **Kontrollen** | Lösungslänge ↔ Score, Lösungslänge je Modell, Bias nach Prüfungsjahr (Kontamination), Wiederholbarkeit | `kontrollen.md` + CSVs, u. a. Bias mit/ohne Längen-Kontrolle je Korrektor, leere/abgeschnittene Lösungen, Anonymisierungs-Eingriffe |

Einheitlicher Stil in `analysis/style.py`: farbenblind-geprüfte Palette, Farbe folgt der Entität (Familie, Aufgabentyp) über alle Grafiken, feine Gitterlinien, Titel + Untertitel + Quelle.

## Daten und Urheberrecht

Die Aufgaben stammen aus dem **gemeinsamen Abituraufgabenpool der Länder (IQB)** und vom **ISB Bayern**. Sie sind für Unterricht und Prüfungsvorbereitung veröffentlicht. Deshalb gilt:

- `data/` und `results/` sind in `.gitignore`. **Keine Aufgabentexte oder PDFs ins Repo committen** und nicht weiterveröffentlichen.
- In Präsentation und Bericht nur kurze Ausschnitte mit Quellenangabe zeigen.
- Den Datensatz kann jede Person mit dem Skript selbst neu erzeugen. Damit ist er reproduzierbar, ohne ihn zu verbreiten.

## Kontamination

Ältere Abituraufgaben stehen wahrscheinlich in den Trainingsdaten der Modelle. Für die Hauptauswertung deshalb **die neuesten Jahrgänge** bevorzugen (`min_year` in `configs/dataset.yaml`). Das Jahr als Kontrollvariable mitführen.
