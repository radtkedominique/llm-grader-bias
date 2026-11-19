# Studiendesign (Kurzfassung)

Ausführliche Version: Notion, Kurs *Vertiefung KI* → „Themenidee: Self- und Family-Bias bei LLM-Korrektur von Hausaufgaben“.

## Aufbau

- **Modelle:** 6 Modelle = 3 Familien × 2 Größen (4 Familien wären stabiler, siehe Risiken).
- **Aufgaben:** Abituraufgaben mit offiziellem Erwartungshorizont. Nur Teilaufgaben, die ohne Abbildung lösbar sind. Fächer: Text (Deutsch/Englisch), Mathe, Naturwissenschaften, Informatik/Coding.
- **Lösen:** Jedes Modell löst jede Teilaufgabe.
- **Korrigieren:** Jedes Modell korrigiert jede Lösung anonym nach BE, in zwei Bedingungen:
  1. ohne Erwartungshorizont
  2. mit Erwartungshorizont
- **Pilot:** 30 Teilaufgaben × 6 Lösende × 6 Korrigierende × 2 Bedingungen = 2.160 Bewertungen.

## Bias-Maß

```
neutrale Note(Lösung, Korrektor) = Mittel der Noten aller Korrektoren,
                                    deren Familie ≠ Familie(Korrektor) und ≠ Familie(Autor)
Abweichung                       = Note(Korrektor) − neutrale Note
Self-Bias                        = Ø Abweichung(eigene Lösungen)       − Ø Abweichung(fremde Lösungen)
Family-Bias                      = Ø Abweichung(gleiche Familie, nicht eigene) − Ø Abweichung(fremde Lösungen)
```

Alle Noten werden als Anteil der erreichbaren BE normiert (0–1).

## Kontrollen

- Länge der Lösung (Tokens)
- Wiederholungsläufe (Temperatur 0 bzw. mehrere Seeds)
- optional: Herkunft verraten, stilneutrale Paraphrase, Handkorrektur von ca. 30 Lösungen

## Auswertung

```
abweichung ~ beziehung * erwartungshorizont * aufgabentyp + länge + (1|aufgabe) + (1|korrektor)
```

`beziehung` ∈ {selbst, gleiche Familie, fremd}

## Risiken

- Mit 3 Familien stützt sich die neutrale Note bei fremden Lösungen auf nur eine Familie.
- Kontamination älterer Jahrgänge → neueste Jahrgänge bevorzugen.
- Aufgabenschwierigkeit kann Bias überdecken (vgl. JudgeBench) → Aufgabe als Random Effect.
