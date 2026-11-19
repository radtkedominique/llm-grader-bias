"""Tests für Pilot-Auswahl, Prompts und den Lauf Lösen → Korrigieren mit Mock-Modellen."""

import json
import shutil
from pathlib import Path

import yaml

from llm_grader_bias import grade, pilot, solve
from llm_grader_bias.prompting import anonymize, load_prompt, parse_grade, task_values

ROOT = Path(__file__).resolve().parents[1]


def make_rows():
    rows = []
    spec = [("deutsch", "sprache", 2024, None, 60), ("englisch", "sprache", 2025, None, 30),
            ("mathematik", "mint", 2025, 5, None), ("physik", "mint", 2026, 4, None),
            ("biologie", "mint", 2018, 6, None), ("informatik", "informatik", None, 4, None)]
    for fach, typ, jahr, be, gew in spec:
        for t in range(4):
            for teil in "abc":
                rows.append({"id": f"iqb_{fach}_{t}/1-{teil}", "fach": fach, "aufgabentyp": typ, "jahr": jahr,
                             "niveau": "erhoeht", "quelle": "iqb", "kontext": "Ein Gedicht über Claude Monet.",
                             "aufgabe": f"Interpretieren Sie {t}{teil}.", "erwartungshorizont": "Musterlösung",
                             "max_punkte": be, "gewichtung_prozent": gew, "quelle_url": "u"})
    return rows


def test_pilot_select_is_stratified_and_reproducible():
    cfg = {"seed": 1, "quotas": {"sprache": 4, "mint": 4, "informatik": 4}, "balance_subjects": True,
           "min_year": 2024, "allow_without_year": ["informatik"], "max_per_task": 2, "max_context_chars": 0}
    a, warn = pilot.select(make_rows(), cfg)
    b, _ = pilot.select(make_rows(), cfg)
    assert [r["id"] for r in a] == [r["id"] for r in b]                 # reproduzierbar
    assert not warn and len(a) == 12
    assert all(r["fach"] != "biologie" for r in a)                       # 2018 < min_year
    per_task = {}
    for r in a:
        per_task[r["id"].split("/")[0]] = per_task.get(r["id"].split("/")[0], 0) + 1
    assert max(per_task.values()) <= 2
    assert {r["fach"] for r in a if r["aufgabentyp"] == "mint"} == {"mathematik", "physik"}


def test_prompts_render_and_scale():
    p = load_prompt(ROOT / "prompts/grade_with_reference_v1.md")
    lang = make_rows()[0]                                                 # Deutsch: nur Gewichtung
    values = task_values(lang)
    values["loesung"] = "Meine Lösung"
    msgs = p.render(**values)
    assert '"max_punkte": 15' in msgs[0]["content"]                       # Notenpunkte-Skala
    assert "Musterlösung" in msgs[1]["content"] and "Meine Lösung" in msgs[1]["content"]
    s = load_prompt(ROOT / "prompts/solve_v1.md").render(**task_values(make_rows()[26]))
    assert "5 Bewertungseinheiten" in s[1]["content"]


def test_anonymize_and_parse():
    text, removed = anonymize("Gerne!\nAls KI kann ich das. Claude Monet malte Seerosen. ChatGPT ist toll.",
                              "Gedicht über Claude Monet")
    assert "ChatGPT" not in text and "Claude Monet" in text and "Als KI" not in text
    assert parse_grade('```json\n{"punkte": 4, "begruendung": "ok"}\n```', 5)["punkte"] == 4
    assert parse_grade('{"punkte": 9}', 5)["punkte"] == 5                 # geklammert
    assert parse_grade("keine Ahnung", 5)["parse_ok"] is False


def test_end_to_end_mock(tmp_path: Path, monkeypatch):
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    (tmp_path / "configs").mkdir()
    cfg = {"defaults": {"rpm": 100000}, "models": [
        {"id": f"mock-{f}-{s}", "family": f"mock-{f}", "size": s, "provider": "mock", "model": "mock",
         **({"extra": {"marker": "MARK", "bias": 10}} if (f, s) == ("a", "large") else {})}
        for f in "abc" for s in ("small", "large")]}
    (tmp_path / "configs/models.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    tasks = make_rows()[:5] + make_rows()[24:27]
    (tmp_path / "data/pilot").mkdir(parents=True)
    (tmp_path / "data/pilot/pilot.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    monkeypatch.chdir(tmp_path)

    assert solve.main([]) == 0
    sols = [json.loads(l) for l in open("data/solutions/solutions.jsonl")]
    assert len(sols) == 8 * 6
    assert grade.main([]) == 0
    grs = [json.loads(l) for l in open("data/gradings/gradings.jsonl")]
    assert len(grs) == 48 * 6 * 2
    assert {g["relation"] for g in grs} == {"self", "family", "foreign"}
    assert sum(g["relation"] == "self" for g in grs) == 48 * 2
    # zweiter Lauf: alles erledigt, nichts Neues
    assert grade.main([]) == 0
    assert sum(1 for _ in open("data/gradings/gradings.jsonl")) == len(grs)


def test_analysis_recovers_injected_self_bias(tmp_path: Path, monkeypatch):
    from llm_grader_bias import analysis
    test_end_to_end_mock(tmp_path, monkeypatch)          # erzeugt Lösungen + Bewertungen mit Bias bei mock-a-large
    assert analysis.main(["--bootstrap", "100", "--no-mixed-model"]) == 0
    import pandas as pd
    by = pd.read_csv("results/bias_by_grader.csv").set_index("grader_model")
    assert by.loc["mock-a-large", "self_bias"] > 0.2
    assert by.loc["mock-a-large", "self_ci_low"] > 0
    others = by.drop(index="mock-a-large")["self_bias"].abs()
    assert (others < 0.2).all()
    df = pd.read_csv("results/gradings_with_neutral.csv")
    # neutrale Note nie aus der Familie von Korrektor oder Autor
    assert df["neutral"].notna().all() and len(df) > 0
    assert "Self-Bias" in Path("results/report.md").read_text()


def test_neutral_excludes_own_families():
    import pandas as pd
    from llm_grader_bias.analysis import add_neutral
    rows = []
    fams = {"a1": "A", "b1": "B", "c1": "C"}
    scores = {"a1": 1.0, "b1": 0.5, "c1": 0.2}
    for g, fam in fams.items():
        rows.append({"grading_id": g, "solution_id": "s", "condition": "x", "grader_model": g,
                     "grader_family": fam, "author_family": "A", "score": scores[g]})
    df = add_neutral(pd.DataFrame(rows)).set_index("grader_model")
    assert df.loc["a1", "neutral"] == (0.5 + 0.2) / 2      # Korrektor A, Autor A → B und C
    assert df.loc["b1", "neutral"] == 0.2                   # Korrektor B, Autor A → nur C
    assert df.loc["c1", "neutral"] == 0.5


def test_controls_and_descriptive(tmp_path: Path, monkeypatch):
    from llm_grader_bias.analysis import controls, descriptive
    test_end_to_end_mock(tmp_path, monkeypatch)
    assert grade.main(["--repeats", "2"]) == 0                       # zweite Bewertungsrunde
    assert controls.main(["--bootstrap", "50"]) == 0
    out = Path("results/kontrollen")
    assert (out / "01_laenge_score.png").exists() and (out / "04_wiederholbarkeit.png").exists()
    import pandas as pd
    rep = pd.read_csv(out / "wiederholbarkeit.csv")
    assert rep.loc[rep["grader_model"] == "alle", "identisch"].iloc[0] == 1.0   # Mock ohne Rauschen: identisch
    blc = pd.read_csv(out / "bias_laengenkontrolle.csv")
    assert set(blc["variante"]) == {"ohne Längen-Kontrolle", "mit Längen-Kontrolle"}

    # Block 1 mit Mini-Datensatz
    proc = Path("data/processed")
    proc.mkdir(parents=True)
    rows = make_rows()
    subs = [{"subtask_id": r["id"], "subject": r["fach"], "split_confidence": "high", "split_notes": [],
             "fragment_ratio": 0.0, "mentions_figure": False, "text_only_candidate": True} for r in rows]
    for name, data in [("catalog", [{"task_id": "x"}] * 5), ("tasks", [{"task_id": "x"}] * 5),
                       ("subtasks", subs), ("dataset", rows)]:
        (proc / f"{name}.jsonl").write_text("\n".join(json.dumps(d) for d in data) + "\n")
    assert descriptive.main([]) == 0
    assert (Path("results/datensatz") / "01_trichter.png").exists()
    assert "Kennzahlen" in Path("results/datensatz/datensatz.md").read_text()


def test_task_cluster():
    from llm_grader_bias.prompting import task_cluster
    assert task_cluster("iqb_mathematik_2025x/2-b") == "iqb_mathematik_2025x"
    assert task_cluster("isb_informatik_illupa_inf_gA/III-1-c") == "isb_informatik_illupa_inf_gA/III"
