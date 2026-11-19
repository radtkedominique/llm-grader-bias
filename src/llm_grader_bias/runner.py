"""Gemeinsame Bausteine für solve und grade: JSONL fortsetzen, parallel pro Modell, Kostenschätzung."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JsonlSink:
    """Thread-sicheres Anhängen an eine JSONL-Datei; kennt bereits erledigte IDs (Fortsetzen)."""

    def __init__(self, path: Path, id_field: str):
        self.path = path
        self.id_field = id_field
        self.done = {r[id_field] for r in read_jsonl(path)}
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.done.add(row[self.id_field])


def run_per_model(jobs_by_model: dict[str, list], worker: Callable, desc: str) -> tuple[int, int]:
    """Ein Thread pro Modell (jedes Modell hat seine eigene Drosselung). Gibt (ok, fehler) zurück."""
    total = sum(len(v) for v in jobs_by_model.values())
    ok = err = 0
    bar = tqdm(total=total, desc=desc, unit="call")
    lock = threading.Lock()

    def run_model(model_id: str, jobs: list):
        nonlocal ok, err
        for job in jobs:
            try:
                worker(job)
                with lock:
                    ok += 1
            except Exception as e:  # noqa: BLE001
                with lock:
                    err += 1
                log.error("[%s] Fehler: %s", model_id, e)
            bar.update(1)

    with ThreadPoolExecutor(max_workers=max(1, len(jobs_by_model))) as ex:
        futures = [ex.submit(run_model, m, j) for m, j in jobs_by_model.items() if j]
        for f in as_completed(futures):
            f.result()
    bar.close()
    return ok, err


def estimate(messages_iter: Iterable[list[dict]], out_tokens: int) -> dict:
    """Grobe Token-Schätzung (4 Zeichen ≈ 1 Token)."""
    n = chars = 0
    for msgs in messages_iter:
        n += 1
        chars += sum(len(m["content"]) for m in msgs)
    return {"calls": n, "input_tokens": chars // 4, "output_tokens": n * out_tokens}
