"""Ein Client für alle Modellzugänge: Azure OpenAI, OpenAI-kompatibel (OpenRouter, Ollama, LM Studio,
Gemini-OpenAI-Endpoint), Anthropic und ein Mock für Tests.

Jedes Modell ist ein Eintrag in ``configs/models.yaml``. Schlüssel kommen aus Umgebungsvariablen
bzw. ``.env`` – nie aus der Config.

Funktionen:
- Antwort-Cache auf Platte (``data/cache/``): Gleicher Aufruf → gleiche Antwort, Läufe lassen sich
  abbrechen und fortsetzen, ohne doppelt zu zahlen.
- Retries mit Backoff bei 429/5xx, Drosselung über ``rpm`` (Anfragen pro Minute).
- Parameter, die ein Modell nicht kennt (z. B. ``temperature`` bei Reasoning-Modellen), lassen sich
  pro Modell abschalten.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- Konfiguration

@dataclass
class ModelSpec:
    id: str                      # kurze, eindeutige ID in der Studie, z. B. "gpt-large"
    family: str                  # "openai" | "anthropic" | "google" | ...
    size: str                    # "small" | "large"
    provider: str                # "azure" | "openai_compatible" | "anthropic" | "mock"
    model: str                   # Modellname bzw. Azure-Deployment-Name
    base_url: str | None = None  # für openai_compatible; bei azure der Endpoint
    base_url_env: str | None = None
    api_key_env: str | None = None
    api_version: str = "2024-10-21"          # nur azure
    temperature: float | None = 0.0          # None = Parameter nicht senden
    max_tokens: int = 2000
    token_param: str = "max_tokens"          # Reasoning-Modelle bei OpenAI/Azure: "max_completion_tokens"
    seed: int | None = 1234                  # None = nicht senden
    rpm: int = 30
    timeout: int = 180
    extra: dict = field(default_factory=dict)  # weitere Body-Parameter, z. B. {"reasoning_effort": "low"}
    json_mode: bool = True                     # beim Korrigieren response_format=json_object senden (falls unterstützt)
    model_version: str | None = None           # zur Dokumentation (z. B. Datum des Modell-Snapshots)

    def resolved_base_url(self) -> str | None:
        if self.base_url_env:
            return os.environ.get(self.base_url_env) or self.base_url
        return self.base_url

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


def load_env(path: Path = Path(".env")) -> None:
    """Minimaler .env-Leser (KEY=VALUE), überschreibt keine gesetzten Variablen."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_models(path: Path, only: set[str] | None = None) -> list[ModelSpec]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    specs = []
    for m in cfg["models"]:
        if m.get("enabled", True) is False:
            continue
        data = {**defaults, **{k: v for k, v in m.items() if k != "enabled"}}
        spec = ModelSpec(**data)
        if only and spec.id not in only:
            continue
        specs.append(spec)
    ids = [s.id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Doppelte Modell-IDs in {path}: {ids}")
    return specs


# --------------------------------------------------------------------------- Ergebnis

@dataclass
class Completion:
    text: str
    model_reported: str | None
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    latency_s: float
    cached: bool = False
    raw: dict | None = None


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- Client

class LLMClient:
    def __init__(self, spec: ModelSpec, cache_dir: Path | None = Path("data/cache"), retries: int = 5):
        self.spec = spec
        self.cache_dir = cache_dir / spec.id if cache_dir else None
        self.retries = retries
        self._last = 0.0
        self.session = requests.Session()

    # ---- Cache
    def _key(self, messages: list[dict], salt: str = "") -> str:
        s = self.spec
        payload = json.dumps({"provider": s.provider, "model": s.model, "messages": messages, "salt": salt,
                              "temperature": s.temperature, "max_tokens": s.max_tokens, "seed": s.seed,
                              "extra": s.extra}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> Completion | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            d.pop("raw", None)
            return Completion(**{**d, "cached": True})
        return None

    def _cache_put(self, key: str, c: Completion) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        d = c.__dict__.copy()
        d["cached"] = False
        (self.cache_dir / f"{key}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    # ---- Drosselung
    def _throttle(self) -> None:
        gap = 60.0 / max(1, self.spec.rpm)
        wait = self._last + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    # ---- Aufruf
    def complete(self, messages: list[dict], use_cache: bool = True, salt: str = "",
                 json_mode: bool = False) -> Completion:
        """salt trennt bewusste Wiederholungen im Cache (z. B. "rep2"); json_mode erzwingt JSON-Antworten."""
        json_mode = json_mode and self.spec.json_mode
        key = self._key(messages, salt + ("|json" if json_mode else ""))
        self._json = json_mode
        if use_cache and (hit := self._cache_get(key)):
            return hit
        self._salt = salt
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._throttle()
            t0 = time.monotonic()
            try:
                c = self._call(messages)
                c.latency_s = round(time.monotonic() - t0, 2)
                self._cache_put(key, c)
                return c
            except _Retryable as e:
                last_err = e
                delay = min(60, 2 ** attempt) + random.random()
                log.warning("[%s] %s – neuer Versuch %d/%d in %.0fs", self.spec.id, e, attempt, self.retries, delay)
                time.sleep(delay)
        raise LLMError(f"[{self.spec.id}] aufgegeben nach {self.retries} Versuchen: {last_err}")

    def _call(self, messages: list[dict]) -> Completion:
        p = self.spec.provider
        if p == "mock":
            return _mock(self.spec, messages, self._salt)
        if p == "anthropic":
            return self._anthropic(messages)
        if p in ("azure", "openai_compatible"):
            return self._openai_style(messages)
        raise LLMError(f"Unbekannter Provider: {p}")

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        try:
            r = self.session.post(url, headers=headers, json=body, timeout=self.spec.timeout)
        except requests.RequestException as e:
            raise _Retryable(f"Netzwerkfehler: {e}") from e
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            raise _Retryable(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise LLMError(f"[{self.spec.id}] HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _openai_style(self, messages: list[dict]) -> Completion:
        s = self.spec
        base = (s.resolved_base_url() or "").rstrip("/")
        key = s.api_key()
        if not base:
            raise LLMError(f"[{s.id}] keine base_url (Config oder Umgebungsvariable {s.base_url_env})")
        body: dict = {"messages": messages, s.token_param: s.max_tokens, **s.extra}
        if s.temperature is not None:
            body["temperature"] = s.temperature
        if s.seed is not None:
            body["seed"] = s.seed
        if getattr(self, "_json", False):
            body["response_format"] = {"type": "json_object"}
        if s.provider == "azure":
            url = f"{base}/openai/deployments/{s.model}/chat/completions?api-version={s.api_version}"
            headers = {"api-key": key or ""}
        else:
            url = f"{base}/chat/completions"
            body["model"] = s.model
            headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = self._post(url, headers, body)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        # Denkende Modelle (z. B. Qwen3 über Ollama) liefern ihren Denkprozess teils im Text mit
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        usage = data.get("usage") or {}
        return Completion(text=text, model_reported=data.get("model"),
                          input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                          finish_reason=choice.get("finish_reason"), latency_s=0.0)

    def _anthropic(self, messages: list[dict]) -> Completion:
        s = self.spec
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        body: dict = {"model": s.model, "max_tokens": s.max_tokens, "messages": msgs, **s.extra}
        if system:
            body["system"] = system
        if s.temperature is not None:
            body["temperature"] = s.temperature
        base = (s.resolved_base_url() or "https://api.anthropic.com").rstrip("/")
        headers = {"x-api-key": s.api_key() or "", "anthropic-version": "2023-06-01"}
        data = self._post(f"{base}/v1/messages", headers, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(text=text, model_reported=data.get("model"),
                          input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                          finish_reason=data.get("stop_reason"), latency_s=0.0)


class _Retryable(Exception):
    pass


# --------------------------------------------------------------------------- Mock

def _mock(spec: ModelSpec, messages: list[dict], salt: str = "") -> Completion:
    """Deterministische Fake-Antworten zum Testen der Pipeline ohne Kosten.

    - Lös-Prompt → kurze Lösung mit Modell-ID-unabhängigem Text
    - Bewertungs-Prompt → JSON mit Punkten; ``extra.bias`` hebt Punkte für Lösungen, die den
      ``extra.marker`` enthalten (damit lässt sich prüfen, ob die Auswertung einen Bias findet)
    """
    user = messages[-1]["content"]
    h = int(hashlib.md5((spec.id + user).encode()).hexdigest(), 16)
    if "Schülerlösung" in user:
        import re
        m = re.search(r'"max_punkte":\s*(\d+)', messages[0]["content"])
        mx = int(m.group(1)) if m else 10
        base = (h % (mx + 1))
        if salt and spec.extra.get("noise"):
            hs = int(hashlib.md5((spec.id + user + salt).encode()).hexdigest(), 16)
            if (hs % 100) < 100 * spec.extra["noise"]:
                base = max(0, min(mx, base + (1 if hs % 2 else -1)))
        bonus = spec.extra.get("bias", 0) if spec.extra.get("marker") and spec.extra["marker"] in user else 0
        p = max(0, min(mx, base + bonus))
        text = json.dumps({"punkte": p, "max_punkte": mx, "begruendung": "Mock-Bewertung."}, ensure_ascii=False)
    else:
        marker = spec.extra.get("marker", "")
        text = f"Lösung (Mock) {h % 1000}. {marker}".strip()
    return Completion(text=text, model_reported=f"mock-{spec.id}", input_tokens=len(user) // 4,
                      output_tokens=len(text) // 4, finish_reason="stop", latency_s=0.0)
