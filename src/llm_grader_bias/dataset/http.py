"""Höflicher HTTP-Client: feste Pause zwischen Requests, Retries, Datei-Cache."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class PoliteClient:
    def __init__(
        self,
        user_agent: str,
        delay_seconds: float = 1.0,
        timeout_seconds: float = 60,
        retries: int = 3,
    ) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.delay = delay_seconds
        self.timeout = timeout_seconds
        self.retries = retries
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def get(self, url: str, allow_404: bool = False) -> requests.Response | None:
        """GET mit Retries. Gibt bei 404 und allow_404=True None zurück."""
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._wait()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("Fehler bei %s (Versuch %d/%d): %s", url, attempt, self.retries, exc)
                time.sleep(2 * attempt)
                continue
            if resp.status_code == 404 and allow_404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                log.warning("HTTP %d bei %s (Versuch %d/%d)", resp.status_code, url, attempt, self.retries)
                time.sleep(5 * attempt)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Abruf fehlgeschlagen: {url}") from last_exc

    def get_text(self, url: str) -> str:
        resp = self.get(url)
        assert resp is not None
        if "charset" in resp.headers.get("Content-Type", "").lower():
            return resp.text
        # Ohne Charset-Angabe würde requests ISO-8859-1 annehmen → Umlaute kaputt
        try:
            return resp.content.decode("utf-8")
        except UnicodeDecodeError:
            return resp.content.decode(resp.apparent_encoding or "latin-1", errors="replace")

    def download(self, url: str, target: Path, allow_404: bool = False) -> Path | None:
        """Lädt eine Datei nach `target`. Existiert sie schon, wird nichts geladen (Cache)."""
        sidecar = target.with_name(target.name + ".url")
        if target.exists() and target.stat().st_size > 0:
            # Cache-Treffer nur, wenn die Datei auch von dieser URL stammt
            if sidecar.exists() and sidecar.read_text().strip() != url:
                return None
            return target
        resp = self.get(url, allow_404=allow_404)
        if resp is None:
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not resp.content.startswith(b"%PDF"):
            log.warning("Keine PDF unter %s (Content-Type: %s) – übersprungen", url, content_type)
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(target)
        sidecar.write_text(url)
        return target


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
