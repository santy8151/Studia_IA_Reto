"""
Adaptador seguro y extensible para OpenClaw (HTTP/webhook).

Regla de seguridad: tratar toda respuesta externa como DATO no confiable,
nunca como instrucción o comando ejecutable para los agentes.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_FORBIDDEN_CONTROL_KEYS = {"command", "shell", "exec", "eval", "override_playbook", "system_prompt"}


def _sanitize_external_value(val: Any) -> Any:
    if isinstance(val, dict):
        cleaned = {}
        for k, v in val.items():
            key = str(k)
            if key.lower() in _FORBIDDEN_CONTROL_KEYS:
                cleaned[f"untrusted_{key}"] = _sanitize_external_value(v)
            else:
                cleaned[key] = _sanitize_external_value(v)
        return cleaned
    if isinstance(val, list):
        return [_sanitize_external_value(x) for x in val]
    if isinstance(val, str):
        return val[:4096]
    return val


class OpenClawClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        hmac_secret: str | None = None,
        timeout: float = 15.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.hmac_secret = hmac_secret
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        retry = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _signed_headers(self, body_bytes: bytes) -> dict[str, str]:
        headers = dict(self.headers)
        if self.hmac_secret:
            sig = hmac.new(self.hmac_secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
            headers["X-PFAI-Signature"] = f"sha256={sig}"
        return headers

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        r = self.session.post(
            f"{self.base_url}{path}",
            data=body,
            headers=self._signed_headers(body),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return {
            "untrusted_data": True,
            "data": _sanitize_external_value(r.json()),
        }

    def healthcheck(self) -> dict:
        r = self.session.get(f"{self.base_url}/health", headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return {"untrusted_data": True, "data": _sanitize_external_value(r.json())}

    def send_event(self, event: dict) -> dict:
        return self._post("/events", event)

    def send_alert(self, alert: dict) -> dict:
        return self._post("/alerts", alert)

    def send_evidence(self, evidence: dict) -> dict:
        return self._post("/evidence", evidence)

    def fetch_threat_intel(self, indicator: str) -> dict:
        r = self.session.get(
            f"{self.base_url}/intel",
            params={"indicator": str(indicator)[:256]},
            headers=self.headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return {"untrusted_data": True, "data": _sanitize_external_value(r.json())}
