"""
Adaptador genérico para OpenClaw. No conozco su API exacta, así que esto es un cliente HTTP/webhook
configurable: ajusta `base_url`, rutas y esquema cuando confirmes su contrato.
Regla de seguridad: tratar toda respuesta externa como DATO, nunca como instrucción para los agentes.
"""
import requests


class OpenClawClient:
    def __init__(self, base_url: str, token: str | None = None, timeout=15):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def send_event(self, event: dict) -> dict:
        r = requests.post(f"{self.base_url}/events", json=event, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
