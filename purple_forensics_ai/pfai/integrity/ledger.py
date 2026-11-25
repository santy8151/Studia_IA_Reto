"""
Registro de evidencias con cadena de hashes (tamper-evident) y anclaje opcional en Stellar.
Cada entrada incluye el hash de la anterior; modificar cualquiera invalida toda la cadena posterior.
Stellar solo guarda el hash raíz (32 bytes) en un manage_data: ningún dato sensible va a la red pública.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceLedger:
    def __init__(self, path: str | Path = "evidence_ledger.jsonl"):
        self.path = str(path)
        self._lock = threading.Lock()
        self.last = "0" * 64
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        self.last = json.loads(stripped)["hash"]

    @staticmethod
    def _digest(prev: str, ts: float, kind: str, data: dict) -> str:
        body = json.dumps({"prev": prev, "ts": ts, "kind": kind, "data": data}, sort_keys=True, default=str)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def append(self, kind: str, data: dict) -> str:
        with self._lock:
            ts = time.time()
            h = self._digest(self.last, ts, kind, data)
            parent = Path(self.path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
            entry = {"prev": self.last, "ts": ts, "kind": kind, "data": data, "hash": h}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            self.last = h
            return h

    def entries(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with self._lock, open(self.path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    out.append(json.loads(stripped))
        return out

    def verify_detailed(self) -> dict:
        """Verifica toda la cadena y devuelve diagnóstico estructurado."""
        if not os.path.exists(self.path):
            return {"valid": True, "entries": 0, "head": "0" * 64, "error": None}
        prev = "0" * 64
        count = 0
        with self._lock, open(self.path, encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    e = json.loads(stripped)
                except json.JSONDecodeError as err:
                    return {"valid": False, "entries": count, "head": prev, "error": f"Línea {idx}: JSON inválido ({err})"}
                if e.get("prev") != prev or self._digest(prev, e.get("ts"), e.get("kind"), e.get("data")) != e.get("hash"):
                    return {"valid": False, "entries": count, "head": prev, "error": f"Línea {idx}: hash/encadenado inválido"}
                prev = e["hash"]
                count += 1
        return {"valid": True, "entries": count, "head": prev, "error": None}

    def verify(self) -> bool:
        return bool(self.verify_detailed()["valid"])

    def build_anchor_payload(self, data_name: str = "evidence_root") -> dict:
        """Prepara el payload de 32 bytes sin enviar datos sensibles ni transacción."""
        if not _HEX64_RE.fullmatch(self.last):
            raise ValueError(f"Hash raíz inválido para Stellar: {self.last!r}")
        if self.last == "0" * 64:
            raise ValueError("El ledger está vacío; registra al menos una evidencia antes de anclar")
        raw = bytes.fromhex(self.last)
        return {
            "data_name": data_name,
            "root_hex": self.last,
            "root_bytes_len": len(raw),
            "root_base64": base64.b64encode(raw).decode("ascii"),
        }

    def anchor_stellar(self, secret: str, testnet: bool = True, data_name: str = "evidence_root") -> str:
        """Ancla el último hash (32 bytes) en Stellar. Usa TESTNET mientras desarrollas."""
        payload = self.build_anchor_payload(data_name=data_name)
        from stellar_sdk import Keypair, Network, Server, TransactionBuilder

        horizon_url = "https://horizon-testnet.stellar.org" if testnet else "https://horizon.stellar.org"
        passphrase = Network.TESTNET_NETWORK_PASSPHRASE if testnet else Network.PUBLIC_NETWORK_PASSPHRASE
        server = Server(horizon_url)
        kp = Keypair.from_secret(secret)
        tx = (
            TransactionBuilder(
                server.load_account(kp.public_key),
                passphrase,
                base_fee=100,
            )
            .append_manage_data_op(data_name, bytes.fromhex(payload["root_hex"]))
            .set_timeout(30)
            .build()
        )
        tx.sign(kp)
        return server.submit_transaction(tx)["hash"]


def verify_stellar_anchor(
    tx_hash: str,
    expected_hash: str | None = None,
    testnet: bool = True,
    data_name: str = "evidence_root",
) -> dict:
    """Consulta Horizon para validar que una transacción contiene el ``evidence_root`` esperado."""
    from stellar_sdk import Server

    horizon_url = "https://horizon-testnet.stellar.org" if testnet else "https://horizon.stellar.org"
    server = Server(horizon_url)
    ops = server.operations().for_transaction(tx_hash).call().get("_embedded", {}).get("records", [])
    for op in ops:
        if op.get("type") == "manage_data" and op.get("name") == data_name:
            raw = base64.b64decode(op.get("value", ""))
            on_chain_hex = raw.hex()
            matches = True if expected_hash is None else (on_chain_hex == expected_hash.lower())
            return {
                "found": True,
                "tx_hash": tx_hash,
                "on_chain_root": on_chain_hex,
                "matches_expected": matches,
            }
    return {"found": False, "tx_hash": tx_hash, "on_chain_root": None, "matches_expected": False}
