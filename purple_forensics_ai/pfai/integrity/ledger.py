"""
Registro de evidencias con cadena de hashes (tamper-evident) y anclaje opcional en Stellar.
Cada entrada incluye el hash de la anterior; modificar cualquiera invalida toda la cadena posterior.
Stellar solo guarda el hash raíz (32 bytes) en un manage_data: ningún dato sensible va a la red pública.
"""
import hashlib
import json
import os
import time


class EvidenceLedger:
    def __init__(self, path="evidence_ledger.jsonl"):
        self.path = path
        self.last = "0" * 64
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    self.last = json.loads(line)["hash"]

    @staticmethod
    def _digest(prev: str, ts: float, kind: str, data: dict) -> str:
        body = json.dumps({"prev": prev, "ts": ts, "kind": kind, "data": data}, sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()

    def append(self, kind: str, data: dict) -> str:
        ts = time.time()
        h = self._digest(self.last, ts, kind, data)
        with open(self.path, "a") as f:
            f.write(json.dumps({"prev": self.last, "ts": ts, "kind": kind, "data": data, "hash": h}, default=str) + "\n")
        self.last = h
        return h

    def verify(self) -> bool:
        prev = "0" * 64
        with open(self.path) as f:
            for line in f:
                e = json.loads(line)
                if e["prev"] != prev or self._digest(prev, e["ts"], e["kind"], e["data"]) != e["hash"]:
                    return False
                prev = e["hash"]
        return True

    def anchor_stellar(self, secret: str, testnet: bool = True) -> str:
        """Ancla el último hash en Stellar. Usa TESTNET mientras desarrollas."""
        from stellar_sdk import Keypair, Network, Server, TransactionBuilder
        server = Server("https://horizon-testnet.stellar.org" if testnet else "https://horizon.stellar.org")
        kp = Keypair.from_secret(secret)
        tx = (TransactionBuilder(server.load_account(kp.public_key),
                                 Network.TESTNET_NETWORK_PASSPHRASE if testnet else Network.PUBLIC_NETWORK_PASSPHRASE,
                                 base_fee=100)
              .append_manage_data_op("evidence_root", bytes.fromhex(self.last))
              .set_timeout(30).build())
        tx.sign(kp)
        return server.submit_transaction(tx)["hash"]
