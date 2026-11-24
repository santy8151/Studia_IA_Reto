"""
Red de agentes sobre un bus asíncrono publish/subscribe:

  flow ──► DetectorAgent (LVGAN) ──alert──► TriageAgent (MoE) ──triage──► ForensicAgent (tshark/nmap/TSK)
                                                                              │evidence
  sample ─► SandboxAgent ──────────────────────────────────────────────────►  ReportAgent (ledger)

Adaptación: la tabla PLAYBOOKS decide qué herramientas lanzar por familia de ataque. Para planificación
dinámica puedes sustituir `plan()` por un LLM, manteniendo SIEMPRE: allowlist de herramientas,
Scope de red y argumentos validados (el LLM propone, el código restringe).
"""
import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pfai.models.moe import FAMILIES
from pfai.tools import forensics as T


@dataclass
class Message:
    topic: str
    payload: dict
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class Bus:
    def __init__(self):
        self.subs = defaultdict(list)
        self.q: asyncio.Queue = asyncio.Queue()

    def subscribe(self, topic, agent):
        self.subs[topic].append(agent)

    async def publish(self, topic, payload, trace_id=None):
        await self.q.put(Message(topic, payload, trace_id or uuid.uuid4().hex[:8]))

    async def run(self, idle_timeout=0.5):
        while True:
            try:
                msg = await asyncio.wait_for(self.q.get(), idle_timeout)
            except asyncio.TimeoutError:
                return
            for agent in self.subs[msg.topic]:
                try:
                    await agent.handle(msg)
                except Exception as e:  # un agente caído no tumba al resto
                    await self.publish("error", {"agent": type(agent).__name__, "error": repr(e)}, msg.trace_id)


class Agent:
    topics: tuple = ()

    def __init__(self, bus: Bus):
        self.bus = bus
        for t in self.topics:
            bus.subscribe(t, self)

    async def handle(self, msg: Message): ...


class DetectorAgent(Agent):
    topics = ("flow",)

    def __init__(self, bus, lvgan):
        super().__init__(bus)
        self.model = lvgan

    async def handle(self, msg):
        x = np.asarray(msg.payload["features"], dtype=np.float32)[None]
        score = float(self.model.score(x)[0])
        if score > self.model.threshold:
            await self.bus.publish("alert", {**msg.payload, "anomaly_score": score}, msg.trace_id)


class TriageAgent(Agent):
    topics = ("alert",)

    def __init__(self, bus, moe, min_conf=0.5):
        super().__init__(bus)
        self.moe, self.min_conf = moe, min_conf

    async def handle(self, msg):
        x = np.asarray(msg.payload["features"], dtype=np.float32)[None]
        cls, conf, expert = self.moe.predict(x)
        family = FAMILIES[int(cls[0])] if conf[0] >= self.min_conf else "unknown"
        await self.bus.publish("triage", {**msg.payload, "family": family, "confidence": float(conf[0]),
                                          "expert": int(expert[0])}, msg.trace_id)


PLAYBOOKS = {
    "probe": ["tshark_conversations", "nmap_exposure"],
    "bruteforce": ["tshark_conversations"],
    "exfil": ["tshark_conversations", "tshark_hierarchy"],
    "dos": ["tshark_hierarchy"],
    "unknown": ["tshark_conversations"],
}


class ForensicAgent(Agent):
    topics = ("triage",)

    def __init__(self, bus, scope: T.Scope):
        super().__init__(bus)
        self.scope = scope

    def _run_tool(self, name, meta):
        pcap = meta.get("pcap")
        if name == "tshark_conversations" and pcap:
            return T.tshark_conversations(pcap)
        if name == "tshark_hierarchy" and pcap:
            return T.tshark_protocol_hierarchy(pcap)
        if name == "nmap_exposure" and meta.get("dst_ip"):
            return T.nmap_service_scan(meta["dst_ip"], self.scope)
        return "omitido: faltan parámetros"

    async def handle(self, msg):
        meta = msg.payload.get("meta", {})
        results = {}
        for name in PLAYBOOKS.get(msg.payload["family"], PLAYBOOKS["unknown"]):
            try:
                results[name] = await asyncio.to_thread(self._run_tool, name, meta)
            except (T.ToolUnavailable, T.OutOfScope, RuntimeError, ValueError) as e:
                results[name] = f"error: {e}"
        await self.bus.publish("evidence", {"family": msg.payload["family"], "meta": meta,
                                            "anomaly_score": msg.payload["anomaly_score"],
                                            "confidence": msg.payload["confidence"], "results": results}, msg.trace_id)


class SandboxAgent(Agent):
    topics = ("sample",)

    def __init__(self, bus, backend):
        super().__init__(bus)
        self.backend = backend

    async def handle(self, msg):
        res = await asyncio.to_thread(self.backend.analyze, msg.payload["path"])
        await self.bus.publish("evidence", {"family": "sample", "meta": msg.payload, "results": res}, msg.trace_id)


class ReportAgent(Agent):
    topics = ("evidence", "error")

    def __init__(self, bus, ledger):
        super().__init__(bus)
        self.ledger = ledger
        self.records = []

    async def handle(self, msg):
        h = self.ledger.append(msg.topic, {"trace": msg.trace_id, **msg.payload})
        self.records.append((msg.trace_id, msg.topic, h))
