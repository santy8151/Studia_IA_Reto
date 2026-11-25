"""
Red dinámica de agentes Purple Team sobre un bus asíncrono publish/subscribe:

  simulate ─► RedTeamAgent (LVGAN xp / evasión) ─► flow
                                                     │
  flow ──► DetectorAgent (LVGAN) ──alert──► TriageAgent (MoE) ──triage──► ForensicAgent (tshark/nmap/TSK)
                │ (si simulated & !alert => evasion)                              │ evidence
  sample ─► SandboxAgent (Docker / VMax / Static) ──────────────────────────────► ReportAgent (ledger + resumen)
                                                                                  └─► OpenClawAgent (opcional)

Seguridad: `dynamic_plan` / `PLAYBOOKS` (o un LLM en su lugar) proponen acciones,
pero el código SIEMPRE restringe con `ALLOWED_TOOLS`, `Scope` y validación de parámetros.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pfai.models.moe import FAMILIES
from pfai.tools import forensics as T

PLAYBOOKS: dict[str, list[str]] = {
    "probe": ["tshark_conversations", "nmap_exposure"],
    "bruteforce": ["tshark_conversations"],
    "exfil": ["tshark_conversations", "tshark_hierarchy"],
    "dos": ["tshark_hierarchy"],
    "unknown": ["tshark_conversations"],
}

ALLOWED_TOOLS = frozenset({
    "tshark_conversations",
    "tshark_hierarchy",
    "nmap_exposure",
    "tsk_partitions",
    "tsk_list_files",
})


def dynamic_plan(
    family: str,
    anomaly_score: float,
    confidence: float,
    meta: dict,
    threshold: float | None = None,
) -> list[str]:
    """Planificador dinámico con filtrado estricto contra ``ALLOWED_TOOLS``."""
    plan = list(PLAYBOOKS.get(family, PLAYBOOKS["unknown"]))
    if confidence < 0.70 and "tshark_hierarchy" not in plan:
        plan.append("tshark_hierarchy")
    if threshold is not None and anomaly_score >= threshold * 2.0 and meta.get("dst_ip") and "nmap_exposure" not in plan:
        plan.append("nmap_exposure")
    if meta.get("disk_image"):
        plan.extend(["tsk_partitions", "tsk_list_files"])
    return [tool for tool in dict.fromkeys(plan) if tool in ALLOWED_TOOLS]


@dataclass
class Message:
    topic: str
    payload: dict
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class Bus:
    def __init__(self):
        self.subs: dict[str, list["Agent"]] = defaultdict(list)
        self.q: asyncio.Queue[Message] = asyncio.Queue()
        self.published_counts: Counter[str] = Counter()

    def subscribe(self, topic: str, agent: "Agent") -> None:
        self.subs[topic].append(agent)

    async def publish(self, topic: str, payload: dict, trace_id: str | None = None) -> str:
        tid = trace_id or uuid.uuid4().hex[:8]
        self.published_counts[topic] += 1
        await self.q.put(Message(topic, payload, tid))
        return tid

    async def run(self, idle_timeout: float = 0.15) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(self.q.get(), idle_timeout)
            except asyncio.TimeoutError:
                return
            try:
                agents = list(self.subs[msg.topic])
                if agents:
                    results = await asyncio.gather(*(a.handle(msg) for a in agents), return_exceptions=True)
                    for agent, res in zip(agents, results):
                        if isinstance(res, Exception):
                            await self.publish(
                                "error",
                                {"agent": type(agent).__name__, "error": repr(res)},
                                msg.trace_id,
                            )
            finally:
                self.q.task_done()


class Agent:
    topics: tuple[str, ...] = ()

    def __init__(self, bus: Bus):
        self.bus = bus
        for t in self.topics:
            bus.subscribe(t, self)

    async def handle(self, msg: Message): ...


class RedTeamAgent(Agent):
    """Agente adversario Purple Team: genera variantes sintéticas ``xp`` desde LVGAN y mide evasión."""
    topics = ("simulate",)

    def __init__(self, bus: Bus, lvgan):
        super().__init__(bus)
        self.model = lvgan

    async def handle(self, msg: Message) -> None:
        mode = msg.payload.get("mode", "prior")
        meta = dict(msg.payload.get("meta", {}))
        if mode == "evasion" and "seed_features" in msg.payload:
            seed = np.asarray(msg.payload["seed_features"], dtype=np.float32)
            if seed.ndim == 1:
                seed = seed[None]
            samples = self.model.generate_evasion_variants(
                seed,
                steps=int(msg.payload.get("steps", 20)),
                step_size=float(msg.payload.get("step_size", 0.08)),
            )
        else:
            n = int(msg.payload.get("n", 1))
            samples = self.model.sample(n)

        for idx, vec in enumerate(samples):
            item_meta = {**meta, "simulated": True, "sim_mode": mode, "sim_index": idx}
            await self.bus.publish("flow", {"features": vec, "meta": item_meta}, f"{msg.trace_id}-{idx}")


class DetectorAgent(Agent):
    topics = ("flow",)

    def __init__(self, bus: Bus, lvgan):
        super().__init__(bus)
        self.model = lvgan

    async def handle(self, msg: Message) -> None:
        if self.model.threshold is None:
            raise RuntimeError("LVGAN requiere calibrate() antes de procesar flujos")
        x = np.asarray(msg.payload["features"], dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        comp = self.model.score_components(x)
        score = float(comp["total"][0])
        if score > self.model.threshold:
            await self.bus.publish(
                "alert",
                {
                    **msg.payload,
                    "anomaly_score": score,
                    "rec_error": float(comp["rec_error"][0]),
                    "feat_error": float(comp["feat_error"][0]),
                    "threshold": float(self.model.threshold),
                },
                msg.trace_id,
            )
        elif msg.payload.get("meta", {}).get("simulated"):
            await self.bus.publish(
                "purple_miss",
                {
                    "meta": msg.payload.get("meta", {}),
                    "anomaly_score": score,
                    "threshold": float(self.model.threshold),
                },
                msg.trace_id,
            )


class TriageAgent(Agent):
    topics = ("alert",)

    def __init__(self, bus: Bus, moe, min_conf: float = 0.5):
        super().__init__(bus)
        self.moe, self.min_conf = moe, min_conf

    async def handle(self, msg: Message) -> None:
        x = np.asarray(msg.payload["features"], dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        cls, conf, expert = self.moe.predict(x)
        family = FAMILIES[int(cls[0])] if conf[0] >= self.min_conf else "unknown"
        await self.bus.publish(
            "triage",
            {
                **msg.payload,
                "family": family,
                "confidence": float(conf[0]),
                "expert": int(expert[0]),
            },
            msg.trace_id,
        )


class ForensicAgent(Agent):
    topics = ("triage",)

    def __init__(
        self,
        bus: Bus,
        scope: T.Scope,
        planner: Callable[[str, float, float, dict, float | None], list[str]] | None = None,
    ):
        super().__init__(bus)
        self.scope = scope
        self.planner = planner or dynamic_plan

    def _run_tool(self, name: str, meta: dict):
        if name not in ALLOWED_TOOLS:
            raise PermissionError(f"Herramienta fuera de allowlist: {name!r}")
        pcap = meta.get("pcap")
        disk_image = meta.get("disk_image")
        if name == "tshark_conversations" and pcap:
            return T.tshark_conversations(pcap)
        if name == "tshark_hierarchy" and pcap:
            return T.tshark_protocol_hierarchy(pcap)
        if name == "nmap_exposure" and meta.get("dst_ip"):
            return T.nmap_service_scan(meta["dst_ip"], self.scope)
        if name == "tsk_partitions" and disk_image:
            return T.tsk_partitions(disk_image)
        if name == "tsk_list_files" and disk_image:
            return T.tsk_list_files(disk_image, offset=meta.get("offset"), deleted_only=bool(meta.get("deleted_only", False)))
        return "omitido: faltan parámetros"

    async def handle(self, msg: Message) -> None:
        meta = msg.payload.get("meta", {})
        proposed = self.planner(
            msg.payload["family"],
            float(msg.payload["anomaly_score"]),
            float(msg.payload["confidence"]),
            meta,
            msg.payload.get("threshold"),
        )
        # El planificador propone, el código restringe
        tools = [t for t in proposed if t in ALLOWED_TOOLS]
        results = {}
        for name in tools:
            try:
                results[name] = await asyncio.to_thread(self._run_tool, name, meta)
            except (T.ToolUnavailable, T.OutOfScope, FileNotFoundError, PermissionError, RuntimeError, ValueError) as e:
                results[name] = f"error: {e}"
        await self.bus.publish(
            "evidence",
            {
                "family": msg.payload["family"],
                "meta": meta,
                "anomaly_score": msg.payload["anomaly_score"],
                "confidence": msg.payload["confidence"],
                "expert": msg.payload.get("expert"),
                "tools": tools,
                "results": results,
            },
            msg.trace_id,
        )


class SandboxAgent(Agent):
    topics = ("sample",)

    def __init__(self, bus: Bus, backend):
        super().__init__(bus)
        self.backend = backend

    async def handle(self, msg: Message) -> None:
        res = await asyncio.to_thread(self.backend.analyze, msg.payload["path"])
        await self.bus.publish(
            "evidence",
            {"family": "sample", "meta": msg.payload, "results": res},
            msg.trace_id,
        )


class OpenClawAgent(Agent):
    """Despacha resúmenes hacia OpenClaw tratando respuestas externas como dato no confiable."""
    topics = ("evidence", "alert")

    def __init__(self, bus: Bus, client):
        super().__init__(bus)
        self.client = client
        self.sent_responses: list[dict] = []

    async def handle(self, msg: Message) -> None:
        summary = {
            "trace_id": msg.trace_id,
            "topic": msg.topic,
            "family": msg.payload.get("family"),
            "anomaly_score": msg.payload.get("anomaly_score"),
            "confidence": msg.payload.get("confidence"),
        }
        if msg.topic == "evidence":
            resp = await asyncio.to_thread(self.client.send_evidence, summary)
        else:
            resp = await asyncio.to_thread(self.client.send_alert, summary)
        self.sent_responses.append({"trace_id": msg.trace_id, "response": resp})


class ReportAgent(Agent):
    topics = ("evidence", "purple_miss", "error")

    def __init__(self, bus: Bus, ledger):
        super().__init__(bus)
        self.ledger = ledger
        self.records: list[tuple[str, str, str]] = []
        self.events: list[dict] = []

    async def handle(self, msg: Message) -> None:
        record_data = {"trace": msg.trace_id, "topic": msg.topic, **msg.payload}
        h = self.ledger.append(msg.topic, record_data)
        self.records.append((msg.trace_id, msg.topic, h))
        self.events.append(record_data)

    def summary(self) -> dict:
        by_topic = Counter(topic for _, topic, _ in self.records)
        by_family = Counter(
            e.get("family", "unknown") for e in self.events if e.get("topic") == "evidence"
        )
        sim_detected = sum(
            1 for e in self.events if e.get("topic") == "evidence" and e.get("meta", {}).get("simulated")
        )
        sim_missed = sum(1 for e in self.events if e.get("topic") == "purple_miss")
        sim_total = sim_detected + sim_missed
        return {
            "total_records": len(self.records),
            "by_topic": dict(by_topic),
            "by_family": dict(by_family),
            "purple_team": {
                "simulated_total": sim_total,
                "detected": sim_detected,
                "evaded": sim_missed,
                "detection_rate": (sim_detected / sim_total) if sim_total else None,
            },
            "ledger": self.ledger.verify_detailed(),
        }
