"""Demo end-to-end con datos sintéticos de flujos (sin tocar nada real)."""
import asyncio
import numpy as np
from sklearn.metrics import roc_auc_score

from pfai.agents.orchestrator import Bus, DetectorAgent, ForensicAgent, ReportAgent, TriageAgent
from pfai.integrity.ledger import EvidenceLedger
from pfai.models.lvgan import LVGAN, LVGANConfig
from pfai.models.moe import FAMILIES, MoEClassifier
from pfai.tools.forensics import Scope

rng = np.random.default_rng(0)
D = 12


def make(n, fam):
    """Familias con 'huellas' distintas en el espacio de features (juguete)."""
    x = rng.normal(0, 1, (n, D))
    if fam == "dos":
        x[:, 0] += 4
    if fam == "probe":
        x[:, 1] += 4; x[:, 2] -= 3
    if fam == "bruteforce":
        x[:, 3] += 4
    if fam == "exfil":
        x[:, 4] += 4; x[:, 5] += 3
    return x.astype(np.float32)


benign = make(6000, "benign")
Xtr, Xval = benign[:5000], benign[5000:]
lv = LVGAN(LVGANConfig(d_in=D, epochs=40)).fit(Xtr)
lv.calibrate(Xval, 0.99)

atk_X = np.vstack([make(300, f) for f in FAMILIES[1:]])
atk_y = np.repeat(np.arange(1, 5), 300)
auc = roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(len(atk_X))], lv.score(np.vstack([Xval, atk_X])))
print(f"AUC detección de anomalías: {auc:.3f}")

Xm = np.vstack([benign[:1500], atk_X]); ym = np.r_[np.zeros(1500, int), atk_y]
moe = MoEClassifier(D).fit(Xm, ym, epochs=40)

async def main():
    bus = Bus()
    DetectorAgent(bus, lv); TriageAgent(bus, moe)
    ForensicAgent(bus, Scope(["10.0.0.0/8", "192.168.0.0/16"]))
    rep = ReportAgent(bus, EvidenceLedger("demo_ledger.jsonl"))
    for fam in ["probe", "exfil", "benign"]:
        await bus.publish("flow", {"features": make(1, fam)[0], "meta": {"dst_ip": "10.0.0.5", "fam_true": fam}})
    await bus.run()
    for tid, topic, h in rep.records:
        print(tid, topic, h[:12])
    print("Ledger íntegro:", rep.ledger.verify())

asyncio.run(main())
