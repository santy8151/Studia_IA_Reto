import json

from pfai.cli import main
from pfai.integrity.ledger import EvidenceLedger


def test_cli_capabilities_and_verify_ledger(tmp_path, capsys):
    assert main(["capabilities"]) == 0
    caps = json.loads(capsys.readouterr().out)
    assert "tshark" in caps

    ledger_path = tmp_path / "cli_ledger.jsonl"
    ledger = EvidenceLedger(ledger_path)
    ledger.append("alert", {"score": 1.2})

    assert main(["verify-ledger", "--ledger", str(ledger_path)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["valid"] is True
    assert res["entries"] == 1


def test_cli_analyze_sample_static(tmp_path, capsys):
    sample = tmp_path / "test.sh"
    sample.write_bytes(b"#!/bin/sh\necho hi\n")
    assert main(["analyze-sample", str(sample), "--backend", "static"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["file_type"] == "script"
