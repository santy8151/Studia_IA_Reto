import pytest
from pfai.integrity.ledger import EvidenceLedger
from pfai.tools.forensics import OutOfScope, Scope, nmap_service_scan


def test_scope_blocks_foreign_ip():
    with pytest.raises(OutOfScope):
        nmap_service_scan("8.8.8.8", Scope(["10.0.0.0/8"]))


def test_scope_rejects_hostnames():
    with pytest.raises(ValueError):
        Scope(["10.0.0.0/8"]).check("example.com; rm -rf /")


def test_ledger_detects_tampering(tmp_path):
    p = tmp_path / "l.jsonl"
    L = EvidenceLedger(str(p))
    L.append("a", {"x": 1}); L.append("b", {"y": 2})
    assert L.verify()
    p.write_text(p.read_text().replace('"x": 1', '"x": 9'))
    assert not L.verify()
