import pytest
from pfai.integrity.ledger import EvidenceLedger
from pfai.tools.forensics import OutOfScope, Scope, nmap_service_scan


def test_scope_blocks_foreign_ip():
    with pytest.raises(OutOfScope):
        nmap_service_scan("8.8.8.8", Scope(["10.0.0.0/8"]))


def test_scope_rejects_hostnames():
    with pytest.raises(ValueError):
        Scope(["10.0.0.0/8"]).check("example.com; rm -rf /")


def test_scope_supports_ipv4_ipv6_and_is_allowed():
    scope = Scope(["10.0.0.0/8", "fd00::/8"])
    assert scope.is_allowed("10.1.2.3") is True
    assert scope.is_allowed("fd00::1") is True
    assert scope.is_allowed("1.1.1.1") is False
    assert scope.is_allowed("not-an-ip") is False


def test_ledger_detects_tampering_and_handles_empty_lines(tmp_path):
    p = tmp_path / "l.jsonl"
    L = EvidenceLedger(str(p))
    assert L.verify_detailed() == {"valid": True, "entries": 0, "head": "0" * 64, "error": None}
    L.append("a", {"x": 1})
    L.append("b", {"y": 2})
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n")
    det = L.verify_detailed()
    assert det["valid"] is True
    assert det["entries"] == 2
    payload = L.build_anchor_payload()
    assert payload["root_bytes_len"] == 32

    p.write_text(p.read_text().replace('"x": 1', '"x": 9'))
    assert not L.verify()
    assert L.verify_detailed()["valid"] is False
