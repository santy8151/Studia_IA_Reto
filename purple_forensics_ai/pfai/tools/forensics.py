"""
Wrappers seguros (sin shell, con timeout y allowlist de alcance) para:
  - Wireshark  -> tshark (CLI) + parsers estructurados
  - Nmap       -> solo contra activos AUTORIZADOS (Scope)
  - Autopsy    -> su motor es The Sleuth Kit (mmls/fls/icat); Autopsy (GUI) puede abrir las mismas imágenes
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_INODE_RE = re.compile(r"^\d+(?:-\d+){0,2}$")
_CONV_ROW_RE = re.compile(
    r"^(?P<src>\S+)\s+<->\s+(?P<dst>\S+)\s+"
    r"(?P<rx_frames>[\d,]+)\s+(?P<rx_bytes>[\d.,]+(?:\s*(?:bytes|[kMG]?B))?)\s+"
    r"(?P<tx_frames>[\d,]+)\s+(?P<tx_bytes>[\d.,]+(?:\s*(?:bytes|[kMG]?B))?)\s+"
    r"(?P<total_frames>[\d,]+)\s+(?P<total_bytes>[\d.,]+(?:\s*(?:bytes|[kMG]?B))?)$"
)
_FLS_ROW_RE = re.compile(
    r"^(?P<kind>[a-zA-Z]/[a-zA-Z]|[-a-zA-Z]+)\s+(?:(?P<deleted>\*)\s+)?(?P<inode>\d+(?:-\d+){0,2})(?:\([^)]*\))?:\s+(?P<path>.+)$"
)


class ToolUnavailable(RuntimeError):
    pass


class OutOfScope(PermissionError):
    pass


@dataclass
class Scope:
    """Redes que el equipo está autorizado a escanear. Todo lo demás se rechaza."""
    allowed_cidrs: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._networks = [ipaddress.ip_network(c, strict=False) for c in self.allowed_cidrs]

    def check(self, target: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        ip = ipaddress.ip_address(target)  # lanza ValueError si no es una IP literal
        if not any(ip in net for net in self._networks):
            raise OutOfScope(f"{target} fuera del alcance autorizado {self.allowed_cidrs}")
        return ip

    def is_allowed(self, target: str) -> bool:
        try:
            self.check(target)
            return True
        except (ValueError, OutOfScope):
            return False


def forensic_capabilities() -> dict[str, bool]:
    """Informa qué binarios forenses/sandbox están instalados en el sistema."""
    tools = ("tshark", "nmap", "mmls", "fls", "icat", "docker")
    return {name: shutil.which(name) is not None for name in tools}


def _validate_existing_file(path: str | Path, label: str = "archivo") -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No existe el {label}: {path}")
    return p


def _run(cmd: list[str], timeout: int = 120) -> str:
    if shutil.which(cmd[0]) is None:
        raise ToolUnavailable(f"{cmd[0]} no está instalado")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} falló ({r.returncode}): {r.stderr[:500]}")
    return r.stdout


def _run_bytes(cmd: list[str], timeout: int = 120) -> bytes:
    if shutil.which(cmd[0]) is None:
        raise ToolUnavailable(f"{cmd[0]} no está instalado")
    r = subprocess.run(cmd, capture_output=True, text=False, timeout=timeout, shell=False)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{cmd[0]} falló ({r.returncode}): {err}")
    return r.stdout


# ---------------- Wireshark / tshark ----------------
def tshark_capture(iface: str, seconds: int, out_pcap: str) -> str:
    if not iface or not re.fullmatch(r"[a-zA-Z0-9_.:-]+", iface):
        raise ValueError(f"Nombre de interfaz inválido: {iface!r}")
    dur = max(1, int(seconds))
    _run(["tshark", "-i", iface, "-a", f"duration:{dur}", "-w", out_pcap], timeout=dur + 30)
    return out_pcap


def tshark_conversations(pcap: str) -> str:
    _validate_existing_file(pcap, "PCAP")
    return _run(["tshark", "-r", pcap, "-q", "-z", "conv,tcp", "-z", "conv,udp"])


def parse_tshark_conversations(raw_output: str) -> list[dict]:
    """Extrae conversaciones estructuradas desde la salida de ``tshark -z conv,...``."""
    items: list[dict] = []
    proto = "unknown"
    for line in raw_output.splitlines():
        s = line.strip()
        if s.startswith("TCP Conversations"):
            proto = "tcp"
            continue
        if s.startswith("UDP Conversations"):
            proto = "udp"
            continue
        m = _CONV_ROW_RE.match(s)
        if m:
            items.append(
                {
                    "proto": proto,
                    "src": m.group("src"),
                    "dst": m.group("dst"),
                    "rx_frames": int(m.group("rx_frames").replace(",", "")),
                    "tx_frames": int(m.group("tx_frames").replace(",", "")),
                    "total_frames": int(m.group("total_frames").replace(",", "")),
                    "total_bytes": m.group("total_bytes").strip(),
                }
            )
    return items


def tshark_conversations_parsed(pcap: str) -> list[dict]:
    return parse_tshark_conversations(tshark_conversations(pcap))


def tshark_protocol_hierarchy(pcap: str) -> str:
    _validate_existing_file(pcap, "PCAP")
    return _run(["tshark", "-r", pcap, "-q", "-z", "io,phs"])


def parse_tshark_protocol_hierarchy(raw_output: str) -> list[dict]:
    """Extrae jerarquía de protocolos desde la salida de ``tshark -z io,phs``."""
    out: list[dict] = []
    pat = re.compile(r"^(?P<proto>[a-zA-Z0-9_.-]+)\s+frames:(?P<frames>\d+)\s+bytes:(?P<bytes>\d+)")
    for line in raw_output.splitlines():
        m = pat.match(line.strip())
        if m:
            out.append(
                {
                    "protocol": m.group("proto"),
                    "frames": int(m.group("frames")),
                    "bytes": int(m.group("bytes")),
                }
            )
    return out


def tshark_protocol_hierarchy_parsed(pcap: str) -> list[dict]:
    return parse_tshark_protocol_hierarchy(tshark_protocol_hierarchy(pcap))


# ---------------- Nmap ----------------
def nmap_service_scan(target: str, scope: Scope, top_ports: int = 100) -> list[dict]:
    """Verificación de exposición de un activo PROPIO (descubrimiento de servicios)."""
    scope.check(target)
    n_ports = max(1, min(65535, int(top_ports)))
    xml = _run(["nmap", "-sV", "-T3", "--top-ports", str(n_ports), "-oX", "-", target], timeout=600)
    out = []
    for port in ET.fromstring(xml).iter("port"):
        st = port.find("state")
        svc = port.find("service")
        if st is not None and st.get("state") == "open":
            out.append(
                {
                    "port": int(port.get("portid")),
                    "proto": port.get("protocol"),
                    "service": svc.get("name") if svc is not None else None,
                    "product": svc.get("product") if svc is not None else None,
                    "version": svc.get("version") if svc is not None else None,
                }
            )
    return out


# ---------------- Sleuth Kit (motor de Autopsy) ----------------
def sha256_file(path: str | Path) -> str:
    _validate_existing_file(path, "archivo")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tsk_partitions(image: str) -> str:
    _validate_existing_file(image, "imagen de disco")
    return _run(["mmls", image])


def parse_tsk_partitions(raw_output: str) -> list[dict]:
    """Parsea la tabla de particiones devuelta por ``mmls``."""
    parts: list[dict] = []
    pat = re.compile(r"^\d+:\s+(?P<slot>\S+)\s+(?P<start>\d+)\s+(?P<end>\d+)\s+(?P<length>\d+)\s+(?P<desc>.+)$")
    for line in raw_output.splitlines():
        m = pat.match(line.strip())
        if m:
            parts.append(
                {
                    "slot": m.group("slot"),
                    "start": int(m.group("start")),
                    "end": int(m.group("end")),
                    "length": int(m.group("length")),
                    "description": m.group("desc").strip(),
                }
            )
    return parts


def tsk_partitions_parsed(image: str) -> list[dict]:
    return parse_tsk_partitions(tsk_partitions(image))


def tsk_list_files(image: str, offset: int | None = None, deleted_only: bool = False) -> str:
    _validate_existing_file(image, "imagen de disco")
    cmd = ["fls", "-r", "-p"]
    if deleted_only:
        cmd.append("-d")
    if offset is not None:
        cmd += ["-o", str(int(offset))]
    return _run(cmd + [image], timeout=600)


def parse_tsk_list_files(raw_output: str) -> list[dict]:
    """Parsea entradas de sistema de archivos devueltas por ``fls -r -p``."""
    entries: list[dict] = []
    for line in raw_output.splitlines():
        m = _FLS_ROW_RE.match(line.strip())
        if m:
            entries.append(
                {
                    "kind": m.group("kind"),
                    "deleted": bool(m.group("deleted")),
                    "inode": m.group("inode"),
                    "path": m.group("path").strip(),
                }
            )
    return entries


def tsk_list_files_parsed(image: str, offset: int | None = None, deleted_only: bool = False) -> list[dict]:
    return parse_tsk_list_files(tsk_list_files(image, offset=offset, deleted_only=deleted_only))


def tsk_extract_file(image: str, inode: str, out_path: str | Path, offset: int | None = None) -> dict:
    """Extrae un artefacto por inode usando ``icat`` y devuelve ruta + SHA-256."""
    _validate_existing_file(image, "imagen de disco")
    if not _INODE_RE.fullmatch(str(inode)):
        raise ValueError(f"Inode inválido: {inode!r}")
    cmd = ["icat"]
    if offset is not None:
        cmd += ["-o", str(int(offset))]
    cmd += [image, str(inode)]
    raw = _run_bytes(cmd, timeout=600)
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)
    return {
        "out_path": str(dst),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
