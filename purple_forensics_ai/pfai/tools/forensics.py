"""
Wrappers seguros (sin shell, con timeout y allowlist de alcance) para:
  - Wireshark  -> tshark (CLI)
  - Nmap       -> solo contra activos AUTORIZADOS (Scope)
  - Autopsy    -> su motor es The Sleuth Kit (mmls/fls/icat); Autopsy (GUI) puede abrir las mismas imágenes
"""
import hashlib
import ipaddress
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


class ToolUnavailable(RuntimeError):
    pass


class OutOfScope(PermissionError):
    pass


@dataclass
class Scope:
    """Redes que el equipo está autorizado a escanear. Todo lo demás se rechaza."""
    allowed_cidrs: list[str] = field(default_factory=list)

    def check(self, target: str):
        ip = ipaddress.ip_address(target)          # lanza ValueError si no es una IP literal
        if not any(ip in ipaddress.ip_network(c) for c in self.allowed_cidrs):
            raise OutOfScope(f"{target} fuera del alcance autorizado {self.allowed_cidrs}")


def _run(cmd: list[str], timeout=120) -> str:
    if shutil.which(cmd[0]) is None:
        raise ToolUnavailable(f"{cmd[0]} no está instalado")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} falló ({r.returncode}): {r.stderr[:500]}")
    return r.stdout


# ---------------- Wireshark / tshark ----------------
def tshark_capture(iface: str, seconds: int, out_pcap: str):
    _run(["tshark", "-i", iface, "-a", f"duration:{int(seconds)}", "-w", out_pcap], timeout=seconds + 30)
    return out_pcap


def tshark_conversations(pcap: str) -> str:
    return _run(["tshark", "-r", pcap, "-q", "-z", "conv,tcp", "-z", "conv,udp"])


def tshark_protocol_hierarchy(pcap: str) -> str:
    return _run(["tshark", "-r", pcap, "-q", "-z", "io,phs"])


# ---------------- Nmap ----------------
def nmap_service_scan(target: str, scope: Scope, top_ports=100) -> list[dict]:
    """Verificación de exposición de un activo PROPIO (descubrimiento de servicios)."""
    scope.check(target)
    xml = _run(["nmap", "-sV", "-T3", "--top-ports", str(int(top_ports)), "-oX", "-", target], timeout=600)
    out = []
    for port in ET.fromstring(xml).iter("port"):
        st = port.find("state")
        svc = port.find("service")
        if st is not None and st.get("state") == "open":
            out.append({"port": int(port.get("portid")), "proto": port.get("protocol"),
                        "service": svc.get("name") if svc is not None else None,
                        "product": svc.get("product") if svc is not None else None,
                        "version": svc.get("version") if svc is not None else None})
    return out


# ---------------- Sleuth Kit (motor de Autopsy) ----------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tsk_partitions(image: str) -> str:
    return _run(["mmls", image])


def tsk_list_files(image: str, offset: int | None = None, deleted_only=False) -> str:
    cmd = ["fls", "-r", "-p"]
    if deleted_only:
        cmd.append("-d")
    if offset is not None:
        cmd += ["-o", str(int(offset))]
    return _run(cmd + [image], timeout=600)
