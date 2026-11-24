"""
Aislamiento para analizar muestras sospechosas.

Backends:
  DockerSandbox : contenedor sin red, rootfs de solo lectura, sin capabilities, límites de recursos.
                  Con runtime="runsc" (gVisor) o "kata" el aislamiento sube de forma notable.
  VMaxSandbox   : punto de extensión para tu tecnología de virtualización (ver TODO).

ADVERTENCIA: Docker estándar comparte kernel con el host; no es frontera de seguridad suficiente
para detonar malware real. Para detonación usa gVisor/Kata/Firecracker o una VM dedicada y
sin ruta a tu red de producción.
"""
import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod

STATIC_ANALYZER = r'''
import hashlib, math, collections, json, re
d = open("/sample/target", "rb").read()
c = collections.Counter(d); n = len(d) or 1
ent = -sum(v / n * math.log2(v / n) for v in c.values())
strings = re.findall(rb"[ -~]{6,}", d)
print(json.dumps({"sha256": hashlib.sha256(d).hexdigest(), "size": len(d),
                  "entropy": round(ent, 3), "magic": d[:4].hex(), "n_strings": len(strings),
                  "urls": [s.decode() for s in strings if s.startswith((b"http://", b"https://"))][:20]}))
'''


class SandboxBackend(ABC):
    @abstractmethod
    def analyze(self, sample_path: str, timeout: int = 30) -> dict: ...


class DockerSandbox(SandboxBackend):
    def __init__(self, image="python:3.12-slim", runtime: str | None = None, mem="256m", cpus="1"):
        self.image, self.runtime, self.mem, self.cpus = image, runtime, mem, cpus

    def _base_cmd(self, sample_path: str) -> list[str]:
        cmd = ["docker", "run", "--rm", "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", "64", "--memory", self.mem, "--cpus", self.cpus,
               "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
               "-v", f"{os.path.abspath(sample_path)}:/sample/target:ro"]
        if self.runtime:
            cmd += ["--runtime", self.runtime]
        return cmd

    def analyze(self, sample_path: str, timeout: int = 30) -> dict:
        """Análisis ESTÁTICO dentro del contenedor (la muestra nunca se ejecuta)."""
        if shutil.which("docker") is None:
            return {"status": "unavailable", "reason": "docker no instalado"}
        cmd = self._base_cmd(sample_path) + [self.image, "python", "-c", STATIC_ANALYZER]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if r.returncode != 0:
            return {"status": "error", "stderr": r.stderr[:500]}
        return {"status": "ok", **json.loads(r.stdout)}


class VMaxSandbox(SandboxBackend):
    """TODO: implementar con la API/CLI real de VMax (snapshot -> copiar muestra -> ejecutar -> recoger
    telemetría -> revertir snapshot). Mantén red aislada y el mismo contrato de retorno que DockerSandbox."""

    def analyze(self, sample_path: str, timeout: int = 30) -> dict:
        raise NotImplementedError("Conecta aquí tu backend VMax")
