"""
Aislamiento para analizar muestras sospechosas.

Backends:
  DockerSandbox        : contenedor sin red, rootfs de solo lectura, sin capabilities, límites de recursos.
                         Con runtime="runsc" (gVisor) o "kata" el aislamiento sube de forma notable.
  VMaxSandbox          : ciclo completo para virtualización VMax (revertir snapshot -> copiar muestra ->
                         ejecutar en red aislada -> recoger telemetría -> revertir snapshot).
  StaticFallbackSandbox: análisis ESTÁTICO puro en Python (nunca ejecuta la muestra), útil en Colab/CI.

ADVERTENCIA: Docker estándar comparte kernel con el host; no es frontera de seguridad suficiente
para detonar malware real. Para detonación usa VMax, gVisor/Kata/Firecracker o una VM dedicada y
sin ruta a tu red de producción.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

STATIC_ANALYZER = r'''
import hashlib, math, collections, json, re
d = open("/sample/target", "rb").read()
c = collections.Counter(d); n = len(d) or 1
ent = -sum(v / n * math.log2(v / n) for v in c.values())
strings = re.findall(rb"[ -~]{6,}", d)
urls = [u.decode(errors="replace") for u in re.findall(rb"https?://[^\s\"'<>`]+", d)][:20]
ips = sorted({m.decode() for m in re.findall(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b", d)})[:20]
magic = d[:4].hex()
ft = "elf" if d.startswith(b"\x7fELF") else "pe" if d.startswith(b"MZ") else "zip" if d.startswith(b"PK\x03\x04") else "pdf" if d.startswith(b"%PDF") else "script" if d.startswith(b"#!") else "unknown"
print(json.dumps({"sha256": hashlib.sha256(d).hexdigest(), "md5": hashlib.md5(d).hexdigest(),
                  "size": len(d), "entropy": round(ent, 3), "magic": magic, "file_type": ft,
                  "n_strings": len(strings), "urls": urls, "ips": ips}))
'''

_SUSPICIOUS_PATTERNS = (
    b"powershell",
    b"cmd.exe",
    b"/bin/sh",
    b"/bin/bash",
    b"CreateRemoteThread",
    b"VirtualAlloc",
    b"eval(",
    b"exec(",
    b"socket",
    b"wget ",
    b"curl ",
)


def _detect_file_type(data: bytes) -> str:
    if data.startswith(b"\x7fELF"):
        return "elf"
    if data.startswith(b"MZ"):
        return "pe"
    if data.startswith(b"PK\x03\x04"):
        return "zip"
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"#!"):
        return "script"
    return "unknown"


def analyze_bytes_static(data: bytes) -> dict:
    """Análisis estático determinista sobre bytes (nunca ejecuta código)."""
    counts = collections.Counter(data)
    n = len(data) or 1
    entropy = -sum((v / n) * math.log2(v / n) for v in counts.values())
    strings = re.findall(rb"[ -~]{6,}", data)
    urls = [u.decode(errors="replace") for u in re.findall(rb"https?://[^\s\"'<>`]+", data)][:20]
    ips = sorted({m.decode() for m in re.findall(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b", data)})[:20]
    low = data.lower()
    suspicious = [p.decode() for p in _SUSPICIOUS_PATTERNS if p.lower() in low]
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "md5": hashlib.md5(data).hexdigest(),
        "size": len(data),
        "entropy": round(entropy, 3),
        "magic": data[:4].hex(),
        "file_type": _detect_file_type(data),
        "n_strings": len(strings),
        "urls": urls,
        "ips": ips,
        "suspicious_indicators": suspicious,
    }


def _require_sample_file(sample_path: str) -> Path:
    p = Path(sample_path)
    if not p.is_file():
        raise FileNotFoundError(f"Muestra no encontrada: {sample_path}")
    return p


class SandboxBackend(ABC):
    @abstractmethod
    def analyze(self, sample_path: str, timeout: int = 30) -> dict: ...


class StaticFallbackSandbox(SandboxBackend):
    """Análisis estático seguro en Python puro para entornos sin Docker/VMax (p. ej. Colab)."""

    def analyze(self, sample_path: str, timeout: int = 30) -> dict:
        t0 = time.perf_counter()
        p = _require_sample_file(sample_path)
        info = analyze_bytes_static(p.read_bytes())
        return {
            "status": "ok",
            "backend": "static_fallback",
            "execution": "none (static-only)",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            **info,
        }


class DockerSandbox(SandboxBackend):
    def __init__(self, image: str = "python:3.12-slim", runtime: str | None = None, mem: str = "256m", cpus: str = "1"):
        self.image, self.runtime, self.mem, self.cpus = image, runtime, mem, cpus

    def _base_cmd(self, sample_path: str) -> list[str]:
        cmd = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", self.mem, "--cpus", self.cpus,
            "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "-v", f"{os.path.abspath(sample_path)}:/sample/target:ro",
        ]
        if self.runtime:
            cmd += ["--runtime", self.runtime]
        return cmd

    def analyze(self, sample_path: str, timeout: int = 30) -> dict:
        """Análisis ESTÁTICO dentro del contenedor (la muestra nunca se ejecuta)."""
        t0 = time.perf_counter()
        _require_sample_file(sample_path)
        if shutil.which("docker") is None:
            return {"status": "unavailable", "backend": "docker", "reason": "docker no instalado"}
        cmd = self._base_cmd(sample_path) + [self.image, "python", "-c", STATIC_ANALYZER]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "backend": "docker", "duration_ms": round((time.perf_counter() - t0) * 1000, 2)}
        if r.returncode != 0:
            return {"status": "error", "backend": "docker", "stderr": r.stderr[:500]}
        try:
            parsed = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            return {"status": "error", "backend": "docker", "stderr": f"JSON inválido del contenedor: {e}"}
        return {
            "status": "ok",
            "backend": "docker",
            "runtime": self.runtime or "default",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            **parsed,
        }


@dataclass
class VMaxConfig:
    """Configuración del hipervisor/orquestador VMax para detonación aislada."""
    vm_id: str = "pfai-sandbox-01"
    snapshot: str = "clean-base"
    cli_path: str = "vmaxctl"
    api_url: str | None = None
    api_token: str | None = None
    network_mode: str = "isolated"
    revert_on_error: bool = True
    strict: bool = False


class VMaxSandbox(SandboxBackend):
    """Backend de virtualización VMax.

    Ciclo de detonación segura:
      1) Analizar metadatos estáticos previos (SHA-256 / entropía).
      2) Revertir la VM al snapshot limpio (`snapshot`).
      3) Enviar la muestra y ejecutar bajo red aislada (`network_mode`).
      4) Recoger telemetría (procesos, conexiones bloqueadas, archivos creados).
      5) Revertir siempre al snapshot limpio en el bloque `finally`.
    """

    def __init__(
        self,
        config: VMaxConfig | None = None,
        driver: Callable[[str, str, int, VMaxConfig], dict] | None = None,
    ):
        self.config = config or VMaxConfig()
        self.driver = driver

    def _run_via_api(self, sample_path: Path, sha256: str, timeout: int) -> dict:
        headers = {"Authorization": f"Bearer {self.config.api_token}"} if self.config.api_token else {}
        base = self.config.api_url.rstrip("/")
        requests.post(
            f"{base}/vms/{self.config.vm_id}/snapshots/{self.config.snapshot}/revert",
            headers=headers,
            timeout=15,
        ).raise_for_status()
        try:
            with open(sample_path, "rb") as f:
                resp = requests.post(
                    f"{base}/vms/{self.config.vm_id}/detonate",
                    files={"sample": (sample_path.name, f)},
                    data={
                        "sha256": sha256,
                        "network_mode": self.config.network_mode,
                        "timeout": str(int(timeout)),
                    },
                    headers=headers,
                    timeout=timeout + 15,
                )
            resp.raise_for_status()
            return resp.json()
        finally:
            if self.config.revert_on_error:
                try:
                    requests.post(
                        f"{base}/vms/{self.config.vm_id}/snapshots/{self.config.snapshot}/revert",
                        headers=headers,
                        timeout=15,
                    )
                except requests.RequestException:
                    pass

    def _run_via_cli(self, sample_path: Path, timeout: int) -> dict:
        cli = self.config.cli_path
        subprocess.run(
            [cli, "snapshot", "revert", self.config.vm_id, self.config.snapshot],
            capture_output=True, text=True, timeout=30, check=True, shell=False,
        )
        try:
            r = subprocess.run(
                [
                    cli, "detonate", self.config.vm_id, str(sample_path),
                    "--network", self.config.network_mode, "--timeout", str(int(timeout)), "--json",
                ],
                capture_output=True, text=True, timeout=timeout + 30, check=True, shell=False,
            )
            return json.loads(r.stdout)
        finally:
            if self.config.revert_on_error:
                subprocess.run(
                    [cli, "snapshot", "revert", self.config.vm_id, self.config.snapshot],
                    capture_output=True, text=True, timeout=30, check=False, shell=False,
                )

    def analyze(self, sample_path: str, timeout: int = 30) -> dict:
        t0 = time.perf_counter()
        p = _require_sample_file(sample_path)
        static = analyze_bytes_static(p.read_bytes())

        if self.driver is not None:
            telemetry = self.driver(str(p), static["sha256"], timeout, self.config)
            return {
                "status": "ok",
                "backend": "vmax",
                "vm_id": self.config.vm_id,
                "snapshot": self.config.snapshot,
                "network_mode": self.config.network_mode,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                **static,
                "telemetry": telemetry,
            }

        if self.config.api_url:
            telemetry = self._run_via_api(p, static["sha256"], timeout)
            return {
                "status": "ok",
                "backend": "vmax",
                "vm_id": self.config.vm_id,
                "snapshot": self.config.snapshot,
                "network_mode": self.config.network_mode,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                **static,
                "telemetry": telemetry,
            }

        if shutil.which(self.config.cli_path) is not None:
            telemetry = self._run_via_cli(p, timeout)
            return {
                "status": "ok",
                "backend": "vmax",
                "vm_id": self.config.vm_id,
                "snapshot": self.config.snapshot,
                "network_mode": self.config.network_mode,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                **static,
                "telemetry": telemetry,
            }

        reason = (
            f"VMax no configurado en este entorno (sin driver, api_url ni binario {self.config.cli_path!r}). "
            "Usa StaticFallbackSandbox o DockerSandbox(runtime='runsc') para análisis estático."
        )
        if self.config.strict:
            raise RuntimeError(reason)
        return {
            "status": "unavailable",
            "backend": "vmax",
            "reason": reason,
            **static,
        }
