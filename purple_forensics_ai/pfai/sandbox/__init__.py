"""Backends de sandboxing para análisis seguro de muestras sospechosas."""
from pfai.sandbox.sandbox import (
    DockerSandbox,
    SandboxBackend,
    StaticFallbackSandbox,
    VMaxConfig,
    VMaxSandbox,
    analyze_bytes_static,
)

__all__ = [
    "DockerSandbox",
    "SandboxBackend",
    "StaticFallbackSandbox",
    "VMaxConfig",
    "VMaxSandbox",
    "analyze_bytes_static",
]
