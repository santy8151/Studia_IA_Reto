"""Generador tabular realista (mixto numérico + categórico) de flujos de red para pruebas,
benchmark PPML y experimentación en Google Colab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATTACK_PROFILES = {
    "benign": {
        "duration": (0.8, 0.7),
        "src_bytes": (1200.0, 600.0),
        "dst_bytes": (3200.0, 1400.0),
        "pkts": (14.0, 6.0),
        "failed_logins": 0.03,
        "port_entropy": (1.1, 0.35),
        "syn_ratio": (0.18, 0.08),
        "proto": (["tcp", "udp", "icmp"], [0.76, 0.22, 0.02]),
        "service": (["https", "dns", "ssh", "smtp"], [0.55, 0.25, 0.12, 0.08]),
        "flag": (["SF", "S1", "REJ"], [0.88, 0.08, 0.04]),
    },
    "dos": {
        "duration": (0.05, 0.04),
        "src_bytes": (180.0, 90.0),
        "dst_bytes": (40.0, 35.0),
        "pkts": (85.0, 28.0),
        "failed_logins": 0.05,
        "port_entropy": (0.5, 0.2),
        "syn_ratio": (0.92, 0.05),
        "proto": (["tcp", "udp", "icmp"], [0.82, 0.12, 0.06]),
        "service": (["https", "dns", "ssh", "smtp"], [0.72, 0.18, 0.05, 0.05]),
        "flag": (["S0", "REJ", "SF"], [0.78, 0.17, 0.05]),
    },
    "probe": {
        "duration": (0.12, 0.08),
        "src_bytes": (95.0, 40.0),
        "dst_bytes": (65.0, 40.0),
        "pkts": (6.0, 3.0),
        "failed_logins": 0.10,
        "port_entropy": (4.6, 0.55),
        "syn_ratio": (0.75, 0.10),
        "proto": (["tcp", "udp", "icmp"], [0.78, 0.14, 0.08]),
        "service": (["ssh", "https", "dns", "smtp"], [0.40, 0.35, 0.15, 0.10]),
        "flag": (["REJ", "S0", "RSTO"], [0.55, 0.30, 0.15]),
    },
    "bruteforce": {
        "duration": (1.6, 0.5),
        "src_bytes": (680.0, 220.0),
        "dst_bytes": (420.0, 150.0),
        "pkts": (18.0, 5.0),
        "failed_logins": 6.5,
        "port_entropy": (0.4, 0.15),
        "syn_ratio": (0.28, 0.08),
        "proto": (["tcp", "udp", "icmp"], [0.97, 0.02, 0.01]),
        "service": (["ssh", "https", "smtp", "dns"], [0.72, 0.18, 0.08, 0.02]),
        "flag": (["RSTO", "SF", "REJ"], [0.48, 0.32, 0.20]),
    },
    "exfil": {
        "duration": (12.5, 4.0),
        "src_bytes": (48000.0, 14000.0),
        "dst_bytes": (950.0, 400.0),
        "pkts": (110.0, 35.0),
        "failed_logins": 0.04,
        "port_entropy": (2.2, 0.45),
        "syn_ratio": (0.12, 0.05),
        "proto": (["tcp", "udp", "icmp"], [0.80, 0.18, 0.02]),
        "service": (["https", "dns", "ssh", "smtp"], [0.45, 0.35, 0.15, 0.05]),
        "flag": (["SF", "S1", "RSTO"], [0.85, 0.10, 0.05]),
    },
}


def generate_netflow_dataset(
    counts: dict[str, int] | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Genera un DataFrame tabular mixto de eventos de red por familia.

    Columnas numéricas:
        ``duration``, ``src_bytes``, ``dst_bytes``, ``pkts``, ``bytes_per_pkt``,
        ``failed_logins``, ``port_entropy``, ``syn_ratio``.
    Columnas categóricas:
        ``proto``, ``service``, ``flag``.
    Etiqueta:
        ``family`` (benign, dos, probe, bruteforce, exfil).
    """
    if counts is None:
        counts = {"benign": 1200, "dos": 250, "probe": 250, "bruteforce": 200, "exfil": 200}
    rng = np.random.default_rng(seed)
    frames = []
    for family, n in counts.items():
        if n <= 0:
            continue
        if family not in ATTACK_PROFILES:
            raise ValueError(f"Familia desconocida: {family!r}")
        p = ATTACK_PROFILES[family]
        duration = np.clip(rng.normal(*p["duration"], size=n), 0.001, 120.0)
        src_bytes = np.clip(rng.normal(*p["src_bytes"], size=n), 20.0, 200_000.0)
        dst_bytes = np.clip(rng.normal(*p["dst_bytes"], size=n), 0.0, 200_000.0)
        pkts = np.clip(np.round(rng.normal(*p["pkts"], size=n)), 1, 5000).astype(int)
        bytes_per_pkt = np.round((src_bytes + dst_bytes) / pkts, 2)
        failed_logins = rng.poisson(p["failed_logins"], size=n).astype(int)
        port_entropy = np.clip(rng.normal(*p["port_entropy"], size=n), 0.0, 8.0)
        syn_ratio = np.clip(rng.normal(*p["syn_ratio"], size=n), 0.0, 1.0)

        proto_vals, proto_probs = p["proto"]
        svc_vals, svc_probs = p["service"]
        flag_vals, flag_probs = p["flag"]

        frames.append(
            pd.DataFrame(
                {
                    "duration": np.round(duration, 4),
                    "src_bytes": np.round(src_bytes, 1),
                    "dst_bytes": np.round(dst_bytes, 1),
                    "pkts": pkts,
                    "bytes_per_pkt": bytes_per_pkt,
                    "failed_logins": failed_logins,
                    "port_entropy": np.round(port_entropy, 3),
                    "syn_ratio": np.round(syn_ratio, 3),
                    "proto": rng.choice(proto_vals, size=n, p=proto_probs),
                    "service": rng.choice(svc_vals, size=n, p=svc_probs),
                    "flag": rng.choice(flag_vals, size=n, p=flag_probs),
                    "family": family,
                }
            )
        )

    if not frames:
        raise ValueError("counts debe incluir al menos una familia con n > 0")
    df = pd.concat(frames, ignore_index=True)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
