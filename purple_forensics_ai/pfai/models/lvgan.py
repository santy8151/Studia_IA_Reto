"""
Latent-Variable GAN (estilo VAE-GAN) para detección de anomalías y simulación Purple Team.

  Encoder       E: x -> (mu, logvar), z = mu + sigma * eps
  Decoder/Gen   G: z -> x~   (y generador: zp ~ N(0, I) -> xp)
  Discriminador D: x -> (logit, features_l)

Pérdidas
  L_GAN       : adversarial; D separa x real de x~ (reconstrucción) y xp (prior)
  L_Dislike   : ||D_l(x) - D_l(x~)||^2 + ||x - x~||^2 (reconstrucción en espacio de features y entrada)
  L_prior     : KL(q(z|x) || N(0, I))

Anomalía = error de reconstrucción (entrada + features de D). Se entrena SOLO con
datos benignos y se calibra el umbral con un percentil de validación benigna.
Además incluye generación de variantes de evasión en espacio latente para Red/Purple Team.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i: int, h: int, o: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, d_in: int, d_z: int, h: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, h),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(h, d_z)
        self.logvar = nn.Linear(h, d_z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)


class Discriminator(nn.Module):
    def __init__(self, d_in: int, h: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(d_in, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, h),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.feat(x)
        return self.head(f), f


@dataclass
class LVGANConfig:
    d_in: int
    d_z: int = 16
    hidden: int = 128
    lr: float = 5e-4
    beta: float = 0.1        # peso L_prior (KL)
    gamma: float = 1.0       # peso L_Dislike (features de D)
    rec_weight: float = 1.0  # peso reconstrucción en espacio de entrada
    gan_weight: float = 0.5  # peso término adversarial de G
    lam_feat: float = 1.0    # peso de features de D en el anomaly score
    grad_clip: float = 5.0
    epochs: int = 60
    batch_size: int = 128
    device: str = "cpu"
    seed: int | None = 42


class LVGAN(nn.Module):
    def __init__(self, cfg: LVGANConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        self.E = Encoder(cfg.d_in, cfg.d_z, cfg.hidden)
        self.G = mlp(cfg.d_z, cfg.hidden, cfg.d_in)
        self.D = Discriminator(cfg.d_in, cfg.hidden)
        self.optD = torch.optim.Adam(self.D.parameters(), cfg.lr, betas=(0.5, 0.999))
        self.optEG = torch.optim.Adam(
            list(self.E.parameters()) + list(self.G.parameters()),
            cfg.lr,
            betas=(0.5, 0.999),
        )
        self.threshold: float | None = None
        self.history_: list[dict[str, float]] = []
        self.to(cfg.device)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.E(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar

    def train_step(self, x: torch.Tensor) -> dict[str, float]:
        bce = F.binary_cross_entropy_with_logits
        ones = torch.ones(len(x), 1, device=x.device)
        zeros = torch.zeros(len(x), 1, device=x.device)

        z, mu, logvar = self._encode(x)
        x_rec = self.G(z)
        x_p = self.G(torch.randn_like(z))

        # 1) Discriminador: real=1, reconstruido=0, muestra del prior=0
        d_real, _ = self.D(x)
        d_rec, _ = self.D(x_rec.detach())
        d_p, _ = self.D(x_p.detach())
        loss_D = bce(d_real, ones) + 0.5 * (bce(d_rec, zeros) + bce(d_p, zeros))
        self.optD.zero_grad(set_to_none=True)
        loss_D.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.D.parameters(), self.cfg.grad_clip)
        self.optD.step()

        # 2) Encoder + Decoder/Generador (reutilizando D actualizado)
        with torch.no_grad():
            _, f_real = self.D(x)
        d_rec_g, f_rec = self.D(x_rec)
        d_p_g, _ = self.D(x_p)
        l_dislike = F.mse_loss(f_rec, f_real)
        l_rec = F.mse_loss(x_rec, x)
        l_prior = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        l_gan_g = 0.5 * (bce(d_rec_g, ones) + bce(d_p_g, ones))
        loss_EG = (
            self.cfg.beta * l_prior
            + self.cfg.gamma * l_dislike
            + self.cfg.rec_weight * l_rec
            + self.cfg.gan_weight * l_gan_g
        )
        self.optEG.zero_grad(set_to_none=True)
        loss_EG.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.E.parameters()) + list(self.G.parameters()),
                self.cfg.grad_clip,
            )
        self.optEG.step()

        return {
            "loss_D": float(loss_D.item()),
            "loss_EG": float(loss_EG.item()),
            "rec": float(l_rec.item()),
            "prior": float(l_prior.item()),
            "dislike": float(l_dislike.item()),
        }

    def fit(self, X_benign: np.ndarray, verbose: bool = True) -> "LVGAN":
        arr = np.asarray(X_benign, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.cfg.d_in or len(arr) < 2:
            raise ValueError(f"Se esperaba X_benign con forma (>=2, {self.cfg.d_in}), recibido {arr.shape}")

        self.train()
        self.history_ = []
        X = torch.tensor(arr, dtype=torch.float32, device=self.cfg.device)
        n = len(X)

        for ep in range(self.cfg.epochs):
            perm = torch.randperm(n, device=X.device)
            ep_stats: dict[str, list[float]] = {k: [] for k in ("loss_D", "loss_EG", "rec", "prior", "dislike")}
            for i in range(0, n, self.cfg.batch_size):
                batch = X[perm[i : i + self.cfg.batch_size]]
                if len(batch) < 2:
                    continue
                stats = self.train_step(batch)
                for k, v in stats.items():
                    ep_stats[k].append(v)
            summary = {k: float(np.mean(v)) for k, v in ep_stats.items() if v}
            self.history_.append(summary)
            if verbose and summary and (ep % 10 == 0 or ep == self.cfg.epochs - 1):
                print(f"[LVGAN] ep {ep:3d} " + " ".join(f"{k}={v:.4f}" for k, v in summary.items()))
        return self

    def _score_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, _ = self.E(x)
        x_rec = self.G(mu)
        rec = (x - x_rec).pow(2).mean(dim=1)
        _, f = self.D(x)
        _, f_r = self.D(x_rec)
        feat = (f - f_r).pow(2).mean(dim=1)
        total = rec + self.cfg.lam_feat * feat
        return total, rec, feat

    @torch.no_grad()
    def score_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Devuelve el score total y sus componentes (error de entrada y features de D)."""
        was_training = self.training
        self.eval()
        try:
            x = torch.tensor(np.asarray(X, dtype=np.float32), device=self.cfg.device)
            if x.ndim != 2 or x.shape[1] != self.cfg.d_in:
                raise ValueError(f"Se esperaba X con forma (n, {self.cfg.d_in}), recibido {tuple(x.shape)}")
            total, rec, feat = self._score_tensor(x)
            return {
                "total": total.cpu().numpy(),
                "rec_error": rec.cpu().numpy(),
                "feat_error": feat.cpu().numpy(),
            }
        finally:
            self.train(was_training)

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        return self.score_components(X)["total"]

    def calibrate(self, X_val_benign: np.ndarray, q: float = 0.99) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError("El percentil q debe estar en (0, 1)")
        self.threshold = float(np.quantile(self.score(X_val_benign), q))
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Umbral no calibrado; llama a calibrate() primero")
        return (self.score(X) > self.threshold).astype(int)

    @torch.no_grad()
    def sample(self, n: int, z: np.ndarray | None = None) -> np.ndarray:
        """Genera muestras sintéticas ``xp`` desde el prior ``zp ~ N(0, I)``."""
        if n <= 0 and z is None:
            raise ValueError("n debe ser positivo")
        was_training = self.training
        self.eval()
        try:
            if z is None:
                zp = torch.randn(n, self.cfg.d_z, device=self.cfg.device)
            else:
                zp = torch.tensor(np.asarray(z, dtype=np.float32), device=self.cfg.device)
            return self.G(zp).cpu().numpy()
        finally:
            self.train(was_training)

    def generate_evasion_variants(
        self,
        X_seed: np.ndarray,
        steps: int = 25,
        step_size: float = 0.08,
        l2_reg: float = 0.05,
        target_score: float | None = None,
    ) -> np.ndarray:
        """Simulación Purple Team: optimiza códigos latentes cerca de ``X_seed`` para producir
        variantes adversarias ``xp = G(z)`` que intenten reducir su score de anomalía
        sin perder proximidad latente al ataque original.
        """
        was_training = self.training
        self.eval()
        try:
            x0 = torch.tensor(np.asarray(X_seed, dtype=np.float32), device=self.cfg.device)
            if x0.ndim != 2 or x0.shape[1] != self.cfg.d_in:
                raise ValueError(f"Se esperaba X_seed con forma (n, {self.cfg.d_in}), recibido {tuple(x0.shape)}")
            with torch.no_grad():
                z_init, _ = self.E(x0)
            z = z_init.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=step_size)
            target = (
                self.threshold * 0.9 if target_score is None and self.threshold is not None else target_score
            )
            params = list(self.G.parameters()) + list(self.D.parameters())
            prev_req = [p.requires_grad for p in params]
            for p in params:
                p.requires_grad_(False)
            try:
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    xp = self.G(z)
                    sc, _, _ = self._score_tensor(xp)
                    if target is not None:
                        sc_loss = F.relu(sc - target).mean()
                    else:
                        sc_loss = sc.mean()
                    prox_loss = F.mse_loss(z, z_init)
                    loss = sc_loss + l2_reg * prox_loss
                    loss.backward()
                    opt.step()
            finally:
                for p, req in zip(params, prev_req):
                    p.requires_grad_(req)
            with torch.no_grad():
                return self.G(z).cpu().numpy()
        finally:
            self.train(was_training)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "threshold": self.threshold,
                "history": self.history_,
                "state_dict": self.state_dict(),
            },
            p,
        )
        return p

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "LVGAN":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
        cfg_dict = dict(ckpt["config"])
        if device is not None:
            cfg_dict["device"] = device
        model = cls(LVGANConfig(**cfg_dict))
        model.load_state_dict(ckpt["state_dict"])
        model.threshold = ckpt.get("threshold")
        model.history_ = list(ckpt.get("history", []))
        model.eval()
        return model
