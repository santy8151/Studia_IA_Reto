"""
Latent-Variable GAN (estilo VAE-GAN) para detección de anomalías y generación.

  Encoder  E: x -> (mu, logvar), z = mu + sigma*eps
  Decoder  G: z -> x~   (también actúa como generador: zp ~ N(0,I) -> xp)
  Discriminador D: x -> (logit, features_l)

Pérdidas
  L_GAN       : adversarial; D separa x real de x~ (reconstrucción) y xp (prior)
  L_Dislike   : ||D_l(x) - D_l(x~)||^2  (reconstrucción en espacio de features de D)
  L_prior     : KL(q(z|x) || N(0,I))

Anomalía = error de reconstrucción (entrada + features de D). Se entrena SOLO con
datos benignos y se calibra el umbral con un percentil de validación benigna.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, d_in, d_z, h):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2))
        self.mu, self.logvar = nn.Linear(h, d_z), nn.Linear(h, d_z)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h).clamp(-8, 8)


class Discriminator(nn.Module):
    def __init__(self, d_in, h):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(d_in, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2))
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        f = self.feat(x)
        return self.head(f), f


@dataclass
class LVGANConfig:
    d_in: int
    d_z: int = 16
    hidden: int = 128
    lr: float = 2e-4
    beta: float = 1.0       # peso L_prior
    gamma: float = 1.0      # peso L_Dislike
    lam_feat: float = 1.0   # peso de features de D en el score
    epochs: int = 60
    batch_size: int = 256
    device: str = "cpu"


class LVGAN(nn.Module):
    def __init__(self, cfg: LVGANConfig):
        super().__init__()
        self.cfg = cfg
        self.E = Encoder(cfg.d_in, cfg.d_z, cfg.hidden)
        self.G = mlp(cfg.d_z, cfg.hidden, cfg.d_in)
        self.D = Discriminator(cfg.d_in, cfg.hidden)
        self.optD = torch.optim.Adam(self.D.parameters(), cfg.lr, betas=(0.5, 0.999))
        self.optEG = torch.optim.Adam(list(self.E.parameters()) + list(self.G.parameters()), cfg.lr, betas=(0.5, 0.999))
        self.threshold = None
        self.to(cfg.device)

    def _encode(self, x):
        mu, logvar = self.E(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar

    def train_step(self, x):
        bce = F.binary_cross_entropy_with_logits
        ones, zeros = torch.ones(len(x), 1, device=x.device), torch.zeros(len(x), 1, device=x.device)
        z, mu, logvar = self._encode(x)
        x_rec = self.G(z)
        x_p = self.G(torch.randn_like(z))

        # 1) Discriminador: real=1, reconstruido=0, muestra del prior=0
        d_real, _ = self.D(x)
        d_rec, _ = self.D(x_rec.detach())
        d_p, _ = self.D(x_p.detach())
        loss_D = bce(d_real, ones) + bce(d_rec, zeros) + bce(d_p, zeros)
        self.optD.zero_grad()
        loss_D.backward()
        self.optD.step()

        # 2) Encoder + Decoder/Generador
        _, f_real = self.D(x)
        d_rec, f_rec = self.D(x_rec)
        d_p, _ = self.D(x_p)
        l_dislike = F.mse_loss(f_rec, f_real.detach())
        l_prior = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        l_gan_g = bce(d_rec, ones) + bce(d_p, ones)
        loss_EG = self.cfg.beta * l_prior + self.cfg.gamma * l_dislike + l_gan_g + F.mse_loss(x_rec, x)
        self.optEG.zero_grad()
        loss_EG.backward()
        self.optEG.step()
        return dict(loss_D=loss_D.item(), loss_EG=loss_EG.item(), prior=l_prior.item(), dislike=l_dislike.item())

    def fit(self, X_benign: np.ndarray, verbose=True):
        X = torch.tensor(X_benign, dtype=torch.float32, device=self.cfg.device)
        n = len(X)
        for ep in range(self.cfg.epochs):
            perm = torch.randperm(n, device=X.device)
            for i in range(0, n, self.cfg.batch_size):
                batch = X[perm[i:i + self.cfg.batch_size]]
                if len(batch) < 2:
                    continue
                stats = self.train_step(batch)
            if verbose and (ep % 10 == 0 or ep == self.cfg.epochs - 1):
                print(f"[LVGAN] ep {ep:3d} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        x = torch.tensor(X, dtype=torch.float32, device=self.cfg.device)
        mu, _ = self.E(x)
        x_rec = self.G(mu)
        rec = (x - x_rec).pow(2).mean(1)
        _, f = self.D(x)
        _, f_r = self.D(x_rec)
        feat = (f - f_r).pow(2).mean(1)
        return (rec + self.cfg.lam_feat * feat).cpu().numpy()

    def calibrate(self, X_val_benign: np.ndarray, q: float = 0.99):
        self.threshold = float(np.quantile(self.score(X_val_benign), q))
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.threshold is not None, "llama calibrate() primero"
        return (self.score(X) > self.threshold).astype(int)

    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray:
        """Variantes sintéticas (xp), zp ~ N(0,I), para probar las defensas (lado red team)."""
        zp = torch.randn(n, self.cfg.d_z, device=self.cfg.device)
        return self.G(zp).cpu().numpy()
