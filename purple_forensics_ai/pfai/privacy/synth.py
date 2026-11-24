"""Síntesis tabular: CTGAN y Gaussian Copula (SDV) + DP-GAN (Opacus)."""
import numpy as np
import pandas as pd


def _metadata(df: pd.DataFrame):
    try:
        from sdv.metadata import Metadata
        return Metadata.detect_from_dataframe(df, table_name="events")
    except Exception:
        from sdv.metadata import SingleTableMetadata
        m = SingleTableMetadata()
        m.detect_from_dataframe(df)
        return m


def fit_ctgan(df: pd.DataFrame, epochs=300, **kw):
    from sdv.single_table import CTGANSynthesizer
    s = CTGANSynthesizer(_metadata(df), epochs=epochs, **kw)
    s.fit(df)
    return s


def fit_gaussian_copula(df: pd.DataFrame):
    from sdv.single_table import GaussianCopulaSynthesizer
    s = GaussianCopulaSynthesizer(_metadata(df))
    s.fit(df)
    return s


class DPTabGAN:
    """
    GAN tabular con DP-SGD (Opacus) aplicado al discriminador, el único componente que
    ve registros reales; el generador solo recibe gradientes ya privatizados, por lo
    que hereda la garantía (eps, delta) por post-procesamiento.
    Opera sobre la matriz codificada (TabularEncoder).
    NOTA: CTGAN/SDV NO son diferencialmente privados por sí solos; para DP real usa esta clase.
    """

    def __init__(self, d_in, d_z=32, hidden=128, epsilon=5.0, delta=1e-5, epochs=30,
                 batch_size=256, max_grad_norm=1.0, lr=2e-4, device="cpu"):
        self.d_in, self.d_z, self.hidden = d_in, d_z, hidden
        self.epsilon, self.delta, self.epochs = epsilon, delta, epochs
        self.batch_size, self.max_grad_norm, self.lr, self.device = batch_size, max_grad_norm, lr, device
        self.G = None
        self.epsilon_spent = None

    def fit(self, X: np.ndarray):
        import torch
        import torch.nn.functional as F
        from opacus import PrivacyEngine
        from torch.utils.data import DataLoader, TensorDataset
        from pfai.models.lvgan import mlp

        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=self.batch_size)
        self.G = mlp(self.d_z, self.hidden, self.d_in).to(self.device)
        D = mlp(self.d_in, self.hidden, 1).to(self.device)      # sin BatchNorm (incompatible con DP-SGD)
        optG = torch.optim.Adam(self.G.parameters(), self.lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), self.lr, betas=(0.5, 0.999))
        pe = PrivacyEngine()
        D, optD, loader = pe.make_private_with_epsilon(
            module=D, optimizer=optD, data_loader=loader, target_epsilon=self.epsilon,
            target_delta=self.delta, epochs=self.epochs, max_grad_norm=self.max_grad_norm)
        for _ in range(self.epochs):
            for (real,) in loader:
                b = len(real)
                if b == 0:
                    continue
                real = real.to(self.device)
                fake = self.G(torch.randn(b, self.d_z, device=self.device)).detach()
                inp = torch.cat([real, fake])
                y = torch.cat([torch.ones(b, 1), torch.zeros(b, 1)]).to(self.device)
                optD.zero_grad()
                F.binary_cross_entropy_with_logits(D(inp), y).backward()
                optD.step()
                D.disable_hooks()                                  # el paso de G no toca datos reales
                optG.zero_grad()
                lg = F.binary_cross_entropy_with_logits(
                    D(self.G(torch.randn(b, self.d_z, device=self.device))), torch.ones(b, 1, device=self.device))
                lg.backward()
                optG.step()
                D.enable_hooks()
        self.epsilon_spent = pe.get_epsilon(self.delta)
        print(f"[DP] epsilon gastado = {self.epsilon_spent:.2f} (delta={self.delta})")
        return self

    def sample(self, n):
        import torch
        with torch.no_grad():
            return self.G(torch.randn(n, self.d_z, device=self.device)).cpu().numpy()
