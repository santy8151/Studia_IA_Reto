"""Síntesis tabular: CTGAN y Gaussian Copula (SDV) + DP-GAN con DP-SGD (Opacus)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PrivacyBudget:
    """Registro formal de privacidad diferencial (ε, δ) por entrenamiento."""
    epsilon_target: float
    epsilon_spent: float
    delta: float
    max_grad_norm: float
    noise_multiplier: float | None
    epochs: int
    batch_size: int
    n_records: int
    mechanism: str = "DP-SGD (Opacus)"

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_metadata(df: pd.DataFrame, table_name: str = "events"):
    from sdv.metadata import Metadata

    if hasattr(Metadata, "detect_from_dataframes"):
        return Metadata.detect_from_dataframes(data={table_name: df}, verbose=False), True
    if hasattr(Metadata, "detect_from_dataframe"):
        return Metadata.detect_from_dataframe(df, table_name=table_name), False
    from sdv.metadata import SingleTableMetadata

    m = SingleTableMetadata()
    m.detect_from_dataframe(df)
    return m, False


class _SDVSynthesizerAdapter:
    """Normaliza `.sample()` para devolver siempre un ``pd.DataFrame`` en SDV 1.x y 2.x."""

    def __init__(self, synthesizer, table_name: str = "events", is_v2: bool = False):
        self.synthesizer = synthesizer
        self.table_name = table_name
        self.is_v2 = is_v2

    def sample(self, num_rows: int) -> pd.DataFrame:
        out = self.synthesizer.sample(num_rows=num_rows)
        if isinstance(out, dict):
            return out[self.table_name]
        return out

    def __getattr__(self, name: str):
        return getattr(self.synthesizer, name)


def fit_ctgan(df: pd.DataFrame, epochs: int = 300, **kw):
    from sdv.single_table import CTGANSynthesizer

    table_name = "events"
    meta, is_v2 = _detect_metadata(df, table_name=table_name)
    s = CTGANSynthesizer(meta, epochs=epochs, **kw)
    s.fit({table_name: df} if is_v2 else df)
    return _SDVSynthesizerAdapter(s, table_name=table_name, is_v2=is_v2)


def fit_gaussian_copula(df: pd.DataFrame):
    from sdv.single_table import GaussianCopulaSynthesizer

    table_name = "events"
    meta, is_v2 = _detect_metadata(df, table_name=table_name)
    s = GaussianCopulaSynthesizer(meta)
    s.fit({table_name: df} if is_v2 else df)
    return _SDVSynthesizerAdapter(s, table_name=table_name, is_v2=is_v2)


class DPTabGAN:
    """
    GAN tabular con DP-SGD (Opacus) aplicado al discriminador, el único componente que
    ve registros reales; el generador solo recibe gradientes ya privatizados, por lo
    que hereda la garantía (eps, delta) por post-procesamiento.
    Opera sobre la matriz codificada (TabularEncoder).
    NOTA: CTGAN/SDV NO son diferencialmente privados por sí solos; para DP real usa esta clase.
    """

    def __init__(
        self,
        d_in: int,
        d_z: int = 16,
        hidden: int = 128,
        epsilon: float = 5.0,
        delta: float = 1e-5,
        epochs: int = 30,
        batch_size: int = 128,
        max_grad_norm: float = 1.0,
        lr: float = 1e-3,
        device: str = "cpu",
        encoder=None,
        seed: int | None = 42,
    ):
        if epsilon <= 0 or not 0 < delta < 1:
            raise ValueError("epsilon debe ser > 0 y delta en (0, 1)")
        self.d_in, self.d_z, self.hidden = d_in, d_z, hidden
        self.epsilon, self.delta, self.epochs = float(epsilon), float(delta), int(epochs)
        self.batch_size, self.max_grad_norm = int(batch_size), float(max_grad_norm)
        self.lr, self.device = float(lr), device
        self.encoder = encoder
        self.seed = seed
        self.G = None
        self.epsilon_spent: float | None = None
        self.budget_: PrivacyBudget | None = None

    def fit(self, X: np.ndarray, verbose: bool = True) -> "DPTabGAN":
        import torch
        import torch.nn.functional as F
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        from torch.utils.data import DataLoader, TensorDataset
        from pfai.models.lvgan import mlp

        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.d_in or len(arr) < 2:
            raise ValueError(f"Se esperaba X con forma (>=2, {self.d_in}), recibido {arr.shape}")

        if self.seed is not None:
            torch.manual_seed(self.seed)

        eff_bs = min(self.batch_size, len(arr))
        loader = DataLoader(TensorDataset(torch.tensor(arr, dtype=torch.float32)), batch_size=eff_bs, shuffle=True)
        self.G = mlp(self.d_z, self.hidden, self.d_in).to(self.device)
        D = mlp(self.d_in, self.hidden, 1).to(self.device)  # sin BatchNorm (incompatible con DP-SGD)
        if not ModuleValidator.is_valid(D):
            D = ModuleValidator.fix(D).to(self.device)

        optG = torch.optim.Adam(self.G.parameters(), self.lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), self.lr, betas=(0.5, 0.999))

        pe = PrivacyEngine()
        D, optD, loader = pe.make_private_with_epsilon(
            module=D,
            optimizer=optD,
            data_loader=loader,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
        )

        self.G.train()
        D.train()
        for _ in range(self.epochs):
            for (real,) in loader:
                b = len(real)
                if b == 0:
                    continue
                real = real.to(self.device)
                fake = self.G(torch.randn(b, self.d_z, device=self.device)).detach()
                inp = torch.cat([real, fake], dim=0)
                y = torch.cat([torch.ones(b, 1), torch.zeros(b, 1)], dim=0).to(self.device)

                optD.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(D(inp), y).backward()
                optD.step()

                # El paso del generador no debe registrar gradientes por muestra en D
                D.disable_hooks()
                optG.zero_grad(set_to_none=True)
                lg = F.binary_cross_entropy_with_logits(
                    D(self.G(torch.randn(b, self.d_z, device=self.device))),
                    torch.ones(b, 1, device=self.device),
                )
                lg.backward()
                optG.step()
                D.enable_hooks()

        self.epsilon_spent = float(pe.get_epsilon(self.delta))
        noise_mult = getattr(optD, "noise_multiplier", None)
        self.budget_ = PrivacyBudget(
            epsilon_target=self.epsilon,
            epsilon_spent=self.epsilon_spent,
            delta=self.delta,
            max_grad_norm=self.max_grad_norm,
            noise_multiplier=float(noise_mult) if noise_mult is not None else None,
            epochs=self.epochs,
            batch_size=eff_bs,
            n_records=len(arr),
        )
        if verbose:
            print(f"[DP] epsilon gastado = {self.epsilon_spent:.2f} (objetivo={self.epsilon}, delta={self.delta})")
        return self

    def sample(self, n: int) -> np.ndarray:
        if self.G is None:
            raise RuntimeError("DPTabGAN no está entrenado; llama a fit() primero")
        if n <= 0:
            raise ValueError("n debe ser positivo")
        import torch
        was_training = self.G.training
        self.G.eval()
        try:
            with torch.no_grad():
                return self.G(torch.randn(n, self.d_z, device=self.device)).cpu().numpy()
        finally:
            self.G.train(was_training)

    def sample_df(self, n: int, encoder=None) -> pd.DataFrame:
        """Genera ``n`` muestras sintéticas y las decodifica a DataFrame usando ``TabularEncoder``."""
        enc = encoder or self.encoder
        if enc is None:
            raise ValueError("Se requiere un TabularEncoder ajustado para reconstruir el DataFrame")
        return enc.inverse_transform(self.sample(n))

    def privacy_report(self) -> dict:
        if self.budget_ is None:
            raise RuntimeError("DPTabGAN no está entrenado; aún no hay presupuesto registrado")
        return self.budget_.to_dict()


def benchmark_synthesizers(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    label: str | None = None,
    n_synth: int | None = None,
    methods: tuple[str, ...] = ("gaussian_copula", "ctgan", "dp_tabgan"),
    ctgan_epochs: int = 50,
    dp_epochs: int = 20,
    dp_epsilon: float = 5.0,
    dp_delta: float = 1e-5,
    encoder=None,
) -> dict[str, dict]:
    """Entrena y audita múltiples generadores tabulares (SDV y DP-GAN) sobre el mismo split."""
    from pfai.preprocess import TabularEncoder
    from pfai.privacy.audit import full_report, privacy_utility_verdict

    n_gen = n_synth or len(train_df)
    results: dict[str, dict] = {}

    for method in methods:
        if method == "gaussian_copula":
            synth_model = fit_gaussian_copula(train_df)
            synth_df = synth_model.sample(num_rows=n_gen)
            budget = None
        elif method == "ctgan":
            synth_model = fit_ctgan(train_df, epochs=ctgan_epochs)
            synth_df = synth_model.sample(num_rows=n_gen)
            budget = None
        elif method == "dp_tabgan":
            enc = encoder or TabularEncoder.from_dataframe(train_df).fit(train_df)
            X_enc = enc.transform(train_df)
            dp = DPTabGAN(
                d_in=X_enc.shape[1],
                epsilon=dp_epsilon,
                delta=dp_delta,
                epochs=dp_epochs,
                encoder=enc,
            ).fit(X_enc, verbose=False)
            synth_df = dp.sample_df(n_gen)
            budget = dp.privacy_report()
        else:
            raise ValueError(f"Método de síntesis desconocido: {method!r}")

        rep = full_report(train_df, holdout_df, synth_df, label=label)
        rep["privacy_budget"] = budget
        rep["verdict"] = privacy_utility_verdict(rep, budget)
        results[method] = rep

    return results
