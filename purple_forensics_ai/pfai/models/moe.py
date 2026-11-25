"""Mixture-of-Experts para clasificar la familia de ataque (top-k gating + balanceo de carga)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

FAMILIES = ["benign", "dos", "probe", "bruteforce", "exfil"]


class MoEClassifier(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_classes: int = len(FAMILIES),
        n_experts: int = 4,
        top_k: int = 2,
        hidden: int = 64,
        aux_weight: float = 0.01,
        dropout: float = 0.0,
        device: str = "cpu",
        seed: int | None = 42,
    ):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k ({top_k}) debe estar entre 1 y n_experts ({n_experts})")
        if seed is not None:
            torch.manual_seed(seed)
        self.d_in = d_in
        self.n_classes = n_classes
        self.n_experts = n_experts
        self.top_k = top_k
        self.hidden = hidden
        self.aux_weight = aux_weight
        self.dropout = dropout
        self.device = device

        self.gate = nn.Linear(d_in, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
            for _ in range(n_experts)
        )
        self.history_: list[dict[str, float]] = []
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = F.softmax(self.gate(x), dim=-1)                       # B, E
        topv, topi = probs.topk(self.top_k, dim=-1)                   # B, k
        topv = topv / topv.sum(dim=-1, keepdim=True)
        outs = torch.stack([e(x) for e in self.experts], dim=1)       # B, E, C
        sel = torch.gather(outs, 1, topi.unsqueeze(-1).expand(-1, -1, outs.size(-1)))
        logits = (topv.unsqueeze(-1) * sel).sum(dim=1)
        importance = probs.mean(dim=0)
        load = F.one_hot(topi[:, 0], self.n_experts).float().mean(dim=0)
        aux = self.n_experts * (importance * load).sum()              # balanceo estilo Switch
        return logits, aux, topi

    def _compute_class_weights(self, y: np.ndarray, class_weights) -> torch.Tensor | None:
        if class_weights is None:
            return None
        if class_weights == "balanced":
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            w = counts.sum() / (self.n_classes * counts)
            return torch.tensor(w, dtype=torch.float32, device=self.device)
        w_arr = np.asarray(class_weights, dtype=np.float32)
        if w_arr.shape != (self.n_classes,):
            raise ValueError(f"class_weights debe tener longitud {self.n_classes}")
        return torch.tensor(w_arr, dtype=torch.float32, device=self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 256,
        class_weights=None,
        verbose: bool = True,
    ) -> "MoEClassifier":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.d_in:
            raise ValueError(f"Se esperaba X con forma (n, {self.d_in}), recibido {X_arr.shape}")
        if len(X_arr) != len(y_arr) or len(X_arr) == 0:
            raise ValueError("X e y deben ser no vacíos y tener igual número de filas")

        self.train()
        self.history_ = []
        weights = self._compute_class_weights(y_arr, class_weights)
        Xt = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y_arr, dtype=torch.long, device=self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for ep in range(epochs):
            perm = torch.randperm(len(Xt), device=self.device)
            losses = []
            for i in range(0, len(Xt), batch_size):
                idx = perm[i : i + batch_size]
                logits, aux, _ = self(Xt[idx])
                loss = F.cross_entropy(logits, yt[idx], weight=weights) + self.aux_weight * aux
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            mean_loss = float(np.mean(losses))
            self.history_.append({"loss": mean_loss})
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"[MoE] ep {ep:3d} loss={mean_loss:.4f}")
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
            if Xt.ndim != 2 or Xt.shape[1] != self.d_in:
                raise ValueError(f"Se esperaba X con forma (n, {self.d_in}), recibido {tuple(Xt.shape)}")
            logits, _, _ = self(Xt)
            return F.softmax(logits, dim=-1).cpu().numpy()
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        was_training = self.training
        self.eval()
        try:
            Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
            if Xt.ndim != 2 or Xt.shape[1] != self.d_in:
                raise ValueError(f"Se esperaba X con forma (n, {self.d_in}), recibido {tuple(Xt.shape)}")
            logits, _, topi = self(Xt)
            p = F.softmax(logits, dim=-1)
            return (
                p.argmax(dim=-1).cpu().numpy(),
                p.max(dim=-1).values.cpu().numpy(),
                topi[:, 0].cpu().numpy(),
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def expert_utilization(self, X: np.ndarray, y: np.ndarray | None = None) -> dict:
        """Reporta la carga de cada experto y, si se pasa ``y``, la especialización por familia."""
        _, _, primary = self.predict(X)
        counts = np.bincount(primary, minlength=self.n_experts).astype(float)
        report: dict = {
            "overall_load": [float(c / max(1, len(primary))) for c in counts],
        }
        if y is not None:
            y_arr = np.asarray(y, dtype=int)
            per_family = {}
            for cls_id in sorted(set(y_arr.tolist())):
                mask = y_arr == cls_id
                name = FAMILIES[cls_id] if 0 <= cls_id < len(FAMILIES) else str(cls_id)
                fam_counts = np.bincount(primary[mask], minlength=self.n_experts).astype(float)
                per_family[name] = [float(c / max(1, mask.sum())) for c in fam_counts]
            report["by_family"] = per_family
        return report

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_true = np.asarray(y, dtype=int)
        y_pred, conf, _ = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mean_confidence": float(np.mean(conf)),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "d_in": self.d_in,
                    "n_classes": self.n_classes,
                    "n_experts": self.n_experts,
                    "top_k": self.top_k,
                    "hidden": self.hidden,
                    "aux_weight": self.aux_weight,
                    "dropout": self.dropout,
                    "device": self.device,
                },
                "history": self.history_,
                "state_dict": self.state_dict(),
            },
            p,
        )
        return p

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "MoEClassifier":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
        cfg = dict(ckpt["config"])
        if device is not None:
            cfg["device"] = device
        model = cls(**cfg, seed=None)
        model.load_state_dict(ckpt["state_dict"])
        model.history_ = list(ckpt.get("history", []))
        model.eval()
        return model
