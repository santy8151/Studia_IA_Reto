"""Mixture-of-Experts para clasificar la familia de ataque (top-k gating + balanceo de carga)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

FAMILIES = ["benign", "dos", "probe", "bruteforce", "exfil"]


class MoEClassifier(nn.Module):
    def __init__(self, d_in, n_classes=len(FAMILIES), n_experts=4, top_k=2, hidden=64, aux_weight=0.01):
        super().__init__()
        self.n_experts, self.top_k, self.aux_weight = n_experts, top_k, aux_weight
        self.gate = nn.Linear(d_in, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)) for _ in range(n_experts)
        )

    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)                    # B,E
        topv, topi = probs.topk(self.top_k, dim=-1)                 # B,k
        topv = topv / topv.sum(-1, keepdim=True)
        outs = torch.stack([e(x) for e in self.experts], dim=1)     # B,E,C (denso: suficiente a esta escala)
        sel = torch.gather(outs, 1, topi.unsqueeze(-1).expand(-1, -1, outs.size(-1)))
        logits = (topv.unsqueeze(-1) * sel).sum(1)
        importance = probs.mean(0)
        load = F.one_hot(topi[:, 0], self.n_experts).float().mean(0)
        aux = self.n_experts * (importance * load).sum()            # pérdida de balanceo (Switch)
        return logits, aux, topi

    def fit(self, X, y, epochs=40, lr=1e-3, batch_size=256, verbose=True):
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.parameters(), lr)
        for ep in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                logits, aux, _ = self(X[idx])
                loss = F.cross_entropy(logits, y[idx]) + self.aux_weight * aux
                opt.zero_grad(); loss.backward(); opt.step()
            if verbose and ep % 10 == 0:
                print(f"[MoE] ep {ep} loss={loss.item():.4f}")
        return self

    @torch.no_grad()
    def predict(self, X):
        logits, _, topi = self(torch.tensor(X, dtype=torch.float32))
        p = F.softmax(logits, -1)
        return p.argmax(-1).numpy(), p.max(-1).values.numpy(), topi[:, 0].numpy()
