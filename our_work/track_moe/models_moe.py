"""Mixture-of-Experts classifier from MoE-FL_Gateway_Design_Document.docx, applied to WUSTL.

Copied unchanged from experiments/common.py: embedding 36->64->64 (LayerNorm, ReLU), gate 64->32->K,
K=4 experts (64->64->2), top-2 routing, gate noise 0.1 in training, Switch-style load-balancing term
(weight 0.01 where used).  All experts run on the full batch and are masked by the gate weights --
required so Opacus per-sample-gradient hooks see every layer on the whole batch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    def __init__(self, d=36, c=2, K=4, h=64, topk=2, gate_noise=0.1):
        super().__init__()
        self.K, self.topk, self.gate_noise = K, topk, gate_noise
        self.emb = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(),
                                 nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, K))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, c)) for _ in range(K)])
        self.aux = torch.zeros(())

    def route(self, z):
        logits = self.gate(z)
        if self.training and self.gate_noise > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(self.topk, dim=-1).indices
        mask = torch.zeros_like(probs).scatter(-1, top, 1.0)
        w = probs * mask
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        return probs, mask, w

    def forward(self, x):
        z = self.emb(x)
        probs, mask, w = self.route(z)
        outs = torch.stack([e(z) for e in self.experts], dim=1)          # (B, K, C)
        out = (outs * w.unsqueeze(-1)).sum(1)
        self.aux = self.K * (mask.mean(0) / self.topk * probs.mean(0)).sum()
        return out

    @torch.no_grad()
    def routing_stats(self, X):
        """(mean gate probabilities, share of top-k assignments per expert)."""
        self.eval()
        z = self.emb(X)
        probs, mask, _ = self.route(z)
        return probs.mean(0).numpy(), (mask.mean(0) / self.topk).numpy()

    @torch.no_grad()
    def routing_entropy(self, X):
        """Mean per-sample gate entropy, normalised to [0,1] (1 = uniform routing, 0 = one expert)."""
        self.eval()
        probs, _, _ = self.route(self.emb(X))
        h = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        return float(h / np.log(self.K))
