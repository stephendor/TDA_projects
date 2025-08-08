"""PersLay-like learnable embedding layer for persistence diagrams.

Topology-only: operates directly on (birth, death) pairs (optionally birth, lifetime) with
Gaussian kernel dictionary whose centers and bandwidths are learned.
No statistical proxy features (mean/std etc.) are introduced.

References (conceptual): Carrière et al., PersLay (ICLR 2020) – simplified variant here.

Design choices (validation-first, deterministic init):
 - Kernel centers initialized via k-means++ over sampled training points (external caller supplies points)
 - Log-sigmas parameterization ensures positivity of bandwidths
 - Lifetime weighting (l^gamma) emphasizes persistent features; gamma configurable (fixed, not learned by default)
 - Separate embedding per homology dimension concatenated
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PersLayConfig:
    num_kernels: int = 32
    homology_dims: Sequence[int] = (0, 1)
    persistence_power: float = 1.0
    use_lifetime_coordinates: bool = True
    device: str = 'cpu'
    normalize_input: bool = False
    average_pool: bool = True  # new: divide kernel responses by number of points
    l2_normalize: bool = True  # new: optional L2 norm of concatenated embedding
    sigma_min: float = 1e-3    # new: clamp for numerical stability
    sigma_max: float = 5.0
    hidden_dim: int = 0          # new: optional hidden layer size
    dropout: float = 0.0          # new: dropout probability if hidden layer used


class PersLayLayer(nn.Module):
    def __init__(self, cfg: PersLayConfig, init_centers: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        K = cfg.num_kernels
        if init_centers is None:
            # fallback deterministic tiny gaussian grid
            grid_b = torch.linspace(0.0, 1.0, int(math.sqrt(K)))
            grid_l = torch.linspace(0.0, 1.0, int(math.sqrt(K)))
            mesh = torch.cartesian_prod(grid_b, grid_l)
            if mesh.shape[0] > K:
                mesh = mesh[:K]
            elif mesh.shape[0] < K:
                pad = K - mesh.shape[0]
                mesh = torch.cat([mesh, mesh[:pad]], dim=0)
            init_centers = mesh
        else:
            if init_centers.shape[0] != K:
                raise ValueError(f"init_centers must have shape (K,2) with K={K}")
        if init_centers.shape[1] != 2:
            raise ValueError("init_centers must have 2 columns (birth, lifetime or birth, death)")
        self.centers = nn.Parameter(init_centers.to(cfg.device).clone())
        # log sigma to ensure positivity; initialize to small value relative to spread
        self.log_sigmas = nn.Parameter(torch.full((K,), math.log(0.1), device=cfg.device))
        # initialize alphas with negative bias to damp early magnitude
        self.alphas = nn.Parameter(torch.full((K,), -2.0, device=cfg.device))  # start neutral; will learn weights
        # Optional per-dimension scale & shift for normalization
        if cfg.normalize_input:
            self.register_parameter('shift', nn.Parameter(torch.zeros(2, device=cfg.device)))
            self.register_parameter('scale', nn.Parameter(torch.ones(2, device=cfg.device)))
        else:
            self.shift = None
            self.scale = None

    def _sanitize(self, pts: torch.Tensor) -> torch.Tensor:
        if pts.numel() == 0:
            return pts
        mask = torch.isfinite(pts).all(dim=1)
        if not mask.all():
            pts = pts[mask]
        return pts

    def _transform_points(self, pts: torch.Tensor) -> torch.Tensor:
        """Transform raw (birth, death) to (birth, lifetime) if configured; expects shape (n,2)."""
        if pts.numel() == 0:
            return pts
        if self.cfg.use_lifetime_coordinates:
            b = pts[:, 0]
            l = torch.clamp(pts[:, 1] - pts[:, 0], min=0.0)
            pts = torch.stack([b, l], dim=1)
        if self.cfg.normalize_input and self.shift is not None and self.scale is not None:
            pts = (pts + self.shift) * self.scale
        return pts

    def _diagram_embedding(self, pts: torch.Tensor) -> torch.Tensor:
        """Compute embedding for a single diagram (points tensor shape (n,2))."""
        if pts.numel() == 0:
            # Return zero vector for empty diagram (stable neutral element under sum pooling)
            return torch.zeros(self.cfg.num_kernels, device=self.centers.device)
        pts = self._sanitize(pts)
        if pts.numel() == 0:
            return torch.zeros(self.cfg.num_kernels, device=self.centers.device)
        pts = self._transform_points(pts)
        pts = self._sanitize(pts)
        centers = self.centers
        sigmas = torch.exp(self.log_sigmas).clamp(self.cfg.sigma_min, self.cfg.sigma_max)
        # Compute squared distances: (n,K)
        x2 = (pts**2).sum(dim=1, keepdim=True)  # (n,1)
        c2 = (centers**2).sum(dim=1).unsqueeze(0)  # (1,K)
        xc = pts @ centers.T  # (n,K)
        d2 = torch.clamp(x2 + c2 - 2 * xc, min=0.0)
        # Clamp extremely large distances to avoid exp underflow -> keep numeric stability
        d2 = torch.clamp(d2, max=1e6)
        gauss = torch.exp(-d2 / (2.0 * (sigmas**2)))  # (n,K)
        if not torch.isfinite(gauss).all():
            gauss = torch.nan_to_num(gauss, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.persistence_power != 0.0:
            lifetime = torch.clamp(pts[:, 1], min=0.0)
            w = lifetime ** self.cfg.persistence_power  # (n,)
            gauss = gauss * w.unsqueeze(1)
        emb = gauss.sum(dim=0)  # (K,)
        if self.cfg.average_pool and pts.shape[0] > 0:
            emb = emb / max(1.0, float(pts.shape[0]))
        emb = emb * torch.exp(self.alphas)
        if not torch.isfinite(emb).all():
            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        return emb

    def forward(self, diagrams_per_dim: List[torch.Tensor]) -> torch.Tensor:
        """diagrams_per_dim: list of tensors (n_i,2) for each homology dimension in cfg.homology_dims order."""
        embeddings = []
        for d in self.cfg.homology_dims:
            if d < len(diagrams_per_dim) and diagrams_per_dim[d] is not None:
                pts = diagrams_per_dim[d]
                if not torch.is_tensor(pts):
                    pts = torch.as_tensor(pts, dtype=torch.float32, device=self.centers.device)
            else:
                pts = torch.zeros((0, 2), device=self.centers.device)
            emb_d = self._diagram_embedding(pts)
            embeddings.append(emb_d)
        out = torch.cat(embeddings, dim=0)
        if self.cfg.l2_normalize:
            out = out / (out.norm(p=2) + 1e-8)
        return out


class PersLayClassifier(nn.Module):
    """Wrapper model: PersLayLayer -> optional MLP -> binary logit output."""
    def __init__(self, layer: PersLayLayer, hidden: Optional[int] = None, dropout: Optional[float] = None):
        super().__init__()
        self.layer = layer
        in_dim = layer.cfg.num_kernels * len(layer.cfg.homology_dims)
        if hidden is None:
            hidden = layer.cfg.hidden_dim
        if dropout is None:
            dropout = layer.cfg.dropout
        if hidden and hidden > 0:
            mods = [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout and dropout > 0:
                mods.append(nn.Dropout(p=dropout))
            mods.append(nn.Linear(hidden, 1))
            self.head = nn.Sequential(*mods)
        else:
            self.head = nn.Linear(in_dim, 1)

    def forward(self, diagrams_per_dim_batch: List[List[torch.Tensor]]) -> torch.Tensor:
        # Expect batch as list over batch items, each is list over dims
        batch_emb = []
        for dpd in diagrams_per_dim_batch:
            emb = self.layer(dpd)
            batch_emb.append(emb)
        X = torch.stack(batch_emb, dim=0)
        return self.head(X).squeeze(-1)


def initialize_centers_kmeanspp(sample_pts: torch.Tensor, k: int, device: str = 'cpu', seed: int = 42) -> torch.Tensor:
    """Deterministic k-means++ like initialization for kernel centers.
    sample_pts: (N,2) already transformed (birth,lifetime) if desired.
    """
    if sample_pts.shape[0] < k:
        # Pad by repeating deterministically
        reps = math.ceil(k / sample_pts.shape[0])
        sample_pts = sample_pts.repeat((reps, 1))[:k]
        return sample_pts.to(device)
    # Sanitize sample points (remove non-finite rows)
    mask = torch.isfinite(sample_pts).all(dim=1)
    sample_pts = sample_pts[mask]
    if sample_pts.shape[0] == 0:
        raise ValueError("No finite sample points available for initialization")
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    centers = []
    idx0 = int(torch.randint(0, sample_pts.shape[0], (1,), generator=g).item())
    centers.append(sample_pts[idx0])
    for _ in range(1, k):
        pts_stack = sample_pts
        c_stack = torch.stack(centers, dim=0)
        d2 = ((pts_stack.unsqueeze(1) - c_stack.unsqueeze(0))**2).sum(dim=2)  # (N, m)
        min_d2, _ = d2.min(dim=1)  # (N,)
        probs = min_d2 / min_d2.sum()
        r = torch.rand((), generator=g, device=probs.device)
        cdf = torch.cumsum(probs, dim=0)
        pick = torch.searchsorted(cdf, r).item()
        if pick >= probs.shape[0]:
            pick = probs.shape[0] - 1
        pick = int(pick)
        centers.append(pts_stack[pick])
    return torch.stack(centers, dim=0).to(device)


__all__ = [
    'PersLayConfig',
    'PersLayLayer',
    'PersLayClassifier',
    'initialize_centers_kmeanspp'
]
