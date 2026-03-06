import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Elementwise soft-thresholding: sign(x)*max(|x|-lam,0)."""
    if lam <= 0:
        return x
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


class SMBPLSNet(nn.Module):
    """
    Sparse Multi-Block PLS-style REGRESSION network.

    Core structure:
      - multiple input blocks X^(b)
      - block-specific loadings W^(b)
      - block scores t^(b) = X^(b) W^(b)
      - weighted sum of block scores
      - optional sparsity on weights and scores
      - linear regression on latent components

    Output:
      - single continuous variable (or multiple if output_dim > 1)
    """

    def __init__(
        self,
        block_dims: Dict[str, int],
        n_components: int = 2,
        output_dim: int = 1, # Added this parameter
        block_weights: Optional[Dict[str, float]] = None,
        lam_w: float = 0.05,
        lam_t: float = 0.0,
        normalize_loadings: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.block_names: List[str] = list(block_dims.keys())
        self.block_dims = block_dims
        self.K = n_components # Number of PLS components
        self.output_dim = output_dim # Store it
        self.lam_w = float(lam_w)
        self.lam_t = float(lam_t)
        self.normalize_loadings = normalize_loadings
        self.eps = eps

        # block weights α_b
        if block_weights is None:
            w = 1.0 / max(len(self.block_names), 1)
            self.alpha = {b: w for b in self.block_names}
        else:
            self.alpha = {b: float(block_weights.get(b, 0.0)) for b in self.block_names}

        # block projections: (n, p_b) → (n, K)
        self.proj = nn.ModuleDict({
            b: nn.Linear(block_dims[b], self.K, bias=False)
            for b in self.block_names
        })

        # regression head: (n, K) → (n, output_dim)
        self.regressor = nn.Linear(self.K, self.output_dim, bias=True) # Changed 1 to self.output_dim

        # initialization
        for b in self.block_names:
            nn.init.normal_(self.proj[b].weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.regressor.bias)

    @torch.no_grad()
    def apply_weight_sparsity_and_normalize(self) -> None:
        """
        Proximal step:
          - soft-threshold variable loadings
          - L2-normalize each component within each block
        """
        for b in self.block_names:
            W = self.proj[b].weight  # (K, p_b)

            # sparsity
            W.copy_(soft_threshold(W, self.lam_w))

    def forward(self, X_blocks):
        t = None

        for b in self.block_names:
            tb = self.proj[b](X_blocks[b])
            ab = self.alpha[b]
            t = tb * ab if t is None else t + tb * ab

        # optional sparsity ONLY — no normalization
        if self.lam_t > 0:
            t = soft_threshold(t, self.lam_t)

        y_hat = self.regressor(t)

        if self.output_dim == 1:
          y_hat = y_hat.squeeze(-1)

        return y_hat, t


    def predict(self, X_blocks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(X_blocks)
def covariance_loss(T: torch.Tensor, y: torch.Tensor, eps=1e-8):
    """
    Correct PLS2 covariance loss

    T: (n, K)
    y: (n, D)
    """
    # center
    T = T - T.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    n = T.shape[0]

    # covariance matrix (K × D)
    cov = (T.T @ y) / (n - 1 + eps)

    # maximize covariance
    return -(cov ** 2).sum()
def r2_score_torch(y_hat, y, eps=1e-8): # r^2 coefficient
    ss_res = torch.sum((y - y_hat) ** 2, dim=0)
    ss_tot = torch.sum((y - torch.mean(y, dim=0)) ** 2, dim=0)
    return 1 - ss_res / (ss_tot + eps)
