import torch

def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Elementwise soft-thresholding: sign(x)*max(|x|-lam,0)."""
    if lam <= 0:
        return x
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)
