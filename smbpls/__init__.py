"""
SMBPLS: Sparse Multi-Block Partial Least Squares implemented in PyTorch.

This package provides:
- SMBPLS: SCVI and PyTorch implementation of sparse multi-block PLS
- simulate_mudata: synthetic data generator for experiments
- soft_threshold: LASSO regularization used to apply sparsity
"""

from .model import SMBPLS
from .utils import soft_threshold
from .data import simulate_mudata

__all__ = ["SMBPLS", "simulate_mudata", "soft_threshold"]
