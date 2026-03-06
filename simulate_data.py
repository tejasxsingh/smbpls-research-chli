import torch
import sys
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def generate(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    n = 1000
    p1 = 50 # number of input features after PCA
    p2 = 80
    p3 = 30
    k = 2
    z = torch.normal(mean=0, std=1, size=(n, k)) # assume data is already standardized

    stdev_scaler = 0

    true_w3 = torch.randn(p1, k)
    x1 = z @ true_w3.T + stdev_scaler * torch.randn(n, p1) # x: original input data (n, p)

    true_w4 = torch.randn(p2, k)
    x2 = z @ true_w4.T + stdev_scaler * torch.randn(n, p2)

    true_w5 = torch.randn(p3, k)
    x3 = z @ true_w5.T + stdev_scaler * torch.randn(n, p3)

    # sparse version with diffferent cols set to 0
    z_sparse = z.clone()
    z_sparse[:500, 0] = 0
    z_sparse[500:, 1] = 0

    x1_sparse = z_sparse @ true_w3.T + stdev_scaler * torch.randn(n, p1)
    x2_sparse = z_sparse @ true_w4.T + stdev_scaler * torch.randn(n, p2)
    x3_sparse = z_sparse @ true_w5.T + stdev_scaler * torch.randn(n, p3)


    # implement train test split
    indices = range(len(z_sparse))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)
    x1_train, x1_test = x1[train_indices], x1[test_indices]
    x2_train, x2_test = x2[train_indices], x2[test_indices]
    x3_train, x3_test = x3[train_indices], x3[test_indices]
    z_train, z_test = z[train_indices], z[test_indices]

    x1_sparse_train, x1_sparse_test = x1_sparse[train_indices], x1_sparse[test_indices]
    x2_sparse_train, x2_sparse_test = x2_sparse[train_indices], x2_sparse[test_indices]
    x3_sparse_train, x3_sparse_test = x3_sparse[train_indices], x3_sparse[test_indices]
    z_sparse_train, z_sparse_test = z_sparse[train_indices], z_sparse[test_indices]

    return z, z_sparse, x1, x2, x3, x1_sparse, x2_sparse, x3_sparse, z_train, z_test, z_sparse_train, z_sparse_test, x1_train, x2_train, x3_train, x1_sparse_train, x2_sparse_train, x3_sparse_train, p1, p2, p3, x1_test, x2_test, x3_test, x1_sparse_test, x2_sparse_test, x3_sparse_test

z, z_sparse, x1, x2, x3, x1_sparse, x2_sparse, x3_sparse, z_train, z_test, z_sparse_train, z_sparse_test, x1_train, x2_train, x3_train, x1_sparse_train, x2_sparse_train, x3_sparse_train, p1, p2, p3, x1_test, x2_test, x3_test, x1_sparse_test, x2_sparse_test, x3_sparse_test = generate(RANDOM_SEED)
