import torch
import torch.nn as nn
import torch.nn.functional as F
from smbpls_model import soft_threshold, SMBPLSNet, covariance_loss, r2_score_torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import sys

def train_smbpls(
    RANDOM_SEED,
    z_sparse,
    z_sparse_train,
    x1_train, x2_train, x3_train,
    x1_sparse_train, x2_sparse_train, x3_sparse_train,
    z_sparse_test, x1_sparse_test, x2_sparse_test, x3_sparse_test,
    p1, p2, p3,
    n_iterations=50,
    lr=1e-3,
    show_plots=True,
):
    # Example usage with sparse 2-dimensional output data

    torch.manual_seed(RANDOM_SEED)

    # Determine output_dim from the target 'y' (z_sparse) that is used below.
    # In the global scope, `z_sparse` has shape (1000, 2), so output_dim should be 2.
    actual_output_dim = z_sparse.shape[1]

    model = SMBPLSNet(
        block_dims={"rna": p1, "atac": p2, 'prot': p3},
        n_components=2,
        output_dim=actual_output_dim,  # Pass the determined output_dim
        lam_w=0.05,
        lam_t=0.0,
    )

    # The local 'n' = 64 is not used for defining X and y in this block,
    # as they are taken from the global scope where n=1000.
    n_local = 64

    X = {
        "rna": x1_train,
        "atac": x2_train,
        'prot': x3_train
    }

    X_sparse = {
        "rna": x1_sparse_train,
        "atac": x2_sparse_train,
        'prot': x3_sparse_train
    }

    y = z_sparse_train  # continuous target, shape (1000, 2)

    Xte = {"rna": x1_sparse_test, "atac": x2_sparse_test, "prot": x3_sparse_test}
    yte = z_sparse_test
        

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    mse_values = []
    r2_values = []

    for step in range(500):
        model.train()
        opt.zero_grad()

        y_hat, T = model(X_sparse)

        loss_pred = F.mse_loss(y_hat, y)

        loss_cov = covariance_loss(T, y)  # include covariance term

        loss = 0.5 * loss_pred + loss_cov

        loss.backward()
        opt.step()        

        mse_values.append(float(loss_pred))  # store mse values at each step
        r2_values.append(float(r2_score_torch(y_hat, y).mean()))  # store r^2 correlation coefficient at each step

        if step % n_iterations == 0:
            model.apply_weight_sparsity_and_normalize()

        if step % 30 == 0:
            model.eval()
            with torch.no_grad():
                yhat, _ = model(Xte)
                print(f"At step {step} | Train MSE = {float(loss_pred)} | Test MSE = {float(F.mse_loss(yhat, yte))}")

    print("final loss:", float(loss))
    y_hat, t_loadings = model(X_sparse)
    y_hat = soft_threshold(y_hat, model.lam_w) # enforce final soft thresholding on outputs

    # plot predicted vs. true values by output dimension
    if show_plots: 
      for i in range(y.shape[1]):
          plt.figure()
          plt.scatter(y_hat.detach()[:, i], y[:, i])
          plt.plot(y[:, i], y[:, i], c='r')
          plt.xlabel("Predicted")
          plt.ylabel("True")
          plt.title(f"Predicted vs True (Output Dimension {i})")
          plt.show()
    return model, mse_values, r2_values, y_hat.detach(), y
