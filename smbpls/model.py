# smbpls/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scvi.module.base import BaseModuleClass, LossOutput
from .utils import soft_threshold
import numpy as np
import anndata as ad
import mudata as mu
from sklearn.model_selection import train_test_split


class SMBPLSModule(BaseModuleClass):
    """
    Core sparse multi-block PLS module.

    Each modality has a linear projection to latent space.
    The latent scores are combined and optionally sparsified.
    """
    def __init__(self, n_input_per_mod, n_output=2, n_components=2, lam_w=0.05, lam_t=0.0):
        super().__init__()
        self.mod_names = list(n_input_per_mod.keys())
        self.K = n_components
        self.lam_w = lam_w
        self.lam_t = lam_t
        alpha = 1.0 / len(self.mod_names)  # equal block weights
        self.alpha = {m: alpha for m in self.mod_names}

        # linear projection per modality
        self.proj = nn.ModuleDict({
            m: nn.Linear(n_input_per_mod[m], self.K, bias=False)
            for m in self.mod_names
        })

        # regression head from latent to output
        self.regressor = nn.Linear(self.K, n_output, bias=True)

        for m in self.mod_names:
            nn.init.normal_(self.proj[m].weight, std=0.02)
        nn.init.zeros_(self.regressor.bias)

    @torch.no_grad()
    def apply_sparsity(self):
        """Apply L1 soft-thresholding to loadings."""
        for m in self.mod_names:
            W = self.proj[m].weight
            W.copy_(soft_threshold(W, self.lam_w))

    def _get_inference_input(self, tensors):
        return {m: tensors[m] for m in self.mod_names}

    def _get_generative_input(self, tensors, inference_outputs):
        return {'t': inference_outputs['t']}

    def inference(self, **block_data):
        """Compute latent composite score t."""
        t = None
        for m in self.mod_names:
            tb = self.proj[m](block_data[m])
            t = tb * self.alpha[m] if t is None else t + tb * self.alpha[m]

        if self.lam_t > 0:
            t = soft_threshold(t, self.lam_t)
        return {'t': t}

    def generative(self, t):
        """Map latent scores to predicted output."""
        y_hat = self.regressor(t)
        return {'y_hat': y_hat}

    def loss(self, tensors, inference_outputs, generative_outputs):
        """Compute regression + PLS covariance + orthogonality loss."""
        y = tensors['y'].float() if isinstance(tensors['y'], torch.Tensor) else torch.tensor(tensors['y'], dtype=torch.float32)
        y_hat = generative_outputs['y_hat']
        t = inference_outputs['t']

        # regression MSE
        mse = F.mse_loss(y_hat, y)

        # PLS-style covariance objective
        T = t - t.mean(0)
        yc = y - y.mean(0)
        cov = (T.T @ yc) / (T.shape[0] - 1)
        cov_loss = -(cov ** 2).sum() / (T.shape[1] * y.shape[1])

        # orthogonality constraint on latent scores
        Tn = T / (T.norm(dim=0, keepdim=True) + 1e-8)
        I = torch.eye(self.K, device=T.device)
        orth_loss = torch.norm(Tn.T @ Tn - I, p="fro") ** 2

        loss = mse + 0.1 * cov_loss + 0.1 * orth_loss
        return LossOutput(loss=loss, n_obs_minibatch=t.shape[0])


class SMBPLS:
    """
    User-facing Sparse Multi-Block PLS class for MuData.
    Wraps SMBPLSModule and handles data setup, training, and evaluation.
    """

    def __init__(self, mdata, n_outputs=2, n_components=2, lam_w=0.05, lam_t=0.0):
        self.mdata = mdata
        n_input = {m: mdata[m].n_vars for m in mdata.mod_names}
        self.module = SMBPLSModule(
            n_input_per_mod=n_input,
            n_output=n_outputs,
            n_components=n_components,
            lam_w=lam_w,
            lam_t=lam_t,
        )

    @classmethod
    def setup_mudata(cls, mdata, modalities, y_obsm_key='y', y_mod='rna'):
        """Register modalities and target on the MuData object."""
        mdata.uns['smbpls_modalities'] = modalities
        mdata.uns['smbpls_y_mod'] = y_mod
        mdata.uns['smbpls_y_key'] = y_obsm_key
        print(f"Registered modalities: {modalities}, target: {y_mod}.obsm['{y_obsm_key}']")

    def train(self, max_epochs=300, lr=1e-3, batch_size=256, sparsity_freq=50):
        """Train the SMBPLS model."""
        mdata = self.mdata
        mods = mdata.uns['smbpls_modalities']
        y_mod = mdata.uns['smbpls_y_mod']
        y_key = mdata.uns['smbpls_y_key']

        X_blocks = {m: torch.tensor(mdata[m].X, dtype=torch.float32) for m in mods}
        y = torch.tensor(mdata[y_mod].obsm[y_key], dtype=torch.float32)

        n = y.shape[0]
        idx = np.arange(n)
        tr, te = train_test_split(idx, test_size=0.2, random_state=42)

        X_tr = {m: X_blocks[m][tr] for m in mods}
        X_te = {m: X_blocks[m][te] for m in mods}
        y_tr, y_te = y[tr], y[te]

        opt = torch.optim.Adam(self.module.parameters(), lr=lr)
        self._train_losses, self._val_losses = [], []

        self.module.train()
        for epoch in range(max_epochs):
            perm = torch.randperm(len(tr))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(tr), batch_size):
                batch_idx = perm[start:start + batch_size]
                Xb = {m: X_tr[m][batch_idx] for m in mods}
                yb = y_tr[batch_idx]

                opt.zero_grad()
                inf = self.module.inference(**Xb)
                gen = self.module.generative(inf['t'])
                loss_out = self.module.loss({'y': yb}, inf, gen)
                loss_out.loss.backward()
                opt.step()
                epoch_loss += loss_out.loss.item()
                n_batches += 1

            if epoch % sparsity_freq == 0:
                self.module.apply_sparsity()

            avg_loss = epoch_loss / n_batches
            self._train_losses.append(avg_loss)

            if epoch % 50 == 0:
                self.module.eval()
                with torch.no_grad():
                    inf_te = self.module.inference(**X_te)
                    gen_te = self.module.generative(inf_te['t'])
                    val_mse = F.mse_loss(gen_te['y_hat'], y_te).item()
                self._val_losses.append((epoch, val_mse))
                print(f'epoch {epoch:>4} | train loss {avg_loss:.4f} | val MSE {val_mse:.4f}')
                self.module.train()

        self.is_trained_ = True
        print('Training done.')

    @torch.no_grad()
    def get_latent_representation(self, mdata=None):
        """Compute latent scores t for all cells."""
        if mdata is None:
            mdata = self.mdata
        mods = mdata.uns['smbpls_modalities']
        X_blocks = {m: torch.tensor(mdata[m].X, dtype=torch.float32) for m in mods}

        self.module.eval()
        inf = self.module.inference(**X_blocks)
        T = inf['t'].numpy()
        mdata.obsm['X_smbpls'] = T
        print(f"Latent scores stored in mdata.obsm['X_smbpls'], shape {T.shape}")
        return T

    @torch.no_grad()
    def get_loadings(self):
        """Return modality-wise loadings as pandas DataFrames."""
        mdata = self.mdata
        mods = mdata.uns['smbpls_modalities']
        loadings = {}
        for m in mods:
            W = self.module.proj[m].weight.numpy()
            var_names = mdata[m].var_names.tolist()
            df = pd.DataFrame(W.T, index=var_names,
                              columns=[f'component_{k+1}' for k in range(self.module.K)])
            loadings[m] = df
        return loadings

    def save(self, path):
        torch.save(self.module.state_dict(), path)
        print(f'Model weights saved to {path}')

    def load(self, path):
        self.module.load_state_dict(torch.load(path))
        print(f'Model weights loaded from {path}')

    def plot_training(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self._train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.show()
