import torch
import numpy as np
import anndata as ad
import mudata as mu
from sklearn.model_selection import train_test_split

def simulate_mudata(seed=42, n=5000, noise=0.0):
    torch.manual_seed(seed)
    p1, p2, p3, k = 50, 80, 30, 2
    z = torch.normal(0, 1, size=(n, k))
    z_sp = z.clone()
    z_sp[:n//2, 0] = 0
    z_sp[n//2:, 1] = 0

    w1, w2, w3 = torch.randn(p1, k), torch.randn(p2, k), torch.randn(p3, k)
    x1, x2, x3 = z_sp @ w1.T + noise*torch.randn(n, p1), z_sp @ w2.T + noise*torch.randn(n, p2), z_sp @ w3.T + noise*torch.randn(n, p3)

    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    split = np.array(['train']*n)
    split[te_idx] = 'test'

    def make_adata(X, name):
        adata = ad.AnnData(X=X.numpy())
        adata.obs['split'] = split
        adata.var_names = [f'{name}_{i}' for i in range(X.shape[1])]
        return adata

    mod1, mod2, mod3 = make_adata(x1, 'rna'), make_adata(x2, 'atac'), make_adata(x3, 'prot')
    mod1.obsm['y'] = z_sp.numpy()

    mdata = mu.MuData({'rna': mod1, 'atac': mod2, 'prot': mod3})
    return mdata
