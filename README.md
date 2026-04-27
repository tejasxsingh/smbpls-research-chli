# SMBPLS

Sparse Multi-Block Partial Least Squares (**sMBPLS**) for multi-modal datasets stored in **MuData** objects.

This package learns shared latent components across multiple modalities (e.g. RNA, ATAC, protein) while enforcing sparsity on feature loadings.

The GitHub repository is **`smbpls_2026`**, but the Python package you import is **`smbpls`**.

---

# Installation

The following example installs the package from GitHub directly into a google colaboratory notebook. 

```bash
# install package from GitHub
! pip uninstall smbpls # uninstall first just to ensure you're getting in newest version
! pip install --no-cache-dir git+https://github.com/CompBio-Lab/smbpls_2026.git
```

# Example usage

Here's an example on how to run SMBPLS on our simulated dataset

```python
import matplotlib.pyplot as plt
from smbpls import SMBPLS, simulate_mudata

# simulate multi-modal dataset
mdata = simulate_mudata()
print(mdata)

# register modalities and target latent variables
SMBPLS.setup_mudata(
    mdata,
    modalities=['rna', 'atac', 'prot'],
    y_obsm_key='y',
    y_mod='rna'
)

# visualize true latent components
Z = mdata['rna'].obsm['y']
n_cells = Z.shape[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i, ax in enumerate(axes):
    ax.scatter(range(n_cells), Z[:, i])
    ax.set_title(f"True component {i+1}")

plt.tight_layout()
plt.show()

# initialize the model
model = SMBPLS(
    mdata,
    n_components=2,
    lam_w=0.05
)

# train the model
model.train(
    max_epochs=500,
    lr=5e-4,
    batch_size=256
)
