from setuptools import setup, find_packages

setup(
    name="smbpls",  # this is the pip / import name
    version="0.1.0",
    description="Sparse Multi-Block Partial Least Squares for multi-omics MuData",
    author="Ray Zhang",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scvi-tools",
        "anndata",
        "mudata",
        "scanpy",
        "matplotlib",
        "scikit-learn",
    ],
)
