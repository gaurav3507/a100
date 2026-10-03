"""CausalBench loader for discrepancy_vae, PERTURBATION-level zero-shot split.
Their split_scdata holds out CELLS; we hold out whole GENES. c is one-hot over the
TRAIN gene list (their map_ptb_features contract: c_dim = #trainable genes).
Control cells get all-zero c. Control partners reshuffle each epoch via resample().
"""
import json, numpy as np, anndata as ad, torch
from torch.utils.data import Dataset

class CBDataset(Dataset):
    def __init__(self, h5ad, split_json, mode="train", seed=0):
        a = ad.read_h5ad(h5ad)
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        self.genes = list(a.var_names)
        guide = a.obs["guide_ids"].astype(str).values
        sp = json.load(open(split_json))
        self.train_genes = sp["train_perturbations"]
        self.held_genes  = sp["heldout_perturbations"]
        self.g2c = {g: i for i, g in enumerate(self.train_genes)}
        self.c_dim = len(self.train_genes)
        self.ctrl = X[guide == ""].astype(np.float64)
        want = set(self.train_genes) if mode == "train" else set(self.held_genes)
        keep = np.array([g in want for g in guide])
        self.Xp = X[keep].astype(np.float64)
        self.gp = guide[keep]
        self.dim = X.shape[1]
        self.rng = np.random.default_rng(seed)
        self.resample()

    def resample(self):
        self.rc = self.ctrl[self.rng.choice(len(self.ctrl), len(self.Xp), replace=True)]

    def gene_onehot(self, g):
        c = np.zeros(self.c_dim)
        if g in self.g2c: c[self.g2c[g]] = 1.0
        return c

    def __len__(self): return len(self.Xp)
    def __getitem__(self, i):
        return (torch.from_numpy(self.rc[i]).double(),
                torch.from_numpy(self.Xp[i]).double(),
                torch.from_numpy(self.gene_onehot(self.gp[i])).double())
