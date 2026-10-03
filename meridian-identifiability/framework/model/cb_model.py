"""CMVAE subclass, case A. Zero-shot readout via DECODER PROBING (respects the
nonlinear decoder). For a held-out gene g we cannot read a program row. Instead we
push each one-hot program through decode(), giving z_dim gene-space outputs, and pick
the program whose decoded output has the largest component on gene g. That program
becomes the intervention target. Must pass cb_oracle.py before any zero-shot number
is trusted.
"""
import sys, torch
sys.path.insert(0, "/workspace/external/discrepancy_vae/src")
from model import CMVAE

class CMVAE_CB(CMVAE):
    def bind(self, train_genes, all_genes):
        self.train_genes = list(train_genes)
        self.gene_col = {g: i for i, g in enumerate(all_genes)}

    def program_gene_map(self):
        """decode each one-hot program -> (z_dim, dim). Cached per eval call."""
        E = torch.eye(self.z_dim, dtype=self.d2.weight.dtype, device=self.d2.weight.device)
        with torch.no_grad():
            return self.decode(E)                        # (z_dim, dim)

    def zeroshot_target(self, gene, pg_map=None, temp=1.0):
        if pg_map is None: pg_map = self.program_gene_map()
        j = self.gene_col[gene]
        col = pg_map[:, j]                                # (z_dim,) each program's effect on g
        bc = self.sftmx((col.abs() * temp).unsqueeze(0))
        csz = self.c_shift.median().detach().reshape(1)
        return bc, csz

    def seen_target(self, gene, n, temp=1.0):
        c = torch.zeros(n, len(self.train_genes),
                        dtype=self.d2.weight.dtype, device=self.d2.weight.device)
        c[:, self.train_genes.index(gene)] = 1.0
        return self.c_encode(c, temp)

    def predict(self, x_ctrl, gene, seen, pg_map=None, temp=1.0):
        mu, var = self.encode(x_ctrl)
        z = self.reparametrize(mu, var)
        if seen:
            bc, csz = self.seen_target(gene, x_ctrl.size(0), temp)
        else:
            bc, csz = self.zeroshot_target(gene, pg_map, temp)
            bc = bc.expand(x_ctrl.size(0), -1); csz = csz.expand(x_ctrl.size(0))
        u = self.dag(z, bc, csz, bc, csz, num_interv=1)
        return self.decode(u)

    def control_pred(self, x_ctrl):
        """model's predicted OBSERVATIONAL output -- baseline for the shift metric."""
        mu, var = self.encode(x_ctrl)
        z = self.reparametrize(mu, var)
        u = self.dag(z, None, None, None, None, num_interv=0)
        return self.decode(u)
