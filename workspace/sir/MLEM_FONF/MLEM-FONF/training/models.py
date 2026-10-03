"""Learned-baseline architectures (published configurations).

RED-CNN: Chen et al., TMI 2017 — 5 conv + 5 deconv, 96 filters, residual
skips. LPD: Adler & Oektem, TMI 2018 — 10 unrolled iterations, 3-layer
primal/dual blocks, 32 filters, 5 memory channels. DIP: compact hourglass
for per-image optimization (Baguer et al., 2020 setting).
"""
import torch
import torch.nn as nn


class REDCNN(nn.Module):
    def __init__(self, ch=96):
        super().__init__()
        C = lambda i, o: nn.Conv2d(i, o, 5, padding=2)
        D = lambda i, o: nn.ConvTranspose2d(i, o, 5, padding=2)
        self.c1, self.c2, self.c3, self.c4, self.c5 = C(1, ch), C(ch, ch), C(ch, ch), C(ch, ch), C(ch, ch)
        self.d1, self.d2, self.d3, self.d4, self.d5 = D(ch, ch), D(ch, ch), D(ch, ch), D(ch, ch), D(ch, 1)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        s0 = x
        x = self.r(self.c1(x)); s1 = x
        x = self.r(self.c2(x))
        x = self.r(self.c3(x)); s2 = x
        x = self.r(self.c4(x))
        x = self.r(self.c5(x))
        x = self.r(self.d1(x) + s2)
        x = self.r(self.d2(x))
        x = self.r(self.d3(x) + s1)
        x = self.r(self.d4(x))
        return self.r(self.d5(x) + s0)


class _Block(nn.Module):
    def __init__(self, cin, cout, ch=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), nn.PReLU(),
                                 nn.Conv2d(ch, ch, 3, padding=1), nn.PReLU(),
                                 nn.Conv2d(ch, cout, 3, padding=1))

    def forward(self, x):
        return self.net(x)


class LPD(nn.Module):
    """Learned Primal-Dual over the certified torch projector."""

    def __init__(self, fwd, bwd, n, n_bins, n_angles, n_iter=10, mem=5):
        super().__init__()
        self.fwd, self.bwd = fwd, bwd
        self.n, self.nb, self.na, self.T, self.M = n, n_bins, n_angles, n_iter, mem
        self.primal = nn.ModuleList([_Block(mem + 1, mem) for _ in range(n_iter)])
        self.dual = nn.ModuleList([_Block(mem + 2, mem) for _ in range(n_iter)])

    def forward(self, g):                          # g: (B, m)
        B = g.shape[0]
        f = torch.zeros(B, self.M, self.n, self.n, device=g.device)
        h = torch.zeros(B, self.M, self.na, self.nb, device=g.device)
        gs = g.view(B, self.na, self.nb)
        for t in range(self.T):
            Af = self.fwd(f[:, 0].reshape(B, -1)).view(B, self.na, self.nb)
            h = h + self.dual[t](torch.cat([h, Af[:, None], gs[:, None]], 1))
            Ath = self.bwd(h[:, 0].reshape(B, -1)).view(B, 1, self.n, self.n)
            f = f + self.primal[t](torch.cat([f, Ath], 1))
        return f[:, :1]                        # unconstrained (per LPD);
        # positivity is enforced at inference/benchmark time


class DIPNet(nn.Module):
    def __init__(self, ch=64, depth=4):
        super().__init__()
        enc, dec = [], []
        c = 1
        for _ in range(depth):
            enc.append(nn.Sequential(nn.Conv2d(c, ch, 3, 2, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.1)))
            c = ch
        for _ in range(depth):
            dec.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     nn.Conv2d(c, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.1)))
        self.enc, self.dec = nn.ModuleList(enc), nn.ModuleList(dec)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, z):
        x = z
        for e in self.enc:
            x = e(x)
        for d in self.dec:
            x = d(x)
        return torch.relu(self.out(x))


class TinyUNet(nn.Module):
    """Compact time-conditioned UNet for the diffusion baseline (importable
    by both the DGX trainer and the DPS sampler)."""

    def __init__(self, ch=64):
        super().__init__()
        self.t1 = nn.Linear(1, ch)
        def blk(i, o):
            return nn.Sequential(nn.GroupNorm(8, i), nn.SiLU(),
                                 nn.Conv2d(i, o, 3, padding=1))
        self.e1 = nn.Conv2d(1, ch, 3, padding=1)
        self.e2 = nn.Sequential(blk(ch, ch), nn.AvgPool2d(2))
        self.e3 = nn.Sequential(blk(ch, 2 * ch), nn.AvgPool2d(2))
        self.mid = blk(2 * ch, 2 * ch)
        self.d3 = blk(2 * ch, ch)
        self.d2 = blk(2 * ch, ch)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)

    def forward(self, x, t):
        te = self.t1(t[:, None])[:, :, None, None]
        h1 = self.e1(x) + te
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        mm = self.mid(h3)
        u3 = self.up(self.d3(mm))
        u2 = self.up(self.d2(torch.cat([u3, h2], 1)))
        return self.out(u2 + h1)


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = (nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity())

    def forward(self, x, te):
        h = self.c1(nn.functional.silu(self.n1(x)))
        h = h + self.t(te)[:, :, None, None]
        h = self.c2(nn.functional.silu(self.n2(h)))
        return h + self.skip(x)


class BigUNet(nn.Module):
    """Deep multi-scale score model for the XL diffusion baseline.

    4 resolutions (256/128/64/32 at n=256), channel multipliers (1,2,3,4),
    two residual blocks per level, sinusoidal-free learned time embedding.
    Same (x, t) -> eps interface as TinyUNet; DPS-compatible."""

    def __init__(self, ch=128):
        super().__init__()
        tdim = ch * 4
        self.temb = nn.Sequential(nn.Linear(1, tdim), nn.SiLU(),
                                  nn.Linear(tdim, tdim))
        mults = (1, 2, 3, 4)
        cs = [ch * m_ for m_ in mults]
        self.stem = nn.Conv2d(1, cs[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        cprev = cs[0]
        for c in cs:
            self.down.append(nn.ModuleList([_ResBlock(cprev, c, tdim),
                                            _ResBlock(c, c, tdim)]))
            self.pool.append(nn.AvgPool2d(2))
            cprev = c
        self.mid1 = _ResBlock(cs[-1], cs[-1], tdim)
        self.mid2 = _ResBlock(cs[-1], cs[-1], tdim)
        self.up = nn.ModuleList()
        for i, c in enumerate(reversed(cs)):
            cskip = c
            cout = cs[max(len(cs) - 2 - i, 0)] if i < len(cs) - 1 else cs[0]
            self.up.append(nn.ModuleList([_ResBlock(c + cskip, cout, tdim),
                                          _ResBlock(cout, cout, tdim)]))
        self.out_n = nn.GroupNorm(8, cs[0])
        self.out = nn.Conv2d(cs[0], 1, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)

    def forward(self, x, t):
        te = self.temb(t[:, None])
        h = self.stem(x)
        skips = []
        for blocks, pool in zip(self.down, self.pool):
            for b in blocks:
                h = b(h, te)
            skips.append(h)
            h = pool(h)
        h = self.mid2(self.mid1(h, te), te)
        for blocks in self.up:
            h = self.upsample(h)
            h = torch.cat([h, skips.pop()], 1)
            for b in blocks:
                h = b(h, te)
        return self.out(nn.functional.silu(self.out_n(h)))
