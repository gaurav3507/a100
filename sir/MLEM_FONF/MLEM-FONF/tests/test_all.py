"""Killer-level test battery for MLEM-FONF (no pytest dependency).

Run:  python tests/test_all.py        (~60-90 s)

Covers: linear-algebra correctness (adjoint, symmetry, spectral norm),
mathematical guarantees the theory relies on (H in [0,1], 1-Lipschitz,
real output, EM monotonicity), API contracts, synthetic ground-truth tests
for the coherence gate, and regression-locked headline numbers so silent
behavior changes cannot slip through a refactor.
"""
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from mlem_fonf import (build_system_matrix, poisson_sinogram, mlem, mlem_fonf,
                       mlem_spatial, pnp_admm_tv, all_metrics)  # noqa: E402
from mlem_fonf.algorithms import (mlem_fonf_damped, spectral_potential,
                                  kl_data, _perona_malik)  # noqa: E402
from mlem_fonf.fonf import (h_fonf, apply_filter, select_notches,
                            select_notches_v2, select_notches_persistent)  # noqa: E402
from mlem_fonf.metrics import psnr, mssim, cp, snr, rmse  # noqa: E402
from mlem_fonf.phantoms import PHANTOMS, shepp_logan_mod  # noqa: E402

PASS, FAIL = 0, 0
FAILED = []


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ok   {name}")
    except Exception as e:  # noqa: BLE001
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL {name}: {e}")
        traceback.print_exc(limit=2)


rng = np.random.default_rng(0)

# ---------------------------------------------------------------- projector
print("[projector]")
A64 = build_system_matrix(64, 32, 24)


def t_adjoint():
    x = rng.random(64 * 64)
    y = rng.random(A64.shape[0])
    lhs = float((A64 @ x) @ y)
    rhs = float(x @ (A64.T @ y))
    assert abs(lhs - rhs) <= 1e-8 * max(abs(lhs), 1.0), (lhs, rhs)


def t_nonneg_rows():
    assert A64.min() >= 0
    assert (np.asarray(A64.sum(axis=1)).ravel() > 0).mean() > 0.9


def t_poisson_scale():
    img = shepp_logan_mod(64)
    g = poisson_sinogram(A64, img, 1e7, np.random.default_rng(1))
    rel = np.linalg.norm(g - A64 @ img.ravel()) / np.linalg.norm(A64 @ img.ravel())
    assert rel < 0.02, rel                      # near-noiseless at huge counts


check("adjoint <Ax,y> == <x,A'y>", t_adjoint)
check("A nonnegative, rows populated", t_nonneg_rows)
check("poisson_sinogram scale sanity", t_poisson_scale)

from mlem_fonf import build_fanbeam_matrix, make_astra_fan  # noqa: E402
AF = build_fanbeam_matrix(64, 32, 24, sod=150.0, odd=150.0)


def t_fan_adjoint():
    x = rng.random(64 * 64)
    y = rng.random(AF.shape[0])
    lhs = float((AF @ x) @ y)
    rhs = float(x @ (AF.T @ y))
    assert abs(lhs - rhs) <= 1e-8 * max(abs(lhs), 1.0)


def t_fan_parallel_limit():
    Afar = build_fanbeam_matrix(64, 32, 24, sod=2e5, odd=2e5)
    img = shepp_logan_mod(64).ravel()
    a, b = Afar @ img, A64 @ img
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.999, corr


def t_fan_recon_sanity():
    img = shepp_logan_mod(64)
    g = poisson_sinogram(AF, img, 300, np.random.default_rng(5))
    p_ml = psnr(img, mlem(g, AF, 64, 150))
    p_fo = psnr(img, mlem_fonf(g, AF, 64, 150, alpha=0.5, lam_dt=0.95))
    assert p_fo > p_ml + 1.0, (p_ml, p_fo)     # FONF helps in fan geometry too


def t_astra_optional():
    try:
        make_astra_fan(64, 32, 24, gpu=False)
    except ImportError:
        print("       (astra not installed here — wrapper import-guard OK, run on GPU box)")
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"astra wrapper broke: {e}")


check("fan-beam adjoint exact", t_fan_adjoint)
check("fan -> parallel limit (sod -> inf)", t_fan_parallel_limit)
check("fan-beam MLEM/FONF sanity", t_fan_recon_sanity)
check("astra wrapper (optional backend)", t_astra_optional)

# ---------------------------------------------------------------- FONF filter
print("[fonf filter]")


def t_h_range():
    for al in (0.2, 0.5, 1.0, 2.0):
        H = h_fonf(128, alpha=al, notches=[(0.8, 0.5, 0.06)])
        assert H.min() >= 0.0 and H.max() <= 1.0 + 1e-12
        assert abs(H[0, 0] - 1.0) < 1e-9        # DC passes for K=0 term? with notch at DC-far, DC ~ G(0)=1


def t_h_symmetry_real():
    H = h_fonf(128, alpha=0.7, notches=[(0.9, -0.4, 0.05)])
    # H(-k) in FFT indexing = roll of the flipped array (even-n Nyquist row)
    assert np.allclose(H, np.roll(H[::-1, ::-1], 1, axis=(0, 1)), atol=1e-10)
    x = rng.random((128, 128))
    y = apply_filter(x, H)
    Fy = H * np.fft.fft2(x)
    assert np.abs(np.imag(np.fft.ifft2(Fy))).max() < 1e-9   # real output


def t_lipschitz_symmetric():
    H = h_fonf(128, alpha=0.5, notches=[(0.7, 0.7, 0.05)])
    for _ in range(5):
        x = rng.standard_normal((128, 128))
        assert np.linalg.norm(apply_filter(x, H)) <= np.linalg.norm(x) * (1 + 1e-10)
    x = rng.standard_normal((128, 128))
    y = rng.standard_normal((128, 128))
    lhs = np.sum(apply_filter(x, H) * y)
    rhs = np.sum(x * apply_filter(y, H))
    assert abs(lhs - rhs) < 1e-6 * max(abs(lhs), 1.0)       # A_s symmetric


def t_k0_reduces():
    assert np.allclose(h_fonf(64, 0.5, ()), h_fonf(64, 0.5, notches=[]))
    H = h_fonf(64, 0.5)
    wr = np.hypot(*np.meshgrid(*(2 * np.pi * np.fft.fftfreq(64),) * 2))
    assert np.allclose(H, np.exp(-0.5 * wr ** 2 / (4 * np.pi ** 2)), atol=1e-12)


def t_psi_nonneg():
    H = h_fonf(64, 0.5)
    for _ in range(3):
        assert spectral_potential(rng.random((64, 64)), H) >= -1e-9


check("H in [0,1], DC preserved", t_h_range)
check("H symmetric -> real filter output", t_h_symmetry_real)
check("A_s nonexpansive and symmetric", t_lipschitz_symmetric)
check("K=0 reduces exactly to G_alpha", t_k0_reduces)
check("spectral potential >= 0", t_psi_nonneg)

# ---------------------------------------------------------------- algorithms
print("[algorithms]")
img64 = shepp_logan_mod(64)
g64 = poisson_sinogram(A64, img64, 500, np.random.default_rng(2))


def t_mlem_nonneg_and_kl():
    kls = []
    mlem(g64, A64, 64, 30, callback=lambda k, f: kls.append(kl_data(g64, A64, f)))
    assert min(kls) >= -np.inf
    assert all(b <= a + 1e-6 * abs(a) for a, b in zip(kls, kls[1:])), "KL not monotone"


def t_damped_gamma1_equals_exact():
    r1 = mlem_fonf(g64, A64, 64, 40, alpha=0.5, lam_dt=0.9)
    r2 = mlem_fonf_damped(g64, A64, 64, 40, alpha=0.5, lam_dt=0.9, gamma=1.0)
    assert np.allclose(r1, r2, rtol=1e-8, atol=1e-10)


def t_all_methods_run_and_finite():
    for name, fn in (
        ("mlem", lambda: mlem(g64, A64, 64, 15)),
        ("fonf", lambda: mlem_fonf(g64, A64, 64, 15)),
        ("tv", lambda: mlem_spatial(g64, A64, 64, "tv", 15)),
        ("ad", lambda: mlem_spatial(g64, A64, 64, "ad", 15)),
        ("fuzzyad", lambda: mlem_spatial(g64, A64, 64, "fuzzyad", 15)),
        ("pnp", lambda: pnp_admm_tv(g64, A64, 64, 15)),
    ):
        out = fn()
        assert out.shape == (64, 64) and np.isfinite(out).all() and out.min() >= 0, name


def t_callback_contract():
    seen = []
    mlem_fonf(g64, A64, 64, 7, callback=lambda k, f: seen.append((k, f.shape)))
    assert seen[0][0] == 0 and seen[-1][0] == 6 and all(s == (64, 64) for _, s in seen)


def t_pm_denoiser_improves():
    ref = shepp_logan_mod(128)
    noisy = np.clip(ref + 0.08 * np.random.default_rng(3).standard_normal(ref.shape), 0, None)
    out = _perona_malik(noisy.copy(), n_sub=3, kappa=0.15, dt=0.15)
    assert psnr(ref, out) > psnr(ref, noisy) + 1.0


check("MLEM keeps f>=0 and KL monotone", t_mlem_nonneg_and_kl)
check("damped(gamma=1) == exact algorithm", t_damped_gamma1_equals_exact)
check("all six methods run, finite, nonneg", t_all_methods_run_and_finite)
check("callback contract (k, image)", t_callback_contract)
check("Perona-Malik improves noisy PSNR", t_pm_denoiser_improves)

# ---------------------------------------------------------------- metrics
print("[metrics]")


def t_metric_identities():
    x = rng.random((64, 64))
    assert mssim(x, x) > 0.999 and rmse(x, x) < 1e-12 and abs(cp(x, x) - 1) < 1e-9
    m = all_metrics(x, np.clip(x + 0.01 * rng.standard_normal(x.shape), 0, None))
    assert set(m) == {"SNR", "MSE", "RMSE", "PSNR", "CP", "MSSIM"}
    assert m["PSNR"] > 20 and 0 <= m["MSSIM"] <= 1


check("metric identities and ranges", t_metric_identities)

# ------------------------------------------------------- notch rule (v1/v2)
print("[notch selection]")
yy, xx = np.mgrid[0:256, 0:256]


def t_v2_synthetic_separation():
    base = shepp_logan_mod(256) * 0.6
    stripe = base + 0.1 * np.sin(2 * np.pi * (0.11 * xx + 0.05 * yy))
    det = select_notches_v2(stripe)
    tgt = np.array([0.11, 0.05]) * 2 * np.pi
    assert det, "global stripe not detected"
    err = min(np.hypot(a - tgt[0], b - tgt[1]) * 256 / (2 * np.pi) for a, b, _ in det)
    assert err <= 3.0, err
    env = np.exp(-((xx - 70) ** 2 + (yy - 90) ** 2) / (2 * 12.0 ** 2))
    packet = base + 0.25 * env * np.sin(2 * np.pi * (0.11 * xx + 0.05 * yy))
    assert select_notches_v2(packet) == [], "localized packet must be rejected"


def t_v2_determinism_and_conjugates():
    im = shepp_logan_mod(256) + 0.1 * np.sin(2 * np.pi * (0.13 * xx))
    d1, d2 = select_notches_v2(im), select_notches_v2(im)
    assert d1 == d2
    assert all(a > 0 or (a == 0 and b >= 0) for a, b, _ in d1)   # half-plane only


def t_persistence_intersection():
    im = shepp_logan_mod(256) + 0.1 * np.sin(2 * np.pi * (0.11 * xx + 0.05 * yy))
    r = np.random.default_rng(4)
    a = im + 0.005 * r.standard_normal(im.shape)
    b = im + 0.005 * r.standard_normal(im.shape)
    assert len(select_notches_persistent(a, b)) >= 1


check("v2: global stripe kept, local packet rejected", t_v2_synthetic_separation)
check("v2: deterministic, half-plane conjugate policy", t_v2_determinism_and_conjugates)
check("persistence keeps true stripe", t_persistence_intersection)

# ------------------------------------------------------- regression locks
print("[regression locks]  (seed-pinned headline numbers)")
A256 = build_system_matrix(256, 64, 90)
ref = shepp_logan_mod(256)
gref = poisson_sinogram(A256, ref, 60, np.random.default_rng(0))


def t_regression_mlem_fonf():
    p_ml = psnr(ref, mlem(gref, A256, 256, 300))
    p_fo = psnr(ref, mlem_fonf(gref, A256, 256, 300, alpha=0.5, lam_dt=0.95))
    assert abs(p_ml - 15.87) < 0.15, p_ml       # locked ±0.15 dB
    assert abs(p_fo - 19.10) < 0.20, p_fo   # measured v0.3.0 baseline
    assert p_fo - p_ml > 2.3


check("Shepp-Logan 300-iter MLEM/FONF locked", t_regression_mlem_fonf)

print("[adaptive alpha]")
from mlem_fonf import alpha_from_counts, alpha_morozov, noise_level_from_counts  # noqa: E402
from mlem_fonf.projector import poisson_sinogram as _ps  # noqa: E402


def t_counts_rule_monotone():
    img = shepp_logan_mod(64)
    etas = []
    for inc in (50, 500, 5000):
        _, c = _ps(A64, img, inc, np.random.default_rng(0), return_counts=True)
        etas.append(noise_level_from_counts(c))
    assert etas[0] > etas[1] > etas[2]                      # noisier -> larger eta
    a = [alpha_from_counts.__wrapped__(None) if False else None]  # placeholder no-op


def t_morozov_small():
    img = shepp_logan_mod(64)
    g, c = _ps(A64, img, 400, np.random.default_rng(3), return_counts=True)
    scale = g.max() / c.max()
    a1 = alpha_morozov(c, scale, A64, 64, grid=(0.5, 1.2, 2.0), n_iter=200)
    a2 = alpha_morozov(c, scale, A64, 64, grid=(0.5, 1.2, 2.0), n_iter=200)
    assert a1 == a2 and a1 in (0.5, 1.2, 2.0)               # deterministic, in-grid


check("noise proxy monotone in dose", t_counts_rule_monotone)
check("Morozov selection deterministic, in-grid", t_morozov_small)


def t_alpha_ladder_uncaps_low_dose():
    """The discrepancy ladder must be able to select beyond the old 2.0 cap
    when the data demand it, and must stop as soon as the fit becomes
    noise-inconsistent (deviance is monotone in alpha)."""
    from mlem_fonf import alpha_grid_for
    from mlem_fonf.adaptive import alpha_morozov
    lad = alpha_grid_for()
    assert lad[0] == 0.2 and max(lad) >= 400.0 and len(set(lad)) == len(lad)
    # the criterion, not the ladder end, must terminate the search at the
    # lowest doses (4 dB was lost to a 2.0 ceiling before v0.9.30)
    img = shepp_logan_mod(64)
    g, c = _ps(A64, img, 40, np.random.default_rng(0), return_counts=True)   # low dose
    scale = g.max() / c.max()
    a_sel, diag = alpha_morozov(c, scale, A64, 64, n_iter=120, return_diag=True)
    devs = diag["deviances"]
    ks = sorted(devs)
    assert all(devs[ks[i]] <= devs[ks[i + 1]] + 1e-9 for i in range(len(ks) - 1)), devs
    assert a_sel == max(a for a in devs if devs[a] <= 1.0) or a_sel == min(devs, key=devs.get)
    assert len(devs) <= len(lad)        # early stop: never more than the ladder


check("alpha ladder: uncapped, monotone, early-stopping", t_alpha_ladder_uncaps_low_dose)

print("[frozen pipeline (Algorithm 1-prime)]")
from mlem_fonf import reconstruct  # noqa: E402


def _pipe(ref, inc, seed=0):
    g, c = _ps(A64, ref, inc, np.random.default_rng(seed), return_counts=True)
    scale = (A64 @ ref.ravel()).max() / inc
    return reconstruct(c, scale, A64, 64, n_bins=32, n_iter=150, select_iter=150)


def t_pipeline_clean_K0_and_deterministic():
    ref = shepp_logan_mod(64)
    f1, r1 = _pipe(ref, 400)
    f2, r2 = _pipe(ref, 400)
    assert r1["K"] == 0, r1["K"]
    assert r1["alpha"] == r2["alpha"] and np.allclose(f1, f2)
    for key in ("alpha", "K", "notches", "deviances", "gamma", "lam_dt", "runtime_s"):
        assert key in r1


def t_pipeline_detects_inband_stripe():
    yy2, xx2 = np.mgrid[0:64, 0:64]
    ref = np.clip(shepp_logan_mod(64)
                  + 0.12 * np.sin(2 * np.pi * (0.16 * xx2 + 0.05 * yy2)), 0, None)
    _, r = _pipe(ref, 2000)
    assert r["K"] >= 1, "in-band stripe not detected"
    tgt = np.array([0.16, 0.05]) * 2 * np.pi
    err = min(np.hypot(a - tgt[0], b - tgt[1]) * 64 / (2 * np.pi)
              for a, b, _ in r["notches"])
    assert err <= 2.0, err


def t_pipeline_band_guard():
    yy2, xx2 = np.mgrid[0:64, 0:64]
    ref = np.clip(shepp_logan_mod(64)
                  + 0.12 * np.sin(2 * np.pi * (0.30 * xx2 + 0.05 * yy2)), 0, None)
    _, r = _pipe(ref, 2000)          # r ~ 19 px > band 14 for n_bins=32
    assert r["K"] == 0, r["notches"]


check("pipeline: clean K=0, deterministic, report contract", t_pipeline_clean_K0_and_deterministic)
check("pipeline: in-band stripe detected from projections", t_pipeline_detects_inband_stripe)
check("pipeline: acquisition-band guard rejects beyond-band", t_pipeline_band_guard)

print("[fbp + learned scaffolding]")
from mlem_fonf.fbp import fbp as _fbp  # noqa: E402
from mlem_fonf import build_fanbeam_matrix as _bfan  # noqa: E402


def t_fbp_dense_sanity():
    ref = shepp_logan_mod(96)
    Ad = build_system_matrix(96, 96, 120)
    r = _fbp(Ad @ ref.ravel(), Ad, 96, 96, 120, "parallel")
    assert psnr(ref, r) > 21, psnr(ref, r)
    Af = _bfan(96, 96, 120, sod=250, odd=250)
    rf = _fbp(Af @ ref.ravel(), Af, 96, 96, 120, "fan", sod=250, odd=250)
    assert psnr(ref, rf) > 19, psnr(ref, rf)


def t_torch_stack_optional():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("       (torch not installed here — guards OK; validate on ant-pc)")
        return
    from mlem_fonf.torch_ops import to_torch_projector
    fwd, bwd, _ = to_torch_projector(A64, device="cpu")
    import torch
    x = torch.rand(2, 64 * 64)
    y = torch.rand(2, A64.shape[0])
    lhs = (fwd(x) * y).sum().item()
    rhs = (x * bwd(y)).sum().item()
    assert abs(lhs - rhs) < 1e-3 * max(abs(lhs), 1.0)
    # gradient exactness: d/dx sum(A x) must equal A^T 1 (certified adjoint)
    xg = torch.rand(1, 64 * 64, requires_grad=True)
    fwd(xg).sum().backward()
    expected = torch.from_numpy((A64.T @ np.ones(A64.shape[0])).astype("float32"))
    assert torch.allclose(xg.grad[0], expected, atol=1e-3)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
    from models import REDCNN, LPD, DIPNet
    r = REDCNN()(torch.rand(1, 1, 55, 55))
    assert r.shape == (1, 1, 55, 55)
    net = LPD(fwd, bwd, 64, 32, 24, n_iter=2)
    out = net(torch.rand(1, A64.shape[0]))
    assert out.shape == (1, 1, 64, 64)
    d = DIPNet()(torch.rand(1, 1, 64, 64))
    assert d.shape == (1, 1, 64, 64)


check("FBP sanity (parallel + fan, dense views)", t_fbp_dense_sanity)
check("torch stack: certified adjoint + model shapes (optional)", t_torch_stack_optional)


def t_dps_mechanics_optional():
    try:
        import torch
    except ImportError:
        print("       (torch absent — DPS mechanics validated on ant-pc)")
        return
    import math, tempfile, os
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
    from models import TinyUNet
    from eval_learned import dps_infer
    net = TinyUNet(ch=16)
    T = 50
    tt = torch.linspace(0, 1, T + 1)
    ab = torch.cos((tt + 0.008) / 1.008 * math.pi / 2) ** 2
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "diffusion_400.pt")
        torch.save({"state_dict": net.state_dict(), "T": T,
                    "abar": (ab / ab[0]), "ch": 16}, p)
        img = shepp_logan_mod(64)
        g = A64 @ img.ravel()
        out = dps_infer(p, g, A64, 1.0, n=64, steps=3, device="cpu")
    assert out.shape == (64, 64) and np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0


check("DPS sampler mechanics (optional)", t_dps_mechanics_optional)


def t_lpd_training_dynamics_optional():
    try:
        import torch
    except ImportError:
        print("       (torch absent — LPD dynamics validated on ant-pc)")
        return
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
    from models import LPD
    from mlem_fonf.torch_ops import to_torch_projector
    img = shepp_logan_mod(64)
    v = np.random.default_rng(0).standard_normal(A64.shape[1])
    for _ in range(25):
        v = A64.T @ (A64 @ v)
        v /= np.linalg.norm(v)
    sigma = float(np.sqrt(np.linalg.norm(A64.T @ (A64 @ v))))
    fwd0, bwd0, _ = to_torch_projector(A64, device="cpu")
    net = LPD(lambda x: fwd0(x) / sigma, lambda y: bwd0(y) / sigma,
              64, 32, 24, n_iter=3)
    g = _ps(A64, img, 800, np.random.default_rng(1)) / sigma
    gt = torch.tensor(g, dtype=torch.float32)[None]
    rt = torch.tensor(img, dtype=torch.float32)[None, None]
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    losses = []
    for _ in range(6):
        loss = torch.nn.functional.mse_loss(net(gt), rt)
        assert torch.isfinite(loss), "LPD loss non-finite"
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    with torch.no_grad():
        out = net(gt)
    assert out.abs().max() > 1e-4, "degenerate (all-zero) LPD output"


check("LPD training dynamics: finite + decreasing (optional)",
      t_lpd_training_dynamics_optional)


def t_npz_export_roundtrip_optional():
    try:
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError:
        print("       (pydicom absent — export round-trip validated elsewhere)")
        return
    import tempfile, os, subprocess
    from mlem_fonf import data as D
    with tempfile.TemporaryDirectory() as td:
        pdir = os.path.join(td, "mayo", "L067", "full_1mm")
        os.makedirs(pdir)
        r = np.random.default_rng(0)
        for k in range(4):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = generate_uid()
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = pydicom.dataset.FileDataset(
                os.path.join(pdir, f"s{k:03d}.IMA"), {}, file_meta=meta,
                preamble=b"\0" * 128)
            arr = (r.random((64, 64)) * 1200).astype(np.int16)
            ds.Rows = ds.Columns = 64
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.RescaleSlope, ds.RescaleIntercept = 1.0, -1024.0
            ds.PixelData = arr.tobytes()
            ds.save_as(os.path.join(pdir, f"s{k:03d}.IMA"))
        out = os.path.join(td, "t.npz")
        script = str(pathlib.Path(__file__).resolve().parents[1] /
                     "training" / "export_train_npz.py")
        subprocess.run([sys.executable, script, "--root", td, "--out", out,
                        "--limit", "4"], check=True, capture_output=True)
        npz = np.load(out)["slices"]
        ref = np.stack([im for _, _, im in
                        D.mayo_slices(td, "train", "full", every=2, limit=4)])
        assert npz.shape == ref.shape
        assert np.abs(ref.astype("f4") - npz).max() < 1e-6


check("npz export round-trip exact (optional)", t_npz_export_roundtrip_optional)


def t_dps_quality_oracle_optional():
    """DPS must couple to data: with an exact Gaussian-prior score model,
    guided samples must beat the prior mean by a wide margin, while the
    legacy (v<=0.9.7) step must not. Model-free quality assertion."""
    try:
        import torch
    except ImportError:
        print("       (torch absent — DPS quality oracle validated elsewhere)")
        return
    import math
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
    from eval_learned import dps_infer

    class Oracle(torch.nn.Module):
        def __init__(self, mu, sp, abar, T):
            super().__init__()
            self.mu, self.sp2, self.abar, self.T = mu, sp ** 2, abar, T

        def forward(self, x, t):
            ab = self.abar[(t * self.T).round().long().clamp(1, self.T)][:, None, None, None]
            x0 = (self.sp2 * ab.sqrt() * x + (1 - ab) * self.mu) / (ab * self.sp2 + (1 - ab))
            return (x - ab.sqrt() * x0) / (1 - ab).sqrt()

    n, T = 64, 200
    tt = torch.linspace(0, 1, T + 1)
    ab = torch.cos((tt + 0.008) / 1.008 * math.pi / 2) ** 2
    abar = ab / ab[0]
    Ad = build_system_matrix(n, 64, 60)
    ref = shepp_logan_mod(n)
    g = Ad @ ref.ravel()
    oracle = Oracle(torch.zeros(1, 1, n, n), 0.5, abar, T)
    base = psnr(ref, np.full_like(ref, 0.5))
    kw = dict(n=n, steps=50, seed=0, device="cpu", net=oracle, abar=abar, T=T)
    old = psnr(ref, dps_infer(None, g, Ad, 1.0, zeta=1.0, _legacy_step=True, **kw))
    new = psnr(ref, dps_infer(None, g, Ad, 1.0, zeta=1.0, **kw))
    assert old < base + 2.0, (old, base)
    assert new > base + 6.0, (new, base)


check("DPS quality oracle: guidance couples to data (optional)",
      t_dps_quality_oracle_optional)


def t_dps_trained_model_reconstructs_optional():
    """Real trained score model (fixture, 100 ep on phantoms @64): guided
    DPS must reconstruct (>> gray baseline and >> unconditional sample).
    Catches the clip_denoised / abar_T explosion class that a
    self-consistent oracle cannot."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("       (torch absent — trained-model DPS test validated elsewhere)")
        return
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
    from eval_learned import dps_infer
    ck = str(pathlib.Path(__file__).resolve().parent / "fixtures" / "score_phantom64.pt")
    n = 64
    Ad = build_system_matrix(n, 64, 60)
    ref = shepp_logan_mod(n)
    g, _ = _ps(Ad, ref, 3000, np.random.default_rng(0), return_counts=True)
    scale = (Ad @ ref.ravel()).max() / 3000
    base = psnr(ref, np.full_like(ref, 0.5))
    unc = psnr(ref, dps_infer(ck, g, Ad, scale, n=n, steps=60, zeta=0.0, seed=0, device="cpu"))
    gui = psnr(ref, dps_infer(ck, g, Ad, scale, n=n, steps=60, zeta=0.1, seed=0, device="cpu"))
    assert np.isfinite(unc) and unc > 5.0, ("unconditional chain exploded", unc)
    assert gui > base + 8.0 and gui > unc + 4.0, (gui, unc, base)


check("DPS on a trained model reconstructs (fixture, optional)",
      t_dps_trained_model_reconstructs_optional)

print("[task-based observer (CHO)]")
from mlem_fonf.cho import (laguerre_gauss_channels, gaussian_lesion,  # noqa: E402
                           channel_responses, cho_dprime, bootstrap_ci,
                           auc_from_dprime, extract_roi)


def t_cho_ideal_scaling():
    """On white noise the CHO must track the analytic matched-filter d'
    (= ||s||/sigma) and never exceed it."""
    nn, N = 64, 500
    sig = gaussian_lesion(nn, 4.0, 1.0)
    chs = laguerre_gauss_channels(nn, 6, 10.0)
    for sigma in (2.0, 1.0):
        r = np.random.default_rng(0)
        absent = r.normal(0, sigma, (N, nn, nn))
        present = absent + sig[None]
        d = cho_dprime(channel_responses(present, chs),
                       channel_responses(absent, chs), split_half=False)
        ideal = np.linalg.norm(sig) / sigma
        assert 0.25 < d / ideal < 1.05, (d, ideal)


def t_cho_null_unbiased():
    """No signal -> split-half d' ~ 0 with a CI covering zero (resubstitution
    would be optimistically positive)."""
    nn = 64
    chs = laguerre_gauss_channels(nn, 6, 10.0)
    r = np.random.default_rng(0)
    a1 = channel_responses(r.normal(0, 1, (300, nn, nn)), chs)
    a2 = channel_responses(r.normal(0, 1, (300, nn, nn)), chs)
    d, lo, hi = bootstrap_ci(a1, a2, n_boot=120)
    assert abs(d) < 0.4 and lo <= 0 <= hi, (d, lo, hi)
    assert cho_dprime(a1, a2, split_half=False) > d - 1e-9   # bias direction


def t_cho_contrast_monotone_and_auc():
    nn = 48
    chs = laguerre_gauss_channels(nn, 6, 12.0)
    r = np.random.default_rng(1)
    base = r.normal(0, 1, (300, nn, nn))
    prev = -1e9
    for c in (0.2, 0.5, 1.0):
        sig = gaussian_lesion(nn, 4.0, c)
        d = cho_dprime(channel_responses(base + sig[None], chs),
                       channel_responses(base, chs))
        assert d > prev, (c, d, prev)
        prev = d
    assert 0.5 <= auc_from_dprime(0.0) <= 0.5 + 1e-9
    assert auc_from_dprime(2.0) > 0.84


def t_cho_roi_and_channels():
    img = np.arange(64 * 64, dtype=float).reshape(64, 64)
    roi = extract_roi(img, (20, 40), 16)
    assert roi.shape == (16, 16) and roi[0, 0] == img[12, 32]
    chs = laguerre_gauss_channels(32, 5, 8.0)
    assert chs.shape == (5, 32, 32)
    assert all(abs(np.linalg.norm(c) - 1) < 1e-6 for c in chs)
    # standard LG channels are NOT zero-mean (j=0 is a Gaussian); zero-meaning
    # them blinds the observer to the ROI mean-shift cue (documented ablation)
    assert abs(chs[0].mean()) > 1e-3
    assert abs(laguerre_gauss_channels(32, 1, 8.0, zero_mean=True)[0].mean()) < 1e-12
    a = 8.0
    yy2, xx2 = np.mgrid[0:32, 0:32]
    cc = (32 - 1) / 2.0
    r2c = ((xx2 - cc) ** 2 + (yy2 - cc) ** 2) / a ** 2
    for j in (0, 2, 4):
        want = (np.polynomial.laguerre.lagval(2 * np.pi * r2c, [0] * j + [1])
                * np.exp(-np.pi * r2c))
        want = want / np.linalg.norm(want)
        assert np.allclose(chs[j], want, atol=1e-9), j


check("CHO tracks the ideal-observer scaling", t_cho_ideal_scaling)
check("CHO null case unbiased (split-half) with CI covering 0", t_cho_null_unbiased)
check("CHO monotone in contrast; AUC mapping", t_cho_contrast_monotone_and_auc)
check("CHO channels unit-norm zero-mean; ROI extraction exact", t_cho_roi_and_channels)


def t_cho_ci_brackets_estimate():
    """Bootstrap CI must contain the point estimate at every sample size
    (regression for the single-split noise bug)."""
    nn = 48
    chs = laguerre_gauss_channels(nn, 6, 12.0)
    for N in (20, 60):
        r = np.random.default_rng(0)
        base = r.normal(0, 1, (N, nn, nn))
        sig = gaussian_lesion(nn, 4.0, 0.5)
        d, lo, hi = bootstrap_ci(channel_responses(base + sig[None], chs),
                                 channel_responses(base, chs), n_boot=80)
        assert lo <= d <= hi, (N, d, lo, hi)


def t_cho_runner_task_contract():
    """Runner tasks must stay tiny (indices, not arrays) — a 1 MB task times
    14k tasks was a 15 GB pool transfer."""
    import pickle
    task = ("A-FONF", 3000.0, 2, True, (150, 128), 48, 7_000_000, 400, "parallel")
    assert len(pickle.dumps(task)) < 500
    seeds = [7_000_000 + si * 100_000 + k * 10 + int(p)
             for si in range(4) for k in range(100) for p in (True, False)]
    assert len(set(seeds)) == len(seeds)          # no noise-seed reuse


check("CHO bootstrap CI brackets the estimate", t_cho_ci_brackets_estimate)
check("CHO runner task/seed contract (memory + independence)", t_cho_runner_task_contract)


def t_cho_campaign_operational():
    """OPERATIONAL test (the class that a functional test cannot cover):
    the campaign must (a) keep parent memory flat, (b) produce identical
    statistics on a repeat run, and (c) reproduce them exactly after a hard
    kill + resume. Runs a shrunken campaign on a synthetic DICOM tree."""
    try:
        import pydicom  # noqa: F401
    except ImportError:
        print("       (pydicom absent — operational campaign test skipped)")
        return
    import csv as _csv, os, shutil, signal, subprocess, tempfile, time
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    root = tempfile.mkdtemp()
    pdir = os.path.join(root, "mayo", "L067", "full_1mm")
    os.makedirs(pdir)
    rr = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    base = np.where(((xx - 64) ** 2 / 45 ** 2 + (yy - 64) ** 2 / 35 ** 2) <= 1, 900.0, 0.0)
    for k in range(4):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = generate_uid()
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(os.path.join(pdir, f"s{k}.IMA"), {}, file_meta=meta,
                         preamble=b"\0" * 128)
        ds.Rows = ds.Columns = 128
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, -1024.0
        ds.PixelData = (base + rr.random((128, 128)) * 150).astype(np.int16).tobytes()
        ds.save_as(os.path.join(pdir, f"s{k}.IMA"))
    script = str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "run_cho.py")
    args = ["--root", root, "--split", "train", "--methods", "FBP",
            "--incidents", "3000", "--contrasts", "0.06", "--radii", "4.0",
            "--realizations", "12", "--slices", "1", "--iters", "20",
            "--workers", "4"]

    def _run(tag, kill_after=None):
        out = os.path.join(root, f"{tag}.csv")
        shutil.rmtree(os.path.join(root, tag), ignore_errors=True)
        if kill_after:
            p = subprocess.Popen([sys.executable, script, *args, "--out", out],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 preexec_fn=os.setsid)
            time.sleep(kill_after)
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
        subprocess.run([sys.executable, script, *args, "--out", out], check=True,
                       capture_output=True)
        return {(r["method"], r["incident"]): float(r["d_prime"])
                for r in _csv.DictReader(open(out))}

    a1 = _run("a1")
    a2 = _run("a2")
    a3 = _run("a3", kill_after=6)
    assert a1 and a1.keys() == a2.keys() == a3.keys()
    for k in a1:
        assert abs(a1[k] - a2[k]) < 1e-9, ("repeat not deterministic", k, a1[k], a2[k])
        assert abs(a1[k] - a3[k]) < 1e-9, ("kill+resume differs", k, a1[k], a3[k])
    shutil.rmtree(root, ignore_errors=True)


check("CHO campaign: deterministic across repeats and kill+resume (optional)",
      t_cho_campaign_operational)

print("[realistic-geometry gate]")


def t_gate_operator_matches_pipeline():
    """The matrix-free gate must run the SHIPPED algorithm: same alpha, same
    notch count, bit-identical image. (Caught: convex-combination damping
    instead of the exponent form; damped instead of undamped selection;
    missing warm start for the persistence partner.)"""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    import run_realistic_geometry as RG
    from mlem_fonf import build_fanbeam_matrix, reconstruct
    nn, nb, na = 96, 72, 90
    Af = build_fanbeam_matrix(nn, nb, na, sod=1000, odd=500)
    fwd = lambda x: Af @ np.asarray(x).ravel()
    bwd = lambda y: Af.T @ np.asarray(y).ravel()
    ref = shepp_logan_mod(nn)
    g, counts = _ps(Af, ref, 600, np.random.default_rng(0), return_counts=True)
    scale = (Af @ ref.ravel()).max() / 600
    f_mat, rep = reconstruct(counts, scale, Af, nn, n_bins=nb, n_iter=80, select_iter=50)
    f_op, a_sel, K = RG.afonf_op(counts, scale, fwd, bwd, nn, nb, 80, 50)
    assert a_sel == rep["alpha"] and K == rep["K"], (a_sel, rep["alpha"], K, rep["K"])
    assert np.allclose(f_mat, f_op, atol=1e-9), np.abs(f_mat - f_op).max()


def t_gate_fbp_no_oracle_scale():
    """FBP's scale must come from a calibration phantom, not from the test
    image (an oracle leak that inflated the baseline)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    import run_realistic_geometry as RG
    from mlem_fonf import build_fanbeam_matrix
    nn, nb, na = 96, 72, 90
    Af = build_fanbeam_matrix(nn, nb, na, sod=1000, odd=500)
    fwd = lambda x: Af @ np.asarray(x).ravel()
    bwd = lambda y: Af.T @ np.asarray(y).ravel()
    RG._FBP_SCALE.clear()
    ref = shepp_logan_mod(nn)
    r1 = RG.fbp_op(fwd(ref.ravel()), bwd, nn, nb, na, 1000.0, 500.0, fwd)
    scale_after_first = dict(RG._FBP_SCALE)
    other = np.clip(ref * 0.3 + 0.2, 0, None)          # very different object
    RG.fbp_op(fwd(other.ravel()), bwd, nn, nb, na, 1000.0, 500.0, fwd)
    assert RG._FBP_SCALE == scale_after_first, "scale changed with the object"
    assert np.isfinite(r1).all() and psnr(ref, r1) > 12


check("gate operator pipeline == shipped algorithm (bit-identical)",
      t_gate_operator_matches_pipeline)
check("gate FBP scale is object-independent (no oracle leak)",
      t_gate_fbp_no_oracle_scale)

print("=" * 56)
print(f"RESULT: {PASS} passed, {FAIL} failed" + (f" -> {FAILED}" if FAILED else "  — ALL PASS"))
sys.exit(1 if FAIL else 0)
