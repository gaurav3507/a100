"""Test phantoms: modified Shepp-Logan, simulated CT, simulated elliptical,
and a synthetic thorax-like phantom (the paper's real thoracic CT slice is
not redistributable; drop your own 256x256 slice into data/ to reproduce
that column)."""
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def _ellipse(n, cx, cy, a, b, ang, val, img):
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - cx) / a
    y = (yy - cy) / b
    c, s = np.cos(ang), np.sin(ang)
    img[(x * c + y * s) ** 2 + (-x * s + y * c) ** 2 <= 1.0] += val
    return img


def shepp_logan_mod(n=256):
    """Modified Shepp-Logan with enhanced low-contrast internal structures."""
    p = resize(shepp_logan_phantom(), (n, n), anti_aliasing=True)
    return np.clip(p / p.max(), 0, 1)


def ct_sim(n=256):
    img = np.zeros((n, n))
    _ellipse(n, n/2, n/2, n*0.42, n*0.34, 0, 0.55, img)
    _ellipse(n, n*0.40, n*0.42, n*0.10, n*0.16, 0.4, 0.45, img)
    _ellipse(n, n*0.62, n*0.55, n*0.06, n*0.06, 0, -0.25, img)
    _ellipse(n, n*0.52, n*0.32, n*0.05, n*0.03, 0, 0.35, img)
    return np.clip(img, 0, 1)


def elliptical(n=256):
    img = np.zeros((n, n))
    _ellipse(n, n/2, n/2, n*0.44, n*0.30, 0, 0.6, img)
    _ellipse(n, n*0.38, n*0.48, n*0.09, n*0.07, 0, 0.30, img)
    _ellipse(n, n*0.60, n*0.44, n*0.07, n*0.10, 0.5, 0.25, img)
    _ellipse(n, n*0.55, n*0.62, n*0.045, n*0.045, 0, -0.30, img)
    return np.clip(img, 0, 1)


def thorax_like(n=256):
    img = np.zeros((n, n))
    _ellipse(n, n/2, n/2, n*0.46, n*0.34, 0, 0.50, img)          # body
    _ellipse(n, n*0.35, n*0.50, n*0.14, n*0.20, 0.15, -0.35, img) # left lung
    _ellipse(n, n*0.65, n*0.50, n*0.14, n*0.20, -0.15, -0.35, img)
    _ellipse(n, n*0.50, n*0.55, n*0.09, n*0.11, 0, 0.40, img)     # heart
    _ellipse(n, n*0.50, n*0.30, n*0.035, n*0.05, 0, 0.45, img)    # mediastinum
    _ellipse(n, n*0.50, n*0.78, n*0.05, n*0.035, 0, 0.35, img)    # spine
    return np.clip(img, 0, 1)


PHANTOMS = {"shepp_logan": shepp_logan_mod, "ct_sim": ct_sim,
            "elliptical": elliptical, "thorax_like": thorax_like}
