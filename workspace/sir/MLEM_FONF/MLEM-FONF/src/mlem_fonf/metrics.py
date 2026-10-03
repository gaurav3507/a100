"""Evaluation metrics: SNR, MSE, RMSE, PSNR, CP (edge correlation), MSSIM."""
import numpy as np
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity


def _flat(ref, rec):
    return ref.ravel().astype(float), rec.ravel().astype(float)


def mse(ref, rec):
    a, b = _flat(ref, rec)
    return float(np.mean((a - b) ** 2))


def rmse(ref, rec):
    return float(np.sqrt(mse(ref, rec)))


def psnr(ref, rec):
    return float(10 * np.log10((ref.max() ** 2) / max(mse(ref, rec), 1e-15)))


def snr(ref, rec):
    a, b = _flat(ref, rec)
    return float(10 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-15)))


def cp(ref, rec):
    """Edge correlation: Pearson correlation of Sobel gradient magnitudes."""
    def gm(x):
        return np.hypot(sobel(x, 0), sobel(x, 1)).ravel()
    a, b = gm(ref), gm(rec)
    a -= a.mean(); b -= b.mean()
    return float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-15))


def mssim(ref, rec):
    return float(structural_similarity(ref, rec, data_range=ref.max() - ref.min()))


def all_metrics(ref, rec):
    return dict(SNR=snr(ref, rec), MSE=mse(ref, rec), RMSE=rmse(ref, rec),
                PSNR=psnr(ref, rec), CP=cp(ref, rec), MSSIM=mssim(ref, rec))
