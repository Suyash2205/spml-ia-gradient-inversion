"""Reconstruction-quality metrics: MSE, PSNR, SSIM, and (optional) LPIPS.

All functions take HWC uint8 images (0-255) or [0,1] floats. LPIPS is loaded
lazily and degrades gracefully to None if the package or its weights are
unavailable, so an offline run never hard-fails.
"""
import numpy as np

_LPIPS_MODEL = None
_LPIPS_TRIED = False


def _to01(img):
    img = np.asarray(img).astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0, 1)


def mse(a, b):
    a, b = _to01(a), _to01(b)
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return 99.0 if m == 0 else float(10 * np.log10(1.0 / m))


def ssim(a, b):
    try:
        from skimage.metrics import structural_similarity as _ssim
        return float(_ssim(_to01(a), _to01(b), channel_axis=2, data_range=1.0))
    except Exception:
        return None


def lpips_distance(a, b):
    """Perceptual LPIPS (AlexNet). Returns None if unavailable/offline."""
    global _LPIPS_MODEL, _LPIPS_TRIED
    try:
        import torch
        if not _LPIPS_TRIED:
            _LPIPS_TRIED = True
            try:
                import lpips
                _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False)
            except Exception:
                _LPIPS_MODEL = None
        if _LPIPS_MODEL is None:
            return None
        def t(x):
            x = _to01(x)
            return torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            return float(_LPIPS_MODEL(t(a), t(b)).item())
    except Exception:
        return None


def all_metrics(recon, gt):
    """Return a dict of every available metric (LPIPS may be None offline)."""
    return {
        "mse": mse(recon, gt),
        "psnr_db": psnr(recon, gt),
        "ssim": ssim(recon, gt),
        "lpips": lpips_distance(recon, gt),
    }
