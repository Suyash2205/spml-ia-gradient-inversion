"""Defences applied to the gradient a client uploads in federated learning.

Each defence transforms the gradient the honest-but-curious server receives,
so it is exactly the signal the gradient-inversion attacker tries to invert.
The same transform, applied during training, has a model-utility cost -- that
is the privacy-utility trade-off we measure.

Mapping to the course modules:
    * gaussian_noise  -> Differential Privacy (Gaussian mechanism; DP-SGD)
    * gradient_prune  -> Compressive Privacy (sparsification / Top-k)
    * quantize        -> Compressive Privacy (bit-depth reduction)
    * local_dp_noise  -> Local Differential Privacy (per-client randomisation)
"""
import torch


def clip_gradient(grads, max_norm):
    """Global-norm clip, as in DP-SGD."""
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    scale = min(1.0, max_norm / (total + 1e-12))
    return [g * scale, ] if False else [g * scale for g in grads]


def gaussian_noise(grads, sigma, clip=None):
    """DP Gaussian mechanism: optional clip, then add N(0, sigma^2)."""
    if clip is not None:
        grads = clip_gradient(grads, clip)
    return [g + torch.randn_like(g) * sigma for g in grads]


def local_dp_noise(grads, sigma, clip=1.0):
    """Local DP: each client clips and perturbs its own update before upload.
    Same mechanism as Gaussian DP but framed as client-side (no trusted server);
    we keep it separate so the report can discuss the trust-model difference."""
    return gaussian_noise(grads, sigma, clip=clip)


def gradient_prune(grads, prune_ratio):
    """Zero the smallest-magnitude fraction `prune_ratio` of all entries
    (the DLG paper's own defence; a compressive / sparsification defence)."""
    if prune_ratio <= 0:
        return [g.clone() for g in grads]
    flat = torch.cat([g.abs().flatten() for g in grads])
    k = int(len(flat) * prune_ratio)
    if k <= 0:
        return [g.clone() for g in grads]
    thresh = torch.kthvalue(flat, k).values
    out = []
    for g in grads:
        mask = (g.abs() > thresh).float()
        out.append(g * mask)
    return out


def quantize(grads, bits):
    """Uniform per-tensor quantisation to `bits` bits (compressive defence)."""
    out = []
    levels = 2 ** bits - 1
    for g in grads:
        lo, hi = g.min(), g.max()
        if hi - lo < 1e-12:
            out.append(g.clone())
            continue
        q = torch.round((g - lo) / (hi - lo) * levels) / levels
        out.append(q * (hi - lo) + lo)
    return out


def apply_defense(grads, name, strength):
    """Dispatch by name. `strength` meaning depends on the defence."""
    if name == "none" or strength == 0:
        return [g.clone() for g in grads]
    if name == "gaussian":
        return gaussian_noise(grads, sigma=strength, clip=1.0)
    if name == "local_dp":
        return local_dp_noise(grads, sigma=strength, clip=1.0)
    if name == "prune":
        return gradient_prune(grads, prune_ratio=strength)
    if name == "quantize":
        return quantize(grads, bits=int(strength))
    raise ValueError(f"unknown defense {name}")
