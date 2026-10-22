"""CIFAR-10 loading, cached as tensors so every experiment is fast and the
dataset is read from disk only once.
"""
import os

import numpy as np
import torch

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
_CACHE = os.path.join("data", "cifar_tensors.pt")


def _load_via_torchvision():
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    tr = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    te = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor([te[i][1] for i in range(len(te))])
    return xtr, ytr, xte, yte


def get_cifar(cache=True):
    """Return (x_train, y_train, x_test, y_test) as normalised tensors."""
    if cache and os.path.exists(_CACHE):
        d = torch.load(_CACHE)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    xtr, ytr, xte, yte = _load_via_torchvision()
    if cache:
        torch.save({"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}, _CACHE)
    return xtr, ytr, xte, yte


def denormalize(x):
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1).to(x.device)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1).to(x.device)
    return (x * std + mean).clamp(0, 1)


def to_image(x):
    """CHW normalised tensor -> HWC uint8 image."""
    img = denormalize(x)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
