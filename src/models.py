"""Model architectures used across the project.

A deliberately small LeNet-style CNN is used for the gradient-inversion
demonstration: untrained shallow networks with smooth activations leak the
cleanest reconstructions (Geiping et al., 2020), and it fits easily in the
GTX 1650's 4 GB of VRAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetCIFAR(nn.Module):
    """Small CNN for 32x32x3 images.

    Uses Sigmoid activations because gradient-inversion attacks reconstruct
    more faithfully through smooth, twice-differentiable non-linearities than
    through ReLU (the original DLG / Inverting-Gradients finding).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1), act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1), act(),
        )
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class LeNetReLU(nn.Module):
    """ReLU CNN used only to MEASURE UTILITY (test accuracy) under a defence.

    The gradient-inversion attack uses the smooth-activation LeNetCIFAR because
    Sigmoid leaks the cleanest reconstructions, but an all-Sigmoid net trains
    poorly as a classifier. For the privacy-utility trade-off we therefore
    measure the accuracy cost of each defence on this trainable ReLU variant of
    the same small-CNN family; the defence mechanism (clip/noise/prune/quantize)
    is what the utility axis is about, and its cost is architecture-robust.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.body(x))


class SmallMLP(nn.Module):
    """Plain MLP for tabular membership-inference experiments."""

    def __init__(self, in_features: int, num_classes: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def weights_init(m):
    """Small uniform init; a fresh random net leaks cleaner gradients."""
    if hasattr(m, "weight") and m.weight is not None:
        try:
            m.weight.data.uniform_(-0.5, 0.5)
        except Exception:
            pass
    if hasattr(m, "bias") and m.bias is not None:
        try:
            m.bias.data.uniform_(-0.5, 0.5)
        except Exception:
            pass
