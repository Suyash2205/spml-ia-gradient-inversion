"""Experiment: how well does each defence stop the gradient-inversion attack,
and what does it cost in model accuracy?  (privacy vs utility)

For each defence (Gaussian/DP noise, gradient pruning, quantization, local-DP)
and each strength:
    PRIVACY  = run the inversion attack on the DEFENDED gradient, measure how
               poorly it reconstructs the private image (low PSNR/SSIM, high
               LPIPS = strong protection).
    UTILITY  = train a classifier whose gradients are passed through the SAME
               defence each step, measure clean test accuracy.

Outputs: results/defense_sweep/metrics.json, reconstruction grid PNGs, and a
privacy-utility trade-off plot per defence.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LeNetCIFAR, LeNetReLU, weights_init  # noqa: E402
from data_util import get_cifar, to_image, CLASSES  # noqa: E402
from metrics import all_metrics  # noqa: E402
from attacks.gradient_inversion import (  # noqa: E402
    compute_true_gradients, invert_from_gradients, infer_label_from_gradient)
from defenses.gradient_defenses import apply_defense  # noqa: E402

SWEEPS = {
    "gaussian": [0.0, 0.001, 0.01, 0.05, 0.1],      # noise sigma (DP)
    "prune":    [0.0, 0.5, 0.7, 0.9, 0.99],          # fraction zeroed
    "quantize": [8, 6, 4, 2],                        # bits (fewer = stronger)
    "local_dp": [0.0, 0.001, 0.01, 0.05, 0.1],       # per-client noise sigma
}
MODULE = {
    "gaussian": "Module 2: Differential Privacy",
    "local_dp": "Module 3: Local Differential Privacy",
    "prune": "Module 5: Compressive Privacy (sparsification)",
    "quantize": "Module 5: Compressive Privacy (quantization)",
}


def train_utility(xtr, ytr, xte, yte, device, defense, strength,
                  epochs=12, bs=128, lr=1e-3, subset=8000):
    """Train a trainable (ReLU) classifier with the defence applied to each
    step's gradient; return clean test accuracy. Measures the utility cost of
    the defence mechanism."""
    idx = torch.randperm(len(xtr))[:subset]
    ds = TensorDataset(xtr[idx], ytr[idx])
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    model = LeNetReLU().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            grads = torch.autograd.grad(loss, model.parameters())
            dg = apply_defense(list(grads), defense, strength)
            with torch.no_grad():
                for p, g in zip(model.parameters(), dg):
                    p.grad = g
            opt.step()
    # evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(xte), 512):
            xb = xte[i:i + 512].to(device)
            pred = model(xb).argmax(1).cpu()
            correct += (pred == yte[i:i + 512]).sum().item()
    return correct / len(xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", type=int, default=25)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--util-epochs", type=int, default=8)
    ap.add_argument("--out", default="results/defense_sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    grid_dir = os.path.join(args.out, "reconstructions")
    os.makedirs(grid_dir, exist_ok=True)

    xtr, ytr, xte, yte = get_cifar()
    gt = xtr[args.image_id:args.image_id + 1].to(device)
    gt_label = ytr[args.image_id:args.image_id + 1].to(device)
    gt_img = to_image(gt[0])
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(args.out, "ground_truth.png"), gt_img)

    model = LeNetCIFAR().to(device)
    model.apply(weights_init)
    model.eval()
    true_grads = compute_true_gradients(gt, gt_label, model)
    rec_label = infer_label_from_gradient(true_grads[-2])

    results = {"image_id": args.image_id,
               "true_label": int(gt_label.item()),
               "true_class": CLASSES[int(gt_label.item())],
               "recovered_label": int(rec_label),
               "device": device, "defenses": {}}

    for defense, strengths in SWEEPS.items():
        print(f"\n=== defence: {defense} ({MODULE[defense]}) ===")
        results["defenses"][defense] = {"module": MODULE[defense], "points": []}
        for s in strengths:
            torch.manual_seed(args.seed)
            dg = apply_defense([g.clone() for g in true_grads], defense, s)
            best, _ = invert_from_gradients(
                dg, model, gt, device, iters=args.iters,
                restarts=args.restarts, rec_label=rec_label,
                group_consistency=True)
            rec_img = to_image(best["img"][0])
            m = all_metrics(rec_img, gt_img)
            try:
                acc = train_utility(xtr, ytr, xte, yte, device, defense, s,
                                    epochs=args.util_epochs)
            except Exception as e:
                print(f"  [util fail @ {s}: {e}]")
                acc = None
            fname = f"{defense}_{s}.png"
            imageio.imwrite(os.path.join(grid_dir, fname), rec_img)
            point = {"strength": s, "test_accuracy": acc,
                     "reconstruction": m, "image": fname}
            results["defenses"][defense]["points"].append(point)
            print(f"  strength={s}: acc={acc} PSNR={m['psnr_db']:.2f} "
                  f"SSIM={m['ssim']} LPIPS={m['lpips']}")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {args.out}/metrics.json")

    # ---- privacy-utility trade-off plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (defense, d) in zip(axes.ravel(), results["defenses"].items()):
        pts = d["points"]
        accs = [p["test_accuracy"] for p in pts]
        psnr = [p["reconstruction"]["psnr_db"] for p in pts]
        xs = [p["strength"] for p in pts]
        ax2 = ax.twinx()
        ax.plot(xs, accs, "o-", color="tab:blue", label="test accuracy")
        ax2.plot(xs, psnr, "s--", color="tab:red", label="recon PSNR (↓=safer)")
        ax.set_xlabel(f"{defense} strength")
        ax.set_ylabel("test accuracy", color="tab:blue")
        ax2.set_ylabel("recon PSNR dB", color="tab:red")
        ax.set_title(d["module"], fontsize=9)
    fig.suptitle("Privacy–Utility Trade-off: defences vs gradient inversion")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "privacy_utility.png"), dpi=150)
    print(f"[done] wrote {args.out}/privacy_utility.png")


if __name__ == "__main__":
    main()
