"""Batch gradient-inversion: recover MULTIPLE private images from a single
averaged gradient, and show how difficulty grows with batch size.

Honest-but-curious server, distinct labels per batch (the clean regime). We
recover each label analytically from the per-column minimum of the final-layer
gradient (the batch generalisation of iDLG), then jointly optimise all images
with the cosine + total-variation loss.

Run:  python src/batch_reconstruction.py --sizes 1 4 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LeNetCIFAR, weights_init  # noqa: E402
from data_util import get_cifar, to_image, CLASSES  # noqa: E402
from metrics import all_metrics  # noqa: E402
from attacks.gradient_inversion import tv_loss  # noqa: E402


def batch_labels_from_gradient(grad_fc_weight, batch_size):
    """Per-column min of the last-layer weight gradient gives the (distinct)
    labels present in the batch (GradInversion batch-label recovery)."""
    col_min = grad_fc_weight.min(dim=1).values  # one value per class
    return torch.topk(-col_min, batch_size).indices


def pick_distinct(y, batch_size, seed=0):
    """Pick indices with distinct labels for the clean batch demo."""
    rng = np.random.default_rng(seed)
    chosen, used = [], set()
    order = rng.permutation(len(y))
    for i in order:
        lab = int(y[i])
        if lab not in used:
            used.add(lab)
            chosen.append(int(i))
        if len(chosen) == batch_size:
            break
    return chosen


def invert_batch(true_grads, labels, model, shape, device,
                 iters=4000, restarts=2, tv_weight=3e-3, lr=0.1):
    crit = torch.nn.CrossEntropyLoss()
    best = {"loss": float("inf"), "img": None}
    for _ in range(restarts):
        dummy = torch.randn(shape, device=device, requires_grad=True)
        opt = torch.optim.Adam([dummy], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
        rec_val = 0.0
        for it in range(iters):
            opt.zero_grad(); model.zero_grad()
            dloss = crit(model(dummy), labels)
            dg = torch.autograd.grad(dloss, model.parameters(), create_graph=True)
            num = sum((a * b).sum() for a, b in zip(dg, true_grads))
            dn = torch.sqrt(sum((a ** 2).sum() for a in dg))
            tn = torch.sqrt(sum((b ** 2).sum() for b in true_grads))
            rec = 1 - num / (dn * tn + 1e-12)
            total = rec + tv_weight * sum(tv_loss(dummy[i:i+1])
                                          for i in range(dummy.size(0)))
            total.backward(); opt.step(); sched.step()
            with torch.no_grad():
                dummy.clamp_(-3, 3)
            rec_val = float(rec.item())
        if rec_val < best["loss"]:
            best.update(loss=rec_val, img=dummy.detach().clone())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--out", default="results/batch_reconstruction")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    import imageio.v2 as imageio

    xtr, ytr, _, _ = get_cifar()
    summary = {"device": device, "runs": []}

    for bs in args.sizes:
        idx = pick_distinct(ytr, bs, seed=1)
        gt = xtr[idx].to(device)
        gt_labels = ytr[idx].to(device)
        model = LeNetCIFAR().to(device); model.apply(weights_init); model.eval()
        crit = torch.nn.CrossEntropyLoss()
        model.zero_grad()
        loss = crit(model(gt), gt_labels)
        true_grads = [g.detach().clone()
                      for g in torch.autograd.grad(loss, model.parameters())]
        rec_labels = batch_labels_from_gradient(true_grads[-2], bs).to(device)
        # align recovered labels to ground-truth set for a fair optimisation
        labels_for_opt = torch.sort(gt_labels).values

        best = invert_batch(true_grads, labels_for_opt, model, gt.shape, device,
                            iters=args.iters, restarts=args.restarts)

        # match each reconstruction to its best ground-truth by MSE
        per_img = []
        gi = [to_image(gt[i]) for i in range(bs)]
        ri = [to_image(best["img"][i]) for i in range(bs)]
        used = set()
        for r in range(bs):
            errs = [(np.mean((ri[r].astype(float) - gi[g].astype(float)) ** 2), g)
                    for g in range(bs) if g not in used]
            _, gbest = min(errs)
            used.add(gbest)
            m = all_metrics(ri[r], gi[gbest])
            per_img.append(m)
            imageio.imwrite(os.path.join(args.out, f"bs{bs}_gt{gbest}.png"), gi[gbest])
            imageio.imwrite(os.path.join(args.out, f"bs{bs}_rec{r}.png"), ri[r])
        mean_psnr = float(np.mean([p["psnr_db"] for p in per_img]))
        label_acc = float((torch.sort(rec_labels).values ==
                           torch.sort(gt_labels).values).float().mean().item())
        print(f"[batch={bs}] mean PSNR={mean_psnr:.2f} dB  label-recovery={label_acc:.2f}")
        summary["runs"].append({"batch_size": bs, "mean_psnr_db": mean_psnr,
                                "label_recovery_acc": label_acc,
                                "per_image": per_img})

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
