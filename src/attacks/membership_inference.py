"""Membership Inference Attack (LiRA) + baselines.

Implements the Likelihood Ratio Attack of Carlini et al., "Membership
Inference Attacks From First Principles" (IEEE S&P 2022), plus the Yeom
loss-threshold baseline, and evaluates them the way that paper argues for:
TPR at low FPR on a log-log ROC, not just balanced accuracy / AUC.

Pipeline:
    1. Train N shadow models, each on a random half of a population pool, so
       every target example is IN roughly half of them and OUT the other half.
    2. For each example, collect the logit-scaled confidence of the true class
       from the IN shadows and the OUT shadows, and fit a Gaussian to each.
    3. For a target model, score each example by the likelihood ratio of its
       observed confidence under the IN vs OUT Gaussians.
    4. Report AUC and TPR@{0.1%,1%} FPR; save a log-log ROC.

Designed to finish overnight on a GTX 1650: a small CNN, a CIFAR-10 subset,
and a configurable (modest) number of shadow models.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LeNetReLU  # noqa: E402

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]


def load_pool(n_pool, seed=0):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_pool]
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx])
    return xs, ys


def train_model(x, y, device, epochs=20, lr=0.05, bs=128, dp=None):
    """Train a LeNet on (x,y). If dp is a dict, train with DP-SGD (Opacus)."""
    model = LeNetReLU().to(device)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    privacy_engine = None
    if dp is not None:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, opt, dl = privacy_engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=dl,
            epochs=epochs, target_epsilon=dp["epsilon"],
            target_delta=dp["delta"], max_grad_norm=dp["max_grad_norm"])
    model.train()
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    eps = None
    if privacy_engine is not None:
        eps = privacy_engine.get_epsilon(dp["delta"])
    model.eval()
    return model, eps


def logit_scaled_conf(model, x, y, device, bs=512):
    """LiRA logit-scaled confidence of the true class: log(p / (1 - p)).

    The probability is clamped away from 0 and 1 so that confidently-fit
    (overfit) models do not produce +-inf, which keeps the downstream Gaussian
    fit and ROC computation numerically stable.
    """
    confs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs].to(device)
            yb = y[i:i + bs].to(device)
            prob = F.softmax(model(xb), dim=1)
            p_y = prob.gather(1, yb.view(-1, 1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
            conf = torch.log(p_y) - torch.log1p(-p_y)
            confs.append(conf.cpu())
    out = torch.cat(confs).numpy()
    return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0)


def tpr_at_fpr(fpr, tpr, target):
    return float(np.interp(target, fpr, tpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadows", type=int, default=32)
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="results/membership_inference")
    ap.add_argument("--dp-epsilon", type=float, default=None,
                    help="if set, the TARGET model is trained with this DP epsilon")
    ap.add_argument("--tag", default="undefended")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    xs, ys = load_pool(args.pool)
    n = len(xs)
    half = n // 2

    # membership matrix: in_mask[s, i] = was example i IN shadow s's train set
    rng = np.random.default_rng(123)
    in_mask = np.zeros((args.shadows, n), dtype=bool)
    confs = np.zeros((args.shadows, n), dtype=np.float32)
    print(f"[lira] training {args.shadows} shadow models on {n} examples "
          f"({args.epochs} epochs each) device={device}")
    for s in range(args.shadows):
        members = rng.permutation(n)[:half]
        in_mask[s, members] = True
        m, _ = train_model(xs[members], ys[members], device, epochs=args.epochs)
        confs[s] = logit_scaled_conf(m, xs, ys, device)
        print(f"  shadow {s + 1}/{args.shadows} done")
        del m
        torch.cuda.empty_cache()

    # ---- target model: half are true members ----
    target_members = rng.permutation(n)[:half]
    target_in = np.zeros(n, dtype=bool)
    target_in[target_members] = True
    dp = None
    if args.dp_epsilon is not None:
        dp = {"epsilon": args.dp_epsilon, "delta": 1e-5, "max_grad_norm": 1.0}
    target_model, eps = train_model(xs[target_members], ys[target_members],
                                    device, epochs=args.epochs, dp=dp)
    target_conf = logit_scaled_conf(target_model, xs, ys, device)

    # ---- LiRA offline/online: per-example IN and OUT Gaussians ----
    scores = np.zeros(n, dtype=np.float32)
    g_mu, g_sd = float(confs.mean()), float(confs.std()) + 1e-6
    for i in range(n):
        in_c = confs[in_mask[:, i], i]
        out_c = confs[~in_mask[:, i], i]
        # fall back to global stats when a point was always IN or always OUT
        mu_in = in_c.mean() if in_c.size else g_mu
        sd_in = (in_c.std() if in_c.size else g_sd) + 1e-6
        mu_out = out_c.mean() if out_c.size else g_mu
        sd_out = (out_c.std() if out_c.size else g_sd) + 1e-6
        c = target_conf[i]
        ll_in = -0.5 * ((c - mu_in) / sd_in) ** 2 - np.log(sd_in)
        ll_out = -0.5 * ((c - mu_out) / sd_out) ** 2 - np.log(sd_out)
        scores[i] = ll_in - ll_out  # higher => more likely a member
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Yeom loss-threshold baseline: higher confidence => member
    baseline_scores = target_conf

    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    plt.figure(figsize=(5, 5))
    for name, sc in [("LiRA", scores), ("Yeom loss-threshold", baseline_scores)]:
        fpr, tpr, _ = roc_curve(target_in, sc)
        auc = roc_auc_score(target_in, sc)
        results[name] = {
            "auc": float(auc),
            "tpr_at_0.1pct": tpr_at_fpr(fpr, tpr, 1e-3),
            "tpr_at_1pct": tpr_at_fpr(fpr, tpr, 1e-2),
        }
        plt.loglog(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.7, label="chance")
    plt.xlim(1e-4, 1); plt.ylim(1e-4, 1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    title = f"Membership Inference ROC ({args.tag}"
    if eps is not None:
        title += f", DP ε={eps:.2f}"
    title += ")"
    plt.title(title); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"roc_{args.tag}.png"), dpi=150)

    import json
    out = {"tag": args.tag, "shadows": args.shadows, "pool": n,
           "epochs": args.epochs, "dp_epsilon_requested": args.dp_epsilon,
           "dp_epsilon_achieved": float(eps) if eps else None,
           "results": results}
    with open(os.path.join(args.out, f"metrics_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[lira] results:", json.dumps(results, indent=2))
    if eps is not None:
        print(f"[lira] target model DP epsilon achieved = {eps:.3f}")


if __name__ == "__main__":
    main()
