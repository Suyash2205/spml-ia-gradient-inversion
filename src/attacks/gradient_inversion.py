"""Gradient-inversion attack on federated learning.

Threat model (honest-but-curious server):
    In federated learning a client never uploads raw data. It computes the
    gradient of the loss on its *private* local batch and uploads only that
    gradient. We show a curious server can invert the gradient and recover the
    client's private input pixel-by-pixel.

Method: Inverting Gradients (Geiping et al., 2020)
    - match dummy and true gradients by COSINE SIMILARITY (scale invariant,
      far more reliable than the L2 matching of the original DLG paper),
    - regularise the candidate image with anisotropic Total Variation,
    - optimise the image with Adam + cosine-annealed LR, several restarts,
    - recover the ground-truth label analytically from the sign of the
      last-layer gradient (iDLG trick) instead of optimising it.

Usage:
    python src/attacks/gradient_inversion.py --image-id 25 --iters 4000
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LeNetCIFAR, weights_init  # noqa: E402

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616])


def tv_loss(x):
    """Anisotropic total variation; suppresses high-frequency noise."""
    dx = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()
    dy = (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    return dx + dy


def infer_label_from_gradient(grad_last_layer_weight):
    """iDLG: for a single sample with cross-entropy, the row of dL/dW in the
    final layer corresponding to the true class is the only one whose gradient
    points the opposite way; its summed gradient is negative."""
    per_class = grad_last_layer_weight.sum(dim=1)
    return int(per_class.argmin().item())


def to_image(x):
    """De-normalise a CHW tensor to a HWC uint8 image for saving."""
    mean = CIFAR_MEAN.view(3, 1, 1).to(x.device)
    std = CIFAR_STD.view(3, 1, 1).to(x.device)
    img = (x * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def compute_true_gradients(ground_truth, true_label, model):
    """Victim side: the gradient a client would upload for this batch."""
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    loss = criterion(model(ground_truth), true_label)
    grads = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in grads]


def invert_from_gradients(true_grads, model, template, device,
                          iters=4000, restarts=3, tv_weight=3e-3, lr=0.1,
                          rec_label=None, group_consistency=True,
                          frame_dir=None, log_every=100):
    """Attacker side: reconstruct the input from (possibly defended) gradients.

    `template` only supplies the shape/dtype of the target batch. If
    `group_consistency` is set, the final image is the pixel-wise median over
    restarts (the near-free GradInversion trick) rather than best-of-N alone.
    """
    criterion = torch.nn.CrossEntropyLoss()
    if rec_label is None:
        rec_label = infer_label_from_gradient(true_grads[-2])
    n = template.size(0)
    labels = torch.tensor([rec_label] * n, device=device)

    best = {"loss": float("inf"), "img": None, "frames": None}
    candidates = []
    for r in range(restarts):
        dummy = torch.randn_like(template, requires_grad=True)
        opt = torch.optim.Adam([dummy], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
        frames = []
        rec_val = 0.0
        for it in range(iters):
            opt.zero_grad()
            model.zero_grad()
            dloss = criterion(model(dummy), labels)
            dgrads = torch.autograd.grad(dloss, model.parameters(),
                                         create_graph=True)
            num = sum((dg * tg).sum() for dg, tg in zip(dgrads, true_grads))
            dn = torch.sqrt(sum((dg ** 2).sum() for dg in dgrads))
            tn = torch.sqrt(sum((tg ** 2).sum() for tg in true_grads))
            rec = 1 - num / (dn * tn + 1e-12)
            total = rec + tv_weight * tv_loss(dummy)
            total.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                dummy.clamp_(-3, 3)
            if frame_dir is not None and r == 0 and (it % log_every == 0):
                frames.append((it, float(rec.item()),
                               to_image(dummy[0].detach())))
            rec_val = float(rec.item())
        candidates.append(dummy.detach().clone())
        if rec_val < best["loss"]:
            best.update(loss=rec_val, img=dummy.detach().clone(),
                        frames=frames if r == 0 else best["frames"])
    if group_consistency and len(candidates) > 1:
        stacked = torch.stack(candidates, dim=0)
        best["img_median"] = stacked.median(dim=0).values
    else:
        best["img_median"] = best["img"]
    return best, rec_label


def run_attack(ground_truth, true_label, model, device,
               iters=4000, restarts=3, tv_weight=3e-3, lr=0.1,
               frame_dir=None, log_every=100):
    """Convenience wrapper: compute the true gradient, then invert it."""
    true_grads = compute_true_gradients(ground_truth, true_label, model)
    return invert_from_gradients(
        true_grads, model, ground_truth, device, iters=iters,
        restarts=restarts, tv_weight=tv_weight, lr=lr,
        frame_dir=frame_dir, log_every=log_every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", type=int, default=25)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--tv", type=float, default=3e-3)
    ap.add_argument("--out", default="results/gradient_inversion")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    frame_dir = os.path.join(args.out, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # ---- load one CIFAR-10 image as the client's private sample ----
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN.tolist(), CIFAR_STD.tolist()),
    ])
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    img, label = ds[args.image_id]
    gt = img.unsqueeze(0).to(device)
    gt_label = torch.tensor([label], device=device)

    model = LeNetCIFAR().to(device)
    model.apply(weights_init)
    model.eval()

    print(f"[victim]  private image id={args.image_id} true label={label} "
          f"({ds.classes[label]})")
    t0 = time.time()
    best, rec_label = run_attack(gt, gt_label, model, device,
                                 iters=args.iters, restarts=args.restarts,
                                 tv_weight=args.tv, frame_dir=frame_dir)
    dt = time.time() - t0
    print(f"[attacker] recovered label={rec_label} ({ds.classes[rec_label]}) "
          f"| label correct: {rec_label == label}")

    gt_img = to_image(gt[0])
    rec_img = to_image(best["img"][0])
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(args.out, "ground_truth.png"), gt_img)
    imageio.imwrite(os.path.join(args.out, "reconstruction.png"), rec_img)
    for it, rl, fr in best["frames"]:
        imageio.imwrite(os.path.join(frame_dir, f"frame_{it:05d}.png"), fr)

    final_psnr = psnr(gt_img, rec_img)
    print(f"[result]  final cosine-loss={best['loss']:.4f} "
          f"PSNR={final_psnr:.2f} dB | wall-clock={dt:.1f}s | frames={len(best['frames'])}")
    with open(os.path.join(args.out, "metrics.txt"), "w") as f:
        f.write(f"image_id={args.image_id}\ntrue_label={label}\n"
                f"recovered_label={rec_label}\nlabel_correct={rec_label == label}\n"
                f"final_cosine_loss={best['loss']:.6f}\nPSNR_dB={final_psnr:.4f}\n"
                f"iters={args.iters}\nrestarts={args.restarts}\n"
                f"wall_clock_s={dt:.2f}\ndevice={device}\n")


if __name__ == "__main__":
    main()
