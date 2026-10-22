"""Gradient-inversion attack on *external* (non-dataset) images.

This proves the attack is a property of federated learning itself, not an
artefact of CIFAR-10: we take ordinary photographs — including a photo of a
person — treat each one as a client's single private training sample, upload
only its gradient, and reconstruct the pixels from that gradient alone.

Sources are standard, public-domain / permissively-licensed test photographs
that ship inside scikit-image (`skimage.data`), so no personal, private, or
scraped data is used (honouring the IA's data-use rule). They are resized to
32x32 to fit the same LeNet client model used throughout the project; the
attack scales to higher resolutions with larger models (Geiping et al., 2020),
which the 4 GB GPU here does not have room for.

Run:  python src/attacks/external_image_attack.py --iters 3000 --restarts 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LeNetCIFAR, weights_init  # noqa: E402
from metrics import all_metrics  # noqa: E402
from attacks.gradient_inversion import (  # noqa: E402
    CIFAR_MEAN, CIFAR_STD, compute_true_gradients, invert_from_gradients,
    to_image)

# (skimage.data name, caption, client-assigned label 0-9 from CIFAR's space)
# The label is arbitrary for an out-of-distribution photo — what matters is that
# the server recovers whatever label the client used, then the pixels.
SOURCES = [
    ("astronaut", "a person (astronaut)", 0),
    ("chelsea",   "a cat",                3),
    ("coffee",    "a coffee cup",         5),
    ("rocket",    "a rocket",             0),
]


def load_photo(name, device):
    """Load a skimage test photo, centre-crop to square, resize to 32x32,
    and normalise with the CIFAR statistics the client model expects."""
    from skimage import data
    from skimage.transform import resize
    img = getattr(data, name)().astype(np.float32) / 255.0
    if img.ndim == 2:  # grayscale -> 3 channels
        img = np.stack([img] * 3, axis=-1)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = resize(img, (32, 32), anti_aliasing=True)
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    mean = CIFAR_MEAN.view(3, 1, 1)
    std = CIFAR_STD.view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--tv", type=float, default=3e-3)
    ap.add_argument("--out", default="results/external_images")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    summary = []
    for name, caption, label in SOURCES:
        sub = os.path.join(args.out, name)
        os.makedirs(sub, exist_ok=True)
        gt = load_photo(name, device)
        gt_label = torch.tensor([label], device=device)

        model = LeNetCIFAR().to(device)
        model.apply(weights_init)
        model.eval()

        print(f"\n[victim]   private photo = '{name}' ({caption}), "
              f"client-assigned label={label}")
        true_grads = compute_true_gradients(gt, gt_label, model)
        t0 = time.time()
        best, rec_label = invert_from_gradients(
            true_grads, model, gt, device, iters=args.iters,
            restarts=args.restarts, tv_weight=args.tv)
        dt = time.time() - t0

        gt_img = to_image(gt[0])
        rec_img = to_image(best["img"][0])
        import imageio.v2 as imageio
        imageio.imwrite(os.path.join(sub, "ground_truth.png"), gt_img)
        imageio.imwrite(os.path.join(sub, "reconstruction.png"), rec_img)

        m = all_metrics(rec_img, gt_img)
        ok = (rec_label == label)
        print(f"[attacker] label recovered={rec_label} ({'correct' if ok else 'wrong'})"
              f" | PSNR={m['psnr_db']:.2f} dB  SSIM={m['ssim']:.3f}  ({dt:.0f}s)")
        rec = {"name": name, "caption": caption, "assigned_label": label,
               "recovered_label": rec_label, "label_correct": ok,
               "seconds": round(dt, 1), **m}
        summary.append(rec)
        with open(os.path.join(sub, "metrics.json"), "w") as f:
            json.dump(rec, f, indent=2)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"source": "skimage.data (public-domain test photographs)",
                   "note": "resized to 32x32; same LeNet client model as the "
                           "rest of the project",
                   "iters": args.iters, "restarts": args.restarts,
                   "device": device, "results": summary}, f, indent=2)
    print(f"\n[done] {len(summary)} external images attacked -> {args.out}")


if __name__ == "__main__":
    main()
