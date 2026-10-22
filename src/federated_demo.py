"""The hero demonstration, scripted for screen recording.

A single-host simulation of one round of federated learning between a hospital
client and a central server, with an explicit, logged privacy boundary:

    HOSPITAL CLIENT                         CENTRAL SERVER (honest-but-curious)
    - holds one private patient image       - never receives the image
    - computes the gradient of the loss     - receives ONLY the gradient tensor
      on its local data                       (a list of floats)
    - uploads the gradient ------------------> runs the gradient-inversion attack
                                               and reconstructs the private image

The console output is deliberately verbose and staged so it can be captured as
a clean terminal recording; every "frame" of the reconstruction is saved so the
image can be shown re-emerging from noise.

Run:  python src/federated_demo.py --image-id 25 --iters 3000 --restarts 2
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LeNetCIFAR, weights_init  # noqa: E402
from data_util import get_cifar, to_image, CLASSES  # noqa: E402
from metrics import all_metrics  # noqa: E402
from attacks.gradient_inversion import (  # noqa: E402
    compute_true_gradients, invert_from_gradients, infer_label_from_gradient)


def banner(text):
    line = "=" * 74
    print(f"\n{line}\n  {text}\n{line}", flush=True)


def slow(msg, d=0.0):
    print(msg, flush=True)
    if d:
        time.sleep(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", type=int, default=25)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to pause between stages (use >0 when recording)")
    ap.add_argument("--out", default="results/federated_demo")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    frame_dir = os.path.join(args.out, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    banner("FEDERATED LEARNING PRIVACY ATTACK  -  gradient inversion demo")
    slow(f"[setup]   device = {device}", args.pause)
    slow("[setup]   scenario: a hospital trains a shared model via federated "
         "learning.", args.pause)
    slow("[setup]   the hospital is promised: 'your patient data never leaves "
         "the building'.", args.pause)

    xtr, ytr, _, _ = get_cifar()
    gt = xtr[args.image_id:args.image_id + 1].to(device)
    gt_label = ytr[args.image_id:args.image_id + 1].to(device)
    gt_img = to_image(gt[0])
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(args.out, "ground_truth.png"), gt_img)

    banner("STEP 1  -  HOSPITAL CLIENT (holds the private image)")
    slow(f"[client]  loaded 1 private training image  (true class = "
         f"'{CLASSES[int(gt_label.item())]}')", args.pause)
    slow(f"[client]  image tensor shape = {tuple(gt.shape)}  "
         f"({gt.numel()} private pixel values)", args.pause)

    model = LeNetCIFAR().to(device)
    model.apply(weights_init)
    model.eval()
    slow("[client]  received the current global model from the server.",
         args.pause)
    slow("[client]  computing gradient of the loss on the private image ...",
         args.pause)
    true_grads = compute_true_gradients(gt, gt_label, model)
    n_floats = sum(g.numel() for g in true_grads)
    slow(f"[client]  gradient computed: {len(true_grads)} tensors, "
         f"{n_floats} floating-point numbers.", args.pause)
    slow("[client]  uploading ONLY the gradient to the server. "
         "(no pixels are sent)", args.pause)

    banner("STEP 2  -  CENTRAL SERVER (honest-but-curious)")
    slow("[server]  received the gradient upload.", args.pause)
    slow("[server]  the server never saw the image - but gradients are not "
         "nothing.", args.pause)
    slow("[server]  step 2a: recovering the private LABEL analytically (iDLG "
         "trick) ...", args.pause)
    rec_label = infer_label_from_gradient(true_grads[-2])
    ok = "correct" if rec_label == int(gt_label.item()) else "WRONG"
    slow(f"[server]  recovered label = '{CLASSES[rec_label]}'  ({ok})",
         args.pause)
    slow("[server]  step 2b: inverting the gradient to reconstruct the PIXELS "
         "...", args.pause)
    slow(f"[server]  optimising a dummy image to match the uploaded gradient "
         f"({args.iters} iters x {args.restarts} restarts) ...", args.pause)

    t0 = time.time()
    best, _ = invert_from_gradients(
        true_grads, model, gt, device, iters=args.iters,
        restarts=args.restarts, rec_label=rec_label,
        group_consistency=True, frame_dir=frame_dir, log_every=50)
    dt = time.time() - t0

    rec_img = to_image(best["img"][0])
    imageio.imwrite(os.path.join(args.out, "reconstruction.png"), rec_img)
    for it, rl, fr in best["frames"]:
        imageio.imwrite(os.path.join(frame_dir, f"frame_{it:05d}.png"), fr)
    m = all_metrics(rec_img, gt_img)

    banner("RESULT  -  the 'private' image has been reconstructed")
    slow(f"[server]  reconstruction finished in {dt:.1f}s.", args.pause)
    slow(f"[server]  PSNR = {m['psnr_db']:.2f} dB   SSIM = {m['ssim']}   "
         f"LPIPS = {m['lpips']}", args.pause)
    slow(f"[server]  recovered label '{CLASSES[rec_label]}' + the pixels of the "
         f"patient's private image.", args.pause)
    slow("[takeaway] sharing gradients is NOT the same as keeping data private.",
         args.pause)
    slow("[takeaway] this is exactly why the defences in Modules 2-5 exist.",
         args.pause)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        import json
        json.dump({"image_id": args.image_id,
                   "true_class": CLASSES[int(gt_label.item())],
                   "recovered_class": CLASSES[rec_label],
                   "label_correct": rec_label == int(gt_label.item()),
                   "metrics": m, "iters": args.iters,
                   "restarts": args.restarts, "seconds": dt,
                   "gradient_floats_uploaded": int(n_floats)}, f, indent=2)
    print(f"\n[saved]   {args.out}/reconstruction.png  +  {len(best['frames'])} "
          f"frames  +  metrics.json", flush=True)


if __name__ == "__main__":
    main()
