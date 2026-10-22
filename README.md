# Breaking Privacy in Federated Learning
### Gradient-Inversion Attack & a Comparative Study of Privacy Defences

**Security and Privacy for Machine Learning (216H17C701) — IA-1 + IA-2**
Suyash Humne · Roll 16014223085 · KJSSE AI&DS · Sem VII

A unified benchmark that **(1)** reconstructs a client's private training image from
the gradient it shares in federated learning, and **(2)** measures four privacy
defences from the course's five modules against that same attack.

> **In one line:** federated learning promises *"only gradients are shared, your
> data never leaves the device."* This project shows a curious server can rebuild
> the private image from that gradient alone — then measures how well each
> syllabus defence (Differential Privacy, Local DP, anonymity/aggregation,
> Compressive Privacy) stops it.

---

## Quick start — run the live attack in 2 steps

**Step 1 — install** (one-time). Pick the line matching the machine:

```bash
# CPU-only (most laptops):
pip install torch torchvision

# OR NVIDIA GPU, CUDA 12.6:
pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# then the rest (both cases):
pip install -r requirements.txt
```

<details>
<summary>Recommended: do it inside a virtual environment</summary>

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision
.\.venv\Scripts\pip install -r requirements.txt
```
```bash
# macOS / Linux
python3 -m venv .venv
. .venv/bin/activate
pip install torch torchvision && pip install -r requirements.txt
```
</details>

**Step 2 — run the attack.** CIFAR-10 downloads automatically on the first run (~170 MB, once).

```bash
python src/federated_demo.py --image-id 25
```

You'll watch it narrate each step and end with the reconstructed image:

```
[client]  loaded 1 private training image  (true class = 'frog')
[server]  the server never saw the image - but gradients are not nothing.
[server]  step 2a: recovering the private LABEL analytically (iDLG trick) ...
[server]  recovered label = 'frog'  (correct)
[server]  step 2b: inverting the gradient to reconstruct the PIXELS ...
  RESULT  -  the 'private' image has been reconstructed
[server]  PSNR = 24.53 dB   SSIM = 0.90
[takeaway] sharing gradients is NOT the same as keeping data private.
```

Then open the two images side by side to see it worked:

| | |
|---|---|
| `results/federated_demo/ground_truth.png`   | the **original** private image |
| `results/federated_demo/reconstruction.png` | what the **server rebuilt** from the gradient |

**Runtime:** ~1–2 min on an NVIDIA GPU, a few minutes on CPU. For a faster live
demo use fewer iterations (quality is slightly lower but the attack still works):

```bash
python src/federated_demo.py --image-id 25 --iters 800 --restarts 1   # ~30-60s
```

---

## How to use — try it on different images

The attack picks one image out of CIFAR-10 by its index with `--image-id N`. The
true class is detected automatically, so you can target any category. Examples:

| Command | Target image |
|---|---|
| `python src/federated_demo.py --image-id 29` | airplane |
| `python src/federated_demo.py --image-id 4`  | automobile |
| `python src/federated_demo.py --image-id 6`  | bird |
| `python src/federated_demo.py --image-id 9`  | cat |
| `python src/federated_demo.py --image-id 3`  | deer |
| `python src/federated_demo.py --image-id 27` | dog |
| `python src/federated_demo.py --image-id 0`  | frog |
| `python src/federated_demo.py --image-id 7`  | horse |
| `python src/federated_demo.py --image-id 8`  | ship |
| `python src/federated_demo.py --image-id 1`  | truck |

Any index from `0` to `49999` works — pick a random number and it'll attack
whatever private image sits there.

### Options you can change
| Flag | Default | What it does |
|---|---|---|
| `--image-id N` | `25` | which private image (0–49999) to attack |
| `--iters N`    | `3000` | optimisation steps — more = sharper, slower |
| `--restarts N` | `2` | retries from fresh noise, best one is kept |
| `--out PATH`   | `results/federated_demo` | where images + `metrics.json` are saved |

> **Tip:** each run overwrites `results/federated_demo/`. To keep several side by
> side, give each its own folder, e.g. `--out results/demo_cat --image-id 9`.

### The same attack, as a plain tool (no narration)
```bash
python src/attacks/gradient_inversion.py --image-id 6 --out results/my_bird
```

---

## Reproduce *everything* in the reports

```bash
python src/run_all.py        # runs all 9 experiments (attack, defence sweep, membership inference, video)
python src/build_reports.py  # regenerates the three PDFs from the fresh numbers
```
All randomness is seeded, so the numbers in the reports come back the same.

---

## What's in this repo
| Path | Purpose |
|---|---|
| `src/attacks/gradient_inversion.py` | The gradient-inversion attack (cosine + TV + iDLG) |
| `src/attacks/membership_inference.py` | LiRA membership-inference attack + baselines |
| `src/defenses/gradient_defenses.py` | DP noise, local-DP, pruning, quantization |
| `src/federated_demo.py` | Scripted one-round FL demo (the recorded hero attack) |
| `src/run_defense_sweep.py` | Privacy–utility sweep of every defence |
| `src/batch_reconstruction.py` | Multi-image batch reconstruction |
| `src/run_all.py` | Master orchestrator — runs every experiment |
| `src/make_video.py` | Builds the edited demo video (ffmpeg) |
| `src/build_reports.py` | Renders IA-1, IA-2 and study-guide PDFs |
| `results/` | All metrics (JSON), figures, reconstructions, the demo video |
| `report/` | The three generated PDFs + HTML templates |

## Environment
Built on Python 3.13, PyTorch 2.14 (CUDA 12.6), NVIDIA GTX 1650 (4 GB). Runs on
CPU too (slower). `.venv/` and `data/` are not committed — they are recreated by
the install step and the first run, respectively.

## Data & ethics
CIFAR-10 (public, non-personal; Krizhevsky 2009), used as a reproducible proxy
for private image data. No real personal or sensitive data is used.

## Key references
Geiping et al., *Inverting Gradients* (NeurIPS 2020) · Zhu et al., *Deep Leakage
from Gradients* (NeurIPS 2019) · Carlini et al., *Membership Inference From First
Principles* (S&P 2022) · Abadi et al., *Deep Learning with Differential Privacy*
(CCS 2016).
