# Breaking Privacy in Federated Learning
### Gradient-Inversion Attack & a Comparative Study of Privacy Defences

**Security and Privacy for Machine Learning (216H17C701) — IA-1 + IA-2**
Suyash Humne · Roll 16014223085 · KJSSE AI&DS · Sem VII

A unified benchmark that (1) reconstructs a client's private training image from
the gradient it shares in federated learning, and (2) measures four privacy
defences from the course's five modules against that same attack.

## What's here
| Path | Purpose |
|---|---|
| `src/models.py` | LeNet (attack) + MLP (tabular) architectures |
| `src/attacks/gradient_inversion.py` | The gradient-inversion attack (cosine + TV + iDLG) |
| `src/attacks/membership_inference.py` | LiRA membership-inference attack + baselines |
| `src/defenses/gradient_defenses.py` | DP noise, local-DP, pruning, quantization |
| `src/federated_demo.py` | Scripted one-round FL demo (the recorded hero attack) |
| `src/run_defense_sweep.py` | Privacy–utility sweep of every defence |
| `src/batch_reconstruction.py` | Multi-image batch reconstruction |
| `src/run_all.py` | Master orchestrator — runs every experiment |
| `src/make_video.py` | Builds the edited demo video (ffmpeg) |
| `src/build_reports.py` | Renders IA-1, IA-2 and study-guide PDFs |
| `results/` | All metrics (JSON), figures, reconstructions, the video |
| `report/` | The three generated PDFs + HTML templates |

## Reproduce
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
# run everything (downloads CIFAR-10 on first run):
.\.venv\Scripts\python -I src\run_all.py
# build the PDFs:
.\.venv\Scripts\python -I src\build_reports.py
```
Single hero demo only: `python src\federated_demo.py --image-id 25`

## Environment
Python 3.13, PyTorch 2.14 (CUDA 12.6), NVIDIA GTX 1650 (4 GB). CPU works but is
slower. All randomness is seeded for reproducibility.

## Data & ethics
CIFAR-10 (public, non-personal; Krizhevsky 2009). Used as a reproducible proxy
for private image data. No real personal or sensitive data is used.

## Key references
Geiping et al., *Inverting Gradients* (NeurIPS 2020) · Zhu et al., *Deep Leakage
from Gradients* (NeurIPS 2019) · Carlini et al., *Membership Inference From First
Principles* (S&P 2022) · Abadi et al., *Deep Learning with Differential Privacy*
(CCS 2016).
