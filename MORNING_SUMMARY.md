# Good morning, Suyash — it's all done ✅

Everything is finished, verified, and ready to show. Here's what you have and the results.

## What to submit / show
All in `E:\Extra work\spml-ia\`:

| File | What it is |
|---|---|
| `report\16014223085_Suyash_Humne_SPML_IA1.pdf` | **IA-1 report** (problem, 5-module map, dataset, working attack, plan) — 9 pages |
| `report\16014223085_Suyash_Humne_SPML_IA2.pdf` | **IA-2 report** (full implementation, all defences, evaluation, conclusions) — 8 pages |
| `report\16014223085_Suyash_Humne_SPML_StudyGuide.pdf` | **Your viva prep** — every concept explained + 14 likely questions with answers |
| `results\video\demo.mp4` | **The edited demo video** (38s, captioned, no voice needed) — show this |
| `results\video\reconstruction.gif` | The attack converging, as a GIF |

I've pushed the 3 PDFs and the video to your other device too.

## The project (one line)
I showed federated learning's "only gradients are shared, so your data is private" claim is false — a curious server reconstructs a client's private image from the gradient alone — then measured every syllabus defence against that attack.

## The headline results (all real, all reproducible)
1. **The attack works:** reconstructed a private "frog" image at **24.5 dB PSNR / SSIM 0.90**, label recovered correctly, from just **19,438 gradient values** — no pixels ever sent.
2. **Defences vs the attack (privacy–utility):**
   - **Differential Privacy (noise):** destroys reconstruction (PSNR 24→7) — accuracy cost 60%→32%.
   - **Pruning (compressive):** most efficient — keeps **57% accuracy at 90% pruning** while wrecking reconstruction.
   - **Quantization:** only 2-bit defends; 4–8 bit the attack survives.
3. **Membership inference + DP audit:** undefended **LiRA AUC 0.89 (TPR@1%FPR = 17.6%)** → under DP-SGD it collapses to **AUC 0.55 (ε≈8)** and **0.51 (ε≈1)** — the DP guarantee empirically verified.

## How it covers all 5 modules
M1 threat model + attack · M2 Differential Privacy · M3 Local DP · M4 anonymity/aggregation · M5 Compressive privacy (pruning + quantization). Every module is a real experiment, not just a paragraph.

## One tiny thing to confirm
I used **Batch B2** on the cover (your most recent files said B2; an older one said A3). If it should be A3, tell me and I'll regenerate the PDFs in seconds.

## To re-run or demo live
```powershell
cd "E:\Extra work\spml-ia"
.\.venv\Scripts\python -I src\federated_demo.py   # the live attack
```
Full details in `README.md`. All code is in `src\`, all results in `results\`.

## Honest notes (so nothing surprises you in the viva)
- The attack model uses Sigmoid activations (best for reconstruction); utility/membership experiments use a trainable ReLU model — this is explained in the reports and study guide.
- Small model + CIFAR-10 (public, non-personal data used deliberately for ethics). Absolute numbers scale, trade-offs hold.
- Batch reconstruction degrades past batch size 1 (label recovery 100%→37%) — shown honestly as a limitation/finding.

You missed the lectures, so **read the study guide once** — it's written so you can defend every choice as your own. Good luck. 🍀
