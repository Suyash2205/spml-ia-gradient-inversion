"""Master orchestrator: run every experiment in sequence, log progress, and
never let one failure abort the rest. Safe to launch once and leave overnight.

Progress is written to results/run_all.log and results/run_all_status.json so
the state is inspectable at any time.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "run_all.log")
STATUS = os.path.join(ROOT, "results", "run_all_status.json")

STAGES = [
    ("hero_demo",
     [PY, "-I", "src/federated_demo.py", "--image-id", "25",
      "--iters", "4000", "--restarts", "4"]),
    ("gallery_bird",
     [PY, "-I", "src/attacks/gradient_inversion.py", "--image-id", "6",
      "--iters", "3500", "--restarts", "2", "--out", "results/gallery/img6"]),
    ("gallery_car",
     [PY, "-I", "src/attacks/gradient_inversion.py", "--image-id", "4",
      "--iters", "3500", "--restarts", "2", "--out", "results/gallery/img4"]),
    ("batch_reconstruction",
     [PY, "-I", "src/batch_reconstruction.py", "--sizes", "1", "4", "8",
      "--iters", "4000", "--restarts", "2"]),
    ("defense_sweep",
     [PY, "-I", "src/run_defense_sweep.py", "--image-id", "25",
      "--iters", "2500", "--restarts", "2", "--util-epochs", "8"]),
    ("mia_undefended",
     [PY, "-I", "src/attacks/membership_inference.py", "--shadows", "32",
      "--pool", "5000", "--epochs", "40", "--tag", "undefended"]),
    ("mia_dp_eps1",
     [PY, "-I", "src/attacks/membership_inference.py", "--shadows", "32",
      "--pool", "5000", "--epochs", "40", "--dp-epsilon", "1.0", "--tag", "dp_eps1"]),
    ("mia_dp_eps8",
     [PY, "-I", "src/attacks/membership_inference.py", "--shadows", "32",
      "--pool", "5000", "--epochs", "40", "--dp-epsilon", "8.0", "--tag", "dp_eps8"]),
    ("make_video",
     [PY, "-I", "src/make_video.py"]),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_status(status):
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}
    save_status(status)
    log("=== run_all started ===")
    t_all = time.time()
    for name, cmd in STAGES:
        log(f"--- stage '{name}' starting ---")
        status["stages"][name] = {"state": "running",
                                  "started": time.strftime("%H:%M:%S")}
        save_status(status)
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               timeout=4 * 3600)
            dt = time.time() - t0
            tail = (r.stdout or "")[-1500:]
            errtail = (r.stderr or "")[-800:]
            if r.returncode == 0:
                status["stages"][name] = {"state": "ok", "seconds": round(dt, 1)}
                log(f"--- stage '{name}' OK in {dt:.0f}s ---")
            else:
                status["stages"][name] = {"state": "failed",
                                          "seconds": round(dt, 1),
                                          "returncode": r.returncode,
                                          "stderr_tail": errtail}
                log(f"!!! stage '{name}' FAILED rc={r.returncode} in {dt:.0f}s")
                log(f"    stderr: {errtail}")
            if tail.strip():
                log(f"    stdout tail: {tail.strip()[-600:]}")
        except subprocess.TimeoutExpired:
            status["stages"][name] = {"state": "timeout"}
            log(f"!!! stage '{name}' TIMED OUT")
        except Exception as e:
            status["stages"][name] = {"state": "error", "error": str(e)}
            log(f"!!! stage '{name}' errored: {e}")
        save_status(status)
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["total_minutes"] = round((time.time() - t_all) / 60, 1)
    save_status(status)
    log(f"=== run_all finished in {status['total_minutes']} min ===")


if __name__ == "__main__":
    main()
