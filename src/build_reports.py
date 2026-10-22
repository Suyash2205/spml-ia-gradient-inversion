"""Render the IA-1 report, IA-2 report, and viva study guide to PDF.

Each document is an HTML template in report/templates/ with {{TOKEN}} markers.
This script fills the tokens from the experiment result JSONs (so every number
in the report is the real measured value), then prints each HTML to PDF with
headless Chrome.

Run after the experiments:  python src/build_reports.py
"""
import base64
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TPL = os.path.join(ROOT, "report", "templates")
OUT = os.path.join(ROOT, "report")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def load_json(path, default=None):
    p = os.path.join(RES, path)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def img_b64(path):
    """Embed an image as a data URI so the PDF is self-contained."""
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return ""
    ext = os.path.splitext(full)[1].lstrip(".").lower()
    with open(full, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return f"data:image/{ext};base64,{b}"


def fmt(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_tokens():
    """Assemble every {{TOKEN}} value from the result files."""
    hero = load_json("federated_demo/metrics.json")
    sweep = load_json("defense_sweep/metrics.json")
    batch = load_json("batch_reconstruction/metrics.json")
    mia_u = load_json("membership_inference/metrics_undefended.json")
    mia_1 = load_json("membership_inference/metrics_dp_eps1.json")
    mia_8 = load_json("membership_inference/metrics_dp_eps8.json")

    t = {}
    style_path = os.path.join(TPL, "_style.css")
    t["STYLE"] = open(style_path, encoding="utf-8").read() if os.path.exists(style_path) else ""
    # dataset provenance: SHA-256 and download date of the CIFAR archive
    import datetime
    import hashlib
    tgz = os.path.join(ROOT, "data", "cifar-10-python.tar.gz")
    if os.path.exists(tgz):
        h = hashlib.sha256()
        with open(tgz, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        t["DL_HASH"] = h.hexdigest()
        t["DL_DATE"] = datetime.date.fromtimestamp(
            os.path.getmtime(tgz)).isoformat()
    else:
        t["DL_HASH"] = "n/a"
        t["DL_DATE"] = datetime.date.today().isoformat()
    hm = hero.get("metrics", {})
    t["HERO_PSNR"] = fmt(hm.get("psnr_db"))
    t["HERO_SSIM"] = fmt(hm.get("ssim"), 3)
    t["HERO_LPIPS"] = fmt(hm.get("lpips"), 3)
    t["HERO_CLASS"] = str(hero.get("true_class", "?"))
    t["HERO_REC_CLASS"] = str(hero.get("recovered_class", "?"))
    t["HERO_LABEL_OK"] = "correctly" if hero.get("label_correct") else "incorrectly"
    t["HERO_SECONDS"] = fmt(hero.get("seconds"), 0)
    t["HERO_FLOATS"] = f"{hero.get('gradient_floats_uploaded', 0):,}"
    t["HERO_IMG"] = img_b64("results/federated_demo/reconstruction.png")
    t["HERO_GT"] = img_b64("results/federated_demo/ground_truth.png")
    t["PU_IMG"] = img_b64("results/defense_sweep/privacy_utility.png")

    # MIA table
    def mia_row(label, j):
        r = j.get("results", {}).get("LiRA", {})
        eps = j.get("dp_epsilon_achieved")
        return (f"<tr><td>{label}</td><td>{fmt(eps,2) if eps else '&infin;'}</td>"
                f"<td>{fmt(r.get('auc'),3)}</td>"
                f"<td>{fmt(r.get('tpr_at_0.1pct'),4)}</td>"
                f"<td>{fmt(r.get('tpr_at_1pct'),4)}</td></tr>")
    rows = ""
    if mia_u:
        rows += mia_row("Undefended", mia_u)
    if mia_8:
        rows += mia_row("DP-SGD (&epsilon;&asymp;8)", mia_8)
    if mia_1:
        rows += mia_row("DP-SGD (&epsilon;&asymp;1)", mia_1)
    t["MIA_ROWS"] = rows or "<tr><td colspan=5>pending</td></tr>"
    t["MIA_ROC_U"] = img_b64("results/membership_inference/roc_undefended.png")

    # defence sweep summary table
    dr = ""
    for dname, d in (sweep.get("defenses", {}) or {}).items():
        pts = d.get("points", [])
        if not pts:
            continue
        strong = pts[-1]
        weak = pts[0]
        dr += (f"<tr><td>{dname}</td><td>{d.get('module','')}</td>"
               f"<td>{fmt(weak['reconstruction']['psnr_db'])} &rarr; "
               f"{fmt(strong['reconstruction']['psnr_db'])}</td>"
               f"<td>{fmt(weak['test_accuracy'],3)} &rarr; "
               f"{fmt(strong['test_accuracy'],3)}</td></tr>")
    t["SWEEP_ROWS"] = dr or "<tr><td colspan=4>pending</td></tr>"

    # batch recon table
    br = ""
    for run in (batch.get("runs", []) or []):
        br += (f"<tr><td>{run['batch_size']}</td>"
               f"<td>{fmt(run['mean_psnr_db'])}</td>"
               f"<td>{fmt(run['label_recovery_acc'],2)}</td></tr>")
    t["BATCH_ROWS"] = br or "<tr><td colspan=3>pending</td></tr>"

    # reconstruction gallery
    gal = ""
    for g in sorted(glob.glob(os.path.join(RES, "gallery", "*", "reconstruction.png"))):
        d = os.path.dirname(g)
        gt = os.path.join(d, "ground_truth.png")
        gal += (f"<div class=gal><img src='{img_b64(gt)}'><span>&rarr;</span>"
                f"<img src='{img_b64(g)}'></div>")
    t["GALLERY"] = gal

    # external (non-dataset) photographs — proof the attack generalises
    ext = load_json("external_images/summary.json")
    egal, erows = "", ""
    for r in (ext.get("results", []) or []):
        sub = f"results/external_images/{r['name']}"
        gt = img_b64(f"{sub}/ground_truth.png")
        rc = img_b64(f"{sub}/reconstruction.png")
        egal += (
            "<div style='display:inline-block;text-align:center;margin:6px 10px;"
            "vertical-align:top'>"
            f"<img src='{gt}' style='width:78px;image-rendering:pixelated;"
            "border:1px solid #bbb'>"
            "<span style='margin:0 5px;color:#b55'>&rarr;</span>"
            f"<img src='{rc}' style='width:78px;image-rendering:pixelated;"
            "border:1px solid #bbb'>"
            f"<div class='small'>{r['caption']} &middot; {fmt(r['psnr_db'])} dB</div>"
            "</div>")
        erows += (f"<tr><td>{r['caption']}</td>"
                  f"<td>{'yes' if r.get('label_correct') else 'no'}</td>"
                  f"<td>{fmt(r.get('psnr_db'))}</td>"
                  f"<td>{fmt(r.get('ssim'), 3)}</td></tr>")
    t["EXTERNAL_GALLERY"] = egal
    t["EXTERNAL_ROWS"] = erows or "<tr><td colspan=4>pending</td></tr>"
    return t


def render(template_name, out_pdf, tokens):
    src = os.path.join(TPL, template_name)
    with open(src, encoding="utf-8") as f:
        html = f.read()

    def repl(m):
        return str(tokens.get(m.group(1), m.group(0)))
    html = re.sub(r"\{\{(\w+)\}\}", repl, html)
    filled = os.path.join(OUT, template_name.replace(".tmpl", ""))
    with open(filled, "w", encoding="utf-8") as f:
        f.write(html)
    url = "file:///" + filled.replace("\\", "/")
    prof = os.path.join(os.environ.get("TEMP", "."), "chrome_pdf_profile")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                    f"--user-data-dir={prof}", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", url],
                   timeout=90, capture_output=True)
    ok = os.path.exists(out_pdf)
    print(f"[report] {out_pdf} {'OK' if ok else 'FAILED'}")
    return ok


def main():
    tokens = build_tokens()
    jobs = [
        ("ia1.html.tmpl", "16014223085_Suyash_Humne_SPML_IA1.pdf"),
        ("ia2.html.tmpl", "16014223085_Suyash_Humne_SPML_IA2.pdf"),
        ("study_guide.html.tmpl", "16014223085_Suyash_Humne_SPML_StudyGuide.pdf"),
    ]
    for tpl, pdf in jobs:
        if os.path.exists(os.path.join(TPL, tpl)):
            render(tpl, os.path.join(OUT, pdf), tokens)
        else:
            print(f"[report] template missing: {tpl}")


if __name__ == "__main__":
    main()
