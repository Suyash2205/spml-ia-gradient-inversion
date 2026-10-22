"""Assemble a polished, captioned demonstration video from experiment outputs.

Pure ffmpeg + Pillow (both already installed); no voice-over needed. Builds:
  1. a title card,
  2. a 'setup' explainer card,
  3. the reconstruction converging from noise (with a slow zoom-in + caption),
  4. a ground-truth vs reconstruction reveal,
  5. the privacy-utility defence results,
  6. a closing takeaway card,
and concatenates them into results/video/demo.mp4.

Everything that is missing is skipped gracefully, so a partial run still yields
a video. Run after the experiments:  python src/make_video.py
"""
import glob
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (16, 20, 24)
FG = (235, 238, 240)
ACCENT = (235, 92, 92)
FPS = 25
OUT = "results/video"
WORK = os.path.join(OUT, "_work")


def _font(size, bold=False):
    names = (["segoeuib.ttf", "arialbd.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def card(path, title, lines=None, title_color=FG, subtitle=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 10, H], fill=ACCENT)
    tf = _font(58, bold=True)
    tlines = wrap(d, title, tf, W - 160)
    y = 120
    for ln in tlines:
        d.text((80, y), ln, font=tf, fill=title_color)
        y += 72
    if subtitle:
        sf = _font(30)
        d.text((80, y + 6), subtitle, font=sf, fill=ACCENT)
        y += 60
    if lines:
        bf = _font(32)
        y += 40
        for ln in lines:
            for wl in wrap(d, ln, bf, W - 200):
                d.text((100, y), wl, font=bf, fill=FG)
                y += 48
            y += 12
    img.save(path)


def clip_from_image(png, seconds, out, zoom=False):
    vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x101418,setsar=1"
    if zoom:
        vf += (",zoompan=z='min(zoom+0.0008,1.4)':d=" + str(int(seconds * FPS)) +
               ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720:fps=" + str(FPS))
    ffmpeg(["-loop", "1", "-t", str(seconds), "-i", png, "-vf", vf,
            "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", out])


def caption_png(src_png, dst_png, text):
    """Overlay a caption with a translucent box at the bottom of an image."""
    base = Image.open(src_png).convert("RGB").resize((W, H))
    d = ImageDraw.Draw(base, "RGBA")
    f = _font(34, bold=True)
    lines = wrap(d, text, f, W - 160)
    box_h = 30 + len(lines) * 44
    d.rectangle([0, H - box_h, W, H], fill=(0, 0, 0, 160))
    y = H - box_h + 14
    for ln in lines:
        tw = d.textlength(ln, font=f)
        d.text(((W - tw) / 2, y), ln, font=f, fill=FG)
        y += 44
    base.save(dst_png)


def side_by_side(gt, rec, out_png):
    g = Image.open(gt).convert("RGB").resize((360, 360), Image.NEAREST)
    r = Image.open(rec).convert("RGB").resize((360, 360), Image.NEAREST)
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    canvas.paste(g, (180, 200))
    canvas.paste(r, (740, 200))
    f = _font(34, bold=True)
    d.text((180 + 180 - d.textlength("ORIGINAL (private)", font=f) / 2, 150),
           "ORIGINAL (private)", font=f, fill=FG)
    rec_label = "RECONSTRUCTED by server"
    d.text((740 + 180 - d.textlength(rec_label, font=f) / 2, 150),
           rec_label, font=f, fill=ACCENT)
    tf = _font(48, bold=True)
    d.text((80, 60), "The server never saw this image.", font=tf, fill=FG)
    canvas.save(out_png)


def build():
    os.makedirs(WORK, exist_ok=True)
    segments = []

    # 1. title
    c = os.path.join(WORK, "c_title.png")
    card(c, "Breaking Privacy in Federated Learning",
         subtitle="Gradient-Inversion Attack  |  Security & Privacy for ML",
         lines=["A curious server reconstructs a client's private training",
                "image from the shared gradient alone."])
    s = os.path.join(WORK, "s_title.mp4"); clip_from_image(c, 4, s); segments.append(s)

    # 2. setup
    c = os.path.join(WORK, "c_setup.png")
    card(c, "The Promise", subtitle="\"Your data never leaves the device\"",
         lines=["Federated learning: clients share only GRADIENTS,",
                "never raw data. This demo shows why that promise fails."])
    s = os.path.join(WORK, "s_setup.mp4"); clip_from_image(c, 4.5, s); segments.append(s)

    # 3. reconstruction convergence
    frames = sorted(glob.glob("results/federated_demo/frames/frame_*.png"))
    if not frames:
        frames = sorted(glob.glob("results/gradient_inversion/frames/frame_*.png"))
    if frames:
        cap_dir = os.path.join(WORK, "capframes"); os.makedirs(cap_dir, exist_ok=True)
        for i, fr in enumerate(frames):
            caption_png(fr, os.path.join(cap_dir, f"f_{i:05d}.png"),
                        "Server optimising a dummy image to match the gradient...")
        s = os.path.join(WORK, "s_recon.mp4")
        ffmpeg(["-framerate", "6", "-i", os.path.join(cap_dir, "f_%05d.png"),
                "-vf", f"scale={W}:{H}:flags=neighbor,setsar=1,"
                       f"zoompan=z='min(zoom+0.0006,1.3)':d=1:s={W}x{H}:fps={FPS}",
                "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", s])
        # hold last frame 2s
        last_cap = os.path.join(cap_dir, f"f_{len(frames)-1:05d}.png")
        s2 = os.path.join(WORK, "s_recon_hold.mp4"); clip_from_image(last_cap, 2, s2)
        segments += [s, s2]

    # 4. reveal
    gt = "results/federated_demo/ground_truth.png"
    rec = "results/federated_demo/reconstruction.png"
    if os.path.exists(gt) and os.path.exists(rec):
        c = os.path.join(WORK, "c_reveal.png"); side_by_side(gt, rec, c)
        s = os.path.join(WORK, "s_reveal.mp4"); clip_from_image(c, 5, s); segments.append(s)
        # metrics caption card
        mp = "results/federated_demo/metrics.json"
        if os.path.exists(mp):
            m = json.load(open(mp))
            met = m.get("metrics", {})
            _ssim = met.get('ssim')
            _ssim_s = f"{_ssim:.3f}" if isinstance(_ssim, (int, float)) else "?"
            card(os.path.join(WORK, "c_metrics.png"), "Attack Succeeded",
                 subtitle=f"label recovered: '{m.get('recovered_class','?')}'",
                 lines=[f"PSNR: {met.get('psnr_db','?'):.2f} dB    SSIM: {_ssim_s}",
                        f"reconstructed from {m.get('gradient_floats_uploaded','?')} "
                        f"gradient values in {m.get('seconds',0):.0f}s",
                        "No pixels were ever transmitted."])
            s = os.path.join(WORK, "s_metrics.mp4")
            clip_from_image(os.path.join(WORK, "c_metrics.png"), 5, s); segments.append(s)

    # 5. defences
    pu = "results/defense_sweep/privacy_utility.png"
    if os.path.exists(pu):
        card(os.path.join(WORK, "c_def.png"), "The Defences",
             lines=["Differential Privacy, Local DP, and Compressive Privacy",
                    "degrade the attack - at a measurable cost to accuracy."])
        s = os.path.join(WORK, "s_defcard.mp4")
        clip_from_image(os.path.join(WORK, "c_def.png"), 4, s); segments.append(s)
        cap = os.path.join(WORK, "c_pu.png")
        caption_png(pu, cap, "Privacy-utility trade-off: stronger defence -> "
                             "worse reconstruction, lower accuracy")
        s = os.path.join(WORK, "s_pu.mp4"); clip_from_image(cap, 6, s); segments.append(s)

    # 6. closing
    card(os.path.join(WORK, "c_end.png"), "Sharing gradients is not privacy.",
         subtitle="Suyash Humne  |  16014223085",
         lines=["Attack: Inverting Gradients (Geiping et al., 2020)",
                "Defences: DP-SGD, Local DP, Compressive Privacy"])
    s = os.path.join(WORK, "s_end.mp4")
    clip_from_image(os.path.join(WORK, "c_end.png"), 4.5, s); segments.append(s)

    # concat
    lst = os.path.join(WORK, "list.txt")
    with open(lst, "w") as f:
        for seg in segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")
    os.makedirs(OUT, exist_ok=True)
    final = os.path.join(OUT, "demo.mp4")
    ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(FPS), final])
    print(f"[video] wrote {final}  ({len(segments)} segments)")

    # also a shareable GIF of just the reconstruction
    if frames:
        gif = os.path.join(OUT, "reconstruction.gif")
        ffmpeg(["-framerate", "8", "-i",
                os.path.join(WORK, "capframes", "f_%05d.png"),
                "-vf", "scale=400:400:flags=neighbor", gif])
        print(f"[video] wrote {gif}")


if __name__ == "__main__":
    try:
        build()
    except subprocess.CalledProcessError as e:
        print(f"[video] ffmpeg error: {e}", file=sys.stderr)
        sys.exit(1)
