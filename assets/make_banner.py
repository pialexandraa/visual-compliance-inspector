"""Generates assets/banner.png (original procedural art). Run: python3 assets/make_banner.py
AI-Generated banner with customizations
"""
import io
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "assets"
W, H = 1280, 640
FONT = ["Helvetica Neue", "Arial", "DejaVu Sans"]

STEPS = [
    ("OpenCV", 118),
    ("Cosmos 3 Reasoner", 206),
    ("SOP Audit Engine", 192),
    ("NVIDIA Nemotron", 184),
]
NEMOCLAW_W = 132


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


C_DEEP = hex_rgb("#041912")
C_FOREST = hex_rgb("#0d4a32")
C_EMERALD = hex_rgb("#1f824e")
C_LEAF = hex_rgb("#65be52")
C_LIME = hex_rgb("#c8e258")


def add_grain(rgb, amount=0.009, seed=29):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amount, size=(rgb.shape[0], rgb.shape[1], 1)).astype(np.float32)
    return np.clip(rgb + noise, 0.0, 1.0)


def build_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = xx / W, yy / H

    base_grad = np.clip(0.35 * nx + 0.25 * (1.0 - ny), 0, 1)
    bg = C_DEEP * (1 - base_grad[..., None]) + C_FOREST * (base_grad[..., None] * 0.65)

    glow = np.exp(-(((nx - 0.82) / 0.45) ** 2 + ((ny - 0.35) / 0.55) ** 2))
    bg += glow[..., None] * C_EMERALD * 0.35

    rng = np.random.default_rng(29)

    calm = np.exp(-(((nx - 0.50) / 0.37) ** 2 + ((ny - 0.50) / 0.26) ** 2))
    vis_mask = 1.0 - 0.82 * calm

    # Layer 1: Out-of-focus bokeh orbs
    bokeh = np.zeros((H, W, 3), dtype=np.float32)
    for _ in range(26):
        px = rng.uniform(0, W)
        py = rng.uniform(0, H)
        r = rng.uniform(22, 62)
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        disc = np.clip(1.0 - d2 / (r * r), 0.0, 1.0) ** 0.65
        col = C_EMERALD if rng.random() < 0.6 else C_LEAF
        bokeh += disc[..., None] * col * rng.uniform(0.05, 0.12)
    bokeh = gaussian_filter(bokeh, sigma=(6, 6, 0))

    # Layer 2 & 3: Mid & fine luminous "green snow" motes
    snow = np.zeros((H, W, 3), dtype=np.float32)
    for count, r_range, alpha_range in [(95, (2.2, 4.8), (0.18, 0.42)), (220, (0.9, 2.0), (0.15, 0.48))]:
        for _ in range(count):
            px = rng.uniform(0, W)
            py = rng.uniform(0, H)
            r = rng.uniform(*r_range)
            x0, x1 = max(0, int(px - r * 3)), min(W, int(px + r * 3) + 1)
            y0, y1 = max(0, int(py - r * 3)), min(H, int(py + r * 3) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            d2 = (xx[y0:y1, x0:x1] - px) ** 2 + (yy[y0:y1, x0:x1] - py) ** 2
            kern = np.exp(-d2 / (2.0 * (r * 0.55) ** 2))
            mix = rng.uniform(0.0, 1.0)
            col = C_LEAF * (1 - mix) + C_LIME * mix
            snow[y0:y1, x0:x1] += kern[..., None] * col * rng.uniform(*alpha_range)

    bg = bg + (bokeh + snow) * vis_mask[..., None]
    return add_grain(np.clip(bg, 0, 1), amount=0.009, seed=29)


def draw_centered_content(ax, title_y=395, sub_y=328, pill_y=212, pill_h=54):
    ax.text(
        W / 2, title_y, "Visual compliance inspector",
        color="white", fontsize=42, fontweight="bold", family=FONT,
        ha="center", va="center", zorder=10
    )
    ax.text(
        W / 2, sub_y, "Product assembly verification pipeline with AI reasoning",
        color="#d4ecd6", fontsize=16.5, family=FONT,
        ha="center", va="center", zorder=10
    )

    arrow_span = 26
    div_pad = 18
    total_w = sum(w for _, w in STEPS) + arrow_span * (len(STEPS) - 1) + div_pad * 2 + NEMOCLAW_W
    cx = (W - total_w) / 2.0
    cy = pill_y

    for i, (label, w) in enumerate(STEPS):
        ax.add_patch(
            FancyBboxPatch(
                (cx, cy), w, pill_h, boxstyle=f"round,pad=0,rounding_size={pill_h/2}",
                fc=(0.02, 0.11, 0.08, 0.62), ec=(0.78, 0.94, 0.58, 0.58),
                lw=1.1, zorder=9
            )
        )
        ax.text(
            cx + w / 2, cy + pill_h / 2 - 0.5, label,
            color="#eef8e3", fontsize=11.5, family=FONT,
            ha="center", va="center", zorder=10
        )
        cx += w
        if i < len(STEPS) - 1:
            ax.annotate(
                "", xy=(cx + arrow_span - 3, cy + pill_h / 2), xytext=(cx + 3, cy + pill_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#cfe35a", lw=1.15, mutation_scale=8),
                zorder=10
            )
            cx += arrow_span

    cx += div_pad
    ax.plot(
        [cx, cx], [cy + 9, cy + pill_h - 9],
        color=(0.80, 0.95, 0.60, 0.42), lw=1.1, zorder=9
    )
    cx += div_pad
    ax.add_patch(
        FancyBboxPatch(
            (cx, cy), NEMOCLAW_W, pill_h, boxstyle=f"round,pad=0,rounding_size={pill_h/2}",
            fc=(0.02, 0.11, 0.08, 0.38), ec=(0.80, 0.95, 0.60, 0.58),
            lw=1.1, ls=(0, (3.5, 2.5)), zorder=9
        )
    )
    ax.text(
        cx + NEMOCLAW_W / 2, cy + pill_h / 2 - 0.5, "NemoClaw",
        color="#eef8e3", fontsize=11.5, family=FONT,
        ha="center", va="center", zorder=10
    )


bg_rgb = build_background()
fig = plt.figure(figsize=(W / 100, H / 100), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.imshow(bg_rgb, extent=[0, W, 0, H], origin="lower", aspect="auto", interpolation="bilinear", zorder=0)
draw_centered_content(ax)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300)
plt.close(fig)
buf.seek(0)

out_path = os.path.join(OUT, "banner.png")
img = Image.open(buf).resize((W, H), Image.Resampling.LANCZOS)
img.save(out_path, optimize=True)
print(f"saved {out_path} ({img.size[0]}x{img.size[1]}px)")