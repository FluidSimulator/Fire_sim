"""
src/renderer.py
───────────────
Color mapping and pixel-buffer render kernel.

Temperature → color ramp:
  0.0        →  black          (no emission)
  0.0–0.4    →  deep ember     (dark crimson glow)
  0.4–1.2    →  deep red       (nascent flame)
  1.2–2.5    →  red-orange     (main flame body)
  2.5–4.2    →  orange-yellow  (high energy zone)
  4.2–7.0    →  yellow-white   (searing core)
  7.0+       →  pure white     (plasma core)

Low temperature + high density → gray smoke.
Optional 3×3 max-neighbourhood bloom adds a radiant glow.
"""

import taichi as ti
from .fields  import temperature, density, pixels
from .config  import GRID_W, GRID_H, BLOOM_ENABLED, GAMMA, FIRE_THRESHOLD

# Scalar constants baked into kernels at JIT compile time
W              = GRID_W
H              = GRID_H
_BLOOM         = BLOOM_ENABLED
_GAMMA         = GAMMA
_FTHR          = FIRE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
#  Color ramp  (pure @ti.func – inlined, no call overhead)
# ─────────────────────────────────────────────────────────────────────────────

@ti.func
def fire_color(t: float, d: float) -> ti.Vector:
    """
    Map (temperature t, density d) → linear-RGB vec3 in [0,1].

    The ramp uses linear blending between hand-tuned anchor colors so the
    transition from black → ember → red → orange → yellow → white looks
    physically plausible and visually immersive.
    """
    r = ti.f32(0.0)
    g = ti.f32(0.0)
    b = ti.f32(0.0)

    if t < 0.0:
        # No emission
        pass
    elif t < 0.40:
        # Black → deep crimson ember
        s = t / 0.40
        r = s * 0.35
        g = s * 0.01
        b = 0.0
    elif t < 1.20:
        # Deep crimson → bright red
        s = (t - 0.40) / 0.80
        r = 0.35 + s * 0.65
        g = 0.01 + s * 0.05
        b = s * 0.02
    elif t < 2.50:
        # Bright red → orange
        s = (t - 1.20) / 1.30
        r = 1.0
        g = 0.06 + s * 0.38
        b = 0.02 * (1.0 - s)
    elif t < 4.20:
        # Orange → yellow-orange
        s = (t - 2.50) / 1.70
        r = 1.0
        g = 0.44 + s * 0.46
        b = s * 0.10
    elif t < 7.00:
        # Yellow-orange → near-white
        s = (t - 4.20) / 2.80
        r = 1.0
        g = 0.90 + s * 0.10
        b = 0.10 + s * 0.80
    else:
        # Plasma white-hot core
        r = 1.0
        g = 1.0
        b = 1.0

    # ── Smoke blend ─────────────────────────────────────────────────────────
    # High density + low temperature → gray/black smoke
    smoke_frac = ti.max(0.0, 1.0 - t / (_FTHR + 1.0))
    smoke_gray = ti.min(d * 0.42, 0.62)
    r = r + smoke_frac * (smoke_gray - r)
    g = g + smoke_frac * (smoke_gray - g)
    b = b + smoke_frac * (smoke_gray - b)

    # ── Clamp ───────────────────────────────────────────────────────────────
    r = ti.max(0.0, ti.min(r, 1.0))
    g = ti.max(0.0, ti.min(g, 1.0))
    b = ti.max(0.0, ti.min(b, 1.0))

    # ── Gamma correction – brightens midtones for a more vivid look ─────────
    r = ti.pow(r, _GAMMA)
    g = ti.pow(g, _GAMMA)
    b = ti.pow(b, _GAMMA)

    return ti.Vector([r, g, b])


# ─────────────────────────────────────────────────────────────────────────────
#  Render kernel
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def render_pixels():
    """
    Fill the pixel buffer.

    If BLOOM_ENABLED: sample a 3×3 neighbourhood and take the maximum
    temperature weighted at 0.72 for off-centre cells.  This creates a
    soft glow / radiant-heat halo around hot regions without a separate
    blur pass.

    Otherwise: direct mapping temperature[i,j] → color.
    """
    for i, j in pixels:
        d_val = density[i, j]

        if ti.static(_BLOOM):
            # Max-neighbourhood bloom: brightest nearby cell bleeds outward
            t_bloom = temperature[i, j]
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    if di != 0 or dj != 0:   # skip centre (already in t_bloom)
                        ni = ti.max(0, ti.min(i + di, W - 1))
                        nj = ti.max(0, ti.min(j + dj, H - 1))
                        t_bloom = ti.max(t_bloom, temperature[ni, nj] * 0.72)
            pixels[i, j] = fire_color(t_bloom, d_val)
        else:
            pixels[i, j] = fire_color(temperature[i, j], d_val)
