"""
src/renderer.py
───────────────
Fast renderer: orange fire, translucent smoke, red 3D sphere.
No gamma correction (saves expensive per-pixel log/exp).
"""

import taichi as ti
from .fields  import temperature, density, pixels, obstacle, ml_class
from .config  import GRID_W, GRID_H, FIRE_THRESHOLD

W      = GRID_W
H      = GRID_H
_FTHR  = FIRE_THRESHOLD

# Obstacle position (set from sim.py each frame)
obs_cx_f = ti.field(ti.f32, shape=())
obs_cy_f = ti.field(ti.f32, shape=())
obs_r_f  = ti.field(ti.f32, shape=())


@ti.func
def fire_color_ramp(t: float) -> ti.Vector:
    """Orange-dominant fire: deep red → orange → light orange → white core."""
    r = 0.0
    g = 0.0
    b = 0.0
    if t > 0.05:
        r = ti.min(1.0, t * 2.0)
        g = ti.min(0.55, ti.max(0.0, (t - 0.5) * 0.30))
        b = ti.min(0.15, ti.max(0.0, (t - 3.5) * 0.08))
        if t > 6.0:
            boost = ti.min(1.0, (t - 6.0) * 0.12)
            g = g + (0.85 - g) * boost
            b = b + (0.5 - b) * boost * 0.3
    return ti.Vector([r, g, b])


@ti.kernel
def render_pixels():
    ocx = obs_cx_f[None]
    ocy = obs_cy_f[None]
    orad = obs_r_f[None]

    for i, j in pixels:
        if obstacle[i, j] > 0.5:
            # ── Red 3D sphere ─────────────────────────────────────────────
            dx = float(i) - ocx
            dy = float(j) - ocy
            dist = ti.sqrt(dx * dx + dy * dy)
            nd = dist / ti.max(orad, 1.0)

            if nd < 1.0:
                nx = dx / ti.max(orad, 1.0)
                ny = dy / ti.max(orad, 1.0)
                light = ti.max(0.0, -0.5 * nx + 0.6 * ny + 0.5)
                spec = light * light * light * 0.25
                shade = 0.3 + 0.7 * light
                edge = 1.0 - nd * nd
                mix = edge * 0.7 + 0.3
                r = ti.min(1.0, 0.85 * shade * mix + spec)
                g = ti.min(1.0, 0.12 * shade * mix + spec * 0.2)
                b = ti.min(1.0, 0.10 * shade * mix + spec * 0.15)
                pixels[i, j] = ti.Vector([r, g, b])
            else:
                pixels[i, j] = ti.Vector([0.03, 0.03, 0.04])
        else:
            t_val = temperature[i, j]
            d_val = density[i, j]

            # ML classification
            ml = ml_class[i, j]
            p_fire  = ml[1]
            p_smoke = ml[2]
            ml_active = (p_fire + p_smoke) > 0.01

            # Fire
            fire_col = fire_color_ramp(t_val)
            fire_int = ti.min(1.0, t_val * 0.45)

            # Smoke opacity (translucent wisps)
            heat_ratio = ti.min(1.0, t_val / (_FTHR + 0.5))
            smoke_factor = 1.0 - heat_ratio

            r = 0.0
            g = 0.0
            b = 0.0

            if ml_active:
                # ML-guided fire
                r = fire_col[0] * fire_int * ti.max(p_fire, 0.3)
                g = fire_col[1] * fire_int * ti.max(p_fire, 0.3)
                b = fire_col[2] * fire_int * ti.max(p_fire, 0.3)
                # ML-guided smoke
                smoke_op = ti.min(0.30, p_smoke * 0.45) * smoke_factor
                r += 0.12 * smoke_op
                g += 0.10 * smoke_op
                b += 0.08 * smoke_op
            else:
                # Physics fallback with smoke
                smoke_op = ti.min(0.35, d_val * 0.06) * smoke_factor
                r = 0.12 * smoke_op + fire_col[0] * fire_int
                g = 0.10 * smoke_op + fire_col[1] * fire_int
                b = 0.08 * smoke_op + fire_col[2] * fire_int

            # Clamp only (no gamma = faster)
            r = ti.max(0.0, ti.min(r, 1.0))
            g = ti.max(0.0, ti.min(g, 1.0))
            b = ti.max(0.0, ti.min(b, 1.0))

            pixels[i, j] = ti.Vector([r, g, b])
