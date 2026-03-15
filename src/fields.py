"""
src/fields.py
─────────────
All Taichi field allocations.

IMPORTANT: this module must be imported AFTER ti.init() is called in main.py.
           Taichi fields are allocated on import.

Grid conventions
────────────────
  Cell (i, j) covers x ∈ [i, i+1),  y ∈ [j, j+1)
  Cell centre at (i + 0.5,  j + 0.5)

Staggered (MAC) velocity:
  vel_x[i, j]  –  u-component on the LEFT   face of cell (i, j)
                   physical position: (i,       j + 0.5)   shape (W+1, H)
  vel_y[i, j]  –  v-component on the BOTTOM face of cell (i, j)
                   physical position: (i + 0.5, j      )   shape (W,   H+1)
"""

import taichi as ti
from .config import GRID_W, GRID_H

W = GRID_W
H = GRID_H

# ── Velocity  –  staggered MAC grid ───────────────────────────────────────────
vel_x     = ti.field(ti.f32, shape=(W + 1, H    ))   # u: left faces
vel_y     = ti.field(ti.f32, shape=(W,     H + 1))   # v: bottom faces
vel_x_tmp = ti.field(ti.f32, shape=(W + 1, H    ))   # advection back-buffer
vel_y_tmp = ti.field(ti.f32, shape=(W,     H + 1))

# ── Pressure / divergence  –  cell-centred ────────────────────────────────────
pressure   = ti.field(ti.f32, shape=(W, H))   # current pressure iterate
pressure_b = ti.field(ti.f32, shape=(W, H))   # Jacobi back-buffer
divergence = ti.field(ti.f32, shape=(W, H))   # ∇·u  (right-hand side)

# ── Scalars  –  cell-centred ──────────────────────────────────────────────────
density       = ti.field(ti.f32, shape=(W, H))   # smoke / soot density
density_tmp   = ti.field(ti.f32, shape=(W, H))
temperature   = ti.field(ti.f32, shape=(W, H))   # heat  (drives buoyancy + color)
temperature_tmp = ti.field(ti.f32, shape=(W, H))

# ── Vorticity  –  cell-centred 2D scalar (curl of velocity) ───────────────────
curl = ti.field(ti.f32, shape=(W, H))

# ── Pixel output ──────────────────────────────────────────────────────────────
pixels = ti.Vector.field(3, ti.f32, shape=(W, H))
