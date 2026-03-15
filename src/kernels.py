"""
src/kernels.py
──────────────
All @ti.kernel and @ti.func definitions for the 2-D Eulerian fire solver.

Pipeline each frame:
  inject_fire  →  apply_buoyancy_force  →  compute_curl_field
  →  apply_vorticity_confinement  →  enforce_boundary_conditions
  →  compute_divergence  →  [jacobi_iteration × N]
  →  subtract_pressure_gradient  →  enforce_boundary_conditions
  →  advect_velocity  →  swap_velocity_buffers
  →  advect_and_cool_scalars  →  swap_scalar_buffers
"""

import taichi as ti

from .fields import (
    vel_x, vel_y, vel_x_tmp, vel_y_tmp,
    pressure, pressure_b, divergence,
    density, density_tmp,
    temperature, temperature_tmp,
    curl,
)
from .config import (
    GRID_W, GRID_H,
    DT,
    BUOYANCY_STRENGTH,
    VORTICITY_STRENGTH,
    DISSIPATION_DENSITY,
    DISSIPATION_TEMP,
    COOLING_FIRE,
    COOLING_SMOKE,
    FIRE_THRESHOLD,
    OVER_RELAXATION,
    SOURCE_NOISE,
)

# Module-level aliases baked into kernels at JIT-compile time
W  = GRID_W
H  = GRID_H
_DT   = DT
_BUOY = BUOYANCY_STRENGTH
_VORT = VORTICITY_STRENGTH
_DISS_D = DISSIPATION_DENSITY
_DISS_T = DISSIPATION_TEMP
_COOL_F = COOLING_FIRE
_COOL_S = COOLING_SMOKE
_FTHR   = FIRE_THRESHOLD
_SOR    = OVER_RELAXATION
_NOISE  = SOURCE_NOISE


# ─────────────────────────────────────────────────────────────────────────────
#  Bilinear interpolation helpers  (inlined by @ti.func → zero call overhead)
# ─────────────────────────────────────────────────────────────────────────────

@ti.func
def bilerp_vx(px: float, py: float) -> float:
    """
    Sample vel_x at world position (px, py).
    vel_x[i,j] is located at (i, j+0.5), so:
      x-grid lines are at integer positions 0 … W
      y-grid lines are at half-integer positions 0.5 … H-0.5
    """
    xi  = int(ti.floor(px))
    yf  = py - 0.5
    yi  = int(ti.floor(yf))
    tx  = px - ti.floor(px)
    ty  = yf  - ti.floor(yf)
    xi  = ti.max(0, ti.min(xi, W - 1))
    yi  = ti.max(0, ti.min(yi, H - 2))
    v00 = vel_x[xi,     yi    ]
    v10 = vel_x[xi + 1, yi    ]
    v01 = vel_x[xi,     yi + 1]
    v11 = vel_x[xi + 1, yi + 1]
    return (1.0-tx)*(1.0-ty)*v00 + tx*(1.0-ty)*v10 + (1.0-tx)*ty*v01 + tx*ty*v11


@ti.func
def bilerp_vy(px: float, py: float) -> float:
    """
    Sample vel_y at world position (px, py).
    vel_y[i,j] is located at (i+0.5, j), so:
      x-grid lines are at half-integer positions 0.5 … W-0.5
      y-grid lines are at integer positions 0 … H
    """
    xf  = px - 0.5
    xi  = int(ti.floor(xf))
    yi  = int(ti.floor(py))
    tx  = xf  - ti.floor(xf)
    ty  = py  - ti.floor(py)
    xi  = ti.max(0, ti.min(xi, W - 2))
    yi  = ti.max(0, ti.min(yi, H - 1))
    v00 = vel_y[xi,     yi    ]
    v10 = vel_y[xi + 1, yi    ]
    v01 = vel_y[xi,     yi + 1]
    v11 = vel_y[xi + 1, yi + 1]
    return (1.0-tx)*(1.0-ty)*v00 + tx*(1.0-ty)*v10 + (1.0-tx)*ty*v01 + tx*ty*v11


@ti.func
def bilerp_cell(field: ti.template(), px: float, py: float) -> float:
    """
    Bilinear sample of a cell-centred field.
    Cell (i,j) centred at (i+0.5, j+0.5).
    """
    xf  = px - 0.5
    yf  = py - 0.5
    xi  = int(ti.floor(xf))
    yi  = int(ti.floor(yf))
    tx  = xf - ti.floor(xf)
    ty  = yf - ti.floor(yf)
    xi  = ti.max(0, ti.min(xi, W - 2))
    yi  = ti.max(0, ti.min(yi, H - 2))
    v00 = field[xi,     yi    ]
    v10 = field[xi + 1, yi    ]
    v01 = field[xi,     yi + 1]
    v11 = field[xi + 1, yi + 1]
    return (1.0-tx)*(1.0-ty)*v00 + tx*(1.0-ty)*v10 + (1.0-tx)*ty*v01 + tx*ty*v11


@ti.func
def vel_at_center(i: int, j: int) -> ti.Vector:
    """Average the four surrounding staggered values to get velocity at cell centre (i,j)."""
    u = 0.5 * (vel_x[i, j] + vel_x[i + 1, j])
    v = 0.5 * (vel_y[i, j] + vel_y[i, j + 1])
    return ti.Vector([u, v])


# ─────────────────────────────────────────────────────────────────────────────
#  Step A – Semi-Lagrangian velocity advection
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def advect_velocity():
    """
    Advect vel_x and vel_y using backward-trace from each face centre.
    Reads from vel_x/vel_y and writes to vel_x_tmp/vel_y_tmp (double-buffer).

    u-faces (vel_x): centre at (i, j+0.5)
    v-faces (vel_y): centre at (i+0.5, j)
    """
    # ── u component ─────────────────────────────────────────────────────────
    for i, j in ti.ndrange((1, W), H):
        cx = float(i)
        cy = float(j) + 0.5
        u  = vel_x[i, j]
        # Cross-component at this face: average surrounding v values
        v  = 0.25 * (vel_y[i-1, j  ] + vel_y[i-1, j+1]
                   + vel_y[i,   j  ] + vel_y[i,   j+1])
        # Clamp backward-trace position to grid interior
        px = ti.max(1.0,          ti.min(cx - _DT * u, float(W) - 1.0))
        py = ti.max(0.5 + 1e-4,   ti.min(cy - _DT * v, float(H) - 0.5 - 1e-4))
        vel_x_tmp[i, j] = bilerp_vx(px, py)

    # ── v component ─────────────────────────────────────────────────────────
    for i, j in ti.ndrange(W, (1, H)):
        cx = float(i) + 0.5
        cy = float(j)
        u  = 0.25 * (vel_x[i,   j-1] + vel_x[i+1, j-1]
                   + vel_x[i,   j  ] + vel_x[i+1, j  ])
        v  = vel_y[i, j]
        px = ti.max(0.5 + 1e-4,   ti.min(cx - _DT * u, float(W) - 0.5 - 1e-4))
        py = ti.max(1.0,          ti.min(cy - _DT * v, float(H) - 1.0))
        vel_y_tmp[i, j] = bilerp_vy(px, py)


@ti.kernel
def swap_velocity_buffers():
    """Copy tmp → current, clear tmp (single pass to reduce memory traffic)."""
    for i, j in ti.ndrange(W + 1, H):
        vel_x[i, j]    = vel_x_tmp[i, j]
        vel_x_tmp[i, j] = 0.0
    for i, j in ti.ndrange(W, H + 1):
        vel_y[i, j]    = vel_y_tmp[i, j]
        vel_y_tmp[i, j] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Step B – Scalar advection: density + temperature  (one combined kernel)
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def advect_and_cool_scalars():
    """
    Semi-Lagrangian advection of density and temperature.
    Applied in one kernel to share the velocity evaluation cost.

    After advection applies:
      • Global dissipation multiplier  (slow uniform decay)
      • Differential cooling:  fire (hot) cools fast, smoke cools slowly
    """
    for i, j in ti.ndrange(W, H):
        cx = float(i) + 0.5
        cy = float(j) + 0.5
        uv = vel_at_center(i, j)

        px = ti.max(0.5 + 1e-4, ti.min(cx - _DT * uv[0], float(W) - 0.5 - 1e-4))
        py = ti.max(0.5 + 1e-4, ti.min(cy - _DT * uv[1], float(H) - 0.5 - 1e-4))

        d = bilerp_cell(density,     px, py) * _DISS_D
        t = bilerp_cell(temperature, px, py) * _DISS_T

        # Differential cooling: fire cools fast, residual smoke lingers
        cool_rate = ti.select(t > _FTHR, _COOL_F, _COOL_S)
        t *= cool_rate

        density_tmp[i, j]     = d
        temperature_tmp[i, j] = t


@ti.kernel
def swap_scalar_buffers():
    """Commit the advected scalars."""
    for i, j in ti.ndrange(W, H):
        density[i, j]     = density_tmp[i, j]
        temperature[i, j] = temperature_tmp[i, j]


# ─────────────────────────────────────────────────────────────────────────────
#  Step C – Buoyancy force  (warm air rises)
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def apply_buoyancy_force():
    """
    F_buoyancy = α · T · Δt  in the +y direction.
    Applied to vertical-velocity faces (vel_y), averaged over the two
    cells sharing each face so the force is smooth.
    """
    for i, j in ti.ndrange(W, (1, H)):
        t_avg = 0.5 * (temperature[i, j-1] + temperature[i, j])
        vel_y[i, j] += _BUOY * t_avg * _DT


# ─────────────────────────────────────────────────────────────────────────────
#  Step D – Vorticity confinement  (re-energise swirls)
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def compute_curl_field():
    """
    2D vorticity / curl:   ω = ∂v/∂x − ∂u/∂y   (scalar in 2D)
    Central differences on the staggered velocity grid.
    """
    for i, j in ti.ndrange((1, W-1), (1, H-1)):
        dv_dx = 0.5 * (vel_y[i+1, j] - vel_y[i-1, j])
        du_dy = 0.5 * (vel_x[i, j+1] - vel_x[i, j-1])
        curl[i, j] = dv_dx - du_dy


@ti.kernel
def apply_vorticity_confinement():
    """
    Vorticity confinement counters numerical dissipation by amplifying swirls.

    F_conf = ε · ω · N̂⊥
    where  N̂ = ∇|ω| / |∇|ω||   (points toward increasing vorticity magnitude)
    and    N̂⊥ is the perpendicular direction (rotated 90°)

    In 2D this resolves to:
      Fx =  ε · ω · ny
      Fy = −ε · ω · nx
    """
    for i, j in ti.ndrange((1, W-1), (1, H-1)):
        # Gradient of |ω| using central differences
        eta_x = 0.5 * (ti.abs(curl[i+1, j]) - ti.abs(curl[i-1, j]))
        eta_y = 0.5 * (ti.abs(curl[i, j+1]) - ti.abs(curl[i, j-1]))
        mag   = ti.sqrt(eta_x * eta_x + eta_y * eta_y) + 1e-6
        nx = eta_x / mag
        ny = eta_y / mag

        omega = curl[i, j]
        fx    =  _VORT * _DT * ny * omega
        fy    = -_VORT * _DT * nx * omega

        # Distribute force conservatively to all four surrounding faces
        vel_x[i,     j] += fx * 0.5
        vel_x[i + 1, j] += fx * 0.5
        vel_y[i, j    ] += fy * 0.5
        vel_y[i, j + 1] += fy * 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Step E – Incompressible projection  (Jacobi pressure solve)
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def compute_divergence():
    """
    div(u)[i,j] = (u_{i+1,j} − u_{i,j}) + (v_{i,j+1} − v_{i,j})
    Grid spacing dx = 1 throughout.
    """
    for i, j in ti.ndrange(W, H):
        divergence[i, j] = (vel_x[i+1, j] - vel_x[i, j]
                          + vel_y[i, j+1] - vel_y[i, j])


@ti.kernel
def jacobi_iteration():
    """
    One SOR-Jacobi step for  ∇²p = div(u):
      p_new[i,j] = (p[i±1,j] + p[i,j±1] − div[i,j]) / 4
    with over-relaxation:
      p_b[i,j]   = p[i,j] + ω · (p_new − p[i,j])

    Reads from pressure, writes to pressure_b (pure Jacobi → embarrassingly parallel).
    Boundary cells (i=0, i=W-1, j=0, j=H-1) remain 0 (Dirichlet BC).
    """
    for i, j in ti.ndrange((1, W-1), (1, H-1)):
        p_sum = (pressure[i-1, j] + pressure[i+1, j]
               + pressure[i, j-1] + pressure[i, j+1])
        p_new = (p_sum - divergence[i, j]) * 0.25
        pressure_b[i, j] = pressure[i, j] + _SOR * (p_new - pressure[i, j])


@ti.kernel
def swap_pressure_buffers():
    """Commit the new pressure iterate."""
    for i, j in ti.ndrange(W, H):
        pressure[i, j] = pressure_b[i, j]


@ti.kernel
def subtract_pressure_gradient():
    """
    Make velocity divergence-free by subtracting ∇p:
      vel_x[i,j] −= p[i,j] − p[i-1,j]   (interior x-faces)
      vel_y[i,j] −= p[i,j] − p[i,j-1]   (interior y-faces)
    """
    for i, j in ti.ndrange((1, W), H):
        vel_x[i, j] -= pressure[i, j] - pressure[i-1, j]
    for i, j in ti.ndrange(W, (1, H)):
        vel_y[i, j] -= pressure[i, j] - pressure[i, j-1]


# ─────────────────────────────────────────────────────────────────────────────
#  Step F – Boundary conditions  (no-slip walls on all four sides)
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def enforce_boundary_conditions():
    """
    Zero normal velocity on all four walls:
      Left / right walls  → vel_x = 0 on boundary faces
      Bottom / top walls  → vel_y = 0 on boundary faces
    """
    for j in range(H):
        vel_x[0, j] = 0.0
        vel_x[W, j] = 0.0
    for i in range(W):
        vel_y[i, 0] = 0.0
        vel_y[i, H] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Source injection  (fire emitter)
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def inject_fire(cx: int, cy: int, radius: int, d_str: float, t_str: float):
    """
    Inject density + temperature inside a circle at (cx, cy) with given radius.

    Falloff: quadratic  →  concentrated at centre, fading at edges.
    Noise:   per-cell random variation for organic, flickering look.
    Kick:    small upward velocity impulse proportional to heat.
    """
    for i, j in ti.ndrange(W, H):
        dx   = float(i) - float(cx)
        dy   = float(j) - float(cy)
        dist = ti.sqrt(dx*dx + dy*dy)
        if dist < float(radius):
            falloff = 1.0 - dist / float(radius)
            falloff = falloff * falloff                           # quadratic
            noise   = 1.0 + _NOISE * (ti.random(ti.f32)*2.0 - 1.0)
            scale   = falloff * noise * _DT * 30.0

            density[i, j]     += d_str * scale
            temperature[i, j] += t_str * scale
            # Small upward impulse to kick the flame
            vel_y[i, j + 1]   += t_str * 0.055 * falloff * _DT * 30.0


# ─────────────────────────────────────────────────────────────────────────────
#  Reset
# ─────────────────────────────────────────────────────────────────────────────

@ti.kernel
def reset_all_fields():
    """Zero every field – called on startup and Space key."""
    for i, j in ti.ndrange(W + 1, H):
        vel_x[i, j]     = 0.0
        vel_x_tmp[i, j] = 0.0
    for i, j in ti.ndrange(W, H + 1):
        vel_y[i, j]     = 0.0
        vel_y_tmp[i, j] = 0.0
    for i, j in ti.ndrange(W, H):
        pressure[i, j]      = 0.0
        pressure_b[i, j]    = 0.0
        divergence[i, j]    = 0.0
        density[i, j]       = 0.0
        density_tmp[i, j]   = 0.0
        temperature[i, j]   = 0.0
        temperature_tmp[i, j] = 0.0
        curl[i, j]          = 0.0
