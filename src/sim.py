"""
src/sim.py
──────────
High-level simulation step: calls kernels in the correct order each frame.

Separated from kernels.py so the pipeline is easy to read at a glance
and easy to modify (e.g. disable vorticity, change solver order, etc.).
"""

from .config import (
    GRID_W, GRID_H,
    SOURCE_RADIUS,
    SOURCE_DENSITY_STR,
    SOURCE_TEMP_STR,
    JACOBI_ITERS,
)
from .kernels import (
    advect_velocity,
    swap_velocity_buffers,
    advect_and_cool_scalars,
    swap_scalar_buffers,
    apply_buoyancy_force,
    compute_curl_field,
    apply_vorticity_confinement,
    compute_divergence,
    jacobi_iteration,
    swap_pressure_buffers,
    subtract_pressure_gradient,
    enforce_boundary_conditions,
    inject_fire,
    reset_all_fields,
)
from .renderer import render_pixels

W = GRID_W
H = GRID_H

# Fixed emitter position (bottom-centre, a few cells above the floor)
_EMITTER_X = W // 2
_EMITTER_Y = 4


def simulation_step() -> None:
    """
    Execute one full simulation frame.

    Order matters:
      1. Source injection      – add new energy/mass
      2. External forces       – buoyancy, vorticity confinement
      3. Pressure projection   – enforce ∇·u = 0 (incompressibility)
      4. Velocity advection    – move the velocity field with itself
      5. Scalar advection      – move density & temperature, apply cooling
      6. Render                – write color to pixel buffer
    """

    # ── 1. Inject fire at the bottom-centre emitter ──────────────────────────
    inject_fire(
        _EMITTER_X, _EMITTER_Y,
        SOURCE_RADIUS,
        SOURCE_DENSITY_STR,
        SOURCE_TEMP_STR,
    )

    # ── 2a. Buoyancy: warm air rises ─────────────────────────────────────────
    apply_buoyancy_force()

    # ── 2b. Vorticity confinement: re-energise swirls ────────────────────────
    compute_curl_field()
    apply_vorticity_confinement()

    # ── 3. Pressure projection (Jacobi / SOR iterations) ─────────────────────
    enforce_boundary_conditions()
    compute_divergence()
    for _ in range(JACOBI_ITERS):
        jacobi_iteration()
        swap_pressure_buffers()
    subtract_pressure_gradient()
    enforce_boundary_conditions()

    # ── 4. Semi-Lagrangian velocity advection ─────────────────────────────────
    advect_velocity()
    swap_velocity_buffers()
    enforce_boundary_conditions()

    # ── 5. Scalar advection + differential cooling ───────────────────────────
    advect_and_cool_scalars()
    swap_scalar_buffers()

    # ── 6. Render to pixel buffer ─────────────────────────────────────────────
    render_pixels()
