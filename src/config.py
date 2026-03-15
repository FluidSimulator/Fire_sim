"""
src/config.py
─────────────
All simulation parameters in one place.
Edit these freely – no other file needs to change.
"""

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_W         = 128        # cells wide
GRID_H         = 256        # cells tall  (tall lets the flame rise naturally)
DISPLAY_SCALE  = 3          # window pixels per grid cell  (2 or 3 recommended)
DT             = 0.032      # time-step in seconds  (keep ≤ 0.04 for stability)

# ── Fire source (bottom-centre emitter) ───────────────────────────────────────
SOURCE_RADIUS       = 24    # emitter radius in cells
SOURCE_DENSITY_STR  = 5.5   # smoke density added per emission call
SOURCE_TEMP_STR     = 12.0  # temperature added per emission call
SOURCE_NOISE        = 0.45  # fractional random noise on source  (0 = uniform)

# ── Physics ───────────────────────────────────────────────────────────────────
BUOYANCY_STRENGTH   = 2.2   # upward acceleration per unit temperature
VORTICITY_STRENGTH  = 0.45  # vorticity confinement epsilon (swirl amplifier)
DISSIPATION_DENSITY = 0.993 # density multiplier per frame   (< 1.0 → fade)
DISSIPATION_TEMP    = 0.987 # temperature multiplier per frame

# ── Cooling ───────────────────────────────────────────────────────────────────
COOLING_FIRE        = 0.956 # cooling factor when temp >  FIRE_THRESHOLD  (fast)
COOLING_SMOKE       = 0.993 # cooling factor when temp <= FIRE_THRESHOLD  (slow)
FIRE_THRESHOLD      = 1.5   # temperature boundary between fire and smoke zones

# ── Pressure solver ───────────────────────────────────────────────────────────
JACOBI_ITERS    = 30        # iterations of the pressure Poisson solve (20–50)
OVER_RELAXATION = 1.80      # SOR factor  (1.0 = pure Jacobi, < 2.0 for stability)

# ── Rendering ─────────────────────────────────────────────────────────────────
BLOOM_ENABLED   = True      # soft max-neighbourhood glow around hot regions
GAMMA           = 0.88      # gamma < 1 brightens midtones (more vivid flames)
