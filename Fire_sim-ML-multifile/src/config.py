"""
src/config.py
"""

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_W         = 128
GRID_H         = 256
DISPLAY_SCALE  = 2          # 2 for speed (3 looks nicer but slower)
DT             = 0.032

# ── Fire source ───────────────────────────────────────────────────────────────
SOURCE_RADIUS       = 30
SOURCE_DENSITY_STR  = 6.0
SOURCE_TEMP_STR     = 18.0
SOURCE_NOISE        = 0.50

# ── Physics ───────────────────────────────────────────────────────────────────
BUOYANCY_STRENGTH   = 2.8
VORTICITY_STRENGTH  = 0.50
DISSIPATION_DENSITY = 0.975
DISSIPATION_TEMP    = 0.993

# ── Cooling ───────────────────────────────────────────────────────────────────
COOLING_FIRE        = 0.978
COOLING_SMOKE       = 0.970
FIRE_THRESHOLD      = 1.5

# ── Pressure solver ───────────────────────────────────────────────────────────
JACOBI_ITERS    = 30
OVER_RELAXATION = 1.00

# ── Rendering ─────────────────────────────────────────────────────────────────
BLOOM_ENABLED   = False     # off for speed
GAMMA           = 0.85

# ── Obstacle ──────────────────────────────────────────────────────────────────
OBSTACLE_ENABLED = True
OBSTACLE_SHAPE   = "circle"
OBSTACLE_CX      = 64
OBSTACLE_CY      = 80
OBSTACLE_RADIUS  = 16
OBSTACLE_RX      = 48
OBSTACLE_RY      = 72
OBSTACLE_RW      = 32
OBSTACLE_RH      = 16

# ── ML – Pressure Solver ─────────────────────────────────────────────────────
ML_ENABLED          = True
ML_COLLECT_FRAMES   = 40
ML_TRAIN_EPOCHS     = 40
ML_LEARNING_RATE    = 0.001
ML_PATCH_SIZE       = 3
ML_HIDDEN_1         = 32
ML_HIDDEN_2         = 16
ML_JACOBI_AFTER_ML  = 15
ML_BATCH_SIZE       = 256
ML_SAMPLES_PER_FRAME = 1500

# ── ML – Fire Spread Classifier ──────────────────────────────────────────────
ML_FIRE_ENABLED         = True
ML_FIRE_COLLECT_FRAMES  = 60
ML_FIRE_TRAIN_EPOCHS    = 50
ML_FIRE_PATCH_SIZE      = 3
ML_FIRE_HIDDEN          = 32
ML_FIRE_LR              = 0.002
ML_FIRE_SAMPLES_PER_FRAME = 2000
ML_FIRE_INFER_EVERY     = 8     # run classifier every N frames (big speed save)
