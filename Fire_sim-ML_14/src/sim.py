"""
src/sim.py
──────────
Simulation with ML pressure solver + ML fire classifier.
Classifier runs every N frames for performance.
"""

import numpy as np

from .config import (
    GRID_W, GRID_H,
    SOURCE_RADIUS, SOURCE_DENSITY_STR, SOURCE_TEMP_STR,
    JACOBI_ITERS,
    OBSTACLE_ENABLED, OBSTACLE_SHAPE,
    OBSTACLE_CX, OBSTACLE_CY, OBSTACLE_RADIUS,
    OBSTACLE_RX, OBSTACLE_RY, OBSTACLE_RW, OBSTACLE_RH,
    ML_ENABLED, ML_COLLECT_FRAMES, ML_TRAIN_EPOCHS,
    ML_LEARNING_RATE, ML_PATCH_SIZE, ML_HIDDEN_1, ML_HIDDEN_2,
    ML_JACOBI_AFTER_ML, ML_BATCH_SIZE, ML_SAMPLES_PER_FRAME,
    ML_FIRE_ENABLED, ML_FIRE_COLLECT_FRAMES, ML_FIRE_TRAIN_EPOCHS,
    ML_FIRE_HIDDEN, ML_FIRE_LR, ML_FIRE_SAMPLES_PER_FRAME,
    ML_FIRE_INFER_EVERY,
)
from .kernels import (
    advect_velocity, swap_velocity_buffers,
    advect_and_cool_scalars, swap_scalar_buffers,
    apply_buoyancy_force, compute_curl_field, apply_vorticity_confinement,
    compute_divergence, jacobi_iteration, swap_pressure_buffers,
    subtract_pressure_gradient, enforce_boundary_conditions,
    inject_fire, reset_all_fields, clamp_velocity, reset_pressure,
    init_obstacle_circle, init_obstacle_rect, clear_obstacle,
)
from .fields import (
    divergence, pressure, pressure_b,
    temperature, density, vel_x, vel_y, ml_class,
)
from .renderer import render_pixels, obs_cx_f, obs_cy_f, obs_r_f

W = GRID_W
H = GRID_H
_EMITTER_X = W // 2
_EMITTER_Y = 4

# ── ML instances ──────────────────────────────────────────────────────────────
ml_solver = None
fire_classifier = None
_frame_count = 0

# Current obstacle position (updated from main.py)
_obs_cx = OBSTACLE_CX
_obs_cy = OBSTACLE_CY
_obs_r  = OBSTACLE_RADIUS

if ML_ENABLED:
    try:
        from .ml_solver import NeuralPressureSolver
        ml_solver = NeuralPressureSolver(
            patch_size=ML_PATCH_SIZE, hidden1=ML_HIDDEN_1,
            hidden2=ML_HIDDEN_2, lr=ML_LEARNING_RATE,
        )
        print(f"  [ML] Pressure Solver: {ML_PATCH_SIZE**2}->{ML_HIDDEN_1}->{ML_HIDDEN_2}->1")
    except Exception as e:
        print(f"  [ML] Pressure solver init failed: {e}")

if ML_FIRE_ENABLED:
    try:
        from .ml_fire_classifier import FireSpreadClassifier
        fire_classifier = FireSpreadClassifier(hidden=ML_FIRE_HIDDEN, lr=ML_FIRE_LR)
        print(f"  [ML] Fire Classifier: 20->{ML_FIRE_HIDDEN}->3 (softmax)")
        print(f"       Runs every {ML_FIRE_INFER_EVERY} frames for speed.\n")
    except Exception as e:
        print(f"  [ML] Classifier init failed: {e}")


# ── Obstacle ──────────────────────────────────────────────────────────────────

def init_obstacle():
    clear_obstacle()
    if OBSTACLE_ENABLED:
        if OBSTACLE_SHAPE == "circle":
            init_obstacle_circle(OBSTACLE_CX, OBSTACLE_CY, OBSTACLE_RADIUS)
        elif OBSTACLE_SHAPE == "rect":
            init_obstacle_rect(OBSTACLE_RX, OBSTACLE_RY, OBSTACLE_RW, OBSTACLE_RH)

def move_obstacle_to(cx, cy, radius):
    global _obs_cx, _obs_cy, _obs_r
    clear_obstacle()
    cx = max(radius + 1, min(cx, W - radius - 1))
    cy = max(radius + 1, min(cy, H - radius - 1))
    init_obstacle_circle(cx, cy, radius)
    _obs_cx = cx
    _obs_cy = cy
    _obs_r = radius


# ── Pressure solvers ──────────────────────────────────────────────────────────

def _pressure_solve_jacobi(n_iters):
    reset_pressure()
    compute_divergence()
    for _ in range(n_iters):
        jacobi_iteration()
        swap_pressure_buffers()
    subtract_pressure_gradient()

def _pressure_solve_ml():
    reset_pressure()
    compute_divergence()
    div_np = divergence.to_numpy()
    pred = ml_solver.predict_pressure(div_np)
    pred = np.clip(pred, -50.0, 50.0)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pressure.from_numpy(pred)
    pressure_b.from_numpy(pred)
    for _ in range(ML_JACOBI_AFTER_ML):
        jacobi_iteration()
        swap_pressure_buffers()
    subtract_pressure_gradient()


# ── ML Fire Classification ────────────────────────────────────────────────────

def _update_fire_classification():
    if fire_classifier is None or not fire_classifier.trained:
        return
    try:
        probs = fire_classifier.predict(
            temperature.to_numpy(), density.to_numpy(),
            vel_x.to_numpy(), vel_y.to_numpy())
        ml_class.from_numpy(probs)
    except:
        pass


# ── Status ────────────────────────────────────────────────────────────────────

def get_ml_status():
    parts = []
    if ml_solver:
        parts.append(ml_solver.get_status_string())
    if fire_classifier:
        parts.append(fire_classifier.get_status_string())
    return " | ".join(parts) if parts else "ML-off"


# ── Main simulation step ─────────────────────────────────────────────────────

def simulation_step():
    global _frame_count

    inject_fire(_EMITTER_X, _EMITTER_Y,
                SOURCE_RADIUS, SOURCE_DENSITY_STR, SOURCE_TEMP_STR)

    apply_buoyancy_force()
    compute_curl_field()
    apply_vorticity_confinement()
    clamp_velocity()

    # Pressure solve
    enforce_boundary_conditions()
    if ML_ENABLED and ml_solver is not None:
        try:
            if not ml_solver.trained:
                _pressure_solve_jacobi(JACOBI_ITERS)
                if _frame_count < ML_COLLECT_FRAMES:
                    ml_solver.collect_sample(
                        divergence.to_numpy(), pressure.to_numpy(),
                        n_samples=ML_SAMPLES_PER_FRAME)
                if _frame_count == ML_COLLECT_FRAMES:
                    ml_solver.train(epochs=ML_TRAIN_EPOCHS, batch_size=ML_BATCH_SIZE)
            else:
                _pressure_solve_ml()
        except:
            _pressure_solve_jacobi(JACOBI_ITERS)
    else:
        _pressure_solve_jacobi(JACOBI_ITERS)

    enforce_boundary_conditions()
    clamp_velocity()

    advect_velocity()
    swap_velocity_buffers()
    enforce_boundary_conditions()

    advect_and_cool_scalars()
    swap_scalar_buffers()

    # Fire classifier (runs every N frames for speed)
    if ML_FIRE_ENABLED and fire_classifier is not None:
        try:
            if not fire_classifier.trained:
                if _frame_count < ML_FIRE_COLLECT_FRAMES:
                    fire_classifier.collect_sample(
                        temperature.to_numpy(), density.to_numpy(),
                        vel_x.to_numpy(), vel_y.to_numpy(),
                        n_samples=ML_FIRE_SAMPLES_PER_FRAME)
                if _frame_count == ML_FIRE_COLLECT_FRAMES:
                    fire_classifier.train(epochs=ML_FIRE_TRAIN_EPOCHS, batch_size=512)
            else:
                if _frame_count % ML_FIRE_INFER_EVERY == 0:
                    _update_fire_classification()
        except Exception as e:
            print(f"  [ML Fire] Error: {e}")

    # Pass obstacle position to renderer
    obs_cx_f[None] = float(_obs_cx)
    obs_cy_f[None] = float(_obs_cy)
    obs_r_f[None]  = float(_obs_r)

    render_pixels()
    _frame_count += 1
