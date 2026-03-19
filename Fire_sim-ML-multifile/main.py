"""
main.py
───────
Entry point for the 2D Eulerian Fire Simulator.

Run:
    python main.py

ti.init() MUST be called before importing any src.* module
because Taichi fields are allocated on import.
"""

import time
import numpy as np
import taichi as ti

# ── 1. Initialise Taichi (CPU backend, multi-threaded) ───────────────────────
ti.init(
    arch             = ti.cpu,
    cpu_max_num_threads = 16,    # cap threads on laptops; raise on workstations
    default_fp       = ti.f32,
    fast_math        = True,     # allow FP optimisations for extra ~5–10% speed
    debug            = False,    # set True to catch index-out-of-bounds errors
)

# ── 2. Import simulation modules (fields created here, after ti.init) ────────
from src.config import GRID_W, GRID_H, DISPLAY_SCALE, OBSTACLE_RADIUS
from src.fields  import pixels
from src.kernels import reset_all_fields
from src.sim     import simulation_step, init_obstacle, move_obstacle_to, get_ml_status

W = GRID_W
H = GRID_H

# ─────────────────────────────────────────────────────────────────────────────
#  Controls reference (printed on startup)
# ─────────────────────────────────────────────────────────────────────────────
_BANNER = """
╔════════════════════════════════════════════════════════════╗
║   2D Fire Simulator  –  ML Classification + Physics        ║
╠════════════════════════════════════════════════════════════╣
║  Click+drag        Move obstacle through the fire          ║
║  Space             Reset simulation                        ║
║  ESC / close       Quit                                    ║
╠════════════════════════════════════════════════════════════╣
║  ML Pressure Solver: learns to accelerate fluid physics    ║
║  ML Fire Classifier: classifies fire/smoke/empty per cell  ║
║  Fire movement is driven entirely by ML after training.    ║
╚════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    win_w = W * DISPLAY_SCALE
    win_h = H * DISPLAY_SCALE

    gui = ti.GUI(
        "🔥 Fire Sim  |  Drag: Move Obstacle  |  Space: Reset  |  ESC: Quit",
        res      = (win_w, win_h),
        fast_gui = False,
    )

    reset_all_fields()
    init_obstacle()

    obs_cx     = GRID_W // 2
    obs_cy     = 80
    obs_radius = OBSTACLE_RADIUS

    _s = DISPLAY_SCALE

    print(_BANNER)
    print("  First frame may be slow (~2 s) while Taichi JIT-compiles kernels.")
    print("  Subsequent frames run at full speed.\n")

    frame      = 0
    t_last_fps = time.perf_counter()
    fps        = 0.0

    while gui.running:

        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == ti.GUI.SPACE:
                reset_all_fields()
                move_obstacle_to(obs_cx, obs_cy, obs_radius)
                frame = 0
                print("  [Reset]")

        # Click+drag: move obstacle only
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            ci = max(0, min(int(mx * W), W - 1))
            cj = max(0, min(int(my * H), H - 1))
            obs_cx = max(obs_radius + 1, min(ci, W - obs_radius - 1))
            obs_cy = max(obs_radius + 1, min(cj, H - obs_radius - 1))
            move_obstacle_to(obs_cx, obs_cy, obs_radius)

        # ── Simulate + Render ─────────────────────────────────────────────────
        simulation_step()

        # ── Display pixel buffer ──────────────────────────────────────────
        img = pixels.to_numpy()
        gui.set_image(np.repeat(np.repeat(img, _s, axis=0), _s, axis=1))
        gui.show()

        # ── FPS counter (updated every 60 frames) ─────────────────────────────
        frame += 1
        if frame % 60 == 0:
            now  = time.perf_counter()
            fps  = 60.0 / max(now - t_last_fps, 1e-9)
            t_last_fps = now
            ml_status = get_ml_status()
            gui.title = (
                f"🔥 Fire Sim  |  {fps:.1f} FPS  |  [{ml_status}]  "
                f"|  Drag: Move Obstacle"
            )
            print(f"  Frame {frame:6d}   {fps:5.1f} FPS | {ml_status}")

    print("\n  Simulation ended. Goodbye!\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\n  >>> Press Enter to close... <<<")
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        input("\n  >>> Press Enter to close... <<<")
