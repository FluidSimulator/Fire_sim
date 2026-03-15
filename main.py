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
from src.config import GRID_W, GRID_H, DISPLAY_SCALE
from src.fields  import pixels
from src.kernels import reset_all_fields, inject_fire
from src.sim     import simulation_step

W = GRID_W
H = GRID_H

# ─────────────────────────────────────────────────────────────────────────────
#  Controls reference (printed on startup)
# ─────────────────────────────────────────────────────────────────────────────
_BANNER = """
╔══════════════════════════════════════════════════════╗
║       2D Eulerian Fire Simulator  –  Taichi CPU      ║
╠══════════════════════════════════════════════════════╣
║  Left-click          Add fire burst at cursor        ║
║  Space               Reset simulation                ║
║  ESC / close         Quit                            ║
╠══════════════════════════════════════════════════════╣
║  Tweak parameters in  src/config.py  and rerun       ║
╚══════════════════════════════════════════════════════╝
"""


def main() -> None:
    win_w = W * DISPLAY_SCALE
    win_h = H * DISPLAY_SCALE

    gui = ti.GUI(
        "🔥 Fire Simulator  |  Click: Add Fire  |  Space: Reset  |  ESC: Quit",
        res      = (win_w, win_h),
        fast_gui = True,          # skips extra Python-side copies for speed
    )

    # Warm-up: reset all fields to a clean zero state
    reset_all_fields()

    print(_BANNER)
    print("  First frame may be slow (~2 s) while Taichi JIT-compiles kernels.")
    print("  Subsequent frames run at full speed.\n")

    frame      = 0
    t_last_fps = time.perf_counter()
    fps        = 0.0

    while gui.running:

        # ── Event handling ────────────────────────────────────────────────────
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == ti.GUI.SPACE:
                reset_all_fields()
                frame = 0
                print("  [Reset]")

        # Left-click: inject an extra fire burst at the cursor
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            # ti.GUI cursor: (0,0) = bottom-left, (1,1) = top-right → grid coords
            ci = max(3, min(int(mx * W), W - 4))
            cj = max(3, min(int(my * H), H - 4))
            inject_fire(ci, cj,
                        radius = 16,
                        d_str  = 10.0,
                        t_str  = 20.0)

        # ── Simulate + Render ─────────────────────────────────────────────────
        simulation_step()

        # ── Display pixel buffer ──────────────────────────────────────────────
        gui.set_image(pixels)
        gui.show()

        # ── FPS counter (updated every 60 frames) ─────────────────────────────
        frame += 1
        if frame % 60 == 0:
            now  = time.perf_counter()
            fps  = 60.0 / max(now - t_last_fps, 1e-9)
            t_last_fps = now
            gui.title = (
                f"🔥 Fire Simulator  |  {fps:.1f} FPS  "
                f"|  Click: Add Fire  |  Space: Reset"
            )
            print(f"  Frame {frame:6d}   {fps:5.1f} FPS")

    print("\n  Simulation ended. Goodbye!\n")


if __name__ == "__main__":
    main()
