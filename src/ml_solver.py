"""
src/ml_solver.py
────────────────
Lightweight Neural Pressure Solver — pure NumPy, fully vectorized.

ML CONCEPTS: MLP, ReLU, MSE, Backprop, Adam, Mini-batch SGD, Normalisation
"""

import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view


class NeuralPressureSolver:
    """
    Small MLP: local divergence patch → pressure value.
    Architecture: Input(9) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1)
    """

    def __init__(self, patch_size=3, hidden1=32, hidden2=16, lr=0.001):
        self.ps  = patch_size
        self.pad = patch_size // 2
        n_in = patch_size * patch_size

        # He weight init
        self.W1 = (np.random.randn(n_in, hidden1) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = (np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)).astype(np.float32)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = (np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

        # Adam state
        self.lr = lr
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0

        # Normalisation
        self.X_mean = None
        self.X_std  = None
        self.y_mean = 0.0
        self.y_std  = 1.0

        # Data storage
        self.X_data = []
        self.y_data = []
        self.trained = False
        self.total_params = n_in * hidden1 + hidden1 + hidden1 * hidden2 + hidden2 + hidden2 + 1

    def _adam_update(self, name, param, grad):
        if name not in self.adam_m:
            self.adam_m[name] = np.zeros_like(param)
            self.adam_v[name] = np.zeros_like(param)
        self.adam_t += 1
        self.adam_m[name] = 0.9 * self.adam_m[name] + 0.1 * grad
        self.adam_v[name] = 0.999 * self.adam_v[name] + 0.001 * (grad ** 2)
        m_hat = self.adam_m[name] / (1 - 0.9 ** self.adam_t)
        v_hat = self.adam_v[name] / (1 - 0.999 ** self.adam_t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def _forward(self, X):
        """Forward pass with intermediates stored for backprop."""
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0, self._z2)
        return self._a2 @ self.W3 + self.b3

    def _forward_fast(self, X):
        """Inference-only forward pass — no intermediates stored."""
        a1 = np.maximum(0, X @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        return a2 @ self.W3 + self.b3

    # ── Data Collection ───────────────────────────────────────────────────

    def collect_sample(self, div_np, pres_np, n_samples=1500):
        """Randomly sample cells and extract patches as training data."""
        div_hw  = div_np.T.astype(np.float32)
        pres_hw = pres_np.T.astype(np.float32)
        H, W = div_hw.shape

        padded = np.pad(div_hw, self.pad, mode='constant')

        ri = np.random.randint(0, H, size=n_samples)
        rj = np.random.randint(0, W, size=n_samples)

        ps = self.ps
        patches = np.empty((n_samples, ps * ps), dtype=np.float32)
        targets = np.empty((n_samples, 1), dtype=np.float32)

        for k in range(n_samples):
            ii, jj = ri[k], rj[k]
            patches[k] = padded[ii:ii+ps, jj:jj+ps].ravel()
            targets[k, 0] = pres_hw[ii, jj]

        self.X_data.append(patches)
        self.y_data.append(targets)

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, epochs=40, batch_size=256):
        """Train the network on collected data."""
        print("\n  ╔═════════════════════════════════════════╗")
        print("  ║   NEURAL NETWORK TRAINING               ║")
        print(f"  ║   Frames: {len(self.X_data)}, Arch: 9→32→16→1       ║")
        print("  ╚═════════════════════════════════════════╝\n")

        t0 = time.perf_counter()

        X = np.vstack(self.X_data).astype(np.float32)
        y = np.vstack(self.y_data).astype(np.float32)
        N = len(X)

        print(f"    Samples: {N:,}  Memory: {X.nbytes/1024/1024:.1f} MB")

        # Normalise
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std  = y.std() + 1e-8
        Xn = (X - self.X_mean) / self.X_std
        yn = (y - self.y_mean) / self.y_std

        for epoch in range(epochs):
            perm = np.random.permutation(N)
            total_loss = 0.0
            nb = 0

            for s in range(0, N, batch_size):
                idx = perm[s:s+batch_size]
                Xb, yb = Xn[idx], yn[idx]
                bs = len(Xb)

                pred = self._forward(Xb)
                diff = pred - yb
                loss = float(np.mean(diff ** 2))
                total_loss += loss
                nb += 1

                # Backprop
                dl = (2.0 / bs) * diff
                dW3 = self._a2.T @ dl
                db3 = dl.sum(axis=0)
                da2 = dl @ self.W3.T * (self._z2 > 0)
                dW2 = self._a1.T @ da2
                db2 = da2.sum(axis=0)
                da1 = da2 @ self.W2.T * (self._z1 > 0)
                dW1 = Xb.T @ da1
                db1 = da1.sum(axis=0)

                self.W1 = self._adam_update('W1', self.W1, dW1)
                self.b1 = self._adam_update('b1', self.b1, db1)
                self.W2 = self._adam_update('W2', self.W2, dW2)
                self.b2 = self._adam_update('b2', self.b2, db2)
                self.W3 = self._adam_update('W3', self.W3, dW3)
                self.b3 = self._adam_update('b3', self.b3, db3)

            avg = total_loss / max(nb, 1)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}  Loss: {avg:.6f}")

        self.trained = True
        self.X_data.clear()
        self.y_data.clear()
        print(f"\n    Done in {time.perf_counter()-t0:.1f}s — ML solver ACTIVE\n")

    # ── Vectorized Inference ──────────────────────────────────────────────

    def predict_pressure(self, div_np):
        """
        Predict full pressure field — FULLY VECTORIZED, no Python loops.

        Uses numpy sliding_window_view to extract all patches at once,
        then runs the entire grid through the network in one matrix multiply.
        Memory: ~7 MB for 128×256 grid — safe.
        """
        div_hw = div_np.T.astype(np.float32)
        H, W = div_hw.shape
        ps = self.ps

        padded = np.pad(div_hw, self.pad, mode='constant')

        # Extract ALL patches in one numpy call (zero-copy view)
        windows = sliding_window_view(padded, (ps, ps))  # (H, W, ps, ps)
        patches = np.ascontiguousarray(
            windows.reshape(H * W, ps * ps),
            dtype=np.float32
        )

        # Normalise → forward pass → denormalise (all vectorized)
        Xn = (patches - self.X_mean) / self.X_std
        pred_n = self._forward_fast(Xn)
        result = (pred_n.ravel() * self.y_std + self.y_mean).reshape(H, W)

        return result.T.astype(np.float32)  # back to Taichi (W, H)

    def get_status_string(self):
        if self.trained:
            return "ML-ACTIVE"
        elif len(self.X_data) > 0:
            return f"ML-collecting({len(self.X_data)})"
        return "ML-idle"
