"""
src/ml_fire_classifier.py
─────────────────────────
ML Classification model for fire spread prediction.

Uses a neural network to classify each cell into:
  Class 0: Empty (background)
  Class 1: Fire  (hot, glowing)
  Class 2: Smoke (translucent, fading)

The classifier learns from the physics simulation and then
GUIDES the fire rendering each frame, determining where fire
and smoke should appear based on local field patterns.

ML CONCEPTS:
  - Multi-class classification (softmax)
  - Cross-entropy loss
  - Feature engineering (local patches of temp + density + velocity)
  - Softmax activation for probability output
  - Trained online from simulation data
"""

import numpy as np
import time


class FireSpreadClassifier:
    """
    MLP classifier: local features → fire/smoke/empty class.

    Input features per cell (3×3 patch):
      - 9 temperature values
      - 9 density values
      - 2 velocity components at center
      Total: 20 features

    Output: 3 class probabilities (softmax)
    Architecture: 20 → 32 → 3
    """

    def __init__(self, hidden=32, lr=0.002):
        n_in = 20  # 9 temp + 9 density + 2 velocity
        n_out = 3  # empty, fire, smoke

        self.W1 = (np.random.randn(n_in, hidden) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (np.random.randn(hidden, n_out) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)

        self.lr = lr
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0

        self.X_data = []
        self.y_data = []
        self.trained = False
        self.total_params = n_in * hidden + hidden + hidden * n_out + n_out

        # Normalisation
        self.X_mean = None
        self.X_std  = None

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

    @staticmethod
    def _softmax(z):
        """Softmax: convert logits to probabilities."""
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / (e.sum(axis=1, keepdims=True) + 1e-8)

    def _forward(self, X):
        """Forward pass with stored intermediates."""
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._probs = self._softmax(self._z2)
        return self._probs

    def _forward_fast(self, X):
        """Inference only."""
        a1 = np.maximum(0, X @ self.W1 + self.b1)
        z2 = a1 @ self.W2 + self.b2
        return self._softmax(z2)

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_features(self, temp_hw, dens_hw, vx_hw, vy_hw, n_samples=None):
        """
        Extract features for cells.

        Per cell: 3×3 temperature patch (9) + 3×3 density patch (9) +
                  velocity at center (2) = 20 features.
        """
        H, W = temp_hw.shape
        t_pad = np.pad(temp_hw, 1, mode='constant')
        d_pad = np.pad(dens_hw, 1, mode='constant')

        if n_samples is not None:
            # Random subsample
            ri = np.random.randint(0, H, size=n_samples)
            rj = np.random.randint(0, W, size=n_samples)
            features = np.empty((n_samples, 20), dtype=np.float32)
            for k in range(n_samples):
                i, j = ri[k], rj[k]
                features[k, :9] = t_pad[i:i+3, j:j+3].ravel()
                features[k, 9:18] = d_pad[i:i+3, j:j+3].ravel()
                features[k, 18] = vx_hw[i, j] if i < vx_hw.shape[0] and j < vx_hw.shape[1] else 0.0
                features[k, 19] = vy_hw[i, j] if i < vy_hw.shape[0] and j < vy_hw.shape[1] else 0.0
            return features, ri, rj
        else:
            # Full grid (for inference)
            from numpy.lib.stride_tricks import sliding_window_view
            t_windows = sliding_window_view(t_pad, (3, 3)).reshape(H * W, 9)
            d_windows = sliding_window_view(d_pad, (3, 3)).reshape(H * W, 9)

            # Velocity at cell centres
            vx_flat = vx_hw[:H, :W].ravel() if vx_hw.shape[0] >= H else np.zeros(H * W, dtype=np.float32)
            vy_flat = vy_hw[:H, :W].ravel() if vy_hw.shape[0] >= H else np.zeros(H * W, dtype=np.float32)

            features = np.hstack([
                t_windows.astype(np.float32),
                d_windows.astype(np.float32),
                vx_flat.reshape(-1, 1).astype(np.float32),
                vy_flat.reshape(-1, 1).astype(np.float32),
            ])
            return features

    def _classify_cell(self, temp, dens):
        """Determine ground-truth class from physics."""
        if temp > 1.0:
            return 1  # Fire
        elif dens > 0.3:
            return 2  # Smoke
        else:
            return 0  # Empty

    # ── Data Collection ───────────────────────────────────────────────────

    def collect_sample(self, temp_np, dens_np, vx_np, vy_np, n_samples=2000):
        """Collect training data from current simulation state."""
        temp_hw = temp_np.T.astype(np.float32)
        dens_hw = dens_np.T.astype(np.float32)
        H, W = temp_hw.shape

        # Velocity: approximate cell-centre from staggered grid
        vx_hw = (vx_np[:W, :H] if vx_np.shape[0] > W else vx_np[:, :H]).T.astype(np.float32)
        vy_hw = (vy_np[:W, :H] if vy_np.shape[1] > H else vy_np[:, :H]).T.astype(np.float32)

        features, ri, rj = self._extract_features(temp_hw, dens_hw, vx_hw, vy_hw, n_samples)

        labels = np.empty(n_samples, dtype=np.int32)
        for k in range(n_samples):
            labels[k] = self._classify_cell(temp_hw[ri[k], rj[k]], dens_hw[ri[k], rj[k]])

        self.X_data.append(features)
        self.y_data.append(labels)

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, epochs=50, batch_size=512):
        """Train the classifier using cross-entropy loss."""
        print("\n  ╔═════════════════════════════════════════╗")
        print("  ║   FIRE CLASSIFIER TRAINING              ║")
        print(f"  ║   Architecture: 20 → 32 → 3 (softmax)  ║")
        print(f"  ║   Classes: Empty / Fire / Smoke         ║")
        print("  ╚═════════════════════════════════════════╝\n")

        t0 = time.perf_counter()

        X = np.vstack(self.X_data).astype(np.float32)
        y = np.concatenate(self.y_data)
        N = len(X)

        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        Xn = (X - self.X_mean) / self.X_std

        # One-hot encode labels
        y_oh = np.zeros((N, 3), dtype=np.float32)
        y_oh[np.arange(N), y] = 1.0

        # Class counts
        for c in range(3):
            print(f"    Class {c} ({'Empty' if c==0 else 'Fire' if c==1 else 'Smoke'}): {(y==c).sum():,}")

        for epoch in range(epochs):
            perm = np.random.permutation(N)
            total_loss = 0.0
            nb = 0

            for s in range(0, N, batch_size):
                idx = perm[s:s+batch_size]
                Xb, yb = Xn[idx], y_oh[idx]
                bs = len(Xb)

                probs = self._forward(Xb)

                # Cross-entropy loss
                loss = -np.mean(np.sum(yb * np.log(probs + 1e-8), axis=1))
                total_loss += loss
                nb += 1

                # Backprop through softmax + cross-entropy
                dl = (probs - yb) / bs

                dW2 = self._a1.T @ dl
                db2 = dl.sum(axis=0)
                da1 = dl @ self.W2.T * (self._z1 > 0)
                dW1 = Xb.T @ da1
                db1 = da1.sum(axis=0)

                self.W1 = self._adam_update('W1', self.W1, dW1)
                self.b1 = self._adam_update('b1', self.b1, db1)
                self.W2 = self._adam_update('W2', self.W2, dW2)
                self.b2 = self._adam_update('b2', self.b2, db2)

            avg = total_loss / max(nb, 1)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}  Loss: {avg:.4f}")

        self.trained = True
        self.X_data.clear()
        self.y_data.clear()

        # Compute accuracy on last batch
        preds = np.argmax(probs, axis=1)
        true_labels = np.argmax(yb, axis=1)
        acc = np.mean(preds == true_labels)
        print(f"\n    ✓ Done in {time.perf_counter()-t0:.1f}s — Accuracy: {acc:.1%}")
        print(f"    ✓ Fire Classifier ACTIVE\n")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, temp_np, dens_np, vx_np, vy_np):
        """
        Predict class probabilities for every cell.

        Returns (W, H, 3) array of [P(empty), P(fire), P(smoke)] per cell.
        """
        temp_hw = temp_np.T.astype(np.float32)
        dens_hw = dens_np.T.astype(np.float32)
        H, W = temp_hw.shape

        vx_hw = (vx_np[:W, :H] if vx_np.shape[0] > W else vx_np[:, :H]).T.astype(np.float32)
        vy_hw = (vy_np[:W, :H] if vy_np.shape[1] > H else vy_np[:, :H]).T.astype(np.float32)

        features = self._extract_features(temp_hw, dens_hw, vx_hw, vy_hw)

        Xn = (features - self.X_mean) / self.X_std
        probs = self._forward_fast(Xn)

        # Reshape to (H, W, 3) then transpose to (W, H, 3)
        probs_hw = probs.reshape(H, W, 3)
        return np.transpose(probs_hw, (1, 0, 2)).astype(np.float32)

    def get_status_string(self):
        if self.trained:
            return "Classifier-ACTIVE"
        elif len(self.X_data) > 0:
            return f"Classifier-collecting({len(self.X_data)})"
        return "Classifier-idle"
