from __future__ import annotations

import numpy as np

import run_factor_upgrade as research


def fixed_cap_training(X: np.ndarray, y: np.ndarray, cap: int = 450_000):
    if len(y) <= cap:
        return X, y
    rng = np.random.default_rng(research.SEED)
    index = np.sort(rng.choice(len(y), size=cap, replace=False))
    return X[index], y[index]


research.cap_training = fixed_cap_training

if __name__ == "__main__":
    research.main()
