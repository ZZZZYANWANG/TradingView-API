from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_nonnormal_operator_round21 as experiment  # noqa: E402

_CACHE: dict[tuple[bytes, int], np.ndarray] = {}


def fast_slow_mode_value(a: np.ndarray, x: np.ndarray, horizon: int) -> float:
    key = (np.ascontiguousarray(a).tobytes(), int(horizon))
    row = _CACHE.get(key)
    if row is None:
        try:
            vals, vecs = np.linalg.eig(a)
            idx = int(np.argmax(np.abs(vals)))
            inv = np.linalg.pinv(vecs)
            lam = complex(vals[idx])
            if abs(1.0 - lam) < 1e-8:
                scale = complex(float(horizon), 0.0)
            else:
                scale = lam * (1.0 - lam ** horizon) / (1.0 - lam)
            mode_matrix = np.outer(vecs[:, idx], inv[idx, :]) * scale
            row = np.real(mode_matrix[0, :]).astype(float)
        except Exception:  # noqa: BLE001
            row = np.full(a.shape[0], np.nan, dtype=float)
        _CACHE[key] = row
        if len(_CACHE) > 50000:
            # Operators are updated daily; retaining the most recent half bounds memory.
            for old_key in list(_CACHE)[:25000]:
                _CACHE.pop(old_key, None)
    value = float(row @ x)
    return value if np.isfinite(value) else np.nan


experiment._slow_mode_value = fast_slow_mode_value

if __name__ == "__main__":
    try:
        experiment.main()
    except Exception as exc:  # noqa: BLE001
        failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        experiment.write_json(experiment.OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
