from __future__ import annotations

import os
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1.0e-12
FPAR_OUTPUT = "frenetic_path_activity_response_20d"
RRAR_OUTPUT = "return_range_antisymmetric_response_20d"


@dataclass(frozen=True)
class FPARConfig:
    direction_window: int = 40
    activity_window: int = 20
    surprise_window: int = 126
    surprise_clip: float = 3.0
    activity_scale: float = 3.0
    return_components: bool = False

    def validate_structure(self) -> None:
        if self.direction_window < 2 or self.activity_window < 2 or self.surprise_window < 20:
            raise ValueError("invalid FPAR windows")
        if self.surprise_clip <= 0 or self.activity_scale <= 0:
            raise ValueError("invalid FPAR scale")


@dataclass(frozen=True)
class RRARConfig:
    return_rms_window: int = 20
    range_state_window: int = 20
    surprise_window: int = 126
    response_window: int = 252
    surprise_clip: float = 3.0
    return_components: bool = False

    def validate_structure(self) -> None:
        if min(self.return_rms_window, self.range_state_window, self.surprise_window, self.response_window) < 2:
            raise ValueError("invalid RRAR windows")
        if self.surprise_clip <= 0:
            raise ValueError("invalid RRAR clip")


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(values).rolling(window, min_periods=window).sum().to_numpy(float)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(values).rolling(window, min_periods=window).mean().to_numpy(float)


def strict_prior_robust_z(values: np.ndarray, window: int, clip: float) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if len(x) <= window:
        return out
    prior = sliding_window_view(x, window_shape=window)[:-1]
    finite = np.isfinite(prior).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(prior, axis=1)
        mad = np.nanmedian(np.abs(prior - median[:, None]), axis=1)
    current = x[window:]
    z = np.clip((current - median) / (1.4826 * mad + EPS), -clip, clip)
    z[~finite | ~np.isfinite(current)] = np.nan
    out[window:] = z
    return out


def fpar_arrays(returns: np.ndarray, activity: np.ndarray, eligible: np.ndarray, *, config: FPARConfig = FPARConfig()):
    config.validate_structure()
    r = np.asarray(returns, dtype=float)
    a = np.asarray(activity, dtype=float)
    e = np.asarray(eligible)
    displacement = rolling_sum(r, config.direction_window)
    energy = rolling_sum(r * r, config.direction_window)
    direction = displacement / np.sqrt(energy + EPS)
    state = rolling_mean(a, config.activity_window)
    surprise = np.full_like(state, np.nan)
    for j in range(state.shape[1]):
        surprise[:, j] = strict_prior_robust_z(state[:, j], config.surprise_window, config.surprise_clip)
    score = -direction * surprise / config.activity_scale
    score[~e | ~np.isfinite(direction) | ~np.isfinite(surprise)] = np.nan
    return {
        FPAR_OUTPUT: score,
        "fpar_energy_direction_40d": direction,
        "fpar_activity_mean_20d": state,
        "fpar_activity_surprise_126d": surprise,
    }


def rrar_arrays(returns: np.ndarray, high: np.ndarray, low: np.ndarray, eligible: np.ndarray, *, config: RRARConfig = RRARConfig()):
    config.validate_structure()
    r = np.asarray(returns, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    e = np.asarray(eligible)
    valid = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0) & (h >= l)
    log_range = np.full_like(r, np.nan)
    log_range[valid] = np.log(h[valid] / l[valid])
    return_rms = np.sqrt(rolling_mean(r * r, config.return_rms_window))
    current = r / (return_rms + EPS)
    range_state = rolling_mean(log_range, config.range_state_window)
    surprise = np.full_like(range_state, np.nan)
    for j in range(range_state.shape[1]):
        surprise[:, j] = strict_prior_robust_z(range_state[:, j], config.surprise_window, config.surprise_clip)
    forward = np.full_like(r, np.nan)
    reverse = np.full_like(r, np.nan)
    forward[1:] = surprise[:-1] * current[1:]
    reverse[1:] = current[:-1] * surprise[1:]
    edge = forward - reverse
    prior = np.full_like(edge, np.nan)
    prior[1:] = edge[:-1]
    response = rolling_mean(prior, config.response_window)
    score = response * surprise
    score[~e | ~np.isfinite(score)] = np.nan
    return {
        RRAR_OUTPUT: score,
        "rrar_price_current_20d": current,
        "rrar_range_state_20d": range_state,
        "rrar_range_surprise_126d": surprise,
        "rrar_antisymmetric_response_252d": response,
    }


def install_contract_modules() -> None:
    names = [
        "futures_factor_library",
        "futures_factor_library.mechanisms",
        "futures_factor_library.mechanisms.return_dynamics",
    ]
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    fpar_module = types.ModuleType("futures_factor_library.mechanisms.return_dynamics.frenetic_path_activity_response")
    fpar_module.FPARConfig = FPARConfig
    fpar_module.fpar_arrays = fpar_arrays
    sys.modules[fpar_module.__name__] = fpar_module

    rrar_module = types.ModuleType("futures_factor_library.mechanisms.return_dynamics.return_range_antisymmetric_response")
    rrar_module.RRARConfig = RRARConfig
    rrar_module.rrar_arrays = rrar_arrays
    sys.modules[rrar_module.__name__] = rrar_module


root = Path.cwd().resolve()
(root / "packages" / "futures-factor-library" / "src").mkdir(parents=True, exist_ok=True)
os.environ["TRADE_ROOT"] = str(root)
os.environ["TRADE_COMMIT"] = "5a2379028e4a7d8566c7c055f4417253a9fc923a"
install_contract_modules()

import run_factor_upgrade as research  # noqa: E402

if __name__ == "__main__":
    research.main()
