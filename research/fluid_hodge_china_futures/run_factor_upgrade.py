from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261121
EPS = 1.0e-12
HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "run.py"
OUT = Path("research_outputs/futures_response_upgrade_20261121")
OUT.mkdir(parents=True, exist_ok=True)

TRADE_ROOT = Path(os.environ.get("TRADE_ROOT", Path.cwd())).resolve()
FUTURES_PACKAGE = TRADE_ROOT / "packages" / "futures-factor-library" / "src"
if not FUTURES_PACKAGE.exists():
    raise RuntimeError(f"Trade futures package not found at {FUTURES_PACKAGE}")
sys.path.insert(0, str(FUTURES_PACKAGE))

from futures_factor_library.mechanisms.return_dynamics.frenetic_path_activity_response import (  # noqa: E402
    FPARConfig,
    fpar_arrays,
)
from futures_factor_library.mechanisms.return_dynamics.return_range_antisymmetric_response import (  # noqa: E402
    RRARConfig,
    rrar_arrays,
)

HORIZON = 20
BOOTSTRAPS = 3999
ONE_WAY_COST_BPS = 5.0
STAGES = {
    "development": (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-01")),
    "validation": (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
    "holdout": (pd.Timestamp("2023-01-01"), pd.Timestamp("2027-01-01")),
}


def load_base():
    spec = importlib.util.spec_from_file_location("china_futures_public_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load China-futures base module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def stage_mask(index: pd.DatetimeIndex, stage: str) -> np.ndarray:
    start, end = STAGES[stage]
    return np.asarray((index >= start) & (index < end))


def strict_prior_z(frame: pd.DataFrame, window: int, minimum: int) -> pd.DataFrame:
    prior = frame.shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0)
    return ((frame - mean) / (std + EPS)).clip(-10.0, 10.0)


def complete_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(values).rolling(window, min_periods=window).mean().to_numpy(float)


def lag_matrix(values: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return values.copy()
    return np.vstack([np.full((lag, values.shape[1]), np.nan), values[:-lag]])


def product_ic(signal: pd.DataFrame, target: pd.DataFrame, mask: np.ndarray, method: str = "pearson", minimum: int = 100) -> pd.Series:
    rows: dict[str, float] = {}
    for product in signal.columns:
        x = signal.loc[mask, product].to_numpy(float)
        y = target.loc[mask, product].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < minimum:
            continue
        if np.nanstd(x[valid]) <= EPS or np.nanstd(y[valid]) <= EPS:
            continue
        value = pearsonr(x[valid], y[valid]).statistic if method == "pearson" else spearmanr(x[valid], y[valid]).statistic
        if np.isfinite(value):
            rows[str(product)] = float(value)
    return pd.Series(rows, dtype=float).sort_index()


def bootstrap_products(values: pd.Series, reps: int = BOOTSTRAPS) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) < 8:
        return {"mean": float(np.nanmean(x)) if len(x) else float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_nonpositive": float("nan"), "n": int(len(x))}
    rng = np.random.default_rng(SEED + len(x))
    draw = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(draw, 0.025)),
        "ci_high": float(np.quantile(draw, 0.975)),
        "p_nonpositive": float((1.0 + np.sum(draw <= 0.0)) / (reps + 1.0)),
        "n": int(len(x)),
    }


def yearly_equal_product_ic(signal: pd.DataFrame, target: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    years = sorted(set(signal.index[mask].year))
    for year in years:
        local_mask = mask & (signal.index.year == year)
        values = product_ic(signal, target, local_mask, minimum=40)
        if len(values):
            out[str(int(year))] = float(values.mean())
    return out


def causal_scale(signal: pd.DataFrame, development_mask: np.ndarray) -> pd.Series:
    scale = signal.loc[development_mask].abs().median(axis=0)
    pooled = float(np.nanmedian(scale.to_numpy(float)))
    if not np.isfinite(pooled) or pooled <= EPS:
        pooled = 1.0
    return scale.where(np.isfinite(scale) & (scale > EPS), pooled)


def opportunity_payments(signal: pd.DataFrame, target: pd.DataFrame, mask: np.ndarray, development_mask: np.ndarray) -> pd.Series:
    scale = causal_scale(signal, development_mask)
    position = np.tanh(signal.div(scale, axis=1)).clip(-1.0, 1.0)
    gross = position * np.expm1(target)
    cost = (2.0 * ONE_WAY_COST_BPS / 10_000.0) * position.abs()
    net = (gross - cost).where(np.broadcast_to(mask[:, None], gross.shape))
    values = net.stack(dropna=True)
    return values.groupby(level=1).mean()


def flatten_features(features: list[pd.DataFrame], target: pd.DataFrame, date_mask: np.ndarray, required: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack([frame.to_numpy(float) for frame in features], axis=2)
    y = target.to_numpy(float)
    selected = np.broadcast_to(date_mask[:, None], y.shape) & np.isfinite(y)
    if required is not None:
        selected &= np.isfinite(required.to_numpy(float))
    rows, cols = np.where(selected)
    return stack[rows, cols, :], y[rows, cols], target.index.to_numpy()[rows]


def fixed_training_sample(X_base: np.ndarray, X_aug: np.ndarray, y: np.ndarray, cap: int = 350_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) <= cap:
        return X_base, X_aug, y
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(y), size=cap, replace=False))
    return X_base[idx], X_aug[idx], y[idx]


def daily_loss_inference(dates: np.ndarray, loss_diff: np.ndarray, block: int = 60, reps: int = 1999) -> dict[str, float]:
    series = pd.DataFrame({"date": pd.to_datetime(dates), "diff": loss_diff}).groupby("date", sort=True).mean().diff.dropna().to_numpy(float)
    if len(series) < max(100, 2 * block):
        return {"mean": float(np.nanmean(series)) if len(series) else float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_nonpositive": float("nan"), "n": int(len(series))}
    rng = np.random.default_rng(SEED + len(series))
    n = len(series)
    blocks = int(math.ceil(n / block))
    max_start = max(1, n - block + 1)
    draws = np.empty(reps, dtype=float)
    for b in range(reps):
        starts = rng.integers(0, max_start, size=blocks)
        sample = np.concatenate([series[s : s + block] for s in starts])[:n]
        draws[b] = float(sample.mean())
    return {
        "mean": float(series.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_nonpositive": float((1.0 + np.sum(draws <= 0.0)) / (reps + 1.0)),
        "n": int(n),
    }


def pooled_rankic(dates: np.ndarray, pred: np.ndarray, y: np.ndarray) -> float:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "pred": pred, "y": y})
    daily: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        valid = np.isfinite(group.pred) & np.isfinite(group.y)
        if int(valid.sum()) < 8:
            continue
        value = spearmanr(group.loc[valid, "pred"], group.loc[valid, "y"]).statistic
        if np.isfinite(value):
            daily.append(float(value))
    return float(np.mean(daily)) if daily else float("nan")


def model_comparison(base_features: list[pd.DataFrame], candidate: pd.DataFrame, target: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray, stage: str) -> dict[str, Any]:
    Xb, y, _ = flatten_features(base_features, target, train_mask, required=candidate)
    Xa, ya, _ = flatten_features(base_features + [candidate], target, train_mask, required=candidate)
    Xbt, yt, dates = flatten_features(base_features, target, test_mask, required=candidate)
    Xat, yat, dates_a = flatten_features(base_features + [candidate], target, test_mask, required=candidate)
    if not (np.array_equal(y, ya) and np.array_equal(yt, yat) and np.array_equal(dates, dates_a)):
        raise AssertionError("model samples differ")
    Xb, Xa, y = fixed_training_sample(Xb, Xa, y)
    output: dict[str, Any] = {"stage": stage, "n_train": int(len(y)), "n_test": int(len(yt))}
    models = {
        "ridge": (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=20.0)),
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=20.0)),
        ),
        "hgb": (
            HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=120, l2_regularization=10.0, random_state=SEED),
            HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=120, l2_regularization=10.0, random_state=SEED),
        ),
    }
    for name, (mb, ma) in models.items():
        mb.fit(Xb, y)
        ma.fit(Xa, y)
        pb = np.asarray(mb.predict(Xbt), dtype=float)
        pa = np.asarray(ma.predict(Xat), dtype=float)
        lb = np.square(yt - pb)
        la = np.square(yt - pa)
        mse_b = float(lb.mean())
        mse_a = float(la.mean())
        output[name] = {
            "mse_base": mse_b,
            "mse_augmented": mse_a,
            "gain_pct": float(100.0 * (mse_b - mse_a) / mse_b) if mse_b > 0 else float("nan"),
            "daily_rankic_base": pooled_rankic(dates, pb, yt),
            "daily_rankic_augmented": pooled_rankic(dates, pa, yt),
            "paired_loss_inference": daily_loss_inference(dates, lb - la),
        }
    return output


def select_candidate(candidates: dict[str, pd.DataFrame], parent: pd.DataFrame, target: pd.DataFrame, development: np.ndarray) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    parent_ic = product_ic(parent, target, development)
    rows: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        ic = product_ic(candidate, target, development)
        pair = pd.concat([ic.rename("candidate"), parent_ic.rename("parent")], axis=1).dropna()
        years = yearly_equal_product_ic(candidate, target, development)
        positive_share = float(np.mean([v > 0 for v in years.values()])) if years else 0.0
        increment = float((pair.candidate - pair.parent).mean()) if len(pair) else float("nan")
        score = float(ic.mean() + 0.25 * increment + 0.005 * positive_share) if len(ic) else -np.inf
        rows.append({"candidate": name, "development_ic": float(ic.mean()) if len(ic) else float("nan"), "development_parent_ic": float(parent_ic.mean()) if len(parent_ic) else float("nan"), "development_increment": increment, "development_positive_year_share": positive_share, "development_score": score, "development_yearly_ic": json.dumps(years, sort_keys=True)})
    table = pd.DataFrame(rows).sort_values("development_score", ascending=False)
    selected_name = str(table.iloc[0].candidate)
    return selected_name, candidates[selected_name], table


def evaluate_family(name: str, parent: pd.DataFrame, candidate: pd.DataFrame, target: pd.DataFrame, base_features: list[pd.DataFrame], development: np.ndarray, validation: np.ndarray, holdout: np.ndarray) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage, mask in (("validation", validation), ("holdout", holdout)):
        ci = product_ic(candidate, target, mask)
        pi = product_ic(parent, target, mask)
        pair = pd.concat([ci.rename("candidate"), pi.rename("parent")], axis=1).dropna()
        diff = pair.candidate - pair.parent
        candidate_pay = opportunity_payments(candidate, target, mask, development)
        parent_pay = opportunity_payments(parent, target, mask, development)
        pay_pair = pd.concat([candidate_pay.rename("candidate"), parent_pay.rename("parent")], axis=1).dropna()
        years = yearly_equal_product_ic(candidate, target, mask)
        stages[stage] = {
            "candidate_equal_product_ic": float(ci.mean()),
            "parent_equal_product_ic": float(pi.mean()),
            "ic_increment": float(diff.mean()),
            "ic_increment_inference": bootstrap_products(diff),
            "candidate_yearly_ic": years,
            "positive_year_share": float(np.mean([v > 0 for v in years.values()])) if years else float("nan"),
            "candidate_equal_product_net_payment_5bp_one_way": float(candidate_pay.mean()),
            "parent_equal_product_net_payment_5bp_one_way": float(parent_pay.mean()),
            "payment_increment_inference": bootstrap_products(pay_pair.candidate - pay_pair.parent),
            "n_products_ic": int(len(ci)),
            "n_products_payment": int(len(candidate_pay)),
        }
    models = {
        "validation": model_comparison(base_features, candidate, target, development, validation, "validation"),
        "holdout": model_comparison(base_features, candidate, target, development | validation, holdout, "holdout"),
    }
    v = stages["validation"]
    h = stages["holdout"]
    vm = models["validation"]["hgb"]
    hm = models["holdout"]["hgb"]
    gates = {
        "direct_increment_positive_both": bool(v["ic_increment"] > 0.002 and h["ic_increment"] > 0.002),
        "direct_increment_inference_both": bool(v["ic_increment_inference"]["p_nonpositive"] <= 0.10 and h["ic_increment_inference"]["p_nonpositive"] <= 0.10),
        "strong_model_gain_positive_both": bool(vm["gain_pct"] > 0.0 and hm["gain_pct"] > 0.0),
        "strong_model_loss_inference_both": bool(vm["paired_loss_inference"]["p_nonpositive"] <= 0.10 and hm["paired_loss_inference"]["p_nonpositive"] <= 0.10),
        "costed_payment_positive_both": bool(v["candidate_equal_product_net_payment_5bp_one_way"] > 0.0 and h["candidate_equal_product_net_payment_5bp_one_way"] > 0.0),
        "costed_payment_beats_parent_both": bool(v["candidate_equal_product_net_payment_5bp_one_way"] > v["parent_equal_product_net_payment_5bp_one_way"] and h["candidate_equal_product_net_payment_5bp_one_way"] > h["parent_equal_product_net_payment_5bp_one_way"]),
        "holdout_breadth": bool(h["positive_year_share"] >= 0.60),
    }
    return {"family": name, "stages": stages, "models": models, "gates": gates, "passed": bool(all(gates.values()))}


def main() -> None:
    base = load_base()
    table, download_records, panels = base.download_panel()
    open_, high, low, close, volume, hold = [panels[k] for k in ["open", "high", "low", "close", "volume", "hold"]]
    index = close.index
    for frame in (open_, high, low, volume, hold):
        index = index.union(frame.index)
    open_, high, low, close, volume, hold = [frame.reindex(index).sort_index() for frame in (open_, high, low, close, volume, hold)]

    coverage: list[dict[str, Any]] = []
    products: list[str] = []
    for product in close.columns:
        counts = {}
        for stage in STAGES:
            mask = stage_mask(index, stage)
            counts[stage] = int((close.loc[mask, product].notna() & hold.loc[mask, product].notna()).sum())
        coverage.append({"product": product, **counts})
        if counts["development"] >= 750 and counts["validation"] >= 500 and counts["holdout"] >= 400:
            products.append(product)
    if len(products) < 14:
        raise RuntimeError(f"insufficient eligible futures products: {len(products)} {products}")
    open_, high, low, close, volume, hold = [frame[products] for frame in (open_, high, low, close, volume, hold)]

    returns = np.log(close / close.shift(1))
    activity = np.log((volume.clip(lower=0.0) + 1.0) / (hold.clip(lower=0.0) + 1.0))
    eligible = (open_ > 0) & (high > 0) & (low > 0) & (close > 0) & (volume >= 0) & (hold >= 0)
    target = np.log(open_.shift(-(HORIZON + 1)) / open_.shift(-1))

    r = returns.to_numpy(float)
    a = activity.to_numpy(float)
    e = eligible.to_numpy(bool)
    fpar = fpar_arrays(r, a, e, config=FPARConfig(return_components=True))
    rrar = rrar_arrays(r, high.to_numpy(float), low.to_numpy(float), e, config=RRARConfig(return_components=True))
    fpar_parent = pd.DataFrame(fpar["frenetic_path_activity_response_20d"], index=index, columns=products)
    q = fpar["fpar_energy_direction_40d"]
    u = fpar["fpar_activity_surprise_126d"]
    rrar_parent = pd.DataFrame(rrar["return_range_antisymmetric_response_20d"], index=index, columns=products)
    x = rrar["rrar_price_current_20d"]
    v = rrar["rrar_range_surprise_126d"]

    fpar_candidates: dict[str, pd.DataFrame] = {}
    positive = np.maximum(u, 0.0)
    negative = np.maximum(-u, 0.0)
    for alpha in (0.0, 0.25, 0.50, 0.75):
        raw = -q * (positive - alpha * negative) / 3.0
        raw[~e | ~np.isfinite(raw)] = np.nan
        fpar_candidates[f"FPAR_ASYMMETRIC_LINEAR_A{alpha:g}"] = pd.DataFrame(raw, index=index, columns=products)
        saturated = -q * (np.tanh(positive) - alpha * np.tanh(negative))
        saturated[~e | ~np.isfinite(saturated)] = np.nan
        fpar_candidates[f"FPAR_ASYMMETRIC_TANH_A{alpha:g}"] = pd.DataFrame(saturated, index=index, columns=products)

    lag_responses: dict[int, np.ndarray] = {}
    for lag in (1, 2, 3, 5):
        edge = lag_matrix(v, lag) * x - lag_matrix(x, lag) * v
        prior_edge = lag_matrix(edge, 1)
        lag_responses[lag] = complete_rolling_mean(prior_edge, 252)
    weight_sets = {
        "EQUAL": {1: 0.25, 2: 0.25, 3: 0.25, 5: 0.25},
        "FAST_DECAY": {1: 0.50, 2: 0.25, 3: 0.15, 5: 0.10},
        "SLOW_HEAVY": {1: 0.10, 2: 0.20, 3: 0.30, 5: 0.40},
        "LAG1_3": {1: 0.50, 2: 0.30, 3: 0.20},
    }
    rrar_candidates: dict[str, pd.DataFrame] = {}
    for label, weights in weight_sets.items():
        response = sum(float(weight) * lag_responses[lag] for lag, weight in weights.items())
        score = response * v
        score[~e | ~np.isfinite(score)] = np.nan
        rrar_candidates[f"RRAR_MULTILAG_{label}"] = pd.DataFrame(score, index=index, columns=products)

    development = stage_mask(index, "development")
    validation = stage_mask(index, "validation")
    holdout = stage_mask(index, "holdout")
    fpar_name, fpar_selected, fpar_selection = select_candidate(fpar_candidates, fpar_parent, target, development)
    rrar_name, rrar_selected, rrar_selection = select_candidate(rrar_candidates, rrar_parent, target, development)
    fpar_selection.to_csv(OUT / "fpar_candidate_selection.csv", index=False)
    rrar_selection.to_csv(OUT / "rrar_candidate_selection.csv", index=False)

    range_log = np.log(high / low)
    oi_log = np.log1p(hold.clip(lower=0.0))
    vol_log = np.log1p(volume.clip(lower=0.0))
    base_features = [
        strict_prior_z(returns.rolling(5, min_periods=5).sum(), 252, 126),
        strict_prior_z(returns.rolling(20, min_periods=20).sum(), 252, 126),
        strict_prior_z(returns.rolling(40, min_periods=35).sum(), 252, 126),
        strict_prior_z(returns.rolling(60, min_periods=50).sum(), 252, 126),
        strict_prior_z(returns.rolling(120, min_periods=100).sum(), 252, 126),
        strict_prior_z(returns.rolling(20, min_periods=15).std(ddof=0), 252, 126),
        strict_prior_z(returns.rolling(60, min_periods=45).std(ddof=0), 252, 126),
        strict_prior_z(range_log.rolling(20, min_periods=15).mean(), 252, 126),
        strict_prior_z(activity.rolling(20, min_periods=15).mean(), 252, 126),
        strict_prior_z(oi_log.diff(5), 252, 126),
        strict_prior_z(oi_log.diff(20), 252, 126),
        strict_prior_z(vol_log.diff(5), 252, 126),
        fpar_parent,
        rrar_parent,
    ]

    fpar_result = evaluate_family("FPAR", fpar_parent, fpar_selected, target, base_features, development, validation, holdout)
    fpar_result["selected_candidate"] = fpar_name
    rrar_result = evaluate_family("RRAR", rrar_parent, rrar_selected, target, base_features, development, validation, holdout)
    rrar_result["selected_candidate"] = rrar_name
    admitted = [result["selected_candidate"] for result in (fpar_result, rrar_result) if result["passed"]]

    decision = {
        "study": "China-futures FPAR asymmetric response and RRAR multi-lag response upgrades",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "trade_commit": os.environ.get("TRADE_COMMIT", "unknown"),
        "data_source": "Akshare Sina continuous-main daily futures public interface",
        "history_status": "historical replay; development-only formula selection; validation and later holdout not used for candidate selection within this run",
        "products": products,
        "product_count": int(len(products)),
        "date_start": str(index.min().date()),
        "date_end": str(index.max().date()),
        "coverage": coverage,
        "download_records": download_records,
        "families": {"FPAR": fpar_result, "RRAR": rrar_result},
        "passed_candidates": admitted,
        "passed": bool(admitted),
        "catalog_action": "eligible_for_default_off_non_independent_registration" if admitted else "no_trade_repository_update",
        "limitations": [
            "Public continuous-main data is an independent robustness panel, not the canonical RQData actual-contract roll-safe panel.",
            "Cost diagnostics are overlapping fixed 20-session opportunities, not a continuous capital account.",
            "Any pass supports only a default-off historical representation upgrade pending canonical-panel and fresh-forward confirmation.",
        ],
    }
    write_json(OUT / "decision.json", decision)
    print("DECISION_JSON=" + json.dumps(decision, ensure_ascii=False, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
