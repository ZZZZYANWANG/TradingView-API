from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import baostock as bs
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261121
EPS = 1.0e-12
HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "run_research.py"
OUT = Path("research_outputs/stock_reversal_upgrade_20261121")
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

MAX_CODES = 650
MIN_ASSETS = 120
HORIZON = 20
BOOTSTRAPS = 1999
BLOCK = 40
ONE_WAY_COST_BPS = 10.0
STAGES = {
    "development": (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-01-01")),
    "validation": (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
    "holdout": (pd.Timestamp("2023-01-01"), pd.Timestamp("2027-01-01")),
}


def load_base():
    spec = importlib.util.spec_from_file_location("ashare_event_absorption_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base research module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.CACHE = CACHE
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def board_name(code: str) -> str:
    value = str(code)
    if value.startswith("sh.688"):
        return "STAR"
    if value.startswith("sh.6"):
        return "SH_MAIN"
    if value.startswith("sz.30"):
        return "CHINEXT"
    if value.startswith("sz.0"):
        return "SZ_MAIN"
    return "OTHER"


def pivot(frame: pd.DataFrame, field: str, dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    return frame.pivot(index="date", columns="code", values=field).reindex(index=dates, columns=codes)


def cs_rank(values: pd.DataFrame, eligible: pd.DataFrame, minimum: int = MIN_ASSETS) -> pd.DataFrame:
    valid = eligible & values.notna() & np.isfinite(values)
    ranked = values.where(valid).rank(axis=1, pct=True, method="average") - 0.5
    ranked = ranked.where(valid)
    ranked.loc[valid.sum(axis=1) < minimum] = np.nan
    return ranked


def cs_z(values: pd.DataFrame, eligible: pd.DataFrame, minimum: int = MIN_ASSETS) -> pd.DataFrame:
    valid = eligible & values.notna() & np.isfinite(values)
    masked = values.where(valid)
    mean = masked.mean(axis=1)
    std = masked.std(axis=1, ddof=0)
    out = masked.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    out = out.clip(-8.0, 8.0).where(valid)
    out.loc[valid.sum(axis=1) < minimum] = np.nan
    return out


def strict_prior_z(values: pd.DataFrame, window: int, minimum: int) -> pd.DataFrame:
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0)
    return ((values - mean) / (std + EPS)).clip(-8.0, 8.0)


def strict_prior_beta(y: pd.DataFrame, x: pd.DataFrame, window: int = 120, minimum: int = 60) -> pd.DataFrame:
    yp = y.shift(1)
    xp = x.shift(1)
    my = yp.rolling(window, min_periods=minimum).mean()
    mx = xp.rolling(window, min_periods=minimum).mean()
    cov = (yp * xp).rolling(window, min_periods=minimum).mean() - my * mx
    var = xp.pow(2).rolling(window, min_periods=minimum).mean() - mx.pow(2)
    return (cov / (var + EPS)).clip(-5.0, 5.0)


def stage_mask(dates: pd.DatetimeIndex, stage: str) -> np.ndarray:
    start, end = STAGES[stage]
    return np.asarray((dates >= start) & (dates < end))


def daily_rankic(signal: pd.DataFrame, target: pd.DataFrame, date_mask: np.ndarray) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date in signal.index[date_mask]:
        x = signal.loc[date].to_numpy(float)
        y = target.loc[date].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < MIN_ASSETS:
            continue
        if np.nanstd(x[valid]) <= EPS or np.nanstd(y[valid]) <= EPS:
            continue
        value = spearmanr(x[valid], y[valid]).statistic
        if np.isfinite(value):
            values[pd.Timestamp(date)] = float(value)
    return pd.Series(values, dtype=float).sort_index()


def moving_block_bootstrap(values: pd.Series, *, block: int = BLOCK, reps: int = BOOTSTRAPS) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) < max(80, 2 * block):
        return {"mean": float(np.nanmean(x)) if len(x) else float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_nonpositive": float("nan"), "n": int(len(x))}
    rng = np.random.default_rng(SEED + len(x))
    n = len(x)
    draws = np.empty(reps, dtype=float)
    starts_max = max(1, n - block + 1)
    blocks_needed = int(math.ceil(n / block))
    for b in range(reps):
        starts = rng.integers(0, starts_max, size=blocks_needed)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        draws[b] = float(np.mean(sample))
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_nonpositive": float((1.0 + np.sum(draws <= 0.0)) / (reps + 1.0)),
        "n": int(n),
    }


def yearly_rankic(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {}
    return {str(int(year)): float(value) for year, value in series.groupby(series.index.year).mean().items() if np.isfinite(value)}


def opportunity_spread(signal: pd.DataFrame, target: pd.DataFrame, date_mask: np.ndarray, tail: float = 0.10) -> pd.Series:
    rows: dict[pd.Timestamp, float] = {}
    for date in signal.index[date_mask]:
        x = signal.loc[date]
        y = target.loc[date]
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        n = int(valid.sum())
        if n < MIN_ASSETS:
            continue
        count = max(1, int(math.floor(tail * n)))
        order = x[valid].sort_values()
        short_ids = order.index[:count]
        long_ids = order.index[-count:]
        rows[pd.Timestamp(date)] = float(y.loc[long_ids].mean() - y.loc[short_ids].mean())
    return pd.Series(rows, dtype=float).sort_index()


def flatten_features(features: list[pd.DataFrame], target: pd.DataFrame, date_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack([frame.to_numpy(float) for frame in features], axis=2)
    y = target.to_numpy(float)
    selected = np.broadcast_to(date_mask[:, None], y.shape) & np.isfinite(y)
    rows, cols = np.where(selected)
    X = stack[rows, cols, :]
    yy = y[rows, cols]
    dd = target.index.to_numpy()[rows]
    return X, yy, dd


def cap_training(X: np.ndarray, y: np.ndarray, cap: int = 450_000) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= cap:
        return X, y
    rng = np.random.default_rng(SEED + X.shape[1])
    index = np.sort(rng.choice(len(y), size=cap, replace=False))
    return X[index], y[index]


def prediction_rankic(dates: np.ndarray, pred: np.ndarray, y: np.ndarray) -> float:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "pred": pred, "y": y})
    values: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        valid = np.isfinite(group.pred) & np.isfinite(group.y)
        if int(valid.sum()) < MIN_ASSETS:
            continue
        value = spearmanr(group.loc[valid, "pred"], group.loc[valid, "y"]).statistic
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def model_comparison(base_features: list[pd.DataFrame], candidate: pd.DataFrame, target: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray, label: str) -> dict[str, Any]:
    X_train_base, y_train, _ = flatten_features(base_features, target, train_mask)
    X_train_aug, y_train_aug, _ = flatten_features(base_features + [candidate], target, train_mask)
    X_test_base, y_test, d_test = flatten_features(base_features, target, test_mask)
    X_test_aug, y_test_aug, d_test_aug = flatten_features(base_features + [candidate], target, test_mask)
    if not (np.array_equal(y_train, y_train_aug) and np.array_equal(y_test, y_test_aug) and np.array_equal(d_test, d_test_aug)):
        raise AssertionError("baseline and augmented samples differ")
    X_train_base, y_train_cap = cap_training(X_train_base, y_train)
    X_train_aug, y_train_aug_cap = cap_training(X_train_aug, y_train_aug)
    if not np.array_equal(y_train_cap, y_train_aug_cap):
        raise AssertionError("training cap differs")

    results: dict[str, Any] = {"stage": label, "n_train": int(len(y_train_cap)), "n_test": int(len(y_test))}
    models = {
        "ridge": (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=20.0)),
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=20.0)),
        ),
        "hgb": (
            HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=10.0, random_state=SEED),
            HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=10.0, random_state=SEED),
        ),
    }
    for name, (base_model, aug_model) in models.items():
        base_model.fit(X_train_base, y_train_cap)
        aug_model.fit(X_train_aug, y_train_aug_cap)
        pred_base = np.asarray(base_model.predict(X_test_base), dtype=float)
        pred_aug = np.asarray(aug_model.predict(X_test_aug), dtype=float)
        loss_base = np.square(y_test - pred_base)
        loss_aug = np.square(y_test - pred_aug)
        mse_base = float(np.mean(loss_base))
        mse_aug = float(np.mean(loss_aug))
        loss_frame = pd.DataFrame({"date": pd.to_datetime(d_test), "diff": loss_base - loss_aug}).groupby("date", sort=True).mean().diff
        inference = moving_block_bootstrap(loss_frame)
        results[name] = {
            "mse_base": mse_base,
            "mse_augmented": mse_aug,
            "gain_pct": float(100.0 * (mse_base - mse_aug) / mse_base) if mse_base > 0 else float("nan"),
            "rankic_base": prediction_rankic(d_test, pred_base, y_test),
            "rankic_augmented": prediction_rankic(d_test, pred_aug, y_test),
            "paired_loss_inference": inference,
        }
    return results


def main() -> None:
    base = load_base()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        membership = base.fetch_membership()
        counts = membership.groupby("code").size().sort_values(ascending=False)
        codes = counts.head(MAX_CODES).index.astype(str).tolist()
        daily = base.fetch_daily_panel(codes)
    finally:
        bs.logout()

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    for column in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "tradestatus", "isST"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.drop_duplicates(["date", "code"], keep="last")
    codes = sorted(set(codes) & set(daily.code.astype(str).unique()))
    dates = pd.DatetimeIndex(sorted(daily.date.unique()))

    open_ = pivot(daily, "open", dates, codes)
    high = pivot(daily, "high", dates, codes)
    low = pivot(daily, "low", dates, codes)
    close = pivot(daily, "close", dates, codes)
    amount = pivot(daily, "amount", dates, codes)
    turnover = pivot(daily, "turn", dates, codes)
    trade = pivot(daily, "tradestatus", dates, codes).eq(1.0)
    nonst = pivot(daily, "isST", dates, codes).eq(0.0)
    available = trade & nonst & (open_ > 0) & (high > 0) & (low > 0) & (close > 0) & (amount > 0)

    membership = membership.copy()
    membership["snapshot_date"] = pd.to_datetime(membership["snapshot_date"])
    snapshots = {int(year): set(group.code.astype(str)) for year, group in membership.groupby(membership.snapshot_date.dt.year)}
    member_mask = pd.DataFrame(False, index=dates, columns=codes)
    for date in dates:
        members = snapshots.get(int(date.year) - 1, set())
        if members:
            member_mask.loc[date, list(set(codes) & members)] = True
    eligible = available & member_mask

    log_close = np.log(close)
    returns = log_close.diff()
    ret5 = returns.rolling(5, min_periods=5).sum()
    ret20 = returns.rolling(20, min_periods=20).sum()
    ret60 = returns.rolling(60, min_periods=50).sum()
    vol20 = returns.rolling(20, min_periods=15).std(ddof=0)
    vol60 = returns.rolling(60, min_periods=45).std(ddof=0)
    log_amount = np.log(amount)
    log_turnover = np.log(turnover.clip(lower=1.0e-6))
    log_range = np.log(high / low)
    close_location = (close - low) / (high - low + EPS)
    path_efficiency = ret20 / (returns.abs().rolling(20, min_periods=20).sum() + EPS)

    market_return = returns.where(eligible).mean(axis=1)
    market_panel = pd.DataFrame(np.repeat(market_return.to_numpy()[:, None], len(codes), axis=1), index=dates, columns=codes)
    board_labels = {code: board_name(code) for code in codes}
    board_panel = pd.DataFrame(index=dates, columns=codes, dtype=float)
    for board in sorted(set(board_labels.values())):
        members = [code for code, name in board_labels.items() if name == board]
        if not members:
            continue
        board_return = returns[members].where(eligible[members]).mean(axis=1)
        board_panel.loc[:, members] = np.repeat(board_return.to_numpy()[:, None], len(members), axis=1)

    beta_market = strict_prior_beta(returns, market_panel)
    residual_market = returns - beta_market * market_panel
    board_excess = board_panel - market_panel
    beta_board = strict_prior_beta(residual_market, board_excess)
    idio_return = residual_market - beta_board * board_excess
    idio_ret20 = idio_return.rolling(20, min_periods=20).sum()

    turnover_shock = strict_prior_z(log_turnover, 60, 30)
    amount_shock = strict_prior_z(log_amount, 60, 30)
    range_shock = strict_prior_z(log_range, 60, 30)
    pressure = cs_rank(
        (cs_rank(turnover_shock, eligible) + cs_rank(amount_shock, eligible) + cs_rank(range_shock, eligible) - cs_rank(close_location, eligible)) / 4.0,
        eligible,
    )

    parent = cs_rank(-ret20, eligible)
    idio_parent = cs_rank(-idio_ret20, eligible)
    blend = cs_rank(0.5 * parent + 0.5 * idio_parent, eligible)
    candidates: dict[str, pd.DataFrame] = {
        "REV20_IDIOSYNCRATIC": idio_parent,
        "REV20_RAW_IDIO_BLEND": blend,
    }
    for label, base_signal in (("RAW", parent), ("IDIO", idio_parent), ("BLEND", blend)):
        active = base_signal.clip(lower=0.0)
        for weight in (0.5, 1.0, 1.5):
            candidates[f"REV20_{label}_PRESSURE_W{weight:g}"] = cs_rank(base_signal + weight * active * pressure, eligible)

    target = np.log(close.shift(-HORIZON) / open_.shift(-1))
    dev = stage_mask(dates, "development")
    val = stage_mask(dates, "validation")
    hold = stage_mask(dates, "holdout")

    parent_dev_ic = daily_rankic(parent, target, dev)
    selection_rows: list[dict[str, Any]] = []
    for name, signal in candidates.items():
        ic = daily_rankic(signal, target, dev)
        aligned = pd.concat([ic.rename("candidate"), parent_dev_ic.rename("parent")], axis=1).dropna()
        diff = aligned.candidate - aligned.parent
        years = yearly_rankic(ic)
        positive_share = float(np.mean([value > 0 for value in years.values()])) if years else 0.0
        score = float(ic.mean() + 0.25 * diff.mean() + 0.005 * positive_share) if len(ic) else -np.inf
        selection_rows.append({
            "candidate": name,
            "development_rankic": float(ic.mean()) if len(ic) else float("nan"),
            "development_parent_rankic": float(parent_dev_ic.mean()) if len(parent_dev_ic) else float("nan"),
            "development_increment": float(diff.mean()) if len(diff) else float("nan"),
            "development_positive_year_share": positive_share,
            "development_score": score,
            "development_yearly_rankic": years,
        })
    selection = pd.DataFrame(selection_rows).sort_values("development_score", ascending=False)
    selection.to_csv(OUT / "candidate_selection.csv", index=False)
    selected_name = str(selection.iloc[0].candidate)
    selected = candidates[selected_name]

    direct: dict[str, Any] = {"selected_candidate": selected_name, "stages": {}}
    for stage, mask in (("validation", val), ("holdout", hold)):
        candidate_ic = daily_rankic(selected, target, mask)
        parent_ic = daily_rankic(parent, target, mask)
        pair = pd.concat([candidate_ic.rename("candidate"), parent_ic.rename("parent")], axis=1).dropna()
        diff = pair.candidate - pair.parent
        candidate_spread = opportunity_spread(selected, target, mask)
        parent_spread = opportunity_spread(parent, target, mask)
        spread_pair = pd.concat([candidate_spread.rename("candidate"), parent_spread.rename("parent")], axis=1).dropna()
        cost = 2.0 * ONE_WAY_COST_BPS / 10_000.0
        years = yearly_rankic(candidate_ic)
        direct["stages"][stage] = {
            "candidate_rankic": float(candidate_ic.mean()),
            "parent_rankic": float(parent_ic.mean()),
            "rankic_increment": float(diff.mean()),
            "rankic_increment_inference": moving_block_bootstrap(diff),
            "candidate_yearly_rankic": years,
            "candidate_positive_year_share": float(np.mean([v > 0 for v in years.values()])) if years else float("nan"),
            "candidate_gross_tail_spread": float(candidate_spread.mean()),
            "parent_gross_tail_spread": float(parent_spread.mean()),
            "candidate_net_tail_spread_10bp_one_way": float(candidate_spread.mean() - cost),
            "parent_net_tail_spread_10bp_one_way": float(parent_spread.mean() - cost),
            "tail_spread_increment_inference": moving_block_bootstrap(spread_pair.candidate - spread_pair.parent),
            "n_ic_dates": int(len(candidate_ic)),
            "n_spread_dates": int(len(candidate_spread)),
        }

    market20 = market_panel.rolling(20, min_periods=20).sum()
    board20 = board_panel.rolling(20, min_periods=20).sum()
    base_features = [
        parent,
        cs_rank(ret5, eligible),
        cs_rank(ret20, eligible),
        cs_rank(ret60, eligible),
        cs_rank(vol20, eligible),
        cs_rank(vol60, eligible),
        cs_rank(turnover_shock, eligible),
        cs_rank(amount_shock, eligible),
        cs_rank(range_shock, eligible),
        cs_rank(close_location, eligible),
        cs_rank(path_efficiency, eligible),
        cs_rank(log_amount.rolling(20, min_periods=15).mean(), eligible),
        cs_rank(market20, eligible),
        cs_rank(board20, eligible),
        idio_parent,
        pressure,
    ]
    model_results = {
        "validation": model_comparison(base_features, selected, target, dev, val, "validation"),
        "holdout": model_comparison(base_features, selected, target, dev | val, hold, "holdout"),
    }

    val_direct = direct["stages"]["validation"]
    hold_direct = direct["stages"]["holdout"]
    val_hgb = model_results["validation"]["hgb"]
    hold_hgb = model_results["holdout"]["hgb"]
    gates = {
        "direct_increment_positive_both": bool(val_direct["rankic_increment"] > 0.001 and hold_direct["rankic_increment"] > 0.001),
        "direct_increment_inference_both": bool(val_direct["rankic_increment_inference"]["p_nonpositive"] <= 0.10 and hold_direct["rankic_increment_inference"]["p_nonpositive"] <= 0.10),
        "strong_model_gain_positive_both": bool(val_hgb["gain_pct"] > 0.0 and hold_hgb["gain_pct"] > 0.0),
        "strong_model_loss_inference_both": bool(val_hgb["paired_loss_inference"]["p_nonpositive"] <= 0.10 and hold_hgb["paired_loss_inference"]["p_nonpositive"] <= 0.10),
        "costed_spread_positive_both": bool(val_direct["candidate_net_tail_spread_10bp_one_way"] > 0.0 and hold_direct["candidate_net_tail_spread_10bp_one_way"] > 0.0),
        "costed_spread_beats_parent_both": bool(val_direct["candidate_net_tail_spread_10bp_one_way"] > val_direct["parent_net_tail_spread_10bp_one_way"] and hold_direct["candidate_net_tail_spread_10bp_one_way"] > hold_direct["parent_net_tail_spread_10bp_one_way"]),
        "holdout_breadth": bool(hold_direct["candidate_positive_year_share"] >= 0.60),
    }
    passed = bool(all(gates.values()))
    decision = {
        "study": "A-share 20-session reversal conditional upgrade",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_source": "Baostock daily adjusted prices and lagged annual HS300/ZZ500 membership snapshots",
        "history_status": "historical replay; development-only candidate selection; validation and holdout untouched by formula selection within this run",
        "selected_candidate": selected_name,
        "candidate_family_size": int(len(candidates)),
        "universe_codes": int(len(codes)),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "eligible_cells": int(eligible.to_numpy().sum()),
        "direct": direct,
        "models": model_results,
        "gates": gates,
        "passed": passed,
        "catalog_action": "eligible_for_default_off_non_independent_registration" if passed else "no_trade_repository_update",
        "limitations": [
            "Public-source robustness panel differs from the canonical RQData full A-share research panel.",
            "The diagnostic 20-session opportunities overlap and are not a continuous capital account.",
            "A pass would support a default-off historical component, not fresh-forward production activation.",
        ],
    }
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "direct_results.json", direct)
    write_json(OUT / "model_results.json", model_results)
    print("DECISION_JSON=" + json.dumps(decision, ensure_ascii=False, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
