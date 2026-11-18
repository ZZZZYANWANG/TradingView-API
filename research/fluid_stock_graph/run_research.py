from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import baostock as bs
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 20261118
RNG = np.random.default_rng(SEED)
EPS = 1.0e-12
OUT = Path("research_outputs/fluid_stock_graph")
OUT.mkdir(parents=True, exist_ok=True)

START_DATE = "2017-01-01"
END_DATE = "2026-11-17"
MAX_CODES = 180
PEERS = 8
GRAPH_LOOKBACK = 120
GRAPH_STEP = 20
HORIZONS = [1, 5, 10, 20]
CANDIDATE_NAMES = [
    "graph_pressure_absorption",
    "peer_sweeping_reattachment",
    "vorticity_enstrophy_release",
    "boundary_layer_reattachment",
    "relaminarization_continuation",
    "reynolds_stress_release",
    "graph_pressure_poisson_absorption",
]


def _rows(result: Any) -> pd.DataFrame:
    rows: list[list[str]] = []
    if result.error_code != "0":
        return pd.DataFrame()
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _query_snapshot(date: pd.Timestamp) -> list[str]:
    for lag in [0, 1, 2, 3, 4, 5, 6]:
        query_date = (date - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        try:
            result = bs.query_hs300_stocks(date=query_date)
            frame = _rows(result)
            if len(frame) >= 150 and "code" in frame:
                return sorted(frame["code"].astype(str).unique().tolist())
        except Exception:
            continue
    return []


def load_membership_snapshots() -> tuple[dict[pd.Timestamp, set[str]], dict[str, Any]]:
    snapshots: dict[pd.Timestamp, set[str]] = {}
    for year in range(2017, 2027):
        for month in [1, 4, 7, 10]:
            stamp = pd.Timestamp(year=year, month=month, day=15)
            codes = _query_snapshot(stamp)
            if codes:
                snapshots[stamp] = set(codes)
                print(f"snapshot {stamp.date()} members={len(codes)}", flush=True)
    try:
        current = _rows(bs.query_hs300_stocks())
        if len(current) >= 150:
            snapshots[pd.Timestamp(END_DATE)] = set(current["code"].astype(str))
    except Exception:
        pass
    if not snapshots:
        raise RuntimeError("No CSI300 membership snapshots were available")
    counts = Counter(code for members in snapshots.values() for code in members)
    selected = [code for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_CODES]]
    selected_set = set(selected)
    clipped = {date: members & selected_set for date, members in snapshots.items()}
    meta = {
        "snapshot_count": len(snapshots),
        "snapshot_min": str(min(snapshots)),
        "snapshot_max": str(max(snapshots)),
        "union_members": len(counts),
        "selected_codes": len(selected),
        "selection_rule": "top historical CSI300 membership frequency, code tiebreak",
        "snapshot_sizes_after_selection": {str(date.date()): len(members) for date, members in clipped.items()},
    }
    return clipped, meta


def download_daily(code: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,isST"
    error = ""
    for attempt in range(3):
        try:
            result = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="d",
                adjustflag="2",
            )
            frame = _rows(result)
            if not frame.empty:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = _safe_numeric(
                    frame,
                    ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "tradestatus", "isST"],
                )
                return frame.dropna(subset=["date", "code", "open", "close"])
            error = result.error_msg
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
        time.sleep(0.5 * (attempt + 1))
    print(f"daily failed {code}: {error}", flush=True)
    return pd.DataFrame()


def download_hourly(code: str) -> pd.DataFrame:
    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    error = ""
    for attempt in range(3):
        try:
            result = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="60",
                adjustflag="2",
            )
            frame = _rows(result)
            if not frame.empty:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = _safe_numeric(frame, ["open", "high", "low", "close", "volume", "amount"])
                return frame.dropna(subset=["date", "time", "code", "open", "high", "low", "close"])
            error = result.error_msg
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
        time.sleep(0.5 * (attempt + 1))
    print(f"hourly failed {code}: {error}", flush=True)
    return pd.DataFrame()


def aggregate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values(["code", "date", "time"]).copy()
    frame["block"] = frame.groupby(["code", "date"], sort=False).cumcount()
    counts = frame.groupby(["code", "date"], sort=False)["block"].transform("size")
    frame = frame.loc[(counts >= 4) & (frame["block"] < 4)].copy()
    frame["bar_return"] = np.log(frame["close"] / frame["open"])
    frame["bar_impact"] = np.log(frame["high"] / frame["low"]) / np.sqrt(frame["amount"].clip(lower=0.0) + 1.0)
    total_amount = frame.groupby(["code", "date"], sort=False)["amount"].transform("sum")
    frame["bar_share"] = frame["amount"] / (total_amount + EPS)
    pieces: list[pd.DataFrame] = []
    for value, prefix in [("bar_return", "bret"), ("bar_share", "bshare"), ("bar_impact", "bimpact")]:
        wide = frame.pivot_table(index=["date", "code"], columns="block", values=value, aggfunc="last")
        wide.columns = [f"{prefix}{int(column)}" for column in wide.columns]
        pieces.append(wide)
    daily = pd.concat(pieces, axis=1).reset_index()
    required = [f"bret{i}" for i in range(4)] + [f"bshare{i}" for i in range(4)] + [f"bimpact{i}" for i in range(4)]
    return daily.dropna(subset=required)


def build_membership_matrix(dates: pd.DatetimeIndex, codes: list[str], snapshots: dict[pd.Timestamp, set[str]]) -> pd.DataFrame:
    snap_dates = pd.DatetimeIndex(sorted(snapshots))
    base = pd.DataFrame(False, index=snap_dates, columns=codes)
    for date, members in snapshots.items():
        valid = [code for code in members if code in base.columns]
        base.loc[date, valid] = True
    all_dates = base.index.union(dates).sort_values()
    return base.reindex(all_dates).ffill().reindex(dates).fillna(False).astype(bool)


def pivot(panel: pd.DataFrame, value: str, dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    return panel.pivot(index="date", columns="code", values=value).reindex(index=dates, columns=codes)


def rolling_corr(left: pd.DataFrame, right: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    out = pd.DataFrame(index=left.index, columns=left.columns, dtype=float)
    for column in left.columns:
        out[column] = left[column].rolling(window, min_periods=min_periods).corr(right[column])
    return out


def dynamic_peers(
    returns: pd.DataFrame,
    eligible: pd.DataFrame,
    values: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    outputs = {name: pd.DataFrame(np.nan, index=returns.index, columns=returns.columns) for name in values}
    graph_records: list[dict[str, Any]] = []
    n_dates, n_codes = returns.shape
    for start in range(GRAPH_LOOKBACK, n_dates, GRAPH_STEP):
        stop = min(start + GRAPH_STEP, n_dates)
        active = eligible.iloc[start].to_numpy(bool)
        history = returns.iloc[start - GRAPH_LOOKBACK : start].copy()
        coverage = history.notna().mean().to_numpy() >= 0.80
        active &= coverage
        active_idx = np.flatnonzero(active)
        if len(active_idx) < 30:
            continue
        hist = history.iloc[:, active_idx]
        hist = hist.sub(hist.mean(axis=1), axis=0)
        corr = hist.corr(min_periods=80).to_numpy()
        weights = np.zeros((n_codes, n_codes), dtype=float)
        degrees: list[int] = []
        for local_i, global_i in enumerate(active_idx):
            row = corr[local_i].copy()
            row[local_i] = np.nan
            order = np.argsort(np.nan_to_num(row, nan=-np.inf))[::-1]
            keep = [index for index in order if np.isfinite(row[index]) and row[index] > 0.05][:PEERS]
            if len(keep) < 3:
                keep = [index for index in order if np.isfinite(row[index])][:PEERS]
            if not keep:
                continue
            raw = np.clip(row[keep], 0.02, None)
            raw = raw / raw.sum()
            weights[global_i, active_idx[np.asarray(keep)]] = raw
            degrees.append(len(keep))
        for name, value in values.items():
            segment = value.iloc[start:stop].to_numpy(float)
            finite = np.isfinite(segment)
            numerator = np.nan_to_num(segment, nan=0.0) @ weights.T
            denominator = finite.astype(float) @ weights.T
            result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.50)
            outputs[name].iloc[start:stop] = result
        graph_records.append(
            {
                "start_date": str(returns.index[start].date()),
                "end_date": str(returns.index[stop - 1].date()),
                "active_nodes": int(len(active_idx)),
                "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
                "min_degree": int(min(degrees)) if degrees else 0,
            }
        )
        if len(graph_records) % 20 == 0:
            print(f"graph windows={len(graph_records)} last={returns.index[start].date()}", flush=True)
    return outputs, graph_records


def cs_rank(signal: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    ranked = signal.where(eligible).rank(axis=1, pct=True, method="average") - 0.5
    return ranked * 2.0


def cs_z(signal: pd.DataFrame, eligible: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    x = signal.where(eligible)
    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=0)
    return x.sub(mean, axis=0).div(std + EPS, axis=0).clip(-clip, clip)


def future_label(open_price: pd.DataFrame, close_price: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return np.log(close_price.shift(-horizon) / open_price.shift(-1))


def rowwise_spearman(signal: pd.DataFrame, target: pd.DataFrame, eligible: pd.DataFrame, min_count: int = 35) -> pd.Series:
    x = signal.where(eligible).rank(axis=1, pct=True)
    y = target.where(eligible).rank(axis=1, pct=True)
    valid = x.notna() & y.notna()
    count = valid.sum(axis=1)
    xm = x.where(valid).sub(x.where(valid).mean(axis=1), axis=0)
    ym = y.where(valid).sub(y.where(valid).mean(axis=1), axis=0)
    numerator = (xm * ym).sum(axis=1)
    denominator = np.sqrt((xm * xm).sum(axis=1) * (ym * ym).sum(axis=1))
    out = numerator / denominator.replace(0.0, np.nan)
    return out.where(count >= min_count)


def period_mask(index: pd.DatetimeIndex, name: str) -> np.ndarray:
    if name == "development":
        return (index >= pd.Timestamp("2017-01-01")) & (index <= pd.Timestamp("2019-12-31"))
    if name == "validation":
        return (index >= pd.Timestamp("2020-01-01")) & (index <= pd.Timestamp("2022-12-31"))
    if name == "later":
        return index >= pd.Timestamp("2023-01-01")
    raise ValueError(name)


def ic_summary(series: pd.Series, mask: np.ndarray) -> dict[str, float]:
    values = series.loc[mask].dropna()
    if len(values) < 20:
        return {"n": int(len(values)), "mean": float("nan"), "std": float("nan"), "t": float("nan"), "positive_share": float("nan")}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return {
        "n": int(len(values)),
        "mean": mean,
        "std": std,
        "t": mean / (std / math.sqrt(len(values)) + EPS),
        "positive_share": float((values > 0).mean()),
    }


def block_bootstrap(series: pd.Series, mask: np.ndarray, block: int = 20, reps: int = 1500) -> dict[str, float]:
    values = series.loc[mask].dropna().to_numpy(float)
    n = len(values)
    if n < 40:
        return {"n": n, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_nonpositive": float("nan")}
    starts = np.arange(max(n - block + 1, 1))
    means = np.empty(reps, dtype=float)
    blocks_needed = math.ceil(n / block)
    for rep in range(reps):
        chosen = RNG.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block] for start in chosen])[:n]
        means[rep] = sample.mean()
    return {
        "n": n,
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_nonpositive": float((means <= 0.0).mean()),
    }


def flatten_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(float).reshape(-1)


def date_vector(index: pd.DatetimeIndex, columns: list[str]) -> np.ndarray:
    return np.repeat(index.to_numpy(), len(columns))


def long_daily_ic(date_values: np.ndarray, factor: np.ndarray, target: np.ndarray, period: np.ndarray) -> float:
    frame = pd.DataFrame({"date": date_values[period], "factor": factor[period], "target": target[period]}).dropna()
    if frame.empty:
        return float("nan")
    values: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) >= 35:
            corr = spearmanr(group["factor"], group["target"]).statistic
            if np.isfinite(corr):
                values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def model_diagnostics(
    base_features: list[pd.DataFrame],
    signal: pd.DataFrame,
    label: pd.DataFrame,
    eligible: pd.DataFrame,
) -> dict[str, Any]:
    columns = list(signal.columns)
    dates = date_vector(signal.index, columns)
    elig = eligible.to_numpy(bool).reshape(-1)
    x = np.column_stack([flatten_matrix(feature) for feature in base_features])
    f = flatten_matrix(signal)
    y = flatten_matrix(label)
    dev = elig & (dates >= np.datetime64("2017-01-01")) & (dates <= np.datetime64("2019-12-31"))
    val = elig & (dates >= np.datetime64("2020-01-01")) & (dates <= np.datetime64("2022-12-31"))
    later = elig & (dates >= np.datetime64("2023-01-01"))
    row_ok = np.isfinite(y) & np.isfinite(f) & (np.isfinite(x).mean(axis=1) >= 0.75)
    dev &= row_ok
    val &= row_ok
    later &= row_ok
    if dev.sum() < 5000 or val.sum() < 3000:
        return {"available": False, "counts": {"dev": int(dev.sum()), "val": int(val.sum()), "later": int(later.sum())}}
    train_idx = np.flatnonzero(dev)
    if len(train_idx) > 160000:
        train_idx = RNG.choice(train_idx, size=160000, replace=False)
    medians = np.nanmedian(x[train_idx], axis=0)
    medians[~np.isfinite(medians)] = 0.0
    x_filled = np.where(np.isfinite(x), x, medians)
    base = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    aug = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    base.fit(x_filled[train_idx], y[train_idx])
    aug.fit(np.column_stack([x_filled[train_idx], f[train_idx]]), y[train_idx])
    recon = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    recon.fit(x_filled[train_idx], f[train_idx])
    out: dict[str, Any] = {"available": True, "counts": {"dev": int(dev.sum()), "val": int(val.sum()), "later": int(later.sum())}}
    for name, mask in [("validation", val), ("later", later)]:
        if mask.sum() < 1000:
            continue
        p0 = base.predict(x_filled[mask])
        p1 = aug.predict(np.column_stack([x_filled[mask], f[mask]]))
        mse0 = mean_squared_error(y[mask], p0)
        mse1 = mean_squared_error(y[mask], p1)
        fhat = recon.predict(x_filled[mask])
        residual = f[mask] - fhat
        local_dates = dates[mask]
        out[name] = {
            "mse_base": float(mse0),
            "mse_augmented": float(mse1),
            "mse_gain_pct": float((mse0 - mse1) / (mse0 + EPS) * 100.0),
            "prediction_rankic_base": long_daily_ic(local_dates, p0, y[mask], np.ones(mask.sum(), dtype=bool)),
            "prediction_rankic_augmented": long_daily_ic(local_dates, p1, y[mask], np.ones(mask.sum(), dtype=bool)),
            "factor_reconstruction_r2": float(r2_score(f[mask], fhat)),
            "residual_rankic": long_daily_ic(local_dates, residual, y[mask], np.ones(mask.sum(), dtype=bool)),
        }
    return out


def nonlinear_increment(
    base_features: list[pd.DataFrame], signal: pd.DataFrame, label: pd.DataFrame, eligible: pd.DataFrame
) -> dict[str, Any]:
    columns = list(signal.columns)
    dates = date_vector(signal.index, columns)
    elig = eligible.to_numpy(bool).reshape(-1)
    x = np.column_stack([flatten_matrix(feature) for feature in base_features])
    f = flatten_matrix(signal)
    y = flatten_matrix(label)
    dev = elig & (dates >= np.datetime64("2017-01-01")) & (dates <= np.datetime64("2019-12-31"))
    val = elig & (dates >= np.datetime64("2020-01-01")) & (dates <= np.datetime64("2022-12-31"))
    later = elig & (dates >= np.datetime64("2023-01-01"))
    ok = np.isfinite(y) & np.isfinite(f) & (np.isfinite(x).mean(axis=1) >= 0.75)
    dev &= ok
    val &= ok
    later &= ok
    train = np.flatnonzero(dev)
    if len(train) > 90000:
        train = RNG.choice(train, size=90000, replace=False)
    med = np.nanmedian(x[train], axis=0)
    med[~np.isfinite(med)] = 0.0
    x = np.where(np.isfinite(x), x, med)
    params = dict(max_iter=120, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=2.0, random_state=SEED)
    base = HistGradientBoostingRegressor(**params).fit(x[train], y[train])
    aug = HistGradientBoostingRegressor(**params).fit(np.column_stack([x[train], f[train]]), y[train])
    out: dict[str, Any] = {}
    for name, mask in [("validation", val), ("later", later)]:
        if mask.sum() < 1000:
            continue
        p0 = base.predict(x[mask])
        p1 = aug.predict(np.column_stack([x[mask], f[mask]]))
        mse0 = mean_squared_error(y[mask], p0)
        mse1 = mean_squared_error(y[mask], p1)
        out[name] = {"mse_gain_pct": float((mse0 - mse1) / (mse0 + EPS) * 100.0)}
    return out


def overlapping_portfolio(
    signal: pd.DataFrame,
    next_return: pd.DataFrame,
    eligible: pd.DataFrame,
    period: str,
    hold: int,
    cost_bp: float,
) -> dict[str, float]:
    mask = period_mask(signal.index, period)
    sig = signal.where(eligible).loc[mask]
    ret = next_return.where(eligible).loc[mask]
    weights = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    for date in sig.index:
        row = sig.loc[date].dropna()
        if len(row) < 50:
            continue
        low = row.quantile(0.20)
        high = row.quantile(0.80)
        long_names = row.index[row >= high]
        short_names = row.index[row <= low]
        if len(long_names):
            weights.loc[date, long_names] = 0.5 / len(long_names)
        if len(short_names):
            weights.loc[date, short_names] = -0.5 / len(short_names)
    held = sum(weights.shift(lag).fillna(0.0) for lag in range(1, hold + 1)) / float(hold)
    pnl_gross = (held * ret).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    pnl = pnl_gross - turnover * cost_bp * 1.0e-4
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pnl) < 20:
        return {"n": int(len(pnl)), "annual_return": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "n": int(len(pnl)),
        "annual_return": float(pnl.mean() * 252.0),
        "sharpe": float(pnl.mean() / (pnl.std(ddof=1) + EPS) * math.sqrt(252.0)),
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.loc[pnl.index].mean()),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    valid = [(name, value) for name, value in pvalues.items() if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (m - rank) * value)
        adjusted[name] = min(running, 1.0)
    return adjusted


def main() -> None:
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        snapshots, membership_meta = load_membership_snapshots()
        counts = Counter(code for members in snapshots.values() for code in members)
        codes = [code for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_CODES]]
        daily_parts: list[pd.DataFrame] = []
        hourly_parts: list[pd.DataFrame] = []
        for index, code in enumerate(codes, start=1):
            daily = download_daily(code)
            hourly = download_hourly(code)
            if not daily.empty:
                daily_parts.append(daily)
            if not hourly.empty:
                hourly_parts.append(hourly)
            if index % 20 == 0:
                print(f"downloaded {index}/{len(codes)} daily={len(daily_parts)} hourly={len(hourly_parts)}", flush=True)
        if len(hourly_parts) < 80:
            raise RuntimeError(f"insufficient hourly coverage: {len(hourly_parts)} codes")
        daily = pd.concat(daily_parts, ignore_index=True).drop_duplicates(["date", "code"], keep="last")
        hourly = pd.concat(hourly_parts, ignore_index=True).drop_duplicates(["date", "time", "code"], keep="last")
    finally:
        bs.logout()

    intraday = aggregate_hourly(hourly)
    panel = daily.merge(intraday, on=["date", "code"], how="inner")
    panel = panel.loc[
        (panel["tradestatus"] == 1)
        & (panel["isST"] == 0)
        & (panel[["open", "high", "low", "close", "amount"]] > 0).all(axis=1)
    ].copy()
    codes = sorted(panel["code"].unique())
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    membership = build_membership_matrix(dates, codes, snapshots)
    available = pivot(panel.assign(available=True), "available", dates, codes).fillna(False).astype(bool)
    eligible = membership & available

    open_price = pivot(panel, "open", dates, codes)
    high_price = pivot(panel, "high", dates, codes)
    low_price = pivot(panel, "low", dates, codes)
    close_price = pivot(panel, "close", dates, codes)
    amount = pivot(panel, "amount", dates, codes)
    turnover = pivot(panel, "turn", dates, codes)
    log_close = np.log(close_price)
    returns = log_close.diff()
    daily_range = np.log(high_price / low_price)
    daily_impact = daily_range / np.sqrt(amount.clip(lower=0.0) + 1.0)
    bret = [pivot(panel, f"bret{i}", dates, codes) for i in range(4)]
    bshare = [pivot(panel, f"bshare{i}", dates, codes) for i in range(4)]
    bimpact = [pivot(panel, f"bimpact{i}", dates, codes) for i in range(4)]

    values = {"ret": returns, "impact": daily_impact, "amount": np.log1p(amount)}
    for index in range(4):
        values[f"bret{index}"] = bret[index]
        values[f"bshare{index}"] = bshare[index]
        values[f"bimpact{index}"] = bimpact[index]
    peers, graph_records = dynamic_peers(returns, eligible, values)

    peer_return = peers["ret"]
    residual_return = returns - peer_return
    residual_blocks = [bret[index] - peers[f"bret{index}"] for index in range(4)]
    participation_gradient = [bshare[index] - peers[f"bshare{index}"] for index in range(4)]
    impact_gap = sum(bimpact[index] - peers[f"bimpact{index}"] for index in range(4)) / 4.0
    coherence = rolling_corr(returns, peer_return, 20, 14).clip(-1.0, 1.0)
    coherence_gate = ((coherence + 1.0) / 2.0).clip(0.0, 1.0)
    low_impact_gate = np.exp(-cs_z(impact_gap, eligible).clip(lower=0.0))
    participation_divergence = sum(component.abs() for component in participation_gradient)

    own_mom5 = returns.rolling(5, min_periods=3).sum()
    peer_mom5 = peer_return.rolling(5, min_periods=3).sum()
    residual_mom3 = residual_return.rolling(3, min_periods=2).sum()
    residual_mom5 = residual_return.rolling(5, min_periods=3).sum()
    morning_residual = residual_blocks[0] + residual_blocks[1]
    afternoon_residual = residual_blocks[2] + residual_blocks[3]
    own_morning = bret[0] + bret[1]
    own_afternoon = bret[2] + bret[3]
    peer_morning = peers["bret0"] + peers["bret1"]
    peer_afternoon = peers["bret2"] + peers["bret3"]
    participation_rise = (bshare[2] + bshare[3]) - (bshare[0] + bshare[1])

    area = sum(0.5 * (residual_blocks[index] * participation_gradient[index + 1] - participation_gradient[index] * residual_blocks[index + 1]) for index in range(3))
    enstrophy_morning = residual_blocks[0].pow(2) + residual_blocks[1].pow(2)
    enstrophy_afternoon = residual_blocks[2].pow(2) + residual_blocks[3].pow(2)
    enstrophy_decay = (enstrophy_morning - enstrophy_afternoon).clip(lower=0.0)
    stress = sum(residual_blocks[index] * participation_gradient[index] for index in range(4))
    pressure_source = sum(participation_gradient[index] * residual_blocks[index].abs() for index in range(4))

    candidates: dict[str, pd.DataFrame] = {
        "graph_pressure_absorption": -residual_mom3 * participation_divergence * low_impact_gate * coherence_gate,
        "peer_sweeping_reattachment": (peer_mom5 - own_mom5) * cs_z(participation_rise, eligible).clip(lower=0.0) * low_impact_gate * coherence_gate,
        "vorticity_enstrophy_release": -area * (enstrophy_morning + enstrophy_afternoon) * low_impact_gate,
        "boundary_layer_reattachment": np.sign(peer_mom5) * (-own_morning * peer_morning).clip(lower=0.0) * (own_afternoon * peer_afternoon).clip(lower=0.0) * low_impact_gate,
        "relaminarization_continuation": peer_mom5 * enstrophy_decay * (coherence - coherence.shift(5)).clip(lower=0.0) * low_impact_gate,
        "reynolds_stress_release": -stress * (afternoon_residual - morning_residual) * low_impact_gate,
        "graph_pressure_poisson_absorption": -residual_mom5 * pressure_source * low_impact_gate * coherence_gate,
    }
    candidates = {name: cs_rank(value, eligible) for name, value in candidates.items()}

    labels = {horizon: future_label(open_price, close_price, horizon) for horizon in HORIZONS}
    selection_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    dev_mask = period_mask(dates, "development")
    for name, signal in candidates.items():
        best: dict[str, Any] | None = None
        for horizon in HORIZONS:
            ic = rowwise_spearman(signal, labels[horizon], eligible)
            summary = ic_summary(ic, dev_mask)
            orientation = 1.0 if summary["mean"] >= 0 else -1.0
            year_means = (ic.loc[dev_mask] * orientation).groupby(ic.loc[dev_mask].index.year).mean()
            stable_share = float((year_means > 0).mean()) if len(year_means) else 0.0
            score = abs(summary["t"]) * (0.5 + 0.5 * stable_share) if np.isfinite(summary["t"]) else -np.inf
            row = {"candidate": name, "horizon": horizon, "orientation": orientation, "dev_score": score, "dev_year_positive_share": stable_share, **{f"dev_{key}": value for key, value in summary.items()}}
            selection_rows.append(row)
            if best is None or score > best["dev_score"]:
                best = row
        assert best is not None
        selected[name] = best

    top_names = sorted(selected, key=lambda name: selected[name]["dev_score"], reverse=True)[:4]
    consensus = sum(cs_rank(candidates[name] * selected[name]["orientation"], eligible) for name in top_names) / len(top_names)
    candidates["fluid_graph_consensus"] = cs_rank(consensus, eligible)
    best_consensus: dict[str, Any] | None = None
    for horizon in HORIZONS:
        ic = rowwise_spearman(candidates["fluid_graph_consensus"], labels[horizon], eligible)
        summary = ic_summary(ic, dev_mask)
        orientation = 1.0 if summary["mean"] >= 0 else -1.0
        year_means = (ic.loc[dev_mask] * orientation).groupby(ic.loc[dev_mask].index.year).mean()
        stable_share = float((year_means > 0).mean()) if len(year_means) else 0.0
        score = abs(summary["t"]) * (0.5 + 0.5 * stable_share) if np.isfinite(summary["t"]) else -np.inf
        row = {"candidate": "fluid_graph_consensus", "horizon": horizon, "orientation": orientation, "dev_score": score, "dev_year_positive_share": stable_share, **{f"dev_{key}": value for key, value in summary.items()}}
        selection_rows.append(row)
        if best_consensus is None or score > best_consensus["dev_score"]:
            best_consensus = row
    assert best_consensus is not None
    selected["fluid_graph_consensus"] = best_consensus

    baseline = [
        cs_z(returns, eligible), cs_z(returns.rolling(5, min_periods=3).sum(), eligible), cs_z(returns.rolling(10, min_periods=6).sum(), eligible),
        cs_z(returns.rolling(20, min_periods=12).sum(), eligible), cs_z(returns.rolling(60, min_periods=36).sum(), eligible),
        cs_z(returns.rolling(5, min_periods=3).std(ddof=0), eligible), cs_z(returns.rolling(20, min_periods=12).std(ddof=0), eligible),
        cs_z(returns.rolling(60, min_periods=36).std(ddof=0), eligible), cs_z(daily_range, eligible), cs_z(np.log1p(amount), eligible),
        cs_z(turnover, eligible), cs_z(daily_impact, eligible), cs_z(peer_return, eligible), cs_z(peer_return.rolling(5, min_periods=3).sum(), eligible),
        cs_z(peer_return.rolling(20, min_periods=12).sum(), eligible), cs_z(residual_return, eligible),
        cs_z(residual_return.rolling(5, min_periods=3).sum(), eligible), cs_z(residual_return.rolling(20, min_periods=12).sum(), eligible),
        cs_z(coherence, eligible), cs_z(own_morning, eligible), cs_z(own_afternoon, eligible), cs_z(participation_rise, eligible),
        cs_z((close_price - low_price) / (high_price - low_price + EPS), eligible),
    ]

    evaluation_rows: list[dict[str, Any]] = []
    bootstrap_results: dict[str, dict[str, float]] = {}
    model_results: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    next_return = np.log(close_price.shift(-1) / close_price)
    portfolio_rows: list[dict[str, Any]] = []
    for name, spec in selected.items():
        horizon = int(spec["horizon"])
        orientation = float(spec["orientation"])
        signal = candidates[name] * orientation
        ic = rowwise_spearman(signal, labels[horizon], eligible)
        for period in ["development", "validation", "later"]:
            summary = ic_summary(ic, period_mask(dates, period))
            evaluation_rows.append({"candidate": name, "horizon": horizon, "period": period, **summary})
        bootstrap = block_bootstrap(ic, period_mask(dates, "validation"))
        bootstrap_results[name] = bootstrap
        pvalues[name] = bootstrap["p_nonpositive"]
        model_results[name] = model_diagnostics(baseline, signal, labels[horizon], eligible)
        for period in ["development", "validation", "later"]:
            for cost in [0.0, 5.0, 10.0]:
                metrics = overlapping_portfolio(signal, next_return, eligible, period, max(1, min(horizon, 20)), cost)
                portfolio_rows.append({"candidate": name, "horizon": horizon, "period": period, "cost_bp": cost, **metrics})

    holm = holm_adjust(pvalues)
    evaluation = pd.DataFrame(evaluation_rows)
    portfolio = pd.DataFrame(portfolio_rows)
    admissions: list[dict[str, Any]] = []
    for name, spec in selected.items():
        horizon = int(spec["horizon"])
        rows = evaluation.loc[evaluation["candidate"] == name].set_index("period")
        model = model_results[name]
        val_model = model.get("validation", {})
        later_model = model.get("later", {})
        val_port = portfolio.loc[(portfolio["candidate"] == name) & (portfolio["period"] == "validation") & (portfolio["cost_bp"] == 5.0)]
        later_port = portfolio.loc[(portfolio["candidate"] == name) & (portfolio["period"] == "later") & (portfolio["cost_bp"] == 5.0)]
        criteria = {
            "validation_rankic_positive": bool(rows.loc["validation", "mean"] > 0.015),
            "later_rankic_positive": bool(rows.loc["later", "mean"] > 0.0),
            "validation_block_p_below_5pct": bool(bootstrap_results[name]["p_nonpositive"] < 0.05),
            "validation_ridge_increment_positive": bool(val_model.get("mse_gain_pct", -np.inf) > 0.0),
            "validation_residual_rankic_positive": bool(val_model.get("residual_rankic", -np.inf) > 0.008),
            "parent_reconstruction_below_75pct": bool(val_model.get("factor_reconstruction_r2", np.inf) < 0.75),
            "validation_5bp_return_positive": bool((not val_port.empty) and val_port.iloc[0]["annual_return"] > 0.0),
            "later_5bp_return_positive": bool((not later_port.empty) and later_port.iloc[0]["annual_return"] > 0.0),
        }
        historical_admission = all(criteria.values())
        independent = historical_admission and holm.get(name, 1.0) < 0.10 and later_model.get("mse_gain_pct", -np.inf) > 0.0
        admissions.append({"candidate": name, "horizon": horizon, "orientation": float(spec["orientation"]), "criteria": criteria, "historical_default_off_admission": historical_admission, "independent_alpha": independent, "holm_adjusted_p": holm.get(name)})

    admitted_names = [row["candidate"] for row in admissions if row["historical_default_off_admission"]]
    best_name = admitted_names[0] if admitted_names else max(selected, key=lambda name: evaluation.loc[(evaluation["candidate"] == name) & (evaluation["period"] == "validation"), "mean"].iloc[0])
    best_spec = selected[best_name]
    best_signal = candidates[best_name] * float(best_spec["orientation"])
    best_nonlinear = nonlinear_increment(baseline, best_signal, labels[int(best_spec["horizon"])], eligible)

    pd.DataFrame(selection_rows).to_csv(OUT / "development_selection.csv", index=False)
    evaluation.to_csv(OUT / "period_rankic.csv", index=False)
    portfolio.to_csv(OUT / "cost_sweep.csv", index=False)
    pd.DataFrame(graph_records).to_csv(OUT / "graph_windows.csv", index=False)
    pd.DataFrame(admissions).to_json(OUT / "admission_table.json", orient="records", indent=2, force_ascii=False)
    with (OUT / "model_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(model_results, handle, indent=2, ensure_ascii=False)
    with (OUT / "bootstrap.json").open("w", encoding="utf-8") as handle:
        json.dump(bootstrap_results, handle, indent=2, ensure_ascii=False)

    data_audit = {
        "daily_rows": int(len(daily)), "hourly_rows": int(len(hourly)), "intraday_product_days": int(len(intraday)), "merged_rows": int(len(panel)),
        "codes": int(len(codes)), "dates": int(len(dates)), "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        "eligible_cells": int(eligible.to_numpy().sum()), "membership": membership_meta, "graph_windows": int(len(graph_records)),
        "candidate_count_before_consensus": len(CANDIDATE_NAMES), "consensus_members": top_names,
    }
    with (OUT / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(data_audit, handle, indent=2, ensure_ascii=False)

    summary = {
        "status": "completed", "research_scope": "A-share CSI300 historical-snapshot sample with 60-minute intraday bars",
        "splits": {"development": "2017-2019", "validation": "2020-2022", "later_history": "2023-latest"},
        "data_audit": data_audit, "selected_specs": selected, "admissions": admissions, "admitted_candidates": admitted_names,
        "best_candidate_for_review": best_name, "best_candidate_nonlinear_increment": best_nonlinear,
        "governance": {"candidate_orientation_and_horizon_selected_on_development_only": True, "validation_and_later_history_not_used_for_formula_selection": True, "survivorship_limit": "download universe is capped by historical CSI300 membership frequency; eligibility follows dated snapshots", "production_certified": False},
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
