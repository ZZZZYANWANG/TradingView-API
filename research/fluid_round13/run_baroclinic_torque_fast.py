from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_baroclinic_torque as r  # noqa: E402

OUT = Path("research_outputs/fluid_baroclinic_torque_round13_fast")
OUT.mkdir(parents=True, exist_ok=True)
r.OUT = OUT


def _columnwise_corr(x: np.ndarray, y: np.ndarray, minimum: int = 8) -> np.ndarray:
    y2 = np.broadcast_to(y[:, None], x.shape)
    valid = np.isfinite(x) & np.isfinite(y2)
    n = valid.sum(axis=0).astype(float)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y2, 0.0)
    sx = xv.sum(axis=0)
    sy = yv.sum(axis=0)
    sxx = (xv * xv).sum(axis=0)
    syy = (yv * yv).sum(axis=0)
    sxy = (xv * yv).sum(axis=0)
    cov = sxy - sx * sy / np.maximum(n, 1.0)
    vx = sxx - sx * sx / np.maximum(n, 1.0)
    vy = syy - sy * sy / np.maximum(n, 1.0)
    out = cov / np.sqrt(np.maximum(vx * vy, 0.0) + r.EPS)
    out[(n < minimum) | (vx <= r.EPS) | (vy <= r.EPS)] = np.nan
    return out


def _rank_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].rank(method="average").to_numpy(dtype=float)


def _batch_for_horizon(panel: pd.DataFrame, factors: list[str], horizon: int) -> pd.DataFrame:
    target = f"y_{horizon}"
    dev = r.issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon)
    symbols = list(r.SYMBOLS)
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    m = len(factors)

    ic_times: list[pd.Timestamp] = []
    ic_values: list[np.ndarray] = []
    gross_rows: list[np.ndarray] = []
    turnover_rows: list[np.ndarray] = []
    previous = np.zeros((len(symbols), m), dtype=float)

    for timestamp, group in dev.groupby("time", sort=True):
        work = group[["symbol", target] + factors].replace([np.inf, -np.inf], np.nan)
        y = work[target].to_numpy(dtype=float)
        x_rank = _rank_matrix(work, factors)
        y_rank = work[target].rank(method="average").to_numpy(dtype=float)
        corr = _columnwise_corr(x_rank, y_rank, minimum=8)
        ic_times.append(pd.Timestamp(timestamp))
        ic_values.append(corr)

        x = work[factors].to_numpy(dtype=float)
        gross = np.full(m, np.nan, dtype=float)
        turnover = np.full(m, np.nan, dtype=float)
        current_symbols = work["symbol"].astype(str).tolist()
        locations = np.asarray([symbol_index[s] for s in current_symbols], dtype=int)
        for j in range(m):
            valid = np.isfinite(x[:, j]) & np.isfinite(y)
            valid_idx = np.flatnonzero(valid)
            if len(valid_idx) < 8:
                continue
            count = max(2, int(math.floor(len(valid_idx) * 0.30)))
            order = valid_idx[np.argsort(x[valid_idx, j], kind="mergesort")]
            low = order[:count]
            high = order[-count:]
            weights = np.zeros(len(symbols), dtype=float)
            weights[locations[low]] -= 0.5 / count
            weights[locations[high]] += 0.5 / count
            target_by_symbol = np.zeros(len(symbols), dtype=float)
            target_by_symbol[locations[valid_idx]] = y[valid_idx]
            gross[j] = float(weights @ target_by_symbol)
            turnover[j] = float(np.abs(weights - previous[:, j]).sum())
            previous[:, j] = weights
        gross_rows.append(gross)
        turnover_rows.append(turnover)

    ic = pd.DataFrame(np.vstack(ic_values), index=pd.DatetimeIndex(ic_times), columns=factors)
    cs_mean = ic.mean(axis=0, skipna=True)
    year_means = ic.groupby(ic.index.year).mean()
    min_year = year_means.min(axis=0, skipna=True)
    year_json: dict[str, str] = {}
    for factor in factors:
        payload = {
            str(int(year)): float(value)
            for year, value in year_means[factor].dropna().items()
        }
        year_json[factor] = json.dumps(payload, sort_keys=True)

    ts_rows: list[np.ndarray] = []
    for _, group in dev.groupby("symbol", sort=False):
        work = group[[target] + factors].replace([np.inf, -np.inf], np.nan)
        x_rank = _rank_matrix(work, factors)
        y_rank = work[target].rank(method="average").to_numpy(dtype=float)
        ts_rows.append(_columnwise_corr(x_rank, y_rank, minimum=80))
    ts = np.nanmean(np.vstack(ts_rows), axis=0)

    gross_matrix = np.vstack(gross_rows)
    turnover_matrix = np.vstack(turnover_rows)
    annual_periods = 365.25 * 24.0 / horizon

    rows: list[dict[str, Any]] = []
    for j, factor in enumerate(factors):
        raw_cs = float(cs_mean[factor])
        raw_ts = float(ts[j])
        raw_metric = 0.75 * (0.0 if not np.isfinite(raw_cs) else raw_cs) + 0.25 * (
            0.0 if not np.isfinite(raw_ts) else raw_ts
        )
        sign = 1 if raw_metric >= 0.0 else -1
        signed_cs = sign * raw_cs if np.isfinite(raw_cs) else float("nan")
        signed_ts = sign * raw_ts if np.isfinite(raw_ts) else float("nan")
        signed_min_year = sign * float(min_year[factor]) if np.isfinite(min_year[factor]) else float("nan")
        net = sign * gross_matrix[:, j] - 2.0e-4 * turnover_matrix[:, j]
        finite = np.isfinite(net)
        if int(finite.sum()) >= 20:
            net_values = net[finite]
            annual_return = float(np.mean(net_values) * annual_periods)
            sharpe = float(np.mean(net_values) / (np.std(net_values, ddof=0) + r.EPS) * np.sqrt(annual_periods))
        else:
            annual_return = float("nan")
            sharpe = float("nan")
        score = (
            (signed_min_year if np.isfinite(signed_min_year) else -1.0)
            + 0.30 * (signed_cs if np.isfinite(signed_cs) else -1.0)
            + 0.10 * (signed_ts if np.isfinite(signed_ts) else -1.0)
            + 0.01 * np.clip(sharpe, -2.0, 2.0)
        )
        signed_years = {
            key: sign * float(value)
            for key, value in json.loads(year_json[factor]).items()
        }
        rows.append({
            "factor": factor,
            "horizon": horizon,
            "sign": sign,
            "development_cs_rankic": signed_cs,
            "development_ts_ic": signed_ts,
            "development_min_year_rankic": signed_min_year,
            "development_2bp_return": annual_return,
            "development_2bp_sharpe": sharpe,
            "development_years": json.dumps(signed_years, sort_keys=True),
            "development_score": float(score),
        })
    return pd.DataFrame(rows)


def batch_development(panel: pd.DataFrame, candidate_map: dict[str, list[str]]) -> pd.DataFrame:
    reverse = {
        factor: family
        for family, factors in candidate_map.items()
        for factor in factors
    }
    factors = list(reverse)
    pieces: list[pd.DataFrame] = []
    for horizon in r.HORIZONS:
        part = _batch_for_horizon(panel, factors, horizon)
        part["family"] = part["factor"].map(reverse)
        pieces.append(part)
        print(f"batch_development_horizon={horizon} rows={len(part)}", flush=True)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    np.random.seed(r.SEED)
    klines, kline_audit = r.load_klines()
    metrics, funding, external_audit = r.load_external()
    merged = r.merge_external(klines, metrics, funding)

    local_parts: list[pd.DataFrame] = []
    for symbol, group in merged.groupby("symbol", sort=False):
        local = r.compute_local_features(group)
        local["symbol"] = symbol
        local_parts.append(local)
        print(f"local_features={symbol} rows={len(local)}", flush=True)
    local_panel = pd.concat(local_parts, ignore_index=True)
    weights, directed, graph_audit = r.build_graph(local_panel)
    graph_panel = r.graph_state(local_panel, weights, directed)
    panel = local_panel.merge(graph_panel, on=["time", "symbol"], how="left", validate="one_to_one")
    panel = r.add_candidate_states(panel)

    candidate_map = r.candidate_columns()
    development = batch_development(panel, candidate_map)
    development.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)
    winners = (
        development.sort_values(["family", "development_score"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(OUT / "FAMILY_WINNERS.csv", index=False)

    validation_rows: list[dict[str, Any]] = []
    for row in winners.itertuples(index=False):
        result = r.evaluate(panel, row.factor, int(row.horizon), int(row.sign), 2022, 2023, 2024)
        result["family"] = row.family
        validation_rows.append(result)
        print(f"validation_complete={row.family} factor={row.factor}", flush=True)
    validation = pd.DataFrame(validation_rows)
    raw_p = [float(x) if np.isfinite(x) else 1.0 for x in validation["bootstrap_p_nonpositive"]]
    validation["holm_p"] = r.holm_adjust(raw_p)
    validation["validation_gate"] = (
        (validation["cs_rankic"] > 0.010)
        & (validation["bootstrap_p_nonpositive"] <= 0.10)
        & (validation["ridge_mse_gain"] > 0.0)
        & (validation["residual_cs_rankic"] > 0.005)
        & (validation["max_parent_corr"] < 0.85)
        & (validation["annual_log_return_5bp"] > 0.0)
        & (validation["holm_p"] <= 0.10)
    )
    validation.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    opened = validation.loc[validation["validation_gate"]].copy()
    for row in opened.itertuples(index=False):
        result = r.evaluate(panel, row.factor, int(row.horizon), int(row.sign), 2024, 2025, 2026)
        result["family"] = row.family
        result["validation_holm_p"] = row.holm_p
        holdout_rows.append(result)
        print(f"holdout_complete={row.family} factor={row.factor}", flush=True)
    holdout = pd.DataFrame(holdout_rows)
    if not holdout.empty:
        holdout.to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    else:
        pd.DataFrame(columns=list(validation.columns)).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)

    promoted: list[dict[str, Any]] = []
    for row in holdout.itertuples(index=False):
        yearly = json.loads(row.yearly_rankic)
        yearly_positive = all(np.isfinite(value) and value > 0.0 for value in yearly.values()) if yearly else False
        passed = (
            row.cs_rankic > 0.010
            and row.bootstrap_p_nonpositive <= 0.10
            and row.ridge_mse_gain > 0.0
            and row.residual_cs_rankic > 0.005
            and row.max_parent_corr < 0.85
            and row.annual_log_return_5bp > 0.0
            and yearly_positive
        )
        if passed:
            promoted.append({
                "family": row.family,
                "factor": row.factor,
                "horizon": int(row.horizon),
                "sign": int(row.sign),
                "validation_holm_p": float(row.validation_holm_p),
                "holdout_cs_rankic": float(row.cs_rankic),
                "holdout_ridge_mse_gain": float(row.ridge_mse_gain),
                "holdout_residual_cs_rankic": float(row.residual_cs_rankic),
                "holdout_annual_log_return_5bp": float(row.annual_log_return_5bp),
            })

    diagnostics: list[dict[str, Any]] = []
    for row in opened.itertuples(index=False):
        for horizon in r.HORIZONS:
            for period_name, train_end, start, end in [
                ("validation", 2022, 2023, 2024),
                ("historical_holdout", 2024, 2025, 2026),
            ]:
                result = r.evaluate(panel, row.factor, horizon, int(row.sign), train_end, start, end)
                diagnostics.append({
                    "family": row.family,
                    "factor": row.factor,
                    "selected_horizon": int(row.horizon),
                    "tested_horizon": horizon,
                    "period": period_name,
                    "cs_rankic": result["cs_rankic"],
                    "ts_ic": result["ts_ic"],
                    "ridge_mse_gain": result["ridge_mse_gain"],
                    "residual_cs_rankic": result["residual_cs_rankic"],
                    "annual_log_return_5bp": result["annual_log_return_5bp"],
                })
    pd.DataFrame(diagnostics).to_csv(OUT / "MULTI_HORIZON_DIAGNOSTICS.csv", index=False)

    data_audit = {
        "klines": kline_audit,
        "external": external_audit,
        "graph": graph_audit,
        "merged_rows": int(len(merged)),
        "panel_rows": int(len(panel)),
        "finite_by_candidate": {
            factor: int(np.isfinite(panel[factor].to_numpy(float)).sum())
            for factors in candidate_map.values() for factor in factors
        },
        "availability_policy": {
            "klines": "factor at completed hour; label enters next-hour open",
            "metrics": "latest archived observation shifted by one complete hour",
            "funding": "last published event as-of joined and shifted by one complete hour",
        },
        "development_engine": "batched_vectorized_equivalent",
    }
    r.write_json(OUT / "DATA_AUDIT.json", data_audit)
    r.write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "candidate_families": len(r.FAMILIES),
        "candidate_columns": sum(len(v) for v in candidate_map.values()),
        "development_tests": int(len(development)),
        "validation_winners": int(len(validation)),
        "validation_gate_passed": int(validation["validation_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "promoted": promoted,
        "decision": "promote_only_if_PROMOTED_nonempty",
        "development_engine": "batched_vectorized_equivalent",
    }
    r.write_json(OUT / "SUMMARY.json", summary)
    (OUT / "REPORT.md").write_text(
        r.report(data_audit, winners, validation, holdout, promoted), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        r.write_json(OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
