from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT = Path("research/fluid_stock_graph/run_research.py")
OUT = Path("research_outputs/fluid_stock_daily")
CACHE = OUT / "cache"
MAX_CODES = 160
WORKERS = 10
HORIZONS = [1, 5, 10, 20]
EPS = 1.0e-12


def load_module():
    spec = importlib.util.spec_from_file_location("fluid_graph_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared research module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_CODES = MAX_CODES
    module.OUT = OUT
    return module


def download_one(code: str) -> dict[str, object]:
    module = load_module()
    login = bs.login()
    if login.error_code != "0":
        return {"code": code, "ok": False, "error": login.error_msg}
    try:
        frame = module.download_daily(code)
    finally:
        bs.logout()
    if frame.empty:
        return {"code": code, "ok": False, "error": "empty_daily"}
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE / f"{code.replace('.', '_')}.pkl")
    return {"code": code, "ok": True, "rows": len(frame)}


def rolling_cov(left: pd.DataFrame, right: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    out = pd.DataFrame(index=left.index, columns=left.columns, dtype=float)
    for column in left.columns:
        out[column] = left[column].rolling(window, min_periods=min_periods).cov(right[column])
    return out


def daily_ic_from_long(dates: np.ndarray, signal: np.ndarray, target: np.ndarray, minimum: int = 35) -> float:
    frame = pd.DataFrame({"date": dates, "signal": signal, "target": target}).dropna()
    values: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) >= minimum:
            value = spearmanr(group["signal"], group["target"]).statistic
            if np.isfinite(value):
                values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    module = load_module()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        snapshots, membership_meta = module.load_membership_snapshots()
    finally:
        bs.logout()
    counts = Counter(code for members in snapshots.values() for code in members)
    codes = [code for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_CODES]]

    receipts: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                receipt = future.result()
            except Exception as exc:  # noqa: BLE001
                receipt = {"code": code, "ok": False, "error": f"{type(exc).__name__}:{exc}"}
            receipts.append(receipt)
            if index % 20 == 0:
                print(f"daily downloads={index}/{len(codes)} usable={sum(bool(x.get('ok')) for x in receipts)}", flush=True)
    with (OUT / "download_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(receipts, handle, ensure_ascii=False, indent=2)
    good = [str(row["code"]) for row in receipts if row.get("ok")]
    if len(good) < 100:
        raise RuntimeError(f"insufficient daily universe: {len(good)}")

    daily = pd.concat([pd.read_pickle(CACHE / f"{code.replace('.', '_')}.pkl") for code in good], ignore_index=True)
    daily = daily.drop_duplicates(["date", "code"], keep="last")
    daily = daily.loc[
        (daily["tradestatus"] == 1)
        & (daily["isST"] == 0)
        & (daily[["open", "high", "low", "close", "amount"]] > 0).all(axis=1)
    ].copy()
    codes = sorted(daily["code"].unique())
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    membership = module.build_membership_matrix(dates, codes, snapshots)
    available = module.pivot(daily.assign(available=True), "available", dates, codes).fillna(False).astype(bool)
    eligible = membership & available

    open_price = module.pivot(daily, "open", dates, codes)
    high_price = module.pivot(daily, "high", dates, codes)
    low_price = module.pivot(daily, "low", dates, codes)
    close_price = module.pivot(daily, "close", dates, codes)
    amount = module.pivot(daily, "amount", dates, codes)
    turnover = module.pivot(daily, "turn", dates, codes)
    returns = np.log(close_price).diff()
    daily_range = np.log(high_price / low_price)
    log_amount = np.log1p(amount)
    impact = daily_range / np.sqrt(amount.clip(lower=0.0) + 1.0)

    peer_values, graph_records = module.dynamic_peers(
        returns,
        eligible,
        {"ret": returns, "amount": log_amount, "turn": turnover, "range": daily_range, "impact": impact},
    )
    peer_return = peer_values["ret"]
    peer_amount = peer_values["amount"]
    peer_turnover = peer_values["turn"]
    peer_range = peer_values["range"]
    peer_impact = peer_values["impact"]

    residual = returns - peer_return
    activity_gap = module.cs_z(log_amount - peer_amount, eligible)
    turnover_gap = module.cs_z(turnover - peer_turnover, eligible)
    impact_gap = module.cs_z(np.log(impact + EPS) - np.log(peer_impact + EPS), eligible)
    range_gap = module.cs_z(daily_range - peer_range, eligible)
    coherence = module.rolling_corr(returns, peer_return, 20, 14).clip(-1.0, 1.0)
    coherence_gate = ((coherence + 1.0) / 2.0).clip(0.0, 1.0)
    low_impact_gate = np.exp(-impact_gap.clip(lower=0.0))

    own_mom3 = returns.rolling(3, min_periods=2).sum()
    own_mom5 = returns.rolling(5, min_periods=3).sum()
    own_mom10 = returns.rolling(10, min_periods=6).sum()
    peer_mom3 = peer_return.rolling(3, min_periods=2).sum()
    peer_mom5 = peer_return.rolling(5, min_periods=3).sum()
    peer_mom10 = peer_return.rolling(10, min_periods=6).sum()
    residual_mom3 = residual.rolling(3, min_periods=2).sum()
    residual_mom5 = residual.rolling(5, min_periods=3).sum()
    residual_vol5 = residual.rolling(5, min_periods=3).std(ddof=0)
    residual_vol20 = residual.rolling(20, min_periods=12).std(ddof=0)
    laplacian = peer_return - returns
    pressure_source = laplacian * (activity_gap + 0.5 * turnover_gap)
    pressure_memory = pressure_source.rolling(5, min_periods=3).mean()
    participation_divergence = activity_gap.abs() + turnover_gap.abs()

    circulation = (residual * activity_gap.shift(1) - activity_gap * residual.shift(1)).rolling(5, min_periods=3).sum()
    enstrophy = residual.pow(2) + activity_gap.pow(2)
    enstrophy_fast = enstrophy.rolling(5, min_periods=3).mean()
    enstrophy_slow = enstrophy.shift(5).rolling(15, min_periods=9).mean()
    enstrophy_decay = (enstrophy_slow - enstrophy_fast).clip(lower=0.0)
    coherence_rise = (coherence - coherence.shift(5)).clip(lower=0.0)
    reynolds_stress = rolling_cov(residual, activity_gap, 20, 12)
    local_accel = residual.diff()
    convective = residual.shift(1) * residual.diff() / (residual.abs() + residual.shift(1).abs() + 0.01)
    forcing_coherence = module.rolling_corr(local_accel + convective, activity_gap.diff(), 20, 12).abs().clip(0.0, 1.0)

    prior_separation = (-residual_mom3.shift(3) * peer_mom3.shift(3)).clip(lower=0.0)
    current_reattachment = (residual_mom3 * peer_mom3).clip(lower=0.0)
    compression = (range_gap + impact_gap).clip(lower=0.0)
    source_absorption = (-pressure_source * residual_mom3).clip(lower=0.0)

    raw_candidates = {
        "daily_graph_pressure_absorption": -residual_mom3 * participation_divergence * low_impact_gate * coherence_gate,
        "daily_pressure_poisson_recovery": -residual_mom5 * source_absorption * low_impact_gate,
        "daily_vorticity_enstrophy_release": -circulation * enstrophy_decay * low_impact_gate,
        "daily_boundary_reattachment": np.sign(peer_mom10) * prior_separation * current_reattachment * (1.0 + activity_gap.abs()) * low_impact_gate,
        "daily_relaminarization_continuation": peer_mom10 * enstrophy_decay * coherence_rise * low_impact_gate,
        "daily_reynolds_stress_dissipation": -reynolds_stress * (residual_vol20 - residual_vol5).clip(lower=0.0) * np.sign(peer_mom10),
        "daily_unforced_material_derivative": -np.sign(peer_mom5) * (local_accel + convective).rolling(3, min_periods=2).mean() * (1.0 - forcing_coherence),
        "daily_compression_absorption_front": -residual_mom3 * compression * source_absorption * coherence_gate,
        "daily_peer_sweeping_reentry": (peer_mom5 - own_mom5) * (activity_gap.diff(3)).clip(lower=0.0) * current_reattachment * low_impact_gate,
    }
    candidates = {name: module.cs_rank(value, eligible) for name, value in raw_candidates.items()}
    labels = {horizon: module.future_label(open_price, close_price, horizon) for horizon in HORIZONS}

    dev_mask = module.period_mask(dates, "development")
    selection_rows: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    for name, signal in candidates.items():
        best = None
        for horizon in HORIZONS:
            ic = module.rowwise_spearman(signal, labels[horizon], eligible)
            summary = module.ic_summary(ic, dev_mask)
            orientation = 1.0 if summary["mean"] >= 0.0 else -1.0
            years = (ic.loc[dev_mask] * orientation).groupby(ic.loc[dev_mask].index.year).mean()
            stable_share = float((years > 0).mean()) if len(years) else 0.0
            score = abs(summary["t"]) * (0.5 + 0.5 * stable_share) if np.isfinite(summary["t"]) else -np.inf
            row = {"candidate": name, "horizon": horizon, "orientation": orientation, "dev_score": score, "dev_year_positive_share": stable_share, **{f"dev_{key}": value for key, value in summary.items()}}
            selection_rows.append(row)
            if best is None or score > best["dev_score"]:
                best = row
        selected[name] = best

    top_names = sorted(selected, key=lambda name: selected[name]["dev_score"], reverse=True)[:4]
    candidates["daily_fluid_graph_consensus"] = module.cs_rank(sum(module.cs_rank(candidates[name] * float(selected[name]["orientation"]), eligible) for name in top_names) / len(top_names), eligible)
    best = None
    for horizon in HORIZONS:
        ic = module.rowwise_spearman(candidates["daily_fluid_graph_consensus"], labels[horizon], eligible)
        summary = module.ic_summary(ic, dev_mask)
        orientation = 1.0 if summary["mean"] >= 0.0 else -1.0
        years = (ic.loc[dev_mask] * orientation).groupby(ic.loc[dev_mask].index.year).mean()
        stable_share = float((years > 0).mean()) if len(years) else 0.0
        score = abs(summary["t"]) * (0.5 + 0.5 * stable_share) if np.isfinite(summary["t"]) else -np.inf
        row = {"candidate": "daily_fluid_graph_consensus", "horizon": horizon, "orientation": orientation, "dev_score": score, "dev_year_positive_share": stable_share, **{f"dev_{key}": value for key, value in summary.items()}}
        selection_rows.append(row)
        if best is None or score > best["dev_score"]:
            best = row
    selected["daily_fluid_graph_consensus"] = best

    baseline = [
        module.cs_z(returns, eligible), module.cs_z(own_mom5, eligible), module.cs_z(own_mom10, eligible),
        module.cs_z(returns.rolling(20, min_periods=12).sum(), eligible), module.cs_z(returns.rolling(60, min_periods=36).sum(), eligible),
        module.cs_z(returns.rolling(5, min_periods=3).std(ddof=0), eligible), module.cs_z(returns.rolling(20, min_periods=12).std(ddof=0), eligible),
        module.cs_z(daily_range, eligible), module.cs_z(log_amount, eligible), module.cs_z(turnover, eligible), module.cs_z(impact, eligible),
        module.cs_z(peer_return, eligible), module.cs_z(peer_mom5, eligible), module.cs_z(peer_mom10, eligible),
        module.cs_z(residual, eligible), module.cs_z(residual_mom3, eligible), module.cs_z(residual_mom5, eligible),
        module.cs_z(residual_vol5, eligible), module.cs_z(residual_vol20, eligible), module.cs_z(coherence, eligible),
        activity_gap, turnover_gap, impact_gap, range_gap, module.cs_z(laplacian, eligible), module.cs_z(pressure_source, eligible),
        module.cs_z((close_price - low_price) / (high_price - low_price + EPS), eligible),
    ]

    rankic_rows: list[dict[str, object]] = []
    bootstrap: dict[str, dict[str, float]] = {}
    model_results: dict[str, object] = {}
    portfolio_rows: list[dict[str, object]] = []
    next_return = np.log(close_price.shift(-1) / close_price)
    pvalues: dict[str, float] = {}
    for name, spec in selected.items():
        horizon = int(spec["horizon"])
        signal = candidates[name] * float(spec["orientation"])
        ic = module.rowwise_spearman(signal, labels[horizon], eligible)
        for period in ["development", "validation", "later"]:
            rankic_rows.append({"candidate": name, "horizon": horizon, "period": period, **module.ic_summary(ic, module.period_mask(dates, period))})
        bootstrap[name] = module.block_bootstrap(ic, module.period_mask(dates, "validation"))
        pvalues[name] = bootstrap[name]["p_nonpositive"]
        model_results[name] = module.model_diagnostics(baseline, signal, labels[horizon], eligible)
        for period in ["development", "validation", "later"]:
            for cost in [0.0, 5.0, 10.0]:
                portfolio_rows.append({"candidate": name, "horizon": horizon, "period": period, "cost_bp": cost, **module.overlapping_portfolio(signal, next_return, eligible, period, max(1, min(horizon, 20)), cost)})

    rankic = pd.DataFrame(rankic_rows)
    portfolios = pd.DataFrame(portfolio_rows)
    holm = module.holm_adjust(pvalues)
    admissions: list[dict[str, object]] = []
    for name, spec in selected.items():
        rows = rankic.loc[rankic["candidate"] == name].set_index("period")
        model = model_results[name]
        val_model = model.get("validation", {}) if isinstance(model, dict) else {}
        later_model = model.get("later", {}) if isinstance(model, dict) else {}
        val_port = portfolios.loc[(portfolios["candidate"] == name) & (portfolios["period"] == "validation") & (portfolios["cost_bp"] == 5.0)]
        later_port = portfolios.loc[(portfolios["candidate"] == name) & (portfolios["period"] == "later") & (portfolios["cost_bp"] == 5.0)]
        criteria = {
            "validation_rankic_gt_0015": bool(rows.loc["validation", "mean"] > 0.015),
            "later_rankic_positive": bool(rows.loc["later", "mean"] > 0.0),
            "validation_block_p_lt_005": bool(bootstrap[name]["p_nonpositive"] < 0.05),
            "validation_ridge_gain_positive": bool(val_model.get("mse_gain_pct", -np.inf) > 0.0),
            "validation_residual_rankic_gt_0008": bool(val_model.get("residual_rankic", -np.inf) > 0.008),
            "reconstruction_r2_lt_075": bool(val_model.get("factor_reconstruction_r2", np.inf) < 0.75),
            "validation_5bp_return_positive": bool((not val_port.empty) and val_port.iloc[0]["annual_return"] > 0.0),
            "later_5bp_return_positive": bool((not later_port.empty) and later_port.iloc[0]["annual_return"] > 0.0),
        }
        historical = all(criteria.values())
        independent = historical and holm.get(name, 1.0) < 0.10 and later_model.get("mse_gain_pct", -np.inf) > 0.0
        admissions.append({"candidate": name, "horizon": int(spec["horizon"]), "orientation": float(spec["orientation"]), "criteria": criteria, "historical_default_off_admission": historical, "independent_alpha": independent, "holm_adjusted_p": holm.get(name)})

    admitted = [row["candidate"] for row in admissions if row["historical_default_off_admission"]]
    review = admitted[0] if admitted else max(selected, key=lambda name: rankic.loc[(rankic["candidate"] == name) & (rankic["period"] == "validation"), "mean"].iloc[0])
    review_signal = candidates[review] * float(selected[review]["orientation"])
    nonlinear = module.nonlinear_increment(baseline, review_signal, labels[int(selected[review]["horizon"])], eligible)

    pd.DataFrame(selection_rows).to_csv(OUT / "development_selection.csv", index=False)
    rankic.to_csv(OUT / "period_rankic.csv", index=False)
    portfolios.to_csv(OUT / "cost_sweep.csv", index=False)
    pd.DataFrame(graph_records).to_csv(OUT / "graph_windows.csv", index=False)
    with (OUT / "bootstrap.json").open("w", encoding="utf-8") as handle:
        json.dump(bootstrap, handle, ensure_ascii=False, indent=2)
    with (OUT / "model_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(model_results, handle, ensure_ascii=False, indent=2)
    with (OUT / "admission_table.json").open("w", encoding="utf-8") as handle:
        json.dump(admissions, handle, ensure_ascii=False, indent=2)
    audit = {
        "daily_rows": len(daily), "codes": len(codes), "dates": len(dates), "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        "eligible_cells": int(eligible.to_numpy().sum()), "graph_windows": len(graph_records), "membership": membership_meta,
        "candidate_count": len(raw_candidates), "consensus_members": top_names,
    }
    with (OUT / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    summary = {
        "status": "completed", "scope": "A-share daily dynamic peer graph fluid mechanics",
        "splits": {"development": "2017-2019", "validation": "2020-2022", "later": "2023-latest"},
        "audit": audit, "selected_specs": selected, "admissions": admissions, "admitted_candidates": admitted,
        "review_candidate": review, "nonlinear_increment": nonlinear,
        "governance": {"selection_on_development_only": True, "point_in_time_membership_eligibility": True, "historical_membership_frequency_sample_cap": True, "production_certified": False},
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
