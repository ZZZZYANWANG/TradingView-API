from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

CORE_PATH = Path("research/fluid_round9/run.py")
spec = importlib.util.spec_from_file_location("fluid_round9_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load round9 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

OUT = Path("research_outputs/fluid_futures_graph_hodge_round10")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
GRAPH_HISTORY = 720
GRAPH_REFRESH = 168
GRAPH_TOP_K = 3
HORIZONS = [4, 12, 24, 72, 120]
SMOOTHING = [12, 24, 72]
FAMILIES = [
    "cross_basis_pressure_release",
    "cross_shear_absorption",
    "cross_cycle_trap",
    "cross_cycle_unwind",
    "cross_pressure_basis_conversion",
    "cross_hodge_consensus",
]


def _pivot(panel: pd.DataFrame, column: str, times: pd.Index, symbols: list[str]) -> np.ndarray:
    return (
        panel.pivot(index="time", columns="symbol", values=column)
        .reindex(index=times, columns=symbols)
        .to_numpy(dtype=float)
    )


def _graph(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corr = pd.DataFrame(window).corr(min_periods=max(360, int(0.60 * len(window)))).to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    strength = np.clip(np.abs(corr) - 0.10, 0.0, None) ** 2
    directed = np.zeros_like(strength)
    for i in range(len(strength)):
        candidates = np.where(strength[i] > 0.0)[0]
        if len(candidates):
            keep = candidates[np.argsort(strength[i, candidates])[-min(GRAPH_TOP_K, len(candidates)):]]
            directed[i, keep] = strength[i, keep]
    weights = np.maximum(directed, directed.T)
    signed = np.sign(corr)
    signed[weights <= 0.0] = 0.0
    degree = weights.sum(axis=1)
    regularized = np.diag(degree) - weights + np.diag(0.10 + 0.05 * degree)
    return weights, signed, degree, regularized


def add_graph_states(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["time", "symbol"]).copy()
    times = pd.Index(sorted(p["time"].dropna().unique()))
    symbols = sorted(p["symbol"].dropna().unique().tolist())
    shear = _pivot(p, "shear_z", times, symbols)
    activity = _pivot(p, "activity_gap_z", times, symbols)
    flow = _pivot(p, "flow_gap_z", times, symbols)
    basis_velocity = _pivot(p, "basis_velocity_z", times, symbols)
    p_u = _pivot(p, "p_u", times, symbols)
    p_mom24 = _pivot(p, "p_mom24", times, symbols)

    graph_source = basis_velocity - np.nanmean(basis_velocity, axis=1, keepdims=True)
    n_times, n_symbols = shear.shape
    outputs = {
        "cross_graph_pressure": np.full((n_times, n_symbols), np.nan),
        "cross_graph_divergence": np.full((n_times, n_symbols), np.nan),
        "cross_graph_cycle_share": np.full((n_times, n_symbols), np.nan),
        "cross_graph_potential_share": np.full((n_times, n_symbols), np.nan),
        "cross_graph_peer_shear": np.full((n_times, n_symbols), np.nan),
        "cross_graph_degree": np.full((n_times, n_symbols), np.nan),
    }

    weights = signed = degree = regularized = None
    for t in range(GRAPH_HISTORY, n_times):
        if weights is None or ((t - GRAPH_HISTORY) % GRAPH_REFRESH == 0):
            weights, signed, degree, regularized = _graph(graph_source[t - GRAPH_HISTORY:t])
        assert weights is not None and signed is not None and degree is not None and regularized is not None
        velocity = shear[t]
        lag_activity = activity[t - 1] + 0.50 * flow[t - 1]
        valid = np.isfinite(velocity) & np.isfinite(lag_activity)
        if int(valid.sum()) < 6:
            continue
        v0 = np.where(np.isfinite(velocity), velocity, 0.0)
        a0 = np.where(np.isfinite(lag_activity), lag_activity, 0.0)
        signed_weights = weights * signed
        edge_flux = signed_weights * (v0[:, None] * a0[None, :] - a0[:, None] * v0[None, :])
        divergence = edge_flux.sum(axis=1)
        try:
            pressure = np.linalg.solve(regularized, divergence)
        except np.linalg.LinAlgError:
            pressure = np.linalg.pinv(regularized, rcond=1.0e-10) @ divergence
        pressure -= np.mean(pressure)
        gradient_flux = weights * (pressure[:, None] - pressure[None, :])
        cycle_flux = edge_flux - gradient_flux
        edge_energy = np.sqrt(np.sum(edge_flux ** 2, axis=1) / (degree + EPS))
        grad_energy = np.sqrt(np.sum(gradient_flux ** 2, axis=1) / (degree + EPS))
        cycle_energy = np.sqrt(np.sum(cycle_flux ** 2, axis=1) / (degree + EPS))
        scale = edge_energy + 0.25 * np.median(edge_energy[np.isfinite(edge_energy)]) + EPS
        outputs["cross_graph_pressure"][t] = pressure / scale
        outputs["cross_graph_divergence"][t] = divergence / (np.sqrt(degree + 1.0) * (edge_energy + EPS))
        outputs["cross_graph_cycle_share"][t] = cycle_energy / (edge_energy + grad_energy + cycle_energy + EPS)
        outputs["cross_graph_potential_share"][t] = grad_energy / (edge_energy + grad_energy + cycle_energy + EPS)
        outputs["cross_graph_peer_shear"][t] = (signed_weights @ v0) / (degree + EPS)
        outputs["cross_graph_degree"][t] = degree
        for values in outputs.values():
            values[t, ~valid] = np.nan

    state = pd.DataFrame({
        "time": np.repeat(times.to_numpy(), n_symbols),
        "symbol": np.tile(np.asarray(symbols, dtype=object), n_times),
    })
    for name, values in outputs.items():
        state[name] = values.reshape(-1)
    p = p.merge(state, on=["time", "symbol"], how="left", validate="one_to_one")

    parts = []
    for _, group in p.sort_values(["symbol", "time"]).groupby("symbol", sort=False):
        g = group.copy()
        pressure = g["cross_graph_pressure"].clip(-12.0, 12.0)
        divergence = g["cross_graph_divergence"].clip(-12.0, 12.0)
        cycle = g["cross_graph_cycle_share"].clip(0.0, 1.5)
        potential = g["cross_graph_potential_share"].clip(0.0, 1.5)
        cycle_change = cycle - cycle.shift(24)
        direction = np.sign(g["p_mom24"]).replace(0.0, np.nan)
        raw = {
            "cross_basis_pressure_release": -pressure * potential,
            "cross_shear_absorption": -g["p_u"] * divergence,
            "cross_cycle_trap": -direction * cycle,
            "cross_cycle_unwind": -direction * cycle_change,
            "cross_pressure_basis_conversion": -pressure * g["basis_z"],
        }
        normalized = {name: core.causal_z(values, 720, 240) for name, values in raw.items()}
        normalized["cross_hodge_consensus"] = (
            normalized["cross_basis_pressure_release"]
            + normalized["cross_shear_absorption"]
            + normalized["cross_cycle_trap"]
            + normalized["cross_cycle_unwind"]
        ) / 4.0
        for family, values in normalized.items():
            for span in SMOOTHING:
                g[f"{family}_ewm{span}"] = values.ewm(
                    span=span, min_periods=max(8, int(0.65 * span)), adjust=False
                ).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


BASE = core.FUTURES_BASE + [
    "cross_graph_pressure", "cross_graph_divergence", "cross_graph_cycle_share",
    "cross_graph_potential_share", "cross_graph_peer_shear", "cross_graph_degree",
]


def portfolio(frame: pd.DataFrame, factor: str, target: str, cost_bp: float, annual_periods: float) -> dict:
    previous = {}
    returns, turns, times = [], [], []
    for timestamp, group in frame.groupby("time", sort=True):
        z = group[["symbol", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 6:
            continue
        raw = np.tanh(z[factor].to_numpy(dtype=float) / 2.0)
        raw -= np.mean(raw)
        denominator = float(np.sum(np.abs(raw)))
        if denominator < EPS:
            continue
        w = raw / denominator
        weights = dict(zip(z["symbol"], w))
        targets = dict(zip(z["symbol"], z[target].to_numpy(dtype=float)))
        gross = float(sum(weights[s] * targets[s] for s in weights))
        names = set(previous) | set(weights)
        turnover = float(sum(abs(weights.get(s, 0.0) - previous.get(s, 0.0)) for s in names))
        returns.append(gross - cost_bp / 10000.0 * turnover)
        turns.append(turnover)
        times.append(timestamp)
        previous = weights
    values = np.asarray(returns, dtype=float)
    if len(values) < 20:
        return {"sharpe": np.nan, "annual_log_return": np.nan, "avg_turnover": np.nan, "years": {}}
    years = pd.to_datetime(pd.Series(times), utc=True).dt.year.to_numpy()
    return {
        "sharpe": float(np.mean(values) / (np.std(values, ddof=0) + EPS) * np.sqrt(annual_periods)),
        "annual_log_return": float(np.mean(values) * annual_periods),
        "avg_turnover": float(np.mean(turns)),
        "years": {str(y): float(np.mean(values[years == y]) * annual_periods) for y in np.unique(years)},
    }


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, sign: int, train_end: int, test_start: int, test_end: int) -> dict:
    target = f"y_{horizon}"
    work = panel.copy()
    work["__factor"] = sign * work[factor]
    train = core.issue_subset(work.loc[work["time"].dt.year <= train_end].copy(), horizon, "futures")
    test = core.issue_subset(work.loc[(work["time"].dt.year >= test_start) & (work["time"].dt.year <= test_end)].copy(), horizon, "futures")
    ts = core.time_series_ic(test, "__factor", target)
    cs, cs_values = core.cross_sectional_ic(test, "__factor", target, "time", 6)
    boot = core.block_bootstrap(cs_values, max(12, int(120 / horizon)))
    ridge = core.ridge_increment(train, test, BASE, "__factor", target, 200000)
    test["__residual"] = core.residual_values(train, test, BASE, "__factor")
    residual = core.time_series_ic(test, "__residual", target)
    corr = core.maximum_baseline_correlation(test, "__factor", BASE)
    account = portfolio(test, "__factor", target, 2.0, 365.25 * 24.0 / horizon)
    return {
        "factor": factor, "horizon": horizon, "sign": sign,
        "ts_ic": ts["mean"], "ts_positive_share": ts["positive_share"],
        "cs_ic": cs["mean"], "cs_bootstrap_low": boot["low"],
        "ridge_gain": ridge["gain"], "residual_ts_ic": residual["mean"],
        "max_baseline_corr": corr["max_abs"], "max_corr_feature": corr["feature"],
        "net_sharpe_2bp": account["sharpe"], "annual_log_return_2bp": account["annual_log_return"],
        "avg_turnover": account["avg_turnover"], "years": json.dumps(account["years"], sort_keys=True),
    }


def development(panel: pd.DataFrame, factor: str, horizon: int) -> dict:
    target = f"y_{horizon}"
    dev = core.issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon, "futures")
    ts = core.time_series_ic(dev, factor, target)
    sign = 1 if np.isfinite(ts["mean"]) and ts["mean"] >= 0.0 else -1
    return {"factor": factor, "family": factor.rsplit("_ewm", 1)[0], "horizon": horizon, "sign": sign,
            "development_ts_ic": abs(ts["mean"]) if np.isfinite(ts["mean"]) else np.nan,
            "development_positive_share": ts["positive_share"] if sign > 0 else 1.0 - ts["positive_share"]}


def score(row: dict) -> float:
    vals = [row["ts_ic"], row["residual_ts_ic"], row["ridge_gain"], row["net_sharpe_2bp"], row["max_baseline_corr"]]
    if not all(np.isfinite(vals)):
        return -np.inf
    return row["ts_ic"] + 0.8 * row["residual_ts_ic"] + 10.0 * row["ridge_gain"] + 0.003 * np.clip(row["net_sharpe_2bp"], -2.0, 2.0)


try:
    raw, metadata = core.load_futures()
    parts = []
    for symbol, frame in raw.groupby("symbol", sort=False):
        z = core.compute_futures_symbol(frame)
        z["symbol"] = symbol
        parts.append(z)
    panel = add_graph_states(pd.concat(parts, ignore_index=True))
    candidates = [f"{family}_ewm{span}" for family in FAMILIES for span in SMOOTHING]
    dev = pd.DataFrame([development(panel, factor, horizon) for factor in candidates for horizon in HORIZONS])
    winners = dev.sort_values(["family", "development_ts_ic"], ascending=[True, False]).groupby("family", as_index=False).head(1)
    rows = []
    for _, chosen in winners.iterrows():
        row = evaluate(panel, str(chosen["factor"]), int(chosen["horizon"]), int(chosen["sign"]), 2022, 2023, 2024)
        row["family"] = chosen["family"]
        row["development_ts_ic"] = chosen["development_ts_ic"]
        row["selection_score"] = score(row)
        row["validation_gate"] = bool(row["ts_ic"] > 0.008 and row["ts_positive_share"] >= 0.625 and row["residual_ts_ic"] > 0.004 and row["ridge_gain"] > 0.0 and row["net_sharpe_2bp"] > 0.0 and row["max_baseline_corr"] < 0.80)
        rows.append(row)
    validation = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    passers = validation.loc[validation["validation_gate"]]
    if len(passers):
        chosen = passers.iloc[0]
        selected_factor, selected_horizon, selected_sign = str(chosen["factor"]), int(chosen["horizon"]), int(chosen["sign"])
        confirmation = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2024, 2025, 2025)
        shadow = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2025, 2026, 2026)
        final_pass = bool(confirmation["ts_ic"] > 0.006 and confirmation["residual_ts_ic"] > 0.003 and confirmation["ridge_gain"] > 0.0 and confirmation["net_sharpe_2bp"] > 0.0 and shadow["ts_ic"] > 0.0 and shadow["residual_ts_ic"] > 0.0 and shadow["net_sharpe_2bp"] > 0.0)
    else:
        selected_factor = selected_horizon = selected_sign = confirmation = shadow = None
        final_pass = False
    dev.to_csv(OUT / "DEVELOPMENT_GRID.csv", index=False)
    winners.to_csv(OUT / "FAMILY_WINNERS.csv", index=False)
    validation.to_csv(OUT / "VALIDATION.csv", index=False)
    if confirmation is not None:
        pd.DataFrame([confirmation]).to_csv(OUT / "CONFIRMATION_2025.csv", index=False)
        pd.DataFrame([shadow]).to_csv(OUT / "SHADOW_2026.csv", index=False)
    summary = {"completed_utc": str(pd.Timestamp.now(tz="UTC")), "data": metadata,
               "mechanism": "cross-asset graph Hodge decomposition of spot-perpetual shear/activity edge flow",
               "selected_factor": selected_factor, "selected_horizon": selected_horizon, "selected_sign": selected_sign,
               "confirmation": confirmation, "shadow": shadow, "final_pass": final_pass,
               "governance": {"development": "2021-2022", "validation": "2023-2024", "confirmation": "2025", "shadow": "2026", "confirmation_sealed_until_validation": True}}
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable([summary] if final_pass else []), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Cross-Asset Futures Graph Hodge Test", "", "## Development family winners", "", winners.to_markdown(index=False), "", "## 2023-2024 validation", "", validation.to_markdown(index=False), "", f"Final pass: **{final_pass}**"]
    if confirmation is not None:
        lines += ["", "## 2025 confirmation", "", pd.DataFrame([confirmation]).to_markdown(index=False), "", "## 2026 shadow", "", pd.DataFrame([shadow]).to_markdown(index=False)]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
