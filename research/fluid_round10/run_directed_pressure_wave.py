from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("research/fluid_round10/run_graph_hodge.py")
spec = importlib.util.spec_from_file_location("fluid_graph_hodge", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load graph Hodge source")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
core = source.core

OUT = Path("research_outputs/fluid_directed_pressure_wave_round10")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
GRAPH_HISTORY = 252
GRAPH_REFRESH = 21
GRAPH_TOP_K = 6
HORIZONS = [1, 5, 10]
SMOOTHING = [3, 5, 10]


def _pivot(panel: pd.DataFrame, column: str, dates: pd.Index, symbols: list[str]) -> np.ndarray:
    return panel.pivot(index="date", columns="symbol", values=column).reindex(index=dates, columns=symbols).to_numpy(dtype=float)


def _sector_neutral(values: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    out = values.copy()
    for sector in np.unique(sectors):
        idx = np.where(sectors == sector)[0]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(out[:, idx], axis=1, keepdims=True)
        out[:, idx] -= mean
    return out


def _directed_graph(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = window.shape[1]
    lead = np.zeros((n, n), dtype=float)
    for target in range(n):
        y = window[1:, target]
        for source_idx in range(n):
            if source_idx == target:
                continue
            x = window[:-1, source_idx]
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < 140:
                continue
            sx, sy = np.std(x[valid]), np.std(y[valid])
            if sx < EPS or sy < EPS:
                continue
            lead[target, source_idx] = np.corrcoef(x[valid], y[valid])[0, 1]
    strength = np.clip(np.abs(lead) - 0.03, 0.0, None) ** 2
    weights = np.zeros_like(strength)
    for target in range(n):
        candidates = np.where(strength[target] > 0.0)[0]
        if len(candidates):
            keep = candidates[np.argsort(strength[target, candidates])[-min(GRAPH_TOP_K, len(candidates)):]]
            weights[target, keep] = strength[target, keep]
    signed = np.sign(lead)
    signed[weights <= 0.0] = 0.0
    return weights, signed


def add_directed_states(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    p = panel.sort_values(["date", "symbol"]).copy()
    dates = pd.Index(sorted(p["date"].dropna().unique()))
    symbols = sorted(p["symbol"].dropna().unique().tolist())
    sectors = np.asarray([core.STOCK_SECTORS[s] for s in symbols], dtype=object)
    r = _pivot(p, "r1", dates, symbols)
    u = _pivot(p, "u", dates, symbols)
    activity = _pivot(p, "signed_activity", dates, symbols)
    qz = _pivot(p, "qz", dates, symbols)
    residual = _sector_neutral(r, sectors)
    n_dates, n_symbols = r.shape
    states = {
        "directed_peer_u": np.full((n_dates, n_symbols), np.nan),
        "directed_peer_activity": np.full((n_dates, n_symbols), np.nan),
        "directed_peer_qz": np.full((n_dates, n_symbols), np.nan),
        "directed_arrival_gap": np.full((n_dates, n_symbols), np.nan),
        "directed_degree": np.full((n_dates, n_symbols), np.nan),
    }
    weights = signed = None
    for t in range(GRAPH_HISTORY, n_dates):
        if weights is None or ((t - GRAPH_HISTORY) % GRAPH_REFRESH == 0):
            weights, signed = _directed_graph(residual[t - GRAPH_HISTORY:t])
        assert weights is not None and signed is not None
        signed_weights = weights * signed
        degree = weights.sum(axis=1)
        u0 = np.where(np.isfinite(u[t]), u[t], 0.0)
        a0 = np.where(np.isfinite(activity[t]), activity[t], 0.0)
        q0 = np.where(np.isfinite(qz[t]), qz[t], 0.0)
        peer_u = (signed_weights @ u0) / (degree + EPS)
        peer_a = (signed_weights @ a0) / (degree + EPS)
        peer_q = (weights @ q0) / (degree + EPS)
        valid = np.isfinite(u[t]) & (degree > 0.0)
        states["directed_peer_u"][t] = peer_u
        states["directed_peer_activity"][t] = peer_a
        states["directed_peer_qz"][t] = peer_q
        states["directed_arrival_gap"][t] = peer_u - u[t]
        states["directed_degree"][t] = degree
        for values in states.values():
            values[t, ~valid] = np.nan
    state = pd.DataFrame({"date": np.repeat(dates.to_numpy(), n_symbols), "symbol": np.tile(np.asarray(symbols, dtype=object), n_dates)})
    for name, values in states.items():
        state[name] = values.reshape(-1)
    p = p.merge(state, on=["date", "symbol"], how="left", validate="one_to_one")

    parts = []
    names: list[str] = []
    for _, group in p.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.copy()
        peer_u = g["directed_peer_u"].clip(-12.0, 12.0)
        peer_a = g["directed_peer_activity"].clip(-12.0, 12.0)
        peer_q = g["directed_peer_qz"].clip(-12.0, 12.0)
        gap = g["directed_arrival_gap"].clip(-12.0, 12.0)
        own_a = g["signed_activity"].clip(-12.0, 12.0)
        raw = {
            "directed_wave_absorption": gap * (peer_a - own_a),
            "directed_unabsorbed_pressure": peer_u * np.exp(-g["u"].abs()) * (1.0 + peer_a.abs()),
            "directed_pressure_overshoot": -gap * g["u"].abs() * (1.0 + peer_q.clip(lower=0.0)),
            "directed_wave_coherence": peer_u * peer_a,
            "directed_activity_arrival": gap * peer_a,
            "directed_pressure_acceleration": peer_u.diff() * peer_a,
            "directed_wave_impedance": gap / (1.0 + (peer_a - own_a).abs()),
        }
        for base, values in raw.items():
            state_value = core.causal_z(values, 252, 100)
            for span in SMOOTHING:
                name = f"{base}_ewm{span}"
                g[name] = state_value.ewm(span=span, min_periods=max(3, int(0.65 * span)), adjust=False).mean()
                if name not in names:
                    names.append(name)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out, names


BASE = source.GRAPH_BASE + ["directed_peer_u", "directed_peer_activity", "directed_peer_qz", "directed_arrival_gap", "directed_degree"]


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, sign: int, train_end: int, start: int, end: int) -> dict:
    target = f"y_{horizon}"
    work = panel.copy(); work["__factor"] = sign * work[factor]
    train = core.issue_subset(work.loc[work["date"].dt.year <= train_end].copy(), horizon, "stocks")
    test = core.issue_subset(work.loc[(work["date"].dt.year >= start) & (work["date"].dt.year <= end)].copy(), horizon, "stocks")
    cs, values = core.cross_sectional_ic(test, "__factor", target, "date", 20)
    boot = core.block_bootstrap(values, max(4, int(20 / horizon)))
    ridge = core.ridge_increment(train, test, BASE, "__factor", target, 180000)
    test["__residual"] = core.residual_values(train, test, BASE, "__factor")
    residual, _ = core.cross_sectional_ic(test, "__residual", target, "date", 20)
    corr = core.maximum_baseline_correlation(test, "__factor", BASE)
    account = core.rank_portfolio(test, "__factor", target, "date", 10.0, 252.0 / horizon, 20)
    return {"factor": factor, "horizon": horizon, "sign": sign, "cs_ic": cs["mean"], "positive_share": cs["positive_share"], "n_times": cs["n_times"], "bootstrap_low": boot["low"], "bootstrap_high": boot["high"], "p_nonpositive": boot["p_nonpositive"], "ridge_gain": ridge["gain"], "residual_cs_ic": residual["mean"], "max_baseline_corr": corr["max_abs"], "max_corr_feature": corr["feature"], "net_sharpe_10bp": account["sharpe"], "annual_log_return_10bp": account["annual_log_return"], "avg_turnover": account["avg_turnover"], "years": json.dumps(account["years"], sort_keys=True)}


def development(panel: pd.DataFrame, factor: str, horizon: int) -> dict:
    target = f"y_{horizon}"
    dev = core.issue_subset(panel.loc[(panel["date"].dt.year >= 2014) & (panel["date"].dt.year <= 2020)].copy(), horizon, "stocks")
    cs, values = core.cross_sectional_ic(dev, factor, target, "date", 20)
    sign = 1 if np.isfinite(cs["mean"]) and cs["mean"] >= 0.0 else -1
    signed = sign * values
    boot = core.block_bootstrap(signed, max(4, int(20 / horizon)))
    dev["__factor"] = sign * dev[factor]
    account = core.rank_portfolio(dev, "__factor", target, "date", 10.0, 252.0 / horizon, 20)
    score = abs(cs["mean"]) + 0.004 * np.clip(account["sharpe"], -2.0, 2.0) + 0.20 * max(boot["low"], -0.02)
    return {"factor": factor, "horizon": horizon, "sign": sign, "development_cs_ic": abs(cs["mean"]) if np.isfinite(cs["mean"]) else np.nan, "development_bootstrap_low": boot["low"], "development_net_sharpe_10bp": account["sharpe"], "development_turnover": account["avg_turnover"], "development_score": score}


try:
    panel, candidates = add_directed_states(source.panel.copy())
    dev = pd.DataFrame([development(panel, factor, horizon) for factor in candidates for horizon in HORIZONS])
    eligible = dev.loc[(dev["development_cs_ic"] > 0.008) & (dev["development_net_sharpe_10bp"] > 0.0)].copy()
    chosen = (eligible if len(eligible) else dev).sort_values("development_score", ascending=False).iloc[0]
    selected_factor, selected_horizon, selected_sign = str(chosen["factor"]), int(chosen["horizon"]), int(chosen["sign"])
    validation = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2020, 2021, 2023)
    validation_gate = bool(validation["cs_ic"] > 0.008 and validation["residual_cs_ic"] > 0.005 and validation["ridge_gain"] > 0.0 and validation["bootstrap_low"] > -0.002 and validation["net_sharpe_10bp"] > 0.10 and validation["max_baseline_corr"] < 0.80)
    if validation_gate:
        confirmation = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2023, 2024, 2025)
        shadow = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2025, 2026, 2026)
        final_pass = bool(confirmation["cs_ic"] > 0.006 and confirmation["residual_cs_ic"] > 0.003 and confirmation["ridge_gain"] > 0.0 and confirmation["net_sharpe_10bp"] > 0.10 and shadow["cs_ic"] > 0.0 and shadow["residual_cs_ic"] > 0.0 and shadow["net_sharpe_10bp"] > 0.0)
    else:
        confirmation = shadow = None; final_pass = False
    dev.to_csv(OUT / "DEVELOPMENT_GRID.csv", index=False)
    pd.DataFrame([validation]).to_csv(OUT / "VALIDATION_2021_2023.csv", index=False)
    if confirmation is not None:
        pd.DataFrame([confirmation]).to_csv(OUT / "CONFIRMATION_2024_2025.csv", index=False)
        pd.DataFrame([shadow]).to_csv(OUT / "SHADOW_2026.csv", index=False)
    summary = {"completed_utc": str(pd.Timestamp.now(tz="UTC")), "mechanism": "strict-prior directed lag-network pressure-wave propagation and incomplete absorption", "selected_factor": selected_factor, "selected_horizon": selected_horizon, "selected_sign": selected_sign, "validation": validation, "validation_gate": validation_gate, "confirmation": confirmation, "shadow": shadow, "final_pass": final_pass, "governance": {"development": "2014-2020", "validation": "2021-2023", "confirmation": "2024-2025 sealed until gate", "shadow": "2026", "directed_graph_uses_only_prior_returns": True, "baseline_contains_directed_parent_states": True, "fixed_universe_warning": True}}
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable([summary] if final_pass else []), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Directed Pressure-Wave Propagation Test", "", "## Development grid", "", dev.sort_values("development_score", ascending=False).to_markdown(index=False), "", "## Frozen validation", "", pd.DataFrame([validation]).to_markdown(index=False), "", f"Validation gate: **{validation_gate}**", "", f"Final pass: **{final_pass}**"]
    if confirmation is not None:
        lines += ["", "## 2024-2025 confirmation", "", pd.DataFrame([confirmation]).to_markdown(index=False), "", "## 2026 shadow", "", pd.DataFrame([shadow]).to_markdown(index=False)]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"]); raise
