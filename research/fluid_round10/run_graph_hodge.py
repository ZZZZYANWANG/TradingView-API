from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REFINEMENT_PATH = Path("research/fluid_round9/run_stock_refinement.py")
spec = importlib.util.spec_from_file_location("fluid_stock_refinement", REFINEMENT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load stock refinement module")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
core = ref.core

OUT = Path("research_outputs/fluid_graph_hodge_round10")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
GRAPH_HISTORY = 126
GRAPH_REFRESH = 21
GRAPH_TOP_K = 6
HORIZONS = [5, 10, 20]
SMOOTHING = [5, 10, 20]
RAW_FAMILIES = [
    "potential_pressure_release",
    "divergence_absorption",
    "cycle_trap",
    "cycle_unwind",
    "pressure_cycle_conversion",
    "selective_pressure_release",
    "pressure_absorption_consensus",
    "hodge_state_consensus",
]


def _pivot(panel: pd.DataFrame, column: str, dates: pd.Index, symbols: list[str]) -> np.ndarray:
    return (
        panel.pivot(index="date", columns="symbol", values=column)
        .reindex(index=dates, columns=symbols)
        .to_numpy(dtype=float)
    )


def _sector_neutral(values: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    out = values.copy()
    for sector in np.unique(sectors):
        idx = np.where(sectors == sector)[0]
        block = out[:, idx]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(block, axis=1, keepdims=True)
        out[:, idx] = block - mean
    return out


def _graph_from_history(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(window)
    corr = frame.corr(min_periods=max(60, int(0.60 * len(frame)))).to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    strength = np.clip(np.abs(corr) - 0.15, 0.0, None) ** 2
    directed = np.zeros_like(strength)
    for i in range(len(strength)):
        row = strength[i]
        finite = np.where(row > 0.0)[0]
        if len(finite):
            keep = finite[np.argsort(row[finite])[-min(GRAPH_TOP_K, len(finite)):]]
            directed[i, keep] = row[keep]
    weights = np.maximum(directed, directed.T)
    signed = np.sign(corr)
    signed[weights <= 0.0] = 0.0
    degree = weights.sum(axis=1)
    laplacian = np.diag(degree) - weights
    regularized = laplacian + np.diag(0.12 + 0.04 * degree)
    return weights, signed, degree, regularized


def add_hodge_states(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["date", "symbol"]).copy()
    dates = pd.Index(sorted(p["date"].dropna().unique()))
    symbols = sorted(p["symbol"].dropna().unique().tolist())
    sectors = np.asarray([core.STOCK_SECTORS[symbol] for symbol in symbols], dtype=object)

    r = _pivot(p, "r1", dates, symbols)
    u = _pivot(p, "u", dates, symbols)
    activity = _pivot(p, "signed_activity", dates, symbols)
    qz = _pivot(p, "qz", dates, symbols)
    range_z = _pivot(p, "range_z", dates, symbols)
    mom20_gap = _pivot(p, "mom20_gap", dates, symbols)
    residual_returns = _sector_neutral(r, sectors)

    n_dates, n_symbols = r.shape
    outputs = {
        "graph_pressure": np.full((n_dates, n_symbols), np.nan),
        "graph_divergence": np.full((n_dates, n_symbols), np.nan),
        "graph_cycle_share": np.full((n_dates, n_symbols), np.nan),
        "graph_potential_share": np.full((n_dates, n_symbols), np.nan),
        "graph_peer_u": np.full((n_dates, n_symbols), np.nan),
        "graph_peer_activity": np.full((n_dates, n_symbols), np.nan),
        "graph_degree": np.full((n_dates, n_symbols), np.nan),
    }

    weights = signed = degree = regularized = None
    for t in range(GRAPH_HISTORY, n_dates):
        if weights is None or ((t - GRAPH_HISTORY) % GRAPH_REFRESH == 0):
            hist = residual_returns[t - GRAPH_HISTORY:t]
            weights, signed, degree, regularized = _graph_from_history(hist)
        assert weights is not None and signed is not None and degree is not None and regularized is not None

        current_u = u[t]
        lag_activity = activity[t - 1]
        current_activity = activity[t]
        valid = np.isfinite(current_u) & np.isfinite(lag_activity)
        if int(valid.sum()) < 20:
            continue
        u0 = np.where(np.isfinite(current_u), current_u, 0.0)
        a0 = np.where(np.isfinite(lag_activity), lag_activity, 0.0)
        signed_weights = weights * signed
        edge_flux = signed_weights * (
            u0[:, None] * a0[None, :] - a0[:, None] * u0[None, :]
        )
        divergence = edge_flux.sum(axis=1)
        try:
            pressure = np.linalg.solve(regularized, divergence)
        except np.linalg.LinAlgError:
            pressure = np.linalg.pinv(regularized, rcond=1.0e-10) @ divergence
        pressure = pressure - np.nanmean(pressure)
        gradient_flux = weights * (pressure[:, None] - pressure[None, :])
        circulation_flux = edge_flux - gradient_flux

        edge_energy = np.sqrt(np.sum(edge_flux ** 2, axis=1) / (degree + EPS))
        gradient_energy = np.sqrt(np.sum(gradient_flux ** 2, axis=1) / (degree + EPS))
        cycle_energy = np.sqrt(np.sum(circulation_flux ** 2, axis=1) / (degree + EPS))
        pressure_scale = edge_energy + 0.25 * np.nanmedian(edge_energy[np.isfinite(edge_energy)]) + EPS
        divergence_scale = np.sqrt(degree + 1.0) * (edge_energy + EPS)

        outputs["graph_pressure"][t] = pressure / pressure_scale
        outputs["graph_divergence"][t] = divergence / divergence_scale
        outputs["graph_cycle_share"][t] = cycle_energy / (edge_energy + gradient_energy + cycle_energy + EPS)
        outputs["graph_potential_share"][t] = gradient_energy / (edge_energy + gradient_energy + cycle_energy + EPS)
        outputs["graph_peer_u"][t] = (signed_weights @ u0) / (degree + EPS)
        a_current0 = np.where(np.isfinite(current_activity), current_activity, 0.0)
        outputs["graph_peer_activity"][t] = (signed_weights @ a_current0) / (degree + EPS)
        outputs["graph_degree"][t] = degree
        for name in outputs:
            outputs[name][t, ~valid] = np.nan

    state = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), n_symbols),
        "symbol": np.tile(np.asarray(symbols, dtype=object), n_dates),
    })
    for name, values in outputs.items():
        state[name] = values.reshape(-1)
    p = p.merge(state, on=["date", "symbol"], how="left", validate="one_to_one")

    records = []
    for _, group in p.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.copy()
        pressure = g["graph_pressure"].clip(-12.0, 12.0)
        divergence = g["graph_divergence"].clip(-12.0, 12.0)
        cycle = g["graph_cycle_share"].clip(0.0, 1.5)
        potential = g["graph_potential_share"].clip(0.0, 1.5)
        momentum_sign = np.sign(g["mom20_gap"]).replace(0.0, np.nan)
        cycle_change = cycle - cycle.shift(5)

        raw = {
            "potential_pressure_release": -pressure * potential,
            "divergence_absorption": -g["u"] * divergence,
            "cycle_trap": -momentum_sign * cycle,
            "cycle_unwind": -momentum_sign * cycle_change,
            "pressure_cycle_conversion": -pressure * cycle_change,
            "selective_pressure_release": -pressure * potential * np.exp(-2.0 * cycle),
        }
        normalized = {
            name: core.causal_z(values, 252, 100)
            for name, values in raw.items()
        }
        normalized["pressure_absorption_consensus"] = (
            normalized["potential_pressure_release"]
            + normalized["divergence_absorption"]
        ) / 2.0
        normalized["hodge_state_consensus"] = (
            normalized["potential_pressure_release"]
            + normalized["divergence_absorption"]
            + normalized["cycle_trap"]
            + normalized["cycle_unwind"]
        ) / 4.0
        for family, values in normalized.items():
            for span in SMOOTHING:
                g[f"{family}_ewm{span}"] = values.ewm(
                    span=span, min_periods=max(4, int(0.65 * span)), adjust=False
                ).mean()
        records.append(g)
    return pd.concat(records, ignore_index=True)


GRAPH_BASE = core.STOCK_BASE + [
    "graph_pressure",
    "graph_divergence",
    "graph_cycle_share",
    "graph_potential_share",
    "graph_peer_u",
    "graph_peer_activity",
    "graph_degree",
]


def evaluate(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
    sign: int,
    train_end: int,
    test_start: int,
    test_end: int,
    cost_bp: float = 10.0,
) -> dict:
    target = f"y_{horizon}"
    work = panel.copy()
    signed_factor = "__signed_factor"
    work[signed_factor] = sign * work[factor]
    train = work.loc[work["date"].dt.year <= train_end].copy()
    test = work.loc[
        (work["date"].dt.year >= test_start)
        & (work["date"].dt.year <= test_end)
    ].copy()
    train = core.issue_subset(train, horizon, "stocks")
    test = core.issue_subset(test, horizon, "stocks")
    cs, values = core.cross_sectional_ic(test, signed_factor, target, "date", 20)
    bootstrap = core.block_bootstrap(values, max(4, int(20 / horizon)))
    ridge = core.ridge_increment(train, test, GRAPH_BASE, signed_factor, target, 180000)
    test["__residual"] = core.residual_values(train, test, GRAPH_BASE, signed_factor)
    residual, _ = core.cross_sectional_ic(test, "__residual", target, "date", 20)
    correlation = core.maximum_baseline_correlation(test, signed_factor, GRAPH_BASE)
    portfolio = core.rank_portfolio(
        test, signed_factor, target, "date", cost_bp, 252.0 / horizon, 20
    )
    return {
        "factor": factor,
        "horizon": horizon,
        "sign": sign,
        "cs_ic": cs["mean"],
        "positive_share": cs["positive_share"],
        "n_times": cs["n_times"],
        "bootstrap_low": bootstrap["low"],
        "bootstrap_high": bootstrap["high"],
        "p_nonpositive": bootstrap["p_nonpositive"],
        "ridge_gain": ridge["gain"],
        "residual_cs_ic": residual["mean"],
        "max_baseline_corr": correlation["max_abs"],
        "max_corr_feature": correlation["feature"],
        "net_sharpe_10bp": portfolio["sharpe"],
        "annual_log_return_10bp": portfolio["annual_log_return"],
        "avg_turnover": portfolio["avg_turnover"],
        "years": json.dumps(portfolio["years"], sort_keys=True),
    }


def development_stat(panel: pd.DataFrame, factor: str, horizon: int) -> dict:
    target = f"y_{horizon}"
    dev = panel.loc[
        (panel["date"].dt.year >= 2014)
        & (panel["date"].dt.year <= 2020)
    ].copy()
    dev = core.issue_subset(dev, horizon, "stocks")
    result, values = core.cross_sectional_ic(dev, factor, target, "date", 20)
    sign = 1 if (np.isfinite(result["mean"]) and result["mean"] >= 0.0) else -1
    signed_values = sign * values
    boot = core.block_bootstrap(signed_values, max(4, int(20 / horizon)))
    return {
        "factor": factor,
        "family": factor.rsplit("_ewm", 1)[0],
        "horizon": horizon,
        "sign": sign,
        "development_cs_ic": abs(result["mean"]) if np.isfinite(result["mean"]) else np.nan,
        "development_positive_share": float(np.mean(signed_values > 0.0)) if len(signed_values) else np.nan,
        "development_bootstrap_low": boot["low"],
    }


def score(row: dict) -> float:
    values = [
        row["cs_ic"], row["residual_cs_ic"], row["ridge_gain"],
        row["net_sharpe_10bp"], row["max_baseline_corr"],
    ]
    if not all(np.isfinite(values)):
        return -np.inf
    return (
        row["cs_ic"]
        + 0.80 * row["residual_cs_ic"]
        + 10.0 * row["ridge_gain"]
        + 0.004 * np.clip(row["net_sharpe_10bp"], -2.0, 2.0)
        - 0.006 * max(row["max_baseline_corr"] - 0.65, 0.0)
    )


try:
    panel = add_hodge_states(ref.panel.copy())
    candidate_columns = [
        f"{family}_ewm{span}"
        for family in RAW_FAMILIES
        for span in SMOOTHING
    ]

    development_rows = [
        development_stat(panel, factor, horizon)
        for factor in candidate_columns
        for horizon in HORIZONS
    ]
    development = pd.DataFrame(development_rows)
    winners = (
        development.sort_values(
            ["family", "development_cs_ic", "development_bootstrap_low"],
            ascending=[True, False, False],
        )
        .groupby("family", as_index=False)
        .head(1)
    )

    validation_rows = []
    for _, chosen in winners.iterrows():
        row = evaluate(
            panel,
            str(chosen["factor"]),
            int(chosen["horizon"]),
            int(chosen["sign"]),
            train_end=2020,
            test_start=2021,
            test_end=2023,
        )
        row["family"] = chosen["family"]
        row["development_cs_ic"] = chosen["development_cs_ic"]
        row["selection_score"] = score(row)
        row["validation_gate"] = bool(
            row["cs_ic"] > 0.008
            and row["residual_cs_ic"] > 0.005
            and row["ridge_gain"] > 0.0
            and row["bootstrap_low"] > 0.0
            and row["net_sharpe_10bp"] > 0.0
            and row["max_baseline_corr"] < 0.75
        )
        validation_rows.append(row)
    validation = pd.DataFrame(validation_rows).sort_values(
        "selection_score", ascending=False
    )
    passers = validation.loc[validation["validation_gate"]]

    if len(passers):
        chosen = passers.iloc[0]
        selected_factor = str(chosen["factor"])
        selected_horizon = int(chosen["horizon"])
        selected_sign = int(chosen["sign"])
        confirmation = evaluate(
            panel, selected_factor, selected_horizon, selected_sign,
            train_end=2023, test_start=2024, test_end=2025,
        )
        shadow = evaluate(
            panel, selected_factor, selected_horizon, selected_sign,
            train_end=2025, test_start=2026, test_end=2026,
        )
        final_pass = bool(
            confirmation["cs_ic"] > 0.006
            and confirmation["residual_cs_ic"] > 0.003
            and confirmation["ridge_gain"] > 0.0
            and confirmation["bootstrap_low"] > -0.002
            and confirmation["net_sharpe_10bp"] > 0.10
            and confirmation["max_baseline_corr"] < 0.75
            and shadow["cs_ic"] > 0.0
            and shadow["residual_cs_ic"] > 0.0
            and shadow["ridge_gain"] > -0.0002
            and shadow["net_sharpe_10bp"] > 0.0
        )
    else:
        selected_factor = None
        selected_horizon = None
        selected_sign = None
        confirmation = None
        shadow = None
        final_pass = False

    development.to_csv(OUT / "DEVELOPMENT_GRID.csv", index=False)
    winners.to_csv(OUT / "FAMILY_WINNERS.csv", index=False)
    validation.to_csv(OUT / "VALIDATION.csv", index=False)
    if confirmation is not None:
        pd.DataFrame([confirmation]).to_csv(OUT / "CONFIRMATION_2024_2025.csv", index=False)
        pd.DataFrame([shadow]).to_csv(OUT / "SHADOW_2026.csv", index=False)

    summary = {
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "mechanism": "dynamic correlation-network Helmholtz-Hodge decomposition of lagged activity/velocity edge flow",
        "graph": {
            "history_sessions": GRAPH_HISTORY,
            "refresh_sessions": GRAPH_REFRESH,
            "top_k": GRAPH_TOP_K,
            "strict_prior_graph": True,
            "sector_neutral_graph_estimation": True,
        },
        "candidate_families": RAW_FAMILIES,
        "smoothing_spans": SMOOTHING,
        "horizons": HORIZONS,
        "development": "2014-2020",
        "validation": "2021-2023",
        "confirmation": "2024-2025, opened only after validation gate",
        "shadow": "2026 through latest available date",
        "selected_factor": selected_factor,
        "selected_horizon": selected_horizon,
        "selected_sign": selected_sign,
        "confirmation_result": confirmation,
        "shadow_result": shadow,
        "final_pass": final_pass,
        "governance": {
            "one_winner_per_mechanism_family_before_validation": True,
            "confirmation_read_only_after_validation_gate": True,
            "strong_baseline_includes_graph_parent_states": True,
            "fixed_current_universe_survivorship_warning": True,
            "production_certification": False,
        },
    }
    (OUT / "SUMMARY.json").write_text(
        json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT / "PROMOTED.json").write_text(
        json.dumps(core.jsonable([summary] if final_pass else []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Dynamic Graph Helmholtz–Hodge Fluid Factor Test",
        "",
        "The graph is estimated only from preceding sector-neutral returns and is refreshed every 21 sessions.",
        "Antisymmetric lagged activity–velocity edge flow is decomposed into a graph-gradient pressure field and a circulation residual.",
        "",
        "## Development family winners",
        "",
        winners.to_markdown(index=False),
        "",
        "## 2021–2023 validation",
        "",
        validation.to_markdown(index=False),
        "",
        f"Selected: `{selected_factor}` / {selected_horizon} sessions / sign {selected_sign}"
        if selected_factor else "No candidate cleared the frozen validation gate; confirmation remained sealed.",
        "",
        f"Final pass: **{final_pass}**",
    ]
    if confirmation is not None:
        lines.extend([
            "", "## 2024–2025 confirmation", "",
            pd.DataFrame([confirmation]).to_markdown(index=False),
            "", "## 2026 shadow", "",
            pd.DataFrame([shadow]).to_markdown(index=False),
        ])
    lines.extend([
        "",
        "The current large-cap universe is fixed and therefore retains survivorship-selection risk.",
        "Even a final pass is historical/default-off pending point-in-time A-share replication.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }
    (OUT / "FAILURE.json").write_text(
        json.dumps(failure, indent=2), encoding="utf-8"
    )
    print(failure["traceback"])
    raise
