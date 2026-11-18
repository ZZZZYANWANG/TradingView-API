from __future__ import annotations

import importlib.util
import json
import math
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

OUT = Path("research_outputs/fluid_round9_stock_horizon")
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = [5, 10, 20]


def add_slow_release_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in panel.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.copy()
        flux = g[ref.BASE_FACTOR]
        source = g[ref.SOURCE_FACTOR]
        residual5 = ref.prior_single_residual(flux, g["mom5_gap"])
        residual20 = ref.prior_single_residual(residual5, g["mom20_gap"])
        linear = 0.70 * residual20 + 0.30 * source
        sign_state = np.sign(linear)
        candidates = {
            "flux_source_release_ma5": linear.rolling(5, min_periods=4).mean(),
            "flux_source_release_ma10": linear.rolling(10, min_periods=7).mean(),
            "flux_source_release_ma20": linear.rolling(20, min_periods=14).mean(),
            "flux_source_release_ewm5": linear.ewm(span=5, min_periods=4, adjust=False).mean(),
            "flux_source_release_ewm10": linear.ewm(span=10, min_periods=7, adjust=False).mean(),
            "flux_source_release_persistence10": (
                linear.rolling(10, min_periods=7).mean()
                * sign_state.rolling(10, min_periods=7).mean().abs()
            ),
            "flux_source_release_persistence20": (
                linear.rolling(20, min_periods=14).mean()
                * sign_state.rolling(20, min_periods=14).mean().abs()
            ),
        }
        for name, values in candidates.items():
            g[name] = core.causal_z(values, 252, 100)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


CANDIDATES = [
    "flux_source_release_ma5",
    "flux_source_release_ma10",
    "flux_source_release_ma20",
    "flux_source_release_ewm5",
    "flux_source_release_ewm10",
    "flux_source_release_persistence10",
    "flux_source_release_persistence20",
]


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, train_end: int, test_start: int, test_end: int):
    target = f"y_{horizon}"
    train = panel.loc[panel["date"].dt.year <= train_end].copy()
    test = panel.loc[
        (panel["date"].dt.year >= test_start)
        & (panel["date"].dt.year <= test_end)
    ].copy()
    train = core.issue_subset(train, horizon, "stocks")
    test = core.issue_subset(test, horizon, "stocks")
    cs, values = core.cross_sectional_ic(test, factor, target, "date", 20)
    bootstrap = core.block_bootstrap(values, max(5, int(20 / horizon)))
    ridge = core.ridge_increment(train, test, core.STOCK_BASE, factor, target, 180000)
    test["residual"] = core.residual_values(train, test, core.STOCK_BASE, factor)
    residual, _ = core.cross_sectional_ic(test, "residual", target, "date", 20)
    correlation = core.maximum_baseline_correlation(test, factor, core.STOCK_BASE)
    portfolio = core.rank_portfolio(
        test, factor, target, "date", 10.0, 252.0 / horizon, 20
    )
    return {
        "factor": factor,
        "horizon": horizon,
        "cs_ic": cs["mean"],
        "positive_share": cs["positive_share"],
        "bootstrap_low": bootstrap["low"],
        "bootstrap_high": bootstrap["high"],
        "ridge_gain": ridge["gain"],
        "residual_cs_ic": residual["mean"],
        "max_baseline_corr": correlation["max_abs"],
        "max_corr_feature": correlation["feature"],
        "net_sharpe_10bp": portfolio["sharpe"],
        "annual_log_return_10bp": portfolio["annual_log_return"],
        "avg_turnover": portfolio["avg_turnover"],
        "years": json.dumps(portfolio["years"], sort_keys=True),
    }


def score(row: dict) -> float:
    needed = [
        row["cs_ic"], row["residual_cs_ic"], row["ridge_gain"],
        row["net_sharpe_10bp"], row["max_baseline_corr"],
    ]
    if not all(np.isfinite(needed)):
        return -np.inf
    return (
        row["cs_ic"]
        + 0.75 * row["residual_cs_ic"]
        + 8.0 * row["ridge_gain"]
        + 0.004 * np.clip(row["net_sharpe_10bp"], -2.0, 2.0)
        - 0.004 * max(row["max_baseline_corr"] - 0.75, 0.0)
    )


try:
    panel = add_slow_release_candidates(ref.panel.copy())
    rows = []
    for factor in CANDIDATES:
        for horizon in HORIZONS:
            row = evaluate(panel, factor, horizon, 2020, 2021, 2023)
            row["selection_score"] = score(row)
            row["validation_gate"] = bool(
                row["cs_ic"] > 0.006
                and row["residual_cs_ic"] > 0.004
                and row["ridge_gain"] > 0.0
                and row["bootstrap_low"] > -0.002
                and row["net_sharpe_10bp"] > 0.0
                and row["max_baseline_corr"] < 0.85
            )
            rows.append(row)
    validation = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    passers = validation.loc[validation["validation_gate"]]
    if len(passers):
        chosen = passers.iloc[0]
        selected_factor = str(chosen["factor"])
        selected_horizon = int(chosen["horizon"])
        confirmation = evaluate(
            panel, selected_factor, selected_horizon, 2023, 2024, 2025
        )
        shadow = evaluate(
            panel, selected_factor, selected_horizon, 2025, 2026, 2026
        )
        final_pass = bool(
            confirmation["cs_ic"] > 0.006
            and confirmation["residual_cs_ic"] > 0.003
            and confirmation["ridge_gain"] > 0.0
            and confirmation["net_sharpe_10bp"] > 0.10
            and shadow["cs_ic"] > 0.0
            and shadow["residual_cs_ic"] > 0.0
            and shadow["net_sharpe_10bp"] > 0.0
        )
    else:
        selected_factor = None
        selected_horizon = None
        confirmation = None
        shadow = None
        final_pass = False

    validation.to_csv(OUT / "VALIDATION_GRID.csv", index=False)
    if confirmation is not None:
        pd.DataFrame([confirmation]).to_csv(OUT / "CONFIRMATION_2024_2025.csv", index=False)
        pd.DataFrame([shadow]).to_csv(OUT / "SHADOW_2026.csv", index=False)
    summary = {
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "candidate_forms": CANDIDATES,
        "candidate_horizons": HORIZONS,
        "selected_factor": selected_factor,
        "selected_horizon": selected_horizon,
        "confirmation": confirmation,
        "shadow": shadow,
        "final_pass": final_pass,
        "governance": {
            "selection_period": "2021-2023",
            "confirmation_period": "2024-2025",
            "shadow_period": "2026-01 through 2026-10",
            "confirmation_read_only_after_validation_gate": True,
            "same_mechanism_not_new_formula_family": True,
            "fixed_current_universe_survivorship_warning": True,
        },
    }
    (OUT / "SUMMARY.json").write_text(
        json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    promoted = [summary] if final_pass else []
    (OUT / "PROMOTED.json").write_text(
        json.dumps(core.jsonable(promoted), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Fluid Stock Horizon and Persistence Test",
        "",
        validation.to_markdown(index=False),
        "",
        f"Selected: `{selected_factor}` at {selected_horizon} sessions"
        if selected_factor else "No candidate cleared the validation-cost gate; confirmation stayed sealed.",
        "",
        f"Final pass: **{final_pass}**",
    ]
    if confirmation is not None:
        lines.extend([
            "", "## 2024-2025 confirmation", "",
            pd.DataFrame([confirmation]).to_markdown(index=False),
            "", "## 2026 shadow", "",
            pd.DataFrame([shadow]).to_markdown(index=False),
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
