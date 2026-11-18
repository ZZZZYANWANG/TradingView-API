from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import pandas as pd

SOURCE = Path("research/fluid_round10/run_futures_expanded_hodge.py")
spec = importlib.util.spec_from_file_location("expanded_hodge", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load expanded Hodge source")
expanded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expanded)
source = expanded.source
core = expanded.core

OUT = Path("research_outputs/fluid_fixed_cycle_unwind_confirmation_round10")
OUT.mkdir(parents=True, exist_ok=True)
FACTOR = "cross_cycle_unwind_ewm12"
HORIZON = 120
SIGN = 1


def evaluate(panel: pd.DataFrame, train_end: int, start: int, end: int) -> dict:
    target = f"y_{HORIZON}"
    work = panel.copy(); work["__factor"] = SIGN * work[FACTOR]
    train = core.issue_subset(work.loc[work["time"].dt.year <= train_end].copy(), HORIZON, "futures")
    test = core.issue_subset(work.loc[(work["time"].dt.year >= start) & (work["time"].dt.year <= end)].copy(), HORIZON, "futures")
    cs, values = core.cross_sectional_ic(test, "__factor", target, "time", 8)
    boot = core.block_bootstrap(values, 12)
    ridge = core.ridge_increment(train, test, source.BASE, "__factor", target, 250000)
    test["__residual"] = core.residual_values(train, test, source.BASE, "__factor")
    residual_cs, _ = core.cross_sectional_ic(test, "__residual", target, "time", 8)
    ts = core.time_series_ic(test, "__factor", target)
    corr = core.maximum_baseline_correlation(test, "__factor", source.BASE)
    account = source.portfolio(test, "__factor", target, 2.0, 365.25 * 24.0 / HORIZON)
    return {
        "period": f"{start}-{end}", "factor": FACTOR, "horizon": HORIZON, "sign": SIGN,
        "cs_ic": cs["mean"], "positive_share": cs["positive_share"], "n_times": cs["n_times"],
        "bootstrap_low": boot["low"], "bootstrap_high": boot["high"], "p_nonpositive": boot["p_nonpositive"],
        "ridge_gain": ridge["gain"], "residual_cs_ic": residual_cs["mean"],
        "ts_ic_diagnostic": ts["mean"], "ts_positive_share": ts["positive_share"],
        "max_baseline_corr": corr["max_abs"], "max_corr_feature": corr["feature"],
        "net_sharpe_2bp": account["sharpe"], "annual_log_return_2bp": account["annual_log_return"],
        "avg_turnover": account["avg_turnover"], "years": json.dumps(account["years"], sort_keys=True),
    }


try:
    core.FUTURES_SYMBOLS = expanded.EXPANDED_SYMBOLS
    raw, metadata = core.load_futures()
    parts = []
    for symbol, frame in raw.groupby("symbol", sort=False):
        z = core.compute_futures_symbol(frame); z["symbol"] = symbol; parts.append(z)
    panel = source.add_graph_states(pd.concat(parts, ignore_index=True))
    if FACTOR not in panel.columns:
        raise RuntimeError(f"missing frozen factor {FACTOR}")

    prior_replication = evaluate(panel, 2022, 2023, 2024)
    confirmation = evaluate(panel, 2024, 2025, 2025)
    shadow = evaluate(panel, 2025, 2026, 2026)
    final_pass = bool(
        confirmation["cs_ic"] > 0.010
        and confirmation["residual_cs_ic"] > 0.005
        and confirmation["ridge_gain"] > 0.0
        and confirmation["bootstrap_low"] > -0.005
        and confirmation["net_sharpe_2bp"] > 0.25
        and confirmation["max_baseline_corr"] < 0.80
        and shadow["cs_ic"] > 0.0
        and shadow["residual_cs_ic"] > 0.0
        and shadow["ridge_gain"] > -0.0002
        and shadow["net_sharpe_2bp"] > 0.0
    )
    results = pd.DataFrame([prior_replication, confirmation, shadow])
    results.to_csv(OUT / "PERIOD_RESULTS.csv", index=False)
    summary = {
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "data": metadata,
        "requested_symbols": expanded.EXPANDED_SYMBOLS,
        "factor": FACTOR,
        "formula": "EWM12{Z_prior720[-sign(perpetual_momentum_24h) * (graph_cycle_share_t - graph_cycle_share_t-24h)]}",
        "horizon_hours": HORIZON,
        "sign": SIGN,
        "frozen_from": "8-pair 2023-2024 graph-Hodge validation; no formula or horizon search in this run",
        "prior_replication": prior_replication,
        "confirmation_2025": confirmation,
        "shadow_2026": shadow,
        "final_pass": final_pass,
        "governance": {
            "2025_is_first_confirmation_after_formula_freeze": True,
            "2026_is_shadow": True,
            "strong_baseline_contains_graph_parent_states": True,
            "cost_bp_one_way": 2.0,
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable([summary] if final_pass else []), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "REPORT.md").write_text("\n".join([
        "# Frozen Cross-Cycle-Unwind Confirmation",
        "",
        "The formula, 120-hour horizon and positive cross-sectional direction were frozen before this expanded-universe run.",
        "",
        results.to_markdown(index=False),
        "",
        f"Final pass: **{final_pass}**",
    ]), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"]); raise
