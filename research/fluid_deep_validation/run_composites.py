from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run as research
import run_fixed as clock

SEED = 20261118
EPS = 1.0e-12
HORIZONS = research.HORIZONS

COMPOSITES = [
    "compression_absorption_switch",
    "compression_absorption_consensus",
    "boundary_absorption_release",
    "boundary_shock_release",
    "dissipation_circulation_consensus",
    "flux_dissipation_transfer",
    "lagrangian_shock_release",
    "multiphysics_consensus",
]


def aligned_geometric_mean(left: pd.Series, right: pd.Series) -> pd.Series:
    aligned = np.sign(left) == np.sign(right)
    value = np.sign(left + right) * np.sqrt((left * right).abs())
    return value.where(aligned, 0.0)


def add_composites(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("time").copy()
    shock = g["shock_exhaustion"]
    continuity = g["continuity_absorption"]
    boundary = g["bl_event_age_reversal"]
    reynolds = g["reynolds_dissipation_release"]
    circulation = g["circulation_release"]
    flux = g["sgs_flux_gradient"]
    lagrangian = g["lagrangian_stretch_release"]

    raw: dict[str, pd.Series] = {}
    raw["compression_absorption_switch"] = pd.Series(
        np.where(
            continuity.abs() >= shock.abs(),
            continuity,
            shock,
        ),
        index=g.index,
    )
    raw["compression_absorption_consensus"] = aligned_geometric_mean(
        shock, continuity
    )
    raw["boundary_absorption_release"] = boundary * np.tanh(
        continuity.clip(lower=0.0) / 2.0
    )
    raw["boundary_shock_release"] = boundary * np.tanh(
        shock.clip(lower=0.0) / 2.0
    )
    raw["dissipation_circulation_consensus"] = aligned_geometric_mean(
        reynolds, circulation
    )
    raw["flux_dissipation_transfer"] = flux * reynolds
    raw["lagrangian_shock_release"] = lagrangian * np.tanh(
        shock.clip(lower=0.0) / 2.0
    )
    state = pd.concat(
        [shock, continuity, boundary, reynolds, circulation], axis=1
    )
    agreement = state.apply(np.sign).mean(axis=1).abs()
    robust_location = state.median(axis=1)
    raw["multiphysics_consensus"] = robust_location * agreement

    for name, values in raw.items():
        g[name] = research.causal_z(
            pd.Series(values, index=g.index).replace([np.inf, -np.inf], np.nan),
            720,
            240,
        ).astype("float32")
    return g


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def main() -> None:
    output = Path("research_outputs/fluid_closed_loop_composites")
    output.mkdir(parents=True, exist_ok=True)
    raw, data_meta = research.load_data()
    parts = []
    for symbol, group in raw.groupby("symbol", sort=True):
        print(f"feature_engineering={symbol} rows={len(group)}", flush=True)
        parts.append(add_composites(research.compute_symbol(group)))
    panel = pd.concat(parts, ignore_index=True)
    coverage = panel.groupby("symbol")["time"].agg(["min", "max", "count"])
    accepted = coverage.loc[
        (coverage["count"] >= 45_000)
        & (coverage["min"] <= pd.Timestamp("2021-01-02", tz="UTC"))
        & (coverage["max"] >= pd.Timestamp("2026-09-30", tz="UTC"))
    ].index.tolist()
    panel = panel.loc[panel["symbol"].isin(accepted)].copy()

    ledger = []
    best = None
    for factor in COMPOSITES:
        for horizon in HORIZONS:
            frame = clock.evaluation_rows(
                panel, factor, horizon, "2021-01-01", "2023-01-01"
            )
            if len(frame) < 500:
                continue
            ic = research.mean_symbol_spearman(frame)
            score = abs(ic) * math.sqrt(min(len(frame), 50_000) / 10_000.0)
            row = {
                "factor": factor,
                "horizon_hours": horizon,
                "development_ic": ic,
                "development_rows": int(len(frame)),
                "selection_score": score,
            }
            ledger.append(row)
            if best is None or score > best["selection_score"]:
                best = row
    if best is None:
        raise RuntimeError("No composite candidate had sufficient development data")
    config = {
        "factor": best["factor"],
        "horizon_hours": int(best["horizon_hours"]),
        "sign": 1.0 if best["development_ic"] >= 0.0 else -1.0,
        "development_ic_after_sign": abs(float(best["development_ic"])),
        "development_rows": int(best["development_rows"]),
    }
    print(f"selected={config}", flush=True)

    factor = config["factor"]
    horizon = config["horizon_hours"]
    train = clock.evaluation_rows(
        panel, factor, horizon, "2021-01-01", "2023-01-01"
    )
    validation = clock.evaluation_rows(
        panel, factor, horizon, "2023-01-01", "2025-01-01"
    )
    train["factor"] *= config["sign"]
    validation["factor"] *= config["sign"]
    validation_ic = research.mean_symbol_spearman(validation)
    validation_rank = research.block_inference(
        research.contribution_series(validation, "factor", "y"), horizon
    )
    validation_ridge = research.ridge_increment(train, validation, horizon)
    validation_cost2 = research.strategy_metrics(validation, horizon, 2.0)
    validation_cost5 = research.strategy_metrics(validation, horizon, 5.0)
    validation_correlation = research.max_baseline_correlation(validation)
    passed = bool(
        validation_ic > 0.0
        and validation_rank["p_one_sided"] < 0.15
        and validation_ridge["relative_improvement"] > 0.0
        and validation_ridge["loss_difference_inference"]["mean"] > 0.0
        and validation_cost2["net_annualized_log_return"] > 0.0
    )
    validation_result = {
        "rows": int(len(validation)),
        "mean_symbol_spearman": validation_ic,
        "rank_contribution_inference": validation_rank,
        "ridge": validation_ridge,
        "strategy_2bps": validation_cost2,
        "strategy_5bps": validation_cost5,
        "baseline_correlation": validation_correlation,
        "passed_to_holdout": passed,
    }
    print(
        f"validation ic={validation_ic:.6f} p={validation_rank['p_one_sided']:.5f} "
        f"ridge={validation_ridge['relative_improvement']:.7f} "
        f"net2={validation_cost2['net_annualized_log_return']:.6f} pass={passed}",
        flush=True,
    )

    holdout_result: dict[str, Any] = {}
    decision = {
        "status": "reject_current_configuration",
        "independent_alpha": False,
        "default_enabled": False,
    }
    if passed:
        train_full = clock.evaluation_rows(
            panel, factor, horizon, "2021-01-01", "2025-01-01"
        )
        holdout = clock.evaluation_rows(panel, factor, horizon, "2025-01-01", None)
        train_full["factor"] *= config["sign"]
        holdout["factor"] *= config["sign"]
        holdout_ic = research.mean_symbol_spearman(holdout)
        holdout_rank = research.block_inference(
            research.contribution_series(holdout, "factor", "y"), horizon
        )
        holdout_ridge = research.ridge_increment(train_full, holdout, horizon)
        holdout_hgb = research.hgb_increment(train_full, holdout, horizon)
        holdout_cost2 = research.strategy_metrics(holdout, horizon, 2.0)
        holdout_cost5 = research.strategy_metrics(holdout, horizon, 5.0)
        holdout_corr = research.max_baseline_correlation(holdout)
        years_ok = holdout_cost5["positive_years"] >= max(
            1, holdout_cost5["year_count"] - 1
        )
        independent = bool(
            holdout_rank["p_one_sided"] < 0.05
            and holdout_ic > 0.0
            and holdout_ridge["loss_difference_inference"]["p_one_sided"] < 0.10
            and holdout_ridge["relative_improvement"] > 0.0
            and holdout_ridge["residual_spearman"] > 0.0
            and holdout_hgb["relative_improvement"] > 0.0
            and holdout_hgb["loss_difference_inference"]["p_one_sided"] < 0.20
            and holdout_cost5["net_annualized_log_return"] > 0.0
            and holdout_cost5["net_sharpe"] > 0.0
            and holdout_corr["max_abs"] < 0.75
            and years_ok
        )
        component = bool(
            not independent
            and holdout_ic > 0.0
            and holdout_ridge["relative_improvement"] > 0.0
            and holdout_cost2["net_annualized_log_return"] > 0.0
        )
        decision = {
            "status": (
                "admit_independent_alpha"
                if independent
                else "admit_default_off_component"
                if component
                else "reject_current_configuration"
            ),
            "independent_alpha": independent,
            "default_enabled": False,
        }
        holdout_result = {
            "rows": int(len(holdout)),
            "mean_symbol_spearman": holdout_ic,
            "rank_contribution_inference": holdout_rank,
            "ridge": holdout_ridge,
            "hist_gradient_boosting": holdout_hgb,
            "strategy_2bps": holdout_cost2,
            "strategy_5bps": holdout_cost5,
            "baseline_correlation": holdout_corr,
        }
        print(
            f"holdout ic={holdout_ic:.6f} p={holdout_rank['p_one_sided']:.5f} "
            f"ridge={holdout_ridge['relative_improvement']:.7f} "
            f"hgb={holdout_hgb['relative_improvement']:.7f} "
            f"net5={holdout_cost5['net_annualized_log_return']:.6f} "
            f"decision={decision['status']}",
            flush=True,
        )

    result = clean({
        "research_id": "fluid_closed_loop_composites_20261118",
        "data": {**data_meta, "accepted_symbols": accepted, "rows_after_filter": int(len(panel))},
        "candidate_count": len(COMPOSITES),
        "candidates": COMPOSITES,
        "selection": config,
        "validation": validation_result,
        "holdout": holdout_result,
        "decision": decision,
        "holdout_accessed": bool(passed),
        "limitations": [
            "Composite structures were frozen before this validation run.",
            "2025-2026 holdout is accessed only if the 2023-2024 gate passes.",
            "Public Binance USDT perpetual hourly bars; funding and nonlinear impact excluded.",
        ],
    })
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    pd.DataFrame(ledger).to_csv(output / "selection_ledger.csv", index=False)
    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    print("RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    main()
