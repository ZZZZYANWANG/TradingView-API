from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUND22 = HERE.parent / "fluid_round22"
ROUND23 = HERE.parent / "fluid_round23"
for path in [ROUND22, ROUND23]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_timeseries_operator_round22 as r22  # noqa: E402
import run_dimensionless_operator_round23 as r23  # noqa: E402

base = r22.base
EPS = r22.EPS
SEED = 20261119
OUT = Path("research_outputs/fluid_regime_constitutive_round24")
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_SYMBOLS = list(r22.DISCOVERY_SYMBOLS)
REPLICATION_SYMBOLS = list(r22.REPLICATION_SYMBOLS)
LEAD_WINDOW = 720
LEAD_MIN_PERIODS = 240
RIDGE_ALPHA = 50.0
NO_TRADE_THRESHOLD = 0.35
POSITION_SCALE = 1.25

REGIME_COLUMNS = [
    "regime_lead_direction",
    "regime_source_loading",
    "regime_impedance_match",
    "regime_perp_dominance",
]

SPECS = [
    {
        "family": "regime_constitutive_impulse",
        "parent": "impulse_balance_direction_ewm24",
        "horizon": 120,
        "factor": "regime_constitutive_impulse_120h",
        "mechanism": "source-pressure impulse imbalance read through a development-frozen constitutive coefficient conditioned on lead direction, leverage loading, impedance matching and venue dominance",
    },
    {
        "family": "regime_constitutive_slowmode",
        "parent": "slow_mode_damped_nonnormal_ewm72",
        "horizon": 72,
        "factor": "regime_constitutive_slowmode_72h",
        "mechanism": "slow-mode-damped non-normal transient read through the same development-frozen propagation-regime basis",
    },
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def causal_regime_states(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("time").copy()
    spot = pd.to_numeric(g["spot_u"], errors="coerce")
    perp = pd.to_numeric(g["perp_u"], errors="coerce")

    # Each rolling pair ends at t-1. No current or future observation enters the regime.
    spot_to_perp = spot.shift(2).rolling(LEAD_WINDOW, min_periods=LEAD_MIN_PERIODS).corr(perp.shift(1))
    perp_to_spot = perp.shift(2).rolling(LEAD_WINDOW, min_periods=LEAD_MIN_PERIODS).corr(spot.shift(1))
    lead_raw = spot_to_perp - perp_to_spot
    g["regime_lead_direction"] = np.tanh(base.causal_z(lead_raw))

    source_raw = pd.to_numeric(g["oi_source_z"], errors="coerce") + 0.5 * pd.to_numeric(g["funding_change_z"], errors="coerce")
    g["regime_source_loading"] = np.tanh(source_raw / 1.5)

    pressure_gap = pd.to_numeric(g["pressure_gap"], errors="coerce").abs()
    g["regime_impedance_match"] = 1.0 - 2.0 * np.tanh(pressure_gap / 2.0)
    g["regime_perp_dominance"] = np.tanh(pd.to_numeric(g["quote_share_z"], errors="coerce"))
    return g


def build_panel(symbols: list[str], out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, audit, _ = r22.build_panel(symbols, out)
    parts = [causal_regime_states(group) for _, group in panel.groupby("symbol", sort=False)]
    result = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    return result, audit


def design_matrix(frame: pd.DataFrame, parent: str) -> tuple[np.ndarray, list[str]]:
    p = pd.to_numeric(frame[parent], errors="coerce").to_numpy(dtype=float)
    lead = pd.to_numeric(frame["regime_lead_direction"], errors="coerce").to_numpy(dtype=float)
    source = pd.to_numeric(frame["regime_source_loading"], errors="coerce").to_numpy(dtype=float)
    impedance = pd.to_numeric(frame["regime_impedance_match"], errors="coerce").to_numpy(dtype=float)
    dominance = pd.to_numeric(frame["regime_perp_dominance"], errors="coerce").to_numpy(dtype=float)

    columns = [
        "parent",
        "parent_x_lead",
        "parent_x_source",
        "parent_x_impedance",
        "parent_x_dominance",
        "parent_x_lead_x_source",
        "parent_x_lead_x_impedance",
        "parent_x_source_x_impedance",
        "parent_x_lead_x_dominance",
    ]
    matrix = np.column_stack([
        p,
        p * lead,
        p * source,
        p * impedance,
        p * dominance,
        p * lead * source,
        p * lead * impedance,
        p * source * impedance,
        p * lead * dominance,
    ])
    return matrix, columns


def fit_model(panel: pd.DataFrame, parent: str, horizon: int, start_year: int, end_year: int) -> dict[str, Any]:
    frame = r22.issue_frame(panel, parent, horizon, start_year, end_year)
    target = f"y_{horizon}"
    x, columns = design_matrix(frame, parent)
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    symbols = frame["symbol"].astype(str).to_numpy()

    asset_scale = frame.assign(__target=y).groupby("symbol", sort=False)["__target"].std(ddof=0).replace(0.0, np.nan)
    scale = np.array([asset_scale.get(symbol, np.nan) for symbol in symbols], dtype=float)
    y_scaled = y / scale

    valid = np.isfinite(y_scaled) & np.isfinite(x).all(axis=1)
    if int(valid.sum()) < 5000:
        raise RuntimeError(f"insufficient development observations for {parent}: {int(valid.sum())}")

    x_train = x[valid]
    y_train = y_scaled[valid]
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    x_std = (x_train - mean) / std
    y_mean = float(np.nanmean(y_train))
    beta = np.linalg.solve(x_std.T @ x_std + RIDGE_ALPHA * np.eye(x_std.shape[1]), x_std.T @ (y_train - y_mean))
    raw_prediction = y_mean + x_std @ beta
    prediction_scale = float(np.nanstd(raw_prediction))
    if not np.isfinite(prediction_scale) or prediction_scale <= 1.0e-8:
        prediction_scale = 1.0

    return {
        "parent": parent,
        "horizon": int(horizon),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "columns": columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "beta": beta.tolist(),
        "y_mean": y_mean,
        "prediction_scale": prediction_scale,
        "n_train": int(valid.sum()),
    }


def apply_model(panel: pd.DataFrame, model: dict[str, Any], output: str) -> None:
    x, columns = design_matrix(panel, str(model["parent"]))
    if columns != list(model["columns"]):
        raise RuntimeError("design matrix contract changed")
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    beta = np.asarray(model["beta"], dtype=float)
    raw = float(model["y_mean"]) + ((x - mean) / std) @ beta
    bounded = np.tanh(raw / float(model["prediction_scale"]) / POSITION_SCALE)
    magnitude = ((np.abs(bounded) - NO_TRADE_THRESHOLD) / (1.0 - NO_TRADE_THRESHOLD)).clip(0.0, 1.0)
    panel[output] = np.sign(bounded) * magnitude


def strong_parents(panel: pd.DataFrame, parent: str) -> list[str]:
    candidates = list(r22.PARENT_CANDIDATES) + [parent] + REGIME_COLUMNS
    return list(dict.fromkeys(column for column in candidates if column in panel.columns))


def evaluate(panel: pd.DataFrame, specs: list[dict[str, Any]], start_year: int, end_year: int, train_end: int, stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        factor = str(spec["factor"])
        parent = str(spec["parent"])
        horizon = int(spec["horizon"])
        frame = r22.issue_frame(panel, factor, horizon, start_year, end_year)
        direct = r22.ts_summary(frame, factor, horizon)
        boot = r22.asset_bootstrap(direct["asset_tsics"])
        yearly = {
            str(year): r22.ts_summary(frame.loc[frame["time"].dt.year == year], factor, horizon)["ts_ic"]
            for year in range(start_year, end_year + 1)
        }

        parents = strong_parents(panel, parent)
        residual_column = f"__residual_{factor}_{stage}"
        panel[residual_column] = r22.residual_factor(panel, factor, parents, train_end, start_year, end_year)
        residual_frame = r22.issue_frame(panel, residual_column, horizon, start_year, end_year)
        residual = r22.ts_summary(residual_frame, residual_column, horizon)
        increment = r22.model_increment(panel, factor, horizon, parents, train_end, start_year, end_year)
        corr = r22.max_parent_corr(frame, factor, parents)
        account = r23.execution_account(frame, factor, horizon, 5.0)
        panel.drop(columns=[residual_column], inplace=True)

        rows.append({
            "stage": stage,
            "family": str(spec["family"]),
            "parent": parent,
            "factor": factor,
            "horizon": horizon,
            "ts_ic": float(direct["ts_ic"]),
            "positive_asset_share": float(direct["positive_asset_share"]),
            "asset_count": int(direct["asset_count"]),
            "bootstrap_ci_low": float(boot["ci_low"]),
            "bootstrap_ci_high": float(boot["ci_high"]),
            "bootstrap_p_nonpositive": float(boot["p_nonpositive"]),
            "yearly_ts_ic": json.dumps(yearly, sort_keys=True),
            "residual_ts_ic": float(residual["ts_ic"]),
            "residual_positive_asset_share": float(residual["positive_asset_share"]),
            "ridge_mse_gain": float(increment),
            "max_parent_corr": float(corr),
            "annual_log_return_5bp": float(account["annual_log_return"]),
            "sharpe_5bp": float(account["sharpe"]),
            "avg_turnover": float(account["avg_turnover"]),
            "portfolio_years": json.dumps(account["years"], sort_keys=True),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_p"] = r22.holm(result["bootstrap_p_nonpositive"].tolist())
    gates: list[bool] = []
    for _, row in result.iterrows():
        years = json.loads(str(row["yearly_ts_ic"]))
        year_ok = all(np.isfinite(value) and value > 0.0 for value in years.values())
        gates.append(bool(
            row["ts_ic"] > 0.020
            and row["positive_asset_share"] >= 0.60
            and row["bootstrap_p_nonpositive"] <= 0.10
            and row["holm_p"] <= 0.10
            and row["residual_ts_ic"] > 0.010
            and row["ridge_mse_gain"] > 0.0
            and row["max_parent_corr"] < 0.80
            and row["annual_log_return_5bp"] > 0.0
            and year_ok
        ))
    result["gate"] = gates
    return result


def development_crossfit(panel: pd.DataFrame, spec: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for train_year, test_year in [(2021, 2022), (2022, 2021)]:
        model = fit_model(panel, str(spec["parent"]), int(spec["horizon"]), train_year, train_year)
        temporary = f"__crossfit_{spec['factor']}_{test_year}"
        apply_model(panel, model, temporary)
        frame = r22.issue_frame(panel, temporary, int(spec["horizon"]), test_year, test_year)
        summary = r22.ts_summary(frame, temporary, int(spec["horizon"]))
        account = r23.execution_account(frame, temporary, int(spec["horizon"]), 5.0)
        panel.drop(columns=[temporary], inplace=True)
        output.append({
            "factor": spec["factor"],
            "train_year": train_year,
            "test_year": test_year,
            "ts_ic": float(summary["ts_ic"]),
            "positive_asset_share": float(summary["positive_asset_share"]),
            "annual_log_return_5bp": float(account["annual_log_return"]),
            "sharpe_5bp": float(account["sharpe"]),
        })
    return output


def main() -> None:
    np.random.seed(SEED)
    discovery, discovery_audit = build_panel(DISCOVERY_SYMBOLS, OUT / "discovery_cache")
    models: dict[str, dict[str, Any]] = {}
    coefficient_rows: list[dict[str, Any]] = []
    crossfit_rows: list[dict[str, Any]] = []

    for spec in SPECS:
        model = fit_model(discovery, str(spec["parent"]), int(spec["horizon"]), 2021, 2022)
        models[str(spec["factor"])] = model
        apply_model(discovery, model, str(spec["factor"]))
        crossfit_rows.extend(development_crossfit(discovery, spec))
        for column, coefficient in zip(model["columns"], model["beta"]):
            coefficient_rows.append({
                "factor": spec["factor"],
                "parent": spec["parent"],
                "horizon": spec["horizon"],
                "term": column,
                "coefficient": coefficient,
            })

    discovery_validation = evaluate(discovery, SPECS, 2023, 2024, 2022, "discovery_validation")
    discovery_holdout = evaluate(discovery, SPECS, 2025, 2026, 2024, "discovery_holdout")

    replication, replication_audit = build_panel(REPLICATION_SYMBOLS, OUT / "replication_cache")
    for spec in SPECS:
        apply_model(replication, models[str(spec["factor"])], str(spec["factor"]))

    replication_validation = evaluate(replication, SPECS, 2023, 2024, 2022, "replication_validation")
    replication_holdout = evaluate(replication, SPECS, 2025, 2026, 2024, "replication_holdout")

    pd.DataFrame(coefficient_rows).to_csv(OUT / "FROZEN_COEFFICIENTS.csv", index=False)
    pd.DataFrame(crossfit_rows).to_csv(OUT / "DEVELOPMENT_CROSSFIT.csv", index=False)
    discovery_validation.to_csv(OUT / "DISCOVERY_VALIDATION_RESULTS.csv", index=False)
    discovery_holdout.to_csv(OUT / "DISCOVERY_HOLDOUT_RESULTS.csv", index=False)
    replication_validation.to_csv(OUT / "REPLICATION_VALIDATION_RESULTS.csv", index=False)
    replication_holdout.to_csv(OUT / "REPLICATION_HOLDOUT_RESULTS.csv", index=False)

    confirmed: list[dict[str, Any]] = []
    for spec in SPECS:
        factor = str(spec["factor"])
        records = []
        for table in [discovery_validation, discovery_holdout, replication_validation, replication_holdout]:
            matched = table.loc[table["factor"] == factor]
            if matched.empty:
                records = []
                break
            records.append(matched.iloc[0])
        if records and all(bool(record["gate"]) for record in records):
            confirmed.append({
                "factor": factor,
                "parent": spec["parent"],
                "horizon": spec["horizon"],
                "all_four_stage_gates": True,
                "minimum_ts_ic": float(min(record["ts_ic"] for record in records)),
                "minimum_residual_ts_ic": float(min(record["residual_ts_ic"] for record in records)),
                "minimum_ridge_mse_gain": float(min(record["ridge_mse_gain"] for record in records)),
                "minimum_annual_log_return_5bp": float(min(record["annual_log_return_5bp"] for record in records)),
            })

    summary = {
        "frozen_models": len(SPECS),
        "model_terms": 9,
        "development_crossfit_rows": len(crossfit_rows),
        "discovery_validation_passed": int(discovery_validation["gate"].sum()),
        "discovery_holdout_passed": int(discovery_holdout["gate"].sum()),
        "replication_validation_passed": int(replication_validation["gate"].sum()),
        "replication_holdout_passed": int(replication_holdout["gate"].sum()),
        "confirmed": confirmed,
        "decision": "eligible_for_trade_only_if_confirmed_nonempty",
    }

    write_json(OUT / "SUMMARY.json", summary)
    write_json(OUT / "CONFIRMED.json", confirmed)
    write_json(OUT / "FROZEN_MODELS.json", models)
    write_json(OUT / "MODEL_CONTRACTS.json", {
        spec["factor"]: {
            "family": spec["family"],
            "parent": spec["parent"],
            "horizon_hours": spec["horizon"],
            "mechanism": spec["mechanism"],
            "regime_basis": REGIME_COLUMNS,
            "interaction_terms": models[spec["factor"]]["columns"],
            "fit_period": "2021-2022 discovery universe only",
            "ridge_alpha": RIDGE_ALPHA,
            "execution_map": "deadzone(tanh(prediction / development_prediction_std / 1.25), 0.35)",
        }
        for spec in SPECS
    })
    write_json(OUT / "DATA_AUDIT.json", {
        "seed": SEED,
        "discovery_symbols": DISCOVERY_SYMBOLS,
        "replication_symbols": REPLICATION_SYMBOLS,
        "discovery_data": discovery_audit,
        "replication_data": replication_audit,
        "lead_regime": {
            "window_hours": LEAD_WINDOW,
            "minimum_hours": LEAD_MIN_PERIODS,
            "definition": "tanh(causal_z(corr(spot[t-2],perp[t-1])-corr(perp[t-2],spot[t-1])))",
        },
        "source_regime": "tanh((oi_source_z+0.5*funding_change_z)/1.5)",
        "impedance_regime": "1-2*tanh(abs(pressure_gap)/2)",
        "dominance_regime": "tanh(quote_share_z)",
        "research_boundary": "all constitutive coefficients are fit on 2021-2022 discovery data only and applied unchanged to discovery validation/holdout and the disjoint replication universe",
    })

    report = [
        "# Regime-dependent constitutive operator research round24",
        "",
        "Round24 tests whether the sign and strength of previously observed non-normal transport depends on a strictly-prior propagation regime rather than on a fixed readout map.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen coefficients",
        pd.DataFrame(coefficient_rows).to_markdown(index=False),
        "",
        "## Development cross-fit",
        pd.DataFrame(crossfit_rows).to_markdown(index=False),
        "",
        "## Discovery validation",
        discovery_validation.to_markdown(index=False),
        "",
        "## Discovery historical holdout",
        discovery_holdout.to_markdown(index=False),
        "",
        "## Disjoint-universe validation",
        replication_validation.to_markdown(index=False),
        "",
        "## Disjoint-universe historical holdout",
        replication_holdout.to_markdown(index=False),
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        write_json(OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
