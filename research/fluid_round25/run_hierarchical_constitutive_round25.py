from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUND24 = HERE.parent / "fluid_round24"
ROUND22 = HERE.parent / "fluid_round22"
ROUND23 = HERE.parent / "fluid_round23"
for path in [ROUND24, ROUND22, ROUND23]:
    import sys
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_regime_constitutive_round24 as r24  # noqa: E402
import run_timeseries_operator_round22 as r22  # noqa: E402
import run_dimensionless_operator_round23 as r23  # noqa: E402

base = r22.base
EPS = r22.EPS
SEED = 20261119
OUT = Path("research_outputs/fluid_hierarchical_constitutive_round25")
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_SYMBOLS = list(r22.DISCOVERY_SYMBOLS)
REPLICATION_SYMBOLS = list(r22.REPLICATION_SYMBOLS)

PERSIST_SPAN = 72
GLOBAL_ALPHA = 50.0
ASSET_ALPHA = 500.0
ROUNDTRIP_COST = 10.0e-4
DEADZONE = 0.20

PERSISTENCE_COLUMNS = [f"{column}_persist" for column in r24.REGIME_COLUMNS]

SPECS = [
    {
        "family": "hierarchical_impulse_constitutive",
        "parent": "impulse_balance_direction_ewm24",
        "horizon": 120,
        "factor": "hierarchical_impulse_constitutive_120h",
        "mechanism": "asset-specific source-pressure impulse response partially pooled toward a global constitutive law; current and persistent propagation regimes jointly control direction and amplitude",
    },
    {
        "family": "hierarchical_slowmode_constitutive",
        "parent": "slow_mode_damped_nonnormal_ewm72",
        "horizon": 72,
        "factor": "hierarchical_slowmode_constitutive_72h",
        "mechanism": "asset-specific slow-mode-damped transient response partially pooled toward a global law with regime-persistence-conditioned direction and magnitude",
    },
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def logistic(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def add_persistence(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("time").copy()
    for column in r24.REGIME_COLUMNS:
        values = pd.to_numeric(g[column], errors="coerce")
        g[f"{column}_persist"] = values.shift(1).ewm(
            span=PERSIST_SPAN,
            adjust=False,
            min_periods=PERSIST_SPAN,
        ).mean()
    return g


def build_panel(symbols: list[str], out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, audit = r24.build_panel(symbols, out)
    parts = [add_persistence(group) for _, group in panel.groupby("symbol", sort=False)]
    result = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    return result, audit


def basis_matrix(frame: pd.DataFrame, parent: str) -> tuple[np.ndarray, list[str]]:
    p = pd.to_numeric(frame[parent], errors="coerce").to_numpy(dtype=float)
    current = [
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for column in r24.REGIME_COLUMNS
    ]
    persistent = [
        pd.to_numeric(frame[f"{column}_persist"], errors="coerce").to_numpy(dtype=float)
        for column in r24.REGIME_COLUMNS
    ]
    columns = [
        "parent",
        "parent_x_lead",
        "parent_x_source",
        "parent_x_impedance",
        "parent_x_dominance",
        "parent_x_lead_persist",
        "parent_x_source_persist",
        "parent_x_impedance_persist",
        "parent_x_dominance_persist",
    ]
    matrix = np.column_stack([p] + [p * values for values in current] + [p * values for values in persistent])
    return matrix, columns


def build_partial_pool_design(
    standardized_basis: np.ndarray,
    symbols: np.ndarray,
    trained_symbols: list[str],
) -> np.ndarray:
    n_rows, n_terms = standardized_basis.shape
    design = np.zeros((n_rows, n_terms * (1 + len(trained_symbols))), dtype=float)
    design[:, :n_terms] = standardized_basis
    index = {symbol: position for position, symbol in enumerate(trained_symbols)}
    for symbol, position in index.items():
        mask = symbols == symbol
        if not np.any(mask):
            continue
        start = n_terms * (1 + position)
        design[mask, start : start + n_terms] = standardized_basis[mask]
    return design


def fit_model(
    panel: pd.DataFrame,
    spec: dict[str, Any],
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    parent = str(spec["parent"])
    horizon = int(spec["horizon"])
    frame = r22.issue_frame(panel, parent, horizon, start_year, end_year)
    target = f"y_{horizon}"

    basis, columns = basis_matrix(frame, parent)
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    symbols = frame["symbol"].astype(str).to_numpy()

    asset_vol = frame.assign(__target=y).groupby("symbol", sort=False)["__target"].std(ddof=0).replace(0.0, np.nan)
    vol = np.array([asset_vol.get(symbol, np.nan) for symbol in symbols], dtype=float)
    net_edge = np.sign(y) * np.maximum(np.abs(y) - ROUNDTRIP_COST, 0.0)
    y_scaled = net_edge / vol
    direction_target = np.sign(y_scaled)
    amplitude_target = np.log1p(np.abs(y_scaled))

    valid = (
        np.isfinite(basis).all(axis=1)
        & np.isfinite(direction_target)
        & np.isfinite(amplitude_target)
        & np.isfinite(vol)
    )
    if int(valid.sum()) < 5000:
        raise RuntimeError(f"insufficient hierarchical observations for {parent}: {int(valid.sum())}")

    x = basis[valid]
    train_symbols = symbols[valid]
    trained_symbols = sorted(set(train_symbols.tolist()))
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    x_std = (x - mean) / std
    design = build_partial_pool_design(x_std, train_symbols, trained_symbols)

    penalty = np.full(design.shape[1], ASSET_ALPHA, dtype=float)
    penalty[: x_std.shape[1]] = GLOBAL_ALPHA
    xtx = design.T @ design + np.diag(penalty)
    targets = np.column_stack([direction_target[valid], amplitude_target[valid]])
    target_mean = np.nanmean(targets, axis=0)
    beta = np.linalg.solve(xtx, design.T @ (targets - target_mean))

    fitted = target_mean + design @ beta
    direction_scale = float(np.nanstd(fitted[:, 0]))
    amplitude_center = float(np.nanmedian(fitted[:, 1]))
    amplitude_scale = float(np.nanstd(fitted[:, 1]))
    direction_scale = direction_scale if direction_scale > 1.0e-8 else 1.0
    amplitude_scale = amplitude_scale if amplitude_scale > 1.0e-8 else 1.0

    return {
        "family": str(spec["family"]),
        "parent": parent,
        "factor": str(spec["factor"]),
        "horizon": horizon,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "columns": columns,
        "trained_symbols": trained_symbols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target_mean": target_mean.tolist(),
        "beta": beta.tolist(),
        "direction_scale": direction_scale,
        "amplitude_center": amplitude_center,
        "amplitude_scale": amplitude_scale,
        "global_alpha": GLOBAL_ALPHA,
        "asset_alpha": ASSET_ALPHA,
        "roundtrip_cost": ROUNDTRIP_COST,
        "deadzone": DEADZONE,
        "n_train": int(valid.sum()),
    }


def apply_model(panel: pd.DataFrame, model: dict[str, Any], output: str) -> None:
    basis, columns = basis_matrix(panel, str(model["parent"]))
    if columns != list(model["columns"]):
        raise RuntimeError("hierarchical basis contract changed")
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    standardized = (basis - mean) / std
    symbols = panel["symbol"].astype(str).to_numpy()
    design = build_partial_pool_design(standardized, symbols, list(model["trained_symbols"]))
    beta = np.asarray(model["beta"], dtype=float)
    target_mean = np.asarray(model["target_mean"], dtype=float)
    prediction = target_mean + design @ beta

    direction = np.tanh(prediction[:, 0] / float(model["direction_scale"]))
    amplitude = logistic((prediction[:, 1] - float(model["amplitude_center"])) / float(model["amplitude_scale"]))
    raw = direction * amplitude
    magnitude = ((np.abs(raw) - DEADZONE) / (1.0 - DEADZONE)).clip(0.0, 1.0)
    panel[output] = np.sign(raw) * magnitude
    panel[f"{output}__direction"] = direction
    panel[f"{output}__amplitude"] = amplitude


def strong_parents(panel: pd.DataFrame, parent: str) -> list[str]:
    candidates = (
        list(r22.PARENT_CANDIDATES)
        + [parent]
        + list(r24.REGIME_COLUMNS)
        + PERSISTENCE_COLUMNS
    )
    return list(dict.fromkeys(column for column in candidates if column in panel.columns))


def evaluate(
    panel: pd.DataFrame,
    specs: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    train_end: int,
    stage: str,
) -> pd.DataFrame:
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
        account_years = json.loads(str(row["portfolio_years"]))
        year_ok = all(np.isfinite(value) and value > 0.0 for value in years.values())
        account_year_ok = all(np.isfinite(value) and value > 0.0 for value in account_years.values())
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
            and account_year_ok
        ))
    result["gate"] = gates
    return result


def development_crossfit(panel: pd.DataFrame, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_year, test_year in [(2021, 2022), (2022, 2021)]:
        model = fit_model(panel, spec, train_year, train_year)
        temporary = f"__crossfit_{spec['factor']}_{test_year}"
        apply_model(panel, model, temporary)
        frame = r22.issue_frame(panel, temporary, int(spec["horizon"]), test_year, test_year)
        direct = r22.ts_summary(frame, temporary, int(spec["horizon"]))
        account = r23.execution_account(frame, temporary, int(spec["horizon"]), 5.0)
        panel.drop(columns=[temporary, f"{temporary}__direction", f"{temporary}__amplitude"], inplace=True)
        rows.append({
            "factor": spec["factor"],
            "train_year": train_year,
            "test_year": test_year,
            "ts_ic": float(direct["ts_ic"]),
            "positive_asset_share": float(direct["positive_asset_share"]),
            "annual_log_return_5bp": float(account["annual_log_return"]),
            "sharpe_5bp": float(account["sharpe"]),
        })
    return rows


def main() -> None:
    np.random.seed(SEED)
    discovery, discovery_audit = build_panel(DISCOVERY_SYMBOLS, OUT / "discovery_cache")
    models: dict[str, dict[str, Any]] = {}
    crossfit_rows: list[dict[str, Any]] = []

    for spec in SPECS:
        model = fit_model(discovery, spec, 2021, 2022)
        models[str(spec["factor"])] = model
        apply_model(discovery, model, str(spec["factor"]))
        crossfit_rows.extend(development_crossfit(discovery, spec))

    discovery_validation = evaluate(discovery, SPECS, 2023, 2024, 2022, "discovery_validation")
    discovery_holdout = evaluate(discovery, SPECS, 2025, 2026, 2024, "discovery_holdout")

    replication, replication_audit = build_panel(REPLICATION_SYMBOLS, OUT / "replication_cache")
    for spec in SPECS:
        apply_model(replication, models[str(spec["factor"])], str(spec["factor"]))
    replication_validation = evaluate(replication, SPECS, 2023, 2024, 2022, "replication_validation")
    replication_holdout = evaluate(replication, SPECS, 2025, 2026, 2024, "replication_holdout")

    crossfit = pd.DataFrame(crossfit_rows)
    crossfit.to_csv(OUT / "DEVELOPMENT_CROSSFIT.csv", index=False)
    discovery_validation.to_csv(OUT / "DISCOVERY_VALIDATION_RESULTS.csv", index=False)
    discovery_holdout.to_csv(OUT / "DISCOVERY_HOLDOUT_RESULTS.csv", index=False)
    replication_validation.to_csv(OUT / "REPLICATION_VALIDATION_RESULTS.csv", index=False)
    replication_holdout.to_csv(OUT / "REPLICATION_HOLDOUT_RESULTS.csv", index=False)
    write_json(OUT / "FROZEN_MODELS.json", models)

    confirmed: list[dict[str, Any]] = []
    for spec in SPECS:
        factor = str(spec["factor"])
        crossfit_factor = crossfit.loc[crossfit["factor"] == factor]
        crossfit_ok = bool(
            len(crossfit_factor) == 2
            and (crossfit_factor["ts_ic"] > 0.0).all()
            and (crossfit_factor["annual_log_return_5bp"] > 0.0).all()
        )
        records = []
        for table in [discovery_validation, discovery_holdout, replication_validation, replication_holdout]:
            match = table.loc[table["factor"] == factor]
            if match.empty:
                records = []
                break
            records.append(match.iloc[0])
        if crossfit_ok and records and all(bool(record["gate"]) for record in records):
            confirmed.append({
                "factor": factor,
                "parent": spec["parent"],
                "horizon": spec["horizon"],
                "development_crossfit_passed": True,
                "all_four_stage_gates": True,
                "minimum_ts_ic": float(min(record["ts_ic"] for record in records)),
                "minimum_residual_ts_ic": float(min(record["residual_ts_ic"] for record in records)),
                "minimum_ridge_mse_gain": float(min(record["ridge_mse_gain"] for record in records)),
                "minimum_annual_log_return_5bp": float(min(record["annual_log_return_5bp"] for record in records)),
            })

    summary = {
        "spec_count": len(SPECS),
        "development_crossfit_rows": len(crossfit),
        "discovery_validation_passed": int(discovery_validation["gate"].sum()),
        "discovery_holdout_passed": int(discovery_holdout["gate"].sum()),
        "replication_validation_passed": int(replication_validation["gate"].sum()),
        "replication_holdout_passed": int(replication_holdout["gate"].sum()),
        "confirmed": confirmed,
        "decision": "eligible_for_trade_only_if_confirmed_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    write_json(OUT / "CONFIRMED.json", confirmed)
    write_json(OUT / "DATA_AUDIT.json", {
        "seed": SEED,
        "discovery_symbols": DISCOVERY_SYMBOLS,
        "replication_symbols": REPLICATION_SYMBOLS,
        "discovery_data": discovery_audit,
        "replication_data": replication_audit,
        "partial_pooling": {
            "global_alpha": GLOBAL_ALPHA,
            "asset_alpha": ASSET_ALPHA,
            "regime_persistence_span_hours": PERSIST_SPAN,
            "direction_target": "sign(max(abs(return)-10bp,0))",
            "amplitude_target": "log1p(abs(net_return)/asset_volatility)",
            "unseen_replication_assets": "global_coefficients_only",
        },
        "execution": {"deadzone": DEADZONE, "cost_bp_one_way": 5.0},
        "confirmation": "development crossfit plus discovery validation/holdout and disjoint replication validation/holdout must all pass",
        "research_boundary": "historical; no parameter selected on validation, holdout or replication stages",
    })
    write_json(OUT / "CANDIDATE_CONTRACTS.json", {
        spec["factor"]: {
            "family": spec["family"],
            "parent": spec["parent"],
            "horizon_hours": spec["horizon"],
            "mechanism": spec["mechanism"],
        }
        for spec in SPECS
    })

    report = [
        "# Hierarchical regime-constitutive research round25",
        "",
        "Round25 tests whether asset-specific constitutive coefficients, partially pooled toward a global law, can stabilize the regime-dependent non-normal transport states identified in Rounds22-24. Direction and magnitude are learned separately on a return target net of a frozen 5bp one-way execution hurdle.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Development crossfit",
        crossfit.to_markdown(index=False),
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
