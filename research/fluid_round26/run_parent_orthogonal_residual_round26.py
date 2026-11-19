from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUND25 = HERE.parent / "fluid_round25"
if str(ROUND25) not in sys.path:
    sys.path.insert(0, str(ROUND25))

import run_hierarchical_constitutive_round25 as r25  # noqa: E402

r22 = r25.r22
r23 = r25.r23
EPS = r22.EPS
SEED = 20261119
OUT = Path("research_outputs/fluid_parent_orthogonal_round26")
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_SYMBOLS = list(r25.DISCOVERY_SYMBOLS)
REPLICATION_SYMBOLS = list(r25.REPLICATION_SYMBOLS)
PARENT_ALPHA = 10.0
RESIDUAL_GLOBAL_ALPHA = 50.0
RESIDUAL_ASSET_ALPHA = 500.0
DEADZONE = 0.20
POSITION_SCALE = 1.0

SPECS = [
    {
        "family": "parent_orthogonal_impulse_constitutive",
        "parent": "impulse_balance_direction_ewm24",
        "horizon": 120,
        "factor": "parent_orthogonal_impulse_constitutive_120h",
        "mechanism": "hierarchical source-pressure constitutive law trained only on strictly out-of-fold strong-parent residual returns",
    },
    {
        "family": "parent_orthogonal_slowmode_constitutive",
        "parent": "slow_mode_damped_nonnormal_ewm72",
        "horizon": 72,
        "factor": "parent_orthogonal_slowmode_constitutive_72h",
        "mechanism": "hierarchical slow-mode-damped non-normal law trained only on strictly out-of-fold strong-parent residual returns",
    },
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    xs = (x - mean) / std
    y_mean = float(np.nanmean(y))
    beta = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ (y - y_mean))
    fitted = y_mean + xs @ beta
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "y_mean": y_mean,
        "beta": beta.tolist(),
        "target_std": float(np.nanstd(y)),
        "fit_mse": float(np.nanmean((y - fitted) ** 2)),
        "n": int(len(y)),
    }


def predict_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    beta = np.asarray(model["beta"], dtype=float)
    return float(model["y_mean"]) + ((x - mean) / std) @ beta


def parent_columns(panel: pd.DataFrame, parent: str) -> list[str]:
    return r25.strong_parents(panel, parent)


def fit_parent_models(panel: pd.DataFrame, spec: dict[str, Any], start_year: int, end_year: int) -> dict[str, Any]:
    parent = str(spec["parent"])
    horizon = int(spec["horizon"])
    target = f"y_{horizon}"
    columns = parent_columns(panel, parent)
    models: dict[str, Any] = {}
    for symbol, group in panel.groupby("symbol", sort=False):
        frame = group.loc[(group["time"].dt.year >= start_year) & (group["time"].dt.year <= end_year)]
        frame = r22.issue_frame(frame, parent, horizon, start_year, end_year)
        local = frame[columns + [target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(local) < max(200, 8 * len(columns)):
            continue
        models[str(symbol)] = fit_ridge(local[columns].to_numpy(float), local[target].to_numpy(float), PARENT_ALPHA)
    return {"columns": columns, "models": models, "horizon": horizon, "parent": parent}


def parent_predictions(
    panel: pd.DataFrame,
    fitted: dict[str, Any],
    start_year: int,
    end_year: int,
) -> tuple[pd.Series, pd.Series]:
    predictions = pd.Series(np.nan, index=panel.index, dtype=float)
    scales = pd.Series(np.nan, index=panel.index, dtype=float)
    columns = list(fitted["columns"])
    for symbol, model in dict(fitted["models"]).items():
        mask = (panel["symbol"].astype(str) == symbol) & panel["time"].dt.year.between(start_year, end_year)
        local = panel.loc[mask, columns].apply(pd.to_numeric, errors="coerce")
        valid = local.notna().all(axis=1)
        if not valid.any():
            continue
        predictions.loc[local.index[valid]] = predict_ridge(model, local.loc[valid].to_numpy(float))
        scale = max(float(model["target_std"]), 1.0e-8)
        scales.loc[local.index[valid]] = scale
    return predictions, scales


def development_oof_residuals(panel: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    horizon = int(spec["horizon"])
    target = f"y_{horizon}"
    scaled = pd.Series(np.nan, index=panel.index, dtype=float)
    raw = pd.Series(np.nan, index=panel.index, dtype=float)
    audit: dict[str, Any] = {}
    for test_year, train_year in [(2021, 2022), (2022, 2021)]:
        parent_model = fit_parent_models(panel, spec, train_year, train_year)
        pred, scale = parent_predictions(panel, parent_model, test_year, test_year)
        y = pd.to_numeric(panel[target], errors="coerce")
        residual = y - pred
        valid = residual.notna() & scale.notna()
        raw.loc[valid] = residual.loc[valid]
        scaled.loc[valid] = residual.loc[valid] / scale.loc[valid]
        audit[str(test_year)] = {
            "train_year": int(train_year),
            "rows": int(valid.sum()),
            "assets": int(panel.loc[valid, "symbol"].nunique()),
            "parent_mse": float(np.nanmean(residual.loc[valid].to_numpy(float) ** 2)) if valid.any() else float("nan"),
        }
    return scaled, raw, audit


def fit_residual_model(panel: pd.DataFrame, spec: dict[str, Any], scaled_residual: pd.Series) -> dict[str, Any]:
    parent = str(spec["parent"])
    basis, columns = r25.basis_matrix(panel, parent)
    symbols = panel["symbol"].astype(str).to_numpy()
    y = pd.to_numeric(scaled_residual, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(basis).all(axis=1) & np.isfinite(y) & panel["time"].dt.year.between(2021, 2022).to_numpy()
    minimum = max(120, 8 * basis.shape[1])
    if int(valid.sum()) < minimum:
        raise RuntimeError(f"insufficient OOF residual observations for {parent}: {int(valid.sum())}")
    x = basis[valid]
    train_symbols = symbols[valid]
    trained_symbols = sorted(set(train_symbols.tolist()))
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    x_std = (x - mean) / std
    design = r25.build_partial_pool_design(x_std, train_symbols, trained_symbols)
    penalty = np.full(design.shape[1], RESIDUAL_ASSET_ALPHA, dtype=float)
    penalty[: x_std.shape[1]] = RESIDUAL_GLOBAL_ALPHA
    y_mean = float(np.nanmean(y[valid]))
    beta = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ (y[valid] - y_mean))
    fitted = y_mean + design @ beta
    scale = float(np.nanstd(fitted))
    if not np.isfinite(scale) or scale <= 1.0e-8:
        scale = 1.0
    return {
        "family": str(spec["family"]),
        "factor": str(spec["factor"]),
        "parent": parent,
        "horizon": int(spec["horizon"]),
        "columns": columns,
        "trained_symbols": trained_symbols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "y_mean": y_mean,
        "beta": beta.tolist(),
        "prediction_scale": scale,
        "global_alpha": RESIDUAL_GLOBAL_ALPHA,
        "asset_alpha": RESIDUAL_ASSET_ALPHA,
        "n_train": int(valid.sum()),
        "fit_mse": float(np.nanmean((y[valid] - fitted) ** 2)),
        "training_target": "strict leave-one-development-year-out strong-parent residual standardized by parent target volatility",
    }


def apply_residual_model(panel: pd.DataFrame, model: dict[str, Any], output: str) -> None:
    basis, columns = r25.basis_matrix(panel, str(model["parent"]))
    if columns != list(model["columns"]):
        raise RuntimeError("residual constitutive basis contract changed")
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    standardized = (basis - mean) / std
    symbols = panel["symbol"].astype(str).to_numpy()
    design = r25.build_partial_pool_design(standardized, symbols, list(model["trained_symbols"]))
    beta = np.asarray(model["beta"], dtype=float)
    raw_scaled = float(model["y_mean"]) + design @ beta
    bounded = np.tanh(raw_scaled / float(model["prediction_scale"]))
    magnitude = ((np.abs(bounded) - DEADZONE) / (1.0 - DEADZONE)).clip(0.0, 1.0)
    panel[output] = np.sign(bounded) * magnitude
    panel[f"{output}__raw_scaled"] = raw_scaled


def asset_corr(frame: pd.DataFrame, x: str, y: str) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        local = group[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(local) < 40:
            continue
        value = local[x].corr(local[y], method="spearman")
        if np.isfinite(value):
            values[str(symbol)] = float(value)
    return pd.Series(values, dtype=float)


def evaluate_stage(
    panel: pd.DataFrame,
    spec: dict[str, Any],
    residual_model: dict[str, Any],
    parent_model: dict[str, Any],
    start_year: int,
    end_year: int,
    stage: str,
) -> dict[str, Any]:
    factor = str(spec["factor"])
    horizon = int(spec["horizon"])
    target = f"y_{horizon}"
    raw_column = f"{factor}__raw_scaled"
    parent_pred, target_scale = parent_predictions(panel, parent_model, start_year, end_year)
    panel[f"__parent_{factor}_{stage}"] = parent_pred
    panel[f"__scale_{factor}_{stage}"] = target_scale
    panel[f"__residual_target_{factor}_{stage}"] = pd.to_numeric(panel[target], errors="coerce") - parent_pred
    panel[f"__combined_{factor}_{stage}"] = parent_pred + pd.to_numeric(panel[raw_column], errors="coerce") * target_scale

    frame = r22.issue_frame(panel, factor, horizon, start_year, end_year)
    needed = [factor, raw_column, target, f"__parent_{factor}_{stage}", f"__residual_target_{factor}_{stage}", f"__combined_{factor}_{stage}"]
    frame = frame.loc[frame[needed].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)].copy()
    direct_assets = asset_corr(frame, factor, target)
    residual_assets = asset_corr(frame, factor, f"__residual_target_{factor}_{stage}")
    boot = r22.asset_bootstrap(residual_assets)
    yearly_direct: dict[str, float] = {}
    yearly_residual: dict[str, float] = {}
    for year in range(start_year, end_year + 1):
        local = frame.loc[frame["time"].dt.year == year]
        yearly_direct[str(year)] = float(asset_corr(local, factor, target).mean())
        yearly_residual[str(year)] = float(asset_corr(local, factor, f"__residual_target_{factor}_{stage}").mean())

    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    pred_parent = pd.to_numeric(frame[f"__parent_{factor}_{stage}"], errors="coerce").to_numpy(float)
    pred_combined = pd.to_numeric(frame[f"__combined_{factor}_{stage}"], errors="coerce").to_numpy(float)
    parent_mse = float(np.mean((y - pred_parent) ** 2))
    combined_mse = float(np.mean((y - pred_combined) ** 2))
    mse_gain = (parent_mse - combined_mse) / (parent_mse + EPS)
    parents = parent_columns(panel, str(spec["parent"]))
    max_corr = r22.max_parent_corr(frame, factor, parents)
    account = r23.execution_account(frame, factor, horizon, 5.0)

    direct_ts = float(direct_assets.mean()) if len(direct_assets) else float("nan")
    residual_ts = float(residual_assets.mean()) if len(residual_assets) else float("nan")
    positive_share = float((residual_assets > 0).mean()) if len(residual_assets) else float("nan")
    year_ok = all(np.isfinite(v) and v > 0.0 for v in yearly_residual.values())
    account_year_ok = all(np.isfinite(v) and v > 0.0 for v in dict(account["years"]).values())
    gate = bool(
        direct_ts > 0.010
        and residual_ts > 0.010
        and positive_share >= 0.60
        and float(boot["p_nonpositive"]) <= 0.10
        and mse_gain > 0.0
        and max_corr < 0.80
        and float(account["annual_log_return"]) > 0.0
        and year_ok
        and account_year_ok
    )
    result = {
        "stage": stage,
        "family": str(spec["family"]),
        "factor": factor,
        "parent": str(spec["parent"]),
        "horizon": horizon,
        "direct_ts_ic": direct_ts,
        "residual_ts_ic": residual_ts,
        "residual_positive_asset_share": positive_share,
        "asset_count": int(len(residual_assets)),
        "bootstrap_ci_low": float(boot["ci_low"]),
        "bootstrap_ci_high": float(boot["ci_high"]),
        "bootstrap_p_nonpositive": float(boot["p_nonpositive"]),
        "yearly_direct_ts_ic": json.dumps(yearly_direct, sort_keys=True),
        "yearly_residual_ts_ic": json.dumps(yearly_residual, sort_keys=True),
        "parent_mse": parent_mse,
        "combined_mse": combined_mse,
        "combined_mse_gain": float(mse_gain),
        "max_parent_corr": float(max_corr),
        "annual_log_return_5bp": float(account["annual_log_return"]),
        "sharpe_5bp": float(account["sharpe"]),
        "avg_turnover": float(account["avg_turnover"]),
        "portfolio_years": json.dumps(account["years"], sort_keys=True),
        "gate": gate,
    }
    panel.drop(columns=[f"__parent_{factor}_{stage}", f"__scale_{factor}_{stage}", f"__residual_target_{factor}_{stage}", f"__combined_{factor}_{stage}"], inplace=True)
    return result


def main() -> None:
    np.random.seed(SEED)
    discovery, discovery_audit = r25.build_panel(DISCOVERY_SYMBOLS, OUT / "discovery_cache")
    models: dict[str, Any] = {}
    parent_models_discovery: dict[str, Any] = {}
    oof_audit: dict[str, Any] = {}
    for spec in SPECS:
        scaled_residual, _, local_audit = development_oof_residuals(discovery, spec)
        residual_model = fit_residual_model(discovery, spec, scaled_residual)
        apply_residual_model(discovery, residual_model, str(spec["factor"]))
        parent_model = fit_parent_models(discovery, spec, 2021, 2022)
        models[str(spec["factor"])] = residual_model
        parent_models_discovery[str(spec["factor"])] = parent_model
        oof_audit[str(spec["factor"])] = local_audit

    discovery_validation = pd.DataFrame([
        evaluate_stage(discovery, spec, models[str(spec["factor"])], parent_models_discovery[str(spec["factor"])], 2023, 2024, "discovery_validation")
        for spec in SPECS
    ])
    discovery_holdout = pd.DataFrame([
        evaluate_stage(discovery, spec, models[str(spec["factor"])], parent_models_discovery[str(spec["factor"])], 2025, 2026, "discovery_holdout")
        for spec in SPECS
    ])

    replication, replication_audit = r25.build_panel(REPLICATION_SYMBOLS, OUT / "replication_cache")
    parent_models_replication: dict[str, Any] = {}
    for spec in SPECS:
        apply_residual_model(replication, models[str(spec["factor"])], str(spec["factor"]))
        parent_models_replication[str(spec["factor"])] = fit_parent_models(replication, spec, 2021, 2022)

    replication_validation = pd.DataFrame([
        evaluate_stage(replication, spec, models[str(spec["factor"])], parent_models_replication[str(spec["factor"])], 2023, 2024, "replication_validation")
        for spec in SPECS
    ])
    replication_holdout = pd.DataFrame([
        evaluate_stage(replication, spec, models[str(spec["factor"])], parent_models_replication[str(spec["factor"])], 2025, 2026, "replication_holdout")
        for spec in SPECS
    ])

    tables = {
        "DISCOVERY_VALIDATION_RESULTS.csv": discovery_validation,
        "DISCOVERY_HOLDOUT_RESULTS.csv": discovery_holdout,
        "REPLICATION_VALIDATION_RESULTS.csv": replication_validation,
        "REPLICATION_HOLDOUT_RESULTS.csv": replication_holdout,
    }
    for name, table in tables.items():
        table.to_csv(OUT / name, index=False)

    confirmed: list[dict[str, Any]] = []
    for spec in SPECS:
        factor = str(spec["factor"])
        records = []
        for table in tables.values():
            matched = table.loc[table["factor"] == factor]
            if matched.empty:
                records = []
                break
            records.append(matched.iloc[0])
        if records and all(bool(record["gate"]) for record in records):
            confirmed.append({
                "factor": factor,
                "horizon": int(spec["horizon"]),
                "all_four_stage_gates": True,
                "minimum_direct_ts_ic": float(min(record["direct_ts_ic"] for record in records)),
                "minimum_residual_ts_ic": float(min(record["residual_ts_ic"] for record in records)),
                "minimum_combined_mse_gain": float(min(record["combined_mse_gain"] for record in records)),
                "minimum_annual_log_return_5bp": float(min(record["annual_log_return_5bp"] for record in records)),
            })

    summary = {
        "spec_count": len(SPECS),
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
    write_json(OUT / "OOF_PARENT_AUDIT.json", oof_audit)
    write_json(OUT / "CANDIDATE_CONTRACTS.json", {spec["factor"]: spec for spec in SPECS})
    write_json(OUT / "DATA_AUDIT.json", {
        "seed": SEED,
        "discovery_symbols": DISCOVERY_SYMBOLS,
        "replication_symbols": REPLICATION_SYMBOLS,
        "discovery_data": discovery_audit,
        "replication_data": replication_audit,
        "parent_training": "per-asset Ridge using only 2021-2022; development residual target formed by leave-one-development-year-out parent predictions",
        "residual_training": "global plus strongly shrunk asset deviations fitted to strict OOF parent residuals only",
        "replication": "residual constitutive coefficients frozen from discovery development; only replication parent baseline is fitted on replication 2021-2022",
        "cost_bp_one_way": 5.0,
        "confirmation": "same gate required in discovery validation, discovery holdout, disjoint replication validation and disjoint replication holdout",
    })

    report = [
        "# Parent-orthogonal residual constitutive research round26",
        "",
        "The fluid constitutive layer is trained only on strict out-of-fold residual returns left unexplained by the strong parent state model. The final test is whether adding the frozen residual prediction improves raw-return MSE, residual TSIC and 5bp net execution across four stages.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
    ]
    for title, table in [
        ("Discovery validation", discovery_validation),
        ("Discovery historical holdout", discovery_holdout),
        ("Disjoint-universe validation", replication_validation),
        ("Disjoint-universe historical holdout", replication_holdout),
    ]:
        report.extend(["", f"## {title}", table.to_markdown(index=False)])
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
