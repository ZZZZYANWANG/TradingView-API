from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUND21 = HERE.parent / "fluid_round21"
if str(ROUND21) not in sys.path:
    sys.path.insert(0, str(ROUND21))

import run_nonnormal_operator_round21_fast as r21fast  # noqa: E402

r21 = r21fast.experiment
base = r21.base
EPS = base.EPS
SEED = 20261119
OUT = Path("research_outputs/fluid_timeseries_operator_round22")
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_SYMBOLS = list(r21.DISCOVERY_SYMBOLS)
REPLICATION_SYMBOLS = list(r21.REPLICATION_SYMBOLS)
HORIZONS = [24, 72, 120]
STATE_SPANS = [6, 24, 72]
NO_TRADE_THRESHOLD = 0.35
POSITION_SCALE = 1.25
RIDGE_ALPHA = 10.0
BOOTSTRAP_REPS = 10000

PARENT_CANDIDATES = [
    "perp_u", "spot_u", "basis_z", "basis_change_z", "pressure_gap",
    "oi_source_z", "funding_change_z", "quote_share_z", "imb6_z",
    "perp_quote_z", "spot_quote_z", "oi_level_z",
    "operator_cumulative_shear_h24", "operator_cumulative_shear_h72",
    "nonnormal_shear_h24", "nonnormal_shear_h72",
    "transient_growth_direction_h24", "operator_rotation_shear_h24",
    "source_impulse_shear_h24", "pressure_impulse_shear_h24",
    "slow_mode_shear_h24",
]

FAMILY_RAW: dict[str, list[str]] = {
    "intrinsic_transient_direction": [
        "intrinsic_nonnormal_residual_h24",
        "nonnormal_term_slope",
        "rotation_supported_transient",
    ],
    "source_supported_transport": [
        "source_supported_nonnormal",
        "source_supported_transient",
        "source_pressure_consensus",
    ],
    "pressure_supported_transport": [
        "pressure_supported_nonnormal",
        "pressure_supported_transient",
        "impulse_balance_direction",
    ],
    "modal_direction_amplitude": [
        "slow_mode_damped_nonnormal",
        "endogenous_operator_transport",
        "transient_minus_slow_mode",
    ],
}

CONTRACTS = {
    "intrinsic_nonnormal_residual_h24": {
        "equation": "NN_24 - beta_prior(OP_24, NN_24)*OP_24",
        "mechanism": "non-normal transient direction orthogonal to the ordinary locally linear operator forecast",
    },
    "nonnormal_term_slope": {
        "equation": "NN_24-NN_72",
        "mechanism": "short-minus-long non-normal transport term structure",
    },
    "rotation_supported_transient": {
        "equation": "TG_24*tanh(rotation_24)",
        "mechanism": "transient growth retained only when the antisymmetric operator rotation supports the same direction",
    },
    "source_supported_nonnormal": {
        "equation": "NN_24*tanh(source_impulse_24)",
        "mechanism": "non-normal shear amplification supported by observable position and participation sources",
    },
    "source_supported_transient": {
        "equation": "TG_24*tanh(source_impulse_24)",
        "mechanism": "signed transient growth conditional on source-to-shear transmission",
    },
    "source_pressure_consensus": {
        "equation": "NN_24*tanh(source_impulse_24+pressure_impulse_24)",
        "mechanism": "transient shear supported jointly by source and boundary-pressure impulses",
    },
    "pressure_supported_nonnormal": {
        "equation": "NN_24*tanh(pressure_impulse_24)",
        "mechanism": "non-normal transport supported by pressure and basis propagation",
    },
    "pressure_supported_transient": {
        "equation": "TG_24*tanh(pressure_impulse_24)",
        "mechanism": "transient growth conditional on pressure-to-shear transmission",
    },
    "impulse_balance_direction": {
        "equation": "NN_24*tanh(source_impulse_24-pressure_impulse_24)",
        "mechanism": "directional shear response to source-pressure impulse imbalance",
    },
    "slow_mode_damped_nonnormal": {
        "equation": "NN_24/(1+abs(slow_mode_24))",
        "mechanism": "transient non-normal direction attenuated when a slow stable mode dominates amplitude",
    },
    "endogenous_operator_transport": {
        "equation": "OP_24-source_impulse_24-pressure_impulse_24",
        "mechanism": "operator-implied shear attributable to endogenous state rotation rather than observable source or pressure impulses",
    },
    "transient_minus_slow_mode": {
        "equation": "TG_24-slow_mode_24",
        "mechanism": "directional transient response separated from the slow modal contribution",
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def causal_beta(x: pd.Series, y: pd.Series, window: int = 336, ridge: float = 0.10) -> pd.Series:
    xs = pd.to_numeric(x, errors="coerce").shift(1)
    ys = pd.to_numeric(y, errors="coerce").shift(1)
    minimum = max(120, window // 2)
    mx = xs.rolling(window, min_periods=minimum).mean()
    my = ys.rolling(window, min_periods=minimum).mean()
    cov = (xs * ys).rolling(window, min_periods=minimum).mean() - mx * my
    var = xs.pow(2).rolling(window, min_periods=minimum).mean() - mx.pow(2)
    return cov / (var.clip(lower=0.0) + ridge)


def add_round22_candidates(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    g = group.sort_values("time").copy()
    op24 = pd.to_numeric(g["operator_cumulative_shear_h24"], errors="coerce")
    nn24 = pd.to_numeric(g["nonnormal_shear_h24"], errors="coerce")
    nn72 = pd.to_numeric(g["nonnormal_shear_h72"], errors="coerce")
    tg24 = pd.to_numeric(g["transient_growth_direction_h24"], errors="coerce")
    rot24 = pd.to_numeric(g["operator_rotation_shear_h24"], errors="coerce")
    source24 = pd.to_numeric(g["source_impulse_shear_h24"], errors="coerce")
    pressure24 = pd.to_numeric(g["pressure_impulse_shear_h24"], errors="coerce")
    slow24 = pd.to_numeric(g["slow_mode_shear_h24"], errors="coerce")

    beta = causal_beta(op24, nn24)
    raw: dict[str, pd.Series] = {
        "intrinsic_nonnormal_residual_h24": nn24 - beta * op24,
        "nonnormal_term_slope": nn24 - nn72,
        "rotation_supported_transient": tg24 * np.tanh(rot24),
        "source_supported_nonnormal": nn24 * np.tanh(source24),
        "source_supported_transient": tg24 * np.tanh(source24),
        "source_pressure_consensus": nn24 * np.tanh(source24 + pressure24),
        "pressure_supported_nonnormal": nn24 * np.tanh(pressure24),
        "pressure_supported_transient": tg24 * np.tanh(pressure24),
        "impulse_balance_direction": nn24 * np.tanh(source24 - pressure24),
        "slow_mode_damped_nonnormal": nn24 / (1.0 + slow24.abs()),
        "endogenous_operator_transport": op24 - source24 - pressure24,
        "transient_minus_slow_mode": tg24 - slow24,
    }
    name_to_family = {name: family for family, names in FAMILY_RAW.items() for name in names}
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for name, values in raw.items():
        z = base.causal_z(values)
        g[f"raw_{name}"] = values
        for span in STATE_SPANS:
            column = f"{name}_ewm{span}"
            g[column] = z.ewm(span=span, adjust=False, min_periods=span).mean()
            candidate_map[name_to_family[name]].append(column)
    return g, candidate_map


def build_panel(symbols: list[str], out: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    r21.configure_round20_driver()
    panel, audit, _ = r21.r20.build_panel(symbols, out)
    parts: list[pd.DataFrame] = []
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for symbol, group in panel.groupby("symbol", sort=False):
        enriched, local_map = add_round22_candidates(group)
        parts.append(enriched)
        candidate_map = local_map
        print(f"round22_candidates={symbol} rows={len(enriched)}", flush=True)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True), audit, candidate_map


def issue_frame(panel: pd.DataFrame, factor: str, horizon: int, start_year: int, end_year: int) -> pd.DataFrame:
    target = f"y_{horizon}"
    frame = panel.loc[(panel["time"].dt.year >= start_year) & (panel["time"].dt.year <= end_year)].copy()
    frame = base.issue_subset(frame, horizon)
    return frame.loc[np.isfinite(pd.to_numeric(frame[factor], errors="coerce")) & np.isfinite(pd.to_numeric(frame[target], errors="coerce"))].copy()


def asset_tsics(frame: pd.DataFrame, factor: str, target: str) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        local = group[[factor, target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(local) < 40:
            continue
        value = local[factor].corr(local[target], method="spearman")
        if np.isfinite(value):
            values[str(symbol)] = float(value)
    return pd.Series(values, dtype=float)


def ts_summary(frame: pd.DataFrame, factor: str, horizon: int) -> dict[str, Any]:
    target = f"y_{horizon}"
    by_asset = asset_tsics(frame, factor, target)
    return {
        "ts_ic": float(by_asset.mean()) if len(by_asset) else float("nan"),
        "positive_asset_share": float((by_asset > 0).mean()) if len(by_asset) else float("nan"),
        "asset_count": int(len(by_asset)),
        "asset_tsics": by_asset,
    }


def asset_bootstrap(by_asset: pd.Series, reps: int = BOOTSTRAP_REPS) -> dict[str, float]:
    values = by_asset.dropna().to_numpy(dtype=float)
    if len(values) < 5:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "p_nonpositive": 1.0}
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return {
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_nonpositive": float(np.mean(draws <= 0.0)),
    }


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    xtr = (x_train - mean) / std
    xte = (x_test - mean) / std
    y_mean = float(np.nanmean(y_train))
    yc = y_train - y_mean
    beta = np.linalg.solve(xtr.T @ xtr + alpha * np.eye(xtr.shape[1]), xtr.T @ yc)
    return y_mean + xte @ beta


def available_parents(panel: pd.DataFrame, factor: str) -> list[str]:
    return [column for column in PARENT_CANDIDATES if column in panel.columns and column != factor]


def residual_factor(panel: pd.DataFrame, factor: str, parents: list[str], train_end: int, stage_start: int, stage_end: int) -> pd.Series:
    residual = pd.Series(np.nan, index=panel.index, dtype=float)
    for _, group in panel.groupby("symbol", sort=False):
        train = group.loc[group["time"].dt.year <= train_end]
        test = group.loc[(group["time"].dt.year >= stage_start) & (group["time"].dt.year <= stage_end)]
        columns = parents + [factor]
        train_valid = train[columns].apply(pd.to_numeric, errors="coerce").dropna()
        test_valid = test[columns].apply(pd.to_numeric, errors="coerce").dropna()
        if len(train_valid) < 500 or len(test_valid) < 80:
            continue
        pred = ridge_fit_predict(train_valid[parents].to_numpy(float), train_valid[factor].to_numpy(float), test_valid[parents].to_numpy(float))
        residual.loc[test_valid.index] = test_valid[factor].to_numpy(float) - pred
    return residual


def model_increment(panel: pd.DataFrame, factor: str, horizon: int, parents: list[str], train_end: int, stage_start: int, stage_end: int) -> float:
    target = f"y_{horizon}"
    se_base: list[np.ndarray] = []
    se_aug: list[np.ndarray] = []
    for _, group in panel.groupby("symbol", sort=False):
        train = group.loc[group["time"].dt.year <= train_end]
        test = group.loc[(group["time"].dt.year >= stage_start) & (group["time"].dt.year <= stage_end)]
        base_cols = parents + [target]
        aug_cols = parents + [factor, target]
        train_base = train[base_cols].apply(pd.to_numeric, errors="coerce").dropna()
        train_aug = train[aug_cols].apply(pd.to_numeric, errors="coerce").dropna()
        test_aug = test[aug_cols].apply(pd.to_numeric, errors="coerce").dropna()
        common_train = train_base.index.intersection(train_aug.index)
        if len(common_train) < 500 or len(test_aug) < 80:
            continue
        x_base = train.loc[common_train, parents].to_numpy(float)
        x_aug = train.loc[common_train, parents + [factor]].to_numpy(float)
        y_train = train.loc[common_train, target].to_numpy(float)
        pred_base = ridge_fit_predict(x_base, y_train, test_aug[parents].to_numpy(float))
        pred_aug = ridge_fit_predict(x_aug, y_train, test_aug[parents + [factor]].to_numpy(float))
        y_test = test_aug[target].to_numpy(float)
        se_base.append((y_test - pred_base) ** 2)
        se_aug.append((y_test - pred_aug) ** 2)
    if not se_base:
        return float("nan")
    base_mse = float(np.mean(np.concatenate(se_base)))
    aug_mse = float(np.mean(np.concatenate(se_aug)))
    return (base_mse - aug_mse) / (base_mse + EPS)


def max_parent_corr(frame: pd.DataFrame, factor: str, parents: list[str]) -> float:
    values: list[float] = []
    for parent in parents:
        local = frame[[factor, parent]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(local) < 200:
            continue
        corr = local[factor].corr(local[parent], method="spearman")
        if np.isfinite(corr):
            values.append(abs(float(corr)))
    return max(values) if values else float("nan")


def ts_account(frame: pd.DataFrame, factor: str, horizon: int, cost_bp: float = 5.0) -> dict[str, Any]:
    target = f"y_{horizon}"
    work = frame[["time", "symbol", factor, target]].copy()
    work[factor] = pd.to_numeric(work[factor], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna()
    if work.empty:
        return {"annual_log_return": float("nan"), "sharpe": float("nan"), "avg_turnover": float("nan"), "years": {}}
    score = np.tanh(work[factor] / POSITION_SCALE)
    score = score.where(score.abs() >= NO_TRADE_THRESHOLD, 0.0)
    work["position"] = score.clip(-1.0, 1.0)
    previous = work.groupby("symbol", sort=False)["position"].shift(1).fillna(0.0)
    work["turnover"] = (work["position"] - previous).abs()
    work["net"] = work["position"] * work[target] - work["turnover"] * cost_bp * 1.0e-4
    issue_net = work.groupby("time", sort=True)["net"].mean()
    issue_turnover = work.groupby("time", sort=True)["turnover"].mean()
    scale = 365.25 * 24.0 / float(horizon)
    annual = float(issue_net.mean() * scale)
    sharpe = float(issue_net.mean() / (issue_net.std(ddof=0) + EPS) * math.sqrt(scale))
    years = {str(int(year)): float(values.mean() * scale) for year, values in issue_net.groupby(issue_net.index.year)}
    return {"annual_log_return": annual, "sharpe": sharpe, "avg_turnover": float(issue_turnover.mean()), "years": years}


def development_score(panel: pd.DataFrame, factor: str, horizon: int, sign: int) -> dict[str, Any]:
    oriented = f"__oriented_{factor}"
    panel[oriented] = sign * pd.to_numeric(panel[factor], errors="coerce")
    frame = issue_frame(panel, oriented, horizon, 2021, 2022)
    summary = ts_summary(frame, oriented, horizon)
    yearly = {}
    for year in [2021, 2022]:
        local = frame.loc[frame["time"].dt.year == year]
        yearly[str(year)] = ts_summary(local, oriented, horizon)["ts_ic"]
    account = ts_account(frame, oriented, horizon, 5.0)
    finite_years = [value for value in yearly.values() if np.isfinite(value)]
    min_year = min(finite_years) if finite_years else -1.0
    ts_ic = float(summary["ts_ic"])
    score = min_year + 0.40 * ts_ic + 0.05 * max(float(account["annual_log_return"]), -0.20)
    panel.drop(columns=[oriented], inplace=True)
    return {
        "factor": factor,
        "sign": int(sign),
        "horizon": int(horizon),
        "development_ts_ic": ts_ic,
        "development_positive_asset_share": float(summary["positive_asset_share"]),
        "development_min_year": float(min_year),
        "development_years": json.dumps(yearly, sort_keys=True),
        "development_annual_log_return_5bp": float(account["annual_log_return"]),
        "development_sharpe_5bp": float(account["sharpe"]),
        "development_score": float(score),
    }


def robust_select(panel: pd.DataFrame, candidate_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for family, factors in candidate_map.items():
        for factor in factors:
            for horizon in HORIZONS:
                for sign in [1, -1]:
                    result = development_score(panel, factor, horizon, sign)
                    result["family"] = family
                    rows.append(result)
    scan = pd.DataFrame(rows)
    selected: list[pd.Series] = []
    for family, group in scan.groupby("family", sort=False):
        eligible = group.loc[
            (group["development_ts_ic"] > 0.010)
            & (group["development_min_year"] > -0.005)
            & (group["development_annual_log_return_5bp"] > 0.0)
            & (group["development_positive_asset_share"] >= 0.55)
        ]
        if eligible.empty:
            continue
        selected.append(eligible.sort_values("development_score", ascending=False).iloc[0])
    selection = pd.DataFrame(selected).reset_index(drop=True)
    if not selection.empty:
        selection["oriented_factor"] = selection["factor"].map(lambda value: f"oriented_{value}")
        for row in selection.itertuples(index=False):
            panel[str(row.oriented_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")
    return scan, selection


def holm(values: list[float]) -> list[float]:
    return base.holm_adjust([float(value) if np.isfinite(value) else 1.0 for value in values])


def evaluate_stage(panel: pd.DataFrame, selection: pd.DataFrame, start_year: int, end_year: int, train_end: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        factor = str(row.oriented_factor)
        horizon = int(row.horizon)
        frame = issue_frame(panel, factor, horizon, start_year, end_year)
        direct = ts_summary(frame, factor, horizon)
        boot = asset_bootstrap(direct["asset_tsics"])
        yearly = {}
        for year in range(start_year, end_year + 1):
            yearly[str(year)] = ts_summary(frame.loc[frame["time"].dt.year == year], factor, horizon)["ts_ic"]
        parents = available_parents(panel, factor)
        residual_column = f"__residual_{factor}_{start_year}_{end_year}"
        panel[residual_column] = residual_factor(panel, factor, parents, train_end, start_year, end_year)
        residual_frame = issue_frame(panel, residual_column, horizon, start_year, end_year)
        residual = ts_summary(residual_frame, residual_column, horizon)
        increment = model_increment(panel, factor, horizon, parents, train_end, start_year, end_year)
        corr = max_parent_corr(frame, factor, parents)
        account = ts_account(frame, factor, horizon, 5.0)
        panel.drop(columns=[residual_column], inplace=True)
        rows.append({
            "family": str(row.family),
            "primitive_factor": str(row.factor),
            "factor": factor,
            "sign": int(row.sign),
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
    result["holm_p"] = holm(result["bootstrap_p_nonpositive"].tolist())
    gates = []
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


def apply_selection(panel: pd.DataFrame, selection: pd.DataFrame) -> None:
    for row in selection.itertuples(index=False):
        panel[str(row.oriented_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")


def main() -> None:
    np.random.seed(SEED)
    discovery_panel, discovery_audit, candidate_map = build_panel(DISCOVERY_SYMBOLS, OUT / "discovery_cache")
    scan, selection = robust_select(discovery_panel, candidate_map)
    scan.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)
    selection.to_csv(OUT / "FROZEN_SELECTION.csv", index=False)

    validation = evaluate_stage(discovery_panel, selection, 2023, 2024, 2022)
    validation.to_csv(OUT / "DISCOVERY_VALIDATION_RESULTS.csv", index=False)
    promoted = validation.loc[validation.get("gate", pd.Series(dtype=bool))].copy() if not validation.empty else pd.DataFrame()

    holdout = pd.DataFrame()
    confirmed_discovery = pd.DataFrame()
    if not promoted.empty:
        frozen = selection.loc[selection["oriented_factor"].isin(promoted["factor"])].copy()
        holdout = evaluate_stage(discovery_panel, frozen, 2025, 2026, 2024)
        holdout.to_csv(OUT / "DISCOVERY_HOLDOUT_RESULTS.csv", index=False)
        confirmed_discovery = holdout.loc[holdout.get("gate", pd.Series(dtype=bool))].copy()
    else:
        pd.DataFrame().to_csv(OUT / "DISCOVERY_HOLDOUT_RESULTS.csv", index=False)

    replication_validation = pd.DataFrame()
    replication_holdout = pd.DataFrame()
    confirmed = pd.DataFrame()
    replication_audit: dict[str, Any] | None = None
    if not confirmed_discovery.empty:
        frozen = selection.loc[selection["oriented_factor"].isin(confirmed_discovery["factor"])].copy()
        replication_panel, replication_audit, _ = build_panel(REPLICATION_SYMBOLS, OUT / "replication_cache")
        apply_selection(replication_panel, frozen)
        replication_validation = evaluate_stage(replication_panel, frozen, 2023, 2024, 2022)
        replication_validation.to_csv(OUT / "REPLICATION_VALIDATION_RESULTS.csv", index=False)
        passed_replication = replication_validation.loc[replication_validation.get("gate", pd.Series(dtype=bool))].copy()
        if not passed_replication.empty:
            frozen2 = frozen.loc[frozen["oriented_factor"].isin(passed_replication["factor"])].copy()
            replication_holdout = evaluate_stage(replication_panel, frozen2, 2025, 2026, 2024)
            replication_holdout.to_csv(OUT / "REPLICATION_HOLDOUT_RESULTS.csv", index=False)
            confirmed = replication_holdout.loc[replication_holdout.get("gate", pd.Series(dtype=bool))].copy()
    else:
        pd.DataFrame().to_csv(OUT / "REPLICATION_VALIDATION_RESULTS.csv", index=False)
        pd.DataFrame().to_csv(OUT / "REPLICATION_HOLDOUT_RESULTS.csv", index=False)

    summary = {
        "raw_primitives": int(sum(len(values) for values in FAMILY_RAW.values())),
        "candidate_columns": int(sum(len(values) for values in candidate_map.values())),
        "development_tests": int(len(scan)),
        "frozen_identities": int(len(selection)),
        "discovery_validation_passed": int(validation.get("gate", pd.Series(dtype=bool)).sum()) if not validation.empty else 0,
        "discovery_holdout_opened": int(len(holdout)),
        "discovery_holdout_passed": int(holdout.get("gate", pd.Series(dtype=bool)).sum()) if not holdout.empty else 0,
        "replication_validation_opened": int(len(replication_validation)),
        "replication_validation_passed": int(replication_validation.get("gate", pd.Series(dtype=bool)).sum()) if not replication_validation.empty else 0,
        "replication_holdout_opened": int(len(replication_holdout)),
        "confirmed": confirmed.to_dict(orient="records") if not confirmed.empty else [],
        "decision": "eligible_for_trade_only_if_confirmed_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    write_json(OUT / "CONFIRMED.json", summary["confirmed"])
    write_json(OUT / "CANDIDATE_CONTRACTS.json", CONTRACTS)
    write_json(OUT / "DATA_AUDIT.json", {
        "seed": SEED,
        "discovery_symbols": DISCOVERY_SYMBOLS,
        "replication_symbols": REPLICATION_SYMBOLS,
        "discovery_data": discovery_audit,
        "replication_data": replication_audit,
        "horizons_hours": HORIZONS,
        "development": "2021-2022",
        "validation": "2023-2024",
        "historical_holdout": "2025-2026-09",
        "selection_rule": "one time-series identity per mechanism family, frozen using 2021-2022 TSIC, yearly sign, positive-asset share and 5bp non-overlapping account",
        "execution": {"position": "tanh(factor/1.25)", "no_trade_threshold": NO_TRADE_THRESHOLD, "cost_bp_one_way": 5.0},
        "gate": "mean asset TSIC>0.02, positive asset share>=0.60, asset bootstrap and Holm<=0.10, residual TSIC>0.01, Ridge MSE gain>0, max parent corr<0.80, 5bp net return>0, every stage year positive",
        "research_boundary": "historical research; replication universe remains closed unless discovery validation and holdout both pass",
    })
    report = [
        "# Time-series non-normal transport research round22",
        "",
        "This experiment reads the local spot-perpetual propagation operator as an asset-internal time-series state rather than a cross-sectional score. Direction and amplitude are separated, positions use a fixed no-trade region, and validation requires parent-residual TSIC, strong-model increment and 5bp net returns.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen selection",
        selection.to_markdown(index=False) if not selection.empty else "No family passed the development freeze.",
        "",
        "## Discovery validation",
        validation.to_markdown(index=False) if not validation.empty else "Validation remained closed.",
        "",
        "## Discovery holdout",
        holdout.to_markdown(index=False) if not holdout.empty else "Holdout remained closed.",
        "",
        "## Replication",
        replication_validation.to_markdown(index=False) if not replication_validation.empty else "Replication remained closed.",
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
