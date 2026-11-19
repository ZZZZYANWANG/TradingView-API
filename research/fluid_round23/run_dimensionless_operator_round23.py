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
if str(ROUND22) not in sys.path:
    sys.path.insert(0, str(ROUND22))

import run_timeseries_operator_round22 as r22  # noqa: E402

base = r22.base
EPS = r22.EPS
SEED = 20261119
OUT = Path("research_outputs/fluid_dimensionless_operator_round23")
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_SYMBOLS = list(r22.DISCOVERY_SYMBOLS)
REPLICATION_SYMBOLS = list(r22.REPLICATION_SYMBOLS)
POSITION_SCALE = r22.POSITION_SCALE
NO_TRADE_THRESHOLD = r22.NO_TRADE_THRESHOLD

# Frozen directly from Round22. No formula, sign, horizon, scale or threshold is
# selected on Round23 validation, holdout or replication data.
FROZEN_IDENTITIES = [
    {
        "family": "pressure_supported_transport",
        "primitive_factor": "impulse_balance_direction_ewm24",
        "sign": 1,
        "horizon": 120,
        "factor": "execution_impulse_balance_direction_120h",
    },
    {
        "family": "modal_direction_amplitude",
        "primitive_factor": "slow_mode_damped_nonnormal_ewm72",
        "sign": 1,
        "horizon": 72,
        "factor": "execution_slow_mode_damped_nonnormal_72h",
    },
]

CONTRACTS = {
    "execution_impulse_balance_direction_120h": {
        "equation": "deadzone[tanh(impulse_balance_direction_ewm24/1.25), 0.35]",
        "mechanism": "dimensionless tradable excitation generated when source and pressure impulses create an imbalanced non-normal shear response",
        "parent": "impulse_balance_direction_ewm24",
        "horizon_hours": 120,
    },
    "execution_slow_mode_damped_nonnormal_72h": {
        "equation": "deadzone[tanh(slow_mode_damped_nonnormal_ewm72/1.25), 0.35]",
        "mechanism": "bounded transient direction after attenuation by the slow stable mode; the same map was frozen as Round22 execution policy",
        "parent": "slow_mode_damped_nonnormal_ewm72",
        "horizon_hours": 72,
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def execution_coordinate(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce")
    bounded = np.tanh(raw / POSITION_SCALE)
    magnitude = ((np.abs(bounded) - NO_TRADE_THRESHOLD) / (1.0 - NO_TRADE_THRESHOLD)).clip(0.0, 1.0)
    return np.sign(bounded) * magnitude


def add_round23_coordinates(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    for identity in FROZEN_IDENTITIES:
        parent = str(identity["primitive_factor"])
        output = str(identity["factor"])
        sign = int(identity["sign"])
        if parent not in result:
            raise KeyError(f"missing frozen Round22 parent {parent}")
        result[output] = execution_coordinate(sign * pd.to_numeric(result[parent], errors="coerce"))
    return result


def build_panel(symbols: list[str], out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, audit, _ = r22.build_panel(symbols, out)
    return add_round23_coordinates(panel), audit


def fixed_selection() -> pd.DataFrame:
    rows = []
    for identity in FROZEN_IDENTITIES:
        row = dict(identity)
        row["oriented_factor"] = row["factor"]
        rows.append(row)
    return pd.DataFrame(rows)


def execution_account(frame: pd.DataFrame, factor: str, horizon: int, cost_bp: float = 5.0) -> dict[str, Any]:
    target = f"y_{horizon}"
    work = frame[["time", "symbol", factor, target]].copy()
    work[factor] = pd.to_numeric(work[factor], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna().sort_values(["symbol", "time"])
    if work.empty:
        return {"annual_log_return": float("nan"), "sharpe": float("nan"), "avg_turnover": float("nan"), "years": {}}
    work["position"] = work[factor].clip(-1.0, 1.0)
    previous = work.groupby("symbol", sort=False)["position"].shift(1).fillna(0.0)
    work["turnover"] = (work["position"] - previous).abs()
    work["net"] = work["position"] * work[target] - work["turnover"] * cost_bp * 1.0e-4
    issue_net = work.groupby("time", sort=True)["net"].mean()
    issue_turnover = work.groupby("time", sort=True)["turnover"].mean()
    scale = 365.25 * 24.0 / float(horizon)
    annual = float(issue_net.mean() * scale)
    sharpe = float(issue_net.mean() / (issue_net.std(ddof=0) + EPS) * np.sqrt(scale))
    years = {str(int(year)): float(values.mean() * scale) for year, values in issue_net.groupby(issue_net.index.year)}
    return {"annual_log_return": annual, "sharpe": sharpe, "avg_turnover": float(issue_turnover.mean()), "years": years}


def evaluate(panel: pd.DataFrame, selection: pd.DataFrame, start_year: int, end_year: int, train_end: int, stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        factor = str(row.factor)
        horizon = int(row.horizon)
        frame = r22.issue_frame(panel, factor, horizon, start_year, end_year)
        direct = r22.ts_summary(frame, factor, horizon)
        boot = r22.asset_bootstrap(direct["asset_tsics"])
        yearly = {
            str(year): r22.ts_summary(frame.loc[frame["time"].dt.year == year], factor, horizon)["ts_ic"]
            for year in range(start_year, end_year + 1)
        }
        parents = r22.available_parents(panel, factor)
        residual_column = f"__residual_{factor}_{stage}"
        panel[residual_column] = r22.residual_factor(panel, factor, parents, train_end, start_year, end_year)
        residual_frame = r22.issue_frame(panel, residual_column, horizon, start_year, end_year)
        residual = r22.ts_summary(residual_frame, residual_column, horizon)
        increment = r22.model_increment(panel, factor, horizon, parents, train_end, start_year, end_year)
        corr = r22.max_parent_corr(frame, factor, parents)
        account = execution_account(frame, factor, horizon, 5.0)
        panel.drop(columns=[residual_column], inplace=True)
        rows.append({
            "stage": stage,
            "family": str(row.family),
            "primitive_factor": str(row.primitive_factor),
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


def main() -> None:
    np.random.seed(SEED)
    selection = fixed_selection()
    selection.to_csv(OUT / "FROZEN_SELECTION.csv", index=False)

    discovery, discovery_audit = build_panel(DISCOVERY_SYMBOLS, OUT / "discovery_cache")
    discovery_validation = evaluate(discovery, selection, 2023, 2024, 2022, "discovery_validation")
    discovery_holdout = evaluate(discovery, selection, 2025, 2026, 2024, "discovery_holdout")
    discovery_validation.to_csv(OUT / "DISCOVERY_VALIDATION_RESULTS.csv", index=False)
    discovery_holdout.to_csv(OUT / "DISCOVERY_HOLDOUT_RESULTS.csv", index=False)

    replication, replication_audit = build_panel(REPLICATION_SYMBOLS, OUT / "replication_cache")
    replication_validation = evaluate(replication, selection, 2023, 2024, 2022, "replication_validation")
    replication_holdout = evaluate(replication, selection, 2025, 2026, 2024, "replication_holdout")
    replication_validation.to_csv(OUT / "REPLICATION_VALIDATION_RESULTS.csv", index=False)
    replication_holdout.to_csv(OUT / "REPLICATION_HOLDOUT_RESULTS.csv", index=False)

    confirmed: list[dict[str, Any]] = []
    for identity in FROZEN_IDENTITIES:
        factor = str(identity["factor"])
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
                "primitive_factor": identity["primitive_factor"],
                "horizon": identity["horizon"],
                "all_four_stage_gates": True,
                "minimum_ts_ic": float(min(record["ts_ic"] for record in records)),
                "minimum_residual_ts_ic": float(min(record["residual_ts_ic"] for record in records)),
                "minimum_ridge_mse_gain": float(min(record["ridge_mse_gain"] for record in records)),
                "minimum_annual_log_return_5bp": float(min(record["annual_log_return_5bp"] for record in records)),
            })

    summary = {
        "frozen_identities": int(len(selection)),
        "discovery_validation_passed": int(discovery_validation["gate"].sum()),
        "discovery_holdout_passed": int(discovery_holdout["gate"].sum()),
        "replication_validation_passed": int(replication_validation["gate"].sum()),
        "replication_holdout_passed": int(replication_holdout["gate"].sum()),
        "confirmed": confirmed,
        "decision": "eligible_for_trade_only_if_confirmed_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    write_json(OUT / "CONFIRMED.json", confirmed)
    write_json(OUT / "CANDIDATE_CONTRACTS.json", CONTRACTS)
    write_json(OUT / "DATA_AUDIT.json", {
        "seed": SEED,
        "discovery_symbols": DISCOVERY_SYMBOLS,
        "replication_symbols": REPLICATION_SYMBOLS,
        "discovery_data": discovery_audit,
        "replication_data": replication_audit,
        "research_design": "Round22 frozen physical coordinates transformed only by the previously frozen bounded execution map; no formula, sign, scale, threshold or horizon selected on Round23 stages",
        "execution": {"map": "deadzone(tanh(parent/1.25),0.35)", "cost_bp_one_way": 5.0},
        "confirmation": "requires discovery 2023-2024, discovery 2025-2026, disjoint replication 2023-2024 and disjoint replication 2025-2026 all to pass the same gate",
        "research_boundary": "historical; discovery validation motivated the readout diagnosis, therefore disjoint-universe replication is mandatory",
    })
    report = [
        "# Dimensionless operator readout research round23",
        "",
        "Round22 found monotone asset-internal information but negative linear MSE increments. Round23 freezes the exact bounded no-trade execution coordinate already declared in Round22 and tests whether dimensional scale, rather than information content, caused the reader mismatch.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen identities",
        selection.to_markdown(index=False),
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
