from __future__ import annotations

import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_baroclinic_torque as r  # noqa: E402

SEED = 20261119
OUT = Path("research_outputs/fluid_baroclinic_ensemble_round13")
OUT.mkdir(parents=True, exist_ok=True)
r.OUT = OUT

DEVELOPMENT_SCAN_PATH = Path(
    "research_outputs/fluid_baroclinic_torque_round13_fast/DEVELOPMENT_SCAN.csv"
)


def root_name(factor: str) -> str:
    return re.sub(r"_(?:z|ewm\d+)$", "", factor)


def signed_median(frame: pd.DataFrame, members: list[tuple[str, int]]) -> pd.Series:
    matrix = np.column_stack(
        [sign * pd.to_numeric(frame[factor], errors="coerce").to_numpy(float) for factor, sign in members]
    )
    with np.errstate(all="ignore"):
        values = np.nanmedian(matrix, axis=1)
    values[np.all(~np.isfinite(matrix), axis=1)] = np.nan
    return pd.Series(values, index=frame.index, dtype=float)


def finite_rows(frame: pd.DataFrame, target: str, factor: str) -> pd.DataFrame:
    work = frame.loc[
        np.isfinite(pd.to_numeric(frame[target], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame[factor], errors="coerce"))
    ].copy()
    return work


def impute_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    medians = train[features].median(axis=0, skipna=True)
    usable = [column for column in features if np.isfinite(medians[column])]
    if not usable:
        return np.empty((len(train), 0)), np.empty((len(test), 0)), []
    medians = medians[usable]
    x = train[usable].fillna(medians).to_numpy(float)
    xt = test[usable].fillna(medians).to_numpy(float)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > r.EPS, std, 1.0)
    return (x - mean) / std, (xt - mean) / std, usable


def ridge_increment_imputed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    factor: str,
    target: str,
) -> dict[str, float]:
    tr = finite_rows(train, target, factor)
    te = finite_rows(test, target, factor)
    if len(tr) > 300_000:
        tr = tr.iloc[np.linspace(0, len(tr) - 1, 300_000, dtype=int)]
    if len(tr) < 5_000 or len(te) < 1_000:
        return {
            "mse_gain": float("nan"),
            "base_rankic": float("nan"),
            "full_rankic": float("nan"),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "usable_parent_count": 0,
        }
    xb, xbt, usable = impute_train_test(tr, te, r.BASE_FEATURES)
    factor_median = float(tr[factor].median())
    xf_raw = np.column_stack([tr[usable].fillna(tr[usable].median()).to_numpy(float), tr[factor].fillna(factor_median).to_numpy(float)])
    xft_raw = np.column_stack([te[usable].fillna(tr[usable].median()).to_numpy(float), te[factor].fillna(factor_median).to_numpy(float)])
    mean = np.mean(xf_raw, axis=0)
    std = np.std(xf_raw, axis=0)
    std = np.where(std > r.EPS, std, 1.0)
    xf = (xf_raw - mean) / std
    xft = (xft_raw - mean) / std
    y = tr[target].to_numpy(float)
    yt = te[target].to_numpy(float)
    base = Ridge(alpha=20.0).fit(xb, y)
    full = Ridge(alpha=20.0).fit(xf, y)
    pb = base.predict(xbt)
    pf = full.predict(xft)
    mse_b = float(np.mean((yt - pb) ** 2))
    mse_f = float(np.mean((yt - pf) ** 2))
    pred = te[["time", "symbol"]].copy()
    pred["target"] = yt
    pred["base"] = pb
    pred["full"] = pf
    base_ic, _ = r.cross_sectional_ic(pred, "base", "target")
    full_ic, _ = r.cross_sectional_ic(pred, "full", "target")
    return {
        "mse_gain": float((mse_b - mse_f) / (mse_b + r.EPS)),
        "base_rankic": base_ic,
        "full_rankic": full_ic,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "usable_parent_count": int(len(usable)),
    }


def residual_imputed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    factor: str,
) -> pd.Series:
    result = pd.Series(np.nan, index=test.index, dtype=float)
    tr = train.loc[np.isfinite(pd.to_numeric(train[factor], errors="coerce"))].copy()
    te = test.loc[np.isfinite(pd.to_numeric(test[factor], errors="coerce"))].copy()
    if len(tr) > 300_000:
        tr = tr.iloc[np.linspace(0, len(tr) - 1, 300_000, dtype=int)]
    if len(tr) < 5_000 or len(te) < 1_000:
        return result
    x, xt, usable = impute_train_test(tr, te, r.BASE_FEATURES)
    if not usable:
        return result
    model = Ridge(alpha=20.0).fit(x, tr[factor].to_numpy(float))
    result.loc[te.index] = te[factor].to_numpy(float) - model.predict(xt)
    return result


def evaluate_imputed(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
    train_year_end: int,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    target = f"y_{horizon}"
    train = r.issue_subset(panel.loc[panel["time"].dt.year <= train_year_end].copy(), horizon)
    test = r.issue_subset(
        panel.loc[(panel["time"].dt.year >= start_year) & (panel["time"].dt.year <= end_year)].copy(),
        horizon,
    )
    cs, daily = r.cross_sectional_ic(test, factor, target)
    ts, ts_positive, ts_n = r.time_series_ic(test, factor, target)
    block = max(4, int(math.ceil(horizon / min(max(horizon, 4), 24))))
    boot = r.moving_block_bootstrap(daily, block=block, repetitions=5000)
    ridge = ridge_increment_imputed(train, test, factor, target)
    test["__residual"] = residual_imputed(train, test, factor)
    residual_cs, _ = r.cross_sectional_ic(test, "__residual", target)
    residual_ts, _, _ = r.time_series_ic(test, "__residual", target)
    max_corr, corr_feature = r.maximum_parent_correlation(test, factor)
    account2 = r.portfolio(test, factor, target, horizon, 2.0)
    account5 = r.portfolio(test, factor, target, horizon, 5.0)
    years = r.yearly_rankic(test, factor, target)
    return {
        "factor": factor,
        "horizon": int(horizon),
        "train_year_end": int(train_year_end),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "cs_rankic": cs,
        "ts_ic": ts,
        "ts_positive_share": ts_positive,
        "ts_products": ts_n,
        "bootstrap_low": boot["low"],
        "bootstrap_high": boot["high"],
        "bootstrap_p_nonpositive": boot["p_nonpositive"],
        "ridge_mse_gain": ridge["mse_gain"],
        "ridge_base_rankic": ridge["base_rankic"],
        "ridge_full_rankic": ridge["full_rankic"],
        "ridge_n_train": ridge["n_train"],
        "ridge_n_test": ridge["n_test"],
        "usable_parent_count": ridge["usable_parent_count"],
        "residual_cs_rankic": residual_cs,
        "residual_ts_ic": residual_ts,
        "max_parent_corr": max_corr,
        "max_parent_feature": corr_feature,
        "annual_log_return_2bp": account2["annual_log_return"],
        "sharpe_2bp": account2["sharpe"],
        "annual_log_return_5bp": account5["annual_log_return"],
        "sharpe_5bp": account5["sharpe"],
        "avg_turnover": account5["avg_turnover"],
        "portfolio_n": account5["n"],
        "yearly_rankic": json.dumps(years, sort_keys=True),
        "yearly_5bp": json.dumps(account5["years"], sort_keys=True),
    }


def development_consensus_score(panel: pd.DataFrame, factor: str, horizon: int) -> dict[str, Any]:
    target = f"y_{horizon}"
    dev = r.issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon)
    cs, _ = r.cross_sectional_ic(dev, factor, target)
    ts, _, _ = r.time_series_ic(dev, factor, target)
    years = r.yearly_rankic(dev, factor, target)
    finite_years = [float(value) for value in years.values() if np.isfinite(value)]
    min_year = min(finite_years) if finite_years else float("nan")
    account = r.portfolio(dev, factor, target, horizon, 2.0)
    score = (
        (min_year if np.isfinite(min_year) else -1.0)
        + 0.30 * (cs if np.isfinite(cs) else -1.0)
        + 0.10 * (ts if np.isfinite(ts) else -1.0)
        + 0.01 * np.clip(account["sharpe"], -2.0, 2.0)
    )
    return {
        "factor": factor,
        "horizon": int(horizon),
        "development_cs_rankic": cs,
        "development_ts_ic": ts,
        "development_min_year_rankic": min_year,
        "development_2bp_return": account["annual_log_return"],
        "development_2bp_sharpe": account["sharpe"],
        "development_years": json.dumps(years, sort_keys=True),
        "development_score": float(score),
    }


def choose_primitive_members(scan: pd.DataFrame, family: str, horizon: int) -> list[tuple[str, int]]:
    subset = scan.loc[(scan["family"] == family) & (scan["horizon"] == horizon)].copy()
    subset["root"] = subset["factor"].map(root_name)
    subset = subset.sort_values("development_score", ascending=False)
    winners = subset.groupby("root", as_index=False).head(1)
    positive = winners.loc[winners["development_min_year_rankic"] > 0.0]
    if len(positive) >= 2:
        winners = positive
    else:
        winners = winners.head(min(3, len(winners)))
    return [(str(row.factor), int(row.sign)) for row in winners.itertuples(index=False)]


def build_ensembles(panel: pd.DataFrame, scan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[tuple[str, int]]]]:
    selection_rows: list[dict[str, Any]] = []
    member_map: dict[str, list[tuple[str, int]]] = {}
    for family in r.FAMILIES:
        for horizon in r.HORIZONS:
            members = choose_primitive_members(scan, family, horizon)
            name = f"ensemble_{family}_h{horizon}"
            panel[name] = signed_median(panel, members)
            member_map[name] = members
            row = development_consensus_score(panel, name, horizon)
            row["family"] = family
            row["member_count"] = len(members)
            row["members"] = json.dumps(members)
            selection_rows.append(row)
    family_grid = pd.DataFrame(selection_rows)
    family_winners = (
        family_grid.sort_values(["family", "development_score"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    winner_names = dict(zip(family_winners["family"], family_winners["factor"]))
    ranked_families = family_winners.sort_values("development_score", ascending=False)["family"].tolist()
    composite_specs = {
        "ensemble_cross_family_top4": ranked_families[:4],
        "ensemble_torque_pv_source": [
            family for family in ["baroclinic_torque", "potential_vorticity", "vortex_stretching", "source_coupling"]
            if family in winner_names
        ],
        "ensemble_circulation_reattachment": [
            family for family in ["kelvin_circulation", "ekman_impedance", "baroclinic_reattachment"]
            if family in winner_names
        ],
    }
    composite_rows: list[dict[str, Any]] = []
    for composite, families in composite_specs.items():
        member_names = [(winner_names[family], 1) for family in families]
        panel[composite] = signed_median(panel, member_names)
        member_map[composite] = member_names
        for horizon in r.HORIZONS:
            row = development_consensus_score(panel, composite, horizon)
            row["family"] = composite
            row["member_count"] = len(member_names)
            row["members"] = json.dumps(member_names)
            composite_rows.append(row)
    composite_grid = pd.DataFrame(composite_rows)
    composite_winners = (
        composite_grid.sort_values(["family", "development_score"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    selection = pd.concat([family_winners, composite_winners], ignore_index=True)
    grids = pd.concat([family_grid, composite_grid], ignore_index=True)
    return selection, grids, member_map


def pairwise_sequential(panel: pd.DataFrame, promoted: pd.DataFrame) -> pd.DataFrame:
    if len(promoted) <= 1:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    factors = promoted["factor"].tolist()
    for factor in factors:
        others = [name for name in factors if name != factor]
        horizon = int(promoted.loc[promoted["factor"] == factor, "horizon"].iloc[0])
        target = f"y_{horizon}"
        train = r.issue_subset(panel.loc[panel["time"].dt.year <= 2024].copy(), horizon)
        test = r.issue_subset(panel.loc[panel["time"].dt.year >= 2025].copy(), horizon)
        tr = finite_rows(train, target, factor)
        te = finite_rows(test, target, factor)
        extra = r.BASE_FEATURES + others
        x, xt, usable = impute_train_test(tr, te, extra)
        xf_raw = np.column_stack([tr[usable].fillna(tr[usable].median()).to_numpy(float), tr[factor].to_numpy(float)])
        xft_raw = np.column_stack([te[usable].fillna(tr[usable].median()).to_numpy(float), te[factor].to_numpy(float)])
        mean = np.mean(xf_raw, axis=0)
        std = np.where(np.std(xf_raw, axis=0) > r.EPS, np.std(xf_raw, axis=0), 1.0)
        xf = (xf_raw - mean) / std
        xft = (xft_raw - mean) / std
        y = tr[target].to_numpy(float)
        yt = te[target].to_numpy(float)
        base = Ridge(alpha=20.0).fit(x, y)
        full = Ridge(alpha=20.0).fit(xf, y)
        mse_b = float(np.mean((yt - base.predict(xt)) ** 2))
        mse_f = float(np.mean((yt - full.predict(xft)) ** 2))
        rows.append({
            "factor": factor,
            "conditional_on": json.dumps(others),
            "horizon": horizon,
            "sequential_mse_gain": float((mse_b - mse_f) / (mse_b + r.EPS)),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    if not DEVELOPMENT_SCAN_PATH.exists():
        raise FileNotFoundError(DEVELOPMENT_SCAN_PATH)
    scan = pd.read_csv(DEVELOPMENT_SCAN_PATH)

    klines, kline_audit = r.load_klines()
    metrics, funding, external_audit = r.load_external()
    merged = r.merge_external(klines, metrics, funding)
    local_parts: list[pd.DataFrame] = []
    for symbol, group in merged.groupby("symbol", sort=False):
        local = r.compute_local_features(group)
        local["symbol"] = symbol
        local_parts.append(local)
        print(f"local_features={symbol} rows={len(local)}", flush=True)
    local_panel = pd.concat(local_parts, ignore_index=True)
    weights, directed, graph_audit = r.build_graph(local_panel)
    graph_panel = r.graph_state(local_panel, weights, directed)
    panel = local_panel.merge(graph_panel, on=["time", "symbol"], how="left", validate="one_to_one")
    panel = r.add_candidate_states(panel)

    selection, grids, member_map = build_ensembles(panel, scan)
    grids.to_csv(OUT / "ENSEMBLE_DEVELOPMENT_GRID.csv", index=False)
    selection.to_csv(OUT / "ENSEMBLE_FROZEN_SELECTION.csv", index=False)
    r.write_json(OUT / "ENSEMBLE_MEMBERS.json", member_map)

    validation_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        result = evaluate_imputed(panel, row.factor, int(row.horizon), 2022, 2023, 2024)
        result["family"] = row.family
        result["members"] = row.members
        validation_rows.append(result)
        print(f"validation={row.factor} horizon={row.horizon}", flush=True)
    validation = pd.DataFrame(validation_rows)
    raw_p = [float(value) if np.isfinite(value) else 1.0 for value in validation["bootstrap_p_nonpositive"]]
    validation["holm_p"] = r.holm_adjust(raw_p)
    validation["validation_gate"] = (
        (validation["cs_rankic"] > 0.010)
        & (validation["bootstrap_p_nonpositive"] <= 0.10)
        & (validation["ridge_mse_gain"] > 0.0)
        & (validation["residual_cs_rankic"] > 0.005)
        & (validation["max_parent_corr"] < 0.85)
        & (validation["annual_log_return_5bp"] > 0.0)
        & (validation["holm_p"] <= 0.10)
    )
    validation.to_csv(OUT / "ENSEMBLE_VALIDATION.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    opened = validation.loc[validation["validation_gate"]].copy()
    for row in opened.itertuples(index=False):
        result = evaluate_imputed(panel, row.factor, int(row.horizon), 2024, 2025, 2026)
        result["family"] = row.family
        result["validation_holm_p"] = row.holm_p
        result["members"] = row.members
        holdout_rows.append(result)
        print(f"holdout={row.factor} horizon={row.horizon}", flush=True)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=validation.columns).to_csv(OUT / "ENSEMBLE_HOLDOUT.csv", index=False)
    else:
        holdout.to_csv(OUT / "ENSEMBLE_HOLDOUT.csv", index=False)

    promoted_rows: list[dict[str, Any]] = []
    for row in holdout.itertuples(index=False):
        years = json.loads(row.yearly_rankic)
        years_positive = bool(years) and all(np.isfinite(value) and float(value) > 0.0 for value in years.values())
        passed = (
            row.cs_rankic > 0.010
            and row.bootstrap_p_nonpositive <= 0.10
            and row.ridge_mse_gain > 0.0
            and row.residual_cs_rankic > 0.005
            and row.max_parent_corr < 0.85
            and row.annual_log_return_5bp > 0.0
            and years_positive
        )
        if passed:
            promoted_rows.append({
                "factor": row.factor,
                "family": row.family,
                "horizon": int(row.horizon),
                "validation_holm_p": float(row.validation_holm_p),
                "holdout_cs_rankic": float(row.cs_rankic),
                "holdout_ridge_mse_gain": float(row.ridge_mse_gain),
                "holdout_residual_cs_rankic": float(row.residual_cs_rankic),
                "holdout_annual_log_return_5bp": float(row.annual_log_return_5bp),
                "holdout_sharpe_5bp": float(row.sharpe_5bp),
                "holdout_max_parent_corr": float(row.max_parent_corr),
                "members": json.loads(row.members),
            })
    promoted = pd.DataFrame(promoted_rows)
    r.write_json(OUT / "PROMOTED.json", promoted_rows)
    sequential = pairwise_sequential(panel, promoted) if not promoted.empty else pd.DataFrame()
    sequential.to_csv(OUT / "SEQUENTIAL_INDEPENDENCE.csv", index=False)

    summary = {
        "primitive_scan_rows": int(len(scan)),
        "family_ensembles": int((selection["family"].isin(r.FAMILIES)).sum()),
        "composite_ensembles": int((~selection["family"].isin(r.FAMILIES)).sum()),
        "validation_candidates": int(len(validation)),
        "validation_gate_passed": int(validation["validation_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "promoted": promoted_rows,
        "strong_parent_policy": "training-median imputation plus training standardization",
        "decision": "promote_only_if_PROMOTED_nonempty_and_sequential_gain_positive_when_multiple",
    }
    r.write_json(OUT / "SUMMARY.json", summary)
    r.write_json(
        OUT / "DATA_AUDIT.json",
        {
            "klines": kline_audit,
            "external": external_audit,
            "graph": graph_audit,
            "merged_rows": int(len(merged)),
            "panel_rows": int(len(panel)),
            "availability_policy": {
                "klines": "completed hour; label starts at next-hour open",
                "metrics": "as-of joined and shifted one complete hour",
                "funding": "last published event as-of joined and shifted one complete hour",
            },
        },
    )
    (OUT / "REPORT.md").write_text(
        "# Baroclinic Ensemble Round 13\n\n"
        + "## Frozen selection\n\n"
        + selection.to_markdown(index=False)
        + "\n\n## Validation\n\n"
        + validation.to_markdown(index=False)
        + "\n\n## Holdout\n\n"
        + (holdout.to_markdown(index=False) if not holdout.empty else "No candidate opened holdout.")
        + "\n\n## Promotion\n\n```json\n"
        + json.dumps(promoted_rows, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        r.write_json(OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
