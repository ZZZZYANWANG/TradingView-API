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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_baroclinic_torque as r  # noqa: E402
import run_baroclinic_ensemble as e  # noqa: E402
import run_baroclinic_torque_fast as fast  # noqa: E402

SEED = 20261119
OUT = Path("research_outputs/fluid_baroclinic_memory_round13")
OUT.mkdir(parents=True, exist_ok=True)
r.OUT = OUT
SPANS = [24, 72, 168, 336, 480]

PRIMITIVE_PARENTS = [
    "baroclinic_funding_z", "baroclinic_flow_z", "baroclinic_total_z",
    "pv_transport_z", "pv_source_z", "pv_pressure_release_z",
    "vortex_stretch_z", "enstrophy_release_z", "kelvin_residual_z",
    "torque_circulation_alignment_z", "baroclinic_reattach_z",
    "source_supported_torque_z", "deleveraging_torque_release_z",
]


def event_age(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).to_numpy(bool)
    out = np.zeros(len(values), dtype=float)
    age = 0.0
    for i, value in enumerate(values):
        age = age + 1.0 if value else 0.0
        out[i] = age
    return pd.Series(out, index=mask.index, dtype=float)


def add_memory_candidates(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    families: dict[str, list[str]] = {
        "torque_reservoir": [],
        "torque_relaxation": [],
        "potential_vorticity_budget": [],
        "circulation_memory": [],
        "reattachment_event": [],
        "source_pressure_action": [],
        "generation_storage_release": [],
    }
    parts: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        torque = g["baroclinic_total_z"]
        funding_torque = g["baroclinic_funding_z"]
        flow_torque = g["baroclinic_flow_z"]
        pv_budget = g["pv_transport_z"] + g["pv_source_z"] + g["pv_pressure_release_z"]
        pv_closure = g["pv_source_z"] + g["pv_pressure_release_z"] - g["pv_transport_z"]
        kelvin = g["kelvin_residual_z"]
        circulation_action = g["torque_circulation_alignment_z"]
        source_action = torque * g["oi_source_z"]
        funding_action = funding_torque * g["funding_change_z"]
        flow_action = flow_torque * g["metric_taker_z"]
        deleveraging = g["deleveraging_torque_release_z"]
        reattach = g["baroclinic_reattach_z"]
        separation = torque.abs() > 1.0
        age = event_age(separation)
        contraction = (torque.abs().shift(1) - torque.abs()).clip(lower=0.0)
        reattach_impulse = np.sign(g["mom24"]) * age.shift(1) * contraction * np.exp(-g["div_u"].abs())

        component_cache: dict[tuple[str, int], str] = {}
        for span in SPANS:
            def register(family: str, stem: str, raw: pd.Series) -> str:
                name = f"{stem}_{span}"
                state = raw.ewm(span=span, min_periods=max(12, span // 2), adjust=False).mean()
                g[name] = r.causal_z(state, 720, 240)
                if name not in families[family]:
                    families[family].append(name)
                component_cache[(stem, span)] = name
                return name

            torque_name = register("torque_reservoir", "torque_reservoir", torque)
            register("torque_reservoir", "funding_torque_reservoir", funding_torque)
            register("torque_reservoir", "flow_torque_reservoir", flow_torque)

            torque_res = torque.ewm(span=span, min_periods=max(12, span // 2), adjust=False).mean()
            relax = np.sign(g["mom24"]) * (torque_res.abs().shift(1) - torque_res.abs()).clip(lower=0.0)
            reversal = -np.sign(torque_res.shift(1)) * torque_res.diff()
            register("torque_relaxation", "torque_relaxation", relax)
            register("torque_relaxation", "torque_reversal", reversal)

            pv_name = register("potential_vorticity_budget", "pv_budget_reservoir", pv_budget)
            register("potential_vorticity_budget", "pv_closure_reservoir", pv_closure)

            kelvin_name = register("circulation_memory", "kelvin_memory", kelvin)
            register("circulation_memory", "circulation_action_memory", circulation_action)
            register("circulation_memory", "kelvin_action_gap", kelvin - circulation_action)

            reattach_name = register("reattachment_event", "reattachment_memory", reattach)
            register("reattachment_event", "reattachment_impulse", reattach_impulse)
            register("reattachment_event", "separation_age_signed", np.sign(g["mom24"]) * age)

            source_name = register("source_pressure_action", "oi_source_torque_action", source_action)
            register("source_pressure_action", "funding_torque_action", funding_action)
            register("source_pressure_action", "flow_torque_action", flow_action)
            register("source_pressure_action", "deleveraging_release_memory", deleveraging)

            if span in {72, 168, 336}:
                matrix = np.column_stack([
                    g[torque_name].to_numpy(float),
                    g[pv_name].to_numpy(float),
                    g[kelvin_name].to_numpy(float),
                    g[reattach_name].to_numpy(float),
                    g[source_name].to_numpy(float),
                ])
                with np.errstate(all="ignore"):
                    composite = np.nanmedian(matrix, axis=1)
                composite[np.all(~np.isfinite(matrix), axis=1)] = np.nan
                name = f"generation_storage_release_{span}"
                g[name] = pd.Series(composite, index=g.index)
                if name not in families["generation_storage_release"]:
                    families["generation_storage_release"].append(name)
        parts.append(g)
    result = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    return result, families


def batch_selection(panel: pd.DataFrame, candidate_map: dict[str, list[str]]) -> pd.DataFrame:
    reverse = {factor: family for family, factors in candidate_map.items() for factor in factors}
    factors = list(reverse)
    pieces: list[pd.DataFrame] = []
    original_horizons = list(r.HORIZONS)
    for horizon in original_horizons:
        target = f"y_{horizon}"
        dev = r.issue_subset(panel.loc[panel["time"].dt.year <= 2024].copy(), horizon)
        dev = dev.loc[np.isfinite(pd.to_numeric(dev[target], errors="coerce"))].copy()
        symbols = list(r.SYMBOLS)
        symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
        m = len(factors)
        ic_times: list[pd.Timestamp] = []
        ic_values: list[np.ndarray] = []
        gross_rows: list[np.ndarray] = []
        turnover_rows: list[np.ndarray] = []
        previous = np.zeros((len(symbols), m), dtype=float)
        for timestamp, group in dev.groupby("time", sort=True):
            work = group[["symbol", target] + factors].replace([np.inf, -np.inf], np.nan)
            y = work[target].to_numpy(float)
            x_rank = fast._rank_matrix(work, factors)
            y_rank = work[target].rank(method="average").to_numpy(float)
            ic_values.append(fast._columnwise_corr(x_rank, y_rank, minimum=8))
            ic_times.append(pd.Timestamp(timestamp))
            x = work[factors].to_numpy(float)
            gross = np.full(m, np.nan)
            turnover = np.full(m, np.nan)
            locations = np.asarray([symbol_index[str(s)] for s in work["symbol"]], dtype=int)
            for j in range(m):
                valid = np.isfinite(x[:, j]) & np.isfinite(y)
                idx = np.flatnonzero(valid)
                if len(idx) < 8:
                    continue
                count = max(2, int(math.floor(len(idx) * 0.30)))
                order = idx[np.argsort(x[idx, j], kind="mergesort")]
                low, high = order[:count], order[-count:]
                weights = np.zeros(len(symbols), dtype=float)
                weights[locations[low]] -= 0.5 / count
                weights[locations[high]] += 0.5 / count
                target_map = np.zeros(len(symbols), dtype=float)
                target_map[locations[idx]] = y[idx]
                gross[j] = weights @ target_map
                turnover[j] = np.abs(weights - previous[:, j]).sum()
                previous[:, j] = weights
            gross_rows.append(gross)
            turnover_rows.append(turnover)
        ic = pd.DataFrame(np.vstack(ic_values), index=pd.DatetimeIndex(ic_times), columns=factors)
        mean_ic = ic.mean()
        annual = ic.groupby(ic.index.year).mean()
        ts_rows = []
        for _, group in dev.groupby("symbol", sort=False):
            work = group[[target] + factors].replace([np.inf, -np.inf], np.nan)
            ts_rows.append(fast._columnwise_corr(
                fast._rank_matrix(work, factors),
                work[target].rank(method="average").to_numpy(float),
                minimum=80,
            ))
        ts = np.nanmean(np.vstack(ts_rows), axis=0)
        gross_matrix = np.vstack(gross_rows)
        turnover_matrix = np.vstack(turnover_rows)
        annual_periods = 365.25 * 24.0 / horizon
        rows = []
        for j, factor in enumerate(factors):
            raw_years = {str(int(year)): float(value) for year, value in annual[factor].dropna().items()}
            raw_metric = 0.75 * float(mean_ic[factor]) + 0.25 * float(ts[j])
            sign = 1 if raw_metric >= 0 else -1
            signed_years = {key: sign * value for key, value in raw_years.items()}
            min_year = min(signed_years.values()) if signed_years else float("nan")
            net = sign * gross_matrix[:, j] - 2.0e-4 * turnover_matrix[:, j]
            finite = np.isfinite(net)
            if finite.sum() >= 20:
                ann_return = float(np.mean(net[finite]) * annual_periods)
                sharpe = float(np.mean(net[finite]) / (np.std(net[finite]) + r.EPS) * np.sqrt(annual_periods))
            else:
                ann_return = sharpe = float("nan")
            signed_cs = sign * float(mean_ic[factor])
            signed_ts = sign * float(ts[j])
            score = min_year + 0.30 * signed_cs + 0.10 * signed_ts + 0.01 * np.clip(sharpe, -2.0, 2.0)
            rows.append({
                "family": reverse[factor], "factor": factor, "horizon": horizon, "sign": sign,
                "selection_cs_rankic": signed_cs, "selection_ts_ic": signed_ts,
                "selection_min_year_rankic": min_year, "selection_2bp_return": ann_return,
                "selection_2bp_sharpe": sharpe, "selection_years": json.dumps(signed_years, sort_keys=True),
                "selection_score": float(score),
            })
        pieces.append(pd.DataFrame(rows))
        print(f"selection_horizon={horizon} candidates={len(rows)}", flush=True)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    np.random.seed(SEED)
    klines, kline_audit = r.load_klines()
    metrics, funding, external_audit = r.load_external()
    merged = r.merge_external(klines, metrics, funding)
    local_parts = []
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
    panel, candidate_map = add_memory_candidates(panel)

    for parent in PRIMITIVE_PARENTS:
        if parent not in r.BASE_FEATURES:
            r.BASE_FEATURES.append(parent)

    scan = batch_selection(panel, candidate_map)
    scan.to_csv(OUT / "SELECTION_SCAN_2021_2024.csv", index=False)
    winners = (
        scan.sort_values(["family", "selection_score"], ascending=[True, False])
        .groupby("family", as_index=False).head(1).reset_index(drop=True)
    )
    winners.to_csv(OUT / "FROZEN_FAMILY_WINNERS.csv", index=False)

    holdout_rows = []
    for row in winners.itertuples(index=False):
        panel["__factor"] = int(row.sign) * panel[row.factor]
        result = e.evaluate_imputed(panel, "__factor", int(row.horizon), 2024, 2025, 2026)
        result["family"] = row.family
        result["canonical_source"] = row.factor
        result["sign"] = int(row.sign)
        holdout_rows.append(result)
        print(f"holdout={row.family} factor={row.factor} horizon={row.horizon}", flush=True)
    holdout = pd.DataFrame(holdout_rows)
    holdout["holm_p"] = r.holm_adjust([
        float(value) if np.isfinite(value) else 1.0
        for value in holdout["bootstrap_p_nonpositive"]
    ])
    holdout.to_csv(OUT / "FINAL_HOLDOUT_RESULTS.csv", index=False)

    promoted = []
    for row in holdout.itertuples(index=False):
        years = json.loads(row.yearly_rankic)
        year_pass = bool(years) and all(np.isfinite(value) and float(value) > 0 for value in years.values())
        passed = (
            row.cs_rankic > 0.010
            and row.bootstrap_p_nonpositive <= 0.10
            and row.holm_p <= 0.10
            and row.ridge_mse_gain > 0.0
            and row.residual_cs_rankic > 0.005
            and row.max_parent_corr < 0.85
            and row.annual_log_return_5bp > 0.0
            and year_pass
        )
        if passed:
            promoted.append({
                "family": row.family,
                "canonical_source": row.canonical_source,
                "sign": int(row.sign),
                "horizon": int(row.horizon),
                "holdout_cs_rankic": float(row.cs_rankic),
                "holdout_bootstrap_low": float(row.bootstrap_low),
                "holdout_holm_p": float(row.holm_p),
                "holdout_ridge_mse_gain": float(row.ridge_mse_gain),
                "holdout_residual_cs_rankic": float(row.residual_cs_rankic),
                "holdout_max_parent_corr": float(row.max_parent_corr),
                "holdout_annual_log_return_5bp": float(row.annual_log_return_5bp),
                "holdout_sharpe_5bp": float(row.sharpe_5bp),
            })
    r.write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "selection_period": "2021-2024",
        "final_holdout": "2025-2026-09",
        "candidate_families": len(candidate_map),
        "candidate_columns": sum(len(values) for values in candidate_map.values()),
        "selection_tests": int(len(scan)),
        "holdout_winners": int(len(holdout)),
        "promoted": promoted,
        "strong_parent_count": len(r.BASE_FEATURES),
        "decision": "promote_only_if_PROMOTED_nonempty",
    }
    r.write_json(OUT / "SUMMARY.json", summary)
    r.write_json(OUT / "DATA_AUDIT.json", {
        "klines": kline_audit, "external": external_audit, "graph": graph_audit,
        "merged_rows": int(len(merged)), "panel_rows": int(len(panel)),
        "availability": "metrics and funding shifted one complete hour; labels begin at next-hour open",
    })
    (OUT / "REPORT.md").write_text(
        "# Baroclinic Memory Round 13\n\n## Frozen winners\n\n"
        + winners.to_markdown(index=False)
        + "\n\n## Final holdout\n\n"
        + holdout.to_markdown(index=False)
        + "\n\n## Promotion\n\n```json\n"
        + json.dumps(promoted, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        r.write_json(OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
