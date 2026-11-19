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
ROUND14 = HERE.parent / "fluid_round14"
ROUND13 = HERE.parent / "fluid_round13"
for path in (ROUND14, ROUND13):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_characteristic_wave_action as c  # noqa: E402
import run_baroclinic_ensemble as e  # noqa: E402
import run_baroclinic_torque as r  # noqa: E402

SEED = 20261119
OUT = Path("research_outputs/fluid_characteristic_wave_crossuniverse_round15")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(SEED)

REPLICATION_SYMBOLS = [
    "TRXUSDT", "EOSUSDT", "ATOMUSDT", "XTZUSDT", "ZECUSDT", "DASHUSDT",
    "FILUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "AVAXUSDT", "COMPUSDT",
]
FROZEN = [
    {"identity": "riemann_transmission_defect_ewm6", "horizon": 480, "orientation": 1},
    {"identity": "impedance_pressure_release_ewm24", "horizon": 72, "orientation": 1},
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def yearly_positive(payload: str) -> bool:
    values = json.loads(payload)
    finite = [float(v) for v in values.values() if np.isfinite(float(v))]
    return bool(finite) and min(finite) > 0.0


def validation_gate(row: pd.Series, holm_p: float) -> bool:
    return bool(
        row["cs_rankic"] > 0.012
        and row["bootstrap_p_nonpositive"] <= 0.10
        and row["ridge_mse_gain"] > 0.0
        and row["residual_cs_rankic"] > 0.005
        and row["max_parent_corr"] < 0.80
        and row["annual_log_return_5bp"] > 0.0
        and holm_p <= 0.10
        and yearly_positive(row["yearly_rankic"])
    )


def main() -> None:
    np.random.seed(SEED)
    r.SYMBOLS = REPLICATION_SYMBOLS
    r.MONTHS = list(pd.period_range("2021-01", "2026-09", freq="M"))
    r.OUT = OUT
    c.OUT = OUT

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
    panel = c.add_local_characteristics(panel)
    characteristic_panel, characteristic_audit = c.build_characteristic_features(panel, weights)
    panel = panel.merge(characteristic_panel, on=["time", "symbol"], how="left", validate="one_to_one")
    panel, _ = c.add_candidate_states(panel)

    extra_parents = [name for name in c.PRIOR_FLUID_PARENTS if name in panel.columns]
    r.BASE_FEATURES = list(dict.fromkeys(r.BASE_FEATURES + extra_parents))

    missing = [item["identity"] for item in FROZEN if item["identity"] not in panel.columns]
    if missing:
        raise RuntimeError(f"frozen identities missing: {missing}")

    validation_rows: list[dict[str, Any]] = []
    for item in FROZEN:
        identity = item["identity"]
        oriented = f"oriented_{identity}"
        panel[oriented] = int(item["orientation"]) * panel[identity]
        result = e.evaluate_imputed(panel, oriented, int(item["horizon"]), 2022, 2023, 2024)
        result["identity"] = identity
        result["orientation"] = int(item["orientation"])
        validation_rows.append(result)
    validation = pd.DataFrame(validation_rows)
    validation["holm_p"] = r.holm_adjust([
        float(v) if np.isfinite(v) else 1.0 for v in validation["bootstrap_p_nonpositive"]
    ])
    validation["validation_gate"] = [
        validation_gate(row, float(row["holm_p"])) for _, row in validation.iterrows()
    ]
    validation.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    for _, row in validation.loc[validation["validation_gate"]].iterrows():
        result = e.evaluate_imputed(
            panel, str(row["factor"]), int(row["horizon"]), 2024, 2025, 2026
        )
        result["identity"] = row["identity"]
        result["orientation"] = int(row["orientation"])
        result["validation_holm_p"] = float(row["holm_p"])
        holdout_rows.append(result)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=list(validation.columns)).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    else:
        holdout.to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)

    promoted: list[dict[str, Any]] = []
    for _, row in holdout.iterrows():
        passed = bool(
            row["cs_rankic"] > 0.012
            and row["bootstrap_p_nonpositive"] <= 0.10
            and row["ridge_mse_gain"] > 0.0
            and row["residual_cs_rankic"] > 0.005
            and row["max_parent_corr"] < 0.80
            and row["annual_log_return_5bp"] > 0.0
            and yearly_positive(row["yearly_rankic"])
        )
        if passed:
            promoted.append({
                "identity": row["identity"],
                "orientation": int(row["orientation"]),
                "horizon": int(row["horizon"]),
                "validation_holm_p": float(row["validation_holm_p"]),
                "holdout_cs_rankic": float(row["cs_rankic"]),
                "holdout_ts_ic": float(row["ts_ic"]),
                "holdout_ridge_mse_gain": float(row["ridge_mse_gain"]),
                "holdout_residual_cs_rankic": float(row["residual_cs_rankic"]),
                "holdout_max_parent_corr": float(row["max_parent_corr"]),
                "holdout_annual_log_return_5bp": float(row["annual_log_return_5bp"]),
            })

    audit = {
        "symbols": REPLICATION_SYMBOLS,
        "klines": kline_audit,
        "external": external_audit,
        "graph": graph_audit,
        "characteristic": characteristic_audit,
        "panel_rows": int(len(panel)),
        "strong_parent_count": len(r.BASE_FEATURES),
        "prior_fluid_parents": extra_parents,
        "frozen_identities": FROZEN,
        "selection_policy": "formula, orientation and horizon frozen from round14 before replication universe was read",
    }
    write_json(OUT / "DATA_AUDIT.json", audit)
    write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "replication_symbols": len(REPLICATION_SYMBOLS),
        "frozen_hypotheses": len(FROZEN),
        "validation_passed": int(validation["validation_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "promoted": promoted,
        "decision": "promote_only_if_both_replication_stages_pass",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = [
        "# Frozen characteristic-wave cross-universe replication round15",
        "",
        "No formula, sign or horizon was reselected in this universe.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
        "",
        "## Historical holdout",
        holdout.to_markdown(index=False) if not holdout.empty else "Holdout remained closed.",
        "",
        "## Promotion",
        "```json",
        json.dumps(promoted, indent=2, sort_keys=True),
        "```",
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
