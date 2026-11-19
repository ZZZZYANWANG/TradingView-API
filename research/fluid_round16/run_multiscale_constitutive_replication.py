from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_multiscale_constitutive as r16  # noqa: E402

base = r16.base
SEED = 20261119
OUT = Path("research_outputs/fluid_multiscale_constitutive_round16_replication")
OUT.mkdir(parents=True, exist_ok=True)

REPLICATION_SYMBOLS = [
    "TRXUSDT", "EOSUSDT", "ATOMUSDT", "XTZUSDT", "ZECUSDT", "DASHUSDT",
    "FILUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "AVAXUSDT", "COMPUSDT",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def years_positive(payload: str) -> bool:
    values = json.loads(payload)
    finite = [float(value) for value in values.values() if np.isfinite(float(value))]
    return bool(finite) and min(finite) > 0.0


def gate(row: pd.Series, holm_p: float | None = None) -> bool:
    result = bool(
        row["cs_rankic"] > 0.012
        and row["bootstrap_p_nonpositive"] <= 0.10
        and row["ridge_mse_gain"] > 0.0
        and row["residual_cs_rankic"] > 0.005
        and row["max_parent_corr"] < 0.80
        and row["annual_log_return_5bp"] > 0.0
        and years_positive(row["yearly_rankic"])
    )
    return result and (holm_p is None or holm_p <= 0.10)


def rebuild_frozen(panel: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    raw_selection = selection.loc[selection["family"].isin(r16.FAMILY_RAW)].copy()
    for row in raw_selection.itertuples(index=False):
        panel[str(row.signed_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")

    best = raw_selection.sort_values("development_score", ascending=False).groupby("family", as_index=False).head(1)
    family_signed = {str(row.family): str(row.signed_factor) for row in best.itertuples(index=False)}
    specs = {
        "conservation_closure_consensus": [name for name in ["favre_momentum_budget", "kelvin_circulation"] if name in family_signed],
        "cascade_constitutive_consensus": [name for name in ["onsager_cascade", "nonnewtonian_constitutive", "scale_locality"] if name in family_signed],
        "round16_all_family_consensus": list(family_signed),
    }
    for name, families in specs.items():
        rows = selection.loc[selection["family"] == name]
        if rows.empty:
            continue
        members = [family_signed[family] for family in families]
        ranks = [panel.groupby("time")[member].rank(pct=True) for member in members]
        panel[name] = pd.concat(ranks, axis=1).mean(axis=1)
        row = rows.iloc[0]
        panel[str(row["signed_factor"])] = int(row["sign"]) * panel[name]
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--promoted", required=True)
    args = parser.parse_args()

    selection = pd.read_csv(args.selection)
    promoted = json.loads(Path(args.promoted).read_text(encoding="utf-8"))
    if not promoted:
        summary = {
            "replication_symbols": REPLICATION_SYMBOLS,
            "frozen_hypotheses": 0,
            "validation_passed": 0,
            "holdout_opened": 0,
            "confirmed": [],
            "decision": "upstream_promoted_empty_replication_skipped",
        }
        write_json(OUT / "SUMMARY.json", summary)
        write_json(OUT / "CONFIRMED.json", [])
        print(json.dumps(summary, indent=2), flush=True)
        return

    np.random.seed(SEED)
    base.SYMBOLS = REPLICATION_SYMBOLS
    base.MONTHS = list(pd.period_range("2021-01", "2026-09", freq="M"))
    base.OUT = OUT
    raw, data_audit = base.load_public_data()
    parts: list[pd.DataFrame] = []
    for symbol, group in raw.groupby("symbol", sort=False):
        local = base.local_features(group)
        enriched, _ = r16.add_mechanisms(local)
        parts.append(enriched)
        print(f"replication_mechanisms={symbol} rows={len(enriched)}", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    panel = rebuild_frozen(panel, selection)

    validation_rows: list[dict[str, Any]] = []
    for item in promoted:
        factor = str(item["signed_factor"])
        result = base.evaluate(panel, factor, int(item["horizon"]), 2022, 2023, 2024)
        result.update({
            "family": item["family"],
            "primitive_factor": item["primitive_factor"],
            "original_validation_holm_p": item["validation_holm_p"],
        })
        validation_rows.append(result)
    validation = pd.DataFrame(validation_rows)
    validation["holm_p"] = base.holm_adjust([
        float(value) if np.isfinite(value) else 1.0 for value in validation["bootstrap_p_nonpositive"]
    ])
    validation["replication_gate"] = [gate(row, float(row["holm_p"])) for _, row in validation.iterrows()]
    validation.to_csv(OUT / "REPLICATION_VALIDATION.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    for _, row in validation.loc[validation["replication_gate"]].iterrows():
        result = base.evaluate(panel, str(row["factor"]), int(row["horizon"]), 2024, 2025, 2026)
        result.update({
            "family": row["family"],
            "primitive_factor": row["primitive_factor"],
            "replication_validation_holm_p": float(row["holm_p"]),
        })
        holdout_rows.append(result)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=validation.columns).to_csv(OUT / "REPLICATION_HOLDOUT.csv", index=False)
    else:
        holdout.to_csv(OUT / "REPLICATION_HOLDOUT.csv", index=False)

    confirmed: list[dict[str, Any]] = []
    for _, row in holdout.iterrows():
        if gate(row):
            confirmed.append({
                "family": row["family"],
                "primitive_factor": row["primitive_factor"],
                "signed_factor": row["factor"],
                "horizon": int(row["horizon"]),
                "replication_validation_holm_p": float(row["replication_validation_holm_p"]),
                "replication_holdout_cs_rankic": float(row["cs_rankic"]),
                "replication_holdout_ts_ic": float(row["ts_ic"]),
                "replication_holdout_ridge_mse_gain": float(row["ridge_mse_gain"]),
                "replication_holdout_residual_cs_rankic": float(row["residual_cs_rankic"]),
                "replication_holdout_annual_log_return_5bp": float(row["annual_log_return_5bp"]),
                "replication_holdout_sharpe_5bp": float(row["sharpe_5bp"]),
            })

    write_json(OUT / "DATA_AUDIT.json", {
        "symbols": REPLICATION_SYMBOLS,
        "data": data_audit,
        "panel_rows": int(len(panel)),
        "frozen_selection_source": str(args.selection),
        "promoted_source": str(args.promoted),
        "selection_policy": "formula, sign, smoothing and horizon frozen before replication universe data were read",
    })
    write_json(OUT / "CONFIRMED.json", confirmed)
    summary = {
        "replication_symbols": REPLICATION_SYMBOLS,
        "frozen_hypotheses": int(len(promoted)),
        "validation_passed": int(validation["replication_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "confirmed": confirmed,
        "decision": "eligible_for_trade_review_only_if_confirmed_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = [
        "# Frozen cross-universe replication for fluid round16",
        "",
        "No formula, sign, state span, family composition or horizon was selected in the replication universe.",
        "",
        "## Validation",
        validation.to_markdown(index=False),
        "",
        "## Historical holdout",
        holdout.to_markdown(index=False) if not holdout.empty else "Holdout remained closed.",
        "",
        "## Confirmed",
        "```json",
        json.dumps(confirmed, indent=2, sort_keys=True),
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
