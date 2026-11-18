from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as core
import refine_enstrophy as refined
import refine_dynamic_graph as dynamic

OUT = Path(os.environ.get("OUT_DIR", "research_outputs/fluid_hodge_residual"))
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12

# Frozen before this residualized-factor run.
DEV = dynamic.DEV
VAL = dynamic.VAL
HOLD = dynamic.HOLD
DEV_SUBS = dynamic.DEV_SUBS
CONFIGS = [(3, 0.05, 1.0), (5, 0.05, 0.5), (5, 0.05, 1.0)]
SMOOTH = [16, 32, 64]
HORIZONS = [10, 20]
RAW_FAMILIES = [
    "curl_gradient_conversion",
    "production_dissipation_balance",
    "vortex_reconnection",
    "dissipation_load_innovation",
]
PARENT_SETS = {
    "pressure": ["pressure_energy"],
    "pressure_enstrophy": ["pressure_energy", "enstrophy"],
    "pressure_peer": ["pressure_energy", "peer_mom20"],
    "full_mechanical": [
        "pressure_energy",
        "enstrophy",
        "peer_mom20",
        "divergence",
        "pressure",
    ],
}


def annual_unsupervised_residual(
    raw: pd.DataFrame,
    parents: dict[str, pd.DataFrame],
    parent_names: list[str],
    cols: list[str],
    start_year: int,
    end_year: int,
    lookback_years: int = 5,
    alpha: float = 20.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """Residualize a state using only the preceding calendar years.

    The regression target is the state itself, never a future return.  Each
    application year receives coefficients fitted on years y-lookback..y-1.
    """
    result = pd.DataFrame(np.nan, index=raw.index, columns=cols)
    audit: list[dict] = []
    for year in range(start_year, end_year + 1):
        cal_start = pd.Timestamp(f"{year - lookback_years}-01-01")
        cal_end = pd.Timestamp(f"{year - 1}-12-31")
        app_start = pd.Timestamp(f"{year}-01-01")
        app_end = pd.Timestamp(f"{year}-12-31")
        cal_dates = raw.index[(raw.index >= cal_start) & (raw.index <= cal_end)]
        app_dates = raw.index[(raw.index >= app_start) & (raw.index <= app_end)]
        if len(cal_dates) < 250 or len(app_dates) == 0:
            continue

        y_cal = raw.loc[cal_dates, cols].to_numpy(dtype=float).reshape(-1)
        x_cal = np.column_stack(
            [parents[name].loc[cal_dates, cols].to_numpy(dtype=float).reshape(-1) for name in parent_names]
        )
        ok_cal = np.isfinite(y_cal) & np.isfinite(x_cal).all(axis=1)
        if ok_cal.sum() < 2000:
            continue
        x_mu = x_cal[ok_cal].mean(axis=0)
        x_sd = x_cal[ok_cal].std(axis=0, ddof=0)
        x_sd = np.where(x_sd > EPS, x_sd, 1.0)
        model = Ridge(alpha=alpha)
        model.fit((x_cal[ok_cal] - x_mu) / x_sd, y_cal[ok_cal])

        y_app = raw.loc[app_dates, cols].to_numpy(dtype=float)
        x_app = np.stack(
            [parents[name].loc[app_dates, cols].to_numpy(dtype=float) for name in parent_names],
            axis=-1,
        )
        flat_x = x_app.reshape(-1, len(parent_names))
        flat_y = y_app.reshape(-1)
        ok_app = np.isfinite(flat_y) & np.isfinite(flat_x).all(axis=1)
        flat_residual = np.full_like(flat_y, np.nan, dtype=float)
        if ok_app.any():
            prediction = model.predict((flat_x[ok_app] - x_mu) / x_sd)
            flat_residual[ok_app] = flat_y[ok_app] - prediction
        result.loc[app_dates, cols] = flat_residual.reshape(len(app_dates), len(cols))
        audit.append(
            {
                "application_year": year,
                "calibration_start": str(cal_start.date()),
                "calibration_end": str(cal_end.date()),
                "n_calibration_rows": int(ok_cal.sum()),
                "parent_names": "|".join(parent_names),
                "intercept": float(model.intercept_),
                "coefficients": json.dumps({name: float(value) for name, value in zip(parent_names, model.coef_)}, sort_keys=True),
            }
        )
    return result, audit


def parent_frames(states: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "pressure_energy": states["grad_p_energy"],
        "enstrophy": states["enstrophy"],
        "peer_mom20": states["peer_velocity"].rolling(20, min_periods=10).sum(),
        "divergence": states["divergence"],
        "pressure": states["pressure"],
    }


def main() -> None:
    core.DEV, core.VAL, core.HOLD, core.DEV_SUBS = DEV, VAL, HOLD, DEV_SUBS
    fields = core.download_market()
    open_, high, low, close, volume = [fields[k] for k in ["Open", "High", "Low", "Close", "Volume"]]
    index = close.index
    for frame in [open_, high, low, volume]:
        index = index.union(frame.index)
    open_, high, low, close, volume = [frame.reindex(index).sort_index() for frame in [open_, high, low, close, volume]]

    eligible: list[str] = []
    coverage = []
    for ticker in close.columns:
        counts = {}
        for label, bounds in [("dev", DEV), ("validation", VAL), ("holdout", HOLD)]:
            dates = core.period_index(close.index, bounds)
            counts[label] = int(close.loc[dates, ticker].notna().sum())
        coverage.append({"ticker": ticker, **counts})
        if counts["dev"] >= 800 and counts["validation"] >= 800 and counts["holdout"] >= 600:
            eligible.append(ticker)
    if len(eligible) < 14:
        raise RuntimeError(f"only {len(eligible)} eligible products: {eligible}")

    open_, high, low, close, volume = [frame[eligible] for frame in [open_, high, low, close, volume]]
    ret1 = np.log(close / close.shift(1))
    forward = {h: np.log(close.shift(-h) / open_.shift(-1)) for h in HORIZONS}
    development_activity = (close * volume).loc[core.period_index(close.index, DEV)].median().sort_values(ascending=False)
    ordered = list(development_activity.index)
    split = max(7, len(ordered) // 2)
    high_pool, secondary_pool = ordered[:split], ordered[split:]
    groups = {ticker: core.TICKER_GROUP.get(ticker, "unknown") for ticker in eligible}

    (OUT / "data_audit.json").write_text(
        json.dumps(
            {
                "source": "Yahoo Finance continuous futures OHLCV",
                "factor_clock": "annual graph and unsupervised residual coefficients use preceding five calendar years only",
                "eligible": eligible,
                "coverage": coverage,
                "development": [str(DEV[0].date()), str(DEV[1].date())],
                "validation": [str(VAL[0].date()), str(VAL[1].date())],
                "holdout": [str(HOLD[0].date()), str(HOLD[1].date())],
                "data_end": str(close.index.max().date()),
                "parent_sets": PARENT_SETS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate_rows: list[dict] = []
    audit_rows: list[dict] = []
    winners: dict[str, dict] = {}
    factor_store: dict[str, pd.DataFrame] = {}
    baseline_store: dict[str, dict[str, pd.DataFrame]] = {}

    for k, lam, density in CONFIGS:
        config = f"rolling5y_k{k}_l{lam:g}_d{density:g}"
        states = dynamic.dynamic_states(ret1, close, high, low, volume, eligible, k, lam, density)
        baseline = core.build_baseline(ret1, open_, high, low, close, volume, eligible, states)
        raw_map = refined.make_raw_families(states, ret1, close, groups, eligible)
        parents = parent_frames(states)
        for raw_family in RAW_FAMILIES:
            raw = raw_map[raw_family]
            for parent_set_name, parent_names in PARENT_SETS.items():
                residual, audit = annual_unsupervised_residual(
                    raw,
                    parents,
                    parent_names,
                    eligible,
                    start_year=DEV[0].year,
                    end_year=min(HOLD[1].year, close.index.max().year),
                )
                for row in audit:
                    audit_rows.append({"config": config, "raw_family": raw_family, "parent_set": parent_set_name, **row})
                family = f"{raw_family}_residual_{parent_set_name}"
                for smooth in SMOOTH:
                    factor = core.ts_z_prior(residual, 120, 40).clip(-8.0, 8.0)
                    factor = factor.ewm(span=smooth, adjust=False, min_periods=max(8, smooth // 2)).mean()
                    for horizon in HORIZONS:
                        score, direction, cells = core.dev_score(factor, forward[horizon], high_pool, secondary_pool)
                        candidate_id = f"{family}|{config}|smooth={smooth}|h={horizon}|dir={direction}"
                        row = {
                            "candidate_id": candidate_id,
                            "family": family,
                            "raw_family": raw_family,
                            "parent_set": parent_set_name,
                            "config": config,
                            "smooth": smooth,
                            "horizon": horizon,
                            "direction": direction,
                            "dev_score": score,
                            "dev_min_cell": min(cells) if cells else np.nan,
                            "dev_mean_cell": float(np.mean(cells)) if cells else np.nan,
                        }
                        candidate_rows.append(row)
                        if np.isfinite(score) and (family not in winners or score > winners[family]["dev_score"]):
                            winners[family] = {**row, "factor": direction * factor, "baseline": baseline}

    pd.DataFrame(candidate_rows).sort_values("dev_score", ascending=False).to_csv(OUT / "candidate_scan.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "residualization_audit.csv", index=False)
    pd.DataFrame([{k: v for k, v in item.items() if k not in {"factor", "baseline"}} for item in winners.values()]).to_csv(OUT / "family_winners.csv", index=False)

    validation_rows = []
    for family, winner in winners.items():
        horizon = int(winner["horizon"])
        result = core.evaluate(
            winner["candidate_id"],
            winner["factor"],
            horizon,
            forward[horizon],
            winner["baseline"],
            VAL,
            eligible,
            ret1,
        )
        result.update({key: winner[key] for key in ["family", "raw_family", "parent_set", "config", "smooth", "direction"]})
        validation_rows.append(result)
        factor_store[winner["candidate_id"]] = winner["factor"]
        baseline_store[winner["candidate_id"]] = winner["baseline"]

    adjusted = core.holm_adjust({row["candidate_id"]: row["p_nonpositive"] for row in validation_rows})
    for row in validation_rows:
        row["holm_p"] = adjusted.get(row["candidate_id"], np.nan)
        row["validation_pass"] = bool(
            row["rankic"] > 0.012
            and row["holm_p"] <= 0.10
            and row["mse_gain_pct"] > 0.0
            and row["residual_rankic"] > 0.005
            and row["max_parent_corr"] < 0.75
            and row["net5_annual_return"] > 0.0
        )
    validation = pd.DataFrame(validation_rows).sort_values("rankic", ascending=False)
    validation.to_csv(OUT / "validation_results.csv", index=False)

    relaxed = validation[
        (validation["rankic"] > 0.008)
        & (validation["residual_rankic"] > 0.003)
        & (validation["max_parent_corr"] < 0.80)
        & (validation["net5_annual_return"] > 0.0)
    ].copy()
    relaxed["selection_score"] = (
        relaxed["rankic"]
        + 0.5 * relaxed["residual_rankic"]
        + 0.02 * relaxed["mse_gain_pct"].clip(-1.0, 1.0)
        + 0.005 * relaxed["net5_sharpe"].clip(-3.0, 3.0)
    )
    relaxed = relaxed.sort_values("selection_score", ascending=False)

    finalists = []
    for _, row in relaxed.head(6).iterrows():
        candidate_id = row["candidate_id"]
        finalists.append(
            {
                "candidate_id": candidate_id,
                "factor": factor_store[candidate_id],
                "baseline": baseline_store[candidate_id],
                "horizon": int(row["horizon"]),
                "source_ids": [candidate_id],
            }
        )

    holdout_rows = []
    for item in finalists:
        horizon = item["horizon"]
        result = core.evaluate(
            item["candidate_id"],
            item["factor"],
            horizon,
            forward[horizon],
            item["baseline"],
            HOLD,
            eligible,
            ret1,
        )
        yearly = core.yearly_rankic(item["factor"], forward[horizon], HOLD, eligible)
        finite_years = [value for value in yearly.values() if np.isfinite(value)]
        result["source_ids"] = "||".join(item["source_ids"])
        result["yearly_rankic"] = json.dumps(yearly, sort_keys=True)
        result["positive_year_share"] = float(np.mean([value > 0 for value in finite_years])) if finite_years else np.nan
        holdout_rows.append(result)

    holdout_adjusted = core.holm_adjust({row["candidate_id"]: row["p_nonpositive"] for row in holdout_rows})
    for row in holdout_rows:
        row["holm_p"] = holdout_adjusted.get(row["candidate_id"], np.nan)
        row["final_pass"] = bool(
            row["rankic"] > 0.015
            and row["holm_p"] <= 0.10
            and row["mse_gain_pct"] > 0.0
            and row["residual_rankic"] > 0.005
            and row["max_parent_corr"] < 0.75
            and row["net5_annual_return"] > 0.0
            and row["positive_year_share"] >= 0.80
        )
    holdout = pd.DataFrame(holdout_rows)
    if len(holdout):
        holdout = holdout.sort_values(["final_pass", "rankic"], ascending=[False, False])
    holdout.to_csv(OUT / "holdout_results.csv", index=False)

    admitted = holdout[holdout["final_pass"] == True] if len(holdout) else pd.DataFrame()
    decision = {
        "status": "admit_historical_default_off" if len(admitted) else "no_catalog_admission",
        "family_count": len(winners),
        "validation_pass_count": int(validation["validation_pass"].sum()) if len(validation) else 0,
        "finalist_count": len(finalists),
        "holdout_pass_count": int(holdout["final_pass"].sum()) if len(holdout) else 0,
        "data_end": str(close.index.max().date()),
        "eligible_count": len(eligible),
    }
    if len(admitted):
        decision["selected"] = admitted.iloc[0].to_dict()
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float), encoding="utf-8")
    (OUT / "selected_factor_spec.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "validation": validation.to_dict(orient="records"),
                "holdout": holdout.to_dict(orient="records") if len(holdout) else [],
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, default=float))


if __name__ == "__main__":
    main()
