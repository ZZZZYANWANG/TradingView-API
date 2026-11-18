from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as core
import refine_enstrophy as refined

OUT = Path(os.environ.get("OUT_DIR", "research_outputs/fluid_hodge_dynamic"))
OUT.mkdir(parents=True, exist_ok=True)

# Frozen protocol before the dynamic-graph run.
DEV = (pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-31"))
VAL = (pd.Timestamp("2017-01-01"), pd.Timestamp("2021-12-31"))
HOLD = (pd.Timestamp("2022-01-01"), pd.Timestamp("2027-01-01"))
DEV_SUBS = [
    (pd.Timestamp("2012-01-01"), pd.Timestamp("2013-12-31"), "dev_2012_2013"),
    (pd.Timestamp("2014-01-01"), pd.Timestamp("2015-12-31"), "dev_2014_2015"),
    (pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31"), "dev_2016"),
]
CONFIGS = [(3, 0.05, 0.5), (3, 0.05, 1.0), (5, 0.05, 0.5), (5, 0.05, 1.0)]
SMOOTH = [8, 16, 32]
HORIZONS = [5, 10, 20]
FAMILIES = {
    "slow_dissipation_load",
    "dissipation_load_innovation",
    "enstrophy_release",
    "curl_gradient_conversion",
    "pressure_supported_release_s2",
    "pressure_supported_release_s4",
}
STATE_KEYS = [
    "divergence", "pressure", "gamma", "separation", "lead_prediction",
    "peer_velocity", "volume_z", "range_z", "grad_p_energy", "enstrophy",
]


def dynamic_states(ret1, close, high, low, volume, cols, k, lam, density):
    pieces = {key: [] for key in STATE_KEYS}
    original_dev = core.DEV
    try:
        for year in range(2012, min(2026, close.index.max().year) + 1):
            calibration = (pd.Timestamp(f"{year-5}-01-01"), pd.Timestamp(f"{year-1}-12-31"))
            core.DEV = calibration
            states = core.compute_graph_states(ret1, close, high, low, volume, cols, k, lam, density)
            year_dates = states["pressure"].index[states["pressure"].index.year == year]
            for key in STATE_KEYS:
                pieces[key].append(states[key].loc[year_dates])
    finally:
        core.DEV = original_dev
    out = {}
    for key, frames in pieces.items():
        out[key] = pd.concat(frames).sort_index().reindex(close.index)
    return out


def main():
    core.DEV, core.VAL, core.HOLD, core.DEV_SUBS = DEV, VAL, HOLD, DEV_SUBS
    fields = core.download_market()
    open_, high, low, close, volume = [fields[k] for k in ["Open", "High", "Low", "Close", "Volume"]]
    idx = close.index
    for df in [open_, high, low, volume]:
        idx = idx.union(df.index)
    open_, high, low, close, volume = [df.reindex(idx).sort_index() for df in [open_, high, low, close, volume]]
    eligible = []
    coverage = []
    for ticker in close.columns:
        counts = {}
        for label, bounds in [("dev", DEV), ("validation", VAL), ("holdout", HOLD)]:
            counts[label] = int(close.loc[core.period_index(close.index, bounds), ticker].notna().sum())
        coverage.append({"ticker": ticker, **counts})
        if counts["dev"] >= 800 and counts["validation"] >= 800 and counts["holdout"] >= 600:
            eligible.append(ticker)
    if len(eligible) < 14:
        raise RuntimeError(f"only {len(eligible)} eligible products")
    open_, high, low, close, volume = [df[eligible] for df in [open_, high, low, close, volume]]
    ret1 = np.log(close / close.shift(1))
    fwd = {h: np.log(close.shift(-h) / open_.shift(-1)) for h in HORIZONS}
    dv = (close * volume).loc[core.period_index(close.index, DEV)].median().sort_values(ascending=False)
    ordered = list(dv.index)
    split = max(7, len(ordered) // 2)
    high_pool, low_pool = ordered[:split], ordered[split:]
    groups = {t: core.TICKER_GROUP.get(t, "unknown") for t in eligible}
    (OUT / "data_audit.json").write_text(json.dumps({
        "source": "Yahoo Finance continuous futures OHLCV",
        "graph_recalibration": "annual using the preceding five calendar years only",
        "eligible": eligible,
        "coverage": coverage,
        "development": [str(DEV[0].date()), str(DEV[1].date())],
        "validation": [str(VAL[0].date()), str(VAL[1].date())],
        "holdout": [str(HOLD[0].date()), str(HOLD[1].date())],
        "data_end": str(close.index.max().date()),
    }, indent=2), encoding="utf-8")

    scan, winners, factor_store, baseline_store = [], {}, {}, {}
    for k, lam, density in CONFIGS:
        cfg = f"rolling5y_k{k}_l{lam:g}_d{density:g}"
        states = dynamic_states(ret1, close, high, low, volume, eligible, k, lam, density)
        baseline = core.build_baseline(ret1, open_, high, low, close, volume, eligible, states)
        raw = refined.make_raw_families(states, ret1, close, groups, eligible)
        for family, raw_factor in raw.items():
            if family not in FAMILIES:
                continue
            for smooth in SMOOTH:
                factor = core.ts_z_prior(raw_factor, 120, 40).clip(-8.0, 8.0)
                factor = factor.ewm(span=smooth, adjust=False, min_periods=max(4, smooth // 2)).mean()
                for h in HORIZONS:
                    score, direction, cells = core.dev_score(factor, fwd[h], high_pool, low_pool)
                    cid = f"{family}|{cfg}|smooth={smooth}|h={h}|dir={direction}"
                    row = {"candidate_id": cid, "family": family, "config": cfg, "smooth": smooth, "horizon": h, "direction": direction, "dev_score": score, "dev_min_cell": min(cells) if cells else np.nan, "dev_mean_cell": float(np.mean(cells)) if cells else np.nan}
                    scan.append(row)
                    if np.isfinite(score) and (family not in winners or score > winners[family]["dev_score"]):
                        winners[family] = {**row, "factor": direction * factor, "baseline": baseline}
    pd.DataFrame(scan).sort_values("dev_score", ascending=False).to_csv(OUT / "candidate_scan.csv", index=False)
    pd.DataFrame([{k: v for k, v in w.items() if k not in {"factor", "baseline"}} for w in winners.values()]).to_csv(OUT / "family_winners.csv", index=False)

    val_rows = []
    for family, w in winners.items():
        h = int(w["horizon"])
        res = core.evaluate(w["candidate_id"], w["factor"], h, fwd[h], w["baseline"], VAL, eligible, ret1)
        res.update({k: w[k] for k in ["family", "config", "smooth", "direction"]})
        val_rows.append(res)
        factor_store[w["candidate_id"]] = w["factor"]
        baseline_store[w["candidate_id"]] = w["baseline"]
    adj = core.holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in val_rows})
    for r in val_rows:
        r["holm_p"] = adj.get(r["candidate_id"], np.nan)
        r["validation_pass"] = bool(r["rankic"] > 0.01 and r["holm_p"] <= 0.10 and r["mse_gain_pct"] > 0 and r["residual_rankic"] > 0.005 and r["max_parent_corr"] < 0.80 and r["net5_annual_return"] > 0)
    val = pd.DataFrame(val_rows).sort_values("rankic", ascending=False)
    val.to_csv(OUT / "validation_results.csv", index=False)

    relaxed = val[(val.rankic > 0.005) & (val.residual_rankic > 0) & (val.max_parent_corr < 0.85) & (val.net5_annual_return > 0)].copy()
    relaxed["selection_score"] = relaxed.rankic + 0.5 * relaxed.residual_rankic + 0.02 * relaxed.mse_gain_pct.clip(-1, 1) + 0.005 * relaxed.net5_sharpe.clip(-3, 3)
    relaxed = relaxed.sort_values("selection_score", ascending=False)
    finalists = []
    for _, row in relaxed.head(5).iterrows():
        cid = row.candidate_id
        finalists.append({"candidate_id": cid, "factor": factor_store[cid], "baseline": baseline_store[cid], "horizon": int(row.horizon), "source_ids": [cid]})

    hold_rows = []
    for item in finalists:
        h = item["horizon"]
        res = core.evaluate(item["candidate_id"], item["factor"], h, fwd[h], item["baseline"], HOLD, eligible, ret1)
        years = core.yearly_rankic(item["factor"], fwd[h], HOLD, eligible)
        finite = [v for v in years.values() if np.isfinite(v)]
        res["source_ids"] = "||".join(item["source_ids"])
        res["yearly_rankic"] = json.dumps(years, sort_keys=True)
        res["positive_year_share"] = float(np.mean([v > 0 for v in finite])) if finite else np.nan
        hold_rows.append(res)
    hadj = core.holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in hold_rows})
    for r in hold_rows:
        r["holm_p"] = hadj.get(r["candidate_id"], np.nan)
        r["final_pass"] = bool(r["rankic"] > 0.015 and r["holm_p"] <= 0.10 and r["mse_gain_pct"] > 0 and r["residual_rankic"] > 0.005 and r["max_parent_corr"] < 0.80 and r["net5_annual_return"] > 0 and r["positive_year_share"] >= 0.60)
    hold = pd.DataFrame(hold_rows)
    if len(hold):
        hold = hold.sort_values(["final_pass", "rankic"], ascending=[False, False])
    hold.to_csv(OUT / "holdout_results.csv", index=False)
    admitted = hold[hold.final_pass == True] if len(hold) else pd.DataFrame()
    decision = {"status": "admit_historical_default_off" if len(admitted) else "no_catalog_admission", "family_count": len(winners), "validation_pass_count": int(val.validation_pass.sum()) if len(val) else 0, "finalist_count": len(finalists), "holdout_pass_count": int(hold.final_pass.sum()) if len(hold) else 0, "data_end": str(close.index.max().date())}
    if len(admitted):
        decision["selected"] = admitted.iloc[0].to_dict()
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float), encoding="utf-8")
    (OUT / "selected_factor_spec.json").write_text(json.dumps({"decision": decision, "validation": val.to_dict(orient="records"), "holdout": hold.to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=float))


if __name__ == "__main__":
    main()
