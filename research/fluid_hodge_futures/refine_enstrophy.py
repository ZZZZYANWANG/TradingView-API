from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as core

OUT = Path(os.environ.get("OUT_DIR", "research_outputs/fluid_hodge_enstrophy"))
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
CONFIGS = [(3, 0.05, 0.5), (3, 0.05, 1.0), (5, 0.05, 0.5)]
SMOOTH = [8, 16, 32]
HORIZONS = [10, 20]
PRESSURE_SCALE = [1.0, 2.0, 4.0]


def group_neutral(frame: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    for group in sorted(set(groups.values())):
        cols = [c for c in frame.columns if groups.get(c) == group]
        if len(cols) >= 2:
            out.loc[:, cols] = frame[cols].sub(frame[cols].mean(axis=1), axis=0)
    return out


def daily_mean_abs_corr(a: pd.DataFrame, b: pd.DataFrame, bounds, cols: list[str]) -> float:
    dates = core.period_index(a.index, bounds)
    vals = []
    for dt in dates:
        x = a.loc[dt, cols].to_numpy(float)
        y = b.loc[dt, cols].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 6:
            c = spearmanr(x[ok], y[ok]).correlation
            if np.isfinite(c):
                vals.append(abs(float(c)))
    return float(np.mean(vals)) if vals else np.nan


def make_raw_families(states: dict, ret1: pd.DataFrame, close: pd.DataFrame, groups: dict[str, str], cols: list[str]):
    r = ret1[cols]
    vol = r.shift(1).rolling(20, min_periods=10).std(ddof=0)
    u = (r / (vol + EPS)).clip(-8.0, 8.0)
    u = u.sub(u.median(axis=1), axis=0)
    gamma = states["gamma"]
    enstrophy = states["enstrophy"]
    peer = states["peer_velocity"]
    pressure = states["pressure"]
    divergence = states["divergence"]
    gpe = states["grad_p_energy"]
    viscosity = np.exp((0.5 * states["range_z"] - 0.5 * states["volume_z"]).clip(-2.0, 2.0))
    load = enstrophy * viscosity
    load_z = core.ts_z_prior(load, 120, 40).clip(-8.0, 8.0)
    enstrophy_z = core.ts_z_prior(enstrophy, 120, 40).clip(-8.0, 8.0)
    pressure_z = core.ts_z_prior(pressure, 120, 40).clip(-8.0, 8.0)

    load_relax = load_z.shift(1).clip(lower=0.0) * (load_z.shift(1) - load_z).clip(lower=0.0)
    enstrophy_release = enstrophy_z.shift(1).clip(lower=0.0) * (enstrophy_z.shift(1) - enstrophy_z).clip(lower=0.0)
    stretching = gamma * (peer - u)
    production_balance = stretching - load
    reconnection = (-gamma.shift(1) * gamma.diff()).clip(lower=0.0)
    curl_gradient_conversion = (gpe.shift(1) - gpe) + (enstrophy - enstrophy.shift(1))
    pressure_dilatation = -pressure * divergence
    dissipation_innovation = load_z - load_z.shift(1).ewm(span=20, adjust=False, min_periods=10).mean()

    raw = {
        "slow_dissipation_load": -load,
        "dissipation_load_innovation": -dissipation_innovation,
        "enstrophy_release": enstrophy_release,
        "dissipation_relaxation": load_relax,
        "vortex_stretching": stretching,
        "production_dissipation_balance": production_balance,
        "vortex_reconnection": reconnection,
        "curl_gradient_conversion": curl_gradient_conversion,
        "pressure_dilatation_release": pressure_dilatation * enstrophy_release,
        "group_neutral_dissipation_relaxation": group_neutral(load_relax, groups),
    }
    for scale in PRESSURE_SCALE:
        raw[f"pressure_relief_relaxation_s{scale:g}"] = load_relax * np.exp(-pressure_z.abs() / scale)
        raw[f"pressure_supported_release_s{scale:g}"] = enstrophy_release * np.exp(-pressure_z.abs() / scale)
    return raw


def main() -> None:
    fields = core.download_market()
    open_, high, low, close, volume = [fields[k] for k in ["Open", "High", "Low", "Close", "Volume"]]
    common = close.index
    for df in [open_, high, low, volume]:
        common = common.union(df.index)
    open_, high, low, close, volume = [df.reindex(common).sort_index() for df in [open_, high, low, close, volume]]

    eligible = []
    for ticker in close.columns:
        counts = [int(close.loc[core.period_index(close.index, bounds), ticker].notna().sum()) for bounds in [core.DEV, core.VAL, core.HOLD]]
        if counts[0] >= 900 and counts[1] >= 800 and counts[2] >= 600:
            eligible.append(ticker)
    if len(eligible) < 14:
        raise RuntimeError(f"insufficient eligible products: {eligible}")
    open_, high, low, close, volume = [df[eligible] for df in [open_, high, low, close, volume]]
    ret1 = np.log(close / close.shift(1))
    fwd = {h: np.log(close.shift(-h) / open_.shift(-1)) for h in HORIZONS}
    dv = (close * volume).loc[core.period_index(close.index, core.DEV)].median().sort_values(ascending=False)
    split = max(7, len(dv) // 2)
    high_pool, low_pool = list(dv.index[:split]), list(dv.index[split:])
    groups = {t: core.TICKER_GROUP.get(t, "unknown") for t in eligible}

    scan = []
    winners = {}
    factor_store = {}
    baseline_store = {}
    for k, lam, density in CONFIGS:
        cfg = f"k{k}_l{lam:g}_d{density:g}"
        states = core.compute_graph_states(ret1, close, high, low, volume, eligible, k, lam, density)
        baseline = core.build_baseline(ret1, open_, high, low, close, volume, eligible, states)
        raw_families = make_raw_families(states, ret1, close, groups, eligible)
        for family, raw in raw_families.items():
            for smooth in SMOOTH:
                factor = core.ts_z_prior(raw, 120, 40).clip(-8.0, 8.0)
                factor = factor.ewm(span=smooth, adjust=False, min_periods=max(4, smooth // 2)).mean()
                for horizon in HORIZONS:
                    score, direction, cells = core.dev_score(factor, fwd[horizon], high_pool, low_pool)
                    cid = f"{family}|{cfg}|smooth={smooth}|h={horizon}|dir={direction}"
                    row = {
                        "candidate_id": cid, "family": family, "config": cfg, "smooth": smooth,
                        "horizon": horizon, "direction": direction, "dev_score": score,
                        "dev_min_cell": min(cells) if cells else np.nan,
                        "dev_mean_cell": float(np.mean(cells)) if cells else np.nan,
                    }
                    scan.append(row)
                    if np.isfinite(score) and (family not in winners or score > winners[family]["dev_score"]):
                        winners[family] = {**row, "factor": direction * factor, "baseline": baseline}
    pd.DataFrame(scan).sort_values("dev_score", ascending=False).to_csv(OUT / "candidate_scan.csv", index=False)
    pd.DataFrame([{k: v for k, v in x.items() if k not in {"factor", "baseline"}} for x in winners.values()]).to_csv(OUT / "family_winners.csv", index=False)

    val_rows = []
    for family, w in winners.items():
        res = core.evaluate(w["candidate_id"], w["factor"], w["horizon"], fwd[w["horizon"]], w["baseline"], core.VAL, eligible, ret1)
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

    relaxed = val[(val.rankic > 0.005) & (val.residual_rankic > 0.0) & (val.max_parent_corr < 0.85) & (val.net5_annual_return > 0.0)].copy()
    relaxed["selection_score"] = relaxed.rankic + 0.5 * relaxed.residual_rankic + 0.02 * relaxed.mse_gain_pct.clip(-1, 1) + 0.005 * relaxed.net5_sharpe.clip(-3, 3)
    relaxed = relaxed.sort_values("selection_score", ascending=False)

    finalists = []
    for _, row in relaxed.head(5).iterrows():
        cid = row.candidate_id
        finalists.append({"candidate_id": cid, "factor": factor_store[cid], "baseline": baseline_store[cid], "horizon": int(row.horizon), "source_ids": [cid]})
    selected_ids = []
    for _, row in relaxed.iterrows():
        cid = row.candidate_id
        if all(daily_mean_abs_corr(factor_store[cid], factor_store[p], core.VAL, eligible) < 0.70 for p in selected_ids):
            selected_ids.append(cid)
        if len(selected_ids) >= 4:
            break
    if len(selected_ids) >= 2:
        h = int(relaxed[relaxed.candidate_id.isin(selected_ids)].horizon.mode().iloc[0])
        consensus = sum(core.xs_rank(factor_store[c]) for c in selected_ids) / len(selected_ids)
        finalists.append({"candidate_id": "enstrophy_dissipation_consensus", "factor": consensus, "baseline": baseline_store[selected_ids[0]], "horizon": h, "source_ids": selected_ids})

    hold_rows = []
    for item in finalists:
        h = item["horizon"]
        res = core.evaluate(item["candidate_id"], item["factor"], h, fwd[h], item["baseline"], core.HOLD, eligible, ret1)
        years = core.yearly_rankic(item["factor"], fwd[h], core.HOLD, eligible)
        res["source_ids"] = "||".join(item["source_ids"])
        res["yearly_rankic"] = json.dumps(years, sort_keys=True)
        finite_years = [v for v in years.values() if np.isfinite(v)]
        res["positive_year_share"] = float(np.mean([v > 0 for v in finite_years])) if finite_years else np.nan
        hold_rows.append(res)
    hold_adj = core.holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in hold_rows})
    for r in hold_rows:
        r["holm_p"] = hold_adj.get(r["candidate_id"], np.nan)
        r["final_pass"] = bool(r["rankic"] > 0.015 and r["holm_p"] <= 0.10 and r["mse_gain_pct"] > 0 and r["residual_rankic"] > 0.005 and r["max_parent_corr"] < 0.80 and r["net5_annual_return"] > 0 and r["positive_year_share"] >= 0.60)
    hold = pd.DataFrame(hold_rows)
    if len(hold):
        hold = hold.sort_values(["final_pass", "rankic"], ascending=[False, False])
    hold.to_csv(OUT / "holdout_results.csv", index=False)

    admitted = hold[hold.final_pass == True] if len(hold) else pd.DataFrame()
    decision = {
        "status": "admit_historical_default_off" if len(admitted) else "no_catalog_admission",
        "family_count": len(winners),
        "validation_pass_count": int(val.validation_pass.sum()) if len(val) else 0,
        "finalist_count": len(finalists),
        "holdout_pass_count": int(hold.final_pass.sum()) if len(hold) else 0,
        "eligible_tickers": eligible,
        "data_end": str(close.index.max().date()),
    }
    if len(admitted):
        decision["selected"] = admitted.iloc[0].to_dict()
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float), encoding="utf-8")
    (OUT / "selected_factor_spec.json").write_text(json.dumps({"decision": decision, "validation": val.to_dict(orient="records"), "holdout": hold.to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=float))


if __name__ == "__main__":
    main()
