from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fluid_hodge_futures"))
import run as core
import refine_enstrophy as refined

OUT = Path(os.environ.get("OUT_DIR", "research_outputs/fluid_hodge_china_futures"))
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12

# Frozen before observing any China-futures result.
core.DEV = (pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31"))
core.VAL = (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"))
core.HOLD = (pd.Timestamp("2023-01-01"), pd.Timestamp("2027-01-01"))
core.DEV_SUBS = [
    (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"), "dev_2015_2016"),
    (pd.Timestamp("2017-01-01"), pd.Timestamp("2018-06-30"), "dev_2017_2018h1"),
    (pd.Timestamp("2018-07-01"), pd.Timestamp("2019-12-31"), "dev_2018h2_2019"),
]
HORIZONS = [5, 10, 20]
CONFIGS = [(3, 0.05, 0.5), (3, 0.05, 1.0), (5, 0.05, 0.5)]
SMOOTH = [8, 16, 32]


def to_numeric_frame(df: pd.DataFrame, column: str, symbol: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float, name=symbol)
    out = pd.to_numeric(df[column], errors="coerce")
    out.index = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.name = symbol
    return out


def get_symbol_table() -> pd.DataFrame:
    table = ak.futures_display_main_sina().copy()
    table.columns = [str(c).lower() for c in table.columns]
    required = {"symbol", "exchange", "name"}
    if not required.issubset(table.columns):
        raise RuntimeError(f"unexpected futures display columns: {table.columns.tolist()}")
    table["symbol"] = table["symbol"].astype(str).str.upper()
    table["exchange"] = table["exchange"].astype(str).str.lower()
    # DCE public history is frequently blocked by upstream anti-bot controls.
    table = table[~table.exchange.eq("dce")]
    table = table[table.symbol.str.endswith("0")]
    table = table.drop_duplicates("symbol").sort_values(["exchange", "symbol"])
    return table.reset_index(drop=True)


def download_panel():
    table = get_symbol_table()
    records = []
    series = {k: [] for k in ["open", "high", "low", "close", "volume", "hold"]}
    for i, row in table.iterrows():
        symbol = row.symbol
        last_error = None
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = ak.futures_zh_daily_sina(symbol=symbol)
                if not df.empty:
                    break
            except Exception as exc:
                last_error = repr(exc)
                time.sleep(1.5 * (attempt + 1))
        if df.empty:
            records.append({"symbol": symbol, "exchange": row.exchange, "name": row["name"], "status": "failed", "error": last_error})
            continue
        df.columns = [str(c).lower() for c in df.columns]
        if "date" not in df.columns:
            records.append({"symbol": symbol, "exchange": row.exchange, "name": row["name"], "status": "bad_schema", "error": str(df.columns.tolist())})
            continue
        for column in series:
            series[column].append(to_numeric_frame(df, column, symbol))
        records.append({
            "symbol": symbol,
            "exchange": row.exchange,
            "name": row["name"],
            "status": "ok",
            "rows": int(len(df)),
            "start": str(pd.to_datetime(df.date).min().date()),
            "end": str(pd.to_datetime(df.date).max().date()),
        })
        time.sleep(0.15)
    panels = {k: pd.concat(v, axis=1).sort_index() if v else pd.DataFrame() for k, v in series.items()}
    return table, records, panels


def extend_baseline(base: dict[str, pd.DataFrame], hold: pd.DataFrame, volume: pd.DataFrame, cols: list[str]):
    log_oi = np.log1p(hold[cols].clip(lower=0.0))
    doi = log_oi.diff()
    log_vol = np.log1p(volume[cols].clip(lower=0.0))
    extra = {
        "oi_level_z60": core.ts_z_prior(log_oi, 60, 20),
        "oi_change_1": doi,
        "oi_change_5": doi.rolling(5, min_periods=3).sum(),
        "oi_change_20": doi.rolling(20, min_periods=10).sum(),
        "volume_oi_ratio_z60": core.ts_z_prior(log_vol - log_oi, 60, 20),
    }
    return {**base, **{k: core.xs_rank(v) for k, v in extra.items()}}


def main() -> None:
    table, download_records, panels = download_panel()
    open_, high, low, close, volume, hold = [panels[k] for k in ["open", "high", "low", "close", "volume", "hold"]]
    if close.empty:
        raise RuntimeError("no China-futures data downloaded")
    index = close.index
    for df in [open_, high, low, volume, hold]:
        index = index.union(df.index)
    open_, high, low, close, volume, hold = [df.reindex(index).sort_index() for df in [open_, high, low, close, volume, hold]]

    eligible = []
    coverage = []
    for symbol in close.columns:
        counts = {}
        for label, bounds in [("dev", core.DEV), ("validation", core.VAL), ("holdout", core.HOLD)]:
            dates = core.period_index(close.index, bounds)
            counts[label] = int((close.loc[dates, symbol].notna() & hold.loc[dates, symbol].notna()).sum())
        coverage.append({"symbol": symbol, **counts})
        if counts["dev"] >= 750 and counts["validation"] >= 500 and counts["holdout"] >= 400:
            eligible.append(symbol)
    if len(eligible) < 14:
        raise RuntimeError(f"only {len(eligible)} eligible products: {eligible}")

    open_, high, low, close, volume, hold = [df[eligible] for df in [open_, high, low, close, volume, hold]]
    ret1 = np.log(close / close.shift(1))
    fwd = {h: np.log(close.shift(-h) / open_.shift(-1)) for h in HORIZONS}
    dev_activity = np.sqrt((volume * hold).clip(lower=0.0)).loc[core.period_index(close.index, core.DEV)].median()
    ordered = list(dev_activity.sort_values(ascending=False).index)
    split = max(7, len(ordered) // 2)
    high_pool, low_pool = ordered[:split], ordered[split:]
    exchange_map = table.set_index("symbol").exchange.to_dict()
    groups = {s: exchange_map.get(s, "unknown") for s in eligible}

    (OUT / "download_audit.json").write_text(json.dumps({
        "interface": "akshare.futures_zh_daily_sina",
        "records": download_records,
        "coverage": coverage,
        "eligible": eligible,
        "groups": groups,
        "high_liquidity_pool": high_pool,
        "secondary_liquidity_pool": low_pool,
        "data_start": str(close.index.min().date()),
        "data_end": str(close.index.max().date()),
        "development": [str(core.DEV[0].date()), str(core.DEV[1].date())],
        "validation": [str(core.VAL[0].date()), str(core.VAL[1].date())],
        "holdout": [str(core.HOLD[0].date()), str(core.HOLD[1].date())],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    scan_rows, winners, factor_store, baseline_store = [], {}, {}, {}
    density_activity = np.sqrt((volume * hold).clip(lower=0.0))
    for k, lam, density in CONFIGS:
        cfg = f"k{k}_l{lam:g}_d{density:g}"
        states = core.compute_graph_states(ret1, close, high, low, density_activity, eligible, k, lam, density)
        baseline = core.build_baseline(ret1, open_, high, low, close, density_activity, eligible, states)
        baseline = extend_baseline(baseline, hold, volume, eligible)
        raw = refined.make_raw_families(states, ret1, close, groups, eligible)
        for family, raw_factor in raw.items():
            for smooth in SMOOTH:
                factor = core.ts_z_prior(raw_factor, 120, 40).clip(-8.0, 8.0)
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
                    scan_rows.append(row)
                    if np.isfinite(score) and (family not in winners or score > winners[family]["dev_score"]):
                        winners[family] = {**row, "factor": direction * factor, "baseline": baseline}
    pd.DataFrame(scan_rows).sort_values("dev_score", ascending=False).to_csv(OUT / "candidate_scan.csv", index=False)
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
    selected = []
    for _, row in relaxed.iterrows():
        cid = row.candidate_id
        if all(refined.daily_mean_abs_corr(factor_store[cid], factor_store[p], core.VAL, eligible) < 0.70 for p in selected):
            selected.append(cid)
        if len(selected) >= 4:
            break
    if len(selected) >= 2:
        h = int(relaxed[relaxed.candidate_id.isin(selected)].horizon.mode().iloc[0])
        consensus = sum(core.xs_rank(factor_store[c]) for c in selected) / len(selected)
        finalists.append({"candidate_id": "china_enstrophy_dissipation_consensus", "factor": consensus, "baseline": baseline_store[selected[0]], "horizon": h, "source_ids": selected})

    hold_rows = []
    for item in finalists:
        h = item["horizon"]
        res = core.evaluate(item["candidate_id"], item["factor"], h, fwd[h], item["baseline"], core.HOLD, eligible, ret1)
        years = core.yearly_rankic(item["factor"], fwd[h], core.HOLD, eligible)
        res["source_ids"] = "||".join(item["source_ids"])
        res["yearly_rankic"] = json.dumps(years, sort_keys=True)
        finite = [v for v in years.values() if np.isfinite(v)]
        res["positive_year_share"] = float(np.mean([v > 0 for v in finite])) if finite else np.nan
        hold_rows.append(res)
    hadj = core.holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in hold_rows})
    for r in hold_rows:
        r["holm_p"] = hadj.get(r["candidate_id"], np.nan)
        r["final_pass"] = bool(r["rankic"] > 0.015 and r["holm_p"] <= 0.10 and r["mse_gain_pct"] > 0 and r["residual_rankic"] > 0.005 and r["max_parent_corr"] < 0.80 and r["net5_annual_return"] > 0 and r["positive_year_share"] >= 0.60)
    hold_df = pd.DataFrame(hold_rows)
    if len(hold_df):
        hold_df = hold_df.sort_values(["final_pass", "rankic"], ascending=[False, False])
    hold_df.to_csv(OUT / "holdout_results.csv", index=False)
    admitted = hold_df[hold_df.final_pass == True] if len(hold_df) else pd.DataFrame()
    decision = {
        "status": "admit_historical_default_off" if len(admitted) else "no_catalog_admission",
        "eligible_count": len(eligible),
        "eligible_symbols": eligible,
        "validation_pass_count": int(val.validation_pass.sum()) if len(val) else 0,
        "finalist_count": len(finalists),
        "holdout_pass_count": int(hold_df.final_pass.sum()) if len(hold_df) else 0,
        "data_end": str(close.index.max().date()),
    }
    if len(admitted):
        decision["selected"] = admitted.iloc[0].to_dict()
    (OUT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (OUT / "selected_factor_spec.json").write_text(json.dumps({"decision": decision, "validation": val.to_dict(orient="records"), "holdout": hold_df.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
