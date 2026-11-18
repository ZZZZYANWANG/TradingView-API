from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as core
import refine_enstrophy as refined

OUT = Path(os.environ.get("OUT_DIR", "research_outputs/fluid_hodge_timeseries"))
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
SEED = 20261118
CONFIGS = [(3, 0.05, 0.5), (3, 0.05, 1.0), (5, 0.05, 0.5)]
SMOOTH = [8, 16, 32]
HORIZONS = [5, 10, 20]


def product_tsic(factor: pd.DataFrame, target: pd.DataFrame, bounds, cols: list[str], min_obs: int = 120) -> pd.Series:
    dates = core.period_index(factor.index, bounds)
    out = {}
    for col in cols:
        pair = pd.concat([factor.loc[dates, col], target.loc[dates, col]], axis=1).dropna()
        if len(pair) < min_obs or pair.iloc[:, 0].std(ddof=0) <= EPS or pair.iloc[:, 1].std(ddof=0) <= EPS:
            continue
        value = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).correlation
        if np.isfinite(value):
            out[col] = float(value)
    return pd.Series(out, dtype=float)


def product_bootstrap(values: pd.Series, reps: int = 5000, seed: int = SEED) -> dict:
    x = values.dropna().to_numpy(float)
    if len(x) < 10:
        return {"ci_low": np.nan, "ci_high": np.nan, "p_nonpositive": np.nan}
    rng = np.random.default_rng(seed + len(x))
    means = np.mean(rng.choice(x, size=(reps, len(x)), replace=True), axis=1)
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0.0)),
    }


def dev_score_ts(factor: pd.DataFrame, target: pd.DataFrame, high_pool: list[str], low_pool: list[str]):
    full = product_tsic(factor, target, core.DEV, high_pool + low_pool, min_obs=250)
    if len(full) < 12 or abs(full.mean()) < 1e-8:
        return -np.inf, 1, []
    direction = 1 if full.mean() > 0 else -1
    oriented = direction * factor
    cells = []
    for start, end, _ in core.DEV_SUBS:
        for pool in [high_pool, low_pool]:
            values = product_tsic(oriented, target, (start, end), pool, min_obs=100)
            if len(values) >= max(4, len(pool) // 3):
                cells.append(float(values.mean()))
    if len(cells) < 6:
        return -np.inf, direction, cells
    return float(min(cells) + 0.25 * np.mean(cells)), direction, cells


def build_ts_baseline(ret1, open_, high, low, close, volume, cols, states, horizon):
    r = ret1[cols]
    logdv = np.log1p((close[cols] * volume[cols]).clip(lower=0.0))
    market = r.median(axis=1)
    market_var = market.rolling(60, min_periods=30).var(ddof=0)
    beta60 = r.rolling(60, min_periods=30).cov(market).div(market_var + EPS, axis=0)
    corr60 = r.rolling(60, min_periods=30).corr(market)
    raw = {
        "ret1": r,
        "mom5": np.log(close[cols] / close[cols].shift(5)),
        "mom20": np.log(close[cols] / close[cols].shift(20)),
        "mom60": np.log(close[cols] / close[cols].shift(60)),
        "mom120": np.log(close[cols] / close[cols].shift(120)),
        "vol20": r.rolling(20, min_periods=10).std(ddof=0),
        "vol60": r.rolling(60, min_periods=30).std(ddof=0),
        "range": np.log(high[cols] / low[cols]),
        "activity": logdv,
        "beta60": beta60,
        "corr_market60": corr60,
        "peer_velocity": states["peer_velocity"],
        "divergence": states["divergence"],
        "pressure": states["pressure"],
        "circulation": states["gamma"],
        "separation": states["separation"],
        "lead_prediction": states["lead_prediction"],
        "pressure_energy": states["grad_p_energy"],
        "enstrophy": states["enstrophy"],
    }
    return {name: core.ts_z_prior(frame, 120, 40).clip(-8.0, 8.0) for name, frame in raw.items()}


def stack_ts(features: dict[str, pd.DataFrame], target: pd.DataFrame, bounds, cols: list[str]):
    dates = core.period_index(target.index, bounds)
    names = list(features)
    rows, ys, keys, products = [], [], [], []
    for j, col in enumerate(cols):
        X = np.column_stack([features[n].loc[dates, col].to_numpy(float) for n in names])
        y = target.loc[dates, col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() == 0:
            continue
        rows.append(X[ok])
        ys.append(y[ok])
        valid_dates = dates[ok]
        keys.extend([(d, col) for d in valid_dates])
        products.extend([j] * int(ok.sum()))
    if not rows:
        return np.empty((0, len(names) + len(cols))), np.empty(0), [], names
    X = np.vstack(rows)
    product_index = np.asarray(products, dtype=int)
    onehot = np.eye(len(cols), dtype=float)[product_index]
    return np.column_stack([X, onehot]), np.concatenate(ys), keys, names


def model_increment_ts(factor, target_scaled, baseline, bounds, cols):
    Xtr, ytr, train_keys, _ = stack_ts(baseline, target_scaled, core.DEV, cols)
    Xte, yte, test_keys, _ = stack_ts(baseline, target_scaled, bounds, cols)
    fmap = factor.stack()
    ftr = np.array([fmap.get(k, np.nan) for k in train_keys], float)
    fte = np.array([fmap.get(k, np.nan) for k in test_keys], float)
    oktr, okte = np.isfinite(ftr), np.isfinite(fte)
    Xtr, ytr, ftr = Xtr[oktr], ytr[oktr], ftr[oktr]
    Xte, yte, fte = Xte[okte], yte[okte], fte[okte]
    test_keys = [k for k, ok in zip(test_keys, okte) if ok]
    if len(ytr) < 5000 or len(yte) < 2000:
        return {"mse_gain_pct": np.nan, "reconstruction_r2": np.nan, "residual_tsic": np.nan}
    base = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    plus = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    base.fit(Xtr, ytr)
    plus.fit(np.column_stack([Xtr, ftr]), ytr)
    p0 = base.predict(Xte)
    p1 = plus.predict(np.column_stack([Xte, fte]))
    mse0, mse1 = mean_squared_error(yte, p0), mean_squared_error(yte, p1)
    recon = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    recon.fit(Xtr, ftr)
    fhat = recon.predict(Xte)
    residual = fte - fhat
    residual_panel = pd.DataFrame(np.nan, index=factor.index, columns=cols)
    for key, value in zip(test_keys, residual):
        residual_panel.loc[key[0], key[1]] = value
    rvals = product_tsic(residual_panel, target_scaled, bounds, cols, min_obs=120)
    return {
        "mse_gain_pct": float(100.0 * (mse0 - mse1) / (mse0 + EPS)),
        "reconstruction_r2": float(r2_score(fte, fhat)),
        "residual_tsic": float(rvals.mean()) if len(rvals) else np.nan,
    }


def max_parent_corr_ts(factor, baseline, bounds, cols):
    best, best_name = -np.inf, ""
    for name, frame in baseline.items():
        vals = []
        for col in cols:
            dates = core.period_index(factor.index, bounds)
            pair = pd.concat([factor.loc[dates, col], frame.loc[dates, col]], axis=1).dropna()
            if len(pair) >= 120:
                c = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).correlation
                if np.isfinite(c):
                    vals.append(abs(float(c)))
        value = float(np.mean(vals)) if vals else np.nan
        if np.isfinite(value) and value > best:
            best, best_name = value, name
    return best, best_name


def portfolio_ts(factor, ret1, horizon, bounds, cols, cost_bp):
    signal = np.tanh(factor[cols] / 2.0)
    signal = signal.shift(1).rolling(horizon, min_periods=max(1, horizon // 3)).mean()
    gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
    weights = signal.div(gross, axis=0).fillna(0.0)
    pnl = (weights * ret1[cols]).sum(axis=1, min_count=1)
    turnover = 0.5 * (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = pnl - turnover * cost_bp * 1e-4
    x = net.loc[core.period_index(net.index, bounds)].dropna()
    if len(x) < 100:
        return {"annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "turnover": np.nan}
    equity = np.exp(x.cumsum())
    dd = equity / equity.cummax() - 1.0
    return {
        "annual_return": float(x.mean() * 252),
        "sharpe": float(x.mean() / (x.std(ddof=0) + EPS) * np.sqrt(252)),
        "max_drawdown": float(dd.min()),
        "turnover": float(turnover.loc[x.index].mean()),
    }


def evaluate_ts(candidate_id, factor, target, target_scaled, baseline, horizon, bounds, cols, ret1):
    values = product_tsic(factor, target, bounds, cols, min_obs=120)
    boot = product_bootstrap(values)
    model = model_increment_ts(factor, target_scaled, baseline, bounds, cols)
    corr, parent = max_parent_corr_ts(factor, baseline, bounds, cols)
    p5 = portfolio_ts(factor, ret1, horizon, bounds, cols, 5.0)
    p10 = portfolio_ts(factor, ret1, horizon, bounds, cols, 10.0)
    return {
        "candidate_id": candidate_id,
        "horizon": horizon,
        "mean_product_tsic": float(values.mean()) if len(values) else np.nan,
        "median_product_tsic": float(values.median()) if len(values) else np.nan,
        "positive_product_share": float((values > 0).mean()) if len(values) else np.nan,
        "n_products": int(len(values)),
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "p_nonpositive": boot["p_nonpositive"],
        "mse_gain_pct": model["mse_gain_pct"], "reconstruction_r2": model["reconstruction_r2"], "residual_tsic": model["residual_tsic"],
        "max_parent_corr": corr, "max_parent_name": parent,
        "net5_annual_return": p5["annual_return"], "net5_sharpe": p5["sharpe"], "net5_max_drawdown": p5["max_drawdown"],
        "net10_annual_return": p10["annual_return"], "net10_sharpe": p10["sharpe"], "avg_turnover": p5["turnover"],
    }


def main():
    fields = core.download_market()
    open_, high, low, close, volume = [fields[k] for k in ["Open", "High", "Low", "Close", "Volume"]]
    idx = close.index
    for df in [open_, high, low, volume]:
        idx = idx.union(df.index)
    open_, high, low, close, volume = [df.reindex(idx).sort_index() for df in [open_, high, low, close, volume]]
    eligible = []
    for ticker in close.columns:
        counts = [int(close.loc[core.period_index(close.index, b), ticker].notna().sum()) for b in [core.DEV, core.VAL, core.HOLD]]
        if counts[0] >= 900 and counts[1] >= 800 and counts[2] >= 600:
            eligible.append(ticker)
    if len(eligible) < 14:
        raise RuntimeError(f"only {len(eligible)} eligible products")
    open_, high, low, close, volume = [df[eligible] for df in [open_, high, low, close, volume]]
    ret1 = np.log(close / close.shift(1))
    fwd = {h: np.log(close.shift(-h) / open_.shift(-1)) for h in HORIZONS}
    vol20 = ret1.shift(1).rolling(20, min_periods=10).std(ddof=0)
    target_scaled = {h: fwd[h] / (vol20 * np.sqrt(h) + EPS) for h in HORIZONS}
    dev_dv = (close * volume).loc[core.period_index(close.index, core.DEV)].median().sort_values(ascending=False)
    ordered = list(dev_dv.index)
    split = max(7, len(ordered) // 2)
    high_pool, low_pool = ordered[:split], ordered[split:]
    groups = {t: core.TICKER_GROUP.get(t, "unknown") for t in eligible}

    scan, winners, factor_store, baseline_store = [], {}, {}, {}
    for k, lam, density in CONFIGS:
        cfg = f"k{k}_l{lam:g}_d{density:g}"
        states = core.compute_graph_states(ret1, close, high, low, volume, eligible, k, lam, density)
        raw = refined.make_raw_families(states, ret1, close, groups, eligible)
        for family, raw_factor in raw.items():
            for smooth in SMOOTH:
                factor = core.ts_z_prior(raw_factor, 120, 40).clip(-8.0, 8.0)
                factor = factor.ewm(span=smooth, adjust=False, min_periods=max(4, smooth // 2)).mean()
                for h in HORIZONS:
                    score, direction, cells = dev_score_ts(factor, fwd[h], high_pool, low_pool)
                    cid = f"{family}|{cfg}|smooth={smooth}|h={h}|dir={direction}"
                    row = {"candidate_id": cid, "family": family, "config": cfg, "smooth": smooth, "horizon": h, "direction": direction, "dev_score": score, "dev_min_cell": min(cells) if cells else np.nan, "dev_mean_cell": float(np.mean(cells)) if cells else np.nan}
                    scan.append(row)
                    if np.isfinite(score) and (family not in winners or score > winners[family]["dev_score"]):
                        baseline = build_ts_baseline(ret1, open_, high, low, close, volume, eligible, states, h)
                        winners[family] = {**row, "factor": direction * factor, "baseline": baseline}
    pd.DataFrame(scan).sort_values("dev_score", ascending=False).to_csv(OUT / "candidate_scan.csv", index=False)
    pd.DataFrame([{k: v for k, v in w.items() if k not in {"factor", "baseline"}} for w in winners.values()]).to_csv(OUT / "family_winners.csv", index=False)

    val_rows = []
    for family, w in winners.items():
        h = int(w["horizon"])
        res = evaluate_ts(w["candidate_id"], w["factor"], fwd[h], target_scaled[h], w["baseline"], h, core.VAL, eligible, ret1)
        res.update({k: w[k] for k in ["family", "config", "smooth", "direction"]})
        val_rows.append(res)
        factor_store[w["candidate_id"]] = w["factor"]
        baseline_store[w["candidate_id"]] = w["baseline"]
    adj = core.holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in val_rows})
    for r in val_rows:
        r["holm_p"] = adj.get(r["candidate_id"], np.nan)
        r["validation_pass"] = bool(r["mean_product_tsic"] > 0.02 and r["holm_p"] <= 0.10 and r["mse_gain_pct"] > 0 and r["residual_tsic"] > 0.005 and r["max_parent_corr"] < 0.80 and r["net5_annual_return"] > 0 and r["positive_product_share"] >= 0.60)
    val = pd.DataFrame(val_rows).sort_values("mean_product_tsic", ascending=False)
    val.to_csv(OUT / "validation_results.csv", index=False)

    relaxed = val[(val.mean_product_tsic > 0.01) & (val.residual_tsic > 0) & (val.max_parent_corr < 0.85) & (val.net5_annual_return > 0)].copy()
    relaxed["selection_score"] = relaxed.mean_product_tsic + 0.5 * relaxed.residual_tsic + 0.02 * relaxed.mse_gain_pct.clip(-1, 1) + 0.005 * relaxed.net5_sharpe.clip(-3, 3)
    relaxed = relaxed.sort_values("selection_score", ascending=False)
    finalists = []
    for _, row in relaxed.head(6).iterrows():
        cid = row.candidate_id
        finalists.append({"candidate_id": cid, "factor": factor_store[cid], "baseline": baseline_store[cid], "horizon": int(row.horizon), "source_ids": [cid]})

    hold_rows = []
    for item in finalists:
        h = item["horizon"]
        res = evaluate_ts(item["candidate_id"], item["factor"], fwd[h], target_scaled[h], item["baseline"], h, core.HOLD, eligible, ret1)
        yearly = {}
        for year in range(2022, close.index.max().year + 1):
            b = (pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
            vals = product_tsic(item["factor"], fwd[h], b, eligible, min_obs=80)
            yearly[str(year)] = float(vals.mean()) if len(vals) else np.nan
        finite = [v for v in yearly.values() if np.isfinite(v)]
        res["source_ids"] = "||".join(item["source_ids"])
        res["yearly_tsic"] = json.dumps(yearly, sort_keys=True)
        res["positive_year_share"] = float(np.mean([v > 0 for v in finite])) if finite else np.nan
        hold_rows.append(res)
    hadj = core.holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in hold_rows})
    for r in hold_rows:
        r["holm_p"] = hadj.get(r["candidate_id"], np.nan)
        r["final_pass"] = bool(r["mean_product_tsic"] > 0.02 and r["holm_p"] <= 0.10 and r["mse_gain_pct"] > 0 and r["residual_tsic"] > 0.005 and r["max_parent_corr"] < 0.80 and r["net5_annual_return"] > 0 and r["positive_product_share"] >= 0.60 and r["positive_year_share"] >= 0.60)
    hold = pd.DataFrame(hold_rows)
    if len(hold):
        hold = hold.sort_values(["final_pass", "mean_product_tsic"], ascending=[False, False])
    hold.to_csv(OUT / "holdout_results.csv", index=False)
    admitted = hold[hold.final_pass == True] if len(hold) else pd.DataFrame()
    decision = {"status": "admit_historical_default_off" if len(admitted) else "no_catalog_admission", "eligible_count": len(eligible), "validation_pass_count": int(val.validation_pass.sum()) if len(val) else 0, "finalist_count": len(finalists), "holdout_pass_count": int(hold.final_pass.sum()) if len(hold) else 0, "data_end": str(close.index.max().date())}
    if len(admitted):
        decision["selected"] = admitted.iloc[0].to_dict()
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float), encoding="utf-8")
    (OUT / "selected_factor_spec.json").write_text(json.dumps({"decision": decision, "validation": val.to_dict(orient="records"), "holdout": hold.to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=float))


if __name__ == "__main__":
    main()
