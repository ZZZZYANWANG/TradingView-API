from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = Path(os.environ.get("OUT_DIR", "research_outputs/fluid_hodge_futures"))
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
SEED = 20261118
RNG = np.random.default_rng(SEED)

TICKER_GROUP = {
    "ES=F": "equity", "NQ=F": "equity", "YM=F": "equity", "RTY=F": "equity",
    "ZB=F": "rates", "ZN=F": "rates", "ZF=F": "rates",
    "6E=F": "fx", "6J=F": "fx", "6B=F": "fx", "6A=F": "fx", "6C=F": "fx", "DX=F": "fx",
    "CL=F": "energy", "BZ=F": "energy", "NG=F": "energy", "RB=F": "energy", "HO=F": "energy",
    "GC=F": "metals", "SI=F": "metals", "HG=F": "metals", "PL=F": "metals", "PA=F": "metals",
    "ZC=F": "agriculture", "ZS=F": "agriculture", "ZW=F": "agriculture", "ZM=F": "agriculture",
    "ZL=F": "agriculture", "KC=F": "agriculture", "SB=F": "agriculture", "CT=F": "agriculture",
    "CC=F": "agriculture", "LE=F": "livestock", "HE=F": "livestock",
}
TICKERS = list(TICKER_GROUP)
START = "2007-01-01"
DEV = (pd.Timestamp("2010-01-01"), pd.Timestamp("2016-12-31"))
VAL = (pd.Timestamp("2017-01-01"), pd.Timestamp("2021-12-31"))
HOLD = (pd.Timestamp("2022-01-01"), pd.Timestamp("2027-01-01"))
DEV_SUBS = [
    (pd.Timestamp("2010-01-01"), pd.Timestamp("2012-12-31"), "dev_2010_2012"),
    (pd.Timestamp("2013-01-01"), pd.Timestamp("2014-12-31"), "dev_2013_2014"),
    (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"), "dev_2015_2016"),
]
HORIZONS = [5, 10, 20]
GRAPH_K = [3, 5]
GRAPH_LAMBDA = [0.05, 0.20]
DENSITY_SCALE = [0.50, 1.00]
SMOOTH_SPAN = [3, 8]
REATTACH_PRESSURE_SCALE = [1.0, 2.0, 4.0]


def dump_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            out = raw[field].copy()
        elif field in raw.columns.get_level_values(1):
            out = raw.xs(field, axis=1, level=1).copy()
        else:
            return pd.DataFrame(index=raw.index)
    else:
        if len(tickers) != 1 or field not in raw.columns:
            return pd.DataFrame(index=raw.index)
        out = raw[[field]].copy()
        out.columns = tickers
    out.columns = [str(c) for c in out.columns]
    return out.reindex(columns=[c for c in tickers if c in out.columns])


def download_market() -> dict[str, pd.DataFrame]:
    last_error = None
    raw = pd.DataFrame()
    for attempt in range(4):
        try:
            raw = yf.download(
                tickers=TICKERS,
                start=START,
                end=None,
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="column",
                threads=True,
                timeout=60,
            )
            if not raw.empty:
                break
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(5 * (attempt + 1))
    if raw.empty:
        raise RuntimeError(f"yfinance batch download failed: {last_error}")

    fields = {}
    for fld in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        fields[fld] = extract_field(raw, fld, TICKERS)
    close = fields["Adj Close"] if fields["Adj Close"].notna().sum().sum() > 0 else fields["Close"]
    fields["Close"] = close

    missing = [t for t in TICKERS if t not in close.columns or close[t].notna().sum() < 500]
    for ticker in missing:
        for attempt in range(3):
            try:
                one = yf.download(
                    ticker,
                    start=START,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=45,
                )
                if one.empty:
                    continue
                if isinstance(one.columns, pd.MultiIndex):
                    one.columns = one.columns.get_level_values(0)
                for fld in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                    if fld in one:
                        s = one[fld]
                        if fld == "Adj Close":
                            fields["Close"][ticker] = s
                        else:
                            fields[fld][ticker] = s
                break
            except Exception:
                time.sleep(4 * (attempt + 1))

    for key, df in fields.items():
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
        fields[key] = df[~df.index.duplicated(keep="last")].sort_index()
    return fields


def ts_z_prior(x: pd.DataFrame, window: int = 120, min_obs: int = 40) -> pd.DataFrame:
    prior = x.shift(1)
    mu = prior.rolling(window, min_periods=min_obs).mean()
    sd = prior.rolling(window, min_periods=min_obs).std(ddof=0)
    return (x - mu) / (sd + EPS)


def xs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True, method="average") - 0.5


def period_index(index: pd.Index, bounds: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Index:
    return index[(index >= bounds[0]) & (index <= bounds[1])]


def daily_ic_series(factor: pd.DataFrame, target: pd.DataFrame, dates: pd.Index, cols: list[str]) -> pd.Series:
    out = {}
    common_dates = dates.intersection(factor.index).intersection(target.index)
    for dt in common_dates:
        a = factor.loc[dt, cols].to_numpy(dtype=float)
        b = target.loc[dt, cols].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 6:
            continue
        if np.nanstd(a[ok]) <= EPS or np.nanstd(b[ok]) <= EPS:
            continue
        value = spearmanr(a[ok], b[ok]).correlation
        if np.isfinite(value):
            out[dt] = float(value)
    return pd.Series(out, dtype=float).sort_index()


def ic_summary(factor: pd.DataFrame, target: pd.DataFrame, bounds, cols: list[str]) -> dict:
    s = daily_ic_series(factor, target, period_index(factor.index, bounds), cols)
    return {
        "mean": float(s.mean()) if len(s) else np.nan,
        "n_days": int(len(s)),
        "positive_share": float((s > 0).mean()) if len(s) else np.nan,
        "series": s,
    }


def moving_block_bootstrap(series: pd.Series, block: int = 20, reps: int = 2000, seed: int = SEED) -> dict:
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < max(40, 2 * block):
        return {"n": n, "ci_low": np.nan, "ci_high": np.nan, "p_nonpositive": np.nan}
    rng = np.random.default_rng(seed + n)
    starts = np.arange(0, n - block + 1)
    means = np.empty(reps, dtype=float)
    need = int(math.ceil(n / block))
    for r in range(reps):
        chosen = rng.choice(starts, size=need, replace=True)
        sample = np.concatenate([x[s:s + block] for s in chosen])[:n]
        means[r] = sample.mean()
    return {
        "n": n,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_nonpositive": float((means <= 0).mean()),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    clean = [(k, float(v)) for k, v in pvalues.items() if np.isfinite(v)]
    clean.sort(key=lambda kv: kv[1])
    m = len(clean)
    adjusted = {}
    running = 0.0
    for rank, (key, p) in enumerate(clean):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adjusted[key] = running
    return adjusted


def build_graph(ret1: pd.DataFrame, cols: list[str], k: int, lam: float):
    dev_ret = ret1.loc[period_index(ret1.index, DEV), cols]
    corr = dev_ret.corr(min_periods=250).fillna(0.0).to_numpy(dtype=float)
    n = len(cols)
    np.fill_diagonal(corr, 0.0)
    weighted = np.zeros_like(corr)
    for i in range(n):
        order = np.argsort(np.abs(corr[i]))[::-1][: min(k, n - 1)]
        weighted[i, order] = np.abs(corr[i, order])
    W = np.maximum(weighted, weighted.T)
    np.fill_diagonal(W, 0.0)
    Wrow = W / np.maximum(W.sum(axis=1, keepdims=True), EPS)

    lagcorr = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = dev_ret.iloc[:, i]
        for j in range(n):
            if i == j or W[i, j] <= 0:
                continue
            yj = dev_ret.iloc[:, j]
            pair = pd.concat([xi, yj.shift(1)], axis=1).dropna()
            if len(pair) >= 250:
                lagcorr[i, j] = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    A = 0.5 * (lagcorr - lagcorr.T) * (W > 0)
    B = lagcorr * (W > 0)
    B = B / np.maximum(np.abs(B).sum(axis=1, keepdims=True), EPS)
    L = np.diag(W.sum(axis=1)) - W
    pressure_operator = np.linalg.pinv(L + lam * np.eye(n) + np.ones((n, n)) / n)
    return W, Wrow, A, B, pressure_operator


def compute_graph_states(
    ret1: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    cols: list[str],
    k: int,
    lam: float,
    density_scale: float,
) -> dict[str, pd.DataFrame]:
    W, Wrow, A, B, pressure_operator = build_graph(ret1, cols, k, lam)
    r = ret1[cols]
    vol_prior = r.shift(1).rolling(20, min_periods=10).std(ddof=0)
    u = (r / (vol_prior + EPS)).clip(-8.0, 8.0)
    u = u.sub(u.median(axis=1), axis=0)
    log_dollar_volume = np.log1p((close[cols] * volume[cols]).clip(lower=0.0))
    volume_z = ts_z_prior(log_dollar_volume, 60, 20).clip(-4.0, 4.0)
    range_state = ts_z_prior(np.log(high[cols] / low[cols]), 60, 20).clip(-4.0, 4.0)
    rho = np.exp((density_scale * volume_z).clip(-2.5, 2.5))

    U = u.fillna(0.0).to_numpy(dtype=float)
    RHO = rho.fillna(1.0).to_numpy(dtype=float)
    valid = close[cols].notna().to_numpy()
    t_count, n = U.shape
    div = np.full((t_count, n), np.nan)
    pressure = np.full((t_count, n), np.nan)
    gamma = np.full((t_count, n), np.nan)
    grad_p_energy = np.full((t_count, n), np.nan)
    grad_rho_energy = np.full((t_count, n), np.nan)
    misalignment = np.full((t_count, n), np.nan)

    for t in range(t_count):
        uu = U[t]
        rr = RHO[t]
        du = uu[None, :] - uu[:, None]
        mean_rho = 0.5 * (rr[:, None] + rr[None, :])
        edge_flux = W * mean_rho * du
        d = edge_flux.sum(axis=1)
        p = pressure_operator @ (-d)
        p = p - np.nanmean(p)
        g = A @ (rr * uu)
        dp = p[None, :] - p[:, None]
        dr = rr[None, :] - rr[:, None]
        gp = np.sum(W * dp * dp, axis=1)
        gr = np.sum(W * dr * dr, axis=1)
        cross = np.sum(W * dp * dr, axis=1)
        mis = np.sqrt(np.maximum(gp * gr - cross * cross, 0.0))
        mask = valid[t]
        d[~mask] = np.nan
        p[~mask] = np.nan
        g[~mask] = np.nan
        gp[~mask] = np.nan
        gr[~mask] = np.nan
        mis[~mask] = np.nan
        div[t], pressure[t], gamma[t] = d, p, g
        grad_p_energy[t], grad_rho_energy[t], misalignment[t] = gp, gr, mis

    idx = ret1.index
    DIV = pd.DataFrame(div, index=idx, columns=cols)
    P = pd.DataFrame(pressure, index=idx, columns=cols)
    GAMMA = pd.DataFrame(gamma, index=idx, columns=cols)
    GPE = pd.DataFrame(grad_p_energy, index=idx, columns=cols)
    GRE = pd.DataFrame(grad_rho_energy, index=idx, columns=cols)
    MIS = pd.DataFrame(misalignment, index=idx, columns=cols)
    PEER = pd.DataFrame(U @ Wrow.T, index=idx, columns=cols).where(close[cols].notna())
    LEAD = pd.DataFrame(U @ B.T, index=idx, columns=cols).where(close[cols].notna())
    SEP = (u - PEER).abs()
    ENSTROPHY = GAMMA * GAMMA
    VISCOSITY = np.exp((0.5 * range_state - 0.5 * volume_z).clip(-2.0, 2.0))
    mom20 = np.log(close[cols] / close[cols].shift(20))
    pressure_z = ts_z_prior(P, 120, 40).clip(-6.0, 6.0)
    sep_lag = SEP.shift(1)
    reattach_core = np.sign(mom20) * sep_lag * (sep_lag - SEP).clip(lower=0.0)

    raw = {
        "pressure_dilatation": -P * DIV,
        "baroclinic_vorticity": np.sign(GAMMA) * MIS / (1.0 + GPE + GRE),
        "potential_curl_conversion": (0.5 * GPE.shift(1) - 0.5 * GPE) + (ENSTROPHY - ENSTROPHY.shift(1)),
        "circulation_transport": GAMMA * PEER,
        "enstrophy_dissipation": -ENSTROPHY * VISCOSITY,
        "leadlag_advection_gap": LEAD - u,
        "hodge_energy_imbalance": (GPE - ENSTROPHY) / (GPE + ENSTROPHY + EPS),
    }
    for scale in REATTACH_PRESSURE_SCALE:
        raw[f"pressure_reattachment_s{scale:g}"] = reattach_core * np.exp(-pressure_z.abs() / scale)

    return {
        "raw": raw,
        "divergence": DIV,
        "pressure": P,
        "gamma": GAMMA,
        "separation": SEP,
        "lead_prediction": LEAD,
        "peer_velocity": PEER,
        "volume_z": volume_z,
        "range_z": range_state,
        "grad_p_energy": GPE,
        "enstrophy": ENSTROPHY,
        "W": W,
    }


def mean_parent_correlation(factor: pd.DataFrame, features: dict[str, pd.DataFrame], bounds, cols: list[str]) -> tuple[float, str]:
    dates = period_index(factor.index, bounds)
    best_value, best_name = -np.inf, ""
    for name, feat in features.items():
        vals = []
        for dt in dates:
            a = factor.loc[dt, cols].to_numpy(dtype=float)
            b = feat.loc[dt, cols].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 6:
                continue
            c = spearmanr(a[ok], b[ok]).correlation
            if np.isfinite(c):
                vals.append(abs(float(c)))
        value = float(np.mean(vals)) if vals else np.nan
        if np.isfinite(value) and value > best_value:
            best_value, best_name = value, name
    return best_value, best_name


def build_baseline(
    ret1: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cols: list[str],
    states: dict,
) -> dict[str, pd.DataFrame]:
    r = ret1[cols]
    market = r.median(axis=1)
    market_var = market.rolling(60, min_periods=30).var(ddof=0)
    beta60 = r.rolling(60, min_periods=30).cov(market).div(market_var + EPS, axis=0)
    corr60 = r.rolling(60, min_periods=30).corr(market)
    logdv = np.log1p((close[cols] * volume[cols]).clip(lower=0.0))
    features = {
        "rev1": -r,
        "mom5": np.log(close[cols] / close[cols].shift(5)),
        "mom20": np.log(close[cols] / close[cols].shift(20)),
        "mom60": np.log(close[cols] / close[cols].shift(60)),
        "mom120": np.log(close[cols] / close[cols].shift(120)),
        "vol20": r.rolling(20, min_periods=10).std(ddof=0),
        "vol60": r.rolling(60, min_periods=30).std(ddof=0),
        "range_z": ts_z_prior(np.log(high[cols] / low[cols]), 60, 20),
        "volume_z20": ts_z_prior(logdv, 20, 10),
        "volume_z60": ts_z_prior(logdv, 60, 20),
        "beta60": beta60,
        "corr_market60": corr60,
        "peer_mom20": states["peer_velocity"].rolling(20, min_periods=10).sum(),
        "divergence": states["divergence"],
        "pressure": states["pressure"],
        "circulation": states["gamma"],
        "separation": states["separation"],
        "lead_prediction": states["lead_prediction"],
        "pressure_energy": states["grad_p_energy"],
        "enstrophy": states["enstrophy"],
    }
    return {k: xs_rank(v[cols]) for k, v in features.items()}


def stack_data(features: dict[str, pd.DataFrame], target: pd.DataFrame, bounds, cols: list[str]):
    dates = period_index(target.index, bounds)
    names = list(features)
    Xs, ys, keys = [], [], []
    for dt in dates:
        X = np.column_stack([features[n].loc[dt, cols].to_numpy(dtype=float) for n in names])
        y = target.loc[dt, cols].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() == 0:
            continue
        Xs.append(X[ok])
        ys.append(y[ok])
        keys.extend([(dt, cols[i]) for i in np.where(ok)[0]])
    if not Xs:
        return np.empty((0, len(names))), np.empty(0), [], names
    return np.vstack(Xs), np.concatenate(ys), keys, names


def model_increment(factor: pd.DataFrame, target: pd.DataFrame, baseline: dict[str, pd.DataFrame], horizon: int, test_bounds, cols: list[str]) -> dict:
    ranked_target = xs_rank(target)
    ranked_factor = xs_rank(factor)
    Xtr, ytr, _, _ = stack_data(baseline, ranked_target, DEV, cols)
    Xte, yte, keys, _ = stack_data(baseline, ranked_target, test_bounds, cols)
    ftr_map = ranked_factor.stack(dropna=False)
    fte_map = ranked_factor.stack(dropna=False)
    ftr = np.array([ftr_map.get(key, np.nan) for key in [(d, c) for d in period_index(ranked_target.index, DEV) for c in cols]], dtype=float)

    # Rebuild aligned factor vectors with the same keys as stack_data.
    _, _, train_keys, _ = stack_data(baseline, ranked_target, DEV, cols)
    ftr = np.array([ftr_map.get(k, np.nan) for k in train_keys], dtype=float)
    fte = np.array([fte_map.get(k, np.nan) for k in keys], dtype=float)
    ok_tr = np.isfinite(ftr)
    ok_te = np.isfinite(fte)
    Xtr, ytr, ftr = Xtr[ok_tr], ytr[ok_tr], ftr[ok_tr]
    Xte, yte, fte = Xte[ok_te], yte[ok_te], fte[ok_te]
    keys = [k for k, ok in zip(keys, ok_te) if ok]
    if len(ytr) < 1000 or len(yte) < 500:
        return {"mse_gain_pct": np.nan, "reconstruction_r2": np.nan, "residual_rankic": np.nan}

    base_model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    plus_model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    base_model.fit(Xtr, ytr)
    plus_model.fit(np.column_stack([Xtr, ftr]), ytr)
    p0 = base_model.predict(Xte)
    p1 = plus_model.predict(np.column_stack([Xte, fte]))
    mse0, mse1 = mean_squared_error(yte, p0), mean_squared_error(yte, p1)

    recon = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    recon.fit(Xtr, ftr)
    fhat = recon.predict(Xte)
    residual = fte - fhat
    r2 = r2_score(fte, fhat)
    residual_panel = pd.DataFrame(np.nan, index=factor.index, columns=cols)
    for key, value in zip(keys, residual):
        residual_panel.loc[key[0], key[1]] = value
    ric = ic_summary(residual_panel, target, test_bounds, cols)["mean"]
    return {
        "mse_gain_pct": float(100.0 * (mse0 - mse1) / (mse0 + EPS)),
        "reconstruction_r2": float(r2),
        "residual_rankic": float(ric),
        "base_mse": float(mse0),
        "plus_mse": float(mse1),
    }


def portfolio_diagnostic(factor: pd.DataFrame, ret1: pd.DataFrame, horizon: int, bounds, cols: list[str], cost_bp: float) -> dict:
    ranks = factor[cols].rank(axis=1, pct=True)
    signal = pd.DataFrame(0.0, index=factor.index, columns=cols)
    long_mask = ranks >= 0.80
    short_mask = ranks <= 0.20
    long_count = long_mask.sum(axis=1).replace(0, np.nan)
    short_count = short_mask.sum(axis=1).replace(0, np.nan)
    signal = signal.add(long_mask.div(long_count, axis=0) * 0.5, fill_value=0.0)
    signal = signal.sub(short_mask.div(short_count, axis=0) * 0.5, fill_value=0.0)
    weights = signal.shift(1).rolling(horizon, min_periods=1).mean().fillna(0.0)
    pnl = (weights * ret1[cols]).sum(axis=1, min_count=1)
    turnover = 0.5 * (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = pnl - turnover * cost_bp * 1e-4
    dates = period_index(net.index, bounds)
    x = net.loc[dates].dropna()
    if len(x) < 100:
        return {"annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "turnover": np.nan}
    equity = np.exp(x.cumsum())
    drawdown = equity / equity.cummax() - 1.0
    return {
        "annual_return": float(x.mean() * 252),
        "annual_vol": float(x.std(ddof=0) * np.sqrt(252)),
        "sharpe": float(x.mean() / (x.std(ddof=0) + EPS) * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.loc[dates].mean()),
        "n_days": int(len(x)),
    }


def dev_score(factor: pd.DataFrame, target: pd.DataFrame, high_pool: list[str], low_pool: list[str]) -> tuple[float, int, list[float]]:
    overall = ic_summary(factor, target, DEV, high_pool + low_pool)["mean"]
    if not np.isfinite(overall) or abs(overall) < 1e-6:
        return -np.inf, 1, []
    direction = 1 if overall > 0 else -1
    oriented = direction * factor
    vals = []
    for start, end, _ in DEV_SUBS:
        for pool in [high_pool, low_pool]:
            value = ic_summary(oriented, target, (start, end), pool)["mean"]
            if np.isfinite(value):
                vals.append(float(value))
    if len(vals) < 6:
        return -np.inf, direction, vals
    score = min(vals) + 0.25 * float(np.mean(vals))
    return float(score), direction, vals


def evaluate(
    candidate_id: str,
    factor: pd.DataFrame,
    horizon: int,
    target: pd.DataFrame,
    baseline: dict[str, pd.DataFrame],
    bounds,
    cols: list[str],
    ret1: pd.DataFrame,
) -> dict:
    direct = ic_summary(factor, target, bounds, cols)
    boot = moving_block_bootstrap(direct["series"], block=max(10, horizon), reps=2000, seed=SEED + horizon)
    model = model_increment(factor, target, baseline, horizon, bounds, cols)
    max_corr, max_parent = mean_parent_correlation(factor, baseline, bounds, cols)
    port2 = portfolio_diagnostic(factor, ret1, horizon, bounds, cols, 2.0)
    port5 = portfolio_diagnostic(factor, ret1, horizon, bounds, cols, 5.0)
    return {
        "candidate_id": candidate_id,
        "horizon": horizon,
        "rankic": direct["mean"],
        "n_ic_days": direct["n_days"],
        "positive_share": direct["positive_share"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_nonpositive": boot["p_nonpositive"],
        "mse_gain_pct": model.get("mse_gain_pct"),
        "reconstruction_r2": model.get("reconstruction_r2"),
        "residual_rankic": model.get("residual_rankic"),
        "max_parent_corr": max_corr,
        "max_parent_name": max_parent,
        "net2_annual_return": port2["annual_return"],
        "net2_sharpe": port2["sharpe"],
        "net5_annual_return": port5["annual_return"],
        "net5_sharpe": port5["sharpe"],
        "net5_max_drawdown": port5["max_drawdown"],
        "avg_turnover": port5["turnover"],
    }


def yearly_rankic(factor: pd.DataFrame, target: pd.DataFrame, bounds, cols: list[str]) -> dict[str, float]:
    out = {}
    for year in range(bounds[0].year, min(bounds[1].year, factor.index.max().year) + 1):
        b = (pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
        out[str(year)] = ic_summary(factor, target, b, cols)["mean"]
    return out


def main() -> None:
    fields = download_market()
    open_, high, low, close, volume = [fields[k] for k in ["Open", "High", "Low", "Close", "Volume"]]
    common_index = close.index
    for df in [open_, high, low, volume]:
        common_index = common_index.union(df.index)
    open_, high, low, close, volume = [df.reindex(common_index).sort_index() for df in [open_, high, low, close, volume]]

    ret1_all = np.log(close / close.shift(1))
    eligible = []
    coverage_rows = []
    for ticker in close.columns:
        counts = {}
        for label, bounds in [("dev", DEV), ("validation", VAL), ("holdout", HOLD)]:
            idx = period_index(close.index, bounds)
            counts[label] = int(close.loc[idx, ticker].notna().sum())
        coverage_rows.append({"ticker": ticker, **counts})
        if counts["dev"] >= 900 and counts["validation"] >= 800 and counts["holdout"] >= 600:
            eligible.append(ticker)
    if len(eligible) < 14:
        raise RuntimeError(f"only {len(eligible)} eligible futures: {eligible}")

    open_, high, low, close, volume = [df[eligible] for df in [open_, high, low, close, volume]]
    ret1 = np.log(close / close.shift(1))
    fwd = {h: np.log(close.shift(-h) / open_.shift(-1)) for h in HORIZONS}
    dev_dv = (close * volume).loc[period_index(close.index, DEV)].median()
    ordered = list(dev_dv.sort_values(ascending=False).index)
    split = max(7, len(ordered) // 2)
    high_pool, low_pool = ordered[:split], ordered[split:]

    dump_json("data_audit.json", {
        "source": "Yahoo Finance continuous futures OHLCV",
        "download_start": START,
        "rows": int(len(close)),
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "eligible_tickers": eligible,
        "groups": {t: TICKER_GROUP.get(t, "unknown") for t in eligible},
        "high_liquidity_pool": high_pool,
        "secondary_liquidity_pool": low_pool,
        "coverage": coverage_rows,
        "development": [str(DEV[0].date()), str(DEV[1].date())],
        "validation": [str(VAL[0].date()), str(VAL[1].date())],
        "holdout": [str(HOLD[0].date()), str(HOLD[1].date())],
    })

    winners = {}
    scan_rows = []
    states_cache = {}

    for k in GRAPH_K:
        for lam in GRAPH_LAMBDA:
            for density_scale in DENSITY_SCALE:
                config_key = f"k{k}_l{lam:g}_d{density_scale:g}"
                states = compute_graph_states(ret1, close, high, low, volume, eligible, k, lam, density_scale)
                states_cache[config_key] = states
                for raw_name, raw_factor in states["raw"].items():
                    family = raw_name.split("_s")[0] if raw_name.startswith("pressure_reattachment") else raw_name
                    for smooth in SMOOTH_SPAN:
                        factor = ts_z_prior(raw_factor, 120, 40).clip(-8.0, 8.0)
                        factor = factor.ewm(span=smooth, adjust=False, min_periods=max(2, smooth // 2)).mean()
                        for horizon in HORIZONS:
                            score, direction, cells = dev_score(factor, fwd[horizon], high_pool, low_pool)
                            cid = f"{family}|{config_key}|raw={raw_name}|smooth={smooth}|h={horizon}|dir={direction}"
                            row = {
                                "candidate_id": cid,
                                "family": family,
                                "config": config_key,
                                "raw_name": raw_name,
                                "smooth": smooth,
                                "horizon": horizon,
                                "direction": direction,
                                "dev_score": score,
                                "dev_min_cell": min(cells) if cells else np.nan,
                                "dev_mean_cell": float(np.mean(cells)) if cells else np.nan,
                            }
                            scan_rows.append(row)
                            if np.isfinite(score) and (family not in winners or score > winners[family]["dev_score"]):
                                winners[family] = {
                                    **row,
                                    "factor": direction * factor,
                                }

    pd.DataFrame(scan_rows).sort_values("dev_score", ascending=False).to_csv(OUT / "candidate_scan.csv", index=False)
    pd.DataFrame([{k: v for k, v in item.items() if k != "factor"} for item in winners.values()]).to_csv(OUT / "family_winners.csv", index=False)

    validation_rows = []
    factor_map = {}
    baseline_map = {}
    for family, winner in winners.items():
        states = states_cache[winner["config"]]
        baseline = build_baseline(ret1, open_, high, low, close, volume, eligible, states)
        result = evaluate(winner["candidate_id"], winner["factor"], winner["horizon"], fwd[winner["horizon"]], baseline, VAL, eligible, ret1)
        result.update({"family": family, "config": winner["config"], "raw_name": winner["raw_name"], "smooth": winner["smooth"], "direction": winner["direction"]})
        validation_rows.append(result)
        factor_map[winner["candidate_id"]] = winner["factor"]
        baseline_map[winner["candidate_id"]] = baseline

    p_adj = holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in validation_rows})
    for row in validation_rows:
        row["holm_p"] = p_adj.get(row["candidate_id"], np.nan)
        row["validation_pass"] = bool(
            row["rankic"] > 0.01
            and row["holm_p"] <= 0.10
            and row["mse_gain_pct"] > 0.0
            and row["residual_rankic"] > 0.005
            and row["max_parent_corr"] < 0.80
            and row["net5_annual_return"] > 0.0
        )
    validation_df = pd.DataFrame(validation_rows).sort_values("rankic", ascending=False)
    validation_df.to_csv(OUT / "validation_results.csv", index=False)

    eligible_for_final = validation_df[
        (validation_df["rankic"] > 0.005)
        & (validation_df["residual_rankic"] > 0.0)
        & (validation_df["max_parent_corr"] < 0.90)
        & (validation_df["net5_annual_return"] > 0.0)
    ].copy()
    eligible_for_final["selection_score"] = (
        eligible_for_final["rankic"]
        + 0.5 * eligible_for_final["residual_rankic"]
        + 0.02 * eligible_for_final["mse_gain_pct"].clip(-1, 1)
        + 0.005 * eligible_for_final["net5_sharpe"].clip(-3, 3)
    )
    eligible_for_final = eligible_for_final.sort_values("selection_score", ascending=False)

    finalists = []
    if len(eligible_for_final):
        best = eligible_for_final.iloc[0]
        cid = best["candidate_id"]
        finalists.append({
            "finalist_id": "best_individual",
            "source_ids": [cid],
            "factor": factor_map[cid],
            "horizon": int(best["horizon"]),
            "baseline": baseline_map[cid],
        })

        selected = []
        for _, row in eligible_for_final.iterrows():
            cid2 = row["candidate_id"]
            f2 = factor_map[cid2]
            accept = True
            for prior_id in selected:
                dates = period_index(f2.index, VAL)
                corrs = []
                for dt in dates:
                    a = f2.loc[dt, eligible].to_numpy(float)
                    b = factor_map[prior_id].loc[dt, eligible].to_numpy(float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() >= 6:
                        c = spearmanr(a[ok], b[ok]).correlation
                        if np.isfinite(c):
                            corrs.append(abs(c))
                if corrs and np.mean(corrs) >= 0.75:
                    accept = False
                    break
            if accept:
                selected.append(cid2)
            if len(selected) >= 4:
                break
        if len(selected) >= 2:
            h_counts = pd.Series([int(validation_df.set_index("candidate_id").loc[c, "horizon"]) for c in selected]).value_counts()
            horizon = int(h_counts.index[0])
            components = [xs_rank(factor_map[c]) for c in selected]
            consensus = sum(components) / len(components)
            base_id = selected[0]
            finalists.append({
                "finalist_id": "diversified_hodge_consensus",
                "source_ids": selected,
                "factor": consensus,
                "horizon": horizon,
                "baseline": baseline_map[base_id],
            })

    holdout_rows = []
    holdout_objects = {}
    for finalist in finalists:
        h = finalist["horizon"]
        result = evaluate(finalist["finalist_id"], finalist["factor"], h, fwd[h], finalist["baseline"], HOLD, eligible, ret1)
        years = yearly_rankic(finalist["factor"], fwd[h], HOLD, eligible)
        result["source_ids"] = "||".join(finalist["source_ids"])
        result["yearly_rankic"] = json.dumps(years, sort_keys=True)
        result["positive_year_share"] = float(np.mean([v > 0 for v in years.values() if np.isfinite(v)])) if years else np.nan
        holdout_rows.append(result)
        holdout_objects[finalist["finalist_id"]] = finalist

    if holdout_rows:
        hold_adj = holm_adjust({r["candidate_id"]: r["p_nonpositive"] for r in holdout_rows})
        for row in holdout_rows:
            row["holm_p"] = hold_adj.get(row["candidate_id"], np.nan)
            row["final_pass"] = bool(
                row["rankic"] > 0.015
                and row["holm_p"] <= 0.10
                and row["mse_gain_pct"] > 0.0
                and row["residual_rankic"] > 0.005
                and row["max_parent_corr"] < 0.80
                and row["net5_annual_return"] > 0.0
                and row["positive_year_share"] >= 0.60
            )
    holdout_df = pd.DataFrame(holdout_rows)
    if len(holdout_df):
        holdout_df = holdout_df.sort_values(["final_pass", "rankic"], ascending=[False, False])
    holdout_df.to_csv(OUT / "holdout_results.csv", index=False)

    admitted = holdout_df[holdout_df["final_pass"] == True] if len(holdout_df) else pd.DataFrame()
    if len(admitted):
        chosen = admitted.iloc[0]
        fid = chosen["candidate_id"]
        obj = holdout_objects[fid]
        status = "admit_historical_default_off"
        selected_spec = {
            "factor_id": "graph_hodge_pressure_reattachment_consensus_20d",
            "name": "Graph Hodge pressure-reattachment consensus",
            "status": status,
            "primary_horizon": int(obj["horizon"]),
            "source_ids": obj["source_ids"],
            "validation": validation_df.to_dict(orient="records"),
            "holdout": chosen.to_dict(),
            "default_enabled": False,
            "independent_alpha": False,
            "fresh_holdout_required": True,
        }
    else:
        status = "no_catalog_admission"
        selected_spec = {
            "status": status,
            "reason": "no finalist passed holdout statistical, incremental, independence and 5bp cost gates simultaneously",
            "validation": validation_df.to_dict(orient="records"),
            "holdout": holdout_df.to_dict(orient="records") if len(holdout_df) else [],
        }

    dump_json("selected_factor_spec.json", selected_spec)
    dump_json("decision.json", {
        "status": status,
        "family_count": len(winners),
        "validation_pass_count": int(validation_df["validation_pass"].sum()) if len(validation_df) else 0,
        "finalist_count": len(finalists),
        "holdout_pass_count": int(holdout_df["final_pass"].sum()) if len(holdout_df) else 0,
        "data_end": str(close.index.max().date()),
        "eligible_tickers": eligible,
    })

    readme = [
        "# Fluid Hodge Futures Research",
        "",
        f"- Eligible futures: {len(eligible)}",
        f"- Data range: {close.index.min().date()} to {close.index.max().date()}",
        f"- Development: {DEV[0].date()} to {DEV[1].date()}",
        f"- Validation: {VAL[0].date()} to {VAL[1].date()}",
        f"- Holdout: {HOLD[0].date()} onward",
        f"- Decision: **{status}**",
        "",
        "The study freezes a development-period correlation and lead-lag graph, constructs graph pressure, divergence, circulation, baroclinic, conversion, dissipation, advection and reattachment coordinates, then applies validation and untouched holdout gates.",
    ]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"decision": status, "eligible": len(eligible), "validation_pass": int(validation_df["validation_pass"].sum()), "holdout_pass": int(holdout_df["final_pass"].sum()) if len(holdout_df) else 0}, indent=2))


if __name__ == "__main__":
    main()
