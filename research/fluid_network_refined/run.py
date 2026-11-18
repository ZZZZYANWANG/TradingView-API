from __future__ import annotations

import io
import json
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUT = Path("research_outputs/fluid_network_refined")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
RNG = np.random.default_rng(20261118)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "ETCUSDT", "XLMUSDT", "DOTUSDT", "SOLUSDT",
]
HORIZONS = [12, 24, 72, 120, 480]


def causal_z(s: pd.Series, window: int = 720, min_periods: int = 240) -> pd.Series:
    p = s.shift(1)
    return ((s - p.rolling(window, min_periods).mean()) / (p.rolling(window, min_periods).std(ddof=0) + EPS)).clip(-12, 12)


def zframe(x: pd.DataFrame, window: int = 720, min_periods: int = 240) -> pd.DataFrame:
    return x.apply(lambda s: causal_z(s, window, min_periods))


def norm_momentum(s: pd.Series, w: int) -> pd.Series:
    return (s.rolling(w, w).sum() / np.sqrt(s.pow(2).rolling(w, w).sum() + EPS)).clip(-12, 12)


def download_month(symbol: str, period: str):
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{period}.zip"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=40, headers={"User-Agent": "fluid-network-refined/1.0"})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                name = [x for x in zf.namelist() if x.endswith(".csv")][0]
                x = pd.read_csv(zf.open(name), header=None, low_memory=False).iloc[:, :12]
            x.columns = ["ot", "open", "high", "low", "close", "volume", "ct", "qv", "trades", "tb", "tbq", "ignore"]
            x["ot"] = pd.to_numeric(x["ot"], errors="coerce")
            x = x[x.ot.notna()].copy()
            for col in ["open", "high", "low", "close", "qv", "trades", "tbq"]:
                x[col] = pd.to_numeric(x[col], errors="coerce")
            x["time"] = pd.to_datetime(x.ot.astype("int64"), unit="ms", utc=True)
            x["symbol"] = symbol
            return x[["time", "symbol", "open", "high", "low", "close", "qv", "trades", "tbq"]]
        except Exception:
            time.sleep(attempt + 1)
    return None


def load_data():
    periods = [str(x) for x in pd.period_range("2021-01", "2026-10", freq="M")]
    frames = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(download_month, s, p) for s in SYMBOLS for p in periods]
        for i, fut in enumerate(as_completed(futures), 1):
            x = fut.result()
            if x is not None and not x.empty:
                frames.append(x)
            if i % 100 == 0:
                print(f"download={i}/{len(futures)} usable={len(frames)}", flush=True)
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["time", "symbol"]).sort_values(["time", "symbol"])
    count = d.groupby("symbol").size()
    keep = sorted(count[count >= 40000].index.tolist())
    return d[d.symbol.isin(keep)].copy(), keep


def wide(d: pd.DataFrame, col: str, symbols: list[str]) -> pd.DataFrame:
    return d.pivot(index="time", columns="symbol", values=col).sort_index().reindex(columns=symbols)


def csic_series(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    xr = x.rank(axis=1, method="average", pct=True)
    yr = y.rank(axis=1, method="average", pct=True)
    mask = xr.notna() & yr.notna()
    n = mask.sum(axis=1)
    xm = xr.where(mask).mean(axis=1)
    ym = yr.where(mask).mean(axis=1)
    xc = xr.sub(xm, axis=0).where(mask)
    yc = yr.sub(ym, axis=0).where(mask)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    out = num / den.replace(0.0, np.nan)
    return out.where(n >= 8)


def event_age(event: pd.DataFrame) -> pd.DataFrame:
    out = np.full(event.shape, np.nan)
    a = event.to_numpy(dtype=bool)
    for j in range(a.shape[1]):
        last = -10**9
        for i in range(a.shape[0]):
            if a[i, j]:
                last = i
            if last > -10**8:
                out[i, j] = i - last
    return pd.DataFrame(out, index=event.index, columns=event.columns)


def block_bootstrap(s: pd.Series, reps: int = 2000, block: int = 14):
    a = s.dropna().resample("1D").mean().dropna().to_numpy()
    n = len(a)
    draws = np.empty(reps)
    for k in range(reps):
        z = []
        while len(z) < n:
            st = int(RNG.integers(n))
            z.extend(a[(st + np.arange(block)) % n].tolist())
        draws[k] = np.mean(z[:n])
    return {
        "mean": float(np.mean(a)), "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)), "p_nonpositive": float(np.mean(draws <= 0)),
    }


def holm(pvals: dict[str, float]):
    q = sorted(pvals.items(), key=lambda z: z[1])
    m = len(q)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(q):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def stack_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([x.stack().rename(k) for k, x in frames.items()], axis=1).reset_index(names=["time", "symbol"])


def ridge_increment(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, parents: list[str]):
    tr = train.dropna(subset=parents + [factor, target])
    te = test.dropna(subset=parents + [factor, target])
    sc0 = StandardScaler().fit(tr[parents])
    sc1 = StandardScaler().fit(tr[parents + [factor]])
    m0 = Ridge(alpha=25).fit(sc0.transform(tr[parents]), tr[target])
    m1 = Ridge(alpha=25).fit(sc1.transform(tr[parents + [factor]]), tr[target])
    p0 = m0.predict(sc0.transform(te[parents]))
    p1 = m1.predict(sc1.transform(te[parents + [factor]]))
    y = te[target].to_numpy()
    mse0 = np.mean((y - p0) ** 2)
    mse1 = np.mean((y - p1) ** 2)
    return float(100 * (mse0 - mse1) / (mse0 + EPS))


def residual_diagnostics(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, parents: list[str]):
    tr = train.dropna(subset=parents + [factor])
    te = test.dropna(subset=parents + [factor, target]).copy()
    sc = StandardScaler().fit(tr[parents])
    model = Ridge(alpha=25).fit(sc.transform(tr[parents]), tr[factor])
    fit = model.predict(sc.transform(tr[parents]))
    pred = model.predict(sc.transform(te[parents]))
    r2 = 1 - np.mean((tr[factor].to_numpy() - fit) ** 2) / (np.var(tr[factor].to_numpy()) + EPS)
    te["resid"] = te[factor].to_numpy() - pred
    rx = te.pivot(index="time", columns="symbol", values="resid")
    ry = te.pivot(index="time", columns="symbol", values=target)
    corr = max(abs(tr[factor].corr(tr[p], method="spearman")) for p in parents)
    return float(r2), float(csic_series(rx, ry).mean()), float(corr)


def portfolio(df: pd.DataFrame, factor: str, horizon: int, cost_bp: float):
    x = df.pivot(index="time", columns="symbol", values=factor).sort_index()
    r = df.pivot(index="time", columns="symbol", values="nextret").sort_index()
    rank = x.rank(axis=1, pct=True)
    w = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    w[rank >= .75] = 1.0
    w[rank <= .25] = -1.0
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sleeve = w.shift(1).rolling(horizon, min_periods=1).mean()
    gross = (sleeve * r).sum(axis=1)
    turnover = sleeve.diff().abs().sum(axis=1)
    pnl = gross - turnover * cost_bp * 1e-4
    ann = 24 * 365
    vol = pnl.std(ddof=0) * math.sqrt(ann)
    return {
        f"ret_{int(cost_bp)}bp": float(pnl.mean() * ann),
        f"sharpe_{int(cost_bp)}bp": float(pnl.mean() * ann / (vol + EPS)),
        "avg_turnover": float(turnover.mean()),
    }


def nonoverlap_share(ic: pd.Series, horizon: int):
    epoch = ((ic.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(hours=1)).astype(int)
    offsets = np.unique(np.linspace(0, horizon - 1, min(20, horizon), dtype=int))
    vals = [ic[(epoch % horizon) == off].mean() for off in offsets]
    return float(np.mean(np.asarray(vals) > 0)), [float(x) for x in vals]


def main():
    data, symbols = load_data()
    fields = {c: wide(data, c, symbols) for c in ["open", "high", "low", "close", "qv", "trades", "tbq"]}
    complete = fields["close"].notna().all(axis=1)
    fields = {k: v.loc[complete] for k, v in fields.items()}
    close = fields["close"]
    ret = np.log(close).diff()
    scale = np.sqrt(ret.shift(1).pow(2).ewm(span=48, min_periods=24, adjust=False).mean())
    u = (ret / (scale + EPS)).clip(-12, 12)
    market = u.median(axis=1)
    graph_train = u.index < pd.Timestamp("2023-01-01", tz="UTC")
    beta = u.loc[graph_train].apply(lambda s: s.cov(market.loc[graph_train]) / (market.loc[graph_train].var() + EPS))
    v = u - market.to_numpy()[:, None] * beta.to_numpy()[None, :]
    logq = np.log1p(fields["qv"].clip(lower=0))
    qz = logq.apply(lambda s: causal_z(s, 168, 72))
    rho = np.exp(qz.clip(-3, 3))
    imb = (2 * fields["tbq"] / (fields["qv"] + EPS) - 1).clip(-1, 1)
    range_z = np.log(fields["high"] / fields["low"]).rolling(24, 18).mean().apply(lambda s: causal_z(s, 336, 120))

    dv = v.loc[graph_train].dropna()
    corr = dv.corr().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    W = np.zeros_like(corr)
    for i in range(len(symbols)):
        ids = np.argsort(np.maximum(corr[i], 0))[-3:]
        W[i, ids] = np.maximum(corr[i, ids], 0)
    W = (W + W.T) / 2
    W = W / (W.sum(axis=1, keepdims=True) + EPS)
    L = np.eye(len(symbols)) - W
    pressure_solver = np.linalg.pinv(L + .15 * np.eye(len(symbols)))
    X = dv.shift(1).dropna()
    Y = dv.loc[X.index]
    reg = Ridge(alpha=25).fit(X, Y)
    B = reg.coef_.copy()
    np.fill_diagonal(B, 0)

    va, ra = v.to_numpy(), rho.to_numpy()
    peer = va @ W.T
    adv = va @ B.T - va
    div = np.full_like(va, np.nan)
    for t in range(len(va)):
        if np.isfinite(va[t]).all() and np.isfinite(ra[t]).all():
            flux = W * (.5 * (ra[t, :, None] + ra[t, None, :])) * (va[t, None, :] - va[t, :, None])
            div[t] = flux.sum(axis=1)
    centered = div - np.nanmean(div, axis=1, keepdims=True)
    pressure = np.full_like(centered, np.nan)
    good = np.isfinite(centered).all(axis=1)
    pressure[good] = -centered[good] @ pressure_solver.T
    continuity = -(np.log(rho + EPS).diff().to_numpy() + div)
    tau = (va * va) @ W.T - peer * peer
    subgrid = tau * (peer - va)
    separation = np.abs(va - peer)

    idx, cols = v.index, v.columns
    raw = {
        "adv_z": zframe(pd.DataFrame(adv, index=idx, columns=cols)),
        "pressure_z": zframe(pd.DataFrame(pressure, index=idx, columns=cols)),
        "continuity_z": zframe(pd.DataFrame(continuity, index=idx, columns=cols)),
        "subgrid_z": zframe(pd.DataFrame(subgrid, index=idx, columns=cols)),
        "separation_z": zframe(pd.DataFrame(separation, index=idx, columns=cols)),
    }
    prior_sep = raw["separation_z"].shift(1).clip(lower=0)
    relief = (prior_sep - raw["separation_z"]).clip(lower=0)
    base_reattach = -v * prior_sep * relief * np.exp(-np.abs(range_z) / 2)

    candidates: dict[str, dict[str, pd.DataFrame]] = {}
    families: dict[str, list[str]] = {}
    def add(family: str, name: str, x: pd.DataFrame):
        candidates[name] = {"matrix": zframe(x), "family": family}
        families.setdefault(family, []).append(name)

    for w in [6, 12, 24, 48]:
        add("multiscale_accumulation", f"reattach_mean_{w}", base_reattach.rolling(w, max(3, w // 2)).mean())
    for threshold in [.5, 1.0, 1.5]:
        age = event_age(raw["separation_z"] > threshold)
        for decay in [6, 12, 24]:
            add("event_clock", f"event_clock_t{threshold}_d{decay}", base_reattach * np.exp(-age / decay))
    for s in [1.0, 2.0, 4.0]:
        add("pressure_gate", f"pressure_gate_{s}", base_reattach * np.exp(-np.abs(raw["pressure_z"]) / s))
        add("advection_gate", f"advection_gate_{s}", base_reattach * np.exp(-np.abs(raw["adv_z"]) / s))
    sink_gate = (1 + np.tanh(-raw["continuity_z"])) / 2
    add("source_sink_gate", "sink_absorption", base_reattach * sink_gate)
    peer_gate = (1 + np.tanh(v * pd.DataFrame(peer, index=idx, columns=cols))) / 2
    add("coherence_gate", "peer_coherent_reattach", base_reattach * peer_gate)
    p_prior = raw["pressure_z"].shift(1)
    pressure_release = -v * p_prior.clip(lower=0) * (p_prior - raw["pressure_z"]).clip(lower=0)
    for w in [6, 12, 24]:
        add("pressure_release", f"pressure_release_{w}", pressure_release.rolling(w, max(3, w // 2)).mean())
    components = [
        zframe(base_reattach.rolling(12, 6).mean()).rank(axis=1, pct=True) - .5,
        zframe(base_reattach * sink_gate).rank(axis=1, pct=True) - .5,
        zframe(base_reattach * np.exp(-np.abs(raw["pressure_z"]) / 2)).rank(axis=1, pct=True) - .5,
        zframe(base_reattach * peer_gate).rank(axis=1, pct=True) - .5,
    ]
    add("recovery_consensus", "recovery_consensus", sum(components) / len(components))

    base_features = {
        "u": u,
        "mom4": u.apply(lambda s: norm_momentum(s, 4)),
        "mom12": u.apply(lambda s: norm_momentum(s, 12)),
        "mom24": u.apply(lambda s: norm_momentum(s, 24)),
        "mom72": u.apply(lambda s: norm_momentum(s, 72)),
        "rv24": np.log(np.sqrt(u.pow(2).rolling(24, 18).mean()) + EPS).apply(lambda s: causal_z(s, 336, 120)),
        "rv72": np.log(np.sqrt(u.pow(2).rolling(72, 54).mean()) + EPS).apply(lambda s: causal_z(s, 336, 120)),
        "qz": qz,
        "imb6": imb.rolling(6, 4).mean().apply(lambda s: causal_z(s, 336, 120)),
        "imb24": imb.rolling(24, 18).mean().apply(lambda s: causal_z(s, 336, 120)),
        "range24": range_z,
        "peer": pd.DataFrame(peer, index=idx, columns=cols),
        **raw,
        "raw_reattach": zframe(base_reattach),
    }
    labels = {f"y{h}": np.log(close.shift(-h - 1) / close.shift(-1)) / (scale * math.sqrt(h) + EPS) for h in HORIZONS}
    frames = {**base_features, **{k: v0["matrix"] for k, v0 in candidates.items()}, **labels, "nextret": ret.shift(-1)}
    panel = stack_panel(frames).replace([np.inf, -np.inf], np.nan)
    pre = panel[panel.time < pd.Timestamp("2025-01-01", tz="UTC")].copy()
    hold = panel[panel.time >= pd.Timestamp("2025-01-01", tz="UTC")].copy()

    periods = {
        "p1_2021_2022": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
        "p2_2023": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
        "p3_2024": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    }
    scan_rows, selected = [], {}
    for family, names in families.items():
        best = None
        for name in names:
            for h in HORIZONS:
                vals = []
                for pname, (start, end) in periods.items():
                    ic = csic_series(candidates[name]["matrix"].loc[start:end - pd.Timedelta(hours=1)], labels[f"y{h}"].loc[start:end - pd.Timedelta(hours=1)]).mean()
                    vals.append(float(ic))
                score = min(vals) + .25 * float(np.mean(vals))
                row = {"family": family, "candidate": name, "horizon": h, "score": score, **dict(zip(periods, vals))}
                scan_rows.append(row)
                if best is None or score > best[0]:
                    best = (score, name, h, vals)
        selected[family] = {"candidate": best[1], "horizon": int(best[2]), "preperiod_ics": best[3], "score": float(best[0])}
    pd.DataFrame(scan_rows).to_csv(OUT / "pre2025_candidate_scan.csv", index=False)

    parents = list(base_features)
    hold_results, pvals = [], {}
    for family, spec in selected.items():
        name, h = spec["candidate"], spec["horizon"]
        target = f"y{h}"
        xh = candidates[name]["matrix"].loc[pd.Timestamp("2025-01-01", tz="UTC"):]
        yh = labels[target].loc[xh.index]
        ic = csic_series(xh, yh)
        boot = block_bootstrap(ic)
        pvals[family] = boot["p_nonpositive"]
        pre2 = pre.copy(); hold2 = hold.copy()
        mse = ridge_increment(pre2, hold2, name, target, parents)
        r2, resid, maxcorr = residual_diagnostics(pre2, hold2, name, target, parents)
        p2 = portfolio(hold2, name, h, 2)
        p5 = portfolio(hold2, name, h, 5)
        positive_share, offset_values = nonoverlap_share(ic, h)
        yearly = {str(year): float(ic[ic.index.year == year].mean()) for year in sorted(set(ic.index.year))}
        hold_results.append({
            "family": family, "candidate": name, "horizon": h,
            "preperiod_ics": spec["preperiod_ics"], "holdout_rankic": float(ic.mean()),
            **boot, "mse_gain_pct": mse, "reconstruction_r2": r2,
            "residual_rankic": resid, "max_parent_corr": maxcorr,
            **p2, **p5, "nonoverlap_positive_share": positive_share,
            "nonoverlap_values": offset_values, "yearly_rankic": yearly,
        })
    adjusted = holm(pvals)
    admitted = []
    for row in hold_results:
        row["holm_p"] = adjusted[row["family"]]
        years_positive = all(v > 0 for v in row["yearly_rankic"].values())
        row["pass"] = bool(
            row["holdout_rankic"] > .015 and row["holm_p"] <= .10
            and row["mse_gain_pct"] > 0 and row["residual_rankic"] > .005
            and row["max_parent_corr"] < .80 and row["ret_2bp"] > 0
            and row["sharpe_2bp"] > 0 and row["ret_5bp"] > 0
            and row["nonoverlap_positive_share"] >= .70 and years_positive
        )
        if row["pass"]:
            admitted.append(row)
    pd.DataFrame(hold_results).to_csv(OUT / "final_holdout_results.csv", index=False)
    summary = {
        "data": {"symbols": symbols, "rows": int(len(data)), "min": str(data.time.min()), "max": str(data.time.max())},
        "graph_training": "2021-2022 only",
        "selection": "all pre-2025; maximize worst subperiod IC plus 0.25 mean across 2021-22, 2023, 2024; fixed physical sign",
        "final_holdout": "2025-2026-10, opened once for eight selected mechanism families",
        "selected": selected, "holdout_results": hold_results,
        "admitted": admitted, "admitted_count": len(admitted),
        "multiple_testing": "Holm across eight final holdout mechanism families",
        "parent_set": parents,
        "frozen_state": {"symbols": symbols, "beta": beta.to_dict(), "W": W.tolist(), "B": B.tolist()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"admitted_count": len(admitted), "admitted": [x["family"] for x in admitted]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
