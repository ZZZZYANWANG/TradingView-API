from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RESERVOIR_PATH = Path("research/fluid_reattachment_reservoir/run.py")
spec = importlib.util.spec_from_file_location("fluid_reattachment_reservoir", RESERVOIR_PATH)
reservoir = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reservoir)
base = reservoir.base
base.SYMBOLS = [
    "AAVEUSDT", "ALGOUSDT", "ATOMUSDT", "AVAXUSDT", "COMPUSDT",
    "EOSUSDT", "FILUSDT", "NEARUSDT", "SUSHIUSDT", "TRXUSDT",
    "UNIUSDT", "XTZUSDT", "ZECUSDT",
]

OUT = Path("research_outputs/fluid_backscatter_crossuniverse")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
HORIZONS = [480]
CADENCES = [5, 14]
RNG = np.random.default_rng(2026111803)


def zframe(x: pd.DataFrame) -> pd.DataFrame:
    return x.apply(lambda s: base.causal_z(s, 336, 120))


def daily_zframe(x: pd.DataFrame, window: int = 180, min_obs: int = 60) -> pd.DataFrame:
    def one(s: pd.Series) -> pd.Series:
        p = s.shift(1)
        return ((s - p.rolling(window, min_obs).mean()) /
                (p.rolling(window, min_obs).std(ddof=0) + EPS)).clip(-12, 12)
    return x.apply(one)


def daily_sample(x: pd.DataFrame, smooth_days: int = 1) -> pd.DataFrame:
    y = x.shift(1)
    y = y.loc[(y.index.hour == 0) & (y.index.minute == 0)]
    if smooth_days > 1:
        y = y.rolling(smooth_days, max(2, smooth_days // 2)).mean()
    return daily_zframe(y)


def rank_consensus(mats: list[pd.DataFrame]) -> pd.DataFrame:
    arr = np.stack([(m.rank(axis=1, pct=True) - 0.5).to_numpy(dtype=float) for m in mats], axis=2)
    return pd.DataFrame(np.nanmedian(arr, axis=2), index=mats[0].index, columns=mats[0].columns)


def stack_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([v.stack().rename(k) for k, v in frames.items()], axis=1).reset_index(names=["time", "symbol"])


def portfolio(
    factor: pd.DataFrame,
    next_daily_return: pd.DataFrame,
    risk: pd.DataFrame,
    *,
    hold_days: int,
    cadence: int,
    cost_bp: float,
) -> dict[str, float]:
    factor, next_daily_return = factor.align(next_daily_return, join="inner", axis=0)
    risk = risk.reindex(index=factor.index, columns=factor.columns).clip(lower=1.0e-4)
    rank = factor.rank(axis=1, pct=True)
    issue = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    issue[rank >= 0.75] = 1.0
    issue[rank <= 0.25] = -1.0
    issue = issue.div(risk)
    issue = issue.div(issue.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    issue_mask = np.zeros(len(issue), dtype=bool)
    issue_mask[::cadence] = True
    issue.loc[~issue_mask, :] = 0.0
    issuance = pd.Series(issue_mask.astype(float), index=issue.index)
    active = issuance.rolling(hold_days, min_periods=1).sum().replace(0, np.nan)
    sleeve = issue.rolling(hold_days, min_periods=1).sum().div(active, axis=0).fillna(0.0)
    gross = (sleeve * next_daily_return).sum(axis=1)
    turnover = sleeve.diff().abs().sum(axis=1).fillna(sleeve.abs().sum(axis=1))
    pnl = gross - turnover * cost_bp * 1.0e-4
    ann = 365.0
    vol = pnl.std(ddof=0) * math.sqrt(ann)
    wealth = np.exp(pnl.cumsum())
    dd = wealth / wealth.cummax() - 1.0
    return {
        f"ret_{int(cost_bp)}bp": float(pnl.mean() * ann),
        f"sharpe_{int(cost_bp)}bp": float(pnl.mean() * ann / (vol + EPS)),
        f"max_drawdown_{int(cost_bp)}bp": float(dd.min()),
        "avg_daily_turnover": float(turnover.mean()),
    }


def ridge_increment(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, parents: list[str]) -> float:
    tr = train.dropna(subset=parents + [factor, target])
    te = test.dropna(subset=parents + [factor, target])
    s0 = StandardScaler().fit(tr[parents])
    s1 = StandardScaler().fit(tr[parents + [factor]])
    m0 = Ridge(alpha=25).fit(s0.transform(tr[parents]), tr[target])
    m1 = Ridge(alpha=25).fit(s1.transform(tr[parents + [factor]]), tr[target])
    y = te[target].to_numpy()
    p0 = m0.predict(s0.transform(te[parents]))
    p1 = m1.predict(s1.transform(te[parents + [factor]]))
    mse0 = np.mean((y - p0) ** 2)
    mse1 = np.mean((y - p1) ** 2)
    return float(100.0 * (mse0 - mse1) / (mse0 + EPS))


def residual_diagnostics(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, parents: list[str]):
    tr = train.dropna(subset=parents + [factor])
    te = test.dropna(subset=parents + [factor, target]).copy()
    scaler = StandardScaler().fit(tr[parents])
    model = Ridge(alpha=25).fit(scaler.transform(tr[parents]), tr[factor])
    fit = model.predict(scaler.transform(tr[parents]))
    pred = model.predict(scaler.transform(te[parents]))
    r2 = 1.0 - np.mean((tr[factor].to_numpy() - fit) ** 2) / (np.var(tr[factor].to_numpy()) + EPS)
    te["resid"] = te[factor].to_numpy() - pred
    rx = te.pivot(index="time", columns="symbol", values="resid")
    ry = te.pivot(index="time", columns="symbol", values=target)
    maxcorr = max(abs(tr[factor].corr(tr[p], method="spearman")) for p in parents)
    return float(r2), float(base.csic_series(rx, ry).mean()), float(maxcorr)


def nonoverlap(ic: pd.Series, hold_days: int) -> tuple[float, list[float]]:
    days = ((ic.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")) /
            pd.Timedelta(days=1)).astype(int)
    vals = [float(ic[(days % hold_days) == off].mean()) for off in range(min(20, hold_days))]
    return float(np.mean(np.asarray(vals) > 0)), vals


def main() -> None:
    data, symbols = base.load_data()
    fields = {c: base.wide(data, c, symbols) for c in ["open", "high", "low", "close", "qv", "trades", "tbq"]}
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
    qz = logq.apply(lambda s: base.causal_z(s, 168, 72))
    rho = np.exp(qz.clip(-3, 3))
    imbalance = (2 * fields["tbq"] / (fields["qv"] + EPS) - 1).clip(-1, 1)
    range_state = np.log(fields["high"] / fields["low"]).rolling(24, 18).mean().apply(
        lambda s: base.causal_z(s, 336, 120)
    )

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
    pressure_solver = np.linalg.pinv(L + 0.15 * np.eye(len(symbols)))

    lag_x = dv.shift(1).dropna()
    lag_y = dv.loc[lag_x.index]
    reg = Ridge(alpha=25).fit(lag_x, lag_y)
    B = reg.coef_.copy()
    np.fill_diagonal(B, 0)
    S = 0.5 * (B + B.T)
    R = 0.5 * (B - B.T)

    x0 = lag_x.to_numpy(dtype=float)
    y0 = lag_y.to_numpy(dtype=float)
    x0 = (x0 - x0.mean(axis=0)) / (x0.std(axis=0) + EPS)
    y0 = (y0 - y0.mean(axis=0)) / (y0.std(axis=0) + EPS)
    C = y0.T @ x0 / max(1, len(x0) - 1)
    A = C - C.T

    va, ra = v.to_numpy(dtype=float), rho.to_numpy(dtype=float)
    peer = va @ W.T
    adv = va @ B.T - va
    div = np.full_like(va, np.nan)
    for t in range(len(va)):
        if np.isfinite(va[t]).all() and np.isfinite(ra[t]).all():
            flux = W * (0.5 * (ra[t, :, None] + ra[t, None, :])) * (va[t, None, :] - va[t, :, None])
            div[t] = flux.sum(axis=1)
    centered = div - np.nanmean(div, axis=1, keepdims=True)
    pressure = np.full_like(centered, np.nan)
    good = np.isfinite(centered).all(axis=1)
    pressure[good] = -centered[good] @ pressure_solver.T
    continuity = -(np.log(rho + EPS).diff().to_numpy() + div)
    tau = (va * va) @ W.T - peer * peer
    subgrid = tau * (peer - va)
    separation = np.abs(va - peer)
    mass_velocity = ra * va
    circulation = mass_velocity @ A.T
    strain = mass_velocity @ S.T
    rotation = mass_velocity @ R.T
    qcriterion = (rotation * rotation - strain * strain) / (rotation * rotation + strain * strain + EPS)
    peer_pressure = pressure @ W.T
    peer_density = ra @ W.T
    baroclinic = (ra - peer_density) * (pressure - peer_pressure)

    idx, cols = v.index, v.columns
    def df(a): return pd.DataFrame(a, index=idx, columns=cols)
    states = {
        "adv": zframe(df(adv)),
        "pressure": zframe(df(pressure)),
        "divergence": zframe(df(div)),
        "continuity": zframe(df(continuity)),
        "subgrid": zframe(df(subgrid)),
        "separation": zframe(df(separation)),
        "circulation": zframe(df(circulation)),
        "qcriterion": zframe(df(qcriterion)),
        "baroclinic": zframe(df(baroclinic)),
    }
    peer_df = df(peer)
    pressure_work = zframe(-states["pressure"] * states["divergence"])
    prior_p = states["pressure"].shift(1)
    pressure_relief = -v * prior_p.abs() * (prior_p.abs() - states["pressure"].abs()).clip(lower=0)
    prior_pw = pressure_work.shift(1)
    compression_release = -v * prior_pw.clip(lower=0) * (prior_pw - pressure_work).clip(lower=0)
    expansion_release = v * (-prior_pw).clip(lower=0) * ((-prior_pw) - (-pressure_work)).clip(lower=0)
    backscatter = v * (-states["subgrid"]).clip(lower=0)
    forward_cascade_exhaustion = -v * states["subgrid"].clip(lower=0)
    source_absorption = -v * (-states["continuity"]).clip(lower=0) * np.exp(-states["pressure"].abs() / 2)
    baroclinic_release = -v * states["baroclinic"].shift(1) * (
        states["baroclinic"].shift(1).abs() - states["baroclinic"].abs()
    ).clip(lower=0)
    prior_q = states["qcriterion"].shift(1)
    topology_release = -v * prior_q.clip(lower=0) * (prior_q - states["qcriterion"]).clip(lower=0)
    prior_circ = states["circulation"].shift(1)
    circulation_release = -v * prior_circ.abs() * (prior_circ.abs() - states["circulation"].abs()).clip(lower=0)
    coherent_backscatter = backscatter * (1 + np.tanh(v * peer_df)) / 2

    primitives = {
        "pressure_relief": pressure_relief,
        "compression_release": compression_release,
        "expansion_release": expansion_release,
        "backscatter": backscatter,
        "forward_cascade_exhaustion": forward_cascade_exhaustion,
        "source_absorption": source_absorption,
        "baroclinic_release": baroclinic_release,
        "topology_release": topology_release,
        "circulation_release": circulation_release,
        "coherent_backscatter": coherent_backscatter,
    }

    candidates: dict[str, dict[str, object]] = {}
    def add(name: str, matrix: pd.DataFrame, cadence: int):
        candidates[name] = {"matrix": matrix, "cadence": cadence}

    cb = coherent_backscatter
    pressure_gate = (1 + np.tanh(-pressure_work)) / 2
    low_circulation_gate = np.exp(-states["circulation"].abs() / 2)
    high_circulation_gate = (1 + np.tanh(states["circulation"].abs())) / 2
    threshold_event = cb.where((-states["subgrid"]) >= 0.0, 0.0)

    add(
        "backscatter_threshold_front_20d",
        daily_sample(zframe(threshold_event.rolling(6, 3).mean()), 3),
        14,
    )
    add(
        "backscatter_high_circulation_20d",
        daily_sample(zframe((cb * high_circulation_gate).rolling(6, 3).mean()), 3),
        14,
    )
    add(
        "backscatter_low_circulation_20d",
        daily_sample(zframe((cb * low_circulation_gate).rolling(6, 3).mean()), 3),
        14,
    )
    add(
        "backscatter_pressure_gate_20d",
        daily_sample(zframe((cb * pressure_gate).rolling(6, 3).mean()), 3),
        5,
    )

    hour0 = (idx.hour == 0) & (idx.minute == 0)
    daily_close = close.shift(1).loc[hour0]
    daily_return = np.log(daily_close).diff()
    next_daily_return = np.log(close.shift(-25) / close.shift(-1)).loc[hour0]
    daily_risk = (scale * math.sqrt(24)).loc[hour0]

    base_features = {
        "mom3d": daily_return.apply(lambda s: base.norm_momentum(s, 3)),
        "mom5d": daily_return.apply(lambda s: base.norm_momentum(s, 5)),
        "mom10d": daily_return.apply(lambda s: base.norm_momentum(s, 10)),
        "mom20d": daily_return.apply(lambda s: base.norm_momentum(s, 20)),
        "rv5d": daily_zframe(np.log(np.sqrt(daily_return.pow(2).rolling(5, 4).mean()) + EPS)),
        "rv20d": daily_zframe(np.log(np.sqrt(daily_return.pow(2).rolling(20, 15).mean()) + EPS)),
        "qz": daily_sample(qz),
        "imb6": daily_sample(imbalance.rolling(6, 4).mean().apply(lambda s: base.causal_z(s, 336, 120))),
        "imb24": daily_sample(imbalance.rolling(24, 18).mean().apply(lambda s: base.causal_z(s, 336, 120))),
        "range24": daily_sample(range_state),
        "peer": daily_sample(peer_df),
        **{f"state_{k}": daily_sample(v0) for k, v0 in states.items()},
        "pressure_work": daily_sample(pressure_work),
        "coherent_backscatter_raw": daily_sample(zframe(coherent_backscatter)),
        "reattach_mean6": daily_sample(zframe((-v * states["separation"].shift(1).clip(lower=0) *
                                                (states["separation"].shift(1) - states["separation"]).clip(lower=0) *
                                                np.exp(-range_state.abs() / 2)).rolling(6, 3).mean())),
    }
    labels = {
        f"y{h}": np.log(close.shift(-h - 1) / close.shift(-1)).loc[hour0]
        for h in HORIZONS
    }
    frames = {**base_features, **{k: v0["matrix"] for k, v0 in candidates.items()}, **labels,
              "next_daily_return": next_daily_return}
    panel = stack_panel(frames).replace([np.inf, -np.inf], np.nan)
    pre = panel[panel.time < pd.Timestamp("2025-01-01", tz="UTC")].copy()
    hold = panel[panel.time >= pd.Timestamp("2025-01-01", tz="UTC")].copy()

    periods = {
        "p1_2021_2022": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
        "p2_2023": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
        "p3_2024": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    }

    frozen = {
        "backscatter_threshold_front_20d": {"horizon": 480, "cadence": 14},
        "backscatter_high_circulation_20d": {"horizon": 480, "cadence": 14},
        "backscatter_low_circulation_20d": {"horizon": 480, "cadence": 14},
        "backscatter_pressure_gate_20d": {"horizon": 480, "cadence": 5},
    }
    parents = list(base_features)
    train = panel[
        (panel.time >= pd.Timestamp("2021-01-01", tz="UTC"))
        & (panel.time < pd.Timestamp("2023-01-01", tz="UTC"))
    ].copy()
    evaluation_periods = {
        "validation_2023_2024": (
            pd.Timestamp("2023-01-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
        ),
        "later_2025_2026": (
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2026-11-01", tz="UTC"),
        ),
    }
    results = []
    pvalues = {period: {} for period in evaluation_periods}
    for period, (start, end) in evaluation_periods.items():
        test = panel[(panel.time >= start) & (panel.time < end)].copy()
        for name, frozen_spec in frozen.items():
            h = int(frozen_spec["horizon"])
            cadence = int(frozen_spec["cadence"])
            target = f"y{h}"
            factor = candidates[name]["matrix"].loc[start:end - pd.Timedelta(days=1)]
            label = labels[target].loc[factor.index]
            ic = base.csic_series(factor, label)
            boot = base.block_bootstrap(ic, reps=5000, block=20)
            pvalues[period][name] = boot["p_nonpositive"]
            mse = ridge_increment(train, test, name, target, parents)
            r2, resid, maxcorr = residual_diagnostics(train, test, name, target, parents)
            p2 = portfolio(
                factor,
                next_daily_return.loc[factor.index],
                daily_risk.loc[factor.index],
                hold_days=20,
                cadence=cadence,
                cost_bp=2,
            )
            p5 = portfolio(
                factor,
                next_daily_return.loc[factor.index],
                daily_risk.loc[factor.index],
                hold_days=20,
                cadence=cadence,
                cost_bp=5,
            )
            p10 = portfolio(
                factor,
                next_daily_return.loc[factor.index],
                daily_risk.loc[factor.index],
                hold_days=20,
                cadence=cadence,
                cost_bp=10,
            )
            positive_share, offsets = nonoverlap(ic, 20)
            yearly = {
                str(year): float(ic[ic.index.year == year].mean())
                for year in sorted(set(ic.index.year))
            }
            results.append({
                "period": period,
                "candidate": name,
                "horizon_hours": h,
                "cadence_days": cadence,
                "rankic": float(ic.mean()),
                **boot,
                "mse_gain_pct": mse,
                "reconstruction_r2": r2,
                "residual_rankic": resid,
                "max_parent_corr": maxcorr,
                **p2,
                **p5,
                **p10,
                "nonoverlap_positive_share": positive_share,
                "nonoverlap_values": offsets,
                "yearly_rankic": yearly,
            })

    adjusted = {period: base.holm(values) for period, values in pvalues.items()}
    for row in results:
        row["holm_p"] = adjusted[row["period"]][row["candidate"]]
        row["period_pass"] = bool(
            row["rankic"] > 0.015
            and row["ci_low"] > 0
            and row["holm_p"] <= 0.10
            and row["mse_gain_pct"] > 0
            and row["residual_rankic"] > 0.005
            and row["max_parent_corr"] < 0.85
            and row["ret_5bp"] > 0
            and row["ret_10bp"] > 0
            and row["nonoverlap_positive_share"] >= 0.70
            and all(value > 0 for value in row["yearly_rankic"].values())
        )

    admitted = []
    for name in frozen:
        rows = [row for row in results if row["candidate"] == name]
        if len(rows) == 2 and all(row["period_pass"] for row in rows):
            admitted.append({"candidate": name, "period_results": rows})

    pd.DataFrame(results).to_csv(OUT / "cross_universe_results.csv", index=False)
    (OUT / "admitted.json").write_text(
        json.dumps(admitted, indent=2, default=str), encoding="utf-8"
    )

    summary = {
        "data": {"symbols": symbols, "rows": int(len(data)), "min": str(data.time.min()), "max": str(data.time.max())},
        "mechanism": "frozen peer-coherent subgrid-backscatter formulas replicated on a disjoint asset universe",
        "graph_training": "2021-2022 only",
        "selection": "none on replication universe; formula, direction, horizon and cadence frozen from the original universe",
        "holdout": "disjoint-universe validation in 2023-2024 and later evaluation in 2025-2026-10",
        "selected": frozen,
        "results": results,
        "admitted": admitted,
        "admitted_count": len(admitted),
        "parents": parents,
        "frozen_state": {"symbols": symbols, "beta": beta.to_dict(), "W": W.tolist(), "B": B.tolist(), "A": A.tolist()},
        "governance": "Cross-universe historical replication; any admitted identity remains default-off and fresh-forward-required.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "admitted.json").write_text(json.dumps(admitted, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"admitted_count": len(admitted),
                      "admitted": [{"family": x["family"], "candidate": x["candidate"]} for x in admitted]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
