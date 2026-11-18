from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path("research/fluid_network_refined/run.py")
spec = importlib.util.spec_from_file_location("fluid_network_refined", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

OUT = Path("research_outputs/fluid_reattachment_reservoir")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
RNG = np.random.default_rng(2026111802)
HORIZONS = [72, 120, 240, 480]
CADENCES = [1, 2, 3, 5]


def daily_causal_z(s: pd.Series, window: int = 180, min_periods: int = 60) -> pd.Series:
    p = s.shift(1)
    return ((s - p.rolling(window, min_periods).mean()) /
            (p.rolling(window, min_periods).std(ddof=0) + EPS)).clip(-12, 12)


def daily_zframe(x: pd.DataFrame, window: int = 180, min_periods: int = 60) -> pd.DataFrame:
    return x.apply(lambda s: daily_causal_z(s, window, min_periods))


def sample_daily(x: pd.DataFrame, smooth_days: int = 1) -> pd.DataFrame:
    # At 00:00 UTC, only information through the preceding hourly bar is admitted.
    z = x.shift(1)
    z = z.loc[(z.index.hour == 0) & (z.index.minute == 0)]
    if smooth_days > 1:
        z = z.rolling(smooth_days, max(2, smooth_days // 2)).mean()
    return daily_zframe(z)


def reset_reservoir(
    impulse: pd.DataFrame,
    separation_z: pd.DataFrame,
    *,
    threshold: float,
    half_life: float,
    gate: pd.DataFrame | None = None,
    absolute: bool = False,
) -> pd.DataFrame:
    x = impulse.to_numpy(dtype=float)
    sep = separation_z.to_numpy(dtype=float)
    g = np.ones_like(x) if gate is None else gate.to_numpy(dtype=float)
    out = np.full_like(x, np.nan)
    decay = float(0.5 ** (1.0 / half_life))
    for j in range(x.shape[1]):
        state = 0.0
        ready = False
        for i in range(x.shape[0]):
            if np.isfinite(sep[i, j]) and sep[i, j] > threshold:
                state = 0.0
                ready = True
            if np.isfinite(x[i, j]) and np.isfinite(g[i, j]):
                value = abs(x[i, j]) if absolute else x[i, j]
                state = decay * state + value * g[i, j]
                ready = True
            elif ready:
                state *= decay
            if ready:
                out[i, j] = state
    return pd.DataFrame(out, index=impulse.index, columns=impulse.columns)


def rank_consensus(mats: list[pd.DataFrame]) -> pd.DataFrame:
    centered = [(m.rank(axis=1, pct=True) - 0.5) for m in mats]
    arr = np.stack([m.to_numpy(dtype=float) for m in centered], axis=2)
    return pd.DataFrame(np.nanmedian(arr, axis=2), index=mats[0].index, columns=mats[0].columns)


def stack_daily(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([x.stack().rename(k) for k, x in frames.items()], axis=1).reset_index(names=["time", "symbol"])


def daily_portfolio(
    factor: pd.DataFrame,
    daily_return: pd.DataFrame,
    *,
    hold_days: int,
    cadence: int,
    cost_bp: float,
) -> dict[str, float]:
    factor, daily_return = factor.align(daily_return, join="inner", axis=0)
    rank = factor.rank(axis=1, pct=True)
    issue = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    issue[rank >= 0.75] = 1.0
    issue[rank <= 0.25] = -1.0
    issue = issue.div(issue.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    issue_mask = np.zeros(len(issue), dtype=bool)
    issue_mask[::cadence] = True
    issue = issue.where(issue_mask[:, None], 0.0)
    issue_count = pd.Series(issue_mask.astype(float), index=issue.index)
    active_count = issue_count.rolling(hold_days, min_periods=1).sum().replace(0, np.nan)
    sleeve = issue.rolling(hold_days, min_periods=1).sum().div(active_count, axis=0).fillna(0.0)
    gross = (sleeve * daily_return).sum(axis=1)
    turnover = sleeve.diff().abs().sum(axis=1).fillna(sleeve.abs().sum(axis=1))
    pnl = gross - turnover * cost_bp * 1e-4
    ann = 365.0
    vol = pnl.std(ddof=0) * math.sqrt(ann)
    wealth = np.exp(pnl.cumsum())
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        f"ret_{int(cost_bp)}bp": float(pnl.mean() * ann),
        f"sharpe_{int(cost_bp)}bp": float(pnl.mean() * ann / (vol + EPS)),
        f"max_drawdown_{int(cost_bp)}bp": float(drawdown.min()),
        "avg_daily_turnover": float(turnover.mean()),
    }


def daily_nonoverlap_share(ic: pd.Series, hold_days: int) -> tuple[float, list[float]]:
    day_index = ((ic.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")) /
                 pd.Timedelta(days=1)).astype(int)
    offsets = np.arange(min(20, hold_days))
    vals = [float(ic[(day_index % hold_days) == off].mean()) for off in offsets]
    return float(np.mean(np.asarray(vals) > 0)), vals


def ridge_increment(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, parents: list[str]) -> float:
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
    maxcorr = max(abs(tr[factor].corr(tr[p], method="spearman")) for p in parents)
    return float(r2), float(base.csic_series(rx, ry).mean()), float(maxcorr)


def main():
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
    imb = (2 * fields["tbq"] / (fields["qv"] + EPS) - 1).clip(-1, 1)
    range_z = np.log(fields["high"] / fields["low"]).rolling(24, 18).mean().apply(
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

    va, ra = v.to_numpy(), rho.to_numpy()
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

    idx, cols = v.index, v.columns
    peer_df = pd.DataFrame(peer, index=idx, columns=cols)
    raw = {
        "adv_z": base.zframe(pd.DataFrame(adv, index=idx, columns=cols)),
        "pressure_z": base.zframe(pd.DataFrame(pressure, index=idx, columns=cols)),
        "continuity_z": base.zframe(pd.DataFrame(continuity, index=idx, columns=cols)),
        "subgrid_z": base.zframe(pd.DataFrame(subgrid, index=idx, columns=cols)),
        "separation_z": base.zframe(pd.DataFrame(separation, index=idx, columns=cols)),
    }
    prior_sep = raw["separation_z"].shift(1).clip(lower=0)
    relief = (prior_sep - raw["separation_z"]).clip(lower=0)
    base_reattach = -v * prior_sep * relief * np.exp(-np.abs(range_z) / 2)
    sink_gate = (1 + np.tanh(-raw["continuity_z"])) / 2
    peer_gate = (1 + np.tanh(v * peer_df)) / 2

    candidates: dict[str, dict[str, object]] = {}
    families: dict[str, list[str]] = {}

    def add(family: str, name: str, matrix: pd.DataFrame):
        candidates[name] = {"matrix": matrix, "family": family}
        families.setdefault(family, []).append(name)

    plain_hourly: dict[tuple[float, int], pd.DataFrame] = {}
    abs_hourly: dict[tuple[float, int], pd.DataFrame] = {}
    for threshold in [0.5, 1.0, 1.5]:
        for half_life in [12, 24, 48, 72]:
            q = reset_reservoir(base_reattach, raw["separation_z"], threshold=threshold, half_life=half_life)
            qa = reset_reservoir(base_reattach, raw["separation_z"], threshold=threshold, half_life=half_life, absolute=True)
            plain_hourly[(threshold, half_life)] = q
            abs_hourly[(threshold, half_life)] = qa
            for smooth in [1, 3]:
                add("plain_reservoir", f"plain_t{threshold}_h{half_life}_s{smooth}", sample_daily(q, smooth))
            ratio = q / (qa + EPS)
            add("normalized_reservoir", f"normalized_t{threshold}_h{half_life}", sample_daily(ratio, 1))

    for half_life in [24, 48, 72]:
        for scale_gate in [1.0, 2.0, 4.0]:
            q = reset_reservoir(
                base_reattach, raw["separation_z"], threshold=1.0, half_life=half_life,
                gate=np.exp(-np.abs(raw["pressure_z"]) / scale_gate),
            )
            add("pressure_relaxed_reservoir", f"pressure_h{half_life}_s{scale_gate}", sample_daily(q, 1))
            q = reset_reservoir(
                base_reattach, raw["separation_z"], threshold=1.0, half_life=half_life,
                gate=np.exp(-np.abs(raw["adv_z"]) / scale_gate),
            )
            add("advection_relaxed_reservoir", f"advection_h{half_life}_s{scale_gate}", sample_daily(q, 1))
        q = reset_reservoir(
            base_reattach, raw["separation_z"], threshold=1.0, half_life=half_life, gate=sink_gate,
        )
        add("sink_absorption_reservoir", f"sink_h{half_life}", sample_daily(q, 1))
        q = reset_reservoir(
            base_reattach, raw["separation_z"], threshold=1.0, half_life=half_life, gate=peer_gate,
        )
        add("peer_coherent_reservoir", f"peer_h{half_life}", sample_daily(q, 1))

    for w in [24, 48, 72, 120]:
        add("daily_integrated_reattachment", f"integrated_{w}", sample_daily(base_reattach.rolling(w, w // 2).mean(), 1))

    consensus_inputs = [
        sample_daily(plain_hourly[(1.0, 24)], 1),
        sample_daily(plain_hourly[(1.0, 48)], 1),
        sample_daily(reset_reservoir(base_reattach, raw["separation_z"], threshold=1.0, half_life=48, gate=sink_gate), 1),
        sample_daily(reset_reservoir(base_reattach, raw["separation_z"], threshold=1.0, half_life=48, gate=peer_gate), 1),
        sample_daily(reset_reservoir(base_reattach, raw["separation_z"], threshold=1.0, half_life=48,
                                     gate=np.exp(-np.abs(raw["pressure_z"]) / 2.0)), 1),
    ]
    add("reservoir_consensus", "reservoir_consensus", daily_zframe(rank_consensus(consensus_inputs)))

    hour0 = (idx.hour == 0) & (idx.minute == 0)
    daily_close = close.shift(1).loc[hour0]
    daily_ret = np.log(daily_close).diff()
    daily_next_return = np.log(close.shift(-25) / close.shift(-1)).loc[hour0]

    base_features = {
        "mom3d": daily_ret.apply(lambda s: base.norm_momentum(s, 3)),
        "mom5d": daily_ret.apply(lambda s: base.norm_momentum(s, 5)),
        "mom10d": daily_ret.apply(lambda s: base.norm_momentum(s, 10)),
        "mom20d": daily_ret.apply(lambda s: base.norm_momentum(s, 20)),
        "rv5d": daily_zframe(np.log(np.sqrt(daily_ret.pow(2).rolling(5, 4).mean()) + EPS)),
        "rv20d": daily_zframe(np.log(np.sqrt(daily_ret.pow(2).rolling(20, 15).mean()) + EPS)),
        "qz": sample_daily(qz),
        "imb6": sample_daily(imb.rolling(6, 4).mean().apply(lambda s: base.causal_z(s, 336, 120))),
        "imb24": sample_daily(imb.rolling(24, 18).mean().apply(lambda s: base.causal_z(s, 336, 120))),
        "range24": sample_daily(range_z),
        "peer": sample_daily(peer_df),
        "adv_z": sample_daily(raw["adv_z"]),
        "pressure_z": sample_daily(raw["pressure_z"]),
        "continuity_z": sample_daily(raw["continuity_z"]),
        "subgrid_z": sample_daily(raw["subgrid_z"]),
        "separation_z": sample_daily(raw["separation_z"]),
        "raw_reattach": sample_daily(base.zframe(base_reattach)),
        "reattach_mean6": sample_daily(base.zframe(base_reattach.rolling(6, 3).mean())),
        "reattach_mean24": sample_daily(base.zframe(base_reattach.rolling(24, 12).mean())),
    }
    labels = {
        f"y{h}": (np.log(close.shift(-h - 1) / close.shift(-1)) /
                   (scale * math.sqrt(h) + EPS)).loc[hour0]
        for h in HORIZONS
    }
    frames = {**base_features, **{k: v0["matrix"] for k, v0 in candidates.items()}, **labels,
              "daily_next_return": daily_next_return}
    panel = stack_daily(frames).replace([np.inf, -np.inf], np.nan)
    pre = panel[panel.time < pd.Timestamp("2025-01-01", tz="UTC")].copy()
    hold = panel[panel.time >= pd.Timestamp("2025-01-01", tz="UTC")].copy()

    periods = {
        "p1_2021_2022": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
        "p2_2023": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
        "p3_2024": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    }

    scan_rows = []
    selected: dict[str, dict[str, object]] = {}
    for family, names in families.items():
        best = None
        for name in names:
            for h in HORIZONS:
                values = []
                for _, (start, end) in periods.items():
                    factor_slice = candidates[name]["matrix"].loc[start:end - pd.Timedelta(days=1)]
                    label_slice = labels[f"y{h}"].loc[factor_slice.index]
                    values.append(float(base.csic_series(factor_slice, label_slice).mean()))
                score = min(values) + 0.25 * float(np.mean(values))
                scan_rows.append({"family": family, "candidate": name, "horizon": h,
                                  "score": score, **dict(zip(periods, values))})
                if best is None or score > best[0]:
                    best = (score, name, h, values)
        assert best is not None
        name, h = best[1], int(best[2])
        cadence_rows = []
        hold_days = h // 24
        for cadence in CADENCES:
            sharpes = []
            returns = []
            for _, (start, end) in periods.items():
                p = daily_portfolio(
                    candidates[name]["matrix"].loc[start:end - pd.Timedelta(days=1)],
                    daily_next_return.loc[start:end - pd.Timedelta(days=1)],
                    hold_days=hold_days, cadence=cadence, cost_bp=5,
                )
                sharpes.append(p["sharpe_5bp"])
                returns.append(p["ret_5bp"])
            cadence_score = min(sharpes) + 0.25 * float(np.mean(sharpes))
            cadence_rows.append({"family": family, "candidate": name, "horizon": h,
                                 "cadence": cadence, "cadence_score": cadence_score,
                                 "min_pre_return_5bp": min(returns), "mean_pre_return_5bp": float(np.mean(returns)),
                                 "min_pre_sharpe_5bp": min(sharpes), "mean_pre_sharpe_5bp": float(np.mean(sharpes))})
        best_cadence = max(cadence_rows, key=lambda row: row["cadence_score"])
        selected[family] = {
            "candidate": name, "horizon": h, "preperiod_ics": best[3], "score": float(best[0]),
            "cadence": int(best_cadence["cadence"]),
            "pre_execution": best_cadence,
        }
        pd.DataFrame(cadence_rows).to_csv(OUT / f"cadence_{family}.csv", index=False)
    pd.DataFrame(scan_rows).to_csv(OUT / "pre2025_candidate_scan.csv", index=False)

    parents = list(base_features)
    hold_results = []
    pvals = {}
    for family, spec0 in selected.items():
        name = str(spec0["candidate"])
        h = int(spec0["horizon"])
        cadence = int(spec0["cadence"])
        target = f"y{h}"
        factor_hold = candidates[name]["matrix"].loc[pd.Timestamp("2025-01-01", tz="UTC"):]
        label_hold = labels[target].loc[factor_hold.index]
        ic = base.csic_series(factor_hold, label_hold)
        boot = base.block_bootstrap(ic, reps=4000, block=20)
        pvals[family] = boot["p_nonpositive"]
        mse = ridge_increment(pre, hold, name, target, parents)
        r2, resid, maxcorr = residual_diagnostics(pre, hold, name, target, parents)
        p2 = daily_portfolio(factor_hold, daily_next_return.loc[factor_hold.index],
                             hold_days=h // 24, cadence=cadence, cost_bp=2)
        p5 = daily_portfolio(factor_hold, daily_next_return.loc[factor_hold.index],
                             hold_days=h // 24, cadence=cadence, cost_bp=5)
        p10 = daily_portfolio(factor_hold, daily_next_return.loc[factor_hold.index],
                              hold_days=h // 24, cadence=cadence, cost_bp=10)
        positive_share, offset_values = daily_nonoverlap_share(ic, h // 24)
        yearly = {str(year): float(ic[ic.index.year == year].mean()) for year in sorted(set(ic.index.year))}
        hold_results.append({
            "family": family, "candidate": name, "horizon_hours": h, "hold_days": h // 24,
            "cadence_days": cadence, "preperiod_ics": spec0["preperiod_ics"],
            "pre_execution": spec0["pre_execution"], "holdout_rankic": float(ic.mean()),
            **boot, "mse_gain_pct": mse, "reconstruction_r2": r2,
            "residual_rankic": resid, "max_parent_corr": maxcorr,
            **p2, **p5, **p10, "nonoverlap_positive_share": positive_share,
            "nonoverlap_values": offset_values, "yearly_rankic": yearly,
        })

    adjusted = base.holm(pvals)
    admitted = []
    for row in hold_results:
        row["holm_p"] = adjusted[row["family"]]
        years_positive = all(v > 0 for v in row["yearly_rankic"].values())
        pre_exec = row["pre_execution"]
        row["pass"] = bool(
            min(row["preperiod_ics"]) > 0
            and pre_exec["min_pre_return_5bp"] > 0
            and row["holdout_rankic"] > 0.015 and row["holm_p"] <= 0.10
            and row["mse_gain_pct"] > 0 and row["residual_rankic"] > 0.005
            and row["max_parent_corr"] < 0.85 and row["ret_5bp"] > 0
            and row["sharpe_5bp"] > 0 and row["ret_10bp"] > 0
            and row["nonoverlap_positive_share"] >= 0.70 and years_positive
        )
        if row["pass"]:
            admitted.append(row)

    pd.DataFrame(hold_results).to_csv(OUT / "final_holdout_results.csv", index=False)
    summary = {
        "data": {"symbols": symbols, "rows": int(len(data)), "min": str(data.time.min()), "max": str(data.time.max())},
        "mechanism": "event-reset network-boundary reattachment reservoir sampled once daily",
        "graph_training": "2021-2022 only",
        "selection": "pre-2025 only; worst-subperiod RankIC plus 0.25 mean; cadence chosen by worst preperiod 5bp Sharpe",
        "final_holdout": "2025-2026-10 opened once for eight selected families",
        "selected": selected,
        "holdout_results": hold_results,
        "admitted": admitted,
        "admitted_count": len(admitted),
        "multiple_testing": "Holm across eight selected reservoir families",
        "parent_set": parents,
        "frozen_state": {"symbols": symbols, "beta": beta.to_dict(), "W": W.tolist(), "B": B.tolist()},
        "governance": "Historical holdout only; related reattachment family was previously inspected, so any admission remains default-off pending fresh forward data.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "admitted.json").write_text(json.dumps(admitted, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"admitted_count": len(admitted),
                      "admitted": [{"family": x["family"], "candidate": x["candidate"]} for x in admitted]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
