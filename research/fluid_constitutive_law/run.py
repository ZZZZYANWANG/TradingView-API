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

POOL_A = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "ETCUSDT", "XLMUSDT", "DOTUSDT", "SOLUSDT",
]
POOL_B = [
    "AAVEUSDT", "ALGOUSDT", "ATOMUSDT", "AVAXUSDT", "COMPUSDT",
    "EOSUSDT", "FILUSDT", "NEARUSDT", "SUSHIUSDT", "TRXUSDT",
    "UNIUSDT", "XTZUSDT", "ZECUSDT",
]
base.SYMBOLS = POOL_A + POOL_B
OUT = Path("research_outputs/fluid_constitutive_law")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
HORIZONS = [72, 120, 240, 480]
CADENCES = [1, 2, 3, 5, 7, 10]


def zframe(x: pd.DataFrame, window: int = 720, min_obs: int = 240) -> pd.DataFrame:
    return x.apply(lambda s: base.causal_z(s, window, min_obs))


def daily_zframe(x: pd.DataFrame, window: int = 180, min_obs: int = 60) -> pd.DataFrame:
    def one(s: pd.Series) -> pd.Series:
        prior = s.shift(1)
        return ((s - prior.rolling(window, min_obs).mean()) /
                (prior.rolling(window, min_obs).std(ddof=0) + EPS)).clip(-12, 12)
    return x.apply(one)


def daily_sample(x: pd.DataFrame, smooth_days: int = 1) -> pd.DataFrame:
    y = x.shift(1)
    y = y.loc[(y.index.hour == 0) & (y.index.minute == 0)]
    if smooth_days > 1:
        y = y.rolling(smooth_days, max(2, smooth_days // 2)).mean()
    return daily_zframe(y)


def rolling_beta(x: pd.DataFrame, y: pd.DataFrame, window: int = 720, min_obs: int = 240) -> pd.DataFrame:
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for column in x.columns:
        xp = x[column].shift(1)
        yp = y[column].shift(1)
        cov = xp.rolling(window, min_obs).cov(yp)
        var = xp.rolling(window, min_obs).var(ddof=0)
        out[column] = cov / (var + EPS)
    return out.clip(-8, 8)


def stack_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([value.stack().rename(name) for name, value in frames.items()], axis=1).reset_index(names=["time", "symbol"])


def pool_matrix(x: pd.DataFrame, pool: list[str]) -> pd.DataFrame:
    columns = [column for column in pool if column in x.columns]
    return x.reindex(columns=columns)


def portfolio(
    factor: pd.DataFrame,
    next_return: pd.DataFrame,
    risk: pd.DataFrame,
    *,
    hold_days: int,
    cadence: int,
    cost_bp: float,
) -> dict[str, float]:
    factor, next_return = factor.align(next_return, join="inner", axis=0)
    risk = risk.reindex(index=factor.index, columns=factor.columns).clip(lower=1.0e-4)
    rank = factor.rank(axis=1, pct=True)
    issue = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    issue[rank >= 0.75] = 1.0
    issue[rank <= 0.25] = -1.0
    issue = issue.div(risk)
    issue = issue.div(issue.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    mask = np.zeros(len(issue), dtype=bool)
    mask[::cadence] = True
    issue.loc[~mask, :] = 0.0
    issuance = pd.Series(mask.astype(float), index=issue.index)
    active = issuance.rolling(hold_days, min_periods=1).sum().replace(0, np.nan)
    sleeve = issue.rolling(hold_days, min_periods=1).sum().div(active, axis=0).fillna(0.0)
    gross = (sleeve * next_return).sum(axis=1)
    turnover = sleeve.diff().abs().sum(axis=1).fillna(sleeve.abs().sum(axis=1))
    pnl = gross - turnover * cost_bp * 1.0e-4
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


def nonoverlap(ic: pd.Series, hold_days: int) -> tuple[float, list[float]]:
    days = ((ic.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")) /
            pd.Timedelta(days=1)).astype(int)
    values = [float(ic[(days % hold_days) == offset].mean()) for offset in range(min(20, hold_days))]
    return float(np.mean(np.asarray(values) > 0)), values


def ridge_increment(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, parents: list[str]) -> float:
    tr = train.dropna(subset=parents + [factor, target])
    te = test.dropna(subset=parents + [factor, target])
    if len(tr) < 1000 or len(te) < 500:
        return float("nan")
    scale0 = StandardScaler().fit(tr[parents])
    scale1 = StandardScaler().fit(tr[parents + [factor]])
    model0 = Ridge(alpha=25).fit(scale0.transform(tr[parents]), tr[target])
    model1 = Ridge(alpha=25).fit(scale1.transform(tr[parents + [factor]]), tr[target])
    y = te[target].to_numpy()
    pred0 = model0.predict(scale0.transform(te[parents]))
    pred1 = model1.predict(scale1.transform(te[parents + [factor]]))
    mse0 = np.mean((y - pred0) ** 2)
    mse1 = np.mean((y - pred1) ** 2)
    return float(100 * (mse0 - mse1) / (mse0 + EPS))


def residual_diagnostics(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, parents: list[str]):
    tr = train.dropna(subset=parents + [factor])
    te = test.dropna(subset=parents + [factor, target]).copy()
    if len(tr) < 1000 or len(te) < 500:
        return float("nan"), float("nan"), float("nan")
    scaler = StandardScaler().fit(tr[parents])
    model = Ridge(alpha=25).fit(scaler.transform(tr[parents]), tr[factor])
    fitted = model.predict(scaler.transform(tr[parents]))
    predicted = model.predict(scaler.transform(te[parents]))
    r2 = 1 - np.mean((tr[factor].to_numpy() - fitted) ** 2) / (np.var(tr[factor].to_numpy()) + EPS)
    te["residual"] = te[factor].to_numpy() - predicted
    rx = te.pivot(index="time", columns="symbol", values="residual")
    ry = te.pivot(index="time", columns="symbol", values=target)
    correlations = [abs(tr[factor].corr(tr[parent], method="spearman")) for parent in parents]
    return float(r2), float(base.csic_series(rx, ry).mean()), float(np.nanmax(correlations))


def main() -> None:
    data, symbols = base.load_data()
    pool_a = [symbol for symbol in POOL_A if symbol in symbols]
    pool_b = [symbol for symbol in POOL_B if symbol in symbols]
    if len(pool_a) < 8 or len(pool_b) < 8:
        raise RuntimeError(f"insufficient disjoint pools: pool_a={pool_a}, pool_b={pool_b}")

    fields = {
        column: base.wide(data, column, symbols)
        for column in ["open", "high", "low", "close", "qv", "trades", "tbq"]
    }
    close = fields["close"]
    log_close = np.log(close)
    ret = log_close.diff()
    risk_hourly = np.sqrt(ret.shift(1).pow(2).ewm(span=48, min_periods=24, adjust=False).mean())
    strain = (ret / (risk_hourly + EPS)).clip(-12, 12)
    log_activity = np.log1p(fields["qv"].clip(lower=0))
    activity_z = zframe(log_activity, 336, 120)
    imbalance = (2 * fields["tbq"] / (fields["qv"] + EPS) - 1).clip(-1, 1)
    stress = zframe(imbalance * (1 + activity_z.abs()), 336, 120)
    beta = rolling_beta(stress, strain, 720, 240)
    expected_strain = beta * stress
    constitutive_gap = expected_strain - strain
    viscosity = np.sign(stress) * (np.log1p(stress.abs()) - np.log1p(strain.abs()))
    yield_absorption = np.sign(stress) * (stress.abs() - strain.abs()).clip(lower=0)
    eta = rolling_beta(strain, stress, 720, 240)
    hysteresis_unit = stress.shift(1) * strain - strain.shift(1) * stress
    storage_unit = stress - strain
    previous_gap = constitutive_gap.shift(1)
    absorption_release = previous_gap * (previous_gap.abs() - constitutive_gap.abs()).clip(lower=0)
    impact = np.log(fields["high"] / fields["low"]) / np.sqrt(fields["qv"].clip(lower=0) + 1.0)
    impact_z = zframe(np.log(impact + EPS), 336, 120)

    candidates: dict[str, dict[str, object]] = {}
    families: dict[str, list[str]] = {}

    def add(family: str, name: str, value: pd.DataFrame):
        candidates[name] = {"matrix": value, "family": family}
        families.setdefault(family, []).append(name)
        candidates[name + "__neg"] = {"matrix": -value, "family": family}
        families[family].append(name + "__neg")

    for window in [6, 12, 24, 48, 72, 120]:
        min_obs = max(3, window // 2)
        for smooth in [1, 3, 5]:
            add(
                "constitutive_gap",
                f"gap_w{window}_s{smooth}",
                daily_sample(zframe(constitutive_gap.rolling(window, min_obs).mean(), 336, 120), smooth),
            )
            add(
                "yield_absorption",
                f"yield_w{window}_s{smooth}",
                daily_sample(zframe(yield_absorption.rolling(window, min_obs).mean(), 336, 120), smooth),
            )
            add(
                "viscosity_memory",
                f"viscosity_w{window}_s{smooth}",
                daily_sample(zframe(viscosity.rolling(window, min_obs).mean(), 336, 120), smooth),
            )
            add(
                "stress_storage",
                f"storage_w{window}_s{smooth}",
                daily_sample(zframe(storage_unit.rolling(window, min_obs).sum(), 336, 120), smooth),
            )
            add(
                "hysteresis_area",
                f"hysteresis_w{window}_s{smooth}",
                daily_sample(zframe(hysteresis_unit.rolling(window, min_obs).mean(), 336, 120), smooth),
            )
            add(
                "absorption_release",
                f"release_w{window}_s{smooth}",
                daily_sample(zframe(absorption_release.rolling(window, min_obs).mean(), 336, 120), smooth),
            )

    for relaxation in [6, 12, 24, 48, 72]:
        maxwell = stress + relaxation * stress.diff() - eta * strain
        add(
            "maxwell_residual",
            f"maxwell_l{relaxation}",
            daily_sample(zframe(maxwell.rolling(12, 6).mean(), 336, 120), 3),
        )

    for amplitude_scale in [0.5, 1.0, 2.0, 4.0]:
        thinning = stress * np.exp(-strain.abs() / amplitude_scale)
        thickening = stress * np.tanh(strain.abs() / amplitude_scale)
        add(
            "shear_thinning",
            f"thinning_s{amplitude_scale}",
            daily_sample(zframe(thinning.rolling(12, 6).mean(), 336, 120), 3),
        )
        add(
            "shear_thickening",
            f"thickening_s{amplitude_scale}",
            daily_sample(zframe(thickening.rolling(12, 6).mean(), 336, 120), 3),
        )

    for window in [12, 24, 48, 72]:
        pressure_storage = constitutive_gap * np.exp(-impact_z.abs() / 2)
        add(
            "low_impact_storage",
            f"low_impact_w{window}",
            daily_sample(zframe(pressure_storage.rolling(window, window // 2).mean(), 336, 120), 3),
        )

    hour0 = (close.index.hour == 0) & (close.index.minute == 0)
    daily_close = close.shift(1).loc[hour0]
    daily_return = np.log(daily_close).diff()
    next_daily_return = np.log(close.shift(-25) / close.shift(-1)).loc[hour0]
    daily_risk = (risk_hourly * math.sqrt(24)).loc[hour0]

    parents = {
        "mom1d": daily_return,
        "mom3d": daily_return.apply(lambda s: base.norm_momentum(s, 3)),
        "mom5d": daily_return.apply(lambda s: base.norm_momentum(s, 5)),
        "mom10d": daily_return.apply(lambda s: base.norm_momentum(s, 10)),
        "mom20d": daily_return.apply(lambda s: base.norm_momentum(s, 20)),
        "rv5d": daily_zframe(np.log(np.sqrt(daily_return.pow(2).rolling(5, 4).mean()) + EPS)),
        "rv20d": daily_zframe(np.log(np.sqrt(daily_return.pow(2).rolling(20, 15).mean()) + EPS)),
        "activity": daily_sample(activity_z),
        "imbalance6": daily_sample(zframe(imbalance.rolling(6, 4).mean(), 336, 120)),
        "imbalance24": daily_sample(zframe(imbalance.rolling(24, 18).mean(), 336, 120)),
        "range24": daily_sample(zframe(np.log(fields["high"] / fields["low"]).rolling(24, 18).mean(), 336, 120)),
        "impact": daily_sample(impact_z),
        "stress": daily_sample(stress),
        "strain": daily_sample(strain),
        "beta": daily_sample(beta),
        "gap": daily_sample(zframe(constitutive_gap, 336, 120)),
        "viscosity": daily_sample(zframe(viscosity, 336, 120)),
        "hysteresis24": daily_sample(zframe(hysteresis_unit.rolling(24, 12).mean(), 336, 120)),
    }
    labels = {
        f"y{h}": np.log(close.shift(-h - 1) / close.shift(-1)).loc[hour0]
        for h in HORIZONS
    }
    frames = {**parents, **{name: item["matrix"] for name, item in candidates.items()}, **labels}
    panel = stack_panel(frames).replace([np.inf, -np.inf], np.nan)
    parent_names = list(parents)

    development_years = [2021, 2022]
    scan_rows = []
    selected: dict[str, dict[str, object]] = {}
    for family, names in families.items():
        best = None
        for name in names:
            for horizon in HORIZONS:
                values = []
                for year in development_years:
                    factor = pool_matrix(candidates[name]["matrix"], pool_a)
                    factor = factor[factor.index.year == year]
                    target = pool_matrix(labels[f"y{horizon}"], pool_a).loc[factor.index]
                    values.append(float(base.csic_series(factor, target).mean()))
                score = min(values) + 0.25 * float(np.mean(values))
                scan_rows.append({"family": family, "candidate": name, "horizon": horizon,
                                  "ic_2021": values[0], "ic_2022": values[1], "score": score})
                if best is None or score > best[0]:
                    best = (score, name, horizon, values)
        assert best is not None
        name, horizon = str(best[1]), int(best[2])
        cadence_rows = []
        for cadence in CADENCES:
            metrics = []
            for year in development_years:
                factor = pool_matrix(candidates[name]["matrix"], pool_a)
                factor = factor[factor.index.year == year]
                metrics.append(portfolio(
                    factor,
                    pool_matrix(next_daily_return, pool_a).loc[factor.index],
                    pool_matrix(daily_risk, pool_a).loc[factor.index],
                    hold_days=horizon // 24,
                    cadence=cadence,
                    cost_bp=5,
                ))
            returns = [item["ret_5bp"] for item in metrics]
            sharpes = [item["sharpe_5bp"] for item in metrics]
            cadence_rows.append({"family": family, "candidate": name, "horizon": horizon,
                                 "cadence": cadence,
                                 "score": min(sharpes) + 0.25 * float(np.mean(sharpes)),
                                 "min_return_5bp": min(returns), "mean_return_5bp": float(np.mean(returns)),
                                 "min_sharpe_5bp": min(sharpes), "mean_sharpe_5bp": float(np.mean(sharpes))})
        best_cadence = max(cadence_rows, key=lambda item: item["score"])
        selected[family] = {"candidate": name, "horizon": horizon,
                            "development_ics": best[3], "selection_score": float(best[0]),
                            "cadence": int(best_cadence["cadence"]), "development_execution": best_cadence}
        pd.DataFrame(cadence_rows).to_csv(OUT / f"cadence_{family}.csv", index=False)
    pd.DataFrame(scan_rows).to_csv(OUT / "development_scan.csv", index=False)

    train = panel[(panel.time >= pd.Timestamp("2021-01-01", tz="UTC")) &
                  (panel.time < pd.Timestamp("2023-01-01", tz="UTC")) &
                  panel.symbol.isin(pool_a)].copy()

    def evaluate_period(period: str, start: str, end: str, pool_name: str, pool: list[str], names: list[str]):
        period_rows = []
        pvalues = {}
        start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        test = panel[(panel.time >= start_ts) & (panel.time < end_ts) & panel.symbol.isin(pool)].copy()
        for family in names:
            spec0 = selected[family]
            name = str(spec0["candidate"])
            horizon = int(spec0["horizon"])
            cadence = int(spec0["cadence"])
            factor = pool_matrix(candidates[name]["matrix"], pool).loc[start_ts:end_ts - pd.Timedelta(days=1)]
            target = pool_matrix(labels[f"y{horizon}"], pool).loc[factor.index]
            ic = base.csic_series(factor, target)
            boot = base.block_bootstrap(ic, reps=4000, block=20)
            pvalues[family] = boot["p_nonpositive"]
            mse = ridge_increment(train, test, name, f"y{horizon}", parent_names)
            r2, residual_ic, maxcorr = residual_diagnostics(train, test, name, f"y{horizon}", parent_names)
            p2 = portfolio(factor, pool_matrix(next_daily_return, pool).loc[factor.index],
                           pool_matrix(daily_risk, pool).loc[factor.index],
                           hold_days=horizon // 24, cadence=cadence, cost_bp=2)
            p5 = portfolio(factor, pool_matrix(next_daily_return, pool).loc[factor.index],
                           pool_matrix(daily_risk, pool).loc[factor.index],
                           hold_days=horizon // 24, cadence=cadence, cost_bp=5)
            p10 = portfolio(factor, pool_matrix(next_daily_return, pool).loc[factor.index],
                            pool_matrix(daily_risk, pool).loc[factor.index],
                            hold_days=horizon // 24, cadence=cadence, cost_bp=10)
            positive_share, offsets = nonoverlap(ic, horizon // 24)
            yearly = {str(year): float(ic[ic.index.year == year].mean()) for year in sorted(set(ic.index.year))}
            period_rows.append({"period": period, "pool": pool_name, "family": family,
                                "candidate": name, "horizon_hours": horizon, "cadence_days": cadence,
                                "development_ics": spec0["development_ics"],
                                "development_execution": spec0["development_execution"],
                                "rankic": float(ic.mean()), **boot, "mse_gain_pct": mse,
                                "reconstruction_r2": r2, "residual_rankic": residual_ic,
                                "max_parent_corr": maxcorr, **p2, **p5, **p10,
                                "nonoverlap_positive_share": positive_share,
                                "nonoverlap_values": offsets, "yearly_rankic": yearly})
        adjusted = base.holm(pvalues)
        for row in period_rows:
            row["holm_p"] = adjusted[row["family"]]
            execution = row["development_execution"]
            row["pass"] = bool(
                min(row["development_ics"]) > 0
                and execution["min_return_5bp"] > 0
                and row["rankic"] > 0.015 and row["ci_low"] > 0 and row["holm_p"] <= 0.10
                and row["mse_gain_pct"] > 0 and row["residual_rankic"] > 0.005
                and row["max_parent_corr"] < 0.85
                and row["ret_5bp"] > 0 and row["ret_10bp"] > 0
                and row["nonoverlap_positive_share"] >= 0.70
                and all(value > 0 for value in row["yearly_rankic"].values())
            )
        return period_rows

    families_all = list(selected)
    validation_rows = []
    validation_rows += evaluate_period("validation", "2023-01-01", "2025-01-01", "pool_a", pool_a, families_all)
    validation_rows += evaluate_period("validation", "2023-01-01", "2025-01-01", "pool_b", pool_b, families_all)
    validation_pass = [
        family for family in families_all
        if all(row["pass"] for row in validation_rows if row["family"] == family)
        and len([row for row in validation_rows if row["family"] == family]) == 2
    ]

    later_rows = []
    if validation_pass:
        later_rows += evaluate_period("later", "2025-01-01", "2026-11-01", "pool_a", pool_a, validation_pass)
        later_rows += evaluate_period("later", "2025-01-01", "2026-11-01", "pool_b", pool_b, validation_pass)
    admitted = [
        {"family": family, "selection": selected[family],
         "validation": [row for row in validation_rows if row["family"] == family],
         "later": [row for row in later_rows if row["family"] == family]}
        for family in validation_pass
        if len([row for row in later_rows if row["family"] == family]) == 2
        and all(row["pass"] for row in later_rows if row["family"] == family)
    ]

    all_rows = validation_rows + later_rows
    pd.DataFrame(all_rows).to_csv(OUT / "evaluation_results.csv", index=False)
    summary = {
        "data": {"symbols": symbols, "pool_a": pool_a, "pool_b": pool_b,
                 "rows": int(len(data)), "min": str(data.time.min()), "max": str(data.time.max())},
        "mechanism": "single-asset stress-strain constitutive laws: viscosity, yield absorption, Maxwell relaxation, hysteresis, stored stress and low-impact release",
        "selection": "formula, direction, horizon and cadence selected only on pool A in separate 2021 and 2022 subperiods",
        "validation": "2023-2024 simultaneously on pool A and disjoint pool B",
        "later": "2025-2026 opened only for families passing both validation pools",
        "selected": selected,
        "validation_pass": validation_pass,
        "results": all_rows,
        "admitted": admitted,
        "admitted_count": len(admitted),
        "parent_features": parent_names,
        "governance": "Any admission is historical default-off and fresh-forward-required; no raw data are committed to Trade.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "admitted.json").write_text(json.dumps(admitted, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"pool_a": pool_a, "pool_b": pool_b,
                      "validation_pass": validation_pass,
                      "admitted_count": len(admitted),
                      "admitted": [item["family"] for item in admitted]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
