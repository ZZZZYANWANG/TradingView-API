from pathlib import Path

source = Path("research/fluid_constitutive_law/run.py")
target = Path("research/fluid_constitutive_ts/run.py")
target.parent.mkdir(parents=True, exist_ok=True)
s = source.read_text()
s = s.replace(
    'OUT = Path("research_outputs/fluid_constitutive_law")',
    'OUT = Path("research_outputs/fluid_constitutive_ts")',
)
helpers = r'''

def tsic_components(factor: pd.DataFrame, target: pd.DataFrame):
    factor, target = factor.align(target, join="inner", axis=0)
    factor, target = factor.align(target, join="inner", axis=1)
    xr = factor.rank(axis=0, method="average", pct=True)
    yr = target.rank(axis=0, method="average", pct=True)
    xz = xr.sub(xr.mean(axis=0), axis=1).div(xr.std(axis=0, ddof=0) + EPS, axis=1)
    yz = yr.sub(yr.mean(axis=0), axis=1).div(yr.std(axis=0, ddof=0) + EPS, axis=1)
    daily = (xz * yz).mean(axis=1, skipna=True)
    per_symbol = {
        column: float(xr[column].corr(yr[column]))
        for column in xr.columns
        if xr[column].notna().sum() >= 30 and yr[column].notna().sum() >= 30
    }
    return float(daily.mean()), daily, per_symbol


def ts_portfolio(
    factor: pd.DataFrame,
    next_return: pd.DataFrame,
    risk: pd.DataFrame,
    *,
    hold_days: int,
    cadence: int,
    cost_bp: float,
):
    factor, next_return = factor.align(next_return, join="inner", axis=0)
    factor, next_return = factor.align(next_return, join="inner", axis=1)
    risk = risk.reindex(index=factor.index, columns=factor.columns).clip(lower=1.0e-4)
    issue = np.tanh(factor / 2.0).div(risk)
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


def ts_nonoverlap(factor: pd.DataFrame, target: pd.DataFrame, hold_days: int):
    days = ((factor.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")) /
            pd.Timedelta(days=1)).astype(int)
    values = []
    for offset in range(min(20, hold_days)):
        mask = (days % hold_days) == offset
        value, _, _ = tsic_components(factor.loc[mask], target.loc[mask])
        values.append(float(value))
    return float(np.mean(np.asarray(values) > 0)), values


def residual_ts_diagnostics(train, test, factor, target, parents):
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
    residual = te.pivot(index="time", columns="symbol", values="residual")
    future = te.pivot(index="time", columns="symbol", values=target)
    residual_ic, _, _ = tsic_components(residual, future)
    correlations = [abs(tr[factor].corr(tr[parent], method="spearman")) for parent in parents]
    return float(r2), float(residual_ic), float(np.nanmax(correlations))
'''
marker = '\ndef main() -> None:\n'
if marker not in s:
    raise SystemExit("main insertion marker missing")
s = s.replace(marker, helpers + marker, 1)

start = s.index('    development_years = [2021, 2022]')
end = s.index('    summary = {', start)
replacement = r'''    development_years = [2021, 2022]
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
                    ic, _, _ = tsic_components(factor, target)
                    values.append(ic)
                score = min(values) + 0.25 * float(np.mean(values))
                scan_rows.append({"family": family, "candidate": name, "horizon": horizon,
                                  "tsic_2021": values[0], "tsic_2022": values[1], "score": score})
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
                metrics.append(ts_portfolio(
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

    def evaluate_period(period, start, end, pool_name, pool, names):
        rows = []
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
            tsic, contribution, per_symbol = tsic_components(factor, target)
            boot = base.block_bootstrap(contribution, reps=4000, block=20)
            pvalues[family] = boot["p_nonpositive"]
            mse = ridge_increment(train, test, name, f"y{horizon}", parent_names)
            r2, residual_ic, maxcorr = residual_ts_diagnostics(train, test, name, f"y{horizon}", parent_names)
            p2 = ts_portfolio(factor, pool_matrix(next_daily_return, pool).loc[factor.index],
                              pool_matrix(daily_risk, pool).loc[factor.index],
                              hold_days=horizon // 24, cadence=cadence, cost_bp=2)
            p5 = ts_portfolio(factor, pool_matrix(next_daily_return, pool).loc[factor.index],
                              pool_matrix(daily_risk, pool).loc[factor.index],
                              hold_days=horizon // 24, cadence=cadence, cost_bp=5)
            p10 = ts_portfolio(factor, pool_matrix(next_daily_return, pool).loc[factor.index],
                               pool_matrix(daily_risk, pool).loc[factor.index],
                               hold_days=horizon // 24, cadence=cadence, cost_bp=10)
            positive_share = float(np.mean(np.asarray(list(per_symbol.values())) > 0)) if per_symbol else float("nan")
            offset_share, offsets = ts_nonoverlap(factor, target, horizon // 24)
            yearly = {}
            for year in sorted(set(factor.index.year)):
                yearly[str(year)] = tsic_components(factor[factor.index.year == year], target[target.index.year == year])[0]
            rows.append({"period": period, "pool": pool_name, "family": family,
                         "candidate": name, "horizon_hours": horizon, "cadence_days": cadence,
                         "development_ics": spec0["development_ics"],
                         "development_execution": spec0["development_execution"],
                         "tsic": tsic, **boot, "positive_symbol_share": positive_share,
                         "per_symbol_tsic": per_symbol,
                         "mse_gain_pct": mse, "reconstruction_r2": r2,
                         "residual_tsic": residual_ic, "max_parent_corr": maxcorr,
                         **p2, **p5, **p10,
                         "nonoverlap_positive_share": offset_share,
                         "nonoverlap_values": offsets, "yearly_tsic": yearly})
        adjusted = base.holm(pvalues)
        for row in rows:
            row["holm_p"] = adjusted[row["family"]]
            execution = row["development_execution"]
            row["pass"] = bool(
                min(row["development_ics"]) > 0
                and execution["min_return_5bp"] > 0
                and row["tsic"] > 0.015 and row["ci_low"] > 0 and row["holm_p"] <= 0.10
                and row["positive_symbol_share"] >= 0.60
                and row["mse_gain_pct"] > 0 and row["residual_tsic"] > 0.005
                and row["max_parent_corr"] < 0.85
                and row["ret_5bp"] > 0 and row["ret_10bp"] > 0
                and row["nonoverlap_positive_share"] >= 0.70
                and all(value > 0 for value in row["yearly_tsic"].values())
            )
        return rows

    families_all = list(selected)
    validation_rows = []
    validation_rows += evaluate_period("validation", "2023-01-01", "2025-01-01", "pool_a", pool_a, families_all)
    validation_rows += evaluate_period("validation", "2023-01-01", "2025-01-01", "pool_b", pool_b, families_all)
    validation_pass = [
        family for family in families_all
        if len([row for row in validation_rows if row["family"] == family]) == 2
        and all(row["pass"] for row in validation_rows if row["family"] == family)
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
'''
s = s[:start] + replacement + s[end:]
s = s.replace(
    '"mechanism": "single-asset stress-strain constitutive laws: viscosity, yield absorption, Maxwell relaxation, hysteresis, stored stress and low-impact release",',
    '"mechanism": "time-series stress-strain constitutive laws: viscosity, yield absorption, Maxwell relaxation, hysteresis, stored stress and low-impact release",',
)
s = s.replace(
    '"validation": "2023-2024 simultaneously on pool A and disjoint pool B",',
    '"validation": "2023-2024 time-series replication simultaneously on pool A and disjoint pool B",',
)
target.write_text(s)
