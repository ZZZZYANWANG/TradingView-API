from pathlib import Path

source = Path("research/fluid_network_energy_budget/run.py")
target = Path("research/fluid_backscatter_crossuniverse/run.py")
target.parent.mkdir(parents=True, exist_ok=True)
s = source.read_text()
s = s.replace(
    'base = reservoir.base\n',
    '''base = reservoir.base
base.SYMBOLS = [
    "AAVEUSDT", "ALGOUSDT", "ATOMUSDT", "AVAXUSDT", "COMPUSDT",
    "EOSUSDT", "FILUSDT", "NEARUSDT", "SUSHIUSDT", "TRXUSDT",
    "UNIUSDT", "XTZUSDT", "ZECUSDT",
]
''',
    1,
)
s = s.replace(
    'OUT = Path("research_outputs/fluid_network_energy_budget")',
    'OUT = Path("research_outputs/fluid_backscatter_crossuniverse")',
)
s = s.replace('HORIZONS = [72, 120, 240, 480]', 'HORIZONS = [480]')
s = s.replace('CADENCES = [1, 2, 3, 5, 7, 10]', 'CADENCES = [5, 14]')

candidate_start = s.index('    candidates: dict[str, dict[str, object]] = {}')
candidate_end = s.index('    hour0 = (idx.hour == 0) & (idx.minute == 0)')
replacement = '''    candidates: dict[str, dict[str, object]] = {}
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

'''
s = s[:candidate_start] + replacement + s[candidate_end:]
needle = '        "pressure_work": daily_sample(pressure_work),\n'
if needle not in s:
    raise SystemExit("base feature insertion point not found")
s = s.replace(
    needle,
    needle + '        "coherent_backscatter_raw": daily_sample(zframe(coherent_backscatter)),\n',
    1,
)

scan_start = s.index('    scan_rows = []')
summary_start = s.index('    summary = {', scan_start)
replication = '''    frozen = {
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

'''
s = s[:scan_start] + replication + s[summary_start:]
s = s.replace(
    '"mechanism": "graph pressure-work, compression-expansion release, subgrid backscatter, source absorption, baroclinic and rotation-strain topology",',
    '"mechanism": "frozen peer-coherent subgrid-backscatter formulas replicated on a disjoint asset universe",',
)
s = s.replace(
    '"selection": "pre-2025 worst-subperiod RankIC; inverse-volatility execution cadence selected pre-2025",',
    '"selection": "none on replication universe; formula, direction, horizon and cadence frozen from the original universe",',
)
s = s.replace(
    '"holdout": "2025-2026-10 opened once for eleven selected mechanism families",',
    '"holdout": "disjoint-universe validation in 2023-2024 and later evaluation in 2025-2026-10",',
)
s = s.replace('"selected": selected,', '"selected": frozen,')
s = s.replace('"admitted_count": len(admitted),', '"admitted_count": len(admitted),')
s = s.replace(
    '"governance": "Historical holdout only; any admitted identity remains default-off pending fresh forward data.",',
    '"governance": "Cross-universe historical replication; any admitted identity remains default-off and fresh-forward-required.",',
)
target.write_text(s)
