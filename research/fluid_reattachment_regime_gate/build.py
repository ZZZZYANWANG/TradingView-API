from pathlib import Path

source = Path("research/fluid_reattachment_frozen_daily/run.py")
target = Path("research/fluid_reattachment_regime_gate/run.py")
target.parent.mkdir(parents=True, exist_ok=True)
s = source.read_text()
s = s.replace(
    "research_outputs/fluid_reattachment_frozen_daily",
    "research_outputs/fluid_reattachment_regime_gate",
)
s = s.replace(
    "CADENCES = [1, 2, 3, 5, 7]",
    "CADENCES = [1, 2, 3, 5, 7, 10]",
)
old = '''    add("frozen_reattach_mean6", "frozen_reattach_mean6", base_features["reattach_mean6"])
    add("frozen_reattach_mean24", "frozen_reattach_mean24", base_features["reattach_mean24"])
    families = {k: v for k, v in families.items() if k.startswith("frozen_reattach_")}
'''
new = '''    base_factor = base_features["reattach_mean6"]
    scalar_frame = pd.DataFrame(index=base_factor.index)
    scalar_frame["separation"] = base_features["separation_z"].median(axis=1)
    scalar_frame["pressure_energy"] = base_features["pressure_z"].abs().median(axis=1)
    scalar_frame["source_energy"] = base_features["continuity_z"].abs().median(axis=1)
    scalar_frame["coherence"] = sample_daily(v * peer_df).median(axis=1)
    scalar_frame["market_volatility"] = base_features["rv20d"].median(axis=1)
    scalar_frame["return_dispersion"] = daily_ret.std(axis=1, ddof=0)
    scalar_frame["market_trend"] = base_features["mom20d"].median(axis=1)
    scalar_frame["reattachment_breadth"] = (base_factor > 0).mean(axis=1)
    scalar_frame = scalar_frame.apply(lambda x: daily_causal_z(x, 180, 60))
    development = scalar_frame.index < pd.Timestamp("2023-01-01", tz="UTC")
    columns = base_factor.columns
    for state in scalar_frame.columns:
        broadcast = pd.DataFrame(
            np.repeat(scalar_frame[[state]].to_numpy(), len(columns), axis=1),
            index=scalar_frame.index,
            columns=columns,
        )
        base_features[f"market_{state}"] = broadcast
        for quantile in [0.20, 0.35, 0.50, 0.65, 0.80]:
            threshold = float(scalar_frame.loc[development, state].quantile(quantile))
            high = scalar_frame[state] >= threshold
            low = scalar_frame[state] <= threshold
            add(
                f"regime_{state}",
                f"{state}_high_q{int(100 * quantile)}",
                base_factor.where(high, 0.0),
            )
            add(
                f"regime_{state}",
                f"{state}_low_q{int(100 * quantile)}",
                base_factor.where(low, 0.0),
            )
    pair_specs = [
        ("coherent_relaxed", "coherence", "high", 0.50, "pressure_energy", "low", 0.50),
        ("ordered_lowvol", "separation", "low", 0.50, "market_volatility", "low", 0.50),
        ("absorbing_ordered", "source_energy", "high", 0.50, "separation", "low", 0.50),
        ("broad_coherent", "reattachment_breadth", "high", 0.50, "coherence", "high", 0.50),
    ]
    for name, s1, side1, q1, s2, side2, q2 in pair_specs:
        t1 = float(scalar_frame.loc[development, s1].quantile(q1))
        t2 = float(scalar_frame.loc[development, s2].quantile(q2))
        g1 = scalar_frame[s1] >= t1 if side1 == "high" else scalar_frame[s1] <= t1
        g2 = scalar_frame[s2] >= t2 if side2 == "high" else scalar_frame[s2] <= t2
        add("regime_composite", name, base_factor.where(g1 & g2, 0.0))
    families = {k: v for k, v in families.items() if k.startswith("regime_")}
'''
if old not in s:
    raise SystemExit("candidate replacement block not found")
s = s.replace(old, new, 1)
s = s.replace(
    '    parents = [p for p in base_features if p not in {"reattach_mean6", "reattach_mean24"}]\n',
    '    parents = list(base_features)\n',
)
s = s.replace(
    "frozen six-hour and twenty-four-hour network reattachment means sampled once daily",
    "market-regime-gated frozen six-hour network reattachment factor",
)
target.write_text(s)
