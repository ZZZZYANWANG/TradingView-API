from pathlib import Path

source = Path("research/fluid_network_energy_budget/run.py")
target = Path("research/fluid_coherent_backscatter/run.py")
target.parent.mkdir(parents=True, exist_ok=True)
s = source.read_text()
s = s.replace("research_outputs/fluid_network_energy_budget", "research_outputs/fluid_coherent_backscatter")
s = s.replace("HORIZONS = [72, 120, 240, 480]", "HORIZONS = [480]")
s = s.replace("CADENCES = [1, 2, 3, 5, 7, 10]", "CADENCES = [3, 5, 7, 10, 14, 20]")
start = s.index("    candidates: dict[str, dict[str, object]] = {}")
end = s.index("    hour0 = (idx.hour == 0) & (idx.minute == 0)")
replacement = '''    candidates: dict[str, dict[str, object]] = {}
    families: dict[str, list[str]] = {}
    def add(family: str, name: str, matrix: pd.DataFrame):
        candidates[name] = {"matrix": matrix, "family": family}
        families.setdefault(family, []).append(name)

    cb = coherent_backscatter
    cb_z = zframe(cb)
    coherence_gate = (1 + np.tanh(v * peer_df)) / 2
    pressure_gate = (1 + np.tanh(-pressure_work)) / 2
    sink_gate = (1 + np.tanh(-states["continuity"])) / 2
    low_circulation_gate = np.exp(-states["circulation"].abs() / 2)
    high_circulation_gate = (1 + np.tanh(states["circulation"].abs())) / 2

    for hours in [3, 6, 12, 24, 48, 72, 120]:
        integrated = cb.rolling(hours, max(2, hours // 2)).mean()
        for smooth in [1, 3, 5, 10]:
            add(
                "backscatter_persistence",
                f"persistence_h{hours}_s{smooth}",
                daily_sample(zframe(integrated), smooth),
            )

    for threshold in [0.0, 0.5, 1.0, 1.5]:
        active = (-states["subgrid"]) >= threshold
        event = cb.where(active, 0.0)
        for hours in [6, 12, 24, 48]:
            add(
                "backscatter_threshold_front",
                f"threshold_{threshold}_h{hours}",
                daily_sample(zframe(event.rolling(hours, max(3, hours // 2)).mean()), 3),
            )

    for amplitude_scale in [0.5, 1.0, 2.0, 4.0]:
        bounded = v * np.tanh((-states["subgrid"]).clip(lower=0) / amplitude_scale) * coherence_gate
        for hours in [6, 12, 24, 48]:
            add(
                "bounded_backscatter",
                f"bounded_s{amplitude_scale}_h{hours}",
                daily_sample(zframe(bounded.rolling(hours, max(3, hours // 2)).mean()), 3),
            )

    gate_map = {
        "pressure": pressure_gate,
        "sink": sink_gate,
        "low_circulation": low_circulation_gate,
        "high_circulation": high_circulation_gate,
    }
    for gate_name, gate in gate_map.items():
        gated = cb * gate
        for hours in [6, 12, 24, 48, 72]:
            add(
                f"backscatter_{gate_name}_gate",
                f"{gate_name}_h{hours}",
                daily_sample(zframe(gated.rolling(hours, max(3, hours // 2)).mean()), 3),
            )

    for threshold in [0.5, 1.0, 1.5]:
        for half_life in [24, 48, 72, 120]:
            accumulated = reservoir.reset_reservoir(
                cb,
                states["subgrid"],
                threshold=threshold,
                half_life=half_life,
            )
            add(
                "backscatter_reservoir",
                f"reservoir_t{threshold}_h{half_life}",
                daily_sample(accumulated, 3),
            )

    for hours in [6, 12, 24, 48]:
        slow = cb.rolling(72, 36).mean()
        front = cb.rolling(hours, max(3, hours // 2)).mean() - slow
        add(
            "backscatter_front",
            f"front_h{hours}",
            daily_sample(zframe(front), 3),
        )

    fixed_consensus = rank_consensus([
        daily_sample(zframe(cb.rolling(6, 3).mean()), 5),
        daily_sample(zframe((cb * pressure_gate).rolling(12, 6).mean()), 3),
        daily_sample(zframe((cb * sink_gate).rolling(12, 6).mean()), 3),
        daily_sample(zframe((cb * low_circulation_gate).rolling(24, 12).mean()), 3),
    ])
    add("backscatter_consensus", "backscatter_consensus", daily_zframe(fixed_consensus))

'''
s = s[:start] + replacement + s[end:]
needle = '        "pressure_work": daily_sample(pressure_work),\n'
if needle not in s:
    raise SystemExit("base feature insertion point not found")
s = s.replace(
    needle,
    needle + '        "coherent_backscatter_raw": daily_sample(cb_z),\n',
    1,
)
s = s.replace(
    '"mechanism": "graph pressure-work, compression-expansion release, subgrid backscatter, source absorption, baroclinic and rotation-strain topology",',
    '"mechanism": "peer-coherent subgrid backscatter persistence, fronts, bounded amplitudes and event-reset reservoirs",',
)
s = s.replace(
    '"holdout": "2025-2026-10 opened once for eleven selected mechanism families",',
    '"holdout": "2025-2026-10 historical evaluation for seven coherent-backscatter subfamilies",',
)
target.write_text(s)
