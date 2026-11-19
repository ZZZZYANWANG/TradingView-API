from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUND13 = HERE.parent / "fluid_round13"
if str(ROUND13) not in sys.path:
    sys.path.insert(0, str(ROUND13))

import run_baroclinic_ensemble as e  # noqa: E402
import run_baroclinic_torque as r  # noqa: E402

SEED = 20261119
EPS = 1.0e-12
OUT = Path("research_outputs/fluid_characteristic_wave_round14")
OUT.mkdir(parents=True, exist_ok=True)
r.OUT = OUT

HORIZONS = [12, 24, 72, 120, 240, 480]
SMOOTHING = [6, 24, 72]
SOUND_WINDOWS = [72, 168, 336]

FAMILY_RAW: dict[str, list[str]] = {
    "riemann_characteristics": [
        "riemann_outgoing_surplus",
        "riemann_incoming_absorption",
        "riemann_balance",
        "riemann_transmission_defect",
        "riemann_reflection_release",
    ],
    "shock_rarefaction": [
        "characteristic_shock",
        "characteristic_rarefaction",
        "shock_entropy_release",
        "shock_source_coupling",
    ],
    "wave_action": [
        "wave_action_divergence",
        "wave_action_residual",
        "wave_action_release",
        "adiabatic_action_defect",
    ],
    "crocco_balance": [
        "crocco_residual",
        "crocco_release",
        "crocco_reattachment",
        "crocco_source_coupling",
    ],
    "helicity_transport": [
        "helicity_level",
        "helicity_transport",
        "helicity_release",
        "helicity_source_coupling",
    ],
    "acoustic_impedance": [
        "impedance_mismatch",
        "impedance_matched_transport",
        "impedance_pressure_release",
        "impedance_energy_absorption",
    ],
    "spinodal_stability": [
        "negative_compressibility",
        "spinodal_pressure_density",
        "spinodal_stabilization",
        "spinodal_entropy_release",
    ],
    "characteristic_reattachment": [
        "characteristic_gap",
        "characteristic_reattachment",
        "incoming_wave_release",
        "outgoing_wave_persistence",
    ],
}

PRIOR_FLUID_PARENTS = [
    "baroclinic_total_z",
    "pv_transport_z",
    "vortex_stretch_z",
    "kelvin_residual_z",
    "impedance_reflection_z",
    "wave_action_gradient_z",
    "baroclinic_reattach_z",
    "torque_circulation_alignment_z",
    "continuity_residual",
    "graph_circulation",
    "simple_density_pressure_alignment",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def causal_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Strict-prior rolling slope of y on x."""
    xp = pd.to_numeric(x, errors="coerce").shift(1)
    yp = pd.to_numeric(y, errors="coerce").shift(1)
    minimum = max(36, int(0.60 * window))
    mx = xp.rolling(window, min_periods=minimum).mean()
    my = yp.rolling(window, min_periods=minimum).mean()
    cov = (xp * yp).rolling(window, min_periods=minimum).mean() - mx * my
    var = xp.pow(2).rolling(window, min_periods=minimum).mean() - mx.pow(2)
    return (cov / (var + EPS)).clip(-16.0, 16.0)


def add_local_characteristics(panel: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        pressure = pd.to_numeric(g["total_pressure"], errors="coerce")
        density = pd.to_numeric(g["oi_level_z"], errors="coerce")
        entropy = (
            0.50 * pd.to_numeric(g["rv24_z"], errors="coerce")
            + 0.35 * pd.to_numeric(g["range24_z"], errors="coerce")
            + 0.15 * pd.to_numeric(g["imb24_z"], errors="coerce").abs()
        )
        rho_eff = np.exp(np.clip(density / 4.0, -3.0, 3.0))
        temperature = np.exp(np.clip(entropy / 4.0, -3.0, 3.0))
        g["entropy_proxy"] = entropy
        g["density_eff"] = rho_eff
        g["temperature_proxy"] = temperature
        g["total_enthalpy"] = pressure / (rho_eff + EPS) + 0.5 * pd.to_numeric(g["u"], errors="coerce").pow(2)
        dp = pressure.diff()
        drho = density.diff()
        for window in SOUND_WINDOWS:
            c2 = causal_beta(dp, drho, window)
            g[f"sound_c2_{window}"] = c2
            g[f"sound_speed_{window}"] = np.sqrt(c2.abs() + 0.05).clip(0.15, 4.0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)


def lag(matrix: np.ndarray, periods: int = 1) -> np.ndarray:
    if periods <= 0:
        return matrix.copy()
    return np.vstack([np.full((periods, matrix.shape[1]), np.nan), matrix[:-periods]])


def flatten(values: np.ndarray, index: pd.MultiIndex) -> np.ndarray:
    return pd.Series(values.reshape(-1), index=index).to_numpy(dtype=float)


def robust_median(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    with np.errstate(all="ignore"):
        out = np.nanmedian(stack, axis=0)
    out[np.all(~np.isfinite(stack), axis=0)] = np.nan
    return out


def build_characteristic_features(
    panel: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    times = pd.Index(sorted(panel["time"].unique()))
    index = pd.MultiIndex.from_product([times, r.SYMBOLS], names=["time", "symbol"])
    base = panel.set_index(["time", "symbol"]).reindex(index)

    def matrix(column: str) -> np.ndarray:
        return base[column].unstack("symbol").reindex(index=times, columns=r.SYMBOLS).to_numpy(dtype=float, copy=True)

    u_raw = matrix("u")
    mask = np.isfinite(u_raw)
    u = r.row_fill(u_raw)
    pressure = r.row_fill(matrix("total_pressure"))
    rho_z = r.row_fill(matrix("oi_level_z"))
    rho = np.exp(np.clip(rho_z / 4.0, -3.0, 3.0))
    entropy = r.row_fill(matrix("entropy_proxy"))
    temperature = r.row_fill(matrix("temperature_proxy"))
    enthalpy = r.row_fill(matrix("total_enthalpy"))
    source = r.row_fill(matrix("oi_source_z"))
    mom = r.row_fill(matrix("mom24"))
    circulation = r.row_fill(matrix("graph_circulation"))
    grad_rho_parent = r.row_fill(matrix("grad_rho"))
    grad_pressure_parent = r.row_fill(matrix("grad_pressure"))

    peer_u = u @ weights.T
    peer_pressure = pressure @ weights.T
    peer_entropy = entropy @ weights.T
    peer_enthalpy = enthalpy @ weights.T
    peer_circulation = circulation @ weights.T
    grad_entropy = peer_entropy - entropy
    grad_enthalpy = peer_enthalpy - enthalpy

    by_name: dict[str, list[np.ndarray]] = {name: [] for names in FAMILY_RAW.values() for name in names}
    window_audit: dict[str, dict[str, float]] = {}

    for window in SOUND_WINDOWS:
        c2 = r.row_fill(matrix(f"sound_c2_{window}"))
        c = r.row_fill(matrix(f"sound_speed_{window}"))
        impedance = rho * c
        peer_impedance = impedance @ weights.T
        reflection = (impedance - peer_impedance) / (impedance + peer_impedance + EPS)
        transmission = 2.0 * peer_impedance / (impedance + peer_impedance + EPS)

        pressure_velocity = pressure / (impedance + EPS)
        j_plus = u + pressure_velocity
        j_minus = u - pressure_velocity
        peer_j_plus = j_plus @ weights.T
        peer_j_minus = j_minus @ weights.T

        outgoing = j_plus - peer_j_plus
        incoming = j_minus - peer_j_minus
        by_name["riemann_outgoing_surplus"].append(outgoing)
        by_name["riemann_incoming_absorption"].append(-incoming)
        by_name["riemann_balance"].append(0.5 * (outgoing - incoming))
        by_name["riemann_transmission_defect"].append(transmission * peer_j_plus - j_plus)
        by_name["riemann_reflection_release"].append(-reflection * peer_j_minus)

        lambda_plus = u + c
        lambda_minus = u - c
        div_plus = lambda_plus @ weights.T - lambda_plus
        div_minus = lambda_minus @ weights.T - lambda_minus
        shock = np.maximum(-div_plus, 0.0) * np.maximum(div_minus, 0.0)
        rarefaction = np.maximum(div_plus, 0.0) * np.maximum(-div_minus, 0.0)
        delta_entropy = entropy - lag(entropy)
        by_name["characteristic_shock"].append(shock)
        by_name["characteristic_rarefaction"].append(rarefaction)
        by_name["shock_entropy_release"].append(-shock * delta_entropy)
        by_name["shock_source_coupling"].append(shock * source)

        acoustic_energy = 0.25 * rho * (j_plus * j_plus + j_minus * j_minus)
        action = acoustic_energy / (c + 0.10)
        flux = 0.25 * rho * ((u + c) * j_plus * j_plus + (u - c) * j_minus * j_minus)
        peer_action = action @ weights.T
        peer_flux = flux @ weights.T
        action_divergence = peer_flux - flux
        action_delta = action - lag(action)
        action_residual = action_delta - action_divergence
        by_name["wave_action_divergence"].append(action_divergence)
        by_name["wave_action_residual"].append(action_residual)
        by_name["wave_action_release"].append(-action_residual * (action_residual - lag(action_residual)))
        c_delta = c - lag(c)
        by_name["adiabatic_action_defect"].append(
            action_delta / (np.abs(lag(action)) + 0.25)
            - c_delta / (np.abs(lag(c)) + 0.25)
        )

        crocco = grad_enthalpy - temperature * grad_entropy - u * circulation
        by_name["crocco_residual"].append(crocco)
        by_name["crocco_release"].append(-crocco * (crocco - lag(crocco)))
        prior_abs_crocco = np.abs(lag(crocco))
        by_name["crocco_reattachment"].append(
            np.sign(mom) * prior_abs_crocco * np.maximum(prior_abs_crocco - np.abs(crocco), 0.0)
        )
        by_name["crocco_source_coupling"].append(crocco * source)

        helicity = u * circulation
        peer_helicity = helicity @ weights.T
        by_name["helicity_level"].append(helicity)
        by_name["helicity_transport"].append(peer_helicity - helicity)
        by_name["helicity_release"].append(-helicity * (helicity - lag(helicity)))
        by_name["helicity_source_coupling"].append(source * 0.5 * (circulation + peer_circulation))

        by_name["impedance_mismatch"].append(reflection)
        by_name["impedance_matched_transport"].append(transmission * peer_u - u)
        by_name["impedance_pressure_release"].append(-reflection * (peer_pressure - pressure))
        peer_energy = acoustic_energy @ weights.T
        by_name["impedance_energy_absorption"].append((peer_energy - acoustic_energy) * (1.0 - np.abs(reflection)))

        negative_c2 = np.maximum(-c2, 0.0)
        prior_negative = lag(negative_c2)
        by_name["negative_compressibility"].append(negative_c2)
        by_name["spinodal_pressure_density"].append(negative_c2 * grad_rho_parent * grad_pressure_parent)
        by_name["spinodal_stabilization"].append(
            np.sign(mom) * prior_negative * np.maximum(prior_negative - negative_c2, 0.0)
        )
        by_name["spinodal_entropy_release"].append(-negative_c2 * delta_entropy)

        characteristic_gap = np.sqrt(outgoing * outgoing + incoming * incoming)
        prior_gap = lag(characteristic_gap)
        prior_incoming = np.abs(lag(incoming))
        by_name["characteristic_gap"].append(characteristic_gap)
        by_name["characteristic_reattachment"].append(
            np.sign(mom) * prior_gap * np.maximum(prior_gap - characteristic_gap, 0.0)
        )
        by_name["incoming_wave_release"].append(
            np.sign(mom) * prior_incoming * np.maximum(prior_incoming - np.abs(incoming), 0.0)
        )
        by_name["outgoing_wave_persistence"].append(
            np.sign(mom) * outgoing * np.exp(-np.clip(characteristic_gap, 0.0, 12.0))
        )

        window_audit[str(window)] = {
            "finite_sound_speed": int(np.isfinite(c[mask]).sum()),
            "median_sound_speed": float(np.nanmedian(c[mask])),
            "negative_c2_share": float(np.nanmean(c2[mask] < 0.0)),
        }

    result = base.reset_index()[["time", "symbol"]].copy()
    finite_counts: dict[str, int] = {}
    for name, arrays in by_name.items():
        values = robust_median(arrays)
        values[~mask] = np.nan
        result[name] = flatten(values, index)
        finite_counts[name] = int(np.isfinite(values).sum())

    audit = {
        "sound_windows": SOUND_WINDOWS,
        "window_diagnostics": window_audit,
        "finite_by_raw_candidate": finite_counts,
        "interpretation": {
            "sound_speed": "strict-prior rolling pressure-density response magnitude",
            "riemann_invariants": "linearized characteristic amplitudes u +/- p/Z",
            "crocco_residual": "graph enthalpy gradient minus entropy gradient and circulation work",
            "wave_action": "acoustic energy divided by characteristic speed with graph flux residual",
        },
    }
    return result, audit


def add_candidate_states(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    parts: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        for family, raw_names in FAMILY_RAW.items():
            for raw in raw_names:
                state_name = f"{raw}_z"
                state = r.causal_z(g[raw], 720, 240)
                g[state_name] = state
                if state_name not in candidate_map[family]:
                    candidate_map[family].append(state_name)
                for span in SMOOTHING:
                    name = f"{raw}_ewm{span}"
                    g[name] = state.ewm(
                        span=span,
                        min_periods=max(4, int(math.ceil(0.65 * span))),
                        adjust=False,
                    ).mean()
                    if name not in candidate_map[family]:
                        candidate_map[family].append(name)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    return out, candidate_map


def signed_median(frame: pd.DataFrame, members: list[tuple[str, int]]) -> pd.Series:
    matrix = np.column_stack([
        sign * pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        for name, sign in members
    ])
    with np.errstate(all="ignore"):
        values = np.nanmedian(matrix, axis=1)
    values[np.all(~np.isfinite(matrix), axis=1)] = np.nan
    return pd.Series(values, index=frame.index, dtype=float)


def development_scan(
    panel: pd.DataFrame,
    candidate_map: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, factors in candidate_map.items():
        for factor in factors:
            for horizon in HORIZONS:
                rows.append(r.development_score(panel, factor, family, horizon))
        print(f"development_complete={family} candidates={len(factors)}", flush=True)
    return pd.DataFrame(rows)


def freeze_family_winners(panel: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    winners = (
        scan.sort_values(["family", "development_score"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    oriented_rows: list[dict[str, Any]] = []
    for row in winners.itertuples(index=False):
        name = f"selected_{row.family}"
        panel[name] = int(row.sign) * pd.to_numeric(panel[row.factor], errors="coerce")
        payload = row._asdict()
        payload["primitive_factor"] = row.factor
        payload["primitive_sign"] = int(row.sign)
        payload["factor"] = name
        payload["sign"] = 1
        oriented_rows.append(payload)
    return pd.DataFrame(oriented_rows)


def add_composites(panel: pd.DataFrame, family_winners: pd.DataFrame) -> pd.DataFrame:
    winner_by_family = dict(zip(family_winners["family"], family_winners["factor"]))
    ranked = family_winners.sort_values("development_score", ascending=False)["family"].tolist()
    specs = {
        "consensus_characteristic_wave": [
            family for family in [
                "riemann_characteristics", "shock_rarefaction", "wave_action", "acoustic_impedance"
            ] if family in winner_by_family
        ],
        "consensus_crocco_rotation": [
            family for family in [
                "crocco_balance", "helicity_transport", "spinodal_stability", "characteristic_reattachment"
            ] if family in winner_by_family
        ],
        "consensus_cross_family_top4": ranked[:4],
    }
    rows: list[dict[str, Any]] = []
    for family, families in specs.items():
        members = [(winner_by_family[name], 1) for name in families]
        raw_name = f"raw_{family}"
        panel[raw_name] = signed_median(panel, members)
        grid = [r.development_score(panel, raw_name, family, horizon) for horizon in HORIZONS]
        best = max(grid, key=lambda x: x["development_score"])
        selected = f"selected_{family}"
        panel[selected] = int(best["sign"]) * panel[raw_name]
        best["primitive_factor"] = raw_name
        best["primitive_sign"] = int(best["sign"])
        best["factor"] = selected
        best["sign"] = 1
        best["members"] = json.dumps(members)
        rows.append(best)
    return pd.DataFrame(rows)


def validation_gate(frame: pd.DataFrame) -> pd.Series:
    yearly_ok: list[bool] = []
    for payload in frame["yearly_rankic"]:
        years = json.loads(payload)
        yearly_ok.append(
            bool(years)
            and all(np.isfinite(float(value)) and float(value) > 0.0 for value in years.values())
        )
    return (
        (frame["cs_rankic"] > 0.012)
        & (frame["bootstrap_p_nonpositive"] <= 0.10)
        & (frame["ridge_mse_gain"] > 0.0)
        & (frame["residual_cs_rankic"] > 0.005)
        & (frame["max_parent_corr"] < 0.80)
        & (frame["annual_log_return_5bp"] > 0.0)
        & (frame["holm_p"] <= 0.10)
        & pd.Series(yearly_ok, index=frame.index)
    )


def report(
    audit: dict[str, Any],
    winners: pd.DataFrame,
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    promoted: list[dict[str, Any]],
) -> str:
    lines = [
        "# Compressible Characteristic-Wave Fluid Research Round 14",
        "",
        "This run tests strict-prior acoustic characteristic amplitudes, shock/rarefaction convergence, wave-action conservation, Crocco balance, helicity transport, impedance matching, negative compressibility and characteristic reattachment on public USD-M perpetual-futures data.",
        "",
        "## Protocol",
        "",
        "- development: 2021-2022",
        "- validation: 2023-2024",
        "- historical holdout opened only for validation-qualified identities: 2025-2026-09",
        "- labels begin at the next completed hourly open",
        "- open-interest metrics and funding are embargoed by one completed hour",
        "- strong parents include conventional states and prior baroclinic, potential-vorticity, circulation, impedance and reattachment coordinates",
        "",
        "## Development-frozen identities",
        "",
        winners.to_markdown(index=False),
        "",
        "## Validation",
        "",
        validation.to_markdown(index=False),
        "",
        "## Historical holdout",
        "",
        holdout.to_markdown(index=False) if not holdout.empty else "No identity passed the validation gate; historical holdout remained closed.",
        "",
        "## Promotion",
        "",
        "```json",
        json.dumps(promoted, indent=2, sort_keys=True),
        "```",
        "",
        "A promoted identity must retain positive validation and holdout RankIC, positive strong-parent Ridge increment, positive residual RankIC, low parent correlation, positive 5bp non-overlap return, positive yearly RankIC and Holm-adjusted validation p <= 0.10. Historical holdout is not fresh live evidence.",
        "",
        "## Data audit summary",
        "",
        "```json",
        json.dumps({
            "symbols": audit["klines"]["symbols"],
            "rows": audit["klines"]["rows"],
            "min_time": audit["klines"]["min_time"],
            "max_time": audit["klines"]["max_time"],
            "sound_windows": audit["characteristic"]["sound_windows"],
        }, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    np.random.seed(SEED)
    klines, kline_audit = r.load_klines()
    metrics, funding, external_audit = r.load_external()
    merged = r.merge_external(klines, metrics, funding)

    local_parts: list[pd.DataFrame] = []
    for symbol, group in merged.groupby("symbol", sort=False):
        local = r.compute_local_features(group)
        local["symbol"] = symbol
        local_parts.append(local)
        print(f"local_features={symbol} rows={len(local)}", flush=True)
    local_panel = pd.concat(local_parts, ignore_index=True)
    weights, directed, graph_audit = r.build_graph(local_panel)
    graph_panel = r.graph_state(local_panel, weights, directed)
    panel = local_panel.merge(graph_panel, on=["time", "symbol"], how="left", validate="one_to_one")
    panel = r.add_candidate_states(panel)
    panel = add_local_characteristics(panel)
    characteristic_panel, characteristic_audit = build_characteristic_features(panel, weights)
    panel = panel.merge(characteristic_panel, on=["time", "symbol"], how="left", validate="one_to_one")
    panel, candidate_map = add_candidate_states(panel)

    extra_parents = [name for name in PRIOR_FLUID_PARENTS if name in panel.columns]
    r.BASE_FEATURES = list(dict.fromkeys(r.BASE_FEATURES + extra_parents))

    scan = development_scan(panel, candidate_map)
    scan.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)
    family_winners = freeze_family_winners(panel, scan)
    composites = add_composites(panel, family_winners)
    selection = pd.concat([family_winners, composites], ignore_index=True)
    selection.to_csv(OUT / "FROZEN_SELECTION.csv", index=False)

    validation_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        result = e.evaluate_imputed(panel, row.factor, int(row.horizon), 2022, 2023, 2024)
        result["family"] = row.family
        result["primitive_factor"] = row.primitive_factor
        result["members"] = getattr(row, "members", "")
        validation_rows.append(result)
        print(f"validation_complete={row.family} factor={row.factor}", flush=True)
    validation = pd.DataFrame(validation_rows)
    raw_p = [float(value) if np.isfinite(value) else 1.0 for value in validation["bootstrap_p_nonpositive"]]
    validation["holm_p"] = r.holm_adjust(raw_p)
    validation["validation_gate"] = validation_gate(validation)
    validation.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)

    opened = validation.loc[validation["validation_gate"]].copy()
    holdout_rows: list[dict[str, Any]] = []
    for row in opened.itertuples(index=False):
        result = e.evaluate_imputed(panel, row.factor, int(row.horizon), 2024, 2025, 2026)
        result["family"] = row.family
        result["validation_holm_p"] = row.holm_p
        holdout_rows.append(result)
        print(f"holdout_complete={row.family} factor={row.factor}", flush=True)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=list(validation.columns)).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    else:
        holdout.to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)

    promoted: list[dict[str, Any]] = []
    for row in holdout.itertuples(index=False):
        years = json.loads(row.yearly_rankic)
        yearly_ok = bool(years) and all(
            np.isfinite(float(value)) and float(value) > 0.0 for value in years.values()
        )
        passed = (
            row.cs_rankic > 0.012
            and row.bootstrap_p_nonpositive <= 0.10
            and row.ridge_mse_gain > 0.0
            and row.residual_cs_rankic > 0.005
            and row.max_parent_corr < 0.80
            and row.annual_log_return_5bp > 0.0
            and yearly_ok
        )
        if passed:
            promoted.append({
                "family": row.family,
                "factor": row.factor,
                "horizon": int(row.horizon),
                "validation_holm_p": float(row.validation_holm_p),
                "holdout_cs_rankic": float(row.cs_rankic),
                "holdout_ts_ic": float(row.ts_ic),
                "holdout_ridge_mse_gain": float(row.ridge_mse_gain),
                "holdout_residual_cs_rankic": float(row.residual_cs_rankic),
                "holdout_max_parent_corr": float(row.max_parent_corr),
                "holdout_annual_log_return_5bp": float(row.annual_log_return_5bp),
            })

    sequential = e.pairwise_sequential(panel, pd.DataFrame(promoted)) if len(promoted) > 1 else pd.DataFrame()
    sequential.to_csv(OUT / "SEQUENTIAL_INDEPENDENCE.csv", index=False)

    diagnostics: list[dict[str, Any]] = []
    for row in opened.itertuples(index=False):
        for horizon in HORIZONS:
            for period, train_end, start, end in [
                ("validation", 2022, 2023, 2024),
                ("historical_holdout", 2024, 2025, 2026),
            ]:
                result = e.evaluate_imputed(panel, row.factor, horizon, train_end, start, end)
                diagnostics.append({
                    "family": row.family,
                    "factor": row.factor,
                    "selected_horizon": int(row.horizon),
                    "tested_horizon": horizon,
                    "period": period,
                    "cs_rankic": result["cs_rankic"],
                    "ts_ic": result["ts_ic"],
                    "ridge_mse_gain": result["ridge_mse_gain"],
                    "residual_cs_rankic": result["residual_cs_rankic"],
                    "annual_log_return_5bp": result["annual_log_return_5bp"],
                })
    pd.DataFrame(diagnostics).to_csv(OUT / "MULTI_HORIZON_DIAGNOSTICS.csv", index=False)

    audit = {
        "klines": kline_audit,
        "external": external_audit,
        "graph": graph_audit,
        "characteristic": characteristic_audit,
        "merged_rows": int(len(merged)),
        "panel_rows": int(len(panel)),
        "strong_parent_count": len(r.BASE_FEATURES),
        "prior_fluid_parents": extra_parents,
        "candidate_families": len(candidate_map),
        "candidate_columns": sum(len(values) for values in candidate_map.values()),
        "availability_policy": {
            "klines": "completed hour only; target begins at next-hour open",
            "metrics": "as-of joined and shifted by one completed hour",
            "funding": "last published event shifted by one completed hour",
            "sound_speed": "rolling pressure-density slope uses t-1 and earlier observations",
        },
    }
    write_json(OUT / "DATA_AUDIT.json", audit)
    write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "candidate_families": len(candidate_map),
        "candidate_columns": sum(len(values) for values in candidate_map.values()),
        "development_tests": int(len(scan)),
        "frozen_identities": int(len(selection)),
        "validation_gate_passed": int(validation["validation_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "promoted": promoted,
        "decision": "promote_only_if_PROMOTED_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    (OUT / "REPORT.md").write_text(
        report(audit, selection, validation, holdout, promoted), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_json(OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
