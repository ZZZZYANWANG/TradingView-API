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
ROUND18 = HERE.parent / "fluid_round18"
ROUND17 = HERE.parent / "fluid_round17"
ROUND16 = HERE.parent / "fluid_round16"
ROUND15 = HERE.parent / "fluid_round15"
for path in (ROUND18, ROUND17, ROUND16, ROUND15):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_dispersion_stability_round18 as r18  # noqa: E402

base = r18.base
r17 = r18.r17
inference = r18.inference
EPS = base.EPS
SEED = 20261119
OUT = Path("research_outputs/fluid_state_dependent_memory_round20")
OUT.mkdir(parents=True, exist_ok=True)

# Neither universe was used to select Round18's formulas.  The second universe is
# never downloaded unless a discovery identity passes validation and holdout.
DISCOVERY_SYMBOLS = [
    "AVAXUSDT", "ATOMUSDT", "NEARUSDT", "AAVEUSDT", "UNIUSDT", "FILUSDT",
    "ALGOUSDT", "XTZUSDT", "THETAUSDT", "RUNEUSDT", "GRTUSDT", "SNXUSDT",
]
REPLICATION_SYMBOLS = [
    "CHZUSDT", "COMPUSDT", "SUSHIUSDT", "DASHUSDT", "ZECUSDT", "EOSUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "LINKUSDT", "ETCUSDT",
]
HORIZONS = [1, 4, 24, 72, 120, 240, 480]
STATE_SPANS = [6, 24, 72]
LAGS = [1, 6, 24, 72, 168]

FAMILY_RAW: dict[str, list[str]] = {
    "source_memory_kernel": [
        "lead_conditioned_source_memory",
        "source_memory_centroid_loading",
        "causal_response_residual_release",
        "causal_response_alignment",
    ],
    "impedance_memory": [
        "impedance_memory_release",
        "impedance_transmission_memory",
        "reflection_hysteresis",
        "matched_source_absorption",
    ],
    "wave_action_memory": [
        "wave_action_memory_release",
        "wave_action_source_phase",
        "action_centroid_shift",
        "cascade_memory_alignment",
    ],
    "regime_switching_closure": [
        "lead_switch_response",
        "closure_breakdown_transmission",
        "memory_kh_reattachment",
        "source_pressure_phase_release",
    ],
}

CONTRACTS: dict[str, dict[str, str]] = {
    "lead_conditioned_source_memory": {
        "equation": "gamma_KH * M_source(72h) * tanh(spot_to_perp_lead_dominance)",
        "mechanism": "position and funding memory loads an unstable shear layer only when the propagation direction supports spot-to-perpetual transmission",
    },
    "source_memory_centroid_loading": {
        "equation": "gamma_KH * Z(memory_centroid_source) * sign(M_source)",
        "mechanism": "the same source impulse has a different effect when its effective memory mass is concentrated near or far in the past",
    },
    "causal_response_residual_release": {
        "equation": "-Z(Delta shear - sum_l beta_l source[t-l]) * Delta Z(residual)",
        "mechanism": "release of generalized-Langevin shear response unexplained by a causal distributed-lag source kernel",
    },
    "causal_response_alignment": {
        "equation": "predicted_shear_response * observed_shear * lead_regime",
        "mechanism": "alignment of the causal memory-kernel response with the current interlayer shear under the frozen propagation regime",
    },
    "impedance_memory_release": {
        "equation": "-R_Z * (pressure + M_pressure) * positive(Delta[-abs(R_Z)])",
        "mechanism": "release of present and remembered pressure when spot and perpetual impedances converge",
    },
    "impedance_transmission_memory": {
        "equation": "(1-|R_Z|)*(M_pressure_short-M_pressure_long)*lead_regime",
        "mechanism": "transmission of pressure-memory innovations through a matched boundary",
    },
    "reflection_hysteresis": {
        "equation": "R_Z*(M_pressure_short-M_pressure_long)*M_source",
        "mechanism": "hysteresis between pressure memory, source memory and impedance reflection",
    },
    "matched_source_absorption": {
        "equation": "(1-|R_Z|)*M_source*shear",
        "mechanism": "observable source flow absorbed by the interlayer shear when impedances are matched",
    },
    "wave_action_memory_release": {
        "equation": "-Z(R_action_memory)*Delta Z(R_action_memory)",
        "mechanism": "relaxation of wave-action disequilibrium after accounting for remembered sources",
    },
    "wave_action_source_phase": {
        "equation": "R_action_memory*M_source*lead_regime",
        "mechanism": "phase alignment of wave-action residual, source memory and propagation direction",
    },
    "action_centroid_shift": {
        "equation": "-Z(action)*Delta Z(memory_centroid_action)",
        "mechanism": "price response to migration of wave-action memory mass from old to recent lags",
    },
    "cascade_memory_alignment": {
        "equation": "joint_signed_cascade*M_source*lead_regime",
        "mechanism": "signed multiscale energy transfer supported by remembered position sources and the causal propagation direction",
    },
    "lead_switch_response": {
        "equation": "predicted_shear_response*lead_regime",
        "mechanism": "the distributed-lag response changes sign when the spot-perpetual leading layer changes",
    },
    "closure_breakdown_transmission": {
        "equation": "response_residual*(1-|R_Z|)",
        "mechanism": "unclosed memory response that can transmit rather than reflect at the interlayer boundary",
    },
    "memory_kh_reattachment": {
        "equation": "gamma_KH*M_source*positive(-Delta separation)*positive(Delta abs(shear))",
        "mechanism": "remembered sources sustain a Kelvin-Helmholtz interface while a separated boundary reattaches",
    },
    "source_pressure_phase_release": {
        "equation": "-Z(M_source*M_pressure)*Delta Z(M_source*M_pressure)*lead_regime",
        "mechanism": "release of the historical phase relationship between observable sources and interlayer pressure",
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def ewm(values: pd.Series, span: int, minimum: int | None = None) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").ewm(
        span=span,
        adjust=False,
        min_periods=minimum if minimum is not None else span,
    ).mean()


def prior_rolling_beta(x: pd.Series, y: pd.Series, window: int = 336, ridge: float = 0.05) -> pd.Series:
    """Causal univariate response coefficient estimated through t-1 only."""
    xs = pd.to_numeric(x, errors="coerce").shift(1)
    ys = pd.to_numeric(y, errors="coerce").shift(1)
    minimum = max(96, window // 2)
    mx = xs.rolling(window, min_periods=minimum).mean()
    my = ys.rolling(window, min_periods=minimum).mean()
    cov = (xs * ys).rolling(window, min_periods=minimum).mean() - mx * my
    var = xs.pow(2).rolling(window, min_periods=minimum).mean() - mx.pow(2)
    return cov / (var.clip(lower=0.0) + ridge)


def prior_lead_dominance(spot_u: pd.Series, perp_u: pd.Series, window: int = 168) -> pd.Series:
    """Positive means spot historically leads perpetual, using pairs ending at t-1."""
    minimum = max(72, window // 2)
    spot_leads = spot_u.shift(2).rolling(window, min_periods=minimum).corr(perp_u.shift(1))
    perp_leads = perp_u.shift(2).rolling(window, min_periods=minimum).corr(spot_u.shift(1))
    return spot_leads - perp_leads


def lag_memory(values: pd.Series, tau: float) -> pd.Series:
    weights = np.exp(-np.asarray(LAGS, dtype=float) / float(tau))
    weights = weights / weights.sum()
    pieces = [float(weight) * pd.to_numeric(values, errors="coerce").shift(lag) for lag, weight in zip(LAGS, weights)]
    return sum(pieces)


def memory_centroid(values: pd.Series) -> pd.Series:
    pieces = [pd.to_numeric(values, errors="coerce").shift(lag).abs() for lag in LAGS]
    denominator = sum(pieces) + EPS
    numerator = sum(float(lag) * piece for lag, piece in zip(LAGS, pieces))
    return numerator / denominator


def add_round20_mechanisms(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    g = group.sort_values("time").copy()
    u_p = pd.to_numeric(g["perp_u"], errors="coerce")
    u_s = pd.to_numeric(g["spot_u"], errors="coerce")
    shear = u_p - u_s
    strain = u_p + u_s
    shear_change = shear.diff()
    basis = pd.to_numeric(g["basis_z"], errors="coerce")
    pressure = pd.to_numeric(g["pressure_gap"], errors="coerce")
    pressure_gradient = pd.to_numeric(g["basis_change_z"], errors="coerce") + 0.25 * pd.to_numeric(g["funding_change_z"], errors="coerce").fillna(0.0)
    source = (
        pd.to_numeric(g["oi_source_z"], errors="coerce").fillna(0.0)
        + 0.50 * pd.to_numeric(g["funding_change_z"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(g["quote_share_z"], errors="coerce").diff().fillna(0.0)
    )

    lead_dom = prior_lead_dominance(u_s, u_p)
    lead_regime = np.tanh(base.causal_z(lead_dom).clip(-4.0, 4.0))
    source_m24 = lag_memory(source, 24.0)
    source_m72 = lag_memory(source, 72.0)
    source_m168 = lag_memory(source, 168.0)
    pressure_m24 = lag_memory(pressure, 24.0)
    pressure_m72 = lag_memory(pressure, 72.0)
    pressure_m168 = lag_memory(pressure, 168.0)
    source_centroid = base.causal_z(memory_centroid(source))

    # Causal diagonal distributed-lag response. Coefficients use pairs ending at t-1.
    predicted_shear = pd.Series(0.0, index=g.index, dtype=float)
    for lag in [1, 6, 24, 72]:
        x = source.shift(lag)
        beta = prior_rolling_beta(x, shear_change)
        predicted_shear = predicted_shear + beta * x
    response_residual = shear_change - predicted_shear

    flow_p = pd.to_numeric(g["imb6_z"], errors="coerce")
    spot_imbalance = (
        2.0 * pd.to_numeric(g["taker_buy_quote_spot"], errors="coerce")
        / (pd.to_numeric(g["quote_volume_spot"], errors="coerce") + EPS)
        - 1.0
    )
    flow_s = base.causal_z(spot_imbalance.rolling(6, min_periods=6).mean())
    rho_p = np.exp(np.clip((0.60 * pd.to_numeric(g["perp_quote_z"], errors="coerce") + 0.40 * pd.to_numeric(g["oi_level_z"], errors="coerce")) / 4.0, -3.0, 3.0))
    rho_s = np.exp(np.clip(pd.to_numeric(g["spot_quote_z"], errors="coerce") / 4.0, -3.0, 3.0))
    density_gradient = np.log(rho_p + EPS) - np.log(rho_s + EPS)
    dm = r18.prior_diffusivity(basis, 168).clip(lower=0.0)
    c_p = np.sqrt(dm / (rho_p + EPS) + EPS)
    c_s = np.sqrt(dm / (rho_s + EPS) + EPS)
    z_p = rho_p * c_p
    z_s = rho_s * c_s
    reflection = (z_p - z_s) / (z_p.abs() + z_s.abs() + EPS)
    matching_gain = (-reflection.abs().diff()).clip(lower=0.0)

    stratification = -pressure * density_gradient
    ri = (stratification / (shear.pow(2) + dm + EPS)).clip(-8.0, 8.0)
    kh_growth = shear.abs() * np.sqrt((1.0 - 4.0 * ri).clip(lower=0.0))
    displacement = density_gradient.abs() / (1.0 + density_gradient.abs())
    adverse = (-shear * pressure_gradient).clip(lower=0.0)
    separation = displacement * adverse / (shear.pow(2) + dm + EPS)
    reattachment = (-separation.diff()).clip(lower=0.0) * shear.abs().diff().clip(lower=0.0)

    energy = 0.5 * (rho_p * u_p.pow(2) + rho_s * u_s.pow(2))
    c_bar = 0.5 * (c_p + c_s)
    action = energy / (c_bar + EPS)
    action_residual = action.diff() + shear * action - source
    action_memory_residual = action_residual + 0.5 * lag_memory(action_residual, 72.0) - source_m72
    action_centroid = base.causal_z(memory_centroid(action))

    log_p = np.log(pd.to_numeric(g["close"], errors="coerce"))
    log_s = np.log(pd.to_numeric(g["close_spot"], errors="coerce"))
    cascades: list[pd.Series] = []
    for scale in [6, 24, 72]:
        dp = log_p.diff(scale) / math.sqrt(scale)
        ds = log_s.diff(scale) / math.sqrt(scale)
        cascades.append(-(dp.pow(3) + ds.pow(3)) / float(scale))
    joint_cascade = r18.robust_median(cascades)

    raw: dict[str, pd.Series] = {}
    raw["lead_conditioned_source_memory"] = kh_growth * source_m72 * lead_regime
    raw["source_memory_centroid_loading"] = kh_growth * source_centroid * np.sign(source_m72)
    residual_z = base.causal_z(response_residual)
    raw["causal_response_residual_release"] = -residual_z * residual_z.diff()
    raw["causal_response_alignment"] = predicted_shear * shear * lead_regime

    raw["impedance_memory_release"] = -reflection * (pressure + pressure_m72) * matching_gain
    raw["impedance_transmission_memory"] = (1.0 - reflection.abs()).clip(lower=0.0) * (pressure_m24 - pressure_m168) * lead_regime
    raw["reflection_hysteresis"] = reflection * (pressure_m24 - pressure_m168) * source_m72
    raw["matched_source_absorption"] = (1.0 - reflection.abs()).clip(lower=0.0) * source_m72 * shear

    action_residual_z = base.causal_z(action_memory_residual)
    raw["wave_action_memory_release"] = -action_residual_z * action_residual_z.diff()
    raw["wave_action_source_phase"] = action_memory_residual * source_m72 * lead_regime
    raw["action_centroid_shift"] = -base.causal_z(action) * action_centroid.diff()
    raw["cascade_memory_alignment"] = joint_cascade * source_m72 * lead_regime

    raw["lead_switch_response"] = predicted_shear * lead_regime
    raw["closure_breakdown_transmission"] = response_residual * (1.0 - reflection.abs()).clip(lower=0.0)
    raw["memory_kh_reattachment"] = kh_growth * source_m72 * reattachment
    phase = source_m24 * pressure_m24 - source_m168 * pressure_m168
    phase_z = base.causal_z(phase)
    raw["source_pressure_phase_release"] = -phase_z * phase_z.diff() * lead_regime

    raw_to_family = {name: family for family, names in FAMILY_RAW.items() for name in names}
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for name, values in raw.items():
        g[f"raw_{name}"] = values
        z = base.causal_z(values)
        for span in STATE_SPANS:
            column = f"{name}_ewm{span}"
            g[column] = ewm(z, span)
            candidate_map[raw_to_family[name]].append(column)
    return g, candidate_map


def build_panel(symbols: list[str], out: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    base.SYMBOLS = symbols
    base.MONTHS = list(pd.period_range("2021-01", "2026-09", freq="M"))
    base.HORIZONS = list(HORIZONS)
    base.OUT = out
    raw, audit = base.load_public_data()
    parts: list[pd.DataFrame] = []
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for symbol, group in raw.groupby("symbol", sort=False):
        local = base.local_features(group)
        enriched, local_map = add_round20_mechanisms(local)
        parts.append(enriched)
        candidate_map = local_map
        print(f"round20_mechanisms={symbol} rows={len(enriched)}", flush=True)
    if not parts:
        raise RuntimeError("no usable discovery data")
    panel = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    return panel, audit, candidate_map


def apply_frozen(panel: pd.DataFrame, selection: pd.DataFrame) -> None:
    for row in selection.itertuples(index=False):
        panel[str(row.oriented_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")


def robust_select(panel: pd.DataFrame, candidate_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Reuse Round18's predeclared multi-horizon rule, but allow only one identity per family.
    scan, selection = r18.robust_select(panel, candidate_map)
    if selection.empty:
        return scan, selection
    selection = (
        selection.sort_values(["family", "robust_multi_horizon_score", "development_score"], ascending=[True, False, False])
        .groupby("family", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    selection["oriented_factor"] = selection["factor"].map(lambda value: f"oriented_{value}")
    apply_frozen(panel, selection)
    return scan, selection


def validate_stage(panel: pd.DataFrame, selection: pd.DataFrame, prefix: str):
    return r18.validate_stage(panel, selection, prefix)


def main() -> None:
    np.random.seed(SEED)
    base.HORIZONS = list(HORIZONS)
    base.moving_block_bootstrap = inference.nonoverlap_issue_block_bootstrap

    discovery_panel, discovery_audit, candidate_map = build_panel(DISCOVERY_SYMBOLS, OUT / "discovery_cache")
    scan, selection = robust_select(discovery_panel, candidate_map)
    scan.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)
    selection.to_csv(OUT / "FROZEN_SELECTION.csv", index=False)
    discovery_validation, discovery_holdout, promoted = validate_stage(discovery_panel, selection, "DISCOVERY")
    write_json(OUT / "DISCOVERY_PROMOTED.json", promoted)

    confirmed: list[dict[str, Any]] = []
    replication_audit: dict[str, Any] | None = None
    replication_validation = pd.DataFrame()
    replication_holdout = pd.DataFrame()
    if promoted:
        frozen = selection.loc[selection["oriented_factor"].isin([item["oriented_factor"] for item in promoted])].copy()
        replication_panel, replication_audit, _ = build_panel(REPLICATION_SYMBOLS, OUT / "replication_cache")
        apply_frozen(replication_panel, frozen)
        replication_validation, replication_holdout, confirmed = validate_stage(replication_panel, frozen, "REPLICATION")
    else:
        pd.DataFrame().to_csv(OUT / "REPLICATION_VALIDATION_RESULTS.csv", index=False)
        pd.DataFrame().to_csv(OUT / "REPLICATION_HOLDOUT_RESULTS.csv", index=False)

    write_json(OUT / "CONFIRMED.json", confirmed)
    audit = {
        "seed": SEED,
        "discovery_symbols": DISCOVERY_SYMBOLS,
        "replication_symbols": REPLICATION_SYMBOLS,
        "discovery_data": discovery_audit,
        "replication_data": replication_audit,
        "raw_mechanisms": FAMILY_RAW,
        "candidate_contracts": CONTRACTS,
        "candidate_columns": int(sum(len(values) for values in candidate_map.values())),
        "horizons_hours": HORIZONS,
        "development": "2021-2022",
        "validation": "2023-2024",
        "historical_holdout": "2025-2026-09",
        "selection_rule": "Round18 multi-horizon rule; exactly one frozen identity per family; no validation or holdout tuning",
        "inference": "non-overlapping issue-grid moving-block bootstrap, 10000 repetitions",
        "gate": "CS>0.012, block p<=0.10, validation Holm<=0.10, Ridge MSE gain>0, residual CS>0.005, max parent corr<0.80, 5bp annual return>0 and all stage years positive",
        "replication_policy": "replication data are downloaded only if an identity passes discovery validation and holdout",
        "research_boundary": "historical research; no fresh blind holdout claim",
    }
    write_json(OUT / "DATA_AUDIT.json", audit)
    write_json(OUT / "CANDIDATE_CONTRACTS.json", CONTRACTS)
    summary = {
        "raw_primitives": int(sum(len(values) for values in FAMILY_RAW.values())),
        "candidate_columns": int(sum(len(values) for values in candidate_map.values())),
        "development_tests": int(len(scan)),
        "frozen_identities": int(len(selection)),
        "discovery_validation_passed": int(discovery_validation.get("validation_gate", pd.Series(dtype=bool)).sum()) if not discovery_validation.empty else 0,
        "discovery_holdout_opened": int(len(discovery_holdout)),
        "discovery_promoted": promoted,
        "replication_validation_passed": int(replication_validation.get("validation_gate", pd.Series(dtype=bool)).sum()) if not replication_validation.empty else 0,
        "replication_holdout_opened": int(len(replication_holdout)),
        "confirmed": confirmed,
        "decision": "eligible_for_trade_only_if_confirmed_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = [
        "# State-dependent non-Markovian fluid closure research round20",
        "",
        "This experiment tests whether spot-perpetual propagation direction, source-memory location and impedance matching explain the sign instability of earlier fixed fluid closures.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen selection",
        selection.to_markdown(index=False) if not selection.empty else "No identity satisfied the frozen multi-horizon development rule.",
        "",
        "## Discovery validation",
        discovery_validation.to_markdown(index=False) if not discovery_validation.empty else "Validation remained closed.",
        "",
        "## Discovery holdout",
        discovery_holdout.to_markdown(index=False) if not discovery_holdout.empty else "Holdout remained closed.",
        "",
        "## Replication validation",
        replication_validation.to_markdown(index=False) if not replication_validation.empty else "Replication remained closed.",
        "",
        "## Confirmed",
        "```json",
        json.dumps(confirmed, indent=2, sort_keys=True),
        "```",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        write_json(OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
