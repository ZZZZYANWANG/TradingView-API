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
ROUND16 = HERE.parent / "fluid_round16"
ROUND15 = HERE.parent / "fluid_round15"
for path in (ROUND16, ROUND15):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_multiscale_constitutive as r16  # noqa: E402
import run_multiscale_constitutive_inference_fix as inference  # noqa: E402

base = r16.base
SEED = 20261119
EPS = base.EPS
OUT = Path("research_outputs/fluid_cascade_direction_round17")
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_SYMBOLS = [
    "TRXUSDT", "EOSUSDT", "ATOMUSDT", "XTZUSDT", "ZECUSDT", "DASHUSDT",
    "FILUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "AVAXUSDT", "COMPUSDT",
]
REPLICATION_SYMBOLS = [
    "ALGOUSDT", "VETUSDT", "NEARUSDT", "GRTUSDT", "SNXUSDT", "RUNEUSDT",
    "YFIUSDT", "CRVUSDT", "MKRUSDT", "EGLDUSDT", "CHZUSDT", "MANAUSDT",
]
SCALES = [6, 24, 72]
STATE_SPANS = [6, 24, 72]
HORIZONS = [1, 4, 24, 72, 120, 240, 480]

FAMILY_RAW: dict[str, list[str]] = {
    "duchon_robert": [
        "joint_dissipation_anomaly", "interlayer_dissipation_gap",
        "coherent_forward_dissipation", "coherent_inverse_backscatter",
        "dissipation_release", "dissipation_basis_work",
    ],
    "cascade_direction": [
        "forward_scale_coherence", "inverse_scale_coherence",
        "layer_direction_mismatch", "cascade_reversal_impulse",
        "cascade_direction_entropy", "directional_persistence",
    ],
    "mixed_yaglom": [
        "perp_orderflow_yaglom", "spot_orderflow_yaglom",
        "interlayer_yaglom_gap", "joint_yaglom_transport",
        "oi_yaglom_source", "basis_yaglom_coupling",
    ],
    "intermittency": [
        "interlayer_flatness_gap", "joint_flatness_state",
        "zeta3_anomaly_gap", "multifractal_width_gap",
        "burst_localization_gap", "intermittency_release",
    ],
    "causal_precedence": [
        "cross_layer_flux_commutator", "perp_to_spot_transfer_efficiency",
        "spot_to_perp_transfer_efficiency", "basis_mediated_flux_transfer",
        "source_mediated_flux_transfer", "causal_flux_propagation_defect",
    ],
    "burgers_entropy": [
        "joint_shock_entropy", "interlayer_shock_entropy_gap",
        "joint_rarefaction_excess", "shock_entropy_release",
        "shock_basis_coupling", "shock_oi_coupling",
    ],
}

CONTRACTS: dict[str, dict[str, str]] = {
    "joint_dissipation_anomaly": {
        "equation": "median_l[-((delta_l u_perp)^3+(delta_l u_spot)^3)/l]",
        "mechanism": "Duchon-Robert-style anomalous dissipation shared by cash and perpetual layers",
    },
    "interlayer_dissipation_gap": {
        "equation": "median_l[-(delta_l u_perp)^3/l+(delta_l u_spot)^3/l]",
        "mechanism": "difference in singular energy dissipation between derivative and cash layers",
    },
    "coherent_forward_dissipation": {
        "equation": "median_l[min(D_perp,D_spot) when both positive]",
        "mechanism": "simultaneous forward cascade in both layers",
    },
    "coherent_inverse_backscatter": {
        "equation": "median_l[min(-D_perp,-D_spot) when both negative]",
        "mechanism": "simultaneous inverse cascade or backscatter in both layers",
    },
    "dissipation_release": {
        "equation": "-Z(D_joint)*d_t Z(D_joint)",
        "mechanism": "relaxation of accumulated anomalous dissipation",
    },
    "dissipation_basis_work": {
        "equation": "D_gap*basis_z",
        "mechanism": "work done by the spot-perpetual pressure boundary on the dissipation gap",
    },
    "forward_scale_coherence": {
        "equation": "mean_l[1(D_perp>0)+1(D_spot>0)]*median_l(|D|)",
        "mechanism": "magnitude-weighted coherence of forward cascade across scales and layers",
    },
    "inverse_scale_coherence": {
        "equation": "mean_l[1(D_perp<0)+1(D_spot<0)]*median_l(|D|)",
        "mechanism": "magnitude-weighted coherence of inverse cascade across scales and layers",
    },
    "layer_direction_mismatch": {
        "equation": "mean_l[sign(D_perp)!=sign(D_spot)]*median_l(|D_perp-D_spot|)",
        "mechanism": "cash and derivative layers transporting energy in opposite directions",
    },
    "cascade_reversal_impulse": {
        "equation": "d_t(forward_coherence-inverse_coherence)*median_l(|D|)",
        "mechanism": "impulse associated with a forward-to-inverse or inverse-to-forward cascade reversal",
    },
    "cascade_direction_entropy": {
        "equation": "binary_entropy(fraction of positive scale-layer fluxes)*median_l(|D|)",
        "mechanism": "uncertainty of cascade direction rather than flux magnitude alone",
    },
    "directional_persistence": {
        "equation": "direction_coherence_t*direction_coherence_{t-1}*median_l(|D|)",
        "mechanism": "persistence versus alternation of cascade direction",
    },
    "perp_orderflow_yaglom": {
        "equation": "median_l[delta_l u_perp*(delta_l imbalance_perp)^2/l]",
        "mechanism": "mixed third-order transport of perpetual order-flow variance",
    },
    "spot_orderflow_yaglom": {
        "equation": "median_l[delta_l u_spot*(delta_l imbalance_spot)^2/l]",
        "mechanism": "mixed third-order transport of spot order-flow variance",
    },
    "interlayer_yaglom_gap": {
        "equation": "Y_perp-Y_spot",
        "mechanism": "difference in mixed price-flow variance transport across layers",
    },
    "joint_yaglom_transport": {
        "equation": "sign(Y_perp+Y_spot)*(|Y_perp|+|Y_spot|)",
        "mechanism": "directional joint Yaglom transport",
    },
    "oi_yaglom_source": {
        "equation": "median_l[delta_l u_perp*(delta_l OI_source)^2/l]",
        "mechanism": "transport of position-source variance by perpetual price increments",
    },
    "basis_yaglom_coupling": {
        "equation": "interlayer_yaglom_gap*basis_z",
        "mechanism": "basis boundary modulating mixed third-order transport",
    },
    "interlayer_flatness_gap": {
        "equation": "F_perp(24h)-F_spot(24h), F=E[du^4]/E[du^2]^2",
        "mechanism": "difference in small-scale intermittency between derivative and cash layers",
    },
    "joint_flatness_state": {
        "equation": "sqrt(F_perp*F_spot)",
        "mechanism": "shared heavy-tailed intermittency state",
    },
    "zeta3_anomaly_gap": {
        "equation": "(zeta3_perp-1)-(zeta3_spot-1)",
        "mechanism": "cross-layer difference in departure from the third-order linear cascade law",
    },
    "multifractal_width_gap": {
        "equation": "(zeta4-2*zeta2)_perp-(zeta4-2*zeta2)_spot",
        "mechanism": "difference in multifractal curvature rather than volatility level",
    },
    "burst_localization_gap": {
        "equation": "max_l|D_l|/sum_l|D_l| difference between layers",
        "mechanism": "whether dissipation is localized at one scale or distributed across scales",
    },
    "intermittency_release": {
        "equation": "-Z(flatness_gap)*d_t Z(flatness_gap)",
        "mechanism": "relaxation of an interlayer intermittency imbalance",
    },
    "cross_layer_flux_commutator": {
        "equation": "D_perp(t-1)D_spot(t)-D_spot(t-1)D_perp(t)",
        "mechanism": "antisymmetric lead-lag circulation of energy flux between layers",
    },
    "perp_to_spot_transfer_efficiency": {
        "equation": "D_perp(t-1)D_spot(t)/(abs(D_perp(t-1))+eps)",
        "mechanism": "causal transfer efficiency from perpetual cascade to spot cascade",
    },
    "spot_to_perp_transfer_efficiency": {
        "equation": "D_spot(t-1)D_perp(t)/(abs(D_spot(t-1))+eps)",
        "mechanism": "causal transfer efficiency from spot cascade to perpetual cascade",
    },
    "basis_mediated_flux_transfer": {
        "equation": "D_perp(t-1)*basis_z(t-1)*D_spot(t)",
        "mechanism": "basis-mediated derivative-to-cash energy propagation",
    },
    "source_mediated_flux_transfer": {
        "equation": "D_perp(t-1)*OI_source(t-1)*D_spot(t)",
        "mechanism": "position-source-mediated energy propagation",
    },
    "causal_flux_propagation_defect": {
        "equation": "D_spot(t)-beta_prior*D_perp(t-1)",
        "mechanism": "spot flux innovation outside a causal derivative-led propagation law",
    },
    "joint_shock_entropy": {
        "equation": "median_l[positive(-delta_l u_perp)^3+positive(-delta_l u_spot)^3]/l",
        "mechanism": "Burgers-style entropy production from compressive price shocks",
    },
    "interlayer_shock_entropy_gap": {
        "equation": "shock_entropy_perp-shock_entropy_spot",
        "mechanism": "difference in compressive shock dissipation across layers",
    },
    "joint_rarefaction_excess": {
        "equation": "median_l[positive(delta_l u_perp)^3+positive(delta_l u_spot)^3]/l",
        "mechanism": "expansive rarefaction energy opposed to shock dissipation",
    },
    "shock_entropy_release": {
        "equation": "-Z(shock_entropy_gap)*d_t Z(shock_entropy_gap)",
        "mechanism": "release of accumulated shock-dissipation asymmetry",
    },
    "shock_basis_coupling": {
        "equation": "shock_entropy_gap*basis_z",
        "mechanism": "basis boundary loading on asymmetric shock entropy production",
    },
    "shock_oi_coupling": {
        "equation": "joint_shock_entropy*OI_source_z",
        "mechanism": "position creation or destruction coincident with compressive shocks",
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


def robust_median(items: list[pd.Series]) -> pd.Series:
    return r16.robust_median(items)


def slope_three(values: dict[int, pd.Series]) -> pd.Series:
    return r16.slope_three(values[6], values[24], values[72])


def binary_entropy(probability: pd.Series) -> pd.Series:
    p = pd.to_numeric(probability, errors="coerce").clip(EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def add_round17_mechanisms(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    g = group.sort_values("time").copy()
    log_p = np.log(pd.to_numeric(g["close"], errors="coerce"))
    log_s = np.log(pd.to_numeric(g["close_spot"], errors="coerce"))
    flow_p = pd.to_numeric(g["imb6_z"], errors="coerce")
    flow_s = (
        2.0 * pd.to_numeric(g["taker_buy_quote_spot"], errors="coerce")
        / (pd.to_numeric(g["quote_volume_spot"], errors="coerce") + EPS)
        - 1.0
    )
    flow_s = base.causal_z(flow_s.rolling(6, min_periods=6).mean())
    oi_source = pd.to_numeric(g["oi_source_z"], errors="coerce").fillna(0.0)
    basis = pd.to_numeric(g["basis_z"], errors="coerce")

    dp: dict[int, pd.Series] = {}
    ds: dict[int, pd.Series] = {}
    dflow_p: dict[int, pd.Series] = {}
    dflow_s: dict[int, pd.Series] = {}
    doi: dict[int, pd.Series] = {}
    diss_p: dict[int, pd.Series] = {}
    diss_s: dict[int, pd.Series] = {}
    yaglom_p: dict[int, pd.Series] = {}
    yaglom_s: dict[int, pd.Series] = {}
    yaglom_oi: dict[int, pd.Series] = {}
    shock_p: dict[int, pd.Series] = {}
    shock_s: dict[int, pd.Series] = {}
    rare_p: dict[int, pd.Series] = {}
    rare_s: dict[int, pd.Series] = {}

    for scale in SCALES:
        dp[scale] = log_p.diff(scale) / math.sqrt(scale)
        ds[scale] = log_s.diff(scale) / math.sqrt(scale)
        dflow_p[scale] = flow_p.diff(scale)
        dflow_s[scale] = flow_s.diff(scale)
        doi[scale] = oi_source.diff(scale)
        diss_p[scale] = -dp[scale].pow(3) / float(scale)
        diss_s[scale] = -ds[scale].pow(3) / float(scale)
        yaglom_p[scale] = dp[scale] * dflow_p[scale].pow(2) / float(scale)
        yaglom_s[scale] = ds[scale] * dflow_s[scale].pow(2) / float(scale)
        yaglom_oi[scale] = dp[scale] * doi[scale].pow(2) / float(scale)
        shock_p[scale] = (-dp[scale]).clip(lower=0.0).pow(3) / float(scale)
        shock_s[scale] = (-ds[scale]).clip(lower=0.0).pow(3) / float(scale)
        rare_p[scale] = dp[scale].clip(lower=0.0).pow(3) / float(scale)
        rare_s[scale] = ds[scale].clip(lower=0.0).pow(3) / float(scale)

    raw: dict[str, pd.Series] = {}
    joint_diss = robust_median([diss_p[l] + diss_s[l] for l in SCALES])
    diss_gap = robust_median([diss_p[l] - diss_s[l] for l in SCALES])
    forward = robust_median([
        pd.concat([diss_p[l].clip(lower=0.0), diss_s[l].clip(lower=0.0)], axis=1).min(axis=1)
        for l in SCALES
    ])
    inverse = robust_median([
        pd.concat([(-diss_p[l]).clip(lower=0.0), (-diss_s[l]).clip(lower=0.0)], axis=1).min(axis=1)
        for l in SCALES
    ])
    raw["joint_dissipation_anomaly"] = joint_diss
    raw["interlayer_dissipation_gap"] = diss_gap
    raw["coherent_forward_dissipation"] = forward
    raw["coherent_inverse_backscatter"] = inverse
    jd_z = base.causal_z(joint_diss)
    raw["dissipation_release"] = -jd_z * jd_z.diff()
    raw["dissipation_basis_work"] = diss_gap * basis

    directions = pd.concat(
        [pd.Series((diss_p[l] > 0).astype(float), index=g.index) for l in SCALES]
        + [pd.Series((diss_s[l] > 0).astype(float), index=g.index) for l in SCALES],
        axis=1,
    )
    forward_fraction = directions.mean(axis=1)
    magnitude = robust_median([diss_p[l].abs() + diss_s[l].abs() for l in SCALES])
    mismatch_fraction = pd.concat(
        [pd.Series((np.sign(diss_p[l]) != np.sign(diss_s[l])).astype(float), index=g.index) for l in SCALES],
        axis=1,
    ).mean(axis=1)
    direction_state = 2.0 * forward_fraction - 1.0
    raw["forward_scale_coherence"] = forward_fraction.pow(2) * magnitude
    raw["inverse_scale_coherence"] = (1.0 - forward_fraction).pow(2) * magnitude
    raw["layer_direction_mismatch"] = mismatch_fraction * robust_median([(diss_p[l] - diss_s[l]).abs() for l in SCALES])
    raw["cascade_reversal_impulse"] = direction_state.diff() * magnitude
    raw["cascade_direction_entropy"] = binary_entropy(forward_fraction) * magnitude
    raw["directional_persistence"] = direction_state * direction_state.shift(1) * magnitude

    y_p = robust_median([yaglom_p[l] for l in SCALES])
    y_s = robust_median([yaglom_s[l] for l in SCALES])
    y_gap = y_p - y_s
    raw["perp_orderflow_yaglom"] = y_p
    raw["spot_orderflow_yaglom"] = y_s
    raw["interlayer_yaglom_gap"] = y_gap
    raw["joint_yaglom_transport"] = np.sign(y_p + y_s) * (y_p.abs() + y_s.abs())
    raw["oi_yaglom_source"] = robust_median([yaglom_oi[l] for l in SCALES])
    raw["basis_yaglom_coupling"] = y_gap * basis

    window = 168
    m2_p = ewm(dp[24].pow(2), window, 72)
    m4_p = ewm(dp[24].pow(4), window, 72)
    m2_s = ewm(ds[24].pow(2), window, 72)
    m4_s = ewm(ds[24].pow(4), window, 72)
    flat_p = m4_p / (m2_p.pow(2) + EPS)
    flat_s = m4_s / (m2_s.pow(2) + EPS)
    flat_gap = flat_p - flat_s
    s2_p = {l: ewm(dp[l].abs().pow(2), window, 72) for l in SCALES}
    s3_p = {l: ewm(dp[l].abs().pow(3), window, 72) for l in SCALES}
    s4_p = {l: ewm(dp[l].abs().pow(4), window, 72) for l in SCALES}
    s2_s = {l: ewm(ds[l].abs().pow(2), window, 72) for l in SCALES}
    s3_s = {l: ewm(ds[l].abs().pow(3), window, 72) for l in SCALES}
    s4_s = {l: ewm(ds[l].abs().pow(4), window, 72) for l in SCALES}
    z2_p, z3_p, z4_p = slope_three(s2_p), slope_three(s3_p), slope_three(s4_p)
    z2_s, z3_s, z4_s = slope_three(s2_s), slope_three(s3_s), slope_three(s4_s)
    abs_diss_p = pd.concat([diss_p[l].abs() for l in SCALES], axis=1)
    abs_diss_s = pd.concat([diss_s[l].abs() for l in SCALES], axis=1)
    localization_p = abs_diss_p.max(axis=1) / (abs_diss_p.sum(axis=1) + EPS)
    localization_s = abs_diss_s.max(axis=1) / (abs_diss_s.sum(axis=1) + EPS)
    raw["interlayer_flatness_gap"] = flat_gap
    raw["joint_flatness_state"] = np.sqrt(flat_p.clip(lower=0.0) * flat_s.clip(lower=0.0))
    raw["zeta3_anomaly_gap"] = (z3_p - 1.0) - (z3_s - 1.0)
    raw["multifractal_width_gap"] = (z4_p - 2.0 * z2_p) - (z4_s - 2.0 * z2_s)
    raw["burst_localization_gap"] = localization_p - localization_s
    flat_z = base.causal_z(flat_gap)
    raw["intermittency_release"] = -flat_z * flat_z.diff()

    flux_p = robust_median([diss_p[l] for l in SCALES])
    flux_s = robust_median([diss_s[l] for l in SCALES])
    raw["cross_layer_flux_commutator"] = flux_p.shift(1) * flux_s - flux_s.shift(1) * flux_p
    raw["perp_to_spot_transfer_efficiency"] = flux_p.shift(1) * flux_s / (flux_p.shift(1).abs() + EPS)
    raw["spot_to_perp_transfer_efficiency"] = flux_s.shift(1) * flux_p / (flux_s.shift(1).abs() + EPS)
    raw["basis_mediated_flux_transfer"] = flux_p.shift(1) * basis.shift(1) * flux_s
    raw["source_mediated_flux_transfer"] = flux_p.shift(1) * oi_source.shift(1) * flux_s
    beta = base.causal_beta(flux_s, flux_p.shift(1), 336)
    raw["causal_flux_propagation_defect"] = flux_s - beta * flux_p.shift(1)

    shock_joint = robust_median([shock_p[l] + shock_s[l] for l in SCALES])
    shock_gap = robust_median([shock_p[l] - shock_s[l] for l in SCALES])
    raw["joint_shock_entropy"] = shock_joint
    raw["interlayer_shock_entropy_gap"] = shock_gap
    raw["joint_rarefaction_excess"] = robust_median([rare_p[l] + rare_s[l] for l in SCALES])
    shock_z = base.causal_z(shock_gap)
    raw["shock_entropy_release"] = -shock_z * shock_z.diff()
    raw["shock_basis_coupling"] = shock_gap * basis
    raw["shock_oi_coupling"] = shock_joint * oi_source

    raw_to_family = {name: family for family, names in FAMILY_RAW.items() for name in names}
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for name, values in raw.items():
        z = base.causal_z(values)
        g[f"raw_{name}"] = values
        for span in STATE_SPANS:
            column = f"{name}_ewm{span}"
            g[column] = ewm(z, span)
            candidate_map[raw_to_family[name]].append(column)
    return g, candidate_map


def years_positive(payload: str) -> bool:
    values = json.loads(payload)
    finite = [float(value) for value in values.values() if np.isfinite(float(value))]
    return bool(finite) and min(finite) > 0.0


def gate(row: pd.Series, holm_p: float | None = None) -> bool:
    passed = bool(
        row["cs_rankic"] > 0.012
        and row["bootstrap_p_nonpositive"] <= 0.10
        and row["ridge_mse_gain"] > 0.0
        and row["residual_cs_rankic"] > 0.005
        and row["max_parent_corr"] < 0.80
        and row["annual_log_return_5bp"] > 0.0
        and years_positive(row["yearly_rankic"])
    )
    return passed and (holm_p is None or holm_p <= 0.10)


def build_panel(symbols: list[str], out: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    base.SYMBOLS = symbols
    base.MONTHS = list(pd.period_range("2021-01", "2026-09", freq="M"))
    base.OUT = out
    raw, audit = base.load_public_data()
    parts: list[pd.DataFrame] = []
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for symbol, group in raw.groupby("symbol", sort=False):
        local = base.local_features(group)
        enriched, local_map = add_round17_mechanisms(local)
        parts.append(enriched)
        candidate_map = local_map
        print(f"round17_mechanisms={symbol} rows={len(enriched)}", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    return panel, audit, candidate_map


def select_frozen(panel: pd.DataFrame, candidate_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scan_rows: list[dict[str, Any]] = []
    for family, factors in candidate_map.items():
        for factor in factors:
            for horizon in HORIZONS:
                plus = base.development_score(panel, factor, horizon, 1)
                minus = base.development_score(panel, factor, horizon, -1)
                best = plus if plus["development_score"] >= minus["development_score"] else minus
                best["family"] = family
                scan_rows.append(best)
    scan = pd.DataFrame(scan_rows)
    dev = panel.loc[panel["time"].dt.year <= 2022].copy()
    selected: list[pd.Series] = []
    for family, group in scan.groupby("family", sort=False):
        ordered = group.sort_values("development_score", ascending=False)
        family_selected: list[pd.Series] = []
        for _, row in ordered.iterrows():
            if row["development_cs"] < 0.006 or row["development_min_year"] < -0.015:
                continue
            signed = int(row["sign"]) * pd.to_numeric(dev[str(row["factor"])], errors="coerce")
            independent = True
            for chosen in family_selected:
                other = int(chosen["sign"]) * pd.to_numeric(dev[str(chosen["factor"])], errors="coerce")
                corr = abs(base.safe_spearman(signed, other, minimum=1000))
                if np.isfinite(corr) and corr >= 0.85:
                    independent = False
                    break
            if independent:
                family_selected.append(row)
            if len(family_selected) >= 2:
                break
        if not family_selected:
            family_selected.append(ordered.iloc[0])
        selected.extend(family_selected)
    selection = pd.DataFrame(selected).reset_index(drop=True)
    selection["oriented_factor"] = selection["factor"].map(lambda value: f"oriented_{value}")
    for row in selection.itertuples(index=False):
        panel[str(row.oriented_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")
    return scan, selection


def validate_stage(panel: pd.DataFrame, selection: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        result = base.evaluate(panel, str(row.oriented_factor), int(row.horizon), 2022, 2023, 2024)
        result.update({
            "family": str(row.family),
            "primitive_factor": str(row.factor),
            "sign": int(row.sign),
            "development_score": float(row.development_score),
        })
        validation_rows.append(result)
    validation = pd.DataFrame(validation_rows)
    validation["holm_p"] = base.holm_adjust([
        float(value) if np.isfinite(value) else 1.0 for value in validation["bootstrap_p_nonpositive"]
    ])
    validation["validation_gate"] = [gate(row, float(row["holm_p"])) for _, row in validation.iterrows()]
    validation.to_csv(OUT / f"{prefix}_VALIDATION_RESULTS.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    for _, row in validation.loc[validation["validation_gate"]].iterrows():
        result = base.evaluate(panel, str(row["factor"]), int(row["horizon"]), 2024, 2025, 2026)
        result.update({
            "family": row["family"],
            "primitive_factor": row["primitive_factor"],
            "sign": int(row["sign"]),
            "validation_holm_p": float(row["holm_p"]),
        })
        holdout_rows.append(result)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=validation.columns).to_csv(OUT / f"{prefix}_HOLDOUT_RESULTS.csv", index=False)
    else:
        holdout.to_csv(OUT / f"{prefix}_HOLDOUT_RESULTS.csv", index=False)

    passed: list[dict[str, Any]] = []
    for _, row in holdout.iterrows():
        if gate(row):
            passed.append({
                "family": row["family"],
                "primitive_factor": row["primitive_factor"],
                "oriented_factor": row["factor"],
                "sign": int(row["sign"]),
                "horizon": int(row["horizon"]),
                "validation_holm_p": float(row["validation_holm_p"]),
                "holdout_cs_rankic": float(row["cs_rankic"]),
                "holdout_ts_ic": float(row["ts_ic"]),
                "holdout_ridge_mse_gain": float(row["ridge_mse_gain"]),
                "holdout_residual_cs_rankic": float(row["residual_cs_rankic"]),
                "holdout_max_parent_corr": float(row["max_parent_corr"]),
                "holdout_annual_log_return_5bp": float(row["annual_log_return_5bp"]),
                "holdout_sharpe_5bp": float(row["sharpe_5bp"]),
            })
    return validation, holdout, passed


def apply_frozen(panel: pd.DataFrame, selection: pd.DataFrame) -> None:
    for row in selection.itertuples(index=False):
        panel[str(row.oriented_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")


def main() -> None:
    np.random.seed(SEED)
    base.moving_block_bootstrap = inference.nonoverlap_issue_block_bootstrap

    discovery_dir = OUT / "discovery_cache"
    discovery_panel, discovery_audit, candidate_map = build_panel(DISCOVERY_SYMBOLS, discovery_dir)
    scan, selection = select_frozen(discovery_panel, candidate_map)
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
        replication_dir = OUT / "replication_cache"
        replication_panel, replication_audit, _ = build_panel(REPLICATION_SYMBOLS, replication_dir)
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
        "inference": "circular moving-block bootstrap on the already non-overlapping issue grid, 4 issue observations, 10000 repetitions",
        "selection_rule": "up to two development candidates per family subject to CS>=0.006, minimum yearly CS>=-0.015 and within-family |rho|<0.85; otherwise family top one; formula, sign, span and horizon frozen before validation",
        "gate": "CS>0.012, block p<=0.10, Holm<=0.10 in validation, Ridge MSE gain>0, residual CS>0.005, max parent corr<0.80, 5bp annual return>0 and every stage year positive",
        "replication_policy": "replication universe is not downloaded unless a discovery candidate passes both discovery validation and discovery historical holdout",
    }
    write_json(OUT / "DATA_AUDIT.json", audit)
    write_json(OUT / "CANDIDATE_CONTRACTS.json", CONTRACTS)
    summary = {
        "raw_primitives": int(sum(len(values) for values in FAMILY_RAW.values())),
        "candidate_columns": int(sum(len(values) for values in candidate_map.values())),
        "development_tests": int(len(scan)),
        "frozen_identities": int(len(selection)),
        "discovery_validation_passed": int(discovery_validation["validation_gate"].sum()),
        "discovery_holdout_opened": int(len(discovery_holdout)),
        "discovery_promoted": promoted,
        "replication_validation_passed": int(replication_validation.get("validation_gate", pd.Series(dtype=bool)).sum()) if not replication_validation.empty else 0,
        "replication_holdout_opened": int(len(replication_holdout)),
        "confirmed": confirmed,
        "decision": "eligible_for_trade_only_if_confirmed_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = [
        "# Fluid cascade direction, dissipation anomaly and intermittency round17",
        "",
        "This fresh-universe experiment freezes physically derived Duchon-Robert dissipation, cascade-direction, mixed Yaglom, intermittency, causal-precedence and Burgers entropy states before reading validation or historical holdout data.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen selection",
        selection.to_markdown(index=False),
        "",
        "## Discovery validation",
        discovery_validation.to_markdown(index=False),
        "",
        "## Discovery historical holdout",
        discovery_holdout.to_markdown(index=False) if not discovery_holdout.empty else "Discovery holdout remained closed.",
        "",
        "## Replication validation",
        replication_validation.to_markdown(index=False) if not replication_validation.empty else "Replication remained closed.",
        "",
        "## Replication historical holdout",
        replication_holdout.to_markdown(index=False) if not replication_holdout.empty else "Replication holdout remained closed.",
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
