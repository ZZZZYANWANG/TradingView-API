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
ROUND15 = HERE.parent / "fluid_round15"
if str(ROUND15) not in sys.path:
    sys.path.insert(0, str(ROUND15))

import run_spot_perp_characteristics_v4 as fast  # noqa: E402

base = fast.base
SEED = 20261119
EPS = base.EPS
OUT = Path("research_outputs/fluid_multiscale_constitutive_round16")
OUT.mkdir(parents=True, exist_ok=True)

SCALES = [6, 24, 72]
STATE_SPANS = [6, 24, 72]
CONSTITUTIVE_WINDOWS = [168, 336, 720]

FAMILY_RAW: dict[str, list[str]] = {
    "onsager_cascade": [
        "interlayer_subgrid_flux_gap",
        "joint_forward_cascade",
        "interlayer_backscatter_gap",
        "cascade_dissipation_release",
        "cascade_basis_coupling",
    ],
    "favre_momentum_budget": [
        "favre_momentum_closure_defect",
        "source_compensated_momentum_defect",
        "favre_stress_gap",
        "favre_stress_release",
        "favre_source_absorption",
    ],
    "nonnewtonian_constitutive": [
        "constitutive_closure_residual",
        "constitutive_residual_release",
        "effective_viscosity_break",
        "shear_thickening_excess",
        "constitutive_hysteresis_work",
        "constitutive_relaxation_load",
    ],
    "kelvin_circulation": [
        "kelvin_circulation_defect",
        "kelvin_defect_release",
        "baroclinic_circulation_injection",
        "source_circulation_alignment",
        "circulation_restoration",
    ],
    "scale_locality": [
        "third_order_flux_gap",
        "joint_third_order_cascade",
        "cascade_scaling_slope_gap",
        "scale_locality_break",
        "basis_scale_flux_coupling",
    ],
}

CONTRACTS: dict[str, dict[str, str]] = {
    "interlayer_subgrid_flux_gap": {
        "equation": "median_l[-tau_perp(l)*d_t ubar_perp(l) + tau_spot(l)*d_t ubar_spot(l)]",
        "mechanism": "difference in coarse-grained subgrid kinetic-energy flux between perpetual and spot layers",
    },
    "joint_forward_cascade": {
        "equation": "median_l[positive(Pi_perp(l))+positive(Pi_spot(l))]",
        "mechanism": "joint forward cascade or dissipation load across the two coupled layers",
    },
    "interlayer_backscatter_gap": {
        "equation": "median_l[positive(-Pi_perp(l))-positive(-Pi_spot(l))]",
        "mechanism": "difference in negative-viscosity backscatter between derivative and cash layers",
    },
    "cascade_dissipation_release": {
        "equation": "-Z(Pi_gap)*d_t Z(Pi_gap)",
        "mechanism": "release of an accumulated cross-layer cascade imbalance",
    },
    "cascade_basis_coupling": {
        "equation": "median_l[Pi_gap(l)*basis_z]",
        "mechanism": "coupling of scale-to-scale energy transport with the spot-perpetual pressure boundary",
    },
    "favre_momentum_closure_defect": {
        "equation": "d_t[(rho u~)_p-(rho u~)_s] + div_layer[rho(u~^2+tau_F)+p]",
        "mechanism": "density-weighted two-layer momentum-budget defect",
    },
    "source_compensated_momentum_defect": {
        "equation": "Favre momentum defect - (OI source + funding source + participation source)",
        "mechanism": "momentum residual after observable market source terms are removed",
    },
    "favre_stress_gap": {
        "equation": "median_l[tau_F,perp(l)-tau_F,spot(l)]",
        "mechanism": "cross-layer difference in unresolved density-weighted Reynolds stress",
    },
    "favre_stress_release": {
        "equation": "-Z(tau_F gap)*d_t Z(tau_F gap)",
        "mechanism": "relaxation of unresolved cross-layer stress",
    },
    "favre_source_absorption": {
        "equation": "-source_compensated_defect * observable_source",
        "mechanism": "whether position and funding sources are absorbed rather than transmitted into price momentum",
    },
    "constitutive_closure_residual": {
        "equation": "sigma-[eta1*gamma_dot+eta3*gamma_dot^3+lambda*d_t gamma_dot], coefficients estimated from strict-prior windows",
        "mechanism": "innovation outside a causal nonlinear viscoelastic constitutive law",
    },
    "constitutive_residual_release": {
        "equation": "-Z(residual)*d_t Z(residual)",
        "mechanism": "release of stored constitutive disequilibrium",
    },
    "effective_viscosity_break": {
        "equation": "d_t[eta1+eta3*gamma_dot^2]*gamma_dot^2",
        "mechanism": "change in effective market viscosity conditional on current shear",
    },
    "shear_thickening_excess": {
        "equation": "eta3*gamma_dot^3",
        "mechanism": "nonlinear shear-thickening or shear-thinning component of the spot-perpetual response",
    },
    "constitutive_hysteresis_work": {
        "equation": "residual_t*gamma_dot_{t-1}-residual_{t-1}*gamma_dot_t",
        "mechanism": "oriented hysteresis-loop area in stress-shear phase space",
    },
    "constitutive_relaxation_load": {
        "equation": "lambda*d_t gamma_dot",
        "mechanism": "estimated viscoelastic memory load",
    },
    "kelvin_circulation_defect": {
        "equation": "d_t(u_perp-u_spot)-[J_t,layer(p,log rho)+observable_source]",
        "mechanism": "Kelvin circulation production unexplained by the discrete baroclinic Jacobian and sources",
    },
    "kelvin_defect_release": {
        "equation": "-Z(circulation defect)*d_t Z(circulation defect)",
        "mechanism": "relaxation of circulation-theorem violation",
    },
    "baroclinic_circulation_injection": {
        "equation": "J_t,layer(p,log rho)*(u_perp-u_spot)",
        "mechanism": "baroclinic generation aligned with existing interlayer circulation",
    },
    "source_circulation_alignment": {
        "equation": "observable_source*(u_perp-u_spot)",
        "mechanism": "position and funding sources reinforcing or opposing circulation",
    },
    "circulation_restoration": {
        "equation": "-circulation*circulation_defect",
        "mechanism": "restoring versus destabilizing circulation force",
    },
    "third_order_flux_gap": {
        "equation": "median_l[(Delta_l u_perp)^3/l-(Delta_l u_spot)^3/l]",
        "mechanism": "two-layer difference in signed third-order cascade flux",
    },
    "joint_third_order_cascade": {
        "equation": "median_l[sign(Delta_l u_perp+Delta_l u_spot)*(|flux_perp|+|flux_spot|)]",
        "mechanism": "directional joint cascade across multiple horizons",
    },
    "cascade_scaling_slope_gap": {
        "equation": "slope_log_l |third-order flux_perp| - slope_log_l |third-order flux_spot|",
        "mechanism": "difference in scale-local cascade exponents",
    },
    "scale_locality_break": {
        "equation": "|flux_24|-sqrt(|flux_6*flux_72|), layer difference",
        "mechanism": "departure from multiplicative scale locality at the intermediate scale",
    },
    "basis_scale_flux_coupling": {
        "equation": "third_order_flux_gap*basis_z",
        "mechanism": "interaction between multiscale cascade and the interlayer pressure boundary",
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def robust_median(series: list[pd.Series]) -> pd.Series:
    matrix = np.column_stack([pd.to_numeric(item, errors="coerce").to_numpy(float) for item in series])
    with np.errstate(all="ignore"):
        values = np.nanmedian(matrix, axis=1)
    values[np.all(~np.isfinite(matrix), axis=1)] = np.nan
    return pd.Series(values, index=series[0].index, dtype=float)


def ewm(values: pd.Series, span: int, minimum: int | None = None) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").ewm(
        span=span,
        adjust=False,
        min_periods=minimum if minimum is not None else span,
    ).mean()


def slope_three(y6: pd.Series, y24: pd.Series, y72: pd.Series) -> pd.Series:
    x = np.log(np.asarray([6.0, 24.0, 72.0]))
    xc = x - x.mean()
    denom = float(np.sum(xc**2))
    matrix = np.column_stack([
        np.log(np.abs(pd.to_numeric(y6, errors="coerce").to_numpy(float)) + EPS),
        np.log(np.abs(pd.to_numeric(y24, errors="coerce").to_numpy(float)) + EPS),
        np.log(np.abs(pd.to_numeric(y72, errors="coerce").to_numpy(float)) + EPS),
    ])
    valid = np.isfinite(matrix).all(axis=1)
    result = np.full(len(matrix), np.nan)
    result[valid] = np.sum((matrix[valid] - matrix[valid].mean(axis=1, keepdims=True)) * xc, axis=1) / denom
    return pd.Series(result, index=y6.index, dtype=float)


def add_mechanisms(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    g = group.sort_values("time").copy()
    u_p = pd.to_numeric(g["perp_u"], errors="coerce")
    u_s = pd.to_numeric(g["spot_u"], errors="coerce")
    shear = u_p - u_s
    basis = pd.to_numeric(g["basis_z"], errors="coerce")
    p_p = basis + 0.35 * pd.to_numeric(g["range_perp_z"], errors="coerce") + 0.25 * pd.to_numeric(g["funding_z"], errors="coerce")
    p_s = -basis + 0.35 * pd.to_numeric(g["range_spot_z"], errors="coerce")
    rho_p = np.exp(np.clip((0.60 * g["perp_quote_z"] + 0.40 * g["oi_level_z"]) / 4.0, -3.0, 3.0))
    rho_s = np.exp(np.clip(g["spot_quote_z"] / 4.0, -3.0, 3.0))
    source = (
        pd.to_numeric(g["oi_source_z"], errors="coerce").fillna(0.0)
        + 0.50 * pd.to_numeric(g["funding_change_z"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(g["quote_share_z"], errors="coerce").diff().fillna(0.0)
    )

    cascade_gap_parts: list[pd.Series] = []
    forward_parts: list[pd.Series] = []
    backscatter_parts: list[pd.Series] = []
    coupling_parts: list[pd.Series] = []
    favre_defect_parts: list[pd.Series] = []
    favre_comp_parts: list[pd.Series] = []
    favre_stress_parts: list[pd.Series] = []

    third_p: dict[int, pd.Series] = {}
    third_s: dict[int, pd.Series] = {}

    for scale in SCALES:
        up_bar = ewm(u_p, scale)
        us_bar = ewm(u_s, scale)
        tau_p = ewm(u_p.pow(2), scale) - up_bar.pow(2)
        tau_s = ewm(u_s.pow(2), scale) - us_bar.pow(2)
        pi_p = -tau_p * up_bar.diff()
        pi_s = -tau_s * us_bar.diff()
        gap = pi_p - pi_s
        cascade_gap_parts.append(gap)
        forward_parts.append(pi_p.clip(lower=0.0) + pi_s.clip(lower=0.0))
        backscatter_parts.append((-pi_p).clip(lower=0.0) - (-pi_s).clip(lower=0.0))
        coupling_parts.append(gap * basis)

        mass_p = ewm(rho_p, scale)
        mass_s = ewm(rho_s, scale)
        favre_p = ewm(rho_p * u_p, scale) / (mass_p + EPS)
        favre_s = ewm(rho_s * u_s, scale) / (mass_s + EPS)
        stress_p = ewm(rho_p * u_p.pow(2), scale) / (mass_p + EPS) - favre_p.pow(2)
        stress_s = ewm(rho_s * u_s.pow(2), scale) / (mass_s + EPS) - favre_s.pow(2)
        momentum_gap = mass_p * favre_p - mass_s * favre_s
        layer_flux = mass_p * (favre_p.pow(2) + stress_p) + p_p - mass_s * (favre_s.pow(2) + stress_s) - p_s
        defect = momentum_gap.diff() + layer_flux
        favre_defect_parts.append(defect)
        favre_comp_parts.append(defect - source)
        favre_stress_parts.append(stress_p - stress_s)

        dp = np.log(pd.to_numeric(g["close"], errors="coerce")).diff(scale) / math.sqrt(scale)
        ds = np.log(pd.to_numeric(g["close_spot"], errors="coerce")).diff(scale) / math.sqrt(scale)
        third_p[scale] = dp.pow(3) / float(scale)
        third_s[scale] = ds.pow(3) / float(scale)

    raw: dict[str, pd.Series] = {}
    raw["interlayer_subgrid_flux_gap"] = robust_median(cascade_gap_parts)
    raw["joint_forward_cascade"] = robust_median(forward_parts)
    raw["interlayer_backscatter_gap"] = robust_median(backscatter_parts)
    gap_z = base.causal_z(raw["interlayer_subgrid_flux_gap"])
    raw["cascade_dissipation_release"] = -gap_z * gap_z.diff()
    raw["cascade_basis_coupling"] = robust_median(coupling_parts)

    raw["favre_momentum_closure_defect"] = robust_median(favre_defect_parts)
    raw["source_compensated_momentum_defect"] = robust_median(favre_comp_parts)
    raw["favre_stress_gap"] = robust_median(favre_stress_parts)
    stress_z = base.causal_z(raw["favre_stress_gap"])
    raw["favre_stress_release"] = -stress_z * stress_z.diff()
    raw["favre_source_absorption"] = -raw["source_compensated_momentum_defect"] * source

    sigma = (
        pd.to_numeric(g["basis_change_z"], errors="coerce")
        + 0.35 * pd.to_numeric(g["funding_change_z"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(g["oi_source_z"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(g["quote_share_z"], errors="coerce").diff().fillna(0.0)
    )
    constitutive_residuals: list[pd.Series] = []
    viscosity_breaks: list[pd.Series] = []
    thickening_parts: list[pd.Series] = []
    hysteresis_parts: list[pd.Series] = []
    relaxation_parts: list[pd.Series] = []
    d_shear = shear.diff()
    for window in CONSTITUTIVE_WINDOWS:
        eta1 = base.causal_beta(sigma, shear, window)
        residual1 = sigma - eta1 * shear
        eta3 = base.causal_beta(residual1, shear.pow(3), window)
        residual2 = residual1 - eta3 * shear.pow(3)
        memory = base.causal_beta(residual2, d_shear, window)
        residual = residual2 - memory * d_shear
        effective_viscosity = eta1 + eta3 * shear.pow(2)
        constitutive_residuals.append(residual)
        viscosity_breaks.append(effective_viscosity.diff() * shear.pow(2))
        thickening_parts.append(eta3 * shear.pow(3))
        hysteresis_parts.append(residual * shear.shift(1) - residual.shift(1) * shear)
        relaxation_parts.append(memory * d_shear)
    raw["constitutive_closure_residual"] = robust_median(constitutive_residuals)
    constitutive_z = base.causal_z(raw["constitutive_closure_residual"])
    raw["constitutive_residual_release"] = -constitutive_z * constitutive_z.diff()
    raw["effective_viscosity_break"] = robust_median(viscosity_breaks)
    raw["shear_thickening_excess"] = robust_median(thickening_parts)
    raw["constitutive_hysteresis_work"] = robust_median(hysteresis_parts)
    raw["constitutive_relaxation_load"] = robust_median(relaxation_parts)

    layer_p = p_p - p_s
    layer_rho = np.log(rho_p + EPS) - np.log(rho_s + EPS)
    mean_p = 0.5 * (p_p + p_s)
    mean_rho = 0.5 * (np.log(rho_p + EPS) + np.log(rho_s + EPS))
    baroclinic = mean_p.diff() * layer_rho - layer_p * mean_rho.diff()
    circulation_forcing = baroclinic + source
    circulation_defect = shear.diff() - circulation_forcing
    raw["kelvin_circulation_defect"] = circulation_defect
    circulation_z = base.causal_z(circulation_defect)
    raw["kelvin_defect_release"] = -circulation_z * circulation_z.diff()
    raw["baroclinic_circulation_injection"] = baroclinic * shear
    raw["source_circulation_alignment"] = source * shear
    raw["circulation_restoration"] = -shear * circulation_defect

    third_gap = {scale: third_p[scale] - third_s[scale] for scale in SCALES}
    joint_third = {
        scale: np.sign(third_p[scale] + third_s[scale]) * (third_p[scale].abs() + third_s[scale].abs())
        for scale in SCALES
    }
    raw["third_order_flux_gap"] = robust_median([third_gap[scale] for scale in SCALES])
    raw["joint_third_order_cascade"] = robust_median([joint_third[scale] for scale in SCALES])
    raw["cascade_scaling_slope_gap"] = slope_three(third_p[6], third_p[24], third_p[72]) - slope_three(third_s[6], third_s[24], third_s[72])
    locality_p = third_p[24].abs() - np.sqrt(third_p[6].abs() * third_p[72].abs())
    locality_s = third_s[24].abs() - np.sqrt(third_s[6].abs() * third_s[72].abs())
    raw["scale_locality_break"] = locality_p - locality_s
    raw["basis_scale_flux_coupling"] = raw["third_order_flux_gap"] * basis

    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    raw_to_family = {name: family for family, names in FAMILY_RAW.items() for name in names}
    for name, values in raw.items():
        z = base.causal_z(values)
        g[f"raw_{name}"] = values
        for span in STATE_SPANS:
            column = f"{name}_ewm{span}"
            g[column] = ewm(z, span)
            candidate_map[raw_to_family[name]].append(column)
    return g, candidate_map


def signed_years_positive(payload: str) -> bool:
    values = json.loads(payload)
    finite = [float(value) for value in values.values() if np.isfinite(float(value))]
    return bool(finite) and min(finite) > 0.0


def validation_gate(row: pd.Series, holm_p: float) -> bool:
    return bool(
        row["cs_rankic"] > 0.012
        and row["bootstrap_p_nonpositive"] <= 0.10
        and row["ridge_mse_gain"] > 0.0
        and row["residual_cs_rankic"] > 0.005
        and row["max_parent_corr"] < 0.80
        and row["annual_log_return_5bp"] > 0.0
        and holm_p <= 0.10
        and signed_years_positive(row["yearly_rankic"])
    )


def main() -> None:
    np.random.seed(SEED)
    base.OUT = OUT
    raw, data_audit = base.load_public_data()
    parts: list[pd.DataFrame] = []
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for symbol, group in raw.groupby("symbol", sort=False):
        local = base.local_features(group)
        enriched, local_map = add_mechanisms(local)
        parts.append(enriched)
        for family, columns in local_map.items():
            candidate_map[family] = columns
        print(f"mechanisms={symbol} rows={len(enriched)}", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)

    scan_rows: list[dict[str, Any]] = []
    for family, factors in candidate_map.items():
        for factor in factors:
            for horizon in base.HORIZONS:
                plus = base.development_score(panel, factor, horizon, 1)
                minus = base.development_score(panel, factor, horizon, -1)
                best = plus if plus["development_score"] >= minus["development_score"] else minus
                best["family"] = family
                scan_rows.append(best)
    scan = pd.DataFrame(scan_rows)
    scan.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)

    selected_rows: list[pd.Series] = []
    dev = panel.loc[panel["time"].dt.year <= 2022].copy()
    for family, group in scan.groupby("family", sort=False):
        ordered = group.sort_values("development_score", ascending=False)
        family_selected: list[pd.Series] = []
        for _, row in ordered.iterrows():
            if row["development_cs"] < 0.006 or row["development_min_year"] < -0.015:
                continue
            signed = int(row["sign"]) * pd.to_numeric(dev[row["factor"]], errors="coerce")
            independent = True
            for chosen in family_selected:
                other = int(chosen["sign"]) * pd.to_numeric(dev[chosen["factor"]], errors="coerce")
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
        selected_rows.extend(family_selected)

    selection = pd.DataFrame(selected_rows).reset_index(drop=True)
    for i, row in selection.iterrows():
        signed_name = f"frozen_signed_{i}"
        panel[signed_name] = int(row["sign"]) * panel[str(row["factor"])]
        selection.loc[i, "signed_factor"] = signed_name

    best_by_family = selection.sort_values("development_score", ascending=False).groupby("family", as_index=False).head(1)
    family_signed = {str(row.family): str(row.signed_factor) for row in best_by_family.itertuples(index=False)}
    composite_specs = {
        "conservation_closure_consensus": [name for name in ["favre_momentum_budget", "kelvin_circulation"] if name in family_signed],
        "cascade_constitutive_consensus": [name for name in ["onsager_cascade", "nonnewtonian_constitutive", "scale_locality"] if name in family_signed],
        "round16_all_family_consensus": list(family_signed),
    }
    composite_rows: list[dict[str, Any]] = []
    for name, families in composite_specs.items():
        members = [family_signed[family] for family in families]
        ranks = [panel.groupby("time")[member].rank(pct=True) for member in members]
        panel[name] = pd.concat(ranks, axis=1).mean(axis=1)
        choices = [base.development_score(panel, name, horizon, sign) for horizon in base.HORIZONS for sign in [1, -1]]
        best = max(choices, key=lambda item: item["development_score"])
        best["family"] = name
        best["signed_factor"] = f"frozen_composite_{len(composite_rows)}"
        panel[best["signed_factor"]] = int(best["sign"]) * panel[name]
        composite_rows.append(best)
    selection = pd.concat([selection, pd.DataFrame(composite_rows)], ignore_index=True)
    selection.to_csv(OUT / "FROZEN_SELECTION.csv", index=False)

    validation_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        result = base.evaluate(panel, str(row.signed_factor), int(row.horizon), 2022, 2023, 2024)
        result.update({
            "family": str(row.family),
            "primitive_factor": str(row.factor),
            "sign": int(row.sign),
            "development_score": float(row.development_score),
        })
        validation_rows.append(result)
        print(f"validation={row.family} factor={row.factor} horizon={row.horizon}", flush=True)
    validation = pd.DataFrame(validation_rows)
    validation["holm_p"] = base.holm_adjust([
        float(value) if np.isfinite(value) else 1.0 for value in validation["bootstrap_p_nonpositive"]
    ])
    validation["validation_gate"] = [
        validation_gate(row, float(row["holm_p"])) for _, row in validation.iterrows()
    ]
    validation.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)

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
        pd.DataFrame(columns=validation.columns).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    else:
        holdout.to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)

    promoted: list[dict[str, Any]] = []
    for _, row in holdout.iterrows():
        passed = bool(
            row["cs_rankic"] > 0.012
            and row["bootstrap_p_nonpositive"] <= 0.10
            and row["ridge_mse_gain"] > 0.0
            and row["residual_cs_rankic"] > 0.005
            and row["max_parent_corr"] < 0.80
            and row["annual_log_return_5bp"] > 0.0
            and signed_years_positive(row["yearly_rankic"])
        )
        if passed:
            promoted.append({
                "family": row["family"],
                "primitive_factor": row["primitive_factor"],
                "signed_factor": row["factor"],
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

    audit = {
        "seed": SEED,
        "data": data_audit,
        "panel_rows": int(len(panel)),
        "families": {family: len(columns) for family, columns in candidate_map.items()},
        "raw_mechanisms": FAMILY_RAW,
        "candidate_contracts": CONTRACTS,
        "horizons_hours": base.HORIZONS,
        "development": "2021-2022",
        "validation": "2023-2024",
        "historical_holdout": "2025-2026-09",
        "selection_rule": "up to two development candidates per family subject to CS>=0.006, minimum yearly CS>=-0.015 and within-family |rho|<0.85; otherwise family top one; sign and horizon frozen before validation",
        "gate": "CS>0.012, block p<=0.10, Holm<=0.10, Ridge MSE gain>0, residual CS>0.005, max parent corr<0.80, 5bp annual return>0 and every stage year positive",
    }
    write_json(OUT / "DATA_AUDIT.json", audit)
    write_json(OUT / "CANDIDATE_CONTRACTS.json", CONTRACTS)
    write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "raw_primitives": int(sum(len(values) for values in FAMILY_RAW.values())),
        "candidate_columns": int(sum(len(values) for values in candidate_map.values())),
        "development_tests": int(len(scan)),
        "frozen_identities": int(len(selection)),
        "validation_passed": int(validation["validation_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "promoted": promoted,
        "decision": "promote_only_after_validation_and_historical_holdout_both_pass",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = [
        "# Fluid multiscale conservation and constitutive research round16",
        "",
        "The experiment implements coarse-grained Onsager energy flux, Favre momentum closure, a causal nonlinear viscoelastic constitutive residual, Kelvin circulation production and third-order scale-locality diagnostics on coupled spot/perpetual hourly states.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen selection",
        selection.to_markdown(index=False),
        "",
        "## Validation",
        validation.to_markdown(index=False),
        "",
        "## Historical holdout",
        holdout.to_markdown(index=False) if not holdout.empty else "Holdout remained closed.",
        "",
        "## Promotion",
        "```json",
        json.dumps(promoted, indent=2, sort_keys=True),
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
