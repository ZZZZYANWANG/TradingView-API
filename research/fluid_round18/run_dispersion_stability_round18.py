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
ROUND17 = HERE.parent / "fluid_round17"
ROUND16 = HERE.parent / "fluid_round16"
ROUND15 = HERE.parent / "fluid_round15"
for path in (ROUND17, ROUND16, ROUND15):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_cascade_direction_round17 as r17  # noqa: E402
import run_multiscale_constitutive as r16  # noqa: E402
import run_multiscale_constitutive_inference_fix as inference  # noqa: E402

base = r17.base
SEED = 20261119
EPS = base.EPS
OUT = Path("research_outputs/fluid_dispersion_stability_round18")
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_SYMBOLS = [
    "KSMUSDT", "KAVAUSDT", "ZILUSDT", "IOTAUSDT", "QTUMUSDT", "ONTUSDT",
    "ICXUSDT", "ZRXUSDT", "BATUSDT", "ANKRUSDT", "LRCUSDT", "FTMUSDT",
]
REPLICATION_SYMBOLS = [
    "1INCHUSDT", "ENJUSDT", "SANDUSDT", "COTIUSDT", "CTSIUSDT", "STORJUSDT",
    "SKLUSDT", "ALICEUSDT", "OGNUSDT", "BANDUSDT", "CELOUSDT", "HBARUSDT",
]
HORIZONS = [1, 4, 24, 72, 120, 240, 480]
STATE_SPANS = [6, 24, 72]
CLOSURE_WINDOWS = [72, 168, 336]

FAMILY_RAW: dict[str, list[str]] = {
    "taylor_aris_dispersion": [
        "imbalance_taylor_dispersion", "basis_taylor_dispersion",
        "liquidity_taylor_dispersion", "taylor_flux_residual",
        "taylor_residual_release", "dispersion_capacity_ratio",
    ],
    "kelvin_helmholtz": [
        "richardson_criticality", "kh_growth_rate", "kh_unstable_energy",
        "kh_stability_restoration", "atwood_shear_instability", "kh_source_loading",
    ],
    "boundary_layer": [
        "adverse_pressure_separation", "skin_friction_collapse",
        "separation_bubble_energy", "reattachment_impulse",
        "reattachment_persistence", "basis_boundary_recovery",
    ],
    "potential_vorticity": [
        "two_layer_potential_vorticity", "pv_material_defect",
        "pv_source_residual", "pv_residual_release",
        "pv_basis_coupling", "pv_cascade_alignment",
    ],
    "vortex_enstrophy": [
        "enstrophy_production", "enstrophy_dissipation",
        "vortex_stretching_residual", "enstrophy_release",
        "strain_vorticity_alignment", "vortex_source_loading",
    ],
    "prandtl_mixing": [
        "prandtl_eddy_viscosity", "mixing_length_momentum_flux",
        "eddy_viscosity_closure_residual", "mixing_length_breakdown",
        "turbulent_schmidt_gap", "eddy_flux_release",
    ],
}

CONTRACTS: dict[str, dict[str, str]] = {
    "imbalance_taylor_dispersion": {
        "equation": "D_eff(c_imb)=D_m(c_imb)+tau(c_imb)*(u_perp-u_spot)^2",
        "mechanism": "Taylor-Aris enhancement of order-flow scalar dispersion by cross-layer price shear",
    },
    "basis_taylor_dispersion": {
        "equation": "D_eff(basis)=D_m(basis)+tau(basis)*(u_perp-u_spot)^2",
        "mechanism": "shear-enhanced dispersion of the spot-perpetual basis",
    },
    "liquidity_taylor_dispersion": {
        "equation": "D_eff(log rho_perp-log rho_spot)",
        "mechanism": "shear-enhanced dispersion of the liquidity-density contrast",
    },
    "taylor_flux_residual": {
        "equation": "median_c[Delta c+2*(-D_eff(c)*c)-observable_source_c]",
        "mechanism": "unexplained scalar transport after causal Taylor dispersion and observable sources",
    },
    "taylor_residual_release": {
        "equation": "-Z(R_Taylor)*Delta Z(R_Taylor)",
        "mechanism": "relaxation of accumulated shear-dispersion closure error",
    },
    "dispersion_capacity_ratio": {
        "equation": "median_c[(D_eff-D_m)/(D_m+epsilon)]",
        "mechanism": "fractional dispersion capacity supplied by interlayer shear rather than baseline diffusion",
    },
    "richardson_criticality": {
        "equation": "1/4-Ri_g, Ri_g=N^2/(shear^2+D_m+epsilon)",
        "mechanism": "distance to the Kelvin-Helmholtz critical Richardson threshold",
    },
    "kh_growth_rate": {
        "equation": "abs(shear)*sqrt(max(1-4*Ri_g,0))",
        "mechanism": "linearized growth rate of two-layer shear instability",
    },
    "kh_unstable_energy": {
        "equation": "gamma_KH*(u_perp^2+u_spot^2+basis^2)",
        "mechanism": "kinetic and boundary energy exposed to Kelvin-Helmholtz instability",
    },
    "kh_stability_restoration": {
        "equation": "-Z(1/4-Ri_g)*Delta Z(1/4-Ri_g)",
        "mechanism": "restoration versus further departure from the stability boundary",
    },
    "atwood_shear_instability": {
        "equation": "Atwood(rho_perp,rho_spot)*gamma_KH",
        "mechanism": "density-asymmetric shear instability",
    },
    "kh_source_loading": {
        "equation": "gamma_KH*(OI_source+funding_source+participation_source)",
        "mechanism": "position and funding sources loading an unstable shear layer",
    },
    "adverse_pressure_separation": {
        "equation": "delta_star*positive(-shear*Delta pressure)/(shear^2+D_m+epsilon)",
        "mechanism": "boundary-layer separation under an adverse basis-pressure gradient",
    },
    "skin_friction_collapse": {
        "equation": "-abs(shear)/sqrt(D_m+epsilon)",
        "mechanism": "collapse of normalized wall shear preceding separation",
    },
    "separation_bubble_energy": {
        "equation": "beta_sep*(basis^2+shear^2)",
        "mechanism": "energy stored in a separated spot-perpetual boundary layer",
    },
    "reattachment_impulse": {
        "equation": "positive(-Delta beta_sep)*positive(Delta abs(shear))",
        "mechanism": "simultaneous collapse of the separation bubble and recovery of wall shear",
    },
    "reattachment_persistence": {
        "equation": "EWM[1(beta_sep decreases and abs(shear) increases)]",
        "mechanism": "persistence of a reattachment episode",
    },
    "basis_boundary_recovery": {
        "equation": "-basis*Delta pressure/(1+beta_sep)",
        "mechanism": "basis recovery conditional on boundary-layer attachment",
    },
    "two_layer_potential_vorticity": {
        "equation": "(u_perp-u_spot)*(log rho_perp-log rho_spot)",
        "mechanism": "two-layer potential-vorticity analogue coupling circulation and stratification",
    },
    "pv_material_defect": {
        "equation": "Delta q+u_bar*(q-EWM_24(q))",
        "mechanism": "material derivative of potential vorticity outside local advection",
    },
    "pv_source_residual": {
        "equation": "Dq/Dt-[baroclinic torque+observable_source*shear]",
        "mechanism": "potential-vorticity production not explained by baroclinicity or position sources",
    },
    "pv_residual_release": {
        "equation": "-Z(R_PV)*Delta Z(R_PV)",
        "mechanism": "relaxation of potential-vorticity nonconservation",
    },
    "pv_basis_coupling": {
        "equation": "q*basis_z",
        "mechanism": "potential vorticity interacting with the interlayer pressure boundary",
    },
    "pv_cascade_alignment": {
        "equation": "q*median_l[-(delta_l u_perp^3+delta_l u_spot^3)/l]",
        "mechanism": "alignment of rotational state with the signed energy cascade",
    },
    "enstrophy_production": {
        "equation": "omega^2*strain, omega=u_perp-u_spot",
        "mechanism": "vortex stretching production of cross-layer enstrophy",
    },
    "enstrophy_dissipation": {
        "equation": "nu_t*(Delta omega)^2",
        "mechanism": "mixing-length dissipation of cross-layer enstrophy",
    },
    "vortex_stretching_residual": {
        "equation": "Delta(omega^2/2)-omega^2*strain+nu_t*(Delta omega)^2-source*omega",
        "mechanism": "unclosed two-layer enstrophy budget",
    },
    "enstrophy_release": {
        "equation": "-Z(R_enstrophy)*Delta Z(R_enstrophy)",
        "mechanism": "release of accumulated enstrophy-budget disequilibrium",
    },
    "strain_vorticity_alignment": {
        "equation": "omega*strain",
        "mechanism": "alignment of interlayer vorticity with common-mode strain",
    },
    "vortex_source_loading": {
        "equation": "omega*(OI_source+funding_source+participation_source)",
        "mechanism": "observable sources injecting or removing circulation",
    },
    "prandtl_eddy_viscosity": {
        "equation": "nu_t=l_m^2*abs(shear), l_m=1/sqrt(rho_perp+rho_spot)",
        "mechanism": "Prandtl mixing-length estimate of turbulent market viscosity",
    },
    "mixing_length_momentum_flux": {
        "equation": "-nu_t*shear",
        "mechanism": "turbulent momentum flux across spot and perpetual layers",
    },
    "eddy_viscosity_closure_residual": {
        "equation": "(u_spot-u_perp)-(-nu_t*shear)",
        "mechanism": "observed interlayer momentum transfer outside the mixing-length closure",
    },
    "mixing_length_breakdown": {
        "equation": "closure_residual/(abs(observed_transfer)+abs(model_flux)+epsilon)",
        "mechanism": "normalized breakdown of the Prandtl closure",
    },
    "turbulent_schmidt_gap": {
        "equation": "nu_t/D_m(imbalance)-nu_t/D_m(basis)",
        "mechanism": "difference between momentum and scalar turbulent transport efficiencies",
    },
    "eddy_flux_release": {
        "equation": "-Z(closure_residual)*Delta Z(closure_residual)",
        "mechanism": "relaxation of turbulent momentum-flux mispricing",
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


def prior_diffusivity(values: pd.Series, window: int) -> pd.Series:
    minimum = max(36, int(0.60 * window))
    return values.diff().pow(2).shift(1).ewm(span=window, adjust=False, min_periods=minimum).mean()


def prior_relaxation(values: pd.Series, window: int) -> pd.Series:
    current = pd.to_numeric(values, errors="coerce").shift(1)
    lagged = pd.to_numeric(values, errors="coerce").shift(2)
    minimum = max(36, int(0.60 * window))
    mx = current.rolling(window, min_periods=minimum).mean()
    my = lagged.rolling(window, min_periods=minimum).mean()
    cov = (current * lagged).rolling(window, min_periods=minimum).mean() - mx * my
    vx = current.pow(2).rolling(window, min_periods=minimum).mean() - mx.pow(2)
    vy = lagged.pow(2).rolling(window, min_periods=minimum).mean() - my.pow(2)
    rho = (cov / (np.sqrt(vx.clip(lower=0.0) * vy.clip(lower=0.0)) + EPS)).clip(-0.90, 0.90)
    return ((1.0 + rho) / (1.0 - rho + EPS)).clip(0.25, 24.0)


def robust_median(items: list[pd.Series]) -> pd.Series:
    return r16.robust_median(items)


def add_round18_mechanisms(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    g = group.sort_values("time").copy()
    u_p = pd.to_numeric(g["perp_u"], errors="coerce")
    u_s = pd.to_numeric(g["spot_u"], errors="coerce")
    shear = u_p - u_s
    strain = u_p + u_s
    basis = pd.to_numeric(g["basis_z"], errors="coerce")
    pressure = pd.to_numeric(g["pressure_gap"], errors="coerce")
    pressure_gradient = pd.to_numeric(g["basis_change_z"], errors="coerce") + 0.25 * pd.to_numeric(g["funding_change_z"], errors="coerce").fillna(0.0)
    source = (
        pd.to_numeric(g["oi_source_z"], errors="coerce").fillna(0.0)
        + 0.50 * pd.to_numeric(g["funding_change_z"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(g["quote_share_z"], errors="coerce").diff().fillna(0.0)
    )
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
    scalars = {
        "imbalance": flow_p - flow_s,
        "basis": basis,
        "liquidity": density_gradient,
    }

    diffusivities: dict[str, list[pd.Series]] = {key: [] for key in scalars}
    effective: dict[str, list[pd.Series]] = {key: [] for key in scalars}
    scalar_residuals: list[pd.Series] = []
    capacity_parts: list[pd.Series] = []
    for window in CLOSURE_WINDOWS:
        for key, scalar in scalars.items():
            dm = prior_diffusivity(scalar, window)
            tau = prior_relaxation(scalar, window)
            deff = dm + tau * shear.pow(2)
            flux = -deff * scalar
            scalar_source = source if key != "basis" else pd.to_numeric(g["funding_change_z"], errors="coerce").fillna(0.0)
            residual = scalar.diff() + 2.0 * flux - scalar_source
            diffusivities[key].append(dm)
            effective[key].append(deff)
            scalar_residuals.append(residual)
            capacity_parts.append((deff - dm) / (dm + EPS))

    dm_imb = robust_median(diffusivities["imbalance"])
    dm_basis = robust_median(diffusivities["basis"])
    dm_liq = robust_median(diffusivities["liquidity"])
    deff_imb = robust_median(effective["imbalance"])
    deff_basis = robust_median(effective["basis"])
    deff_liq = robust_median(effective["liquidity"])
    taylor_residual = robust_median(scalar_residuals)

    raw: dict[str, pd.Series] = {}
    raw["imbalance_taylor_dispersion"] = deff_imb
    raw["basis_taylor_dispersion"] = deff_basis
    raw["liquidity_taylor_dispersion"] = deff_liq
    raw["taylor_flux_residual"] = taylor_residual
    residual_z = base.causal_z(taylor_residual)
    raw["taylor_residual_release"] = -residual_z * residual_z.diff()
    raw["dispersion_capacity_ratio"] = robust_median(capacity_parts)

    stratification = -pressure * density_gradient
    reference_diffusion = np.sqrt((dm_imb + dm_basis + dm_liq).clip(lower=0.0) / 3.0)
    ri = (stratification / (shear.pow(2) + reference_diffusion + EPS)).clip(-8.0, 8.0)
    criticality = 0.25 - ri
    kh_growth = shear.abs() * np.sqrt((1.0 - 4.0 * ri).clip(lower=0.0))
    atwood = (rho_p - rho_s) / (rho_p + rho_s + EPS)
    raw["richardson_criticality"] = criticality
    raw["kh_growth_rate"] = kh_growth
    raw["kh_unstable_energy"] = kh_growth * (u_p.pow(2) + u_s.pow(2) + basis.pow(2))
    critical_z = base.causal_z(criticality)
    raw["kh_stability_restoration"] = -critical_z * critical_z.diff()
    raw["atwood_shear_instability"] = atwood * kh_growth
    raw["kh_source_loading"] = kh_growth * source

    displacement = density_gradient.abs() / (1.0 + density_gradient.abs())
    adverse = (-shear * pressure_gradient).clip(lower=0.0)
    sep_beta = displacement * adverse / (shear.pow(2) + reference_diffusion + EPS)
    skin_friction = shear / (np.sqrt(reference_diffusion.clip(lower=0.0)) + EPS)
    reattach_event = (-sep_beta.diff()).clip(lower=0.0) * shear.abs().diff().clip(lower=0.0)
    reattach_flag = ((sep_beta.diff() < 0.0) & (shear.abs().diff() > 0.0)).astype(float)
    raw["adverse_pressure_separation"] = sep_beta
    raw["skin_friction_collapse"] = -skin_friction.abs()
    raw["separation_bubble_energy"] = sep_beta * (basis.pow(2) + shear.pow(2))
    raw["reattachment_impulse"] = reattach_event
    raw["reattachment_persistence"] = ewm(reattach_flag, 24, 6)
    raw["basis_boundary_recovery"] = -basis * pressure_gradient / (1.0 + sep_beta)

    mean_velocity = 0.5 * (u_p + u_s)
    baroclinic = pressure_gradient * density_gradient - pressure * density_gradient.diff()
    pv = shear * density_gradient
    pv_material = pv.diff() + mean_velocity * (pv - ewm(pv, 24, 12))
    pv_residual = pv_material - (baroclinic + source * shear)
    log_p = np.log(pd.to_numeric(g["close"], errors="coerce"))
    log_s = np.log(pd.to_numeric(g["close_spot"], errors="coerce"))
    cascade_parts: list[pd.Series] = []
    for scale in [6, 24, 72]:
        dp = log_p.diff(scale) / math.sqrt(scale)
        ds = log_s.diff(scale) / math.sqrt(scale)
        cascade_parts.append(-(dp.pow(3) + ds.pow(3)) / float(scale))
    joint_cascade = robust_median(cascade_parts)
    raw["two_layer_potential_vorticity"] = pv
    raw["pv_material_defect"] = pv_material
    raw["pv_source_residual"] = pv_residual
    pv_z = base.causal_z(pv_residual)
    raw["pv_residual_release"] = -pv_z * pv_z.diff()
    raw["pv_basis_coupling"] = pv * basis
    raw["pv_cascade_alignment"] = pv * joint_cascade

    mixing_length = 1.0 / np.sqrt(rho_p + rho_s + EPS)
    nu_t = mixing_length.pow(2) * shear.abs()
    enstrophy = 0.5 * shear.pow(2)
    enstrophy_prod = shear.pow(2) * strain
    enstrophy_diss = nu_t * shear.diff().pow(2)
    enstrophy_residual = enstrophy.diff() - enstrophy_prod + enstrophy_diss - source * shear
    raw["enstrophy_production"] = enstrophy_prod
    raw["enstrophy_dissipation"] = enstrophy_diss
    raw["vortex_stretching_residual"] = enstrophy_residual
    ens_z = base.causal_z(enstrophy_residual)
    raw["enstrophy_release"] = -ens_z * ens_z.diff()
    raw["strain_vorticity_alignment"] = shear * strain
    raw["vortex_source_loading"] = shear * source

    model_flux = -nu_t * shear
    observed_transfer = u_s - u_p
    eddy_residual = observed_transfer - model_flux
    raw["prandtl_eddy_viscosity"] = nu_t
    raw["mixing_length_momentum_flux"] = model_flux
    raw["eddy_viscosity_closure_residual"] = eddy_residual
    raw["mixing_length_breakdown"] = eddy_residual / (observed_transfer.abs() + model_flux.abs() + EPS)
    raw["turbulent_schmidt_gap"] = nu_t / (dm_imb + EPS) - nu_t / (dm_basis + EPS)
    eddy_z = base.causal_z(eddy_residual)
    raw["eddy_flux_release"] = -eddy_z * eddy_z.diff()

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
        enriched, local_map = add_round18_mechanisms(local)
        parts.append(enriched)
        candidate_map = local_map
        print(f"round18_mechanisms={symbol} rows={len(enriched)}", flush=True)
    if not parts:
        raise RuntimeError("no usable discovery data")
    panel = pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    return panel, audit, candidate_map


def robust_select(panel: pd.DataFrame, candidate_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hasattr(base, "_DEV_CACHE"):
        base._DEV_CACHE.clear()
    rows: list[dict[str, Any]] = []
    for family, factors in candidate_map.items():
        for factor in factors:
            for horizon in HORIZONS:
                for sign in [1, -1]:
                    result = base.development_score(panel, factor, horizon, sign)
                    result["family"] = family
                    rows.append(result)
    scan = pd.DataFrame(rows)

    factor_candidates: list[pd.Series] = []
    for (family, factor), group in scan.groupby(["family", "factor"], sort=False):
        sign_options: list[dict[str, Any]] = []
        for sign, signed in group.groupby("sign", sort=False):
            supported = signed.loc[
                (signed["development_cs"] > 0.005)
                & (signed["development_min_year"] > -0.015)
            ]
            short_count = int((supported["horizon"] <= 120).sum())
            slow_count = int((supported["horizon"] >= 120).sum())
            support_count = int(len(supported))
            robust_score = float(signed["development_score"].median())
            sign_options.append({
                "sign": int(sign),
                "support_count": support_count,
                "short_support": short_count,
                "slow_support": slow_count,
                "robust_score": robust_score,
            })
        eligible_signs = [item for item in sign_options if item["support_count"] >= 3 and item["short_support"] >= 1 and item["slow_support"] >= 1]
        if not eligible_signs:
            continue
        chosen_sign = max(eligible_signs, key=lambda item: (item["robust_score"], item["support_count"]))
        signed = group.loc[group["sign"] == chosen_sign["sign"]].copy()
        supported_target = signed.loc[
            (signed["horizon"] <= 240)
            & (signed["development_cs"] > 0.005)
            & (signed["development_min_year"] > -0.015)
        ]
        if supported_target.empty:
            continue
        chosen = supported_target.sort_values("development_score", ascending=False).iloc[0].copy()
        chosen["multi_horizon_support"] = chosen_sign["support_count"]
        chosen["short_support"] = chosen_sign["short_support"]
        chosen["slow_support"] = chosen_sign["slow_support"]
        chosen["robust_multi_horizon_score"] = chosen_sign["robust_score"]
        factor_candidates.append(chosen)

    candidates = pd.DataFrame(factor_candidates)
    if candidates.empty:
        return scan, candidates

    dev = panel.loc[panel["time"].dt.year <= 2022].copy()
    selected: list[pd.Series] = []
    for family, group in candidates.groupby("family", sort=False):
        family_selected: list[pd.Series] = []
        for _, row in group.sort_values(["robust_multi_horizon_score", "development_score"], ascending=False).iterrows():
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
        selected.extend(family_selected)
    selection = pd.DataFrame(selected).reset_index(drop=True)
    if not selection.empty:
        selection["oriented_factor"] = selection["factor"].map(lambda value: f"oriented_{value}")
        for row in selection.itertuples(index=False):
            panel[str(row.oriented_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")
    return scan, selection


def apply_frozen(panel: pd.DataFrame, selection: pd.DataFrame) -> None:
    for row in selection.itertuples(index=False):
        panel[str(row.oriented_factor)] = int(row.sign) * pd.to_numeric(panel[str(row.factor)], errors="coerce")


def validate_stage(panel: pd.DataFrame, selection: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        result = base.evaluate(panel, str(row.oriented_factor), int(row.horizon), 2022, 2023, 2024)
        result.update({
            "family": str(row.family),
            "primitive_factor": str(row.factor),
            "sign": int(row.sign),
            "development_score": float(row.development_score),
            "multi_horizon_support": int(row.multi_horizon_support),
            "robust_multi_horizon_score": float(row.robust_multi_horizon_score),
        })
        validation_rows.append(result)
    validation = pd.DataFrame(validation_rows)
    if validation.empty:
        validation.to_csv(OUT / f"{prefix}_VALIDATION_RESULTS.csv", index=False)
        pd.DataFrame().to_csv(OUT / f"{prefix}_HOLDOUT_RESULTS.csv", index=False)
        return validation, pd.DataFrame(), []
    validation["holm_p"] = base.holm_adjust([
        float(value) if np.isfinite(value) else 1.0 for value in validation["bootstrap_p_nonpositive"]
    ])
    validation["validation_gate"] = [r17.gate(row, float(row["holm_p"])) for _, row in validation.iterrows()]
    validation.to_csv(OUT / f"{prefix}_VALIDATION_RESULTS.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    for _, row in validation.loc[validation["validation_gate"]].iterrows():
        result = base.evaluate(panel, str(row["factor"]), int(row["horizon"]), 2024, 2025, 2026)
        result.update({
            "family": row["family"],
            "primitive_factor": row["primitive_factor"],
            "sign": int(row["sign"]),
            "validation_holm_p": float(row["holm_p"]),
            "multi_horizon_support": int(row["multi_horizon_support"]),
        })
        holdout_rows.append(result)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=validation.columns).to_csv(OUT / f"{prefix}_HOLDOUT_RESULTS.csv", index=False)
    else:
        holdout.to_csv(OUT / f"{prefix}_HOLDOUT_RESULTS.csv", index=False)

    passed: list[dict[str, Any]] = []
    for _, row in holdout.iterrows():
        if r17.gate(row):
            passed.append({
                "family": row["family"],
                "primitive_factor": row["primitive_factor"],
                "oriented_factor": row["factor"],
                "sign": int(row["sign"]),
                "horizon": int(row["horizon"]),
                "multi_horizon_support": int(row["multi_horizon_support"]),
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
        "selection_rule": "orientation selected by median development score across all seven horizons; require at least three supported horizons, at least one <=120h and at least one >=120h; target horizon restricted to <=240h; up to two independent factors per family with |rho|<0.85",
        "inference": "circular moving-block bootstrap on non-overlapping issue grid, 4 issue observations, 10000 repetitions",
        "gate": "CS>0.012, block p<=0.10, Holm<=0.10 in validation, Ridge MSE gain>0, residual CS>0.005, max parent corr<0.80, 5bp annual return>0 and every stage year positive",
        "replication_policy": "replication universe is not downloaded unless a discovery candidate passes discovery validation and historical holdout",
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
        "# Fluid shear dispersion, stability and boundary-layer research round18",
        "",
        "This experiment implements Taylor-Aris shear dispersion, Kelvin-Helmholtz Richardson criticality, adverse-pressure boundary separation and reattachment, two-layer potential-vorticity conservation, enstrophy stretching, and Prandtl mixing-length closure on completely fresh discovery and replication universes.",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen selection",
        selection.to_markdown(index=False) if not selection.empty else "No identity satisfied the predeclared multi-horizon development rule.",
        "",
        "## Discovery validation",
        discovery_validation.to_markdown(index=False) if not discovery_validation.empty else "Validation remained closed.",
        "",
        "## Discovery holdout",
        discovery_holdout.to_markdown(index=False) if not discovery_holdout.empty else "Discovery holdout remained closed.",
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
