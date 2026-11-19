from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent / "ashare_fluid_round10"
if str(ROUND10) not in sys.path:
    sys.path.insert(0, str(ROUND10))

import run_baostock as source  # noqa: E402

SEED = 20261119
EPS = 1.0e-12
RNG = np.random.default_rng(SEED)
OUT = Path("research_outputs/ashare_fluid_round15")
OUT.mkdir(parents=True, exist_ok=True)

DEV0, DEV1 = pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")
VAL0, VAL1 = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")
HOLD0, HOLD1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-11-15")
HORIZONS = [1, 5, 10, 20]

FAMILIES: dict[str, list[str]] = {
    "riemann_transmission": [
        "riemann_transmission_defect",
        "riemann_incoming_absorption",
        "impedance_matched_transport",
    ],
    "impedance_release": [
        "impedance_pressure_release",
        "impedance_energy_absorption",
        "impedance_reflection_trend",
    ],
    "characteristic_reattachment": [
        "characteristic_reattachment",
        "incoming_wave_release",
        "outgoing_wave_persistence",
    ],
    "wave_action": [
        "wave_action_divergence",
        "wave_action_residual_release",
        "adiabatic_action_defect",
    ],
    "shock_rarefaction": [
        "characteristic_shock_release",
        "characteristic_rarefaction",
        "shock_source_coupling",
    ],
    "baroclinic_crocco": [
        "baroclinic_circulation",
        "crocco_residual_release",
        "crocco_source_coupling",
    ],
    "helicity_transport": [
        "helicity_transport",
        "helicity_release",
        "helicity_source_coupling",
    ],
    "source_characteristics": [
        "source_supported_transmission",
        "characteristic_continuity_residual",
        "source_absorption_release",
    ],
}

BASE_RAW = [
    "mom1", "mom5", "mom10", "mom20", "mom60",
    "vol5", "vol20", "vol60", "range5", "range20",
    "amount_z", "volume_z", "turnover_z", "amihud_z",
    "gap", "close_loc", "skew20", "kurt20",
    "density_z", "pressure", "source", "sound_c2", "sound_speed",
    "impedance", "peer_u", "peer_pressure", "peer_density",
    "graph_circulation", "continuity_residual_parent",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, minimum: int = 8) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < minimum:
        return float("nan")
    if np.nanstd(a[mask]) <= EPS or np.nanstd(b[mask]) <= EPS:
        return float("nan")
    value = spearmanr(a[mask], b[mask]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def prior_z(series: pd.Series, codes: pd.Series, window: int = 120, minimum: int = 60) -> pd.Series:
    prior = series.groupby(codes, sort=False).shift(1)
    group = prior.groupby(codes, sort=False)
    mean = group.transform(lambda x: x.rolling(window, min_periods=minimum).mean())
    std = group.transform(lambda x: x.rolling(window, min_periods=minimum).std(ddof=0))
    return ((series - mean) / (std + EPS)).clip(-12.0, 12.0)


def rolling_beta_prior(y: pd.Series, x: pd.Series, codes: pd.Series, window: int = 120) -> pd.Series:
    frame = pd.DataFrame({"code": codes, "x": x, "y": y})
    pieces: list[pd.Series] = []
    for _, group in frame.groupby("code", sort=False):
        xp = group["x"].shift(1)
        yp = group["y"].shift(1)
        minimum = max(60, int(0.6 * window))
        mx = xp.rolling(window, min_periods=minimum).mean()
        my = yp.rolling(window, min_periods=minimum).mean()
        cov = (xp * yp).rolling(window, min_periods=minimum).mean() - mx * my
        var = xp.pow(2).rolling(window, min_periods=minimum).mean() - mx.pow(2)
        pieces.append((cov / (var + EPS)).clip(-16.0, 16.0))
    return pd.concat(pieces).sort_index()


def cross_sectional_z(frame: pd.DataFrame, column: str) -> pd.Series:
    def transform(values: pd.Series) -> pd.Series:
        low, high = values.quantile(0.01), values.quantile(0.99)
        clipped = values.clip(low, high)
        return (clipped - clipped.mean()) / (clipped.std(ddof=0) + EPS)
    return frame.groupby("date", sort=False)[column].transform(transform)


def daily_rankic(frame: pd.DataFrame, factor: str, target: str, minimum: int = 30) -> pd.Series:
    work = frame[["date", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return pd.Series(dtype=float)
    values = work.groupby("date", sort=False).apply(
        lambda g: safe_spearman(g[factor], g[target], minimum=minimum),
        include_groups=False,
    )
    return pd.to_numeric(values, errors="coerce")


def time_series_ic(frame: pd.DataFrame, factor: str, target: str, minimum: int = 60) -> tuple[float, float, int]:
    values: list[float] = []
    for _, group in frame.groupby("code", sort=False):
        value = safe_spearman(group[factor], group[target], minimum=minimum)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.mean(array > 0.0)), int(len(array))


def moving_block_bootstrap(values: pd.Series, block: int = 20, repetitions: int = 2500) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(array) < max(30, block * 2):
        return {"mean": float(np.nanmean(array)) if len(array) else float("nan"), "low": float("nan"), "high": float("nan"), "p_nonpositive": float("nan")}
    starts = np.arange(0, len(array) - block + 1)
    draws = np.empty(repetitions, dtype=float)
    blocks = int(math.ceil(len(array) / block))
    for index in range(repetitions):
        chosen = RNG.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([array[start:start + block] for start in chosen])[: len(array)]
        draws[index] = np.mean(sample)
    return {
        "mean": float(np.mean(array)),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "p_nonpositive": float(np.mean(draws <= 0.0)),
    }


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    array = np.asarray([value if np.isfinite(value) else 1.0 for value in pvalues], dtype=float)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 0.0
    n = len(array)
    for rank, position in enumerate(order):
        candidate = min(1.0, (n - rank) * array[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted


def period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    if period == "development":
        return frame["date"].between(DEV0, DEV1)
    if period == "validation":
        return frame["date"].between(VAL0, VAL1)
    if period == "holdout":
        return frame["date"].between(HOLD0, HOLD1)
    raise KeyError(period)


def build_fixed_graph(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    development = frame.loc[frame["date"].between(DEV0, DEV1)].copy()
    codes = sorted(frame["code"].unique())
    ret = development.pivot(index="date", columns="code", values="u").reindex(columns=codes)
    act = development.pivot(index="date", columns="code", values="amount_z").reindex(columns=codes)
    ret_corr = ret.corr(min_periods=250).fillna(0.0).to_numpy(float)
    act_corr = act.corr(min_periods=250).fillna(0.0).to_numpy(float)
    similarity = 0.75 * np.maximum(ret_corr, 0.0) + 0.25 * np.maximum(act_corr, 0.0)
    np.fill_diagonal(similarity, 0.0)
    weights = np.zeros_like(similarity)
    k = min(10, len(codes) - 1)
    for row in range(len(codes)):
        order = np.argsort(similarity[row])[-k:]
        selected = order[similarity[row, order] > 0.03]
        if len(selected) == 0:
            selected = order[-min(3, len(order)):]
        values = similarity[row, selected]
        weights[row, selected] = values / (values.sum() + EPS)

    # Strictly development-estimated directed lag matrix for graph circulation.
    values = ret.to_numpy(float)
    directed = np.zeros_like(similarity)
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            x = values[:-1, i]
            y = values[1:, j]
            reverse_x = values[:-1, j]
            reverse_y = values[1:, i]
            forward = safe_spearman(x, y, minimum=250)
            reverse = safe_spearman(reverse_x, reverse_y, minimum=250)
            if np.isfinite(forward) and np.isfinite(reverse):
                asym = forward - reverse
                directed[i, j] = asym
                directed[j, i] = -asym
    scale = np.sum(np.abs(directed), axis=1, keepdims=True)
    directed = directed / (scale + EPS)
    audit = {
        "codes": len(codes),
        "neighbors": k,
        "mean_nonzero_neighbors": float(np.mean((weights > 0).sum(axis=1))),
        "mean_weighted_similarity": float(np.sum(weights * similarity) / max(1, np.count_nonzero(weights))),
        "directed_nonzero": int(np.count_nonzero(np.abs(directed) > 0)),
        "graph_period": f"{DEV0.date()} to {DEV1.date()}",
    }
    return weights, directed, codes, audit


def graph_apply(frame: pd.DataFrame, column: str, weights: np.ndarray, codes: list[str]) -> pd.Series:
    pivot = frame.pivot(index="date", columns="code", values=column).reindex(columns=codes)
    values = pivot.to_numpy(float)
    row_median = np.nanmedian(values, axis=1)
    filled = np.where(np.isfinite(values), values, row_median[:, None])
    filled = np.where(np.isfinite(filled), filled, 0.0)
    peer = filled @ weights.T
    stacked = pd.DataFrame(peer, index=pivot.index, columns=codes).stack(dropna=False)
    lookup = pd.MultiIndex.from_arrays([frame["date"], frame["code"]])
    return stacked.reindex(lookup).to_numpy(float)


def graph_directed(frame: pd.DataFrame, column: str, directed: np.ndarray, codes: list[str]) -> pd.Series:
    pivot = frame.pivot(index="date", columns="code", values=column).reindex(columns=codes)
    values = pivot.to_numpy(float)
    row_median = np.nanmedian(values, axis=1)
    filled = np.where(np.isfinite(values), values, row_median[:, None])
    filled = np.where(np.isfinite(filled), filled, 0.0)
    result = filled @ directed.T
    stacked = pd.DataFrame(result, index=pivot.index, columns=codes).stack(dropna=False)
    lookup = pd.MultiIndex.from_arrays([frame["date"], frame["code"]])
    return stacked.reindex(lookup).to_numpy(float)


def build_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], list[str], dict[str, Any]]:
    frame = raw.sort_values(["code", "date"]).copy()
    group = frame.groupby("code", sort=False)
    frame["r1"] = group["close"].transform(lambda x: np.log(x).diff())
    frame["sigma20"] = np.sqrt(group["r1"].transform(lambda x: x.shift(1).pow(2).rolling(20, min_periods=12).mean()))
    frame["u"] = (frame["r1"] / (frame["sigma20"] + EPS)).clip(-12.0, 12.0)
    for horizon in [1, 5, 10, 20, 60]:
        frame[f"mom{horizon}"] = group["r1"].transform(lambda x, h=horizon: x.rolling(h, min_periods=max(1, int(0.7 * h))).sum())
    for window in [5, 20, 60]:
        frame[f"vol{window}"] = np.sqrt(group["r1"].transform(lambda x, w=window: x.pow(2).rolling(w, min_periods=max(3, int(0.7 * w))).mean()))
    range1 = np.log(frame["high"] / frame["low"])
    frame["range1"] = range1
    for window in [5, 20]:
        frame[f"range{window}"] = range1.groupby(frame["code"]).transform(lambda x, w=window: x.rolling(w, min_periods=max(3, int(0.7 * w))).mean())
    frame["amount_z"] = prior_z(np.log1p(frame["amount"]), frame["code"])
    frame["volume_z"] = prior_z(np.log1p(frame["volume"]), frame["code"])
    frame["turnover_z"] = prior_z(frame["turnover"], frame["code"])
    frame["range_z"] = prior_z(range1, frame["code"])
    frame["amihud_z"] = prior_z(np.log1p(frame["r1"].abs() / (frame["amount"] / 1e8 + EPS)), frame["code"])
    frame["gap"] = np.log(frame["open"] / group["close"].shift(1))
    frame["close_loc"] = (frame["close"] - frame["low"]) / (frame["high"] - frame["low"] + EPS) - 0.5
    frame["skew20"] = group["r1"].transform(lambda x: x.rolling(20, min_periods=14).skew())
    frame["kurt20"] = group["r1"].transform(lambda x: x.rolling(20, min_periods=14).kurt())

    frame["density_z"] = 0.40 * frame["amount_z"] + 0.30 * frame["turnover_z"] + 0.30 * frame["volume_z"]
    frame["rho"] = np.exp(np.clip(frame["density_z"] / 3.0, -3.0, 3.0))
    frame["entropy"] = 0.50 * prior_z(frame["vol20"], frame["code"]) + 0.30 * frame["range_z"] + 0.20 * frame["amihud_z"].abs()
    frame["temperature"] = np.exp(np.clip(frame["entropy"] / 4.0, -3.0, 3.0))
    pressure_raw = 0.45 * frame["u"].pow(2) + 0.30 * frame["range_z"] + 0.25 * frame["amihud_z"]
    frame["pressure"] = prior_z(pressure_raw, frame["code"])
    dp = frame.groupby("code", sort=False)["pressure"].diff()
    drho = frame.groupby("code", sort=False)["density_z"].diff()
    frame["sound_c2"] = rolling_beta_prior(dp, drho, frame["code"], window=120)
    frame["sound_speed"] = np.sqrt(frame["sound_c2"].abs() + 0.05).clip(0.15, 4.0)
    frame["impedance"] = frame["rho"] * frame["sound_speed"]
    frame["source"] = frame.groupby("code", sort=False)["density_z"].diff()
    frame["total_enthalpy"] = frame["pressure"] / (frame["rho"] + EPS) + 0.5 * frame["u"].pow(2)

    weights, directed, codes, graph_audit = build_fixed_graph(frame)
    for column, output in [
        ("u", "peer_u"),
        ("pressure", "peer_pressure"),
        ("rho", "peer_rho"),
        ("density_z", "peer_density"),
        ("entropy", "peer_entropy"),
        ("temperature", "peer_temperature"),
        ("total_enthalpy", "peer_enthalpy"),
        ("impedance", "peer_impedance"),
        ("source", "peer_source"),
    ]:
        frame[output] = graph_apply(frame, column, weights, codes)
    frame["graph_circulation"] = graph_directed(frame, "u", directed, codes)
    frame["grad_pressure"] = frame["peer_pressure"] - frame["pressure"]
    frame["grad_density"] = frame["peer_density"] - frame["density_z"]
    frame["continuity_residual_parent"] = frame["source"] + frame["rho"] * (frame["peer_u"] - frame["u"])

    z_self = frame["impedance"]
    z_peer = frame["peer_impedance"]
    reflection = (z_self - z_peer) / (z_self + z_peer + EPS)
    transmission = 2.0 * z_peer / (z_self + z_peer + EPS)
    pressure_velocity = frame["pressure"] / (z_self + EPS)
    j_plus = frame["u"] + pressure_velocity
    j_minus = frame["u"] - pressure_velocity
    peer_j_plus = graph_apply(pd.DataFrame({**{column: frame[column] for column in ["date", "code"]}, "value": j_plus}), "value", weights, codes)
    peer_j_minus = graph_apply(pd.DataFrame({**{column: frame[column] for column in ["date", "code"]}, "value": j_minus}), "value", weights, codes)
    frame["j_plus"] = j_plus
    frame["j_minus"] = j_minus
    frame["peer_j_plus"] = peer_j_plus
    frame["peer_j_minus"] = peer_j_minus

    raw: dict[str, pd.Series] = {}
    raw["riemann_transmission_defect"] = transmission * peer_j_plus - j_plus
    raw["riemann_incoming_absorption"] = -(j_minus - peer_j_minus)
    raw["impedance_matched_transport"] = (1.0 - reflection.abs()) * (transmission * frame["peer_u"] - frame["u"])
    raw["impedance_pressure_release"] = -reflection * frame["grad_pressure"]
    acoustic_energy = 0.25 * frame["rho"] * (j_plus.pow(2) + j_minus.pow(2))
    peer_energy = graph_apply(pd.DataFrame({"date": frame["date"], "code": frame["code"], "value": acoustic_energy}), "value", weights, codes)
    raw["impedance_energy_absorption"] = (peer_energy - acoustic_energy) * (1.0 - reflection.abs())
    raw["impedance_reflection_trend"] = -reflection * np.sign(frame["mom20"])

    gap = np.sqrt((j_plus - peer_j_plus) ** 2 + (j_minus - peer_j_minus) ** 2)
    prior_gap = gap.groupby(frame["code"], sort=False).shift(1)
    raw["characteristic_reattachment"] = np.sign(frame["mom20"]) * prior_gap * (prior_gap - gap).clip(lower=0.0) * np.exp(-frame["range_z"].abs() / 2.0)
    prior_incoming = (j_minus - peer_j_minus).abs().groupby(frame["code"], sort=False).shift(1)
    raw["incoming_wave_release"] = np.sign(frame["mom20"]) * prior_incoming * (prior_incoming - (j_minus - peer_j_minus).abs()).clip(lower=0.0)
    raw["outgoing_wave_persistence"] = np.sign(frame["mom20"]) * (j_plus - peer_j_plus) * np.exp(-gap.clip(0.0, 12.0))

    action = acoustic_energy / (frame["sound_speed"] + 0.10)
    flux = 0.25 * frame["rho"] * ((frame["u"] + frame["sound_speed"]) * j_plus.pow(2) + (frame["u"] - frame["sound_speed"]) * j_minus.pow(2))
    peer_flux = graph_apply(pd.DataFrame({"date": frame["date"], "code": frame["code"], "value": flux}), "value", weights, codes)
    action_div = peer_flux - flux
    action_delta = action.groupby(frame["code"], sort=False).diff()
    action_residual = action_delta - action_div
    raw["wave_action_divergence"] = action_div
    raw["wave_action_residual_release"] = -action_residual * action_residual.groupby(frame["code"], sort=False).diff()
    c_delta = frame.groupby("code", sort=False)["sound_speed"].diff()
    raw["adiabatic_action_defect"] = action_delta / (action.groupby(frame["code"], sort=False).shift(1).abs() + 0.25) - c_delta / (frame.groupby("code", sort=False)["sound_speed"].shift(1).abs() + 0.25)

    lambda_plus = frame["u"] + frame["sound_speed"]
    lambda_minus = frame["u"] - frame["sound_speed"]
    peer_lambda_plus = graph_apply(pd.DataFrame({"date": frame["date"], "code": frame["code"], "value": lambda_plus}), "value", weights, codes)
    peer_lambda_minus = graph_apply(pd.DataFrame({"date": frame["date"], "code": frame["code"], "value": lambda_minus}), "value", weights, codes)
    shock = (-(peer_lambda_plus - lambda_plus)).clip(lower=0.0) * (peer_lambda_minus - lambda_minus).clip(lower=0.0)
    rarefaction = (peer_lambda_plus - lambda_plus).clip(lower=0.0) * (-(peer_lambda_minus - lambda_minus)).clip(lower=0.0)
    entropy_delta = frame.groupby("code", sort=False)["entropy"].diff()
    raw["characteristic_shock_release"] = -shock * entropy_delta
    raw["characteristic_rarefaction"] = rarefaction
    raw["shock_source_coupling"] = shock * frame["source"]

    crocco = (frame["peer_enthalpy"] - frame["total_enthalpy"]) - frame["temperature"] * (frame["peer_entropy"] - frame["entropy"]) - frame["u"] * frame["graph_circulation"]
    raw["baroclinic_circulation"] = frame["grad_density"] * frame["grad_pressure"] * frame["graph_circulation"]
    raw["crocco_residual_release"] = -crocco * crocco.groupby(frame["code"], sort=False).diff()
    raw["crocco_source_coupling"] = crocco * frame["source"]
    helicity = frame["u"] * frame["graph_circulation"]
    peer_helicity = graph_apply(pd.DataFrame({"date": frame["date"], "code": frame["code"], "value": helicity}), "value", weights, codes)
    raw["helicity_transport"] = peer_helicity - helicity
    raw["helicity_release"] = -helicity * helicity.groupby(frame["code"], sort=False).diff()
    raw["helicity_source_coupling"] = frame["source"] * 0.5 * (frame["graph_circulation"] + graph_apply(pd.DataFrame({"date": frame["date"], "code": frame["code"], "value": frame["graph_circulation"]}), "value", weights, codes))

    raw["source_supported_transmission"] = raw["riemann_transmission_defect"] * frame["source"]
    raw["characteristic_continuity_residual"] = -(j_plus.groupby(frame["code"], sort=False).diff() + lambda_plus * (peer_j_plus - j_plus) - frame["source"])
    prior_source = frame["source"].abs().groupby(frame["code"], sort=False).shift(1)
    raw["source_absorption_release"] = np.sign(frame["mom20"]) * prior_source * (prior_source - frame["source"].abs()).clip(lower=0.0) * (1.0 - reflection.abs())

    family_candidates: dict[str, list[str]] = {family: [] for family in FAMILIES}
    for family, names in FAMILIES.items():
        for name in names:
            frame[name] = raw[name].clip(-50.0, 50.0)
            candidates = [name]
            for span in [5, 20]:
                column = f"{name}_ewm{span}"
                frame[column] = frame.groupby("code", sort=False)[name].transform(lambda x, s=span: x.ewm(span=s, adjust=False, min_periods=max(3, s // 2)).mean())
                candidates.append(column)
            column = f"{name}_z120"
            frame[column] = prior_z(frame[name], frame["code"], 120, 60)
            candidates.append(column)
            family_candidates[family].extend(candidates)

    # Point-in-time liquidity strata and targets.
    frame["liq20"] = frame.groupby("code", sort=False)["amount"].transform(lambda x: x.shift(1).rolling(20, min_periods=12).median())
    frame["liq_pct"] = frame.groupby(["date", "market"], sort=False)["liq20"].rank(pct=True)
    frame["bucket"] = np.where(frame["liq_pct"] >= 2 / 3, "high", np.where(frame["liq_pct"] >= 1 / 3, "secondary", "tail"))
    for horizon in HORIZONS:
        future = np.log(frame.groupby("code", sort=False)["close"].shift(-horizon) / frame.groupby("code", sort=False)["open"].shift(-1))
        frame[f"target{horizon}_raw"] = future
        frame[f"target{horizon}"] = future - future.groupby([frame["date"], frame["market"]]).transform("median")

    base_features: list[str] = []
    for column in BASE_RAW:
        if column not in frame:
            continue
        output = f"cs_{column}"
        frame[output] = cross_sectional_z(frame, column)
        base_features.append(output)
    return frame, family_candidates, base_features, graph_audit


def development_score(frame: pd.DataFrame, factor: str, horizon: int) -> dict[str, Any]:
    target = f"target{horizon}"
    work = frame.loc[period_mask(frame, "development") & frame["bucket"].isin(["high", "secondary"])].copy()
    overall = daily_rankic(work, factor, target, minimum=50).mean()
    orientation = 1 if not np.isfinite(overall) or overall >= 0 else -1
    work["__factor"] = orientation * work[factor]
    groups: list[float] = []
    scopes = {
        "early": work["date"].le("2020-06-30"),
        "late": work["date"].ge("2020-07-01"),
        "SH": work["market"].eq("SH"),
        "SZ": work["market"].eq("SZ"),
        "high": work["bucket"].eq("high"),
        "secondary": work["bucket"].eq("secondary"),
    }
    scope_values: dict[str, float] = {}
    for name, mask in scopes.items():
        value = daily_rankic(work.loc[mask], "__factor", target, minimum=18).mean()
        scope_values[name] = float(value) if np.isfinite(value) else float("nan")
        if np.isfinite(value):
            groups.append(float(value))
    ts, _, _ = time_series_ic(work, "__factor", target, minimum=100)
    score = (min(groups) if groups else -1.0) + 0.35 * (np.mean(groups) if groups else -1.0) + 0.10 * (ts if np.isfinite(ts) else -1.0)
    return {"factor": factor, "horizon": horizon, "orientation": orientation, "development_rankic": float(orientation * overall), "development_ts_ic": ts, "development_score": float(score), **scope_values}


def freeze_identities(frame: pd.DataFrame, family_candidates: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for family, candidates in family_candidates.items():
        for factor in candidates:
            for horizon in HORIZONS:
                row = development_score(frame, factor, horizon)
                row["family"] = family
                rows.append(row)
    scan = pd.DataFrame(rows)
    winners = scan.sort_values(["family", "development_score"], ascending=[True, False]).groupby("family", as_index=False).head(1).reset_index(drop=True)
    for row in winners.itertuples(index=False):
        frame[f"oriented_{row.family}"] = int(row.orientation) * frame[row.factor]
    # Two frozen cross-family consensus identities, selected only from family winners.
    ranked = winners.sort_values("development_score", ascending=False)
    top4 = ranked.head(4)["family"].tolist()
    propagation = [name for name in ["riemann_transmission", "impedance_release", "characteristic_reattachment", "source_characteristics"] if name in set(winners["family"])]
    composites = []
    for name, families in [("characteristic_top4_consensus", top4), ("source_riemann_consensus", propagation)]:
        columns = [f"oriented_{family}" for family in families]
        frame[name] = frame[columns].median(axis=1, skipna=True)
        best: dict[str, Any] | None = None
        for horizon in HORIZONS:
            row = development_score(frame, name, horizon)
            row["family"] = name
            row["members"] = json.dumps(families)
            if best is None or row["development_score"] > best["development_score"]:
                best = row
        if best is not None:
            composites.append(best)
    frozen = pd.concat([winners, pd.DataFrame(composites)], ignore_index=True, sort=False)
    return scan, frozen


def fit_matrices(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    medians = train[features].median(axis=0, skipna=True)
    usable = [column for column in features if np.isfinite(medians[column])]
    if not usable:
        return np.empty((len(train), 0)), np.empty((len(test), 0)), []
    x = train[usable].fillna(medians[usable]).to_numpy(float)
    xt = test[usable].fillna(medians[usable]).to_numpy(float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > EPS, std, 1.0)
    return (x - mean) / std, (xt - mean) / std, usable


def model_and_residual(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str, base: list[str]) -> dict[str, Any]:
    required = [target, factor]
    tr = train.loc[np.isfinite(train[target]) & np.isfinite(train[factor])].copy()
    te = test.loc[np.isfinite(test[target]) & np.isfinite(test[factor])].copy()
    if len(tr) > 300_000:
        tr = tr.iloc[np.linspace(0, len(tr) - 1, 300_000, dtype=int)]
    if len(tr) < 10_000 or len(te) < 2_000:
        return {"mse_gain_pct": float("nan"), "residual_rankic": float("nan"), "reconstruction_r2": float("nan"), "max_parent_corr": float("nan"), "max_parent_feature": None}
    xb, xbt, usable = fit_matrices(tr, te, base)
    factor_median = float(tr[factor].median())
    xf_raw = np.column_stack([tr[usable].fillna(tr[usable].median()).to_numpy(float), tr[factor].fillna(factor_median).to_numpy(float)])
    xft_raw = np.column_stack([te[usable].fillna(tr[usable].median()).to_numpy(float), te[factor].fillna(factor_median).to_numpy(float)])
    mean = xf_raw.mean(axis=0)
    std = np.where(xf_raw.std(axis=0) > EPS, xf_raw.std(axis=0), 1.0)
    xf = (xf_raw - mean) / std
    xft = (xft_raw - mean) / std
    y = tr[target].to_numpy(float)
    yt = te[target].to_numpy(float)
    base_model = Ridge(alpha=20.0).fit(xb, y)
    full_model = Ridge(alpha=20.0).fit(xf, y)
    pb = base_model.predict(xbt)
    pf = full_model.predict(xft)
    mse_b = mean_squared_error(yt, pb)
    mse_f = mean_squared_error(yt, pf)

    recon = Ridge(alpha=20.0).fit(xb, tr[factor].to_numpy(float))
    predicted_factor = recon.predict(xbt)
    residual = te[factor].to_numpy(float) - predicted_factor
    temp = te[["date", target]].copy()
    temp["residual"] = residual
    residual_ic = daily_rankic(temp, "residual", target, minimum=30).mean()
    correlations = {column: abs(safe_spearman(te[factor], te[column], minimum=1000)) for column in usable}
    correlations = {key: value for key, value in correlations.items() if np.isfinite(value)}
    max_feature = max(correlations, key=correlations.get) if correlations else None
    return {
        "mse_gain_pct": float(100.0 * (mse_b - mse_f) / (mse_b + EPS)),
        "residual_rankic": float(residual_ic),
        "reconstruction_r2": float(r2_score(te[factor], predicted_factor)),
        "max_parent_corr": float(correlations[max_feature]) if max_feature else float("nan"),
        "max_parent_feature": max_feature,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def cost_diagnostic(frame: pd.DataFrame, factor: str, target_raw: str, horizon: int, cost_bp: float = 5.0) -> dict[str, float]:
    work = frame[["date", factor, target_raw]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return {"annual_log_return": float("nan"), "sharpe": float("nan"), "positive_offset_share": float("nan"), "offsets": 0}
    unique_dates = pd.Index(sorted(work["date"].unique()))
    date_number = {date: index for index, date in enumerate(unique_dates)}
    work["date_number"] = work["date"].map(date_number)
    offset_returns: list[float] = []
    offset_sharpes: list[float] = []
    for offset in range(max(1, horizon)):
        sample = work.loc[work["date_number"].mod(max(1, horizon)).eq(offset)]
        spreads: list[float] = []
        for _, group in sample.groupby("date", sort=False):
            if len(group) < 40:
                continue
            low, high = group[factor].quantile([0.10, 0.90])
            long_return = group.loc[group[factor] >= high, target_raw].mean()
            short_return = group.loc[group[factor] <= low, target_raw].mean()
            if np.isfinite(long_return) and np.isfinite(short_return):
                spreads.append(float(long_return - short_return - 4.0 * cost_bp * 1e-4))
        if len(spreads) >= 5:
            array = np.asarray(spreads, dtype=float)
            annual = float(array.mean() * 252.0 / max(1, horizon))
            sharpe = float(array.mean() / (array.std(ddof=0) + EPS) * math.sqrt(252.0 / max(1, horizon)))
            offset_returns.append(annual)
            offset_sharpes.append(sharpe)
    if not offset_returns:
        return {"annual_log_return": float("nan"), "sharpe": float("nan"), "positive_offset_share": float("nan"), "offsets": 0}
    return {
        "annual_log_return": float(np.mean(offset_returns)),
        "sharpe": float(np.mean(offset_sharpes)),
        "positive_offset_share": float(np.mean(np.asarray(offset_returns) > 0.0)),
        "offsets": int(len(offset_returns)),
    }


def yearly_rankic(frame: pd.DataFrame, factor: str, target: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for year, group in frame.groupby(frame["date"].dt.year, sort=True):
        value = daily_rankic(group, factor, target, minimum=30).mean()
        result[str(int(year))] = float(value) if np.isfinite(value) else float("nan")
    return result


def evaluate(frame: pd.DataFrame, factor: str, horizon: int, train_period: str, test_period: str, base: list[str]) -> dict[str, Any]:
    target = f"target{horizon}"
    target_raw = f"target{horizon}_raw"
    train = frame.loc[period_mask(frame, train_period) & frame["bucket"].isin(["high", "secondary"])].copy()
    test = frame.loc[period_mask(frame, test_period) & frame["bucket"].isin(["high", "secondary"])].copy()
    direct = daily_rankic(test, factor, target, minimum=30)
    block = moving_block_bootstrap(direct, block=max(10, horizon), repetitions=2500)
    ts, ts_share, ts_n = time_series_ic(test, factor, target, minimum=60)
    model = model_and_residual(train, test, factor, target, base)
    cost = cost_diagnostic(test, factor, target_raw, horizon, cost_bp=5.0)
    years = yearly_rankic(test, factor, target)
    return {
        "factor": factor,
        "horizon": int(horizon),
        "train_period": train_period,
        "test_period": test_period,
        "cs_rankic": float(direct.mean()),
        "positive_date_share": float((direct > 0).mean()),
        "n_dates": int(direct.notna().sum()),
        "bootstrap_low": block["low"],
        "bootstrap_high": block["high"],
        "bootstrap_p_nonpositive": block["p_nonpositive"],
        "ts_ic": ts,
        "ts_positive_share": ts_share,
        "ts_products": ts_n,
        **model,
        "annual_log_return_5bp": cost["annual_log_return"],
        "sharpe_5bp": cost["sharpe"],
        "positive_offset_share": cost["positive_offset_share"],
        "cost_offsets": cost["offsets"],
        "yearly_rankic": json.dumps(years, sort_keys=True),
    }


def validation_gate(frame: pd.DataFrame) -> pd.Series:
    yearly_ok: list[bool] = []
    for payload in frame["yearly_rankic"]:
        years = json.loads(payload)
        finite = [float(value) for value in years.values() if np.isfinite(float(value))]
        yearly_ok.append(bool(finite) and min(finite) > 0.0)
    return (
        frame["cs_rankic"].gt(0.012)
        & frame["bootstrap_p_nonpositive"].le(0.10)
        & frame["mse_gain_pct"].gt(0.0)
        & frame["residual_rankic"].gt(0.005)
        & frame["max_parent_corr"].lt(0.75)
        & frame["annual_log_return_5bp"].gt(0.0)
        & frame["holm_p"].le(0.10)
        & pd.Series(yearly_ok, index=frame.index)
    )


def main() -> None:
    raw, source_audit = source.load_baostock()
    frame, family_candidates, base_features, graph_audit = build_panel(raw)
    scan, frozen = freeze_identities(frame, family_candidates)
    scan.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)
    frozen.to_csv(OUT / "FROZEN_IDENTITIES.csv", index=False)

    validation_rows: list[dict[str, Any]] = []
    for row in frozen.itertuples(index=False):
        factor = str(row.factor)
        orientation = int(row.orientation)
        oriented = f"__validation_{row.family}"
        frame[oriented] = orientation * frame[factor]
        result = evaluate(frame, oriented, int(row.horizon), "development", "validation", base_features)
        result["family"] = row.family
        result["primitive_factor"] = factor
        result["orientation"] = orientation
        validation_rows.append(result)
        print(f"validation_complete family={row.family} factor={factor} horizon={row.horizon}", flush=True)
    validation = pd.DataFrame(validation_rows)
    validation["holm_p"] = holm_adjust(validation["bootstrap_p_nonpositive"].tolist())
    validation["validation_gate"] = validation_gate(validation)
    validation.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    for row in validation.loc[validation["validation_gate"]].itertuples(index=False):
        factor = str(row.factor)
        result = evaluate(frame, factor, int(row.horizon), "validation", "holdout", base_features)
        result["family"] = row.family
        result["primitive_factor"] = row.primitive_factor
        result["orientation"] = row.orientation
        result["validation_holm_p"] = row.holm_p
        holdout_rows.append(result)
        print(f"holdout_complete family={row.family} factor={factor}", flush=True)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=list(validation.columns)).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    else:
        holdout.to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)

    promoted: list[dict[str, Any]] = []
    for row in holdout.itertuples(index=False):
        years = json.loads(row.yearly_rankic)
        finite_years = [float(value) for value in years.values() if np.isfinite(float(value))]
        passed = (
            row.cs_rankic > 0.012
            and row.bootstrap_p_nonpositive <= 0.10
            and row.mse_gain_pct > 0.0
            and row.residual_rankic > 0.005
            and row.max_parent_corr < 0.75
            and row.annual_log_return_5bp > 0.0
            and row.positive_offset_share >= 0.60
            and bool(finite_years)
            and min(finite_years) > 0.0
        )
        if passed:
            promoted.append({
                "family": row.family,
                "factor": row.factor,
                "primitive_factor": row.primitive_factor,
                "orientation": int(row.orientation),
                "horizon": int(row.horizon),
                "validation_holm_p": float(row.validation_holm_p),
                "holdout_rankic": float(row.cs_rankic),
                "holdout_mse_gain_pct": float(row.mse_gain_pct),
                "holdout_residual_rankic": float(row.residual_rankic),
                "holdout_max_parent_corr": float(row.max_parent_corr),
                "holdout_annual_log_return_5bp": float(row.annual_log_return_5bp),
                "holdout_positive_offset_share": float(row.positive_offset_share),
            })

    audit = {
        "source": source_audit,
        "graph": graph_audit,
        "panel_rows": int(len(frame)),
        "codes": int(frame["code"].nunique()),
        "dates": int(frame["date"].nunique()),
        "min_date": str(frame["date"].min().date()),
        "max_date": str(frame["date"].max().date()),
        "base_features": base_features,
        "base_feature_count": len(base_features),
        "families": {family: len(candidates) for family, candidates in family_candidates.items()},
        "candidate_identities": int(sum(len(values) for values in family_candidates.values()) * len(HORIZONS)),
        "availability": {
            "factor_time": "after completed daily close",
            "target_entry": "next completed session open",
            "rolling_states": "strict prior t-1 and earlier",
            "graph": "frozen from development period only",
            "universe": "Baostock quarterly historical CSI300 snapshots, forward-filled until next snapshot",
        },
    }
    write_json(OUT / "DATA_AUDIT.json", audit)
    write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "candidate_families": len(family_candidates),
        "candidate_state_columns": sum(len(values) for values in family_candidates.values()),
        "development_tests": int(len(scan)),
        "frozen_identities": int(len(frozen)),
        "validation_gate_passed": int(validation["validation_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "promoted": promoted,
        "decision": "promote_only_if_PROMOTED_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = [
        "# A-share characteristic-network fluid research round 15",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen identities",
        "",
        frozen.to_markdown(index=False),
        "",
        "## Validation",
        "",
        validation.to_markdown(index=False),
        "",
        "## Historical holdout",
        "",
        holdout.to_markdown(index=False) if not holdout.empty else "Historical holdout remained closed because no frozen identity passed every validation gate.",
        "",
        "The experiment estimates the stock network and all orientations only in development. A factor is promoted only if direct IC, moving-block inference, strong-parent Ridge increment, parent residual, correlation, annual direction and a conservative 5bp non-overlap account all pass in validation and the conditionally opened holdout.",
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
