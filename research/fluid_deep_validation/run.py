from __future__ import annotations

import io
import json
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SEED = 20261118
EPS = 1.0e-12
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "BCHUSDT", "LINKUSDT", "ETCUSDT",
    "XLMUSDT", "DOTUSDT", "SOLUSDT",
]
MONTHS = list(pd.period_range("2021-01", "2026-10", freq="M"))
HORIZONS = [1, 4, 12, 24, 72, 120, 480]
BASE_FEATURES = [
    "u", "mom4", "mom12", "mom24", "mom72", "mom168",
    "rv24_z", "rv72_z", "range24_z", "qz", "imb6_z", "imb24_z",
    "trades_z", "skew24_z", "kurt24_z", "peclet_proxy",
    "cascade_coherence_proxy",
]
FAMILIES = {
    "boundary_event": [
        "bl_event_age_reversal", "bl_reattachment_impulse",
        "bl_separation_hazard", "bl_reattach_persistence",
    ],
    "subgrid_flux": [
        "sgs_flux_persistence", "sgs_flux_gradient",
        "sgs_flux_reversal", "sgs_forward_cascade",
    ],
    "reynolds_orthogonal": [
        "reynolds_production_innovation", "reynolds_balance_innovation",
        "reynolds_dissipation_release",
    ],
    "lagrangian_geometry": [
        "lagrangian_stretch_release", "lagrangian_stretch_directional",
        "state_helicity", "helicity_trend_coupling",
    ],
    "shock_cavitation": [
        "shock_exhaustion", "shock_supported_transport",
        "cavitation_refill_continuation", "cavitation_refill_reversal",
    ],
    "circulation_shedding": [
        "circulation_release", "circulation_transport", "karman_alternation",
    ],
    "continuity_control_volume": [
        "continuity_source_sink", "continuity_absorption",
    ],
}
CANDIDATES = [item for values in FAMILIES.values() for item in values]


def causal_z(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    prior = s.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((s - mean) / (std + EPS)).clip(-12.0, 12.0)


def norm_momentum(u: pd.Series, window: int) -> pd.Series:
    numerator = u.rolling(window, min_periods=window).sum()
    denominator = np.sqrt(u.pow(2).rolling(window, min_periods=window).sum())
    return (numerator / (denominator + EPS)).clip(-12.0, 12.0)


def causal_rls_residual(
    target: pd.Series,
    design: np.ndarray,
    forgetting: float = 0.997,
    ridge: float = 10.0,
) -> pd.Series:
    y = target.to_numpy(dtype=float)
    x = np.asarray(design, dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    covariance = np.eye(x.shape[1], dtype=float) / ridge
    residual = np.full(len(y), np.nan, dtype=float)
    for index in range(len(y)):
        row = x[index]
        value = y[index]
        if not (np.isfinite(value) and np.all(np.isfinite(row))):
            continue
        prediction = float(row @ beta)
        error = value - prediction
        residual[index] = error
        projected = covariance @ row
        gain = projected / (forgetting + row @ projected)
        beta += gain * error
        covariance = (covariance - np.outer(gain, row) @ covariance) / forgetting
        if index % 2000 == 0:
            covariance = (covariance + covariance.T) / 2.0
    return pd.Series(residual, index=target.index)


def event_age(event: pd.Series, cap: int = 72) -> pd.Series:
    values = event.fillna(False).to_numpy(dtype=bool)
    output = np.full(len(values), np.nan, dtype=float)
    age = cap
    observed = False
    for index, flag in enumerate(values):
        if flag:
            age = 0
            observed = True
        elif observed:
            age = min(cap, age + 1)
        if observed:
            output[index] = age
    return pd.Series(output, index=event.index)


def download_month(symbol: str, period: pd.Period) -> tuple[pd.DataFrame | None, str | None]:
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/1h/{symbol}-1h-{period.year:04d}-{period.month:02d}.zip"
    )
    headers = {"User-Agent": "fluid-deep-validation/20261118"}
    error: str | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=40, headers=headers)
            if response.status_code == 404:
                return None, "404"
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = [name for name in archive.namelist() if name.endswith(".csv")]
                if not names:
                    return None, "missing_csv"
                frame = pd.read_csv(archive.open(names[0]), header=None, low_memory=False)
            frame = frame.iloc[:, :12]
            frame.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ]
            frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
            frame = frame.loc[frame["open_time"].notna()].copy()
            for column in [
                "open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_quote",
            ]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["time"] = pd.to_datetime(
                frame["open_time"].astype("int64"), unit="ms", utc=True
            )
            frame["symbol"] = symbol
            keep = [
                "time", "symbol", "open", "high", "low", "close",
                "volume", "quote_volume", "trades", "taker_buy_quote",
            ]
            frame = frame[keep].replace([np.inf, -np.inf], np.nan)
            frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
            frame = frame.loc[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
            return frame, None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return None, error


def load_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = [(symbol, period) for symbol in SYMBOLS for period in MONTHS]
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        jobs = {
            pool.submit(download_month, symbol, period): (symbol, period)
            for symbol, period in tasks
        }
        for count, future in enumerate(as_completed(jobs), start=1):
            symbol, period = jobs[future]
            frame, error = future.result()
            if frame is not None and not frame.empty:
                frames.append(frame)
            elif error != "404":
                errors.append({"symbol": symbol, "month": str(period), "error": str(error)})
            if count % 100 == 0:
                print(f"download_progress={count}/{len(tasks)} parts={len(frames)}", flush=True)
    if not frames:
        raise RuntimeError("No public Binance data downloaded")
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["symbol", "time"], keep="last")
    data = data.sort_values(["symbol", "time"]).reset_index(drop=True)
    metadata = {
        "requested_symbols": SYMBOLS,
        "requested_months": [str(MONTHS[0]), str(MONTHS[-1])],
        "rows": int(len(data)),
        "min_time": str(data["time"].min()),
        "max_time": str(data["time"].max()),
        "errors": errors[:100],
    }
    return data, metadata


def compute_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("time").reset_index(drop=True).copy()
    log_close = np.log(g["close"])
    returns = log_close.diff()
    prior_scale = np.sqrt(
        returns.shift(1).pow(2).ewm(span=48, min_periods=24, adjust=False).mean()
    )
    u = (returns / (prior_scale + EPS)).clip(-12.0, 12.0)
    qz = causal_z(np.log1p(g["quote_volume"].clip(lower=0.0)), 168, 72)
    imbalance = (
        2.0 * g["taker_buy_quote"] / (g["quote_volume"] + EPS) - 1.0
    ).clip(-1.0, 1.0)
    imbalance_z = causal_z(imbalance, 168, 72)
    trades_z = causal_z(np.log1p(g["trades"].clip(lower=0.0)), 168, 72)
    intrabar_range = np.log(g["high"] / g["low"]).clip(lower=0.0)

    g["u"] = u
    g["qz"] = qz
    g["trades_z"] = trades_z
    g["rv24_price"] = np.sqrt(returns.pow(2).rolling(24, min_periods=18).mean())
    for window in [4, 12, 24, 72, 168]:
        g[f"mom{window}"] = norm_momentum(u, window)
    rv24 = np.sqrt(u.pow(2).rolling(24, min_periods=18).mean())
    rv72 = np.sqrt(u.pow(2).rolling(72, min_periods=54).mean())
    g["rv24_z"] = causal_z(np.log(rv24 + EPS), 336, 120)
    g["rv72_z"] = causal_z(np.log(rv72 + EPS), 336, 120)
    g["range24_z"] = causal_z(
        intrabar_range.rolling(24, min_periods=18).mean(), 336, 120
    )
    g["imb6_z"] = causal_z(imbalance.rolling(6, min_periods=4).mean(), 336, 120)
    g["imb24_z"] = causal_z(
        imbalance.rolling(24, min_periods=18).mean(), 336, 120
    )
    g["skew24_z"] = causal_z(u.rolling(24, min_periods=18).skew(), 336, 120)
    g["kurt24_z"] = causal_z(u.rolling(24, min_periods=18).kurt(), 336, 120)

    activity_weight = np.exp(qz.clip(-2.5, 2.5))
    weight_sum = activity_weight.rolling(24, min_periods=18).sum()
    advection = (activity_weight * u).rolling(24, min_periods=18).sum() / (
        weight_sum + EPS
    )
    second = (activity_weight * u.pow(2)).rolling(24, min_periods=18).sum() / (
        weight_sum + EPS
    )
    diffusion = np.sqrt((second - advection.pow(2)).clip(lower=0.0))
    g["peclet_proxy"] = causal_z(advection / (diffusion + EPS), 720, 240)
    coherence_parts = []
    for scale in [1, 2, 4, 8]:
        delta = u.rolling(scale, min_periods=scale).sum()
        lagged = delta.shift(scale)
        numerator = 2.0 * (delta * lagged).rolling(96, min_periods=72).mean()
        denominator = (delta.pow(2) + lagged.pow(2)).rolling(
            96, min_periods=72
        ).mean()
        coherence_parts.append((numerator / (denominator + EPS)).clip(-1.0, 1.0))
    coherence = pd.concat(coherence_parts, axis=1).mean(axis=1)
    g["cascade_coherence_proxy"] = causal_z(-g["mom24"] * coherence, 720, 240)

    direction = np.sign(g["mom24"]).replace(0.0, np.nan)
    raw: dict[str, pd.Series] = {}

    # Boundary layer as an event-clock process rather than a one-period state.
    shear6 = (activity_weight * u).rolling(6, min_periods=4).sum() / (
        (activity_weight * u.abs()).rolling(6, min_periods=4).sum() + EPS
    )
    shear24 = (activity_weight * u).rolling(24, min_periods=18).sum() / (
        (activity_weight * u.abs()).rolling(24, min_periods=18).sum() + EPS
    )
    high48 = g["high"].rolling(48, min_periods=36).max()
    low48 = g["low"].rolling(48, min_periods=36).min()
    range_scale = intrabar_range.rolling(48, min_periods=36).mean()
    distance_up = np.log(high48 / g["close"]).clip(lower=0.0)
    distance_down = np.log(g["close"] / low48).clip(lower=0.0)
    proximity = pd.Series(
        np.where(
            direction >= 0.0,
            np.exp(-distance_up / (range_scale + EPS)),
            np.exp(-distance_down / (range_scale + EPS)),
        ),
        index=g.index,
    ).clip(0.0, 1.0)
    adverse_shear = (-direction * shear6).clip(lower=0.0)
    shear_drop = (direction * (shear24 - shear6)).clip(lower=0.0)
    separation = proximity * adverse_shear * np.tanh(shear_drop)
    threshold = separation.shift(1).rolling(720, min_periods=240).quantile(0.8)
    separation_event = (separation > threshold) & (separation > 0.0)
    age = event_age(separation_event, 72)
    memory = separation.ewm(halflife=12, min_periods=4, adjust=False).mean()
    aligned_shear = (direction * shear6).clip(lower=0.0)
    shear_acceleration = (direction * shear6.diff()).clip(lower=0.0)
    reattachment = (
        direction * memory.shift(1) * aligned_shear * shear_acceleration * proximity
    )
    raw["bl_event_age_reversal"] = -direction * memory * np.tanh(age / 12.0)
    raw["bl_reattachment_impulse"] = reattachment
    raw["bl_separation_hazard"] = -direction * separation * (
        1.0 + np.tanh(age / 24.0)
    )
    raw["bl_reattach_persistence"] = (
        direction * reattachment.rolling(6, min_periods=3).sum()
    )

    # Germano-style coarse-grained subgrid energy transfer.
    fluxes: dict[int, pd.Series] = {}
    for scale in [4, 12, 24, 48]:
        mean_flow = u.rolling(scale, min_periods=max(3, int(0.75 * scale))).mean()
        stress = (
            u.pow(2).rolling(scale, min_periods=max(3, int(0.75 * scale))).mean()
            - mean_flow.pow(2)
        )
        fluxes[scale] = -stress * mean_flow.diff()
    flux_frame = pd.DataFrame({f"pi{scale}": value for scale, value in fluxes.items()})
    flux_z = {scale: causal_z(value, 720, 240) for scale, value in fluxes.items()}
    consistency = flux_frame.apply(np.sign).mean(axis=1)
    magnitude = flux_frame.abs().mean(axis=1)
    raw["sgs_flux_persistence"] = (
        direction
        * consistency
        * np.tanh(causal_z(magnitude, 720, 240).clip(lower=0.0))
    )
    raw["sgs_flux_gradient"] = direction * (flux_z[4] - flux_z[48])
    raw["sgs_flux_reversal"] = -direction * (flux_z[4] - flux_z[24]).diff()
    raw["sgs_forward_cascade"] = direction * pd.concat(
        [flux_z[4], flux_z[12], flux_z[24]], axis=1
    ).mean(axis=1)

    # Reynolds production innovations orthogonalized causally to standard states.
    mean_flow = u.ewm(span=24, min_periods=18, adjust=False).mean()
    u_prime = u - mean_flow
    mean_activity = qz.ewm(span=24, min_periods=18, adjust=False).mean()
    q_prime = qz - mean_activity
    stress = (u_prime * q_prime).rolling(24, min_periods=18).mean()
    production = -stress * mean_flow.diff()
    dissipation = u_prime.diff().pow(2).rolling(24, min_periods=18).mean()
    production_z = causal_z(production, 336, 120)
    dissipation_z = causal_z(np.log1p(dissipation), 336, 120)
    controls = np.column_stack([
        np.ones(len(g)), g["mom24"], qz, g["rv24_z"], trades_z, g["imb24_z"],
    ])
    production_residual = causal_rls_residual(production_z, controls)
    balance_residual = causal_rls_residual(production_z - dissipation_z, controls)
    raw["reynolds_production_innovation"] = (
        np.sign(mean_flow) * np.tanh(production_residual)
    )
    raw["reynolds_balance_innovation"] = (
        np.sign(mean_flow) * np.tanh(balance_residual)
    )
    raw["reynolds_dissipation_release"] = (
        -np.sign(mean_flow)
        * np.tanh(dissipation_z)
        * np.tanh(production_residual.abs())
    )

    # Lagrangian stretching and signed state-space helicity.
    vx = u
    vy = qz.diff().clip(-6.0, 6.0)
    vz = imbalance_z.diff().clip(-6.0, 6.0)
    displacement = np.sqrt(vx.diff().pow(2) + vy.diff().pow(2) + vz.diff().pow(2))
    stretch = np.log((displacement + 1.0e-4) / (displacement.shift(1) + 1.0e-4)).clip(-8.0, 8.0)
    stretch_z = causal_z(stretch, 720, 240)
    alignment = (
        vx.diff() * vx + vy.diff() * vy + vz.diff() * vz
    ) / (
        displacement * np.sqrt(vx.pow(2) + vy.pow(2) + vz.pow(2)) + EPS
    )
    raw["lagrangian_stretch_release"] = (
        -direction * stretch_z.clip(lower=0.0) * alignment.abs()
    )
    raw["lagrangian_stretch_directional"] = direction * stretch_z * alignment
    current = np.column_stack([vx, vy, vz])
    lag1 = np.roll(current, 1, axis=0)
    lag2 = np.roll(current, 2, axis=0)
    triple = np.einsum("ij,ij->i", current, np.cross(lag1, lag2))
    denominator = (
        np.linalg.norm(current, axis=1)
        * np.linalg.norm(lag1, axis=1)
        * np.linalg.norm(lag2, axis=1)
        + EPS
    )
    helicity = pd.Series(np.clip(triple / denominator, -1.0, 1.0), index=g.index)
    helicity_state = helicity.rolling(12, min_periods=8).mean()
    raw["state_helicity"] = helicity_state
    raw["helicity_trend_coupling"] = g["mom24"] * helicity_state

    # Shock compression and cavitation-refill paths.
    local_mean = u.rolling(24, min_periods=18).mean()
    sound_speed = np.sqrt((u - local_mean).pow(2).rolling(24, min_periods=18).mean())
    mach = (local_mean.abs() / (sound_speed + EPS)).clip(0.0, 10.0)
    mach_z = causal_z(mach, 720, 240)
    compression = (
        mach_z.clip(lower=0.0)
        * g["range24_z"].clip(lower=0.0)
        * qz.clip(lower=0.0)
    )
    flow_support = (
        direction * imbalance.rolling(6, min_periods=4).mean()
    ).clip(lower=0.0)
    raw["shock_exhaustion"] = (
        -direction * compression * (1.0 - flow_support.clip(0.0, 1.0))
    )
    raw["shock_supported_transport"] = direction * compression * flow_support
    cavitation = (
        (-qz).clip(lower=0.0)
        * (-trades_z).clip(lower=0.0)
        * causal_z(u.abs(), 336, 120).clip(lower=0.0)
    )
    refill = qz.diff().clip(lower=0.0) * trades_z.diff().clip(lower=0.0)
    raw["cavitation_refill_continuation"] = direction * cavitation.shift(6) * refill
    raw["cavitation_refill_reversal"] = -direction * cavitation.shift(6) * refill

    # State-space circulation and Karman-like alternating shedding.
    circulation = u * qz.diff() - qz * u.diff()
    circulation_z = causal_z(circulation, 720, 240)
    flips = (np.sign(circulation) != np.sign(circulation.shift(1))).astype(float)
    amplitude = circulation_z.abs().rolling(12, min_periods=8).mean()
    alternation = flips.rolling(12, min_periods=8).mean()
    lag_correlation = circulation_z.rolling(48, min_periods=36).corr(
        circulation_z.shift(6)
    )
    raw["circulation_release"] = -direction * amplitude * alternation
    raw["circulation_transport"] = (
        direction * circulation_z.rolling(6, min_periods=4).mean()
    )
    raw["karman_alternation"] = (
        -direction * lag_correlation.clip(upper=0.0).abs() * amplitude
    )

    # A control-volume continuity source/sink residual.
    density = np.exp(qz.clip(-3.0, 3.0))
    mass_flux = density * u
    continuity = qz.diff() + mass_flux.diff()
    continuity_z = causal_z(continuity, 720, 240)
    raw["continuity_source_sink"] = -direction * continuity_z
    raw["continuity_absorption"] = (
        direction * (-continuity_z).clip(lower=0.0) * flow_support
    )

    for name, values in raw.items():
        g[name] = causal_z(
            pd.Series(values, index=g.index).replace([np.inf, -np.inf], np.nan),
            720,
            240,
        ).astype("float32")
    for column in BASE_FEATURES + ["rv24_price"]:
        g[column] = g[column].astype("float32")
    for horizon in HORIZONS:
        g[f"y_{horizon}"] = np.log(
            g["open"].shift(-(horizon + 1)) / g["open"].shift(-1)
        ).astype("float32")
    keep = [
        "time", "symbol", "rv24_price", *BASE_FEATURES, *CANDIDATES,
        *[f"y_{horizon}" for horizon in HORIZONS],
    ]
    return g[keep]


def evaluation_rows(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    mask = panel["time"] >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        mask &= panel["time"] < pd.Timestamp(end, tz="UTC")
    epoch_hours = panel["time"].astype("int64") // 3_600_000_000_000
    mask &= epoch_hours % horizon == 0
    columns = ["time", "symbol", factor, f"y_{horizon}", "rv24_price", *BASE_FEATURES]
    result = panel.loc[mask, columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return result.rename(columns={factor: "factor", f"y_{horizon}": "y"})


def mean_symbol_spearman(
    frame: pd.DataFrame,
    x_column: str = "factor",
    y_column: str = "y",
    min_observations: int = 24,
) -> float:
    values = []
    for _, group in frame.groupby("symbol"):
        subset = group[[x_column, y_column]].dropna()
        if (
            len(subset) < min_observations
            or subset[x_column].nunique() < 3
            or subset[y_column].nunique() < 3
        ):
            continue
        value = spearmanr(subset[x_column], subset[y_column]).statistic
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def contribution_series(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.Series:
    work = frame[["time", "symbol", x_column, y_column]].dropna().copy()
    work["xr"] = work.groupby("symbol")[x_column].rank(method="average", pct=True)
    work["yr"] = work.groupby("symbol")[y_column].rank(method="average", pct=True)
    for column in ["xr", "yr"]:
        mean = work.groupby("symbol")[column].transform("mean")
        std = work.groupby("symbol")[column].transform(lambda values: values.std(ddof=0))
        work[f"z_{column}"] = (work[column] - mean) / (std + EPS)
    work["contribution"] = work["z_xr"] * work["z_yr"]
    return work.groupby("time")["contribution"].mean().sort_index()


def block_inference(series: pd.Series, horizon: int, repetitions: int = 1600) -> dict[str, Any]:
    values = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(values) < 12:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_one_sided": None, "blocks": 0}
    block_length = max(1, int(round(168 / max(horizon, 1))))
    block_length = min(block_length, max(1, len(values) // 8))
    block_count = len(values) // block_length
    blocks = values[: block_count * block_length].reshape(block_count, block_length).mean(axis=1)
    observed = float(blocks.mean())
    rng = np.random.default_rng(SEED + horizon + len(values))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, block_count))
    null_means = (signs * blocks[None, :]).mean(axis=1)
    p_value = float((1.0 + np.sum(null_means >= observed)) / (repetitions + 1.0))
    indices = rng.integers(0, block_count, size=(repetitions, block_count))
    bootstrap = blocks[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_one_sided": p_value,
        "blocks": int(block_count),
        "block_length": int(block_length),
    }


def design_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    symbols: list[str],
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = frame[columns].to_numpy(dtype=np.float64)
    if mean is None:
        mean = np.nanmean(numeric, axis=0)
    if std is None:
        std = np.nanstd(numeric, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    numeric = np.nan_to_num((numeric - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    codes = pd.Categorical(frame["symbol"], categories=symbols).codes
    one_hot = np.zeros((len(frame), len(symbols)), dtype=np.float64)
    valid = codes >= 0
    one_hot[np.arange(len(frame))[valid], codes[valid]] = 1.0
    return np.concatenate([numeric, one_hot], axis=1), mean, std


def ridge_increment(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> dict[str, Any]:
    symbols = sorted(set(train["symbol"]) | set(test["symbol"]))
    base_train, base_mean, base_std = design_matrix(train, BASE_FEATURES, symbols)
    base_test, _, _ = design_matrix(test, BASE_FEATURES, symbols, base_mean, base_std)
    enhanced_train, enhanced_mean, enhanced_std = design_matrix(
        train, [*BASE_FEATURES, "factor"], symbols
    )
    enhanced_test, _, _ = design_matrix(
        test, [*BASE_FEATURES, "factor"], symbols, enhanced_mean, enhanced_std
    )
    y_train = train["y"].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)
    alpha = max(1.0, 0.001 * len(train))
    base_model = Ridge(alpha=alpha).fit(base_train, y_train)
    enhanced_model = Ridge(alpha=alpha).fit(enhanced_train, y_train)
    base_prediction = base_model.predict(base_test)
    enhanced_prediction = enhanced_model.predict(enhanced_test)
    base_loss = (y_test - base_prediction) ** 2
    enhanced_loss = (y_test - enhanced_prediction) ** 2
    loss_difference = pd.DataFrame({
        "time": test["time"].to_numpy(),
        "difference": base_loss - enhanced_loss,
    }).groupby("time")["difference"].mean()
    residual_frame = test[["time", "symbol", "factor"]].copy()
    residual_frame["residual"] = y_test - base_prediction
    return {
        "mse_base": float(base_loss.mean()),
        "mse_enhanced": float(enhanced_loss.mean()),
        "relative_improvement": float(
            (base_loss.mean() - enhanced_loss.mean()) / (base_loss.mean() + EPS)
        ),
        "loss_difference_inference": block_inference(loss_difference, horizon),
        "residual_spearman": mean_symbol_spearman(
            residual_frame, "factor", "residual", 20
        ),
        "alpha": float(alpha),
    }


def hgb_increment(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> dict[str, Any]:
    symbols = sorted(set(train["symbol"]) | set(test["symbol"]))
    if len(train) > 160_000:
        train = train.iloc[np.linspace(0, len(train) - 1, 160_000, dtype=int)].copy()
    base_train, base_mean, base_std = design_matrix(train, BASE_FEATURES, symbols)
    base_test, _, _ = design_matrix(test, BASE_FEATURES, symbols, base_mean, base_std)
    enhanced_train, enhanced_mean, enhanced_std = design_matrix(
        train, [*BASE_FEATURES, "factor"], symbols
    )
    enhanced_test, _, _ = design_matrix(
        test, [*BASE_FEATURES, "factor"], symbols, enhanced_mean, enhanced_std
    )
    y_train = train["y"].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)
    parameters = dict(
        loss="squared_error", learning_rate=0.05, max_iter=160,
        max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=10.0,
        early_stopping=True, validation_fraction=0.15, random_state=SEED,
    )
    base_model = HistGradientBoostingRegressor(**parameters).fit(base_train, y_train)
    enhanced_model = HistGradientBoostingRegressor(**parameters).fit(
        enhanced_train, y_train
    )
    base_loss = (y_test - base_model.predict(base_test)) ** 2
    enhanced_loss = (y_test - enhanced_model.predict(enhanced_test)) ** 2
    loss_difference = pd.DataFrame({
        "time": test["time"].to_numpy(),
        "difference": base_loss - enhanced_loss,
    }).groupby("time")["difference"].mean()
    return {
        "mse_base": float(base_loss.mean()),
        "mse_enhanced": float(enhanced_loss.mean()),
        "relative_improvement": float(
            (base_loss.mean() - enhanced_loss.mean()) / (base_loss.mean() + EPS)
        ),
        "loss_difference_inference": block_inference(loss_difference, horizon),
        "fit_rows": int(len(train)),
    }


def strategy_metrics(frame: pd.DataFrame, horizon: int, cost_bps: float) -> dict[str, Any]:
    signal = frame.pivot(index="time", columns="symbol", values="factor").sort_index()
    returns = frame.pivot(index="time", columns="symbol", values="y").reindex(
        index=signal.index, columns=signal.columns
    )
    risk = frame.pivot(index="time", columns="symbol", values="rv24_price").reindex(
        index=signal.index, columns=signal.columns
    )
    s = signal.to_numpy(dtype=float)
    y = returns.to_numpy(dtype=float)
    v = risk.to_numpy(dtype=float)
    valid = np.isfinite(s) & np.isfinite(y) & np.isfinite(v) & (v > 0.0)
    raw = np.zeros_like(s)
    raw[valid] = np.tanh(s[valid] / 2.0) / np.maximum(v[valid], 1.0e-6)
    gross_exposure = np.sum(np.abs(raw), axis=1)
    positions = np.divide(
        raw,
        gross_exposure[:, None],
        out=np.zeros_like(raw),
        where=gross_exposure[:, None] > 0.0,
    )
    gross = np.nansum(np.where(valid, positions * y, 0.0), axis=1)
    previous = np.vstack([np.zeros((1, positions.shape[1])), positions[:-1]])
    turnover = np.sum(np.abs(positions - previous), axis=1)
    net = gross - cost_bps * 1.0e-4 * turnover
    frequency = 8760.0 / horizon
    wealth = np.exp(np.cumsum(net))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    yearly = pd.Series(net, index=signal.index).groupby(signal.index.year).sum()
    return {
        "observations": int(len(net)),
        "gross_annualized_log_return": float(np.mean(gross) * frequency),
        "net_annualized_log_return": float(np.mean(net) * frequency),
        "gross_sharpe": float(np.mean(gross) / (np.std(gross) + EPS) * math.sqrt(frequency)),
        "net_sharpe": float(np.mean(net) / (np.std(net) + EPS) * math.sqrt(frequency)),
        "max_drawdown": float(np.min(drawdown)),
        "mean_turnover": float(np.mean(turnover)),
        "positive_years": int((yearly > 0.0).sum()),
        "year_count": int(len(yearly)),
        "yearly_net_log_return": {
            str(int(year)): float(value) for year, value in yearly.items()
        },
    }


def max_baseline_correlation(frame: pd.DataFrame) -> dict[str, Any]:
    values = {}
    for column in BASE_FEATURES:
        working = frame[["symbol", "factor", column]].rename(columns={column: "control"})
        values[column] = mean_symbol_spearman(working, "factor", "control", 20)
    valid = {key: value for key, value in values.items() if np.isfinite(value)}
    feature = max(valid, key=lambda key: abs(valid[key]))
    return {
        "max_abs": float(abs(valid[feature])),
        "feature": feature,
        "signed": float(valid[feature]),
        "all": values,
    }


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(
        [(key, value) for key, value in values.items() if np.isfinite(value)],
        key=lambda item: item[1],
    )
    result = {key: float("nan") for key in values}
    running = 0.0
    count = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        result[key] = float(running)
    return result


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def main() -> None:
    output = Path("research_outputs/fluid_deep_validation")
    output.mkdir(parents=True, exist_ok=True)
    raw, data_metadata = load_data()
    parts = []
    for symbol, group in raw.groupby("symbol", sort=True):
        print(f"feature_engineering={symbol} rows={len(group)}", flush=True)
        parts.append(compute_symbol(group))
    panel = pd.concat(parts, ignore_index=True)
    coverage = panel.groupby("symbol")["time"].agg(["min", "max", "count"])
    accepted = coverage.loc[
        (coverage["count"] >= 45_000)
        & (coverage["min"] <= pd.Timestamp("2021-01-02", tz="UTC"))
        & (coverage["max"] >= pd.Timestamp("2026-09-30", tz="UTC"))
    ].index.tolist()
    panel = panel.loc[panel["symbol"].isin(accepted)].copy()
    print(f"accepted_symbols={accepted} rows={len(panel)}", flush=True)

    selection_ledger: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for family, variants in FAMILIES.items():
        best = None
        for variant in variants:
            for horizon in HORIZONS:
                frame = evaluation_rows(
                    panel, variant, horizon, "2021-01-01", "2023-01-01"
                )
                if len(frame) < 500:
                    continue
                ic = mean_symbol_spearman(frame)
                score = abs(ic) * math.sqrt(min(len(frame), 50_000) / 10_000.0)
                row = {
                    "family": family,
                    "variant": variant,
                    "horizon_hours": horizon,
                    "development_ic": ic,
                    "development_rows": int(len(frame)),
                    "selection_score": score,
                }
                selection_ledger.append(row)
                if best is None or score > best["selection_score"]:
                    best = row
        if best is not None:
            selected[family] = {
                "variant": best["variant"],
                "horizon_hours": int(best["horizon_hours"]),
                "sign": 1.0 if best["development_ic"] >= 0.0 else -1.0,
                "development_ic_after_sign": abs(float(best["development_ic"])),
                "development_rows": int(best["development_rows"]),
            }
            print(f"selected={family}:{selected[family]}", flush=True)

    validation: dict[str, Any] = {}
    finalists: list[str] = []
    for family, config in selected.items():
        variant = config["variant"]
        horizon = config["horizon_hours"]
        train = evaluation_rows(panel, variant, horizon, "2021-01-01", "2023-01-01")
        test = evaluation_rows(panel, variant, horizon, "2023-01-01", "2025-01-01")
        train["factor"] *= config["sign"]
        test["factor"] *= config["sign"]
        ic = mean_symbol_spearman(test)
        rank_inference = block_inference(contribution_series(test, "factor", "y"), horizon)
        ridge = ridge_increment(train, test, horizon)
        cost2 = strategy_metrics(test, horizon, 2.0)
        cost5 = strategy_metrics(test, horizon, 5.0)
        correlation = max_baseline_correlation(test)
        passed = bool(
            ic > 0.0
            and rank_inference["p_one_sided"] < 0.15
            and ridge["relative_improvement"] > 0.0
            and ridge["loss_difference_inference"]["mean"] > 0.0
            and cost2["net_annualized_log_return"] > 0.0
        )
        validation[family] = {
            "config": config,
            "rows": int(len(test)),
            "mean_symbol_spearman": ic,
            "rank_contribution_inference": rank_inference,
            "ridge": ridge,
            "strategy_2bps": cost2,
            "strategy_5bps": cost5,
            "baseline_correlation": correlation,
            "passed_to_holdout": passed,
        }
        print(
            f"validation={family} ic={ic:.6f} p={rank_inference['p_one_sided']:.5f} "
            f"ridge={ridge['relative_improvement']:.7f} "
            f"net2={cost2['net_annualized_log_return']:.6f} pass={passed}",
            flush=True,
        )
        if passed:
            finalists.append(family)

    holdout: dict[str, Any] = {}
    for family in finalists:
        config = selected[family]
        variant = config["variant"]
        horizon = config["horizon_hours"]
        train = evaluation_rows(panel, variant, horizon, "2021-01-01", "2025-01-01")
        test = evaluation_rows(panel, variant, horizon, "2025-01-01", None)
        train["factor"] *= config["sign"]
        test["factor"] *= config["sign"]
        ic = mean_symbol_spearman(test)
        rank_inference = block_inference(contribution_series(test, "factor", "y"), horizon)
        ridge = ridge_increment(train, test, horizon)
        hgb = hgb_increment(train, test, horizon)
        cost2 = strategy_metrics(test, horizon, 2.0)
        cost5 = strategy_metrics(test, horizon, 5.0)
        correlation = max_baseline_correlation(test)
        holdout[family] = {
            "config": config,
            "rows": int(len(test)),
            "mean_symbol_spearman": ic,
            "rank_contribution_inference": rank_inference,
            "ridge": ridge,
            "hist_gradient_boosting": hgb,
            "strategy_2bps": cost2,
            "strategy_5bps": cost5,
            "baseline_correlation": correlation,
        }
        print(
            f"holdout={family} ic={ic:.6f} p={rank_inference['p_one_sided']:.5f} "
            f"ridge={ridge['relative_improvement']:.7f} hgb={hgb['relative_improvement']:.7f} "
            f"net5={cost5['net_annualized_log_return']:.6f}",
            flush=True,
        )

    rank_adjusted = holm_adjust({
        family: result["rank_contribution_inference"]["p_one_sided"]
        for family, result in holdout.items()
    })
    ridge_adjusted = holm_adjust({
        family: result["ridge"]["loss_difference_inference"]["p_one_sided"]
        for family, result in holdout.items()
    })
    decisions = {}
    for family, result in holdout.items():
        result["rank_p_holm"] = rank_adjusted[family]
        result["ridge_p_holm"] = ridge_adjusted[family]
        cost5 = result["strategy_5bps"]
        hgb = result["hist_gradient_boosting"]
        years_ok = cost5["positive_years"] >= max(1, cost5["year_count"] - 1)
        independent = bool(
            rank_adjusted[family] < 0.05
            and result["mean_symbol_spearman"] > 0.0
            and ridge_adjusted[family] < 0.10
            and result["ridge"]["relative_improvement"] > 0.0
            and result["ridge"]["residual_spearman"] > 0.0
            and hgb["relative_improvement"] > 0.0
            and hgb["loss_difference_inference"]["p_one_sided"] < 0.20
            and cost5["net_annualized_log_return"] > 0.0
            and cost5["net_sharpe"] > 0.0
            and result["baseline_correlation"]["max_abs"] < 0.75
            and years_ok
        )
        component = bool(
            not independent
            and result["mean_symbol_spearman"] > 0.0
            and result["ridge"]["relative_improvement"] > 0.0
            and result["strategy_2bps"]["net_annualized_log_return"] > 0.0
        )
        decisions[family] = {
            "status": (
                "admit_independent_alpha"
                if independent
                else "admit_default_off_component"
                if component
                else "reject_current_configuration"
            ),
            "independent_alpha": independent,
            "default_enabled": False,
        }

    result = clean({
        "research_id": "fluid_deep_validation_20261118",
        "data": {
            **data_metadata,
            "accepted_symbols": accepted,
            "rows_after_filter": int(len(panel)),
        },
        "splits": {
            "development": "2021-01-01 <= t < 2023-01-01",
            "validation": "2023-01-01 <= t < 2025-01-01",
            "holdout": "2025-01-01 <= t <= 2026-10-31",
            "label": "next-hour open to open after horizon",
            "issuance": "non-overlapping epoch-hour modulo horizon",
        },
        "families": FAMILIES,
        "selection": selected,
        "validation": validation,
        "holdout": holdout,
        "decisions": decisions,
        "independent_alpha_count": int(
            sum(item["independent_alpha"] for item in decisions.values())
        ),
        "component_count": int(
            sum(item["status"] == "admit_default_off_component" for item in decisions.values())
        ),
        "limitations": [
            "Historical public Binance USDT perpetual hourly data.",
            "Time-ordered holdout is not future live evidence.",
            "Kline-level activity is not order-book or participant identity data.",
            "Cost model excludes funding and nonlinear market impact.",
        ],
    })
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    pd.DataFrame(selection_ledger).to_csv(output / "selection_ledger.csv", index=False)
    decision_rows = []
    for family, decision in decisions.items():
        decision_rows.append({"family": family, **decision})
    pd.DataFrame(decision_rows).to_csv(output / "decision_ledger.csv", index=False)
    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    print("RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    main()
