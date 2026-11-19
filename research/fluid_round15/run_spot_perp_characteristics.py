from __future__ import annotations

import io
import json
import math
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

SEED = 20261119
EPS = 1.0e-12
OUT = Path("research_outputs/fluid_spot_perp_round15")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "BCHUSDT", "LINKUSDT", "ETCUSDT",
    "XLMUSDT", "DOTUSDT", "SOLUSDT",
]
MONTHS = list(pd.period_range("2021-01", "2026-09", freq="M"))
HORIZONS = [12, 24, 72, 120, 240, 480]
SOUND_WINDOWS = [72, 168, 336]
STATE_SPANS = [6, 24, 72]

FAMILY_RAW: dict[str, list[str]] = {
    "riemann_transfer": [
        "spot_to_perp_transmission_defect",
        "perp_to_spot_transmission_defect",
        "bidirectional_characteristic_balance",
    ],
    "impedance_release": [
        "impedance_pressure_release",
        "impedance_matched_shear_transport",
        "impedance_energy_absorption",
    ],
    "source_conservation": [
        "source_adjusted_characteristic_residual",
        "source_absorption_release",
        "oi_supported_characteristic_transfer",
    ],
    "wave_action": [
        "wave_action_transfer_defect",
        "wave_action_residual_release",
        "wave_energy_absorption",
    ],
    "shock_rarefaction": [
        "basis_characteristic_shock",
        "basis_characteristic_rarefaction",
        "shock_entropy_release",
    ],
    "characteristic_reattachment": [
        "interlayer_characteristic_gap",
        "interlayer_characteristic_reattachment",
        "incoming_characteristic_release",
    ],
    "interlayer_reynolds": [
        "interlayer_reynolds_stress",
        "interlayer_stress_release",
        "source_supported_stress",
    ],
    "crocco_enthalpy": [
        "interlayer_crocco_residual",
        "interlayer_crocco_release",
        "source_supported_enthalpy_gap",
    ],
}

BASE_FEATURES = [
    "perp_u", "spot_u", "shear", "basis_z", "basis_change_z",
    "mom4", "mom12", "mom24", "mom72", "mom168", "mom480",
    "rv24_z", "rv72_z", "rv168_z", "range_perp_z", "range_spot_z",
    "perp_quote_z", "spot_quote_z", "quote_share_z", "trades_z",
    "imb6_z", "imb24_z", "skew24_z", "kurt24_z",
    "oi_level_z", "oi_source_z", "oi_value_source_z", "funding_z",
    "funding_change_z", "crowding_z", "account_ratio_z", "metric_taker_z",
    "density_gap", "pressure_gap", "simple_reflection", "simple_transmission_gap",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, minimum: int = 8) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < minimum or np.nanstd(a[mask]) <= EPS or np.nanstd(b[mask]) <= EPS:
        return float("nan")
    value = spearmanr(a[mask], b[mask]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def timestamp_from_any(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty:
        unit = "us" if float(finite.abs().median()) > 1.0e15 else "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def read_zip(url: str, *, header: int | None) -> pd.DataFrame:
    response = requests.get(url, timeout=60, headers={"User-Agent": "Trade-fluid-spot-perp-round15/20261119"})
    if response.status_code == 404:
        raise FileNotFoundError(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV in {url}")
        return pd.read_csv(archive.open(names[0]), header=header, low_memory=False)


def download_kline(symbol: str, period: pd.Period, venue: str) -> tuple[pd.DataFrame | None, str | None]:
    if venue == "perp":
        url = (
            "https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{symbol}/1h/{symbol}-1h-{period.year:04d}-{period.month:02d}.zip"
        )
    else:
        url = (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"{symbol}/1h/{symbol}-1h-{period.year:04d}-{period.month:02d}.zip"
        )
    last_error: str | None = None
    for attempt in range(4):
        try:
            frame = read_zip(url, header=None).iloc[:, :12]
            frame.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ]
            frame["time"] = timestamp_from_any(frame["open_time"])
            for column in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["symbol"] = symbol
            frame = frame[[
                "time", "symbol", "open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_quote",
            ]].replace([np.inf, -np.inf], np.nan)
            frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
            frame = frame.loc[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
            return frame, None
        except FileNotFoundError:
            return None, "404"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return None, last_error


def download_metrics(symbol: str, period: pd.Period, kind: str) -> tuple[pd.DataFrame | None, str | None]:
    if kind == "metrics":
        url = (
            "https://data.binance.vision/data/futures/um/monthly/metrics/"
            f"{symbol}/{symbol}-metrics-{period.year:04d}-{period.month:02d}.zip"
        )
    else:
        url = (
            "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
            f"{symbol}/{symbol}-fundingRate-{period.year:04d}-{period.month:02d}.zip"
        )
    last_error: str | None = None
    for attempt in range(4):
        try:
            frame = read_zip(url, header=0)
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            return frame, None
        except FileNotFoundError:
            return None, "404"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return None, last_error


def load_public_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks: list[tuple[str, pd.Period, str]] = []
    for symbol in SYMBOLS:
        for period in MONTHS:
            tasks.extend([(symbol, period, "perp"), (symbol, period, "spot"), (symbol, period, "metrics"), (symbol, period, "funding")])
    buckets: dict[str, list[pd.DataFrame]] = {kind: [] for kind in ["perp", "spot", "metrics", "funding"]}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        jobs = {}
        for symbol, period, kind in tasks:
            if kind in {"perp", "spot"}:
                future = pool.submit(download_kline, symbol, period, kind)
            else:
                future = pool.submit(download_metrics, symbol, period, kind)
            jobs[future] = (symbol, period, kind)
        for count, future in enumerate(as_completed(jobs), 1):
            symbol, period, kind = jobs[future]
            frame, error = future.result()
            if frame is not None and not frame.empty:
                frame["symbol"] = symbol
                buckets[kind].append(frame)
            elif error != "404":
                errors.append({"symbol": symbol, "month": str(period), "kind": kind, "error": str(error)})
            if count % 300 == 0:
                print(f"downloads={count}/{len(tasks)} usable=" + ",".join(f"{k}:{len(v)}" for k, v in buckets.items()), flush=True)

    if not buckets["perp"] or not buckets["spot"]:
        raise RuntimeError("spot or perpetual kline data missing")
    perp = pd.concat(buckets["perp"], ignore_index=True).drop_duplicates(["symbol", "time"], keep="last")
    spot = pd.concat(buckets["spot"], ignore_index=True).drop_duplicates(["symbol", "time"], keep="last")
    spot = spot.rename(columns={column: f"{column}_spot" for column in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]})
    panel = perp.merge(spot, on=["time", "symbol"], how="inner", validate="one_to_one")

    metric_parts: list[pd.DataFrame] = []
    if buckets["metrics"]:
        raw = pd.concat(buckets["metrics"], ignore_index=True)
        aliases = {
            "create_time": "time", "sum_open_interest": "open_interest",
            "sum_open_interest_value": "open_interest_value",
            "count_toptrader_long_short_ratio": "top_count_ratio",
            "sum_toptrader_long_short_ratio": "top_position_ratio",
            "count_long_short_ratio": "account_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ratio",
            "taker_buy_sell_ratio": "taker_ratio",
        }
        raw = raw.rename(columns=aliases)
        if "time" in raw:
            raw["time"] = timestamp_from_any(raw["time"])
            for column in ["open_interest", "open_interest_value", "top_count_ratio", "top_position_ratio", "account_ratio", "taker_ratio"]:
                if column not in raw:
                    raw[column] = np.nan
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            for symbol, group in raw.groupby("symbol", sort=False):
                g = group.set_index("time").sort_index()[["open_interest", "open_interest_value", "top_count_ratio", "top_position_ratio", "account_ratio", "taker_ratio"]].resample("1h").last().reset_index()
                g["time"] = g["time"] + pd.Timedelta(hours=1)
                g["symbol"] = symbol
                metric_parts.append(g)
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame(columns=["time", "symbol"])

    funding_parts: list[pd.DataFrame] = []
    if buckets["funding"]:
        raw = pd.concat(buckets["funding"], ignore_index=True)
        time_column = next((column for column in ["calc_time", "funding_time", "time"] if column in raw), None)
        rate_column = next((column for column in ["last_funding_rate", "funding_rate", "fundingrate"] if column in raw), None)
        if time_column and rate_column:
            raw["time"] = timestamp_from_any(raw[time_column])
            raw["funding_rate"] = pd.to_numeric(raw[rate_column], errors="coerce")
            for symbol, group in raw.groupby("symbol", sort=False):
                g = group[["time", "funding_rate"]].dropna().sort_values("time")
                g["time"] = g["time"] + pd.Timedelta(hours=1)
                g["symbol"] = symbol
                funding_parts.append(g)
    funding = pd.concat(funding_parts, ignore_index=True) if funding_parts else pd.DataFrame(columns=["time", "symbol"])

    merged_parts: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        if not metrics.empty:
            m = metrics.loc[metrics["symbol"] == symbol].sort_values("time").drop(columns="symbol")
            g = pd.merge_asof(g, m, on="time", direction="backward", tolerance=pd.Timedelta(hours=2))
        if not funding.empty:
            f = funding.loc[funding["symbol"] == symbol].sort_values("time").drop(columns="symbol")
            g = pd.merge_asof(g, f, on="time", direction="backward", tolerance=pd.Timedelta(hours=12))
        merged_parts.append(g)
    panel = pd.concat(merged_parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    audit = {
        "requested_downloads": len(tasks),
        "usable_parts": {key: len(value) for key, value in buckets.items()},
        "errors": errors[:100],
        "rows": int(len(panel)),
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "min_time": str(panel["time"].min()),
        "max_time": str(panel["time"].max()),
        "metric_coverage": int(panel.get("open_interest", pd.Series(dtype=float)).notna().sum()),
        "funding_coverage": int(panel.get("funding_rate", pd.Series(dtype=float)).notna().sum()),
    }
    return panel, audit


def causal_z(values: pd.Series, window: int = 720, minimum: int = 240) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0)
    return ((series - mean) / (std + EPS)).clip(-12.0, 12.0)


def causal_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    xp = pd.to_numeric(x, errors="coerce").shift(1)
    yp = pd.to_numeric(y, errors="coerce").shift(1)
    minimum = max(36, int(0.60 * window))
    mx = xp.rolling(window, min_periods=minimum).mean()
    my = yp.rolling(window, min_periods=minimum).mean()
    cov = (xp * yp).rolling(window, min_periods=minimum).mean() - mx * my
    var = xp.pow(2).rolling(window, min_periods=minimum).mean() - mx.pow(2)
    return (cov / (var + EPS)).clip(-16.0, 16.0)


def norm_momentum(u: pd.Series, window: int) -> pd.Series:
    return (u.rolling(window, min_periods=window).sum() / (np.sqrt(u.pow(2).rolling(window, min_periods=window).sum()) + EPS)).clip(-12.0, 12.0)


def robust_median(items: list[pd.Series]) -> pd.Series:
    matrix = np.column_stack([pd.to_numeric(item, errors="coerce").to_numpy(float) for item in items])
    with np.errstate(all="ignore"):
        values = np.nanmedian(matrix, axis=1)
    values[np.all(~np.isfinite(matrix), axis=1)] = np.nan
    return pd.Series(values, index=items[0].index, dtype=float)


def local_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("time").copy()
    log_perp = np.log(pd.to_numeric(g["close"], errors="coerce"))
    log_spot = np.log(pd.to_numeric(g["close_spot"], errors="coerce"))
    g["perp_u"] = log_perp.diff()
    g["spot_u"] = log_spot.diff()
    g["shear"] = g["perp_u"] - g["spot_u"]
    basis = log_perp - log_spot
    g["basis_z"] = causal_z(basis)
    g["basis_change_z"] = causal_z(basis.diff())
    for window in [4, 12, 24, 72, 168, 480]:
        g[f"mom{window}"] = norm_momentum(g["perp_u"], window)
    for window in [24, 72, 168]:
        g[f"rv{window}_z"] = causal_z(g["perp_u"].rolling(window, min_periods=window).std(ddof=0))
    range_perp = np.log(pd.to_numeric(g["high"], errors="coerce") / pd.to_numeric(g["low"], errors="coerce"))
    range_spot = np.log(pd.to_numeric(g["high_spot"], errors="coerce") / pd.to_numeric(g["low_spot"], errors="coerce"))
    g["range_perp_z"] = causal_z(range_perp)
    g["range_spot_z"] = causal_z(range_spot)
    g["perp_quote_z"] = causal_z(np.log1p(pd.to_numeric(g["quote_volume"], errors="coerce")))
    g["spot_quote_z"] = causal_z(np.log1p(pd.to_numeric(g["quote_volume_spot"], errors="coerce")))
    quote_share = np.log((pd.to_numeric(g["quote_volume"], errors="coerce") + EPS) / (pd.to_numeric(g["quote_volume_spot"], errors="coerce") + EPS))
    g["quote_share_z"] = causal_z(quote_share)
    g["trades_z"] = causal_z(np.log1p(pd.to_numeric(g["trades"], errors="coerce")))
    imbalance = 2.0 * pd.to_numeric(g["taker_buy_quote"], errors="coerce") / (pd.to_numeric(g["quote_volume"], errors="coerce") + EPS) - 1.0
    g["imb6_z"] = causal_z(imbalance.rolling(6, min_periods=6).mean())
    g["imb24_z"] = causal_z(imbalance.rolling(24, min_periods=24).mean())
    g["skew24_z"] = causal_z(g["perp_u"].rolling(24, min_periods=24).skew())
    g["kurt24_z"] = causal_z(g["perp_u"].rolling(24, min_periods=24).kurt())
    oi = pd.to_numeric(g.get("open_interest", np.nan), errors="coerce")
    oi_value = pd.to_numeric(g.get("open_interest_value", np.nan), errors="coerce")
    g["oi_level_z"] = causal_z(np.log1p(oi))
    g["oi_source_z"] = causal_z(np.log(oi.clip(lower=EPS)).diff())
    g["oi_value_source_z"] = causal_z(np.log(oi_value.clip(lower=EPS)).diff())
    funding = pd.to_numeric(g.get("funding_rate", np.nan), errors="coerce")
    g["funding_z"] = causal_z(funding)
    g["funding_change_z"] = causal_z(funding.diff())
    top_count = pd.to_numeric(g.get("top_count_ratio", np.nan), errors="coerce")
    top_position = pd.to_numeric(g.get("top_position_ratio", np.nan), errors="coerce")
    account = pd.to_numeric(g.get("account_ratio", np.nan), errors="coerce")
    metric_taker = pd.to_numeric(g.get("taker_ratio", np.nan), errors="coerce")
    g["crowding_z"] = causal_z(np.log(top_position.clip(lower=EPS)) - np.log(top_count.clip(lower=EPS)))
    g["account_ratio_z"] = causal_z(np.log(account.clip(lower=EPS)))
    g["metric_taker_z"] = causal_z(np.log(metric_taker.clip(lower=EPS)))
    g["density_gap"] = g["perp_quote_z"] - g["spot_quote_z"] + 0.35 * g["oi_level_z"]
    g["pressure_gap"] = 2.0 * g["basis_z"] + 0.35 * (g["range_perp_z"] - g["range_spot_z"]) + 0.25 * g["funding_z"]
    g["simple_reflection"] = g["density_gap"] / (g["density_gap"].abs() + 2.0)
    g["simple_transmission_gap"] = g["spot_u"] - g["perp_u"]
    for horizon in HORIZONS:
        g[f"y_{horizon}"] = np.log(pd.to_numeric(g["close"], errors="coerce").shift(-horizon) / pd.to_numeric(g["open"], errors="coerce").shift(-1))
    return g


def characteristic_features(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    g = group.sort_values("time").copy()
    u_p = pd.to_numeric(g["perp_u"], errors="coerce")
    u_s = pd.to_numeric(g["spot_u"], errors="coerce")
    p_p = g["basis_z"] + 0.35 * g["range_perp_z"] + 0.25 * g["funding_z"]
    p_s = -g["basis_z"] + 0.35 * g["range_spot_z"]
    rho_p = np.exp(np.clip((0.65 * g["perp_quote_z"] + 0.35 * g["oi_level_z"]) / 3.0, -3.0, 3.0))
    rho_s = np.exp(np.clip(g["spot_quote_z"] / 3.0, -3.0, 3.0))
    entropy_p = 0.55 * g["rv24_z"] + 0.30 * g["range_perp_z"] + 0.15 * g["imb24_z"].abs()
    entropy_s = 0.65 * causal_z(u_s.rolling(24, min_periods=24).std(ddof=0)) + 0.35 * g["range_spot_z"]
    source = g["oi_source_z"].fillna(0.0) + 0.50 * g["funding_change_z"].fillna(0.0) + 0.25 * g["quote_share_z"].fillna(0.0)

    values: dict[str, list[pd.Series]] = {name: [] for names in FAMILY_RAW.values() for name in names}
    audit: dict[str, Any] = {}
    for window in SOUND_WINDOWS:
        c2_p = causal_beta(p_p.diff(), np.log(rho_p + EPS).diff(), window)
        c2_s = causal_beta(p_s.diff(), np.log(rho_s + EPS).diff(), window)
        c_p = np.sqrt(c2_p.abs() + 0.05).clip(0.15, 4.0)
        c_s = np.sqrt(c2_s.abs() + 0.05).clip(0.15, 4.0)
        z_p = rho_p * c_p
        z_s = rho_s * c_s
        denom = z_p + z_s + EPS
        t_sp = 2.0 * z_p / denom
        t_ps = 2.0 * z_s / denom
        reflection = (z_p - z_s) / denom
        jp_p = u_p + p_p / (z_p + EPS)
        jm_p = u_p - p_p / (z_p + EPS)
        jp_s = u_s + p_s / (z_s + EPS)
        jm_s = u_s - p_s / (z_s + EPS)
        forward = t_sp * jp_s - jp_p
        reverse = t_ps * jm_p - jm_s
        values["spot_to_perp_transmission_defect"].append(forward)
        values["perp_to_spot_transmission_defect"].append(reverse)
        values["bidirectional_characteristic_balance"].append(0.5 * (forward - reverse))
        values["impedance_pressure_release"].append(-reflection * (p_p - p_s))
        values["impedance_matched_shear_transport"].append((1.0 - reflection.abs()) * (u_s - u_p))
        energy_p = 0.25 * rho_p * (jp_p.pow(2) + jm_p.pow(2))
        energy_s = 0.25 * rho_s * (jp_s.pow(2) + jm_s.pow(2))
        values["impedance_energy_absorption"].append((energy_s - energy_p) * (1.0 - reflection.abs()))
        lambda_p = u_p + c_p
        residual = jp_p.diff() - lambda_p * (jp_s - jp_p) - source
        values["source_adjusted_characteristic_residual"].append(residual)
        values["source_absorption_release"].append(-residual * source)
        values["oi_supported_characteristic_transfer"].append(forward * source)
        action_p = energy_p / (c_p + 0.10)
        action_s = energy_s / (c_s + 0.10)
        flux_p = (u_p + c_p) * action_p
        flux_s = (u_s + c_s) * action_s
        action_defect = action_p.diff() - (flux_s - flux_p)
        values["wave_action_transfer_defect"].append(action_defect)
        values["wave_action_residual_release"].append(-action_defect * action_defect.diff())
        values["wave_energy_absorption"].append((action_s - action_p) * (1.0 - reflection.abs()))
        characteristic_divergence = (u_p + c_p) - (u_s + c_s)
        shock = (-characteristic_divergence).clip(lower=0.0) * g["basis_z"].abs()
        rarefaction = characteristic_divergence.clip(lower=0.0) * (-g["basis_z"])
        values["basis_characteristic_shock"].append(shock)
        values["basis_characteristic_rarefaction"].append(rarefaction)
        values["shock_entropy_release"].append(-shock * (entropy_p - entropy_s).diff())
        gap = np.sqrt((jp_p - jp_s).pow(2) + (jm_p - jm_s).pow(2))
        prior_gap = gap.shift(1)
        prior_incoming = (jm_p - jm_s).abs().shift(1)
        values["interlayer_characteristic_gap"].append(gap)
        values["interlayer_characteristic_reattachment"].append(np.sign(g["mom24"]) * prior_gap * (prior_gap - gap).clip(lower=0.0))
        values["incoming_characteristic_release"].append(np.sign(g["mom24"]) * prior_incoming * (prior_incoming - (jm_p - jm_s).abs()).clip(lower=0.0))
        shear = u_p - u_s
        density_gradient = np.log(rho_p + EPS) - np.log(rho_s + EPS)
        stress = shear.rolling(24, min_periods=18).cov(density_gradient)
        values["interlayer_reynolds_stress"].append(stress)
        values["interlayer_stress_release"].append(-stress * stress.diff())
        values["source_supported_stress"].append(stress * source)
        enthalpy_gap = p_p / (rho_p + EPS) - p_s / (rho_s + EPS) + 0.5 * (u_p.pow(2) - u_s.pow(2))
        crocco = enthalpy_gap - 0.5 * (u_p + u_s) * (u_p - u_s) - (entropy_p - entropy_s)
        values["interlayer_crocco_residual"].append(crocco)
        values["interlayer_crocco_release"].append(-crocco * crocco.diff())
        values["source_supported_enthalpy_gap"].append(enthalpy_gap * source)
        audit[str(window)] = {
            "median_c_perp": float(np.nanmedian(c_p)),
            "median_c_spot": float(np.nanmedian(c_s)),
            "negative_c2_perp_share": float(np.nanmean(c2_p < 0.0)),
            "negative_c2_spot_share": float(np.nanmean(c2_s < 0.0)),
        }

    raw = {name: robust_median(items) for name, items in values.items()}
    raw_frame = pd.DataFrame(raw, index=g.index)
    state_columns: dict[str, pd.Series] = {}
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    for family, names in FAMILY_RAW.items():
        for name in names:
            state = causal_z(raw_frame[name])
            z_name = f"{name}_z"
            state_columns[z_name] = state
            candidate_map[family].append(z_name)
            for span in STATE_SPANS:
                ewm_name = f"{name}_ewm{span}"
                state_columns[ewm_name] = state.ewm(span=span, adjust=False, min_periods=max(3, span // 3)).mean()
                candidate_map[family].append(ewm_name)
    return pd.concat([g, raw_frame, pd.DataFrame(state_columns, index=g.index)], axis=1), candidate_map, audit


def issue_subset(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    hours = ((frame["time"] - epoch) // pd.Timedelta(hours=1)).astype("int64")
    return frame.loc[(hours % horizon) == 0].copy()


def cross_sectional_ic(frame: pd.DataFrame, factor: str, target: str) -> tuple[float, pd.Series]:
    rows: list[tuple[pd.Timestamp, float]] = []
    for timestamp, group in frame.groupby("time", sort=True):
        value = safe_spearman(group[factor], group[target], minimum=6)
        if np.isfinite(value):
            rows.append((timestamp, value))
    daily = pd.Series({timestamp: value for timestamp, value in rows}, dtype=float).sort_index()
    return (float(daily.mean()) if len(daily) else float("nan"), daily)


def time_series_ic(frame: pd.DataFrame, factor: str, target: str) -> tuple[float, float, int]:
    values = []
    for _, group in frame.groupby("symbol", sort=False):
        value = safe_spearman(group[factor], group[target], minimum=30)
        if np.isfinite(value):
            values.append(value)
    return (float(np.mean(values)) if values else float("nan"), float(np.mean(np.asarray(values) > 0.0)) if values else float("nan"), len(values))


def yearly_rankic(frame: pd.DataFrame, factor: str, target: str) -> dict[str, float]:
    result = {}
    for year, group in frame.groupby(frame["time"].dt.year):
        value, _ = cross_sectional_ic(group, factor, target)
        result[str(int(year))] = value
    return result


def moving_block_bootstrap(series: pd.Series, block: int, repetitions: int = 5000) -> dict[str, float]:
    values = np.asarray(series.dropna(), dtype=float)
    if len(values) < max(12, 2 * block):
        return {"low": float("nan"), "high": float("nan"), "p_nonpositive": float("nan")}
    rng = np.random.default_rng(SEED)
    starts = np.arange(max(1, len(values) - block + 1))
    means = np.empty(repetitions)
    for i in range(repetitions):
        sample = []
        while len(sample) < len(values):
            start = int(rng.choice(starts))
            sample.extend(values[start:start + block])
        means[i] = np.mean(sample[:len(values)])
    return {"low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975)), "p_nonpositive": float(np.mean(means <= 0.0))}


def rank_weights(group: pd.DataFrame, factor: str) -> pd.Series:
    rank = group[factor].rank(pct=True, method="average")
    weight = pd.Series(0.0, index=group.index)
    long = rank >= 0.70
    short = rank <= 0.30
    if long.any():
        weight.loc[long] = 0.5 / long.sum()
    if short.any():
        weight.loc[short] = -0.5 / short.sum()
    return weight


def portfolio(frame: pd.DataFrame, factor: str, target: str, horizon: int, cost_bp: float) -> dict[str, Any]:
    work = frame.loc[np.isfinite(frame[factor]) & np.isfinite(frame[target])].copy()
    if work.empty:
        return {"annual_log_return": float("nan"), "sharpe": float("nan"), "avg_turnover": float("nan"), "n": 0, "years": {}}
    work["weight"] = work.groupby("time", group_keys=False).apply(lambda g: rank_weights(g, factor), include_groups=False)
    pivot = work.pivot(index="time", columns="symbol", values="weight").fillna(0.0)
    returns = work.pivot(index="time", columns="symbol", values=target).reindex(pivot.index).fillna(0.0)
    turnover = pivot.diff().abs().sum(axis=1).fillna(pivot.abs().sum(axis=1))
    gross = (pivot * returns).sum(axis=1)
    net = gross - turnover * cost_bp * 1.0e-4
    scale = 365.25 * 24.0 / horizon
    annual = float(net.mean() * scale)
    sharpe = float(net.mean() / (net.std(ddof=0) + EPS) * math.sqrt(scale))
    years = {str(int(year)): float(values.mean() * scale) for year, values in net.groupby(net.index.year)}
    return {"annual_log_return": annual, "sharpe": sharpe, "avg_turnover": float(turnover.mean()), "n": int(len(net)), "years": years}


def impute_standardize(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    medians = train[features].median(axis=0, skipna=True)
    usable = [column for column in features if np.isfinite(medians.get(column, np.nan))]
    medians = medians[usable]
    x = train[usable].fillna(medians).to_numpy(float)
    xt = test[usable].fillna(medians).to_numpy(float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > EPS, std, 1.0)
    return (x - mean) / std, (xt - mean) / std, usable


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, train_end: int, start: int, end: int) -> dict[str, Any]:
    target = f"y_{horizon}"
    train = issue_subset(panel.loc[panel["time"].dt.year <= train_end].copy(), horizon)
    test = issue_subset(panel.loc[(panel["time"].dt.year >= start) & (panel["time"].dt.year <= end)].copy(), horizon)
    test = test.loc[np.isfinite(test[factor]) & np.isfinite(test[target])].copy()
    train = train.loc[np.isfinite(train[factor]) & np.isfinite(train[target])].copy()
    cs, daily = cross_sectional_ic(test, factor, target)
    ts, ts_positive, ts_n = time_series_ic(test, factor, target)
    boot = moving_block_bootstrap(daily, block=max(4, int(math.ceil(horizon / 24))))
    if len(train) > 300_000:
        train = train.iloc[np.linspace(0, len(train) - 1, 300_000, dtype=int)]
    xb, xbt, usable = impute_standardize(train, test, BASE_FEATURES)
    y = train[target].to_numpy(float)
    yt = test[target].to_numpy(float)
    factor_median = float(train[factor].median())
    f_train = train[factor].fillna(factor_median).to_numpy(float)[:, None]
    f_test = test[factor].fillna(factor_median).to_numpy(float)[:, None]
    f_mean, f_std = float(f_train.mean()), float(f_train.std())
    f_std = f_std if f_std > EPS else 1.0
    xf = np.column_stack([xb, (f_train - f_mean) / f_std])
    xft = np.column_stack([xbt, (f_test - f_mean) / f_std])
    base = Ridge(alpha=20.0).fit(xb, y)
    full = Ridge(alpha=20.0).fit(xf, y)
    pb, pf = base.predict(xbt), full.predict(xft)
    mse_b = float(np.mean((yt - pb) ** 2))
    mse_f = float(np.mean((yt - pf) ** 2))
    pred = test[["time", "symbol"]].copy()
    pred["target"], pred["base"], pred["full"] = yt, pb, pf
    base_ic, _ = cross_sectional_ic(pred, "base", "target")
    full_ic, _ = cross_sectional_ic(pred, "full", "target")
    parent_model = Ridge(alpha=20.0).fit(xb, train[factor].to_numpy(float))
    residual = test[factor].to_numpy(float) - parent_model.predict(xbt)
    test["__residual"] = residual
    residual_cs, _ = cross_sectional_ic(test, "__residual", target)
    residual_ts, _, _ = time_series_ic(test, "__residual", target)
    correlations = {column: abs(safe_spearman(test[factor], test[column], minimum=100)) for column in usable}
    max_feature = max(correlations, key=lambda key: correlations[key] if np.isfinite(correlations[key]) else -1.0)
    max_corr = correlations[max_feature]
    return {
        "factor": factor, "horizon": horizon, "train_end": train_end, "start": start, "end": end,
        "cs_rankic": cs, "ts_ic": ts, "ts_positive_share": ts_positive, "ts_products": ts_n,
        "bootstrap_low": boot["low"], "bootstrap_high": boot["high"], "bootstrap_p_nonpositive": boot["p_nonpositive"],
        "ridge_mse_gain": float((mse_b - mse_f) / (mse_b + EPS)), "ridge_base_rankic": base_ic, "ridge_full_rankic": full_ic,
        "residual_cs_rankic": residual_cs, "residual_ts_ic": residual_ts,
        "max_parent_corr": max_corr, "max_parent_feature": max_feature,
        "annual_log_return_2bp": portfolio(test, factor, target, horizon, 2.0)["annual_log_return"],
        "annual_log_return_5bp": portfolio(test, factor, target, horizon, 5.0)["annual_log_return"],
        "sharpe_5bp": portfolio(test, factor, target, horizon, 5.0)["sharpe"],
        "avg_turnover": portfolio(test, factor, target, horizon, 5.0)["avg_turnover"],
        "yearly_rankic": json.dumps(yearly_rankic(test, factor, target), sort_keys=True),
        "yearly_5bp": json.dumps(portfolio(test, factor, target, horizon, 5.0)["years"], sort_keys=True),
        "n_train": int(len(train)), "n_test": int(len(test)), "parent_count": len(usable),
    }


def development_score(panel: pd.DataFrame, factor: str, horizon: int, sign: int) -> dict[str, Any]:
    target = f"y_{horizon}"
    dev = issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon)
    temp = f"__signed_{factor}"
    dev[temp] = sign * dev[factor]
    cs, _ = cross_sectional_ic(dev, temp, target)
    ts, _, _ = time_series_ic(dev, temp, target)
    years = yearly_rankic(dev, temp, target)
    finite = [float(value) for value in years.values() if np.isfinite(value)]
    minimum = min(finite) if finite else -1.0
    score = minimum + 0.30 * (cs if np.isfinite(cs) else -1.0) + 0.10 * (ts if np.isfinite(ts) else -1.0)
    return {"factor": factor, "sign": sign, "horizon": horizon, "development_cs": cs, "development_ts": ts, "development_min_year": minimum, "development_years": json.dumps(years, sort_keys=True), "development_score": score}


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    np.random.seed(SEED)
    raw, data_audit = load_public_data()
    local_parts = []
    candidate_map: dict[str, list[str]] = {family: [] for family in FAMILY_RAW}
    characteristic_audit = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        local = local_features(group)
        enriched, local_map, audit = characteristic_features(local)
        local_parts.append(enriched)
        characteristic_audit[symbol] = audit
        for family, columns in local_map.items():
            candidate_map[family] = columns
        print(f"features={symbol} rows={len(enriched)}", flush=True)
    panel = pd.concat(local_parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)

    scan_rows = []
    for family, factors in candidate_map.items():
        for factor in factors:
            for horizon in HORIZONS:
                plus = development_score(panel, factor, horizon, 1)
                minus = development_score(panel, factor, horizon, -1)
                best = plus if plus["development_score"] >= minus["development_score"] else minus
                best["family"] = family
                scan_rows.append(best)
    scan = pd.DataFrame(scan_rows)
    scan.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)
    winners = scan.sort_values(["family", "development_score"], ascending=[True, False]).groupby("family", as_index=False).head(1).reset_index(drop=True)

    for row in winners.itertuples(index=False):
        panel[f"signed_{row.factor}"] = int(row.sign) * panel[row.factor]
    ranked_families = winners.sort_values("development_score", ascending=False)["family"].tolist()
    composite_specs = {
        "spot_perp_top4_consensus": ranked_families[:4],
        "source_riemann_impedance_consensus": [name for name in ["source_conservation", "riemann_transfer", "impedance_release"] if name in set(winners["family"])],
        "wave_shock_reattachment_consensus": [name for name in ["wave_action", "shock_rarefaction", "characteristic_reattachment"] if name in set(winners["family"])],
    }
    composite_rows = []
    winner_map = {row.family: f"signed_{row.factor}" for row in winners.itertuples(index=False)}
    for name, families in composite_specs.items():
        members = [winner_map[family] for family in families]
        ranks = [panel.groupby("time")[member].rank(pct=True) for member in members]
        panel[name] = pd.concat(ranks, axis=1).mean(axis=1)
        choices = [development_score(panel, name, horizon, sign) for horizon in HORIZONS for sign in [1, -1]]
        best = max(choices, key=lambda item: item["development_score"])
        best["family"] = name
        composite_rows.append(best)
    selection = pd.concat([winners, pd.DataFrame(composite_rows)], ignore_index=True)
    selection.to_csv(OUT / "FROZEN_SELECTION.csv", index=False)

    validation_rows = []
    for row in selection.itertuples(index=False):
        factor = row.factor if str(row.factor).startswith("spot_perp_") or str(row.factor).endswith("_consensus") else row.factor
        signed_name = f"validation_signed_{len(validation_rows)}"
        panel[signed_name] = int(row.sign) * panel[factor]
        result = evaluate(panel, signed_name, int(row.horizon), 2022, 2023, 2024)
        result.update({"family": row.family, "primitive_factor": factor, "sign": int(row.sign)})
        validation_rows.append(result)
        print(f"validation={row.family} factor={factor}", flush=True)
    validation = pd.DataFrame(validation_rows)
    raw_p = [float(value) if np.isfinite(value) else 1.0 for value in validation["bootstrap_p_nonpositive"]]
    validation["holm_p"] = holm_adjust(raw_p)
    yearly_ok = []
    for payload in validation["yearly_rankic"]:
        values = json.loads(payload)
        yearly_ok.append(bool(values) and all(np.isfinite(float(v)) and float(v) > 0.0 for v in values.values()))
    validation["validation_gate"] = (
        (validation["cs_rankic"] > 0.012)
        & (validation["bootstrap_p_nonpositive"] <= 0.10)
        & (validation["ridge_mse_gain"] > 0.0)
        & (validation["residual_cs_rankic"] > 0.005)
        & (validation["max_parent_corr"] < 0.80)
        & (validation["annual_log_return_5bp"] > 0.0)
        & (validation["holm_p"] <= 0.10)
        & pd.Series(yearly_ok, index=validation.index)
    )
    validation.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)

    opened = validation.loc[validation["validation_gate"]].copy()
    holdout_rows = []
    for row in opened.itertuples(index=False):
        result = evaluate(panel, row.factor, int(row.horizon), 2024, 2025, 2026)
        result.update({"family": row.family, "validation_holm_p": row.holm_p, "primitive_factor": row.primitive_factor, "sign": row.sign})
        holdout_rows.append(result)
    holdout = pd.DataFrame(holdout_rows)
    if holdout.empty:
        pd.DataFrame(columns=validation.columns).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    else:
        holdout.to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)

    promoted = []
    for row in holdout.itertuples(index=False):
        years = json.loads(row.yearly_rankic)
        passed = (
            row.cs_rankic > 0.012 and row.bootstrap_p_nonpositive <= 0.10
            and row.ridge_mse_gain > 0.0 and row.residual_cs_rankic > 0.005
            and row.max_parent_corr < 0.80 and row.annual_log_return_5bp > 0.0
            and bool(years) and all(float(v) > 0.0 for v in years.values())
        )
        if passed:
            promoted.append({
                "family": row.family, "factor": row.primitive_factor, "sign": int(row.sign), "horizon": int(row.horizon),
                "validation_holm_p": float(row.validation_holm_p), "holdout_cs_rankic": float(row.cs_rankic),
                "holdout_ridge_mse_gain": float(row.ridge_mse_gain), "holdout_residual_cs_rankic": float(row.residual_cs_rankic),
                "holdout_max_parent_corr": float(row.max_parent_corr), "holdout_annual_log_return_5bp": float(row.annual_log_return_5bp),
            })

    audit = {
        "data": data_audit,
        "characteristic": characteristic_audit,
        "panel_rows": int(len(panel)),
        "candidate_families": len(candidate_map),
        "candidate_columns": sum(len(values) for values in candidate_map.values()),
        "base_features": BASE_FEATURES,
        "availability_policy": {
            "spot_and_perp_klines": "completed hour only; target begins at next-hour open",
            "metrics": "resampled and shifted one completed hour",
            "funding": "publication time shifted one completed hour",
            "sound_speed": "strict-prior rolling pressure-density response",
        },
    }
    write_json(OUT / "DATA_AUDIT.json", audit)
    write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "candidate_families": len(candidate_map), "candidate_columns": sum(len(values) for values in candidate_map.values()),
        "development_tests": int(len(scan)), "frozen_identities": int(len(selection)),
        "validation_gate_passed": int(validation["validation_gate"].sum()), "holdout_opened": int(len(holdout)),
        "promoted": promoted, "decision": "promote_only_if_PROMOTED_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    report = [
        "# Spot-Perpetual Characteristic Transport Round 15", "",
        "This experiment tests a two-layer spot/perpetual compressible characteristic system with real basis, funding and open-interest source terms.", "",
        "## Protocol", "", "- development: 2021-2022", "- validation: 2023-2024",
        "- 2025-2026 historical holdout opens only after the full validation gate", "- target begins at the next completed hourly open", "",
        "## Frozen identities", "", selection.to_markdown(index=False), "", "## Validation", "", validation.to_markdown(index=False), "",
        "## Historical holdout", "", holdout.to_markdown(index=False) if not holdout.empty else "No identity passed validation; holdout remained closed.", "",
        "## Promoted", "", "```json", json.dumps(promoted, indent=2, sort_keys=True), "```",
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
