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

SEED = 20261118
RNG = np.random.default_rng(SEED)
EPS = 1.0e-12
OUT = Path("research_outputs/fluid_round9")
OUT.mkdir(parents=True, exist_ok=True)

FUTURES_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "SOLUSDT",
]
FUTURES_HORIZONS = [1, 4, 12, 24, 72, 120, 480]
STOCK_HORIZONS = [1, 5, 10, 20]
MONTHS = list(pd.period_range("2021-01", "2026-10", freq="M"))

STOCK_SECTORS = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "AVGO": "technology", "ORCL": "technology", "CSCO": "technology",
    "IBM": "technology", "ADBE": "technology",
    "GOOGL": "communication", "META": "communication", "NFLX": "communication",
    "DIS": "communication", "CMCSA": "communication",
    "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
    "HD": "consumer_discretionary", "MCD": "consumer_discretionary",
    "NKE": "consumer_discretionary", "SBUX": "consumer_discretionary",
    "WMT": "consumer_staples", "COST": "consumer_staples",
    "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "MRK": "healthcare", "ABBV": "healthcare", "LLY": "healthcare",
    "JPM": "financials", "BAC": "financials", "WFC": "financials",
    "GS": "financials", "MS": "financials", "V": "financials", "MA": "financials",
    "CAT": "industrials", "DE": "industrials", "GE": "industrials",
    "RTX": "industrials", "HON": "industrials", "UPS": "industrials",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
    "LIN": "materials", "SHW": "materials", "FCX": "materials",
    "AMT": "real_estate", "PLD": "real_estate",
}

FUTURES_BASE = [
    "p_u", "s_u", "p_mom4", "p_mom24", "p_mom72", "s_mom24",
    "basis_z", "basis_velocity_z", "shear_z", "p_qz", "s_qz",
    "p_imb_z", "s_imb_z", "activity_gap_z", "flow_gap_z",
    "p_rv_z", "s_rv_z", "p_range_z", "s_range_z",
]
STOCK_BASE = [
    "r1", "mom5", "mom20", "mom60", "mom120", "vol20_z",
    "qz", "range_z", "close_location", "mom5_gap", "mom20_gap",
    "sector_peer_u", "market_u", "size_proxy",
]

FUTURES_FAMILIES = {
    "interfacial_energy": ["interfacial_energy_flux", "shear_relaxation", "shear_drive"],
    "kelvin_helmholtz": ["kh_onset", "kh_release"],
    "interfacial_vorticity": ["interfacial_vorticity_6", "interfacial_vorticity_24"],
    "continuity_pressure": ["continuity_residual", "pressure_release", "pressure_drive"],
    "sweeping_impedance": ["spot_sweeping_lead", "interfacial_impedance", "impedance_mismatch"],
    "interfacial_geometry": ["interfacial_torsion"],
}
STOCK_FAMILIES = {
    "sector_shear": ["sector_shear_absorption", "sector_shear_drive", "sector_shear_curvature"],
    "pressure_poisson": ["pressure_reversal", "pressure_flow"],
    "sector_vorticity": ["sector_vorticity", "sector_vorticity_release"],
    "source_sink": ["source_sink_absorption", "source_sink_release"],
    "wake_drag": ["sector_wake_catchup", "leader_wake_exhaustion", "drag_efficiency", "drag_breakdown"],
    "boundary_layer": ["boundary_layer_separation", "boundary_layer_reattachment"],
    "multiscale_flux": ["scale_flux_gradient", "scale_flux_persistence"],
}


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (pd.Timestamp, pd.Period)):
        return str(x)
    return x


def causal_z(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    prior = s.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((s - mean) / (std + EPS)).clip(-12.0, 12.0)


def norm_momentum(r: pd.Series, window: int) -> pd.Series:
    numerator = r.rolling(window, min_periods=window).sum()
    denominator = np.sqrt(r.pow(2).rolling(window, min_periods=window).sum())
    return (numerator / (denominator + EPS)).clip(-12.0, 12.0)


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, minimum: int = 6) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < minimum:
        return np.nan
    if np.nanstd(a[mask]) < EPS or np.nanstd(b[mask]) < EPS:
        return np.nan
    return float(spearmanr(a[mask], b[mask]).statistic)


def block_bootstrap(values: np.ndarray, block: int, reps: int = 800) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max(20, 2 * block):
        return {"n": n, "mean": np.nan, "low": np.nan, "high": np.nan, "p_nonpositive": np.nan}
    starts = np.arange(n - block + 1)
    blocks = int(math.ceil(n / block))
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = RNG.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([x[j:j + block] for j in draw])[:n]
        means[i] = np.mean(sample)
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0.0)),
    }


def forward_open_return(g: pd.DataFrame, horizon: int, open_col: str) -> pd.Series:
    return np.log(g[open_col].shift(-(horizon + 1)) / g[open_col].shift(-1))


def _binance_url(market: str, symbol: str, period: pd.Period) -> str:
    root = "futures/um" if market == "perp" else "spot"
    return (
        f"https://data.binance.vision/data/{root}/monthly/klines/"
        f"{symbol}/1h/{symbol}-1h-{period.year:04d}-{period.month:02d}.zip"
    )


def download_binance_month(market: str, symbol: str, period: pd.Period) -> tuple[str, str, pd.Period, pd.DataFrame | None, str | None]:
    headers = {"User-Agent": "Trade-fluid-round9/20261118"}
    error = None
    for attempt in range(4):
        try:
            response = requests.get(_binance_url(market, symbol, period), timeout=45, headers=headers)
            if response.status_code == 404:
                return market, symbol, period, None, "404"
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not names:
                    return market, symbol, period, None, "missing_csv"
                raw = pd.read_csv(archive.open(names[0]), header=None, low_memory=False)
            if raw.shape[1] < 11:
                return market, symbol, period, None, f"short_schema_{raw.shape[1]}"
            raw = raw.iloc[:, :12]
            raw.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ]
            raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
            raw = raw.loc[raw["open_time"].notna()].copy()
            for col in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
            magnitude = float(raw["open_time"].median())
            unit = "us" if magnitude > 1.0e14 else "ms"
            raw["time"] = pd.to_datetime(raw["open_time"].astype("int64"), unit=unit, utc=True)
            raw["symbol"] = symbol
            raw = raw[[
                "time", "symbol", "open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_quote",
            ]]
            raw = raw.replace([np.inf, -np.inf], np.nan)
            raw = raw.dropna(subset=["time", "open", "high", "low", "close"])
            raw = raw.loc[(raw[["open", "high", "low", "close"]] > 0).all(axis=1)]
            return market, symbol, period, raw, None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return market, symbol, period, None, error


def load_futures() -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = [
        (market, symbol, period)
        for market in ("spot", "perp")
        for symbol in FUTURES_SYMBOLS
        for period in MONTHS
    ]
    frames: dict[str, list[pd.DataFrame]] = {"spot": [], "perp": []}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        jobs = [pool.submit(download_binance_month, *task) for task in tasks]
        for count, future in enumerate(as_completed(jobs), start=1):
            market, symbol, period, frame, error = future.result()
            if frame is not None and not frame.empty:
                frames[market].append(frame)
            elif error != "404":
                errors.append({"market": market, "symbol": symbol, "month": str(period), "error": str(error)})
            if count % 160 == 0:
                print(f"binance={count}/{len(tasks)} spot={len(frames['spot'])} perp={len(frames['perp'])}", flush=True)
    if not frames["spot"] or not frames["perp"]:
        raise RuntimeError("spot or perpetual downloads are empty")
    spot = pd.concat(frames["spot"], ignore_index=True)
    perp = pd.concat(frames["perp"], ignore_index=True)
    spot = spot.drop_duplicates(["symbol", "time"], keep="last")
    perp = perp.drop_duplicates(["symbol", "time"], keep="last")
    spot = spot.rename(columns={c: f"s_{c}" for c in spot.columns if c not in {"symbol", "time"}})
    perp = perp.rename(columns={c: f"p_{c}" for c in perp.columns if c not in {"symbol", "time"}})
    panel = perp.merge(spot, on=["symbol", "time"], how="inner", validate="one_to_one")
    panel = panel.sort_values(["symbol", "time"]).reset_index(drop=True)
    expected = len(pd.date_range("2021-01-01", "2026-10-31 23:00", freq="h", tz="UTC"))
    counts = panel.groupby("symbol").size()
    retained = counts[counts >= 0.88 * expected].index.tolist()
    panel = panel.loc[panel["symbol"].isin(retained)].copy()
    if len(retained) < 6:
        raise RuntimeError(f"only {len(retained)} complete coupled symbols")
    metadata = {
        "requested_symbols": FUTURES_SYMBOLS,
        "retained_symbols": retained,
        "rows": int(len(panel)),
        "min_time": str(panel["time"].min()),
        "max_time": str(panel["time"].max()),
        "spot_rows": int(len(spot)),
        "perp_rows": int(len(perp)),
        "errors": errors[:100],
    }
    return panel, metadata


def compute_futures_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("time").reset_index(drop=True).copy()
    p_r = np.log(g["p_close"]).diff()
    s_r = np.log(g["s_close"]).diff()
    p_scale = p_r.shift(1).rolling(48, min_periods=24).std(ddof=0)
    s_scale = s_r.shift(1).rolling(48, min_periods=24).std(ddof=0)
    p_u = (p_r / (p_scale + EPS)).clip(-12.0, 12.0)
    s_u = (s_r / (s_scale + EPS)).clip(-12.0, 12.0)
    g["p_u"], g["s_u"] = p_u, s_u
    for window in [4, 24, 72]:
        g[f"p_mom{window}"] = norm_momentum(p_r, window)
        g[f"s_mom{window}"] = norm_momentum(s_r, window)
    g["p_qz"] = causal_z(np.log1p(g["p_quote_volume"].clip(lower=0.0)), 168, 72)
    g["s_qz"] = causal_z(np.log1p(g["s_quote_volume"].clip(lower=0.0)), 168, 72)
    p_imbalance = (2.0 * g["p_taker_buy_quote"] / (g["p_quote_volume"] + EPS) - 1.0).clip(-1.0, 1.0)
    s_imbalance = (2.0 * g["s_taker_buy_quote"] / (g["s_quote_volume"] + EPS) - 1.0).clip(-1.0, 1.0)
    g["p_imb_z"] = causal_z(p_imbalance, 168, 72)
    g["s_imb_z"] = causal_z(s_imbalance, 168, 72)
    p_range = np.log(g["p_high"] / g["p_low"]).clip(lower=0.0)
    s_range = np.log(g["s_high"] / g["s_low"]).clip(lower=0.0)
    g["p_range_z"] = causal_z(p_range, 336, 120)
    g["s_range_z"] = causal_z(s_range, 336, 120)
    p_rv = np.sqrt(p_r.pow(2).rolling(24, min_periods=18).mean())
    s_rv = np.sqrt(s_r.pow(2).rolling(24, min_periods=18).mean())
    g["p_rv_z"] = causal_z(np.log(p_rv + EPS), 336, 120)
    g["s_rv_z"] = causal_z(np.log(s_rv + EPS), 336, 120)

    basis = np.log(g["p_close"] / g["s_close"])
    basis_velocity = basis.diff()
    basis_acceleration = basis_velocity.diff()
    raw_shear = p_u - s_u
    raw_activity_gap = g["p_qz"] - g["s_qz"]
    raw_flow_gap = g["p_imb_z"] - g["s_imb_z"]
    g["basis_z"] = causal_z(basis, 336, 120)
    g["basis_velocity_z"] = causal_z(basis_velocity, 336, 120)
    g["basis_acceleration_z"] = causal_z(basis_acceleration, 336, 120)
    g["shear_z"] = causal_z(raw_shear, 336, 120)
    g["activity_gap_z"] = causal_z(raw_activity_gap, 336, 120)
    g["flow_gap_z"] = causal_z(raw_flow_gap, 336, 120)

    flux = -g["shear_z"] * g["activity_gap_z"] * g["basis_velocity_z"]
    g["interfacial_energy_flux"] = causal_z(flux, 720, 240)
    relaxation = -g["shear_z"] * (1.0 + g["activity_gap_z"].abs()) * np.exp(-g["basis_z"].abs())
    g["shear_relaxation"] = causal_z(relaxation, 720, 240)
    drive = g["shear_z"] * np.tanh(g["activity_gap_z"]) * np.exp(-g["basis_velocity_z"].abs())
    g["shear_drive"] = causal_z(drive, 720, 240)

    kh_excess = (g["shear_z"].abs() - 0.75 * g["basis_z"].abs()).clip(lower=0.0)
    g["kh_onset"] = causal_z(np.sign(g["shear_z"]) * kh_excess * (1.0 + g["flow_gap_z"].abs()), 720, 240)
    g["kh_release"] = causal_z(-np.sign(g["shear_z"]) * kh_excess * g["basis_acceleration_z"].abs(), 720, 240)

    circulation = g["shear_z"].shift(1) * g["flow_gap_z"] - g["flow_gap_z"].shift(1) * g["shear_z"]
    g["interfacial_vorticity_6"] = causal_z(circulation.rolling(6, min_periods=4).mean(), 720, 240)
    g["interfacial_vorticity_24"] = causal_z(circulation.rolling(24, min_periods=18).mean(), 720, 240)

    continuity = g["basis_velocity_z"] - 0.50 * g["shear_z"] - 0.25 * g["activity_gap_z"]
    g["continuity_residual"] = causal_z(continuity, 720, 240)
    pressure = g["basis_acceleration_z"] - 0.50 * g["shear_z"].diff() + 0.25 * g["flow_gap_z"]
    g["pressure_release"] = causal_z(-pressure * (1.0 + g["shear_z"].abs()), 720, 240)
    g["pressure_drive"] = causal_z(pressure * np.tanh(g["activity_gap_z"]), 720, 240)

    g["spot_sweeping_lead"] = causal_z((s_u - p_u.shift(1)) * (1.0 + g["s_qz"].clip(lower=0.0)), 720, 240)
    g["interfacial_impedance"] = causal_z(g["shear_z"] / (1.0 + g["activity_gap_z"].abs() + g["flow_gap_z"].abs()), 720, 240)
    g["impedance_mismatch"] = causal_z(g["shear_z"] * (g["activity_gap_z"] - g["flow_gap_z"]), 720, 240)

    x, y, z = g["shear_z"], g["activity_gap_z"], g["basis_z"]
    vx, vy, vz = x.diff(), y.diff(), z.diff()
    ax, ay, az = vx.diff(), vy.diff(), vz.diff()
    jx, jy, jz = ax.diff(), ay.diff(), az.diff()
    cx = vy * az - vz * ay
    cy = vz * ax - vx * az
    cz = vx * ay - vy * ax
    triple = cx * jx + cy * jy + cz * jz
    denominator = np.sqrt(cx.pow(2) + cy.pow(2) + cz.pow(2)) * np.sqrt(jx.pow(2) + jy.pow(2) + jz.pow(2))
    torsion = (triple / (denominator + EPS)).clip(-1.0, 1.0)
    g["interfacial_torsion"] = causal_z(torsion.rolling(12, min_periods=8).mean(), 720, 240)

    for horizon in FUTURES_HORIZONS:
        g[f"y_{horizon}"] = forward_open_return(g, horizon, "p_open")
    return g


def download_stooq(ticker: str) -> tuple[str, pd.DataFrame | None, str | None]:
    symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1=20120101&d2=20261031&i=d"
    headers = {"User-Agent": "Mozilla/5.0 Trade-fluid-round9"}
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45, headers=headers)
            response.raise_for_status()
            if "No data" in response.text or len(response.text) < 100:
                raise ValueError("no_data")
            frame = pd.read_csv(io.StringIO(response.text))
            frame.columns = [str(c).lower() for c in frame.columns]
            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(frame.columns):
                raise ValueError(f"schema={list(frame.columns)}")
            frame = frame[list(required)].copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.dropna().sort_values("date")
            frame = frame.loc[(frame[["open", "high", "low", "close"]] > 0).all(axis=1) & (frame["volume"] > 0)]
            if len(frame) < 2000:
                raise ValueError(f"short_history={len(frame)}")
            frame["symbol"] = ticker
            frame["sector"] = STOCK_SECTORS[ticker]
            return ticker, frame, None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return ticker, None, error


def load_stocks() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        jobs = [pool.submit(download_stooq, ticker) for ticker in STOCK_SECTORS]
        for future in as_completed(jobs):
            ticker, frame, error = future.result()
            if frame is not None:
                frames.append(frame)
            else:
                errors.append({"ticker": ticker, "error": str(error)})
    if len(frames) < 35:
        raise RuntimeError(f"only {len(frames)} usable stock histories")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    metadata = {
        "requested": sorted(STOCK_SECTORS),
        "retained": sorted(panel["symbol"].unique().tolist()),
        "rows": int(len(panel)),
        "min_date": str(panel["date"].min()),
        "max_date": str(panel["date"].max()),
        "errors": errors,
        "universe_warning": "fixed current large-cap universe; survivorship selection remains",
    }
    return panel, metadata


def compute_stock_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("date").reset_index(drop=True).copy()
    r = np.log(g["close"]).diff()
    prior_vol = r.shift(1).rolling(20, min_periods=12).std(ddof=0)
    u = (r / (prior_vol + EPS)).clip(-12.0, 12.0)
    dollar_volume = g["close"] * g["volume"]
    intraday_range = np.log(g["high"] / g["low"]).clip(lower=0.0)
    location = ((2.0 * g["close"] - g["high"] - g["low"]) / (g["high"] - g["low"] + EPS)).clip(-1.0, 1.0)
    g["r1"], g["u"] = r, u
    for window in [5, 20, 60, 120]:
        g[f"mom{window}"] = norm_momentum(r, window)
    g["vol20_z"] = causal_z(np.log(prior_vol + EPS), 252, 100)
    g["qz"] = causal_z(np.log1p(dollar_volume), 252, 100)
    g["range_z"] = causal_z(intraday_range, 252, 100)
    g["close_location"] = location
    g["size_proxy"] = causal_z(np.log1p(dollar_volume.rolling(20, min_periods=15).mean()), 504, 200)
    g["acceleration"] = u.diff()
    g["signed_activity"] = location * np.tanh(g["qz"] / 2.0)
    high20 = g["high"].rolling(20, min_periods=15).max()
    low20 = g["low"].rolling(20, min_periods=15).min()
    scale20 = intraday_range.rolling(20, min_periods=15).mean()
    direction = np.sign(g["mom20"]).replace(0.0, np.nan)
    distance = pd.Series(
        np.where(direction >= 0.0, np.log(high20 / g["close"]), np.log(g["close"] / low20)),
        index=g.index,
    ).clip(lower=0.0)
    g["boundary_proximity"] = np.exp(-distance / (scale20 + EPS)).clip(0.0, 1.0)
    g["flow5"] = g["signed_activity"].rolling(5, min_periods=4).mean()
    g["flow20"] = g["signed_activity"].rolling(20, min_periods=15).mean()
    for horizon in STOCK_HORIZONS:
        g[f"y_{horizon}"] = forward_open_return(g, horizon, "open")
    return g


def add_stock_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["date", "symbol"]).copy()
    peer_columns = ["u", "qz", "range_z", "mom5", "mom20", "acceleration", "signed_activity", "flow5", "flow20"]
    for col in peer_columns:
        sector_sum = p.groupby(["date", "sector"])[col].transform("sum")
        sector_count = p.groupby(["date", "sector"])[col].transform("count")
        p[f"sector_peer_{col}"] = (sector_sum - p[col]) / (sector_count - 1).clip(lower=1)
        market_sum = p.groupby("date")[col].transform("sum")
        market_count = p.groupby("date")[col].transform("count")
        p[f"market_{col}"] = (market_sum - p[col]) / (market_count - 1).clip(lower=1)
    p["sector_peer_u"] = p["sector_peer_u"]
    p["market_u"] = p["market_u"]
    p["mom5_gap"] = p["mom5"] - p["sector_peer_mom5"]
    p["mom20_gap"] = p["mom20"] - p["sector_peer_mom20"]
    p["shear"] = p["u"] - p["sector_peer_u"]
    p["activity_gap"] = p["qz"] - p["sector_peer_qz"]
    p["range_gap"] = p["range_z"] - p["sector_peer_range_z"]
    p["acceleration_gap"] = p["acceleration"] - p["sector_peer_acceleration"]
    p["flow_gap"] = p["signed_activity"] - p["sector_peer_signed_activity"]
    p["pressure"] = p["acceleration_gap"] - 0.50 * p["shear"]

    p["sector_shear_absorption"] = -p["shear"] * p["activity_gap"].abs() * np.exp(-p["mom20_gap"].abs())
    p["sector_shear_drive"] = p["shear"] * np.tanh(p["activity_gap"])
    p["sector_shear_curvature"] = -p.groupby("symbol")["shear"].diff() * (1.0 + p["range_gap"].abs())
    p["pressure_reversal"] = -p["pressure"] * (1.0 + p["range_gap"].abs())
    p["pressure_flow"] = -p["pressure"] * np.tanh(p["activity_gap"])

    lag_shear = p.groupby("symbol")["shear"].shift(1)
    lag_activity = p.groupby("symbol")["activity_gap"].shift(1)
    circulation = lag_shear * p["activity_gap"] - lag_activity * p["shear"]
    p["sector_vorticity"] = circulation
    p["sector_vorticity_release"] = -circulation * np.sign(p["mom20_gap"])

    unconverted = (p["activity_gap"].abs() - p["shear"].abs()).clip(lower=0.0)
    p["source_sink_absorption"] = p["flow_gap"] * unconverted
    p["source_sink_release"] = -p["shear"] * unconverted * p["range_gap"].abs()
    p["sector_wake_catchup"] = -p["mom5_gap"] * np.exp(-p["activity_gap"].abs()) * (1.0 + p["sector_peer_mom20"].abs())
    p["leader_wake_exhaustion"] = -p["mom20_gap"] * p["activity_gap"].clip(lower=0.0) * p["range_gap"].clip(lower=0.0)
    p["drag_efficiency"] = p["mom20_gap"] / (1.0 + p["activity_gap"].abs() + p["range_gap"].abs())
    p["drag_breakdown"] = -p["mom20_gap"] * p["range_gap"].abs() / (1.0 + p["activity_gap"].abs())

    direction = np.sign(p["mom20"]).replace(0.0, np.nan)
    separation = (direction * (p["flow20"] - p["flow5"])).clip(lower=0.0)
    adverse = (-direction * p["flow5"]).clip(lower=0.0)
    p["boundary_layer_separation"] = -direction * p["boundary_proximity"] * separation * (1.0 + adverse)
    previous_flow = p.groupby("symbol")["flow5"].shift(5)
    reattachment = (-direction * previous_flow).clip(lower=0.0) * (direction * p["flow5"]).clip(lower=0.0)
    p["boundary_layer_reattachment"] = direction * p["boundary_proximity"] * reattachment
    p["scale_flux_gradient"] = -(p["mom5_gap"].pow(3) - p["mom20_gap"].pow(3))
    p["scale_flux_persistence"] = -p["mom20_gap"] * np.sign(p["mom5_gap"] * p["mom20_gap"]) * np.minimum(p["mom5_gap"].abs(), p["mom20_gap"].abs())

    candidates = sorted({factor for factors in STOCK_FAMILIES.values() for factor in factors})
    p = p.sort_values(["symbol", "date"])
    for factor in candidates:
        p[factor] = p.groupby("symbol", group_keys=False)[factor].transform(lambda s: causal_z(s, 252, 100))
    dates = pd.Index(sorted(p["date"].dropna().unique()))
    date_index = pd.Series(np.arange(len(dates)), index=dates)
    p["session_index"] = p["date"].map(date_index).astype("Int64")
    return p


def split_label(values: pd.Series, kind: str) -> pd.Series:
    years = pd.to_datetime(values).dt.year
    if kind == "futures":
        return pd.Series(np.where(years <= 2022, "development", np.where(years <= 2024, "validation", "holdout")), index=values.index)
    return pd.Series(np.where(years <= 2020, "development", np.where(years <= 2023, "validation", "holdout")), index=values.index)


def issue_subset(frame: pd.DataFrame, horizon: int, kind: str) -> pd.DataFrame:
    if kind == "futures":
        hours = ((pd.to_datetime(frame["time"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(hours=1)).astype("int64")
        return frame.loc[(hours % horizon) == 0]
    return frame.loc[(frame["session_index"].astype(float) % horizon) == 0]


def time_series_ic(frame: pd.DataFrame, factor: str, target: str) -> dict[str, Any]:
    by_asset = {symbol: safe_spearman(g[factor], g[target], 20) for symbol, g in frame.groupby("symbol")}
    values = np.array([v for v in by_asset.values() if np.isfinite(v)], dtype=float)
    return {
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "median": float(np.median(values)) if len(values) else np.nan,
        "positive_share": float(np.mean(values > 0.0)) if len(values) else np.nan,
        "n_assets": int(len(values)),
        "by_asset": by_asset,
    }


def cross_sectional_ic(frame: pd.DataFrame, factor: str, target: str, time_col: str, minimum_assets: int) -> tuple[dict[str, Any], np.ndarray]:
    values = []
    for _, group in frame.groupby(time_col):
        z = group[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) >= minimum_assets:
            value = safe_spearman(z[factor], z[target], minimum_assets)
            if np.isfinite(value):
                values.append(value)
    array = np.asarray(values, dtype=float)
    return ({
        "mean": float(np.mean(array)) if len(array) else np.nan,
        "median": float(np.median(array)) if len(array) else np.nan,
        "positive_share": float(np.mean(array > 0.0)) if len(array) else np.nan,
        "n_times": int(len(array)),
    }, array)


def standard_matrix(frame: pd.DataFrame, columns: list[str], mean: pd.Series | None = None, std: pd.Series | None = None) -> tuple[np.ndarray, pd.Series, pd.Series]:
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    if mean is None:
        mean = x.mean()
        std = x.std(ddof=0).replace(0.0, 1.0)
    matrix = ((x - mean) / std).fillna(0.0).clip(-12.0, 12.0).to_numpy(dtype=float)
    return matrix, mean, std


def ridge_increment(train: pd.DataFrame, test: pd.DataFrame, baseline: list[str], factor: str, target: str, maximum_train: int) -> dict[str, Any]:
    train = train[baseline + [factor, target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[factor, target])
    test = test[baseline + [factor, target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[factor, target])
    if len(train) > maximum_train:
        train = train.sample(maximum_train, random_state=SEED)
    if len(train) < 1000 or len(test) < 500:
        return {"gain": np.nan, "n_train": len(train), "n_test": len(test)}
    xb, mean, std = standard_matrix(train, baseline)
    xbt, _, _ = standard_matrix(test, baseline, mean, std)
    xf, fmean, fstd = standard_matrix(train, [factor])
    xft, _, _ = standard_matrix(test, [factor], fmean, fstd)
    y = train[target].to_numpy(dtype=float)
    yt = test[target].to_numpy(dtype=float)
    base = Ridge(alpha=25.0).fit(xb, y)
    plus = Ridge(alpha=25.0).fit(np.column_stack([xb, xf]), y)
    base_error = (yt - base.predict(xbt)) ** 2
    plus_error = (yt - plus.predict(np.column_stack([xbt, xft]))) ** 2
    return {
        "gain": float((np.mean(base_error) - np.mean(plus_error)) / (np.mean(base_error) + EPS)),
        "mse_base": float(np.mean(base_error)),
        "mse_plus": float(np.mean(plus_error)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }


def residual_values(train: pd.DataFrame, test: pd.DataFrame, baseline: list[str], factor: str) -> pd.Series:
    train = train[baseline + [factor]].replace([np.inf, -np.inf], np.nan).dropna(subset=[factor])
    test_data = test[baseline + [factor]].replace([np.inf, -np.inf], np.nan)
    xb, mean, std = standard_matrix(train, baseline)
    xbt, _, _ = standard_matrix(test_data, baseline, mean, std)
    model = Ridge(alpha=25.0).fit(xb, train[factor].to_numpy(dtype=float))
    values = test_data[factor].to_numpy(dtype=float) - model.predict(xbt)
    values[~np.isfinite(test_data[factor].to_numpy(dtype=float))] = np.nan
    return pd.Series(values, index=test_data.index)


def maximum_baseline_correlation(frame: pd.DataFrame, factor: str, baseline: list[str]) -> dict[str, Any]:
    correlations = {column: safe_spearman(frame[factor], frame[column], 50) for column in baseline}
    finite = {column: value for column, value in correlations.items() if np.isfinite(value)}
    if not finite:
        return {"max_abs": np.nan, "feature": None, "correlations": correlations}
    feature = max(finite, key=lambda c: abs(finite[c]))
    return {"max_abs": float(abs(finite[feature])), "feature": feature, "correlations": correlations}


def rank_portfolio(frame: pd.DataFrame, factor: str, target: str, time_col: str, cost_bp: float, annual_periods: float, minimum_assets: int) -> dict[str, Any]:
    previous: dict[str, float] = {}
    returns, turns, times = [], [], []
    for timestamp, group in frame.groupby(time_col, sort=True):
        z = group[["symbol", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < minimum_assets:
            continue
        ranks = z[factor].rank(pct=True) - 0.5
        denominator = float(ranks.abs().sum())
        if denominator < EPS:
            continue
        weights = dict(zip(z["symbol"], (ranks / denominator).to_numpy(dtype=float)))
        target_map = dict(zip(z["symbol"], z[target].to_numpy(dtype=float)))
        gross = float(sum(weights[symbol] * target_map[symbol] for symbol in weights))
        names = set(previous) | set(weights)
        turnover = float(sum(abs(weights.get(name, 0.0) - previous.get(name, 0.0)) for name in names))
        returns.append(gross - cost_bp / 10000.0 * turnover)
        turns.append(turnover)
        times.append(timestamp)
        previous = weights
    values = np.asarray(returns, dtype=float)
    if len(values) < 20:
        return {"n": int(len(values)), "sharpe": np.nan, "annual_log_return": np.nan, "avg_turnover": np.nan, "years": {}}
    years = pd.to_datetime(pd.Series(times)).dt.year.to_numpy()
    yearly = {str(year): float(values[years == year].sum()) for year in np.unique(years)}
    return {
        "n": int(len(values)),
        "sharpe": float(np.mean(values) / (np.std(values, ddof=0) + EPS) * math.sqrt(annual_periods)),
        "annual_log_return": float(np.mean(values) * annual_periods),
        "avg_turnover": float(np.mean(turns)),
        "years": yearly,
    }


def select_families(panel: pd.DataFrame, families: dict[str, list[str]], horizons: list[int], kind: str) -> pd.DataFrame:
    development = panel.loc[panel["split"] == "development"]
    records = []
    for family, candidates in families.items():
        best: tuple[float, dict[str, Any]] | None = None
        for factor in candidates:
            for horizon in horizons:
                sample = issue_subset(development, horizon, kind)
                target = f"y_{horizon}"
                if kind == "futures":
                    ts = time_series_ic(sample, factor, target)
                    cs, _ = cross_sectional_ic(sample, factor, target, "time", 6)
                    raw = 0.70 * ts["mean"] + 0.30 * cs["mean"]
                    sign = 1.0 if raw >= 0.0 else -1.0
                    score = abs(raw) * math.sqrt(max(1, ts["n_assets"]))
                    record = {
                        "family": family, "factor": factor, "horizon": horizon, "sign": sign,
                        "development_ts_ic": sign * ts["mean"],
                        "development_cs_ic": sign * cs["mean"],
                        "selection_score": score,
                    }
                else:
                    cs, _ = cross_sectional_ic(sample, factor, target, "date", 20)
                    sign = 1.0 if cs["mean"] >= 0.0 else -1.0
                    score = abs(cs["mean"]) * math.sqrt(max(1, cs["n_times"]))
                    record = {
                        "family": family, "factor": factor, "horizon": horizon, "sign": sign,
                        "development_cs_ic": sign * cs["mean"],
                        "development_cs_positive_share": cs["positive_share"],
                        "selection_score": score,
                    }
                if best is None or score > best[0]:
                    best = (score, record)
        if best is not None:
            records.append(best[1])
    return pd.DataFrame(records)


def validate_futures(panel: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    development = panel.loc[panel["split"] == "development"].copy()
    validation_all = panel.loc[panel["split"] == "validation"].copy()
    holdout_all = panel.loc[panel["split"] == "holdout"].copy()
    validation_records, holdout_records, promoted = [], [], []
    for selection in selections.to_dict("records"):
        factor = selection["factor"]
        horizon = int(selection["horizon"])
        signed = f"signed_{factor}"
        sign = float(selection["sign"])
        development[signed] = sign * development[factor]
        validation_all[signed] = sign * validation_all[factor]
        holdout_all[signed] = sign * holdout_all[factor]
        train = issue_subset(development, horizon, "futures")
        validation = issue_subset(validation_all, horizon, "futures").copy()
        target = f"y_{horizon}"
        ts = time_series_ic(validation, signed, target)
        cs, cs_values = cross_sectional_ic(validation, signed, target, "time", 6)
        bootstrap = block_bootstrap(cs_values, max(5, int(120 / horizon)))
        ridge = ridge_increment(train, validation, FUTURES_BASE, signed, target, 180000)
        validation["residual"] = residual_values(train, validation, FUTURES_BASE, signed)
        residual_ts = time_series_ic(validation, "residual", target)
        residual_cs, _ = cross_sectional_ic(validation, "residual", target, "time", 6)
        correlation = maximum_baseline_correlation(validation, signed, FUTURES_BASE)
        portfolio = rank_portfolio(validation, signed, target, "time", 2.0, 365.0 * 24.0 / horizon, 6)
        passes = bool(
            ts["mean"] > 0.010 and ts["positive_share"] >= 0.625 and
            cs["mean"] > 0.008 and bootstrap["low"] > -0.003 and
            ridge["gain"] > 0.0 and residual_ts["mean"] > 0.002 and
            residual_cs["mean"] > 0.0 and portfolio["sharpe"] > 0.0 and
            correlation["max_abs"] < 0.80
        )
        validation_records.append({
            **selection,
            "validation_pass": passes,
            "validation_ts_ic": ts["mean"],
            "validation_ts_positive_share": ts["positive_share"],
            "validation_cs_ic": cs["mean"],
            "validation_cs_positive_share": cs["positive_share"],
            "validation_cs_boot_low": bootstrap["low"],
            "validation_cs_boot_high": bootstrap["high"],
            "validation_ridge_gain": ridge["gain"],
            "validation_residual_ts_ic": residual_ts["mean"],
            "validation_residual_cs_ic": residual_cs["mean"],
            "validation_max_baseline_corr": correlation["max_abs"],
            "validation_max_corr_feature": correlation["feature"],
            "validation_net_sharpe_2bp": portfolio["sharpe"],
            "validation_net_annual_log_return_2bp": portfolio["annual_log_return"],
            "validation_avg_turnover": portfolio["avg_turnover"],
            "validation_years": json.dumps(portfolio["years"], sort_keys=True),
        })
        if not passes:
            continue
        holdout = issue_subset(holdout_all, horizon, "futures").copy()
        combined_train = pd.concat([train, validation], ignore_index=True)
        ts_h = time_series_ic(holdout, signed, target)
        cs_h, cs_h_values = cross_sectional_ic(holdout, signed, target, "time", 6)
        boot_h = block_bootstrap(cs_h_values, max(5, int(120 / horizon)))
        ridge_h = ridge_increment(combined_train, holdout, FUTURES_BASE, signed, target, 220000)
        holdout["residual"] = residual_values(combined_train, holdout, FUTURES_BASE, signed)
        residual_h = time_series_ic(holdout, "residual", target)
        residual_cs_h, _ = cross_sectional_ic(holdout, "residual", target, "time", 6)
        portfolio_h = rank_portfolio(holdout, signed, target, "time", 2.0, 365.0 * 24.0 / horizon, 6)
        positive_years = sum(value > 0.0 for value in portfolio_h["years"].values())
        final_pass = bool(
            ts_h["mean"] > 0.006 and cs_h["mean"] > 0.003 and
            boot_h["mean"] > 0.0 and ridge_h["gain"] > 0.0 and
            residual_h["mean"] > 0.0 and residual_cs_h["mean"] > 0.0 and
            portfolio_h["sharpe"] > 0.0 and
            positive_years >= max(1, len(portfolio_h["years"]) - 1)
        )
        record = {
            **selection,
            "holdout_pass": final_pass,
            "holdout_ts_ic": ts_h["mean"],
            "holdout_ts_positive_share": ts_h["positive_share"],
            "holdout_cs_ic": cs_h["mean"],
            "holdout_boot_low": boot_h["low"],
            "holdout_ridge_gain": ridge_h["gain"],
            "holdout_residual_ts_ic": residual_h["mean"],
            "holdout_residual_cs_ic": residual_cs_h["mean"],
            "holdout_net_sharpe_2bp": portfolio_h["sharpe"],
            "holdout_net_annual_log_return_2bp": portfolio_h["annual_log_return"],
            "holdout_years": json.dumps(portfolio_h["years"], sort_keys=True),
        }
        holdout_records.append(record)
        if final_pass:
            promoted.append({"asset_class": "crypto_perpetual", **record})
    return pd.DataFrame(validation_records), pd.DataFrame(holdout_records), promoted


def validate_stocks(panel: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    development = panel.loc[panel["split"] == "development"].copy()
    validation_all = panel.loc[panel["split"] == "validation"].copy()
    holdout_all = panel.loc[panel["split"] == "holdout"].copy()
    validation_records, holdout_records, promoted = [], [], []
    for selection in selections.to_dict("records"):
        factor = selection["factor"]
        horizon = int(selection["horizon"])
        sign = float(selection["sign"])
        signed = f"signed_{factor}"
        development[signed] = sign * development[factor]
        validation_all[signed] = sign * validation_all[factor]
        holdout_all[signed] = sign * holdout_all[factor]
        train = issue_subset(development, horizon, "stocks")
        validation = issue_subset(validation_all, horizon, "stocks").copy()
        target = f"y_{horizon}"
        cs, values = cross_sectional_ic(validation, signed, target, "date", 20)
        bootstrap = block_bootstrap(values, max(5, int(20 / horizon)))
        ridge = ridge_increment(train, validation, STOCK_BASE, signed, target, 160000)
        validation["residual"] = residual_values(train, validation, STOCK_BASE, signed)
        residual_cs, _ = cross_sectional_ic(validation, "residual", target, "date", 20)
        correlation = maximum_baseline_correlation(validation, signed, STOCK_BASE)
        portfolio = rank_portfolio(validation, signed, target, "date", 10.0, 252.0 / horizon, 20)
        passes = bool(
            cs["mean"] > 0.012 and cs["positive_share"] >= 0.54 and
            bootstrap["low"] > 0.0 and ridge["gain"] > 0.0 and
            residual_cs["mean"] > 0.004 and portfolio["sharpe"] > 0.15 and
            correlation["max_abs"] < 0.80
        )
        validation_records.append({
            **selection,
            "validation_pass": passes,
            "validation_cs_ic": cs["mean"],
            "validation_cs_positive_share": cs["positive_share"],
            "validation_boot_low": bootstrap["low"],
            "validation_boot_high": bootstrap["high"],
            "validation_ridge_gain": ridge["gain"],
            "validation_residual_cs_ic": residual_cs["mean"],
            "validation_max_baseline_corr": correlation["max_abs"],
            "validation_max_corr_feature": correlation["feature"],
            "validation_net_sharpe_10bp": portfolio["sharpe"],
            "validation_net_annual_log_return_10bp": portfolio["annual_log_return"],
            "validation_avg_turnover": portfolio["avg_turnover"],
            "validation_years": json.dumps(portfolio["years"], sort_keys=True),
        })
        if not passes:
            continue
        holdout = issue_subset(holdout_all, horizon, "stocks").copy()
        combined_train = pd.concat([train, validation], ignore_index=True)
        cs_h, values_h = cross_sectional_ic(holdout, signed, target, "date", 20)
        boot_h = block_bootstrap(values_h, max(5, int(20 / horizon)))
        ridge_h = ridge_increment(combined_train, holdout, STOCK_BASE, signed, target, 180000)
        holdout["residual"] = residual_values(combined_train, holdout, STOCK_BASE, signed)
        residual_h, _ = cross_sectional_ic(holdout, "residual", target, "date", 20)
        portfolio_h = rank_portfolio(holdout, signed, target, "date", 10.0, 252.0 / horizon, 20)
        positive_years = sum(value > 0.0 for value in portfolio_h["years"].values())
        final_pass = bool(
            cs_h["mean"] > 0.008 and boot_h["low"] >= -0.002 and
            ridge_h["gain"] > 0.0 and residual_h["mean"] > 0.003 and
            portfolio_h["sharpe"] > 0.10 and
            positive_years >= max(1, len(portfolio_h["years"]) - 1)
        )
        record = {
            **selection,
            "holdout_pass": final_pass,
            "holdout_cs_ic": cs_h["mean"],
            "holdout_boot_low": boot_h["low"],
            "holdout_boot_high": boot_h["high"],
            "holdout_ridge_gain": ridge_h["gain"],
            "holdout_residual_cs_ic": residual_h["mean"],
            "holdout_net_sharpe_10bp": portfolio_h["sharpe"],
            "holdout_net_annual_log_return_10bp": portfolio_h["annual_log_return"],
            "holdout_years": json.dumps(portfolio_h["years"], sort_keys=True),
        }
        holdout_records.append(record)
        if final_pass:
            promoted.append({"asset_class": "equities", **record})
    return pd.DataFrame(validation_records), pd.DataFrame(holdout_records), promoted


def save_results(
    summary: dict[str, Any],
    futures_selection: pd.DataFrame,
    futures_validation: pd.DataFrame,
    futures_holdout: pd.DataFrame,
    stock_selection: pd.DataFrame,
    stock_validation: pd.DataFrame,
    stock_holdout: pd.DataFrame,
) -> None:
    futures_selection.to_csv(OUT / "FUTURES_SELECTION.csv", index=False)
    futures_validation.to_csv(OUT / "FUTURES_VALIDATION.csv", index=False)
    futures_holdout.to_csv(OUT / "FUTURES_HOLDOUT.csv", index=False)
    stock_selection.to_csv(OUT / "STOCK_SELECTION.csv", index=False)
    stock_validation.to_csv(OUT / "STOCK_VALIDATION.csv", index=False)
    stock_holdout.to_csv(OUT / "STOCK_HOLDOUT.csv", index=False)
    (OUT / "SUMMARY.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(jsonable(summary["promoted"]), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Fluid Mechanics Round 9: Coupled Layers and Sector Flow",
        "",
        "Real public data; family-level development, validation and conditional holdout protocol.",
        "Holdout metrics are computed only for validation passers.",
        "",
        "## Futures selections", "", futures_selection.to_markdown(index=False), "",
        "## Futures validation", "", futures_validation.to_markdown(index=False), "",
        "## Futures conditional holdout", "", futures_holdout.to_markdown(index=False) if len(futures_holdout) else "No family passed validation.", "",
        "## Stock selections", "", stock_selection.to_markdown(index=False), "",
        "## Stock validation", "", stock_validation.to_markdown(index=False), "",
        "## Stock conditional holdout", "", stock_holdout.to_markdown(index=False) if len(stock_holdout) else "No family passed validation.", "",
        f"## Promoted: {len(summary['promoted'])}", "",
        "The stock universe is fixed from a current large-cap list and therefore retains survivorship-selection risk.",
        "Any promoted stock factor is historical/default-off until replicated on a point-in-time universe.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    futures_raw, futures_metadata = load_futures()
    futures_parts = []
    for symbol, frame in futures_raw.groupby("symbol", sort=False):
        computed = compute_futures_symbol(frame)
        computed["symbol"] = symbol
        futures_parts.append(computed)
    futures_panel = pd.concat(futures_parts, ignore_index=True)
    futures_panel["split"] = split_label(futures_panel["time"], "futures")
    futures_selection = select_families(futures_panel, FUTURES_FAMILIES, FUTURES_HORIZONS, "futures")
    futures_validation, futures_holdout, futures_promoted = validate_futures(futures_panel, futures_selection)

    stocks_raw, stocks_metadata = load_stocks()
    stock_parts = []
    for symbol, frame in stocks_raw.groupby("symbol", sort=False):
        computed = compute_stock_symbol(frame)
        computed["symbol"] = symbol
        computed["sector"] = STOCK_SECTORS[symbol]
        stock_parts.append(computed)
    stock_panel = add_stock_cross_section(pd.concat(stock_parts, ignore_index=True))
    stock_panel["split"] = split_label(stock_panel["date"], "stocks")
    stock_selection = select_families(stock_panel, STOCK_FAMILIES, STOCK_HORIZONS, "stocks")
    stock_validation, stock_holdout, stock_promoted = validate_stocks(stock_panel, stock_selection)

    promoted = futures_promoted + stock_promoted
    summary = {
        "seed": SEED,
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "elapsed_seconds": time.time() - started,
        "futures_data": futures_metadata,
        "stocks_data": stocks_metadata,
        "futures_validation_passers": int(futures_validation["validation_pass"].sum()) if len(futures_validation) else 0,
        "stock_validation_passers": int(stock_validation["validation_pass"].sum()) if len(stock_validation) else 0,
        "promoted": promoted,
        "protocol": {
            "futures": {"development": "2021-2022", "validation": "2023-2024", "holdout": "2025-2026-10", "cost_bp_one_way": 2.0},
            "stocks": {"development": "2012-2020", "validation": "2021-2023", "holdout": "2024-2026-10", "cost_bp_one_way": 10.0},
            "holdout_policy": "computed only after family validation pass",
        },
    }
    save_results(summary, futures_selection, futures_validation, futures_holdout, stock_selection, stock_validation, stock_holdout)
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(failure["traceback"], flush=True)
        raise
