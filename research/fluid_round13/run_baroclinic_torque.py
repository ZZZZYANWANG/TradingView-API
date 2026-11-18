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
from huggingface_hub import HfApi, hf_hub_download
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

SEED = 20261119
EPS = 1.0e-12
OUT = Path("research_outputs/fluid_baroclinic_torque_round13")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "BCHUSDT", "LINKUSDT", "ETCUSDT",
    "XLMUSDT", "DOTUSDT", "SOLUSDT",
]
MONTHS = list(pd.period_range("2021-01", "2026-09", freq="M"))
HORIZONS = [4, 12, 24, 72, 120, 240, 480]
SMOOTHING = [6, 12, 24, 72]
HF_REPO = "ismatsamadov/binance-usdt-archive"

FAMILIES: dict[str, list[str]] = {
    "baroclinic_torque": [
        "baroclinic_funding", "baroclinic_flow", "baroclinic_total",
    ],
    "potential_vorticity": [
        "pv_transport", "pv_source", "pv_pressure_release",
    ],
    "vortex_stretching": [
        "vortex_stretch", "enstrophy_release", "misalignment_energy",
    ],
    "kelvin_circulation": [
        "kelvin_residual", "kelvin_reconnection", "torque_circulation_alignment",
    ],
    "ekman_impedance": [
        "ekman_pumping", "impedance_reflection", "wave_action_gradient",
    ],
    "baroclinic_reattachment": [
        "baroclinic_reattach", "torque_contraction", "pressure_density_reconnect",
    ],
    "source_coupling": [
        "source_supported_torque", "deleveraging_torque_release",
    ],
}

BASE_FEATURES = [
    "u", "mom4", "mom12", "mom24", "mom72", "mom168", "mom480",
    "rv24_z", "rv72_z", "rv168_z", "range24_z", "qz", "trades_z",
    "imb6_z", "imb24_z", "skew24_z", "kurt24_z",
    "oi_level_z", "oi_source_z", "oi_value_source_z", "funding_z",
    "funding_change_z", "metric_taker_z", "crowding_z", "account_ratio_z",
    "peer_u", "div_u", "peer_rho", "grad_rho", "peer_pressure",
    "grad_pressure", "graph_circulation", "continuity_residual",
    "simple_density_pressure_alignment", "simple_source_velocity",
    "simple_source_funding", "simple_funding_velocity", "impedance_reflection_parent",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def causal_z(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    prior = s.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((s - mean) / (std + EPS)).clip(-12.0, 12.0)


def norm_momentum(u: pd.Series, window: int) -> pd.Series:
    num = u.rolling(window, min_periods=window).sum()
    den = np.sqrt(u.pow(2).rolling(window, min_periods=window).sum())
    return (num / (den + EPS)).clip(-12.0, 12.0)


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


def timestamp_from_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.to_datetime(numeric, utc=True, errors="coerce")
    unit = "us" if float(finite.abs().median()) > 1.0e15 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def download_kline_month(symbol: str, period: pd.Period) -> tuple[pd.DataFrame | None, str | None]:
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/1h/{symbol}-1h-{period.year:04d}-{period.month:02d}.zip"
    )
    headers = {"User-Agent": "Trade-fluid-baroclinic-round13/20261119"}
    error: str | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45, headers=headers)
            if response.status_code == 404:
                return None, "404"
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not names:
                    return None, "missing_csv"
                frame = pd.read_csv(archive.open(names[0]), header=None, low_memory=False)
            frame = frame.iloc[:, :12]
            frame.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ]
            frame["time"] = timestamp_from_numeric(frame["open_time"])
            for column in [
                "open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_quote",
            ]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
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


def load_klines() -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = [(symbol, period) for symbol in SYMBOLS for period in MONTHS]
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        jobs = {
            pool.submit(download_kline_month, symbol, period): (symbol, period)
            for symbol, period in tasks
        }
        for count, future in enumerate(as_completed(jobs), 1):
            symbol, period = jobs[future]
            frame, error = future.result()
            if frame is not None and not frame.empty:
                frames.append(frame)
            elif error != "404":
                errors.append({"symbol": symbol, "month": str(period), "error": str(error)})
            if count % 150 == 0:
                print(f"kline_parts={count}/{len(tasks)} usable={len(frames)}", flush=True)
    if not frames:
        raise RuntimeError("No Binance futures kline data downloaded")
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["symbol", "time"], keep="last")
    data = data.sort_values(["symbol", "time"]).reset_index(drop=True)
    audit = {
        "requested_parts": len(tasks),
        "usable_parts": len(frames),
        "rows": int(len(data)),
        "symbols": sorted(data["symbol"].unique().tolist()),
        "min_time": str(data["time"].min()),
        "max_time": str(data["time"].max()),
        "errors": errors[:100],
    }
    return data, audit


def resolve_hf_file(files: set[str], config: str, symbol: str) -> str:
    exact = f"{config}/{symbol}.parquet"
    if exact in files:
        return exact
    matches = [name for name in files if config in name and name.endswith(f"/{symbol}.parquet")]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot uniquely resolve {config}/{symbol}: {matches[:5]}")


def load_external() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    api = HfApi()
    info = api.dataset_info(HF_REPO)
    revision = str(info.sha)
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset", revision=revision))
    metrics: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.DataFrame] = {}
    resolved: dict[str, dict[str, str]] = {}
    for symbol in SYMBOLS:
        metric_name = resolve_hf_file(files, "um_metrics", symbol)
        funding_name = resolve_hf_file(files, "um_funding", symbol)
        resolved[symbol] = {"metrics": metric_name, "funding": funding_name}
        metric_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=metric_name,
            repo_type="dataset",
            revision=revision,
        )
        funding_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=funding_name,
            repo_type="dataset",
            revision=revision,
        )
        m = pd.read_parquet(metric_path)
        m.columns = [str(c).strip().lower() for c in m.columns]
        aliases = {
            "sum_open_interest": "open_interest",
            "sum_open_interest_value": "open_interest_value",
            "top_trader_account_ratio": "top_count_ratio",
            "count_toptrader_long_short_ratio": "top_count_ratio",
            "top_trader_position_ratio": "top_position_ratio",
            "sum_toptrader_long_short_ratio": "top_position_ratio",
            "count_long_short_ratio": "account_ratio",
            "taker_buy_sell_ratio": "taker_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ratio",
        }
        m = m.rename(columns=aliases)
        if "time" not in m:
            raise RuntimeError(f"metrics time missing for {symbol}: {m.columns.tolist()}")
        required = [
            "time", "open_interest", "open_interest_value", "top_count_ratio",
            "top_position_ratio", "account_ratio", "taker_ratio",
        ]
        missing = [c for c in required if c not in m]
        if missing:
            raise RuntimeError(f"metrics columns missing for {symbol}: {missing}")
        m["time"] = pd.to_datetime(m["time"], utc=True, errors="coerce")
        for column in required[1:]:
            m[column] = pd.to_numeric(m[column], errors="coerce")
        m = m.dropna(subset=["time", "open_interest", "open_interest_value"])
        m = (
            m.set_index("time")[required[1:]]
            .sort_index()
            .resample("1h")
            .last()
            .dropna(subset=["open_interest", "open_interest_value"])
            .reset_index()
        )
        metrics[symbol] = m

        f = pd.read_parquet(funding_path)
        f.columns = [str(c).strip().lower() for c in f.columns]
        f = f.rename(columns={"funding_time": "time", "last_funding_rate": "rate"})
        if "time" not in f or "rate" not in f:
            raise RuntimeError(f"funding columns missing for {symbol}: {f.columns.tolist()}")
        f["time"] = pd.to_datetime(f["time"], utc=True, errors="coerce")
        f["rate"] = pd.to_numeric(f["rate"], errors="coerce")
        f = f.dropna(subset=["time", "rate"]).sort_values("time")
        funding[symbol] = f[["time", "rate"]].drop_duplicates("time", keep="last")
        print(f"external_loaded={symbol} metrics={len(m)} funding={len(f)}", flush=True)
    audit = {
        "hf_repo": HF_REPO,
        "hf_revision": revision,
        "resolved_files": resolved,
        "metrics_rows": {symbol: int(len(frame)) for symbol, frame in metrics.items()},
        "funding_rows": {symbol: int(len(frame)) for symbol, frame in funding.items()},
    }
    return metrics, funding, audit


def merge_external(
    klines: pd.DataFrame,
    metrics: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    metric_columns = [
        "open_interest", "open_interest_value", "top_count_ratio",
        "top_position_ratio", "account_ratio", "taker_ratio",
    ]
    for symbol, group in klines.groupby("symbol", sort=False):
        left = group.sort_values("time").copy()
        m = metrics[symbol].sort_values("time")
        joined = pd.merge_asof(
            left, m, on="time", direction="backward", tolerance=pd.Timedelta(hours=2)
        )
        f = funding[symbol].rename(columns={"time": "funding_time"}).sort_values("funding_time")
        joined = pd.merge_asof(
            joined.sort_values("time"),
            f,
            left_on="time",
            right_on="funding_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=12),
        )
        joined = joined.sort_values("time").reset_index(drop=True)
        # Conservative availability: only observations known by the previous completed hour.
        for column in metric_columns + ["rate"]:
            joined[column] = joined[column].shift(1)
        joined["symbol"] = symbol
        parts.append(joined.drop(columns=["funding_time"], errors="ignore"))
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)


def compute_local_features(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("time").reset_index(drop=True).copy()
    log_close = np.log(g["close"])
    returns = log_close.diff()
    prior_scale = np.sqrt(
        returns.shift(1).pow(2).ewm(span=48, min_periods=24, adjust=False).mean()
    )
    u = (returns / (prior_scale + EPS)).clip(-12.0, 12.0)
    qz = causal_z(np.log1p(g["quote_volume"].clip(lower=0.0)), 336, 120)
    trades_z = causal_z(np.log1p(g["trades"].clip(lower=0.0)), 336, 120)
    imbalance = (
        2.0 * g["taker_buy_quote"] / (g["quote_volume"] + EPS) - 1.0
    ).clip(-1.0, 1.0)
    intrabar_range = np.log(g["high"] / g["low"]).clip(lower=0.0)

    g["u"] = u
    g["qz"] = qz
    g["trades_z"] = trades_z
    for window in [4, 12, 24, 72, 168, 480]:
        g[f"mom{window}"] = norm_momentum(u, window)
    for window in [24, 72, 168]:
        rv = np.sqrt(u.pow(2).rolling(window, min_periods=max(12, int(0.75 * window))).mean())
        g[f"rv{window}_z"] = causal_z(np.log(rv + EPS), 720, 240)
    g["range24_z"] = causal_z(
        intrabar_range.rolling(24, min_periods=18).mean(), 720, 240
    )
    g["imb6_z"] = causal_z(imbalance.rolling(6, min_periods=4).mean(), 720, 240)
    g["imb24_z"] = causal_z(imbalance.rolling(24, min_periods=18).mean(), 720, 240)
    g["skew24_z"] = causal_z(u.rolling(24, min_periods=18).skew(), 720, 240)
    g["kurt24_z"] = causal_z(u.rolling(24, min_periods=18).kurt(), 720, 240)

    oi = pd.to_numeric(g["open_interest"], errors="coerce").clip(lower=EPS)
    oi_value = pd.to_numeric(g["open_interest_value"], errors="coerce").clip(lower=EPS)
    funding_rate = pd.to_numeric(g["rate"], errors="coerce")
    metric_taker = pd.to_numeric(g["taker_ratio"], errors="coerce").clip(lower=EPS)
    top_position = pd.to_numeric(g["top_position_ratio"], errors="coerce").clip(lower=EPS)
    account_ratio = pd.to_numeric(g["account_ratio"], errors="coerce").clip(lower=EPS)

    g["oi_level_z"] = causal_z(np.log(oi_value), 720, 240)
    g["oi_source_z"] = causal_z(np.log(oi).diff(), 720, 240)
    g["oi_value_source_z"] = causal_z(np.log(oi_value).diff(), 720, 240)
    g["funding_z"] = causal_z(funding_rate, 720, 240)
    g["funding_change_z"] = causal_z(funding_rate.diff(), 720, 240)
    g["metric_taker_z"] = causal_z(np.log(metric_taker), 720, 240)
    g["crowding_z"] = causal_z(np.log(top_position), 720, 240)
    g["account_ratio_z"] = causal_z(np.log(account_ratio), 720, 240)
    g["flow_pressure"] = 0.5 * g["imb24_z"] + 0.5 * g["metric_taker_z"]
    g["total_pressure"] = 0.5 * g["funding_z"] + 0.5 * g["flow_pressure"]

    for horizon in HORIZONS:
        g[f"y_{horizon}"] = np.log(g["close"].shift(-horizon) / g["open"].shift(-1))
    return g


def build_graph(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dev = panel.loc[panel["time"].dt.year <= 2022]
    pivot = dev.pivot(index="time", columns="symbol", values="u").reindex(columns=SYMBOLS)
    corr = pivot.corr(min_periods=3000).fillna(0.0).to_numpy(dtype=float)
    np.fill_diagonal(corr, 0.0)
    weights = np.zeros_like(corr)
    k = min(4, len(SYMBOLS) - 1)
    for i in range(len(SYMBOLS)):
        order = np.argsort(np.abs(corr[i]))[::-1]
        chosen = [j for j in order if j != i][:k]
        for j in chosen:
            weights[i, j] = abs(corr[i, j])
    weights = 0.5 * (weights + weights.T)
    row_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sum, out=np.zeros_like(weights), where=row_sum > EPS)

    directed = np.zeros_like(corr)
    values = pivot.to_numpy(dtype=float)
    for i in range(len(SYMBOLS)):
        for j in range(i + 1, len(SYMBOLS)):
            c_ij = safe_spearman(values[1:, i], values[:-1, j], minimum=1000)
            c_ji = safe_spearman(values[1:, j], values[:-1, i], minimum=1000)
            value = (0.0 if not np.isfinite(c_ij) else c_ij) - (0.0 if not np.isfinite(c_ji) else c_ji)
            directed[i, j] = value
            directed[j, i] = -value
    max_abs = float(np.nanmax(np.abs(directed)))
    if max_abs > EPS:
        directed /= max_abs
    audit = {
        "symbols": SYMBOLS,
        "top_k": k,
        "correlation_matrix": corr.tolist(),
        "weights": weights.tolist(),
        "directed_lead_lag": directed.tolist(),
        "development_rows": int(len(pivot)),
    }
    return weights, directed, audit


def row_fill(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    med = np.nanmedian(out, axis=1)
    med = np.where(np.isfinite(med), med, 0.0)
    rows, cols = np.where(~np.isfinite(out))
    out[rows, cols] = med[rows]
    return out


def graph_state(panel: pd.DataFrame, weights: np.ndarray, directed: np.ndarray) -> pd.DataFrame:
    times = pd.Index(sorted(panel["time"].unique()))
    index = pd.MultiIndex.from_product([times, SYMBOLS], names=["time", "symbol"])
    base = panel.set_index(["time", "symbol"]).reindex(index)

    def matrix(column: str) -> np.ndarray:
        return (
            base[column].unstack("symbol").reindex(index=times, columns=SYMBOLS)
            .to_numpy(dtype=float)
        )

    rho_raw = matrix("oi_level_z")
    source_raw = matrix("oi_source_z")
    u_raw = matrix("u")
    mom_raw = matrix("mom24")
    fund_raw = matrix("funding_z")
    flow_raw = matrix("flow_pressure")
    total_raw = matrix("total_pressure")
    rv_raw = matrix("rv24_z")

    rho = row_fill(rho_raw)
    source = row_fill(source_raw)
    u = row_fill(u_raw)
    mom = row_fill(mom_raw)
    fund = row_fill(fund_raw)
    flow = row_fill(flow_raw)
    total = row_fill(total_raw)
    rv = row_fill(rv_raw)

    peer_u = u @ weights.T
    peer_rho = rho @ weights.T
    peer_pressure = total @ weights.T
    div_u = peer_u - u
    grad_rho = peer_rho - rho
    grad_pressure = peer_pressure - total
    circulation = (directed @ (rho * u).T).T

    n_time, n_symbol = rho.shape
    torque_funding = np.full_like(rho, np.nan)
    torque_flow = np.full_like(rho, np.nan)
    torque_total = np.full_like(rho, np.nan)
    misalignment = np.full_like(rho, np.nan)

    pressure_sets = [fund, flow, total]
    torque_sets = [torque_funding, torque_flow, torque_total]
    for t in range(n_time):
        for i in range(n_symbol):
            w = weights[i]
            dr = w * (rho[t] - rho[t, i])
            for pressure, output in zip(pressure_sets, torque_sets):
                dp = w * (pressure[t] - pressure[t, i])
                output[t, i] = float(dr @ directed @ dp)
            dp_total = w * (total[t] - total[t, i])
            norm = float(np.sqrt(np.sum(dr * dr) * np.sum(dp_total * dp_total)))
            if norm > EPS:
                misalignment[t, i] = 1.0 - abs(float(np.sum(dr * dp_total) / norm))

    impedance = np.exp(np.clip(rho / 4.0, -3.0, 3.0)) * np.exp(np.clip(rv / 4.0, -3.0, 3.0))
    peer_impedance = impedance @ weights.T
    reflection = (impedance - peer_impedance) / (impedance + peer_impedance + EPS)
    wave_action = np.sign(u) * u * u / (impedance + EPS)
    peer_wave_action = wave_action @ weights.T

    delta_torque = np.vstack([np.full((1, n_symbol), np.nan), np.diff(torque_total, axis=0)])
    delta_circulation = np.vstack([np.full((1, n_symbol), np.nan), np.diff(circulation, axis=0)])
    prior_abs_torque = np.vstack([np.full((1, n_symbol), np.nan), np.abs(torque_total[:-1])])
    contraction = np.maximum(prior_abs_torque - np.abs(torque_total), 0.0)
    kelvin_residual = delta_circulation - torque_total
    pv = torque_total / (1.0 + np.abs(rho))

    raw: dict[str, np.ndarray] = {
        "baroclinic_funding": torque_funding,
        "baroclinic_flow": torque_flow,
        "baroclinic_total": torque_total,
        "pv_transport": pv * u,
        "pv_source": pv * source,
        "pv_pressure_release": -pv * np.vstack([np.full((1, n_symbol), np.nan), np.diff(total, axis=0)]),
        "vortex_stretch": torque_total * (-div_u),
        "enstrophy_release": -torque_total * delta_torque,
        "misalignment_energy": np.sign(u) * np.abs(torque_total) * misalignment,
        "kelvin_residual": kelvin_residual,
        "kelvin_reconnection": -np.sign(mom) * kelvin_residual,
        "torque_circulation_alignment": torque_total * circulation,
        "ekman_pumping": -fund * circulation / (1.0 + np.abs(rho)),
        "impedance_reflection": -reflection * peer_u,
        "wave_action_gradient": peer_wave_action - wave_action,
        "baroclinic_reattach": np.sign(mom) * prior_abs_torque * contraction * np.exp(-np.abs(div_u)),
        "torque_contraction": np.sign(u) * contraction * misalignment,
        "pressure_density_reconnect": -np.sign(mom) * grad_rho * grad_pressure * contraction,
        "source_supported_torque": torque_total * source,
        "deleveraging_torque_release": -np.sign(mom) * torque_total * (-source).clip(min=0.0),
    }

    extras: dict[str, np.ndarray] = {
        "peer_u": peer_u,
        "div_u": div_u,
        "peer_rho": peer_rho,
        "grad_rho": grad_rho,
        "peer_pressure": peer_pressure,
        "grad_pressure": grad_pressure,
        "graph_circulation": circulation,
        "continuity_residual": source - div_u,
        "simple_density_pressure_alignment": grad_rho * grad_pressure,
        "simple_source_velocity": source * u,
        "simple_source_funding": source * fund,
        "simple_funding_velocity": fund * u,
        "impedance_reflection_parent": reflection,
    }

    result = base.reset_index()[["time", "symbol"]].copy()
    for name, values in {**extras, **raw}.items():
        flat = pd.DataFrame(values, index=times, columns=SYMBOLS).stack(dropna=False)
        flat.index.names = ["time", "symbol"]
        result[name] = flat.reindex(index).to_numpy(dtype=float)
    return result


def add_candidate_states(panel: pd.DataFrame) -> pd.DataFrame:
    raw_names = [name for names in FAMILIES.values() for name in names]
    parts: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        for name in raw_names:
            state = causal_z(g[name], 720, 240)
            g[f"{name}_z"] = state
            for span in SMOOTHING:
                g[f"{name}_ewm{span}"] = state.ewm(
                    span=span,
                    min_periods=max(4, int(0.65 * span)),
                    adjust=False,
                ).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)


def issue_subset(frame: pd.DataFrame, horizon: int, nonoverlap: bool = False) -> pd.DataFrame:
    step = horizon if nonoverlap else min(max(horizon, 4), 24)
    hours = (frame["time"].astype("int64") // 3_600_000_000_000).astype("int64")
    return frame.loc[(hours % step) == 0].copy()


def cross_sectional_ic(frame: pd.DataFrame, factor: str, target: str) -> tuple[float, pd.Series]:
    rows: list[tuple[pd.Timestamp, float]] = []
    for timestamp, group in frame.groupby("time", sort=True):
        value = safe_spearman(group[factor], group[target], minimum=8)
        if np.isfinite(value):
            rows.append((timestamp, value))
    if not rows:
        return float("nan"), pd.Series(dtype=float)
    series = pd.Series({timestamp: value for timestamp, value in rows}).sort_index()
    return float(series.mean()), series


def time_series_ic(frame: pd.DataFrame, factor: str, target: str) -> tuple[float, float, int]:
    values: list[float] = []
    for _, group in frame.groupby("symbol", sort=False):
        value = safe_spearman(group[factor], group[target], minimum=80)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.mean(array > 0.0)), int(len(array))


def moving_block_bootstrap(series: pd.Series, block: int, repetitions: int = 3000) -> dict[str, float]:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < max(20, 2 * block):
        return {"low": float("nan"), "high": float("nan"), "p_nonpositive": float("nan"), "n": int(len(values))}
    rng = np.random.default_rng(SEED + len(values) + block)
    means = np.empty(repetitions, dtype=float)
    max_start = max(1, len(values) - block + 1)
    blocks_needed = int(math.ceil(len(values) / block))
    for r in range(repetitions):
        starts = rng.integers(0, max_start, size=blocks_needed)
        sample = np.concatenate([values[s:s + block] for s in starts])[: len(values)]
        means[r] = float(np.mean(sample))
    return {
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0.0)),
        "n": int(len(values)),
    }


def prepare_xy(frame: pd.DataFrame, features: list[str], target: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    columns = ["time", "symbol", target] + features
    work = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if limit is not None and len(work) > limit:
        indices = np.linspace(0, len(work) - 1, limit, dtype=int)
        work = work.iloc[indices]
    x = work[features].to_numpy(dtype=float)
    y = work[target].to_numpy(dtype=float)
    return x, y, work


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std > EPS, std, 1.0)
    return (train - mean) / std, (test - mean) / std


def ridge_increment(train: pd.DataFrame, test: pd.DataFrame, factor: str, target: str) -> dict[str, float]:
    common = ["time", "symbol", target] + BASE_FEATURES + [factor]
    tr = train[common].replace([np.inf, -np.inf], np.nan).dropna()
    te = test[common].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) > 300_000:
        tr = tr.iloc[np.linspace(0, len(tr) - 1, 300_000, dtype=int)]
    if len(tr) < 5000 or len(te) < 1000:
        return {"mse_gain": float("nan"), "base_rankic": float("nan"), "full_rankic": float("nan"), "n_train": len(tr), "n_test": len(te)}
    xb = tr[BASE_FEATURES].to_numpy(float)
    xbt = te[BASE_FEATURES].to_numpy(float)
    xf = tr[BASE_FEATURES + [factor]].to_numpy(float)
    xft = te[BASE_FEATURES + [factor]].to_numpy(float)
    xb, xbt = standardize(xb, xbt)
    xf, xft = standardize(xf, xft)
    y = tr[target].to_numpy(float)
    yt = te[target].to_numpy(float)
    base_model = Ridge(alpha=20.0).fit(xb, y)
    full_model = Ridge(alpha=20.0).fit(xf, y)
    pb = base_model.predict(xbt)
    pf = full_model.predict(xft)
    mse_b = float(np.mean((yt - pb) ** 2))
    mse_f = float(np.mean((yt - pf) ** 2))
    pred = te[["time", "symbol"]].copy()
    pred["target"] = yt
    pred["base"] = pb
    pred["full"] = pf
    base_ic, _ = cross_sectional_ic(pred.rename(columns={"base": "factor", "target": "label"}), "factor", "label")
    full_ic, _ = cross_sectional_ic(pred.rename(columns={"full": "factor", "target": "label"}), "factor", "label")
    return {
        "mse_gain": float((mse_b - mse_f) / (mse_b + EPS)),
        "base_rankic": base_ic,
        "full_rankic": full_ic,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def residual_factor(train: pd.DataFrame, test: pd.DataFrame, factor: str) -> pd.Series:
    columns = ["time", "symbol"] + BASE_FEATURES + [factor]
    tr = train[columns].replace([np.inf, -np.inf], np.nan).dropna()
    te = test[columns].replace([np.inf, -np.inf], np.nan).dropna()
    result = pd.Series(np.nan, index=test.index, dtype=float)
    if len(tr) < 5000 or len(te) < 1000:
        return result
    if len(tr) > 300_000:
        tr = tr.iloc[np.linspace(0, len(tr) - 1, 300_000, dtype=int)]
    x = tr[BASE_FEATURES].to_numpy(float)
    xt = te[BASE_FEATURES].to_numpy(float)
    x, xt = standardize(x, xt)
    y = tr[factor].to_numpy(float)
    model = Ridge(alpha=20.0).fit(x, y)
    result.loc[te.index] = te[factor].to_numpy(float) - model.predict(xt)
    return result


def maximum_parent_correlation(frame: pd.DataFrame, factor: str) -> tuple[float, str]:
    values: list[tuple[float, str]] = []
    for feature in BASE_FEATURES:
        daily: list[float] = []
        for _, group in frame.groupby("time", sort=False):
            value = safe_spearman(group[factor], group[feature], minimum=8)
            if np.isfinite(value):
                daily.append(value)
        if daily:
            values.append((abs(float(np.mean(daily))), feature))
    if not values:
        return float("nan"), ""
    return max(values)


def portfolio(frame: pd.DataFrame, factor: str, target: str, horizon: int, cost_bp: float) -> dict[str, Any]:
    work = issue_subset(frame, horizon, nonoverlap=True)
    previous: dict[str, float] = {}
    net: list[float] = []
    gross_values: list[float] = []
    turnovers: list[float] = []
    times: list[pd.Timestamp] = []
    for timestamp, group in work.groupby("time", sort=True):
        z = group[["symbol", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 8:
            continue
        z = z.sort_values(factor)
        count = max(2, int(math.floor(len(z) * 0.30)))
        low = z.head(count)
        high = z.tail(count)
        weights = {symbol: -0.5 / count for symbol in low["symbol"]}
        for symbol in high["symbol"]:
            weights[symbol] = weights.get(symbol, 0.0) + 0.5 / count
        target_map = dict(zip(z["symbol"], z[target].to_numpy(float)))
        gross = float(sum(weight * target_map[symbol] for symbol, weight in weights.items()))
        names = set(previous) | set(weights)
        turnover = float(sum(abs(weights.get(name, 0.0) - previous.get(name, 0.0)) for name in names))
        value = gross - cost_bp / 10000.0 * turnover
        gross_values.append(gross)
        net.append(value)
        turnovers.append(turnover)
        times.append(timestamp)
        previous = weights
    if len(net) < 20:
        return {"annual_log_return": float("nan"), "sharpe": float("nan"), "avg_turnover": float("nan"), "n": len(net), "years": {}}
    array = np.asarray(net, dtype=float)
    annual_periods = 365.25 * 24.0 / horizon
    years = pd.to_datetime(pd.Series(times), utc=True).dt.year.to_numpy()
    yearly = {str(year): float(np.mean(array[years == year]) * annual_periods) for year in np.unique(years)}
    return {
        "annual_log_return": float(np.mean(array) * annual_periods),
        "gross_annual_log_return": float(np.mean(gross_values) * annual_periods),
        "sharpe": float(np.mean(array) / (np.std(array, ddof=0) + EPS) * np.sqrt(annual_periods)),
        "avg_turnover": float(np.mean(turnovers)),
        "n": int(len(array)),
        "years": yearly,
    }


def yearly_rankic(frame: pd.DataFrame, factor: str, target: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for year, group in frame.groupby(frame["time"].dt.year):
        value, _ = cross_sectional_ic(group, factor, target)
        out[str(int(year))] = value
    return out


def evaluate(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
    sign: int,
    train_year_end: int,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    target = f"y_{horizon}"
    work = panel.copy()
    work["__factor"] = sign * work[factor]
    train = issue_subset(work.loc[work["time"].dt.year <= train_year_end].copy(), horizon)
    test = issue_subset(
        work.loc[(work["time"].dt.year >= start_year) & (work["time"].dt.year <= end_year)].copy(),
        horizon,
    )
    cs, daily = cross_sectional_ic(test, "__factor", target)
    ts, ts_positive, ts_n = time_series_ic(test, "__factor", target)
    block = max(4, int(math.ceil(horizon / min(max(horizon, 4), 24))))
    boot = moving_block_bootstrap(daily, block=block)
    ridge = ridge_increment(train, test, "__factor", target)
    test["__residual"] = residual_factor(train, test, "__factor")
    residual_cs, _ = cross_sectional_ic(test, "__residual", target)
    residual_ts, _, _ = time_series_ic(test, "__residual", target)
    max_corr, corr_feature = maximum_parent_correlation(test, "__factor")
    account2 = portfolio(test, "__factor", target, horizon, 2.0)
    account5 = portfolio(test, "__factor", target, horizon, 5.0)
    years = yearly_rankic(test, "__factor", target)
    return {
        "factor": factor,
        "horizon": horizon,
        "sign": sign,
        "train_year_end": train_year_end,
        "start_year": start_year,
        "end_year": end_year,
        "cs_rankic": cs,
        "ts_ic": ts,
        "ts_positive_share": ts_positive,
        "ts_products": ts_n,
        "bootstrap_low": boot["low"],
        "bootstrap_high": boot["high"],
        "bootstrap_p_nonpositive": boot["p_nonpositive"],
        "ridge_mse_gain": ridge["mse_gain"],
        "ridge_base_rankic": ridge["base_rankic"],
        "ridge_full_rankic": ridge["full_rankic"],
        "residual_cs_rankic": residual_cs,
        "residual_ts_ic": residual_ts,
        "max_parent_corr": max_corr,
        "max_parent_feature": corr_feature,
        "annual_log_return_2bp": account2["annual_log_return"],
        "sharpe_2bp": account2["sharpe"],
        "annual_log_return_5bp": account5["annual_log_return"],
        "sharpe_5bp": account5["sharpe"],
        "avg_turnover": account5["avg_turnover"],
        "portfolio_n": account5["n"],
        "yearly_rankic": json.dumps(years, sort_keys=True),
        "yearly_5bp": json.dumps(account5["years"], sort_keys=True),
    }


def development_score(panel: pd.DataFrame, factor: str, family: str, horizon: int) -> dict[str, Any]:
    target = f"y_{horizon}"
    dev = issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon)
    cs, _ = cross_sectional_ic(dev, factor, target)
    ts, _, _ = time_series_ic(dev, factor, target)
    raw_metric = 0.75 * (0.0 if not np.isfinite(cs) else cs) + 0.25 * (0.0 if not np.isfinite(ts) else ts)
    sign = 1 if raw_metric >= 0 else -1
    dev["__factor"] = sign * dev[factor]
    yearly = yearly_rankic(dev, "__factor", target)
    finite_years = [value for value in yearly.values() if np.isfinite(value)]
    min_year = min(finite_years) if finite_years else float("nan")
    signed_cs = sign * cs if np.isfinite(cs) else float("nan")
    signed_ts = sign * ts if np.isfinite(ts) else float("nan")
    account = portfolio(dev, "__factor", target, horizon, 2.0)
    score = (
        (min_year if np.isfinite(min_year) else -1.0)
        + 0.30 * (signed_cs if np.isfinite(signed_cs) else -1.0)
        + 0.10 * (signed_ts if np.isfinite(signed_ts) else -1.0)
        + 0.01 * np.clip(account["sharpe"], -2.0, 2.0)
    )
    return {
        "family": family,
        "factor": factor,
        "horizon": horizon,
        "sign": sign,
        "development_cs_rankic": signed_cs,
        "development_ts_ic": signed_ts,
        "development_min_year_rankic": min_year,
        "development_2bp_return": account["annual_log_return"],
        "development_2bp_sharpe": account["sharpe"],
        "development_years": json.dumps(yearly, sort_keys=True),
        "development_score": float(score),
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adjusted = np.full(n, np.nan, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        p = pvalues[index]
        value = min(1.0, (n - rank) * p)
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def candidate_columns() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for family, raw_names in FAMILIES.items():
        names: list[str] = []
        for raw in raw_names:
            names.append(f"{raw}_z")
            names.extend(f"{raw}_ewm{span}" for span in SMOOTHING)
        out[family] = names
    return out


def report(
    data_audit: dict[str, Any],
    winners: pd.DataFrame,
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    promoted: list[dict[str, Any]],
) -> str:
    lines = [
        "# Fluid Baroclinic Torque Round 13",
        "",
        "This run tests graph baroclinic torque, potential-vorticity transport, vortex stretching, Kelvin-circulation residuals, Ekman pumping, impedance reflection, reattachment and source coupling on public USD-M perpetual-futures data.",
        "",
        "## Data",
        "",
        f"- symbols: {len(SYMBOLS)}",
        f"- kline rows: {data_audit['klines']['rows']}",
        f"- range: {data_audit['klines']['min_time']} to {data_audit['klines']['max_time']}",
        f"- Hugging Face revision: `{data_audit['external']['hf_revision']}`",
        "- development: 2021-2022; validation: 2023-2024; historical holdout: 2025-2026-09",
        "",
        "## Development-frozen family winners",
        "",
        winners.to_markdown(index=False),
        "",
        "## Validation",
        "",
        validation.to_markdown(index=False),
        "",
        "## Historical holdout",
        "",
        holdout.to_markdown(index=False) if not holdout.empty else "No validation-qualified candidate was opened on the historical holdout.",
        "",
        "## Promotion",
        "",
        "```json",
        json.dumps(promoted, indent=2, sort_keys=True),
        "```",
        "",
        "Promotion requires positive validation and holdout RankIC, positive strong-parent Ridge increment, positive residual RankIC, parent correlation below 0.85, positive 5bp non-overlap account return, positive yearly RankIC, and Holm-adjusted validation p <= 0.10. Historical holdout is not fresh live evidence.",
    ]
    return "\n".join(lines)


def main() -> None:
    np.random.seed(SEED)
    klines, kline_audit = load_klines()
    metrics, funding, external_audit = load_external()
    merged = merge_external(klines, metrics, funding)

    local_parts: list[pd.DataFrame] = []
    for symbol, group in merged.groupby("symbol", sort=False):
        local = compute_local_features(group)
        local["symbol"] = symbol
        local_parts.append(local)
        print(f"local_features={symbol} rows={len(local)}", flush=True)
    local_panel = pd.concat(local_parts, ignore_index=True)
    weights, directed, graph_audit = build_graph(local_panel)
    graph_panel = graph_state(local_panel, weights, directed)
    panel = local_panel.merge(graph_panel, on=["time", "symbol"], how="left", validate="one_to_one")
    panel = add_candidate_states(panel)

    candidate_map = candidate_columns()
    development_rows: list[dict[str, Any]] = []
    for family, factors in candidate_map.items():
        for factor in factors:
            for horizon in HORIZONS:
                development_rows.append(development_score(panel, factor, family, horizon))
        print(f"development_complete={family}", flush=True)
    development = pd.DataFrame(development_rows)
    development.to_csv(OUT / "DEVELOPMENT_SCAN.csv", index=False)
    winners = (
        development.sort_values(["family", "development_score"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(OUT / "FAMILY_WINNERS.csv", index=False)

    validation_rows: list[dict[str, Any]] = []
    for row in winners.itertuples(index=False):
        result = evaluate(panel, row.factor, int(row.horizon), int(row.sign), 2022, 2023, 2024)
        result["family"] = row.family
        validation_rows.append(result)
        print(f"validation_complete={row.family} factor={row.factor}", flush=True)
    validation = pd.DataFrame(validation_rows)
    raw_p = [float(x) if np.isfinite(x) else 1.0 for x in validation["bootstrap_p_nonpositive"]]
    validation["holm_p"] = holm_adjust(raw_p)
    validation["validation_gate"] = (
        (validation["cs_rankic"] > 0.010)
        & (validation["bootstrap_p_nonpositive"] <= 0.10)
        & (validation["ridge_mse_gain"] > 0.0)
        & (validation["residual_cs_rankic"] > 0.005)
        & (validation["max_parent_corr"] < 0.85)
        & (validation["annual_log_return_5bp"] > 0.0)
        & (validation["holm_p"] <= 0.10)
    )
    validation.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)

    holdout_rows: list[dict[str, Any]] = []
    opened = validation.loc[validation["validation_gate"]].copy()
    for row in opened.itertuples(index=False):
        result = evaluate(panel, row.factor, int(row.horizon), int(row.sign), 2024, 2025, 2026)
        result["family"] = row.family
        result["validation_holm_p"] = row.holm_p
        holdout_rows.append(result)
        print(f"holdout_complete={row.family} factor={row.factor}", flush=True)
    holdout = pd.DataFrame(holdout_rows)
    if not holdout.empty:
        holdout.to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    else:
        pd.DataFrame(columns=list(validation.columns)).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)

    promoted: list[dict[str, Any]] = []
    for row in holdout.itertuples(index=False):
        yearly = json.loads(row.yearly_rankic)
        yearly_positive = all(np.isfinite(value) and value > 0.0 for value in yearly.values()) if yearly else False
        passed = (
            row.cs_rankic > 0.010
            and row.bootstrap_p_nonpositive <= 0.10
            and row.ridge_mse_gain > 0.0
            and row.residual_cs_rankic > 0.005
            and row.max_parent_corr < 0.85
            and row.annual_log_return_5bp > 0.0
            and yearly_positive
        )
        if passed:
            promoted.append({
                "family": row.family,
                "factor": row.factor,
                "horizon": int(row.horizon),
                "sign": int(row.sign),
                "validation_holm_p": float(row.validation_holm_p),
                "holdout_cs_rankic": float(row.cs_rankic),
                "holdout_ridge_mse_gain": float(row.ridge_mse_gain),
                "holdout_residual_cs_rankic": float(row.residual_cs_rankic),
                "holdout_annual_log_return_5bp": float(row.annual_log_return_5bp),
            })

    # Multi-horizon diagnostics for every holdout-opened candidate.
    diagnostics: list[dict[str, Any]] = []
    for row in opened.itertuples(index=False):
        for horizon in HORIZONS:
            for period_name, train_end, start, end in [
                ("validation", 2022, 2023, 2024),
                ("historical_holdout", 2024, 2025, 2026),
            ]:
                result = evaluate(panel, row.factor, horizon, int(row.sign), train_end, start, end)
                diagnostics.append({
                    "family": row.family,
                    "factor": row.factor,
                    "selected_horizon": int(row.horizon),
                    "tested_horizon": horizon,
                    "period": period_name,
                    "cs_rankic": result["cs_rankic"],
                    "ts_ic": result["ts_ic"],
                    "ridge_mse_gain": result["ridge_mse_gain"],
                    "residual_cs_rankic": result["residual_cs_rankic"],
                    "annual_log_return_5bp": result["annual_log_return_5bp"],
                })
    pd.DataFrame(diagnostics).to_csv(OUT / "MULTI_HORIZON_DIAGNOSTICS.csv", index=False)

    data_audit = {
        "klines": kline_audit,
        "external": external_audit,
        "graph": graph_audit,
        "merged_rows": int(len(merged)),
        "panel_rows": int(len(panel)),
        "finite_by_candidate": {
            factor: int(np.isfinite(panel[factor].to_numpy(float)).sum())
            for factors in candidate_map.values() for factor in factors
        },
        "availability_policy": {
            "klines": "factor at completed hour; label enters next-hour open",
            "metrics": "latest archived observation shifted by one complete hour",
            "funding": "last published event as-of joined and shifted by one complete hour",
        },
    }
    write_json(OUT / "DATA_AUDIT.json", data_audit)
    write_json(OUT / "PROMOTED.json", promoted)
    summary = {
        "candidate_families": len(FAMILIES),
        "candidate_columns": sum(len(v) for v in candidate_map.values()),
        "development_tests": int(len(development)),
        "validation_winners": int(len(validation)),
        "validation_gate_passed": int(validation["validation_gate"].sum()),
        "holdout_opened": int(len(holdout)),
        "promoted": promoted,
        "decision": "promote_only_if_PROMOTED_nonempty",
    }
    write_json(OUT / "SUMMARY.json", summary)
    (OUT / "REPORT.md").write_text(
        report(data_audit, winners, validation, holdout, promoted), encoding="utf-8"
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
