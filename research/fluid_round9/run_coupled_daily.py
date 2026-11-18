from __future__ import annotations

import importlib.util
import io
import json
import math
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CORE_PATH = Path("research/fluid_round9/run.py")
spec = importlib.util.spec_from_file_location("fluid_round9_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load fluid round9 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

OUT = Path("research_outputs/fluid_round9_coupled_daily")
OUT.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "SOLUSDT"]
HORIZONS = [1, 3, 5, 10, 20]
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2026-11-01", tz="UTC")
START_MS = int(START.timestamp() * 1000)
END_MS = int(END.timestamp() * 1000)
MONTHS = list(pd.period_range("2021-01", "2026-10", freq="M"))
BASE = [
    "p_u", "s_u", "p_mom3", "p_mom10", "p_mom20", "p_mom60", "s_mom20",
    "basis_z", "basis_velocity_z", "shear_z", "p_qz", "s_qz",
    "p_imb_z", "s_imb_z", "activity_gap_z", "flow_gap_z",
    "p_rv_z", "s_rv_z", "p_range_z", "s_range_z",
]
FAMILIES = {
    "interfacial_energy": ["energy_flux", "shear_relaxation", "shear_drive"],
    "kelvin_helmholtz": ["kh_onset", "kh_release"],
    "interfacial_vorticity": ["vorticity_3", "vorticity_10"],
    "continuity_pressure": ["continuity_residual", "pressure_release", "pressure_drive"],
    "sweeping_impedance": ["spot_sweeping", "interfacial_impedance", "impedance_mismatch"],
    "interfacial_geometry": ["interfacial_torsion"],
}


def parse_klines(rows, symbol):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.iloc[:, :11]
    frame.columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
    ]
    for column in ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["open_time"].astype("Int64"), unit="ms", utc=True)
    frame["symbol"] = symbol
    frame = frame[["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.loc[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return frame


def fetch_rest(symbol: str, market: str):
    if market == "spot":
        endpoints = [
            "https://data-api.binance.vision/api/v3/klines",
            "https://api.binance.com/api/v3/klines",
        ]
        limit = 1000
    else:
        endpoints = [
            "https://fapi.binance.com/fapi/v1/klines",
            "https://fapi1.binance.com/fapi/v1/klines",
        ]
        limit = 1500
    errors = []
    for endpoint in endpoints:
        try:
            rows = []
            start = START_MS
            while start < END_MS:
                response = requests.get(
                    endpoint,
                    params={"symbol": symbol, "interval": "1d", "startTime": start, "endTime": END_MS, "limit": limit},
                    headers={"User-Agent": "Trade-fluid-coupled-daily/1.0"},
                    timeout=35,
                )
                response.raise_for_status()
                batch = response.json()
                if not isinstance(batch, list) or not batch:
                    break
                rows.extend(batch)
                next_start = int(batch[-1][0]) + 86_400_000
                if next_start <= start:
                    break
                start = next_start
                if len(batch) < limit:
                    break
                time.sleep(0.04)
            frame = parse_klines(rows, symbol)
            if len(frame) >= 1500:
                return market, symbol, frame, None
            errors.append(f"{endpoint}:short={len(frame)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{endpoint}:{type(exc).__name__}:{exc}")
    return market, symbol, None, " | ".join(errors)


def fetch_futures_month(symbol: str, period: pd.Period):
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/1d/{symbol}-1d-{period.year:04d}-{period.month:02d}.zip"
    )
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Trade-fluid-coupled-daily/1.0"})
        if response.status_code == 404:
            return symbol, period, None, "404"
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            raw = pd.read_csv(archive.open(names[0]), header=None, low_memory=False)
        return symbol, period, parse_klines(raw.values.tolist(), symbol), None
    except Exception as exc:  # noqa: BLE001
        return symbol, period, None, f"{type(exc).__name__}:{exc}"


def fallback_futures(symbols):
    frames = []
    errors = []
    tasks = [(symbol, period) for symbol in symbols for period in MONTHS]
    with ThreadPoolExecutor(max_workers=20) as pool:
        jobs = [pool.submit(fetch_futures_month, *task) for task in tasks]
        for future in as_completed(jobs):
            symbol, period, frame, error = future.result()
            if frame is not None and not frame.empty:
                frames.append(frame)
            elif error != "404":
                errors.append({"symbol": symbol, "month": str(period), "error": error})
    if not frames:
        return pd.DataFrame(), errors
    panel = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    return panel, errors


def load_data():
    results = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = [pool.submit(fetch_rest, symbol, market) for market in ("spot", "perp") for symbol in SYMBOLS]
        for future in as_completed(jobs):
            results.append(future.result())
    frames = {"spot": [], "perp": []}
    errors = []
    for market, symbol, frame, error in results:
        if frame is not None:
            frames[market].append(frame)
        else:
            errors.append({"market": market, "symbol": symbol, "error": error})
    available_perp = {frame["symbol"].iloc[0] for frame in frames["perp"]}
    missing_perp = [symbol for symbol in SYMBOLS if symbol not in available_perp]
    if missing_perp:
        fallback, fallback_errors = fallback_futures(missing_perp)
        if len(fallback):
            frames["perp"].append(fallback)
        errors.extend({"market": "perp_fallback", **item} for item in fallback_errors)
    if not frames["spot"] or not frames["perp"]:
        raise RuntimeError("spot or perpetual daily data unavailable")
    spot = pd.concat(frames["spot"], ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    perp = pd.concat(frames["perp"], ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    spot = spot.rename(columns={column: f"s_{column}" for column in spot.columns if column not in {"date", "symbol"}})
    perp = perp.rename(columns={column: f"p_{column}" for column in perp.columns if column not in {"date", "symbol"}})
    panel = perp.merge(spot, on=["symbol", "date"], how="inner", validate="one_to_one")
    counts = panel.groupby("symbol").size()
    retained = counts[counts >= 1500].index.tolist()
    panel = panel.loc[panel["symbol"].isin(retained)].sort_values(["symbol", "date"]).reset_index(drop=True)
    if len(retained) < 6:
        raise RuntimeError(f"only {len(retained)} coupled daily symbols")
    return panel, {
        "retained_symbols": retained,
        "rows": int(len(panel)),
        "min_date": str(panel["date"].min()),
        "max_date": str(panel["date"].max()),
        "errors": errors,
    }


def compute_symbol(frame):
    g = frame.sort_values("date").reset_index(drop=True).copy()
    p_return = np.log(g["p_close"]).diff()
    s_return = np.log(g["s_close"]).diff()
    p_scale = p_return.shift(1).rolling(20, min_periods=12).std(ddof=0)
    s_scale = s_return.shift(1).rolling(20, min_periods=12).std(ddof=0)
    p_u = (p_return / (p_scale + core.EPS)).clip(-12.0, 12.0)
    s_u = (s_return / (s_scale + core.EPS)).clip(-12.0, 12.0)
    g["p_u"], g["s_u"] = p_u, s_u
    for window in [3, 10, 20, 60]:
        g[f"p_mom{window}"] = core.norm_momentum(p_return, window)
        g[f"s_mom{window}"] = core.norm_momentum(s_return, window)
    g["p_qz"] = core.causal_z(np.log1p(g["p_quote_volume"].clip(lower=0.0)), 60, 30)
    g["s_qz"] = core.causal_z(np.log1p(g["s_quote_volume"].clip(lower=0.0)), 60, 30)
    p_imbalance = (2.0 * g["p_taker_buy_quote"] / (g["p_quote_volume"] + core.EPS) - 1.0).clip(-1.0, 1.0)
    s_imbalance = (2.0 * g["s_taker_buy_quote"] / (g["s_quote_volume"] + core.EPS) - 1.0).clip(-1.0, 1.0)
    g["p_imb_z"] = core.causal_z(p_imbalance, 60, 30)
    g["s_imb_z"] = core.causal_z(s_imbalance, 60, 30)
    p_range = np.log(g["p_high"] / g["p_low"]).clip(lower=0.0)
    s_range = np.log(g["s_high"] / g["s_low"]).clip(lower=0.0)
    g["p_range_z"] = core.causal_z(p_range, 120, 50)
    g["s_range_z"] = core.causal_z(s_range, 120, 50)
    p_rv = np.sqrt(p_return.pow(2).rolling(20, min_periods=12).mean())
    s_rv = np.sqrt(s_return.pow(2).rolling(20, min_periods=12).mean())
    g["p_rv_z"] = core.causal_z(np.log(p_rv + core.EPS), 120, 50)
    g["s_rv_z"] = core.causal_z(np.log(s_rv + core.EPS), 120, 50)

    basis = np.log(g["p_close"] / g["s_close"])
    basis_velocity = basis.diff()
    basis_acceleration = basis_velocity.diff()
    shear = p_u - s_u
    activity_gap = g["p_qz"] - g["s_qz"]
    flow_gap = g["p_imb_z"] - g["s_imb_z"]
    g["basis_z"] = core.causal_z(basis, 120, 50)
    g["basis_velocity_z"] = core.causal_z(basis_velocity, 120, 50)
    g["basis_acceleration_z"] = core.causal_z(basis_acceleration, 120, 50)
    g["shear_z"] = core.causal_z(shear, 120, 50)
    g["activity_gap_z"] = core.causal_z(activity_gap, 120, 50)
    g["flow_gap_z"] = core.causal_z(flow_gap, 120, 50)

    g["energy_flux"] = core.causal_z(-g["shear_z"] * g["activity_gap_z"] * g["basis_velocity_z"], 252, 100)
    g["shear_relaxation"] = core.causal_z(-g["shear_z"] * (1.0 + g["activity_gap_z"].abs()) * np.exp(-g["basis_z"].abs()), 252, 100)
    g["shear_drive"] = core.causal_z(g["shear_z"] * np.tanh(g["activity_gap_z"]) * np.exp(-g["basis_velocity_z"].abs()), 252, 100)
    excess = (g["shear_z"].abs() - 0.75 * g["basis_z"].abs()).clip(lower=0.0)
    g["kh_onset"] = core.causal_z(np.sign(g["shear_z"]) * excess * (1.0 + g["flow_gap_z"].abs()), 252, 100)
    g["kh_release"] = core.causal_z(-np.sign(g["shear_z"]) * excess * g["basis_acceleration_z"].abs(), 252, 100)
    circulation = g["shear_z"].shift(1) * g["flow_gap_z"] - g["flow_gap_z"].shift(1) * g["shear_z"]
    g["vorticity_3"] = core.causal_z(circulation.rolling(3, min_periods=2).mean(), 252, 100)
    g["vorticity_10"] = core.causal_z(circulation.rolling(10, min_periods=7).mean(), 252, 100)
    continuity = g["basis_velocity_z"] - 0.50 * g["shear_z"] - 0.25 * g["activity_gap_z"]
    g["continuity_residual"] = core.causal_z(continuity, 252, 100)
    pressure = g["basis_acceleration_z"] - 0.50 * g["shear_z"].diff() + 0.25 * g["flow_gap_z"]
    g["pressure_release"] = core.causal_z(-pressure * (1.0 + g["shear_z"].abs()), 252, 100)
    g["pressure_drive"] = core.causal_z(pressure * np.tanh(g["activity_gap_z"]), 252, 100)
    g["spot_sweeping"] = core.causal_z((s_u - p_u.shift(1)) * (1.0 + g["s_qz"].clip(lower=0.0)), 252, 100)
    g["interfacial_impedance"] = core.causal_z(g["shear_z"] / (1.0 + g["activity_gap_z"].abs() + g["flow_gap_z"].abs()), 252, 100)
    g["impedance_mismatch"] = core.causal_z(g["shear_z"] * (g["activity_gap_z"] - g["flow_gap_z"]), 252, 100)
    x, y, z = g["shear_z"], g["activity_gap_z"], g["basis_z"]
    vx, vy, vz = x.diff(), y.diff(), z.diff()
    ax, ay, az = vx.diff(), vy.diff(), vz.diff()
    jx, jy, jz = ax.diff(), ay.diff(), az.diff()
    cx, cy, cz = vy * az - vz * ay, vz * ax - vx * az, vx * ay - vy * ax
    triple = cx * jx + cy * jy + cz * jz
    denominator = np.sqrt(cx.pow(2) + cy.pow(2) + cz.pow(2)) * np.sqrt(jx.pow(2) + jy.pow(2) + jz.pow(2))
    g["interfacial_torsion"] = core.causal_z((triple / (denominator + core.EPS)).clip(-1.0, 1.0).rolling(5, min_periods=3).mean(), 252, 100)
    for horizon in HORIZONS:
        g[f"y_{horizon}"] = core.forward_open_return(g, horizon, "p_open")
    return g


def issue(frame, horizon):
    return frame.loc[(frame["session_index"].astype(float) % horizon) == 0]


def split(values):
    year = pd.to_datetime(values).dt.year
    return pd.Series(np.where(year <= 2022, "development", np.where(year <= 2024, "validation", "holdout")), index=values.index)


def select(panel):
    development = panel.loc[panel["split"] == "development"]
    rows = []
    for family, candidates in FAMILIES.items():
        best = None
        for factor in candidates:
            for horizon in HORIZONS:
                sample = issue(development, horizon)
                target = f"y_{horizon}"
                ts = core.time_series_ic(sample, factor, target)
                cs, _ = core.cross_sectional_ic(sample, factor, target, "date", 6)
                raw = 0.70 * ts["mean"] + 0.30 * cs["mean"]
                if not np.isfinite(raw):
                    continue
                sign = 1.0 if raw >= 0 else -1.0
                score = abs(raw) * math.sqrt(max(1, ts["n_assets"]))
                record = {"family": family, "factor": factor, "horizon": horizon, "sign": sign, "development_ts_ic": sign * ts["mean"], "development_cs_ic": sign * cs["mean"], "selection_score": score}
                if best is None or score > best[0]:
                    best = (score, record)
        if best is not None:
            rows.append(best[1])
    return pd.DataFrame(rows)


def evaluate(panel, selection):
    development = panel.loc[panel["split"] == "development"].copy()
    validation_all = panel.loc[panel["split"] == "validation"].copy()
    holdout_all = panel.loc[panel["split"] == "holdout"].copy()
    validation_rows, holdout_rows, promoted = [], [], []
    for rec in selection.to_dict("records"):
        factor, horizon, sign = rec["factor"], int(rec["horizon"]), float(rec["sign"])
        signed = f"signed_{factor}"
        for frame in (development, validation_all, holdout_all):
            frame[signed] = sign * frame[factor]
        train = issue(development, horizon)
        validation = issue(validation_all, horizon).copy()
        target = f"y_{horizon}"
        ts = core.time_series_ic(validation, signed, target)
        cs, values = core.cross_sectional_ic(validation, signed, target, "date", 6)
        boot = core.block_bootstrap(values, max(3, int(20 / horizon)))
        ridge = core.ridge_increment(train, validation, BASE, signed, target, 100000)
        validation["residual"] = core.residual_values(train, validation, BASE, signed)
        residual_ts = core.time_series_ic(validation, "residual", target)
        residual_cs, _ = core.cross_sectional_ic(validation, "residual", target, "date", 6)
        correlation = core.maximum_baseline_correlation(validation, signed, BASE)
        portfolio = core.rank_portfolio(validation, signed, target, "date", 5.0, 365.0 / horizon, 6)
        passed = bool(ts["mean"] > 0.010 and ts["positive_share"] >= 0.625 and cs["mean"] > 0.006 and boot["low"] > -0.004 and ridge["gain"] > 0.0 and residual_ts["mean"] > 0.002 and residual_cs["mean"] > 0.0 and portfolio["sharpe"] > 0.0 and correlation["max_abs"] < 0.80)
        validation_rows.append({**rec, "validation_pass": passed, "validation_ts_ic": ts["mean"], "validation_ts_positive_share": ts["positive_share"], "validation_cs_ic": cs["mean"], "validation_boot_low": boot["low"], "validation_ridge_gain": ridge["gain"], "validation_residual_ts_ic": residual_ts["mean"], "validation_residual_cs_ic": residual_cs["mean"], "validation_max_baseline_corr": correlation["max_abs"], "validation_max_corr_feature": correlation["feature"], "validation_net_sharpe_5bp": portfolio["sharpe"], "validation_net_annual_log_return_5bp": portfolio["annual_log_return"], "validation_years": json.dumps(portfolio["years"], sort_keys=True)})
        if not passed:
            continue
        holdout = issue(holdout_all, horizon).copy()
        combined = pd.concat([train, validation], ignore_index=True)
        ts_h = core.time_series_ic(holdout, signed, target)
        cs_h, values_h = core.cross_sectional_ic(holdout, signed, target, "date", 6)
        boot_h = core.block_bootstrap(values_h, max(3, int(20 / horizon)))
        ridge_h = core.ridge_increment(combined, holdout, BASE, signed, target, 120000)
        holdout["residual"] = core.residual_values(combined, holdout, BASE, signed)
        residual_h = core.time_series_ic(holdout, "residual", target)
        residual_cs_h, _ = core.cross_sectional_ic(holdout, "residual", target, "date", 6)
        portfolio_h = core.rank_portfolio(holdout, signed, target, "date", 5.0, 365.0 / horizon, 6)
        positive_years = sum(value > 0 for value in portfolio_h["years"].values())
        final_pass = bool(ts_h["mean"] > 0.006 and cs_h["mean"] > 0.003 and boot_h["mean"] > 0.0 and ridge_h["gain"] > 0.0 and residual_h["mean"] > 0.0 and residual_cs_h["mean"] > 0.0 and portfolio_h["sharpe"] > 0.0 and positive_years >= max(1, len(portfolio_h["years"]) - 1))
        record = {**rec, "holdout_pass": final_pass, "holdout_ts_ic": ts_h["mean"], "holdout_ts_positive_share": ts_h["positive_share"], "holdout_cs_ic": cs_h["mean"], "holdout_boot_low": boot_h["low"], "holdout_ridge_gain": ridge_h["gain"], "holdout_residual_ts_ic": residual_h["mean"], "holdout_residual_cs_ic": residual_cs_h["mean"], "holdout_net_sharpe_5bp": portfolio_h["sharpe"], "holdout_net_annual_log_return_5bp": portfolio_h["annual_log_return"], "holdout_years": json.dumps(portfolio_h["years"], sort_keys=True)}
        holdout_rows.append(record)
        if final_pass:
            promoted.append(record)
    return pd.DataFrame(validation_rows), pd.DataFrame(holdout_rows), promoted


try:
    raw, metadata = load_data()
    pieces = []
    for symbol, frame in raw.groupby("symbol", sort=False):
        computed = compute_symbol(frame)
        computed["symbol"] = symbol
        pieces.append(computed)
    panel = pd.concat(pieces, ignore_index=True)
    dates = pd.Index(sorted(panel["date"].unique()))
    panel["session_index"] = panel["date"].map(pd.Series(np.arange(len(dates)), index=dates)).astype("Int64")
    panel["split"] = split(panel["date"])
    selection = select(panel)
    validation, holdout, promoted = evaluate(panel, selection)
    selection.to_csv(OUT / "SELECTION.csv", index=False)
    validation.to_csv(OUT / "VALIDATION.csv", index=False)
    holdout.to_csv(OUT / "HOLDOUT.csv", index=False)
    summary = {"completed_utc": str(pd.Timestamp.now(tz="UTC")), "data": metadata, "validation_passers": int(validation["validation_pass"].sum()) if len(validation) else 0, "promoted": promoted, "protocol": {"development": "2021-2022", "validation": "2023-2024", "holdout": "2025-2026-10", "horizons_days": HORIZONS, "cost_bp_one_way": 5.0, "holdout_policy": "computed only after validation pass"}}
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable(promoted), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Coupled Spot–Perpetual Daily Fluid Validation", "", "## Selection", "", selection.to_markdown(index=False), "", "## Validation", "", validation.to_markdown(index=False), "", "## Conditional holdout", "", holdout.to_markdown(index=False) if len(holdout) else "No validation passer; holdout remained sealed.", "", f"Promoted factors: {len(promoted)}"]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
