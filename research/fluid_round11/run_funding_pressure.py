from __future__ import annotations

import importlib.util
import io
import json
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
    raise RuntimeError("cannot load round9 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

OUT = Path("research_outputs/fluid_funding_pressure_round11")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
SYMBOLS = [
    "ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "DOTUSDT", "ETCUSDT", "ETHUSDT", "LINKUSDT", "LTCUSDT",
    "SOLUSDT", "XLMUSDT", "XRPUSDT",
]
HORIZONS = [4, 12, 24, 72, 120]
SMOOTHING = [12, 24, 72]
MONTHS = list(pd.period_range("2021-01", "2026-10", freq="M"))
FAMILIES = {
    "stored_pressure": ["stored_pressure", "stored_pressure_flow"],
    "pressure_discharge": ["pressure_discharge", "premium_discharge"],
    "pressure_work": ["pressure_work", "funding_work"],
    "enthalpy_flux": ["enthalpy_flux", "enthalpy_dissipation"],
    "acoustic_impedance": ["impedance_release", "impedance_mismatch"],
    "boundary_absorption": ["boundary_absorption", "funding_flow_absorption"],
    "baroclinic_torque": ["baroclinic_torque", "source_gradient_torque"],
    "pressure_mismatch": ["premium_funding_mismatch", "mismatch_release"],
}


def _get_zip(url: str) -> bytes | None:
    headers = {"User-Agent": "Trade-fluid-funding-pressure/20261118"}
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"download failed {url}: {error}")


def download_aux(kind: str, symbol: str, period: pd.Period):
    if kind == "premium":
        url = (
            "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/"
            f"{symbol}/1h/{symbol}-1h-{period.year:04d}-{period.month:02d}.zip"
        )
    else:
        url = (
            "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
            f"{symbol}/{symbol}-fundingRate-{period.year:04d}-{period.month:02d}.zip"
        )
    try:
        payload = _get_zip(url)
        if payload is None:
            return kind, symbol, period, None, "404"
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                return kind, symbol, period, None, "missing_csv"
            raw = pd.read_csv(archive.open(names[0]), header=None, low_memory=False)
        if kind == "premium":
            if raw.shape[1] < 5:
                return kind, symbol, period, None, f"short_schema_{raw.shape[1]}"
            timestamp = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
            close = pd.to_numeric(raw.iloc[:, 4], errors="coerce")
            valid = timestamp.notna() & close.notna()
            timestamp = timestamp.loc[valid].astype("int64")
            unit = "us" if float(timestamp.median()) > 1.0e14 else "ms"
            frame = pd.DataFrame({
                "time": pd.to_datetime(timestamp, unit=unit, utc=True),
                "symbol": symbol,
                "premium_close": close.loc[valid].to_numpy(dtype=float),
            })
        else:
            if raw.shape[1] < 2:
                return kind, symbol, period, None, f"short_schema_{raw.shape[1]}"
            timestamp = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
            rate_col = 2 if raw.shape[1] >= 3 else raw.shape[1] - 1
            rate = pd.to_numeric(raw.iloc[:, rate_col], errors="coerce")
            valid = timestamp.notna() & rate.notna()
            timestamp = timestamp.loc[valid].astype("int64")
            unit = "us" if float(timestamp.median()) > 1.0e14 else "ms"
            frame = pd.DataFrame({
                "time": pd.to_datetime(timestamp, unit=unit, utc=True),
                "symbol": symbol,
                "funding_rate": rate.loc[valid].to_numpy(dtype=float),
            })
        return kind, symbol, period, frame.drop_duplicates(["symbol", "time"], keep="last"), None
    except Exception as exc:  # noqa: BLE001
        return kind, symbol, period, None, f"{type(exc).__name__}:{exc}"


def load_auxiliary() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    tasks = [(kind, symbol, period) for kind in ("premium", "funding") for symbol in SYMBOLS for period in MONTHS]
    premium, funding, errors = [], [], []
    with ThreadPoolExecutor(max_workers=24) as pool:
        jobs = [pool.submit(download_aux, *task) for task in tasks]
        for count, future in enumerate(as_completed(jobs), 1):
            kind, symbol, period, frame, error = future.result()
            if frame is not None and not frame.empty:
                (premium if kind == "premium" else funding).append(frame)
            elif error != "404":
                errors.append({"kind": kind, "symbol": symbol, "month": str(period), "error": str(error)})
            if count % 300 == 0:
                print(f"aux={count}/{len(tasks)} premium={len(premium)} funding={len(funding)}", flush=True)
    if not premium or not funding:
        raise RuntimeError("premium or funding auxiliary data empty")
    return pd.concat(premium, ignore_index=True), pd.concat(funding, ignore_index=True), errors


def merge_aux(panel: pd.DataFrame, premium: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    p = panel.merge(premium, on=["symbol", "time"], how="left", validate="one_to_one")
    parts = []
    for symbol, group in p.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        f = funding.loc[funding["symbol"] == symbol, ["time", "funding_rate"]].sort_values("time")
        g = pd.merge_asof(g, f, on="time", direction="backward", tolerance=pd.Timedelta(hours=9))
        g["symbol"] = symbol
        g["funding_rate"] = g["funding_rate"].shift(1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def compute_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    g = core.compute_futures_symbol(frame).copy()
    premium = pd.to_numeric(frame.sort_values("time")["premium_close"].reset_index(drop=True), errors="coerce")
    funding = pd.to_numeric(frame.sort_values("time")["funding_rate"].reset_index(drop=True), errors="coerce")
    premium_z = core.causal_z(premium, 720, 240)
    funding_z = core.causal_z(funding, 720, 120)
    premium_velocity_z = core.causal_z(premium.diff(), 720, 240)
    funding_change_z = core.causal_z(funding.diff(), 720, 120)
    potential = core.causal_z(premium_z + 0.50 * funding_z, 720, 240)
    potential_change = potential.diff()
    activity = g["p_qz"].clip(-8.0, 8.0)
    flow = g["p_imb_z"].clip(-8.0, 8.0)
    velocity = g["p_u"].clip(-8.0, 8.0)
    momentum = g["p_mom24"].clip(-8.0, 8.0)
    impedance = core.causal_z(potential / (velocity.abs() + 0.25), 720, 240)
    impedance_change = impedance.diff()
    flow_change = flow.diff()
    activity_change = activity.diff()

    raw = {
        "stored_pressure": potential * np.exp(-velocity.abs()),
        "stored_pressure_flow": potential * np.exp(-velocity.abs()) * (1.0 + flow.abs()),
        "pressure_discharge": -potential * potential_change,
        "premium_discharge": -premium_z * premium_velocity_z,
        "pressure_work": -velocity * potential_change,
        "funding_work": -velocity * funding_change_z,
        "enthalpy_flux": velocity * (potential + 0.50 * velocity.pow(2)),
        "enthalpy_dissipation": -momentum * (velocity * (potential + 0.50 * velocity.pow(2))).abs(),
        "impedance_release": -np.sign(potential) * impedance_change,
        "impedance_mismatch": potential * impedance,
        "boundary_absorption": -potential * flow * np.exp(-potential_change.abs()),
        "funding_flow_absorption": -funding_z * flow * np.exp(-premium_velocity_z.abs()),
        "baroclinic_torque": premium_velocity_z * activity_change - activity * potential_change,
        "source_gradient_torque": funding_change_z * flow_change - flow * funding_change_z.shift(1),
        "premium_funding_mismatch": premium_z - funding_z,
        "mismatch_release": -(premium_z - funding_z) * (premium_velocity_z - funding_change_z),
    }
    for family, names in FAMILIES.items():
        for name in names:
            state = core.causal_z(raw[name], 720, 240)
            for span in SMOOTHING:
                g[f"{name}_ewm{span}"] = state.ewm(span=span, min_periods=max(8, int(0.65 * span)), adjust=False).mean()
    g["premium_z"] = premium_z
    g["funding_z"] = funding_z
    g["premium_velocity_z"] = premium_velocity_z
    g["funding_change_z"] = funding_change_z
    g["pressure_potential_z"] = potential
    g["acoustic_impedance_z"] = impedance
    return g


BASE = core.FUTURES_BASE + [
    "premium_z", "funding_z", "premium_velocity_z", "funding_change_z",
    "pressure_potential_z", "acoustic_impedance_z",
]


def portfolio(frame: pd.DataFrame, factor: str, target: str, cost_bp: float, annual_periods: float) -> dict:
    previous, returns, turns, times = {}, [], [], []
    for timestamp, group in frame.groupby("time", sort=True):
        z = group[["symbol", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 8:
            continue
        raw = np.tanh(z[factor].to_numpy(dtype=float) / 2.0); raw -= np.mean(raw)
        den = float(np.sum(np.abs(raw)))
        if den < EPS:
            continue
        weights = dict(zip(z["symbol"], raw / den)); targets = dict(zip(z["symbol"], z[target].to_numpy(dtype=float)))
        gross = float(sum(weights[s] * targets[s] for s in weights))
        names = set(previous) | set(weights)
        turnover = float(sum(abs(weights.get(s, 0.0) - previous.get(s, 0.0)) for s in names))
        returns.append(gross - cost_bp / 10000.0 * turnover); turns.append(turnover); times.append(timestamp); previous = weights
    values = np.asarray(returns, dtype=float)
    if len(values) < 20:
        return {"sharpe": np.nan, "annual_log_return": np.nan, "avg_turnover": np.nan, "years": {}}
    years = pd.to_datetime(pd.Series(times), utc=True).dt.year.to_numpy()
    return {"sharpe": float(np.mean(values)/(np.std(values, ddof=0)+EPS)*np.sqrt(annual_periods)), "annual_log_return": float(np.mean(values)*annual_periods), "avg_turnover": float(np.mean(turns)), "years": {str(y): float(np.mean(values[years==y])*annual_periods) for y in np.unique(years)}}


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, sign: int, train_end: int, start: int, end: int) -> dict:
    target = f"y_{horizon}"; work = panel.copy(); work["__factor"] = sign * work[factor]
    train = core.issue_subset(work.loc[work["time"].dt.year <= train_end].copy(), horizon, "futures")
    test = core.issue_subset(work.loc[(work["time"].dt.year >= start)&(work["time"].dt.year <= end)].copy(), horizon, "futures")
    ts = core.time_series_ic(test, "__factor", target)
    cs, cs_values = core.cross_sectional_ic(test, "__factor", target, "time", 8)
    boot = core.block_bootstrap(cs_values, max(12, int(120/horizon)))
    ridge = core.ridge_increment(train, test, BASE, "__factor", target, 250000)
    test["__residual"] = core.residual_values(train, test, BASE, "__factor")
    residual_ts = core.time_series_ic(test, "__residual", target)
    residual_cs, _ = core.cross_sectional_ic(test, "__residual", target, "time", 8)
    corr = core.maximum_baseline_correlation(test, "__factor", BASE)
    account = portfolio(test, "__factor", target, 2.0, 365.25*24.0/horizon)
    return {"factor": factor, "horizon": horizon, "sign": sign, "ts_ic": ts["mean"], "ts_positive_share": ts["positive_share"], "cs_ic": cs["mean"], "cs_positive_share": cs["positive_share"], "bootstrap_low": boot["low"], "bootstrap_high": boot["high"], "ridge_gain": ridge["gain"], "residual_ts_ic": residual_ts["mean"], "residual_cs_ic": residual_cs["mean"], "max_baseline_corr": corr["max_abs"], "max_corr_feature": corr["feature"], "net_sharpe_2bp": account["sharpe"], "annual_log_return_2bp": account["annual_log_return"], "avg_turnover": account["avg_turnover"], "years": json.dumps(account["years"], sort_keys=True)}


def development(panel: pd.DataFrame, factor: str, horizon: int) -> dict:
    target = f"y_{horizon}"; dev = core.issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon, "futures")
    ts = core.time_series_ic(dev, factor, target); cs, _ = core.cross_sectional_ic(dev, factor, target, "time", 8)
    metric = ts["mean"] if abs(ts["mean"]) >= abs(cs["mean"]) else cs["mean"]
    lane = "ts" if abs(ts["mean"]) >= abs(cs["mean"]) else "cs"
    sign = 1 if np.isfinite(metric) and metric >= 0 else -1
    dev["__factor"] = sign * dev[factor]
    account = portfolio(dev, "__factor", target, 2.0, 365.25*24.0/horizon)
    score = abs(metric) + 0.003*np.clip(account["sharpe"], -2.0, 2.0)
    return {"factor": factor, "family": factor.rsplit("_ewm",1)[0].rsplit("_",1)[0], "horizon": horizon, "sign": sign, "lane": lane, "development_ts_ic": sign*ts["mean"], "development_cs_ic": sign*cs["mean"], "development_sharpe": account["sharpe"], "development_score": score}


try:
    core.FUTURES_SYMBOLS = SYMBOLS
    raw, metadata = core.load_futures()
    premium, funding, aux_errors = load_auxiliary()
    merged = merge_aux(raw, premium, funding)
    parts = []
    for symbol, frame in merged.groupby("symbol", sort=False):
        z = compute_symbol(frame); z["symbol"] = symbol; parts.append(z)
    panel = pd.concat(parts, ignore_index=True)
    candidates = [f"{name}_ewm{span}" for names in FAMILIES.values() for name in names for span in SMOOTHING]
    dev = pd.DataFrame([development(panel, factor, horizon) for factor in candidates for horizon in HORIZONS])
    winners = dev.sort_values(["family","development_score"], ascending=[True,False]).groupby("family", as_index=False).head(1)
    rows = []
    for _, chosen in winners.iterrows():
        row = evaluate(panel, str(chosen["factor"]), int(chosen["horizon"]), int(chosen["sign"]), 2022, 2023, 2024)
        row["family"] = chosen["family"]; row["lane"] = chosen["lane"]
        row["validation_gate"] = bool(row["ridge_gain"] > 0.0 and row["max_baseline_corr"] < 0.80 and row["net_sharpe_2bp"] > 0.20 and ((row["ts_ic"] > 0.010 and row["residual_ts_ic"] > 0.005 and row["ts_positive_share"] >= 0.60) or (row["cs_ic"] > 0.012 and row["residual_cs_ic"] > 0.006 and row["bootstrap_low"] > -0.003)))
        rows.append(row)
    validation = pd.DataFrame(rows).sort_values(["validation_gate","ridge_gain","net_sharpe_2bp"], ascending=[False,False,False])
    passers = validation.loc[validation["validation_gate"]]
    confirmation_rows = []
    for _, chosen in passers.iterrows():
        factor, horizon, sign = str(chosen["factor"]), int(chosen["horizon"]), int(chosen["sign"])
        confirm = evaluate(panel, factor, horizon, sign, 2024, 2025, 2025)
        shadow = evaluate(panel, factor, horizon, sign, 2025, 2026, 2026)
        passed = bool(confirm["ridge_gain"] > 0.0 and confirm["net_sharpe_2bp"] > 0.15 and ((confirm["ts_ic"] > 0.006 and confirm["residual_ts_ic"] > 0.003) or (confirm["cs_ic"] > 0.008 and confirm["residual_cs_ic"] > 0.004)) and shadow["ridge_gain"] > -0.0002 and shadow["net_sharpe_2bp"] > 0.0 and ((shadow["ts_ic"] > 0 and shadow["residual_ts_ic"] > 0) or (shadow["cs_ic"] > 0 and shadow["residual_cs_ic"] > 0)))
        confirmation_rows.append({"factor": factor, "horizon": horizon, "sign": sign, "confirmation": confirm, "shadow": shadow, "final_pass": passed})
    promoted = [row for row in confirmation_rows if row["final_pass"]]
    dev.to_csv(OUT/"DEVELOPMENT_GRID.csv", index=False); winners.to_csv(OUT/"FAMILY_WINNERS.csv", index=False); validation.to_csv(OUT/"VALIDATION.csv", index=False)
    (OUT/"CONFIRMATION.json").write_text(json.dumps(core.jsonable(confirmation_rows), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"completed_utc": str(pd.Timestamp.now(tz="UTC")), "data": metadata, "aux_errors": aux_errors[:100], "mechanism": "funding and premium-index boundary pressure, source-sink, impedance and enthalpy transport", "families": list(FAMILIES), "validation_passers": int(len(passers)), "promoted_count": int(len(promoted)), "promoted": promoted, "governance": {"development":"2021-2022","validation":"2023-2024","confirmation":"2025 only after gate","shadow":"2026","strong_baseline_contains_raw_funding_and_premium_states":True,"cost_bp_one_way":2.0}}
    (OUT/"SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"); (OUT/"PROMOTED.json").write_text(json.dumps(core.jsonable(promoted), indent=2, ensure_ascii=False), encoding="utf-8")
    lines=["# Funding–Premium Boundary Pressure Fluid Test","","## Family winners","",winners.to_markdown(index=False),"","## 2023-2024 validation","",validation.to_markdown(index=False),"",f"Validation passers: **{len(passers)}**",f"Final promoted: **{len(promoted)}**"]
    if confirmation_rows: lines += ["","## 2025/2026 confirmation","","```json",json.dumps(core.jsonable(confirmation_rows),indent=2,ensure_ascii=False),"```"]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8"); print(json.dumps(core.jsonable(summary),indent=2,ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure={"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}; (OUT/"FAILURE.json").write_text(json.dumps(failure,indent=2),encoding="utf-8"); print(failure["traceback"]); raise
