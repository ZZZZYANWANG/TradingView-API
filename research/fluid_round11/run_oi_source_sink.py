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

OUT = Path("research_outputs/fluid_oi_source_sink_round11")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
SYMBOLS = [
    "ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "DOTUSDT", "ETCUSDT", "ETHUSDT", "LINKUSDT", "LTCUSDT",
    "SOLUSDT", "XLMUSDT", "XRPUSDT",
]
MONTHS = list(pd.period_range("2021-01", "2026-10", freq="M"))
HORIZONS = [4, 12, 24, 72, 120]
SMOOTHING = [12, 24, 72]
FAMILIES = {
    "mass_source_transport": ["mass_source_transport", "source_supported_transport"],
    "source_sink_absorption": ["source_sink_absorption", "deleveraging_discharge"],
    "continuity_residual": ["continuity_residual", "mass_flux_divergence"],
    "compressible_energy": ["density_kinetic_energy", "specific_kinetic_release"],
    "source_vorticity": ["source_vorticity", "source_activity_torque"],
    "leverage_pressure": ["leverage_pressure", "leverage_pressure_release"],
    "crowding_source": ["crowding_source_torque", "crowding_discharge"],
}


def _download(symbol: str, period: pd.Period):
    url = (
        "https://data.binance.vision/data/futures/um/monthly/metrics/"
        f"{symbol}/{symbol}-metrics-{period.year:04d}-{period.month:02d}.zip"
    )
    headers = {"User-Agent": "Trade-fluid-oi-source-sink/20261118"}
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45, headers=headers)
            if response.status_code == 404:
                return symbol, period, None, "404"
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if not names:
                    return symbol, period, None, "missing_csv"
                raw = pd.read_csv(archive.open(names[0]), low_memory=False)
            raw.columns = [str(c).strip().lower() for c in raw.columns]
            aliases = {
                "create_time": "time",
                "sum_open_interest": "open_interest",
                "sum_open_interest_value": "open_interest_value",
                "count_toptrader_long_short_ratio": "top_count_ratio",
                "sum_toptrader_long_short_ratio": "top_position_ratio",
                "count_long_short_ratio": "account_ratio",
                "sum_taker_long_short_vol_ratio": "taker_ratio",
            }
            raw = raw.rename(columns=aliases)
            needed = ["time", "open_interest", "open_interest_value", "top_count_ratio", "top_position_ratio", "account_ratio", "taker_ratio"]
            missing = [c for c in needed if c not in raw]
            if missing:
                return symbol, period, None, "missing_columns:" + ",".join(missing)
            raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce")
            for column in needed[1:]:
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            raw = raw.dropna(subset=["time", "open_interest", "open_interest_value"])
            raw["symbol"] = symbol
            frame = (
                raw.set_index("time")
                .resample("1h")
                .last()
                .dropna(subset=["open_interest", "open_interest_value"])
                .reset_index()
            )
            frame["symbol"] = symbol
            return symbol, period, frame[["time", "symbol"] + needed[1:]], None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return symbol, period, None, error


def load_metrics():
    tasks = [(symbol, period) for symbol in SYMBOLS for period in MONTHS]
    frames, errors = [], []
    with ThreadPoolExecutor(max_workers=24) as pool:
        jobs = [pool.submit(_download, *task) for task in tasks]
        for count, future in enumerate(as_completed(jobs), 1):
            symbol, period, frame, error = future.result()
            if frame is not None and not frame.empty:
                frames.append(frame)
            elif error != "404":
                errors.append({"symbol": symbol, "month": str(period), "error": str(error)})
            if count % 200 == 0:
                print(f"metrics={count}/{len(tasks)} usable={len(frames)}", flush=True)
    if not frames:
        raise RuntimeError("no metrics data")
    data = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "time"], keep="last")
    return data.sort_values(["symbol", "time"]), errors


def merge_metrics(panel: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for symbol, group in panel.groupby("symbol", sort=False):
        left = group.sort_values("time").copy()
        right = metrics.loc[metrics["symbol"] == symbol].drop(columns="symbol").sort_values("time")
        joined = pd.merge_asof(left, right, on="time", direction="backward", tolerance=pd.Timedelta(hours=2))
        joined["symbol"] = symbol
        for column in ["open_interest", "open_interest_value", "top_count_ratio", "top_position_ratio", "account_ratio", "taker_ratio"]:
            joined[column] = joined[column].shift(1)
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True)


def compute_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("time").reset_index(drop=True)
    g = core.compute_futures_symbol(frame).copy()
    oi = pd.to_numeric(frame["open_interest"], errors="coerce").clip(lower=EPS)
    oi_value = pd.to_numeric(frame["open_interest_value"], errors="coerce").clip(lower=EPS)
    taker_ratio = pd.to_numeric(frame["taker_ratio"], errors="coerce").clip(lower=EPS)
    top_position = pd.to_numeric(frame["top_position_ratio"], errors="coerce").clip(lower=EPS)
    account_ratio = pd.to_numeric(frame["account_ratio"], errors="coerce").clip(lower=EPS)

    source = core.causal_z(np.log(oi).diff(), 720, 240)
    value_source = core.causal_z(np.log(oi_value).diff(), 720, 240)
    density = core.causal_z(np.log(oi_value), 720, 240)
    leverage_source = core.causal_z(value_source - source, 720, 240)
    taker_pressure = core.causal_z(np.log(taker_ratio), 720, 240)
    crowding = core.causal_z(0.5 * np.log(top_position) + 0.5 * np.log(account_ratio), 720, 240)
    flow = g["p_imb_z"].clip(-8.0, 8.0)
    activity = g["p_qz"].clip(-8.0, 8.0)
    velocity = g["p_u"].clip(-8.0, 8.0)
    momentum = g["p_mom24"].clip(-8.0, 8.0)
    source_change = source.diff()
    flow_change = flow.diff()
    activity_change = activity.diff()
    divergence_proxy = core.causal_z(activity_change + flow_change, 720, 240)

    raw = {
        "mass_source_transport": velocity * source * (1.0 + flow.abs()),
        "source_supported_transport": momentum * source.clip(lower=0.0) * (1.0 + taker_pressure.abs()),
        "source_sink_absorption": -velocity * source,
        "deleveraging_discharge": -np.sign(momentum).replace(0.0, np.nan) * (-source).clip(lower=0.0) * velocity.abs(),
        "continuity_residual": source - divergence_proxy,
        "mass_flux_divergence": source - core.causal_z((activity * velocity).diff(), 720, 240),
        "density_kinetic_energy": np.sign(momentum).replace(0.0, np.nan) * density * velocity.pow(2),
        "specific_kinetic_release": -momentum * velocity.pow(2) / (1.0 + density.abs()),
        "source_vorticity": source * flow_change - flow * source_change,
        "source_activity_torque": source * activity_change - activity * source_change,
        "leverage_pressure": leverage_source * g["basis_z"],
        "leverage_pressure_release": -leverage_source * g["basis_velocity_z"],
        "crowding_source_torque": crowding * source_change - source * crowding.diff(),
        "crowding_discharge": -crowding * (-source).clip(lower=0.0) * velocity,
    }
    for names in FAMILIES.values():
        for name in names:
            state = core.causal_z(raw[name], 720, 240)
            for span in SMOOTHING:
                g[f"{name}_ewm{span}"] = state.ewm(span=span, min_periods=max(8, int(0.65 * span)), adjust=False).mean()
    g["oi_source_z"] = source
    g["oi_value_source_z"] = value_source
    g["oi_density_z"] = density
    g["leverage_source_z"] = leverage_source
    g["metrics_taker_pressure_z"] = taker_pressure
    g["metrics_crowding_z"] = crowding
    g["continuity_divergence_z"] = divergence_proxy
    return g


BASE = core.FUTURES_BASE + [
    "oi_source_z", "oi_value_source_z", "oi_density_z", "leverage_source_z",
    "metrics_taker_pressure_z", "metrics_crowding_z", "continuity_divergence_z",
]


def portfolio(frame: pd.DataFrame, factor: str, target: str, cost_bp: float, annual_periods: float) -> dict:
    previous, returns, turns, times = {}, [], [], []
    for timestamp, group in frame.groupby("time", sort=True):
        z = group[["symbol", factor, target]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z) < 8: continue
        raw = np.tanh(z[factor].to_numpy(float)/2.0); raw -= np.mean(raw); den=float(np.sum(np.abs(raw)))
        if den < EPS: continue
        weights=dict(zip(z["symbol"],raw/den)); targets=dict(zip(z["symbol"],z[target].to_numpy(float)))
        gross=float(sum(weights[s]*targets[s] for s in weights)); names=set(previous)|set(weights)
        turnover=float(sum(abs(weights.get(s,0.0)-previous.get(s,0.0)) for s in names))
        returns.append(gross-cost_bp/10000.0*turnover); turns.append(turnover); times.append(timestamp); previous=weights
    values=np.asarray(returns,float)
    if len(values)<20: return {"sharpe":np.nan,"annual_log_return":np.nan,"avg_turnover":np.nan,"years":{}}
    years=pd.to_datetime(pd.Series(times),utc=True).dt.year.to_numpy()
    return {"sharpe":float(np.mean(values)/(np.std(values,ddof=0)+EPS)*np.sqrt(annual_periods)),"annual_log_return":float(np.mean(values)*annual_periods),"avg_turnover":float(np.mean(turns)),"years":{str(y):float(np.mean(values[years==y])*annual_periods) for y in np.unique(years)}}


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, sign: int, train_end: int, start: int, end: int) -> dict:
    target=f"y_{horizon}"; work=panel.copy(); work["__factor"]=sign*work[factor]
    train=core.issue_subset(work.loc[work["time"].dt.year<=train_end].copy(),horizon,"futures")
    test=core.issue_subset(work.loc[(work["time"].dt.year>=start)&(work["time"].dt.year<=end)].copy(),horizon,"futures")
    ts=core.time_series_ic(test,"__factor",target); cs,values=core.cross_sectional_ic(test,"__factor",target,"time",8); boot=core.block_bootstrap(values,max(12,int(120/horizon)))
    ridge=core.ridge_increment(train,test,BASE,"__factor",target,250000); test["__residual"]=core.residual_values(train,test,BASE,"__factor")
    residual_ts=core.time_series_ic(test,"__residual",target); residual_cs,_=core.cross_sectional_ic(test,"__residual",target,"time",8)
    corr=core.maximum_baseline_correlation(test,"__factor",BASE); account=portfolio(test,"__factor",target,2.0,365.25*24.0/horizon)
    return {"factor":factor,"horizon":horizon,"sign":sign,"ts_ic":ts["mean"],"ts_positive_share":ts["positive_share"],"cs_ic":cs["mean"],"cs_positive_share":cs["positive_share"],"bootstrap_low":boot["low"],"bootstrap_high":boot["high"],"ridge_gain":ridge["gain"],"residual_ts_ic":residual_ts["mean"],"residual_cs_ic":residual_cs["mean"],"max_baseline_corr":corr["max_abs"],"max_corr_feature":corr["feature"],"net_sharpe_2bp":account["sharpe"],"annual_log_return_2bp":account["annual_log_return"],"avg_turnover":account["avg_turnover"],"years":json.dumps(account["years"],sort_keys=True)}


def development(panel: pd.DataFrame, factor: str, family: str, horizon: int) -> dict:
    target=f"y_{horizon}"; dev=core.issue_subset(panel.loc[panel["time"].dt.year<=2022].copy(),horizon,"futures")
    ts=core.time_series_ic(dev,factor,target); cs,_=core.cross_sectional_ic(dev,factor,target,"time",8)
    metric=ts["mean"] if abs(ts["mean"])>=abs(cs["mean"]) else cs["mean"]; lane="ts" if abs(ts["mean"])>=abs(cs["mean"]) else "cs"; sign=1 if np.isfinite(metric) and metric>=0 else -1
    dev["__factor"]=sign*dev[factor]; account=portfolio(dev,"__factor",target,2.0,365.25*24.0/horizon)
    return {"factor":factor,"family":family,"horizon":horizon,"sign":sign,"lane":lane,"development_ts_ic":sign*ts["mean"],"development_cs_ic":sign*cs["mean"],"development_sharpe":account["sharpe"],"development_score":abs(metric)+0.003*np.clip(account["sharpe"],-2,2)}


try:
    core.FUTURES_SYMBOLS=SYMBOLS
    raw,metadata=core.load_futures(); metrics,metric_errors=load_metrics(); merged=merge_metrics(raw,metrics)
    parts=[]
    for symbol,frame in merged.groupby("symbol",sort=False):
        z=compute_symbol(frame); z["symbol"]=symbol; parts.append(z)
    panel=pd.concat(parts,ignore_index=True)
    dev_rows=[]
    for family,names in FAMILIES.items():
        for name in names:
            for span in SMOOTHING:
                factor=f"{name}_ewm{span}"
                for horizon in HORIZONS: dev_rows.append(development(panel,factor,family,horizon))
    dev=pd.DataFrame(dev_rows); winners=dev.sort_values(["family","development_score"],ascending=[True,False]).groupby("family",as_index=False).head(1)
    rows=[]
    for _,chosen in winners.iterrows():
        row=evaluate(panel,str(chosen["factor"]),int(chosen["horizon"]),int(chosen["sign"]),2022,2023,2024); row["family"]=chosen["family"]; row["lane"]=chosen["lane"]
        row["validation_gate"]=bool(row["ridge_gain"]>0 and row["max_baseline_corr"]<0.80 and row["net_sharpe_2bp"]>0.20 and ((row["ts_ic"]>0.010 and row["residual_ts_ic"]>0.005 and row["ts_positive_share"]>=0.60) or (row["cs_ic"]>0.012 and row["residual_cs_ic"]>0.006 and row["bootstrap_low"]>-0.003)))
        rows.append(row)
    validation=pd.DataFrame(rows).sort_values(["validation_gate","ridge_gain","net_sharpe_2bp"],ascending=[False,False,False]); passers=validation.loc[validation["validation_gate"]]
    confirmation=[]
    for _,chosen in passers.iterrows():
        factor,horizon,sign=str(chosen["factor"]),int(chosen["horizon"]),int(chosen["sign"]); confirm=evaluate(panel,factor,horizon,sign,2024,2025,2025); shadow=evaluate(panel,factor,horizon,sign,2025,2026,2026)
        passed=bool(confirm["ridge_gain"]>0 and confirm["net_sharpe_2bp"]>0.15 and ((confirm["ts_ic"]>0.006 and confirm["residual_ts_ic"]>0.003) or (confirm["cs_ic"]>0.008 and confirm["residual_cs_ic"]>0.004)) and shadow["ridge_gain"]>-0.0002 and shadow["net_sharpe_2bp"]>0 and ((shadow["ts_ic"]>0 and shadow["residual_ts_ic"]>0) or (shadow["cs_ic"]>0 and shadow["residual_cs_ic"]>0)))
        confirmation.append({"factor":factor,"family":chosen["family"],"horizon":horizon,"sign":sign,"confirmation":confirm,"shadow":shadow,"final_pass":passed})
    promoted=[row for row in confirmation if row["final_pass"]]
    dev.to_csv(OUT/"DEVELOPMENT_GRID.csv",index=False); winners.to_csv(OUT/"FAMILY_WINNERS.csv",index=False); validation.to_csv(OUT/"VALIDATION.csv",index=False); (OUT/"CONFIRMATION.json").write_text(json.dumps(core.jsonable(confirmation),indent=2,ensure_ascii=False),encoding="utf-8")
    summary={"completed_utc":str(pd.Timestamp.now(tz="UTC")),"data":metadata,"metrics_rows":int(len(metrics)),"metric_errors":metric_errors[:100],"mechanism":"real open-interest mass source/sink, compressible transport, continuity residual and crowding torque","families":list(FAMILIES),"validation_passers":int(len(passers)),"promoted_count":int(len(promoted)),"promoted":promoted,"governance":{"development":"2021-2022","validation":"2023-2024","confirmation":"2025 only after gate","shadow":"2026","strong_baseline_contains_raw_oi_and_crowding_states":True,"cost_bp_one_way":2.0}}
    (OUT/"SUMMARY.json").write_text(json.dumps(core.jsonable(summary),indent=2,ensure_ascii=False),encoding="utf-8"); (OUT/"PROMOTED.json").write_text(json.dumps(core.jsonable(promoted),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# Open-Interest Source–Sink Fluid Test","","## Family winners","",winners.to_markdown(index=False),"","## 2023-2024 validation","",validation.to_markdown(index=False),"",f"Validation passers: **{len(passers)}**",f"Final promoted: **{len(promoted)}**"]
    if confirmation: lines += ["","## 2025/2026 confirmation","","```json",json.dumps(core.jsonable(confirmation),indent=2,ensure_ascii=False),"```"]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8"); print(json.dumps(core.jsonable(summary),indent=2,ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure={"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}; (OUT/"FAILURE.json").write_text(json.dumps(failure,indent=2),encoding="utf-8"); print(failure["traceback"]); raise
