from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import baostock as bs
import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

SEED = 20261120
OUT = Path("research_outputs/ashare_event_absorption")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

REPORT_ENDS = pd.date_range("2016-03-31", "2026-09-30", freq="QE")
MEMBERSHIP_DATES = [f"{year}-12-31" for year in range(2015, 2026)]
HORIZONS = (1, 5, 20)
STAGES = {
    "development": (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-01-01")),
    "validation": (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")),
    "historical_holdout": (pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")),
    "recent_extension": (pd.Timestamp("2025-01-01"), pd.Timestamp("2027-01-01")),
}
COSTS_BPS = (10.0, 20.0, 30.0)
BLOCK = 20
BOOTSTRAPS = 1999
MIN_EVENT_COHORT = 8
MIN_TS_EVENTS = 12


@dataclass(frozen=True)
class Contract:
    source_history_min: int = 8
    source_history_window: int = 20
    source_clip: float = 5.0
    response_clip: float = 6.0
    candidate_clip: float = 8.0
    model_train_years: int = 8
    minimum_model_rows: int = 1200
    minimum_model_dates: int = 100
    event_max_report_lag_days: int = 900


CONTRACT = Contract()


def retry_json(url: str, params: dict, attempts: int = 5) -> dict:
    error = None
    for i in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=(10, 60))
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") is False:
                raise RuntimeError(payload.get("message"))
            return payload
        except Exception as exc:
            error = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"request failed after {attempts} attempts: {error!r}")


def fetch_report_period(report_date: pd.Timestamp) -> pd.DataFrame:
    tag = report_date.strftime("%Y%m%d")
    path = CACHE / f"reports_{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    base = {
        "reportName": "RPT_LICO_FN_CPD",
        "columns": "ALL",
        "filter": f"(REPORTDATE='{report_date:%Y-%m-%d}')",
        "pageNumber": 1,
        "pageSize": 500,
        "sortColumns": "NOTICE_DATE,SECURITY_CODE",
        "sortTypes": "1,1",
        "source": "WEB",
        "client": "WEB",
    }
    first = retry_json(url, base)
    result = first.get("result") or {}
    pages = int(result.get("pages") or 0)
    rows = list(result.get("data") or [])
    for page in range(2, pages + 1):
        params = dict(base, pageNumber=page)
        payload = retry_json(url, params)
        rows.extend((payload.get("result") or {}).get("data") or [])
        if page % 10 == 0:
            print("REPORT", tag, page, "/", pages, "rows", len(rows), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


def fetch_reports() -> pd.DataFrame:
    frames = []
    for report_date in REPORT_ENDS:
        frame = fetch_report_period(report_date)
        if len(frame):
            frames.append(frame)
        print("REPORT_DONE", report_date.date(), len(frame), flush=True)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw.to_parquet(OUT / "raw_report_rows.parquet", index=False, compression="zstd")
    return raw


def query_rows(rs) -> list[list[str]]:
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")
    return rows


def fetch_membership() -> pd.DataFrame:
    path = CACHE / "membership.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for date in MEMBERSHIP_DATES:
        for label, func in (("HS300", bs.query_hs300_stocks), ("ZZ500", bs.query_zz500_stocks)):
            rs = func(date)
            for row in query_rows(rs):
                item = dict(zip(rs.fields, row))
                item.update(snapshot_date=date, index_name=label)
                rows.append(item)
        print("MEMBERSHIP", date, len(rows), flush=True)
    frame = pd.DataFrame(rows)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


def code_to_baostock(secucode: str) -> str:
    code, market = secucode.split(".")
    return ("sh." if market == "SH" else "sz.") + code


def filter_events(raw: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame[frame["SECUCODE"].astype(str).str.endswith((".SH", ".SZ"), na=False)]
    frame = frame[frame["SECURITY_CODE"].astype(str).str.match(r"^(0|3|6)")]
    for col in ("REPORTDATE", "NOTICE_DATE", "UPDATE_DATE", "EITIME"):
        frame[col] = pd.to_datetime(frame.get(col), errors="coerce")
    frame["available_time"] = frame[["NOTICE_DATE", "UPDATE_DATE", "EITIME"]].max(axis=1)
    frame = frame[frame["available_time"].notna() & frame["REPORTDATE"].notna()]
    lag = (frame["available_time"].dt.normalize() - frame["REPORTDATE"]).dt.days
    frame = frame[lag.between(0, CONTRACT.event_max_report_lag_days)]
    frame["code"] = frame["SECUCODE"].map(code_to_baostock)
    frame["membership_year"] = frame["available_time"].dt.year - 1

    mem = membership.copy()
    mem["membership_year"] = mem["snapshot_date"].dt.year
    eligible = set(zip(mem["membership_year"], mem["code"]))
    frame = frame[[tuple(x) in eligible for x in frame[["membership_year", "code"]].itertuples(index=False, name=None)]]

    numeric = [
        "BASIC_EPS", "DEDUCT_BASIC_EPS", "TOTAL_OPERATE_INCOME", "PARENT_NETPROFIT",
        "WEIGHTAVG_ROE", "YSTZ", "SJLTZ", "BPS", "MGJYXJJE", "XSMLL",
    ]
    for col in numeric:
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    frame["industry"] = frame.get("BOARD_NAME").fillna(frame.get("PUBLISHNAME")).fillna("UNKNOWN").astype(str)
    frame = frame.sort_values(["code", "REPORTDATE", "available_time"])
    # Earliest fully observable row is used; later revisions are separate information and are not backdated.
    frame = frame.drop_duplicates(["code", "REPORTDATE"], keep="first")
    frame["issue_year"] = frame["available_time"].dt.year
    frame["report_quarter"] = frame["REPORTDATE"].dt.quarter
    frame.to_parquet(OUT / "eligible_events.parquet", index=False, compression="zstd")
    return frame


def fetch_stock_daily(code: str, start="2014-01-01", end="2026-11-20") -> pd.DataFrame:
    safe = code.replace(".", "_")
    path = CACHE / f"daily_{safe}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,isST"
    rs = bs.query_history_k_data_plus(
        code, fields, start_date=start, end_date=end, frequency="d", adjustflag="2"
    )
    rows = query_rows(rs)
    frame = pd.DataFrame(rows, columns=rs.fields)
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"])
        for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["tradestatus"] = pd.to_numeric(frame["tradestatus"], errors="coerce")
        frame["isST"] = pd.to_numeric(frame["isST"], errors="coerce")
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


def fetch_daily_panel(codes: Iterable[str]) -> pd.DataFrame:
    frames = []
    codes = sorted(set(codes))
    for i, code in enumerate(codes, 1):
        try:
            frame = fetch_stock_daily(code)
            if len(frame):
                frames.append(frame)
        except Exception as exc:
            print("DAILY_ERROR", code, repr(exc), flush=True)
        if i % 50 == 0:
            print("DAILY", i, "/", len(codes), "frames", len(frames), flush=True)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["code", "date"])
    panel.to_parquet(OUT / "daily_panel.parquet", index=False, compression="zstd")
    return panel


def fetch_entry_hour(code: str, start="2020-01-01", end="2026-11-20") -> pd.DataFrame:
    safe = code.replace(".", "_")
    path = CACHE / f"hourly_{safe}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    rs = bs.query_history_k_data_plus(
        code, fields, start_date=start, end_date=end, frequency="60", adjustflag="2"
    )
    rows = query_rows(rs)
    frame = pd.DataFrame(rows, columns=rs.fields)
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["time"] = frame["time"].astype(str)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


def fetch_hourly_targets(codes: Iterable[str], needed: dict[str, set[pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    codes = sorted(set(codes))
    for i, code in enumerate(codes, 1):
        dates = needed.get(code, set())
        if not dates:
            continue
        try:
            frame = fetch_entry_hour(code)
            frame = frame[frame["date"].isin(dates)].sort_values(["date", "time"])
            first = frame.groupby("date", as_index=False).first()
            first["return_1h"] = np.log(first["close"] / first["open"])
            rows.append(first[["code", "date", "return_1h"]])
        except Exception as exc:
            print("HOUR_ERROR", code, repr(exc), flush=True)
        if i % 50 == 0:
            print("HOUR", i, "/", len(codes), flush=True)
    if not rows:
        return pd.DataFrame(columns=["code", "date", "return_1h"])
    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(OUT / "hourly_targets.parquet", index=False, compression="zstd")
    return result


def prior_robust_z(values: pd.Series, window: int, min_obs: int, clip: float) -> pd.Series:
    prior = values.shift(1)
    roll = prior.rolling(window, min_periods=min_obs)
    med = roll.median()
    q25 = roll.quantile(0.25)
    q75 = roll.quantile(0.75)
    scale = (q75 - q25) / 1.3489795003921634
    scale = scale.where(scale > 1e-8, roll.std())
    return ((values - med) / scale.replace(0, np.nan)).clip(-clip, clip)


def signed_growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    return 200.0 * (current - previous) / (current.abs() + previous.abs() + 1e-8)


def prepare_sources(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy().sort_values(["code", "REPORTDATE", "available_time"])
    lag_cols = ["TOTAL_OPERATE_INCOME", "PARENT_NETPROFIT", "WEIGHTAVG_ROE", "MGJYXJJE", "XSMLL", "BASIC_EPS"]
    lag = frame[["code", "REPORTDATE", *lag_cols]].copy()
    lag["REPORTDATE"] = lag["REPORTDATE"] + pd.DateOffset(years=1)
    lag = lag.rename(columns={c: f"{c}_lag4" for c in lag_cols})
    frame = frame.merge(lag, on=["code", "REPORTDATE"], how="left")

    frame["raw_profit"] = frame["SJLTZ"].where(frame["SJLTZ"].notna(), signed_growth(frame["PARENT_NETPROFIT"], frame["PARENT_NETPROFIT_lag4"]))
    frame["raw_revenue"] = frame["YSTZ"].where(frame["YSTZ"].notna(), signed_growth(frame["TOTAL_OPERATE_INCOME"], frame["TOTAL_OPERATE_INCOME_lag4"]))
    frame["raw_roe"] = frame["WEIGHTAVG_ROE"] - frame["WEIGHTAVG_ROE_lag4"]
    frame["raw_cashflow"] = frame["MGJYXJJE"] - frame["MGJYXJJE_lag4"]
    frame["raw_margin"] = frame["XSMLL"] - frame["XSMLL_lag4"]
    frame["raw_eps"] = signed_growth(frame["BASIC_EPS"], frame["BASIC_EPS_lag4"])

    transforms = {
        "raw_profit": 50.0,
        "raw_revenue": 30.0,
        "raw_roe": 5.0,
        "raw_cashflow": 0.5,
        "raw_margin": 5.0,
        "raw_eps": 50.0,
    }
    zcols = []
    for col, scale in transforms.items():
        frame[f"{col}_t"] = np.arcsinh(frame[col] / scale)
        zcol = f"z_{col.removeprefix('raw_')}"
        frame[zcol] = frame.groupby("code", sort=False)[f"{col}_t"].transform(
            lambda s: prior_robust_z(s, CONTRACT.source_history_window, CONTRACT.source_history_min, CONTRACT.source_clip)
        )
        zcols.append(zcol)
    count = frame[zcols].notna().sum(axis=1)
    frame["source"] = frame[zcols].mean(axis=1).where(count >= 2)
    signs = np.sign(frame[zcols])
    frame["source_coherence"] = signs.sum(axis=1).abs().div(count).where(count >= 2)
    frame["source_quality"] = frame["source"] * (0.5 + 0.5 * frame["source_coherence"])
    return frame


def add_daily_features(daily: pd.DataFrame, industries: dict[str, str]) -> pd.DataFrame:
    d = daily.copy().sort_values(["code", "date"])
    d = d[(d["tradestatus"] == 1) & d["open"].gt(0) & d["close"].gt(0)]
    d["log_ret"] = np.log(d["close"] / d["preclose"])
    d["gap"] = np.log(d["open"] / d["preclose"])
    d["intraday"] = np.log(d["close"] / d["open"])
    d["range"] = (d["high"] - d["low"]) / d["preclose"]
    grouped = d.groupby("code", sort=False)
    for h in [5, 20, 60]:
        d[f"mom{h}"] = grouped["close"].transform(lambda s: np.log(s / s.shift(h)))
    d["vol20"] = grouped["log_ret"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).std())
    d["amount_med20"] = grouped["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).median())
    d["turn_med20"] = grouped["turn"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).median())
    d["volume_shock"] = np.log1p(d["amount"] / d["amount_med20"].replace(0, np.nan))
    d["turn_shock"] = np.log1p(d["turn"] / d["turn_med20"].replace(0, np.nan))
    d["industry"] = d["code"].map(industries).fillna("UNKNOWN")
    d["market_ret"] = d.groupby("date")["log_ret"].transform("median")
    d["industry_ret"] = d.groupby(["date", "industry"])["log_ret"].transform("median")
    d["abnormal_ret"] = d["log_ret"] - 0.5 * d["market_ret"] - 0.5 * d["industry_ret"]
    d["response_z"] = (d["abnormal_ret"] / d["vol20"].replace(0, np.nan)).clip(-CONTRACT.response_clip, CONTRACT.response_clip)
    d["gap_z"] = (d["gap"] / d["vol20"].replace(0, np.nan)).clip(-CONTRACT.response_clip, CONTRACT.response_clip)
    d["intraday_z"] = (d["intraday"] / d["vol20"].replace(0, np.nan)).clip(-CONTRACT.response_clip, CONTRACT.response_clip)
    d["range_z"] = (d["range"] / d["vol20"].replace(0, np.nan)).clip(0, CONTRACT.response_clip)
    d["sid"] = d.groupby("code").cumcount()
    for h in HORIZONS:
        if h == 1:
            d["return_1d"] = np.log(d.groupby("code")["close"].shift(-1) / d.groupby("code")["open"].shift(-1))
            d["entry_date_1d"] = d.groupby("code")["date"].shift(-1)
            d["exit_date_1d"] = d.groupby("code")["date"].shift(-1)
        else:
            entry = d.groupby("code")["open"].shift(-1)
            exit_price = d.groupby("code")["open"].shift(-(h + 1))
            d[f"return_{h}d"] = np.log(exit_price / entry)
            d[f"entry_date_{h}d"] = d.groupby("code")["date"].shift(-1)
            d[f"exit_date_{h}d"] = d.groupby("code")["date"].shift(-(h + 1))
    return d


def map_events(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in daily.groupby("code")}
    response_cols = [
        "code", "date", "open", "close", "preclose", "response_z", "gap_z", "intraday_z", "range_z",
        "volume_shock", "turn_shock", "mom5", "mom20", "mom60", "vol20", "amount", "turn", "isST",
        "market_ret", "industry_ret", "abnormal_ret",
        "return_1d", "entry_date_1d", "exit_date_1d",
        "return_5d", "entry_date_5d", "exit_date_5d",
        "return_20d", "entry_date_20d", "exit_date_20d",
    ]
    for code, ev in events.groupby("code", sort=False):
        data = daily_by_code.get(code)
        if data is None or data.empty:
            continue
        dates = data["date"].to_numpy(dtype="datetime64[ns]")
        avail = ev["available_time"].dt.normalize().to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(dates, avail, side="right")
        valid = positions < len(data)
        if not valid.any():
            continue
        left = ev.loc[valid].reset_index(drop=True)
        right = data.iloc[positions[valid]][response_cols].reset_index(drop=True)
        right = right.rename(columns={"date": "issue_date"})
        pieces.append(pd.concat([left, right.drop(columns="code")], axis=1))
    mapped = pd.concat(pieces, ignore_index=True)
    mapped = mapped[mapped["isST"].fillna(0).eq(0)]
    mapped = mapped[mapped["source"].notna() & mapped["response_z"].notna()]
    return mapped


def build_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    source = p["source_quality"].clip(-CONTRACT.source_clip, CONTRACT.source_clip)
    response = p["response_z"].clip(-CONTRACT.response_clip, CONTRACT.response_clip)
    signed_absorption = np.sign(source) * np.maximum(np.sign(source) * response, 0.0)
    volume_impedance = np.maximum(p["volume_shock"].fillna(0.0), 0.0)
    range_impedance = np.maximum(p["range_z"].fillna(0.0), 0.0)
    p["earnings_source_absorption_gap"] = (source - response).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    p["earnings_source_reservoir"] = (source / (1.0 + response.abs())).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    p["earnings_liquidity_impedance_reservoir"] = (source / (1.0 + response.abs() + 0.5 * volume_impedance)).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    p["earnings_range_impedance_reservoir"] = (source / (1.0 + response.abs() + 0.5 * range_impedance)).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    p["earnings_coherent_underreaction"] = (np.sign(source) * np.maximum(source.abs() - signed_absorption.abs(), 0.0)).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    p["earnings_overshoot_reversal"] = (-np.sign(response) * np.maximum(response.abs() - source.abs(), 0.0)).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    p["earnings_absorption_efficiency_gap"] = (source * (1.0 - np.clip(signed_absorption.abs() / (source.abs() + 0.5), 0.0, 2.0))).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    p["cashflow_absorption_gap"] = (p["z_cashflow"] - response).clip(-CONTRACT.candidate_clip, CONTRACT.candidate_clip)
    return p


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    xr = rankdata(x[valid], method="average")
    yr = rankdata(y[valid], method="average")
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def daily_ic(frame: pd.DataFrame, factor: str, target: str, min_events=MIN_EVENT_COHORT) -> pd.Series:
    out = {}
    for date, group in frame.groupby("issue_date"):
        valid = group[[factor, target]].dropna()
        if len(valid) >= min_events:
            out[pd.Timestamp(date)] = rank_ic(valid[factor], valid[target])
    return pd.Series(out, dtype=float).sort_index()


def time_series_ic(frame: pd.DataFrame, factor: str, target: str) -> tuple[float, float, int]:
    values = []
    for _, group in frame.groupby("code"):
        valid = group[[factor, target]].dropna()
        if len(valid) >= MIN_TS_EVENTS:
            values.append(rank_ic(valid[factor], valid[target]))
    arr = np.asarray(values, float)
    return (float(np.nanmean(arr)) if len(arr) else np.nan,
            float(np.mean(arr > 0)) if len(arr) else np.nan,
            int(len(arr)))


def stage_mask(frame: pd.DataFrame, stage: str, horizon: str) -> pd.Series:
    start, end = STAGES[stage]
    issue = frame["issue_date"]
    exit_col = "exit_date_1d" if horizon == "1d" else f"exit_date_{horizon}"
    exit_date = pd.to_datetime(frame[exit_col])
    return issue.ge(start) & issue.lt(end) & exit_date.notna() & exit_date.lt(end)


def block_indices(n: int, block: int, reps: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, blocks))
    idx = (starts[..., None] + np.arange(block)) % n
    return idx.reshape(reps, -1)[:, :n]


def family_inference(daily_map: dict[str, pd.Series]) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(sorted(set().union(*(set(s.index) for s in daily_map.values()))))
    names = list(daily_map)
    values = np.column_stack([daily_map[n].reindex(calendar).fillna(0.0).to_numpy() for n in names])
    support = np.column_stack([daily_map[n].reindex(calendar).notna().to_numpy() for n in names])
    estimates = np.array([daily_map[n].mean() for n in names])
    influences = np.zeros_like(values)
    for j in range(len(names)):
        count = support[:, j].sum()
        if count:
            influences[support[:, j], j] = len(calendar) / count * (values[support[:, j], j] - estimates[j])
    idx = block_indices(len(calendar), BLOCK, BOOTSTRAPS)
    deviations = np.empty((BOOTSTRAPS, len(names)))
    for b in range(BOOTSTRAPS):
        deviations[b] = influences[idx[b]].mean(axis=0)
    se = deviations.std(axis=0, ddof=1)
    t = np.divide(np.abs(estimates), se, out=np.zeros_like(estimates), where=se > 0)
    max_t = np.max(np.divide(np.abs(deviations), se, out=np.zeros_like(deviations), where=se > 0), axis=1)
    rows = []
    for j, name in enumerate(names):
        low, high = np.quantile(deviations[:, j], [0.025, 0.975])
        p = (1 + np.sum(np.abs(deviations[:, j]) >= abs(estimates[j]))) / (BOOTSTRAPS + 1)
        pmax = (1 + np.sum(max_t >= t[j])) / (BOOTSTRAPS + 1)
        rows.append({"hypothesis": name, "estimate": estimates[j], "se": se[j],
                     "ci_low": estimates[j] - high, "ci_high": estimates[j] - low,
                     "p_two_sided": p, "p_maxT": pmax,
                     "valid_dates": int(support[:, j].sum())})
    out = pd.DataFrame(rows)
    order = np.argsort(out["p_two_sided"].to_numpy())
    adjusted = np.empty(len(out))
    running = 0.0
    for k, ix in enumerate(order):
        running = max(running, min(1.0, out.loc[ix, "p_two_sided"] * (len(out) - k)))
        adjusted[ix] = running
    out["p_holm"] = adjusted
    return out


BASE_FEATURES = [
    "source", "source_quality", "source_coherence", "z_profit", "z_revenue", "z_roe",
    "z_cashflow", "z_margin", "z_eps", "response_z", "gap_z", "intraday_z", "range_z",
    "volume_shock", "turn_shock", "mom5", "mom20", "mom60", "vol20",
]


class Transform:
    def fit(self, x: np.ndarray):
        self.med = np.nanmedian(x, axis=0)
        self.med = np.nan_to_num(self.med)
        filled = np.where(np.isfinite(x), x, self.med)
        self.lo = np.quantile(filled, 0.01, axis=0)
        self.hi = np.quantile(filled, 0.99, axis=0)
        clipped = np.clip(filled, self.lo, self.hi)
        self.mean = clipped.mean(axis=0)
        self.std = clipped.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(x)
        filled = np.where(~missing, x, self.med)
        clipped = np.clip(filled, self.lo, self.hi)
        return np.concatenate([(clipped - self.mean) / self.std, missing.astype(float)], axis=1)


def fit_predict(xtr, ytr, xte, model):
    transform = Transform().fit(xtr)
    a = transform.transform(xtr)
    b = transform.transform(xte)
    if model == "ridge":
        estimator = Ridge(alpha=0.02 * len(ytr))
    else:
        estimator = HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, min_samples_leaf=80,
            learning_rate=0.05, l2_regularization=10.0, early_stopping=False,
            random_state=SEED,
        )
    estimator.fit(a, np.clip(ytr, -0.35, 0.35))
    return estimator.predict(b)


def model_increment(frame: pd.DataFrame, candidate: str, target: str, exit_col: str) -> pd.DataFrame:
    data = frame.copy().sort_values("issue_date")
    years = data["issue_date"].dt.year.to_numpy()
    target_values = data[target].to_numpy(float)
    exit_dates = pd.to_datetime(data[exit_col]).to_numpy(dtype="datetime64[ns]")
    base = data[BASE_FEATURES].to_numpy(float)
    extra = data[[candidate]].to_numpy(float)
    rows = []
    predictions = []
    for year in sorted(set(years)):
        if year < 2020:
            continue
        test = years == year
        if not test.any():
            continue
        first = np.datetime64(pd.Timestamp(year=year, month=1, day=1))
        train = (exit_dates < first) & np.isfinite(target_values)
        if train.sum() < CONTRACT.minimum_model_rows:
            continue
        for model in ("ridge", "hgb"):
            p0 = fit_predict(base[train], target_values[train], base[test], model)
            p1 = fit_predict(np.column_stack([base[train], extra[train]]), target_values[train],
                             np.column_stack([base[test], extra[test]]), model)
            part = data.loc[test, ["issue_date", "code", target]].copy()
            part["candidate"] = candidate
            part["model"] = model
            part["base_prediction"] = p0
            part["aug_prediction"] = p1
            predictions.append(part)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if pred.empty:
        return pd.DataFrame()
    for stage in ("validation", "historical_holdout", "recent_extension"):
        start, end = STAGES[stage]
        d = pred[pred["issue_date"].between(start, end, inclusive="left")]
        for model, g in d.groupby("model"):
            valid = g[[target, "base_prediction", "aug_prediction"]].dropna()
            if not len(valid):
                continue
            loss0 = (valid[target] - valid["base_prediction"]) ** 2
            loss1 = (valid[target] - valid["aug_prediction"]) ** 2
            daily = pd.DataFrame({"date": g.loc[valid.index, "issue_date"], "loss0": loss0, "loss1": loss1}).groupby("date").mean()
            mse0, mse1 = daily["loss0"].mean(), daily["loss1"].mean()
            ic0 = daily_ic(g, "base_prediction", target).mean()
            ic1 = daily_ic(g, "aug_prediction", target).mean()
            rows.append({"candidate": candidate, "target": target, "stage": stage, "model": model,
                         "relative_mse_gain": (mse0 - mse1) / mse0 if mse0 > 0 else np.nan,
                         "base_rankic": ic0, "aug_rankic": ic1, "rankic_increment": ic1 - ic0,
                         "rows": len(valid), "dates": daily.shape[0]})
    pred.to_parquet(OUT / f"predictions_{candidate}_{target}.parquet", index=False, compression="zstd")
    return pd.DataFrame(rows)


def residual_ic(frame: pd.DataFrame, candidate: str, target: str, exit_col: str) -> pd.DataFrame:
    data = frame.copy().sort_values("issue_date")
    years = data["issue_date"].dt.year.to_numpy()
    exit_dates = pd.to_datetime(data[exit_col]).to_numpy(dtype="datetime64[ns]")
    x = data[BASE_FEATURES].to_numpy(float)
    y = data[candidate].to_numpy(float)
    residual = np.full(len(data), np.nan)
    for year in sorted(set(years)):
        if year < 2020:
            continue
        test = years == year
        first = np.datetime64(pd.Timestamp(year=year, month=1, day=1))
        train = (exit_dates < first) & np.isfinite(y)
        if train.sum() < CONTRACT.minimum_model_rows:
            continue
        transform = Transform().fit(x[train])
        model = Ridge(alpha=0.02 * train.sum()).fit(transform.transform(x[train]), y[train])
        residual[test] = y[test] - model.predict(transform.transform(x[test]))
    data["candidate_residual"] = residual
    rows = []
    for stage in ("validation", "historical_holdout", "recent_extension"):
        d = data[stage_mask(data, stage, target.removeprefix("return_"))]
        cs = daily_ic(d, "candidate_residual", target)
        ts, pos, products = time_series_ic(d, "candidate_residual", target)
        rows.append({"candidate": candidate, "target": target, "stage": stage,
                     "residual_rankic": cs.mean(), "residual_rankic_dates": len(cs),
                     "residual_tsic": ts, "residual_positive_share": pos, "residual_products": products})
    return pd.DataFrame(rows)


def account_diagnostic(frame: pd.DataFrame, candidate: str, target: str) -> pd.DataFrame:
    horizon = target.removeprefix("return_")
    rows = []
    for stage in ("validation", "historical_holdout", "recent_extension"):
        d = frame[stage_mask(frame, stage, horizon)].copy()
        cohorts = []
        for date, group in d.groupby("issue_date"):
            valid = group[[candidate, target, "market_ret"]].dropna().sort_values(candidate)
            if len(valid) < 10:
                continue
            k = max(2, int(math.floor(len(valid) * 0.2)))
            bottom, top = valid.head(k), valid.tail(k)
            gross_ls = top[target].mean() - bottom[target].mean()
            gross_long_active = top[target].mean() - valid[target].mean()
            cohorts.append((date, gross_ls, gross_long_active, len(valid)))
        if not cohorts:
            continue
        cohort = pd.DataFrame(cohorts, columns=["date", "gross_ls", "gross_long_active", "n"])
        for cost in COSTS_BPS:
            # Dollar-neutral long-short and long-only both turn over approximately two gross units per cohort.
            charge = 2.0 * cost * 1e-4
            rows.append({"candidate": candidate, "target": target, "stage": stage,
                         "cost_per_side_bp": cost, "cohorts": len(cohort),
                         "gross_ls_mean": cohort["gross_ls"].mean(),
                         "net_ls_mean": cohort["gross_ls"].mean() - charge,
                         "gross_long_active_mean": cohort["gross_long_active"].mean(),
                         "net_long_active_mean": cohort["gross_long_active"].mean() - charge,
                         "positive_ls_share": float((cohort["gross_ls"] > 0).mean()),
                         "positive_long_active_share": float((cohort["gross_long_active"] > 0).mean())})
    return pd.DataFrame(rows)


def evaluate(panel: pd.DataFrame) -> dict:
    candidates = [
        "earnings_source_absorption_gap", "earnings_source_reservoir",
        "earnings_liquidity_impedance_reservoir", "earnings_range_impedance_reservoir",
        "earnings_coherent_underreaction", "earnings_overshoot_reversal",
        "earnings_absorption_efficiency_gap", "cashflow_absorption_gap",
    ]
    targets = {"1h": "return_1h", "1d": "return_1d", "5d": "return_5d", "20d": "return_20d"}
    direct_rows = []
    daily_maps: dict[str, pd.Series] = {}
    residual_frames = []
    model_frames = []
    accounts = []
    for horizon, target in targets.items():
        if target not in panel or panel[target].notna().sum() < 100:
            continue
        for candidate in candidates:
            key = f"{candidate}|{horizon}"
            for stage in STAGES:
                d = panel[stage_mask(panel, stage, horizon)]
                cs = daily_ic(d, candidate, target)
                ts, pos, products = time_series_ic(d, candidate, target)
                direct_rows.append({"hypothesis": key, "candidate": candidate, "horizon": horizon,
                                    "stage": stage, "rankic": cs.mean(), "rankic_dates": len(cs),
                                    "tsic": ts, "positive_ts_share": pos, "ts_products": products,
                                    "rows": int(d[[candidate, target]].dropna().shape[0])})
                if stage in ("validation", "historical_holdout", "recent_extension"):
                    daily_maps[f"{stage}|{key}"] = cs
            exit_col = "exit_date_1d" if horizon == "1d" else ("entry_date_1d" if horizon == "1h" else f"exit_date_{horizon}")
            residual_frames.append(residual_ic(panel, candidate, target, exit_col))
            model_frames.append(model_increment(panel, candidate, target, exit_col))
            accounts.append(account_diagnostic(panel, candidate, target))
            print("EVALUATED", key, flush=True)
    direct = pd.DataFrame(direct_rows)
    inference = family_inference(daily_maps)
    residuals = pd.concat(residual_frames, ignore_index=True)
    models = pd.concat([x for x in model_frames if len(x)], ignore_index=True)
    account = pd.concat([x for x in accounts if len(x)], ignore_index=True)
    direct.to_csv(OUT / "direct_metrics.csv", index=False)
    inference.to_csv(OUT / "family_inference.csv", index=False)
    residuals.to_csv(OUT / "residual_metrics.csv", index=False)
    models.to_csv(OUT / "model_increment.csv", index=False)
    account.to_csv(OUT / "account_diagnostics.csv", index=False)
    return {"direct": direct, "inference": inference, "residuals": residuals,
            "models": models, "accounts": account, "candidates": candidates, "targets": targets}


def admission(results: dict) -> pd.DataFrame:
    direct = results["direct"]
    inference = results["inference"]
    residuals = results["residuals"]
    models = results["models"]
    accounts = results["accounts"]
    rows = []
    for candidate in results["candidates"]:
        for horizon, target in results["targets"].items():
            ds = direct[(direct.candidate == candidate) & (direct.horizon == horizon)]
            if ds.empty:
                continue
            metric = {r.stage: r for r in ds.itertuples()}
            rs = residuals[(residuals.candidate == candidate) & (residuals.target == target)]
            ms = models[(models.candidate == candidate) & (models.target == target) & (models.model == "hgb")]
            ac = accounts[(accounts.candidate == candidate) & (accounts.target == target) & (accounts.cost_per_side_bp == 20)]
            inf = {stage: inference[inference.hypothesis == f"{stage}|{candidate}|{horizon}"] for stage in ("validation", "historical_holdout", "recent_extension")}
            direction = all(stage in metric and metric[stage].rankic > 0.005 for stage in inf)
            residual = all((rs[rs.stage == stage].residual_rankic > 0.003).any() for stage in inf)
            model = all((ms[ms.stage == stage].relative_mse_gain > 0).any() for stage in inf)
            fwer = all(len(inf[stage]) and float(inf[stage].iloc[0].p_maxT) <= 0.10 for stage in inf)
            cost = all((ac[ac.stage == stage].net_long_active_mean > 0).any() for stage in inf)
            admitted = direction and residual and model and fwer and cost
            rows.append({"candidate": candidate, "horizon": horizon, "target": target,
                         "passes_direction": direction, "passes_residual": residual,
                         "passes_model": model, "passes_fwer": fwer, "passes_cost": cost,
                         "admitted_historical_factor": admitted,
                         "independent_alpha": False, "default_enabled": False,
                         "fresh_forward_required": True})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "admission.csv", index=False)
    return out


def main():
    receipt = {"contract": asdict(CONTRACT), "status": "started"}
    try:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(login.error_msg)
        reports = fetch_reports()
        membership = fetch_membership()
        events = filter_events(reports, membership)
        events = prepare_sources(events)
        codes = sorted(events.loc[events["source"].notna(), "code"].unique())
        print("ELIGIBLE", len(events), "SOURCE", events.source.notna().sum(), "CODES", len(codes), flush=True)
        daily = fetch_daily_panel(codes)
        industries = events.sort_values("available_time").groupby("code")["industry"].last().to_dict()
        daily = add_daily_features(daily, industries)
        panel = map_events(events, daily)
        panel = build_candidates(panel)
        needed = {}
        for code, group in panel.groupby("code"):
            needed[code] = set(pd.to_datetime(group["entry_date_1d"].dropna()).dt.normalize())
        hourly = fetch_hourly_targets(panel["code"].unique(), needed)
        panel = panel.merge(hourly.rename(columns={"date": "entry_date_1d"}), on=["code", "entry_date_1d"], how="left")
        panel.to_parquet(OUT / "research_panel.parquet", index=False, compression="zstd")
        results = evaluate(panel)
        admitted = admission(results)
        summary = {
            "status": "completed",
            "events": len(panel),
            "codes": int(panel.code.nunique()),
            "issue_dates": int(panel.issue_date.nunique()),
            "hourly_labels": int(panel.return_1h.notna().sum()),
            "candidate_count": len(results["candidates"]),
            "admitted": admitted.loc[admitted.admitted_historical_factor, ["candidate", "horizon"]].to_dict("records"),
            "fresh_forward": False,
            "independent_alpha": False,
        }
        receipt.update(summary)
        (OUT / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        bs.logout()
    except Exception as exc:
        receipt.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        (OUT / "FAILURE.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        try:
            bs.logout()
        except Exception:
            pass
        raise
    finally:
        (OUT / "RUN_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
