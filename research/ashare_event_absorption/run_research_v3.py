from __future__ import annotations

import numpy as np
import pandas as pd

import run_research as research


def stage_mask(frame: pd.DataFrame, stage: str, horizon: str) -> pd.Series:
    start, end = research.STAGES[stage]
    issue = pd.to_datetime(frame["issue_date"])
    if horizon == "1h":
        exit_date = pd.to_datetime(frame["entry_date_1d"])
    elif horizon == "1d":
        exit_date = pd.to_datetime(frame["exit_date_1d"])
    else:
        exit_date = pd.to_datetime(frame[f"exit_date_{horizon}"])
    return issue.ge(start) & issue.lt(end) & exit_date.notna() & exit_date.lt(end)


def prepare_sources(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    lag_cols = [
        "TOTAL_OPERATE_INCOME", "PARENT_NETPROFIT", "WEIGHTAVG_ROE",
        "MGJYXJJE", "XSMLL", "BASIC_EPS",
    ]
    lag = frame[["code", "REPORTDATE", "available_time", *lag_cols]].copy()
    lag["REPORTDATE"] = lag["REPORTDATE"] + pd.DateOffset(years=1)
    lag = lag.rename(columns={
        "available_time": "lag_available_time",
        **{column: f"{column}_lag4" for column in lag_cols},
    })
    frame = frame.merge(lag, on=["code", "REPORTDATE"], how="left")
    visible = frame["lag_available_time"].notna() & (frame["lag_available_time"] < frame["available_time"])
    for column in lag_cols:
        frame.loc[~visible, f"{column}_lag4"] = np.nan

    frame["raw_profit"] = frame["SJLTZ"].where(
        frame["SJLTZ"].notna(),
        research.signed_growth(frame["PARENT_NETPROFIT"], frame["PARENT_NETPROFIT_lag4"]),
    )
    frame["raw_revenue"] = frame["YSTZ"].where(
        frame["YSTZ"].notna(),
        research.signed_growth(frame["TOTAL_OPERATE_INCOME"], frame["TOTAL_OPERATE_INCOME_lag4"]),
    )
    frame["raw_roe"] = frame["WEIGHTAVG_ROE"] - frame["WEIGHTAVG_ROE_lag4"]
    frame["raw_cashflow"] = frame["MGJYXJJE"] - frame["MGJYXJJE_lag4"]
    frame["raw_margin"] = frame["XSMLL"] - frame["XSMLL_lag4"]
    frame["raw_eps"] = research.signed_growth(frame["BASIC_EPS"], frame["BASIC_EPS_lag4"])

    transforms = {
        "raw_profit": 50.0,
        "raw_revenue": 30.0,
        "raw_roe": 5.0,
        "raw_cashflow": 0.5,
        "raw_margin": 5.0,
        "raw_eps": 50.0,
    }
    # Strict point-in-time history: an observation enters the stock's scale only
    # after its database availability timestamp, never merely because its report
    # period is earlier.
    frame = frame.sort_values(["code", "available_time", "REPORTDATE"], kind="mergesort")
    zcols = []
    for column, scale in transforms.items():
        frame[f"{column}_t"] = np.arcsinh(frame[column] / scale)
        zcol = f"z_{column.removeprefix('raw_')}"
        frame[zcol] = frame.groupby("code", sort=False)[f"{column}_t"].transform(
            lambda series: research.prior_robust_z(
                series,
                research.CONTRACT.source_history_window,
                research.CONTRACT.source_history_min,
                research.CONTRACT.source_clip,
            )
        )
        zcols.append(zcol)
    count = frame[zcols].notna().sum(axis=1)
    frame["source"] = frame[zcols].mean(axis=1).where(count >= 2)
    signs = np.sign(frame[zcols])
    frame["source_coherence"] = signs.sum(axis=1).abs().div(count).where(count >= 2)
    frame["source_quality"] = frame["source"] * (0.5 + 0.5 * frame["source_coherence"])
    return frame


def add_daily_features(daily: pd.DataFrame, industries: dict[str, str]) -> pd.DataFrame:
    # Industry labels in the public report feed are not guaranteed point-in-time.
    # The causal response control therefore uses only the same-date market median;
    # the event's own industry label remains available for later robustness slices.
    data = daily.copy().sort_values(["code", "date"])
    data = data[(data["tradestatus"] == 1) & data["open"].gt(0) & data["close"].gt(0)]
    data["log_ret"] = np.log(data["close"] / data["preclose"])
    data["gap"] = np.log(data["open"] / data["preclose"])
    data["intraday"] = np.log(data["close"] / data["open"])
    data["range"] = (data["high"] - data["low"]) / data["preclose"]
    grouped = data.groupby("code", sort=False)
    for horizon in [5, 20, 60]:
        data[f"mom{horizon}"] = grouped["close"].transform(lambda series: np.log(series / series.shift(horizon)))
    data["vol20"] = grouped["log_ret"].transform(lambda series: series.shift(1).rolling(20, min_periods=15).std())
    data["amount_med20"] = grouped["amount"].transform(lambda series: series.shift(1).rolling(20, min_periods=15).median())
    data["turn_med20"] = grouped["turn"].transform(lambda series: series.shift(1).rolling(20, min_periods=15).median())
    data["volume_shock"] = np.log1p(data["amount"] / data["amount_med20"].replace(0, np.nan))
    data["turn_shock"] = np.log1p(data["turn"] / data["turn_med20"].replace(0, np.nan))
    data["industry"] = data["code"].map(industries).fillna("UNKNOWN")
    data["market_ret"] = data.groupby("date")["log_ret"].transform("median")
    data["industry_ret"] = data["market_ret"]
    data["abnormal_ret"] = data["log_ret"] - data["market_ret"]
    data["response_z"] = (data["abnormal_ret"] / data["vol20"].replace(0, np.nan)).clip(-research.CONTRACT.response_clip, research.CONTRACT.response_clip)
    data["gap_z"] = (data["gap"] / data["vol20"].replace(0, np.nan)).clip(-research.CONTRACT.response_clip, research.CONTRACT.response_clip)
    data["intraday_z"] = (data["intraday"] / data["vol20"].replace(0, np.nan)).clip(-research.CONTRACT.response_clip, research.CONTRACT.response_clip)
    data["range_z"] = (data["range"] / data["vol20"].replace(0, np.nan)).clip(0, research.CONTRACT.response_clip)
    data["sid"] = data.groupby("code").cumcount()
    data["return_1d"] = np.log(data.groupby("code")["close"].shift(-1) / data.groupby("code")["open"].shift(-1))
    data["entry_date_1d"] = data.groupby("code")["date"].shift(-1)
    data["exit_date_1d"] = data["entry_date_1d"]
    for horizon in [5, 20]:
        entry = data.groupby("code")["open"].shift(-1)
        exit_price = data.groupby("code")["open"].shift(-(horizon + 1))
        data[f"return_{horizon}d"] = np.log(exit_price / entry)
        data[f"entry_date_{horizon}d"] = data.groupby("code")["date"].shift(-1)
        data[f"exit_date_{horizon}d"] = data.groupby("code")["date"].shift(-(horizon + 1))
    return data


research.stage_mask = stage_mask
research.prepare_sources = prepare_sources
research.add_daily_features = add_daily_features


if __name__ == "__main__":
    research.main()
