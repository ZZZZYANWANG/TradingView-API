from __future__ import annotations

import json
import math
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "ashare_event_absorption"))
import run_research_v6_resilient as resilient

base = resilient.research

OUT = Path("research_outputs/ashare_share_supply")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

SOURCE_SCALE = 0.05
SOURCE_CLIP = 5.0
CANDIDATE_CLIP = 8.0
MIN_EVENT_COHORT = 5


def retry_json(url: str, params: dict, attempts: int = 6) -> dict:
    last = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=(10, 60))
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") is False:
                raise RuntimeError(payload.get("message"))
            return payload
        except Exception as exc:
            last = exc
            print("SUPPLY_RETRY", attempt, "/", attempts, params.get("filter"), repr(exc), flush=True)
            if attempt < attempts:
                time.sleep(min(30.0, 3.0 * attempt))
    raise RuntimeError("share-structure request failed") from last


def em_code(code: str) -> str:
    market, number = code.split(".")
    suffix = "SH" if market.lower() == "sh" else "SZ"
    return f"{number}.{suffix}"


def fetch_one_structure(code: str) -> pd.DataFrame:
    safe = code.replace(".", "_")
    path = CACHE / f"share_{safe}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    columns = (
        "SECUCODE,SECURITY_CODE,END_DATE,TOTAL_SHARES,LIMITED_SHARES,UNLIMITED_SHARES,"
        "LISTED_A_SHARES,FREE_SHARES,LIMITED_A_SHARES,CHANGE_REASON"
    )
    params = {
        "reportName": "RPT_F10_EH_EQUITY",
        "columns": columns,
        "quoteColumns": "",
        "filter": f'(SECUCODE="{em_code(code)}")',
        "pageNumber": "1",
        "pageSize": "500",
        "sortTypes": "1",
        "sortColumns": "END_DATE",
        "source": "HSF10",
        "client": "PC",
    }
    first = retry_json(url, params)
    result = first.get("result") or {}
    rows = list(result.get("data") or [])
    pages = int(result.get("pages") or 1)
    for page in range(2, pages + 1):
        rows.extend((retry_json(url, dict(params, pageNumber=str(page))).get("result") or {}).get("data") or [])
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["code"] = code
        frame["END_DATE"] = pd.to_datetime(frame["END_DATE"], errors="coerce")
        for column in [
            "TOTAL_SHARES", "LIMITED_SHARES", "UNLIMITED_SHARES", "LISTED_A_SHARES",
            "FREE_SHARES", "LIMITED_A_SHARES",
        ]:
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        frame = frame.dropna(subset=["END_DATE"]).sort_values("END_DATE").drop_duplicates(
            ["END_DATE", "CHANGE_REASON"], keep="last"
        )
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


def fetch_structures(codes: list[str]) -> pd.DataFrame:
    frames = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fetch_one_structure, code): code for code in codes}
        for number, future in enumerate(as_completed(future_map), 1):
            code = future_map[future]
            try:
                frame = future.result()
                if len(frame):
                    frames.append(frame)
            except Exception as exc:
                print("SUPPLY_ERROR", code, repr(exc), flush=True)
            if number % 50 == 0:
                print("SUPPLY_FETCH", number, "/", len(codes), "frames", len(frames), flush=True)
    result = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    result.to_parquet(OUT / "raw_share_structure.parquet", index=False, compression="zstd")
    return result


def asinh_source(change: pd.Series) -> pd.Series:
    return (-np.arcsinh(change / SOURCE_SCALE)).clip(-SOURCE_CLIP, SOURCE_CLIP)


def prepare_events(raw: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().sort_values(["code", "END_DATE"], kind="mergesort")
    for column in ["TOTAL_SHARES", "LISTED_A_SHARES", "FREE_SHARES", "LIMITED_A_SHARES"]:
        prior = frame.groupby("code", sort=False)[column].shift(1)
        frame[f"dlog_{column.lower()}"] = np.log(frame[column] / prior).replace([np.inf, -np.inf], np.nan)
    reason = frame["CHANGE_REASON"].fillna("").astype(str)
    frame["is_initial"] = reason.str.contains("首发|首次公开|IPO", regex=True)
    frame["is_bonus"] = reason.str.contains("送股|转增|公积金|利润分配", regex=True)
    frame["is_unlock"] = reason.str.contains("限售|解禁|上市流通|股改", regex=True)
    frame["is_issue"] = reason.str.contains("增发|配股|可转债转股|行权|股权激励|发行", regex=True)
    frame["is_buyback"] = reason.str.contains("回购|注销|减资", regex=True)
    frame["membership_year"] = frame["END_DATE"].dt.year - 1
    member = membership.copy()
    member["membership_year"] = pd.to_datetime(member["snapshot_date"]).dt.year
    eligible = set(zip(member["membership_year"], member["code"]))
    frame = frame[[tuple(value) in eligible for value in frame[["membership_year", "code"]].itertuples(index=False, name=None)]]
    frame = frame[frame["END_DATE"].between("2016-01-01", "2026-09-30")]
    frame = frame[~frame["is_initial"] & ~frame["is_bonus"]]

    frame["source_total"] = asinh_source(frame["dlog_total_shares"])
    frame["source_float"] = asinh_source(frame["dlog_listed_a_shares"])
    unlock_change = frame["dlog_listed_a_shares"] - frame["dlog_total_shares"].fillna(0.0)
    frame["source_unlock"] = asinh_source(unlock_change).where(frame["is_unlock"])
    frame["source_issue"] = frame["source_total"].where(frame["is_issue"])
    frame["source_buyback"] = frame["source_total"].where(frame["is_buyback"])
    frame["source"] = np.select(
        [frame["is_unlock"], frame["is_issue"], frame["is_buyback"]],
        [frame["source_float"], frame["source_total"], frame["source_total"]],
        default=np.nan,
    )
    frame["source_quality"] = frame["source"]
    frame["source_coherence"] = 1.0
    frame["available_time"] = frame["END_DATE"]
    frame["industry"] = "UNKNOWN"
    frame["REPORTDATE"] = frame["END_DATE"]
    frame["issue_year"] = frame["END_DATE"].dt.year
    frame["report_quarter"] = frame["END_DATE"].dt.quarter
    frame = frame[frame["source"].notna()]
    frame.to_parquet(OUT / "eligible_share_events.parquet", index=False, compression="zstd")
    return frame


def build_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    response = data["response_z"].clip(-base.CONTRACT.response_clip, base.CONTRACT.response_clip)
    source = data["source"].clip(-SOURCE_CLIP, SOURCE_CLIP)
    signed_absorption = np.sign(source) * np.maximum(np.sign(source) * response, 0.0)
    volume_impedance = np.maximum(data["volume_shock"].fillna(0.0), 0.0)
    data["share_supply_absorption_gap"] = (source - response).clip(-CANDIDATE_CLIP, CANDIDATE_CLIP)
    data["share_supply_reservoir"] = (source / (1.0 + response.abs())).clip(-CANDIDATE_CLIP, CANDIDATE_CLIP)
    data["share_supply_liquidity_reservoir"] = (
        source / (1.0 + response.abs() + 0.5 * volume_impedance)
    ).clip(-CANDIDATE_CLIP, CANDIDATE_CLIP)
    data["share_supply_coherent_underreaction"] = (
        np.sign(source) * np.maximum(source.abs() - signed_absorption.abs(), 0.0)
    ).clip(-CANDIDATE_CLIP, CANDIDATE_CLIP)
    data["share_unlock_absorption_gap"] = (
        data["source_unlock"] - response
    ).where(data["is_unlock"]).clip(-CANDIDATE_CLIP, CANDIDATE_CLIP)
    data["share_financing_absorption_gap"] = (
        data["source_issue"] - response
    ).where(data["is_issue"]).clip(-CANDIDATE_CLIP, CANDIDATE_CLIP)
    data["share_buyback_absorption_gap"] = (
        data["source_buyback"] - response
    ).where(data["is_buyback"]).clip(-CANDIDATE_CLIP, CANDIDATE_CLIP)
    return data


CANDIDATES = [
    "share_supply_absorption_gap",
    "share_supply_reservoir",
    "share_supply_liquidity_reservoir",
    "share_supply_coherent_underreaction",
    "share_unlock_absorption_gap",
    "share_financing_absorption_gap",
    "share_buyback_absorption_gap",
]

base.MIN_EVENT_COHORT = MIN_EVENT_COHORT
base.BASE_FEATURES = [
    "source", "source_total", "source_float", "source_unlock", "source_issue", "source_buyback",
    "is_unlock", "is_issue", "is_buyback", "response_z", "gap_z", "intraday_z", "range_z",
    "volume_shock", "turn_shock", "mom5", "mom20", "mom60", "vol20",
]


def evaluate(panel: pd.DataFrame) -> dict:
    targets = {"1d": "return_1d", "5d": "return_5d", "20d": "return_20d"}
    direct_rows = []
    daily_maps: dict[str, pd.Series] = {}
    residual_frames = []
    model_frames = []
    accounts = []
    for horizon, target in targets.items():
        if target not in panel or panel[target].notna().sum() < 100:
            continue
        for candidate in CANDIDATES:
            key = f"{candidate}|{horizon}"
            for stage in base.STAGES:
                data = panel[base.stage_mask(panel, stage, horizon)]
                cross_section = base.daily_ic(data, candidate, target, min_events=MIN_EVENT_COHORT)
                tsic, positive_share, products = base.time_series_ic(data, candidate, target)
                direct_rows.append({
                    "hypothesis": key, "candidate": candidate, "horizon": horizon,
                    "stage": stage, "rankic": cross_section.mean(), "rankic_dates": len(cross_section),
                    "tsic": tsic, "positive_ts_share": positive_share, "ts_products": products,
                    "rows": int(data[[candidate, target]].dropna().shape[0]),
                })
                if stage in ("validation", "historical_holdout", "recent_extension"):
                    daily_maps[f"{stage}|{key}"] = cross_section
            exit_column = "exit_date_1d" if horizon == "1d" else f"exit_date_{horizon}"
            residual_frames.append(base.residual_ic(panel, candidate, target, exit_column))
            model_frames.append(base.model_increment(panel, candidate, target, exit_column))
            accounts.append(base.account_diagnostic(panel, candidate, target))
            print("SUPPLY_EVALUATED", key, flush=True)
    direct = pd.DataFrame(direct_rows)
    inference = base.family_inference(daily_maps)
    residuals = pd.concat(residual_frames, ignore_index=True)
    models = pd.concat([frame for frame in model_frames if len(frame)], ignore_index=True)
    account = pd.concat([frame for frame in accounts if len(frame)], ignore_index=True)
    direct.to_csv(OUT / "direct_metrics.csv", index=False)
    inference.to_csv(OUT / "family_inference.csv", index=False)
    residuals.to_csv(OUT / "residual_metrics.csv", index=False)
    models.to_csv(OUT / "model_increment.csv", index=False)
    account.to_csv(OUT / "account_diagnostics.csv", index=False)
    return {
        "direct": direct, "inference": inference, "residuals": residuals,
        "models": models, "accounts": account, "candidates": CANDIDATES, "targets": targets,
    }


def main() -> None:
    receipt = {"status": "started", "source_scale": SOURCE_SCALE, "candidates": CANDIDATES}
    try:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(login.error_msg)
        membership = base.fetch_membership()
        codes = sorted(set(membership["code"]))
        raw = fetch_structures(codes)
        events = prepare_events(raw, membership)
        event_codes = sorted(set(events["code"]))
        print("SUPPLY_ELIGIBLE", len(events), "CODES", len(event_codes), flush=True)
        daily = base.fetch_daily_panel(event_codes)
        daily = base.add_daily_features(daily, {})
        panel = base.map_events(events, daily)
        panel = build_candidates(panel)
        panel.to_parquet(OUT / "research_panel.parquet", index=False, compression="zstd")
        results = evaluate(panel)
        admitted = base.admission(results)
        summary = {
            "status": "completed", "events": len(panel), "codes": int(panel.code.nunique()),
            "issue_dates": int(panel.issue_date.nunique()), "candidate_count": len(CANDIDATES),
            "admitted": admitted.loc[
                admitted.admitted_historical_factor, ["candidate", "horizon"]
            ].to_dict("records"),
            "fresh_forward": False, "independent_alpha": False,
            "availability_boundary": "share END_DATE used after close; next-session trading only",
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
