from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "ashare_event_absorption"))
import run_research_v6_resilient as resilient

base = resilient.research

OUT = Path("research_outputs/ashare_financial_revision")
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "TOTAL_OPERATE_INCOME", "PARENT_NETPROFIT", "WEIGHTAVG_ROE",
    "MGJYXJJE", "XSMLL", "BASIC_EPS",
]
COMPONENTS = [
    "revision_profit", "revision_revenue", "revision_roe",
    "revision_cashflow", "revision_margin", "revision_eps",
]
CANDIDATES = [
    "financial_revision_absorption_gap",
    "financial_revision_reservoir",
    "financial_revision_liquidity_reservoir",
    "financial_revision_range_reservoir",
    "financial_revision_coherent_underreaction",
    "financial_revision_overshoot_reversal",
]


def symmetric_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    denominator = current.abs() + previous.abs()
    return (2.0 * (current - previous) / denominator.where(denominator > 1e-12)).clip(-2.0, 2.0)


def filter_events(raw: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame[frame["SECUCODE"].astype(str).str.endswith((".SH", ".SZ"), na=False)]
    frame = frame[frame["SECURITY_CODE"].astype(str).str.match(r"^(0|3|6)")]
    for column in ("REPORTDATE", "NOTICE_DATE", "UPDATE_DATE", "EITIME"):
        frame[column] = pd.to_datetime(frame.get(column), errors="coerce")
    frame["available_time"] = frame[["NOTICE_DATE", "UPDATE_DATE", "EITIME"]].max(axis=1)
    frame = frame[frame["available_time"].notna() & frame["REPORTDATE"].notna()]
    lag = (frame["available_time"].dt.normalize() - frame["REPORTDATE"].dt.normalize()).dt.days
    frame = frame[lag.between(0, base.CONTRACT.event_max_report_lag_days)]
    frame["code"] = frame["SECUCODE"].map(base.code_to_baostock)
    frame["membership_year"] = frame["available_time"].dt.year - 1
    member = membership.copy()
    member["membership_year"] = pd.to_datetime(member["snapshot_date"]).dt.year
    eligible = set(zip(member["membership_year"], member["code"]))
    frame = frame[[tuple(value) in eligible for value in frame[["membership_year", "code"]].itertuples(index=False, name=None)]]
    for column in FIELDS:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["industry"] = frame.get("BOARD_NAME").fillna(frame.get("PUBLISHNAME")).fillna("UNKNOWN").astype(str)
    frame["completeness"] = frame[FIELDS].notna().sum(axis=1)
    frame = frame.sort_values(
        ["code", "REPORTDATE", "available_time", "completeness"],
        ascending=[True, True, True, False], kind="mergesort",
    ).drop_duplicates(["code", "REPORTDATE", "available_time"], keep="first")
    frame["revision_number"] = frame.groupby(["code", "REPORTDATE"], sort=False).cumcount()
    previous_available = frame.groupby(["code", "REPORTDATE"], sort=False)["available_time"].shift(1)
    frame["revision_latency_days"] = (
        frame["available_time"].dt.normalize() - previous_available.dt.normalize()
    ).dt.days
    previous = frame.groupby(["code", "REPORTDATE"], sort=False)[FIELDS].shift(1)
    frame["revision_profit"] = symmetric_change(frame["PARENT_NETPROFIT"], previous["PARENT_NETPROFIT"])
    frame["revision_revenue"] = symmetric_change(frame["TOTAL_OPERATE_INCOME"], previous["TOTAL_OPERATE_INCOME"])
    frame["revision_roe"] = np.arcsinh((frame["WEIGHTAVG_ROE"] - previous["WEIGHTAVG_ROE"]) / 5.0)
    frame["revision_cashflow"] = np.arcsinh((frame["MGJYXJJE"] - previous["MGJYXJJE"]) / 0.5)
    frame["revision_margin"] = np.arcsinh((frame["XSMLL"] - previous["XSMLL"]) / 5.0)
    frame["revision_eps"] = symmetric_change(frame["BASIC_EPS"], previous["BASIC_EPS"])
    frame = frame[frame["revision_number"].ge(1)]
    count = frame[COMPONENTS].notna().sum(axis=1)
    frame["source"] = frame[COMPONENTS].mean(axis=1).where(count >= 2).clip(-base.CONTRACT.source_clip, base.CONTRACT.source_clip)
    frame["source_coherence"] = np.sign(frame[COMPONENTS]).sum(axis=1).abs().div(count).where(count >= 2)
    frame["source_quality"] = frame["source"] * (0.5 + 0.5 * frame["source_coherence"])
    frame["revision_latency_t"] = np.log1p(frame["revision_latency_days"].clip(lower=0))
    frame["issue_year"] = frame["available_time"].dt.year
    frame["report_quarter"] = frame["REPORTDATE"].dt.quarter
    frame = frame[frame["source"].notna()]
    frame.to_parquet(OUT / "eligible_revision_events.parquet", index=False, compression="zstd")
    return frame


def build_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    source = data["source_quality"].clip(-base.CONTRACT.source_clip, base.CONTRACT.source_clip)
    response = data["response_z"].clip(-base.CONTRACT.response_clip, base.CONTRACT.response_clip)
    signed_absorption = np.sign(source) * np.maximum(np.sign(source) * response, 0.0)
    volume_impedance = np.maximum(data["volume_shock"].fillna(0.0), 0.0)
    range_impedance = np.maximum(data["range_z"].fillna(0.0), 0.0)
    data["financial_revision_absorption_gap"] = (source - response).clip(-base.CONTRACT.candidate_clip, base.CONTRACT.candidate_clip)
    data["financial_revision_reservoir"] = (source / (1.0 + response.abs())).clip(-base.CONTRACT.candidate_clip, base.CONTRACT.candidate_clip)
    data["financial_revision_liquidity_reservoir"] = (
        source / (1.0 + response.abs() + 0.5 * volume_impedance)
    ).clip(-base.CONTRACT.candidate_clip, base.CONTRACT.candidate_clip)
    data["financial_revision_range_reservoir"] = (
        source / (1.0 + response.abs() + 0.5 * range_impedance)
    ).clip(-base.CONTRACT.candidate_clip, base.CONTRACT.candidate_clip)
    data["financial_revision_coherent_underreaction"] = (
        np.sign(source) * np.maximum(source.abs() - signed_absorption.abs(), 0.0)
    ).clip(-base.CONTRACT.candidate_clip, base.CONTRACT.candidate_clip)
    data["financial_revision_overshoot_reversal"] = (
        -np.sign(response) * np.maximum(response.abs() - source.abs(), 0.0)
    ).clip(-base.CONTRACT.candidate_clip, base.CONTRACT.candidate_clip)
    return data


base.MIN_EVENT_COHORT = 5
base.BASE_FEATURES = [
    "source", "source_quality", "source_coherence", *COMPONENTS,
    "revision_latency_t", "revision_number", "response_z", "gap_z", "intraday_z", "range_z",
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
                cross_section = base.daily_ic(data, candidate, target, min_events=5)
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
            print("REVISION_EVALUATED", key, flush=True)
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
    receipt = {"status": "started", "candidates": CANDIDATES}
    try:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(login.error_msg)
        reports = base.fetch_reports()
        membership = base.fetch_membership()
        events = filter_events(reports, membership)
        codes = sorted(set(events["code"]))
        print("REVISION_ELIGIBLE", len(events), "CODES", len(codes), flush=True)
        daily = base.fetch_daily_panel(codes)
        industries = events.sort_values("available_time").groupby("code")["industry"].last().to_dict()
        daily = base.add_daily_features(daily, industries)
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
