from __future__ import annotations

import numpy as np
import pandas as pd

import run_research_v6_resilient as resilient

research = resilient.research

_original_prepare_sources = research.prepare_sources
_original_map_events = research.map_events
_original_build_candidates = research.build_candidates


def prepare_sources(events: pd.DataFrame) -> pd.DataFrame:
    frame = _original_prepare_sources(events)
    notice = pd.to_datetime(frame.get("NOTICE_DATE"), errors="coerce")
    available = pd.to_datetime(frame["available_time"], errors="raise")
    report = pd.to_datetime(frame["REPORTDATE"], errors="raise")
    frame["report_latency_days"] = (available.dt.normalize() - report.dt.normalize()).dt.days.clip(lower=0)
    frame["revision_latency_days"] = (available.dt.normalize() - notice.dt.normalize()).dt.days.clip(lower=0)
    frame["report_latency_t"] = np.log1p(frame["report_latency_days"])
    frame["revision_latency_t"] = np.log1p(frame["revision_latency_days"])
    frame = frame.sort_values(["code", "available_time", "REPORTDATE"], kind="mergesort")
    frame["report_latency_z"] = frame.groupby("code", sort=False)["report_latency_t"].transform(
        lambda series: research.prior_robust_z(
            series,
            research.CONTRACT.source_history_window,
            research.CONTRACT.source_history_min,
            research.CONTRACT.source_clip,
        )
    )
    frame["revision_latency_z"] = frame.groupby("code", sort=False)["revision_latency_t"].transform(
        lambda series: research.prior_robust_z(
            series,
            research.CONTRACT.source_history_window,
            research.CONTRACT.source_history_min,
            research.CONTRACT.source_clip,
        )
    )
    return frame


def map_events(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    mapped = _original_map_events(events, daily)
    market_count = mapped.groupby("issue_date", sort=False)["code"].transform("nunique")
    industry_count = mapped.groupby(["issue_date", "industry"], sort=False)["code"].transform("nunique")
    mapped["market_event_count"] = market_count.astype(float)
    mapped["industry_event_count"] = industry_count.astype(float)
    mapped["industry_event_share"] = (
        mapped["industry_event_count"] / mapped["market_event_count"].replace(0.0, np.nan)
    )
    mapped["industry_congestion"] = np.log1p(mapped["industry_event_count"])
    mapped["market_congestion"] = np.log1p(mapped["market_event_count"])
    return mapped


def build_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    p = _original_build_candidates(panel)
    source = p["source_quality"].clip(-research.CONTRACT.source_clip, research.CONTRACT.source_clip)
    response = p["response_z"].clip(-research.CONTRACT.response_clip, research.CONTRACT.response_clip)
    absorption = 1.0 + response.abs()
    early = 0.5 + 0.5 * np.tanh(-p["report_latency_z"].fillna(0.0))
    revision_quality = 0.5 + 0.5 * np.tanh(-p["revision_latency_z"].fillna(0.0))
    congestion = np.sqrt(1.0 + p["industry_event_count"].fillna(0.0))
    congestion_share = np.sqrt(1.0 + 10.0 * p["industry_event_share"].fillna(0.0))
    p["earnings_timing_quality_reservoir"] = (
        source * early / absorption
    ).clip(-research.CONTRACT.candidate_clip, research.CONTRACT.candidate_clip)
    p["earnings_revision_quality_reservoir"] = (
        source * revision_quality / absorption
    ).clip(-research.CONTRACT.candidate_clip, research.CONTRACT.candidate_clip)
    p["earnings_industry_congestion_reservoir"] = (
        source * congestion / absorption
    ).clip(-research.CONTRACT.candidate_clip, research.CONTRACT.candidate_clip)
    p["earnings_congestion_share_reservoir"] = (
        source * congestion_share / absorption
    ).clip(-research.CONTRACT.candidate_clip, research.CONTRACT.candidate_clip)
    p["earnings_timing_congestion_reservoir"] = (
        source * early * congestion_share / absorption
    ).clip(-research.CONTRACT.candidate_clip, research.CONTRACT.candidate_clip)
    return p


TIMING_CANDIDATES = [
    "earnings_timing_quality_reservoir",
    "earnings_revision_quality_reservoir",
    "earnings_industry_congestion_reservoir",
    "earnings_congestion_share_reservoir",
    "earnings_timing_congestion_reservoir",
]

# Raw timing and congestion variables are explicit strong-parent controls.  The
# candidate can only pass if the source-by-capacity relation adds information.
for column in [
    "report_latency_z", "revision_latency_z", "industry_event_count",
    "market_event_count", "industry_event_share", "industry_congestion",
    "market_congestion",
]:
    if column not in research.BASE_FEATURES:
        research.BASE_FEATURES.append(column)


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
        for candidate in TIMING_CANDIDATES:
            key = f"{candidate}|{horizon}"
            for stage in research.STAGES:
                data = panel[research.stage_mask(panel, stage, horizon)]
                cross_section = research.daily_ic(data, candidate, target)
                tsic, positive_share, products = research.time_series_ic(data, candidate, target)
                direct_rows.append({
                    "hypothesis": key, "candidate": candidate, "horizon": horizon,
                    "stage": stage, "rankic": cross_section.mean(),
                    "rankic_dates": len(cross_section), "tsic": tsic,
                    "positive_ts_share": positive_share, "ts_products": products,
                    "rows": int(data[[candidate, target]].dropna().shape[0]),
                })
                if stage in ("validation", "historical_holdout", "recent_extension"):
                    daily_maps[f"{stage}|{key}"] = cross_section
            exit_column = "exit_date_1d" if horizon == "1d" else f"exit_date_{horizon}"
            residual_frames.append(research.residual_ic(panel, candidate, target, exit_column))
            model_frames.append(research.model_increment(panel, candidate, target, exit_column))
            accounts.append(research.account_diagnostic(panel, candidate, target))
            print("TIMING_EVALUATED", key, flush=True)
    direct = pd.DataFrame(direct_rows)
    inference = research.family_inference(daily_maps)
    residuals = pd.concat(residual_frames, ignore_index=True)
    models = pd.concat([frame for frame in model_frames if len(frame)], ignore_index=True)
    account = pd.concat([frame for frame in accounts if len(frame)], ignore_index=True)
    direct.to_csv(research.OUT / "timing_direct_metrics.csv", index=False)
    inference.to_csv(research.OUT / "timing_family_inference.csv", index=False)
    residuals.to_csv(research.OUT / "timing_residual_metrics.csv", index=False)
    models.to_csv(research.OUT / "timing_model_increment.csv", index=False)
    account.to_csv(research.OUT / "timing_account_diagnostics.csv", index=False)
    return {
        "direct": direct, "inference": inference, "residuals": residuals,
        "models": models, "accounts": account,
        "candidates": TIMING_CANDIDATES, "targets": targets,
    }


research.prepare_sources = prepare_sources
research.map_events = map_events
research.build_candidates = build_candidates
research.evaluate = evaluate


if __name__ == "__main__":
    research.main()
