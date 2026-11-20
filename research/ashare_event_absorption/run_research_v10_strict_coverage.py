from __future__ import annotations

import json

import pandas as pd

import run_research_v9_strict as strict

research = strict.research

_original_fetch_daily_panel = research.fetch_daily_panel
_original_map_events = research.map_events


def fetch_daily_panel(codes):
    requested = sorted(set(codes))
    panel = _original_fetch_daily_panel(requested)
    observed = sorted(set(panel["code"])) if len(panel) else []
    missing = sorted(set(requested) - set(observed))
    coverage = len(observed) / len(requested) if requested else 0.0
    audit = {
        "requested_codes": len(requested), "observed_codes": len(observed),
        "coverage": coverage, "missing_codes": missing,
        "minimum_required_coverage": 0.95,
    }
    (research.OUT / "daily_download_coverage.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if coverage < 0.95:
        raise RuntimeError(f"daily download coverage {coverage:.3%} is below 95%")
    return panel


def map_events(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    panel = _original_map_events(events, daily)
    before = len(panel)
    if "REPORTDATE" in panel:
        panel = panel.sort_values(
            ["code", "issue_date", "REPORTDATE", "available_time"], kind="mergesort"
        ).drop_duplicates(["code", "issue_date"], keep="last")
    else:
        panel = panel.sort_values(["code", "issue_date"], kind="mergesort").drop_duplicates(
            ["code", "issue_date"], keep="last"
        )
    audit = {
        "rows_before_unique_stock_day": before,
        "rows_after_unique_stock_day": len(panel),
        "removed_duplicate_weight_rows": before - len(panel),
        "duplicate_stock_day_after": int(panel.duplicated(["code", "issue_date"]).sum()),
        "selection_rule": "latest fiscal report date on a shared tradable event day",
    }
    (research.OUT / "event_weight_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return panel


research.fetch_daily_panel = fetch_daily_panel
research.map_events = map_events


if __name__ == "__main__":
    research.main()
