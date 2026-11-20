from __future__ import annotations

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


research.stage_mask = stage_mask


if __name__ == "__main__":
    research.main()
