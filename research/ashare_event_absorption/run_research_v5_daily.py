from __future__ import annotations

import pandas as pd

import run_research_v4 as corrected

research = corrected.research


def fetch_hourly_targets(codes, needed):
    # Financial-report publication is only date-certified in the public source.
    # The daily-only fallback therefore makes no one-hour claim and avoids an
    # unnecessary second market-data download.
    return pd.DataFrame(columns=["code", "date", "return_1h"])


research.fetch_hourly_targets = fetch_hourly_targets


if __name__ == "__main__":
    research.main()
