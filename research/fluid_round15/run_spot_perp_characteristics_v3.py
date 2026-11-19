from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

import run_spot_perp_characteristics_v2 as v2

HF_REPO = v2.HF_REPO


def load_hf_external_fixed() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    api = HfApi()
    info = api.dataset_info(HF_REPO)
    revision = str(info.sha)
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset", revision=revision))
    metrics: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.DataFrame] = {}
    resolved: dict[str, dict[str, str]] = {}

    for symbol in v2.base.SYMBOLS:
        metric_name = v2.resolve_hf_file(files, "um_metrics", symbol)
        funding_name = v2.resolve_hf_file(files, "um_funding", symbol)
        resolved[symbol] = {"metrics": metric_name, "funding": funding_name}

        metric_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=metric_name,
            repo_type="dataset",
            revision=revision,
        )
        funding_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=funding_name,
            repo_type="dataset",
            revision=revision,
        )

        m = pd.read_parquet(metric_path)
        m.columns = [str(column).strip().lower() for column in m.columns]
        aliases = {
            "sum_open_interest": "open_interest",
            "sum_open_interest_value": "open_interest_value",
            "top_trader_account_ratio": "top_count_ratio",
            "count_toptrader_long_short_ratio": "top_count_ratio",
            "top_trader_position_ratio": "top_position_ratio",
            "sum_toptrader_long_short_ratio": "top_position_ratio",
            "count_long_short_ratio": "account_ratio",
            "taker_buy_sell_ratio": "taker_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ratio",
        }
        m = m.rename(columns=aliases)
        required = [
            "time",
            "open_interest",
            "open_interest_value",
            "top_count_ratio",
            "top_position_ratio",
            "account_ratio",
            "taker_ratio",
        ]
        for column in required[1:]:
            if column not in m:
                m[column] = np.nan
        m["time"] = pd.to_datetime(m["time"], utc=True, errors="coerce")
        for column in required[1:]:
            m[column] = pd.to_numeric(m[column], errors="coerce")
        m = (
            m.dropna(subset=["time", "open_interest", "open_interest_value"])
            .set_index("time")[required[1:]]
            .sort_index()
            .resample("1h")
            .last()
            .dropna(subset=["open_interest", "open_interest_value"])
            .reset_index()
        )
        # One completed-hour embargo after the observation timestamp.
        m["time"] = m["time"] + pd.Timedelta(hours=1)
        metrics[symbol] = m

        f = pd.read_parquet(funding_path)
        f.columns = [str(column).strip().lower() for column in f.columns]
        time_col = next(
            (column for column in ["time", "calc_time", "funding_time"] if column in f),
            None,
        )
        rate_col = next(
            (
                column
                for column in ["funding_rate", "last_funding_rate", "fundingrate", "rate"]
                if column in f
            ),
            None,
        )
        if time_col is None or rate_col is None:
            raise RuntimeError(f"funding columns missing for {symbol}: {f.columns.tolist()}")
        f["time"] = pd.to_datetime(f[time_col], utc=True, errors="coerce")
        f["funding_rate"] = pd.to_numeric(f[rate_col], errors="coerce")
        f = f[["time", "funding_rate"]].dropna().sort_values("time")
        # Funding is usable only after its publication time and one completed hour.
        f["time"] = f["time"] + pd.Timedelta(hours=1)
        funding[symbol] = f

    return metrics, funding, {
        "repo": HF_REPO,
        "revision": revision,
        "resolved": resolved,
        "funding_rate_alias": "rate",
        "embargo_hours": 1,
    }


v2.load_hf_external = load_hf_external_fixed

if __name__ == "__main__":
    v2.base.main()
