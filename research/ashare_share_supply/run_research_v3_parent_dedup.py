from __future__ import annotations

import numpy as np
import pandas as pd

import run_research_v2_blocks as blocks

study = blocks.study

_original_prepare_events = study.prepare_events


def prepare_events(raw: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    frame = _original_prepare_events(raw, membership)
    frame = frame.sort_values(["code", "END_DATE"], kind="mergesort").copy()
    frame["circ_ratio"] = frame["LISTED_A_SHARES"] / frame["TOTAL_SHARES"].replace(0.0, np.nan)
    frame["free_ratio"] = frame["FREE_SHARES"] / frame["TOTAL_SHARES"].replace(0.0, np.nan)
    frame["dlog_circ_ratio"] = frame.groupby("code", sort=False)["circ_ratio"].transform(
        lambda values: np.log(values / values.shift(1))
    )
    frame["dlog_free_ratio"] = frame.groupby("code", sort=False)["free_ratio"].transform(
        lambda values: np.log(values / values.shift(1))
    )
    pieces = []
    for _, group in frame.groupby("code", sort=False):
        group = group.sort_values("END_DATE").copy()
        indexed = group.set_index("END_DATE")
        group["parent_circulation_contraction20_proxy"] = (
            (-indexed["dlog_circ_ratio"]).clip(lower=0).rolling("30D", min_periods=1).sum().to_numpy()
        )
        group["parent_freefloat_contraction20_proxy"] = (
            (-indexed["dlog_free_ratio"]).clip(lower=0).rolling("30D", min_periods=1).sum().to_numpy()
        )
        group["parent_circulation_expansion60_proxy"] = (
            indexed["dlog_circ_ratio"].clip(lower=0).rolling("90D", min_periods=1).sum().to_numpy()
        )
        pieces.append(group)
    frame = pd.concat(pieces, ignore_index=True)
    frame.to_parquet(study.OUT / "eligible_share_events_with_parent_proxies.parquet", index=False, compression="zstd")
    return frame


for column in [
    "circ_ratio", "free_ratio", "dlog_circ_ratio", "dlog_free_ratio",
    "parent_circulation_contraction20_proxy", "parent_freefloat_contraction20_proxy",
    "parent_circulation_expansion60_proxy",
]:
    if column not in study.base.BASE_FEATURES:
        study.base.BASE_FEATURES.append(column)
    if column not in blocks.SUPPLY_BLOCK:
        blocks.SUPPLY_BLOCK.append(column)

study.prepare_events = prepare_events


if __name__ == "__main__":
    study.main()
