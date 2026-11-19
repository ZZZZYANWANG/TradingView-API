from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

import run_spot_perp_characteristics_v3 as v3

base = v3.v2.base
_DEV_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def fast_cross_sectional_ic(
    frame: pd.DataFrame,
    factor: str,
    target: str,
) -> tuple[float, pd.Series]:
    work = frame[["time", factor, target]].copy()
    work[factor] = pd.to_numeric(work[factor], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[factor, target])
    if work.empty:
        return float("nan"), pd.Series(dtype=float)
    groups = work.groupby("time", sort=True)
    work["__n"] = groups[factor].transform("count")
    work = work.loc[work["__n"] >= 6].copy()
    if work.empty:
        return float("nan"), pd.Series(dtype=float)
    groups = work.groupby("time", sort=True)
    work["__x"] = groups[factor].rank(method="average")
    work["__y"] = groups[target].rank(method="average")
    work["__xc"] = work["__x"] - groups["__x"].transform("mean")
    work["__yc"] = work["__y"] - groups["__y"].transform("mean")
    work["__xy"] = work["__xc"] * work["__yc"]
    work["__xx"] = work["__xc"] ** 2
    work["__yy"] = work["__yc"] ** 2
    summed = work.groupby("time", sort=True)[["__xy", "__xx", "__yy"]].sum()
    daily = summed["__xy"] / (np.sqrt(summed["__xx"] * summed["__yy"]) + base.EPS)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    return (float(daily.mean()) if len(daily) else float("nan"), daily)


def fast_development_score(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
    sign: int,
) -> dict[str, Any]:
    key = (factor, int(horizon))
    if key not in _DEV_CACHE:
        target = f"y_{horizon}"
        dev = base.issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon)
        cs, daily = fast_cross_sectional_ic(dev, factor, target)
        ts, _, _ = base.time_series_ic(dev, factor, target)
        years = {
            str(int(year)): float(values.mean())
            for year, values in daily.groupby(daily.index.year)
        }
        _DEV_CACHE[key] = {"cs": cs, "ts": ts, "years": years}
    cached = _DEV_CACHE[key]
    signed_cs = sign * float(cached["cs"])
    signed_ts = sign * float(cached["ts"])
    signed_years = {year: sign * float(value) for year, value in cached["years"].items()}
    finite = [value for value in signed_years.values() if np.isfinite(value)]
    minimum = min(finite) if finite else -1.0
    score = minimum + 0.30 * (signed_cs if np.isfinite(signed_cs) else -1.0) + 0.10 * (signed_ts if np.isfinite(signed_ts) else -1.0)
    return {
        "factor": factor,
        "sign": int(sign),
        "horizon": int(horizon),
        "development_cs": signed_cs,
        "development_ts": signed_ts,
        "development_min_year": minimum,
        "development_years": __import__("json").dumps(signed_years, sort_keys=True),
        "development_score": float(score),
    }


def fast_portfolio(
    frame: pd.DataFrame,
    factor: str,
    target: str,
    horizon: int,
    cost_bp: float,
) -> dict[str, Any]:
    work = frame.loc[np.isfinite(frame[factor]) & np.isfinite(frame[target]), ["time", "symbol", factor, target]].copy()
    if work.empty:
        return {"annual_log_return": float("nan"), "sharpe": float("nan"), "avg_turnover": float("nan"), "n": 0, "years": {}}
    work["__rank"] = work.groupby("time")[factor].rank(pct=True, method="average")
    work["weight"] = 0.0
    long = work["__rank"] >= 0.70
    short = work["__rank"] <= 0.30
    long_count = long.groupby(work["time"]).transform("sum")
    short_count = short.groupby(work["time"]).transform("sum")
    work.loc[long, "weight"] = 0.5 / long_count.loc[long].clip(lower=1)
    work.loc[short, "weight"] = -0.5 / short_count.loc[short].clip(lower=1)
    pivot = work.pivot(index="time", columns="symbol", values="weight").fillna(0.0)
    returns = work.pivot(index="time", columns="symbol", values=target).reindex(index=pivot.index, columns=pivot.columns).fillna(0.0)
    turnover = pivot.diff().abs().sum(axis=1).fillna(pivot.abs().sum(axis=1))
    net = (pivot * returns).sum(axis=1) - turnover * cost_bp * 1.0e-4
    scale = 365.25 * 24.0 / horizon
    annual = float(net.mean() * scale)
    sharpe = float(net.mean() / (net.std(ddof=0) + base.EPS) * math.sqrt(scale))
    years = {str(int(year)): float(values.mean() * scale) for year, values in net.groupby(net.index.year)}
    return {"annual_log_return": annual, "sharpe": sharpe, "avg_turnover": float(turnover.mean()), "n": int(len(net)), "years": years}


# Replace only computational implementations; all research identities, clocks,
# labels, gates and candidate specifications remain frozen.
base.cross_sectional_ic = fast_cross_sectional_ic
base.development_score = fast_development_score
base.portfolio = fast_portfolio

if __name__ == "__main__":
    v3.v2.load_hf_external = v3.load_hf_external_fixed
    v3.v2.base.main()
