from __future__ import annotations

import importlib.util
import io
import json
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CORE_PATH = Path("research/fluid_round9/run.py")
spec = importlib.util.spec_from_file_location("fluid_round9_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load round9 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

OUT = Path("research_outputs/fluid_round9_stock_refinement")
OUT.mkdir(parents=True, exist_ok=True)
START = int(pd.Timestamp("2012-01-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-11-01", tz="UTC").timestamp())
HORIZON = 5
BASE_FACTOR = "scale_flux_persistence"
SOURCE_FACTOR = "source_sink_release"


def download_yahoo(ticker: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": START,
        "period2": END,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0 Trade-fluid-round9-refinement"}
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=35)
            response.raise_for_status()
            result = response.json()["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
            frame = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
                "open_raw": quote["open"],
                "high_raw": quote["high"],
                "low_raw": quote["low"],
                "close_raw": quote["close"],
                "close": adjusted,
                "volume": quote["volume"],
            })
            for column in ["open_raw", "high_raw", "low_raw", "close_raw", "close", "volume"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            ratio = frame["close"] / frame["close_raw"]
            frame["open"] = frame["open_raw"] * ratio
            frame["high"] = frame["high_raw"] * ratio
            frame["low"] = frame["low_raw"] * ratio
            frame = frame[["date", "open", "high", "low", "close", "volume"]]
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
            frame = frame.loc[(frame[["open", "high", "low", "close"]] > 0).all(axis=1) & (frame["volume"] > 0)]
            if len(frame) < 2000:
                raise ValueError(f"short_history={len(frame)}")
            frame["symbol"] = ticker
            frame["sector"] = core.STOCK_SECTORS[ticker]
            return ticker, frame, None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return ticker, None, error


def load_panel():
    frames = []
    errors = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = [pool.submit(download_yahoo, ticker) for ticker in core.STOCK_SECTORS]
        for job in as_completed(jobs):
            ticker, frame, error = job.result()
            if frame is not None:
                frames.append(frame)
            else:
                errors.append({"ticker": ticker, "error": error})
    if len(frames) < 35:
        raise RuntimeError(f"only {len(frames)} Yahoo histories")
    raw = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    parts = []
    for symbol, frame in raw.groupby("symbol", sort=False):
        computed = core.compute_stock_symbol(frame)
        computed["symbol"] = symbol
        computed["sector"] = core.STOCK_SECTORS[symbol]
        parts.append(computed)
    panel = core.add_stock_cross_section(pd.concat(parts, ignore_index=True))
    return panel, {
        "provider": "Yahoo chart API",
        "rows": int(len(panel)),
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "min_date": str(panel["date"].min()),
        "max_date": str(panel["date"].max()),
        "errors": errors,
    }


def prior_single_residual(y: pd.Series, x: pd.Series, window: int = 252, minimum: int = 100) -> pd.Series:
    yp = y.shift(1)
    xp = x.shift(1)
    covariance = yp.rolling(window, min_periods=minimum).cov(xp)
    variance = xp.rolling(window, min_periods=minimum).var(ddof=0)
    beta = covariance / (variance + core.EPS)
    return y - beta * x


def add_refinements(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["symbol", "date"]).copy()
    records = []
    for _, group in p.groupby("symbol", sort=False):
        g = group.copy()
        flux = g[BASE_FACTOR]
        source = g[SOURCE_FACTOR]
        mom5 = g["mom5_gap"]
        mom20 = g["mom20_gap"]
        flux_mom5_residual = prior_single_residual(flux, mom5)
        flux_dual_residual = prior_single_residual(flux_mom5_residual, mom20)
        same_sign = np.sign(flux) == np.sign(source)
        agreement = pd.Series(
            np.where(same_sign, np.sign(flux) * np.sqrt(np.minimum(flux.abs(), source.abs())), 0.0),
            index=g.index,
            dtype=float,
        )
        candidates = {
            "flux_slow5": flux.rolling(5, min_periods=4).mean(),
            "flux_slow10": flux.rolling(10, min_periods=7).mean(),
            "flux_slow20": flux.rolling(20, min_periods=14).mean(),
            "flux_mom5_orth_slow5": flux_mom5_residual.rolling(5, min_periods=4).mean(),
            "flux_dual_orth_slow5": flux_dual_residual.rolling(5, min_periods=4).mean(),
            "flux_source_linear_slow5": (0.70 * flux_dual_residual + 0.30 * source).rolling(5, min_periods=4).mean(),
            "flux_source_agreement_slow5": agreement.rolling(5, min_periods=4).mean(),
            "flux_source_absorption_slow5": (flux_dual_residual * np.exp(-source.abs())).rolling(5, min_periods=4).mean(),
        }
        for name, values in candidates.items():
            g[name] = core.causal_z(values, 252, 100)
        records.append(g)
    return pd.concat(records, ignore_index=True)


CANDIDATES = [
    "flux_slow5",
    "flux_slow10",
    "flux_slow20",
    "flux_mom5_orth_slow5",
    "flux_dual_orth_slow5",
    "flux_source_linear_slow5",
    "flux_source_agreement_slow5",
    "flux_source_absorption_slow5",
]


def evaluate(panel: pd.DataFrame, factor: str, train_end: int, test_start: int, test_end: int, cost_bp: float = 10.0):
    target = f"y_{HORIZON}"
    train = panel.loc[panel["date"].dt.year <= train_end].copy()
    test = panel.loc[(panel["date"].dt.year >= test_start) & (panel["date"].dt.year <= test_end)].copy()
    train = core.issue_subset(train, HORIZON, "stocks")
    test = core.issue_subset(test, HORIZON, "stocks")
    cs, series = core.cross_sectional_ic(test, factor, target, "date", 20)
    bootstrap = core.block_bootstrap(series, 5)
    ridge = core.ridge_increment(train, test, core.STOCK_BASE, factor, target, 180000)
    test["residual"] = core.residual_values(train, test, core.STOCK_BASE, factor)
    residual, _ = core.cross_sectional_ic(test, "residual", target, "date", 20)
    correlation = core.maximum_baseline_correlation(test, factor, core.STOCK_BASE)
    portfolio = core.rank_portfolio(test, factor, target, "date", cost_bp, 252.0 / HORIZON, 20)
    return {
        "factor": factor,
        "cs_ic": cs["mean"],
        "positive_share": cs["positive_share"],
        "bootstrap_low": bootstrap["low"],
        "bootstrap_high": bootstrap["high"],
        "ridge_gain": ridge["gain"],
        "residual_cs_ic": residual["mean"],
        "max_baseline_corr": correlation["max_abs"],
        "max_corr_feature": correlation["feature"],
        "net_sharpe": portfolio["sharpe"],
        "annual_log_return": portfolio["annual_log_return"],
        "avg_turnover": portfolio["avg_turnover"],
        "years": json.dumps(portfolio["years"], sort_keys=True),
    }


def selection_score(row: dict) -> float:
    values = [row["cs_ic"], row["residual_cs_ic"], row["ridge_gain"], row["net_sharpe"]]
    if not all(np.isfinite(values)):
        return -np.inf
    return (
        1.0 * row["cs_ic"]
        + 0.75 * row["residual_cs_ic"]
        + 8.0 * row["ridge_gain"]
        + 0.004 * np.clip(row["net_sharpe"], -2.0, 2.0)
        - 0.005 * max(row["max_baseline_corr"] - 0.75, 0.0)
    )


try:
    panel, metadata = load_panel()
    panel = add_refinements(panel)
    validation_rows = []
    for factor in CANDIDATES:
        row = evaluate(panel, factor, train_end=2020, test_start=2021, test_end=2023)
        row["selection_score"] = selection_score(row)
        row["validation_gate"] = bool(
            row["cs_ic"] > 0.006
            and row["residual_cs_ic"] > 0.004
            and row["ridge_gain"] > 0.0
            and row["net_sharpe"] > 0.0
            and row["max_baseline_corr"] < 0.85
        )
        validation_rows.append(row)
    validation = pd.DataFrame(validation_rows).sort_values("selection_score", ascending=False)
    passers = validation.loc[validation["validation_gate"]]
    if len(passers):
        selected = str(passers.iloc[0]["factor"])
        confirmation = evaluate(panel, selected, train_end=2023, test_start=2024, test_end=2025)
        shadow = evaluate(panel, selected, train_end=2025, test_start=2026, test_end=2026)
        final_pass = bool(
            confirmation["cs_ic"] > 0.006
            and confirmation["residual_cs_ic"] > 0.003
            and confirmation["ridge_gain"] > 0.0
            and confirmation["net_sharpe"] > 0.10
            and shadow["cs_ic"] > 0.0
            and shadow["residual_cs_ic"] > 0.0
            and shadow["net_sharpe"] > 0.0
        )
    else:
        selected = None
        confirmation = None
        shadow = None
        final_pass = False

    validation.to_csv(OUT / "VALIDATION_CANDIDATES.csv", index=False)
    if confirmation is not None:
        pd.DataFrame([confirmation]).to_csv(OUT / "CONFIRMATION_2024_2025.csv", index=False)
        pd.DataFrame([shadow]).to_csv(OUT / "SHADOW_2026.csv", index=False)
    summary = {
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "data": metadata,
        "horizon_sessions": HORIZON,
        "base_factor": BASE_FACTOR,
        "source_factor": SOURCE_FACTOR,
        "validation_period": "2021-2023",
        "confirmation_period": "2024-2025",
        "shadow_period": "2026-01 through 2026-10",
        "selected": selected,
        "confirmation": confirmation,
        "shadow": shadow,
        "final_pass": final_pass,
        "governance": {
            "candidate_count": len(CANDIDATES),
            "all_candidates_mechanistically_frozen_before_confirmation": True,
            "confirmation_read_only_if_validation_gate_passes": True,
            "fixed_current_universe_survivorship_warning": True,
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    promoted = [summary] if final_pass else []
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable(promoted), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Fluid Stock Refinement: Slow Flux and Source–Sink Coupling",
        "",
        validation.to_markdown(index=False),
        "",
        f"Selected after validation gate: `{selected}`" if selected else "No candidate passed validation; confirmation remained sealed.",
        "",
        f"Final pass: **{final_pass}**",
        "",
        "The refinement targets the two near-pass mechanisms from Round 9: multiscale flux persistence and source–sink release.",
        "It reduces turnover with causal smoothing and removes momentum overlap with prior rolling orthogonalization.",
        "The current-universe stock sample retains survivorship-selection risk; even a final pass remains historical/default-off pending point-in-time A-share replication.",
    ]
    if confirmation is not None:
        lines.extend(["", "## 2024–2025 confirmation", "", pd.DataFrame([confirmation]).to_markdown(index=False), "", "## 2026 shadow", "", pd.DataFrame([shadow]).to_markdown(index=False)])
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
