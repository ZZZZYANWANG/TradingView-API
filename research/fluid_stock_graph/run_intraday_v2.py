from __future__ import annotations

import importlib.util
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT = Path("research/fluid_stock_graph/run_research.py")
OUT = Path("research_outputs/fluid_stock_graph_v2")
CACHE = OUT / "cache"
MAX_CODES = 100
WORKERS = 8
DEV_START = np.datetime64("2020-01-01")
DEV_END = np.datetime64("2021-12-31")
VAL_START = np.datetime64("2022-01-01")
VAL_END = np.datetime64("2023-12-31")
LATER_START = np.datetime64("2024-01-01")
EPS = 1.0e-12
SEED = 20261118
RNG = np.random.default_rng(SEED)


def patched_period_mask(index: pd.DatetimeIndex, name: str) -> np.ndarray:
    if name == "development":
        return (index >= pd.Timestamp("2020-01-01")) & (index <= pd.Timestamp("2021-12-31"))
    if name == "validation":
        return (index >= pd.Timestamp("2022-01-01")) & (index <= pd.Timestamp("2023-12-31"))
    if name == "later":
        return index >= pd.Timestamp("2024-01-01")
    raise ValueError(name)


def patch_module(module):
    module.period_mask = patched_period_mask

    def model_diagnostics(base_features, signal, label, eligible):
        columns = list(signal.columns)
        dates = module.date_vector(signal.index, columns)
        elig = eligible.to_numpy(bool).reshape(-1)
        x = np.column_stack([module.flatten_matrix(feature) for feature in base_features])
        f = module.flatten_matrix(signal)
        y = module.flatten_matrix(label)
        dev = elig & (dates >= DEV_START) & (dates <= DEV_END)
        val = elig & (dates >= VAL_START) & (dates <= VAL_END)
        later = elig & (dates >= LATER_START)
        row_ok = np.isfinite(y) & np.isfinite(f) & (np.isfinite(x).mean(axis=1) >= 0.75)
        dev &= row_ok
        val &= row_ok
        later &= row_ok
        if dev.sum() < 5000 or val.sum() < 3000:
            return {"available": False, "counts": {"dev": int(dev.sum()), "val": int(val.sum()), "later": int(later.sum())}}
        train_idx = np.flatnonzero(dev)
        if len(train_idx) > 160000:
            train_idx = RNG.choice(train_idx, size=160000, replace=False)
        medians = np.nanmedian(x[train_idx], axis=0)
        medians[~np.isfinite(medians)] = 0.0
        x_filled = np.where(np.isfinite(x), x, medians)
        base = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        aug = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        base.fit(x_filled[train_idx], y[train_idx])
        aug.fit(np.column_stack([x_filled[train_idx], f[train_idx]]), y[train_idx])
        recon = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        recon.fit(x_filled[train_idx], f[train_idx])
        out = {"available": True, "counts": {"dev": int(dev.sum()), "val": int(val.sum()), "later": int(later.sum())}}
        for period_name, mask in [("validation", val), ("later", later)]:
            if mask.sum() < 1000:
                continue
            p0 = base.predict(x_filled[mask])
            p1 = aug.predict(np.column_stack([x_filled[mask], f[mask]]))
            mse0 = mean_squared_error(y[mask], p0)
            mse1 = mean_squared_error(y[mask], p1)
            fhat = recon.predict(x_filled[mask])
            residual = f[mask] - fhat
            local_dates = dates[mask]
            use = np.ones(mask.sum(), dtype=bool)
            out[period_name] = {
                "mse_base": float(mse0),
                "mse_augmented": float(mse1),
                "mse_gain_pct": float((mse0 - mse1) / (mse0 + EPS) * 100.0),
                "prediction_rankic_base": module.long_daily_ic(local_dates, p0, y[mask], use),
                "prediction_rankic_augmented": module.long_daily_ic(local_dates, p1, y[mask], use),
                "factor_reconstruction_r2": float(r2_score(f[mask], fhat)),
                "residual_rankic": module.long_daily_ic(local_dates, residual, y[mask], use),
            }
        return out

    def nonlinear_increment(base_features, signal, label, eligible):
        columns = list(signal.columns)
        dates = module.date_vector(signal.index, columns)
        elig = eligible.to_numpy(bool).reshape(-1)
        x = np.column_stack([module.flatten_matrix(feature) for feature in base_features])
        f = module.flatten_matrix(signal)
        y = module.flatten_matrix(label)
        dev = elig & (dates >= DEV_START) & (dates <= DEV_END)
        val = elig & (dates >= VAL_START) & (dates <= VAL_END)
        later = elig & (dates >= LATER_START)
        ok = np.isfinite(y) & np.isfinite(f) & (np.isfinite(x).mean(axis=1) >= 0.75)
        dev &= ok
        val &= ok
        later &= ok
        train = np.flatnonzero(dev)
        if len(train) > 90000:
            train = RNG.choice(train, size=90000, replace=False)
        if len(train) < 5000:
            return {"available": False, "counts": {"dev": int(dev.sum()), "val": int(val.sum()), "later": int(later.sum())}}
        med = np.nanmedian(x[train], axis=0)
        med[~np.isfinite(med)] = 0.0
        x = np.where(np.isfinite(x), x, med)
        params = dict(max_iter=120, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=2.0, random_state=SEED)
        base = HistGradientBoostingRegressor(**params).fit(x[train], y[train])
        aug = HistGradientBoostingRegressor(**params).fit(np.column_stack([x[train], f[train]]), y[train])
        out = {"available": True}
        for period_name, mask in [("validation", val), ("later", later)]:
            if mask.sum() < 1000:
                continue
            p0 = base.predict(x[mask])
            p1 = aug.predict(np.column_stack([x[mask], f[mask]]))
            mse0 = mean_squared_error(y[mask], p0)
            mse1 = mean_squared_error(y[mask], p1)
            out[period_name] = {"mse_gain_pct": float((mse0 - mse1) / (mse0 + EPS) * 100.0)}
        return out

    module.model_diagnostics = model_diagnostics
    module.nonlinear_increment = nonlinear_increment
    module.MAX_CODES = MAX_CODES
    module.OUT = OUT
    return module


def load_module():
    spec = importlib.util.spec_from_file_location("fluid_graph_v2", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load research module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return patch_module(module)


def download_one(code: str) -> dict[str, object]:
    module = load_module()
    login = bs.login()
    if login.error_code != "0":
        return {"code": code, "ok": False, "error": login.error_msg}
    try:
        daily = module.download_daily(code)
        hourly = module.download_hourly(code)
    finally:
        bs.logout()
    if daily.empty or hourly.empty:
        return {"code": code, "ok": False, "error": "empty_daily_or_hourly"}
    CACHE.mkdir(parents=True, exist_ok=True)
    stem = code.replace(".", "_")
    daily.to_pickle(CACHE / f"{stem}.daily.pkl")
    hourly.to_pickle(CACHE / f"{stem}.hourly.pkl")
    return {
        "code": code,
        "ok": True,
        "daily_rows": int(len(daily)),
        "hourly_rows": int(len(hourly)),
        "hourly_min": str(hourly["date"].min().date()),
        "hourly_max": str(hourly["date"].max().date()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    module = load_module()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        snapshots, membership_meta = module.load_membership_snapshots()
    finally:
        bs.logout()
    counts = Counter(code for members in snapshots.values() for code in members)
    codes = [code for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_CODES]]

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"code": code, "ok": False, "error": f"{type(exc).__name__}:{exc}"}
            results.append(result)
            if index % 10 == 0:
                print(f"v2 downloads={index}/{len(codes)} usable={sum(bool(row.get('ok')) for row in results)}", flush=True)
    with (OUT / "parallel_download_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    usable = [str(row["code"]) for row in results if row.get("ok")]
    if len(usable) < 60:
        raise RuntimeError(f"only {len(usable)} codes downloaded successfully")

    def cached_daily(code: str) -> pd.DataFrame:
        path = CACHE / f"{code.replace('.', '_')}.daily.pkl"
        return pd.read_pickle(path) if path.exists() else pd.DataFrame()

    def cached_hourly(code: str) -> pd.DataFrame:
        path = CACHE / f"{code.replace('.', '_')}.hourly.pkl"
        return pd.read_pickle(path) if path.exists() else pd.DataFrame()

    module.load_membership_snapshots = lambda: (snapshots, membership_meta)
    module.download_daily = cached_daily
    module.download_hourly = cached_hourly
    module.main()

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["splits"] = {
            "development": "2020-01-02 to 2021-12-31",
            "validation": "2022-01-01 to 2023-12-31",
            "later_history": "2024-01-01 to 2026-11-17",
        }
        summary["data_availability_correction"] = {
            "reason": "BaoStock 60-minute data begins 2020-01-02",
            "probe_commit": "90371649e10a519d32bb0508939b93ee4f08d820",
            "old_2017_2019_development_run_invalidated": True,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
