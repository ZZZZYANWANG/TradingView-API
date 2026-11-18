from pathlib import Path

path = Path(__file__).with_name("run.py")
source = path.read_text(encoding="utf-8")
source = source.replace("to_numpy(dtype=float)", "to_numpy(dtype=float, copy=True)")
source = source.replace("stack(dropna=False)", "stack()")

start = source.index("def daily_ic_series(")
end = source.index("\ndef ic_summary(", start)
daily_replacement = '''def daily_ic_series(factor: pd.DataFrame, target: pd.DataFrame, dates: pd.Index, cols: list[str]) -> pd.Series:
    common_dates = dates.intersection(factor.index).intersection(target.index)
    if len(common_dates) == 0:
        return pd.Series(dtype=float)
    a = factor.loc[common_dates, cols].astype(float)
    b = target.loc[common_dates, cols].astype(float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a.where(valid)
    b = b.where(valid)
    count = valid.sum(axis=1)
    ar = a.rank(axis=1, method="average")
    br = b.rank(axis=1, method="average")
    ac = ar.sub(ar.mean(axis=1), axis=0)
    bc = br.sub(br.mean(axis=1), axis=0)
    numerator = (ac * bc).sum(axis=1, min_count=1)
    denominator = np.sqrt(
        (ac * ac).sum(axis=1, min_count=1)
        * (bc * bc).sum(axis=1, min_count=1)
    )
    result = numerator / denominator.replace(0.0, np.nan)
    result = result.where(count >= 6).dropna()
    return result.astype(float).sort_index()

'''
source = source[:start] + daily_replacement + source[end + 1 :]

start = source.index("def mean_parent_correlation(")
end = source.index("\ndef build_baseline(", start)
parent_replacement = '''def mean_parent_correlation(factor: pd.DataFrame, features: dict[str, pd.DataFrame], bounds, cols: list[str]) -> tuple[float, str]:
    dates = period_index(factor.index, bounds)
    base_factor = factor.loc[dates, cols].astype(float)
    best_value, best_name = -np.inf, ""
    for name, feat in features.items():
        other = feat.loc[dates, cols].astype(float)
        valid = np.isfinite(base_factor) & np.isfinite(other)
        a = base_factor.where(valid)
        b = other.where(valid)
        count = valid.sum(axis=1)
        ar = a.rank(axis=1, method="average")
        br = b.rank(axis=1, method="average")
        ac = ar.sub(ar.mean(axis=1), axis=0)
        bc = br.sub(br.mean(axis=1), axis=0)
        numerator = (ac * bc).sum(axis=1, min_count=1)
        denominator = np.sqrt(
            (ac * ac).sum(axis=1, min_count=1)
            * (bc * bc).sum(axis=1, min_count=1)
        )
        corr = (numerator / denominator.replace(0.0, np.nan)).where(count >= 6).dropna()
        value = float(corr.abs().mean()) if len(corr) else np.nan
        if np.isfinite(value) and value > best_value:
            best_value, best_name = value, name
    return best_value, best_name

'''
source = source[:start] + parent_replacement + source[end + 1 :]
path.write_text(source, encoding="utf-8")
