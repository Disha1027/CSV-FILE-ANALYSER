"""Guarded statistical, anomaly, and time-series calculations for EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def benjamini_hochberg(p_values):
    """Return BH adjusted p-values, preserving input order and missing values."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    values = p[valid]
    if not len(values):
        return out
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def pairwise_analysis(frame: pd.DataFrame, first: str, second: str) -> dict:
    """Analyze a selected pair, reporting sample sizes and effect sizes."""
    x, y = frame[first], frame[second]
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    result = {"columns": [first, second], "n": n, "interpretation": "Observed association does not establish causation."}
    if n < 2:
        return {**result, "error": "At least two complete observations are required."}
    x_num, y_num = pd.api.types.is_numeric_dtype(pair.x), pd.api.types.is_numeric_dtype(pair.y)
    if x_num and y_num:
        if pair.x.nunique() < 2 or pair.y.nunique() < 2:
            return {**result, "error": "Correlation needs at least two distinct values in each column."}
        result["type"] = "numeric-numeric"
        correlations = {}
        for name, function in (("pearson", stats.pearsonr), ("spearman", stats.spearmanr), ("kendall", stats.kendalltau)):
            try:
                test = function(pair.x, pair.y)
                correlations[name] = {"coefficient": _finite_number(test.statistic), "p_value": _finite_number(test.pvalue)}
            except (ValueError, FloatingPointError):
                correlations[name] = {"coefficient": None, "p_value": None}
        result["correlations"] = correlations
        result["sample_size"] = n
        return result
    if x_num != y_num:
        result["type"] = "numeric-categorical"
        numeric = pair.x if x_num else pair.y
        groups = [g.to_numpy(dtype=float) for _, g in numeric.groupby(pair.y if x_num else pair.x, observed=True)]
        result["groups"] = [{"category": str(k), "n": int(len(g)), "mean": float(g.mean()), "median": float(g.median()), "std": _finite_number(g.std(ddof=1)) if len(g)>1 else None}
                            for k, g in numeric.groupby(pair.y if x_num else pair.x, observed=True)]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            try:
                f, p = stats.f_oneway(*groups)
                grand = np.concatenate(groups).mean()
                ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
                ss_total = sum(((g - grand) ** 2).sum() for g in groups)
                result["anova"] = {"statistic": _finite_number(f), "p_value": _finite_number(p), "eta_squared": _finite_number(ss_between / ss_total) if ss_total > 0 else None,
                                   "assumptions": "Independent observations; approximately normal residuals; similar within-group variances."}
                h, kp = stats.kruskal(*groups)
                result["kruskal_wallis"] = {"statistic": _finite_number(h), "p_value": _finite_number(kp), "epsilon_squared": _finite_number(max(0, (h-len(groups)+1)/(n-len(groups)))) if n > len(groups) else None,
                                             "assumptions": "Independent observations; ordinal or continuous outcome; group distributions have comparable shapes for a location interpretation."}
            except (ValueError, FloatingPointError) as exc:
                result["test_note"] = str(exc)
        return result
    result["type"] = "categorical-categorical"
    table = pd.crosstab(pair.x, pair.y)
    result["contingency"] = table
    if min(table.shape) >= 2:
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        n_obs = int(table.to_numpy().sum())
        denom = n_obs * (min(table.shape) - 1)
        result["chi_square"] = {"statistic": float(chi2), "p_value": float(p), "degrees_of_freedom": int(dof),
                                "cramers_v": float(np.sqrt(chi2 / denom)) if denom else None,
                                "minimum_expected_count": float(expected.min()),
                                "assumptions": "Independent observations; expected cell counts should generally not be very small."}
    else:
        result["test_note"] = "Both columns need at least two observed categories."
    return result


def time_series_profile(frame: pd.DataFrame, date_column: str, value_column: str, duplicate_rule: str = "mean", frequency: str | None = None) -> dict:
    """Profile timestamp data without mutating the input. Resampling is opt-in."""
    dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed", utc=True)
    values = pd.to_numeric(frame[value_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ts = pd.DataFrame({"date": dates, "value": values, "source_index": frame.index}).dropna(subset=["date", "value"]).sort_values("date")
    if ts.empty:
        return {"error": "No valid date and numeric value pairs were found."}
    duplicate_count = int(ts.duplicated("date", keep=False).sum())
    if duplicate_count and duplicate_rule == "keep first":
        ts = ts.drop_duplicates("date", keep="first")
    elif duplicate_count and duplicate_rule == "keep last":
        ts = ts.drop_duplicates("date", keep="last")
    elif duplicate_count:
        ts = ts.groupby("date", as_index=False)["value"].mean()
    ts = ts.set_index("date").sort_index()
    inferred = None
    if len(ts) >= 3 and not ts.index.has_duplicates:
        try:
            inferred = pd.infer_freq(ts.index)
        except (ValueError, TypeError):
            pass
    if frequency:
        series = ts["value"].resample(frequency).mean()
    else:
        series = ts["value"]
    gaps = series.isna()
    result = {"data": series, "start": str(series.index.min()), "end": str(series.index.max()), "n": int(series.notna().sum()),
              "duplicate_timestamps": duplicate_count, "inferred_frequency": inferred, "regular_frequency_used": frequency,
              "missing_periods_after_optional_resampling": int(gaps.sum()), "irregular_intervals": inferred is None and frequency is None,
              "duplicate_rule": duplicate_rule, "timezone": "Parsed as UTC for comparable timestamps."}
    clean = series.dropna()
    if len(clean) > 1:
        window = max(2, min(7, len(clean)//3))
        result["rolling_mean"] = clean.rolling(window, min_periods=1).mean()
        result["rolling_std"] = clean.rolling(window, min_periods=2).std()
        result["pct_change"] = clean.pct_change().replace([np.inf, -np.inf], np.nan)
        result["absolute_change"] = clean.diff()
    return result
