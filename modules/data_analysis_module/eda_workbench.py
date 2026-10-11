"""Small, dependency-light EDA calculations used by the Streamlit workspace."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    if len(non_null) and (re.search(r"date|time|timestamp", str(series.name), re.I)):
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.8:
            return "datetime (inferred)"
    if pd.api.types.is_string_dtype(series) or series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
        unique_ratio = non_null.nunique() / max(len(non_null), 1)
        if unique_ratio >= 0.8 and non_null.nunique() > 20:
            return "identifier-like" if re.search(r"(^id$|_id$|^id_|uuid|code)", str(series.name), re.I) else "text"
        return "categorical"
    return "other"


def _format_bytes(size: int) -> str:
    amount = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024


def build_dataset_profile(df: pd.DataFrame) -> dict:
    total = int(df.size)
    kinds = [_kind(df.iloc[:, i]) for i in range(df.shape[1])]
    counts = pd.Series(kinds).value_counts().to_dict()
    profiles = []
    for position, name in enumerate(df.columns):
        series = df.iloc[:, position]
        non_null = series.dropna()
        kind = kinds[position]
        distinct = int(series.nunique(dropna=True))
        row = {
            "Column": str(name), "Type": kind, "Valid": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Missing %": round(float(series.isna().mean() * 100), 2) if len(df) else 0,
            "Distinct": distinct,
            "Distinct %": round(distinct / max(len(df), 1) * 100, 2),
            "Constant": distinct == 1,
        }
        if kind == "numeric":
            values = pd.to_numeric(series, errors="coerce" ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                q = values.quantile([.01, .05, .25, .5, .75, .95, .99])
                mode = values.mode()
                row.update({"Mean": float(values.mean()), "Median": float(q.loc[.5]),
                    "Mode": float(mode.iloc[0]) if len(mode) else None,
                    "Std": float(values.std()) if len(values) > 1 else None,
                    "Variance": float(values.var()) if len(values) > 1 else None,
                    "Min": float(values.min()), "Max": float(values.max()),
                    "Q1": float(q.loc[.25]), "Q3": float(q.loc[.75]),
                    "IQR": float(q.loc[.75] - q.loc[.25]),
                    "Skew": float(values.skew()) if len(values) > 2 else None,
                    "Excess kurtosis": float(values.kurt()) if len(values) > 3 else None,
                    "Zero": int((values == 0).sum()), "Negative": int((values < 0).sum()),
                    "Percentiles (1, 5, 95, 99)": [float(q.loc[p]) for p in (.01, .05, .95, .99)]})
        elif kind.startswith("datetime"):
            values = pd.to_datetime(series, errors="coerce", format="mixed")
            row.update({"Min": str(values.min()) if values.notna().any() else None,
                        "Max": str(values.max()) if values.notna().any() else None,
                        "Invalid dates": int((series.notna() & values.isna()).sum())})
        elif kind in ("categorical", "text", "identifier-like", "boolean"):
            strings = series.dropna().astype(str)
            counts_for_values = strings.value_counts()
            row.update({"Top values": counts_for_values.head(5).to_dict(),
                "Rare categories (<1%)": int((counts_for_values < max(len(strings) * .01, 1)).sum()),
                "Whitespace-only": int(strings.str.strip().eq("").sum()),
                "Trim/case variants": int(strings.str.strip().str.casefold().nunique() < strings.nunique()),
                "Mean text length": float(strings.str.len().mean()) if len(strings) else None,
                "Min text length": int(strings.str.len().min()) if len(strings) else None,
                "Max text length": int(strings.str.len().max()) if len(strings) else None})
        profiles.append(row)
    unique_cols = [str(df.columns[i]) for i in range(df.shape[1]) if df.iloc[:, i].nunique(dropna=True) == len(df) and len(df) > 0]
    constants = [str(df.columns[i]) for i in range(df.shape[1]) if df.iloc[:, i].nunique(dropna=True) == 1]
    return {"rows": len(df), "columns": df.shape[1], "memory_usage": _format_bytes(int(df.memory_usage(deep=True).sum())),
        "memory_bytes": int(df.memory_usage(deep=True).sum()), "missing_cells": int(df.isna().sum().sum()),
        "missing_percentage": float(df.isna().sum().sum() / max(total, 1) * 100),
        "duplicate_rows": int(df.duplicated().sum()), "duplicate_percentage": float(df.duplicated().mean() * 100) if len(df) else 0,
        "rows_with_missing": int(df.isna().any(axis=1).sum()), "complete_rows": int(df.notna().all(axis=1).sum()),
        "unique_columns": unique_cols, "constant_columns": constants,
        "duplicate_column_names": sorted({str(c) for c in df.columns if list(df.columns).count(c) > 1}),
        "type_counts": counts, "columns_profile": profiles}


def _outlier_indices(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    if len(valid) < 2:
        return {"IQR": [], "Z-score": [], "Modified z-score": []}
    q1, q3 = valid.quantile([.25, .75])
    iqr = q3 - q1
    iqr_mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    std = valid.std(ddof=0)
    z_mask = ((values - valid.mean()).abs() / std > 3) if std > 0 else pd.Series(False, index=values.index)
    mad = (valid - valid.median()).abs().median()
    mz_mask = (0.6745 * (values - valid.median()).abs() / mad > 3.5) if mad > 0 else pd.Series(False, index=values.index)
    return {"IQR": values.index[iqr_mask.fillna(False)].tolist(),
            "Z-score": values.index[z_mask.fillna(False)].tolist(),
            "Modified z-score": values.index[mz_mask.fillna(False)].tolist()}


def build_column_profile(df: pd.DataFrame, column: str) -> dict:
    series = df[column]
    kind = _kind(series)
    stats = {"column": str(column), "type": kind, "valid_count": int(series.notna().sum()),
             "missing_count": int(series.isna().sum()),
             "missing_percentage": float(series.isna().mean() * 100) if len(series) else 0,
             "distinct_count": int(series.nunique(dropna=True))}
    if kind == "numeric":
        v = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            stats.update({"mean": float(v.mean()), "median": float(v.median()), "std": float(v.std()) if len(v)>1 else None,
                "variance": float(v.var()) if len(v)>1 else None, "min": float(v.min()), "max": float(v.max()),
                "range": float(v.max()-v.min()), "q1": float(v.quantile(.25)), "q3": float(v.quantile(.75)),
                "skewness": float(v.skew()) if len(v)>2 else None, "excess_kurtosis": float(v.kurt()) if len(v)>3 else None,
                "zero_count": int((v == 0).sum()), "negative_count": int((v < 0).sum())})
    else:
        counts = series.astype("string").fillna("(missing)").value_counts().head(20)
        stats["top_values"] = counts.to_dict()
    return {"kind": kind, "statistics": stats, "outliers": _outlier_indices(series) if kind == "numeric" else {}}


def build_quality_report(df: pd.DataFrame) -> list[dict]:
    issues = []
    for position, name in enumerate(df.columns):
        series = df.iloc[:, position]
        missing_pct = float(series.isna().mean() * 100) if len(df) else 0
        if series.isna().any():
            issues.append({"Severity": "High" if missing_pct >= 50 else "Medium", "Issue": "Missing values",
                "Column": str(name), "Affected rows": int(series.isna().sum()), "Explanation": f"{missing_pct:.1f}% of values are missing.",
                "Suggested action": "Inspect missingness and domain context before imputing or removing rows."})
        if series.nunique(dropna=True) == 1:
            issues.append({"Severity": "Low", "Issue": "Constant column", "Column": str(name),
                "Affected rows": len(df), "Explanation": "Only one non-missing value occurs.", "Suggested action": "Confirm this feature is useful for analysis."})
        if series.dtype == object:
            objects = series.dropna().map(type).nunique()
            if objects > 1:
                issues.append({"Severity": "Medium", "Issue": "Mixed Python types", "Column": str(name),
                    "Affected rows": int(series.notna().sum()), "Explanation": "Non-missing cells contain more than one underlying Python type.",
                    "Suggested action": "Review values and choose an explicit, consistent representation."})
            strings = series.dropna().astype(str)
            trim = strings.ne(strings.str.strip())
            blanks = strings.str.strip().eq("")
            if trim.any() or blanks.any():
                issues.append({"Severity": "Low", "Issue": "Whitespace or blank text", "Column": str(name),
                    "Affected rows": int((trim | blanks).sum()), "Explanation": "Some strings contain outer whitespace or are blank after trimming.",
                    "Suggested action": "Review a normalized preview before applying trimming."})
        if len(df) > 0 and series.nunique(dropna=True) / len(df) >= .9 and series.nunique(dropna=True) > 20:
            issues.append({"Severity": "Info", "Issue": "High cardinality / possible identifier", "Column": str(name),
                "Affected rows": len(df), "Explanation": "Most rows have distinct values.", "Suggested action": "Check whether this is an identifier before using it in correlations or models."})
    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append({"Severity": "Medium", "Issue": "Exact duplicate rows", "Column": "All columns", "Affected rows": duplicates,
            "Explanation": "Rows repeat across every column.", "Suggested action": "Inspect the repeated rows; remove only if duplicates are not meaningful."})
    repeated_names = sorted({str(c) for c in df.columns if list(df.columns).count(c) > 1})
    if repeated_names:
        issues.append({"Severity": "Medium", "Issue": "Duplicate column names", "Column": ", ".join(repeated_names),
            "Affected rows": 0, "Explanation": "The dataframe contains repeated headers.", "Suggested action": "Rename columns explicitly to avoid ambiguous analysis."})
    return issues


def generate_insights(df: pd.DataFrame) -> list[str]:
    findings = []
    if df.size and df.isna().any().any():
        col = df.isna().mean().idxmax()
        findings.append(f"{col!s} has the highest missingness ({df[col].isna().mean()*100:.1f}%; n={len(df)} rows).")
    for col in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) > 2:
            skew = float(values.skew())
            if np.isfinite(skew) and abs(skew) >= 1:
                direction = "right" if skew > 0 else "left"
                findings.append(f"{col!s} is {direction}-skewed (skew={skew:.2f}; n={len(values)} valid values).")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        corr = numeric.corr(min_periods=3)
        pairs = [(abs(corr.iloc[i, j]), i, j, corr.iloc[i, j]) for i in range(len(corr)) for j in range(i+1, len(corr)) if pd.notna(corr.iloc[i,j])]
        if pairs:
            strength, i, j, value = max(pairs)
            if strength >= .7:
                a, b = corr.columns[i], corr.columns[j]
                n = int(numeric[[a,b]].dropna().shape[0])
                findings.append(f"{a!s} and {b!s} have a strong observed Pearson correlation (r={value:.2f}; n={n}); this is an association, not evidence of causation.")
    for col in df.columns:
        if df[col].nunique(dropna=True) == 1 and len(df):
            findings.append(f"{col!s} is constant across all non-missing observations and may not distinguish records.")
    if not findings:
        findings.append("No findings met the current screening rules; this does not imply that the dataset has no meaningful patterns.")
    return findings[:12]
