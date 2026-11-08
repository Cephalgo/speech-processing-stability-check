"""Agreement statistics."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def icc_2_1(data: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measure
    (Shrout & Fleiss 1979; McGraw & Wong 1996).

    ``data`` is an (n subjects x k raters) array; here subjects are recordings
    and raters are processing conditions. Rows with any NaN are dropped.
    """
    m = np.asarray(data, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape if m.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return float("nan")
    grand = m.mean()
    ss_rows = k * np.sum((m.mean(axis=1) - grand) ** 2)
    ss_cols = n * np.sum((m.mean(axis=0) - grand) ** 2)
    ss_total = np.sum((m - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom <= 0:
        return float("nan")
    return float((msr - mse) / denom)


def summarise(features: pd.DataFrame, threshold_pct: float = 10.0,
              reference: str = "original") -> pd.DataFrame:
    """Per feature and condition: shift versus the reference condition.

    ``features`` is long-format with columns ``file``, ``condition`` and one
    column per feature. Returns one row per (feature, condition), plus one row
    per feature with condition ``ALL`` holding the ICC(2,1) across every
    condition including the reference.
    """
    feat_cols = [c for c in features.columns if c not in ("file", "condition")]
    conditions: List[str] = [c for c in features["condition"].unique() if c != reference]
    rows = []
    for feat in feat_cols:
        wide = features.pivot(index="file", columns="condition", values=feat)
        if reference not in wide:
            raise ValueError(f"reference condition '{reference}' missing")
        ref = wide[reference]
        for cond in conditions:
            if cond not in wide:
                continue
            pair = pd.concat([ref, wide[cond]], axis=1).dropna()
            diff = pair.iloc[:, 1] - pair.iloc[:, 0]
            denom = pair.iloc[:, 0].abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = 100 * diff.abs() / denom
            pct = pct.replace([np.inf, -np.inf], np.nan)
            rows.append({
                "feature": feat,
                "condition": cond,
                "n_files": int(len(pair)),
                "mean_original": float(pair.iloc[:, 0].mean()) if len(pair) else np.nan,
                "mean_shift": float(diff.mean()) if len(pair) else np.nan,
                "mean_shift_pct": float(100 * diff.mean() / pair.iloc[:, 0].mean())
                if len(pair) and pair.iloc[:, 0].mean() != 0 else np.nan,
                "median_abs_change_pct": float(pct.median()) if pct.notna().any() else np.nan,
                "icc21_vs_original": icc_2_1(pair.to_numpy()),
                "pct_files_over_threshold": float(100 * (pct > threshold_pct).sum() / len(pair))
                if len(pair) else np.nan,
            })
        cols = [reference] + [c for c in conditions if c in wide]
        rows.append({
            "feature": feat,
            "condition": "ALL",
            "n_files": int(wide[cols].dropna().shape[0]),
            "icc21_vs_original": np.nan,
            "icc21_all_conditions": icc_2_1(wide[cols].to_numpy()),
        })
    out = pd.DataFrame(rows)
    order = ["feature", "condition", "n_files", "mean_original", "mean_shift",
             "mean_shift_pct", "median_abs_change_pct", "icc21_vs_original",
             "pct_files_over_threshold", "icc21_all_conditions"]
    return out.reindex(columns=order)
