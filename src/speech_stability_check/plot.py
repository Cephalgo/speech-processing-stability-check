"""Optional heatmap: median absolute change (%) per feature and condition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_summary(summary: pd.DataFrame, path: Path, threshold_pct: float = 10.0,
                 max_features: int = 30) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = summary[summary["condition"] != "ALL"]
    feats = [f for f in summary["feature"].unique() if not f.startswith("egemaps_")]
    feats = feats or list(summary["feature"].unique())
    feats = feats[:max_features]
    tab = per.pivot(index="feature", columns="condition", values="median_abs_change_pct")
    cond_order = [c for c in per["condition"].unique() if c in tab.columns]
    tab = tab.loc[[f for f in feats if f in tab.index], cond_order]
    vals = tab.to_numpy(dtype=float)
    vmax = max(2 * threshold_pct, float(np.nanmax(vals)) if np.isfinite(vals).any() else 1)

    fig, ax = plt.subplots(figsize=(1.3 * tab.shape[1] + 3.2, 0.42 * tab.shape[0] + 1.8))
    im = ax.imshow(np.clip(vals, 0, vmax), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), [c.replace("_", " ") for c in tab.columns],
                  rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(tab.shape[0]), [f.replace("_", " ") for f in tab.index], fontsize=9)
    for (i, j), v in np.ndenumerate(vals):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 * vmax else "#1f2328")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("median absolute change vs original (%)", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Feature shift by processing condition", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
