"""
report.py - EDA figures and the auto-generated Markdown summary.

Each plot_* function returns a matplotlib Figure and, when `path` is given,
also saves it as PNG. The notebook displays the figures; main.py saves them.
Every number in reports/summary.md is computed from the data - nothing is
typed in by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config

# -----------------------------------------------------------------------------
# Visual style: neutral ink, one accent blue, blue <-> red diverging scale
# -----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
RED = "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#f0efec", RED])

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
}
plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, path: Path | None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


# -----------------------------------------------------------------------------
# 1. Numeric distributions
# -----------------------------------------------------------------------------
def plot_numeric_distributions(raw: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, len(config.NUMERIC_COLUMNS), figsize=(13, 3.2))
    for ax, col in zip(axes, config.NUMERIC_COLUMNS):
        s = raw[col]
        if s.nunique() <= 12:  # small whole numbers -> one bar per value
            counts = s.value_counts().sort_index()
            ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
            ax.set_xticks(counts.index)
        else:
            ax.hist(s, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"{col}  (skew {s.skew():.2f})", loc="left")
        ax.set_ylabel("Orders")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Numeric columns: distributions", x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


# -----------------------------------------------------------------------------
# 2. Categorical distributions
# -----------------------------------------------------------------------------
def plot_categorical_distributions(raw: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, len(config.CATEGORICAL_COLUMNS), figsize=(15, 3.4))
    for ax, col in zip(axes, config.CATEGORICAL_COLUMNS):
        counts = raw[col].fillna("(blank)").value_counts().sort_values()
        colors = [RED if label == "(blank)" else BLUE for label in counts.index]
        ax.barh(counts.index, counts.values, color=colors, height=0.6)
        ax.set_title(col, loc="left")
        ax.set_xlabel("Orders")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Categorical columns: order counts (red = blank values)", x=0.01, ha="left",
                 fontweight="bold")
    return _save(fig, path)


# -----------------------------------------------------------------------------
# 3. Missing values
# -----------------------------------------------------------------------------
def plot_missing_values(report: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    data = report.sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [RED if p > 0 else GRID for p in data["missing_pct"]]
    ax.barh(data.index, data["missing_pct"], color=colors, height=0.6)
    for y, (pct, strategy) in enumerate(data[["missing_pct", "strategy"]].itertuples(index=False)):
        if pct > 0:
            ax.text(pct + 0.6, y, f"{pct:.2f}%  ->  {strategy.replace('_', ' ')}", va="center", fontsize=9)
    for x in (config.DROP_ROWS_MAX_PCT, config.STAT_IMPUTE_MAX_PCT):
        ax.axvline(x, color=INK_2, linestyle="--", linewidth=1)
        ax.text(x + 0.4, -0.9, f"{x:g}%", color=INK_2, fontsize=8)
    ax.set_xlim(0, 50)
    ax.set_ylim(-1.2, len(data) - 0.4)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing values per column and the decision-matrix strategy", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# -----------------------------------------------------------------------------
# 4. Imputation experiment
# -----------------------------------------------------------------------------
def plot_imputation_experiment(example: dict[str, pd.Series], truth: pd.Series,
                               path: Path | None = None) -> plt.Figure:
    bins = np.arange(truth.min() - 0.5, truth.max() + 1.5, 1)
    fig, axes = plt.subplots(1, len(example), figsize=(12, 3.4), sharey=True)
    for ax, (method, series) in zip(axes, example.items()):
        ax.hist(series, bins=bins, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=1,
                label="After imputation")
        ax.hist(truth, bins=bins, histtype="step", color=INK, linewidth=1.6, label="True values")
        ax.set_title(method, loc="left")
        ax.set_xlabel(truth.name)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Orders")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    share = int(config.EXPERIMENT_MISSING_SHARE * 100)
    fig.suptitle(f"{truth.name}: {share}% of values hidden on purpose, then imputed",
                 x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


# -----------------------------------------------------------------------------
# 5. Outliers
# -----------------------------------------------------------------------------
def plot_outliers(raw: pd.DataFrame, model: pd.DataFrame, bounds: dict[str, tuple[float, float]],
                  path: Path | None = None) -> plt.Figure:
    cols = config.NUMERIC_COLUMNS
    fig, axes = plt.subplots(1, len(cols) + 1, figsize=(15, 3.4),
                             gridspec_kw={"width_ratios": [1] * len(cols) + [1.6]})
    box_style = dict(patch_artist=True, widths=0.5,
                     boxprops={"facecolor": BLUE_LIGHT, "edgecolor": BLUE},
                     medianprops={"color": INK, "linewidth": 1.5},
                     whiskerprops={"color": INK_2}, capprops={"color": INK_2},
                     flierprops={"marker": "o", "markersize": 4, "markerfacecolor": RED,
                                 "markeredgecolor": SURFACE})
    for ax, col in zip(axes, cols):
        ax.boxplot(raw[col], **box_style)
        ax.set_xticks([])
        ax.set_title(col, loc="left")
    ax = axes[-1]
    col = config.OUTLIER_CAP_COLUMNS[0]
    ax.boxplot([raw[col], model[col]], vert=False, **box_style)
    ax.set_yticks([1, 2], ["Before", "After capping"])
    upper = bounds[col][1]
    ax.axvline(upper, color=INK_2, linestyle="--", linewidth=1)
    ax.text(upper, 1.5, f" cap {upper:,.0f}", color=INK_2, fontsize=8, va="center")
    ax.set_title(f"{col}: IQR capping", loc="left")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Outliers (red dots) by the IQR rule", x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


# -----------------------------------------------------------------------------
# 6. Correlation of final features
# -----------------------------------------------------------------------------
def plot_correlation(model: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    cols = [c for c in model.columns if c != config.ID_COLUMN]
    corr = model[cols].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):          # annotate only the meaningful cells
        for j in range(len(cols)):
            v = corr.iat[i, j]
            if i != j and abs(v) >= 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.035, label="Pearson r")
    ax.set_title(f"Correlation of the final model features (all pairs <= {config.COLLINEARITY_THRESHOLD})",
                 loc="left")
    return _save(fig, path)


# -----------------------------------------------------------------------------
# Markdown summary
# -----------------------------------------------------------------------------
def _md_table(df: pd.DataFrame) -> str:
    """Tiny Markdown table writer (avoids an extra 'tabulate' dependency)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_summary(raw: pd.DataFrame, analysis: pd.DataFrame, model: pd.DataFrame, pipe,
                  experiment: pd.DataFrame, outliers: pd.DataFrame, path: Path) -> None:
    miss = pipe.missing_report_.reset_index()
    collinear = (pipe.collinearity_table_ if not pipe.collinearity_table_.empty
                 else pd.DataFrame([{"result": "no pair above threshold"}]))
    lines = [
        "# EDA & Feature Engineering - Run Summary",
        "",
        "_Generated automatically by `main.py`. Re-run the pipeline instead of editing this file._",
        "",
        f"- Raw data: **{raw.shape[0]} rows x {raw.shape[1]} columns**",
        f"- Analysis table: **{analysis.shape[0]} rows x {analysis.shape[1]} columns**",
        f"- Model-ready table: **{model.shape[0]} rows x {model.shape[1]} columns** "
        f"(OrderID + {len(pipe.feature_columns_)} features), rows dropped: {pipe.rows_dropped_}",
        f"- Missing values left: analysis **{int(analysis.isna().sum().sum())}**, "
        f"model **{int(model.isna().sum().sum())}**",
        "",
        "## 1. Missing values",
        "",
        _md_table(miss[miss["missing_pct"] > 0]),
        "",
        "## 2. Imputation experiment (Mean vs Median vs KNN)",
        "",
        f"{int(config.EXPERIMENT_MISSING_SHARE * 100)}% of `{config.EXPERIMENT_COLUMN}` hidden at random, "
        f"averaged over {config.EXPERIMENT_REPEATS} repeats. Lower MAE/RMSE/KS is better; "
        "std closer to the true value means the spread is preserved.",
        "",
        _md_table(experiment.reset_index()),
        "",
        "## 3. Outliers",
        "",
        _md_table(outliers.reset_index()),
        "",
        "## 4. Collinearity (|r| > 0.80)",
        "",
        _md_table(collinear),
        "",
        "## 5. Final model features",
        "",
        ", ".join(f"`{c}`" for c in pipe.feature_columns_),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
