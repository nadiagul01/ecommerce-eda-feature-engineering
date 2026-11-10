"""
main.py - Command-line entry point. Runs the whole project in one go:

    raw Excel  ->  validate  ->  clean  ->  engineer features  ->  save outputs + report

Usage:
    python main.py                 # full run (data + figures + summary)
    python main.py --no-report     # data only (used by GitHub Actions)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files; no window needed (works on Windows/Linux/CI)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from feature_pipeline import FeaturePipeline, config  # noqa: E402
from feature_pipeline import report  # noqa: E402
from feature_pipeline.missing import run_imputation_experiment  # noqa: E402
from feature_pipeline.outliers import outlier_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DecodeLabs DS Project 1 - EDA & Feature Engineering")
    parser.add_argument("--input", type=Path, default=config.RAW_DATA_PATH, help="Raw Excel file")
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED_DIR, help="Folder for outputs")
    parser.add_argument("--no-report", action="store_true", help="Skip figures and summary.md")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("main")
    args = parse_args()

    # ---------------- 1. Load
    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        return 1
    raw = pd.read_excel(args.input)
    log.info("Loaded %s: %d rows x %d columns", args.input.name, *raw.shape)

    # ---------------- 2. Run the pipeline
    pipe = FeaturePipeline()
    analysis, model = pipe.fit_transform(raw)

    # ---------------- 3. Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(args.output_dir / "orders_clean.csv", index=False)
    model.to_csv(args.output_dir / "orders_model_ready.csv", index=False)
    model.to_parquet(args.output_dir / "orders_model_ready.parquet", index=False)
    pipe.save_params(config.ARTIFACTS_DIR / "pipeline_params.json")
    log.info("Saved orders_clean.csv, orders_model_ready.csv/.parquet and pipeline_params.json")

    # ---------------- 4. Report (figures + summary.md)
    if not args.no_report:
        fig_dir = config.FIGURES_DIR
        experiment, example, truth = run_imputation_experiment(raw)
        outliers = outlier_summary(raw, config.NUMERIC_COLUMNS)
        experiment.to_csv(config.REPORTS_DIR / "imputation_experiment.csv")
        outliers.to_csv(config.REPORTS_DIR / "outlier_summary.csv")

        for fig in (
            report.plot_numeric_distributions(raw, fig_dir / "01_numeric_distributions.png"),
            report.plot_categorical_distributions(raw, fig_dir / "02_categorical_distributions.png"),
            report.plot_missing_values(pipe.missing_report_, fig_dir / "03_missing_values.png"),
            report.plot_imputation_experiment(example, truth, fig_dir / "04_imputation_experiment.png"),
            report.plot_outliers(raw, model, pipe.outlier_bounds_, fig_dir / "05_outliers.png"),
            report.plot_correlation(model, fig_dir / "06_feature_correlation.png"),
        ):
            plt.close(fig)

        report.write_summary(raw, analysis, model, pipe, experiment, outliers,
                             config.REPORTS_DIR / "summary.md")
        log.info("Report written to reports/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
