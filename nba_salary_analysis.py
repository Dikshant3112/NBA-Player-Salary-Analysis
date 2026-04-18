"""NBA Player salary analysis and forecasting pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class NBAAnalysis:
    """End-to-end NBA salary analysis using historical deal and performance data."""

    FEATURE_COLUMNS = [
        "Age",
        "G",
        "GS",
        "MP",
        "PTS",
        "AST",
        "TRB",
        "STL",
        "BLK",
        "TOV",
        "FG%",
        "3P%",
        "FT%",
        "USG%",
        "OWS",
        "DWS",
        "WS",
        "RFA",
    ]

    TARGET_COLUMN = "Deal Average Salary"

    def __init__(self, workbook_path: str | Path, output_dir: str | Path = "output"):
        self.workbook_path = Path(workbook_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.salary_df: pd.DataFrame | None = None
        self.cap_df: pd.DataFrame | None = None
        self.model: Pipeline | None = None
        self.feature_importance: pd.Series | None = None

    @staticmethod
    def _season_to_start_year(season: str) -> int | None:
        if not isinstance(season, str):
            return None
        digits = "".join(ch for ch in season if ch.isdigit() or ch == "-")
        if len(digits) < 4:
            return None
        try:
            return int(digits[:4])
        except ValueError:
            return None

    def load_data(self) -> None:
        logging.info("Loading workbook: %s", self.workbook_path)
        self.salary_df = pd.read_excel(self.workbook_path, sheet_name="NBA Data")
        cap_raw = pd.read_excel(self.workbook_path, sheet_name="NBA Salary Cap History")

        cap_df = cap_raw.copy()
        cap_df.columns = ["Season", "Salary Cap", "Luxury Tax Line"]
        if str(cap_df.iloc[0]["Season"]).strip().lower() == "season":
            cap_df = cap_df.iloc[1:].copy()

        cap_df["Deal_Year"] = cap_df["Season"].apply(self._season_to_start_year)
        cap_df["Salary Cap"] = pd.to_numeric(cap_df["Salary Cap"], errors="coerce")
        cap_df = cap_df.dropna(subset=["Deal_Year", "Salary Cap"])
        self.cap_df = cap_df[["Deal_Year", "Salary Cap"]]

    def clean_and_engineer_features(self) -> None:
        if self.salary_df is None or self.cap_df is None:
            raise ValueError("Data must be loaded before preprocessing.")

        df = self.salary_df.copy()
        df = df.merge(self.cap_df, how="left", on="Deal_Year")

        numeric_cols = self.FEATURE_COLUMNS + [self.TARGET_COLUMN, "Salary Cap"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)
        df["Salary_to_Cap"] = df[self.TARGET_COLUMN] / df["Salary Cap"]
        df["Log_Salary"] = np.log1p(df[self.TARGET_COLUMN])

        self.salary_df = df
        logging.info("Prepared %s player records for modeling.", len(df))

    def train_regression_model(self) -> dict[str, float]:
        if self.salary_df is None:
            raise ValueError("Data must be prepared before modeling.")

        X = self.salary_df[self.FEATURE_COLUMNS]
        y = self.salary_df[self.TARGET_COLUMN]

        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regression", LinearRegression()),
            ]
        )
        self.model.fit(x_train, y_train)

        predictions = self.model.predict(x_test)
        metrics = {
            "r2": r2_score(y_test, predictions),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        }

        coeffs = self.model.named_steps["regression"].coef_
        self.feature_importance = pd.Series(coeffs, index=self.FEATURE_COLUMNS).sort_values(
            key=np.abs,
            ascending=False,
        )

        self._save_prediction_plot(y_test, predictions)
        return metrics

    def forecast_sheet_players(self) -> pd.DataFrame:
        if self.model is None or self.cap_df is None:
            raise ValueError("Model must be trained before forecasting.")

        forecast_df = pd.read_excel(self.workbook_path, sheet_name="Forecast Data")
        merged = forecast_df.merge(self.cap_df, how="left", on="Deal_Year")
        latest_cap = float(self.cap_df.sort_values("Deal_Year")["Salary Cap"].iloc[-1])
        merged["Salary Cap"] = merged["Salary Cap"].fillna(latest_cap)

        for col in self.FEATURE_COLUMNS:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

        valid = merged.dropna(subset=self.FEATURE_COLUMNS).copy()
        valid["Predicted Salary"] = self.model.predict(valid[self.FEATURE_COLUMNS])
        valid["Predicted Salary"] = valid["Predicted Salary"].clip(lower=0)
        valid["Predicted Salary to Cap"] = valid["Predicted Salary"] / valid["Salary Cap"]

        result = valid[
            ["Player", "Deal_Year", "Predicted Salary", "Predicted Salary to Cap"]
        ].sort_values("Predicted Salary", ascending=False)
        result.to_csv(self.output_dir / "forecast_estimates.csv", index=False)
        return result

    def create_visualizations(self) -> None:
        if self.salary_df is None or self.feature_importance is None:
            raise ValueError("Data and model outputs must exist before visualization.")

        sns.set_theme(style="whitegrid")

        # Salary distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(self.salary_df[self.TARGET_COLUMN], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Salary Distribution")
        axes[0].set_xlabel("Average Annual Salary")

        sns.histplot(self.salary_df["Log_Salary"], bins=30, kde=True, ax=axes[1], color="#7A3E9D")
        axes[1].set_title("Log Salary Distribution")
        axes[1].set_xlabel("log(1 + salary)")

        fig.tight_layout()
        fig.savefig(self.output_dir / "salary_distribution.png", dpi=200)
        plt.close(fig)

        # Correlation heatmap
        heatmap_cols = self.FEATURE_COLUMNS + [self.TARGET_COLUMN, "Salary_to_Cap"]
        corr = self.salary_df[heatmap_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Performance Metrics vs Salary Correlation")
        fig.tight_layout()
        fig.savefig(self.output_dir / "correlation_heatmap.png", dpi=200)
        plt.close(fig)

        # Top feature impacts
        top_coeffs = self.feature_importance.head(10).sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        top_coeffs.plot(kind="barh", ax=ax, color="#2E86AB")
        ax.set_title("Top Regression Coefficients (Standardized)")
        ax.set_xlabel("Coefficient")
        fig.tight_layout()
        fig.savefig(self.output_dir / "feature_importance.png", dpi=200)
        plt.close(fig)

    def _save_prediction_plot(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_true, y=y_pred, ax=ax)
        min_val = float(min(y_true.min(), y_pred.min()))
        max_val = float(max(y_true.max(), y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        ax.set_title("Actual vs Predicted Salary")
        ax.set_xlabel("Actual Salary")
        ax.set_ylabel("Predicted Salary")
        fig.tight_layout()
        fig.savefig(self.output_dir / "actual_vs_predicted.png", dpi=200)
        plt.close(fig)

    def write_report(self, metrics: dict[str, float], forecasts: pd.DataFrame) -> None:
        if self.feature_importance is None:
            raise ValueError("Feature importance not available.")

        top_5 = self.feature_importance.head(5)
        lines = [
            "NBA Salary Analysis Report",
            "=" * 30,
            f"Records modeled: {len(self.salary_df) if self.salary_df is not None else 0}",
            f"Test R^2: {metrics['r2']:.3f}",
            f"Test RMSE: ${metrics['rmse']:,.0f}",
            "",
            "Top 5 standardized feature coefficients:",
        ]
        lines.extend([f"- {name}: {value:.4f}" for name, value in top_5.items()])
        lines.append("")
        lines.append("Forecast estimates:")
        for _, row in forecasts.iterrows():
            lines.append(
                f"- {row['Player']} ({int(row['Deal_Year'])}): "
                f"${row['Predicted Salary']:,.0f} "
                f"({row['Predicted Salary to Cap']:.2%} of cap)"
            )

        report_path = self.output_dir / "analysis_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logging.info("Saved report: %s", report_path)

    def run_analysis(self) -> None:
        self.load_data()
        self.clean_and_engineer_features()
        metrics = self.train_regression_model()
        forecasts = self.forecast_sheet_players()
        self.create_visualizations()
        self.write_report(metrics, forecasts)

        logging.info("Analysis completed.")
        logging.info("Model R^2: %.3f", metrics["r2"])
        logging.info("Model RMSE: $%0.0f", metrics["rmse"])
        logging.info("Forecast summary:\n%s", forecasts.to_string(index=False))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA player salary analysis")
    parser.add_argument(
        "--data",
        default="data/NBA Talent Analysis Part BandC Data.xlsx",
        help="Path to workbook with NBA data sheets.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for analysis artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analysis = NBAAnalysis(args.data, args.output)
    analysis.run_analysis()
