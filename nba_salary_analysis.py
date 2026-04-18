"""
NBA Player Salary Analysis
==========================
This script loads NBA player performance and salary data from an Excel file,
cleans it, merges it with historical salary cap data, and then:
  - Runs exploratory correlation analysis
  - Builds a linear regression model to predict player salaries
  - Generates three publication-ready charts saved to the output/ directory

Usage:
    python nba_salary_analysis.py

The script expects the data file at:
    data/NBA Talent Analysis Part BandC Data.xlsx
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join("data", "NBA Talent Analysis Part BandC Data.xlsx")
OUTPUT_DIR = "output"

# Performance metrics used in regression and correlation analysis
FEATURE_COLS = ["MP", "WS", "FG%", "USG%", "PTS", "TRB", "AST"]


# ── Main analysis class ───────────────────────────────────────────────────────

class NBAAnalysis:
    """
    End-to-end pipeline for NBA player salary analysis.

    Parameters
    ----------
    filepath : str
        Path to the Excel workbook containing 'NBA Data' and
        'NBA Salary Cap History' sheets.
    """

    def __init__(self, filepath: str = DATA_FILE) -> None:
        self.filepath = filepath
        self.df: pd.DataFrame = pd.DataFrame()          # merged, cleaned dataset
        self._cap_map: dict[int, float] = {}            # Deal_Year → salary cap

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading & cleaning
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Read the two relevant sheets from the Excel workbook."""
        logger.info("Loading data from '%s' …", self.filepath)
        xl = pd.ExcelFile(self.filepath)

        self._raw_nba = xl.parse("NBA Data")
        self._raw_cap = xl.parse("NBA Salary Cap History")

        logger.info(
            "  NBA Data  : %d rows × %d columns", *self._raw_nba.shape
        )
        logger.info(
            "  Salary Cap: %d seasons loaded", len(self._raw_cap) - 1
        )

    def _parse_salary_cap(self) -> None:
        """
        The salary-cap sheet stores headers in the first data row.
        We skip that row, rename columns, and build a mapping from
        Deal_Year (e.g. 2019) to the corresponding salary cap value.
        """
        cap = self._raw_cap.copy()
        # Row 0 contains the real column labels
        cap.columns = ["Season", "Salary_Cap", "Luxury_Tax"]
        cap = cap.iloc[1:].reset_index(drop=True)

        # Extract the first year from "2019–20" → 2019
        cap["Deal_Year"] = (
            cap["Season"]
            .astype(str)
            .str.split("–")
            .str[0]
            .str.strip()
            .astype(int)
        )
        cap["Salary_Cap"] = pd.to_numeric(cap["Salary_Cap"])
        self._cap_map = dict(zip(cap["Deal_Year"], cap["Salary_Cap"]))

    def clean_data(self) -> None:
        """
        Clean the NBA player stats sheet:
          - Standardise the salary column name
          - Drop rows without salary information
          - Convert numeric columns that may have been read as strings
        """
        logger.info("Cleaning player data …")
        df = self._raw_nba.copy()

        # Standardise salary column
        df.rename(columns={"Deal Average Salary": "Salary"}, inplace=True)

        # Drop players with no salary (e.g. forecast-only rows)
        df.dropna(subset=["Salary"], inplace=True)

        # Ensure all feature columns are numeric
        for col in FEATURE_COLS + ["Salary", "Deal_Year"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=FEATURE_COLS + ["Salary"], inplace=True)

        self.df = df.reset_index(drop=True)
        logger.info("  %d players retained after cleaning.", len(self.df))

    def merge_datasets(self) -> None:
        """Attach the salary-cap value for each player's contract year."""
        logger.info("Merging salary cap data …")
        self._parse_salary_cap()

        self.df["Salary_Cap"] = self.df["Deal_Year"].map(self._cap_map)
        missing = self.df["Salary_Cap"].isna().sum()
        if missing:
            logger.warning(
                "  %d rows could not be matched to a salary cap year.", missing
            )
            self.df.dropna(subset=["Salary_Cap"], inplace=True)

    def feature_engineering(self) -> None:
        """
        Derive two extra columns:
          - Salary_Cap_Ratio : salary as a fraction of the league cap,
            enabling apples-to-apples comparison across different seasons
          - Log_Salary       : log-transformed salary, which follows a
            more symmetric distribution and improves regression fit
        """
        logger.info("Engineering new features …")
        self.df["Salary_Cap_Ratio"] = self.df["Salary"] / self.df["Salary_Cap"]
        self.df["Log_Salary"] = np.log1p(self.df["Salary"])

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def correlation_analysis(self) -> pd.Series:
        """
        Compute Pearson correlations between each performance metric and
        the raw salary.

        Returns
        -------
        pd.Series
            Correlations sorted from highest to lowest (absolute value).
        """
        logger.info("Running correlation analysis …")
        correlations = (
            self.df[FEATURE_COLS + ["Salary"]]
            .corr()["Salary"]
            .drop("Salary")
            .sort_values(ascending=False)
        )

        logger.info("  Top correlations with salary:")
        for metric, corr in correlations.items():
            logger.info("    %-8s  r = %+.3f", metric, corr)

        return correlations

    def regression_analysis(self) -> dict:
        """
        Fit a multiple linear regression model to predict Log_Salary from
        the standardised performance metrics.

        Returns
        -------
        dict
            Dictionary containing the fitted model, R² score, and a
            DataFrame mapping each feature to its coefficient.
        """
        logger.info("Building regression model …")

        X = self.df[FEATURE_COLS]
        y = self.df["Log_Salary"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)

        coef_df = (
            pd.DataFrame(
                {"Feature": FEATURE_COLS, "Coefficient": model.coef_}
            )
            .sort_values("Coefficient", ascending=False)
            .reset_index(drop=True)
        )

        logger.info("  R² = %.4f  (explains %.1f%% of salary variance)", r2, r2 * 100)
        logger.info("  Standardised coefficients:")
        for _, row in coef_df.iterrows():
            logger.info("    %-8s  β = %+.4f", row["Feature"], row["Coefficient"])

        return {"model": model, "r2": r2, "coefficients": coef_df, "scaler": scaler}

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def visualize_salary_distribution(self) -> None:
        """
        Save a two-panel chart showing:
          Left  – raw salary distribution (right-skewed histogram)
          Right – log-transformed salary distribution (more bell-shaped)
        """
        logger.info("Plotting salary distribution …")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("NBA Player Salary Distribution", fontsize=14, fontweight="bold")

        # Raw salary (millions for readability)
        sal_m = self.df["Salary"] / 1_000_000
        axes[0].hist(sal_m, bins=30, color="#1D428A", edgecolor="white", alpha=0.85)
        axes[0].set_title("Raw Salary")
        axes[0].set_xlabel("Salary ($ millions)")
        axes[0].set_ylabel("Number of Players")
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))

        # Log-transformed salary
        axes[1].hist(
            self.df["Log_Salary"], bins=30, color="#C8102E", edgecolor="white", alpha=0.85
        )
        axes[1].set_title("Log-Transformed Salary")
        axes[1].set_xlabel("ln(Salary + 1)")
        axes[1].set_ylabel("Number of Players")

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "salary_distribution.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved → %s", out_path)

    def visualize_correlation_heatmap(self) -> None:
        """
        Save a heatmap of pairwise Pearson correlations among all
        performance metrics and the salary.
        """
        logger.info("Plotting correlation heatmap …")
        cols = FEATURE_COLS + ["Salary"]
        corr_matrix = self.df[cols].corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Correlation Matrix — Performance Metrics vs. Salary", fontsize=13)
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved → %s", out_path)

    def visualize_relationships(self) -> None:
        """
        Save a 2 × 3 grid of scatter plots, each showing one performance
        metric against salary with a linear trend line.
        """
        logger.info("Plotting metric–salary scatter plots …")
        metrics = ["WS", "MP", "PTS", "USG%", "TRB", "AST"]
        labels = {
            "WS": "Win Shares",
            "MP": "Minutes Played",
            "PTS": "Points per Season",
            "USG%": "Usage Rate (%)",
            "TRB": "Total Rebounds",
            "AST": "Assists",
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            "NBA Performance Metrics vs. Player Salary", fontsize=14, fontweight="bold"
        )

        sal_m = self.df["Salary"] / 1_000_000

        for ax, metric in zip(axes.flatten(), metrics):
            ax.scatter(
                self.df[metric],
                sal_m,
                alpha=0.4,
                s=20,
                color="#1D428A",
                edgecolors="none",
            )
            # Trend line
            z = np.polyfit(self.df[metric], sal_m, 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.df[metric].min(), self.df[metric].max(), 200)
            ax.plot(x_line, p(x_line), color="#C8102E", linewidth=1.8)

            ax.set_xlabel(labels[metric])
            ax.set_ylabel("Salary ($ millions)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "relationships.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved → %s", out_path)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline from data loading to charts."""
        self.load_data()
        self.clean_data()
        self.merge_datasets()
        self.feature_engineering()

        self.correlation_analysis()
        self.regression_analysis()

        self.visualize_salary_distribution()
        self.visualize_correlation_heatmap()
        self.visualize_relationships()

        logger.info("All done!  Charts saved to the '%s/' directory.", OUTPUT_DIR)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    analysis = NBAAnalysis(DATA_FILE)
    analysis.run_analysis()


if __name__ == "__main__":
    main()
