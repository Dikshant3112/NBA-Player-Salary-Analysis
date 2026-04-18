# NBA Player Salary Analysis

A production-style data analysis project that explains **what drives NBA player contract value** and generates salary forecasts for upcoming free agents.

## Why this repo is interesting

This project goes beyond static charts by combining:
- Historical player performance and contract data
- Salary-cap normalization across different eras
- A reproducible machine learning pipeline
- Forecasting for players in the workbook's `Forecast Data` sheet

## What it does

The script reads `data/NBA Talent Analysis Part BandC Data.xlsx` and performs:

1. Data loading from `NBA Data` and `NBA Salary Cap History`
2. Data cleaning and numeric feature preparation
3. Feature engineering (`Salary_to_Cap`, `Log_Salary`)
4. Linear regression model training with standardized features
5. Evaluation with test-set **R²** and **RMSE**
6. Forecasting salaries for players in `Forecast Data`
7. Artifact generation in `output/`

## Repository structure

- `nba_salary_analysis.py` – end-to-end analysis + forecasting pipeline
- `data/` – source Excel workbook
- `output/` – generated charts, report, and forecast CSV
- `requirements.txt` – Python dependencies

## Setup

```bash
git clone https://github.com/Dikshant3112/NBA-Player-Salary-Analysis.git
cd NBA-Player-Salary-Analysis
pip install -r requirements.txt
```

## Run

```bash
python nba_salary_analysis.py
```

Optional arguments:

```bash
python nba_salary_analysis.py --data "data/NBA Talent Analysis Part BandC Data.xlsx" --output output
```

## Outputs

After running, you will get:

- `output/salary_distribution.png`
- `output/correlation_heatmap.png`
- `output/feature_importance.png`
- `output/actual_vs_predicted.png`
- `output/forecast_estimates.csv`
- `output/analysis_report.txt`

## Example insights you can extract

- Which stats most strongly increase expected salary
- How much of the salary cap a player is likely to consume
- Whether model predictions track real salary outcomes

## Tech stack

- Python 3.10+
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- openpyxl

## License

MIT License (see `LICENSE`).
