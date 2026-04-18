# NBA Player Salary Analysis 📊

Analyze NBA player performance data to identify key factors influencing player salaries using Python 3, machine learning, and statistical analysis.

## 🎯 Objective

Determine which performance metrics most strongly influence NBA player salaries through comprehensive data analysis, regression modeling, and visualization.

## 📋 Features

### Data Processing
- ✅ Excel file loading with multiple sheets
- ✅ Automatic data cleaning and preprocessing
- ✅ Salary cap normalization for fair year-to-year comparison
- ✅ Missing value handling and data validation

### Analysis
- ✅ Correlation analysis between salary and performance metrics
- ✅ Linear regression modeling with standardized features
- ✅ R-squared model performance metrics
- ✅ Statistical significance testing

### Visualizations
- ✅ Salary distribution histograms (log and normalized)
- ✅ Correlation heatmap
- ✅ Performance metric scatter plots with trend lines
- ✅ Professional-grade charts saved to `output/` directory

## 🛠️ Tech Stack

**Language:** Python 3.8+

**Libraries:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning and preprocessing
- `scipy` - Statistical analysis
- `openpyxl` - Excel file handling

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Dikshant3112/NBA-Player-Salary-Analysis.git
cd NBA-Player-Salary-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your Excel file in the `data/` directory
   - Update the file path in `nba_salary_analysis.py` if needed
   - File should have sheets named "NBA Data" and "NBA Salary Cap History"

4. Run the analysis:
```bash
python nba_salary_analysis.py
```

## 📊 Usage

### Basic Usage
```python
from nba_salary_analysis import NBAAnalysis

# Initialize analysis
analysis = NBAAnalysis("data/NBA_data.xlsx")

# Run complete pipeline
analysis.run_analysis()
```

### Individual Components
```python
# Load and clean data
analysis.load_data()
analysis.clean_data()
analysis.merge_datasets()
analysis.feature_engineering()

# Perform specific analyses
correlations = analysis.correlation_analysis()
regression_results = analysis.regression_analysis()

# Generate visualizations
analysis.visualize_salary_distribution()
analysis.visualize_correlation_heatmap()
analysis.visualize_relationships()
```

## 📈 Output

The analysis generates the following outputs in the `output/` directory:

1. **salary_distribution.png** - Histograms showing salary distribution
2. **correlation_heatmap.png** - Heatmap of metric correlations
3. **relationships.png** - Scatter plots with trend lines
4. **Console logs** - Detailed analysis results and statistics

## 🔍 Key Metrics Analyzed

- **MP** (Minutes Played) - Player usage level
- **WS** (Win Shares) - Player contribution to wins
- **FG%** (Field Goal Percentage) - Shooting efficiency
- **USG%** (Usage Percentage) - Percentage of team plays used
- **Salary/Cap Ratio** - Normalized salary across different eras

## 📚 Methodology

1. **Data Loading** - Read NBA stats and salary cap data from Excel
2. **Data Cleaning** - Handle missing values, standardize column names
3. **Feature Engineering** - Create normalized and log-transformed salary metrics
4. **Exploratory Analysis** - Calculate correlations and distributions
5. **Regression Modeling** - Build predictive models with standardized features
6. **Visualization** - Generate professional charts and heatmaps

## 🎓 Key Insights

- Win Shares shows strong positive correlation with player salaries
- Minutes Played is a key predictor of salary levels
- Salary normalization by cap reveals consistent market patterns across years
- Usage percentage and shooting efficiency also influence salary negotiations

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✉️ Contact

For questions or feedback, please reach out to the repository maintainer.

---

**Last Updated:** 2026-04-18 13:15:53
**Python Version:** 3.8+
**Status:** Active Development