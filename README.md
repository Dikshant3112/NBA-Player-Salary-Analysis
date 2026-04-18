Determinants of NBA Player Salaries 📊

## Objective
Analyze NBA player performance data to identify key factors influencing player salaries.

## Methodology
- Data cleaning and preprocessing
- Feature engineering (salary normalization using salary cap)
- Exploratory Data Analysis (EDA)
- Correlation analysis between salary and performance metrics

## Dataset
- NBA player statistics
- Salary cap data

## Tech Stack
- Python (Pandas, NumPy, Matplotlib)

## Key Insights
- Win Shares and Minutes Played strongly influence player salaries
- Salary normalized using salary cap for fair comparison

## Results

### Salary Distribution
![Salary Distribution](output/salary_distribution.png)

## Sample Output
Correlation between key variables:

- Salary vs Minutes Played → Positive relationship  
- Salary vs Win Shares → Strong positive relationship  

## How to Run
```bash
pip install pandas numpy matplotlib openpyxl
python nba_salary_analysis.py
