import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
nba = pd.read_excel(r"C:\Users\diksh\Desktop\New folder\NBA Talent Analysis Part BandC data.xlsx", sheet_name="NBA Data")
cap = pd.read_excel(r"C:\Users\diksh\Desktop\New folder\NBA Talent Analysis Part BandC data.xlsx", sheet_name="NBA Salary Cap History")

# Clean column names
nba.rename(columns={"FG%": "FG_pct", "3P%": "3P_pct", "USG%": "USG_pct"}, inplace=True)

# Convert numeric columns
nba = nba.apply(pd.to_numeric, errors='ignore')

# Extract year and clean cap data
cap['Deal_Year'] = cap.iloc[:,0].astype(str).str.extract(r'(\d{4})').astype(float)
cap['SalaryCap'] = pd.to_numeric(cap.iloc[:,1], errors='coerce')

# Merge datasets
nba = nba.merge(cap[['Deal_Year','SalaryCap']], on='Deal_Year', how='left')

# Feature engineering
nba['cap_norm_salary'] = nba['Deal Average Salary'] / nba['SalaryCap']
nba['log_cap_salary'] = np.log(nba['cap_norm_salary'])

# Simple visualization
plt.hist(nba['log_cap_salary'].dropna(), bins=30)
plt.title("Distribution of log(salary/cap)")
plt.xlabel("log(salary/cap)")
plt.ylabel("Count")
plt.show()

# Basic correlation
print(nba[['log_cap_salary','MP','WS']].corr())