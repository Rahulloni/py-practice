import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate Dates
dates = pd.date_range(start="2025-01-01", periods=300, freq='B')

# 1. Generate normal stock path
price = 100 + np.cumsum(np.random.normal(0, 1, 300))

# 2. Add Outliers (Sudden spikes)
outlier_indices = np.random.choice(300, 10, replace=False)
price[outlier_indices] += np.random.normal(0, 20, 10)

# 3. Add Jumps (Structural break)
jump_start = 150
price[jump_start:] += 30

# 4. Introduce Missing Data (Gaps)
missing_data_indices = np.random.choice(300, 20, replace=False)
price[missing_data_indices] = np.nan

# Create DataFrame
df = pd.DataFrame({'Date': dates, 'Close': price})

# Save to CSV
df.to_csv('noisy_stock_data.csv', index=False)
print("File 'noisy_stock_data.csv' generated with outliers, jumps, and missing data.")