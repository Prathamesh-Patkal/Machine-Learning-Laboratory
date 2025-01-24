import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load Dataset
file_path = r"data_engineer_jobs_salaries_2024.csv"
data = pd.read_csv(file_path)

# Display initial rows and information about the dataset
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows with missing values (if applicable)
data_cleaned = data.dropna()

# Display unique values in categorical columns (if any)
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}': {data_cleaned[col].unique()}")

# Correlation Matrix (Numerical Data)
numerical_cols = data_cleaned.select_dtypes(include=['number']).columns
if len(numerical_cols) > 1:
    print("\nCorrelation Matrix:")
    correlation_matrix = data_cleaned[numerical_cols].corr()
    print(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

# Visualization: Pairplot
sns.pairplot(data_cleaned)
plt.title("Pairplot of Numerical Features")
plt.show()

# Example Statistical Analysis
if len(numerical_cols) >= 2:
    print("\nT-Test between the first two numerical columns:")
    col1, col2 = numerical_cols[:2]
    t_stat, p_value = stats.ttest_ind(data_cleaned[col1], data_cleaned[col2])
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
