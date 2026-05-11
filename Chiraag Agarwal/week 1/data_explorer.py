import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------- Step 1: Load CSV ----------
try:
    df = pd.read_csv('weather.csv')
except FileNotFoundError:
    print("❌ File not found!")
    exit()

print("Shape:", df.shape)
print("\nHead:\n", df.head())

# ---------- Step 2: Exploration ----------
print("\nDescribe:\n", df.describe())

print("\nInfo:")
df.info()

print("\nMissing Values:\n", df.isnull().sum())

# Select numerical columns
numerical_df = df.select_dtypes(include=[np.number])

print("\nCorrelations:\n", numerical_df.corr())

# ---------- Step 3: Auto Insights ----------
print("\n🔍 Insights:")

# Missing values insight
missing = df.isnull().sum()
for col in missing.index:
    if missing[col] > 0:
        print(f"⚠ {col} has {missing[col]} missing values")

# High correlation insight
corr_matrix = numerical_df.corr()
for i in corr_matrix.columns:
    for j in corr_matrix.columns:
        if i != j and abs(corr_matrix.loc[i, j]) > 0.8:
            print(f"🔥 High correlation between {i} and {j}: {corr_matrix.loc[i, j]:.2f}")

# ---------- Step 4: Visualization ----------
for col in numerical_df.columns:
    # Histogram
    plt.figure()
    df[col].hist(bins=20)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'{col}_hist.png')
    plt.close()

# Scatter (first two columns if exist)
if len(numerical_df.columns) >= 2:
    col1, col2 = numerical_df.columns[:2]
    plt.figure()
    plt.scatter(df[col1], df[col2])
    plt.title(f'{col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.savefig('scatter.png')
    plt.close()

# Boxplot
plt.figure()
numerical_df.boxplot()
plt.savefig('boxplot.png')
plt.close()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png')
plt.close()

# ---------- Step 5: Export ----------
df.describe().to_csv('report.csv')

print("\n✅ Report saved")
print("✅ Visualizations saved")