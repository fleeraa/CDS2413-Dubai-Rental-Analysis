
import pandas as pd
import numpy as np
def descriptive_statistics(sample, field):
    """
    Calculate descriptive statistics for a selected field.
    Parameters:
        sample (pandas.DataFrame): Dataset or selected sample.
        field (str): Column name to analyse.
    Returns:
        pandas.Series: Descriptive statistics for the field.
    """
    if not isinstance(sample, pd.DataFrame):
        raise TypeError("The sample must be a pandas DataFrame.")
    if field not in sample.columns:
        raise ValueError(f"Column '{field}' was not found in the sample.")
    data = sample[field]
    if not pd.api.types.is_numeric_dtype(data):
        raise TypeError(f"Column '{field}' must contain numeric data.")
    data = data.dropna()
    if data.empty:
        raise ValueError(f"Column '{field}' has no valid numeric values.")
    results = {
        "Count": data.count(),
        "Mean": data.mean(),
        "Median": data.median(),
        "Mode": data.mode().tolist(),
        "Standard Deviation": data.std(),
        "Variance": data.var(),
        "Minimum": data.min(),
        "25th Percentile (Q1)": data.quantile(0.25),
        "75th Percentile (Q3)": data.quantile(0.75),
        "Maximum": data.max(),
        "Range": data.max() - data.min(),
        "Skewness": data.skew()
    }
    return pd.Series(results, name=field)
df = pd.read_csv("data/rentals.csv")
statistics = descriptive_statistics(df, "annual_rent_usd")
print("Descriptive Statistics for Annual Rent")
print(statistics)

# Task 7: Random Sampling of 150 Observations
random_sample = df.sample(n=150, random_state=42)
random_sample_stats = descriptive_statistics(
    random_sample,
    "annual_rent_usd"
)
print("\nDescriptive Statistics for Random Sample (150 Observations)")
print(random_sample_stats)
print("\nRandom Sample Size:", len(random_sample))

# Task 8: Systematic Sampling
sample_size = 150
N = len(df)
k = N // sample_size
start = np.random.default_rng(seed=42).integers(0, k)
systematic_sample = df.iloc[
    np.arange(start, N, k)[:sample_size]
].copy()
systematic_stats = descriptive_statistics(
    systematic_sample,
    "annual_rent_usd"
)
print("\nDescriptive Statistics for Systematic Sample")
print(systematic_stats)
print("\nSystematic Sample Size:", len(systematic_sample))
print("Sampling Interval:", k)
print("Starting Position (zero-based):", start)

# Task 10: Visualisation of the Dependent Variable

import matplotlib.pyplot as plt
import seaborn as sns

# Set a consistent visual style
sns.set_theme(style="whitegrid")

# Select the dependent variable
target = "annual_rent_usd"

# Check required columns
if target not in df.columns:
    raise ValueError(f"Required column '{target}' was not found.")

# Ensure the dependent variable is numeric
rent = pd.to_numeric(df[target], errors="coerce").dropna()

# --------------------------------------------------
# 1. Scatter Plot: Annual Rent vs Property Size
# --------------------------------------------------

if "area_sqft" in df.columns:
    scatter_data = df[["area_sqft", target]].copy()
    scatter_data["area_sqft"] = pd.to_numeric(
        scatter_data["area_sqft"], errors="coerce"
    )
    scatter_data[target] = pd.to_numeric(
        scatter_data[target], errors="coerce"
    )
    scatter_data = scatter_data.dropna()

    # Use a reproducible sample for a clearer plot
    scatter_data = scatter_data.sample(
        n=min(2000, len(scatter_data)), random_state=42
    )

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=scatter_data,
        x="area_sqft",
        y=target,
        alpha=0.5
    )
    plt.title("Scatter Plot: Annual Rent vs Property Size")
    plt.xlabel("Property Size (sq. ft.)")
    plt.ylabel("Annual Rent (USD)")
    plt.tight_layout()
    plt.savefig("task10_scatter_plot.png", dpi=300)
    plt.show()
else:
    print("Scatter plot skipped: area_sqft was not found.")


# --------------------------------------------------
# 2. Box Plot: Distribution of Annual Rent
# --------------------------------------------------

plt.figure(figsize=(9, 5))
sns.boxplot(x=rent)
plt.title("Box Plot of Annual Rental Prices")
plt.xlabel("Annual Rent (USD)")
plt.tight_layout()
plt.savefig("task10_box_plot.png", dpi=300)
plt.show()


# --------------------------------------------------
# 3. Histogram: Distribution of Annual Rent
# --------------------------------------------------

plt.figure(figsize=(9, 5))
sns.histplot(rent, bins=40, kde=True)
plt.title("Histogram of Annual Rental Prices")
plt.xlabel("Annual Rent (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("task10_histogram.png", dpi=300)
plt.show()


# --------------------------------------------------
# 4. Heat Map: Correlation Between Numeric Variables
# --------------------------------------------------

candidate_columns = [
    "annual_rent_usd",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "parking_spaces",
    "metro_distance_min"
]

# Keep only columns that exist in the dataset
numeric_columns = [
    col for col in candidate_columns
    if col in df.columns
]

heatmap_data = df[numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)

# Remove columns that contain no usable numeric values
heatmap_data = heatmap_data.dropna(axis=1, how="all")

if heatmap_data.shape[1] >= 2:
    correlation_matrix = heatmap_data.corr(method="pearson")

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0
    )
    plt.title("Heat Map of Numerical Variable Correlations")
    plt.tight_layout()
    plt.savefig("task10_heatmap.png", dpi=300)
    plt.show()
else:
    print("Heat map skipped: at least two numeric columns are needed.")

print("\nAll available Task 10 plots have been generated.")

# Task 11: Correlation Analysis and Hypothesis Testing

from scipy.stats import pearsonr, spearmanr

# Select the independent and dependent variables
independent_var = "area_sqft"
dependent_var = "annual_rent_usd"

# Convert both variables to numeric and remove missing values
correlation_data = df[
    [independent_var, dependent_var]
].copy()

correlation_data[independent_var] = pd.to_numeric(
    correlation_data[independent_var], errors="coerce"
)

correlation_data[dependent_var] = pd.to_numeric(
    correlation_data[dependent_var], errors="coerce"
)

correlation_data = correlation_data.dropna()

# Check that sufficient observations and variation exist
if (
    len(correlation_data) < 3
    or correlation_data[independent_var].nunique() < 2
    or correlation_data[dependent_var].nunique() < 2
):
    raise ValueError(
        "Insufficient valid observations or variation for correlation testing."
    )

x = correlation_data[independent_var]
y = correlation_data[dependent_var]

# State the hypotheses
print("\nTASK 11: CORRELATION HYPOTHESIS TESTING")
print("Independent variable:", independent_var)
print("Dependent variable:", dependent_var)
print("Valid observations:", len(correlation_data))
print("Significance level: 0.05")

print("\nH0: There is no population correlation between property size and annual rent.")
print("H1: There is a population correlation between property size and annual rent.")

# Pearson correlation: tests linear association
pearson_result = pearsonr(x, y)

print("\n--- Pearson Correlation Test ---")
print(f"Correlation coefficient: {pearson_result.statistic:.4f}")
print(f"P-value: {pearson_result.pvalue:.6g}")

if pearson_result.pvalue < 0.05:
    print("Decision: Reject H0.")
    print("Conclusion: The linear correlation is statistically significant.")
else:
    print("Decision: Fail to reject H0.")
    print("Conclusion: There is insufficient evidence of a linear correlation.")

# Spearman correlation: tests monotonic association
spearman_result = spearmanr(x, y)

print("\n--- Spearman Rank Correlation Test ---")
print(f"Correlation coefficient: {spearman_result.statistic:.4f}")
print(f"P-value: {spearman_result.pvalue:.6g}")

if spearman_result.pvalue < 0.05:
    print("Decision: Reject H0.")
    print("Conclusion: The monotonic correlation is statistically significant.")
else:
    print("Decision: Fail to reject H0.")
    print("Conclusion: There is insufficient evidence of a monotonic correlation.")

# Task 12: One-Sample t-Test

from scipy.stats import ttest_1samp

# Dependent variable
target = "annual_rent_usd"

# Use the random sample created in Task 7
sample_rent = pd.to_numeric(
    random_sample[target], errors="coerce"
).dropna()

# Use the full dataset mean as the reference benchmark
reference_mean = pd.to_numeric(
    df[target], errors="coerce"
).dropna().mean()

# Significance level
alpha = 0.05

# Hypotheses
print("\nTASK 12: ONE-SAMPLE T-TEST")
print("H0: The sample mean equals the reference mean.")
print("H1: The sample mean differs from the reference mean.")

# Perform the two-tailed one-sample t-test
test_result = ttest_1samp(
    sample_rent,
    popmean=reference_mean
)

print(f"\nSample size: {len(sample_rent)}")
print(f"Sample mean: USD {sample_rent.mean():,.2f}")
print(f"Reference mean: USD {reference_mean:,.2f}")
print(f"T-statistic: {test_result.statistic:.4f}")
print(f"P-value: {test_result.pvalue:.6f}")
print(f"Significance level: {alpha}")

# Interpret the result
if test_result.pvalue < alpha:
    print("\nDecision: Reject H0.")
    print("Conclusion: The sample mean differs significantly "
          "from the reference mean.")
else:
    print("\nDecision: Fail to reject H0.")
    print("Conclusion: There is insufficient evidence that "
          "the sample mean differs from the reference mean.")
