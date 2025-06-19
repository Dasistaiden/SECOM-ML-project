
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.diagnostic import kstest_normal
from scipy import stats

# Load the dataset
df = pd.read_csv("train_KNN_imputed.csv")

# Separate timestamp and label columns
timestamp = df["timestamp"]
label = df["label"]

# Select only numeric features
numeric_df = df.drop(columns=["timestamp", "label"])

# Perform Lilliefors test to identify non-normal features
non_normal_features = []
for col in numeric_df.columns:
    series = numeric_df[col].dropna()
    if len(series) > 20:  # Ensure enough samples for the test
        stat, p_value = kstest_normal(series)
        if p_value < 0.05:
            non_normal_features.append(col)

# Apply Yeo-Johnson transformation to non-normal features
pt = PowerTransformer(method='yeo-johnson')
df_transformed = numeric_df.copy()

for col in non_normal_features:
    original_data = df_transformed[col].values.reshape(-1, 1)
    transformed_data = pt.fit_transform(original_data)
    df_transformed[col] = transformed_data.flatten()

# Restore timestamp and label
df_transformed["timestamp"] = timestamp
df_transformed["label"] = label

# Save the transformed dataset
df_transformed.to_csv("train_KNN_YeoTransformed_Lilliefors.csv", index=False)

# Visualization: Q-Q plot for feature547 before and after transformation
feature_name = "feature547"
if feature_name in numeric_df.columns:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Q-Q plot before transformation
    stats.probplot(numeric_df[feature_name], dist="norm", plot=axs[0])
    axs[0].set_title(f"{feature_name} - Before Yeo-Johnson")

    # Q-Q plot after transformation
    stats.probplot(df_transformed[feature_name], dist="norm", plot=axs[1])
    axs[1].set_title(f"{feature_name} - After Yeo-Johnson")

    plt.tight_layout()
    plt.savefig("feature547_QQ_plot.png")
