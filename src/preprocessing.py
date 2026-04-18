import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Commenting so I know exactly what's going on in the code when I come back to it later.
# Authored: @AymaRehman

# Original Dataset Columns (14):
# age, sex, cp(chest pain type), trestbps(resting blood pressure), chol, fbs(fasting blood sugar),
# restecg(resting ecg results), thalach(maximum heart rate), exang(exercise-induced angina),
# oldpeak(st depression induced by exercise), slope(the slope of the peak exercise ST segment),
# ca(number of major vessels), thal(thalassemia), target(presence of heart disease)

df = pd.read_csv("data/heart.csv")

# Confirm dataset is in a workable format (.csv)
# "If the dataset retrieved from the repository is not in a format
# that is easy to work with...transform it into the required format."
# The Kaggle version is already .csv but let's just confirm through code anyway.
print("=== Format Check ===")
print(f"Dataset loaded from CSV. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nRaw data snapshot (all columns, first 5 rows):")
print(df.head().to_string())
print()

# Check for and handle textual (non-numeric) values
# "If the values of any features (attributes) are textual values
# ... transform them into numeric values."
print("=== Textual Value Check ===")
non_numeric_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if non_numeric_cols:
    print(f"Non-numeric columns found: {non_numeric_cols}. Encoding to numeric.")
    # LabelEncoder assigns arbitrary integer order. This is suitable for nominal
    # categories only and should not be used for ordinal features without custom mapping.
    for col in non_numeric_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        print(f"  Encoded: {col}")
else:
    print("All feature values are already numeric. No encoding required.")
print()

# Verify target variable
# UCI original uses 0-4
# Kaggle version should be binarised (0/1) already
print("=== Target Variable Check ===")
print(f"Unique target values: {sorted(df['target'].unique())}")
if df["target"].nunique() > 2:
    print("Multi-class target detected. Binarising: 0 = no disease, 1 = disease.")
    df["target"] = (df["target"] > 0).astype(int)
else:
    print("Target is already binary (0 = no disease, 1 = disease). No changes needed.")
print()


# Handle missing values
# "If there are missing values of features (attributes)...the
# student team must find a way to deal with these problems"
print("=== Missing Value Check ===")
missing = df.isnull().sum()
print("Missing values per feature:")
print(missing.to_string())
if missing.sum() == 0:
    print("No missing values found. No action required.")
else:
    rows_before = len(df)
    df = df.dropna()
    print(f"Rows dropped: {rows_before - len(df)}. Remaining: {len(df)}")
print()

# Remove duplicate data objects
# "if the dataset contains...duplicate data objects, the student
# team must find a way to deal with these problems"
print("=== Duplicate Check ===")
rows_before = len(df)
duplicates_found = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates_found}")
df = df.drop_duplicates()
print(f"Rows dropped: {rows_before - len(df)}. Remaining: {len(df)}\n")

continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Handle outliers before the correlation/feature-selection checks so that
# inflated/deflated Pearson r values caused by outliers do not affect
# feature retention decisions.
# "If there are...outliers...the student team must find a way
# to deal with these problems"
# IQR method applied to continuous features only.
print("=== Outlier Removal (IQR method, continuous features only) ===")
rows_before = len(df)
for feature in continuous_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
print(f"Rows removed: {rows_before - len(df)}. Remaining: {len(df)}\n")
