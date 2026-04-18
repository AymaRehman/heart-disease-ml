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
