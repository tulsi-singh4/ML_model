# Feature engineering,scaling, GridSearchCV, plots
# advanced_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_excel("data/dataset.xlsx", sheet_name="Table S1")

df['amine1'] = df['amine1'].fillna('None').astype(str)
df['amine2'] = df['amine2'].fillna('None').astype(str)

df = df.dropna(subset=['CO2_loading'])

# Fill missing numeric values
for col in ['conc1', 'conc2', 'temperature', 'P']:
    df[col].fillna(df[col].median(), inplace=True)

# --------------------------
# Label Encoding
# --------------------------
le1 = LabelEncoder()
le2 = LabelEncoder()

df['amine1_encoded'] = le1.fit_transform(df['amine1'])
df['amine2_encoded'] = le2.fit_transform(df['amine2'])

# --------------------------
# Feature Engineering
# --------------------------
df['total_conc'] = df['conc1'] + df['conc2']
df['conc_ratio'] = df['conc1'] / (df['conc2'] + 0.001)
df['temp_pressure'] = df['temperature'] * df['P']

feature_cols = [
    "amine1_encoded", "amine2_encoded", "conc1", "conc2",
    "temperature", "P", "total_conc", "conc_ratio", "temp_pressure"
]

X = df[feature_cols]
y = df["CO2_loading"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------
# Grid Search for Hyperparameters
# --------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# --------------------------
# Evaluation
# --------------------------
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

print("Train R2:", r2_score(y_train, y_pred_train))
print("Test R2:", r2_score(y_test, y_pred_test))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Test MAE:", mean_absolute_error(y_test, y_pred_test))

# --------------------------
# Feature Importance
# --------------------------
importances = best_rf.feature_importances_
fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance")

plt.figure(figsize=(7, 5))
plt.barh(fi_df['feature'], fi_df['importance'])
plt.title("Feature Importance (Advanced RF)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# --------------------------
# Predicted vs Actual Plot
# --------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_test, alpha=0.5, edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual CO2 Loading")
plt.ylabel("Predicted CO2 Loading")
plt.title("Predicted vs Actual")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------
# Residual Plot
# --------------------------
residuals = y_test - y_pred_test

plt.figure(figsize=(6, 5))
plt.scatter(y_pred_test, residuals, alpha=0.5, edgecolor="k")
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
