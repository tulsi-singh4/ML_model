# Simple model, Label Encoding, Metrics, Feature Importance
# basic_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("data/dataset.csv")

df['amine1'] = df['amine1'].fillna('None').astype(str)
df['amine2'] = df['amine2'].fillna('None').astype(str)

# --------------------------
# Label Encoding
# --------------------------
le1 = LabelEncoder()
le2 = LabelEncoder()

df['amine1_encoded'] = le1.fit_transform(df['amine1'])
df['amine2_encoded'] = le2.fit_transform(df['amine2'])

# --------------------------
# Feature & Target
# --------------------------
X = df[["amine1_encoded", "amine2_encoded", "conc1", "conc2", "temperature", "P"]]
y = df["CO2_loading"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------
# Model Training
# --------------------------
rf = RandomForestRegressor(n_estimators=800, max_depth=60, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# --------------------------
# Evaluation
# --------------------------
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# --------------------------
# Feature Importance Plot
# --------------------------
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Model (Basic)")
plt.tight_layout()
plt.show()
