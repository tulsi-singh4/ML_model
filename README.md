🌿 CO₂ Capture Efficiency Prediction using Machine Learning

🔬 Random Forest Model for Amine Absorption Systems


---

📌 Overview

This project builds a Machine Learning model to predict CO₂ capture efficiency in an amine absorption system, a widely used industrial method for carbon capture.
Using Random Forest Regression, the model learns the relationship between operating variables (like temperature, pressure, amine type, concentration, etc.) and CO₂ loading.

The project includes two versions of the model:

🧪 Basic Model: Simple Random Forest

🚀 Advanced Model: Feature engineering + scaling + GridSearchCV optimization



---

🎯 Problem Statement

Amine-based CO₂ absorption is influenced by multiple interacting variables.
Predicting CO₂ capture efficiency using basic equations is difficult.

This project uses machine learning to build a fast and reliable prediction model that can estimate CO₂ loading without running complex simulations.


---

📊 Key Features

✅ Random Forest Regression (basic & optimized versions)

✅ Label Encoding of categorical amine types

✅ Feature engineering

Total concentration

Concentration ratio

Temperature × Pressure interaction


✅ Hyperparameter tuning with GridSearchCV

✅ Detailed performance metrics: R², RMSE, MAE

✅ Visualizations:

Feature importance

Predicted vs Actual

Residual plots




---

📁 Project Structure

CO2-Capture-ML-Model/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── dataset.csv   (optional)
│
└── src/
    ├── basic_model.py
    └── advanced_model.py


---

🧠 Machine Learning Approach

1️⃣ Basic Model

Label encoding for amine types

Random Forest with fixed hyperparameters

Accuracy achieved: R² ≈ 0.89

Feature importance visualization


2️⃣ Advanced Model

Additional engineered features

Standard scaling

Hyperparameter tuning using GridSearchCV

Residual and error analysis

More stable and generalizable predictions



---

📈 Results

⭐ Model Performance

R² Score: ~0.89 (Basic), improved with GridSearch

RMSE: ~0.095

Pressure and temperature identified as the most influential variables

Predictions align closely with actual CO₂ loading values


🌡 Top Influencing Factors

Pressure (highest)

Temperature

Amine type

Concentration of amine



---

🛠 Tech Stack

Python

NumPy, Pandas

Scikit-learn

Matplotlib

LabelEncoder, RandomForestRegressor, GridSearchCV



---

🚀 How to Run

# Clone the repository
git clone https://github.com/yourusername/CO2-Capture-ML-Model.git

# Install dependencies
pip install -r requirements.txt

# Run basic model
python src/basic_model.py

# Run advanced model
python src/advanced_model.py


---

🌍 Motivation

With rising global CO₂ emissions, carbon capture technologies need better tools to analyze and optimize performance.
Machine learning offers a modern, fast, and highly accurate way to support industrial CO₂ reduction efforts.


---
