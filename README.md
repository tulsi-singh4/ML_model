# 🌿 CO₂ Capture Efficiency Prediction using Machine Learning  
### 🔬 Random Forest Model for Amine Absorption Systems

---

## 📌 Overview  
This project develops a *Machine Learning model* to predict *CO₂ capture efficiency* in an *amine absorption system*, a widely used industrial carbon-capture method.  
Using a *Random Forest Regression* approach, the model learns how variables such as temperature, pressure, amine type, and concentration influence CO₂ loading.

The repository includes *two model versions*:  
- 🧪 *Basic Model* – simple Random Forest  
- 🚀 *Advanced Model* – feature engineering + scaling + GridSearchCV tuning  

---

## 🎯 Problem Statement  
Predicting CO₂ capture efficiency using classical equations is challenging due to multiple interacting operating variables.  
This project builds a *data-driven ML model* that can accurately estimate CO₂ loading without requiring complex thermodynamic simulations.

---

## 📊 Key Features  
- *Random Forest regression* (basic + optimized)  
- *Label Encoding* for categorical amine types  
- *Feature engineering*:  
  - total_conc  
  - conc_ratio  
  - temp_pressure  
- *Hyperparameter tuning* with GridSearchCV  
- Evaluation metrics: *R², **RMSE, **MAE*  
- Visualizations for feature importance, predicted vs actual, residuals  

---
---

## 🧠 Machine Learning Approach  

### *1️⃣ Basic Model*
- Label encoding  
- Random Forest with fixed hyperparameters  
- Achieved *R² ≈ 0.89*  
- Includes feature importance plot  

### *2️⃣ Advanced Model*
- Additional engineered features  
- Standard scaling applied  
- Tuned with *GridSearchCV*  
- Residual + prediction error analysis  

---

## 📈 Results  

### ⭐ Model Performance  
- *R² Score:* ~0.89 (Basic), further improved after tuning  
- *RMSE:* ~0.095  
- Predictions closely match actual CO₂ loading values  

### 🌡 Most Influential Variables  
- Pressure  
- Temperature  
- Amine type  
- Concentration levels  

---

## 🛠 Tech Stack  
- *Python*  
- *NumPy, **Pandas*  
- *Scikit-learn*  
- *Matplotlib*  
- *RandomForestRegressor, **GridSearchCV*  

---
## 🌍 Motivation

Carbon capture is essential to reduce global CO₂ emissions.
Machine learning provides a fast, scalable, and accurate approach to analyze capture performance and optimize operating conditions.
