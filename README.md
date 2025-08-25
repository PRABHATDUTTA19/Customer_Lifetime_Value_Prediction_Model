# 📈 **Customer Lifetime Value (LTV) Prediction & Segmentation**

### This project builds a machine learning pipeline to predict customer lifetime value (LTV) and segment customers into actionable groups. The end result is an interactive Streamlit app that allows exploration of predictions and segment-level insights.

### 🔍 **Workflow**

**Data Preprocessing** – cleaning, missing values, formatting.

**Feature Engineering** – log transforms (AOV, Recency, Tenure, Frequency), LTV drivers.

**Model Training** – trained XGBoost regression on engineered features.

**Segmentation** – grouped customers into 4 LTV-based segments (Low → High).

**Deployment** – packaged into a Streamlit dashboard for exploration.

# 📊 **Results**

## Model Performance

**R²**: 0.986 — explains 98.6% of LTV variation.
***R² values are between 0 and 1. A value of 1 means the model perfectly explains data variability, while 0 means it explains none.*** 

**RMSE**: 1,553 — typical prediction error ~1.5K
***RMSE: Root Mean Squared Error
RMSE is the standard deviation of the residuals (errors) and is calculated by taking the square root of the average of the squared differences between predicted and actual values.***

**MAE**: 900 — average error per customer
***MAE: Mean Absolute Error
MAE represents the average of the absolute differences between the predicted and actual values. It provides a straightforward measure of the magnitude of the errors.***

**MAPE**: 6.94% — predictions within ±7% of actual values
***MAPE: Mean Absolute Percentage Error
This metric calculates the average of the absolute differences between predicted and actual values, expressed as a percentage of the actual values.***

## ✅ **The model is highly reliable for ranking customers by future value.**

### Customer Segmentation

<img width="1005" height="326" alt="image" src="https://github.com/user-attachments/assets/fa3c5388-1ada-441c-a97e-4e85836ac2f6" />


# **Key Insights**

### **Revenue concentration**: Top 25% of customers = ~62% of revenue.

### **Inefficiency at the bottom**: Half of the base contributes <16% of revenue.

### **Customer spread**: High LTV customers spend ~10× more than Low LTV.

### **Order size effect**: AOV grows from 36 → 314 across segments.

## 🚀 Streamlit App
Run locally:
```
streamlit run app/ltv_app.py
```

## 🔑 Why This Project Matters

***Predicts future customer value, not just historical RFM.***

***Enables targeted retention of high-value customers.***

***Provides a business-friendly dashboard for exploration.***

***Supports scalable strategy: upsell, cross-sell, and churn prevention.***
