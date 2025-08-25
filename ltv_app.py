#!/usr/bin/env python
# coding: utf-8

# 
# # 📊 Customer Lifetime Value (LTV) Prediction & Segmentation
# 
# This notebook walks through the **end-to-end process of predicting Customer Lifetime Value (LTV)** using machine learning models, and then segmenting customers into groups for business insights.
# 
# **Objectives:**
# - Understand customer purchasing behavior (Recency, Frequency, Spend, etc.).  
# - Build predictive models to estimate LTV.  
# - Compare models and evaluate accuracy.  
# - Segment customers into business-friendly groups.  
# - Visualize results in an easy-to-understand format.  
# 
# ---
# 

# 
# ## 1. Data Loading & Exploration  
# 
# We start by loading the dataset containing customer purchase behavior.  
# Key columns include:  
# 
# - `Recency`: Days since last purchase  
# - `Tenure`: Days since becoming a customer  
# - `Frequency`: Number of purchases  
# - `TotalSpend`: Total money spent  
# - `AOV`: Average order value  
# - `Predicted_LTV`: Model-predicted Lifetime Value  
# 
# Let's explore the data.
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('D:\Prediction Model\data\synthetic_beverage_sales_data.csv', nrows=1048575)

# Look at first 5 rows
print(df.head())

# Check for missing values and data types
print(df.describe())
print(df.info())
print(df.isnull().sum())


# In[3]:


# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])


# Create a reference point (last date in dataset)
max_date = df["Order_Date"].max()

# Aggregate customer features
customer_df = df.groupby("Customer_ID").agg({
    "Order_Date": [
        lambda x: (max_date - x.max()).days,   # Recency
        lambda x: (x.max() - x.min()).days     # Tenure
    ],
    "Order_ID": "nunique",                     # Frequency
    "Total_Price": ["sum", "mean"]             # Monetary value: total + AOV
}).reset_index()

# Rename columns
customer_df.columns = [
    "Customer_ID", "Recency", "Tenure", "Frequency", "TotalSpend", "AOV"
]

print(customer_df.head())


# In[ ]:





# In[4]:


features = ["Recency", "Tenure", "Frequency", "TotalSpend", "AOV"]

plt.figure(figsize=(15, 8))
for i, col in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(customer_df[col], bins=30, kde=True)
    plt.title(col)

plt.tight_layout()
plt.show()


# In[5]:


plt.figure(figsize=(8,6))
sns.heatmap(customer_df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[6]:


# Only apply log transform to columns that exist in each dataframe

# For df
df["Total_Price_log"] = np.log1p(df["Total_Price"])

# For customer_df
customer_df["AOV_log"] = np.log1p(customer_df["AOV"])
customer_df["Recency_log"] = np.log1p(customer_df["Recency"])
customer_df["Tenure_log"] = np.log1p(customer_df["Tenure"])
customer_df["Frequency_log"] = np.log1p(customer_df["Frequency"])
# Plot before & after for one feature (say TotalSpend)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df["Total_Price"], bins=50, kde=True, ax=axes[0])
axes[0].set_title("Before Log: TotalSpend")

sns.histplot(df["Total_Price_log"], bins=50, kde=True, ax=axes[1])
axes[1].set_title("After Log: TotalSpend")

plt.tight_layout()
plt.show()


# In[7]:


# Before log
print("Skewness (before log):", df["Total_Price"].skew())

# After log
df["Total_Price_log"] = np.log1p(df["Total_Price"])   # log1p handles 0 safely
print("Skewness (after log):", df["Total_Price_log"].skew())


# In[8]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(customer_df["AOV"], bins=50, kde=True, ax=axes[0])
axes[0].set_title("Before Log: AOV")

sns.histplot(customer_df["AOV_log"], bins=50, kde=True, ax=axes[1])
axes[1].set_title("After Log: AOV")

plt.tight_layout()
plt.show()


# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(customer_df["Frequency"], bins=50, kde=True, ax=axes[0])
axes[0].set_title("Before Log: Frequency")

sns.histplot(customer_df["Frequency_log"], bins=50, kde=True, ax=axes[1])
axes[1].set_title("After Log: Frequency")

plt.tight_layout()
plt.show()


# In[10]:


import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Modeling + plots
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


# In[11]:


# Features (inputs) and Target (output)
X = customer_df[['Recency', 'Tenure', 'Frequency', 'AOV']]  # independent variables
y = customer_df['TotalSpend']  # dependent variable (target)


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[13]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[14]:


from sklearn.ensemble import RandomForestRegressor

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Random Forest doesn't need scaling


# In[15]:


# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train) 


# In[16]:


# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


# In[17]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test, scaled=False):
    if scaled:
        preds = model.predict(X_test_scaled)
    else:
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return mae, rmse, r2

# Evaluate each model
print("Linear Regression:", evaluate_model(lr, X_test, y_test, scaled=True))
print("Random Forest:", evaluate_model(rf, X_test, y_test))
print("XGBoost:", evaluate_model(xgb, X_test, y_test))


# In[18]:


# Example: Predict with best model (say Random Forest won)
final_preds = rf.predict(X)

# Add predictions to dataset
customer_df['Predicted_LTV'] = final_preds

# Save to CSV
customer_df.to_csv("Final_LTV_Predictions.csv", index=False)


# In[19]:


# Predict on test set using Random Forest
y_pred_test = rf.predict(X_test)

plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.xlabel("Actual LTV (TotalSpend)")
plt.ylabel("Predicted LTV")
plt.title("Predicted vs Actual LTV")
plt.show()


# In[20]:


# Define segments
quantiles = customer_df["Predicted_LTV"].quantile([0.33, 0.66]).values
def segment(x):
    if x <= quantiles[0]:
        return "Low"
    elif x <= quantiles[1]:
        return "Medium"
    else:
        return "High"

customer_df["LTV_Segment"] = customer_df["Predicted_LTV"].apply(segment)

# Save again
customer_df.to_csv("Final_LTV_Segmented.csv", index=False)


# In[21]:


sns.boxplot(x="LTV_Segment", y="Predicted_LTV", data=customer_df)
plt.title("Customer Segments by Predicted LTV")
plt.show()


# In[22]:


from xgboost import plot_importance

# Plot feature importance
plt.figure(figsize=(8,6))
plot_importance(xgb, importance_type="gain")  
plt.title("Feature Importance by Gain - XGBoost")
plt.show()


# In[23]:


# Weight (split count)
plt.figure(figsize=(8,6))
plot_importance(xgb, importance_type="weight")
plt.title("Feature Importance - Weight")
plt.show()


# In[24]:


# Gain (how much improvement feature brings)
plt.figure(figsize=(8,6))
plot_importance(xgb, importance_type="gain")
plt.title("Feature Importance - Gain")
plt.show()


# In[25]:


# Cover (number of samples split by feature)
plt.figure(figsize=(8,6))
plot_importance(xgb, importance_type="cover")
plt.title("Feature Importance - Cover")
plt.show()


# In[26]:


# Simplified feature importance (average across cover/gain/weight for clarity)
features = ["AOV", "Frequency", "Tenure", "Recency"]
importance = [1400, 1000, 900, 600]  # Example combined scores

plt.figure(figsize=(8,5))
bars = plt.barh(features, importance, color=["#1f77b4","#ff7f0e","#2ca02c","#d62728"])
plt.xlabel("Importance (relative)")
plt.title("What Drives Customer Value Most?")

# Annotate values
for bar, value in zip(bars, importance):
    plt.text(value+20, bar.get_y()+bar.get_height()/2, str(value), va='center')

plt.show()


# In[27]:


import numpy as np

# Example data (replace with your customer dataset)
np.random.seed(42)
AOV = np.random.normal(300, 50, 100)     # Avg order value
Frequency = np.random.normal(10, 3, 100) # Number of purchases

plt.figure(figsize=(7,7))
plt.scatter(AOV, Frequency, alpha=0.6)

# Quadrant thresholds (use your medians or business cutoffs)
aov_cutoff = np.median(AOV)
freq_cutoff = np.median(Frequency)

plt.axvline(aov_cutoff, color="red", linestyle="--")
plt.axhline(freq_cutoff, color="red", linestyle="--")

# Labels for quadrants
plt.text(aov_cutoff+10, freq_cutoff+1, "Superstars", fontsize=12, color="green")
plt.text(aov_cutoff+10, freq_cutoff-3, "Premium Buyers", fontsize=12, color="blue")
plt.text(aov_cutoff-120, freq_cutoff+1, "Steady Shoppers", fontsize=12, color="orange")
plt.text(aov_cutoff-120, freq_cutoff-3, "At-Risk/Low Value", fontsize=12, color="red")

plt.xlabel("AOV (spend per order)")
plt.ylabel("Frequency (number of purchases)")
plt.title("Customer Segmentation by Value")
plt.show()


# In[28]:


import seaborn as sns
import pandas as pd

# Example synthetic data
data = pd.DataFrame({
    "Tenure": np.random.choice(["Low", "High"], 200),
    "Recency": np.random.choice(["Recent", "Not Recent"], 200)
})

# Count of customers in each bucket
heatmap_data = data.groupby(["Tenure","Recency"]).size().unstack()

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Customer Retention Segments")
plt.show()


# In[29]:


# Create quantile-based segmentation
customer_df["LTV_Segment"] = pd.qcut(customer_df["Predicted_LTV"],
                                     q=[0, 0.2, 0.8, 1.0],
                                     labels=["Low", "Medium", "High"])


# In[30]:
import streamlit as st

segment_summary = customer_df.groupby("LTV_Segment").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Tenure": "mean",
    "AOV": "mean",
    "Predicted_LTV": ["mean", "count"]
}).round(2)

st.dataframe(segment_summary)


# In[31]:


sns.boxplot(x="LTV_Segment", y="Predicted_LTV", data=customer_df)
plt.title("Customer Segments by Predicted LTV")
st.pyplot(plt)

sns.barplot(x="LTV_Segment", y="Frequency", data=customer_df)
plt.title("Frequency by Segment")
st.pyplot(plt)


# In[32]:


import pandas as pd

# Create quartile-based segmentation
customer_df["LTV_Segment"] = pd.qcut(
    customer_df["Predicted_LTV"], 
    q=4, 
    labels=["Low", "Lower-Mid", "Upper-Mid", "High"]
)

# Check distribution
print(customer_df["LTV_Segment"].value_counts())


# In[33]:


import matplotlib.pyplot as plt

# Average feature values by segment
segment_analysis = customer_df.groupby("LTV_Segment")[["Recency", "Frequency", "TotalSpend", "AOV", "Predicted_LTV"]].mean()

# Plot bar chart
segment_analysis.plot(kind="bar", figsize=(12,6))
plt.title("Customer Segments: Average Behavior & Predicted LTV", fontsize=14)
plt.ylabel("Average Value")
plt.xlabel("LTV Segments")
plt.xticks(rotation=0)  # keep labels horizontal
plt.legend(title="Features")
plt.show()


# In[34]:


print(customer_df.columns.tolist())
print(customer_df.head())
customer_df.columns = customer_df.columns.str.strip()


# In[35]:


# Load your data
customer_df = pd.read_csv("Final_LTV_Predictions.csv")



# In[36]:


# --- FIX: auto-convert numeric columns ---
numeric_cols = ["Recency","Frequency","TotalSpend","AOV","Predicted_LTV"]
for col in numeric_cols:
    customer_df[col] = pd.to_numeric(customer_df[col], errors="coerce")


# In[37]:


# Sidebar
st.sidebar.title("Customer LTV Segmentation Dashboard")


# In[38]:


# Main Title
st.title("Customer Lifetime Value (LTV) Segmentation")


# In[39]:


# Segment Analysis (averages per LTV segment)
st.subheader("Segment Analysis")

# Ensure 'LTV_Segment' exists
if "LTV_Segment" not in customer_df.columns:
	customer_df["LTV_Segment"] = pd.qcut(
		customer_df["Predicted_LTV"], 
		q=4, 
		labels=["Low", "Lower-Mid", "Upper-Mid", "High"]
	)

segment_analysis = customer_df.groupby("LTV_Segment")[numeric_cols].mean()
st.dataframe(segment_analysis)


# In[40]:


# Boxplot
st.subheader("Predicted LTV Distribution by Segment")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x="LTV_Segment", y="Predicted_LTV", data=customer_df, ax=ax)
plt.title("Customer Segments by Predicted LTV")
st.pyplot(fig)


# In[41]:


# Countplot
st.subheader("Segment Size")
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.countplot(x="LTV_Segment", data=customer_df, ax=ax2, order=segment_analysis.index)
plt.title("Number of Customers per Segment")
st.pyplot(fig2)


# In[42]:


# --- NEW: Download Button for full data ---
st.subheader("Download Full Data")
csv_full = customer_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Full Data (CSV)",
    data=csv_full,
    file_name="Full_LTV_Data.csv",
    mime="text/csv"
)
