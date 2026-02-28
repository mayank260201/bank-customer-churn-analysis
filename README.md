# 🏦 Bank Customer Churn Analysis — End-to-End ML + BI Project

This project predicts **customer churn for a bank** using:
- Python (Pandas, Scikit-learn)
- Machine Learning Models
- Power BI Dashboard
- PostgreSQL for data storage
- End-to-end deployment documentation

The goal is to identify **which customers are most likely to leave the bank**,  
and provide actionable insights for retention.

## 📦 Dataset Information

This project uses the **Bank Customer Churn Dataset** containing:

- 10,000 customer records
- 12 features (demographic + financial)
- Binary churn label (0 = stayed, 1 = churned)

### 📌 Dataset Source & Credit
Dataset: *Bank Customer Churn Dataset*  
Author: **Gaurav Topre**  
Source: Kaggle  
🔗 https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

## 📁 Project Structure

Bank-Customer-Churn-Analysis/
│
├── data/
│   └── raw/
│       └── Bank Customer Churn Prediction.csv
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_evaluation.ipynb
│
├── powerbi/
│   └── churn_dashboard.pbix
│
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── scripts/
│   └── predict.py
│
└── README.md

## 🛠️ Tech Stack

### 🔹 Programming
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)

### 🔹 Database
- PostgreSQL  
- pgAdmin4

### 🔹 BI & Visualization
- Power BI  
- Excel (Pivots / Summary tables)

### 🔹 ML Models
- Logistic Regression  
- Random Forest Classifier  
- Dummy Classifier (baseline)

### 🔹 Deployment Tools
- Pickle (.pkl model exports)
- Power BI integration scripts
- SQL integration workflow

## 🔍 Exploratory Data Analysis (EDA) Summary

### ✔ Key Findings:
- **20% churn rate** → dataset is imbalanced
- **Germany** shows the highest churn
- **Older customers** churn significantly more
- **High balance customers** are at higher churn risk
- **Inactive members** churn more than active ones
- **Customers with 2+ products** churn more
- Salary has almost **no impact** on churn
- Credit score shows mild correlation

### ✔ Churn Drivers (Strongest → Weakest)
1. Age  
2. Balance  
3. Number of products  
4. Country (Germany)  
5. Active member status  
6. Credit score  
7. Salary (very weak)

## 🤖 Modeling Summary

We built the following models:

### 1️⃣ Dummy Classifier (Baseline)
- Accuracy ≈ 80% (misleading due to imbalance)
- Recall for churn = **0**
- Purpose: Baseline benchmark only

### 2️⃣ Logistic Regression
- Good interpretability  
- ROC-AUC ~ 0.75  
- Performs decently but misses many churners

### 3️⃣ Random Forest Classifier (BEST MODEL)
- Highest accuracy  
- Best recall for churners  
- Strong ROC-AUC  
- Best PR Curve performance  
- Provides feature importance

## 📊 Model Evaluation Summary

### ✔ ROC–AUC
- Random Forest performs best with strong class separation.

### ✔ Precision–Recall
- Random Forest gives the highest recall and average precision.
- Important because banks prefer catching more churners.

### ✔ Confusion Matrix
- Random Forest reduces false negatives significantly.

### ✔ Final Decision
**Random Forest (with threshold tuning)** chosen as the final model.

## 🚀 Model Deployment Summary

- Final model exported as: `models/churn_model.pkl`
- Scaler exported as: `models/scaler.pkl`
- Prediction script (`predict.py`) supports new customer scoring
- Power BI integration documented
- PostgreSQL scoring workflow documented

