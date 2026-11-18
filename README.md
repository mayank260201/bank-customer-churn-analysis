# 🏦 Bank Customer Churn Analysis — End-to-End ML + BI Project

This project predicts **which bank customers are likely to leave (churn)** and turns that into something a retention team can actually use: a risk score for every customer, a ranked call list, SQL analysis and a Power BI dashboard.

🔗 **Live demo:** [Bank Churn Risk Checker](https://mayank260201.github.io/bank-customer-churn-analysis/) – enter a customer's details and get a churn risk score
📄 **Product brief:** [docs/PRODUCT_BRIEF.md](docs/PRODUCT_BRIEF.md) – problem, users, metrics, risks, roadmap

Built with:
- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- Machine Learning (Logistic Regression, Random Forest)
- PostgreSQL + pgAdmin4
- Power BI
- Excel

## 📦 Dataset Information

- 10,000 customer records, 12 columns (demographic + financial)
- Binary churn label (0 = stayed, 1 = churned)
- **20.37% churn** → imbalanced dataset

### 📌 Dataset Source & Credit
Dataset: *Bank Customer Churn Dataset*
Author: **Gaurav Topre**
Source: Kaggle
🔗 https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

## 📁 Project Structure

```
bank-customer-churn-analysis/
│
├── data/
│   ├── raw/Bank Customer Churn Prediction.csv     # original dataset
│   ├── processed/customers_scored.csv             # every customer + churn probability + risk band
│   └── sample/new_customers_sample.csv            # 10 customers to test predict.py
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb                 # cleaning + EDA
│   ├── 02_feature_engineering.ipynb               # encoding, scaling, train/test split
│   ├── 03_modeling.ipynb                          # Dummy, Logistic Regression, Random Forest
│   ├── 04_model_evaluation.ipynb                  # ROC, PR curve, CV, threshold tuning
│   └── 05_model_export.ipynb                      # save model, score all customers, export
│
├── models/
│   ├── churn_model.pkl                            # final Random Forest
│   └── scaler.pkl                                 # StandardScaler
│
├── scripts/
│   └── predict.py                                 # score new customers from the command line
│
├── sql/                                           # PostgreSQL: tables, load, checks, analysis
├── powerbi/                                       # Power BI guide, DAX measures, theme (+ .pbix)
├── excel/
│   └── churn_summary.xlsx                         # summary tables + charts + call list
├── docs/
│   ├── PRODUCT_BRIEF.md                           # product / business view
│   └── index.html                                 # live demo page (GitHub Pages)
├── requirements.txt
└── README.md
```

## 🔍 Exploratory Data Analysis (EDA) Summary

| Segment | Churn rate |
|---|---|
| Overall | **20.4%** |
| Germany vs France / Spain | **32.4%** vs 16.2% / 16.7% |
| Age 50–59 | **56.0%** (age 18–29: 7.6%) |
| 1 product / **2 products** / 3 products / 4 products | 27.7% / **7.6%** / 82.7% / 100% |
| Inactive vs active members | **26.9%** vs 14.3% |
| Female vs male | **25.1%** vs 16.5% |
| Zero balance vs has balance | 13.8% vs **24.1%** |

### ✔ Key Findings
- **Age** is the strongest driver – churn peaks at 50–59
- **Germany** churns 2× more than France and Spain
- **2 products = most loyal customers**; 3–4 products → almost everyone leaves
- **Inactive members** churn about 2× more
- Customers **with money in the account** churn more than zero-balance ones → the bank loses **24% of total balance** through churn
- Salary, tenure and credit card have almost **no impact**

## 🤖 Modeling Summary

| Model | Accuracy | Churn Recall | ROC-AUC |
|---|---|---|---|
| Dummy (baseline) | 79.7% | 0.00 | – |
| Logistic Regression | 80.8% | 0.19 | 0.775 |
| **Random Forest** | **86.1%** | 0.46 (0.67 at threshold 0.3) | **0.854** |

- 5-fold cross-validation accuracy (Random Forest): **86.5%**
- **Final model: Random Forest with threshold 0.3** – catches ~2 out of 3 churners (recall 0.67, precision 0.57).
  For a bank, missing a churner costs more than an extra retention call.
- Risk bands on all customers (cross-validated scores): **High** → 83.8% really churned, **Medium** → 40.1%, **Low** → 8.9%

## 💼 Business Use

- **Retention call list**: High/Medium risk customers (SQL view `retention_call_list`, Excel sheet, Power BI page)
- **Prioritisation**: ranking by *churn probability × balance* → the top 1,000 customers hold **51% of all churned balance** (vs ~10% for 1,000 random customers)
- Full product thinking (users, success metrics, A/B test, fairness risks, roadmap): [docs/PRODUCT_BRIEF.md](docs/PRODUCT_BRIEF.md)

## 🚀 How to Run

Needs **Python 3.11+**. Using a virtual environment keeps these versions separate from your other projects:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 1. run the notebooks in order (01 → 05) in Jupyter
# 2. score one example customer
python scripts/predict.py
# 3. score a CSV of new customers
python scripts/predict.py data/sample/new_customers_sample.csv
```

- **SQL:** see [sql/README.md](sql/README.md) (PostgreSQL + pgAdmin4, 5 scripts)
- **Power BI:** see [powerbi/POWER_BI_GUIDE.md](powerbi/POWER_BI_GUIDE.md)
- **Excel:** open `excel/churn_summary.xlsx`

> If `predict.py` can't load the model (different scikit-learn version), install the pinned versions from `requirements.txt`, or re-run `notebooks/05_model_export.ipynb` – it saves the model again.

## 🛠️ Fixes & Lessons Learned

- **Data leakage fix:** at first, notebooks 03 and 04 fitted the scaler on the *whole* dataset before the train/test split. Now the split happens first and the scaler is fit only on training data (like notebook 02). Scores barely changed (RF accuracy 86.05% → 86.1%), but the method is now correct.
- **Missing function:** `plot_roc()` was used in notebook 04 but never defined, so the notebook crashed on re-run. Added it.
- **Wrong EDA conclusion corrected:** I first wrote "2+ products churn more". Checking exact rates in SQL showed 2 products is actually the *safest* group. Correlation (−0.05) hid this because the pattern isn't linear.
- **Accuracy is misleading** on imbalanced data – the Dummy model gets 80% while catching zero churners.

## 🔮 Next Steps
- Add dates → predict churn in the *next 90 days* instead of a one-time snapshot
- Show the top reasons per customer (e.g. SHAP values)
- Test a model without gender (fairness)
- A/B test retention calls to prove they actually reduce churn
