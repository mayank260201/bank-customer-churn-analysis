# SQL (PostgreSQL) — How to Run

I used **PostgreSQL + pgAdmin4** to store the data and answer business questions with SQL.

## Files (run in this order)

| File | What it does |
|---|---|
| `01_create_tables.sql` | Creates `customers` (raw data) and `customers_scored` (data + model scores) |
| `02_load_data.sql` | Loads both CSV files into the tables |
| `03_data_checks.sql` | Row counts, missing values, duplicates (sanity checks) |
| `04_churn_analysis.sql` | 9 business questions: churn by country, age, products, activity, gender, balance |
| `05_model_results.sql` | How good the risk bands are, balance at risk, retention call list (view) |

## Steps in pgAdmin4

1. Open pgAdmin4 → right-click **Databases** → **Create → Database** → name it `bank_churn`.
2. Select `bank_churn` → **Tools → Query Tool**.
3. Open `01_create_tables.sql` → press **F5** (Run).
4. Load data (pick one):
   - Right-click table `customers` → **Import/Export Data** → Import → file = `data/raw/Bank Customer Churn Prediction.csv` → Options: **Header ON**, Delimiter `,` → OK.
     Do the same for `customers_scored` with `data/processed/customers_scored.csv`.
   - OR right-click `bank_churn` → **PSQL Tool** and paste the two `\copy` lines from `02_load_data.sql` (change the path if your folder is somewhere else).
5. Run `03_data_checks.sql` → both tables should show 10000 rows and 0 problems.
6. Run `04_churn_analysis.sql` and `05_model_results.sql`. Each query has the result I got written above it.

> `customers_scored.csv` is created by `notebooks/05_model_export.ipynb`. Run that notebook first if the file is missing.

## Main things SQL showed

- Overall churn **20.37%**
- Germany **32.4%** vs France 16.2% / Spain 16.7%
- Age **50–59** churn **56%**
- **2 products = lowest churn (7.6%)**; 1 product 27.7%; 3–4 products 83–100%
- Inactive members churn **26.9%** vs active 14.3%
- Churned customers took **24%** of the bank's total balance with them
