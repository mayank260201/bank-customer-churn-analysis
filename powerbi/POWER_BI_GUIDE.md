# Power BI Dashboard — Step by Step

Goal: build `powerbi/churn_dashboard.pbix` with **2 pages**:
1. **Churn Overview** – who is leaving
2. **Risk & Retention** – who to call next (uses the ML model scores)

Time: ~30–45 minutes.
Data: `data/processed/customers_scored.csv` (made by `notebooks/05_model_export.ipynb`).

---

## Step 1 – Load data
1. Open **Power BI Desktop** → **Get data → Text/CSV** → select `data/processed/customers_scored.csv` → **Transform Data**.
2. In Power Query check the types:
   - `customer_id` → Whole number (or Text)
   - `balance`, `estimated_salary`, `churn_probability` → Decimal number
   - `age_group`, `balance_group`, `credit_score_group`, `risk_band` → **Text** (Power BI may guess `balance_group` "0" as a number — change it to Text)
3. **Close & Apply**.

*(Optional: load from PostgreSQL instead → Get data → PostgreSQL database → server `localhost`, database `bank_churn` → table `customers_scored`. Same columns.)*

## Step 2 – Theme
**View → Themes → Browse for themes** → `powerbi/churn_theme.json`.

## Step 3 – Measures
Create all measures and the 3 sort columns from `powerbi/DAX_measures.md`.
Then set **Sort by column**: `age_group` → `Age Order`, `balance_group` → `Balance Order`, `risk_band` → `Risk Order`.

## Step 4 – Page 1: "Churn Overview"

| Visual | Fields |
|---|---|
| 4 **Cards** (top row) | Total Customers · Churned Customers · Churn Rate · % Balance Lost |
| **Clustered column** – Churn Rate by Country | X: country · Y: Churn Rate |
| **Clustered column** – Churn Rate by Age Group | X: age_group · Y: Churn Rate |
| **Clustered column** – Churn Rate by Products | X: products_number · Y: Churn Rate |
| **Donut** – Active vs Inactive churners | Legend: active_member · Values: Churned Customers |
| **Slicers** (left side) | country · gender |

Tip: turn on **Data labels** for the column charts.

## Step 5 – Page 2: "Risk & Retention"

| Visual | Fields |
|---|---|
| 3 **Cards** | High Risk Customers · Balance at Risk · Churners Caught % |
| **Clustered column** – Risk band vs actual churn | X: risk_band · Y: Churn Rate |
| **Stacked bar** – Balance by risk band | Y: risk_band · X: Total Balance |
| **Matrix** – Country × Risk band | Rows: country · Columns: risk_band · Values: Total Customers |
| **Table** – Retention call list | customer_id, country, age, balance, products_number, active_member, churn_probability · Filter: risk_band = High · Sort: balance (desc) |
| **Slicer** | risk_band |

## Step 6 – Check numbers (should match SQL + Excel)

| Check | Expected |
|---|---|
| Total Customers | 10,000 |
| Churn Rate | 20.4% |
| Germany churn rate | 32.4% |
| Age 50-59 churn rate | 56.0% |
| 2 products churn rate | 7.6% |
| % Balance Lost | 24.3% |
| High Risk Customers | 955 |
| Churn rate inside "High" band | 83.8% |
| Churners Caught % | ~66.7% |

## Step 7 – Save & export
- **File → Save as** → `powerbi/churn_dashboard.pbix`
- **File → Export → Export to PDF** → `powerbi/churn_dashboard.pdf`
- Take a screenshot of each page → `powerbi/screenshots/overview.png`, `powerbi/screenshots/risk_retention.png` (used in the main README)
