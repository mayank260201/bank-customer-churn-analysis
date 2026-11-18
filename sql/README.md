# SQL (PostgreSQL) — How to Run

I used **PostgreSQL + pgAdmin 4** to store the data and answer business questions with SQL.
(Steps checked against pgAdmin 4 version 9.)

## Files (run in this order)

| File | What it does |
|---|---|
| `01_create_tables.sql` | Creates `customers` (raw data) and `customers_scored` (data + model scores) |
| `02_load_data.sql` | Loads both CSV files into the tables |
| `03_data_checks.sql` | Row counts, missing values, duplicates (sanity checks) |
| `04_churn_analysis.sql` | 9 business questions: churn by country, age, products, activity, gender, balance |
| `05_model_results.sql` | How good the risk bands are, balance at risk, retention call list (view), priority score |

> `customers_scored.csv` is created by `notebooks/05_model_export.ipynb`. Run that notebook first if the file is missing.

## Step 0 – One-time setup (avoids the "file not found / Binary Path" error)

pgAdmin's **Import/Export Data** and **PSQL Tool** use PostgreSQL's `psql.exe`. If pgAdmin doesn't know where it is, you get:
*"…psql.exe file not found. Please correct the Binary Path in the Preferences dialog."*

1. pgAdmin → **File** → **Preferences** → left side: **Paths** → **Binary paths**.
2. Under **PostgreSQL Binary Path**, find the row for your PostgreSQL version (e.g. PostgreSQL 16 or 17).
3. Set the path to the `bin` folder, usually: `C:\Program Files\PostgreSQL\17\bin` (change 17 to your version). Tick it as **Set as default**.
4. Click **Save**.

## Step 1 – Create the database and tables

1. In the **Object Explorer** (left), expand **Servers** → your server (enter your postgres password).
2. Right-click **Databases** → **Create** → **Database…** → Database: `bank_churn` → **Save**.
3. Click `bank_churn` → **Tools** menu → **Query Tool**.
4. In the Query Tool click the **Open File** (folder) icon → pick `sql/01_create_tables.sql` → press **F5** (Execute script).
   Expect: `CREATE TABLE` messages. Two "does not exist, skipping" notices the first time are normal.

## Step 2 – Load the data (pick ONE option)

**Option A – Import/Export Data dialog (easiest)**
1. Object Explorer → `bank_churn` → **Schemas** → **public** → **Tables** (right-click → **Refresh** if empty).
2. Right-click `customers` → **Import/Export Data…**
   - **General** tab: switch set to **Import**; **Filename** = `data/raw/Bank Customer Churn Prediction.csv`; **Format** = `csv`
   - **Options** tab: **Header** = **ON**; **Delimiter** = `,`
   - Click **OK**. A "Process completed" message appears at the bottom-right.
3. Do the same for `customers_scored` with `data/processed/customers_scored.csv`.

**Option B – PSQL Tool**
Right-click `bank_churn` → **PSQL Tool** → paste the two `\copy` lines from `02_load_data.sql` (change the path if your folder is somewhere else; use forward slashes `/`).

⚠️ Load each file **only once**. To reload, run `01_create_tables.sql` again first (it empties the tables).

## Step 3 – Run the checks and the analysis

Open and run (F5) in the Query Tool, in order:
1. `03_data_checks.sql` → both tables **10000** rows, all missing = **0**, duplicates = **no rows**, mismatches = **0**
2. `04_churn_analysis.sql`
3. `05_model_results.sql`

Each query has the result I got written in a comment above it.
Tip: F5 runs the whole file and shows the **last** result. To see one query's result, select (highlight) just that query and press F5.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `psql.exe file not found. Please correct the Binary Path…` | pgAdmin doesn't know where PostgreSQL's `bin` folder is | Step 0 |
| `invalid input syntax for type integer: "customer_id"` | Header row was imported as data | Import again with **Header = ON** (run `01_create_tables.sql` first) |
| `duplicate key value violates unique constraint "customers_pkey"` | The file was loaded twice | Run `01_create_tables.sql`, then load once |
| `relation "customers" does not exist` | Query Tool opened on the wrong database (e.g. `postgres`) | Open the Query Tool from `bank_churn` |
| `extra data after last expected column` / `missing data for column` | Wrong CSV loaded into a table | `customers` ← raw CSV (12 columns); `customers_scored` ← processed CSV (18 columns) |
| `could not open file … for reading: Permission denied` | Used `COPY` (server-side) instead of `\copy` | Use Option A, or `\copy` in the PSQL Tool |
| Tables don't appear in the Object Explorer | Tree not refreshed | Right-click **Tables** → **Refresh** |

## Main things SQL showed

- Overall churn **20.37%**
- Germany **32.4%** vs France 16.2% / Spain 16.7%
- Age **50–59** churn **56%**
- **2 products = lowest churn (7.6%)**; 1 product 27.7%; 3–4 products 83–100%
- Inactive members churn **26.9%** vs active 14.3%
- Churned customers took **24%** of the bank's total balance with them
- Top 1,000 customers by *churn probability × balance* hold **~51%** of all churned balance
