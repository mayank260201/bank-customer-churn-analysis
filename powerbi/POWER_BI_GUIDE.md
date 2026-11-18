# Power BI Dashboard — Step by Step

Goal: build `powerbi/churn_dashboard.pbix` with **2 report pages**:
1. **Churn Overview** – who is leaving
2. **Risk & Retention** – who to call next (uses the ML model scores)

Time: ~45 minutes.
Data: `data/processed/customers_scored.csv` (made by `notebooks/05_model_export.ipynb`).

> Names in this guide match **Power BI Desktop as of November 2026**. Older YouTube videos may use old names. The main renames:
>
> | Old name (in old tutorials) | Current name |
> |---|---|
> | Data view | **Table view** (left side, table icon) |
> | Fields pane | **Data pane** (right side) |
> | Card / Multi-row card | **Card** (the "new" card; it can show several measures at once) |
> | Text slicer | **Input slicer** |
> | Modeling → Sort by column | **Column tools → Sort by column** |
> | Analytics pane | Inside the **Format visual** pane |

---

## Step 1 – Load the data and set the column types

1. Open **Power BI Desktop** → **Blank report**.
2. **Home** tab → **Get data** → **Text/CSV** → select `data/processed/customers_scored.csv` → **Open**.
   *(Newer versions may open a new "Get data" window. Type **Text/CSV** in its search box.)*
3. In the preview window click **Transform Data** (not Load). The **Power Query Editor** opens.
4. Check the type icon at the left of each column header (`123` = whole number, `1.2` = decimal, `ABC` = text). Click an icon to change it:

   | Column | Must be |
   |---|---|
   | `customer_id`, `credit_score`, `age`, `tenure`, `products_number`, `credit_card`, `active_member`, `churn`, `predicted_churn` | **Whole Number** (`123`) |
   | `balance`, `estimated_salary`, `churn_probability` | **Decimal Number** (`1.2`) |
   | `country`, `gender`, `risk_band`, `age_group`, `balance_group`, `credit_score_group` | **Text** (`ABC`) |

   ⚠️ `balance_group` is the one to double-check. Its first value is `0`, so Power BI sometimes guesses "number". That turns `1-50K` etc. into **Error**. If it isn't `ABC`, change it to **Text**. If a "Change Column Type" box appears, click **Replace current**.

Keep the Power Query Editor open for Step 2.

---

## Step 2 – Add helper columns in Power Query (this fixes the "circular dependency" error)

**Why:** charts show `age_group` alphabetically by default, so "60+" lands in the wrong place. To fix that you give each group an order number and use **Sort by column**.
If you make the order number with DAX **New column** (`SWITCH(customers_scored[age_group], ...)`), Power BI shows:

> *A circular dependency was detected*

That happens because the DAX column reads `age_group`, and sorting `age_group` by it makes `age_group` depend on the DAX column too. Each depends on the other.
**The fix is to create the order columns in Power Query instead.** They are then normal data columns, so there is no loop.

In the **Power Query Editor**, do this 4 times (one per row of the table below):

1. **Add Column** tab → **Custom Column**.
2. **New column name:** the name from the table.
3. **Custom column formula:** delete what's there and paste the formula from the table.
4. Click **OK**. Check the bottom-left of the dialog says "No syntax errors have been detected".
5. Set the new column's type: click the `ABC123` icon in its header and pick the type from the table. (If the Custom Column window has a **Data type** box, you can pick it there instead.)

| New column name | Formula to paste | Type |
|---|---|---|
| `age_order` | `if [age_group] = "18-29" then 1 else if [age_group] = "30-39" then 2 else if [age_group] = "40-49" then 3 else if [age_group] = "50-59" then 4 else 5` | Whole Number |
| `balance_order` | `if Text.From([balance_group]) = "0" then 1 else if [balance_group] = "1-50K" then 2 else if [balance_group] = "50K-100K" then 3 else if [balance_group] = "100K-150K" then 4 else 5` | Whole Number |
| `risk_order` | `if [risk_band] = "High" then 1 else if [risk_band] = "Medium" then 2 else 3` | Whole Number |
| `member_status` | `if [active_member] = 1 then "Active" else "Inactive"` | Text |

(`member_status` isn't for sorting. It just shows "Active / Inactive" in charts instead of 1 / 0.)

6. **Home** tab → **Close & Apply**.

---

## Step 3 – Sort the group columns (Sort by column)

1. On the left edge of Power BI, click the **Table view** icon (the small table).
2. In the **Data pane** on the right, expand `customers_scored` and click the column **you want to sort**: `age_group`. (Click the text column, not the order column.)
3. The **Column tools** tab appears at the top → click **Sort by column** → choose `age_order`.
4. Repeat:
   - click `balance_group` → **Sort by column** → `balance_order`
   - click `risk_band` → **Sort by column** → `risk_order`
5. Hide the helper columns so they don't clutter the report: in the Data pane, right-click `age_order` → **Hide in report view**. Do the same for `balance_order` and `risk_order`.

You won't see anything change in Table view. The effect shows up in the charts.

---

## Step 4 – Stop Power BI from adding up ID-type numbers

Power BI adds up every number column by default (Σ icon). That's wrong for IDs and 0/1 flags. For example, a table visual would show the **sum** of `customer_id`.

Still in **Table view**, for each of these columns: click the column in the Data pane → **Column tools** → **Summarization** → **Don't summarize**:

`customer_id`, `credit_score`, `age`, `tenure`, `balance`, `estimated_salary`, `products_number`, `credit_card`, `active_member`, `churn`, `predicted_churn`, `churn_probability`

Also set `customer_id` → **Column tools** → **Format** → **Whole number**, with **thousands separator off** (`,` button not highlighted), so IDs don't show as "15,634,602".

The measures in Step 6 handle all the counting and totals.

---

## Step 5 – Apply the theme

**View** tab → **Themes** dropdown (arrow next to the theme gallery) → **Browse for themes** → select `powerbi/churn_theme.json`.
You should see "Import theme successful".

*(From the August 2026 update, themes also have their own **Theme pane**. If you don't see "Browse for themes", open the Theme pane and use **Import** there.)*

---

## Step 6 – Create the measures

Follow `powerbi/DAX_measures.md` (10 measures, copy-paste).

---

## Step 7 – Page 1: "Churn Overview"

Go back to **Report view** (top icon on the left). Right-click the page tab at the bottom → **Rename page** → `Churn Overview`.

To add a visual: click an empty part of the page → in the **Visualizations** pane pick the visual icon → drag fields from the **Data pane** into the visual's field boxes (**X-axis**, **Y-axis**, **Legend**, **Values**…), shown under the visual icons.

| # | Visual (Visualizations pane) | Put these fields | Extra setting |
|---|---|---|---|
| 1 | **Card** | Drag `Total Customers`, `Churned Customers`, `Churn Rate`, `% Balance Lost` into the card's field box | One Card visual can hold all 4 measures. Make it wide, across the top |
| 2 | **Clustered column chart** – Churn rate by country | X-axis: `country` · Y-axis: `Churn Rate` | |
| 3 | **Clustered column chart** – Churn rate by age group | X-axis: `age_group` · Y-axis: `Churn Rate` | Sort fix below ⬇ |
| 4 | **Clustered column chart** – Churn rate by products | X-axis: `products_number` · Y-axis: `Churn Rate` | Sort fix below ⬇ |
| 5 | **Donut chart** – Who are the churners? | Legend: `member_status` · Values: `Churned Customers` | |
| 6 | **Slicer** | Field: `country` | Format visual → Visual → Slicer settings → Style: **Dropdown** or **Tile** |
| 7 | **Slicer** | Field: `gender` | Same |

**Make the charts follow your sort order:** by default a column chart sorts by the **value** (tallest bar first), not by age group.
For charts 3 and 4: hover the chart → **More options (…)** at the top-right corner → **Sort axis** → pick the axis field (`age_group` / `products_number`) → open **More options (…)** → **Sort axis** again → **Sort ascending**.

**Data labels:** select a chart → **Format visual** (paintbrush icon) → **Visual** → turn **Data labels** **On**.

---

## Step 8 – Page 2: "Risk & Retention"

Click **+** at the bottom to add a page → rename it `Risk & Retention`.

| # | Visual | Put these fields | Extra setting |
|---|---|---|---|
| 1 | **Card** | Drag `High Risk Customers`, `Balance at Risk`, `Churners Caught %` into the card's field box | |
| 2 | **Clustered column chart** – Risk band vs actual churn | X-axis: `risk_band` · Y-axis: `Churn Rate` | More options (…) → Sort axis → `risk_band` → Sort ascending |
| 3 | **Stacked bar chart** – Balance by risk band | Y-axis: `risk_band` · X-axis: `Total Balance` | |
| 4 | **Matrix** – Country × risk band | Rows: `country` · Columns: `risk_band` · Values: `Total Customers` | |
| 5 | **Table** – Retention call list | Columns: `customer_id`, `country`, `age`, `balance`, `products_number`, `member_status`, `churn_probability` | Filter below ⬇ |
| 6 | **Slicer** | Field: `risk_band` | |

**Filter the call list to High risk only:** select the Table visual → open the **Filters** pane (right side) → under **Filters on this visual**, drag `risk_band` in (or find it there) → **Filter type: Basic filtering** → tick **High**.
**Sort it by balance:** click the `balance` column header in the table (click again to switch to largest first).

If the table shows only a few rows, or numbers look added up, go back to **Step 4** (Don't summarize).

---

## Step 9 – Check the numbers

These should match the SQL results and the Excel workbook:

| Check | Expected |
|---|---|
| Total Customers | 10,000 |
| Churn Rate | 20.4% |
| Germany churn rate | 32.4% |
| Age 50-59 churn rate | 56.0% |
| 2 products churn rate | 7.6% |
| % Balance Lost | 24.3% |
| High Risk Customers | 955 |
| Churn rate of the "High" bar (page 2) | 83.8% |
| Churners Caught % | 66.7% |

---

## Step 10 – Save & export

- **File** → **Save as** → `powerbi/churn_dashboard.pbix`
  *(Power BI Desktop now stores the report inside the .pbix in the new "PBIR" format. Nothing changes for you; it's still one .pbix file.)*
- **File** → **Export** → **Export to PDF** → `powerbi/churn_dashboard.pdf`
- Screenshot each page (Windows: **Win + Shift + S**) → save as `powerbi/screenshots/overview.png` and `powerbi/screenshots/risk_retention.png`

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| **"A circular dependency was detected"** when using Sort by column | The order column was made with DAX **New column** reading the same column | Delete the DAX column (right-click → Delete from model) and use the Power Query columns from **Step 2** |
| Sort by column says the order column has **more than one value** for the same group | The order formula gives two numbers to one group | Re-check the Step 2 formula. Each group must get exactly one number |
| `balance_group` shows **Error** in Power Query | Column typed as number | Step 1: set it to **Text** |
| Bars are in the wrong order | Chart sorts by value | Step 7: **More options (…) → Sort axis** |
| Table visual shows few rows / summed IDs | Summarization still on | Step 4: **Don't summarize** |
| Measures show **(Blank)** | Measure created in a different table, or typo in a column name | Recreate it with `customers_scored` selected in the Data pane |
| Percentages show 0.20 instead of 20% | Measure not formatted | Select the measure → **Measure tools** → **%** button |
| "Couldn't find file" after moving the project folder | Saved file path changed | **Home** → **Transform data** dropdown → **Data source settings** → **Change Source…** → pick the CSV again |
| A pane or visual is missing | Pane closed, or visuals list reset | Reopen panes from the small pane tabs on the right edge of the window. For a missing visual icon: **More options (…)** in the Visualizations pane → **Restore default visuals** |
| Decimals read wrongly (e.g. balance 100× too big) | Locale uses comma as decimal separator | **File → Options and settings → Options → Current File → Regional Settings** → Locale for import: **English (United States)** → Home → **Refresh** |
