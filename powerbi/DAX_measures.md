# DAX Measures

**Where:** in **Report view** or **Table view**, first click the table `customers_scored` in the **Data pane** (right side). Then **Home** tab → **New measure**.
A formula bar opens at the top. Delete what's there, paste **one** measure, press **Enter**. Repeat for each.

The new measure appears in the Data pane with a calculator icon.

> ⚠️ Don't create the sort-order columns (`Age Order`, `Balance Order`, `Risk Order`) here with DAX. That causes the **"circular dependency"** error. They are made in Power Query (POWER_BI_GUIDE.md, Step 2).

---

```dax
Total Customers = COUNTROWS(customers_scored)
```

```dax
Churned Customers = SUM(customers_scored[churn])
```

```dax
Churn Rate = DIVIDE([Churned Customers], [Total Customers])
```

```dax
Total Balance = SUM(customers_scored[balance])
```

```dax
Balance Lost = CALCULATE([Total Balance], customers_scored[churn] = 1)
```

```dax
% Balance Lost = DIVIDE([Balance Lost], [Total Balance])
```

```dax
High Risk Customers = CALCULATE([Total Customers], customers_scored[risk_band] = "High")
```

```dax
Balance at Risk = CALCULATE([Total Balance], customers_scored[risk_band] IN {"High", "Medium"})
```

```dax
Avg Churn Probability = AVERAGE(customers_scored[churn_probability])
```

```dax
Churners Caught % =
DIVIDE(
    CALCULATE([Total Customers], customers_scored[churn] = 1, customers_scored[predicted_churn] = 1),
    [Churned Customers]
)
```
(= recall of the model at threshold 0.3, should be about 66.7%)

---

## Formatting

Select the measure in the Data pane → the **Measure tools** tab appears at the top:

| Measure | Format |
|---|---|
| `Churn Rate`, `% Balance Lost`, `Avg Churn Probability`, `Churners Caught %` | Click **%**, set decimals to **1** |
| `Total Customers`, `Churned Customers`, `High Risk Customers` | Click **,** (thousands separator), decimals **0** |
| `Total Balance`, `Balance Lost`, `Balance at Risk` | Click **,**, decimals **0**. *(The dataset doesn't say the currency, so don't add a currency symbol.)* |

## Quick check

Put `Total Customers` in a Card visual → it should show **10,000** (or **10K** with auto display units).
If it shows **(Blank)** or an error, check that the column names match exactly (they are lowercase, e.g. `churn`, `balance`).
