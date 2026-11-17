# DAX Measures

Create these in Power BI: **Home → New measure** (table = `customers_scored`).
Copy one measure at a time.

```dax
Total Customers = COUNTROWS(customers_scored)
```

```dax
Churned Customers = SUM(customers_scored[churn])
```

```dax
Churn Rate = DIVIDE([Churned Customers], [Total Customers])
```
Format: Percentage, 1 decimal.

```dax
Total Balance = SUM(customers_scored[balance])
```

```dax
Balance Lost = CALCULATE(SUM(customers_scored[balance]), customers_scored[churn] = 1)
```

```dax
% Balance Lost = DIVIDE([Balance Lost], [Total Balance])
```
Format: Percentage, 1 decimal.

```dax
High Risk Customers = CALCULATE([Total Customers], customers_scored[risk_band] = "High")
```

```dax
Balance at Risk = CALCULATE([Total Balance], customers_scored[risk_band] IN {"High", "Medium"})
```

```dax
Avg Churn Probability = AVERAGE(customers_scored[churn_probability])
```
Format: Percentage, 1 decimal.

```dax
Churners Caught % = 
DIVIDE(
    CALCULATE([Total Customers], customers_scored[churn] = 1, customers_scored[predicted_churn] = 1),
    [Churned Customers]
)
```
(= recall of the model at threshold 0.3, should be about 67%)

## Sort columns (so charts are in the right order)

`age_group` and `balance_group` sort alphabetically by default, so "60+" and "150K+" end up in the wrong place.
Fix: **Data view → select column → Column tools → Sort by column**, using a helper column:

```dax
Age Order = SWITCH(customers_scored[age_group], "18-29",1, "30-39",2, "40-49",3, "50-59",4, "60+",5)
```

```dax
Balance Order = SWITCH(customers_scored[balance_group], "0",1, "1-50K",2, "50K-100K",3, "100K-150K",4, "150K+",5)
```

```dax
Risk Order = SWITCH(customers_scored[risk_band], "High",1, "Medium",2, "Low",3)
```
(these 3 are **New column**, not New measure)
