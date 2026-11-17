-- =====================================================
-- 04_churn_analysis.sql
-- Business questions answered with SQL
-- (results I got are written in the comments)
-- =====================================================

-- Q1. Overall churn rate
-- Result: 2037 of 10000 customers left = 20.37%
SELECT
    COUNT(*)                                  AS total_customers,
    SUM(churn)                                AS churned_customers,
    ROUND(100.0 * AVG(churn), 2)              AS churn_rate_pct
FROM customers;

-- Q2. Churn rate by country
-- Result: Germany ~32% is double France and Spain (~16%)
SELECT
    country,
    COUNT(*)                         AS customers,
    SUM(churn)                       AS churned,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct
FROM customers
GROUP BY country
ORDER BY churn_rate_pct DESC;

-- Q3. Churn rate by age group
-- Result: churn peaks in the 50-59 group (~56%), young customers rarely leave
SELECT
    CASE
        WHEN age < 30 THEN '18-29'
        WHEN age < 40 THEN '30-39'
        WHEN age < 50 THEN '40-49'
        WHEN age < 60 THEN '50-59'
        ELSE '60+'
    END                              AS age_group,
    COUNT(*)                         AS customers,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct
FROM customers
GROUP BY age_group
ORDER BY age_group;

-- Q4. Churn by number of products
-- Result: 2 products = lowest churn (~8%); 3 products ~83%; 4 products = 100%
SELECT
    products_number,
    COUNT(*)                         AS customers,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct
FROM customers
GROUP BY products_number
ORDER BY products_number;

-- Q5. Active vs inactive members
-- Result: inactive members churn almost 2x more (~27% vs ~14%)
SELECT
    CASE WHEN active_member = 1 THEN 'Active' ELSE 'Inactive' END AS member_status,
    COUNT(*)                         AS customers,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct
FROM customers
GROUP BY member_status;

-- Q6. Gender
-- Result: female ~25% vs male ~16%
SELECT
    gender,
    COUNT(*)                         AS customers,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct
FROM customers
GROUP BY gender;

-- Q7. Zero balance vs customers with money in the account
-- Result: customers WITH balance churn more (~24% vs ~14%)
SELECT
    CASE WHEN balance = 0 THEN 'Zero balance' ELSE 'Has balance' END AS balance_type,
    COUNT(*)                         AS customers,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct,
    ROUND(AVG(balance), 0)           AS avg_balance
FROM customers
GROUP BY balance_type;

-- Q8. How much money (balance) left with churned customers?
-- Result: churned customers held ~185.6M of ~764.9M total balance = 24.26%
-- (dataset does not say the currency, so I just call it 'balance')
SELECT
    ROUND(SUM(balance) FILTER (WHERE churn = 1), 0)                     AS balance_lost,
    ROUND(SUM(balance), 0)                                               AS total_balance,
    ROUND(100.0 * SUM(balance) FILTER (WHERE churn = 1) / SUM(balance), 2) AS pct_of_total_balance
FROM customers;

-- Q9. Highest-risk segment: country + activity together
-- Result: inactive members in Germany churn the most (~41%)
SELECT
    country,
    CASE WHEN active_member = 1 THEN 'Active' ELSE 'Inactive' END AS member_status,
    COUNT(*)                         AS customers,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct
FROM customers
GROUP BY country, member_status
ORDER BY churn_rate_pct DESC;
