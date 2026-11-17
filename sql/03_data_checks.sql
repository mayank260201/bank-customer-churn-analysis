-- =====================================================
-- 03_data_checks.sql
-- Check that the data loaded correctly before doing analysis
-- =====================================================

-- 1. Row counts (both should be 10000)
SELECT 'customers' AS table_name, COUNT(*) AS total_rows FROM customers
UNION ALL
SELECT 'customers_scored', COUNT(*) FROM customers_scored;

-- 2. Any missing values in important columns? (should be 0)
SELECT
    COUNT(*) FILTER (WHERE age IS NULL)      AS missing_age,
    COUNT(*) FILTER (WHERE balance IS NULL)  AS missing_balance,
    COUNT(*) FILTER (WHERE country IS NULL)  AS missing_country,
    COUNT(*) FILTER (WHERE churn IS NULL)    AS missing_churn
FROM customers;

-- 3. Duplicate customer ids (should return no rows)
SELECT customer_id, COUNT(*)
FROM customers
GROUP BY customer_id
HAVING COUNT(*) > 1;

-- 4. Both tables have the same customers and same churn label (should be 0)
SELECT COUNT(*) AS mismatches
FROM customers c
JOIN customers_scored s ON c.customer_id = s.customer_id
WHERE c.churn <> s.churn;
