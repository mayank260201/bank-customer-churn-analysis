-- =====================================================
-- 05_model_results.sql
-- Using the model scores (customers_scored table)
-- =====================================================

-- Q1. How good are the risk bands? (actual churn inside each band)
-- Result: High ~84% really churned, Medium ~40%, Low ~9%
SELECT
    risk_band,
    COUNT(*)                         AS customers,
    SUM(churn)                       AS actually_churned,
    ROUND(100.0 * AVG(churn), 2)     AS actual_churn_rate_pct,
    ROUND(AVG(churn_probability), 3) AS avg_predicted_probability
FROM customers_scored
GROUP BY risk_band
ORDER BY avg_predicted_probability DESC;

-- Q2. Confusion matrix at threshold 0.3 (all 10000 customers, cross-validated scores)
-- Result: TP 1358, FP 987, FN 679, TN 6976 -> catches 1358 of 2037 churners (~67%)
SELECT
    SUM(CASE WHEN predicted_churn = 1 AND churn = 1 THEN 1 ELSE 0 END) AS true_positive,
    SUM(CASE WHEN predicted_churn = 1 AND churn = 0 THEN 1 ELSE 0 END) AS false_positive,
    SUM(CASE WHEN predicted_churn = 0 AND churn = 1 THEN 1 ELSE 0 END) AS false_negative,
    SUM(CASE WHEN predicted_churn = 0 AND churn = 0 THEN 1 ELSE 0 END) AS true_negative
FROM customers_scored;

-- Q3. Balance at risk by risk band
-- Result: High + Medium customers hold ~220M of balance
SELECT
    risk_band,
    COUNT(*)                   AS customers,
    ROUND(SUM(balance), 0)     AS total_balance,
    ROUND(AVG(balance), 0)     AS avg_balance
FROM customers_scored
GROUP BY risk_band
ORDER BY total_balance DESC;

-- Q4. Retention call list: top 50 high-risk customers with the biggest balance
-- (this is what the retention team would actually use)
SELECT
    customer_id, country, age, balance, products_number, active_member,
    churn_probability, risk_band
FROM customers_scored
WHERE risk_band = 'High'
ORDER BY balance DESC
LIMIT 50;

-- Q5. Save the call list as a view so Power BI / Excel can read it directly
CREATE OR REPLACE VIEW retention_call_list AS
SELECT
    customer_id, country, gender, age, balance, products_number,
    active_member, churn_probability, risk_band
FROM customers_scored
WHERE risk_band IN ('High', 'Medium')
ORDER BY churn_probability DESC, balance DESC;

-- Result: 2345 customers in the call list
SELECT COUNT(*) AS customers_in_call_list FROM retention_call_list;

-- Q6. Priority score = churn probability x balance (expected balance we could lose)
-- Idea: the retention team can only call ~1000 people a month, so who first?
-- Result: top 1000 by priority_score contain 731 real churners holding ~51% of all churned balance
--         (top 1000 by probability only: 825 churners but ~41% of churned balance)
SELECT
    customer_id, country, age, balance, churn_probability,
    ROUND(churn_probability * balance, 0) AS priority_score
FROM customers_scored
ORDER BY priority_score DESC
LIMIT 1000;
