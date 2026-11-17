-- =====================================================
-- 01_create_tables.sql
-- Bank Customer Churn Analysis - PostgreSQL
-- Run this first (in pgAdmin4 -> Query Tool) on database: bank_churn
-- =====================================================

-- drop old tables if I am re-running the script
DROP TABLE IF EXISTS customers_scored;
DROP TABLE IF EXISTS customers;

-- raw data (same columns as the Kaggle CSV)
CREATE TABLE customers (
    customer_id       INTEGER PRIMARY KEY,
    credit_score      INTEGER,
    country           VARCHAR(20),
    gender            VARCHAR(10),
    age               INTEGER,
    tenure            INTEGER,
    balance           NUMERIC(12,2),
    products_number   INTEGER,
    credit_card       INTEGER,      -- 1 = has credit card
    active_member     INTEGER,      -- 1 = active
    estimated_salary  NUMERIC(12,2),
    churn             INTEGER       -- 1 = left the bank
);

-- scored data exported from notebooks/05_model_export.ipynb
CREATE TABLE customers_scored (
    customer_id        INTEGER PRIMARY KEY,
    credit_score       INTEGER,
    country            VARCHAR(20),
    gender             VARCHAR(10),
    age                INTEGER,
    tenure             INTEGER,
    balance            NUMERIC(12,2),
    products_number    INTEGER,
    credit_card        INTEGER,
    active_member      INTEGER,
    estimated_salary   NUMERIC(12,2),
    churn              INTEGER,
    churn_probability  NUMERIC(6,4),
    predicted_churn    INTEGER,
    risk_band          VARCHAR(10),   -- High / Medium / Low
    age_group          VARCHAR(10),
    balance_group      VARCHAR(15),
    credit_score_group VARCHAR(25)
);
