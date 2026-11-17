-- =====================================================
-- 02_load_data.sql
-- Load the CSV files into the tables
--
-- OPTION 1 (easiest): pgAdmin4 -> right click table -> Import/Export Data
--   * Import, Format = csv, Header = ON, Delimiter = ,
--   * customers        <- data/raw/Bank Customer Churn Prediction.csv
--   * customers_scored <- data/processed/customers_scored.csv
--
-- OPTION 2: pgAdmin4 -> right click database -> PSQL Tool, then paste the
--   two \copy lines below. Change the path to where YOUR project folder is.
--   (use forward slashes / even on Windows)
-- =====================================================

\copy customers FROM 'C:/Users/Mayank/Desktop/Bank-Customer-Churn-Analysis/data/raw/Bank Customer Churn Prediction.csv' WITH (FORMAT csv, HEADER true);

\copy customers_scored FROM 'C:/Users/Mayank/Desktop/Bank-Customer-Churn-Analysis/data/processed/customers_scored.csv' WITH (FORMAT csv, HEADER true);
