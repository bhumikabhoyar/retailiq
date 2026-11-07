-- Run in the Athena console (Query editor). Replace YOUR-BUCKET with the bucket name from the stack outputs.

CREATE DATABASE IF NOT EXISTS retailiq;

CREATE EXTERNAL TABLE IF NOT EXISTS retailiq.sales (
  invoice      string,
  stock_code   string,
  description  string,
  quantity     bigint,
  order_date   timestamp,
  price        double,
  customer_id  double,
  country      string,
  amount       double
)
PARTITIONED BY (year_month string)
STORED AS PARQUET
LOCATION 's3://YOUR-BUCKET/processed/sales/';

-- Tell Athena about the year_month=... folders (run again after new data arrives)
MSCK REPAIR TABLE retailiq.sales;

-- Test 1: row count (should be about 1,003,214 minus a few cross-file effects; see DEPLOY.md)
SELECT COUNT(*) FROM retailiq.sales;

-- Test 2: monthly revenue with month-over-month growth
WITH m AS (SELECT year_month, SUM(amount) AS revenue FROM retailiq.sales GROUP BY 1)
SELECT year_month, ROUND(revenue) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY year_month))
             / LAG(revenue) OVER (ORDER BY year_month), 1) AS mom_pct
FROM m ORDER BY 1;

-- Test 3: top 10 countries by revenue
SELECT country, ROUND(SUM(amount)) AS revenue, COUNT(DISTINCT customer_id) AS customers
FROM retailiq.sales GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
