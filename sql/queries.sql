-- Run in DuckDB or Athena. Window functions + CTEs on the real Online Retail II data.

-- 1. Monthly revenue with month-over-month growth
WITH m AS (
  SELECT year_month, SUM(amount) AS revenue FROM sales GROUP BY 1
)
SELECT year_month, ROUND(revenue) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY year_month))
             / LAG(revenue) OVER (ORDER BY year_month), 1) AS mom_pct
FROM m ORDER BY 1;

-- 2. Top 3 customers per country by lifetime spend (countries with 30+ customers)
SELECT * FROM (
  SELECT country, customer_id, ROUND(monetary) AS spend,
         ROW_NUMBER() OVER (PARTITION BY country ORDER BY monetary DESC) AS rk
  FROM features
  QUALIFY COUNT(*) OVER (PARTITION BY country) >= 30
) WHERE rk <= 3 ORDER BY country, rk;

-- 3. Churn rate by country
SELECT country, ROUND(AVG(churned) * 100, 1) AS churn_pct, COUNT(*) AS customers
FROM features GROUP BY 1 HAVING COUNT(*) >= 30 ORDER BY 2 DESC;

-- 4. Customers by number of orders (frequency buckets) and their churn
SELECT CASE WHEN frequency = 1 THEN '1 order' WHEN frequency <= 3 THEN '2-3'
            WHEN frequency <= 9 THEN '4-9' ELSE '10+' END AS bucket,
       COUNT(*) AS customers, ROUND(AVG(churned) * 100, 1) AS churn_pct
FROM features GROUP BY 1 ORDER BY 2 DESC;
