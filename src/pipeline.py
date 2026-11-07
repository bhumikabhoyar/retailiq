"""ETL pipeline for the real Online Retail II dataset (UCI).

raw xlsx (2 sheets) -> one DataFrame -> data-quality report -> cleaned sales (Parquet, partitioned by month)
-> leakage-safe customer feature table with a churn label.

Local by default. Set S3_BUCKET to also publish the processed files to S3.
"""
import json
import os
import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
CACHE = RAW_DIR / "online_retail_cache.parquet"  # reading 1M rows from xlsx is slow, so cache once
PROC = Path("data/processed")
CHURN_WINDOW_DAYS = 90

RENAME = {
    "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "order_date", "Price": "price",
    "Customer ID": "customer_id", "Country": "country",
}


def load_raw() -> pd.DataFrame:
    """Read BOTH sheets (2009-2010 and 2010-2011) and stack them into one table."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files found in {RAW_DIR}. Put your Online Retail II file(s) there.")
    frames = []
    for f in files:  # every file, every sheet (works for 1 file with 2 sheets, or 2 files with 1 sheet each)
        print("reading", f.name)
        frames += list(pd.read_excel(f, sheet_name=None).values())
    df = pd.concat(frames, ignore_index=True)
    # these columns mix numbers and text in Excel (e.g. 85123A vs 71053, C489449), so force text
    for c in ["Invoice", "StockCode"]:
        df[c] = df[c].astype(str)
    df["Description"] = df["Description"].astype("string")
    df.to_parquet(CACHE, index=False)
    return df


def clean(df: pd.DataFrame):
    """Return (sales, returns, data_quality_report)."""
    df = df.rename(columns=RENAME)
    report = {"rows_in": int(len(df))}

    # 1. exact duplicate rows. This also removes the overlap between the two sheets (1-9 Dec 2010)
    report["exact_duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 2. cancellations: invoice numbers starting with "C" are returns/credit notes
    is_cancel = df.invoice.str.startswith("C")
    report["cancellation_rows"] = int(is_cancel.sum())
    returns = df[is_cancel].copy()
    df = df[~is_cancel]

    # 3. non-positive quantity or price (damaged stock write-offs, price adjustments)
    bad = (df.quantity <= 0) | (df.price <= 0)
    report["non_positive_qty_or_price"] = int(bad.sum())
    df = df[~bad]

    # 4. non-product codes (POST, D, M, BANK CHARGES, AMAZONFEE ...). Real products start with 5 digits
    non_product = ~df.stock_code.str.match(r"^\d{5}")
    report["non_product_rows"] = int(non_product.sum())
    df = df[~non_product]

    report["missing_customer_id_rows"] = int(df.customer_id.isna().sum())  # kept for revenue, not for customers
    sales = df.copy()
    sales["order_date"] = pd.to_datetime(sales.order_date)
    sales["amount"] = (sales.quantity * sales.price).round(2)
    returns["order_date"] = pd.to_datetime(returns.order_date)
    returns["amount"] = (returns.quantity.abs() * returns.price).round(2)
    report["rows_out_sales"] = int(len(sales))
    report["date_range"] = [str(sales.order_date.min()), str(sales.order_date.max())]
    return sales, returns, report


def build_features(sales: pd.DataFrame, returns: pd.DataFrame, window_days: int = CHURN_WINDOW_DAYS) -> pd.DataFrame:
    """Leakage-safe churn dataset.

    cutoff = last date in data minus 90 days.
      features  -> built ONLY from purchases BEFORE the cutoff
      label     -> churned = 1 if the customer made NO purchase in the 90 days AFTER the cutoff
    """
    end = sales.order_date.max()
    cutoff = end - pd.Timedelta(days=window_days)
    known = sales[sales.customer_id.notna()]
    hist, future = known[known.order_date < cutoff], known[known.order_date >= cutoff]

    inv = hist.groupby(["customer_id", "invoice"]).agg(
        value=("amount", "sum"), items=("quantity", "sum"), date=("order_date", "min")).reset_index()
    f = inv.groupby("customer_id").agg(
        frequency=("invoice", "count"), monetary=("value", "sum"), avg_order_value=("value", "mean"),
        avg_items=("items", "mean"), first_order=("date", "min"), last_order=("date", "max"))
    f["n_products"] = hist.groupby("customer_id").stock_code.nunique()
    f["country"] = hist.groupby("customer_id").country.agg(lambda s: s.mode().iat[0])
    f["recency_days"] = (cutoff - f.last_order).dt.days
    f["tenure_days"] = (cutoff - f.first_order).dt.days
    f["avg_gap_days"] = (f.tenure_days / (f.frequency - 1).clip(lower=1))
    f["is_uk"] = (f.country == "United Kingdom").astype(int)

    r = returns[(returns.customer_id.notna()) & (returns.order_date < cutoff)]
    f["n_returns"] = r.groupby("customer_id").invoice.nunique()
    f["n_returns"] = f.n_returns.fillna(0)
    f["return_rate"] = f.n_returns / f.frequency

    f["future_spend"] = future.groupby("customer_id").amount.sum()
    f["future_spend"] = f.future_spend.fillna(0)
    f["churned"] = (f.future_spend == 0).astype(int)
    return f.drop(columns=["first_order", "last_order"]).reset_index()


def publish_s3(bucket: str):
    import boto3  # lazy import so local runs don't need AWS
    s3 = boto3.client("s3")
    for p in PROC.rglob("*"):
        if p.is_file():
            s3.upload_file(str(p), bucket, f"retailiq/processed/{p.relative_to(PROC)}")
    print(f"published to s3://{bucket}/retailiq/processed/")


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    sales, returns, report = clean(raw)

    sales["year_month"] = sales.order_date.dt.strftime("%Y-%m")
    shutil.rmtree(PROC / "sales", ignore_errors=True)  # otherwise a re-run APPENDS and doubles the data
    sales.to_parquet(PROC / "sales", partition_cols=["year_month"], index=False)  # Glue/Athena-style layout
    feats = build_features(sales, returns)
    feats.to_parquet(PROC / "features.parquet", index=False)

    report.update({"customers_modelled": int(len(feats)), "churn_rate_pct": round(float(feats.churned.mean() * 100), 1)})
    (PROC / "dq_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if os.getenv("S3_BUCKET"):
        publish_s3(os.environ["S3_BUCKET"])


if __name__ == "__main__":
    main()
