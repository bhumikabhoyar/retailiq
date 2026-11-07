"""AWS Lambda: runs when a raw CSV lands in s3://<bucket>/raw/.

It applies the SAME cleaning rules as src/pipeline.py and writes month-partitioned Parquet to
s3://<bucket>/processed/sales/year_month=YYYY-MM/<file>.parquet  (the layout Athena reads).

Needs pandas + pyarrow: provided by the AWS SDK for pandas managed layer (see template.yaml).
"""
import io
import json
import os
import urllib.parse

import boto3
import pandas as pd

s3 = boto3.client("s3")

RENAME = {
    "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "order_date", "Price": "price",
    "Customer ID": "customer_id", "Country": "country",
}


def clean(df: pd.DataFrame):
    """Same rules as src/pipeline.py: returns (sales, report)."""
    df = df.rename(columns=RENAME)
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    report = {"rows_in": int(len(df))}

    report["exact_duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    cancel = df.invoice.str.startswith("C")          # returns / credit notes
    report["cancellation_rows"] = int(cancel.sum())
    df = df[~cancel]

    bad = (df.quantity <= 0) | (df.price <= 0)        # write-offs, price adjustments
    report["non_positive_qty_or_price"] = int(bad.sum())
    df = df[~bad]

    non_product = ~df.stock_code.str.match(r"^\d{5}")  # POST, BANK CHARGES, fees...
    report["non_product_rows"] = int(non_product.sum())
    df = df[~non_product].copy()

    df["order_date"] = pd.to_datetime(df.order_date)
    df["amount"] = (df.quantity * df.price).round(2)
    report["rows_out"] = int(len(df))
    return df, report


def handler(event, context):
    rec = event["Records"][0]["s3"]
    bucket = rec["bucket"]["name"]
    key = urllib.parse.unquote_plus(rec["object"]["key"])
    if not key.startswith("raw/") or not key.endswith(".csv"):
        return {"skipped": key}

    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    df = pd.read_csv(io.BytesIO(body), dtype={"Invoice": str, "StockCode": str})
    sales, report = clean(df)

    name = os.path.splitext(os.path.basename(key))[0]
    sales["year_month"] = sales.order_date.dt.strftime("%Y-%m")
    written = []
    for ym, part in sales.groupby("year_month"):
        buf = io.BytesIO()
        # milliseconds: Athena cannot read nanosecond Parquet timestamps
        part.drop(columns="year_month").to_parquet(buf, index=False, coerce_timestamps="ms",
                                                    allow_truncated_timestamps=True)
        out_key = f"processed/sales/year_month={ym}/{name}.parquet"
        s3.put_object(Bucket=bucket, Key=out_key, Body=buf.getvalue())
        written.append(out_key)

    s3.put_object(Bucket=bucket, Key=f"processed/dq_reports/{name}.json",
                  Body=json.dumps(report, indent=2).encode())
    print(json.dumps({"input": key, **report, "files_written": len(written)}))
    return {"input": key, **report, "files_written": len(written)}
