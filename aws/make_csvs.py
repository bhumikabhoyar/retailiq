"""Run LOCALLY before uploading: turn your Excel file(s) into CSVs, one per calendar year.

    python aws/make_csvs.py

Output: data/csv/online_retail_2009.csv, ..._2010.csv, ..._2011.csv
Why: Lambda reads CSV quickly (Excel is far too slow), and splitting by year gives 3 files that
each trigger one Lambda run. Exact duplicate rows (the 1-9 Dec 2010 overlap between the two
sheets) are removed here, so no duplicates reach S3.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import load_raw  # reuses the same reader and cache as the local pipeline


def main():
    out = ROOT / "data" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    df = load_raw().drop_duplicates()
    df["InvoiceDate"] = df["InvoiceDate"].astype("datetime64[ns]")
    for year, part in df.groupby(df.InvoiceDate.dt.year):
        path = out / f"online_retail_{year}.csv"
        part.to_csv(path, index=False)
        print(f"{path.name}: {len(part):,} rows, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
