"""GenAI layer inside the data pipeline: natural-language question -> safe SQL (DuckDB over Parquet)
-> result -> business summary. Works with OpenAI or Google Gemini; falls back to an offline
demo mode when no API key is set so the project always runs.

    LLM_PROVIDER=openai   OPENAI_API_KEY=...
    LLM_PROVIDER=gemini   GOOGLE_API_KEY=...
"""
import os
import re
from pathlib import Path

import duckdb

PROC = Path("data/processed")
SCHEMA = """
Tables (DuckDB views):
  features(customer_id, frequency, monetary, avg_order_value, avg_items, n_products, country, recency_days,
           tenure_days, avg_gap_days, is_uk, n_returns, return_rate, future_spend, churned)   -- one row per customer
  sales(invoice, stock_code, description, quantity, order_date, price, customer_id, country, amount, year_month)
"""
SQL_PROMPT = (
    "You write one DuckDB SELECT query. Return only SQL, no markdown.\n"
    "Rules: (1) when grouping by country or any category, ALWAYS include COUNT(*) AS customers and add "
    "HAVING COUNT(*) >= 30 so tiny groups do not distort rates; (2) for 'highest' or 'top' questions return "
    "the top 5 rows, not just 1; (3) rates of churned should be ROUND(AVG(churned)*100,1) AS churn_pct.\n"
    "{schema}\nQuestion: {q}"
)
SUM_PROMPT = (
    "Question: {q}\nSQL result (first rows):\n{rows}\n"
    "Give a 2-3 sentence business insight. Use only these numbers. Mention the group sizes. "
    "If a group has fewer than 30 customers, say the figure is unreliable instead of recommending action."
)
BLOCKED = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma|export|install|load)\b", re.I)


def get_con():
    con = duckdb.connect()
    con.execute(f"CREATE VIEW features AS SELECT * FROM '{PROC}/features.parquet'")
    con.execute(f"CREATE VIEW sales AS SELECT * FROM read_parquet('{PROC}/sales/**/*.parquet', hive_partitioning=true)")
    return con


_MODELS = []


def _gemini_models(client) -> list:
    """Model names change often, so ask the API which Flash models this key can use (best first, cached)."""
    if not _MODELS:
        skip = ("lite", "tts", "image", "live", "audio", "embed", "transcribe", "robotics", "omni", "thinking")
        names = [m.name.removeprefix("models/") for m in client.models.list()
                 if "flash" in m.name.lower() and "generateContent" in (m.supported_actions or [])
                 and not any(w in m.name.lower() for w in skip)]
        if not names:
            raise SystemExit("No Gemini Flash model available for this key. Set GEMINI_MODEL manually.")
        _MODELS.extend(sorted(names, reverse=True))
        print(f"[gemini] candidate models: {_MODELS[:3]}")
    return _MODELS


def _gemini_call(prompt: str) -> str:
    """Try each model; retry temporary errors (503 overloaded, 429 rate limit) with growing waits."""
    import time
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    models = [os.environ["GEMINI_MODEL"]] if os.getenv("GEMINI_MODEL") else _gemini_models(client)[:3]
    last = None
    for model in models:
        for attempt in range(3):
            try:
                text = client.models.generate_content(model=model, contents=prompt).text
                print(f"[gemini] answered by {model}")
                return text
            except Exception as e:  # network, quota or overload
                last = e
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                if code in (429, 500, 503) and attempt < 2:
                    time.sleep(3 * (attempt + 1))  # 3s, 6s, then move to the next model
                    continue
                break
    print(f"[gemini] unavailable ({type(last).__name__}: {str(last)[:120]}). Falling back to offline mode.")
    return ""


def llm(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "offline")
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                             messages=[{"role": "user", "content": prompt}], temperature=0)
        return r.choices[0].message.content
    if provider == "gemini":
        return _gemini_call(prompt)
    return ""  # offline


def validate_sql(sql: str) -> str:
    """Guardrail: single SELECT only, no DDL/DML, row cap enforced."""
    sql = sql.strip().strip("`").removeprefix("sql").strip().rstrip(";")
    if not re.match(r"^(select|with)\b", sql, re.I) or ";" in sql or BLOCKED.search(sql):
        raise ValueError(f"Rejected unsafe SQL: {sql[:80]}")
    return f"SELECT * FROM ({sql}) LIMIT 100"


def offline_sql(q: str) -> str:
    ql = q.lower()
    if "churn" in ql:
        return "SELECT country, ROUND(AVG(churned)*100,1) AS churn_pct, COUNT(*) AS customers FROM features GROUP BY 1 HAVING COUNT(*) >= 30 ORDER BY 2 DESC"
    if "product" in ql or "description" in ql:
        return "SELECT description, ROUND(SUM(amount)) AS revenue FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
    if "month" in ql or "revenue" in ql:
        return "SELECT year_month, ROUND(SUM(amount)) AS revenue FROM sales GROUP BY 1 ORDER BY 1"
    return "SELECT country, COUNT(*) AS customers, ROUND(AVG(monetary)) AS avg_spend FROM features GROUP BY 1 HAVING COUNT(*) >= 30 ORDER BY 3 DESC"


def ask(question: str) -> dict:
    con = get_con()
    raw = llm(SQL_PROMPT.format(schema=SCHEMA, q=question)) or offline_sql(question)
    sql = validate_sql(raw)
    df = con.execute(sql).df()
    summary = llm(SUM_PROMPT.format(q=question, rows=df.head(10).to_string(index=False))) \
        or f"(offline mode) {len(df)} rows returned; set LLM_PROVIDER for an AI-written insight."
    return {"sql": sql, "data": df, "insight": summary}


if __name__ == "__main__":
    import sys
    out = ask(" ".join(sys.argv[1:]) or "Which country has the highest churn?")
    print(out["sql"], "\n", out["data"].to_string(index=False), "\n", out["insight"])
