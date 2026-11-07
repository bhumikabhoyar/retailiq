"""Streamlit dashboard: run with `streamlit run app.py`."""
import json
from pathlib import Path

import streamlit as st

from src.genai_insights import ask

st.title("RetailIQ - Ask your data")
m = Path("outputs/metrics.json")
if m.exists():
    st.subheader("Model metrics")
    st.json(json.loads(m.read_text()))

q = st.text_input("Ask a question", "Which country has the highest churn?")
if st.button("Ask") and q:
    r = ask(q)
    st.code(r["sql"], language="sql")
    st.dataframe(r["data"])
    st.info(r["insight"])
