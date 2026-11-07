"""Statistical + ML models covering the JD list:
Linear/Logistic Regression, Decision Tree, Bagging, Boosting, Time Series, Non-parametric tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROC, OUT = Path("data/processed"), Path("outputs")
SKEWED = ["frequency", "monetary", "avg_order_value", "avg_items", "n_products", "avg_gap_days", "tenure_days"]


def prep(feats: pd.DataFrame):
    """log1p on right-skewed money/count columns so linear models behave; trees don't mind."""
    X = feats.drop(columns=["customer_id", "country", "future_spend", "churned"]).copy()
    X[SKEWED] = np.log1p(X[SKEWED])
    return X


def churn_models(feats: pd.DataFrame) -> dict:
    X, y = prep(feats), feats.churned
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=100, random_state=42),
        "boosting": GradientBoostingClassifier(random_state=42),
    }
    out = {}
    for name, m in models.items():
        cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="roc_auc").mean()
        test = roc_auc_score(yte, m.fit(Xtr, ytr).predict_proba(Xte)[:, 1])
        out[name] = {"cv_auc": round(float(cv), 3), "test_auc": round(float(test), 3)}
    imp = pd.Series(models["boosting"].feature_importances_, index=X.columns).sort_values(ascending=False)
    out["boosting_top_features"] = {k: round(float(v), 3) for k, v in imp.head(5).items()}
    return out


def spend_regression(feats: pd.DataFrame) -> dict:
    """Linear regression: how much will a customer spend in the next 90 days (log scale)."""
    # monetary is dropped: it is roughly frequency x avg_order_value, which makes coefficients unstable
    X, y = prep(feats).drop(columns=["monetary"]), np.log1p(feats.future_spend)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
    coef = pd.Series(lr[-1].coef_, index=X.columns).sort_values(key=abs, ascending=False)
    return {"test_r2": round(float(lr.score(Xte, yte)), 3),
            "top_standardised_coefs": {k: round(float(v), 3) for k, v in coef.head(4).items()}}


def forecast_revenue(sales: pd.DataFrame, horizon: int = 3) -> dict:
    monthly = sales.groupby(sales.order_date.dt.to_period("M")).amount.sum().astype(float)
    monthly = monthly.iloc[:-1]  # last month (Dec 2011) is partial, drop it
    monthly.index = monthly.index.to_timestamp()
    train, test = monthly[:-horizon], monthly[-horizon:]
    mape = lambda p: round(float((abs(np.asarray(p) - test.values) / test.values).mean() * 100), 1)
    res = {"train_months": len(train), "holdout_months": [d.strftime("%Y-%m") for d in test.index]}
    res["seasonal_naive_mape_pct"] = mape(monthly.shift(12)[-horizon:].values)  # same month last year
    res["holt_trend_mape_pct"] = mape(ExponentialSmoothing(train, trend="add", damped_trend=True).fit().forecast(horizon))
    # Holt-Winters with a 12-month season needs >= 24 months of training data; we only have 23 full
    # months, so it cannot be fitted honestly. Seasonal-naive (same month last year) is the benchmark.
    res["note"] = "12-month seasonal Holt-Winters not fitted: only 23 full months available (needs 24+)"
    monthly.to_csv(OUT / "monthly_revenue.csv", header=["revenue"])
    return res


def nonparametric(feats: pd.DataFrame) -> dict:
    a, b = feats[feats.churned == 1].monetary, feats[feats.churned == 0].monetary
    top = feats.country.value_counts()
    top = top[top >= 30].index[:5]
    k = stats.kruskal(*[feats[feats.country == c].monetary.values for c in top])
    rho = stats.spearmanr(feats.frequency, feats.monetary)
    return {"mannwhitney_churned_vs_retained_p": float(stats.mannwhitneyu(a, b).pvalue),
            "median_spend_churned": round(float(a.median())), "median_spend_retained": round(float(b.median())),
            "kruskal_across_top_countries_p": float(k.pvalue), "countries_tested": list(top),
            "spearman_frequency_vs_spend_rho": round(float(rho.statistic), 3)}


def main():
    OUT.mkdir(exist_ok=True)
    feats = pd.read_parquet(PROC / "features.parquet")
    sales = pd.read_parquet(PROC / "sales", columns=["order_date", "amount"])
    sales["order_date"] = pd.to_datetime(sales.order_date)
    res = {
        "n_customers": int(len(feats)), "churn_rate_pct": round(float(feats.churned.mean() * 100), 1),
        "churn_models": churn_models(feats),
        "spend_regression": spend_regression(feats),
        "revenue_forecast": forecast_revenue(sales),
        "nonparametric": nonparametric(feats),
    }
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
