import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def correlation_with_price(df: pd.DataFrame) -> pd.Series:
    """Calculates Pearson correlation of numeric features against selling_price."""
    numeric_df = df.select_dtypes(include=[np.number])
    if "selling_price" not in numeric_df.columns:
        raise KeyError("'selling_price' column must be present in DataFrame.")
    return numeric_df.corr()["selling_price"].sort_values(ascending=False)

def transmission_price_test(df: pd.DataFrame) -> dict:
    """Performs Welch's t-test comparing selling price across transmission types."""
    manual = df[df["transmission"] == "Manual"]["selling_price"].dropna()
    automatic = df[df["transmission"] == "Automatic"]["selling_price"].dropna()
    
    t_stat, p_val = stats.ttest_ind(automatic, manual, equal_var=False)
    
    n1, n2 = len(automatic), len(manual)
    s1, s2 = automatic.std(), manual.std()
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (automatic.mean() - manual.mean()) / s_pooled
    
    return {
        "manual_sample_size": n1,
        "automatic_sample_size": n2,
        "manual_mean_price": float(manual.mean()),
        "automatic_mean_price": float(automatic.mean()),
        "welch_t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d)
    }

def age_price_regression(df: pd.DataFrame) -> dict:
    """Fits OLS linear regression predicting selling_price from car_age."""
    X = sm.add_constant(df["car_age"])
    y = df["selling_price"]
    model = sm.OLS(y, X).fit()
    
    return {
        "r_squared": float(model.rsquared),
        "intercept": float(model.params["const"]),
        "slope_car_age": float(model.params["car_age"]),
        "p_value_car_age": float(model.pvalues["car_age"])
    }

def executive_kpis(df: pd.DataFrame) -> dict:
    """Computes high-level summary KPIs for reporting."""
    return {
        "total_vehicles": len(df),
        "avg_price": float(df["selling_price"].mean()),
        "median_price": float(df["selling_price"].median()),
        "avg_km_driven": float(df["km_driven"].mean()),
        "avg_car_age": float(df["car_age"].mean()) if "car_age" in df.columns else None
    }