import pytest
import pandas as pd
from used_cars_analysis.data import clean_data, add_features, validate_data
from used_cars_analysis.analysis import correlation_with_price, transmission_price_test

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "name": ["Car A ", "Car A ", "Car B"],
        "year": [2018, 2018, 2015],
        "selling_price": [500000, 500000, 300000],
        "km_driven": [40000, 40000, 70000],
        "fuel": ["Petrol", "Petrol", "Diesel"],
        "seller_type": ["Individual", "Individual", "Dealer"],
        "transmission": ["Manual", "Manual", "Automatic"],
        "owner": ["First Owner", "First Owner", "Second Owner"]
    })

def test_clean_data_removes_duplicates(sample_data):
    cleaned = clean_data(sample_data)
    assert len(cleaned) == 2
    assert cleaned["name"].iloc[0] == "Car A"

def test_clean_data_adds_car_age(sample_data):
    cleaned = clean_data(sample_data)
    featured = add_features(cleaned)
    assert "car_age" in featured.columns
    assert featured["car_age"].iloc[0] == 0
    assert featured["car_age"].iloc[1] == 3

def test_validation_rejects_missing_columns():
    invalid_df = pd.DataFrame({"year": [2020], "selling_price": [100000]})
    with pytest.raises(ValueError, match="Dataset missing required columns"):
        validate_data(invalid_df)

def test_correlation_output_contains_selling_price(sample_data):
    featured = add_features(clean_data(sample_data))
    corr = correlation_with_price(featured)
    assert "selling_price" in corr.index
    assert "car_age" in corr.index

def test_transmission_test_returns_expected_metrics():
    # 3 manual rows, 2 automatic rows, with automatic priced strictly higher,
    # so the expected direction and magnitude of every metric is known.
    df = pd.DataFrame({
        "selling_price": [400000, 500000, 600000, 900000, 1100000],
        "transmission": ["Manual", "Manual", "Manual", "Automatic", "Automatic"],
    })

    results = transmission_price_test(df)

    expected_keys = [
        "manual_sample_size", "automatic_sample_size",
        "manual_mean_price", "automatic_mean_price",
        "welch_t_statistic", "p_value", "cohens_d"
    ]
    for key in expected_keys:
        assert key in results

    # Sample sizes must map to the correct transmission type, not be swapped.
    assert results["manual_sample_size"] == 3
    assert results["automatic_sample_size"] == 2

    # Mean prices must match their own group, not the other group's.
    assert results["manual_mean_price"] == pytest.approx(500000)
    assert results["automatic_mean_price"] == pytest.approx(1000000)

    # Automatic vehicles are priced higher in this sample, so both the
    # t-statistic and Cohen's d should be positive (automatic - manual).
    assert results["welch_t_statistic"] > 0
    assert results["cohens_d"] == pytest.approx(4.330127, rel=1e-4)