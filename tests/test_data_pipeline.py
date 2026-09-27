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

def test_transmission_test_returns_expected_metrics(sample_data):
    featured = add_features(clean_data(sample_data))
    results = transmission_price_test(featured)
    expected_keys = [
        "manual_sample_size", "automatic_sample_size", 
        "manual_mean_price", "automatic_mean_price", 
        "welch_t_statistic", "p_value", "cohens_d"
    ]
    for key in expected_keys:
        assert key in results