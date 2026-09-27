# Used Cars Price & Market Analysis

## Project Overview
Modular Python data pipeline and analysis suite exploring depreciation, transmission premiums, and seller behaviors in used car markets.

## Setup Instructions
```bash
# Install the package (with Jupyter and pytest) in editable mode.
# All dependency versions live in pyproject.toml.
python -m pip install -e ".[dev]"

# Run tests
python -m pytest
```

## Analysis Workflow

Run the notebooks in order from the project root:

1. `notebooks/01_data_inspection.ipynb` loads and cleans the raw CSV.
2. `notebooks/02_exploratory_data_analysis.ipynb` explores distributions and relationships.
3. `notebooks/03_statistical_analysis.ipynb` runs hypothesis tests and regression.
4. `notebooks/04_executive_conclusions.ipynb` summarizes the main business findings.

The cleaned dataset is written to `data/cleaned_car_data.csv` by the first notebook. Charts produced by notebooks 2 and 3 are saved to `images/generated/` (not tracked in git; regenerated each run).