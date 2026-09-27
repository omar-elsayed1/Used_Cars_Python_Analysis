from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "name", "year", "selling_price", "km_driven", 
    "fuel", "seller_type", "transmission", "owner"
]

def find_project_root(current_path: Path = None) -> Path:
    """Finds project root by searching for pyproject.toml or .git upwards."""
    if current_path is None:
        current_path = Path.cwd()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current_path

def load_raw_data(relative_path: str = "data/CAR DETAILS FROM CAR DEKHO.csv") -> pd.DataFrame:
    """Loads raw dataset relative to project root."""
    root = find_project_root()
    full_path = root / relative_path
    if not full_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {full_path}")
    return pd.read_csv(full_path)

def validate_data(df: pd.DataFrame) -> bool:
    """Validates presence of required columns and non-negative numeric fields."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    numeric_cols = ["year", "selling_price", "km_driven"]
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Column '{col}' contains invalid negative values.")
    return True

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace, removes exact duplicate rows, and cleans string values."""
    df_clean = df.copy()
    
    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    return df_clean

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds reproducible car_age feature relative to max year in dataset."""
    df_feat = df.copy()
    reference_year = df_feat["year"].max()
    df_feat["car_age"] = reference_year - df_feat["year"]
    return df_feat

def save_clean_data(df: pd.DataFrame, relative_path: str = "data/cleaned_car_data.csv") -> Path:
    """Saves cleaned dataset to target CSV path."""
    root = find_project_root()
    output_path = root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path