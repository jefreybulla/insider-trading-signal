from pathlib import Path
import pandas as pd

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def write_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)