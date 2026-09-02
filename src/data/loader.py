"""
loader.py — CSV ingestion for PulseRetain.

Responsibilities:
  - Locate and load the raw HR attrition CSV.
  - Log dataset dimensions.
  - Raise informative errors on missing files or empty data.
"""
import logging
from pathlib import Path

import pandas as pd

from src.config import LOG_FORMAT, LOG_DATE_FORMAT, RAW_DATA_PATH

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw HR attrition CSV and return a DataFrame.

    Args:
        path: Path to the CSV file. Defaults to the configured raw data path.

    Returns:
        pandas DataFrame with the raw dataset.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
        ValueError: If the loaded file is empty.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place HR-Employee-Attrition.csv in data/raw/ and retry."
        )

    logger.info("Loading dataset from: %s", path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset at {path} loaded but is empty.")

    logger.info("Dataset loaded — %d rows × %d columns", *df.shape)

    # Strip trailing garbage rows where Age is not numeric.
    # The IBM HR CSV sometimes has metadata rows appended at the end.
    original_len = len(df)
    if "Age" in df.columns:
        df = df[pd.to_numeric(df["Age"], errors="coerce").notna()].copy()
        df["Age"] = df["Age"].astype(int)
        dropped = original_len - len(df)
        if dropped > 0:
            logger.warning(
                "Stripped %d non-data rows (Age was non-numeric — likely CSV metadata).",
                dropped,
            )
            logger.info("Clean dataset — %d rows × %d columns", *df.shape)

    return df


if __name__ == "__main__":
    from src.data.validator import validate_dataset
    from src.data.preprocessing import generate_data_quality_report, clean_dataset, save_processed_data

    df = load_raw_data()
    report = validate_dataset(df)
    generate_data_quality_report(df, report)

    cleaned = clean_dataset(df)
    save_processed_data(cleaned)
