"""
test_data.py — Phase 1 tests for data ingestion, validation, and preprocessing.
Run with: python -m pytest tests/test_data.py -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import load_raw_data
from src.data.validator import validate_dataset
from src.data.preprocessing import encode_target, drop_non_predictive_columns, clean_dataset
from src.config import TARGET_COLUMN, DROP_COLUMNS, RAW_DATA_PATH


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load the actual CSV once for all tests in this module."""
    return load_raw_data()


@pytest.fixture
def minimal_df():
    """Minimal valid DataFrame for unit tests that don't need the real CSV."""
    return pd.DataFrame({
        "Age": [30, 45],
        "Attrition": ["Yes", "No"],
        "OverTime": ["Yes", "No"],
        "EmployeeNumber": [1, 2],
        "EmployeeCount": [1, 1],
        "StandardHours": [80, 80],
        "Over18": ["Y", "Y"],
        "MonthlyIncome": [5000, 8000],
    })


# ── Loader tests ───────────────────────────────────────────────────────────

class TestLoader:
    def test_loads_real_csv(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)
        assert not raw_df.empty

    def test_expected_row_count(self, raw_df):
        # IBM HR dataset has 1470 rows
        assert len(raw_df) == 1470

    def test_expected_column_count(self, raw_df):
        assert len(raw_df.columns) == 35

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data(Path("data/raw/nonexistent.csv"))


# ── Validator tests ────────────────────────────────────────────────────────

class TestValidator:
    def test_pass_on_real_data(self, raw_df):
        report = validate_dataset(raw_df)
        # Real dataset should have no hard issues
        assert report["issues"] == []

    def test_detects_target_column(self, raw_df):
        report = validate_dataset(raw_df)
        assert TARGET_COLUMN in raw_df.columns
        assert "Yes" in report["target_distribution"]
        assert "No" in report["target_distribution"]

    def test_detects_constant_columns(self, raw_df):
        report = validate_dataset(raw_df)
        # EmployeeCount, StandardHours, Over18 are constant
        assert "EmployeeCount" in report["constant_columns"]
        assert "StandardHours" in report["constant_columns"]
        assert "Over18" in report["constant_columns"]

    def test_detects_id_column(self, raw_df):
        report = validate_dataset(raw_df)
        assert "EmployeeNumber" in report["potential_id_columns"]

    def test_no_missing_values_in_real_data(self, raw_df):
        report = validate_dataset(raw_df)
        assert report["missing_values"] == {}

    def test_no_duplicate_rows_in_real_data(self, raw_df):
        report = validate_dataset(raw_df)
        assert report["duplicate_rows"] == 0

    def test_flags_missing_column(self):
        df = pd.DataFrame({"Age": [30], "Attrition": ["Yes"]})
        report = validate_dataset(df)
        assert len(report["missing_columns"]) > 0
        assert report["status"] == "FAIL"

    def test_flags_duplicate_rows(self):
        df = pd.DataFrame({
            "Age": [30, 30],
            "Attrition": ["Yes", "Yes"],
        })
        report = validate_dataset(df)
        assert report["duplicate_rows"] == 1


# ── Preprocessing tests ────────────────────────────────────────────────────

class TestPreprocessing:
    def test_encode_target_yes_to_1(self, minimal_df):
        result = encode_target(minimal_df)
        assert result[TARGET_COLUMN].iloc[0] == 1

    def test_encode_target_no_to_0(self, minimal_df):
        result = encode_target(minimal_df)
        assert result[TARGET_COLUMN].iloc[1] == 0

    def test_encode_target_dtype_is_int(self, minimal_df):
        result = encode_target(minimal_df)
        assert pd.api.types.is_integer_dtype(result[TARGET_COLUMN])

    def test_encode_target_raises_on_unexpected_value(self):
        df = pd.DataFrame({"Attrition": ["Maybe"]})
        with pytest.raises(ValueError):
            encode_target(df)

    def test_drop_non_predictive_removes_drop_columns(self, minimal_df):
        result = drop_non_predictive_columns(minimal_df)
        for col in DROP_COLUMNS:
            assert col not in result.columns

    def test_clean_dataset_on_real_data(self, raw_df):
        cleaned = clean_dataset(raw_df)
        # Target should be binary integer
        assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1})
        # Drop columns should be gone
        for col in DROP_COLUMNS:
            assert col not in cleaned.columns

    def test_class_imbalance_ratio(self, raw_df):
        """Attrition rate in this dataset is ~16% — verify it's in expected range."""
        cleaned = clean_dataset(raw_df)
        attrition_rate = cleaned[TARGET_COLUMN].mean()
        assert 0.10 < attrition_rate < 0.25, (
            f"Unexpected attrition rate: {attrition_rate:.2%}"
        )
