"""
PulseRetain — Central configuration.
All constants and paths live here; nothing is scattered across the codebase.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_SAMPLE_DIR = ROOT_DIR / "data" / "sample"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_RAW_DIR / "HR-Employee-Attrition.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "employee_features.csv"
MODEL_PATH = MODELS_DIR / "attrition_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ── Target ─────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Attrition"
EMPLOYEE_ID_COLUMN = "EmployeeNumber"

# ── Columns to drop before modelling ──────────────────────────────────────
# Identifiers and administrative constants that carry no predictive signal
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]

# ── Risk thresholds (score = probability × 100) ────────────────────────────
RISK_LOW_THRESHOLD = 30
RISK_MODERATE_THRESHOLD = 60
RISK_HIGH_THRESHOLD = 80

# ── ML ─────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Logging ────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
