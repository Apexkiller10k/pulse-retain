"""
run_app.py — Launch PulseRetain Streamlit app.

Usage:
    python run_app.py
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
app  = ROOT / "app" / "app.py"

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(app),
     "--server.headless", "false"],
    cwd=str(ROOT),
)
