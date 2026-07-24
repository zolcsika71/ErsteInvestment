"""Shared project paths used by import, analysis, and report commands."""

from pathlib import Path
from typing import Final


PROJECT_DIR: Final = Path(__file__).resolve().parent
INPUT_DIR: Final = PROJECT_DIR / "model_portfolios_xls"
DATABASE_PATH: Final = PROJECT_DIR / "db" / "model_portfolio.sqlite"
ANALYSIS_JSON_PATH: Final = PROJECT_DIR / "db" / "investment_analysis.json"
ANALYSIS_XLSX_PATH: Final = PROJECT_DIR / "results" / "investment_analysis.xlsx"
ENV_PATH: Final = PROJECT_DIR / ".env"
