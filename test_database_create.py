"""Tests for the isolated database layer."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from database_create import (
    DatabaseSession,
    MODEL_PORTFOLIOS_COLUMNS,
    date_exists,
    ensure_data_table,
    extract_date,
    table_columns,
)
from main import DEFAULT_DATABASE, DEFAULT_INPUT_DIR, PROJECT_DIR, rebuild_database
from excel_processing import replace_numeric_zeros, translate_values


class DatabaseCreateTests(unittest.TestCase):
    def test_numeric_zeros_are_replaced_with_null_values(self) -> None:
        source = pd.DataFrame({
            "values": [0.0, 0, 1.0, -1.0, "0.0", False, None],
        })

        cleaned = replace_numeric_zeros(source)

        self.assertIsNone(cleaned.at[0, "values"])
        self.assertIsNone(cleaned.at[1, "values"])
        self.assertEqual(cleaned.at[2, "values"], 1.0)
        self.assertEqual(cleaned.at[3, "values"], -1.0)
        self.assertEqual(cleaned.at[4, "values"], "0.0")
        self.assertIs(cleaned.at[5, "values"], False)
        self.assertIsNone(cleaned.at[6, "values"])

    def test_filename_date_uses_slash_format(self) -> None:
        path = Path("PB_Modell_Portfoliok_es_Shortlist_20240702.xls")
        self.assertEqual(extract_date(path), "2024/07/02")

    def test_categorical_values_are_translated_to_english(self) -> None:
        source = pd.DataFrame({
            "Asset Class": ["RÉSZVÉNY", "Kötvény-befektetési kategória"],
            "Sub-Asset Class": ["Fejl?d? piacok", "Globál-Állampapír"],
            "Currency Risk": ["Nincs fedezve", "Részben fedezve"],
            "Sustainability": ["1: ESG-Minimum standard", "2: ESG-Plusz"],
        })

        translated = translate_values(source)

        self.assertEqual(translated["Asset Class"].tolist(), [
            "Equity", "Investment Grade Bond",
        ])
        self.assertEqual(translated["Sub-Asset Class"].tolist(), [
            "Emerging Markets", "Global-Government Bond",
        ])
        self.assertEqual(translated["Currency Risk"].tolist(), [
            "Unhedged", "Partially Hedged",
        ])
        self.assertEqual(translated["Sustainability"].tolist(), [
            "1: ESG-Minimum Standard", "2: ESG-Plus",
        ])

    def test_invalid_currency_risk_values_become_null(self) -> None:
        source = pd.DataFrame({"Currency Risk": ["2", "VALUE!"]})

        translated = translate_values(source)

        self.assertTrue(translated["Currency Risk"].isna().all())

    def test_unknown_category_stops_import(self) -> None:
        source = pd.DataFrame({"Asset Class": ["Ismeretlen kategória"]})

        with self.assertRaisesRegex(ValueError, "No English translation configured"):
            translate_values(source)

    def test_default_project_paths(self) -> None:
        self.assertEqual(DEFAULT_INPUT_DIR, PROJECT_DIR / "model_portfolios_xls")
        self.assertEqual(DEFAULT_DATABASE, PROJECT_DIR / "DB" / "model_portfolio.sqlite")

    def test_duplicate_detection_uses_model_portfolios(self) -> None:
        with closing(sqlite3.connect(":memory:")) as connection:
            self.assertFalse(date_exists(connection, "2026/07/18"))
            ensure_data_table(
                connection,
                "model_portfolios",
                MODEL_PORTFOLIOS_COLUMNS,
            )
            values = ["2026/07/18", *([None] * (len(MODEL_PORTFOLIOS_COLUMNS) - 1))]
            placeholders = ", ".join("?" for _ in values)
            connection.execute(
                f"INSERT INTO model_portfolios VALUES ({placeholders})",
                values,
            )
            self.assertTrue(date_exists(connection, "2026/07/18"))

    def test_data_table_creation(self) -> None:
        with closing(sqlite3.connect(":memory:")) as connection:
            ensure_data_table(connection, "model_portfolios", MODEL_PORTFOLIOS_COLUMNS)
            self.assertEqual(
                table_columns(connection, "model_portfolios"),
                list(MODEL_PORTFOLIOS_COLUMNS),
            )
            self.assertEqual(MODEL_PORTFOLIOS_COLUMNS[0], "Date")

    def test_session_rolls_back_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite"
            with self.assertRaises(RuntimeError):
                with DatabaseSession(path) as connection:
                    connection.execute("CREATE TABLE sample (value TEXT)")
                    connection.execute("INSERT INTO sample VALUES ('not committed')")
                    raise RuntimeError("stop")
            with closing(sqlite3.connect(path)) as connection:
                table = connection.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'sample'"
                ).fetchone()
                if table:
                    self.assertEqual(connection.execute("SELECT * FROM sample").fetchall(), [])

    def test_rebuild_removes_database_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite"
            path.touch()
            rebuild_database(path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
