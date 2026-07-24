"""SQLite schema and import operations for model-portfolio workbooks."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from excel_processing import prepare_rows, read_target_worksheet


DATE_PATTERN: Final = re.compile(r"(\d{8})(?=\.[^.]+$)")
TABLE_NAME_OVERRIDES: Final = {
    "modell portfóliók": "model_portfolios",
    "model portfolios": "model_portfolios",
}
WORKSHEET_COLUMNS: Final = (
    "Portfolio Name", "Product", "ISIN", "Allocation (%)", "Asset Class",
    "Sub-Asset Class", "Currency", "Currency Risk", "Sustainability", "YTD",
    "1 Year", "3 Years", "5 Years", "1Y Sharpe Ratio", "3Y Sharpe Ratio",
    "5Y Sharpe Ratio", "1Y Volatility", "3Y Volatility", "Downside Risk",
    "Information Ratio", "Maximum Drawdown",
)
MODEL_PORTFOLIOS_COLUMNS: Final = ("Date", *WORKSHEET_COLUMNS)
EXPECTED_TABLE_COLUMNS: Final = {
    "model_portfolios": MODEL_PORTFOLIOS_COLUMNS,
}
TEXT_COLUMNS: Final = {
    "Date", "Portfolio Name", "Product", "ISIN", "Asset Class", "Sub-Asset Class",
    "Product Type", "Currency", "Currency Risk", "Sustainability",
}


class DatabaseError(Exception):
    """Report a database-layer failure without exposing sqlite internals."""


class DatabaseSession(AbstractContextManager[sqlite3.Connection]):
    """Own a SQLite connection and its commit/rollback lifecycle."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        try:
            self.connection = sqlite3.connect(self.database_path)
        except sqlite3.Error as error:
            raise DatabaseError(f"Could not open database: {self.database_path}") from error
        return self.connection

    def __exit__(self, error_type: type[BaseException] | None,
                 error: BaseException | None,
                 traceback: TracebackType | None) -> bool | None:
        assert self.connection is not None
        try:
            if error_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        except sqlite3.Error as sqlite_error:
            raise DatabaseError("Could not complete database transaction") from sqlite_error
        finally:
            self.connection.close()
        return None


def normalized(value: Any) -> str:
    """Return a normalized lookup key."""
    return unicodedata.normalize("NFC", str(value).strip()).casefold()


def extract_date(file_path: Path) -> str:
    """Extract the filename date and return it as ``YYYY/MM/DD``."""
    match = DATE_PATTERN.search(file_path.name)
    if match is None:
        raise ValueError(f"Filename does not end in an 8-digit date: {file_path.name}")
    parsed_date = datetime.strptime(match.group(1), "%Y%m%d")
    return parsed_date.strftime("%Y/%m/%d")


def normalize_table_name(sheet_name: str) -> str:
    """Map a supported worksheet name to its private SQLite table name."""
    try:
        return TABLE_NAME_OVERRIDES[normalized(sheet_name)]
    except KeyError as error:
        raise ValueError(f"No table mapping is configured for worksheet: {sheet_name!r}") from error


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_type(column_name: str) -> str:
    return "TEXT" if column_name in TEXT_COLUMNS else "REAL"


def table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    """Return SQLite table columns in order."""
    rows = connection.execute(
        f"PRAGMA table_info({_quote_identifier(table_name)})"
    ).fetchall()
    return [row[1] for row in rows]


def ensure_data_table(connection: sqlite3.Connection, table_name: str,
                      columns: tuple[str, ...]) -> None:
    """Create a data table or reject an incompatible existing schema."""
    existing = table_columns(connection, table_name)
    if existing:
        if existing != list(columns):
            raise ValueError(
                f"Existing table {table_name!r} has incompatible columns. Run with --rebuild."
            )
        return
    definitions = [f"{_quote_identifier(column)} {_sql_type(column)}" for column in columns]
    connection.execute(
        f"CREATE TABLE {_quote_identifier(table_name)} ({', '.join(definitions)})"
    )


def date_exists(connection: sqlite3.Connection, import_date: str) -> bool:
    """Return whether portfolio rows already exist for a workbook date."""
    if not table_columns(connection, "model_portfolios"):
        return False
    return connection.execute(
        'SELECT 1 FROM model_portfolios WHERE "Date" = ? LIMIT 1', (import_date,)
    ).fetchone() is not None


def import_file(file_path: Path, database_path: Path) -> bool:
    """Import a workbook atomically; return False for an existing date."""
    file_path = file_path.expanduser().resolve()
    database_path = database_path.expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    import_date = extract_date(file_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DatabaseSession(database_path) as connection:
            if date_exists(connection, import_date):
                print(f"Skipped: Date {import_date} already exists in {database_path}")
                return False
            worksheets = read_target_worksheet(file_path)
            row_count = 0
            for sheet_name, worksheet in worksheets.items():
                table_name = normalize_table_name(sheet_name)
                columns = EXPECTED_TABLE_COLUMNS[table_name]
                frame = prepare_rows(
                    file_path,
                    sheet_name,
                    worksheet,
                    WORKSHEET_COLUMNS,
                )
                if frame.empty:
                    continue
                ensure_data_table(connection, table_name, columns)
                column_sql = ", ".join(_quote_identifier(name) for name in columns)
                placeholders = ", ".join("?" for _ in columns)
                connection.executemany(
                    f"INSERT INTO {_quote_identifier(table_name)} ({column_sql}) "
                    f"VALUES ({placeholders})",
                    (
                        (import_date, *row)
                        for row in frame.itertuples(index=False, name=None)
                    ),
                )
                row_count += len(frame)
    except DatabaseError:
        raise
    except sqlite3.Error as error:
        raise DatabaseError(f"Database operation failed: {database_path}") from error
    print(f"Imported {row_count} rows for Date {import_date} into {database_path}")
    return True
