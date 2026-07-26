"""Create and populate the shortlist SQLite database.

The command-line entry point passes files from ``model_portfolios_shortlist_xls``
to :func:`import_file` and uses ``db/shortlist.sqlite` as the destination.
The importer deliberately receives the destination path from its caller, so
the existing ``model_portfolio.sqlite`` database is never opened or changed by
the default shortlist workflow.

The database schema and import behavior remain consistent with the original
model-portfolio importer: the same field names are used, the date is derived
from each filename in ``YYYY/MM/DD`` format, and files whose date is already
present are skipped.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Final

from .excel_processing import prepare_rows, read_target_worksheet
from .text_normalization import normalized_key


# Matches an eight-digit date immediately before the file extension.
# Example: ``portfolio_20250726.xlsx`` -> ``20250726``.
DATE_PATTERN: Final = re.compile(r"(\d{8})(?=\.[^.]+$)")


# Worksheet names can differ by language or formatting. These normalized
# names are mapped to stable, application-specific SQLite table names.
# noinspection SpellCheckingInspection
TABLE_NAME_OVERRIDES: Final = {
    "modell portfóliók": "model_portfolios",
    "model portfolios": "model_portfolios",
}


# Column names expected in the source worksheet.
WORKSHEET_COLUMNS: Final = (
    "Portfolio Name",
    "Product",
    "ISIN",
    "Allocation (%)",
    "Asset Class",
    "Sub-Asset Class",
    "Currency",
    "Currency Risk",
    "Sustainability",
    "YTD",
    "1 Year",
    "3 Years",
    "5 Years",
    "1Y Sharpe Ratio",
    "3Y Sharpe Ratio",
    "5Y Sharpe Ratio",
    "1Y Volatility",
    "3Y Volatility",
    "Downside Risk",
    "Information Ratio",
    "Maximum Drawdown",
)


# Database rows include the source worksheet columns plus the import date.
MODEL_PORTFOLIOS_COLUMNS: Final = ("Date", *WORKSHEET_COLUMNS)


# Defines the expected schema for each supported database table.
EXPECTED_TABLE_COLUMNS: Final = {
    "model_portfolios": MODEL_PORTFOLIOS_COLUMNS,
}


# Columns listed here are textual. All other supported columns are numeric.
TEXT_COLUMNS: Final = {
    "Date",
    "Portfolio Name",
    "Product",
    "ISIN",
    "Asset Class",
    "Sub-Asset Class",
    "Product Type",
    "Currency",
    "Currency Risk",
    "Sustainability",
}


class DatabaseError(Exception):
    """Represent a database-layer failure without exposing SQLite internals."""


class DatabaseSession(AbstractContextManager[sqlite3.Connection]):
    """Manage a SQLite connection and its commit/rollback lifecycle."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        """Open the database connection when entering the context manager."""
        try:
            connection = sqlite3.connect(self.database_path)
        except sqlite3.Error as error:
            raise DatabaseError(
                f"Could not open database: {self.database_path}"
            ) from error

        self.connection = connection
        return connection

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Commit successful work or roll back the transaction after an error."""
        assert self.connection is not None

        try:
            if error_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        except sqlite3.Error as sqlite_error:
            raise DatabaseError(
                "Could not complete database transaction"
            ) from sqlite_error
        finally:
            # Always close the connection, regardless of transaction outcome.
            self.connection.close()

        # Returning None allows any original exception to propagate normally.
        return None


def extract_date(file_path: Path) -> str:
    """Extract a filename date and return it in ``YYYY/MM/DD`` format.

    The source filename must end with an eight-digit date immediately before
    its extension, such as ``portfolio_20250726.xlsx``.
    """
    match = DATE_PATTERN.search(file_path.name)

    if match is None:
        raise ValueError(
            f"Filename does not end in an eight-digit date: {file_path.name}"
        )

    parsed_date = datetime.strptime(match.group(1), "%Y%m%d")
    return parsed_date.strftime("%Y/%m/%d")


def normalize_table_name(sheet_name: str) -> str:
    """Convert a supported worksheet name into its SQLite table name."""
    try:
        return TABLE_NAME_OVERRIDES[normalized_key(sheet_name)]
    except KeyError as error:
        raise ValueError(
            f"No table mapping is configured for worksheet: {sheet_name!r}"
        ) from error


def _quote_identifier(identifier: str) -> str:
    """Safely quote a SQLite table or column identifier."""
    # Escaping embedded quotes prevents malformed SQL identifiers.
    return '"' + identifier.replace('"', '""') + '"'


def _sql_type(column_name: str) -> str:
    """Return the SQLite type required for a database column."""
    return "TEXT" if column_name in TEXT_COLUMNS else "REAL"


def table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[str]:
    """Return the columns currently defined for a SQLite table, in order."""
    rows = connection.execute(
        f"PRAGMA table_info({_quote_identifier(table_name)})"
    ).fetchall()

    # PRAGMA table_info returns the column name at index 1.
    return [row[1] for row in rows]


def ensure_data_table(
    connection: sqlite3.Connection,
    table_name: str,
    columns: tuple[str, ...],
) -> None:
    """Create a table or reject an incompatible existing schema.

    Existing tables are not altered automatically. This prevents an import
    from silently writing data into an unexpected schema.
    """
    existing_columns = table_columns(connection, table_name)

    if existing_columns:
        if existing_columns != list(columns):
            raise ValueError(
                f"Existing table {table_name!r} has incompatible columns. "
                "Run with --rebuild."
            )
        return

    column_definitions = [
        f"{_quote_identifier(column)} {_sql_type(column)}"
        for column in columns
    ]
    connection.execute(
        f"CREATE TABLE {_quote_identifier(table_name)} "
        f"({', '.join(column_definitions)})"
    )


def date_exists(
    connection: sqlite3.Connection,
    import_date: str,
) -> bool:
    """Return whether portfolio data already exists for the given date."""
    # If the table has not yet been created, no date can have been imported.
    if not table_columns(connection, "model_portfolios"):
        return False

    # The table is created dynamically by ensure_data_table.
    # noinspection SqlResolve
    result = connection.execute(
        'SELECT 1 FROM model_portfolios WHERE "Date" = ? LIMIT 1',
        (import_date,),
    ).fetchone()
    return result is not None


def import_file(
    file_path: Path,
    database_path: Path,
) -> bool:
    """Import one shortlist ``.xls`` workbook atomically into SQLite.

    Returns:
        ``True`` if rows were imported.
        ``False`` if the workbook date was already present in the database.

    Raises:
        ValueError: If the source file is not an ``.xls`` workbook or its
            filename does not contain a valid date.
    """
    # Resolve paths once so status messages and errors show unambiguous paths.
    file_path = file_path.expanduser().resolve()
    database_path = database_path.expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # The shortlist input directory is defined as an .xls-only source. Fail
    # before opening or creating the database if a different file is passed.
    if file_path.suffix.casefold() != ".xls":
        raise ValueError(
            f"Unsupported input format: {file_path.name}. "
            "Expected an .xls workbook."
        )

    import_date = extract_date(file_path)

    # SQLite can create the database file, but its parent directory must exist.
    database_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with DatabaseSession(database_path) as connection:
            # This date-level check makes the import idempotent and prevents
            # duplicating all rows from a workbook that was already imported.
            if date_exists(connection, import_date):
                print(
                    f"Skipped: Date {import_date} already exists "
                    f"in {database_path}"
                )
                return False

            worksheets = read_target_worksheet(file_path)
            row_count = 0

            for sheet_name, worksheet in worksheets.items():
                table_name = normalize_table_name(sheet_name)
                columns = EXPECTED_TABLE_COLUMNS[table_name]

                # Normalize worksheet data before passing it to the database
                # layer so insertion always receives the expected columns.
                frame = prepare_rows(
                    file_path,
                    sheet_name,
                    worksheet,
                    WORKSHEET_COLUMNS,
                )

                # Empty worksheets do not require table creation or insertion.
                if frame.empty:
                    continue

                ensure_data_table(connection, table_name, columns)

                column_sql = ", ".join(
                    _quote_identifier(column) for column in columns
                )
                placeholders = ", ".join("?" for _ in columns)
                insert_sql = (
                    f"INSERT INTO {_quote_identifier(table_name)} "
                    f"({column_sql}) VALUES ({placeholders})"
                )

                # Prefix every worksheet row with the workbook import date.
                # A generator avoids creating a second full in-memory copy.
                connection.executemany(
                    insert_sql,
                    (
                        (import_date, *row)
                        for row in frame.itertuples(index=False, name=None)
                    ),
                )
                row_count += len(frame)

    except DatabaseError:
        # Preserve errors that already contain database-layer context.
        raise
    except sqlite3.Error as error:
        # Convert raw SQLite exceptions into the public exception type used by
        # callers of this module.
        raise DatabaseError(
            f"Database operation failed: {database_path}"
        ) from error

    print(
        f"Imported {row_count} rows for Date {import_date} "
        f"into {database_path}"
    )
    return True
