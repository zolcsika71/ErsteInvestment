"""Import Model Portfolios worksheets into an SQLite database.

Python: 3.14.6

For legacy .xls files install:
    python -m pip install pandas python-calamine

For .xlsx files also install:
    python -m pip install openpyxl

Usage:
    python main.py

Optional locations:
    python main.py --input-dir model_portfolios_xls --database DB/model_portfolio.sqlite
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd


DEFAULT_INPUT_DIR: Final = Path("model_portfolios_xls")
DEFAULT_DATABASE: Final = Path("DB/model_portfolio.sqlite")
DATE_PATTERN: Final = re.compile(r"(\d{8})(?=\.[^.]+$)")

# English database columns, in the same order as the worksheet columns.
# "Date" is added separately as the first column.
ENGLISH_HEADERS: Final = (
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

# SQL types corresponding to ENGLISH_HEADERS.
SQL_TYPES: Final = (
    "TEXT", "TEXT", "TEXT", "REAL", "TEXT", "TEXT", "TEXT", "TEXT",
    "TEXT", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL",
    "REAL", "REAL", "REAL", "REAL", "REAL",
)

# Translation maps for worksheet columns E, F, H and I (1-based numbering).
ASSET_CLASS_TRANSLATIONS: Final = {
    "alternatív": "Alternative",
    "kötvény": "Fixed Income",
    "kötvény-befektetési kategória": "Fixed Income",
    "kötvény-magas hozamú": "High Yield Fixed Income",
    "pénzpiaci": "Money Market",
    "részvény": "Equity",
}

SUB_ASSET_CLASS_TRANSLATIONS: Final = {
    "abszolút hozamú": "Absolute Return",
    "eur": "EUR",
    "huf": "HUF",
    "usd": "USD",
    "európa": "Europe",
    "európa-vállalatok": "Europe - Corporate Bonds",
    "fejlődő piacok": "Emerging Markets",
    # Some source files contain replacement characters instead of accents.
    "fejl?d? piacok": "Emerging Markets",
    "globál": "Global",
    "globál-állampapír": "Global - Government Bonds",
    "hu-állampapír": "Hungary - Government Bonds",
    "ingatlan": "Real Estate",
    "kötvény - magyar állampapírok": "Hungary - Government Bonds",
    "közép-kelet európai állampapír": (
        "Central and Eastern European Government Bonds"
    ),
    "nyersanyag": "Commodities",
    "részvény - fejlődő piacok": "Equity - Emerging Markets",
    "részvény - fejl?d? piacok": "Equity - Emerging Markets",
    "észak-amerika": "North America",
    "észak-amerika-állampapír": "North America - Government Bonds",
}

CURRENCY_RISK_TRANSLATIONS: Final = {
    "fedezve": "Hedged",
    "nincs fedezve": "Unhedged",
    "részben fedezve": "Partially Hedged",
}

SUSTAINABILITY_TRANSLATIONS: Final = {
    "1: esg-minimum standard": "1: ESG Minimum Standard",
    "2: esg-plusz": "2: ESG Plus",
}


def normalized(value: Any) -> str:
    """Return a trimmed, Unicode-normalized, case-insensitive lookup key."""
    return unicodedata.normalize("NFC", str(value).strip()).casefold()


def extract_date(file_path: Path) -> str:
    """Extract and validate the last eight digits immediately before extension."""
    match = DATE_PATTERN.search(file_path.name)
    if match is None:
        raise ValueError(
            f"Filename does not end in an 8-digit date: {file_path.name}"
        )

    date_text = match.group(1)
    datetime.strptime(date_text, "%Y%m%d")  # Reject impossible calendar dates.
    return date_text


def locate_model_portfolios_sheet(excel_file: pd.ExcelFile) -> str:
    """Find either the English sheet name or its name in the supplied files."""
    accepted_names = {"model portfolios", "modell portfóliók"}
    for sheet_name in excel_file.sheet_names:
        if normalized(sheet_name) in accepted_names:
            return sheet_name
    raise ValueError(
        "The workbook has no 'Model Portfolios' worksheet. "
        f"Available worksheets: {excel_file.sheet_names}"
    )


def read_worksheet(file_path: Path) -> pd.DataFrame:
    """Read only the Model Portfolios worksheet from .xls or .xlsx."""
    suffix = file_path.suffix.casefold()
    if suffix == ".xls":
        # Calamine is robust with legacy BIFF/OLE .xls files, including the
        # supplied files that some xlrd versions report as malformed.
        engine = "calamine"
    elif suffix == ".xlsx":
        engine = "openpyxl"
    else:
        raise ValueError(f"Unsupported Excel extension: {file_path.suffix}")

    try:
        excel_file = pd.ExcelFile(file_path, engine=engine)
    except ImportError as error:
        package = "python-calamine" if engine == "calamine" else "openpyxl"
        raise RuntimeError(
            f"Missing Excel reader. Install it with: "
            f"python -m pip install pandas {package}"
        ) from error

    with excel_file:
        sheet_name = locate_model_portfolios_sheet(excel_file)
        frame = pd.read_excel(excel_file, sheet_name=sheet_name, dtype=object)

    # The supplied Model Portfolios worksheet has exactly 21 source columns.
    if frame.shape[1] != len(ENGLISH_HEADERS):
        raise ValueError(
            f"Expected {len(ENGLISH_HEADERS)} columns, found {frame.shape[1]}."
        )

    # This translates the complete first/header row into English.
    frame.columns = ENGLISH_HEADERS
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame


def translate_column(
    frame: pd.DataFrame,
    column_name: str,
    translations: dict[str, str],
) -> None:
    """Translate known categorical values, preserving unrecognized source data."""
    translated_values: list[Any] = []

    for value in frame[column_name]:
        if pd.isna(value):
            translated_values.append(None)
            continue
        key = normalized(value)
        if key not in translations:
            translated_values.append(value)
        else:
            translated_values.append(translations[key])

    frame[column_name] = translated_values


def prepare_rows(file_path: Path, import_date: str) -> pd.DataFrame:
    """Read, translate, and normalize worksheet data for SQLite."""
    frame = read_worksheet(file_path)

    # Excel columns E, F, H and I after assigning the translated headers.
    translate_column(frame, "Asset Class", ASSET_CLASS_TRANSLATIONS)
    translate_column(frame, "Sub-Asset Class", SUB_ASSET_CLASS_TRANSLATIONS)
    translate_column(frame, "Currency Risk", CURRENCY_RISK_TRANSLATIONS)
    translate_column(frame, "Sustainability", SUSTAINABILITY_TRANSLATIONS)

    # Insert the requested Date column in the first position.
    frame.insert(0, "Date", import_date)

    # Convert pandas missing values to values sqlite3 can bind.
    return frame.astype(object).where(pd.notna(frame), None)


def quote_identifier(identifier: str) -> str:
    """Quote a trusted SQLite identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the relational import-batch and portfolio tables if absent."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS import_batches (
            "Date" TEXT PRIMARY KEY
                CHECK (length("Date") = 8 AND "Date" NOT GLOB '*[^0-9]*'),
            source_file TEXT NOT NULL,
            imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    definitions = ['"Date" TEXT NOT NULL']
    definitions.extend(
        f"{quote_identifier(name)} {sql_type}"
        for name, sql_type in zip(ENGLISH_HEADERS, SQL_TYPES, strict=True)
    )
    definitions.append(
        'FOREIGN KEY ("Date") REFERENCES import_batches("Date") ON DELETE CASCADE'
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS model_portfolios ("
        + ", ".join(definitions)
        + ")"
    )
    connection.execute(
        'CREATE INDEX IF NOT EXISTS idx_model_portfolios_date '
        'ON model_portfolios("Date")'
    )


def date_exists(connection: sqlite3.Connection, import_date: str) -> bool:
    """Return True when this dated workbook has already been imported."""
    result = connection.execute(
        'SELECT 1 FROM import_batches WHERE "Date" = ? LIMIT 1',
        (import_date,),
    ).fetchone()
    return result is not None


def import_file(file_path: Path, database_path: Path) -> bool:
    """Import a workbook atomically; return False when its date already exists."""
    file_path = file_path.expanduser().resolve()
    database_path = database_path.expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    import_date = extract_date(file_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        create_schema(connection)

        # Check before reading the workbook, avoiding unnecessary work.
        if date_exists(connection, import_date):
            print(f"Skipped: Date {import_date} already exists in {database_path}")
            return False

        frame = prepare_rows(file_path, import_date)
        if frame.empty:
            raise ValueError("The Model Portfolios worksheet contains no data rows.")

        columns = list(frame.columns)
        column_sql = ", ".join(quote_identifier(name) for name in columns)
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = (
            f"INSERT INTO model_portfolios ({column_sql}) "
            f"VALUES ({placeholders})"
        )

        # The context manager commits both tables together, or rolls both back
        # if any row fails. The PRIMARY KEY also protects against race/re-import.
        connection.execute(
            'INSERT INTO import_batches ("Date", source_file) VALUES (?, ?)',
            (import_date, file_path.name),
        )
        connection.executemany(insert_sql, frame.itertuples(index=False, name=None))

    print(
        f"Imported {len(frame)} rows for Date {import_date} "
        f"into {database_path}"
    )
    return True


def find_xls_files(input_dir: Path) -> list[Path]:
    """Return root-level .xls files from the configured input directory."""
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".xls"
    )
    if not files:
        raise FileNotFoundError(f"No .xls files found in: {input_dir}")
    return files


def import_directory(input_dir: Path, database_path: Path) -> tuple[int, int]:
    """Import all root-level .xls files from a directory."""
    imported_count = 0
    skipped_count = 0

    for file_path in find_xls_files(input_dir):
        if import_file(file_path, database_path):
            imported_count += 1
        else:
            skipped_count += 1

    return imported_count, skipped_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import all Model Portfolios .xls files into SQLite."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing .xls files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite file (default: {DEFAULT_DATABASE})",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        imported_count, skipped_count = import_directory(
            arguments.input_dir,
            arguments.database,
        )
    except (FileNotFoundError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Done. Imported {imported_count} file(s), "
        f"skipped {skipped_count} already imported file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
