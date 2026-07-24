"""Read and normalize model-portfolio Excel worksheets."""

from __future__ import annotations

import re
from numbers import Real
from pathlib import Path
from typing import Any, Final

import pandas as pd
from python_calamine import load_workbook

from .text_normalization import normalized_key


TARGET_WORKSHEET: Final = "modell portfóliók"

HEADER_TRANSLATIONS: Final = {
    "portfólió neve": "Portfolio Name", "portfolio name": "Portfolio Name",
    "termék": "Product", "product": "Product", "isin": "ISIN",
    "hányad (%)": "Allocation (%)", "allocation (%)": "Allocation (%)",
    "eszközosztály": "Asset Class", "asset class": "Asset Class",
    "aleszközosztály": "Sub-Asset Class", "sub-asset class": "Sub-Asset Class",
    "termék típus": "Product Type", "product type": "Product Type",
    "deviza": "Currency", "currency": "Currency",
    "devizakockázat": "Currency Risk", "currency risk": "Currency Risk",
    "fenntarthatóság": "Sustainability", "sustainability": "Sustainability",
    "ytd": "YTD", "1yr": "1 Year", "1 year": "1 Year",
    "3yr": "3 Years", "3 years": "3 Years", "5yr": "5 Years",
    "5 years": "5 Years", "1y sharpe": "1Y Sharpe Ratio",
    "1y sharpe ratio": "1Y Sharpe Ratio", "3y sharpe": "3Y Sharpe Ratio",
    "3y sharpe ratio": "3Y Sharpe Ratio", "5y sharpe": "5Y Sharpe Ratio",
    "5y sharpe ratio": "5Y Sharpe Ratio", "1y vol.": "1Y Volatility",
    "1y volatility": "1Y Volatility", "3y vol.": "3Y Volatility",
    "3y volatility": "3Y Volatility", "down. risk": "Downside Risk",
    "downside risk": "Downside Risk", "info. ratio": "Information Ratio",
    "information ratio": "Information Ratio", "max. drawd.": "Maximum Drawdown",
    "maximum drawdown": "Maximum Drawdown",
}

VALUE_TRANSLATIONS: Final = {
    "Asset Class": {
        "alternatív": "Alternative",
        "kötvény": "Bond",
        "kötvény - befektetési kategória": "Investment Grade Bond",
        "kötvény-befektetési kategória": "Investment Grade Bond",
        "kötvény - magas hozamú": "High Yield Bond",
        "kötvény-magas hozamú": "High Yield Bond",
        "kötvény-rugalmas": "Flexible Bond",
        "pénzpiac": "Money Market",
        "pénzpiaci": "Money Market",
        "részvény": "Equity",
    },
    "Sub-Asset Class": {
        "abszolút hozamú": "Absolute Return",
        "amerikai dollár": "USD",
        "eur": "EUR",
        "euro": "EUR",
        "európa": "Europe",
        "európa-vállalatok": "Europe-Corporates",
        "európai vállalatok": "Europe-Corporates",
        "fejl?d? piacok": "Emerging Markets",
        "globál": "Global",
        "globál állampapír": "Global-Government Bond",
        "globál-állampapír": "Global-Government Bond",
        "hu-állampapír": "Hungary-Government Bond",
        "huf": "HUF",
        "ingatlan": "Real Estate",
        "kötvény - magyar állampapírok": "Bond - Hungarian Government Bonds",
        "közép-kelet európai állampapír":
            "Central and Eastern European Government Bond",
        "magyar forint": "HUF",
        "magyar állampapírok": "Hungarian Government Bonds",
        "nyersanyag": "Commodities",
        "részvény - fejl?d? piacok": "Equity - Emerging Markets",
        "usd": "USD",
        "észak-amerika": "North America",
        "észak-amerika-állampapír": "North America-Government Bond",
        "észak-amerikai állampapír": "North America-Government Bond",
    },
    "Currency Risk": {
        "fedezve": "Hedged",
        "nincs fedezve": "Unhedged",
        "részben fedezve": "Partially Hedged",
    },
    "Sustainability": {
        "0: nem minősített": "0: Not Rated",
        "1: esg-minimum standard": "1: ESG-Minimum Standard",
        "2: esg-plusz": "2: ESG-Plus",
        "3: esg-impact": "3: ESG-Impact",
    },
}

INVALID_CATEGORY_VALUES: Final = {
    "Currency Risk": {"2", "3", "4", "value!"},
}

MERGED_RANGE_PATTERN: Final = re.compile(
    r"^\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)$"
)


def is_visible_sheet(metadata: Any) -> bool:
    """Return whether worksheet metadata describes a visible sheet."""
    return str(metadata.visible).endswith(".Visible")


def column_label_to_index(label: str) -> int:
    """Convert an Excel column label to a zero-based index."""
    index = 0
    for char in label:
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def parse_merged_range(range_value: Any) -> tuple[int, int, int, int]:
    """Parse a merged-cell range into zero-based boundaries."""
    if isinstance(range_value, str):
        match = MERGED_RANGE_PATTERN.match(range_value.upper())
        if match is None:
            raise ValueError(f"Unsupported merged-cell range: {range_value!r}")
        start_col, start_row, end_col, end_row = match.groups()
        return (int(start_row) - 1, column_label_to_index(start_col),
                int(end_row) - 1, column_label_to_index(end_col))
    if isinstance(range_value, (tuple, list)) and len(range_value) == 4:
        return tuple(int(value) for value in range_value)  # type: ignore[return-value]
    raise ValueError(f"Unsupported merged-cell range: {range_value!r}")


def fill_merged_cells(frame: pd.DataFrame, merged_ranges: list[Any]) -> pd.DataFrame:
    """Repeat each merged range's top-left value across that range."""
    if not merged_ranges:
        return frame
    frame = frame.copy()
    for range_value in merged_ranges:
        start_row, start_col, end_row, end_col = parse_merged_range(range_value)
        if start_row >= frame.shape[0] or start_col >= frame.shape[1]:
            continue
        value = frame.iat[start_row, start_col]
        frame.iloc[start_row:min(end_row, frame.shape[0] - 1) + 1,
                   start_col:min(end_col, frame.shape[1] - 1) + 1] = value
    return frame


def read_target_worksheet(file_path: Path) -> dict[str, pd.DataFrame]:
    """Read only the visible model-portfolio worksheet."""
    workbook = load_workbook(file_path)
    worksheets: dict[str, pd.DataFrame] = {}
    try:
        excel_file = pd.ExcelFile(file_path, engine="calamine")
    except ImportError as error:
        raise RuntimeError(
            "Missing Excel reader. Install it with: "
            "python -m pip install pandas python-calamine"
        ) from error
    with excel_file:
        for index, metadata in enumerate(workbook.sheets_metadata):
            if (
                not is_visible_sheet(metadata)
                or normalized_key(metadata.name) != TARGET_WORKSHEET
            ):
                continue
            sheet = workbook.get_sheet_by_index(index)
            frame = pd.read_excel(excel_file, sheet_name=metadata.name,
                                  dtype=object, header=None)
            frame = fill_merged_cells(frame, sheet.merged_cell_ranges)
            frame = frame.dropna(axis=1, how="all").dropna(how="all").reset_index(drop=True)
            worksheets[metadata.name] = frame
    if not worksheets:
        raise ValueError(
            f"Workbook has no visible 'Modell portfóliók' worksheet: {file_path.name}"
        )
    return worksheets


def translate_headers(headers: list[Any], file_name: str, sheet_name: str) -> list[str]:
    """Translate the worksheet header row into database field names."""
    translated: list[str] = []
    for index, header in enumerate(headers, start=1):
        if pd.isna(header) or not str(header).strip():
            raise ValueError(f"{file_name} / {sheet_name}: blank header in column {index}")
        key = normalized_key(header)
        if key not in HEADER_TRANSLATIONS:
            raise ValueError(
                f"{file_name} / {sheet_name}: no English translation configured "
                f"for header {header!r} in column {index}"
            )
        translated.append(HEADER_TRANSLATIONS[key])
    duplicates = sorted({item for item in translated if translated.count(item) > 1})
    if duplicates:
        raise ValueError(
            f"{file_name} / {sheet_name}: duplicate translated headers: {', '.join(duplicates)}"
        )
    return translated


def translate_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate categorical values and reject unknown non-English categories."""
    frame = frame.copy()
    for column_name, translations in VALUE_TRANSLATIONS.items():
        if column_name not in frame.columns:
            continue
        english_values = {
            normalized_key(value): value for value in translations.values()
        }
        invalid_values = INVALID_CATEGORY_VALUES.get(column_name, set())

        def translate(value: Any) -> Any:
            if value is None or pd.isna(value):
                return value
            key = normalized_key(value)
            if key in translations:
                return translations[key]
            if key in english_values:
                return english_values[key]
            if key in invalid_values:
                return None
            raise ValueError(
                f"No English translation configured for {column_name} "
                f"value: {value!r}"
            )

        frame[column_name] = frame[column_name].map(translate)
    return frame


def replace_numeric_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace numeric zero values with ``None`` for SQLite NULL insertion."""
    return frame.map(
        lambda value: (
            None
            if isinstance(value, Real)
            and not isinstance(value, bool)
            and value == 0
            else value
        )
    )


def _validate_columns(
    headers: list[str],
    expected_columns: tuple[str, ...],
    location: str,
) -> None:
    """Reject missing or unexpected worksheet columns."""
    missing = [column for column in expected_columns if column not in headers]
    extra = [column for column in headers if column not in expected_columns]
    if missing or extra:
        details = ([f"missing: {', '.join(missing)}"] if missing else []) + (
            [f"extra: {', '.join(extra)}"] if extra else []
        )
        raise ValueError(f"{location}: column mismatch ({'; '.join(details)})")


def prepare_rows(file_path: Path, sheet_name: str, frame: pd.DataFrame,
                 expected_columns: tuple[str, ...]) -> pd.DataFrame:
    """Validate headers and return normalized worksheet data rows."""
    if frame.empty:
        raise ValueError(f"{file_path.name} / {sheet_name}: worksheet is empty")
    headers = translate_headers(list(frame.iloc[0]), file_path.name, sheet_name)
    _validate_columns(headers, expected_columns, f"{file_path.name} / {sheet_name}")
    data = frame.iloc[1:].copy()
    data.columns = headers
    data = data.loc[:, list(expected_columns)]
    data = translate_values(data)
    data = replace_numeric_zeros(data)
    return data.astype(object).where(pd.notna(data), None)
