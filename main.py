"""Import /project/model_portfolios_xls/*.xls into the project database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from database_create import DatabaseError, import_file
from investment_analysis import AnalysisError, run_analysis, write_report


PROJECT_DIR: Final = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR: Final = PROJECT_DIR / "model_portfolios_xls"
DEFAULT_DATABASE: Final = PROJECT_DIR / "DB" / "model_portfolio.sqlite"


def find_xls_files(input_dir: Path) -> list[Path]:
    """Return root-level .xls files from the configured directory."""
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(path for path in input_dir.iterdir()
                   if path.is_file() and path.suffix.casefold() == ".xls")
    if not files:
        raise FileNotFoundError(f"No .xls files found in: {input_dir}")
    return files


def import_directory(input_dir: Path, database_path: Path) -> tuple[int, int]:
    """Import all root-level .xls files from a directory."""
    imported_count = skipped_count = 0
    for file_path in find_xls_files(input_dir):
        if import_file(file_path, database_path):
            imported_count += 1
        else:
            skipped_count += 1
    return imported_count, skipped_count


def rebuild_database(database_path: Path) -> None:
    """Remove the generated SQLite file before rebuilding tables."""
    database_path = database_path.expanduser().resolve()
    if database_path.exists():
        database_path.unlink()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line import options."""
    parser = argparse.ArgumentParser(
        description="Import the Modell portfóliók worksheet from .xls files into SQLite."
    )
    parser.add_argument("--input-dir", "-i", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing .xls files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--database", "-d", type=Path, default=DEFAULT_DATABASE,
                        help=f"SQLite file (default: {DEFAULT_DATABASE})")
    parser.add_argument("--rebuild", action="store_true",
                        help="Replace the SQLite database before importing.")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Rank investments and portfolios, then optimize allocations.",
    )
    parser.add_argument(
        "--risk-profile",
        choices=("conservative", "balanced", "dynamic"),
        default="balanced",
        help="Risk profile used for ranking and allocation (default: balanced).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of investment recommendations to return (default: 10).",
    )
    parser.add_argument(
        "--allocation-candidates",
        type=int,
        default=12,
        help="Top candidates considered by the optimizer (default: 12).",
    )
    parser.add_argument(
        "--max-allocation",
        type=float,
        default=20.0,
        help="Maximum allocation per investment as a percentage (default: 20).",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        help="Optional path for the analysis JSON report.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Add an OpenAI explanation; requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--ai-model",
        default="gpt-5.6-terra",
        help="OpenAI model used only with --explain (default: gpt-5.6-terra).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line importer."""
    arguments = parse_arguments()
    try:
        if arguments.rebuild:
            rebuild_database(arguments.database)
        imported, skipped = import_directory(arguments.input_dir, arguments.database)
        if arguments.analyze:
            report = run_analysis(
                arguments.database,
                risk_profile=arguments.risk_profile,
                top_investments=arguments.top,
                allocation_candidates=arguments.allocation_candidates,
                maximum_allocation=arguments.max_allocation / 100,
                explain=arguments.explain,
                model=arguments.ai_model,
            )
            print(write_report(report, arguments.analysis_output))
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        DatabaseError,
        AnalysisError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Done. Imported {imported} file(s), skipped {skipped} already imported file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
