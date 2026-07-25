# Erste Investment Analysis

Import Erste model-portfolio spreadsheets into SQLite, rank investments with a
time-aware XGBoost model, compare named portfolios, calculate constrained
allocations, optionally generate an OpenAI explanation, and export the result to
a formatted Excel workbook.

> The generated rankings and allocations are experimental research signals.
> They are not personalized financial advice and do not guarantee future
> performance.

## Requirements

- Python 3.13 or newer
- [Poetry](https://python-poetry.org/)
- `.xls` source workbooks
- An OpenAI API key only when using `--explain`

Install all runtime and development dependencies:

```bash
poetry install
```

## Project folders

```text
.
├── analysis/                # Modeling, ranking, optimization, OpenAI explanation
├── db_creation/             # Excel processing and SQLite import
├── db/                      # Generated SQLite database and analysis JSON
├── model_portfolios_xls/    # Local source .xls files
├── results/                 # Generated Excel reports
├── tests/                   # Pytest suite
├── export_analysis_excel.py
├── main.py
├── project_config.py
└── pyproject.toml
```

All configured paths are derived from the directory containing
`project_config.py`. If the application is installed in `/project`, the default
paths are:

```text
/project/model_portfolios_xls
/project/db/model_portfolio.sqlite
/project/db/investment_analysis.json
/project/results/investment_analysis.xlsx
```

Source workbooks, generated databases, reports, `.env`, and IDE files are
ignored by Git.

## Import spreadsheets

Place `.xls` files directly inside `model_portfolios_xls/`, then run:

```bash
poetry run python main.py
```

Import behavior:

- Only root-level `.xls` files are processed.
- Only the visible `Modell portfóliók` worksheet is imported.
- The filename must end with an eight-digit date such as `20240702.xls`.
- The date is stored as `2024/07/02`.
- Hungarian headers and configured categorical values are translated to English.
- Numeric zero values are imported as SQLite `NULL`.
- A file is skipped when its date already exists in `model_portfolios`.
- Each workbook is imported in one transaction.

The default database is:

```text
db/model_portfolio.sqlite
```

Rebuild the database after schema or translation changes:

```bash
poetry run python main.py --rebuild
```

Use custom locations when required:

```bash
poetry run python main.py \
  --input-dir /path/to/xls/files \
  --database /path/to/model_portfolio.sqlite
```

## Run quantitative analysis

The `--analyze` option imports any new workbooks and then runs the analysis:

```bash
poetry run python main.py --analyze \
  --risk-profile balanced \
  --top 10 \
  --allocation-candidates 12 \
  --max-allocation 20 \
  --analysis-output db/investment_analysis.json
```

Available risk profiles:

- `conservative`
- `balanced` (default)
- `dynamic`

The analysis pipeline:

1. Collapses repeated portfolio rows into one ISIN/date observation.
2. Creates forward labels using the nearest snapshot 270–455 days later.
3. Trains XGBoost with a leak-resistant chronological validation split.
4. Predicts and ranks investments at the newest database date.
5. Scores each named portfolio using predicted return, risk, and concentration.
6. Uses constrained SLSQP optimization to allocate exactly 100%.

The optimizer enforces:

- A maximum allocation per investment
- Risk-profile-specific asset-class limits
- A currency-exposure limit
- Candidate coverage across available asset classes

## Optional OpenAI explanation

The OpenAI model explains the deterministic report. It cannot modify calculated
rankings, scores, or allocations.

Add the token to the Git-ignored `.env` file:

```dotenv
OPENAI_API_KEY=sk-your-key-here
```

Then run:

```bash
poetry run python main.py --analyze --explain \
  --analysis-output db/investment_analysis.json
```

The default explanatory model is `gpt-5.6-terra`. Select another accessible
model with:

```bash
poetry run python main.py --analyze --explain \
  --ai-model your-model-name
```

A shell or PyCharm environment variable takes precedence over `.env`. Without
`--explain`, the analysis is local and makes no OpenAI request.

If OpenAI reports `insufficient_quota`, add API credits or adjust the
organization/project spend limit. The quantitative analysis remains available
without `--explain`.

## Export the Excel report

Convert `db/investment_analysis.json` into a formatted workbook:

```bash
poetry run python export_analysis_excel.py
```

Default output:

```text
results/investment_analysis.xlsx
```

The workbook contains:

- Summary
- Investments
- Portfolios
- Allocations
- Warnings
- AI Explanation

Custom input and output paths can be passed positionally:

```bash
poetry run python export_analysis_excel.py \
  /path/to/investment_analysis.json \
  /path/to/investment_analysis.xlsx
```

## Tests

The pytest suite is located in `tests/`:

```bash
poetry run pytest
```

Show individual test names:

```bash
poetry run pytest -v
```

## Command-line reference

```bash
poetry run python main.py --help
poetry run python export_analysis_excel.py --help
```

Important analysis options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--risk-profile` | `balanced` | Risk penalty and asset-class limits |
| `--top` | `10` | Number of ranked investments |
| `--allocation-candidates` | `12` | Top candidates considered by optimizer |
| `--max-allocation` | `20` | Maximum percentage in one investment |
| `--analysis-output` | `none` | Optional JSON output path |
| `--explain` | `disabled` | Request an OpenAI explanation |
| `--ai-model` | `gpt-5.6-terra` | Model used for explanation |

## Code structure

- `project_config.py` — shared project paths.
- `db_creation/database_create.py` — SQLite schema and import transactions.
- `db_creation/excel_processing.py` — workbook reading and data normalization.
- `db_creation/text_normalization.py` — shared Unicode lookup normalization.
- `analysis/model.py` — data loading, forward labels, XGBoost, and ranking.
- `analysis/portfolio.py` — portfolio scoring and allocation optimization.
- `analysis/openai_explainer.py` — structured OpenAI explanation and API errors.
- `analysis/types.py` — analysis configuration and result dataclasses.
- `analysis/service.py` — public analysis workflow.
- `export_analysis_excel.py` — formatted JSON-to-Excel reporting.
- `main.py` — command-line import and analysis orchestration.
- `tests/` — pytest test suite.

## Modeling limitations

The database contains irregular portfolio snapshots and overlapping trailing
performance metrics rather than daily NAV histories. Consequently:

- Forward returns are approximated from later snapshots.
- Covariance is estimated from changes in overlapping trailing returns.
- Transaction costs, taxes, liquidity, and macroeconomic data are not modeled.
- Validation results may not generalize to future market regimes.

For production investment decisions, add reliable daily or monthly NAV history,
benchmarks, transaction costs, and independent out-of-sample evaluation.
