# Erste Investment analysis

The project imports `/project/model_portfolios_xls/*.xls` into
`/project/db/model_portfolio.sqlite`. Paths are derived from the folder that
contains `main.py`, so the same code also works outside `/project`.

## Import

```bash
poetry run python main.py
```

Use `--rebuild` after an import schema or translation change.

## Quantitative analysis

```bash
poetry run python main.py --analyze \
  --risk-profile balanced \
  --top 10 \
  --max-allocation 20 \
  --analysis-output db/investment_analysis.json
```

The analysis:

1. Collapses duplicate portfolio rows into one ISIN/date observation.
2. Uses the nearest snapshot 270–455 days later as a forward training label.
3. Trains an XGBoost return model with a time-aware validation split.
4. Ranks the investments and named portfolios at the latest date.
5. Uses constrained SLSQP optimization to allocate 100% across top candidates.
   The optimizer enforces per-investment, risk-profile asset-class, and currency
   caps and includes candidates from every available asset class.

The results are experimental research signals. Sparse snapshots and overlapping
trailing-return metrics are not substitutes for daily NAV history.

## Optional OpenAI explanation

The LLM explains deterministic results; it cannot change ranks or allocations.
Add the key to the Git-ignored `.env` file in the project directory:

```dotenv
OPENAI_API_KEY=sk-your-key-here
```

Then run:

```bash
poetry run python main.py --analyze --explain
```

The default explanatory model is `gpt-5.6-terra`. Override it with
`--ai-model`. A key supplied by the shell or PyCharm takes precedence over
`.env`. Without `--explain`, analysis is local and makes no API request.

## Excel report

Convert the generated JSON report into a formatted, multi-sheet workbook:

```bash
poetry run python export_analysis_excel.py
```

The default output is `results/investment_analysis.xlsx`.

## Tests

All tests are located in `tests/` and run with pytest:

```bash
poetry run pytest
```

## Code structure

- `project_config.py` — shared project paths.
- `db_creation/` — worksheet processing, translations, and SQLite imports.
- `analysis/` — model training, portfolio optimization, OpenAI explanation,
  shared result types, and workflow orchestration.
- `export_analysis_excel.py` — formatted JSON-to-Excel reporting.
- `main.py` — command-line interface.
- `tests/` — pytest test suite.
