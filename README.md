# Erste Investment analysis

The project imports `/project/model_portfolios_xls/*.xls` into
`/project/DB/model_portfolio.sqlite`. Paths are derived from the folder that
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
  --analysis-output DB/investment_analysis.json
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

The default output is `DB/investment_analysis.xlsx`.
