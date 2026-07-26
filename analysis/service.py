"""Public facade for investment ranking and portfolio allocation.

Responsibility splits the implementation:

-:mod:`analysis.model` loads data, engineers features, and predicts returns.
-:mod:`analysis.portfolio` ranks portfolios and optimizes allocations.
-:mod:`analysis.openai_explainer` optionally explains deterministic results.
-:mod:`analysis.types` owns shared configuration and result types.

Imports from this module remain supported for backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from .types import (
    RISK_AVERSION,
    AllocationRecommendation,
    AnalysisError,
    AnalysisReport,
    InvestmentRecommendation,
    ModelDiagnostics,
    PortfolioRecommendation,
    RiskProfile,
)
from .model import (
    add_forward_targets,
    collapse_investment_snapshots,
    load_portfolio_data,
    rank_investments,
    train_and_predict,
)
from .openai_explainer import explain_with_openai, select_best_portfolio_with_openai
from .portfolio import optimize_allocations, rank_portfolios
from project_config import ENV_PATH


# Compatibility alias retained for callers of the original monolithic module.
ENV_FILE = ENV_PATH

ANALYSIS_WARNINGS = (
    "Forecasts are experimental and do not guarantee future performance.",
    "The database contains sparse, irregular snapshots rather than daily NAV history.",
    "Covariance is estimated from changes in overlapping trailing returns.",
    "Allocation uses hard per-investment, asset-class, and currency caps.",
    "Transaction costs, taxes, liquidity, and external macro data are not modeled.",
    "This output is research and not personalized financial advice.",
)


def _json_safe_value(value: object) -> object:
    """Convert pandas/Python scalar values into strict-JSON values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        # Convert NumPy scalar values such as float64 and int64.
        return value.item()
    return value


def _json_safe_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return asset records without pandas timestamps or NumPy scalars."""
    return [
        {key: _json_safe_value(value) for key, value in record.items()}
        for record in records
    ]


def run_analysis(
    database_path: Path,
    risk_profile: RiskProfile = "balanced",
    top_investments: int = 10,
    allocation_candidates: int = 12,
    maximum_allocation: float = 0.20,
    explain: bool = False,
    model: str = "gpt-5.6-terra",
) -> AnalysisReport:
    """Run prediction, ranking, optimization, and optional explanation."""
    if risk_profile not in RISK_AVERSION:
        raise ValueError(f"Unsupported risk profile: {risk_profile}")
    if top_investments < 1 or allocation_candidates < 1:
        raise ValueError("Recommendation limits must be positive")

    raw = load_portfolio_data(database_path)
    snapshots = collapse_investment_snapshots(raw)
    predictions, diagnostics = train_and_predict(snapshots)
    portfolios = rank_portfolios(raw, predictions, risk_profile)
    if not portfolios:
        raise AnalysisError("No eligible portfolios were found")

    # The local ranking still computes the same candidate metrics. When AI is
    # enabled, OpenAI selects among those candidates using preservation-first
    # priorities; it cannot create or modify portfolio metrics.
    if explain:
        selection = select_best_portfolio_with_openai(
            [
                {
                    "portfolio_name": item.portfolio_name,
                    "expected_return": item.expected_return,
                    "expected_volatility": item.expected_volatility,
                    "concentration": item.concentration,
                    "score": item.score,
                    "coverage": item.coverage,
                }
                for item in portfolios
            ],
            model,
        )
        selected_index = next(
            index
            for index, item in enumerate(portfolios)
            if item.portfolio_name == selection
        )
        portfolios = [portfolios[selected_index], *portfolios[:selected_index],
                      *portfolios[selected_index + 1:]]

    best_portfolio = portfolios[0]
    latest = raw[raw["Snapshot Date"] == raw["Snapshot Date"].max()]
    best_assets = latest[
        latest["Portfolio Name"] == best_portfolio.portfolio_name
    ].copy()
    best_assets = best_assets.merge(
        predictions.loc[:, ["ISIN", "Predicted Return", "Risk Score"]],
        on="ISIN",
        how="left",
    )
    asset_coverage = float(best_assets["Predicted Return"].notna().mean())
    # Convert missing numeric values to JSON null instead of NaN, because the
    # export intentionally uses strict JSON (allow_nan=False).
    assets = _json_safe_records(
        best_assets.astype(object)
        .where(best_assets.notna(), None)
        .to_dict(orient="records")
    )

    report = AnalysisReport(
        as_of_date=diagnostics.latest_date,
        risk_profile=risk_profile,
        diagnostics=diagnostics,
        investments=rank_investments(predictions, top_investments),
        portfolios=portfolios,
        allocations=optimize_allocations(
            snapshots,
            predictions,
            risk_profile,
            allocation_candidates,
            maximum_allocation,
        ),
        warnings=list(ANALYSIS_WARNINGS),
        best_portfolio={
            key: value for key, value in {
                "portfolio_name": best_portfolio.portfolio_name,
                "expected_return": best_portfolio.expected_return,
                "expected_volatility": best_portfolio.expected_volatility,
                "concentration": best_portfolio.concentration,
                "score": best_portfolio.score,
                "coverage": best_portfolio.coverage,
                "asset_coverage": asset_coverage,
            }.items()
        },
        assets=assets,
    )
    if explain:
        report = replace(
            report,
            explanation=explain_with_openai(report, model),
        )
    return report


def write_report(report: AnalysisReport, output_path: Path | None) -> str:
    """Write only the selected portfolio and its assets to JSON."""
    if report.best_portfolio is None:
        raise AnalysisError("Analysis report does not contain a best portfolio")

    # Keep the JSON aligned with the Excel input. Ranked alternatives,
    # optimizer candidates, diagnostics, and warnings are intentionally not
    # exported because they are unrelated to the selected portfolio summary.
    export = {
        "as_of_date": report.as_of_date,
        "risk_profile": report.risk_profile,
        "portfolio": report.best_portfolio,
        "assets": report.assets,
        "explanation": report.explanation,
    }
    content = json.dumps(
        export,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content + "\n", encoding="utf-8")
    return content


__all__ = [
    "AllocationRecommendation",
    "AnalysisError",
    "AnalysisReport",
    "InvestmentRecommendation",
    "ModelDiagnostics",
    "PortfolioRecommendation",
    "RiskProfile",
    "add_forward_targets",
    "collapse_investment_snapshots",
    "explain_with_openai",
    "load_portfolio_data",
    "optimize_allocations",
    "rank_investments",
    "rank_portfolios",
    "run_analysis",
    "train_and_predict",
    "write_report",
]
