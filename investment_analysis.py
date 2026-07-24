"""Public facade for investment ranking and portfolio allocation.

The implementation is split by responsibility:

- :mod:`investment_model` loads data, engineers features, and predicts returns.
- :mod:`portfolio_engine` ranks portfolios and optimizes allocations.
- :mod:`openai_explainer` optionally explains deterministic results.
- :mod:`analysis_types` owns shared configuration and result types.

Imports from this module remain supported for backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from analysis_types import (
    RISK_AVERSION,
    AllocationRecommendation,
    AnalysisError,
    AnalysisReport,
    InvestmentRecommendation,
    ModelDiagnostics,
    PortfolioRecommendation,
    RiskProfile,
)
from investment_model import (
    add_forward_targets,
    collapse_investment_snapshots,
    load_portfolio_data,
    rank_investments,
    train_and_predict,
)
from openai_explainer import explain_with_openai
from portfolio_engine import optimize_allocations, rank_portfolios
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
    report = AnalysisReport(
        as_of_date=diagnostics.latest_date,
        risk_profile=risk_profile,
        diagnostics=diagnostics,
        investments=rank_investments(predictions, top_investments),
        portfolios=rank_portfolios(raw, predictions, risk_profile),
        allocations=optimize_allocations(
            snapshots,
            predictions,
            risk_profile,
            allocation_candidates,
            maximum_allocation,
        ),
        warnings=list(ANALYSIS_WARNINGS),
    )
    if explain:
        report = replace(
            report,
            explanation=explain_with_openai(report, model),
        )
    return report


def write_report(report: AnalysisReport, output_path: Path | None) -> str:
    """Serialize a report, optionally writing it to a JSON file."""
    content = json.dumps(
        report.to_dict(),
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
