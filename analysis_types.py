"""Shared types and configuration for investment analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal


RiskProfile = Literal["conservative", "balanced", "dynamic"]

NUMERIC_FEATURES: Final = (
    "YTD", "1 Year", "3 Years", "5 Years", "1Y Sharpe Ratio",
    "3Y Sharpe Ratio", "5Y Sharpe Ratio", "1Y Volatility",
    "3Y Volatility", "Information Ratio", "Maximum Drawdown",
)
CATEGORICAL_FEATURES: Final = (
    "Asset Class", "Sub-Asset Class", "Currency", "Currency Risk",
    "Sustainability",
)
RISK_AVERSION: Final = {
    "conservative": 1.50,
    "balanced": 0.55,
    "dynamic": 0.25,
}
ASSET_CLASS_CAPS: Final = {
    "conservative": {"Equity": 0.35, "Alternative": 0.15},
    "balanced": {"Equity": 0.50, "Alternative": 0.20},
    "dynamic": {"Equity": 0.70, "Alternative": 0.25},
}
DEFAULT_ASSET_CLASS_CAP: Final = 0.60
DEFAULT_CURRENCY_CAP: Final = 0.60


class AnalysisError(Exception):
    """Report invalid data or a failed analysis operation."""


@dataclass(frozen=True)
class ModelDiagnostics:
    training_samples: int
    validation_samples: int
    validation_mae: float | None
    latest_date: str
    target: str


@dataclass(frozen=True)
class InvestmentRecommendation:
    rank: int
    product: str
    isin: str
    asset_class: str
    currency: str
    predicted_return: float
    risk_score: float
    model_score: float


@dataclass(frozen=True)
class PortfolioRecommendation:
    rank: int
    portfolio_name: str
    expected_return: float
    expected_volatility: float
    concentration: float
    score: float
    coverage: float


@dataclass(frozen=True)
class AllocationRecommendation:
    product: str
    isin: str
    asset_class: str
    currency: str
    allocation_percent: float
    predicted_return: float
    risk_score: float


@dataclass(frozen=True)
class AnalysisReport:
    as_of_date: str
    risk_profile: str
    diagnostics: ModelDiagnostics
    investments: list[InvestmentRecommendation]
    portfolios: list[PortfolioRecommendation]
    allocations: list[AllocationRecommendation]
    warnings: list[str]
    explanation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)
