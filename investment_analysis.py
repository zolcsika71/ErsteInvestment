"""Time-aware investment ranking and constrained portfolio allocation.

The model produces research signals, not personalized financial advice.  An LLM
may explain the result, but it never calculates predictions or allocations.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize
import xgboost as xgb


RiskProfile = Literal["conservative", "balanced", "dynamic"]

NUMERIC_FEATURES: Final = (
    "YTD",
    "1 Year",
    "3 Years",
    "5 Years",
    "1Y Sharpe Ratio",
    "3Y Sharpe Ratio",
    "5Y Sharpe Ratio",
    "1Y Volatility",
    "3Y Volatility",
    "Information Ratio",
    "Maximum Drawdown",
)
CATEGORICAL_FEATURES: Final = (
    "Asset Class",
    "Sub-Asset Class",
    "Currency",
    "Currency Risk",
    "Sustainability",
)
IDENTITY_COLUMNS: Final = ("Product", "ISIN")
ENV_FILE: Final = Path(__file__).resolve().parent / ".env"
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


def load_portfolio_data(database_path: Path) -> pd.DataFrame:
    """Load and validate the portfolio snapshots from SQLite."""
    database_path = database_path.expanduser().resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    try:
        with sqlite3.connect(database_path) as connection:
            frame = pd.read_sql_query("SELECT * FROM model_portfolios", connection)
    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        raise AnalysisError(f"Could not read model_portfolios from {database_path}") from error
    if frame.empty:
        raise AnalysisError("The model_portfolios table is empty")

    required = {
        "Date", "Portfolio Name", "Product", "ISIN", "Allocation (%)",
        *NUMERIC_FEATURES, *CATEGORICAL_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AnalysisError(f"Missing analysis columns: {', '.join(missing)}")

    frame["Snapshot Date"] = pd.to_datetime(frame["Date"], format="%Y/%m/%d")
    for column in ("Allocation (%)", *NUMERIC_FEATURES):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def collapse_investment_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse portfolio duplicates into one observation per ISIN and date."""
    aggregations: dict[str, str] = {
        "Date": "first",
        "Product": "first",
        "Asset Class": "first",
        "Sub-Asset Class": "first",
        "Currency": "first",
        "Currency Risk": "first",
        "Sustainability": "first",
        "Allocation (%)": "mean",
    }
    aggregations.update({column: "median" for column in NUMERIC_FEATURES})
    collapsed = (
        frame.groupby(["Snapshot Date", "ISIN"], as_index=False, dropna=False)
        .agg(aggregations)
        .sort_values(["Snapshot Date", "ISIN"])
        .reset_index(drop=True)
    )
    return collapsed


def add_forward_targets(
    frame: pd.DataFrame,
    minimum_days: int = 270,
    maximum_days: int = 455,
) -> pd.DataFrame:
    """Attach the nearest approximately one-year-ahead trailing return.

    A future snapshot's trailing one-year return approximates the realized
    return following the observation date.  ``Target Date`` is retained so the
    time-aware validation split can exclude labels unavailable at training time.
    """
    labeled = frame.copy()
    labeled["Target Return"] = np.nan
    labeled["Target Date"] = pd.NaT

    for _, indexes in labeled.groupby("ISIN").groups.items():
        ordered_indexes = list(
            labeled.loc[indexes].sort_values("Snapshot Date").index
        )
        dates = labeled.loc[ordered_indexes, "Snapshot Date"]
        for position, source_index in enumerate(ordered_indexes[:-1]):
            source_date = dates.loc[source_index]
            candidates: list[tuple[int, int]] = []
            for target_index in ordered_indexes[position + 1:]:
                days = (dates.loc[target_index] - source_date).days
                if days > maximum_days:
                    break
                if days >= minimum_days:
                    candidates.append((abs(days - 365), target_index))
            if not candidates:
                continue
            _, target_index = min(candidates)
            target_return = labeled.at[target_index, "1 Year"]
            if pd.isna(target_return):
                continue
            labeled.at[source_index, "Target Return"] = float(target_return)
            labeled.at[source_index, "Target Date"] = labeled.at[
                target_index, "Snapshot Date"
            ]
    return labeled


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.loc[:, NUMERIC_FEATURES].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    categorical = frame.loc[:, CATEGORICAL_FEATURES].fillna("Unknown").astype(str)
    encoded = pd.get_dummies(
        categorical,
        prefix=list(CATEGORICAL_FEATURES),
        dtype=float,
    )
    return pd.concat(
        [numeric.reset_index(drop=True), encoded.reset_index(drop=True)],
        axis=1,
    )


def _train_regressor(
    features: pd.DataFrame,
    target: pd.Series,
) -> xgb.Booster:
    """Train through XGBoost's native API without a scikit-learn dependency."""
    matrix = xgb.DMatrix(features, label=target)
    return xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.035,
            "min_child_weight": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "alpha": 0.05,
            "lambda": 1.5,
            "seed": 42,
            "nthread": 1,
        },
        matrix,
        num_boost_round=220,
    )


def _predict(model: xgb.Booster, features: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(features))


def train_and_predict(
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, ModelDiagnostics]:
    """Train with time-aware labels and predict the latest snapshot."""
    labeled = add_forward_targets(snapshots)
    train = labeled[labeled["Target Return"].notna()].copy().reset_index(drop=True)
    if len(train) < 50:
        raise AnalysisError(
            f"Only {len(train)} labeled samples are available; at least 50 are required"
        )

    latest_date = snapshots["Snapshot Date"].max()
    latest = snapshots[snapshots["Snapshot Date"] == latest_date].copy()
    combined = pd.concat([train, latest], ignore_index=True)
    all_features = _feature_matrix(combined)
    train_features = all_features.iloc[:len(train)]
    latest_features = all_features.iloc[len(train):]

    observation_dates = sorted(train["Snapshot Date"].unique())
    eligible_splits: list[tuple[pd.Timestamp, pd.Series, pd.Series]] = []
    for raw_cutoff in observation_dates:
        cutoff = pd.Timestamp(raw_cutoff)
        candidate_fit = train["Target Date"] < cutoff
        candidate_validation = train["Snapshot Date"] >= cutoff
        if candidate_fit.sum() >= 50 and candidate_validation.sum() >= 30:
            eligible_splits.append((
                cutoff,
                candidate_fit,
                candidate_validation,
            ))
    if not eligible_splits:
        raise AnalysisError(
            "No leak-free time split has at least 50 training and 30 validation samples"
        )
    _, fit_mask, validation_mask = eligible_splits[-1]
    validation_mae: float | None = None
    validation_count = int(validation_mask.sum())
    validation_model = _train_regressor(
        train_features.loc[fit_mask],
        train.loc[fit_mask, "Target Return"],
    )
    validation_prediction = _predict(
        validation_model,
        train_features.loc[validation_mask],
    )
    validation_mae = float(np.mean(np.abs(
        validation_prediction - train.loc[validation_mask, "Target Return"]
    )))

    model = _train_regressor(train_features, train["Target Return"])
    latest["Predicted Return"] = _predict(model, latest_features)
    latest["Risk Score"] = (
        latest["3Y Volatility"]
        .fillna(latest["1Y Volatility"])
        .fillna(latest["3Y Volatility"].median())
        .fillna(0.0)
        .clip(lower=0.0)
    )
    latest["Model Score"] = (
        latest["Predicted Return"] - 0.35 * latest["Risk Score"]
    )
    diagnostics = ModelDiagnostics(
        training_samples=len(train),
        validation_samples=validation_count,
        validation_mae=validation_mae,
        latest_date=latest_date.strftime("%Y/%m/%d"),
        target="nearest 270-455 day forward one-year return",
    )
    return latest.sort_values("Model Score", ascending=False), diagnostics


def rank_investments(
    predictions: pd.DataFrame,
    limit: int,
) -> list[InvestmentRecommendation]:
    """Return the highest-ranked latest-date investments."""
    recommendations: list[InvestmentRecommendation] = []
    for rank, (_, row) in enumerate(predictions.head(limit).iterrows(), start=1):
        recommendations.append(InvestmentRecommendation(
            rank=rank,
            product=str(row["Product"]),
            isin=str(row["ISIN"]),
            asset_class=str(row["Asset Class"]),
            currency=str(row["Currency"]),
            predicted_return=float(row["Predicted Return"]),
            risk_score=float(row["Risk Score"]),
            model_score=float(row["Model Score"]),
        ))
    return recommendations


def rank_portfolios(
    raw_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    risk_profile: RiskProfile,
) -> list[PortfolioRecommendation]:
    """Rank latest named portfolios using predicted return and observed risk."""
    latest_date = raw_frame["Snapshot Date"].max()
    latest = raw_frame[raw_frame["Snapshot Date"] == latest_date].copy()
    prediction_columns = ["ISIN", "Predicted Return", "Risk Score"]
    merged = latest.merge(
        predictions.loc[:, prediction_columns],
        on="ISIN",
        how="left",
    )
    results: list[PortfolioRecommendation] = []
    penalty = RISK_AVERSION[risk_profile]
    for portfolio_name, group in merged.groupby("Portfolio Name"):
        total_allocation = group["Allocation (%)"].sum()
        covered = group[group["Predicted Return"].notna()].copy()
        covered_allocation = covered["Allocation (%)"].sum()
        if covered.empty or total_allocation <= 0 or covered_allocation <= 0:
            continue
        weights = covered["Allocation (%)"] / covered_allocation
        expected_return = float(np.dot(weights, covered["Predicted Return"]))
        volatility = float(np.dot(weights, covered["Risk Score"]))
        concentration = float(np.square(weights).sum())
        score = expected_return - penalty * volatility - 0.02 * concentration
        results.append(PortfolioRecommendation(
            rank=0,
            portfolio_name=str(portfolio_name),
            expected_return=expected_return,
            expected_volatility=volatility,
            concentration=concentration,
            score=score,
            coverage=float(covered_allocation / total_allocation),
        ))
    results.sort(key=lambda item: item.score, reverse=True)
    return [
        PortfolioRecommendation(rank=index, **{
            key: value
            for key, value in asdict(item).items()
            if key != "rank"
        })
        for index, item in enumerate(results, start=1)
    ]


def _return_covariance(
    snapshots: pd.DataFrame,
    candidate_isins: list[str],
) -> np.ndarray:
    history = snapshots[snapshots["ISIN"].isin(candidate_isins)].pivot_table(
        index="Snapshot Date",
        columns="ISIN",
        values="1 Year",
        aggfunc="median",
    )
    changes = history.diff()
    covariance = changes.cov(min_periods=3).reindex(
        index=candidate_isins,
        columns=candidate_isins,
    )
    diagonal_fallback = float(
        np.nanmedian(np.diag(covariance.to_numpy(dtype=float)))
    )
    if not np.isfinite(diagonal_fallback) or diagonal_fallback <= 0:
        diagonal_fallback = 0.0025
    matrix = covariance.fillna(0.0).to_numpy(dtype=float, copy=True)
    diagonal = np.diag(matrix).copy()
    diagonal[~np.isfinite(diagonal) | (diagonal <= 0)] = diagonal_fallback
    np.fill_diagonal(matrix, diagonal)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < 1e-8:
        matrix += np.eye(len(matrix)) * (1e-8 - eigenvalues.min())
    return matrix


def optimize_allocations(
    snapshots: pd.DataFrame,
    predictions: pd.DataFrame,
    risk_profile: RiskProfile,
    limit: int,
    maximum_allocation: float,
) -> list[AllocationRecommendation]:
    """Optimize long-only allocations among the top-ranked investments."""
    if not 0 < maximum_allocation <= 1:
        raise ValueError("maximum_allocation must be between 0 and 1")
    minimum_assets = math.ceil(1 / maximum_allocation)
    candidate_count = min(max(limit, minimum_assets), len(predictions))
    top_candidates = predictions.head(candidate_count)
    class_representatives = predictions.groupby(
        "Asset Class",
        sort=False,
        group_keys=False,
    ).head(2)
    candidates = (
        pd.concat([top_candidates, class_representatives])
        .drop_duplicates("ISIN")
        .sort_values("Model Score", ascending=False)
        .reset_index(drop=True)
    )
    if len(candidates) < minimum_assets:
        raise AnalysisError(
            f"At least {minimum_assets} candidates are needed for the allocation cap"
        )

    returns = candidates["Predicted Return"].to_numpy(dtype=float)
    covariance = _return_covariance(
        snapshots,
        candidates["ISIN"].astype(str).tolist(),
    )
    risk_aversion = RISK_AVERSION[risk_profile] * 10.0
    concentration_penalty = 0.02

    def objective(weights: np.ndarray) -> float:
        expected_return = float(weights @ returns)
        variance = float(weights @ covariance @ weights)
        concentration = float(weights @ weights)
        return -expected_return + risk_aversion * variance + concentration_penalty * concentration

    initial = np.full(len(candidates), 1 / len(candidates))
    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
    ]
    asset_caps = {
        asset_class: ASSET_CLASS_CAPS[risk_profile].get(
            str(asset_class),
            DEFAULT_ASSET_CLASS_CAP,
        )
        for asset_class in candidates["Asset Class"].dropna().unique()
    }
    if len(asset_caps) > 1 and sum(asset_caps.values()) >= 1:
        for asset_class, cap in asset_caps.items():
            indexes = np.flatnonzero(
                candidates["Asset Class"].to_numpy() == asset_class
            )
            constraints.append({
                "type": "ineq",
                "fun": lambda weights, indexes=indexes, cap=cap: (
                    cap - weights[indexes].sum()
                ),
            })
    currencies = candidates["Currency"].dropna().unique()
    if len(currencies) > 1 and len(currencies) * DEFAULT_CURRENCY_CAP >= 1:
        for currency in currencies:
            indexes = np.flatnonzero(candidates["Currency"].to_numpy() == currency)
            constraints.append({
                "type": "ineq",
                "fun": lambda weights, indexes=indexes: (
                    DEFAULT_CURRENCY_CAP - weights[indexes].sum()
                ),
            })

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, maximum_allocation)] * len(candidates),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise AnalysisError(f"Allocation optimization failed: {result.message}")

    weights = np.where(result.x < 0.0005, 0.0, result.x)
    weights /= weights.sum()
    allocation_rows: list[AllocationRecommendation] = []
    for index in np.argsort(weights)[::-1]:
        if weights[index] <= 0:
            continue
        row = candidates.iloc[index]
        allocation_rows.append(AllocationRecommendation(
            product=str(row["Product"]),
            isin=str(row["ISIN"]),
            asset_class=str(row["Asset Class"]),
            currency=str(row["Currency"]),
            allocation_percent=float(weights[index] * 100),
            predicted_return=float(row["Predicted Return"]),
            risk_score=float(row["Risk Score"]),
        ))
    return allocation_rows


def explain_with_openai(
    report: AnalysisReport,
    model: str,
) -> dict[str, Any]:
    """Ask OpenAI to explain—but not change—the deterministic recommendation."""
    try:
        import openai
        from openai import OpenAI
        from pydantic import BaseModel
    except ImportError as error:
        raise AnalysisError("Install the openai package to use --explain") from error
    load_dotenv(ENV_FILE, override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise AnalysisError(
            f"Set OPENAI_API_KEY in {ENV_FILE} when --explain is used"
        )

    class Explanation(BaseModel):
        summary: str
        best_medium_term_portfolio: str
        key_reasons: list[str]
        allocation_commentary: list[str]
        risks_and_limitations: list[str]
        disclaimer: str

    payload = report.to_dict()
    payload["explanation"] = None
    try:
        response = OpenAI().responses.parse(
            model=model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an investment-research explainer. Explain only the "
                        "provided deterministic analysis. Do not change rankings, "
                        "scores, or allocations. Do not promise future performance. "
                        "State that this is research, not personalized financial advice."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                },
            ],
            text_format=Explanation,
        )
    except openai.AuthenticationError as error:
        raise AnalysisError(
            "OpenAI rejected the API key. Check OPENAI_API_KEY in .env and "
            "confirm that the key belongs to the intended API project."
        ) from error
    except openai.RateLimitError as error:
        body = error.body if isinstance(error.body, dict) else {}
        if body.get("code") == "insufficient_quota":
            raise AnalysisError(
                "OpenAI API quota is unavailable. Add API credits or raise the "
                "organization/project spend limit, then retry. You can run "
                "without --explain to generate the local quantitative report."
            ) from error
        raise AnalysisError(
            "OpenAI rate limit reached. Wait briefly and retry, or run without "
            "--explain."
        ) from error
    except openai.APITimeoutError as error:
        raise AnalysisError(
            "The OpenAI request timed out. Retry or run without --explain."
        ) from error
    except openai.APIConnectionError as error:
        raise AnalysisError(
            "Could not connect to OpenAI. Check the network connection or run "
            "without --explain."
        ) from error
    except openai.APIStatusError as error:
        raise AnalysisError(
            f"OpenAI API request failed with HTTP {error.status_code}. "
            "Check model access and project permissions, or run without --explain."
        ) from error
    except openai.APIError as error:
        raise AnalysisError(
            "OpenAI API request failed. Retry or run without --explain."
        ) from error
    if response.output_parsed is None:
        raise AnalysisError("OpenAI returned no structured explanation")
    return response.output_parsed.model_dump()


def run_analysis(
    database_path: Path,
    risk_profile: RiskProfile = "balanced",
    top_investments: int = 10,
    allocation_candidates: int = 12,
    maximum_allocation: float = 0.20,
    explain: bool = False,
    model: str = "gpt-5.6-terra",
) -> AnalysisReport:
    """Run ranking, portfolio comparison, optimization, and optional explanation."""
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
        warnings=[
            "Forecasts are experimental and do not guarantee future performance.",
            "The database contains sparse, irregular snapshots rather than daily NAV history.",
            "Covariance is estimated from changes in overlapping trailing returns.",
            "Allocation uses hard per-investment, asset-class, and currency caps.",
            "Transaction costs, taxes, liquidity, and external macro data are not modeled.",
            "This output is research and not personalized financial advice.",
        ],
    )
    if explain:
        explanation = explain_with_openai(report, model)
        report = replace(report, explanation=explanation)
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
