"""Named-portfolio scoring and constrained allocation optimization."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import TypedDict

# noinspection PyPackageRequirements
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from .types import (
    ASSET_CLASS_CAPS,
    DEFAULT_ASSET_CLASS_CAP,
    DEFAULT_CURRENCY_CAP,
    RISK_AVERSION,
    AllocationRecommendation,
    AnalysisError,
    PortfolioRecommendation,
    RiskProfile,
)


# noinspection SpellCheckingInspection
class _SlsqpOptions(TypedDict):
    ftol: float
    maxiter: int


def rank_portfolios(
    raw_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    risk_profile: RiskProfile,
) -> list[PortfolioRecommendation]:
    """Rank latest named portfolios using predicted return and observed risk."""
    latest_date = raw_frame["Snapshot Date"].max()
    latest = raw_frame[raw_frame["Snapshot Date"] == latest_date].copy()
    merged = latest.merge(
        predictions.loc[:, ["ISIN", "Predicted Return", "Risk Score"]],
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
        results.append(PortfolioRecommendation(
            rank=0,
            portfolio_name=str(portfolio_name),
            expected_return=expected_return,
            expected_volatility=volatility,
            concentration=concentration,
            score=expected_return - penalty * volatility - 0.02 * concentration,
            coverage=float(covered_allocation / total_allocation),
        ))
    results.sort(key=lambda item: item.score, reverse=True)
    return [
        PortfolioRecommendation(
            rank=index,
            **{key: value for key, value in asdict(item).items() if key != "rank"},
        )
        for index, item in enumerate(results, start=1)
    ]


def _return_covariance(
    snapshots: pd.DataFrame,
    candidate_identifiers: list[str],
) -> np.ndarray:
    history = snapshots[
        snapshots["ISIN"].isin(candidate_identifiers)
    ].pivot_table(
        index="Snapshot Date",
        columns="ISIN",
        values="1 Year",
        aggfunc="median",
    )
    covariance = history.diff().cov(min_periods=3).reindex(
        index=candidate_identifiers,
        columns=candidate_identifiers,
    )
    fallback = float(np.nanmedian(np.diag(covariance.to_numpy(dtype=float))))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.0025
    matrix = covariance.fillna(0.0).to_numpy(dtype=float, copy=True)
    diagonal = np.diag(matrix).copy()
    diagonal[~np.isfinite(diagonal) | (diagonal <= 0)] = fallback
    np.fill_diagonal(matrix, diagonal)
    minimum_eigenvalue = np.linalg.eigvalsh(matrix).min()
    if minimum_eigenvalue < 1e-8:
        matrix += np.eye(len(matrix)) * (1e-8 - minimum_eigenvalue)
    return matrix


def _select_candidates(
    predictions: pd.DataFrame,
    limit: int,
    minimum_assets: int,
) -> pd.DataFrame:
    count = min(max(limit, minimum_assets), len(predictions))
    representatives = predictions.groupby(
        "Asset Class",
        sort=False,
        group_keys=False,
    ).head(2)
    candidates = (
        pd.concat([predictions.head(count), representatives])
        .drop_duplicates("ISIN")
        .sort_values("Model Score", ascending=False)
        .reset_index(drop=True)
    )
    if len(candidates) < minimum_assets:
        raise AnalysisError(
            f"At least {minimum_assets} candidates are needed for the allocation cap"
        )
    return candidates


def _allocation_constraints(
    candidates: pd.DataFrame,
    risk_profile: RiskProfile,
) -> list[LinearConstraint]:
    candidate_count = len(candidates)
    constraints = [
        LinearConstraint(np.ones(candidate_count), lb=1.0, ub=1.0),
    ]
    asset_caps = {
        asset_class: ASSET_CLASS_CAPS[risk_profile].get(
            str(asset_class),
            DEFAULT_ASSET_CLASS_CAP,
        )
        for asset_class in candidates["Asset Class"].dropna().unique()
    }
    if len(asset_caps) > 1 and sum(asset_caps.values()) >= 1:
        for asset_class, asset_cap in asset_caps.items():
            positions = np.flatnonzero(
                candidates["Asset Class"].to_numpy() == asset_class
            )
            coefficients = np.zeros(candidate_count)
            coefficients[positions] = 1.0
            constraints.append(
                LinearConstraint(coefficients, lb=-np.inf, ub=asset_cap)
            )
    currencies = candidates["Currency"].dropna().unique()
    if len(currencies) > 1 and len(currencies) * DEFAULT_CURRENCY_CAP >= 1:
        for currency in currencies:
            positions = np.flatnonzero(
                candidates["Currency"].to_numpy() == currency
            )
            coefficients = np.zeros(candidate_count)
            coefficients[positions] = 1.0
            constraints.append(
                LinearConstraint(
                    coefficients,
                    lb=-np.inf,
                    ub=DEFAULT_CURRENCY_CAP,
                )
            )
    return constraints


def optimize_allocations(
    snapshots: pd.DataFrame,
    predictions: pd.DataFrame,
    risk_profile: RiskProfile,
    limit: int,
    maximum_allocation: float,
) -> list[AllocationRecommendation]:
    """Optimize long-only allocations among diversified top candidates."""
    if not 0 < maximum_allocation <= 1:
        raise ValueError("maximum_allocation must be between 0 and 1")
    minimum_assets = math.ceil(1 / maximum_allocation)
    candidates = _select_candidates(predictions, limit, minimum_assets)
    returns = candidates["Predicted Return"].to_numpy(dtype=float)
    covariance = _return_covariance(
        snapshots,
        candidates["ISIN"].astype(str).tolist(),
    )
    risk_aversion = RISK_AVERSION[risk_profile] * 10.0

    def objective(allocation: np.ndarray) -> float:
        return float(
            -float(allocation @ returns)
            + risk_aversion * float(allocation @ covariance @ allocation)
            + 0.02 * float(allocation @ allocation)
        )

    initial_allocation: np.ndarray = np.full(
        len(candidates),
        1.0 / len(candidates),
        dtype=np.float64,
    )
    bounds: list[tuple[float, float]] = [
        (0.0, maximum_allocation)
    ] * len(candidates)
    # noinspection SpellCheckingInspection
    options: _SlsqpOptions = {"ftol": 1e-12, "maxiter": 1_000}
    # PyCharm cannot resolve the generic minimized overload from scipy-stubs.
    # noinspection PyTypeChecker
    result = minimize(
        objective,
        initial_allocation,
        method="SLSQP",
        bounds=bounds,
        constraints=_allocation_constraints(candidates, risk_profile),
        options=options,
    )
    if not result.success:
        raise AnalysisError(f"Allocation optimization failed: {result.message}")

    weights = np.where(result.x < 0.0005, 0.0, result.x)
    weights /= weights.sum()
    recommendations: list[AllocationRecommendation] = []
    for index in np.argsort(weights)[::-1]:
        if weights[index] <= 0:
            continue
        row = candidates.iloc[index]
        recommendations.append(AllocationRecommendation(
            product=str(row["Product"]),
            isin=str(row["ISIN"]),
            asset_class=str(row["Asset Class"]),
            currency=str(row["Currency"]),
            allocation_percent=float(weights[index] * 100),
            predicted_return=float(row["Predicted Return"]),
            risk_score=float(row["Risk Score"]),
        ))
    return recommendations
