"""Named-portfolio scoring and constrained allocation optimization."""

from __future__ import annotations

import math
import json
from urllib.parse import quote
from urllib.request import Request, urlopen
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


def fetch_external_market_metrics(
    tickers: list[str],
    timeout: int = 10,
) -> pd.DataFrame:
    """Fetch recent Yahoo Finance prices and calculate short-term metrics.

    ``tickers`` must contain explicit Yahoo Finance symbols such as ``VWCE.DE``.
    The endpoint returns daily chart data; no ticker is inferred from an ISIN.
    """
    records: list[dict[str, float | str]] = []
    for ticker in sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()}):
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{quote(ticker)}?range=1y&interval=1d&events=history"
        )
        request = Request(url, headers={"User-Agent": "ErsteInvestment/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            result = payload["chart"]["result"][0]
            closes = [
                float(value)
                for value in result["indicators"]["quote"][0]["close"]
                if value is not None
            ]
        except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(closes) < 20:
            continue

        returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
        peak = closes[0]
        drawdowns: list[float] = []
        for close in closes:
            peak = max(peak, close)
            drawdowns.append(close / peak - 1)
        records.append({
            "Ticker": ticker,
            "External 1Y Return": closes[-1] / closes[0] - 1,
            "External Volatility": (sum(value * value for value in returns) / len(returns)) ** 0.5,
            "External Maximum Drawdown": min(drawdowns),
        })
    return pd.DataFrame.from_records(records)


def merge_external_market_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Merge external metrics into a frame containing a ``Ticker`` column."""
    if "Ticker" not in frame.columns:
        raise ValueError("External market data requires a Ticker column")
    external = fetch_external_market_metrics(frame["Ticker"].dropna().astype(str).tolist())
    return frame.merge(external, on="Ticker", how="left")


def rank_short_term_capital_preservation(
    raw_frame: pd.DataFrame,
    use_external_market_data: bool = True,
) -> list[PortfolioRecommendation]:
    """Rank portfolios for short-term capital preservation.

    This ranking uses only the latest snapshot in ``raw_frame``. Returns are
    rewarded, while volatility, downside risk, drawdown, and concentration are
    penalized. Missing short-term metrics reduce coverage and therefore reduce
    confidence in a portfolio. No external market or web data is consulted.
    """
    if use_external_market_data and "Ticker" in raw_frame.columns:
        raw_frame = merge_external_market_metrics(raw_frame)

    latest_date = raw_frame["Snapshot Date"].max()
    latest = raw_frame[raw_frame["Snapshot Date"] == latest_date].copy()
    results: list[PortfolioRecommendation] = []

    for portfolio_name, group in latest.groupby("Portfolio Name"):
        allocation = pd.to_numeric(group["Allocation (%)"], errors="coerce")
        total = allocation[allocation > 0].sum()
        if not np.isfinite(total) or total <= 0:
            continue

        weights = allocation.clip(lower=0) / total

        def weighted_metric(column: str) -> float | None:
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna() & weights.notna()
            if not valid.any():
                return None
            return float((values[valid] * weights[valid]).sum())

        ytd = weighted_metric("YTD")
        one_year = weighted_metric("1 Year")
        volatility = weighted_metric("1Y Volatility")
        downside = weighted_metric("Downside Risk")
        drawdown = weighted_metric("Maximum Drawdown")
        if "External Volatility" in group:
            volatility = weighted_metric("External Volatility") or volatility
        if "External Maximum Drawdown" in group:
            drawdown = weighted_metric("External Maximum Drawdown") or drawdown
        coverage = float(
            group["1 Year"].notna().sum() / max(len(group), 1)
        )
        concentration = float(np.square(weights.fillna(0)).sum())

        # Missing metrics contribute zero to the score but lower coverage.
        expected_return = float(np.nanmean([value for value in (ytd, one_year)
                                            if value is not None])) if any(
            value is not None for value in (ytd, one_year)
        ) else 0.0
        expected_volatility = volatility or 0.0
        preservation_penalty = (
            0.45 * expected_volatility
            + 0.30 * (downside or 0.0)
            + 0.20 * (drawdown or 0.0)
            + 0.05 * concentration
        )
        score = expected_return - preservation_penalty

        results.append(PortfolioRecommendation(
            rank=0,
            portfolio_name=str(portfolio_name),
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            concentration=concentration,
            score=score,
            coverage=coverage,
        ))

    results.sort(key=lambda item: item.score, reverse=True)
    return [
        PortfolioRecommendation(
            rank=index,
            **{key: value for key, value in asdict(item).items() if key != "rank"},
        )
        for index, item in enumerate(results, start=1)
    ]


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
