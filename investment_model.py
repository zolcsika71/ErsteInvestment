"""SQLite loading, feature engineering, and XGBoost investment ranking."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from analysis_types import (
    AnalysisError,
    CATEGORICAL_FEATURES,
    InvestmentRecommendation,
    ModelDiagnostics,
    NUMERIC_FEATURES,
)


def load_portfolio_data(database_path: Path) -> pd.DataFrame:
    """Load and validate portfolio snapshots from SQLite."""
    database_path = database_path.expanduser().resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    try:
        with sqlite3.connect(database_path) as connection:
            frame = pd.read_sql_query("SELECT * FROM model_portfolios", connection)
    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        raise AnalysisError(
            f"Could not read model_portfolios from {database_path}"
        ) from error
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
    return (
        frame.groupby(["Snapshot Date", "ISIN"], as_index=False, dropna=False)
        .agg(aggregations)
        .sort_values(["Snapshot Date", "ISIN"])
        .reset_index(drop=True)
    )


def add_forward_targets(
    frame: pd.DataFrame,
    minimum_days: int = 270,
    maximum_days: int = 455,
) -> pd.DataFrame:
    """Attach the nearest approximately one-year-ahead trailing return."""
    labeled = frame.copy()
    labeled["Target Return"] = np.nan
    labeled["Target Date"] = pd.NaT

    for indexes in labeled.groupby("ISIN").groups.values():
        ordered = list(labeled.loc[indexes].sort_values("Snapshot Date").index)
        dates = labeled.loc[ordered, "Snapshot Date"]
        for position, source_index in enumerate(ordered[:-1]):
            source_date = dates.loc[source_index]
            candidates: list[tuple[int, int]] = []
            for target_index in ordered[position + 1:]:
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
    numeric = frame.loc[:, NUMERIC_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )
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


def _validation_masks(
    train: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    eligible: list[tuple[pd.Series, pd.Series]] = []
    for raw_cutoff in sorted(train["Snapshot Date"].unique()):
        cutoff = pd.Timestamp(raw_cutoff)
        fit = train["Target Date"] < cutoff
        validation = train["Snapshot Date"] >= cutoff
        if fit.sum() >= 50 and validation.sum() >= 30:
            eligible.append((fit, validation))
    if not eligible:
        raise AnalysisError(
            "No leak-free time split has at least 50 training and "
            "30 validation samples"
        )
    return eligible[-1]


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
    features = _feature_matrix(pd.concat([train, latest], ignore_index=True))
    train_features = features.iloc[:len(train)]
    latest_features = features.iloc[len(train):]

    fit_mask, validation_mask = _validation_masks(train)
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
        validation_samples=int(validation_mask.sum()),
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
    return [
        InvestmentRecommendation(
            rank=rank,
            product=str(row["Product"]),
            isin=str(row["ISIN"]),
            asset_class=str(row["Asset Class"]),
            currency=str(row["Currency"]),
            predicted_return=float(row["Predicted Return"]),
            risk_score=float(row["Risk Score"]),
            model_score=float(row["Model Score"]),
        )
        for rank, (_, row) in enumerate(
            predictions.head(limit).iterrows(),
            start=1,
        )
    ]
