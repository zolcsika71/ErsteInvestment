"""Tests for time-aware investment analysis and allocation."""

import unittest

import numpy as np
import pandas as pd

from investment_analysis import (
    add_forward_targets,
    optimize_allocations,
)


class InvestmentAnalysisTests(unittest.TestCase):
    def test_forward_target_uses_nearest_snapshot_about_one_year_later(self) -> None:
        frame = pd.DataFrame({
            "Snapshot Date": pd.to_datetime([
                "2024-01-01", "2024-12-20", "2025-02-01",
            ]),
            "ISIN": ["TEST", "TEST", "TEST"],
            "1 Year": [0.01, 0.12, 0.20],
        })

        labeled = add_forward_targets(frame)

        self.assertAlmostEqual(labeled.loc[0, "Target Return"], 0.12)
        self.assertEqual(
            labeled.loc[0, "Target Date"],
            pd.Timestamp("2024-12-20"),
        )

    def test_optimizer_respects_cap_and_sums_to_one_hundred_percent(self) -> None:
        isins = [f"ISIN-{index}" for index in range(6)]
        dates = pd.to_datetime([
            "2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01",
        ])
        history_rows = []
        for date_index, date in enumerate(dates):
            history_rows.extend(
                {
                    "Snapshot Date": date,
                    "ISIN": isin,
                    "1 Year": 0.02 + isin_index * 0.01 + date_index * 0.002,
                }
                for isin_index, isin in enumerate(isins)
            )
        snapshots = pd.DataFrame(history_rows)
        predictions = pd.DataFrame({
            "Product": [f"Fund {index}" for index in range(6)],
            "ISIN": isins,
            "Asset Class": ["Equity"] * 6,
            "Currency": ["EUR"] * 6,
            "Predicted Return": np.linspace(0.04, 0.10, 6),
            "Risk Score": np.linspace(0.03, 0.08, 6),
            "Model Score": np.linspace(0.03, 0.09, 6),
        }).sort_values("Model Score", ascending=False)

        allocations = optimize_allocations(
            snapshots,
            predictions,
            "balanced",
            limit=6,
            maximum_allocation=0.20,
        )

        self.assertAlmostEqual(
            sum(item.allocation_percent for item in allocations),
            100.0,
            places=6,
        )
        self.assertLessEqual(
            max(item.allocation_percent for item in allocations),
            20.000001,
        )


if __name__ == "__main__":
    unittest.main()
