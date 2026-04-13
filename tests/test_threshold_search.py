"""Tests for threshold search utilities in engine.py."""

from __future__ import annotations

import torch
import numpy as np
import pytest

from crowdcount.engine import search_optimal_threshold


class TestSearchOptimalThreshold:
    """Tests for search_optimal_threshold with synthetic data."""

    def _make_scores(
        self, n_images: int, n_queries: int, gt_counts: list[int], optimal_t: float
    ) -> list[torch.Tensor]:
        """Create synthetic scores where `optimal_t` yields perfect counting.

        For each image, the top `gt_count` scores are above `optimal_t` and
        the rest are below.
        """
        scores = []
        for gt in gt_counts:
            s = torch.zeros(n_queries)
            if gt > 0:
                # Place gt scores above optimal_t, rest below
                s[:gt] = optimal_t + 0.05
                s[gt:] = optimal_t - 0.05
            scores.append(s)
        return scores

    def test_finds_optimal_threshold(self):
        gt_counts = [10, 20, 15, 30, 5]
        n_queries = 100
        optimal_t = 0.45
        all_scores = self._make_scores(len(gt_counts), n_queries, gt_counts, optimal_t)

        best_t, best_mae, results = search_optimal_threshold(all_scores, gt_counts)

        assert best_mae == 0.0
        # Best threshold should be close to the designed optimal
        assert abs(best_t - optimal_t) <= 0.05

    def test_returns_all_candidates(self):
        gt_counts = [5]
        all_scores = [torch.rand(50)]

        best_t, best_mae, results = search_optimal_threshold(all_scores, gt_counts)

        # Default range [0.1, 0.95] step 0.01 → 86 candidates
        assert len(results) >= 80
        assert all(isinstance(v, float) for v in results.values())

    def test_custom_range(self):
        gt_counts = [10]
        all_scores = [torch.rand(50)]

        best_t, best_mae, results = search_optimal_threshold(
            all_scores, gt_counts, t_min=0.3, t_max=0.7, t_step=0.1
        )

        assert 0.3 <= best_t <= 0.7
        assert len(results) == 5  # 0.3, 0.4, 0.5, 0.6, 0.7

    def test_single_image(self):
        gt_counts = [0]
        all_scores = [torch.zeros(20)]

        best_t, best_mae, results = search_optimal_threshold(all_scores, gt_counts)

        # All scores are 0, so any threshold > 0 gives predict_cnt=0, MAE=0
        assert best_mae == 0.0

    def test_high_count_image(self):
        gt_counts = [80]
        n_queries = 100
        scores = torch.linspace(0.0, 1.0, n_queries)
        all_scores = [scores]

        best_t, best_mae, results = search_optimal_threshold(all_scores, gt_counts)

        # With 100 evenly spaced scores, threshold ~0.2 should give ~80 detections
        assert best_mae < 2.0
        assert 0.15 <= best_t <= 0.25
