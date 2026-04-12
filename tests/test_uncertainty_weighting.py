"""Tests for UncertaintyWeighter (Kendall et al. 2018)."""

from __future__ import annotations

import math

import pytest
import torch

from crowdcount.models.uncertainty_loss import UncertaintyWeighter


@pytest.fixture
def weighter():
    """Default UncertaintyWeighter with known init values."""
    return UncertaintyWeighter(
        init_log_var_den=3.91,
        init_log_var_ce=-0.693,
        init_log_var_reg=8.52,
    )


class TestInit:
    def test_parameters_registered(self, weighter: UncertaintyWeighter):
        names = {n for n, _ in weighter.named_parameters()}
        assert names == {"log_var_den", "log_var_ce", "log_var_reg"}

    def test_initial_values(self, weighter: UncertaintyWeighter):
        assert weighter.log_var_den.item() == pytest.approx(3.91, abs=1e-3)
        assert weighter.log_var_ce.item() == pytest.approx(-0.693, abs=1e-3)
        assert weighter.log_var_reg.item() == pytest.approx(8.52, abs=1e-3)


class TestForward:
    def test_output_is_scalar(self, weighter: UncertaintyWeighter):
        loss = weighter(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
        assert loss.dim() == 0

    def test_formula_correctness(self):
        """Verify forward matches the analytic formula."""
        s_den, s_ce, s_reg = 2.0, 0.0, 4.0
        w = UncertaintyWeighter(s_den, s_ce, s_reg)
        l_den, l_ce, l_reg = 0.5, 1.2, 0.3
        result = w(torch.tensor(l_den), torch.tensor(l_ce), torch.tensor(l_reg))
        expected = (
            0.5 * math.exp(-s_den) * l_den
            + 0.5 * s_den
            + 0.5 * math.exp(-s_ce) * l_ce
            + 0.5 * s_ce
            + 0.5 * math.exp(-s_reg) * l_reg
            + 0.5 * s_reg
        )
        assert result.item() == pytest.approx(expected, rel=1e-5)

    def test_effective_weights_match_fixed(self):
        """When init from fixed weights, effective weights ≈ original values."""
        # s = -log(2w) => 1/(2σ²) = exp(-s)/2 = exp(log(2w))/2 = w
        w_den, w_ce, w_reg = 0.01, 1.0, 0.0002
        uw = UncertaintyWeighter(
            -math.log(2 * w_den),
            -math.log(2 * w_ce),
            -math.log(2 * w_reg),
        )
        weights = uw.get_weights()
        assert weights["w_den"] == pytest.approx(w_den, rel=1e-4)
        assert weights["w_ce"] == pytest.approx(w_ce, rel=1e-4)
        assert weights["w_reg"] == pytest.approx(w_reg, rel=1e-4)


class TestGradients:
    def test_gradients_flow_to_log_vars(self, weighter: UncertaintyWeighter):
        loss = weighter(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
        loss.backward()
        for p in weighter.parameters():
            assert p.grad is not None
            assert p.grad.abs().item() > 0

    def test_gradients_flow_through_task_losses(self):
        """Task losses with requires_grad get gradients through the weighter."""
        w = UncertaintyWeighter(0.0, 0.0, 0.0)
        l_den = torch.tensor(2.0, requires_grad=True)
        l_ce = torch.tensor(3.0, requires_grad=True)
        l_reg = torch.tensor(1.0, requires_grad=True)
        total = w(l_den, l_ce, l_reg)
        total.backward()
        assert l_den.grad is not None
        assert l_ce.grad is not None
        assert l_reg.grad is not None


class TestHelpers:
    def test_get_weights_keys(self, weighter: UncertaintyWeighter):
        w = weighter.get_weights()
        assert set(w.keys()) == {"w_den", "w_ce", "w_reg"}
        assert all(isinstance(v, float) for v in w.values())

    def test_get_log_vars_keys(self, weighter: UncertaintyWeighter):
        lv = weighter.get_log_vars()
        assert set(lv.keys()) == {"log_var_den", "log_var_ce", "log_var_reg"}

    def test_get_weights_positive(self, weighter: UncertaintyWeighter):
        """Effective weights must always be positive."""
        w = weighter.get_weights()
        assert all(v > 0 for v in w.values())


class TestStateDict:
    def test_save_load_roundtrip(self, weighter: UncertaintyWeighter):
        # Modify params
        with torch.no_grad():
            weighter.log_var_den.fill_(99.0)

        state = weighter.state_dict()
        new_w = UncertaintyWeighter(0.0, 0.0, 0.0)
        new_w.load_state_dict(state)
        assert new_w.log_var_den.item() == pytest.approx(99.0)
