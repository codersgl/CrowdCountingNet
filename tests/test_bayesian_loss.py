"""Tests for Bayesian Loss density criterion."""

from __future__ import annotations

import pytest
import torch

from crowdcount.plugins.bayesian_loss import BayesianLoss


# Use spatial sizes that match a typical downsampled density map.
H_IMG, W_IMG = 64, 64
H_D, W_D = 32, 32  # density map at /2 scale


def _make_targets(
    num_per_image: list[int], img_h: int = H_IMG, img_w: int = W_IMG
) -> list[dict]:
    targets: list[dict] = []
    for n in num_per_image:
        if n == 0:
            pts = torch.zeros((0, 2), dtype=torch.float32)
        else:
            torch.manual_seed(n + 1)
            xs = torch.rand(n) * (img_w - 1)
            ys = torch.rand(n) * (img_h - 1)
            pts = torch.stack([xs, ys], dim=-1)
        targets.append({"point": pts})
    return targets


def test_requires_points_attribute() -> None:
    loss = BayesianLoss()
    assert getattr(loss, "requires_points", False) is True


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        BayesianLoss(sigma=0.0)
    with pytest.raises(ValueError):
        BayesianLoss(bg_ratio=-0.1)
    with pytest.raises(ValueError):
        BayesianLoss(count_loss_type="huber")


def test_missing_targets_raises() -> None:
    loss = BayesianLoss()
    pred = torch.zeros(1, 1, H_D, W_D)
    with pytest.raises(ValueError):
        loss(pred)


def test_shape_and_grad_flow() -> None:
    loss = BayesianLoss(sigma=4.0)
    pred = torch.rand(2, 1, H_D, W_D, requires_grad=True)
    targets = _make_targets([5, 7])
    out = loss(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG))
    assert out.dim() == 0  # scalar
    assert torch.isfinite(out).item()
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()


def test_zero_pred_loss_matches_count_with_bg() -> None:
    """With BL+ and zero predicted density, every E[c_n]=0:
    real points contribute |0-1|=1 each, bg contributes |0-0|=0.
    Loss should equal total #points across the batch.
    """
    loss = BayesianLoss(sigma=4.0, use_background=True, count_loss_type="l1")
    pred = torch.zeros(2, 1, H_D, W_D)
    targets = _make_targets([3, 5])
    out = loss(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG))
    assert out.item() == pytest.approx(3 + 5, abs=1e-5)


def test_zero_pred_loss_matches_count_vanilla_bl() -> None:
    """Vanilla BL with zero pred: same |0-1|=1 per point, no bg term."""
    loss = BayesianLoss(sigma=4.0, use_background=False, count_loss_type="l1")
    pred = torch.zeros(2, 1, H_D, W_D)
    targets = _make_targets([4, 2])
    out = loss(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG))
    assert out.item() == pytest.approx(4 + 2, abs=1e-5)


def test_well_placed_density_lowers_loss() -> None:
    """Density with mass concentrated on annotation locations should
    yield substantially lower loss than zero density."""
    loss = BayesianLoss(sigma=4.0, use_background=True, count_loss_type="l1")
    targets = _make_targets([6])

    pred_zero = torch.zeros(1, 1, H_D, W_D)
    loss_zero = loss(pred_zero, None, targets=targets, image_sizes=(H_IMG, W_IMG))

    # Build a density with a count-1 spike at each point's downscaled
    # location so each point's expected count is ~1.
    pred_good = torch.zeros(1, 1, H_D, W_D)
    stride_y = H_IMG / H_D
    stride_x = W_IMG / W_D
    for x_pix, y_pix in targets[0]["point"]:
        cy = int(min(max(y_pix.item() / stride_y, 0), H_D - 1))
        cx = int(min(max(x_pix.item() / stride_x, 0), W_D - 1))
        pred_good[0, 0, cy, cx] += 1.0
    loss_good = loss(pred_good, None, targets=targets, image_sizes=(H_IMG, W_IMG))

    assert loss_good.item() < loss_zero.item() * 0.5


def test_empty_points_handled() -> None:
    """An image with zero annotations must not raise. Following the
    official reference, both BL and BL+ fall back to a global count loss
    ``|sum(D) - 0|`` so the head is supervised to integrate to zero on
    empty crops."""
    targets = _make_targets([0])
    pred = torch.full((1, 1, H_D, W_D), 0.5)
    expected = abs(pred.sum().item())  # |sum(D) - 0|

    bl_plus = BayesianLoss(sigma=4.0, use_background=True, count_loss_type="l1")
    out_plus = bl_plus(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG))
    assert out_plus.item() == pytest.approx(expected, rel=1e-4)

    bl = BayesianLoss(sigma=4.0, use_background=False, count_loss_type="l1")
    out_vanilla = bl(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG))
    assert out_vanilla.item() == pytest.approx(expected, rel=1e-4)


def test_bg_term_adds_penalty_on_uniform_pred() -> None:
    """When the pred density carries spurious mass (uniform > 0),
    BL+ should penalise the background while vanilla BL has no
    such term, so BL+ ≥ BL on the same input."""
    targets = _make_targets([3])
    pred = torch.full((1, 1, H_D, W_D), 0.05)

    bl = BayesianLoss(sigma=4.0, use_background=False)
    bl_plus = BayesianLoss(sigma=4.0, use_background=True)
    loss_bl = bl(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG)).item()
    loss_bl_plus = bl_plus(
        pred, None, targets=targets, image_sizes=(H_IMG, W_IMG)
    ).item()
    assert loss_bl_plus >= loss_bl


def test_l1_vs_mse_modes_differ() -> None:
    targets = _make_targets([4])
    pred = torch.full((1, 1, H_D, W_D), 0.02, requires_grad=True)

    l1 = BayesianLoss(sigma=4.0, count_loss_type="l1")
    mse = BayesianLoss(sigma=4.0, count_loss_type="mse")
    out_l1 = l1(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG))
    out_mse = mse(pred, None, targets=targets, image_sizes=(H_IMG, W_IMG))
    assert torch.isfinite(out_l1).item()
    assert torch.isfinite(out_mse).item()
    assert abs(out_l1.item() - out_mse.item()) > 1e-4
    out_mse.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all().item()


def test_image_sizes_default_to_density_shape() -> None:
    """When image_sizes is None, points should be interpreted in
    density-map coords; passing density-scale points must work."""
    loss = BayesianLoss(sigma=2.0, use_background=False)
    pred = torch.zeros(1, 1, H_D, W_D)
    targets = [{"point": torch.tensor([[5.0, 5.0], [10.0, 10.0]], dtype=torch.float32)}]
    out = loss(pred, None, targets=targets, image_sizes=None)
    # Zero pred, vanilla BL → loss == number of points.
    assert out.item() == pytest.approx(2.0, abs=1e-5)


def test_bg_posterior_is_spatially_varying() -> None:
    """BL+ background distance must depend on the per-pixel min distance
    to the nearest annotation (official formula). A pixel co-located with
    an annotation should produce a near-zero bg posterior; a pixel far
    from every annotation should produce a near-one bg posterior.

    Uses a realistic image size (128 px) so that the bg distance
    ``(r·S)²/min_dist²`` reaches the proper saturation regime.
    """
    H_img, W_img = 128, 128
    H_d, W_d = 32, 32  # stride 4; cell (2, 2) centres on (10, 10) image px
    bl_plus = BayesianLoss(sigma=4.0, use_background=True)
    pts = torch.tensor([[10.0, 10.0]], dtype=torch.float32)

    # Mass placed exactly on the annotation density cell.
    pred_near = torch.zeros(1, 1, H_d, W_d)
    pred_near[0, 0, 2, 2] = 1.0
    out_near = bl_plus(
        pred_near, None, targets=[{"point": pts}], image_sizes=(H_img, W_img)
    )
    # Pt-cell distance ≈ 0  →  bg likelihood ≈ 0
    # →  expected_pt ≈ 1, expected_bg ≈ 0  →  loss ≈ 0.
    assert out_near.item() < 0.05

    # Mass placed at the far corner, well outside the bg transition radius.
    pred_far = torch.zeros(1, 1, H_d, W_d)
    pred_far[0, 0, H_d - 1, W_d - 1] = 1.0
    out_far = bl_plus(
        pred_far, None, targets=[{"point": pts}], image_sizes=(H_img, W_img)
    )
    # Far pixel: bg likelihood dominates  →  expected_bg ≈ 1, expected_pt ≈ 0
    # →  loss ≈ |0 - 1| + |1 - 0| = 2.
    assert out_far.item() > 1.5


def test_bl_plus_strictly_stronger_than_vanilla_on_spurious_mass() -> None:
    """When the prediction puts a unit of mass *far* from every
    annotation, BL+ must penalise it (bg expected count grows) while
    vanilla BL has only the per-point regression target. With per-pixel
    bg distance, BL+ loss should noticeably exceed vanilla BL loss."""
    pts = torch.tensor([[3.0, 3.0]], dtype=torch.float32)
    targets = [{"point": pts}]
    pred = torch.zeros(1, 1, H_D, W_D)
    pred[0, 0, 3, 3] = 1.0  # correct mass at annotation
    pred[0, 0, H_D - 1, W_D - 1] = 1.0  # spurious mass far away

    bl = BayesianLoss(sigma=4.0, use_background=False)
    bl_plus = BayesianLoss(sigma=4.0, use_background=True)
    loss_bl = bl(pred, None, targets=targets, image_sizes=None).item()
    loss_bl_plus = bl_plus(pred, None, targets=targets, image_sizes=None).item()
    # BL only sees: expected ≈ 2 (both pixels routed to single point) → |2-1| = 1
    # BL+ sees: expected_pt ≈ 1, expected_bg ≈ 1 → |1-1|+|1-0| = 1
    # BL+ should be ≥ BL since the spurious-mass pixel goes mostly to bg.
    assert loss_bl_plus > loss_bl


def test_density_head_optimises_toward_correct_expected_counts() -> None:
    """End-to-end sanity: train a tiny "density head" (a learnable density
    map) with Adam on the BL+ loss for a few hundred steps. The head must
    converge so that:
      (a) the loss decreases substantially from its initial value;
      (b) the final integrated count is in the right ballpark of the GT.
    """
    torch.manual_seed(0)
    H_img, W_img = 96, 96
    H_d, W_d = 24, 24  # stride 4
    pts = torch.tensor(
        [[20.0, 20.0], [48.0, 32.0], [72.0, 72.0], [12.0, 80.0]],
        dtype=torch.float32,
    )
    targets = [{"point": pts}]

    # Learnable raw density (apply softplus → non-negative).
    raw = torch.full((1, 1, H_d, W_d), -2.0, requires_grad=True)
    optim = torch.optim.Adam([raw], lr=0.1)
    loss_fn = BayesianLoss(sigma=4.0, use_background=True, count_loss_type="l1")

    losses: list[float] = []
    for _ in range(800):
        optim.zero_grad()
        density = torch.nn.functional.softplus(raw)
        loss = loss_fn(density, None, targets=targets, image_sizes=(H_img, W_img))
        loss.backward()
        optim.step()
        losses.append(loss.item())

    final_density = torch.nn.functional.softplus(raw).detach()
    final_count = float(final_density.sum().item())
    gt_count = float(pts.shape[0])

    # (a) The loss should drop to a small fraction of its starting value.
    assert losses[-1] < losses[0] * 0.2, (
        f"loss did not decrease enough: {losses[0]:.3f} → {losses[-1]:.3f}"
    )
    # (b) Predicted total count should be on the same order of magnitude
    # as the GT count.  BL+ with L1 only constrains *per-point* expected
    # counts to be 1; the integrated density is not strictly count-
    # conservative and can overshoot moderately when the posterior is
    # spatially leaky (a known property of BL — see Ma et al. §4.3).
    assert 0.5 * gt_count < final_count < 3.0 * gt_count, (
        f"predicted count {final_count:.2f} not in [{0.5 * gt_count}, {3.0 * gt_count}]"
    )
