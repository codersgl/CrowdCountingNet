"""Bayesian Loss for crowd counting.

Implements Bayesian Loss (BL) and BL+ (with background pseudo-point) from:

    Ma, Wei, Su, Cao. "Bayesian Loss for Crowd Count Estimation with
    Point Supervision." ICCV 2019. arXiv:1908.03684.

Unlike L2 / SSIM / DM-Count losses which compare a *predicted* density map
against a *generated* GT density map (Gaussian-blurred annotations), BL
operates directly on the **point annotations** by assigning each pixel of
the predicted density an expected contribution to each point and minimising
the deviation of these expected counts from 1.

For a predicted density ``D(x)`` over pixels ``x`` and N annotated points
``y_1,…,y_N`` with isotropic Gaussian likelihoods
``p(x | y_n) ∝ exp(-‖x − y_n‖² / (2σ²))``, the posterior assignment is::

    p(y_n | x) = p(x | y_n) / Σ_m p(x | y_m)

The expected count attributed to point ``y_n`` is
``E[c_n] = Σ_x p(y_n | x) · D(x)`` and the loss is
``Σ_n |E[c_n] − 1|`` (or its squared form).

BL+ (``use_background=True``) adds a single background pseudo-point per
pixel placed on the line from the pixel to its nearest annotation, at a
distance ``d_bg = bg_ratio · max(H, W)`` from that pixel.  The expected
count for the background "point" is regressed toward 0 instead of 1, which
suppresses spurious density mass in empty regions.

API contract
------------
This module exposes ``BayesianLoss`` as a drop-in replacement for the MSE
density criterion used in :func:`crowdcount.engine.train_one_epoch`.  The
class sets ``requires_points = True`` and accepts the additional
``targets`` and ``image_sizes`` kwargs; the engine dispatches accordingly.

The returned scalar is **summed across the batch** to mirror the
``MSELoss(reduction="sum")`` convention (the engine subsequently divides
by the batch size).

Magnitude warning
-----------------
BL is on the scale of ``N`` (the per-image count) whereas
``MSELoss(reduction="sum")/B`` is on the scale of ``H·W·D²`` (typically
much smaller).  When switching from MSE to BL the user should retune
``cfg.density_loss_weight`` (typically increase by 10×–100×).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BayesianLoss(nn.Module):
    """Bayesian Loss / BL+ density criterion.

    Args:
        sigma:          Standard deviation (in image pixels) of the
                        Gaussian likelihood used to assign pixels to
                        annotated points.  Default 8.0 (paper recipe for
                        ShanghaiTech).
        use_background: If True, append a background pseudo-point per
                        pixel and regress its expected count toward 0
                        (BL+).  If False, vanilla BL.
        bg_ratio:       Distance of the background pseudo-point from
                        each pixel, expressed as a fraction of
                        ``max(H_img, W_img)``.  Default 0.15 (paper).
        count_loss_type: ``"l1"`` for ``|E[c] − target|`` or ``"mse"``
                        for ``(E[c] − target)²``.
    """

    requires_points: bool = True

    def __init__(
        self,
        sigma: float = 8.0,
        use_background: bool = True,
        bg_ratio: float = 0.15,
        count_loss_type: str = "l1",
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if bg_ratio <= 0:
            raise ValueError(f"bg_ratio must be > 0, got {bg_ratio}")
        if count_loss_type not in ("l1", "mse"):
            raise ValueError(
                f"count_loss_type must be 'l1' or 'mse', got '{count_loss_type}'"
            )
        self.sigma = float(sigma)
        self.use_background = bool(use_background)
        self.bg_ratio = float(bg_ratio)
        self.count_loss_type = count_loss_type

    def _residual_loss(
        self, expected_count: torch.Tensor, target: float
    ) -> torch.Tensor:
        """|E − target| or (E − target)² applied element-wise then summed."""
        diff = expected_count - target
        if self.count_loss_type == "l1":
            return diff.abs().sum()
        return diff.pow(2).sum()

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        image_sizes: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Compute Bayesian Loss.

        Args:
            pred_density: ``[B, 1, H_d, W_d]`` predicted density map.
            gt_density:   Accepted for API symmetry with MSE/SSIM criteria;
                          unused (point supervision only).
            targets:      List of length ``B`` with ``{"point": Tensor[N, 2]}``
                          entries holding annotation coords in **image
                          pixel** space (x, y).
            image_sizes:  ``(H_img, W_img)`` of the (possibly padded)
                          input image used to build the pixel grid in the
                          density-map coordinate frame.  When ``None``,
                          falls back to ``(H_d, W_d)`` (i.e. assumes
                          point coords are already in density-map space).

        Returns:
            Scalar loss summed over the batch.  Caller is expected to
            divide by the batch size.
        """
        if targets is None:
            raise ValueError("BayesianLoss.forward requires `targets` (point list)")

        del gt_density  # unused; kept for API symmetry.

        device = pred_density.device
        dtype = pred_density.dtype
        B, _, H_d, W_d = pred_density.shape

        if image_sizes is None:
            H_img, W_img = H_d, W_d
        else:
            H_img, W_img = int(image_sizes[0]), int(image_sizes[1])

        # Build pixel-centre coordinate grid in image-pixel units.
        # Each density-map cell covers ``H_img / H_d`` × ``W_img / W_d``
        # input pixels; centring on (i + 0.5) · stride keeps the grid
        # aligned with the average-pooled receptive field.
        stride_y = H_img / max(H_d, 1)
        stride_x = W_img / max(W_d, 1)
        ys = (torch.arange(H_d, device=device, dtype=dtype) + 0.5) * stride_y
        xs = (torch.arange(W_d, device=device, dtype=dtype) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Coords as [H_d * W_d, 2] in (x, y) order to match annotation convention.
        coords = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1
        )  # [P, 2] where P = H_d * W_d

        bg_distance_sq = (self.bg_ratio * float(max(H_img, W_img))) ** 2
        two_sigma_sq = 2.0 * self.sigma * self.sigma

        total_loss = torch.zeros((), device=device, dtype=dtype)

        for b in range(B):
            density_b = pred_density[b, 0].reshape(-1)  # [P]
            pts = targets[b].get("point") if isinstance(targets[b], dict) else None

            if pts is None or pts.numel() == 0:
                # No annotations: fall back to a global count loss
                # ``|Σ D(x) − 0|`` (matching the official reference).
                # This holds for both BL and BL+ so the head is told that
                # an empty crop should integrate to zero, regardless of
                # whether the background pseudo-point is enabled.
                pred_count = density_b.sum().unsqueeze(0)
                total_loss = total_loss + self._residual_loss(pred_count, 0.0)
                continue

            pts = pts.to(device=device, dtype=dtype)
            if pts.dim() != 2 or pts.shape[-1] != 2:
                raise ValueError(
                    f"target['point'] must be [N, 2]; got shape {tuple(pts.shape)}"
                )

            # Pairwise squared distances [P, N].
            # ``coords``: [P, 2]; ``pts``: [N, 2].
            dist_sq = (
                (coords.unsqueeze(1) - pts.unsqueeze(0)).pow(2).sum(dim=-1)
            )  # [P, N]

            if self.use_background:
                # Per-pixel background distance² following the official
                # reference (Ma et al. 2019, post_prob.py):
                #     d_bg²(x) = (r · S)² / (min_n ‖x − y_n‖² + ε)
                # so pixels near an annotation get a *large* bg distance
                # (bg likelihood ≈ 0), while pixels far from every
                # annotation get a *small* bg distance (bg likelihood ≈ 1).
                # Using a constant bg distance instead would make BL+
                # degenerate to vanilla BL with an extra useless point.
                min_dist_sq = dist_sq.min(dim=1, keepdim=True).values.clamp(
                    min=0.0
                )  # [P, 1]
                bg_dist_sq = bg_distance_sq / (min_dist_sq + 1e-5)  # [P, 1]
                dist_sq = torch.cat([dist_sq, bg_dist_sq], dim=1)  # [P, N+1]

            # Likelihood and posterior. Subtract per-row max log-lik for
            # numerical stability (standard log-sum-exp trick).
            log_lik = -dist_sq / two_sigma_sq  # [P, N(+1)]
            log_lik = log_lik - log_lik.max(dim=1, keepdim=True).values
            lik = log_lik.exp()
            posterior = lik / lik.sum(dim=1, keepdim=True).clamp(min=1e-12)  # [P, M]

            # Expected count per (real or background) point: weighted sum of D(x).
            expected = (posterior * density_b.unsqueeze(1)).sum(dim=0)  # [M]

            if self.use_background:
                expected_pts = expected[:-1]
                expected_bg = expected[-1:]
                total_loss = total_loss + self._residual_loss(expected_pts, 1.0)
                total_loss = total_loss + self._residual_loss(expected_bg, 0.0)
            else:
                total_loss = total_loss + self._residual_loss(expected, 1.0)

        return total_loss
