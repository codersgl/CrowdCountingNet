"""Tests for density generation quality analysis helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_density_generation_quality.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_density_generation_quality", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
analysis_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis_module
SPEC.loader.exec_module(analysis_module)

DensityRecord = analysis_module.DensityRecord
_in_bounds_summary = analysis_module._in_bounds_summary
_method_params = analysis_module._method_params
_normalise_method = analysis_module._normalise_method
_summary = analysis_module._summary
collect_density_records = analysis_module.collect_density_records


def _make_tiny_shanghai_dataset(root: Path) -> None:
    img_dir = root / "train_data" / "images"
    gt_dir = root / "train_data" / "ground_truth"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "IMG_0000.jpg"), img)
    with (gt_dir / "GT_0000.txt").open("w") as f:
        f.write("20.0 20.0\n")
        f.write("44.0 44.0\n")
        f.write("100.0 100.0\n")


def test_normalise_method_aliases() -> None:
    assert _normalise_method("geometry-adaptive") == "geometry_adaptive"
    assert _normalise_method("geo") == "geometry_adaptive"
    assert _normalise_method("fixed") == "fixed"
    assert _normalise_method("hybrid") == "hybrid"

    with pytest.raises(ValueError, match="Unknown density method"):
        _normalise_method("unknown")


def test_method_params_use_density_config_values() -> None:
    density_cfg = {
        "fixed_sigma": 6.0,
        "beta": 1.2,
        "min_sigma": 3.0,
        "sigma_base": 4.0,
        "hybrid_min_sigma": 1.5,
        "hybrid_max_sigma": 12.0,
        "hybrid_alpha": 0.7,
        "disparity_input": False,
    }

    fixed = _method_params("fixed", density_cfg)
    assert fixed.method == "fixed"
    assert fixed.mode == "fixed"
    assert fixed.fixed_sigma == 6.0
    assert not fixed.hybrid

    hybrid = _method_params("hybrid", density_cfg)
    assert hybrid.method == "hybrid"
    assert hybrid.mode == "hybrid"
    assert hybrid.hybrid
    assert hybrid.hybrid_max_sigma == 12.0
    assert hybrid.disparity_input is False


def test_summary_density_sum_metrics() -> None:
    records = [
        DensityRecord(
            dataset="demo",
            method="fixed",
            split="train",
            image="a.jpg",
            gt_count=3,
            in_bounds_gt_count=2,
            out_of_bounds_count=1,
            image_height=64,
            image_width=64,
            density_sum=1.5,
            error=-1.5,
            abs_error=1.5,
            rel_abs_error=0.5,
            in_bounds_error=-0.5,
            in_bounds_abs_error=0.5,
            in_bounds_rel_abs_error=0.25,
            cache_dir="cache",
        ),
        DensityRecord(
            dataset="demo",
            method="fixed",
            split="train",
            image="b.jpg",
            gt_count=4,
            in_bounds_gt_count=4,
            out_of_bounds_count=0,
            image_height=64,
            image_width=64,
            density_sum=5.0,
            error=1.0,
            abs_error=1.0,
            rel_abs_error=0.25,
            in_bounds_error=1.0,
            in_bounds_abs_error=1.0,
            in_bounds_rel_abs_error=0.25,
            cache_dir="cache",
        ),
    ]

    row = _summary(records)
    in_bounds_row = _in_bounds_summary(records)

    assert row.dataset == "demo"
    assert row.method == "fixed"
    assert row.n_images == 2
    assert row.gt_total == 7
    assert row.density_total == pytest.approx(6.5)
    assert row.mae == pytest.approx(1.25)
    assert row.rmse == pytest.approx(np.sqrt((2.25 + 1.0) / 2.0))
    assert row.bias == pytest.approx(-0.25)
    assert row.mean_rel_abs_error == pytest.approx(0.375)
    assert row.under_ratio == pytest.approx(0.5)
    assert row.over_ratio == pytest.approx(0.5)

    assert in_bounds_row.gt_total == 7
    assert in_bounds_row.in_bounds_gt_total == 6
    assert in_bounds_row.out_of_bounds_total == 1
    assert in_bounds_row.out_of_bounds_rate == pytest.approx(1 / 7)
    assert in_bounds_row.in_bounds_mae == pytest.approx(0.75)
    assert in_bounds_row.in_bounds_rmse == pytest.approx(np.sqrt((0.25 + 1.0) / 2.0))
    assert in_bounds_row.in_bounds_bias == pytest.approx(0.25)
    assert in_bounds_row.in_bounds_mean_rel_abs_error == pytest.approx(0.25)


def test_collect_density_records_generates_fixed_cache(tmp_path: Path) -> None:
    _make_tiny_shanghai_dataset(tmp_path)
    params = _method_params("fixed", {"fixed_sigma": 2.0})

    records = collect_density_records(
        dataset_name="tiny",
        data_root=tmp_path,
        split="train",
        params=params,
        ensure_cache=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record.dataset == "tiny"
    assert record.method == "fixed"
    assert record.gt_count == 3
    assert record.in_bounds_gt_count == 2
    assert record.out_of_bounds_count == 1
    assert record.density_sum == pytest.approx(2.0, rel=0.05)
    assert record.abs_error == pytest.approx(1.0, rel=0.1)
    assert record.in_bounds_abs_error < 0.1
    assert Path(record.cache_dir).name == "train"
    assert Path(record.cache_dir).parent.name == "gt_density_maps_fixed_s2p00"
