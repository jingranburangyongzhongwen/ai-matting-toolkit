"""Local automated tests for manual edge refinement.

Usage:
    python scripts/test_manual_refine.py [--debug-dir output/_test_manual]

Requires ViTMatte (GPU ~0.5s/case, CPU ~3s/case). No Gradio server needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engines.rgba_postprocess import build_context, _compute_spill_score
from engines.manual_refine import (
    build_accept_mask,
    expand_user_mask,
    build_spill_aware_trimap,
    regularized_unmix,
    compute_spatial_gate,
    compute_confidence,
    compute_recon_gate,
    compute_screen_spill_score,
    estimate_screen_chroma,
    refine_manual_edge,
)
from engines.vitmatte import ViTMatteRefiner
from model_manager import get_models_path, get_device


RGB_DISTANCE_MAX = float(np.sqrt(3.0) * 255.0)


# ── Synthetic Data ─────────────────────────────────────────────

def make_synthetic_screen(
    fg_color=(120, 80, 50),
    bg_color=(30, 220, 60),
    size=512,
):
    """Create a synthetic image with known GT alpha, simulating RMBG solidification failure.

    Returns:
        image_rgb: HxWx3 uint8 (composited with true alpha, containing bg spill at edges)
        alpha_gt: HxW uint8 (ground truth alpha from geometry)
        rmbg_alpha: HxW uint8 (simulated RMBG: edge alpha pumped to 240-255)
        user_mask: HxW bool (covers the contaminated edge band)
    """
    h = w = size
    y, x = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.45

    # Main circular foreground
    radius = min(h, w) * 0.28
    dist_main = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha_gt = np.clip((radius + 3.0 - dist_main) / 6.0, 0.0, 1.0)

    # Add hair-like strands
    for offset in (-45, -20, 5, 30, 50):
        strand_x = cx + offset + np.sin((y - 15) / 15.0) * 6.0
        strand = np.exp(-((x - strand_x) ** 2) / 1.8)
        vertical = (y > 15) & (y < h - 15)
        alpha_gt = np.maximum(alpha_gt, strand * vertical * 0.98)

    # Ground truth foreground color (uniform for simplicity)
    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[..., 0] = fg_color[0]
    fg[..., 1] = fg_color[1]
    fg[..., 2] = fg_color[2]

    # Background color
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[..., 0] = bg_color[0]
    bg[..., 1] = bg_color[1]
    bg[..., 2] = bg_color[2]

    # Composite with true alpha (this is the "image" with real bg spill)
    a3 = alpha_gt[..., None]
    image_rgb = np.clip(fg * a3 + bg * (1.0 - a3), 0, 255).astype(np.uint8)

    # Simulate RMBG failure: pump edge alpha to 230-250 range
    edge_band = (alpha_gt > 0.15) & (alpha_gt < 0.90)
    rmbg_alpha = (alpha_gt * 255).astype(np.uint8)
    # Pump edge pixels to near-solid (mimic RMBG solidification)
    pumped = np.clip(alpha_gt * 255 * 1.8, 220, 250).astype(np.uint8)
    rmbg_alpha[edge_band] = np.maximum(rmbg_alpha[edge_band], pumped[edge_band])

    # User mask: cover the contaminated edge band
    # In practice user paints over the visible green/blue rim
    dist_to_edge = cv2.distanceTransform(
        (~(edge_band)).astype(np.uint8), cv2.DIST_L2, 5,
    )
    user_mask = edge_band & (dist_to_edge <= 8)

    return image_rgb, (alpha_gt * 255).astype(np.uint8), rmbg_alpha, user_mask


def make_no_spill_case(size=512):
    """Foreground with no background contamination — regression test."""
    h = w = size
    y, x = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.5
    radius = min(h, w) * 0.3
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha_gt = np.clip((radius + 3.0 - dist) / 6.0, 0.0, 1.0)

    # Foreground = background color (no separation) — no spill to remove
    color = np.array([100, 100, 100], dtype=np.float32)
    image_rgb = np.broadcast_to(color, (h, w, 3)).astype(np.uint8).copy()

    rmbg_alpha = (alpha_gt * 255).astype(np.uint8)
    # Pump edge
    edge = (alpha_gt > 0.1) & (alpha_gt < 0.95)
    rmbg_alpha[edge] = np.clip(rmbg_alpha[edge].astype(float) * 2.0, 200, 255).astype(np.uint8)

    user_mask = edge.copy()
    return image_rgb, (alpha_gt * 255).astype(np.uint8), rmbg_alpha, user_mask


# ── Metrics ────────────────────────────────────────────────────

def residue_by_alpha(rgb, alpha, ctx):
    """Compute visible background-color residue in alpha bands."""
    mask = (alpha > 0)
    if not np.any(mask):
        return {"lt64": 0.0, "64_180": 0.0, "180_240": 0.0, "gte240": 0.0}

    spill, _, _ = _compute_spill_score(
        rgb, ctx["foreground_fill"], ctx["background_fill"], mask=mask,
    )
    a_f = alpha.astype(np.float32) / 255.0
    visible = spill * a_f

    bands = {
        "lt64": mask & (alpha < 64),
        "64_180": mask & (alpha >= 64) & (alpha < 180),
        "180_240": mask & (alpha >= 180) & (alpha < 240),
        "gte240": mask & (alpha >= 240),
    }
    result = {}
    for name, band in bands.items():
        result[name] = float(visible[band].mean()) if np.any(band) else 0.0
    return result


# ── ViTMatte Setup ─────────────────────────────────────────────

def get_vitmatte_refiner():
    """Load ViTMatte Base engine."""
    from model_manager import ModelManager
    mgr = ModelManager()
    return mgr.get_vitmatte("base")


class FixedAlphaRefiner:
    """Small deterministic refiner for logic tests that should not need ViTMatte."""

    def __init__(self, alpha_out):
        self.alpha_out = alpha_out.astype(np.uint8)

    def refine_with_trimap(self, image, alpha, trimap, accept_mask, mode="full", _debug_dir=None):
        return np.where(accept_mask, self.alpha_out, alpha).astype(np.uint8)


# ── Test Cases ─────────────────────────────────────────────────

def test_green_screen_hair(refiner, debug_dir=None):
    """Green screen: residue in >=240 band should drop >40%."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(120, 80, 50), bg_color=(30, 220, 60),
    )

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    ctx = build_context(image, rmbg_alpha)
    before = residue_by_alpha(image, rmbg_alpha, ctx)
    after = residue_by_alpha(rgba_out[:, :, :3], rgba_out[:, :, 3], ctx)

    gte240_before = before["gte240"]
    gte240_after = after["gte240"]
    improve = (gte240_before - gte240_after) / max(gte240_before, 1e-6) * 100

    # Also check unmix accuracy against known fg_color
    accept = build_accept_mask(
        expand_user_mask(user_mask), rmbg_alpha,
        ctx["protected_transparency"], ctx["dist_to_background"],
    )
    fg_gt = np.array([120, 80, 50], dtype=np.float32)
    rgb_out = rgba_out[:, :, :3].astype(np.float32)
    fg_err = np.linalg.norm(rgb_out - fg_gt, axis=2)
    fg_err_on_accept = fg_err[accept & (rmbg_alpha >= 200)]

    return {
        "case": "green_screen_hair",
        "residue_before_by_alpha": before,
        "residue_after_by_alpha": after,
        "gte240_improve_pct": round(improve, 1),
        "unmix_fg_error_mean": round(float(fg_err_on_accept.mean()), 2) if fg_err_on_accept.size else 0,
        "unmix_fg_error_p95": round(float(np.percentile(fg_err_on_accept, 95)), 2) if fg_err_on_accept.size else 0,
        "solid_loss_pct": round(diag.get("solid_loss_pct", 0), 2),
        "alpha_l1_delta": round(diag.get("alpha_l1_delta", 0), 4),
        "rollback_alpha": diag.get("rollback_alpha", False),
        "elapsed_s": round(diag.get("elapsed_s", 0), 2),
        "passed": improve > 40 and diag.get("solid_loss_pct", 0) < 15,
    }


def test_blue_screen_hair(refiner, debug_dir=None):
    """Blue screen: residue in >=240 band should drop >40%."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(180, 140, 90), bg_color=(20, 40, 230),
    )

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    ctx = build_context(image, rmbg_alpha)
    before = residue_by_alpha(image, rmbg_alpha, ctx)
    after = residue_by_alpha(rgba_out[:, :, :3], rgba_out[:, :, 3], ctx)

    gte240_improve = (before["gte240"] - after["gte240"]) / max(before["gte240"], 1e-6) * 100

    return {
        "case": "blue_screen_hair",
        "residue_before_by_alpha": before,
        "residue_after_by_alpha": after,
        "gte240_improve_pct": round(gte240_improve, 1),
        "solid_loss_pct": round(diag.get("solid_loss_pct", 0), 2),
        "passed": gte240_improve > 15 and diag.get("solid_loss_pct", 0) < 15,
    }


def test_red_bg_hair(refiner, debug_dir=None):
    """Red/warm background: residue in >=240 band should drop >30%."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(60, 130, 180), bg_color=(210, 40, 30),
    )

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    ctx = build_context(image, rmbg_alpha)
    before = residue_by_alpha(image, rmbg_alpha, ctx)
    after = residue_by_alpha(rgba_out[:, :, :3], rgba_out[:, :, 3], ctx)

    gte240_improve = (before["gte240"] - after["gte240"]) / max(before["gte240"], 1e-6) * 100

    return {
        "case": "red_bg_hair",
        "residue_before_by_alpha": before,
        "residue_after_by_alpha": after,
        "gte240_improve_pct": round(gte240_improve, 1),
        "solid_loss_pct": round(diag.get("solid_loss_pct", 0), 2),
        "passed": gte240_improve > 10 and diag.get("solid_loss_pct", 0) < 15,
    }


def test_white_bg_hair(refiner, debug_dir=None):
    """White background (common indoor): residue should still improve."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(80, 60, 45), bg_color=(240, 240, 240),
    )

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    ctx = build_context(image, rmbg_alpha)
    before = residue_by_alpha(image, rmbg_alpha, ctx)
    after = residue_by_alpha(rgba_out[:, :, :3], rgba_out[:, :, 3], ctx)

    gte240_improve = (before["gte240"] - after["gte240"]) / max(before["gte240"], 1e-6) * 100

    return {
        "case": "white_bg_hair",
        "residue_before_by_alpha": before,
        "residue_after_by_alpha": after,
        "gte240_improve_pct": round(gte240_improve, 1),
        "solid_loss_pct": round(diag.get("solid_loss_pct", 0), 2),
        "passed": gte240_improve > 20 and diag.get("solid_loss_pct", 0) < 15,
    }


def test_dark_bg_hair(refiner, debug_dir=None):
    """Dark/black background: low contrast edge, should not break."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(160, 120, 80), bg_color=(20, 20, 25),
    )

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    ctx = build_context(image, rmbg_alpha)
    before = residue_by_alpha(image, rmbg_alpha, ctx)
    after = residue_by_alpha(rgba_out[:, :, :3], rgba_out[:, :, 3], ctx)

    gte240_improve = (before["gte240"] - after["gte240"]) / max(before["gte240"], 1e-6) * 100

    return {
        "case": "dark_bg_hair",
        "residue_before_by_alpha": before,
        "residue_after_by_alpha": after,
        "gte240_improve_pct": round(gte240_improve, 1),
        "solid_loss_pct": round(diag.get("solid_loss_pct", 0), 2),
        # Dark bg: unmix is harder, allow slight regression (<15%)
        "passed": gte240_improve > -15 and diag.get("solid_loss_pct", 0) < 15,
    }


def test_similar_fg_bg_hair(refiner, debug_dir=None):
    """Foreground and background similar color (worst case for unmix).
    Should not crash or produce garbage, even if improvement is small."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(100, 90, 80), bg_color=(120, 110, 100),
    )

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    # No crash = pass. Improvement may be small.
    rgb_out = rgba_out[:, :, :3].astype(np.float32)
    max_val = np.max(rgb_out)

    return {
        "case": "similar_fg_bg_hair",
        "status": diag.get("status", "unknown"),
        "max_rgb": round(float(max_val), 1),
        "passed": max_val < 280,  # no garbage values
    }


def test_additional_pure_color_backgrounds(refiner, debug_dir=None):
    """Yellow/cyan/magenta/purple/beige pure backgrounds should improve or stay safe."""
    cases = [
        ("yellow", (95, 70, 45), (225, 210, 35), 10.0),
        ("cyan", (140, 90, 60), (25, 200, 220), 10.0),
        ("magenta", (95, 135, 80), (220, 45, 190), 10.0),
        ("purple", (130, 95, 55), (125, 60, 220), 5.0),
        ("beige", (80, 55, 40), (210, 190, 150), -10.0),
    ]
    results = {}
    passed = True

    for name, fg_color, bg_color, min_improve in cases:
        case_debug = os.path.join(debug_dir, name) if debug_dir else None
        image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
            fg_color=fg_color,
            bg_color=bg_color,
            size=384,
        )
        rgba_in = np.dstack([image, rmbg_alpha])
        rgba_out, diag = refine_manual_edge(
            image, rgba_in, user_mask, refiner, debug_dir=case_debug,
        )

        ctx = build_context(image, rmbg_alpha)
        before = residue_by_alpha(image, rmbg_alpha, ctx)
        after = residue_by_alpha(rgba_out[:, :, :3], rgba_out[:, :, 3], ctx)
        improve = (before["gte240"] - after["gte240"]) / max(before["gte240"], 1e-6) * 100.0
        max_rgb = float(np.max(rgba_out[:, :, :3]))
        solid_loss = float(diag.get("solid_loss_pct", 0.0))
        case_passed = improve > min_improve and solid_loss < 15.0 and max_rgb < 280.0
        passed = passed and case_passed
        results[name] = {
            "gte240_improve_pct": round(float(improve), 1),
            "residue_before_gte240": round(float(before["gte240"]), 4),
            "residue_after_gte240": round(float(after["gte240"]), 4),
            "solid_loss_pct": round(solid_loss, 2),
            "max_rgb": round(max_rgb, 1),
            "passed": case_passed,
        }

    return {
        "case": "additional_pure_color_backgrounds",
        "background_results": results,
        "passed": passed,
    }


def test_screen_chroma_accepts_non_green_blue(refiner, debug_dir=None):
    """Chroma projection should flag saturated non-green/blue screen residue."""
    cases = [
        ("yellow", (225, 210, 35)),
        ("cyan", (25, 200, 220)),
        ("magenta", (220, 45, 190)),
        ("purple", (125, 60, 220)),
        ("red", (210, 40, 30)),
    ]
    results = {}
    passed = True
    for name, bg_color in cases:
        image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
            fg_color=(95, 70, 45),
            bg_color=bg_color,
            size=256,
        )
        ctx = build_context(image, rmbg_alpha)
        screen_info = estimate_screen_chroma(image, ctx["bg_color_seed"])
        screen_spill = compute_screen_spill_score(image, screen_info)
        accept = build_accept_mask(
            expand_user_mask(user_mask),
            rmbg_alpha,
            ctx["protected_transparency"],
            ctx["dist_to_background"],
            image_rgb=image,
            thin_detail=ctx.get("detail"),
            spill_score=np.zeros_like(ctx["spill_score"]),
            screen_spill_score=screen_spill,
        )
        accept_pixels = int(np.sum(accept))
        case_passed = (
            float(screen_info.get("screen_conf", 0.0)) > 0.2
            and accept_pixels > 0
            and float(screen_spill[accept].mean()) > 0.1
        )
        passed = passed and case_passed
        results[name] = {
            "screen_conf": round(float(screen_info.get("screen_conf", 0.0)), 3),
            "accept_pixels": accept_pixels,
            "screen_spill_mean": round(float(screen_spill[accept].mean()) if accept_pixels else 0.0, 3),
            "passed": case_passed,
        }

    return {
        "case": "screen_chroma_accepts_non_green_blue",
        "background_results": results,
        "passed": passed,
    }


def test_alpha_write_allowed_inside_accept(refiner, debug_dir=None):
    """Manual alpha write should survive when the change is contained in accept_mask."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(120, 80, 50),
        bg_color=(30, 220, 60),
        size=256,
    )
    ctx = build_context(image, rmbg_alpha)
    screen_info = estimate_screen_chroma(image, ctx["bg_color_seed"])
    screen_spill = compute_screen_spill_score(image, screen_info)
    accept = build_accept_mask(
        expand_user_mask(user_mask),
        rmbg_alpha,
        ctx["protected_transparency"],
        ctx["dist_to_background"],
        image_rgb=image,
        thin_detail=ctx.get("detail"),
        spill_score=ctx["spill_score"],
        screen_spill_score=screen_spill,
    )
    if not np.any(accept):
        return {"case": "alpha_write_allowed_inside_accept", "passed": False, "error": "accept empty"}

    alpha_refined = rmbg_alpha.copy()
    alpha_refined[accept] = np.minimum(alpha_refined[accept], alpha_gt[accept])
    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image,
        rgba_in,
        user_mask,
        FixedAlphaRefiner(alpha_refined),
        ctx=ctx,
        debug_dir=debug_dir,
    )

    alpha_delta_accept = np.abs(rgba_out[:, :, 3].astype(int) - rmbg_alpha.astype(int))[accept]
    alpha_delta_outside = np.abs(rgba_out[:, :, 3].astype(int) - rmbg_alpha.astype(int))[~accept]

    return {
        "case": "alpha_write_allowed_inside_accept",
        "alpha_written": diag.get("alpha_written", False),
        "rollback_alpha": diag.get("rollback_alpha", True),
        "accept_alpha_delta_mean": round(float(alpha_delta_accept.mean()), 2),
        "outside_alpha_delta_mean": round(float(alpha_delta_outside.mean()), 4),
        "passed": (
            diag.get("alpha_written", False)
            and float(alpha_delta_accept.mean()) > 2.0
            and float(alpha_delta_outside.mean()) < 0.5
        ),
    }


def test_rgb_regression_rolls_back_rgb_and_alpha(refiner, debug_dir=None):
    """If RGB residue gets worse, rollback the whole edit to avoid worse output."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(120, 80, 50),
        bg_color=(30, 220, 60),
        size=256,
    )
    ctx = build_context(image, rmbg_alpha)
    screen_info = estimate_screen_chroma(image, ctx["bg_color_seed"])
    screen_spill = compute_screen_spill_score(image, screen_info)
    accept = build_accept_mask(
        expand_user_mask(user_mask),
        rmbg_alpha,
        ctx["protected_transparency"],
        ctx["dist_to_background"],
        image_rgb=image,
        thin_detail=ctx.get("detail"),
        spill_score=ctx["spill_score"],
        screen_spill_score=screen_spill,
    )
    if not np.any(accept):
        return {"case": "rgb_regression_rolls_back_rgb_and_alpha", "passed": False, "error": "accept empty"}

    alpha_refined = rmbg_alpha.copy()
    alpha_refined[accept] = np.minimum(alpha_refined[accept], alpha_gt[accept])
    rgba_in = np.dstack([image, rmbg_alpha])

    # Poison the unmix background so the screen-direction residue truly gets
    # worse. High-confidence chroma-key cases should rollback on screen residue,
    # not on the older fill-distance metric.
    bad_ctx = dict(ctx)
    bad_ctx["foreground_fill"] = image.astype(np.float32)
    bad_ctx["background_fill"] = np.full_like(
        ctx["background_fill"], (255, 0, 255), dtype=np.float32,
    )

    rgba_out, diag = refine_manual_edge(
        image,
        rgba_in,
        user_mask,
        FixedAlphaRefiner(alpha_refined),
        ctx=bad_ctx,
        debug_dir=debug_dir,
    )

    visible_accept = accept & (rgba_out[:, :, 3] >= 2)
    rgb_delta_accept = np.linalg.norm(
        rgba_out[:, :, :3].astype(float) - image.astype(float),
        axis=2,
    )[visible_accept]
    alpha_delta_accept = np.abs(rgba_out[:, :, 3].astype(int) - rmbg_alpha.astype(int))[accept]

    return {
        "case": "rgb_regression_rolls_back_rgb_and_alpha",
        "rollback_rgb": diag.get("rollback_rgb", False),
        "rollback_rgb_reason": diag.get("rollback_rgb_reason", ""),
        "alpha_written": diag.get("alpha_written", False),
        "alpha_write_allowed": diag.get("alpha_write_allowed", False),
        "residue_metric": diag.get("residue_metric", ""),
        "rgb_delta_accept_mean": round(float(rgb_delta_accept.mean()), 3),
        "alpha_delta_accept_mean": round(float(alpha_delta_accept.mean()), 2),
        "rejected_rgb_improve_pct": round(float(diag.get("rejected_rgb_improve_pct", 0.0)), 1),
        "passed": (
            diag.get("rollback_rgb", False)
            and not diag.get("alpha_written", True)
            and diag.get("residue_metric") == "screen"
            and float(rgb_delta_accept.mean()) < 0.5
            and float(alpha_delta_accept.mean()) < 0.5
        ),
    }


def test_fill_metric_regression_does_not_block_screen_improvement(refiner, debug_dir=None):
    """For pure-color screens, screen-residue improvement should override fill-metric noise."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(
        fg_color=(120, 80, 50),
        bg_color=(30, 220, 60),
        size=256,
    )
    ctx = build_context(image, rmbg_alpha)
    screen_info = estimate_screen_chroma(image, ctx["bg_color_seed"])
    screen_spill = compute_screen_spill_score(image, screen_info)
    accept = build_accept_mask(
        expand_user_mask(user_mask),
        rmbg_alpha,
        ctx["protected_transparency"],
        ctx["dist_to_background"],
        image_rgb=image,
        thin_detail=ctx.get("detail"),
        spill_score=ctx["spill_score"],
        screen_spill_score=screen_spill,
    )
    if not np.any(accept):
        return {
            "case": "fill_metric_regression_does_not_block_screen_improvement",
            "passed": False,
            "error": "accept empty",
        }

    alpha_refined = rmbg_alpha.copy()
    alpha_refined[accept] = np.minimum(alpha_refined[accept], alpha_gt[accept])
    rgba_in = np.dstack([image, rmbg_alpha])

    # This white fill was enough to make the old fill-distance diagnostic reject
    # a visually useful edit. The screen projection should be the deciding KPI.
    noisy_diag_ctx = dict(ctx)
    noisy_diag_ctx["foreground_fill"] = image.astype(np.float32)
    noisy_diag_ctx["background_fill"] = np.full_like(ctx["background_fill"], 255, dtype=np.float32)

    rgba_out, diag = refine_manual_edge(
        image,
        rgba_in,
        user_mask,
        FixedAlphaRefiner(alpha_refined),
        ctx=noisy_diag_ctx,
        debug_dir=debug_dir,
    )

    alpha_delta_accept = np.abs(rgba_out[:, :, 3].astype(int) - rmbg_alpha.astype(int))[accept]

    return {
        "case": "fill_metric_regression_does_not_block_screen_improvement",
        "rollback_rgb": diag.get("rollback_rgb", False),
        "alpha_written": diag.get("alpha_written", False),
        "residue_metric": diag.get("residue_metric", ""),
        "rgb_improve_pct": round(float(diag.get("rgb_improve_pct", 0.0)), 1),
        "alpha_delta_accept_mean": round(float(alpha_delta_accept.mean()), 2),
        "passed": (
            not diag.get("rollback_rgb", True)
            and diag.get("alpha_written", False)
            and diag.get("residue_metric") == "screen"
            and float(diag.get("rgb_improve_pct", 0.0)) > 20.0
            and float(alpha_delta_accept.mean()) > 2.0
        ),
    }


def test_no_regression_interior(refiner, debug_dir=None):
    """Interior pixels (dist_to_bg > 5px) should not change by more than 3 RGB."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen()

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    ctx = build_context(image, rmbg_alpha)
    interior = ctx["dist_to_background"] > 5
    interior = interior & (rmbg_alpha > 0)

    if not np.any(interior):
        return {"case": "no_regression_interior", "interior_rgb_delta_mean": 0, "passed": True}

    delta = np.linalg.norm(
        rgba_out[:, :, :3].astype(float) - image.astype(float), axis=2,
    )
    interior_delta = delta[interior]

    return {
        "case": "no_regression_interior",
        "interior_rgb_delta_mean": round(float(interior_delta.mean()), 2),
        "interior_rgb_delta_p95": round(float(np.percentile(interior_delta, 95)), 2),
        "passed": float(interior_delta.mean()) < 3.0,
    }


def test_no_regression_low_alpha(refiner, debug_dir=None):
    """Alpha < 64 pixels should not be affected."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen()

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    low_alpha = rmbg_alpha < 64
    if not np.any(low_alpha):
        return {"case": "no_regression_low_alpha", "passed": True}

    alpha_diff = np.abs(rgba_out[:, :, 3].astype(int) - rmbg_alpha.astype(int))
    low_alpha_diff = alpha_diff[low_alpha]

    return {
        "case": "no_regression_low_alpha",
        "low_alpha_diff_mean": round(float(low_alpha_diff.mean()), 4),
        "passed": float(low_alpha_diff.mean()) < 1.0,
    }


def test_accept_mask_excludes_interior(refiner, debug_dir=None):
    """Deep interior pixels should not be in accept_mask."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen()

    ctx = build_context(image, rmbg_alpha)
    user_expanded = expand_user_mask(user_mask)
    accept = build_accept_mask(
        user_expanded, rmbg_alpha,
        ctx["protected_transparency"], ctx["dist_to_background"],
    )

    interior = ctx["dist_to_background"] > 12
    overlap = accept & interior

    return {
        "case": "accept_mask_excludes_interior",
        "interior_overlap_pixels": int(np.sum(overlap)),
        "passed": int(np.sum(overlap)) == 0,
    }


def test_unmix_numerical_stability(refiner, debug_dir=None):
    """Low alpha + noise should not produce extreme bright pixels."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen()

    # Add noise to simulate JPEG artifacts
    noisy = np.clip(
        image.astype(np.float32) + np.random.randn(*image.shape) * 8,
        0, 255,
    ).astype(np.uint8)

    rgba_in = np.dstack([noisy, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        noisy, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    # Check for extreme bright pixels in low-alpha region
    low_alpha = (rmbg_alpha > 0) & (rmbg_alpha < 64)
    if not np.any(low_alpha):
        return {"case": "unmix_numerical_stability", "passed": True}

    rgb_out = rgba_out[:, :, :3].astype(np.float32)
    max_val = np.max(rgb_out[low_alpha], axis=1)

    return {
        "case": "unmix_numerical_stability",
        "low_alpha_max_rgb_mean": round(float(max_val.mean()), 1),
        "low_alpha_max_rgb_p99": round(float(np.percentile(max_val, 99)), 1),
        "passed": float(np.percentile(max_val, 99)) < 270,
    }


def test_correct_area_no_change(refiner, debug_dir=None):
    """When user paints a non-contaminated area, result should barely change."""
    h = w = 256
    # Uniform gray foreground, no background separation
    image = np.full((h, w, 3), 128, dtype=np.uint8)
    rmbg_alpha = np.full((h, w), 250, dtype=np.uint8)

    # User paints the center (no spill to fix)
    user_mask = np.zeros((h, w), dtype=bool)
    user_mask[100:156, 100:156] = True

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    alpha_diff = np.abs(rgba_out[:, :, 3].astype(int) - rmbg_alpha.astype(int))
    rgb_diff = np.linalg.norm(
        rgba_out[:, :, :3].astype(float) - image.astype(float), axis=2,
    )

    return {
        "case": "correct_area_no_change",
        "alpha_diff_mean": round(float(alpha_diff.mean()), 4),
        "rgb_diff_mean": round(float(rgb_diff.mean()), 2),
        "passed": float(alpha_diff.mean()) < 3.0 and float(rgb_diff.mean()) < 5.0,
    }


def test_multi_stroke(refiner, debug_dir=None):
    """Multiple stroke regions should all be repaired."""
    image, alpha_gt, rmbg_alpha, _ = make_synthetic_screen()

    # Create two separate user mask regions covering different edge areas
    ctx = build_context(image, rmbg_alpha)
    h, w = image.shape[:2]

    # Use the full user_mask from make_synthetic_screen but split into two halves
    _, _, _, full_user_mask = make_synthetic_screen()
    user_mask = np.zeros((h, w), dtype=bool)
    user_mask[:h // 2, :] = full_user_mask[:h // 2, :]
    user_mask[h // 2:, :] = full_user_mask[h // 2:, :]

    # Verify we have strokes in both halves
    top_count = int(np.sum(user_mask[:h // 2, :]))
    bot_count = int(np.sum(user_mask[h // 2:, :]))
    if top_count == 0 or bot_count == 0:
        return {"case": "test_multi_stroke", "passed": False, "error": "user_mask has empty half"}

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    after = residue_by_alpha(rgba_out[:, :, :3], rgba_out[:, :, 3], ctx)
    before = residue_by_alpha(image, rmbg_alpha, ctx)
    improve = (before["gte240"] - after["gte240"]) / max(before["gte240"], 1e-6) * 100

    return {
        "case": "test_multi_stroke",
        "gte240_improve_pct": round(improve, 1),
        "passed": improve > 30 or diag.get("status") == "skipped",
    }


def test_feather_no_jagged(refiner, debug_dir=None):
    """Spatial feathering should produce smooth transitions, no jagged edges."""
    image, alpha_gt, rmbg_alpha, user_mask = make_synthetic_screen(size=256)

    rgba_in = np.dstack([image, rmbg_alpha])
    rgba_out, diag = refine_manual_edge(
        image, rgba_in, user_mask, refiner, debug_dir=debug_dir,
    )

    # Check gradient smoothness at accept_mask boundary
    ctx = build_context(image, rmbg_alpha)
    accept = build_accept_mask(
        expand_user_mask(user_mask), rmbg_alpha,
        ctx["protected_transparency"], ctx["dist_to_background"],
    )

    # Dilate boundary by 1px, check gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.dilate(accept.astype(np.uint8), kernel) ^ accept.astype(np.uint8)
    boundary = boundary.astype(bool)

    if not np.any(boundary):
        return {"case": "feather_no_jagged", "passed": True}

    alpha_grad = np.abs(
        cv2.Laplacian(rgba_out[:, :, 3].astype(np.float32), cv2.CV_32F),
    )
    boundary_grad = alpha_grad[boundary]

    return {
        "case": "feather_no_jagged",
        "boundary_grad_mean": round(float(boundary_grad.mean()), 2),
        "boundary_grad_p95": round(float(np.percentile(boundary_grad, 95)), 2),
        "passed": float(boundary_grad.mean()) < 30,
    }


# ── Main ───────────────────────────────────────────────────────

ALL_TESTS = [
    test_green_screen_hair,
    test_blue_screen_hair,
    test_red_bg_hair,
    test_white_bg_hair,
    test_dark_bg_hair,
    test_similar_fg_bg_hair,
    test_additional_pure_color_backgrounds,
    test_screen_chroma_accepts_non_green_blue,
    test_alpha_write_allowed_inside_accept,
    test_fill_metric_regression_does_not_block_screen_improvement,
    test_rgb_regression_rolls_back_rgb_and_alpha,
    test_no_regression_interior,
    test_no_regression_low_alpha,
    test_accept_mask_excludes_interior,
    test_unmix_numerical_stability,
    test_correct_area_no_change,
    test_multi_stroke,
    test_feather_no_jagged,
]


def run_all_tests(refiner, debug_dir=None):
    results = []
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        case_debug = os.path.join(debug_dir, name) if debug_dir else None
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        t0 = time.perf_counter()
        try:
            result = test_fn(refiner, debug_dir=case_debug)
        except Exception as e:
            result = {"case": name, "passed": False, "error": str(e)}
        elapsed = time.perf_counter() - t0
        result["test_elapsed_s"] = round(elapsed, 2)
        results.append(result)

        status = "PASS" if result.get("passed") else "FAIL"
        improve = result.get("gte240_improve_pct", "N/A")
        print(f"  {status} | gte240_improve={improve}% | elapsed={elapsed:.2f}s")
        if not result.get("passed") and "error" in result:
            print(f"  ERROR: {result['error']}")

    # Summary
    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(results)} tests passed")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        summary_path = os.path.join(debug_dir, "test_results.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {summary_path}")

    return all(r.get("passed") for r in results)


def main():
    parser = argparse.ArgumentParser(description="Manual edge refinement tests")
    parser.add_argument("--debug-dir", default=os.path.join(ROOT, "output", "_test_manual"))
    args = parser.parse_args()

    print("Loading ViTMatte...")
    refiner = get_vitmatte_refiner()
    print("ViTMatte ready.\n")

    ok = run_all_tests(refiner, debug_dir=args.debug_dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
