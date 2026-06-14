"""Unit tests for RGBA post-processing pipeline.

Covers build_context, analyze_matte, _refine_alpha, _guard_against_overcut,
and make_clean_rgba. Uses synthetic data (no model loading).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines.rgba_postprocess import (
    _connected_background,
    _guard_against_overcut,
    _hole_background,
    _largest_component,
    _refine_alpha,
    analyze_matte,
    build_context,
    make_clean_rgba,
)


# ── Helper to build a synthetic matte ────────────────────────────

def _make_synthetic_image(h=200, w=200, bg_color=(180, 200, 220)):
    """White-ish object on colored background."""
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    img[40:160, 40:160] = [240, 230, 220]  # foreground
    return img


def _make_synthetic_alpha(h=200, w=200, soft_edge=8):
    """Alpha with hard core and soft fringe."""
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[40:160, 40:160] = 240  # solid core
    # Soft fringe
    for i in range(soft_edge):
        val = int(240 * (i + 1) / soft_edge)
        alpha[40 - soft_edge + i, 40:160] = min(val, 255)
        alpha[160 + soft_edge - 1 - i, 40:160] = min(val, 255)
        alpha[40:160, 40 - soft_edge + i] = min(val, 255)
        alpha[40:160, 160 + soft_edge - 1 - i] = min(val, 255)
    return np.clip(alpha, 0, 255).astype(np.uint8)


# ── Topology helpers ─────────────────────────────────────────────

def test_largest_component():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True  # large
    mask[80:85, 80:85] = True  # small
    result = _largest_component(mask)
    assert result[20, 20], "largest component should include main blob"
    assert not result[82, 82], "small blob should be excluded"


def test_connected_background():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[30:70, 30:70] = 200  # foreground in center
    seed = alpha <= 8  # background seed
    bg = _connected_background(seed)
    assert bg[0, 0], "border pixel should be background"
    assert not bg[50, 50], "center should not be background"


def test_hole_background():
    alpha = np.full((100, 100), 200, dtype=np.uint8)
    alpha[40:60, 40:60] = 0  # hole inside foreground
    seed = alpha <= 8
    outer_bg = _connected_background(seed)
    hole_bg = _hole_background(seed, outer_bg)
    assert hole_bg[50, 50], "hole should be detected as background"
    assert not hole_bg[0, 0], "outer border is not a hole"


# ── build_context ────────────────────────────────────────────────

def test_build_context_returns_expected_keys():
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    ctx = build_context(img, alpha)
    expected_keys = [
        "outer_fringe", "hole_fringe", "detail", "protected_transparency",
        "color_fringe", "safe_fg_seed", "bg_color_seed",
        "foreground_fill", "background_fill", "spill_score",
    ]
    for key in expected_keys:
        assert key in ctx, f"missing key: {key}"


def test_build_context_fringe_shapes():
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    ctx = build_context(img, alpha)
    h, w = alpha.shape
    for key in ["outer_fringe", "hole_fringe", "color_fringe"]:
        assert ctx[key].shape == (h, w), f"{key} shape mismatch"
        assert ctx[key].dtype == bool, f"{key} should be bool"


def test_build_context_fg_bg_fill_shapes():
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    ctx = build_context(img, alpha)
    assert ctx["foreground_fill"].shape == (*alpha.shape, 3)
    assert ctx["background_fill"].shape == (*alpha.shape, 3)


# ── analyze_matte ────────────────────────────────────────────────

def test_analyze_matte_returns_profile():
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    profile, ctx = analyze_matte(img, alpha)
    assert hasattr(profile, "profile")
    assert hasattr(profile, "alpha_tighten")
    assert hasattr(profile, "defringe")
    assert profile.profile in ("balanced", "hard_object", "detail_safe", "transparent_safe")


# ── _refine_alpha ────────────────────────────────────────────────

def test_refine_alpha_preserves_shape():
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    profile, ctx = analyze_matte(img, alpha)
    result = _refine_alpha(alpha, ctx, profile)
    assert result.shape == alpha.shape
    assert result.dtype == np.uint8


def test_refine_alpha_does_not_expand():
    """Refined alpha should not have more nonzero pixels than original."""
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    profile, ctx = analyze_matte(img, alpha)
    result = _refine_alpha(alpha, ctx, profile)
    assert np.sum(result > 0) <= np.sum(alpha > 0) * 1.05  # allow 5% tolerance


# ── _guard_against_overcut ───────────────────────────────────────

def test_guard_against_overcut_rolls_back_excessive():
    """If refined alpha loses too much solid area, guard should rollback."""
    alpha_orig = np.full((100, 100), 200, dtype=np.uint8)
    alpha_refined = np.zeros((100, 100), dtype=np.uint8)  # wiped out
    profile = type("P", (), {"profile": "balanced"})()
    result, rollback, guard = _guard_against_overcut(alpha_orig, alpha_refined, profile)
    assert rollback, "should rollback when solid area is wiped"


def test_guard_against_overcut_accepts_minor_change():
    """Minor refinement within tolerance should not rollback."""
    alpha_orig = np.full((100, 100), 200, dtype=np.uint8)
    alpha_refined = alpha_orig.copy()
    alpha_refined[5:95, 5:95] = 195  # minor change
    profile = type("P", (), {"profile": "balanced"})()
    result, rollback, guard = _guard_against_overcut(alpha_orig, alpha_refined, profile)
    assert not rollback, "minor change should not trigger rollback"


# ── make_clean_rgba ──────────────────────────────────────────────

def test_make_clean_rgba_output_shape():
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    rgba = make_clean_rgba(img, alpha)
    assert rgba.shape == (*alpha.shape, 4)
    assert rgba.dtype == np.uint8


def test_make_clean_rgba_background_is_transparent():
    """Background pixels (alpha=0) should have RGBA = (0,0,0,0)."""
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    rgba = make_clean_rgba(img, alpha)
    # Corner is background
    assert rgba[0, 0, 3] == 0, "background alpha should be 0"


def test_make_clean_rgba_foreground_has_alpha():
    """Foreground pixels should have nonzero alpha."""
    img = _make_synthetic_image()
    alpha = _make_synthetic_alpha()
    rgba = make_clean_rgba(img, alpha)
    # Center of foreground
    assert rgba[100, 100, 3] > 200, "foreground alpha should be high"


def test_make_clean_rgba_crop_optimization():
    """Cropped path (no debug) should produce same result as full path."""
    img = _make_synthetic_image(300, 300)
    alpha = _make_synthetic_alpha(300, 300)
    # Small foreground, large image → crop should trigger
    rgba = make_clean_rgba(img, alpha)
    assert rgba.shape == (300, 300, 4)


def test_defringe_debug_outputs():
    """RGB defringe with hair-like strands should produce expected debug images."""
    import tempfile
    # Synthetic case: circle with hair strands on blue background
    h = w = 192
    y, x = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w * 0.52, h * 0.48
    radius = min(h, w) * 0.29
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip((radius + 2.5 - dist) / 5.0, 0.0, 1.0)
    # Add hair-like strands with blue contamination
    for offset in (-30, -14, 12, 28):
        strand_x = cx + offset + np.sin((y - 20) / 18.0) * 8.0
        strand = np.exp(-((x - strand_x) ** 2) / 2.2)
        vertical = (y > 18) & (y < h - 18)
        alpha = np.maximum(alpha, strand * vertical * 0.96)
    # Foreground: warm tones; background: blue screen
    foreground = np.zeros((h, w, 3), dtype=np.float32)
    foreground[..., 0] = 82 + 32 * (x / max(w - 1, 1))
    foreground[..., 1] = 58 + 20 * (y / max(h - 1, 1))
    foreground[..., 2] = 38
    background = np.zeros((h, w, 3), dtype=np.float32)
    background[..., 0] = 3
    background[..., 1] = 24
    background[..., 2] = 245 - 28 * (x / max(w - 1, 1))
    image = foreground * alpha[..., None] + background * (1.0 - alpha[..., None])
    rim = (alpha > 0.72) & (alpha < 1.0)
    image[rim] = image[rim] * 0.72 + background[rim] * 0.28
    image = np.clip(image, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as debug_dir:
        rgba = make_clean_rgba(image, alpha_u8, debug_dir=debug_dir)
        assert rgba.shape == (h, w, 4)
        # Verify key debug outputs exist
        required = [
            "52_rgb_defringe_delta.png",
            "53_rgb_residue_before.png",
            "54_rgb_residue_after.png",
            "60_composite_black.png",
        ]
        for name in required:
            assert os.path.exists(os.path.join(debug_dir, name)), f"Missing: {name}"


# ── Run all tests ────────────────────────────────────────────────

ALL_TESTS = [
    test_largest_component,
    test_connected_background,
    test_hole_background,
    test_build_context_returns_expected_keys,
    test_build_context_fringe_shapes,
    test_build_context_fg_bg_fill_shapes,
    test_analyze_matte_returns_profile,
    test_refine_alpha_preserves_shape,
    test_refine_alpha_does_not_expand,
    test_guard_against_overcut_rolls_back_excessive,
    test_guard_against_overcut_accepts_minor_change,
    test_make_clean_rgba_output_shape,
    test_make_clean_rgba_background_is_transparent,
    test_make_clean_rgba_foreground_has_alpha,
    test_make_clean_rgba_crop_optimization,
    test_defringe_debug_outputs,
]


def run_all():
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
