"""Unit tests for RMBG2Engine._clean_mask and _smooth_edge.

These are pure numpy functions that require no model loading.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines.rmbg2 import RMBG2Engine


# ── _clean_mask tests ────────────────────────────────────────────

def test_clean_mask_removes_noise():
    """Small isolated foreground blobs should be removed."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    # Large foreground
    alpha[50:150, 50:150] = 200
    # Tiny noise blob (area < 50 px)
    alpha[10:13, 10:13] = 200
    result = RMBG2Engine._clean_mask(alpha.copy())
    assert result[10:13, 10:13].sum() == 0, "noise blob should be removed"
    assert result[100, 100] > 0, "main foreground should remain"


def test_clean_mask_preserves_large_foreground():
    """Large foreground should not be touched."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[20:180, 20:180] = 230
    result = RMBG2Engine._clean_mask(alpha.copy())
    assert (result[30:170, 30:170] > 0).all(), "large foreground preserved"


def test_clean_mask_removes_haze():
    """Isolated faint alpha without solid support should be removed."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    # Solid foreground
    alpha[80:120, 80:120] = 200
    # Isolated haze (faint, not connected to solid)
    alpha[10:30, 170:190] = 50
    result = RMBG2Engine._clean_mask(alpha.copy())
    assert result[10:30, 170:190].sum() == 0, "isolated haze removed"
    assert result[100, 100] > 0, "solid foreground remains"


def test_clean_mask_zeroes_very_low_alpha():
    """Alpha < 10 should be zeroed."""
    alpha = np.full((100, 100), 5, dtype=np.uint8)
    alpha[40:60, 40:60] = 200
    result = RMBG2Engine._clean_mask(alpha.copy())
    assert (result[:40, :] == 0).all(), "very low alpha zeroed"
    assert result[50, 50] > 0, "foreground preserved"


def test_clean_mask_empty():
    """All-zero input should return all-zero."""
    alpha = np.zeros((100, 100), dtype=np.uint8)
    result = RMBG2Engine._clean_mask(alpha.copy())
    assert result.sum() == 0


# ── _smooth_edge tests ──────────────────────────────────────────

def test_smooth_edge_preserves_solid():
    """Solid foreground (alpha > 224) should not change."""
    alpha = np.full((100, 100), 250, dtype=np.uint8)
    result = RMBG2Engine._smooth_edge(alpha)
    np.testing.assert_allclose(result, alpha, atol=1)


def test_smooth_edge_preserves_background():
    """Solid background (alpha < 32) should not change."""
    alpha = np.zeros((100, 100), dtype=np.uint8)
    result = RMBG2Engine._smooth_edge(alpha)
    np.testing.assert_array_equal(result, alpha)


def test_smooth_edge_blends_transition():
    """Transition pixels (32 < alpha < 224) should be blended toward neighbors."""
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[40:60, 40:60] = 128  # transition zone
    alpha[35:40, 40:60] = 240  # solid neighbor
    result = RMBG2Engine._smooth_edge(alpha)
    # Boundary pixel of transition zone (adjacent to solid) should change
    # because Gaussian blur averages with the high-alpha neighbor
    assert result[40, 50] != alpha[40, 50], "boundary transition pixel should be smoothed"
    # Solid neighbor should stay
    np.testing.assert_allclose(result[37, 50], alpha[37, 50], atol=1)


def test_smooth_edge_dtype():
    """Output should be uint8."""
    alpha = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    result = RMBG2Engine._smooth_edge(alpha)
    assert result.dtype == np.uint8


# ── Run all tests ────────────────────────────────────────────────

ALL_TESTS = [
    test_clean_mask_removes_noise,
    test_clean_mask_preserves_large_foreground,
    test_clean_mask_removes_haze,
    test_clean_mask_zeroes_very_low_alpha,
    test_clean_mask_empty,
    test_smooth_edge_preserves_solid,
    test_smooth_edge_preserves_background,
    test_smooth_edge_blends_transition,
    test_smooth_edge_dtype,
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
