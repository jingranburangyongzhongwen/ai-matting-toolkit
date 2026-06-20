"""Test _sam_strict_alpha band behavior: interior solid pixels must stay opaque."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_logic.tab2 import _sam_strict_alpha


def _make_square_mask(size=200, square_size=100):
    """Create a centered square binary mask and mock logits."""
    mask = np.zeros((size, size), dtype=bool)
    offset = (size - square_size) // 2
    mask[offset:offset + square_size, offset:offset + square_size] = True

    # Mock logits: positive inside (strong), negative outside (strong)
    logits = np.full((size, size), -10.0, dtype=np.float32)
    logits[mask] = 10.0  # Very high confidence inside
    # Add some gradient at the edge for realism
    for y in range(size):
        for x in range(size):
            if mask[y, x]:
                dist_to_edge = min(
                    y - offset, offset + square_size - 1 - y,
                    x - offset, offset + square_size - 1 - x,
                )
                if dist_to_edge <= 5:
                    logits[y, x] = float(dist_to_edge + 1)  # 1-6 near edge
    return mask, logits


def test_interior_stays_opaque():
    """Interior solid pixels must remain alpha=255 after band processing."""
    mask, logits = _make_square_mask(size=200, square_size=100)
    image_shape = (200, 200, 3)

    alpha, box = _sam_strict_alpha(
        mask, [], [], None, image_shape, logits=logits,
    )

    # Interior pixels (well inside the mask) must be 255
    interior = mask.copy()
    # Erode 15px to get truly interior pixels
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    interior_core = cv2.erode(interior.astype(np.uint8), kernel) > 0

    interior_alpha = alpha[interior_core]
    assert np.all(interior_alpha == 255), (
        f"Interior alpha should be 255, got min={interior_alpha.min()}, "
        f"mean={interior_alpha.mean():.1f}, "
        f"pixels_below_255={np.sum(interior_alpha < 255)}"
    )


def test_no_alpha_outside_mask():
    """Pixels outside the mask boundary must remain alpha=0."""
    mask, logits = _make_square_mask(size=200, square_size=100)
    image_shape = (200, 200, 3)

    alpha, box = _sam_strict_alpha(
        mask, [], [], None, image_shape, logits=logits,
    )

    # Pixels far outside mask must be 0
    import cv2
    outside = (~mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    outside_core = cv2.erode(outside, kernel) > 0

    outside_alpha = alpha[outside_core]
    assert np.all(outside_alpha == 0), (
        f"Outside alpha should be 0, got max={outside_alpha.max()}, "
        f"pixels_above_0={np.sum(outside_alpha > 0)}"
    )


def test_edge_band_no_exterior_leak():
    """Band processing must not create non-zero alpha outside the mask."""
    mask, logits = _make_square_mask(size=200, square_size=100)
    image_shape = (200, 200, 3)

    alpha, box = _sam_strict_alpha(
        mask, [], [], None, image_shape, logits=logits,
    )

    # No pixels outside the original mask should have non-zero alpha
    # (the old code would leak small sigmoid values outside the mask)
    outside_mask = ~mask
    outside_alpha = alpha[outside_mask]
    assert np.all(outside_alpha == 0), (
        f"No alpha should leak outside mask, but found "
        f"{np.sum(outside_alpha > 0)} non-zero pixels outside, "
        f"max={outside_alpha.max()}"
    )


def test_no_logits_binary_output():
    """Without logits, output should be strictly binary (0 or 255)."""
    mask, _ = _make_square_mask(size=200, square_size=100)
    image_shape = (200, 200, 3)

    alpha, box = _sam_strict_alpha(
        mask, [], [], None, image_shape, logits=None,
    )

    unique = np.unique(alpha)
    assert set(unique.tolist()).issubset({0, 255}), (
        f"Without logits, alpha should be binary, got unique values: {unique}"
    )


def test_thin_structure_preserved():
    """Thin structures (e.g., 20px wide) should maintain high interior alpha."""
    size = 200
    mask = np.zeros((size, size), dtype=bool)
    # 20px wide vertical bar
    mask[20:180, 90:110] = True

    # Mock logits: high inside, gradient at edges
    logits = np.full((size, size), -10.0, dtype=np.float32)
    logits[mask] = 8.0
    # Edge gradient
    for y in range(20, 180):
        for x in range(90, 110):
            dist_to_edge = min(x - 90, 109 - x, y - 20, 179 - y)
            if dist_to_edge <= 3:
                logits[y, x] = float(dist_to_edge + 1)

    image_shape = (size, size, 3)
    alpha, box = _sam_strict_alpha(
        mask, [], [], None, image_shape, logits=logits,
    )

    # Center column of the bar should be fully opaque
    center_col = alpha[100, 90:110]
    # At least the center pixels should be 255
    center_pixels = alpha[100, 95:105]
    assert np.all(center_pixels == 255), (
        f"Thin structure center should be 255, got {center_pixels}"
    )


# ── Run all ────────────────────────────────────────────────────────

ALL_TESTS = [
    test_interior_stays_opaque,
    test_no_alpha_outside_mask,
    test_edge_band_no_exterior_leak,
    test_no_logits_binary_output,
    test_thin_structure_preserved,
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
