"""Verify RGB defringe diagnostics without loading RMBG/SAM models."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engines.rgba_postprocess import make_clean_rgba  # noqa: E402


def _synthetic_case(size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    h = w = size
    y, x = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w * 0.52, h * 0.48
    radius = min(h, w) * 0.29
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip((radius + 2.5 - dist) / 5.0, 0.0, 1.0)

    # Add high-alpha hair-like strands with blue contamination on their rims.
    for offset in (-30, -14, 12, 28):
        strand_x = cx + offset + np.sin((y - 20) / 18.0) * 8.0
        strand = np.exp(-((x - strand_x) ** 2) / 2.2)
        vertical = (y > 18) & (y < h - 18)
        alpha = np.maximum(alpha, strand * vertical * 0.96)

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
    return np.clip(image, 0, 255).astype(np.uint8), np.clip(alpha * 255, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a synthetic postprocess case and verify RGB defringe debug outputs."
    )
    parser.add_argument(
        "--debug-dir",
        default=os.path.join(ROOT, "output", "_verify_rgb_defringe"),
        help="Directory for debug images.",
    )
    args = parser.parse_args()

    image, alpha = _synthetic_case()
    os.makedirs(args.debug_dir, exist_ok=True)
    rgba = make_clean_rgba(image, alpha, debug_dir=args.debug_dir)
    Image.fromarray(rgba, "RGBA").save(os.path.join(args.debug_dir, "result_rgba.png"))

    required = [
        "52_rgb_defringe_delta.png",
        "53_rgb_residue_before.png",
        "54_rgb_residue_after.png",
        "55_rgb_residue_delta.png",
        "56_bg_confidence.png",
        "57_despill_projection.png",
        "58_screen_despill_strength.png",
        "result_rgba.png",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(args.debug_dir, name))]
    if missing:
        raise RuntimeError(f"Missing expected debug outputs: {missing}")

    print(f"[verify] RGB defringe postprocess outputs saved to: {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
