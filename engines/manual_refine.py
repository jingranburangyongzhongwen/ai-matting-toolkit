"""Manual edge refinement: user-painted mask -> ViTMatte alpha re-estimation -> regularized unmix.

Core pipeline:
  1. build_accept_mask (safety filter on user paint)
  2. build_spill_aware_trimap (for ViTMatte)
  3. ViTMatte alpha refinement (ROI-cropped)
  4. regularized_unmix (recover clean foreground RGB)
  5. spatial_gate + confidence gates (safe merge)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from engines.rgba_postprocess import build_context, _compute_spill_score, _safe_foreground_seed, _estimate_background_fill


def build_accept_mask(
    user_mask_expanded: np.ndarray,
    alpha: np.ndarray,
    protected_transparency: np.ndarray,
    dist_to_background: np.ndarray,
) -> np.ndarray:
    """Filter user paint to safe repair candidates. No alpha upper cutoff —
    alpha≈240-255 pseudo-solid pixels are exactly what we need to fix."""
    return (
        user_mask_expanded
        & (alpha > 0)
        & (~protected_transparency)
        & (dist_to_background <= 12)
    )


def expand_user_mask(user_mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Dilate user paint by a few pixels to cover nearby contaminated rim."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(
        user_mask.astype(np.uint8), kernel, iterations=iterations,
    ).astype(bool)


def build_spill_aware_trimap(
    alpha: np.ndarray,
    accept_mask: np.ndarray,
    bg_seed: np.ndarray,
) -> np.ndarray:
    """Build trimap for ViTMatte. Unknown band covers accept_mask edge + some interior."""
    trimap = np.zeros_like(alpha, dtype=np.uint8)

    # known_bg: alpha<=8 or bg_seed
    known_bg = (alpha <= 8) | bg_seed
    trimap[known_bg] = 0

    # Wide unknown band: accept_mask interior up to ~6px from edge
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    accept_eroded = cv2.erode(accept_mask.astype(np.uint8), kernel, iterations=3).astype(bool)
    accept_rim = accept_mask & (~accept_eroded)

    # Also include the original soft edge band outside accept_mask
    soft_edge = (alpha > 8) & (alpha < 240) & (~accept_mask)

    unknown = accept_rim | soft_edge
    trimap[unknown] = 127

    # known_fg: everything else that's not bg and not unknown
    known_fg = (~known_bg) & (~unknown)
    trimap[known_fg] = 255

    return trimap


def regularized_unmix(
    image: np.ndarray,
    alpha_new: np.ndarray,
    alpha_old: np.ndarray,
    background: np.ndarray,
    eps: float = 0.01,
) -> np.ndarray:
    """Tikhonov-regularized foreground recovery: F = a*(I-(1-a)*B)/(a^2+eps).
    Blends back to original RGB at low alpha to avoid noise amplification."""
    a = alpha_new.astype(np.float32)[..., None]
    I = image.astype(np.float32)
    B = background.astype(np.float32)
    if B.ndim == 1:
        B = B[None, None, :]

    F_unmixed = a * (I - (1.0 - a) * B) / (a * a + eps)
    F_unmixed = np.clip(F_unmixed, 0.0, 255.0)

    w = np.clip((alpha_new.astype(np.float32) - 0.05) / 0.15, 0.0, 1.0)[..., None]
    return w * F_unmixed + (1.0 - w) * I


def compute_spatial_gate(
    user_mask_expanded: np.ndarray,
    accept_mask: np.ndarray,
    feather_px: int = 5,
) -> np.ndarray:
    """Continuous feather from user intent region, hard-cutoff by accept_mask."""
    dist = cv2.distanceTransform(
        user_mask_expanded.astype(np.uint8), cv2.DIST_L2, 5,
    )
    feather_field = np.clip(dist / feather_px, 0.0, 1.0)
    return (feather_field * accept_mask).astype(np.float32)


def compute_confidence(
    alpha_new: np.ndarray,
    alpha_old: np.ndarray,
    bg_confidence: np.ndarray,
    fg_bg_sep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-unmix confidence gates. Recon gate computed separately with F_final."""
    delta = np.abs(alpha_new - alpha_old) / 255.0
    delta_gate = np.clip(delta / 0.08, 0.0, 1.0)

    bg_gate = np.clip((bg_confidence - 0.05) / 0.35, 0.0, 1.0)
    sep_gate = np.clip(fg_bg_sep / 0.05, 0.0, 1.0)

    alpha_conf = np.clip(bg_gate * delta_gate * sep_gate, 0.0, 1.0)
    rgb_conf = np.clip(bg_gate * sep_gate, 0.0, 1.0)
    return alpha_conf.astype(np.float32), rgb_conf.astype(np.float32)


def compute_recon_gate(
    image: np.ndarray,
    alpha_new: np.ndarray,
    F_final: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """Recon consistency gate using unmix-produced F_final (not old foreground_fill)."""
    I = image.astype(np.float32)
    a = alpha_new.astype(np.float32)[..., None]
    B = background.astype(np.float32)
    if B.ndim == 1:
        B = B[None, None, :]
    recon = a * F_final + (1.0 - a) * B
    recon_err = np.linalg.norm(I - recon, axis=2) / (np.sqrt(3.0) * 255.0)
    return np.clip(1.0 - recon_err / 0.12, 0.0, 1.0).astype(np.float32)


def _compute_fg_bg_sep(
    image: np.ndarray,
    foreground_fill: np.ndarray,
    background_fill: np.ndarray,
) -> np.ndarray:
    """Per-pixel foreground-background color separation in normalized RGB."""
    fg = foreground_fill.astype(np.float32)
    bg = background_fill.astype(np.float32)
    vec = bg - fg
    return (np.linalg.norm(vec, axis=2) / (np.sqrt(3.0) * 255.0)).astype(np.float32)


def _compute_bg_confidence_simple(
    alpha: np.ndarray,
    bg_color_seed: np.ndarray,
    background_fill: np.ndarray,
    dist_to_background: np.ndarray,
) -> np.ndarray:
    """Lightweight bg confidence: distance to seed * local fill agreement."""
    if not np.any(bg_color_seed):
        return np.zeros(alpha.shape, dtype=np.float32)

    dist_conf = np.clip((8.0 - dist_to_background) / 8.0, 0.0, 1.0)
    return dist_conf.astype(np.float32)


def refine_manual_edge(
    image_rgb: np.ndarray,
    current_rgba: np.ndarray,
    user_mask: np.ndarray,
    vitmatte_refiner,
    ctx: Optional[Dict[str, np.ndarray]] = None,
    debug_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Main entry: user paint -> ViTMatte alpha refinement -> regularized unmix -> safe merge.

    Args:
        image_rgb: original RGB image, HxWx3 uint8
        current_rgba: current RGBA result, HxWx4 uint8
        user_mask: user-painted mask, HxW bool
        vitmatte_refiner: ViTMatteRefiner instance
        ctx: optional pre-computed context from build_context()
        debug_dir: optional directory for debug outputs

    Returns:
        (repaired_rgba, diagnostics_dict)
    """
    import time
    t0 = time.perf_counter()

    alpha_old = current_rgba[:, :, 3].copy()
    rgb_old = current_rgba[:, :, :3].copy()

    # 1. Build context (reuse from rgba_postprocess)
    if ctx is None:
        ctx = build_context(image_rgb, alpha_old)

    user_mask_expanded = expand_user_mask(user_mask)

    # 2. Build accept_mask
    accept_mask = build_accept_mask(
        user_mask_expanded, alpha_old,
        ctx["protected_transparency"], ctx["dist_to_background"],
    )

    n_accept = int(np.sum(accept_mask))
    if n_accept == 0:
        diag = {"status": "skipped", "reason": "accept_mask empty", "elapsed_s": time.perf_counter() - t0}
        return current_rgba.copy(), diag

    # 3. ViTMatte refinement (optional — often changes alpha too little)
    trimap = build_spill_aware_trimap(alpha_old, accept_mask, ctx["background_seed"])
    alpha_vit = vitmatte_refiner.refine_with_trimap(
        image_rgb, alpha_old, trimap, accept_mask,
        mode="full", _debug_dir=debug_dir,
    )
    # Use ViTMatte alpha only where it changed significantly
    vit_delta = np.abs(alpha_vit.astype(float) - alpha_old.astype(float))
    use_vit = vit_delta > 15  # only trust ViTMatte where it made a real change
    alpha_new = np.where(use_vit, alpha_vit, alpha_old)

    # 4. Regularized unmix
    bg = ctx["background_fill"].astype(np.float32)
    alpha_new_f = alpha_new.astype(np.float32) / 255.0
    alpha_old_f = alpha_old.astype(np.float32) / 255.0
    F_final = regularized_unmix(image_rgb, alpha_new_f, alpha_old_f, bg)

    # 6. Spatial gate
    gate = compute_spatial_gate(user_mask_expanded, accept_mask, feather_px=5)

    # 7. Confidence gates
    fg_bg_sep = _compute_fg_bg_sep(image_rgb, ctx["foreground_fill"], bg)
    bg_conf = _compute_bg_confidence_simple(
        alpha_old, ctx["bg_color_seed"], bg, ctx["dist_to_background"],
    )
    alpha_conf, rgb_conf = compute_confidence(
        alpha_new_f, alpha_old_f, bg_conf, fg_bg_sep,
    )

    # 8. Recon gate (must use F_final, not foreground_fill)
    recon_gate = compute_recon_gate(image_rgb, alpha_new_f, F_final, bg)
    rgb_conf = rgb_conf * recon_gate

    # 9. Rollback safety check
    solid_before = int(np.sum(alpha_old >= 127))
    solid_after = int(np.sum(alpha_new >= 127))
    solid_loss = max(0, solid_before - solid_after) / max(solid_before, 1)
    alpha_l1 = float(np.mean(np.abs(alpha_new.astype(float) - alpha_old.astype(float))) / 255.0)

    rollback_alpha = solid_loss > 0.03 or alpha_l1 > 0.08

    # 10. Merge
    if rollback_alpha:
        # alpha risk too high: only accept RGB fix, keep old alpha
        alpha_out = alpha_old
        g3 = gate[..., None] * rgb_conf[..., None]
        rgb_out = rgb_old.astype(np.float32) * (1.0 - g3) + F_final * g3
    else:
        alpha_out_f = alpha_old_f * (1.0 - gate * alpha_conf) + alpha_new_f * gate * alpha_conf
        alpha_out = np.clip(alpha_out_f * 255.0, 0, 255).round().astype(np.uint8)
        g3 = gate[..., None] * rgb_conf[..., None]
        rgb_out = rgb_old.astype(np.float32) * (1.0 - g3) + F_final * g3

    rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)

    # 11. Clean invisible pixels
    result_rgba = np.dstack([rgb_out, alpha_out])
    result_rgba[result_rgba[:, :, 3] < 2, :3] = 0

    elapsed = time.perf_counter() - t0

    # Diagnostics
    # Alpha change diagnostics on accept_mask
    alpha_delta_on_accept = np.abs(
        alpha_new[accept_mask].astype(float) - alpha_old[accept_mask].astype(float)
    ) if n_accept else np.array([0.0])

    # Edge smoothness on accept_mask fringe
    accept_fringe = accept_mask & ctx["fringe"]
    if np.any(accept_fringe):
        a_f = alpha_out.astype(np.float32) / 255.0
        lap = np.abs(cv2.Laplacian(a_f, cv2.CV_32F))
        lap_vals = lap[accept_fringe]
        lap_p95 = float(np.percentile(lap_vals, 95))
        smooth_score = float(np.clip(lap_p95 / 0.3, 0.0, 1.0))
    else:
        lap_p95 = 0.0
        smooth_score = 0.0

    diag: Dict[str, object] = {
        "status": "applied",
        "accept_pixels": n_accept,
        "user_paint_pixels": int(np.sum(user_mask)),
        "solid_loss_pct": solid_loss * 100.0,
        "alpha_l1_delta": alpha_l1,
        "rollback_alpha": rollback_alpha,
        "gate_mean": float(gate[accept_mask].mean()) if n_accept else 0.0,
        "alpha_conf_mean": float(alpha_conf[accept_mask].mean()) if n_accept else 0.0,
        "rgb_conf_mean": float(rgb_conf[accept_mask].mean()) if n_accept else 0.0,
        "alpha_delta_mean": float(alpha_delta_on_accept.mean()),
        "alpha_delta_p95": float(np.percentile(alpha_delta_on_accept, 95)),
        "alpha_new_mean_on_accept": float(alpha_new[accept_mask].mean()) if n_accept else 0.0,
        "alpha_old_mean_on_accept": float(alpha_old[accept_mask].mean()) if n_accept else 0.0,
        "edge_lap_p95": lap_p95,
        "edge_smooth_score": smooth_score,
        "elapsed_s": elapsed,
    }

    # Residue by alpha band
    eval_mask = accept_mask & (alpha_old > 0)
    if np.any(eval_mask):
        spill_before, _, _ = _compute_spill_score(
            rgb_old, ctx["foreground_fill"], ctx["background_fill"], mask=eval_mask,
        )
        spill_after, _, _ = _compute_spill_score(
            rgb_out, ctx["foreground_fill"], ctx["background_fill"], mask=eval_mask,
        )
        a_eval = alpha_old[eval_mask].astype(np.float32) / 255.0
        vis_before = spill_before[eval_mask] * a_eval
        vis_after = spill_after[eval_mask] * a_eval
        bands = {
            "lt64": eval_mask & (alpha_old < 64),
            "64_180": eval_mask & (alpha_old >= 64) & (alpha_old < 180),
            "180_240": eval_mask & (alpha_old >= 180) & (alpha_old < 240),
            "gte240": eval_mask & (alpha_old >= 240),
        }
        residue_before, residue_after = {}, {}
        for name, band in bands.items():
            if np.any(band):
                # compute spill on this band
                sb, _, _ = _compute_spill_score(
                    rgb_old, ctx["foreground_fill"], ctx["background_fill"], mask=band,
                )
                sa, _, _ = _compute_spill_score(
                    rgb_out, ctx["foreground_fill"], ctx["background_fill"], mask=band,
                )
                ab = alpha_old[band].astype(np.float32) / 255.0
                residue_before[name] = float((sb[band] * ab).mean())
                residue_after[name] = float((sa[band] * ab).mean())
            else:
                residue_before[name] = 0.0
                residue_after[name] = 0.0
        diag["residue_before_by_alpha"] = residue_before
        diag["residue_after_by_alpha"] = residue_after
        diag["visible_improve_pct"] = (
            float((vis_before.mean() - vis_after.mean()) / max(vis_before.mean(), 1e-6) * 100.0)
        )

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray(alpha_out, "L").save(os.path.join(debug_dir, "manual_alpha_out.png"))
        Image.fromarray(rgb_out, "RGB").save(os.path.join(debug_dir, "manual_rgb_out.png"))
        if "residue_before_by_alpha" in diag:
            print(f"[manual refine] residue gte240: "
                  f"{diag['residue_before_by_alpha'].get('gte240', 0):.3f} -> "
                  f"{diag['residue_after_by_alpha'].get('gte240', 0):.3f}")
        print(f"[manual refine] elapsed={elapsed:.2f}s accept={n_accept}px "
              f"solid_loss={solid_loss*100:.2f}% rollback_alpha={rollback_alpha}")

    return result_rgba, diag
