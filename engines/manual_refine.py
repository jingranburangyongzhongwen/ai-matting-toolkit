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


RGB_DISTANCE_MAX = float(np.sqrt(3.0) * 255.0)
SCREEN_EPS = 1e-6


def _chroma_plane(image: np.ndarray) -> np.ndarray:
    """Remove luminance so hue-like screen directions work for any pure color."""
    rgb = image.astype(np.float32)
    return rgb - rgb.mean(axis=2, keepdims=True)


def estimate_screen_chroma(
    image: np.ndarray,
    bg_seed: np.ndarray,
) -> Dict[str, object]:
    """Estimate the background chroma direction from reliable bg pixels.

    This is intentionally color-agnostic: green/blue/red/yellow/cyan/etc. all
    become a unit vector in the chroma plane. Low-saturation backgrounds simply
    get low confidence and fall back to existing distance-based gates.
    """
    chroma = _chroma_plane(image)
    seed = bg_seed.astype(bool)
    if not np.any(seed):
        return {
            "S_hat": np.zeros(3, dtype=np.float32),
            "sat": 0.0,
            "consistency": 0.0,
            "screen_conf": 0.0,
        }

    samples = chroma[seed]
    norms = np.linalg.norm(samples, axis=1)
    vivid = norms > 6.0
    if np.any(vivid):
        samples = samples[vivid]
        norms = norms[vivid]

    s_vec = np.median(samples, axis=0).astype(np.float32)
    sat = float(np.linalg.norm(s_vec) / 255.0)
    if sat < SCREEN_EPS:
        return {
            "S_hat": np.zeros(3, dtype=np.float32),
            "sat": sat,
            "consistency": 0.0,
            "screen_conf": 0.0,
        }

    s_hat = s_vec / (np.linalg.norm(s_vec) + SCREEN_EPS)
    unit = samples / (norms[:, None] + SCREEN_EPS)
    consistency = float(np.clip(np.median(unit @ s_hat), 0.0, 1.0))
    sat_gate = float(np.clip((sat - 0.04) / 0.14, 0.0, 1.0))
    screen_conf = float(np.clip(sat_gate * consistency, 0.0, 1.0))
    return {
        "S_hat": s_hat.astype(np.float32),
        "sat": sat,
        "consistency": consistency,
        "screen_conf": screen_conf,
    }


def compute_screen_spill_score(
    image: np.ndarray,
    screen_info: Dict[str, object],
) -> np.ndarray:
    """Screen-color residue score from projection onto the estimated chroma axis."""
    s_hat = np.asarray(screen_info.get("S_hat", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    screen_conf = float(screen_info.get("screen_conf", 0.0))
    if screen_conf <= 0.0 or np.linalg.norm(s_hat) < SCREEN_EPS:
        return np.zeros(image.shape[:2], dtype=np.float32)

    proj = np.tensordot(_chroma_plane(image), s_hat, axes=([2], [0])) / 255.0
    # Keep the threshold low enough for warm/yellow/beige screens, but scale by
    # screen confidence so neutral backgrounds do not become aggressive.
    score = np.clip((proj - 0.025) / 0.22, 0.0, 1.0) * screen_conf
    return score.astype(np.float32)


def compute_screen_direction_gate(
    image: np.ndarray,
    foreground: np.ndarray,
    screen_info: Dict[str, object],
) -> np.ndarray:
    """Accept RGB when the cleaned foreground moves away from screen chroma."""
    s_hat = np.asarray(screen_info.get("S_hat", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    screen_conf = float(screen_info.get("screen_conf", 0.0))
    if screen_conf <= 0.0 or np.linalg.norm(s_hat) < SCREEN_EPS:
        return np.zeros(image.shape[:2], dtype=np.float32)

    before = np.tensordot(_chroma_plane(image), s_hat, axes=([2], [0])) / 255.0
    after = np.tensordot(_chroma_plane(foreground), s_hat, axes=([2], [0])) / 255.0
    reduction = before - after
    return (np.clip((reduction + 0.015) / 0.08, 0.0, 1.0) * screen_conf).astype(np.float32)


def build_accept_mask(
    user_mask_expanded: np.ndarray,
    alpha: np.ndarray,
    protected_transparency: np.ndarray,
    dist_to_background: np.ndarray,
    image_rgb: Optional[np.ndarray] = None,
    thin_detail: Optional[np.ndarray] = None,
    spill_score: Optional[np.ndarray] = None,
    screen_spill_score: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Filter user paint to safe repair candidates. No alpha upper cutoff."""
    if thin_detail is None:
        thin_detail = np.zeros(alpha.shape, dtype=bool)
    else:
        thin_detail = thin_detail.astype(bool)
    if spill_score is None:
        spill_score = np.zeros(alpha.shape, dtype=np.float32)
    else:
        spill_score = spill_score.astype(np.float32)
    if screen_spill_score is None:
        screen_spill_score = np.zeros(alpha.shape, dtype=np.float32)
    else:
        screen_spill_score = screen_spill_score.astype(np.float32)
    repair_score = np.maximum(spill_score, screen_spill_score)

    large_smooth_region = np.zeros(alpha.shape, dtype=bool)
    if image_rgb is not None:
        large_smooth_region = (
            (_local_color_variance(image_rgb) < 0.035)
            & (_local_gradient_magnitude(image_rgb) < 0.045)
            & (~thin_detail)
        )

    safe_to_refine = (
        (dist_to_background <= 6.0)
        | thin_detail
        | (
            (dist_to_background <= 12.0)
            & (repair_score >= 0.38)
            & (~large_smooth_region)
        )
    )

    return (
        user_mask_expanded
        & (alpha > 0)
        & (~protected_transparency)
        & safe_to_refine
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
    """Build trimap for ViTMatte; the whole accepted repair area is unknown."""
    trimap = np.zeros_like(alpha, dtype=np.uint8)

    known_bg = (alpha <= 8) | bg_seed
    trimap[known_bg] = 0

    # Manual repair must re-estimate pseudo-solid hair, so accept_mask is all UNKNOWN.
    soft_edge = (alpha > 8) & (alpha < 240) & (~accept_mask)
    unknown = accept_mask | soft_edge
    trimap[unknown] = 127

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


def _local_color_variance(image: np.ndarray, ksize: int = 11) -> np.ndarray:
    """Normalized local RGB standard deviation; low values indicate smooth regions."""
    rgb = image.astype(np.float32)
    mean = cv2.boxFilter(rgb, cv2.CV_32F, (ksize, ksize), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(rgb * rgb, cv2.CV_32F, (ksize, ksize), normalize=True, borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return (np.sqrt(np.sum(var, axis=2)) / RGB_DISTANCE_MAX).astype(np.float32)


def _local_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Normalized Sobel gradient magnitude; hair-like regions tend to be high frequency."""
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return np.clip(grad / (4.0 * 255.0), 0.0, 1.0).astype(np.float32)


def compute_confidence(
    alpha_new: np.ndarray,
    alpha_old: np.ndarray,
    bg_confidence: np.ndarray,
    fg_bg_sep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-unmix confidence gates. Recon gate computed separately with F_final."""
    # alpha_new/alpha_old are normalized floats in [0, 1].
    delta = np.abs(alpha_new - alpha_old)
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


def _residue_diagnostics(
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    alpha_before: np.ndarray,
    alpha_after: np.ndarray,
    ctx: Dict[str, np.ndarray],
    eval_mask: np.ndarray,
) -> Dict[str, object]:
    """Compute result-level RGB residue metrics on the accepted repair area."""
    if not np.any(eval_mask):
        return {}

    spill_before, _, _ = _compute_spill_score(
        rgb_before, ctx["foreground_fill"], ctx["background_fill"], mask=eval_mask,
    )
    spill_after, _, _ = _compute_spill_score(
        rgb_after, ctx["foreground_fill"], ctx["background_fill"], mask=eval_mask,
    )
    a_before_eval = alpha_before[eval_mask].astype(np.float32) / 255.0
    a_after_eval = alpha_after[eval_mask].astype(np.float32) / 255.0
    vis_before = spill_before[eval_mask] * a_before_eval
    vis_after = spill_after[eval_mask] * a_after_eval

    bands = {
        "lt64": eval_mask & (alpha_before < 64),
        "64_180": eval_mask & (alpha_before >= 64) & (alpha_before < 180),
        "180_240": eval_mask & (alpha_before >= 180) & (alpha_before < 240),
        "gte240": eval_mask & (alpha_before >= 240),
    }
    residue_before, residue_after = {}, {}
    for name, band in bands.items():
        if np.any(band):
            sb, _, _ = _compute_spill_score(
                rgb_before, ctx["foreground_fill"], ctx["background_fill"], mask=band,
            )
            sa, _, _ = _compute_spill_score(
                rgb_after, ctx["foreground_fill"], ctx["background_fill"], mask=band,
            )
            ab = alpha_before[band].astype(np.float32) / 255.0
            aa = alpha_after[band].astype(np.float32) / 255.0
            residue_before[name] = float((sb[band] * ab).mean())
            residue_after[name] = float((sa[band] * aa).mean())
        else:
            residue_before[name] = 0.0
            residue_after[name] = 0.0

    return {
        "residue_metric": "fill",
        "residue_before_by_alpha": residue_before,
        "residue_after_by_alpha": residue_after,
        "visible_improve_pct": float(
            (vis_before.mean() - vis_after.mean()) / max(vis_before.mean(), 1e-6) * 100.0
        ),
    }


def _screen_residue_diagnostics(
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    alpha_before: np.ndarray,
    alpha_after: np.ndarray,
    screen_info: Dict[str, object],
    eval_mask: np.ndarray,
) -> Dict[str, object]:
    """Compute residue using absolute projection onto the estimated screen chroma."""
    if not np.any(eval_mask):
        return {}

    score_before = compute_screen_spill_score(rgb_before, screen_info)
    score_after = compute_screen_spill_score(rgb_after, screen_info)
    a_before_eval = alpha_before[eval_mask].astype(np.float32) / 255.0
    a_after_eval = alpha_after[eval_mask].astype(np.float32) / 255.0
    vis_before = score_before[eval_mask] * a_before_eval
    vis_after = score_after[eval_mask] * a_after_eval

    bands = {
        "lt64": eval_mask & (alpha_before < 64),
        "64_180": eval_mask & (alpha_before >= 64) & (alpha_before < 180),
        "180_240": eval_mask & (alpha_before >= 180) & (alpha_before < 240),
        "gte240": eval_mask & (alpha_before >= 240),
    }
    residue_before, residue_after = {}, {}
    for name, band in bands.items():
        if np.any(band):
            ab = alpha_before[band].astype(np.float32) / 255.0
            aa = alpha_after[band].astype(np.float32) / 255.0
            residue_before[name] = float((score_before[band] * ab).mean())
            residue_after[name] = float((score_after[band] * aa).mean())
        else:
            residue_before[name] = 0.0
            residue_after[name] = 0.0

    return {
        "residue_metric": "screen",
        "residue_before_by_alpha": residue_before,
        "residue_after_by_alpha": residue_after,
        "visible_improve_pct": float(
            (vis_before.mean() - vis_after.mean()) / max(vis_before.mean(), 1e-6) * 100.0
        ),
    }


def _select_residue_diagnostics(
    fill_diag: Dict[str, object],
    screen_diag: Dict[str, object],
    screen_info: Dict[str, object],
    screen_spill_score: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, object]:
    """Prefer screen-anchored residue for high-confidence chroma-key backgrounds."""
    if not screen_diag:
        return fill_diag
    screen_conf = float(screen_info.get("screen_conf", 0.0))
    if screen_conf < 0.45 or not np.any(eval_mask):
        return fill_diag

    visible_screen = screen_spill_score[eval_mask]
    if visible_screen.size and (float(visible_screen.mean()) >= 0.02 or float(visible_screen.max()) >= 0.12):
        return screen_diag
    return fill_diag


def _should_rollback_rgb(residue_diag: Dict[str, object]) -> Tuple[bool, str]:
    """Reject RGB edits when result-level residue clearly regresses."""
    if not residue_diag:
        return False, ""

    improve = float(residue_diag.get("visible_improve_pct", 0.0))
    if improve < -2.0:
        return True, "visible_residue_regressed"

    before = residue_diag.get("residue_before_by_alpha", {})
    after = residue_diag.get("residue_after_by_alpha", {})
    gte240_before = float(before.get("gte240", 0.0))
    gte240_after = float(after.get("gte240", 0.0))
    if gte240_before > 1e-4 and gte240_after > gte240_before * 1.02:
        return True, "gte240_residue_regressed"

    return False, ""


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
    spill_score: Optional[np.ndarray] = None,
    screen_spill_score: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Lightweight bg confidence: distance to seed * local fill agreement."""
    dist_conf = np.clip((8.0 - dist_to_background) / 8.0, 0.0, 1.0)
    if spill_score is None:
        spill_score = np.zeros(alpha.shape, dtype=np.float32)
    if screen_spill_score is None:
        screen_spill_score = np.zeros(alpha.shape, dtype=np.float32)

    repair_score = np.maximum(spill_score.astype(np.float32), screen_spill_score.astype(np.float32))
    spill_conf = np.clip((16.0 - dist_to_background) / 16.0, 0.0, 1.0) * repair_score
    if not np.any(bg_color_seed) and not np.any(repair_score > 0):
        return np.zeros(alpha.shape, dtype=np.float32)

    return np.maximum(dist_conf, spill_conf).astype(np.float32)


def refine_manual_edge(
    image_rgb: np.ndarray,
    current_rgba: np.ndarray,
    user_mask: np.ndarray,
    vitmatte_refiner,
    ctx: Optional[Dict[str, np.ndarray]] = None,
    debug_dir: Optional[str] = None,
    verbose: bool = False,
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
    screen_info = estimate_screen_chroma(image_rgb, ctx["bg_color_seed"])
    screen_spill_score = compute_screen_spill_score(image_rgb, screen_info)

    user_mask_expanded = expand_user_mask(user_mask)

    # 2. Build accept_mask
    accept_mask = build_accept_mask(
        user_mask_expanded, alpha_old,
        ctx["protected_transparency"], ctx["dist_to_background"],
        image_rgb=image_rgb,
        thin_detail=ctx.get("detail"),
        spill_score=ctx.get("spill_score"),
        screen_spill_score=screen_spill_score,
    )

    n_accept = int(np.sum(accept_mask))
    if n_accept == 0:
        diag = {"status": "skipped", "reason": "accept_mask empty", "elapsed_s": time.perf_counter() - t0}
        if verbose or debug_dir:
            print(
                "[manual refine] skipped accept=0 "
                f"screen_conf={float(screen_info.get('screen_conf', 0.0)):.2f}"
            )
        return current_rgba.copy(), diag

    # 3. ViTMatte refinement (optional — often changes alpha too little)
    trimap = build_spill_aware_trimap(alpha_old, accept_mask, ctx["background_seed"])
    alpha_vit = vitmatte_refiner.refine_with_trimap(
        image_rgb, alpha_old, trimap, accept_mask,
        mode="full", _debug_dir=debug_dir,
    )
    # Use ViTMatte alpha only where it changed significantly
    vit_delta = np.abs(alpha_vit.astype(float) - alpha_old.astype(float))
    strong_repair = accept_mask & (
        ctx.get("detail", np.zeros_like(accept_mask, dtype=bool))
        | (np.maximum(ctx["spill_score"], screen_spill_score) >= 0.38)
    )
    use_vit = (vit_delta > 15) | (strong_repair & (vit_delta > 6))
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
        spill_score=ctx["spill_score"],
        screen_spill_score=screen_spill_score,
    )
    alpha_conf, rgb_conf = compute_confidence(
        alpha_new_f, alpha_old_f, bg_conf, fg_bg_sep,
    )

    # 8. Recon gate (must use F_final, not foreground_fill)
    recon_gate = compute_recon_gate(image_rgb, alpha_new_f, F_final, bg)
    direction_gate = compute_screen_direction_gate(image_rgb, F_final, screen_info)
    direction_region = (
        accept_mask
        & (np.maximum(ctx["spill_score"], screen_spill_score) >= 0.38)
        & (ctx.get("detail", np.zeros_like(accept_mask, dtype=bool)) | (ctx["dist_to_background"] <= 12.0))
    )
    effective_recon_gate = recon_gate.copy()
    effective_recon_gate[direction_region] = np.maximum(
        effective_recon_gate[direction_region],
        direction_gate[direction_region],
    )
    rgb_conf = rgb_conf * effective_recon_gate

    # 9. Rollback safety check
    solid_before = int(np.sum(alpha_old >= 127))
    solid_after = int(np.sum(alpha_new >= 127))
    solid_loss = max(0, solid_before - solid_after) / max(solid_before, 1)
    alpha_l1 = float(np.mean(np.abs(alpha_new.astype(float) - alpha_old.astype(float))) / 255.0)

    unpainted_alpha_delta = np.abs(alpha_new.astype(float) - alpha_old.astype(float))[~accept_mask]
    unpainted_alpha_l1 = (
        float(unpainted_alpha_delta.mean() / 255.0)
        if unpainted_alpha_delta.size else 0.0
    )
    accept_alpha_l1 = float(
        np.mean(np.abs(alpha_new[accept_mask].astype(float) - alpha_old[accept_mask].astype(float))) / 255.0
    )
    accept_solid_before = int(np.sum(alpha_old[accept_mask] >= 127))
    accept_solid_after = int(np.sum(alpha_new[accept_mask] >= 127))
    accept_solid_loss = max(0, accept_solid_before - accept_solid_after) / max(accept_solid_before, 1)
    alpha_write_allowed = (
        (unpainted_alpha_l1 <= 0.002)
        and (accept_solid_loss <= 0.45)
        and (accept_alpha_l1 <= 0.35)
    )
    rollback_alpha = not alpha_write_allowed

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

    eval_mask = accept_mask & (alpha_old > 0)
    fill_residue_diag = _residue_diagnostics(rgb_old, rgb_out, alpha_old, alpha_out, ctx, eval_mask)
    screen_residue_diag = _screen_residue_diagnostics(
        rgb_old, rgb_out, alpha_old, alpha_out, screen_info, eval_mask,
    )
    residue_diag = _select_residue_diagnostics(
        fill_residue_diag, screen_residue_diag, screen_info, screen_spill_score, eval_mask,
    )
    rollback_rgb, rollback_rgb_reason = _should_rollback_rgb(residue_diag)
    if rollback_rgb:
        # RGB and alpha are visually coupled: keeping refined alpha after RGB
        # rollback makes contaminated pixels more transparent without cleaning
        # their color, which can look worse than the original straight output.
        rgb_out = rgb_old.copy()
        alpha_out = alpha_old.copy()
        rollback_alpha = True
        fill_after_rollback = _residue_diagnostics(rgb_old, rgb_out, alpha_old, alpha_out, ctx, eval_mask)
        screen_after_rollback = _screen_residue_diagnostics(
            rgb_old, rgb_out, alpha_old, alpha_out, screen_info, eval_mask,
        )
        residue_diag_after_rollback = _select_residue_diagnostics(
            fill_after_rollback, screen_after_rollback, screen_info, screen_spill_score, eval_mask,
        )
    else:
        residue_diag_after_rollback = residue_diag

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
        "accept_solid_loss_pct": accept_solid_loss * 100.0,
        "alpha_l1_delta": alpha_l1,
        "accept_alpha_l1_delta": accept_alpha_l1,
        "unpainted_alpha_l1_delta": unpainted_alpha_l1,
        "rollback_alpha": rollback_alpha,
        "alpha_written": not rollback_alpha,
        "alpha_write_allowed": alpha_write_allowed,
        "rollback_rgb": rollback_rgb,
        "rollback_rgb_reason": rollback_rgb_reason,
        "gate_mean": float(gate[accept_mask].mean()) if n_accept else 0.0,
        "alpha_conf_mean": float(alpha_conf[accept_mask].mean()) if n_accept else 0.0,
        "rgb_conf_mean": float(rgb_conf[accept_mask].mean()) if n_accept else 0.0,
        "screen_conf": float(screen_info.get("screen_conf", 0.0)),
        "screen_sat": float(screen_info.get("sat", 0.0)),
        "screen_consistency": float(screen_info.get("consistency", 0.0)),
        "screen_spill_mean": float(screen_spill_score[accept_mask].mean()) if n_accept else 0.0,
        "direction_gate_mean": float(direction_gate[accept_mask].mean()) if n_accept else 0.0,
        "alpha_delta_mean": float(alpha_delta_on_accept.mean()),
        "alpha_delta_p95": float(np.percentile(alpha_delta_on_accept, 95)),
        "alpha_new_mean_on_accept": float(alpha_new[accept_mask].mean()) if n_accept else 0.0,
        "alpha_old_mean_on_accept": float(alpha_old[accept_mask].mean()) if n_accept else 0.0,
        "edge_lap_p95": lap_p95,
        "edge_smooth_score": smooth_score,
        "elapsed_s": elapsed,
    }

    if residue_diag_after_rollback:
        diag.update(residue_diag_after_rollback)
        diag["rgb_improve_pct"] = diag["visible_improve_pct"]
        if rollback_rgb:
            diag["rejected_residue_by_alpha"] = residue_diag.get("residue_after_by_alpha", {})
            diag["rejected_rgb_improve_pct"] = residue_diag.get("visible_improve_pct", 0.0)

    if verbose or debug_dir:
        if "residue_before_by_alpha" in diag:
            rejected = ""
            if rollback_rgb:
                rejected = (
                    f" rejected_rgb_improve={diag.get('rejected_rgb_improve_pct', 0.0):.1f}%"
                    f" reason={rollback_rgb_reason}"
                )
            print(f"[manual refine] residue gte240: "
                  f"{diag['residue_before_by_alpha'].get('gte240', 0):.3f} -> "
                  f"{diag['residue_after_by_alpha'].get('gte240', 0):.3f} "
                  f"rgb_improve={diag.get('rgb_improve_pct', 0.0):.1f}%"
                  f"{rejected}")
        print(
            "[manual refine] "
            f"elapsed={elapsed:.2f}s accept={n_accept}px "
            f"screen_conf={diag['screen_conf']:.2f} "
            f"rgb_conf={diag['rgb_conf_mean']:.2f} "
            f"rollback_rgb={rollback_rgb} "
            f"alpha_written={diag['alpha_written']} rollback_alpha={rollback_alpha} "
            f"accept_solid_loss={accept_solid_loss*100:.2f}% "
            f"accept_alpha_l1={accept_alpha_l1:.3f} "
            f"unpainted_alpha_l1={unpainted_alpha_l1:.4f}"
        )

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray(alpha_out, "L").save(os.path.join(debug_dir, "manual_alpha_out.png"))
        Image.fromarray(rgb_out, "RGB").save(os.path.join(debug_dir, "manual_rgb_out.png"))

    return result_rgba, diag
