"""RGB fringe cleanup using local background-direction spill suppression.

The module intentionally stays solver-free: it removes only the color component
that points from a local foreground anchor toward a local background estimate.
When local background or foreground separation is unreliable, it skips instead
of inventing colors.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


RGB_DISTANCE_MAX = float(np.sqrt(3.0) * 255.0)

BG_CONF_DISTANCE_PX = 8.0
BG_CONF_VARIANCE_WIDTH = 0.16
BG_CONF_FILL_ERROR_WIDTH = 0.12
FG_BG_MIN_SEPARATION = 0.045
FG_BG_GOOD_SEPARATION = 0.22

SOFT_ALPHA_LOW = 24.0 / 255.0
SOFT_ALPHA_HIGH = 220.0 / 255.0
HIGH_ALPHA_START = 200.0 / 255.0
LOW_ALPHA_DROP = 32.0 / 255.0

SOFT_STRENGTH = 0.55
HIGH_ALPHA_STRENGTH = 0.92
OPAQUE_STRENGTH = 0.86
HARD_OBJECT_BOOST = 1.08
EDGE_BIAS_BOOST = 1.08
SAFE_PROFILE_SCALE = 0.72
DETAIL_SCALE = 0.35
ACTIVE_STRENGTH_FLOOR = 0.01

PROJECTION_GAIN_OFFSET = 0.01
PROJECTION_GAIN_WIDTH = 0.42
SPILL_CONF_WIDTH = 0.62
SOFT_NEED_GAMMA = 0.75
DESPILL_MAX_STRENGTH = 0.96
MAX_RGB_DELTA_SOFT = 72.0
MAX_RGB_DELTA_HIGH = 112.0

# Extra pass for chroma-key-like backgrounds. Projection despill is conservative
# when alpha is nearly opaque; this pass removes only the dominant screen channel
# excess, anchored by local foreground color so green clothes/blue hair are kept.
SCREEN_DOMINANCE_MIN = 0.16
SCREEN_DOMINANCE_GOOD = 0.46
SCREEN_EXCESS_MARGIN = 6.0
SCREEN_EXCESS_WIDTH = 32.0
SCREEN_STRENGTH = 1.18
SCREEN_MAX_STRENGTH = 0.98


def _safe_norm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(values, axis=axis)


def _neighbor_mean(values: np.ndarray, mask: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Mean of values over local masked neighbors, falling back to zeros."""
    weights = mask.astype(np.float32)
    kernel = (ksize, ksize)
    count = cv2.boxFilter(weights, cv2.CV_32F, kernel, normalize=False, borderType=cv2.BORDER_REPLICATE)
    count_safe = np.maximum(count, 1.0)
    out = np.zeros_like(values, dtype=np.float32)
    for c in range(values.shape[2]):
        summed = cv2.boxFilter(
            values[..., c].astype(np.float32) * weights,
            cv2.CV_32F,
            kernel,
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        out[..., c] = summed / count_safe
    return out


def _local_color_variance(values: np.ndarray, seed: np.ndarray, ksize: int = 11) -> np.ndarray:
    """Local color standard deviation normalized to RGB unit distance."""
    mean = _neighbor_mean(values, seed, ksize=ksize)
    mean_sq = _neighbor_mean(values.astype(np.float32) ** 2, seed, ksize=ksize)
    var = np.maximum(mean_sq - mean ** 2, 0.0)
    return np.sqrt(np.sum(var, axis=2)) / RGB_DISTANCE_MAX


def _background_confidence(image: np.ndarray, alpha: np.ndarray,
                           ctx: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Estimate how trustworthy the local background color is near each pixel."""
    bg_seed = ctx.get("bg_color_seed", ctx.get("background_seed", alpha <= 8)).astype(bool)
    if not np.any(bg_seed):
        return np.zeros(alpha.shape, dtype=np.float32), {
            "bg_seed_pixels": 0,
            "bg_seed_near_edge": 0.0,
            "bg_conf_mean": 0.0,
            "bg_conf_p10": 0.0,
            "bg_var_p95": 1.0,
            "bg_fill_error_p95": 1.0,
        }

    dist_to_bg_seed = cv2.distanceTransform((~bg_seed).astype(np.uint8), cv2.DIST_L2, 3)
    dist_conf = np.clip((BG_CONF_DISTANCE_PX - dist_to_bg_seed) / BG_CONF_DISTANCE_PX, 0.0, 1.0)

    rgb = image.astype(np.float32)
    bg_fill = ctx["background_fill"].astype(np.float32)
    bg_var = _local_color_variance(rgb, bg_seed)
    var_conf = np.clip(1.0 - bg_var / BG_CONF_VARIANCE_WIDTH, 0.0, 1.0)

    fill_error_map = _safe_norm(rgb - bg_fill, axis=2) / RGB_DISTANCE_MAX
    seed_error = np.zeros(alpha.shape, dtype=np.float32)
    seed_error[bg_seed] = fill_error_map[bg_seed]
    seed_fill_error = _neighbor_mean(seed_error[..., None], bg_seed, ksize=11)[..., 0]
    fill_conf = np.clip(1.0 - seed_fill_error / BG_CONF_FILL_ERROR_WIDTH, 0.0, 1.0)

    confidence = np.clip(dist_conf * (0.55 + 0.45 * var_conf) * fill_conf, 0.0, 1.0)
    eval_mask = ctx["color_fringe"] & (alpha > 0)
    conf_vals = confidence[eval_mask]
    var_vals = bg_var[eval_mask]
    fill_vals = seed_fill_error[eval_mask]
    near = eval_mask & (dist_to_bg_seed <= BG_CONF_DISTANCE_PX)
    stats = {
        "bg_seed_pixels": int(np.sum(bg_seed)),
        "bg_seed_near_edge": float(np.sum(near) / max(np.sum(eval_mask), 1)),
        "bg_conf_mean": float(conf_vals.mean()) if conf_vals.size else 0.0,
        "bg_conf_p10": float(np.percentile(conf_vals, 10)) if conf_vals.size else 0.0,
        "bg_var_p95": float(np.percentile(var_vals, 95)) if var_vals.size else 0.0,
        "bg_fill_error_p95": float(np.percentile(fill_vals, 95)) if fill_vals.size else 0.0,
    }
    return confidence.astype(np.float32), stats


def _residue_by_alpha(rgb: np.ndarray, alpha: np.ndarray, ctx: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Visible background-color residue split by alpha confidence bands."""
    mask = ctx["color_fringe"] & (alpha > 0)
    result: Dict[str, float] = {}
    if not np.any(mask):
        for name in ("lt64", "64_180", "180_240", "gte240"):
            result[name] = 0.0
        return result
    compute_spill_score = ctx.get("compute_spill_score")
    if compute_spill_score is None:
        raise KeyError("ctx['compute_spill_score'] is required for RGB defringe diagnostics")

    spill, _, _ = compute_spill_score(rgb, ctx["foreground_fill"], ctx["background_fill"], mask=mask)
    visible = spill * (alpha.astype(np.float32) / 255.0)
    bands = {
        "lt64": mask & (alpha < 64),
        "64_180": mask & (alpha >= 64) & (alpha < 180),
        "180_240": mask & (alpha >= 180) & (alpha < 240),
        "gte240": mask & (alpha >= 240),
    }
    for name, band in bands.items():
        vals = visible[band]
        result[name] = float(vals.mean()) if vals.size else 0.0
    return result


def background_direction_defringe(image: np.ndarray, alpha: np.ndarray,
                                  ctx: Dict[str, np.ndarray],
                                  profile) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Suppress background-colored RGB residue on matte fringes."""
    guide_edge = ctx.get("defringe_fringe", ctx["color_fringe"])
    edge = guide_edge & (alpha > 0)
    strength = np.zeros(alpha.shape, dtype=np.float32)
    out = image.astype(np.float32).copy()

    bg_confidence, bg_stats = _background_confidence(image, alpha, ctx)
    if not np.any(edge):
        out[alpha == 0] = 0
        metrics: Dict[str, object] = {
            **bg_stats,
            "method": "background_direction_despill",
            "applied_pixels": 0,
            "soft_pixels": 0,
            "high_alpha_pixels": 0,
            "opaque_pixels": 0,
            "skipped_low_bg_conf": 0,
            "skipped_ambiguous": 0,
            "skipped_projection": 0,
            "skipped_protected": 0,
            "strength_mean": 0.0,
            "strength_p95": 0.0,
            "residue_before_by_alpha": _residue_by_alpha(image, alpha, ctx),
            "residue_after_by_alpha": _residue_by_alpha(out.astype(np.uint8), alpha, ctx),
        }
        return out.astype(np.uint8), strength, metrics

    a = alpha.astype(np.float32) / 255.0
    rgb = image.astype(np.float32)
    fg = ctx["foreground_fill"].astype(np.float32)
    bg = ctx["background_fill"].astype(np.float32)
    spill = ctx["spill_score"].astype(np.float32)

    vector = bg - fg
    vector_norm2 = np.sum(vector * vector, axis=2) + 1e-6
    fg_bg_sep = np.sqrt(vector_norm2) / RGB_DISTANCE_MAX
    projection = np.sum((rgb - fg) * vector, axis=2) / vector_norm2
    projection_conf = np.clip((projection - PROJECTION_GAIN_OFFSET) / PROJECTION_GAIN_WIDTH, 0.0, 1.0)
    sep_conf = np.clip(
        (fg_bg_sep - FG_BG_MIN_SEPARATION) / max(FG_BG_GOOD_SEPARATION - FG_BG_MIN_SEPARATION, 1e-6),
        0.0,
        1.0,
    )

    soft_mask = edge & (a >= SOFT_ALPHA_LOW) & (a < SOFT_ALPHA_HIGH)
    high_alpha_mask = edge & (a >= HIGH_ALPHA_START)
    opaque_mask = edge & (ctx["opaque_rim"] | ctx["solid_rim"])
    candidate = (soft_mask | high_alpha_mask | opaque_mask) & (~ctx["protected_transparency"])

    bg_gate = np.clip((bg_confidence - 0.18) / 0.52, 0.0, 1.0)
    spill_gate = np.clip(spill / SPILL_CONF_WIDTH, 0.0, 1.0)
    soft_need = np.clip((1.0 - a) ** SOFT_NEED_GAMMA, 0.0, 1.0)
    alpha_weight = np.where(
        high_alpha_mask | opaque_mask,
        HIGH_ALPHA_STRENGTH,
        SOFT_STRENGTH * soft_need,
    ).astype(np.float32)
    # Very soft pixels are mostly coverage errors; keep RGB recovery conservative
    # there and let alpha cleanup carry the visual weight.
    alpha_weight = np.where(opaque_mask, np.maximum(alpha_weight, OPAQUE_STRENGTH), alpha_weight)

    strength_map = (
        alpha_weight
        * bg_gate
        * spill_gate
        * projection_conf
        * (0.35 + 0.65 * sep_conf)
    )

    if getattr(profile, "defringe", "balanced") in ("bright_edge", "dark_edge"):
        strength_map *= EDGE_BIAS_BOOST
    if getattr(profile, "profile", "balanced") == "hard_object":
        strength_map *= HARD_OBJECT_BOOST
    if getattr(profile, "profile", "balanced") in ("detail_safe", "transparent_safe"):
        strength_map *= SAFE_PROFILE_SCALE
    strength_map[ctx["detail"]] *= DETAIL_SCALE
    strength_map[ctx["protected_transparency"]] = 0.0
    strength_map[a < LOW_ALPHA_DROP] *= 0.35
    strength_map = np.clip(strength_map, 0.0, DESPILL_MAX_STRENGTH)
    strength_map[~candidate] = 0.0
    strength_map[projection <= 0.0] = 0.0
    strength_map[fg_bg_sep < FG_BG_MIN_SEPARATION] = 0.0

    delta = np.clip(projection, 0.0, 1.0)[..., None] * vector
    max_delta = np.where(high_alpha_mask | opaque_mask, MAX_RGB_DELTA_HIGH, MAX_RGB_DELTA_SOFT)
    delta_norm = np.maximum(_safe_norm(delta, axis=2), 1e-6)
    delta_scale = np.minimum(1.0, max_delta / delta_norm)
    cleaned = rgb - delta * delta_scale[..., None] * strength_map[..., None]
    cleaned = np.clip(cleaned, 0.0, 255.0)

    bg_dom = np.argmax(bg, axis=2)
    dom_one_hot = np.eye(3, dtype=bool)[bg_dom]
    bg_dom_val = np.take_along_axis(bg, bg_dom[..., None], axis=2)[..., 0]
    rgb_dom_val = np.take_along_axis(cleaned, bg_dom[..., None], axis=2)[..., 0]
    fg_dom_val = np.take_along_axis(fg, bg_dom[..., None], axis=2)[..., 0]
    rgb_other_max = np.max(np.where(dom_one_hot, -1e6, cleaned), axis=2)
    bg_other_max = np.max(np.where(dom_one_hot, -1e6, bg), axis=2)
    screen_dominance = np.clip(
        ((bg_dom_val - bg_other_max) / 255.0 - SCREEN_DOMINANCE_MIN)
        / max(SCREEN_DOMINANCE_GOOD - SCREEN_DOMINANCE_MIN, 1e-6),
        0.0,
        1.0,
    )
    # Keep the dominant channel no lower than either local foreground evidence or
    # the non-screen channels; this removes spill without neutralizing real color.
    screen_target = np.maximum(rgb_other_max, fg_dom_val) + SCREEN_EXCESS_MARGIN
    screen_excess = np.maximum(rgb_dom_val - screen_target, 0.0)
    screen_excess_conf = np.clip(screen_excess / SCREEN_EXCESS_WIDTH, 0.0, 1.0)
    screen_strength = np.clip(
        SCREEN_STRENGTH
        * alpha_weight
        * bg_gate
        * spill_gate
        * screen_dominance
        * screen_excess_conf
        * (0.35 + 0.65 * sep_conf),
        0.0,
        SCREEN_MAX_STRENGTH,
    )
    screen_strength[~candidate] = 0.0
    screen_strength[ctx["detail"]] *= DETAIL_SCALE
    screen_strength[ctx["protected_transparency"]] = 0.0

    screen_cleaned = cleaned.copy()
    for c in range(3):
        channel = bg_dom == c
        if np.any(channel):
            target = np.minimum(screen_cleaned[..., c], screen_target)
            screen_cleaned[..., c] = np.where(
                channel,
                screen_cleaned[..., c] * (1.0 - screen_strength) + target * screen_strength,
                screen_cleaned[..., c],
            )
    cleaned = np.clip(screen_cleaned, 0.0, 255.0)

    combined_strength = np.maximum(strength_map, screen_strength)
    active = combined_strength > ACTIVE_STRENGTH_FLOOR
    out[active] = cleaned[active]
    out[alpha == 0] = 0
    strength[active] = combined_strength[active].astype(np.float32)

    protected = edge & ctx["protected_transparency"]
    low_bg_conf = candidate & (bg_gate <= 0.0)
    ambiguous = candidate & (fg_bg_sep < FG_BG_MIN_SEPARATION)
    projection_skip = candidate & (projection <= 0.0)
    screen_active = screen_strength > ACTIVE_STRENGTH_FLOOR
    active_vals = strength[active]
    screen_vals = screen_strength[screen_active]
    metrics = {
        **bg_stats,
        "method": "background_direction_despill",
        "applied_pixels": int(np.sum(active)),
        "soft_pixels": int(np.sum(active & soft_mask)),
        "high_alpha_pixels": int(np.sum(active & high_alpha_mask)),
        "opaque_pixels": int(np.sum(active & opaque_mask)),
        "screen_pixels": int(np.sum(screen_active)),
        "screen_green_pixels": int(np.sum(screen_active & (bg_dom == 1))),
        "screen_blue_pixels": int(np.sum(screen_active & (bg_dom == 2))),
        "screen_strength_mean": float(screen_vals.mean()) if screen_vals.size else 0.0,
        "screen_strength_p95": float(np.percentile(screen_vals, 95)) if screen_vals.size else 0.0,
        "skipped_low_bg_conf": int(np.sum(low_bg_conf)),
        "skipped_ambiguous": int(np.sum(ambiguous)),
        "skipped_projection": int(np.sum(projection_skip)),
        "skipped_protected": int(np.sum(protected)),
        "strength_mean": float(active_vals.mean()) if active_vals.size else 0.0,
        "strength_p95": float(np.percentile(active_vals, 95)) if active_vals.size else 0.0,
        "residue_before_by_alpha": _residue_by_alpha(image, alpha, ctx),
        "residue_after_by_alpha": _residue_by_alpha(np.clip(out, 0, 255).astype(np.uint8), alpha, ctx),
        "bg_confidence_map": bg_confidence,
        "projection_map": np.clip(projection_conf, 0.0, 1.0).astype(np.float32),
        "screen_strength_map": screen_strength.astype(np.float32),
    }
    return np.clip(out, 0, 255).astype(np.uint8), strength, metrics
