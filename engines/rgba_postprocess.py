"""Topology-aware RGBA post-processing for clean background replacement exports."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from engines.rgb_defringe import background_direction_defringe


RGB_DISTANCE_MAX = float(np.sqrt(3.0) * 255.0)

# Spill score normalization. These are tuning gates, not literature constants:
# keep them named so dataset A/B can adjust behavior without hunting literals.
SPILL_BG_LIKENESS_OFFSET = 0.018
SPILL_BG_LIKENESS_WIDTH = 0.18
ROI_FULL_FRAME_RATIO = 0.92


class MatteProfile:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _connected_background(seed: np.ndarray) -> np.ndarray:
    """Return background seed components connected to the image border."""
    padded = np.pad(seed.astype(np.uint8), 1, mode="constant", constant_values=1)
    # Flood-fill the artificial outer border instead of labeling the whole image.
    # This is equivalent to selecting seed components connected to any border.
    cv2.floodFill(padded, None, (0, 0), 2, flags=8)
    return padded[1:-1, 1:-1] == 2


def _hole_background(seed: np.ndarray, outer_background: np.ndarray) -> np.ndarray:
    """Return enclosed low-alpha background regions, such as gaps between limbs."""
    background = seed.astype(bool)
    return background & (~outer_background)


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """OpenCV distance transform computes distance to zeros."""
    if not np.any(mask):
        return np.full(mask.shape, 1e6, dtype=np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def _gaussian_color_fill(image: np.ndarray, seed: np.ndarray, sigma: float = 20.0,
                         scale: int = 4) -> np.ndarray:
    """用 GaussianBlur 加权平均替代迭代漏色：快 ~100x，且各方向无偏。

    颜色填充是 σ≈20 的低频估计，在 1/scale 分辨率上算（σ 同比缩小）再上采样，
    视觉等价但大核高斯的开销下降 ~scale^3。结果只在边缘附近被读取，远端保真无影响。
    """
    h, w = seed.shape
    img_f = image.astype(np.float32)
    wt = seed.astype(np.float32)

    s = int(scale) if (scale > 1 and min(h, w) >= 256) else 1
    if s > 1:
        size = (max(1, w // s), max(1, h // s))
        wt_s = cv2.resize(wt, size, interpolation=cv2.INTER_AREA)
        imgw_s = cv2.resize(img_f * wt[..., None], size, interpolation=cv2.INTER_AREA)
        img_s = cv2.resize(img_f, size, interpolation=cv2.INTER_AREA)
        sig = sigma / s
    else:
        wt_s, imgw_s, img_s, sig = wt, img_f * wt[..., None], img_f, sigma

    w_blur = cv2.GaussianBlur(wt_s, (0, 0), sig)
    w_blur_safe = np.where(w_blur > 1e-6, w_blur, 1.0)
    out_s = np.empty_like(img_s)
    for c in range(3):
        num = cv2.GaussianBlur(imgw_s[..., c], (0, 0), sig)
        out_s[..., c] = np.where(w_blur > 1e-6, num / w_blur_safe, img_s[..., c])

    if s > 1:
        return cv2.resize(out_s, (w, h), interpolation=cv2.INTER_LINEAR)
    return out_s


def _safe_foreground_seed(alpha: np.ndarray, largest_solid: np.ndarray,
                          dist_to_background: np.ndarray) -> np.ndarray:
    """Seed foreground colors from the interior, not from contaminated rims."""
    min_seed = max(48, int(alpha.size * 0.00003))
    core_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_solid = cv2.erode(largest_solid.astype(np.uint8), core_shape, iterations=1).astype(bool)

    seed = (alpha >= 245) & eroded_solid & (dist_to_background >= 2.6)
    if int(np.sum(seed)) >= min_seed:
        return seed

    seed = (alpha >= 220) & largest_solid & (dist_to_background >= 3.4)
    if int(np.sum(seed)) >= min_seed:
        return seed

    seed = (alpha >= 180) & largest_solid & (dist_to_background >= 2.2)
    if np.any(seed):
        return seed

    fallback = (alpha >= 245) & largest_solid
    if np.any(fallback):
        return fallback
    return alpha >= 128


def _estimate_foreground_fill(image: np.ndarray, alpha: np.ndarray,
                              seed: np.ndarray = None) -> np.ndarray:
    """Extend clean foreground core colors into edge pixels."""
    if seed is None:
        seed = alpha >= 245
        if not np.any(seed):
            seed = alpha >= 128
    return _gaussian_color_fill(image, seed, sigma=20.0)


def _estimate_background_fill(image: np.ndarray, alpha: np.ndarray,
                              background_seed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extend local true-background colors toward fringe pixels."""
    color_seed = background_seed & (alpha <= 8)
    if not np.any(color_seed):
        color_seed = background_seed & (alpha <= 24)
    if not np.any(color_seed):
        color_seed = alpha <= 8
    return _gaussian_color_fill(image, color_seed, sigma=20.0), color_seed


def _compute_spill_score(image: np.ndarray, foreground_fill: np.ndarray,
                         background_fill: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return how likely each pixel color is background spill rather than foreground."""
    if mask is None:
        rgb = image.astype(np.float32)
        fg = foreground_fill.astype(np.float32)
        bg = background_fill.astype(np.float32)
        fg_dist = np.linalg.norm(rgb - fg, axis=2) / RGB_DISTANCE_MAX
        bg_dist = np.linalg.norm(rgb - bg, axis=2) / RGB_DISTANCE_MAX
        bg_like = np.clip(
            (fg_dist - bg_dist - SPILL_BG_LIKENESS_OFFSET) / SPILL_BG_LIKENESS_WIDTH,
            0.0,
            1.0,
        )

        edge_lum = rgb.mean(axis=2)
        fg_lum = fg.mean(axis=2)
        color_delta = np.clip(fg_dist / 0.30, 0.0, 1.0)
        bright = np.clip((edge_lum - fg_lum - 6.0) / 64.0, 0.0, 1.0) * color_delta
        dark = np.clip((fg_lum - edge_lum - 6.0) / 64.0, 0.0, 1.0) * color_delta
        spill = np.maximum(bg_like, np.maximum(bright, dark) * 0.82)
        return np.clip(spill, 0.0, 1.0).astype(np.float32), fg_dist.astype(np.float32), bg_dist.astype(np.float32)

    spill = np.zeros(mask.shape, dtype=np.float32)
    fg_dist = np.zeros(mask.shape, dtype=np.float32)
    bg_dist = np.zeros(mask.shape, dtype=np.float32)
    if not np.any(mask):
        return spill, fg_dist, bg_dist

    idx = np.flatnonzero(mask.ravel())
    rgb = image.reshape(-1, 3)[idx].astype(np.float32)
    fg = foreground_fill.reshape(-1, 3)[idx].astype(np.float32)
    bg = background_fill.reshape(-1, 3)[idx].astype(np.float32)
    fg_vals = np.linalg.norm(rgb - fg, axis=1) / RGB_DISTANCE_MAX
    bg_vals = np.linalg.norm(rgb - bg, axis=1) / RGB_DISTANCE_MAX
    bg_like = np.clip(
        (fg_vals - bg_vals - SPILL_BG_LIKENESS_OFFSET) / SPILL_BG_LIKENESS_WIDTH,
        0.0,
        1.0,
    )

    edge_lum = rgb.mean(axis=1)
    fg_lum = fg.mean(axis=1)
    color_delta = np.clip(fg_vals / 0.30, 0.0, 1.0)
    bright = np.clip((edge_lum - fg_lum - 6.0) / 64.0, 0.0, 1.0) * color_delta
    dark = np.clip((fg_lum - edge_lum - 6.0) / 64.0, 0.0, 1.0) * color_delta
    spill_vals = np.maximum(bg_like, np.maximum(bright, dark) * 0.82)
    spill.ravel()[idx] = np.clip(spill_vals, 0.0, 1.0).astype(np.float32)
    fg_dist.ravel()[idx] = fg_vals.astype(np.float32)
    bg_dist.ravel()[idx] = bg_vals.astype(np.float32)
    return spill, fg_dist, bg_dist


def _thin_detail_mask(alpha: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Protect soft wisps that extend away from solid foreground, not contour rings."""
    detail_src = ((alpha > 12) & (alpha < 220)).astype(np.uint8)
    if not np.any(detail_src) or not np.any(solid):
        return np.zeros(alpha.shape, dtype=bool)

    opened = cv2.morphologyEx(
        detail_src,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    thin = (detail_src > 0) & (opened == 0)
    if not np.any(thin):
        return np.zeros(alpha.shape, dtype=bool)

    solid_touch = cv2.dilate(
        solid.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
    protected = np.zeros(alpha.shape, dtype=bool)
    max_area = max(180, int(alpha.size * 0.0025))
    candidate_labels = [
        i for i in range(1, n)
        if 6 <= int(stats[i, cv2.CC_STAT_AREA]) <= max_area
    ]
    if not candidate_labels:
        return protected

    dist_from_solid = _distance_to(solid)

    for i in candidate_labels:
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comp = labels[y:y + h, x:x + w] == i
        major = max(w, h)
        minor = max(1, min(w, h))
        elongation = major / minor
        thickness = area / max(major, 1)
        contact = _ratio(np.sum(comp & solid_touch[y:y + h, x:x + w]), area)
        distances = dist_from_solid[y:y + h, x:x + w][comp]
        p90 = float(np.percentile(distances, 90)) if distances.size else 0.0
        wisp = p90 >= 2.0 and contact < 0.55 and (area < 120 or (elongation >= 4.0 and thickness <= 6.0))
        if wisp:
            region = protected[y:y + h, x:x + w]
            region[comp] = True
    return protected


def _build_context(image: np.ndarray, alpha: np.ndarray,
                   preserve_transparency: bool = False,
                   full_spill: bool = True) -> Dict[str, np.ndarray]:
    """
    Split soft alpha by topology:
    - fringe: soft pixels adjacent to real background, including limb holes.
    - interior_soft: soft pixels away from background; likely transparent material.
    - detail: hair/fur-like wisps that should not be tightened.
    """
    a = np.clip(alpha, 0, 255).astype(np.uint8)
    solid = a > 127
    largest_solid = _largest_component(solid)
    edge = (a > 0) & (a < 245)
    haze = (a > 0) & (a < 80)

    # Low-confidence alpha is a reliable seed for true background. Topology then
    # distinguishes outside background from enclosed limb holes.
    bg_seed = a <= 24
    outer_bg = _connected_background(bg_seed)
    hole_bg = _hole_background(bg_seed, outer_bg)
    background_seed = outer_bg | hole_bg
    dist_to_background = _distance_to(background_seed)
    dist_to_solid = _distance_to(largest_solid) if full_spill else None

    fringe = edge & (dist_to_background <= 3.2)
    outer_fringe = fringe & (dist_to_background <= 3.2) & cv2.dilate(
        outer_bg.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    hole_fringe = fringe & cv2.dilate(
        hole_bg.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)

    detail = _thin_detail_mask(a, largest_solid)
    # Do not infer transparency merely from holes. Protect only soft regions
    # away from background, or all such regions when the user requested it.
    interior_soft = edge & (dist_to_background > (2.0 if preserve_transparency else 3.2))
    protected_transparency = interior_soft & (a > 16)
    safe_fg_seed = _safe_foreground_seed(a, largest_solid, dist_to_background)
    foreground_fill = _estimate_foreground_fill(image, a, seed=safe_fg_seed)
    background_fill, bg_color_seed = _estimate_background_fill(image, a, background_seed)
    spill_mask = None
    if not full_spill:
        spill_mask = (
            fringe
            | ((a >= 200) & (dist_to_background <= 2.4))
            | ((a >= 245) & (dist_to_background <= 3.4))
        )
    spill_score, fg_dist, bg_dist = _compute_spill_score(
        image,
        foreground_fill,
        background_fill,
        mask=spill_mask,
    )

    near_background = (a > 0) & (dist_to_background <= 2.4)
    opaque_rim = (
        near_background
        & (a >= 200)
        & (spill_score >= 0.38)
        & (~detail)
        & (~protected_transparency)
    )
    solid_rim = (
        (a >= 245)
        & (dist_to_background <= 3.4)
        & (~detail)
        & (~protected_transparency)
    )
    color_fringe = (fringe | opaque_rim | solid_rim) & (a > 0)

    return {
        "alpha": a,
        "solid": solid,
        "largest_solid": largest_solid,
        "edge": edge,
        "haze": haze,
        "outer_bg": outer_bg,
        "hole_bg": hole_bg,
        "background_seed": background_seed,
        "bg_color_seed": bg_color_seed,
        "dist_to_background": dist_to_background,
        "dist_to_solid": dist_to_solid,
        "fringe": fringe,
        "color_fringe": color_fringe,
        "solid_rim": solid_rim,
        "outer_fringe": outer_fringe,
        "hole_fringe": hole_fringe,
        "detail": detail,
        "protected_transparency": protected_transparency,
        "safe_fg_seed": safe_fg_seed,
        "foreground_fill": foreground_fill,
        "background_fill": background_fill,
        "spill_score": spill_score,
        "fg_dist": fg_dist,
        "bg_dist": bg_dist,
        "opaque_rim": opaque_rim,
    }


def analyze_matte(image: np.ndarray, alpha: np.ndarray,
                  preserve_transparency: bool = False,
                  full_spill: bool = True) -> Tuple[MatteProfile, Dict[str, np.ndarray]]:
    ctx = _build_context(image, alpha, preserve_transparency=preserve_transparency,
                         full_spill=full_spill)
    a = ctx["alpha"]
    alpha_area = int(np.sum(a > 0))
    solid_area = int(np.sum(a > 127))
    edge_pixels = int(np.sum(ctx["edge"]))
    haze_pixels = int(np.sum(ctx["haze"]))
    fringe_pixels = int(np.sum(ctx["fringe"]))
    outer_fringe_pixels = int(np.sum(ctx["outer_fringe"]))
    hole_fringe_pixels = int(np.sum(ctx["hole_fringe"]))
    color_fringe_pixels = int(np.sum(ctx["color_fringe"]))
    opaque_rim_pixels = int(np.sum(ctx["opaque_rim"]))
    solid_rim_pixels = int(np.sum(ctx["solid_rim"]))
    detail_pixels = int(np.sum(ctx["detail"]))
    protected_transparency_pixels = int(np.sum(ctx["protected_transparency"]))
    safe_fg_seed_pixels = int(np.sum(ctx["safe_fg_seed"]))
    bg_color_seed_pixels = int(np.sum(ctx["bg_color_seed"]))

    edge_ratio = _ratio(edge_pixels, max(alpha_area, 1))
    haze_ratio = _ratio(haze_pixels, max(alpha_area, 1))
    fringe_ratio = _ratio(fringe_pixels, max(edge_pixels, 1))
    opaque_rim_ratio = _ratio(opaque_rim_pixels, max(color_fringe_pixels, 1))
    detail_ratio = _ratio(detail_pixels, max(edge_pixels, 1))
    transparency_ratio = _ratio(protected_transparency_pixels, max(solid_area, 1))
    spill_vals = ctx["spill_score"][ctx["color_fringe"]]
    spill_score_mean = float(spill_vals.mean()) if spill_vals.size else 0.0
    spill_score_p95 = float(np.percentile(spill_vals, 95)) if spill_vals.size else 0.0

    rgb = image.astype(np.float32)
    fg = ctx["foreground_fill"]
    edge_for_color = ctx["color_fringe"] & (a < 252)
    if np.any(edge_for_color):
        edge_rgb = rgb[edge_for_color]
        fg_rgb = fg[edge_for_color]
        edge_lum = edge_rgb.mean(axis=1)
        fg_lum = fg_rgb.mean(axis=1)
        color_dist = np.linalg.norm(edge_rgb - fg_rgb, axis=1) / RGB_DISTANCE_MAX
        bright = np.clip((edge_lum - fg_lum - 8.0) / 72.0, 0.0, 1.0) * np.clip(color_dist * 2.0, 0.0, 1.0)
        dark = np.clip((fg_lum - edge_lum - 8.0) / 72.0, 0.0, 1.0) * np.clip(color_dist * 2.0, 0.0, 1.0)
        bright_score = float(np.percentile(bright, 90))
        dark_score = float(np.percentile(dark, 90))
    else:
        bright_score = 0.0
        dark_score = 0.0

    detail_risk = float(np.clip((detail_ratio - 0.06) / 0.34, 0.0, 1.0))
    transparency_risk = float(np.clip(transparency_ratio * 6.0, 0.0, 1.0))
    halo_score = float(np.clip(
        fringe_ratio * 0.50 + haze_ratio * 2.2 + max(bright_score, dark_score) * 0.35
        + spill_score_p95 * 0.22 + opaque_rim_ratio * 0.12
        - detail_risk * 0.35 - transparency_risk * 0.25,
        0.0,
        1.0,
    ))

    strong_color_halo = (
        halo_score > 0.70
        and max(bright_score, dark_score) > 0.55
        and haze_ratio > 0.06
    )

    if preserve_transparency or (transparency_risk > 0.42 and not strong_color_halo):
        profile = "transparent_safe"
    elif detail_risk > 0.55:
        profile = "detail_safe"
    elif halo_score > 0.55 and edge_ratio > 0.045:
        profile = "hard_object"
    else:
        profile = "balanced"

    if profile == "hard_object":
        alpha_tighten = "medium_strong"
    elif profile in ("detail_safe", "transparent_safe"):
        alpha_tighten = "light"
    else:
        alpha_tighten = "medium"

    if bright_score > 0.22 and bright_score >= dark_score:
        defringe = "bright_edge"
    elif dark_score > 0.22:
        defringe = "dark_edge"
    else:
        defringe = "balanced"

    return MatteProfile(
        alpha_area=alpha_area,
        solid_area=solid_area,
        edge_pixels=edge_pixels,
        haze_pixels=haze_pixels,
        fringe_pixels=fringe_pixels,
        outer_fringe_pixels=outer_fringe_pixels,
        hole_fringe_pixels=hole_fringe_pixels,
        color_fringe_pixels=color_fringe_pixels,
        opaque_rim_pixels=opaque_rim_pixels,
        solid_rim_pixels=solid_rim_pixels,
        detail_pixels=detail_pixels,
        protected_transparency_pixels=protected_transparency_pixels,
        safe_fg_seed_pixels=safe_fg_seed_pixels,
        bg_color_seed_pixels=bg_color_seed_pixels,
        edge_ratio=edge_ratio,
        haze_ratio=haze_ratio,
        fringe_ratio=fringe_ratio,
        opaque_rim_ratio=opaque_rim_ratio,
        detail_ratio=detail_ratio,
        transparency_ratio=transparency_ratio,
        spill_score_mean=spill_score_mean,
        spill_score_p95=spill_score_p95,
        detail_risk=detail_risk,
        transparency_risk=transparency_risk,
        bright_fringe_score=bright_score,
        dark_fringe_score=dark_score,
        halo_score=halo_score,
        profile=profile,
        alpha_tighten=alpha_tighten,
        defringe=defringe,
    ), ctx


def _refine_alpha(alpha: np.ndarray, ctx: Dict[str, np.ndarray],
                  profile: MatteProfile) -> np.ndarray:
    """
    Tighten only topology-confirmed fringe. Never erode the solid mask: fingers,
    spikes and thin rigid structures keep their original alpha support.
    """
    a = alpha.astype(np.float32) / 255.0
    out = a.copy()
    candidate = ctx["fringe"] & (~ctx["detail"]) & (~ctx["protected_transparency"])

    if profile.alpha_tighten == "medium_strong":
        low, high, gamma, strength = 0.10, 0.94, 1.18, 0.88
    elif profile.alpha_tighten == "medium":
        low, high, gamma, strength = 0.06, 0.97, 1.10, 0.62
    else:
        low, high, gamma, strength = 0.035, 0.985, 1.04, 0.32

    if np.any(candidate):
        a_candidate = a[candidate]
        calibrated = np.power(np.clip((a_candidate - low) / max(high - low, 1e-6), 0.0, 1.0), gamma)
        blended = a_candidate * (1.0 - strength) + calibrated * strength
        if profile.profile == "hard_object":
            # Hard objects should have a compact 1-2px transition. RMBG often leaves
            # a broad semi-transparent rim, which reads as halo on replacement.
            t = np.clip((a_candidate - 0.30) / 0.48, 0.0, 1.0)
            compacted = t * t * (3.0 - 2.0 * t)
            hard_blend = a_candidate * 0.08 + compacted * 0.92
            # Background-colored pixels should not be solidified just because the
            # object profile is hard; suppress positive gain by spill confidence.
            positive_gain = np.maximum(hard_blend - a_candidate, 0.0)
            hard_blend -= positive_gain * np.clip(ctx["spill_score"][candidate] * 0.38, 0.0, 0.38)
            out[candidate] = hard_blend
        else:
            # Background-facing fringe should not become more opaque during cleanup.
            out[candidate] = np.minimum(a_candidate, blended)

    # Background-facing haze is almost always residue. Hole fringe receives the
    # same treatment as outer fringe, which fixes arm/leg gap halos.
    haze = candidate & (alpha < (112 if profile.profile == "hard_object" else 72))
    out[haze] *= 0.22 if profile.profile == "hard_object" else 0.62
    if profile.profile == "hard_object":
        out[candidate & (alpha < 56)] = 0.0
        out[candidate & (alpha < 92) & (ctx["spill_score"] > 0.28)] = 0.0
    spill_pull = candidate & (ctx["spill_score"] > 0.55) & (alpha < 180)
    if np.any(spill_pull):
        out[spill_pull] *= np.clip(1.0 - 0.18 * ctx["spill_score"][spill_pull], 0.72, 1.0)
    out[candidate & (alpha < (18 if profile.profile == "hard_object" else 10))] = 0.0

    # Keep original support for confirmed details and interior transparent matte.
    out[ctx["detail"]] = np.maximum(out[ctx["detail"]], a[ctx["detail"]] * 0.96)
    out[ctx["protected_transparency"]] = np.maximum(out[ctx["protected_transparency"]], a[ctx["protected_transparency"]] * 0.98)
    return np.clip(out * 255.0, 0, 255).round().astype(np.uint8)


def _guard_against_overcut(original: np.ndarray, refined: np.ndarray,
                           profile: MatteProfile) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    def measure(candidate: np.ndarray) -> Dict[str, float]:
        before_solid = int(np.sum(original > 127))
        after_solid = int(np.sum(candidate > 127))
        before_area = int(np.sum(original > 0))
        after_area = int(np.sum(candidate > 0))
        delta = candidate.astype(np.int16) - original.astype(np.int16)
        positive = delta > 0
        negative = delta < 0
        return {
            "solid_loss": _ratio(max(0, before_solid - after_solid), max(before_solid, 1)),
            "alpha_area_loss": _ratio(max(0, before_area - after_area), max(before_area, 1)),
            "solid_gain": _ratio(max(0, after_solid - before_solid), max(before_solid, 1)),
            "alpha_area_gain": _ratio(max(0, after_area - before_area), max(before_area, 1)),
            "alpha_l1": float(np.mean(np.abs(delta)) / 255.0),
            "alpha_positive_pixels": int(np.sum(positive)),
            "alpha_negative_pixels": int(np.sum(negative)),
            "alpha_positive_l1": float(np.sum(delta[positive]) / max(original.size * 255.0, 1.0)),
            "alpha_negative_l1": float(np.sum(-delta[negative]) / max(original.size * 255.0, 1.0)),
        }

    guard = measure(refined)
    limit = 0.010 if profile.profile in ("detail_safe", "transparent_safe") else 0.018
    rollback = guard["solid_loss"] > limit
    if rollback:
        blend = 0.45
        refined = np.clip(
            original.astype(np.float32) * (1.0 - blend) + refined.astype(np.float32) * blend,
            0,
            255,
        ).round().astype(np.uint8)
        guard = measure(refined)

    return refined, rollback, guard


def _mask_crop(mask: np.ndarray, margin: int = 64,
               full_frame_ratio: float = ROI_FULL_FRAME_RATIO) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded crop around mask, or None when cropping brings little benefit."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(h, int(ys.max()) + 1 + margin)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(w, int(xs.max()) + 1 + margin)
    if (y2 - y1) * (x2 - x1) >= full_frame_ratio * h * w:
        return None
    return y1, y2, x1, x2


def _defringe_rgb(image: np.ndarray, alpha: np.ndarray, ctx: Dict[str, np.ndarray],
                  profile: MatteProfile) -> Tuple[np.ndarray, np.ndarray]:
    """Remove background-colored RGB residue using local foreground/background colors."""
    ctx["compute_spill_score"] = _compute_spill_score
    try:
        rgb, strength, metrics = background_direction_defringe(image, alpha, ctx, profile)
    finally:
        ctx.pop("compute_spill_score", None)
    ctx["defringe_metrics"] = metrics
    ctx["bg_confidence"] = metrics.get("bg_confidence_map", np.zeros(alpha.shape, dtype=np.float32))
    ctx["despill_projection"] = metrics.get("projection_map", np.zeros(alpha.shape, dtype=np.float32))
    ctx["screen_despill_strength"] = metrics.get("screen_strength_map", np.zeros(alpha.shape, dtype=np.float32))
    return rgb, strength


def _rgb_residue_summary(rgb: np.ndarray, alpha: np.ndarray,
                         ctx: Dict[str, np.ndarray],
                         with_map: bool = False) -> Dict[str, object]:
    """Estimate visible background-colored RGB that remains on matte edges."""
    eval_mask = ctx["color_fringe"] & (alpha > 0)
    count = int(np.sum(eval_mask))
    residue_map = np.zeros(alpha.shape, dtype=np.float32)
    if count == 0:
        summary = {
            "pixels": 0,
            "spill_mean": 0.0,
            "spill_p95": 0.0,
            "visible_mean": 0.0,
            "visible_p95": 0.0,
        }
        if with_map:
            summary["visible_map"] = residue_map
        return summary

    spill_score, fg_dist, bg_dist = _compute_spill_score(
        rgb,
        ctx["foreground_fill"],
        ctx["background_fill"],
        mask=eval_mask,
    )
    spill_vals = spill_score[eval_mask]
    alpha_weight = alpha[eval_mask].astype(np.float32) / 255.0
    visible_vals = spill_vals * alpha_weight
    residue_map[eval_mask] = visible_vals

    summary = {
        "pixels": count,
        "spill_mean": float(spill_vals.mean()),
        "spill_p95": float(np.percentile(spill_vals, 95)),
        "visible_mean": float(visible_vals.mean()),
        "visible_p95": float(np.percentile(visible_vals, 95)),
    }
    if with_map:
        summary["visible_map"] = residue_map
    return summary


def _rgb_residue_diagnostics(image: np.ndarray, rgb: np.ndarray,
                             alpha: np.ndarray, ctx: Dict[str, np.ndarray],
                             with_map: bool = False) -> Dict[str, Dict[str, object]]:
    before = _rgb_residue_summary(image, alpha, ctx, with_map=with_map)
    after = _rgb_residue_summary(rgb, alpha, ctx, with_map=with_map)
    base = max(float(before["visible_mean"]), 1e-6)
    after["visible_improve"] = float((before["visible_mean"] - after["visible_mean"]) / base)
    return {"before": before, "after": after}


def _edge_width(ctx: Dict[str, np.ndarray]) -> Tuple[float, float]:
    vals = ctx["dist_to_solid"][ctx["fringe"]]
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(np.percentile(vals, 95))


def _dump_stats(profile: MatteProfile, width: Tuple[float, float],
                guard: Dict[str, float], rollback: bool,
                strength: np.ndarray, ctx: Dict[str, np.ndarray],
                rgb_residue: Optional[Dict[str, Dict[str, object]]] = None) -> None:
    metrics = ctx.get("defringe_metrics", {})
    active = strength[strength > 0]
    strength_mean = float(active.mean()) if active.size else 0.0
    strength_p95 = float(np.percentile(active, 95)) if active.size else 0.0
    active_pixels = int(np.sum(strength > 0))
    defringe_soft = int(np.sum((strength > 0) & ctx["fringe"]))
    defringe_opaque = int(np.sum((strength > 0) & ctx["opaque_rim"]))
    defringe_solid = int(np.sum((strength > 0) & ctx["solid_rim"]))
    pct = lambda value: value * 100.0
    print(
        "[后处理诊断] Alpha分布: "
        f"edge={pct(profile.edge_ratio):.2f}% haze={pct(profile.haze_ratio):.2f}% "
        f"fringe={profile.fringe_pixels}px({pct(profile.fringe_ratio):.2f}%/edge) solid={profile.solid_area}px"
    )
    print(
        "[后处理诊断] 拓扑边缘: "
        f"outer_fringe={profile.outer_fringe_pixels}px hole_fringe={profile.hole_fringe_pixels}px "
        f"width_mean={width[0]:.2f}px width_p95={width[1]:.2f}px halo_score={profile.halo_score:.2f}"
    )
    print(
        "[后处理诊断] 边色污染: "
        f"bright={profile.bright_fringe_score:.2f} dark={profile.dark_fringe_score:.2f} "
        f"edge_bias={profile.defringe} strength_mean={strength_mean:.3f} strength_p95={strength_p95:.3f}"
    )
    print(
        "[postprocess diag] RGB recovery: "
        f"method={metrics.get('method', 'background_direction_despill')} replace=background_direction "
        f"scope=edge-only applied={active_pixels}px "
        f"soft={metrics.get('soft_pixels', 0)}px "
        f"high_alpha={metrics.get('high_alpha_pixels', 0)}px "
        f"opaque={metrics.get('opaque_pixels', 0)}px "
        f"screen={metrics.get('screen_pixels', 0)}px "
        f"skip_bg={metrics.get('skipped_low_bg_conf', 0)}px "
        f"skip_amb={metrics.get('skipped_ambiguous', 0)}px "
        f"skip_proj={metrics.get('skipped_projection', 0)}px "
        f"skip_protect={metrics.get('skipped_protected', 0)}px"
    )
    print(
        "[postprocess diag] BG confidence: "
        f"seed={metrics.get('bg_seed_pixels', profile.bg_color_seed_pixels)}px "
        f"near_edge={metrics.get('bg_seed_near_edge', 0.0) * 100.0:.2f}% "
        f"conf={metrics.get('bg_conf_mean', 0.0):.3f}/"
        f"{metrics.get('bg_conf_p10', 0.0):.3f}(mean/p10) "
        f"var_p95={metrics.get('bg_var_p95', 0.0):.3f} "
        f"fill_err_p95={metrics.get('bg_fill_error_p95', 0.0):.3f}"
    )
    print(
        "[postprocess diag] Screen despill: "
        f"green={metrics.get('screen_green_pixels', 0)}px "
        f"blue={metrics.get('screen_blue_pixels', 0)}px "
        f"strength={metrics.get('screen_strength_mean', 0.0):.3f}/"
        f"{metrics.get('screen_strength_p95', 0.0):.3f}(mean/p95)"
    )
    print(
        "[postprocess diag] Color model: "
        f"safe_fg_seed={profile.safe_fg_seed_pixels}px bg_seed={profile.bg_color_seed_pixels}px "
        f"spill_mean={profile.spill_score_mean:.3f} spill_p95={profile.spill_score_p95:.3f} "
        f"opaque_rim={profile.opaque_rim_pixels}px({pct(profile.opaque_rim_ratio):.2f}%/color_fringe) "
        f"solid_rim={profile.solid_rim_pixels}px"
    )
    print(
        "[postprocess diag] Defringe coverage: "
        f"soft={defringe_soft}px opaque={defringe_opaque}px solid={defringe_solid}px "
        f"total={active_pixels}px"
    )
    if rgb_residue is not None:
        before = rgb_residue["before"]
        after = rgb_residue["after"]
        print(
            "[postprocess diag] RGB residue(final-alpha): "
            f"pixels={after['pixels']} "
            f"before_spill={before['spill_mean']:.3f}/{before['spill_p95']:.3f} "
            f"after_spill={after['spill_mean']:.3f}/{after['spill_p95']:.3f} "
            f"visible={before['visible_mean']:.3f}->{after['visible_mean']:.3f} "
            f"p95={before['visible_p95']:.3f}->{after['visible_p95']:.3f} "
            f"improve={pct(after['visible_improve']):.2f}%"
        )
        before_bins = metrics.get("residue_before_by_alpha", {})
        after_bins = metrics.get("residue_after_by_alpha", {})
        print(
            "[postprocess diag] RGB residue by alpha: "
            f"<64={before_bins.get('lt64', 0.0):.3f}->{after_bins.get('lt64', 0.0):.3f} "
            f"64-180={before_bins.get('64_180', 0.0):.3f}->{after_bins.get('64_180', 0.0):.3f} "
            f"180-240={before_bins.get('180_240', 0.0):.3f}->{after_bins.get('180_240', 0.0):.3f} "
            f">=240={before_bins.get('gte240', 0.0):.3f}->{after_bins.get('gte240', 0.0):.3f}"
        )
    print(
        "[后处理诊断] 保护区域: "
        f"detail={profile.detail_pixels}px({pct(profile.detail_ratio):.2f}%/edge) "
        f"transparency={profile.protected_transparency_pixels}px({pct(profile.transparency_ratio):.2f}%/solid)"
    )
    print(
        "[后处理诊断] 后处理变化: "
        f"alpha_area_loss={pct(guard['alpha_area_loss']):.2f}% "
        f"solid_area_loss={pct(guard['solid_loss']):.2f}% "
        f"alpha_l1_delta={guard['alpha_l1']:.4f} rollback={rollback}"
    )
    print(
        "[postprocess diag] Alpha delta: "
        f"positive={guard['alpha_positive_pixels']}px negative={guard['alpha_negative_pixels']}px "
        f"pos_l1={guard['alpha_positive_l1']:.4f} neg_l1={guard['alpha_negative_l1']:.4f} "
        f"area_gain={pct(guard['alpha_area_gain']):.2f}% solid_gain={pct(guard['solid_gain']):.2f}%"
    )
    print(
        "[后处理诊断] 决策: "
        f"profile={profile.profile} alpha_tighten={profile.alpha_tighten} edge_bias={profile.defringe}"
    )


def _save_debug(debug_dir: str, image: np.ndarray, rgb: np.ndarray,
                alpha_before: np.ndarray, alpha_after: np.ndarray,
                ctx: Dict[str, np.ndarray], strength: np.ndarray) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(image, "RGB").save(os.path.join(debug_dir, "00_input_rgb.png"))
    Image.fromarray(alpha_before, "L").save(os.path.join(debug_dir, "30_pre_postprocess_alpha.png"))
    Image.fromarray(alpha_after, "L").save(os.path.join(debug_dir, "31_postprocess_alpha.png"))
    delta = alpha_after.astype(np.int16) - alpha_before.astype(np.int16)
    delta_rgb = np.zeros((*delta.shape, 3), dtype=np.uint8)
    delta_rgb[..., 0] = np.clip(-delta, 0, 255).astype(np.uint8)
    delta_rgb[..., 1] = np.clip(255 - np.abs(delta), 0, 255).astype(np.uint8) // 4
    delta_rgb[..., 2] = np.clip(delta, 0, 255).astype(np.uint8)
    Image.fromarray(delta_rgb, "RGB").save(os.path.join(debug_dir, "32_postprocess_alpha_delta.png"))

    masks = {
        "40_outer_fringe_mask.png": ctx["outer_fringe"],
        "41_hole_fringe_mask.png": ctx["hole_fringe"],
        "42_protected_detail_mask.png": ctx["detail"],
        "43_protected_transparency_mask.png": ctx["protected_transparency"],
        "44_defringe_strength.png": strength,
        "45_safe_fg_seed.png": ctx["safe_fg_seed"],
        "46_bg_color_seed.png": ctx["bg_color_seed"],
        "47_opaque_rim_mask.png": ctx["opaque_rim"],
        "48_spill_score.png": ctx["spill_score"],
        "49_solid_rim_mask.png": ctx["solid_rim"],
    }
    for name, mask in masks.items():
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        else:
            mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        Image.fromarray(mask, "L").save(os.path.join(debug_dir, name))

    Image.fromarray(
        np.clip(ctx["foreground_fill"], 0, 255).astype(np.uint8),
        "RGB",
    ).save(os.path.join(debug_dir, "50_foreground_fill.png"))
    Image.fromarray(
        np.clip(ctx["background_fill"], 0, 255).astype(np.uint8),
        "RGB",
    ).save(os.path.join(debug_dir, "51_local_bg_fill.png"))
    if "bg_confidence" in ctx:
        Image.fromarray(
            np.clip(ctx["bg_confidence"] * 255, 0, 255).astype(np.uint8),
            "L",
        ).save(os.path.join(debug_dir, "56_bg_confidence.png"))
    if "despill_projection" in ctx:
        Image.fromarray(
            np.clip(ctx["despill_projection"] * 255, 0, 255).astype(np.uint8),
            "L",
        ).save(os.path.join(debug_dir, "57_despill_projection.png"))
    if "screen_despill_strength" in ctx:
        Image.fromarray(
            np.clip(ctx["screen_despill_strength"] * 255, 0, 255).astype(np.uint8),
            "L",
        ).save(os.path.join(debug_dir, "58_screen_despill_strength.png"))
    rgb_delta = np.max(np.abs(rgb.astype(np.int16) - image.astype(np.int16)), axis=2)
    rgb_delta[(alpha_after == 0) | (strength <= 0)] = 0
    Image.fromarray(np.clip(rgb_delta * 4, 0, 255).astype(np.uint8), "L").save(
        os.path.join(debug_dir, "52_rgb_defringe_delta.png")
    )
    residue_diag = _rgb_residue_diagnostics(image, rgb, alpha_after, ctx, with_map=True)
    before_residue = residue_diag["before"]["visible_map"]
    after_residue = residue_diag["after"]["visible_map"]
    Image.fromarray(np.clip(before_residue * 255, 0, 255).astype(np.uint8), "L").save(
        os.path.join(debug_dir, "53_rgb_residue_before.png")
    )
    Image.fromarray(np.clip(after_residue * 255, 0, 255).astype(np.uint8), "L").save(
        os.path.join(debug_dir, "54_rgb_residue_after.png")
    )
    residue_delta = before_residue - after_residue
    residue_delta_rgb = np.zeros((*alpha_after.shape, 3), dtype=np.uint8)
    residue_delta_rgb[..., 1] = np.clip(residue_delta * 255 * 2, 0, 255).astype(np.uint8)
    residue_delta_rgb[..., 0] = np.clip(-residue_delta * 255 * 2, 0, 255).astype(np.uint8)
    Image.fromarray(residue_delta_rgb, "RGB").save(
        os.path.join(debug_dir, "55_rgb_residue_delta.png")
    )

    for name, color in {
        "60_composite_black.png": (0, 0, 0),
        "61_composite_white.png": (255, 255, 255),
        "62_composite_gray.png": (128, 128, 128),
        "63_composite_green.png": (0, 170, 80),
    }.items():
        bg = np.full((*alpha_after.shape, 3), color, dtype=np.uint8)
        af = alpha_after.astype(np.float32) / 255.0
        comp = rgb.astype(np.float32) * af[..., None] + bg.astype(np.float32) * (1.0 - af[..., None])
        Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), "RGB").save(os.path.join(debug_dir, name))


def _foreground_crop(alpha: np.ndarray, margin: int = 64) -> Optional[Tuple[int, int, int, int]]:
    """Crop around non-zero alpha; return None when the crop is almost full-frame."""
    return _mask_crop(alpha > 0, margin=margin)


def _clean_rgba_core(image_u8: np.ndarray, alpha_u8: np.ndarray, debug_dir: Optional[str],
                     preserve_transparency: bool) -> np.ndarray:
    profile, ctx = analyze_matte(
        image_u8,
        alpha_u8,
        preserve_transparency=preserve_transparency,
        full_spill=debug_dir is not None,
    )
    width = _edge_width(ctx) if debug_dir else (0.0, 0.0)
    alpha_refined = _refine_alpha(alpha_u8, ctx, profile)
    alpha_refined, rollback, guard = _guard_against_overcut(alpha_u8, alpha_refined, profile)

    # 复用首次 context：前景/背景色估计的 seed 取自稳定的实心核与背景，refine 仅收紧
    # 了 <3% 的 fringe，对低频颜色填充无实质影响，故省去二次全图重建。
    ctx["defringe_alpha"] = alpha_u8
    ctx["defringe_fringe"] = ctx["color_fringe"]
    rgb, strength = _defringe_rgb(image_u8, alpha_refined, ctx, profile)

    if debug_dir:
        rgb_residue = _rgb_residue_diagnostics(image_u8, rgb, alpha_refined, ctx)
        _dump_stats(profile, width, guard, rollback, strength, ctx, rgb_residue)
        _save_debug(debug_dir, image_u8, rgb, alpha_u8, alpha_refined, ctx, strength)
    return np.dstack([rgb, alpha_refined])


def make_clean_rgba(image: np.ndarray, alpha: np.ndarray, debug_dir: Optional[str] = None,
                    preserve_transparency: bool = False) -> np.ndarray:
    """Return RGBA using topology-aware alpha calibration and RGB decontamination."""
    import time
    t0 = time.perf_counter()
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(alpha, 0, 255).astype(np.uint8)

    # debug 模式保持全图，保证诊断中间图与原图对齐。
    crop = _foreground_crop(alpha_u8) if debug_dir is None else None
    if crop is not None:
        y1, y2, x1, x2 = crop
        rgba_crop = _clean_rgba_core(image_u8[y1:y2, x1:x2], alpha_u8[y1:y2, x1:x2],
                                     None, preserve_transparency)
        rgba = np.zeros((*alpha_u8.shape, 4), dtype=np.uint8)
        rgba[y1:y2, x1:x2] = rgba_crop
    else:
        rgba = _clean_rgba_core(image_u8, alpha_u8, debug_dir, preserve_transparency)

    print(f"[后处理] RGBA 后处理耗时 {time.perf_counter() - t0:.2f}s")
    return rgba
