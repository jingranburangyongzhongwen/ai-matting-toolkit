# ── Tab 2 后端：精细选区 ────────────────────────────────────────
import hashlib
import os
import threading
import time
from collections import OrderedDict

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from log import get_logger
from model_manager import free_vram_gb, get_output_path

logger = get_logger(__name__)

# ── 模块级状态 ───────────────────────────────────────────────────
import uuid as _uuid
_mgr = None

KEEP_RESIDENT_FREE_GB = 6.0
_SAM_SESSION_LOCK = threading.RLock()
_SAM_CONTEXTS = OrderedDict()
_STALE_SAM_ACTIVE = {}
_MULTI_SESSION_MODE = False
_MAX_SAM_CONTEXTS = 1
# 自动分割选择历史（替代 gr.State 跨事件传递）
_SELECT_HISTORY = {}
ENGINE_MODE_MAP = {"快速": "mobile_sam", "高精度": "sam_hq"}
TAB2_OUTPUT_MODES = {"SAM严格": "sam_strict", "RMBG精修": "rmbg_refine"}
AUTO_SEGMENT_MAX_MASKS = 120
_AGGRESSIVE_UNLOAD_ENV = os.environ.get("MATTING_AGGRESSIVE_UNLOAD")
_AGGRESSIVE_UNLOAD = (
    _AGGRESSIVE_UNLOAD_ENV == "1"
    if _AGGRESSIVE_UNLOAD_ENV is not None else True
)


def init(mgr_instance):
    global _mgr
    _mgr = mgr_instance


# ── SAM 会话管理 ─────────────────────────────────────────────────

def configure_runtime(multi_session: bool, max_sam_sessions: int):
    """启动期配置：默认单 session 低显存友好，多 session 保持模型常驻。"""
    global _MULTI_SESSION_MODE, _AGGRESSIVE_UNLOAD
    _MULTI_SESSION_MODE = bool(multi_session)
    if _AGGRESSIVE_UNLOAD_ENV is None:
        _AGGRESSIVE_UNLOAD = not _MULTI_SESSION_MODE
    else:
        _AGGRESSIVE_UNLOAD = _AGGRESSIVE_UNLOAD_ENV == "1"
    _mgr.set_multi_session_mode(_MULTI_SESSION_MODE)
    _set_max_sam_contexts(max_sam_sessions if _MULTI_SESSION_MODE else 1)
    mode = "multi-session" if _MULTI_SESSION_MODE else "single-session"
    logger.info("runtime mode=%s, max_sam_sessions=%d, aggressive_unload=%s",
                mode, _MAX_SAM_CONTEXTS, _AGGRESSIVE_UNLOAD)


def _set_max_sam_contexts(value: int):
    global _MAX_SAM_CONTEXTS
    _MAX_SAM_CONTEXTS = max(1, int(value))
    with _SAM_SESSION_LOCK:
        _evict_sam_contexts_unlocked()


def _request_session_id(request=None):
    if not _MULTI_SESSION_MODE:
        return "default"
    session_hash = getattr(request, "session_hash", None)
    return str(session_hash) if session_hash else "default"


def _cleanup_sam_context_unlocked(ctx):
    if isinstance(ctx, dict) and ctx.get("cleaned"):
        return
    sam = ctx.get("sam") if isinstance(ctx, dict) else None
    if sam is not None and hasattr(sam, "cleanup"):
        try:
            sam.cleanup()
        except Exception as exc:
            logger.warning("SAM context cleanup failed: %s", exc)
    if isinstance(ctx, dict):
        ctx["cleaned"] = True


_STALE_ACTIVE_TIMEOUT_SECS = 300.0  # 5 分钟无释放视为断连


def _reap_stale_active_contexts():
    """扫描所有上下文，强释 active 超时的孤儿（用户断连）。"""
    now = time.monotonic()
    with _SAM_SESSION_LOCK:
        for key, ctx in list(_SAM_CONTEXTS.items()):
            if ctx.get("active", 0) <= 0:
                continue
            since = ctx.get("_active_since", 0.0)
            if since <= 0 or (now - since) <= _STALE_ACTIVE_TIMEOUT_SECS:
                continue
            logger.warning(
                "SAM context active %.0fs (session=%s), force-releasing",
                now - since, key[0],
            )
            ctx["active"] = 0
            ctx["_active_since"] = 0.0
            if ctx.get("stale"):
                engine_type = ctx.get("engine_type")
                _STALE_SAM_ACTIVE[engine_type] = max(
                    0, int(_STALE_SAM_ACTIVE.get(engine_type, 0)) - 1
                )
                if _STALE_SAM_ACTIVE.get(engine_type, 0) == 0:
                    _STALE_SAM_ACTIVE.pop(engine_type, None)
                ctx["stale"] = False
            lock = ctx.get("lock")
            acquired = lock.acquire(blocking=False) if lock is not None else True
            if not acquired:
                continue
            try:
                _cleanup_sam_context_unlocked(ctx)
            finally:
                lock.release()
            _SAM_CONTEXTS.pop(key, None)
            logger.info("reaped stale SAM context: session=%s engine=%s", key[0], key[1])


def _evict_sam_contexts_unlocked(keep_key=None):
    scans_left = len(_SAM_CONTEXTS)
    while len(_SAM_CONTEXTS) > _MAX_SAM_CONTEXTS and scans_left > 0:
        old_key, old_ctx = _SAM_CONTEXTS.popitem(last=False)
        scans_left -= 1
        if old_key == keep_key or old_ctx.get("active", 0):
            _SAM_CONTEXTS[old_key] = old_ctx
            continue
        lock = old_ctx.get("lock")
        acquired = lock.acquire(blocking=False) if lock is not None else True
        if not acquired:
            _SAM_CONTEXTS[old_key] = old_ctx
            continue
        try:
            _cleanup_sam_context_unlocked(old_ctx)
        finally:
            if lock is not None:
                lock.release()
        logger.info("evicted SAM context: session=%s engine=%s", old_key[0], old_key[1])


def _get_sam_context(request, engine_type, retain=False):
    _reap_stale_active_contexts()
    session_id = _request_session_id(request)
    key = (session_id, engine_type)
    with _SAM_SESSION_LOCK:
        ctx = _SAM_CONTEXTS.get(key)
        if ctx is None:
            engine = _mgr.get_sam_engine(engine_type)
            ctx = {
                "session_id": session_id,
                "engine_type": engine_type,
                "sam": engine.create_session(),
                "fingerprint": None,
                "box_logits": {},
                "free_logits": {},
                "lock": threading.RLock(),
                "active": 0,
                "_active_since": 0.0,
                "stale": False,
                "cleaned": False,
            }
            _SAM_CONTEXTS[key] = ctx
            _evict_sam_contexts_unlocked(keep_key=key)
        else:
            _SAM_CONTEXTS.move_to_end(key)
        if retain:
            ctx["active"] += 1
            if ctx["active"] == 1:
                ctx["_active_since"] = time.monotonic()
        return ctx


def _release_sam_context(ctx):
    should_cleanup = False
    unload_engine_type = None
    with _SAM_SESSION_LOCK:
        ctx["active"] = max(0, int(ctx.get("active", 0)) - 1)
        if ctx["active"] == 0:
            ctx["_active_since"] = 0.0
        should_cleanup = ctx["active"] == 0 and bool(ctx.get("stale"))
        if should_cleanup:
            engine_type = ctx.get("engine_type")
            _STALE_SAM_ACTIVE[engine_type] = max(
                0, int(_STALE_SAM_ACTIVE.get(engine_type, 0)) - 1
            )
            if _STALE_SAM_ACTIVE[engine_type] == 0:
                _STALE_SAM_ACTIVE.pop(engine_type, None)
            live_same_type = any(key[1] == engine_type for key in _SAM_CONTEXTS)
            if not _MULTI_SESSION_MODE and not live_same_type:
                unload_engine_type = engine_type
    if should_cleanup:
        with ctx["lock"]:
            _cleanup_sam_context_unlocked(ctx)
    if unload_engine_type:
        _mgr.unload_sam_engine(unload_engine_type)


def _clear_session_sam_contexts(request=None):
    session_id = _request_session_id(request)
    had_active = False
    removed_types = set()
    with _SAM_SESSION_LOCK:
        stale_keys = [key for key in _SAM_CONTEXTS if key[0] == session_id]
        for key in stale_keys:
            ctx = _SAM_CONTEXTS.pop(key)
            removed_types.add(key[1])
            if ctx.get("active", 0):
                if not ctx.get("stale"):
                    ctx["stale"] = True
                    _STALE_SAM_ACTIVE[key[1]] = _STALE_SAM_ACTIVE.get(key[1], 0) + 1
                had_active = True
                continue
            with ctx["lock"]:
                _cleanup_sam_context_unlocked(ctx)
    if not _MULTI_SESSION_MODE and not had_active:
        live_types = {key[1] for key in _SAM_CONTEXTS}
        for engine_type in removed_types - live_types:
            _mgr.unload_sam_engine(engine_type)


def _clear_all_sam_contexts():
    had_active = False
    with _SAM_SESSION_LOCK:
        while _SAM_CONTEXTS:
            _, ctx = _SAM_CONTEXTS.popitem(last=False)
            if ctx.get("active", 0):
                if not ctx.get("stale"):
                    ctx["stale"] = True
                    engine_type = ctx.get("engine_type")
                    _STALE_SAM_ACTIVE[engine_type] = _STALE_SAM_ACTIVE.get(engine_type, 0) + 1
                had_active = True
                continue
            with ctx["lock"]:
                _cleanup_sam_context_unlocked(ctx)
    if not had_active:
        _mgr.unload_sam()


def _reset_sam_interaction_state_unlocked(ctx):
    ctx["box_logits"] = {}
    ctx["free_logits"] = {}
    sam = ctx.get("sam")
    if sam is None:
        return
    sam._prev_logits = None
    sam._prev_npoints = 0
    sam._cached_mask = None


def _reset_session_sam_interaction_state(request=None):
    session_id = _request_session_id(request)
    with _SAM_SESSION_LOCK:
        contexts = [ctx for key, ctx in _SAM_CONTEXTS.items() if key[0] == session_id]
    for ctx in contexts:
        with ctx["lock"]:
            _reset_sam_interaction_state_unlocked(ctx)


def _unload_unused_for_tab2_sam_hq(keep_grounding_dino=False):
    if not _AGGRESSIVE_UNLOAD:
        return
    if not _MULTI_SESSION_MODE or free_vram_gb() < KEEP_RESIDENT_FREE_GB:
        _mgr.unload_vitmatte()
        if not keep_grounding_dino:
            _mgr.unload_grounding_dino()


def _ensure_sam_ready(image, engine_mode, keep_grounding_dino=False,
                      request=None, retain=False, image_id=None):
    engine_type = ENGINE_MODE_MAP.get(engine_mode, "mobile_sam")
    if not _MULTI_SESSION_MODE:
        with _SAM_SESSION_LOCK:
            current_types = {
                key[1] for key, ctx in _SAM_CONTEXTS.items()
                if not ctx.get("stale")
            }
        if current_types and engine_type not in current_types:
            _clear_session_sam_contexts(request)
    if engine_type == "sam_hq":
        _unload_unused_for_tab2_sam_hq(keep_grounding_dino=keep_grounding_dino)
    ctx = _get_sam_context(request, engine_type, retain=retain)
    if image_id is not None:
        fingerprint = (image.shape, image.dtype.str, image_id)
    else:
        fingerprint = _image_fingerprint(image)
    try:
        with ctx["lock"]:
            sam = ctx["sam"]
            if not sam._image_set or fingerprint != ctx.get("fingerprint"):
                sam.set_image(image)
                ctx["fingerprint"] = fingerprint
                _reset_sam_interaction_state_unlocked(ctx)
    except Exception:
        if retain:
            _release_sam_context(ctx)
        raise
    return ctx


# ── Box 几何工具 ─────────────────────────────────────────────────

def _normalize_box_state(box_state):
    if box_state is None:
        return []
    arr = np.asarray(box_state, dtype=object)
    if arr.ndim == 1 and len(arr) == 4:
        return [[float(v) for v in arr.tolist()]]
    boxes = []
    for item in box_state:
        vals = np.asarray(item, dtype=float).reshape(-1)
        if vals.size != 4:
            continue
        x1, y1, x2, y2 = vals.tolist()
        if x2 > x1 and y2 > y1:
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a, b):
    inter = _box_intersection(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _box_containment(container, inner):
    inner_area = _box_area(inner)
    if inner_area <= 0:
        return 0.0
    return _box_intersection(container, inner) / inner_area


def _filter_text_candidate_boxes(boxes, scores, image_shape, max_prompts=8):
    h, w = image_shape[:2]
    has_scores = scores is not None and len(scores) == len(boxes)
    if not has_scores:
        scores = [0.0] * len(boxes)
    candidates = []
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [float(v) for v in box]
        clipped = [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]
        if _box_area(clipped) >= 64 and (not has_scores or score >= 0.30):
            candidates.append({"box": clipped, "score": float(score), "idx": idx})

    if has_scores:
        candidates.sort(key=lambda item: item["score"], reverse=True)
    stats = {
        "raw": len(candidates),
        "wrapper": 0,
        "duplicate": 0,
        "capped": 0,
        "has_scores": has_scores,
    }
    if not candidates:
        return [], [], stats

    no_wrappers = []
    for i, item in enumerate(candidates):
        box = item["box"]
        area = _box_area(box)
        children = [
            other for j, other in enumerate(candidates)
            if i != j
            and area > _box_area(other["box"]) * 1.8
            and _box_containment(box, other["box"]) >= 0.94
        ]
        if children:
            best_child_score = max(child["score"] for child in children) if has_scores else 0.0
            if len(children) >= 2 or not has_scores or item["score"] <= best_child_score + 0.12:
                stats["wrapper"] += 1
                continue
        no_wrappers.append(item)

    filtered = no_wrappers or candidates
    selected = []
    for item in filtered:
        if any(_box_iou(item["box"], kept["box"]) >= 0.78 for kept in selected):
            stats["duplicate"] += 1
            continue
        selected.append(item)
        if len(selected) >= max_prompts:
            break
    stats["capped"] = max(0, len(filtered) - stats["duplicate"] - len(selected))

    return [item["box"] for item in selected], [item["score"] for item in selected], stats


def _union_boxes(boxes, image_shape):
    boxes = _normalize_box_state(boxes)
    if not boxes:
        return None
    h, w = image_shape[:2]
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return [
        max(0, int(min(xs1))),
        max(0, int(min(ys1))),
        min(w, int(max(xs2))),
        min(h, int(max(ys2))),
    ]


def _expand_box(box, image_shape, ratio=0.08, min_margin=12):
    if box is None:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw, bh = x2 - x1, y2 - y1
    margin = max(min_margin, int(max(bw, bh) * ratio))
    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(w, x2 + margin),
        min(h, y2 + margin),
    ]


def _point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _point_in_any_box(point, boxes):
    x, y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in boxes)


def _box_cache_key(box):
    return tuple(float(v) for v in box)


def _points_for_box(points, labels, box):
    local_points = []
    local_labels = []
    for pt, label in zip(points, labels):
        if label == 0 or _point_in_box(pt, box):
            local_points.append(pt)
            local_labels.append(label)
    return local_points, local_labels


# ── Mask / 指纹工具 ─────────────────────────────────────────────

def _image_fingerprint(image):
    if image is None:
        return None
    arr = np.ascontiguousarray(image)
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return arr.shape, arr.dtype.str, digest


def _component_bbox_at(mask, point):
    if mask is None or point is None:
        return None
    mask_u8 = (mask > 0).astype(np.uint8)
    x, y = int(point[0]), int(point[1])
    if not (0 <= y < mask_u8.shape[0] and 0 <= x < mask_u8.shape[1]):
        return None
    num_labels, labels = cv2.connectedComponents(mask_u8)
    label = labels[y, x]
    if num_labels <= 1 or label == 0:
        return None
    ys, xs = np.where(labels == label)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_bbox(mask, image_shape, fallback_box=None):
    h, w = image_shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        fallback = _union_boxes(fallback_box, image_shape)
        if fallback is None:
            return None
        return fallback
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_bbox_from_seg(segmentation):
    ys, xs = np.where(segmentation)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_seg_contains(seg_parent, seg_child, ratio=0.95):
    seg_parent = np.asarray(seg_parent, dtype=bool)
    seg_child = np.asarray(seg_child, dtype=bool)
    child_area = int(seg_child.sum())
    if child_area <= 0:
        return False
    parent_area = int(seg_parent.sum())
    if parent_area <= child_area:
        return False
    overlap = int((seg_parent & seg_child).sum())
    return overlap >= child_area * ratio


def _merge_auto_masks(masks, indices):
    """合并多个 auto-mask 的 segmentation，返回 merged bool array。"""
    if not indices:
        return None
    seg_shape = masks[indices[0]]["segmentation"].shape
    merged = np.zeros(seg_shape, dtype=bool)
    for idx in indices:
        if 0 <= idx < len(masks):
            seg = np.asarray(masks[idx]["segmentation"], dtype=bool)
            merged |= seg
    return merged


# ── 选择历史管理（模块级，绕过 gr.State 跨事件传递问题）──

def _history_key(auto_masks_state, request=None):
    """生成选择历史的 key（session_id + masks 实例 id）。"""
    session_id = _request_session_id(request)
    return (session_id, id(auto_masks_state) if auto_masks_state else 0)


def _get_history(auto_masks_state, request=None):
    return list(_SELECT_HISTORY.get(_history_key(auto_masks_state, request), []))


def _push_history(auto_masks_state, step, request=None):
    key = _history_key(auto_masks_state, request)
    hist = _SELECT_HISTORY.get(key, [])
    hist.append(step)
    _SELECT_HISTORY[key] = hist


def _pop_history(auto_masks_state, request=None):
    key = _history_key(auto_masks_state, request)
    hist = _SELECT_HISTORY.get(key, [])
    if hist:
        step = hist.pop()
        _SELECT_HISTORY[key] = hist
        return step, list(hist)
    return None, []


def _clear_history(auto_masks_state, request=None):
    _SELECT_HISTORY.pop(_history_key(auto_masks_state, request), None)


# ── SAM 预测 ────────────────────────────────────────────────────

def _set_sam_logits_for_prediction(sam, logits):
    sam._prev_logits = logits
    sam._prev_npoints = 0 if logits is None else 1


def _prompt_signature(points, labels):
    return tuple(
        (float(pt[0]), float(pt[1]), int(label))
        for pt, label in zip(points, labels)
    )


def _prefix_prompt_signature(points, labels):
    if not points:
        return None
    return _prompt_signature(points[:-1], labels[:-1])


def _cache_get_with_prefix(cache, current_key, prefix_key):
    if current_key in cache:
        return cache[current_key]
    if prefix_key is not None:
        return cache.get(prefix_key)
    return None


def _filter_tab2_box_logits(ctx, boxes, *, keep_active):
    active_keys = {_box_cache_key(box) for box in _normalize_box_state(boxes)}
    fingerprint = ctx.get("fingerprint")
    ctx["box_logits"] = {
        key: value for key, value in ctx["box_logits"].items()
        if (
            isinstance(key, tuple) and len(key) >= 4
            and key[0] == fingerprint
            and key[2] in active_keys
        ) == keep_active
    }


def _prune_tab2_free_logits(ctx):
    fingerprint = ctx.get("fingerprint")
    ctx["free_logits"] = {
        key: value for key, value in ctx["free_logits"].items()
        if isinstance(key, tuple) and len(key) >= 3 and key[0] == fingerprint
    }


def _update_positive_prompt_box(mask, box_state, image_shape, point=None):
    mask_box = _component_bbox_at(mask, point) or _mask_bbox(mask, image_shape, fallback_box=None)
    boxes = _normalize_box_state(box_state)
    if not boxes:
        return [_expand_box(mask_box, image_shape)] if mask_box is not None else box_state
    if point is not None:
        for idx, box in enumerate(boxes):
            if _point_in_box(point, box):
                merged = _union_boxes([box, mask_box], image_shape) if mask_box else box
                boxes[idx] = _expand_box(merged, image_shape)
                return boxes
        if mask_box is not None:
            boxes.append(_expand_box(mask_box, image_shape))
        return boxes
    if len(boxes) == 1 and mask_box is not None:
        return [_expand_box(_union_boxes([boxes[0], mask_box], image_shape), image_shape)]
    return boxes


def _exclude_object_at_point(sam, x, y, current_mask):
    saved_logits = sam._prev_logits
    saved_npoints = sam._prev_npoints
    saved_cached = sam._cached_mask
    saved_cached_logits = sam._cached_logits
    try:
        sam._prev_logits = None
        sam._prev_npoints = 0
        exclude_mask = sam.predict_mask([[x, y]], [1], box=None)
        return current_mask & ~exclude_mask
    finally:
        sam._prev_logits = saved_logits
        sam._prev_npoints = saved_npoints
        sam._cached_mask = saved_cached
        sam._cached_logits = saved_cached_logits


def _apply_object_exclusions(sam, points, labels, mask):
    for pt, label in zip(points, labels):
        if label == 0 and mask[int(pt[1]), int(pt[0])]:
            mask = _exclude_object_at_point(sam, pt[0], pt[1], mask)
    return mask


def _predict_tab2_mask(ctx, points, labels, box_state):
    """统一的 SAM 预测入口，支持 box/free 两种模式。"""
    with ctx["lock"]:
        sam = ctx["sam"]
        fingerprint = ctx.get("fingerprint")
        points = list(points or [])
        labels = list(labels or [])
        boxes = _normalize_box_state(box_state)

        # ── box 模式 ──
        if boxes:
            _filter_tab2_box_logits(ctx, boxes, keep_active=True)
            _prune_tab2_free_logits(ctx)
            combined = np.zeros(sam._original_size, dtype=bool)
            combined_logits = None
            single_box_logits = None
            for box in boxes:
                box_key = _box_cache_key(box)
                box_points, box_labels = _points_for_box(points, labels, box)
                sig = _prompt_signature(box_points, box_labels)
                prefix_sig = _prefix_prompt_signature(box_points, box_labels)
                current_key = (fingerprint, "box", box_key, sig)
                prefix_key = (
                    (fingerprint, "box", box_key, prefix_sig)
                    if prefix_sig is not None else None
                )
                _set_sam_logits_for_prediction(
                    sam,
                    _cache_get_with_prefix(ctx["box_logits"], current_key, prefix_key)
                )
                combined |= sam.predict_mask(box_points, box_labels, box=box)
                if sam._cached_logits is not None:
                    if combined_logits is None:
                        combined_logits = sam._cached_logits.copy()
                    else:
                        combined_logits = np.maximum(combined_logits, sam._cached_logits)
                single_box_logits = sam._prev_logits
                ctx["box_logits"][current_key] = single_box_logits
            if combined_logits is not None:
                sam._cached_logits = combined_logits

            extra_points = [
                pt for pt, label in zip(points, labels)
                if label == 1 and not _point_in_any_box(pt, boxes)
            ]
            has_extra_free_mask = bool(extra_points)
            if extra_points:
                extra_labels = [1] * len(extra_points)
                sig = _prompt_signature(extra_points, extra_labels)
                prefix_sig = _prefix_prompt_signature(extra_points, extra_labels)
                current_key = (fingerprint, "free", sig)
                prefix_key = (
                    (fingerprint, "free", prefix_sig)
                    if prefix_sig is not None else None
                )
                _set_sam_logits_for_prediction(
                    sam,
                    _cache_get_with_prefix(ctx["free_logits"], current_key, prefix_key)
                )
                combined |= sam.predict_mask(extra_points, extra_labels, box=None)
                ctx["free_logits"][current_key] = sam._prev_logits

            sam._cached_mask = combined
            if len(boxes) == 1 and not has_extra_free_mask:
                sam._prev_logits = single_box_logits
                sam._prev_npoints = len(_points_for_box(points, labels, boxes[0])[0])
            else:
                sam._prev_logits = None
                sam._prev_npoints = len(points)
            return combined

        # ── free 模式 ──
        ctx["box_logits"] = {}
        _prune_tab2_free_logits(ctx)
        sig = _prompt_signature(points, labels)
        prefix_sig = _prefix_prompt_signature(points, labels)
        current_key = (fingerprint, "free", sig)
        prefix_key = (
            (fingerprint, "free", prefix_sig)
            if prefix_sig is not None else None
        )
        _set_sam_logits_for_prediction(
            sam,
            _cache_get_with_prefix(ctx["free_logits"], current_key, prefix_key)
        )
        mask = sam.predict_mask(points, labels, box=None)
        ctx["free_logits"][current_key] = sam._prev_logits
        return mask


def _predict_tab2_box_batch_initial(ctx, boxes):
    with ctx["lock"]:
        sam = ctx["sam"]
        fingerprint = ctx.get("fingerprint")
        boxes = _normalize_box_state(boxes)
        if not boxes:
            return np.zeros(sam._original_size, dtype=bool)
        _filter_tab2_box_logits(ctx, boxes, keep_active=True)
        _prune_tab2_free_logits(ctx)
        result = sam.predict_box_batch(boxes)
        low_res = result.get("low_res")
        for idx, box in enumerate(boxes):
            current_key = (fingerprint, "box", _box_cache_key(box), ())
            ctx["box_logits"][current_key] = (
                low_res[idx] if low_res is not None and idx < len(low_res) else None
            )
        if len(boxes) != 1:
            sam._prev_logits = None
        return result["mask"]


# ── Overlay 绘制 ────────────────────────────────────────────────

def _draw_tab2_overlay(image, mask, points, labels, box_state=None,
                       mask_color=(255, 0, 0), opacity=0.4):
    if mask is not None and np.any(mask):
        darkened = cv2.addWeighted(image, 1.0 - opacity,  # type: ignore[arg-type]
                                   np.full_like(image, mask_color, dtype=np.uint8),
                                   opacity, 0)
        img = np.where(mask[..., None], darkened, image)
    else:
        img = image.copy()
    for box in _normalize_box_state(box_state):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
    for coord, label in zip(points, labels):
        x, y = int(coord[0]), int(coord[1])
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(img, (x, y), 8, color, -1)
        cv2.circle(img, (x, y), 8, (255, 255, 255), 2)
    return img


_AUTO_SEG_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128),
]


def _draw_auto_segment_overlay(image, masks, selected_idx=None, excluded_indices=None):
    """绘制自动分割结果：每个 mask 半透明彩色，选中的高亮。"""
    if selected_idx is None:
        selected = set()
    elif isinstance(selected_idx, (list, tuple, set)):
        selected = {int(i) for i in selected_idx}
    else:
        selected = {int(selected_idx)}
    del excluded_indices
    overlay = image.copy().astype(np.float32)
    for i, mask_info in enumerate(masks):
        color = _AUTO_SEG_COLORS[i % len(_AUTO_SEG_COLORS)]
        opacity = 0.45 if i in selected else 0.2
        seg = np.asarray(mask_info["segmentation"], dtype=bool)
        for c in range(3):
            overlay[:, :, c] = np.where(
                seg,
                overlay[:, :, c] * (1 - opacity) + color[c] * opacity,
                overlay[:, :, c],
            )
    img = overlay.clip(0, 255).astype(np.uint8)
    for i, mask_info in enumerate(masks):
        meta = mask_info.get("_overlay_meta")
        if meta:
            cx, cy, tw, th, label = meta["cx"], meta["cy"], meta["tw"], meta["th"], meta["label"]
        else:
            seg = np.asarray(mask_info["segmentation"], dtype=bool)
            ys, xs = np.where(seg)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(i + 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (cx - tw // 2 - 4, cy - th - 4),
                      (cx + tw // 2 + 4, cy + 4), (0, 0, 0), -1)
        cv2.putText(img, label, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def _postprocess_auto_masks(masks):
    kept = sorted(masks, key=lambda m: (
        float(m.get("predicted_iou", 0.0)),
        float(m.get("stability_score", 0.0)),
        float(m.get("area", 0.0)),
    ), reverse=True)[:AUTO_SEGMENT_MAX_MASKS]
    kept.sort(key=lambda m: int(m.get("area", 0)), reverse=True)
    for i, mask_info in enumerate(kept):
        # Pre-convert to contiguous bool array to avoid repeated decode/copy
        # in _find_masks_at, _draw_auto_segment_overlay, etc.
        seg = np.asarray(mask_info["segmentation"], dtype=bool)
        if not seg.flags["C_CONTIGUOUS"]:
            seg = np.ascontiguousarray(seg)
        mask_info["segmentation"] = seg
        # Compute centroid from bounding box region (much faster than full np.where)
        bbox = mask_info.get("bbox")  # SAM auto masks provide [x, y, w, h]
        if bbox is not None:
            bx, by, bw, bh = [int(v) for v in bbox]
            if bw > 0 and bh > 0:
                crop = seg[by:by + bh, bx:bx + bw]
                ys_local, xs_local = np.where(crop)
                if len(xs_local) > 0:
                    cx, cy = int(xs_local.mean()) + bx, int(ys_local.mean()) + by
                else:
                    cx, cy = bx + bw // 2, by + bh // 2
            else:
                cx, cy = 0, 0
        else:
            ys, xs = np.where(seg)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
            else:
                cx, cy = 0, 0
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        mask_info["_overlay_meta"] = {
            "cx": cx, "cy": cy, "tw": tw, "th": th, "label": label,
        }
    return kept


def _find_masks_at(masks, x, y):
    hits = []
    for i, mask_info in enumerate(masks):
        seg = mask_info["segmentation"]
        if seg[y, x]:
            hits.append((int(mask_info["area"]), i))
    if not hits:
        return []
    hits.sort()
    min_area = hits[0][0]
    return [idx for area, idx in hits if area <= min_area * 10]


# ── Alpha 生成 ──────────────────────────────────────────────────

def _odd_kernel(value, min_value, max_value):
    value = int(np.clip(int(value), min_value, max_value))
    return value if value % 2 == 1 else value + 1


def _pad_bbox(box, image_shape, ratio=0.25, min_pad=96):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = max(min_pad, int(max(bw, bh) * ratio))
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ]


def _bbox_margin_to_image_edge(box, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    return min(x1, y1, w - x2, h - y2)


def _roi_alpha_touches_border(alpha):
    if alpha.size == 0:
        return False
    h, w = alpha.shape
    band = max(4, min(24, min(h, w) // 32))
    border = np.concatenate([
        alpha[:band, :].ravel(),
        alpha[-band:, :].ravel(),
        alpha[:, :band].ravel(),
        alpha[:, -band:].ravel(),
    ])
    if border.size == 0:
        return False
    strong_ratio = np.mean(border > 0.75)
    solid_ratio = np.mean(border > 0.5)
    return strong_ratio > 0.01 or solid_ratio > 0.03


def _build_sam_constraints(sam_mask, bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    base_dim = max(1, min(x2 - x1, y2 - y1))
    sam_u8 = ((sam_mask > 0).astype(np.uint8) * 255)

    soft_k = _odd_kernel(base_dim * 0.04, 15, 101)
    blur_k = _odd_kernel(base_dim * 0.06, 21, 151)
    soft_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (soft_k, soft_k))
    soft = cv2.dilate(sam_u8, soft_kernel)
    soft = cv2.GaussianBlur(soft, (blur_k, blur_k), 0).astype(np.float32) / 255.0
    soft[sam_mask > 0] = 1.0

    allow_k = _odd_kernel(base_dim * 0.045, 19, 101)
    allow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (allow_k, allow_k))
    hard_allow = cv2.dilate(sam_u8, allow_kernel) > 0

    recover_k = _odd_kernel(base_dim * 0.018, 7, 41)
    recover_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (recover_k, recover_k))
    recover_allow = cv2.dilate(sam_u8, recover_kernel) > 0

    return np.clip(soft, 0.0, 1.0), hard_allow, recover_allow


def _suppress_tab2_extra_fringe(alpha, sam_mask, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    base_dim = max(1, min(x2 - x1, y2 - y1))
    sam_u8 = (sam_mask > 0).astype(np.uint8)

    inner_k = _odd_kernel(base_dim * 0.012, 5, 25)
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_k, inner_k))
    near_sam = cv2.dilate(sam_u8, inner_kernel) > 0

    loose_k = _odd_kernel(base_dim * 0.03, 11, 71)
    loose_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (loose_k, loose_k))
    loose_sam = cv2.dilate(sam_u8, loose_kernel) > 0

    out = alpha.copy()
    outside_inner = ~near_sam
    far_outside = ~loose_sam
    out[outside_inner & (out < 0.42)] = 0.0
    out[outside_inner & (out < 0.72)] *= 0.35
    out[far_outside & (out < 0.90)] = 0.0
    return out


def _make_rgba_result(image, alpha, debug_dir=None, preserve_transparency=False):
    from engines.rgba_postprocess import make_clean_rgba
    return make_clean_rgba(
        image, alpha,
        debug_dir=debug_dir,
        preserve_transparency=preserve_transparency,
    )


def _sam_strict_alpha(sam_mask, points_state, labels_state,
                      box_state, image_shape, logits=None):
    subject_box = _mask_bbox(sam_mask, image_shape, fallback_box=box_state)
    if subject_box is None:
        raise ValueError("SAM 未得到有效主体区域")

    x1, y1, x2, y2 = subject_box
    base_dim = max(1, min(x2 - x1, y2 - y1))

    close_k = _odd_kernel(base_dim * 0.006, 3, 11)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask_u8 = (sam_mask > 0).astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    alpha = (mask_u8 > 0).astype(np.float32)
    if logits is not None and logits.shape == sam_mask.shape:
        band_k = _odd_kernel(base_dim * 0.004, 3, 9)
        band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_k, band_k))
        outer = cv2.dilate(mask_u8, band_kernel) > 0
        inner = cv2.erode(mask_u8, band_kernel) > 0
        band = outer & ~inner
        band_alpha = 1.0 / (1.0 + np.exp(-logits[band].astype(np.float32)))
        # 只在 mask 内部使用 band_alpha，保护实心前景不被 sigmoid 降低
        # mask 外部的 band 像素保持 alpha=0（后处理会抑制外溢）
        inside = band & (mask_u8 > 0)
        inside_band = inside[band]  # bool index aligned to band_alpha
        alpha[inside] = np.maximum(alpha[inside], band_alpha[inside_band])

    return (np.clip(alpha, 0.0, 1.0) * 255).round().astype(np.uint8), subject_box


def _sam_guided_rmbg_alpha(image, sam_mask, points_state, labels_state, box_state):
    h, w = image.shape[:2]
    subject_box = _mask_bbox(sam_mask, image.shape, fallback_box=box_state)
    if subject_box is None:
        raise ValueError("SAM 未得到有效主体区域")

    quality_notes = []
    roi_box = _pad_bbox(subject_box, image.shape, ratio=0.32, min_pad=128)
    x1, y1, x2, y2 = roi_box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI 无效，请重新标记主体")

    roi_img = Image.fromarray(image[y1:y2, x1:x2].astype(np.uint8), "RGB")
    roi_alpha = _mgr.rmbg2.predict_alpha(roi_img, clean=True, smooth=True).astype(np.float32) / 255.0

    if _roi_alpha_touches_border(roi_alpha):
        if _bbox_margin_to_image_edge(roi_box, image.shape) <= 1:
            quality_notes.append("主体接近图像边缘")
        else:
            expanded_box = _pad_bbox(subject_box, image.shape, ratio=0.50, min_pad=224)
            if expanded_box != roi_box:
                ex1, ey1, ex2, ey2 = expanded_box
                expanded_img = Image.fromarray(image[ey1:ey2, ex1:ex2].astype(np.uint8), "RGB")
                roi_alpha = (
                    _mgr.rmbg2.predict_alpha(expanded_img, clean=True, smooth=True).astype(np.float32) / 255.0
                )
                roi_box = expanded_box
                quality_notes.append("ROI自动扩边")

    x1, y1, x2, y2 = roi_box
    full_alpha = np.zeros((h, w), dtype=np.float32)
    full_alpha[y1:y2, x1:x2] = roi_alpha[:y2 - y1, :x2 - x1]

    soft_constraint, hard_allow, recover_allow = _build_sam_constraints(
        sam_mask, subject_box, image.shape,
    )
    final_alpha = np.minimum(full_alpha, soft_constraint)

    high_conf = (full_alpha > 0.95) & recover_allow
    final_alpha[high_conf] = full_alpha[high_conf]
    final_alpha[~hard_allow] = 0.0

    # Trimap 约束（业界标准：SAM erode 内部 = 确定前景，matting 不可压制）
    # Matte Anything / ViTMatte 等均采用此方案：SAM mask → erode → foreground,
    # matting 网络在 foreground 区域输出 ≈ 1.0，仅在 unknown 边界带自由预测。
    sx1, sy1, sx2, sy2 = subject_box
    base_dim = max(1, min(sx2 - sx1, sy2 - sy1))
    erode_k = _odd_kernel(base_dim * 0.02, 5, 21)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    sam_foreground = cv2.erode(
        (sam_mask > 0).astype(np.uint8) * 255, erode_kernel,
    ) > 0
    final_alpha[sam_foreground] = np.maximum(final_alpha[sam_foreground], 0.95)
    final_alpha = _suppress_tab2_extra_fringe(final_alpha, sam_mask, subject_box)

    final_alpha = np.clip(final_alpha, 0.0, 1.0)
    return (final_alpha * 255).round().astype(np.uint8), subject_box, roi_box, quality_notes


# ── Tab 2 回调 ──────────────────────────────────────────────────

def on_image_upload(image, request: gr.Request = None):
    """接收 numpy 数组（来自 gr.Image 上传/粘贴）。"""
    _clear_session_sam_contexts(request)
    if image is None:
        return tuple(gr.update() for _ in range(16))
    try:
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        new_image_id = _uuid.uuid4().hex[:16]
        return (gr.update(value=None, visible=False), gr.update(value=img, visible=True),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
                gr.update(value=None, visible=False), [], [], None, None,
                [], new_image_id, False, "已上传 | 直接点图选取，或输入文字定位，或点击「自动分割」", None, None)
    except Exception:
        return (gr.update(value=None, visible=True), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(value=None, visible=False), [], [], None, None,
                [], None, False, "图片加载失败", None, None)


def on_canvas_clear_source(request: gr.Request = None):
    _clear_session_sam_contexts(request)
    return (gr.update(value=None, visible=True), gr.update(value=None, visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(value=None, visible=False),
            [], [], None, None, [], None, False, "请先上传图片", None, None)


def on_image_click(image, evt: gr.SelectData, mode, engine_mode,
                   points_state, labels_state, box_state,
                   auto_masks_state, auto_select_mode,
                   image_id=None,
                   selected_auto_indices_state=None,
                   request: gr.Request = None):
    _RET_BASE = (image, gr.update(visible=False), gr.update(value=None, visible=False),
                 points_state, labels_state, box_state, auto_select_mode)
    _ret_err = lambda msg: (*_RET_BASE, msg, None, None)
    if image is None:
        return _ret_err("请先上传图片")
    try:
        x, y = evt.index[0], evt.index[1]
    except Exception:
        return _ret_err("无法获取点击坐标")

    # ── 自动分割点选模式 ──
    if auto_select_mode and auto_masks_state:
        h, w = image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return _ret_err("点击超出图像范围")
        hit_indices = _find_masks_at(auto_masks_state, x, y)
        label_val = 1 if mode == "正向选取（我要）" else 0
        prev_indices = selected_auto_indices_state or []
        logger.info("auto-click: pos=(%d,%d), hit=%s, prev=%s, label=%d",
                    x, y, hit_indices, prev_indices, label_val)
        if not hit_indices:
            overlay = _draw_auto_segment_overlay(image, auto_masks_state,
                                                 selected_idx=prev_indices or None)
            return (overlay, gr.update(visible=False), gr.update(value=None, visible=False),
                    [], [], None, True, "未命中任何主体，请点击有色区域", None, None)
        if label_val == 1:
            new_indices = sorted(set(prev_indices) | set(hit_indices))
        else:
            new_indices = sorted(set(prev_indices) - set(hit_indices))
        if not new_indices:
            overlay = _draw_auto_segment_overlay(image, auto_masks_state)
            return (overlay, gr.update(visible=False), gr.update(value=None, visible=False),
                    [], [], None, True, "已排除所有主体，请重新选取", None, None)
        merged_seg = _merge_auto_masks(auto_masks_state, new_indices)
        # 记录本次变更到模块级历史
        added_indices = sorted(set(new_indices) - set(prev_indices))
        removed_indices = sorted(set(prev_indices) - set(new_indices))
        if added_indices:
            _push_history(auto_masks_state, added_indices, request)
        elif removed_indices:
            _push_history(auto_masks_state, (-1, removed_indices), request)
        logger.info("auto-click result: new_indices=%s, added=%s, removed=%s, area=%d",
                    new_indices, added_indices, removed_indices,
                    int(merged_seg.sum()) if merged_seg is not None else 0)
        bbox = _mask_bbox_from_seg(merged_seg)
        if bbox is None:
            return _ret_err("mask 无效")
        mask = merged_seg
        overlay = _draw_tab2_overlay(image, mask, [[x, y]], [1], [bbox])
        total = len(new_indices)
        added = len(set(hit_indices) - set(prev_indices)) if label_val == 1 else 0
        if label_val == 0:
            tag = f"已排除 {len(hit_indices)} 个主体，剩余 {total} 个"
        elif prev_indices:
            tag = f"已累积选中 {total} 个主体（+{added}）"
        elif len(hit_indices) > 1:
            tag = f"已合并 {len(hit_indices)} 个主体"
        else:
            tag = f"已选中主体 #{hit_indices[0] + 1}"
        return (overlay, gr.update(visible=True), gr.update(value=None, visible=False),
                [[x, y]], [1], None, True, f"{tag}，可继续点击添加", merged_seg,
                new_indices)

    # ── 正常点选模式 ──
    label_val = 1 if mode == "正向选取（我要）" else 0
    new_points = list(points_state) + [[x, y]]
    new_labels = list(labels_state) + [label_val]

    try:
        ctx = _ensure_sam_ready(
            image, engine_mode,
            keep_grounding_dino=True, retain=True, request=request,
            image_id=image_id,
        )
        try:
            mask = _predict_tab2_mask(ctx, new_points, new_labels, box_state)
            if label_val == 0:
                mask = _apply_object_exclusions(ctx["sam"], new_points, new_labels, mask)
        finally:
            _release_sam_context(ctx)
        overlay = _draw_tab2_overlay(
            image, mask, new_points, new_labels, box_state
        )
        tag = "正向" if label_val == 1 else "负向"
        status = f"已添加{tag}标记 ({x}, {y})，共 {len(new_points)} 个点"
        return (overlay, gr.update(visible=True), gr.update(value=None, visible=False),
                new_points, new_labels, box_state, False, status, gr.update(), gr.update())
    except Exception as e:
        return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                new_points, new_labels, box_state, False, f"预测失败: {e}", gr.update(),
                gr.update())


def on_auto_segment(image, engine_mode, points_state, labels_state, box_state,
                    image_id=None, request: gr.Request = None):
    # 防止 _SELECT_HISTORY 无限增长：保留最近 50 个 session 的历史
    if len(_SELECT_HISTORY) > 50:
        keys = list(_SELECT_HISTORY.keys())
        for old_key in keys[:-25]:
            _SELECT_HISTORY.pop(old_key, None)
    if image is None:
        return (
            None, points_state, labels_state, box_state, [],
            image_id, False, "请先上传图片", None, None,
        )
    try:
        ctx = _ensure_sam_ready(image, engine_mode, retain=True, request=request, image_id=image_id)
        try:
            masks = ctx["sam"].auto_segment(points_per_side=32)
        finally:
            _release_sam_context(ctx)
        masks = _postprocess_auto_masks(masks)
        if not masks:
            return (
                image, points_state, labels_state, box_state, [],
                image_id, False, "未检测到主体", None, None,
            )
        best_iou = float(masks[0].get("predicted_iou", 0.0))
        best_stability = float(masks[0].get("stability_score", 0.0))
        if best_iou < 0.85 or best_stability < 0.90:
            logger.warning("SAM 自动分割最高质量 mask 偏低 (iou=%.3f, stability=%.3f)，"
                           "建议手动标记精修", best_iou, best_stability)
        overlay = _draw_auto_segment_overlay(image, masks)
        return (
            overlay, points_state, labels_state, box_state, masks,
            image_id, True, f"发现 {len(masks)} 个主体，点击选择", None, None,
        )
    except Exception as e:
        return (
            image, points_state, labels_state, box_state, [],
            image_id, False, f"自动分割失败: {e}", None, None,
        )


def on_text_locate(image, caption, engine_mode, image_id=None, request: gr.Request = None):
    if image is None:
        return (None, gr.update(visible=False), gr.update(value=None, visible=False),
                [], [], None, [], image_id, False, "请先上传图片", None, None)
    if not caption or not caption.strip():
        return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                [], [], None, [], image_id, False, "请输入定位描述", None, None)

    try:
        caption_for_dino = caption.strip()
        if not caption_for_dino.endswith("."):
            caption_for_dino += "."
        boxes, scores = _mgr.grounding_dino.detect(
            image, caption=caption_for_dino,
            box_threshold=0.30, text_threshold=0.25,
            max_boxes=8, return_scores=True,
        )
        if not boxes:
            return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                    [], [], None, [], image_id, False, "未找到匹配物体", None, None)

        h, w = image.shape[:2]
        boxes, scores, filter_stats = _filter_text_candidate_boxes(
            boxes, scores, image.shape, max_prompts=8
        )
        if not boxes:
            return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                    [], [], None, [], image_id, False, "候选框过滤后为空，请换个描述", None, None)

        prompt_boxes = []
        for box_raw in boxes:
            bw, bh = box_raw[2] - box_raw[0], box_raw[3] - box_raw[1]
            margin = max(12, int(max(bw, bh) * 0.08))
            prompt_boxes.append([
                max(0, box_raw[0] - margin),
                max(0, box_raw[1] - margin),
                min(w, box_raw[2] + margin),
                min(h, box_raw[3] + margin),
            ])

        ctx = _ensure_sam_ready(
            image, engine_mode,
            keep_grounding_dino=True, retain=True, request=request,
            image_id=image_id,
        )
        try:
            with ctx["lock"]:
                mask = _predict_tab2_box_batch_initial(ctx, prompt_boxes)
        finally:
            _release_sam_context(ctx)
        overlay = _draw_tab2_overlay(image, mask, [], [], prompt_boxes)
        score_text = (
            ", ".join(f"{s:.2f}" for s in scores[:len(prompt_boxes)])
            if filter_stats["has_scores"] else "N/A"
        )
        score_note = "" if filter_stats["has_scores"] else "（无score，按模型原顺序）"
        filter_text = (
            f"；已过滤 大框{filter_stats['wrapper']} / 重复{filter_stats['duplicate']}"
            if filter_stats["wrapper"] or filter_stats["duplicate"] else ""
        )
        cap_text = f" / 截断{filter_stats['capped']}" if filter_stats["capped"] else ""
        status = (
            f"文本定位: '{caption}' → {len(prompt_boxes)} 个候选框 "
            f"score={score_text}{score_note}{filter_text}{cap_text}；可继续加正/负点修正"
        )
        return (overlay, gr.update(visible=True), gr.update(value=None, visible=False),
                [], [], prompt_boxes, [], image_id, False, status, None, None)
    except Exception as e:
        return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                [], [], None, [], image_id, False, f"定位失败: {e}", None, None)


def _run_alpha_pipeline(image, mask, points_state, labels_state, box_state,
                        mode_key, cached_logits=None):
    """统一的 mask→alpha 管线：根据 mode_key 选择 rmbg_refine 或 sam_strict。

    Returns (alpha, subject_box, roi_box, quality_notes).
    """
    if mode_key == "rmbg_refine":
        try:
            alpha, subject_box, roi_box, quality_notes = _sam_guided_rmbg_alpha(
                image, mask, points_state, labels_state, box_state,
            )
            return alpha, subject_box, roi_box, quality_notes
        except Exception as e:
            logger.warning("RMBG 精修失败 (%s)，回退到 SAM 快速模式", e)
    alpha, subject_box = _sam_strict_alpha(
        mask, points_state, labels_state, box_state, image.shape,
        logits=cached_logits,
    )
    quality_notes = ["RMBG 精修失败，已回退到 SAM 快速模式"] if mode_key == "rmbg_refine" else ["SAM fast boundary"]
    return alpha, subject_box, None, quality_notes


def on_generate_cutout(image, engine_mode, output_mode, points_state,
                       labels_state, box_state, auto_masks_state=None,
                       preserve_transparency=False,
                       save_debug=False, image_id=None,
                       selected_auto_mask=None,
                       selected_auto_indices=None,
                       request: gr.Request = None):
    empty_states = (None, None, None, [], gr.update(visible=False))
    pending_states = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=False))

    if image is None:
        yield (gr.update(), gr.update(), gr.update(value=None, visible=False), gr.update(),
               *empty_states, gr.update(), gr.update())
        return
    if (
        not points_state
        and len(_normalize_box_state(box_state)) == 0
        and selected_auto_mask is None
    ):
        yield (gr.update(), gr.update(), gr.update(value=None, visible=False),
               "请先标记区域或使用文本定位", *empty_states, gr.update(), gr.update())
        return

    try:
        # 自动分割选中的 mask：对每个主体独立处理，避免跨主体背景残留
        if selected_auto_mask is not None and selected_auto_indices and auto_masks_state:
            # 如果用户在自动分割后又打了新点，用 SAM 预测新点的 mask 并合并
            has_new_points = (
                len(points_state) > 1
                or (len(points_state) == 1 and not selected_auto_indices)
            )
            if has_new_points and box_state is None:
                yield (gr.update(), gr.update(), gr.update(value=None, visible=False),
                       "合并自动分割 + 手动标记...", *pending_states, gr.update(), gr.update())
                ctx = _ensure_sam_ready(image, engine_mode, request=request, retain=True, image_id=image_id)
                try:
                    point_mask = _predict_tab2_mask(ctx, points_state, labels_state, box_state)
                finally:
                    _release_sam_context(ctx)
                # 合并 SAM 预测与自动分割 mask
                merged = selected_auto_mask | point_mask
                mode_key = TAB2_OUTPUT_MODES.get(output_mode, "sam_strict")
                alpha, subject_box, roi_box, quality_notes = _run_alpha_pipeline(
                    image, merged, points_state, labels_state, box_state, mode_key,
                )
                if mode_key != "rmbg_refine":
                    quality_notes = ["SAM fast boundary (自动分割+手动标记合并)"]
                cached_logits = None
            else:
                # 纯自动分割，每个主体独立处理
                yield (gr.update(), gr.update(), gr.update(value=None, visible=False),
                       "使用自动分割结果生成中...", *pending_states, gr.update(), gr.update())
                mode_key = TAB2_OUTPUT_MODES.get(output_mode, "sam_strict")
                quality_notes = []
                h, w = image.shape[:2]
                combined_alpha = np.zeros((h, w), dtype=np.uint8)
                subject_boxes = []
                for idx in selected_auto_indices:
                    if idx < 0 or idx >= len(auto_masks_state):
                        continue
                    seg = np.asarray(auto_masks_state[idx]["segmentation"], dtype=bool)
                    if not seg.any():
                        continue
                    single_box = _mask_bbox_from_seg(seg)
                    if single_box is None:
                        continue
                    a, sb, _, _ = _run_alpha_pipeline(
                        image, seg, points_state, labels_state, [single_box], mode_key,
                    )
                    combined_alpha = np.maximum(combined_alpha, a)
                    subject_boxes.append(sb)
                if not subject_boxes:
                    raise ValueError("自动分割 mask 无效")
                # 合并所有主体的 bounding box
                all_y1 = min(b[1] for b in subject_boxes)
                all_x1 = min(b[0] for b in subject_boxes)
                all_y2 = max(b[3] for b in subject_boxes)
                all_x2 = max(b[2] for b in subject_boxes)
                subject_box = [all_x1, all_y1, all_x2, all_y2]
                alpha = combined_alpha
                cached_logits = None
                roi_box = None
                quality_notes.append(f"自动分割 {len(selected_auto_indices)} 个主体独立处理")
        else:
            yield (gr.update(), gr.update(), gr.update(value=None, visible=False),
                   "SAM 分割中...", *pending_states, gr.update(), gr.update())
            ctx = _ensure_sam_ready(image, engine_mode, request=request, retain=True, image_id=image_id)
            try:
                mask = _predict_tab2_mask(
                    ctx, points_state, labels_state, box_state,
                )
                cached_logits = ctx["sam"]._cached_logits
            finally:
                _release_sam_context(ctx)

            mode_key = TAB2_OUTPUT_MODES.get(output_mode, "sam_strict")
            if mode_key == "rmbg_refine":
                yield (gr.update(), gr.update(), gr.update(value=None, visible=False),
                       "RMBG-2.0 精修中...", *pending_states, gr.update(), gr.update())
            else:
                yield (gr.update(), gr.update(), gr.update(value=None, visible=False),
                       "SAM 快速硬边界导出中...", *pending_states, gr.update(), gr.update())
            alpha, subject_box, roi_box, quality_notes = _run_alpha_pipeline(
                image, mask, points_state, labels_state, box_state, mode_key,
                cached_logits=cached_logits,
            )

        output_dir = get_output_path()
        out_path = os.path.join(output_dir, "cutout.png")
        counter = 1
        while os.path.exists(out_path) or (
            save_debug and os.path.isdir(os.path.splitext(out_path)[0] + "_debug")
        ):
            out_path = os.path.join(output_dir, f"cutout_{counter}.png")
            counter += 1

        debug_dir = os.path.splitext(out_path)[0] + "_debug" if save_debug else None
        rgba = _make_rgba_result(
            image, alpha,
            debug_dir=debug_dir,
            preserve_transparency=bool(preserve_transparency),
        )
        result = Image.fromarray(rgba, "RGBA")
        result.save(out_path)

        sx1, sy1, sx2, sy2 = subject_box
        notes = list(quality_notes)
        if preserve_transparency:
            notes.append("已保护透明材质")
        if save_debug:
            notes.append(f"诊断目录: {os.path.basename(debug_dir)}")
        note_text = f"\n质量: {', '.join(sorted(set(notes)))}" if notes else ""
        roi_text = ""
        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            roi_text = f" (RMBG ROI: [{rx1},{ry1},{rx2},{ry2}])"
        yield (
            result, gr.update(visible=True),
            gr.update(value=out_path, visible=True),
            f"已完成: {os.path.basename(out_path)}\n主体区域: [{sx1},{sy1},{sx2},{sy2}]{roi_text}{note_text}",
            image, rgba, rgba, [], gr.update(visible=True), gr.update(), gr.update(),
        )
    except Exception as e:
        yield (gr.update(), gr.update(), gr.update(value=None, visible=False),
               f"生成失败: {e}", *empty_states, None, None)


def on_undo_point(image, points_state, labels_state, box_state,
                  auto_masks_state, auto_select_mode,
                  engine_mode, image_id=None,
                  selected_auto_indices_state=None,
                  request: gr.Request = None):
    """撤销最后一个标记点，重新预测 mask。"""
    if image is None:
        return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                points_state, labels_state, box_state, auto_select_mode,
                "请先上传图片", gr.update(), gr.update())

    # ── 自动分割选中模式：按历史栈撤销 ──
    step, remaining_hist = _pop_history(auto_masks_state, request)
    logger.info("on_undo_point: auto_select_mode=%s, indices=%s, step=%s, hist_remain=%d, points=%s",
                auto_select_mode, selected_auto_indices_state, step, len(remaining_hist), points_state)
    if auto_select_mode and step is not None:
        prev_indices = selected_auto_indices_state or []
        if isinstance(step, tuple) and step[0] == -1:
            restored = step[1]
            new_indices = sorted(set(prev_indices) | set(restored))
        else:
            new_indices = sorted(set(prev_indices) - set(step))
        if new_indices:
            merged_seg = _merge_auto_masks(auto_masks_state, new_indices)
            logger.info("undo auto-select: step=%s, remaining=%s, area=%d",
                        step, new_indices, int(merged_seg.sum()) if merged_seg is not None else 0)
            bbox = _mask_bbox_from_seg(merged_seg) if merged_seg is not None else None
            overlay = _draw_tab2_overlay(image, merged_seg, [], [], [bbox] if bbox else [])
            return (overlay, gr.update(visible=True), gr.update(value=None, visible=False),
                    [], [], box_state, True,
                    f"已撤销选中，剩余 {len(new_indices)} 个主体", gr.update(), new_indices)
        if auto_masks_state:
            overlay = _draw_auto_segment_overlay(image, auto_masks_state)
        else:
            overlay = image
        return (overlay, gr.update(visible=False), gr.update(value=None, visible=False),
                [], [], box_state, True, "已撤销选中，请重新点击选择主体", None, None)

    if not points_state:
        return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                points_state, labels_state, box_state, auto_select_mode,
                "没有可撤销的标记", gr.update(), gr.update())

    new_points = list(points_state[:-1])
    new_labels = list(labels_state[:-1])

    if not new_points:
        _reset_session_sam_interaction_state(request)
        if auto_masks_state:
            overlay = _draw_auto_segment_overlay(image, auto_masks_state)
        else:
            overlay = image
        return (overlay, gr.update(visible=False), gr.update(value=None, visible=False),
                [], [], box_state, False, "已撤销所有标记", None, None)

    try:
        ctx = _ensure_sam_ready(
            image, engine_mode,
            keep_grounding_dino=True, retain=True, request=request,
            image_id=image_id,
        )
        try:
            mask = _predict_tab2_mask(
                ctx, new_points, new_labels, box_state,
            )
        finally:
            _release_sam_context(ctx)

        overlay = _draw_tab2_overlay(image, mask, new_points, new_labels, box_state)

        return (overlay, gr.update(visible=True), gr.update(value=None, visible=False),
                new_points, new_labels, box_state, False,
                f"已撤销，剩余 {len(new_points)} 个标记", gr.update(), gr.update())
    except Exception as e:
        return (image, gr.update(visible=False), gr.update(value=None, visible=False),
                new_points, new_labels, box_state, False, f"撤销失败: {e}",
                gr.update(), gr.update())


def on_clear_points(image, request: gr.Request = None):
    if image is None:
        return (None, gr.update(visible=False), gr.update(value=None, visible=False),
                [], [], None, [], None, False, "", "请先上传图片", None, None)
    _reset_session_sam_interaction_state(request)
    return (None, gr.update(visible=False), gr.update(value=None, visible=False),
            [], [], None, [], None, False, "", "标记和文本定位已清除", None, None)


def on_engine_mode_change(image, request: gr.Request = None):
    _clear_session_sam_contexts(request)
    if image is None:
        return (None, gr.update(visible=False), gr.update(value=None, visible=False),
                [], [], None, [], None, False, "", "请先上传图片", None, None)
    return (None, gr.update(visible=False), gr.update(value=None, visible=False),
            [], [], None, [], None, False, "", "引擎已切换，请重新标记", None, None)


def clear_result_preview_on_start(source, result):
    """点击开始时快速清空旧预览；无原图或无旧预览则保持界面不变。"""
    if source is None or (isinstance(source, np.ndarray) and source.size == 0) or result is None:
        return gr.update(), gr.update(), gr.update()
    return None, gr.update(visible=False), gr.update(value=None, visible=False)


def should_unload_for_tab1():
    """供 Tab1 判断是否需要释放 SAM 以节省显存。"""
    return _AGGRESSIVE_UNLOAD and (not _MULTI_SESSION_MODE or free_vram_gb() < KEEP_RESIDENT_FREE_GB)
