# ── 导入和全局初始化 ─────────────────────────────────────────────
import argparse, gc, hashlib, os, warnings, signal, time, threading
from collections import OrderedDict
_STARTUP_T0 = time.perf_counter()
_STARTUP_LAST = _STARTUP_T0
_DEFAULT_WARMUP_THREAD = None


def _startup_log(stage: str):
    global _STARTUP_LAST
    if os.environ.get("MATTING_STARTUP_LOG", "1") == "0":
        return
    now = time.perf_counter()
    print(
        f"[startup] {stage}: +{now - _STARTUP_LAST:.2f}s "
        f"(total {now - _STARTUP_T0:.2f}s)"
    )
    _STARTUP_LAST = now


import cv2
_startup_log("import cv2")

warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*timm.models.*")
warnings.filterwarnings("ignore", message=".*Overwriting.*in registry.*")

from model_manager import (
    ModelManager, free_vram_gb, get_base_path, get_output_path,
    VITMATTE_VARIANTS, VITMATTE_PROCESS_MODES,
)
_startup_log("import model_manager")
from engines.rgba_postprocess import make_clean_rgba
from engines.manual_refine import refine_manual_edge
_startup_log("import rgba_postprocess")

os.environ["HF_HOME"] = os.path.join(get_base_path(), "models", "cache")
mgr = ModelManager()
_startup_log("initialize globals")


def start_default_model_warmup():
    """Warm up the default RMBG path without blocking the UI server."""
    global _DEFAULT_WARMUP_THREAD
    if os.environ.get("MATTING_PRELOAD_RMBG", "1") == "0":
        return None
    if _DEFAULT_WARMUP_THREAD is not None:
        return _DEFAULT_WARMUP_THREAD

    def _worker():
        t0 = time.perf_counter()
        try:
            mgr.preload_rmbg2()
            print(f"[startup warmup] RMBG-2.0 ready in {time.perf_counter() - t0:.2f}s")
        except Exception as exc:
            print(f"[startup warmup] RMBG-2.0 failed: {exc}")

    _DEFAULT_WARMUP_THREAD = threading.Thread(
        target=_worker, name="rmbg2-warmup", daemon=True
    )
    _DEFAULT_WARMUP_THREAD.start()
    return _DEFAULT_WARMUP_THREAD


import gradio as gr
_startup_log("import gradio")
import numpy as np
_startup_log("import numpy")
from PIL import Image
_startup_log("import PIL")
KEEP_RESIDENT_FREE_GB = 6.0
_SAM_SESSION_LOCK = threading.RLock()
_SAM_CONTEXTS = OrderedDict()
_STALE_SAM_ACTIVE = {}
_MULTI_SESSION_MODE = False
_MAX_SAM_CONTEXTS = 1
ENGINE_MODE_MAP = {
    "快速模式（MobileSAM）": "mobile_sam",
    "高精度模式（SAM-HQ）": "sam_hq",
}
TAB2_OUTPUT_MODES = {
    "SAM严格": "sam_strict",
    "RMBG精修": "rmbg_refine",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_AGGRESSIVE_UNLOAD_ENV = os.environ.get("MATTING_AGGRESSIVE_UNLOAD")
_AGGRESSIVE_UNLOAD = (
    _AGGRESSIVE_UNLOAD_ENV == "1"
    if _AGGRESSIVE_UNLOAD_ENV is not None else True
)


def _configure_runtime(multi_session: bool, max_sam_sessions: int):
    """启动期配置：默认单 session 低显存友好，多 session 保持模型常驻。"""
    global _MULTI_SESSION_MODE, _AGGRESSIVE_UNLOAD
    _MULTI_SESSION_MODE = bool(multi_session)
    if _AGGRESSIVE_UNLOAD_ENV is None:
        _AGGRESSIVE_UNLOAD = not _MULTI_SESSION_MODE
    else:
        _AGGRESSIVE_UNLOAD = _AGGRESSIVE_UNLOAD_ENV == "1"
    mgr.set_multi_session_mode(_MULTI_SESSION_MODE)
    _set_max_sam_contexts(max_sam_sessions if _MULTI_SESSION_MODE else 1)
    mode = "multi-session" if _MULTI_SESSION_MODE else "single-session"
    print(
        f"[runtime] mode={mode}, max_sam_sessions={_MAX_SAM_CONTEXTS}, "
        f"aggressive_unload={_AGGRESSIVE_UNLOAD}"
    )


def _set_max_sam_contexts(value: int):
    """限制常驻的 per-session SAM predictor 数量，超出后按 LRU 回收。"""
    global _MAX_SAM_CONTEXTS
    _MAX_SAM_CONTEXTS = max(1, int(value))
    with _SAM_SESSION_LOCK:
        _evict_sam_contexts_unlocked()


def _request_session_id(request=None):
    """Gradio 每个页面连接都有独立 session_hash；无 request 时退回单机默认。"""
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
            print(f"[session] SAM context cleanup failed: {exc}")
    if isinstance(ctx, dict):
        ctx["cleaned"] = True


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
            # 正在推理的 session 不强制清理，避免中途 reset predictor。
            _SAM_CONTEXTS[old_key] = old_ctx
            continue
        try:
            _cleanup_sam_context_unlocked(old_ctx)
        finally:
            if lock is not None:
                lock.release()
        print(f"[session] evicted SAM context: session={old_key[0]} engine={old_key[1]}")


def _get_sam_context(request, engine_type, retain=False):
    """返回当前 Gradio session + 引擎类型对应的独立 SAM predictor 状态。"""
    session_id = _request_session_id(request)
    key = (session_id, engine_type)
    with _SAM_SESSION_LOCK:
        ctx = _SAM_CONTEXTS.get(key)
        if ctx is None:
            engine = mgr.get_sam_engine(engine_type)
            ctx = {
                "session_id": session_id,
                "engine_type": engine_type,
                "sam": engine.create_session(),
                "fingerprint": None,
                "box_logits": {},
                "free_logits": {},
                "lock": threading.RLock(),
                "active": 0,
                "stale": False,
                "cleaned": False,
            }
            _SAM_CONTEXTS[key] = ctx
            _evict_sam_contexts_unlocked(keep_key=key)
        else:
            _SAM_CONTEXTS.move_to_end(key)
        if retain:
            ctx["active"] += 1
        return ctx


def _release_sam_context(ctx):
    should_cleanup = False
    unload_engine_type = None
    with _SAM_SESSION_LOCK:
        ctx["active"] = max(0, int(ctx.get("active", 0)) - 1)
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
        mgr.unload_sam_engine(unload_engine_type)


def _clear_session_sam_contexts(request=None):
    """清掉当前页面 session 的 SAM predictor/embedding，不影响其它用户。"""
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
            mgr.unload_sam_engine(engine_type)


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
        mgr.unload_sam()


def _reset_sam_interaction_state_unlocked(ctx):
    """清理单个 session 的 SAM 交互先验，保留已缓存的图像 embedding。"""
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


# ── Tab 1 后端：一键抠图 ────────────────────────────────────────
def _unload_sam_and_reset_state():
    """低显存兼容：只在显式启用 aggressive unload 时卸载所有 SAM。"""
    _clear_all_sam_contexts()


def on_auto_process(files, source_img, detect_transparent, vitmatte_variant,
                    process_mode, save_debug=False):
    """generator, yield (preview_img, status_text, result_img, result_view_btn,
    result_download_btn, original_rgb_state, auto_rgba_state, current_rgba_state,
    enter_refine_btn)"""

    # Clear old result on start
    yield (gr.update(), "开始处理...", None, gr.update(visible=False), gr.update(visible=False),
           gr.update(), gr.update(), gr.update(), gr.update(visible=False))
    if not files:
        yield (gr.update(), "请先上传图片", None, gr.update(visible=False), gr.update(visible=False),
               gr.update(), gr.update(), gr.update(), gr.update(visible=False))
        return
    if not _has_source_content(source_img):
        yield (gr.update(), "请等待原图预览加载完成", None, gr.update(visible=False), gr.update(visible=False),
               gr.update(), gr.update(), gr.update(), gr.update(visible=False))
        return

    # 映射 ViTMatte 变体
    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    # 映射推理模式
    refine_mode = VITMATTE_PROCESS_MODES.get(process_mode, "strip")

    needs_vitmatte = variant_key != "none"
    needs_dino = bool(detect_transparent)
    should_unload_unused = (
        _AGGRESSIVE_UNLOAD
        and (not _MULTI_SESSION_MODE or free_vram_gb() < KEEP_RESIDENT_FREE_GB)
    )

    # 一键抠图必走 RMBG-2.0；单 session 默认释放本次不会使用的常驻模型。
    if should_unload_unused:
        _unload_sam_and_reset_state()
        if not needs_dino:
            mgr.unload_grounding_dino()
        if not needs_vitmatte:
            mgr.unload_vitmatte()

    refiner = None
    if needs_vitmatte:
        try:
            refiner = mgr.get_vitmatte(variant_key)
        except FileNotFoundError as e:
            yield gr.update(), f"模型加载失败: {e}", gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(visible=False)
            return
        yield gr.update(), f"ViTMatte ({variant_key}) 已加载", gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(visible=False)

    # 透明物体检测器
    detector = None
    if detect_transparent:
        detector = mgr.grounding_dino
        yield gr.update(), "Grounding-DINO 已加载，开始处理...", gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(visible=False)

    output_dir = get_output_path()
    total = len(files)
    last_result = None
    last_original = None

    for idx, f in enumerate(files):
        # 跳过非图片
        ext = os.path.splitext(str(f))[-1].lower()
        if ext not in VALID_EXTS:
            continue

        fname = os.path.basename(str(f))
        # 只更新状态文字；原图/结果图保持现状，避免无谓的全图重编码重传。
        yield gr.update(), f"[{idx + 1}/{total}] 正在处理: {fname}", gr.update(), gr.update(visible=(last_result is not None)), gr.update(visible=(last_result is not None)), gr.update(), gr.update(), gr.update(), gr.update(visible=False)

        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue

        # 先确定唯一输出路径，调试目录据此对齐，避免重跑同名图覆盖旧结果
        base, ext_out = os.path.splitext(fname)
        out_path = os.path.join(output_dir, base + ".png")
        counter = 1
        while os.path.exists(out_path) or (
            save_debug and os.path.isdir(os.path.splitext(out_path)[0] + "_debug")
        ):
            out_path = os.path.join(output_dir, f"{base}_{counter}.png")
            counter += 1

        debug_dir = os.path.splitext(out_path)[0] + "_debug" if save_debug else None

        last_original = np.array(img)
        # 原图只在这里传一次。
        yield last_original, f"[{idx + 1}/{total}] RMBG-2.0 推理中: {fname}", gr.update(), gr.update(visible=(last_result is not None)), gr.update(visible=(last_result is not None)), gr.update(), gr.update(), gr.update(), gr.update(visible=False)

        result = mgr.rmbg2.remove_background(
            img,
            refiner=refiner,
            transparent_detector=detector,
            refine_mode=refine_mode,
            debug_dir=debug_dir,
        )

        result.save(out_path)

        last_result = out_path
        rgba_arr = np.array(result) if isinstance(result, Image.Image) else result
        if rgba_arr.ndim == 3 and rgba_arr.shape[2] == 4:
            yield (gr.update(), f"[{idx + 1}/{total}] 完成: {fname} → {os.path.basename(out_path)}",
                   result, gr.update(visible=True), gr.update(value=out_path, visible=True),
                   last_original, rgba_arr, rgba_arr, gr.update(visible=True))
        else:
            yield (gr.update(), f"[{idx + 1}/{total}] 完成: {fname} → {os.path.basename(out_path)}",
                   result, gr.update(visible=True), gr.update(value=out_path, visible=True),
                   gr.update(), gr.update(), gr.update(), gr.update(visible=False))

        # 清理
        del img
        if (idx + 1) % 10 == 0:
            gc.collect()

    if last_result is not None:
        done_msg = f"全部完成，共处理 {total} 张，结果保存在 output/"
        yield gr.update(), done_msg, gr.update(), gr.update(visible=True), gr.update(visible=True), gr.update(), gr.update(), gr.update(), gr.update(visible=True)
    else:
        yield gr.update(), "没有有效图片被处理", gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(visible=False)


def on_auto_upload(files):
    """上传后隐藏上传提示，显示原图预览。"""
    if not files:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), None, \
            gr.update(visible=False), gr.update(value=None, visible=False), "请先上传图片"
    first = files[0] if isinstance(files, list) else files
    try:
        img = Image.open(first).convert("RGB")
        return gr.update(visible=False), gr.update(value=np.array(img), visible=True), \
            gr.update(visible=True), gr.update(visible=True), None, \
            gr.update(visible=False), gr.update(value=None, visible=False), "图片已上传，点击开始抠图"
    except Exception:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), None, \
            gr.update(visible=False), gr.update(value=None, visible=False), "图片加载失败"


def on_auto_clear_source():
    """清空原图区：恢复上传提示，隐藏预览和清空按钮。"""
    return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
        gr.update(visible=False), gr.update(visible=False), None, \
        gr.update(visible=False), gr.update(value=None, visible=False), "请先上传图片"


def _has_source_content(source):
    """判断原图区是否已有内容，避免 numpy 数组触发布尔歧义。"""
    if source is None:
        return False
    if isinstance(source, np.ndarray):
        return source.size > 0
    if isinstance(source, (list, tuple, set)):
        return len(source) > 0
    return True


def clear_result_preview_on_start(source, result):
    """点击开始时快速清空旧预览；无原图或无旧预览则保持界面不变。"""
    if not _has_source_content(source) or result is None:
        return gr.update(), gr.update(), gr.update()
    return None, gr.update(visible=False), gr.update(value=None, visible=False)


def on_vitmatte_variant_change(vitmatte_variant):
    """直出不需要 ViTMatte 推理模式；只在精修模型启用时显示。"""
    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    return gr.update(visible=(variant_key != "none"))


# ── Tab 2 后端：精细选区 ────────────────────────────────────────
def _image_fingerprint(image):
    """生成轻量图像指纹，用于判断 SAM 缓存是否对应当前图。"""
    if image is None:
        return None
    arr = np.ascontiguousarray(image)
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return arr.shape, arr.dtype.str, digest


def _filter_tab2_box_logits(ctx, boxes, *, keep_active: bool):
    """按当前候选框过滤 box logits；keep_active=False 用于丢弃初次文本预览先验。"""
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


def _normalize_box_state(box_state):
    """把单框/多框 state 统一成 [[x1, y1, x2, y2], ...]。"""
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


def _box_state_has_boxes(box_state):
    return len(_normalize_box_state(box_state)) > 0


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
    """去掉重复框和明显的大包小 wrapper，保留可交互的多个候选。"""
    h, w = image_shape[:2]
    has_scores = scores is not None and len(scores) == len(boxes)
    if not has_scores:
        scores = [0.0] * len(boxes)
    candidates = []
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [float(v) for v in box]
        clipped = [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]
        if _box_area(clipped) >= 64:
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
    """返回多框外接框，供空 mask fallback 使用。"""
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


def _point_in_any_box(point, boxes):
    x, y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in boxes)


def _point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_cache_key(box):
    return tuple(float(v) for v in box)


def _points_for_box(points, labels, box):
    """正向点只作用于所在框；负向点全局保留用于排除误选。"""
    local_points = []
    local_labels = []
    for pt, label in zip(points, labels):
        if label == 0 or _point_in_box(pt, box):
            local_points.append(pt)
            local_labels.append(label)
    return local_points, local_labels


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


def _prune_tab2_free_logits(ctx):
    fingerprint = ctx.get("fingerprint")
    ctx["free_logits"] = {
        key: value for key, value in ctx["free_logits"].items()
        if isinstance(key, tuple) and len(key) >= 3 and key[0] == fingerprint
    }


def _predict_tab2_mask(ctx, points, labels, box_state):
    """
    支持 Grounding-DINO 多候选框：每个 box 独立跑 SAM 后合并，避免框之间互相污染。
    """
    with ctx["lock"]:
        sam = ctx["sam"]
        fingerprint = ctx.get("fingerprint")
        points = list(points or [])
        labels = list(labels or [])
        boxes = _normalize_box_state(box_state)
        if not boxes:
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

        _filter_tab2_box_logits(ctx, boxes, keep_active=True)
        _prune_tab2_free_logits(ctx)
        combined = np.zeros(sam._original_size, dtype=bool)
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
            single_box_logits = sam._prev_logits
            ctx["box_logits"][current_key] = single_box_logits

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
            # 单框仍可安全保留 SAM 自身先验；多框 union 没有单一 logits 可回喂。
            sam._prev_logits = single_box_logits
            sam._prev_npoints = len(_points_for_box(points, labels, boxes[0])[0])
        else:
            # 多框的迭代先验保存在 ctx["box_logits"] / ctx["free_logits"] 中。
            sam._prev_logits = None
            sam._prev_npoints = len(points)
        return combined


def _draw_tab2_overlay(image, mask, points, labels, box_state=None,
                       mask_color=(255, 0, 0), opacity=0.4):
    overlay = image.copy().astype(np.float32)
    color_array = np.array(mask_color, dtype=np.float32)
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask,
            overlay[:, :, c] * (1 - opacity) + color_array[c] * opacity,
            overlay[:, :, c],
        )
    img = overlay.clip(0, 255).astype(np.uint8)
    for box in _normalize_box_state(box_state):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
    for coord, label in zip(points, labels):
        x, y = int(coord[0]), int(coord[1])
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(img, (x, y), 8, color, -1)
        cv2.circle(img, (x, y), 8, (255, 255, 255), 2)
    return img


def _unload_unused_for_tab2_sam_hq(keep_grounding_dino=False):
    """SAM-HQ 低显存准备：只释放 Tab 2 后续不会马上用到的模型。"""
    if not _AGGRESSIVE_UNLOAD:
        return
    if not _MULTI_SESSION_MODE or free_vram_gb() < KEEP_RESIDENT_FREE_GB:
        # Tab 2 默认走 SAM 严格模式；只有 RMBG 精修模式才会重新加载 RMBG。
        mgr.unload_vitmatte()
        if not keep_grounding_dino:
            mgr.unload_grounding_dino()


def _ensure_sam_ready(
    image,
    engine_mode,
    keep_grounding_dino=False,
    request=None,
    retain=False,
):
    """确保当前 session 的 SAM predictor 就绪，返回其上下文。"""
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


def on_image_upload(files, request: gr.Request = None):
    """上传后隐藏上传提示，显示画布。"""
    _clear_session_sam_contexts(request)
    if not files:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            gr.update(value=None, visible=False), [], [], None, None, "请先上传图片"
    first = files[0] if isinstance(files, list) else files
    try:
        img = Image.open(first).convert("RGB")
        return gr.update(visible=False), gr.update(value=np.array(img), visible=True), \
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), \
            gr.update(value=None, visible=False), [], [], None, None, "图片已上传，点击图片选取区域或用文本定位"
    except Exception:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            gr.update(value=None, visible=False), [], [], None, None, "图片加载失败"


def on_canvas_clear_source(request: gr.Request = None):
    """清空原图区：恢复上传提示，并清空画布、结果和标记。"""
    _clear_session_sam_contexts(request)
    return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(value=None, visible=False), [], [], None, None, "请先上传图片"


def on_image_click(image, evt: gr.SelectData, mode, engine_mode, text_locate_enabled,
                   points_state, labels_state, box_state, request: gr.Request = None):
    if image is None:
        return image, gr.update(visible=False), gr.update(value=None, visible=False), points_state, labels_state, box_state, "请先上传图片"
    try:
        x, y = evt.index[0], evt.index[1]
    except Exception:
        return image, gr.update(visible=False), gr.update(value=None, visible=False), points_state, labels_state, box_state, "无法获取点击坐标"

    label = 1 if mode == "正向选取（我要）" else 0
    new_points = list(points_state) + [[x, y]]
    new_labels = list(labels_state) + [label]

    try:
        ctx = _ensure_sam_ready(
            image,
            engine_mode,
            keep_grounding_dino=bool(text_locate_enabled),
            retain=True,
            request=request,
        )
        try:
            mask = _predict_tab2_mask(ctx, new_points, new_labels, box_state)
        finally:
            _release_sam_context(ctx)
        overlay = _draw_tab2_overlay(
            image, mask, new_points, new_labels, box_state
        )
        tag = "正向" if label == 1 else "负向"
        status = f"已添加{tag}标记 ({x}, {y})，共 {len(new_points)} 个点"
        return overlay, gr.update(visible=True), gr.update(value=None, visible=False), new_points, new_labels, box_state, status
    except Exception as e:
        return image, gr.update(visible=False), gr.update(value=None, visible=False), new_points, new_labels, box_state, f"预测失败: {e}"


def on_text_locate(image, caption, engine_mode, request: gr.Request = None):
    if image is None:
        return None, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "请先上传图片"
    if not caption or not caption.strip():
        return image, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "请输入定位描述"

    try:
        caption_for_dino = caption.strip()
        if not caption_for_dino.endswith("."):
            caption_for_dino += "."
        boxes, scores = mgr.grounding_dino.detect(
            image,
            caption=caption_for_dino,
            box_threshold=0.18,
            text_threshold=0.18,
            max_boxes=16,
            return_scores=True,
        )
        if not boxes:
            return image, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "未找到匹配物体"

        h, w = image.shape[:2]
        boxes, scores, filter_stats = _filter_text_candidate_boxes(
            boxes, scores, image.shape, max_prompts=8
        )
        if not boxes:
            return image, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "候选框过滤后为空，请换个描述"

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
            image,
            engine_mode,
            keep_grounding_dino=True,
            retain=True,
            request=request,
        )
        try:
            with ctx["lock"]:
                _reset_sam_interaction_state_unlocked(ctx)
                mask = _predict_tab2_mask(ctx, [], [], prompt_boxes)
                _filter_tab2_box_logits(ctx, prompt_boxes, keep_active=False)
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
        return overlay, gr.update(visible=True), gr.update(value=None, visible=False), [], [], prompt_boxes, status
    except Exception as e:
        return image, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, f"定位失败: {e}"


def _odd_kernel(value, min_value, max_value):
    """返回 OpenCV 形态学/模糊操作需要的奇数核大小。"""
    value = int(np.clip(int(value), min_value, max_value))
    return value if value % 2 == 1 else value + 1


def _mask_bbox(mask: np.ndarray, image_shape, fallback_box=None):
    """从 SAM mask 取主体 bbox；无 mask 时退回文本定位框。"""
    h, w = image_shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        fallback = _union_boxes(fallback_box, image_shape)
        if fallback is None:
            return None
        return fallback
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _pad_bbox(box, image_shape, ratio=0.25, min_pad=96):
    """给 RMBG ROI 保留足够背景上下文，避免边界截断和指缝误判。"""
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
    """计算 bbox 到图像边界的最近距离，用于判断 ROI 是否太紧。"""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    return min(x1, y1, w - x2, h - y2)


def _roi_alpha_touches_border(alpha: np.ndarray) -> bool:
    """只有明显前景贴着 ROI 边缘时，才认为裁剪上下文不足。"""
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


def _build_sam_constraints(sam_mask: np.ndarray, bbox, image_shape):
    """
    SAM 只提供主体范围先验：原始 mask 内不压暗 RMBG，外扩软边内渐隐，
    大外扩外强制清零。
    """
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

    # Tab 2 is an interactive selection path. Keep the allow band close to SAM
    # so ROI RMBG haze/neighbor subjects do not survive as a visible extra rim.
    allow_k = _odd_kernel(base_dim * 0.045, 19, 101)
    allow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (allow_k, allow_k))
    hard_allow = cv2.dilate(sam_u8, allow_kernel) > 0

    # RMBG 高置信豁免只给 SAM 原 mask 和非常贴近边缘的窄带，
    # 避免 ROI 内第二主体绕过 soft_constraint 被带出来。
    recover_k = _odd_kernel(base_dim * 0.018, 7, 41)
    recover_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (recover_k, recover_k),
    )
    recover_allow = cv2.dilate(sam_u8, recover_kernel) > 0

    return np.clip(soft, 0.0, 1.0), hard_allow, recover_allow


def _suppress_tab2_extra_fringe(alpha: np.ndarray, sam_mask: np.ndarray, bbox) -> np.ndarray:
    """Remove low-confidence RMBG leftovers outside the selected SAM boundary."""
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

    # Low alpha around the selection is usually the "extra edge" seen in Tab 2.
    out[outside_inner & (out < 0.42)] = 0.0
    out[outside_inner & (out < 0.72)] *= 0.35
    out[far_outside & (out < 0.90)] = 0.0
    return out


def _apply_negative_points(alpha: np.ndarray, points, labels, bbox):
    """负向点优先级最高：在用户明确排除区域做局部软擦除。"""
    negative_points = [pt for pt, label in zip(points, labels) if label == 0]
    if not negative_points:
        return alpha

    x1, y1, x2, y2 = [int(v) for v in bbox]
    base_dim = max(1, min(x2 - x1, y2 - y1))
    radius = int(np.clip(base_dim * 0.025, 12, 48))
    blur = _odd_kernel(radius * 2 + 1, 25, 129)

    erase = np.zeros(alpha.shape, dtype=np.float32)
    for x, y in negative_points:
        cv2.circle(erase, (int(round(x)), int(round(y))), radius, 1.0, -1)
    erase = cv2.GaussianBlur(erase, (blur, blur), 0)
    erase = np.clip(erase, 0.0, 1.0)
    return alpha * (1.0 - erase)


def _make_rgba_result(image: np.ndarray, alpha: np.ndarray, debug_dir: str = None,
                      preserve_transparency: bool = False) -> np.ndarray:
    """生成可换任意背景的干净 RGBA，避免两侧 Tab 出口逻辑分叉。"""
    return make_clean_rgba(
        image,
        alpha,
        debug_dir=debug_dir,
        preserve_transparency=preserve_transparency,
    )


def _sam_strict_alpha(sam_mask: np.ndarray, points_state, labels_state,
                      box_state, image_shape):
    """Use SAM as the final boundary; only add a tiny anti-aliased transition."""
    subject_box = _mask_bbox(sam_mask, image_shape, fallback_box=box_state)
    if subject_box is None:
        raise ValueError("SAM 未得到有效主体区域")

    mask_u8 = (sam_mask > 0).astype(np.uint8) * 255
    x1, y1, x2, y2 = subject_box
    base_dim = max(1, min(x2 - x1, y2 - y1))

    # Close tiny SAM holes, then feather only a 1-3 px contour for anti-aliasing.
    close_k = _odd_kernel(base_dim * 0.006, 3, 11)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    blur_k = _odd_kernel(base_dim * 0.004, 3, 9)
    blurred = cv2.GaussianBlur(mask_u8.astype(np.float32), (blur_k, blur_k), 0)

    alpha = mask_u8.astype(np.float32) / 255.0
    contour = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, close_kernel) > 0
    alpha[contour] = blurred[contour] / 255.0

    alpha = _apply_negative_points(alpha, points_state, labels_state, subject_box)
    return (np.clip(alpha, 0.0, 1.0) * 255).round().astype(np.uint8), subject_box


def _sam_guided_rmbg_alpha(image: np.ndarray, sam_mask: np.ndarray,
                           points_state, labels_state, box_state):
    """商业级 Tab 2 融合：SAM 选主体，RMBG 在 ROI 内决定最终 alpha。"""
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
    roi_alpha = mgr.rmbg2.predict_alpha(roi_img, clean=True, smooth=True).astype(np.float32) / 255.0

    # 只在明确裁切到前景时扩大一次，避免为了极端情况牺牲交互速度。
    if _roi_alpha_touches_border(roi_alpha):
        if _bbox_margin_to_image_edge(roi_box, image.shape) <= 1:
            quality_notes.append("主体接近图像边缘")
        else:
            expanded_box = _pad_bbox(subject_box, image.shape, ratio=0.50, min_pad=224)
            if expanded_box != roi_box:
                ex1, ey1, ex2, ey2 = expanded_box
                expanded_img = Image.fromarray(image[ey1:ey2, ex1:ex2].astype(np.uint8), "RGB")
                roi_alpha = (
                    mgr.rmbg2.predict_alpha(expanded_img, clean=True, smooth=True).astype(np.float32) / 255.0
                )
                roi_box = expanded_box
                quality_notes.append("ROI自动扩边")

    x1, y1, x2, y2 = roi_box

    full_alpha = np.zeros((h, w), dtype=np.float32)
    full_alpha[y1:y2, x1:x2] = roi_alpha[:y2 - y1, :x2 - x1]

    soft_constraint, hard_allow, recover_allow = _build_sam_constraints(
        sam_mask,
        subject_box,
        image.shape,
    )
    final_alpha = np.minimum(full_alpha, soft_constraint)

    # 高置信 RMBG 只能在 SAM 原 mask/窄边缘带内豁免，避免抠出 ROI 内其他主体。
    high_conf = (full_alpha > 0.95) & recover_allow
    final_alpha[high_conf] = full_alpha[high_conf]
    final_alpha[~hard_allow] = 0.0
    final_alpha = _suppress_tab2_extra_fringe(final_alpha, sam_mask, subject_box)

    final_alpha = _apply_negative_points(final_alpha, points_state, labels_state, subject_box)
    final_alpha = np.clip(final_alpha, 0.0, 1.0)
    return (final_alpha * 255).round().astype(np.uint8), subject_box, roi_box, quality_notes


def on_generate_cutout(image, engine_mode, output_mode, points_state,
                       labels_state, box_state, preserve_transparency=False,
                       save_debug=False, request: gr.Request = None):
    """generator，yield (result_img, result_view_btn, result_download_btn, status_text)"""
    if image is None:
        yield gr.update(), gr.update(), gr.update(value=None, visible=False), gr.update()
        return
    if not points_state and not _box_state_has_boxes(box_state):
        yield gr.update(), gr.update(), gr.update(value=None, visible=False), "请先标记区域或用文本定位"
        return

    try:
        # SAM 分割（优先用交互 overlay 缓存的 mask，保证一致）
        yield gr.update(), gr.update(), gr.update(value=None, visible=False), "SAM 分割中..."
        ctx = _ensure_sam_ready(image, engine_mode, request=request, retain=True)
        try:
            mask = _predict_tab2_mask(ctx, points_state, labels_state, box_state)
        finally:
            _release_sam_context(ctx)

        mode_key = TAB2_OUTPUT_MODES.get(output_mode, "sam_strict")
        quality_notes = []
        roi_box = None
        if mode_key == "rmbg_refine":
            # Optional soft-alpha path: SAM selects the subject, RMBG may still alter edges.
            yield gr.update(), gr.update(), gr.update(value=None, visible=False), "RMBG-2.0 ROI 精修中..."
            alpha, subject_box, roi_box, quality_notes = _sam_guided_rmbg_alpha(
                image, mask, points_state, labels_state, box_state
            )
        else:
            yield gr.update(), gr.update(), gr.update(value=None, visible=False), "SAM 严格选区输出中..."
            alpha, subject_box = _sam_strict_alpha(
                mask, points_state, labels_state, box_state, image.shape
            )
            quality_notes.append("SAM严格边界")
        # 保存到 output/
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
            image,
            alpha,
            debug_dir=debug_dir,
            preserve_transparency=bool(preserve_transparency),
        )
        result = Image.fromarray(rgba, "RGBA")
        result.save(out_path)

        sx1, sy1, sx2, sy2 = subject_box
        notes = list(quality_notes)
        if preserve_transparency:
            notes.append("透明材质保护")
        if save_debug:
            notes.append(f"诊断目录: {os.path.basename(debug_dir)}")
        note_text = f"\n质量保护: {'、'.join(sorted(set(notes)))}" if notes else ""
        roi_text = ""
        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            roi_text = f"，RMBG扩边ROI: [{rx1},{ry1},{rx2},{ry2}]"
        yield result, gr.update(visible=True), gr.update(value=out_path, visible=True), (
            f"完成！已保存到 {os.path.basename(out_path)}\n"
            f"SAM主体框: [{sx1},{sy1},{sx2},{sy2}]{roi_text}"
            f"{note_text}"
        )
    except Exception as e:
        yield gr.update(), gr.update(), gr.update(value=None, visible=False), f"生成失败: {e}"


def on_clear_points(image, request: gr.Request = None):
    if image is None:
        return None, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "请先上传图片"
    _reset_session_sam_interaction_state(request)
    return None, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "标记和文本定位已清除"


def on_engine_mode_change(image, request: gr.Request = None):
    """切换 SAM 引擎后清空旧选区预览，避免新旧引擎结果混淆。"""
    _clear_session_sam_contexts(request)
    if image is None:
        return None, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "请先上传图片"
    return None, gr.update(visible=False), gr.update(value=None, visible=False), [], [], None, "引擎已切换，请重新标记"


# ── APP_CSS 样式 ────────────────────────────────────────────────
APP_CSS = """
:root {
    --bg: #f0f2f5;
    --ink: #1a1a2e;
    --ink-soft: #4a4a6a;
    --ink-muted: #8a8aa0;
    --line: rgba(0,0,0,0.08);
    --glass: rgba(255,255,255,0.6);
    --panel: rgba(255,255,255,0.85);
    --panel-tint: rgba(240,242,245,0.9);
    --blue: #4f8cff;
    --cyan: #36d1dc;
    --green: #34d399;
    --warn: #f59e0b;
    --radius: 18px;
    --radius-sm: 10px;
    --shadow: 0 8px 32px rgba(0,0,0,0.06);
    --button-primary-shadow: none;
    --button-primary-shadow-hover: 0 4px 16px rgba(79,140,255,0.3);
    --button-primary-shadow-active: none;
    --button-secondary-shadow: none;
    --button-secondary-shadow-hover: none;
    --button-secondary-shadow-active: none;
}

.dark {
    --bg: #0f0f1a;
    --ink: #e8e8f0;
    --ink-soft: #b0b0c8;
    --ink-muted: #6a6a80;
    --line: rgba(255,255,255,0.08);
    --glass: rgba(30,30,50,0.6);
    --panel: rgba(30,30,50,0.85);
    --panel-tint: rgba(20,20,35,0.9);
    --blue: #6a9fff;
    --cyan: #5ad8e8;
    --green: #4ade80;
    --warn: #fbbf24;
    --shadow: 0 8px 32px rgba(0,0,0,0.3);
    --button-primary-shadow: none;
    --button-primary-shadow-hover: 0 4px 16px rgba(106,159,255,0.3);
    --button-primary-shadow-active: none;
    --button-secondary-shadow: none;
    --button-secondary-shadow-hover: none;
    --button-secondary-shadow-active: none;
}

/* ── 全局 ── */
body {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: "Inter", "SF Pro Display", -apple-system, sans-serif;
}

/* ── 控制栏 ── */
.control-rail {
    background: var(--glass) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    padding: 14px !important;
    box-shadow: var(--shadow);
    max-height: calc(100vh - 60px);
    overflow-y: auto;
}
.control-rail::-webkit-scrollbar {
    width: 4px;
}
.control-rail::-webkit-scrollbar-thumb {
    background: var(--ink-muted);
    border-radius: 2px;
}

/* ── 卡片面板 ── */
.panel-card {
    background: var(--panel) !important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    padding: 12px !important;
    box-shadow: var(--shadow);
    min-width: 0 !important;
    overflow: hidden !important;
}

/* ── 标题 ── */
.section-title {
    font-size: 1.1em;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--blue), var(--cyan));
    color: #fff;
    font-size: 0.7em;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
}
.section-hint {
    font-size: 0.75em;
    color: var(--ink-muted);
    font-weight: 400;
    margin-left: 8px;
}

/* ── 上传区虚线边框 ── */
.upload-area {
    border: 2px dashed var(--ink-muted) !important;
    border-radius: var(--radius) !important;
    background: var(--panel-tint) !important;
    transition: border-color 0.2s;
}
.upload-area:hover {
    border-color: var(--blue) !important;
}

/* ── 画布区棋盘格背景 ── */
.checkerboard {
    background-image:
        linear-gradient(45deg, #ccc 25%, transparent 25%),
        linear-gradient(-45deg, #ccc 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #ccc 75%),
        linear-gradient(-45deg, transparent 75%, #ccc 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0;
    border-radius: var(--radius) !important;
    overflow: hidden;
}
.checkerboard,
.checkerboard > div,
.checkerboard .image-container,
.checkerboard [data-testid="image"],
.checkerboard [data-testid="image"] > div {
    max-width: 100% !important;
    width: 100% !important;
    min-width: 0 !important;
}
.checkerboard img,
.checkerboard canvas {
    display: block !important;
    max-width: 100% !important;
    width: 100% !important;
    height: auto !important;
    max-height: min(68vh, 760px) !important;
    object-fit: contain !important;
    border-radius: var(--radius) !important;
}
.checkerboard button[aria-label*="fullscreen" i],
.checkerboard button[aria-label*="全屏"],
.checkerboard button[aria-label*="放大"],
.checkerboard button[title*="fullscreen" i],
.checkerboard button[title*="全屏"],
.checkerboard button[title*="放大"] {
    display: none !important;
}

/* ── 浏览器级图片弹层 ── */
.image-lightbox-overlay {
    position: fixed;
    inset: 0;
    z-index: 2147483647;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 32px;
    background: rgba(7, 10, 18, 0.86);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
.image-lightbox-overlay.open {
    display: flex;
}
.image-lightbox-frame {
    max-width: min(96vw, 1600px);
    max-height: 92vh;
    border-radius: 22px;
    padding: 0;
    background-color: #f7f7fb;
    background-image:
        linear-gradient(45deg, #d7dbe4 25%, transparent 25%),
        linear-gradient(-45deg, #d7dbe4 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d7dbe4 75%),
        linear-gradient(-45deg, transparent 75%, #d7dbe4 75%);
    background-size: 24px 24px;
    background-position: 0 0, 0 12px, 12px -12px, -12px 0;
    box-shadow: 0 24px 80px rgba(0,0,0,0.38);
    overflow: hidden;
}
.image-lightbox-frame img {
    display: block;
    max-width: min(96vw, 1600px);
    max-height: 92vh;
    object-fit: contain;
    border-radius: 22px;
}
.image-lightbox-close {
    position: fixed;
    top: 22px;
    right: 24px;
    width: 42px;
    height: 42px;
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 50%;
    background: rgba(255,255,255,0.14);
    color: #fff;
    font-size: 24px;
    line-height: 1;
    cursor: pointer;
}
.image-lightbox-close:hover {
    background: rgba(255,255,255,0.24);
}
body.image-lightbox-open {
    overflow: hidden;
}
.preview-actions {
    margin-top: 8px;
    display: flex;
    justify-content: flex-end;
}
.preview-actions button,
.preview-open-btn button {
    font-size: 0.84em !important;
}

/* ── 按钮胶囊形 ── */
.btn-primary, .btn-primary button {
    background: linear-gradient(135deg, var(--blue), var(--cyan)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    transition: filter 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease !important;
}
.btn-primary:hover, .btn-primary button:hover {
    filter: brightness(1.1);
    box-shadow: var(--button-primary-shadow-hover) !important;
    transform: translateY(-1px);
}
.btn-secondary, .btn-secondary button {
    background: var(--glass) !important;
    color: var(--ink-soft) !important;
    border: 1px solid var(--line) !important;
    border-radius: 50px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
}
.btn-secondary:hover, .btn-secondary button:hover {
    background: var(--panel) !important;
    color: var(--ink) !important;
}

/* ── 文本定位区域：去掉 Gradio 默认灰底，按钮沿用主操作样式 ── */
.text-locate-panel {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
}
.text-locate-panel > div {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* ── Radio 分段选择器（iOS segmented control）── */
/* DOM: fieldset.segment-control > span(block-info) + div.wrap > label > input + span */
.segment-control {
    background: var(--panel-tint) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-sm) !important;
    padding: 3px !important;
}
.segment-control .wrap {
    display: flex !important;
    gap: 2px !important;
}
.segment-control .wrap label {
    flex: 1;
    display: flex !important;
    align-items: center;
    justify-content: center;
    text-align: center;
    border: none !important;
    border-radius: 8px !important;
    padding: 7px 4px !important;
    font-size: 0.82em !important;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--ink-soft) !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: unset !important;
}
.segment-control .wrap label:hover {
    background: var(--glass) !important;
}
.segment-control .wrap label.selected {
    background: var(--panel) !important;
    color: var(--blue) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    font-weight: 600;
}
.segment-control .wrap label input[type="radio"] {
    display: none !important;
}

/* ── 状态文本 ── */
.status-box textarea {
    background: var(--panel-tint) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-soft) !important;
    font-size: 0.85em;
    resize: none;
}

/* ── Checkbox ── */
input[type="checkbox"] {
    accent-color: var(--blue);
}

/* ── 条件隐藏 ── */
.hidden {
    display: none !important;
}

/* ── 响应式：窄屏 ── */
@media (max-width: 768px) {
    .control-rail {
        max-height: none !important;
        overflow-x: auto;
    }
}

/* ── ImageEditor 修复模式：跟 gr.Image 视觉一致 ── */
.refine-editor,
.refine-editor > div,
.refine-editor > div > div {
    background: transparent !important;
    background-color: transparent !important;
}
.refine-editor {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    overflow: hidden !important;
}
/* 画布填满组件高度 */
.refine-editor [data-testid="image"] {
    height: 100% !important;
    max-height: none !important;
}
.refine-editor canvas {
    width: 100% !important;
    max-height: 100% !important;
    object-fit: contain !important;
    border-radius: var(--radius) !important;
}
/* 工具栏不浮层 */
.refine-editor [class*="toolbar"],
.refine-editor header,
.refine-editor nav {
    position: relative !important;
    float: none !important;
    z-index: 1 !important;
    background: var(--panel) !important;
    border-radius: var(--radius-sm) !important;
    margin-bottom: 4px !important;
}
"""


APP_JS = """
(() => {
    if (window.__mattingLightboxReady) return;
    window.__mattingLightboxReady = true;

    const ensureLightbox = () => {
        let overlay = document.querySelector(".image-lightbox-overlay");
        if (overlay) return overlay;

        overlay = document.createElement("div");
        overlay.className = "image-lightbox-overlay";
        overlay.innerHTML = `
            <button class="image-lightbox-close" type="button" aria-label="关闭预览">×</button>
            <div class="image-lightbox-frame"><img alt="图片预览" /></div>
        `;
        document.body.appendChild(overlay);

        const close = () => {
            overlay.classList.remove("open");
            document.body.classList.remove("image-lightbox-open");
        };
        overlay.addEventListener("click", (event) => {
            if (
                event.target === overlay ||
                event.target.closest(".image-lightbox-close")
            ) {
                close();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && overlay.classList.contains("open")) {
                close();
            }
        });
        return overlay;
    };

    const openLightbox = (src) => {
        if (!src) return;
        const overlay = ensureLightbox();
        const img = overlay.querySelector("img");
        img.src = src;
        overlay.classList.add("open");
        document.body.classList.add("image-lightbox-open");
    };
    window.openMattingLightbox = openLightbox;

    document.addEventListener("click", (event) => {
        const trigger = event.target.closest(".preview-open-btn");
        if (!trigger) return;
        const panel =
            trigger.closest(".panel-card") ||
            trigger.parentElement?.closest(".panel-card");
        const images = Array.from(
            panel?.querySelectorAll(".checkerboard img[src]") || []
        ).filter((img) => img.currentSrc || img.src);
        const image = images[images.length - 1];
        if (!image?.src) return;

        event.preventDefault();
        event.stopPropagation();
        openLightbox(image.currentSrc || image.src);
    }, true);
})();
"""


# ── Manual Refine Helpers ──────────────────────────────────────

def _extract_user_mask(editor_value):
    """Extract painted mask from ImageEditor value. Returns HxW bool, or None."""
    if not editor_value:
        return None
    # ImageEditor returns {"background": ndarray, "layers": [...], "composite": ndarray}
    if isinstance(editor_value, dict):
        # Try layers first
        layers = editor_value.get("layers")
        if layers and len(layers) > 0:
            layer0 = layers[0]
            if isinstance(layer0, dict):
                layer0 = layer0.get("image", layer0.get("composite"))
            if layer0 is not None and hasattr(layer0, 'ndim') and layer0.ndim >= 3:
                mask = layer0[..., 3] > 30 if layer0.shape[2] == 4 else layer0[..., 0] > 200
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        # Fallback: composite vs background diff
        comp = editor_value.get("composite")
        bg = editor_value.get("background")
        if comp is not None and bg is not None:
            diff = np.abs(comp.astype(np.int16) - bg.astype(np.int16))
            mask = (diff[..., 0] > 50) & (diff[..., 1] < 30) & (diff[..., 2] < 30)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return None


def _make_editor_value(rgba):
    """Convert RGBA ndarray to ImageEditor value dict."""
    return {"background": rgba, "layers": [], "composite": rgba}


def on_enter_refine_mode(current_rgba_state):
    """Switch from preview to editor mode."""
    if current_rgba_state is None:
        return [gr.update()] * 6 + ["请先完成抠图"]
    editor_val = {"background": current_rgba_state, "layers": [], "composite": current_rgba_state}
    title = ('<div class="section-title">边缘修复 '
             '<span class="badge">画笔涂抹</span> '
             '<span class="section-hint">涂抹污染区域，涂完点应用</span></div>')
    return (
        gr.update(visible=False),       # auto_result_img
        gr.update(visible=False),       # preview_actions
        gr.update(value=editor_val, visible=True),  # auto_result_editor
        gr.update(visible=True),        # editor_actions
        gr.update(value=title),         # result_title
        gr.update(visible=False),       # enter_refine_btn
        "修复模式：涂抹绿/蓝边，涂完点「应用修复」",
    )


def on_exit_refine_mode():
    """Switch back from editor to preview mode."""
    title = ('<div class="section-title">效果预览 '
             '<span class="badge">透明背景</span></div>')
    return (
        gr.update(visible=True),        # auto_result_img
        gr.update(visible=True),        # preview_actions
        gr.update(visible=False),       # auto_result_editor
        gr.update(visible=False),       # editor_actions
        gr.update(value=title),         # result_title
        gr.update(visible=True),        # enter_refine_btn
        "",
    )


def on_apply_refine(auto_result_editor, original_rgb_state, current_rgba_state,
                    edit_history_state, vitmatte_variant):
    """Apply manual edge refinement."""
    user_mask = _extract_user_mask(auto_result_editor)
    if user_mask is None or not np.any(user_mask):
        return (auto_result_editor, current_rgba_state, edit_history_state,
                "未检测到涂抹区域，请用红色画笔涂抹")

    image_rgb = original_rgb_state
    if image_rgb is None:
        return (auto_result_editor, current_rgba_state, edit_history_state,
                "缺少原图数据")

    # Load ViTMatte
    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    if variant_key == "none":
        # Default to base for manual refine
        variant_key = "base"
    try:
        refiner = mgr.get_vitmatte(variant_key)
    except Exception as e:
        return (auto_result_editor, current_rgba_state, edit_history_state,
                f"ViTMatte 加载失败: {e}")

    try:
        rgba_out, diag = refine_manual_edge(
            image_rgb, current_rgba_state, user_mask, refiner,
        )
    except Exception as e:
        return (auto_result_editor, current_rgba_state, edit_history_state,
                f"修复失败: {e}")

    # Update history (keep last 5)
    history = list(edit_history_state or [])
    history.append(current_rgba_state.copy())
    if len(history) > 5:
        history = history[-5:]

    # Save to output
    output_dir = get_output_path()
    out_path = os.path.join(output_dir, "refined.png")
    counter = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"refined_{counter}.png")
        counter += 1
    Image.fromarray(rgba_out, "RGBA").save(out_path)

    editor_val = _make_editor_value(rgba_out)
    status_parts = []
    status_parts.append(f"accept:{diag.get('accept_pixels',0)}px")
    status_parts.append(f"α变化:{diag.get('alpha_delta_mean',0):.1f}")
    status_parts.append(f"gate:{diag.get('gate_mean',0):.2f} conf:{diag.get('rgb_conf_mean',0):.2f}")
    smooth = diag.get('edge_smooth_score', 0)
    status_parts.append(f"锯齿:{smooth:.2f}({'差' if smooth > 0.6 else '中' if smooth > 0.3 else '好'})")
    if diag.get("residue_after_by_alpha"):
        after = diag["residue_after_by_alpha"]
        before = diag.get("residue_before_by_alpha", {})
        status_parts.append(f">=240: {before.get('gte240',0):.3f}->{after.get('gte240',0):.3f}")
    status_parts.append(os.path.basename(out_path))

    return (gr.update(value=editor_val), rgba_out, history, " | ".join(status_parts))


def on_undo_refine(edit_history_state, current_rgba_state):
    """Undo last refinement."""
    history = list(edit_history_state or [])
    if not history:
        return (current_rgba_state, edit_history_state,
                _make_editor_value(current_rgba_state) if current_rgba_state is not None else gr.update(),
                "没有可撤销的修复")
    prev = history.pop()
    editor_val = _make_editor_value(prev)
    return (prev, history, gr.update(value=editor_val), "已撤销上次修复")


def on_reset_auto(auto_rgba_state):
    """Reset to auto result."""
    if auto_rgba_state is None:
        return (None, [], gr.update(), "没有自动结果可恢复")
    editor_val = _make_editor_value(auto_rgba_state)
    return (auto_rgba_state, [], gr.update(value=editor_val), "已重置到自动结果")


# ── build_ui() 构建界面 ─────────────────────────────────────────
def build_ui(model_concurrency_limit=2):
    with gr.Blocks(title="全自动抠图") as demo:
        gr.Markdown("# 全自动抠图工具")

        # ==================== Tab 1: 一键抠图 ====================
        with gr.Tab("一键抠图"):
            with gr.Row():
                # 左栏：控制面板
                with gr.Column(scale=1, elem_classes="control-rail"):
                    gr.Markdown(
                        '<div class="section-title">一键抠图 '
                        '<span class="badge">RMBG-2.0</span></div>'
                    )

                    detect_transparent = gr.Checkbox(
                        label="检测透明物体（玻璃/水滴等）",
                        value=False,
                    )
                    save_debug = gr.Checkbox(
                        label="保存诊断中间结果",
                        value=False,
                    )

                    vitmatte_variant = gr.Radio(
                        choices=list(VITMATTE_VARIANTS.keys()),
                        value="直出",
                        label="精修模型",
                        elem_classes="segment-control",
                    )
                    with gr.Group(visible=False) as process_mode_group:
                        process_mode = gr.Radio(
                            choices=list(VITMATTE_PROCESS_MODES.keys()),
                            value="条带",
                            label="推理模式",
                            elem_classes="segment-control",
                        )

                    auto_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                        lines=3,
                        elem_classes="status-box",
                    )
                    auto_btn = gr.Button(
                        "开始抠图",
                        variant="primary",
                        elem_classes="btn-primary",
                    )

                # 中栏：原图
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown(
                        '<div class="section-title">原图 '
                        '<span class="section-hint">上传后在这里确认待处理图片</span></div>'
                    )
                    auto_files = gr.File(
                        label="上传原图（支持多张）",
                        file_count="multiple",
                        file_types=["image"],
                        elem_classes="upload-area",
                    )
                    auto_input_img = gr.Image(
                        label="原图",
                        visible=False,
                        interactive=False,
                        elem_classes="checkerboard",
                    )
                    with gr.Row(elem_classes="preview-actions"):
                        auto_input_view_btn = gr.Button(
                            "查看大图",
                            visible=False,
                            elem_classes=["btn-secondary", "preview-open-btn"],
                        )
                    auto_swap_btn = gr.Button(
                        "清空原图区",
                        visible=False,
                        elem_classes="btn-secondary",
                    )

                # 右栏：效果预览 / 边缘修复（二选一，全尺寸）
                with gr.Column(scale=4, elem_classes="panel-card"):
                    # ── 共享标题栏，切换模式时更新文字 ──
                    result_title = gr.Markdown(
                        '<div class="section-title">效果预览 '
                        '<span class="badge">透明背景</span></div>'
                    )
                    # ── 预览模式组件 ──
                    auto_result_img = gr.Image(
                        label="效果预览",
                        interactive=False,
                        visible=True,
                        buttons=[],
                        elem_classes="checkerboard",
                    )
                    preview_actions = gr.Row(elem_classes="preview-actions", visible=True)
                    with preview_actions:
                        auto_result_view_btn = gr.Button(
                            "查看大图",
                            visible=False,
                            elem_classes=["btn-secondary", "preview-open-btn"],
                        )
                        auto_result_download_btn = gr.DownloadButton(
                            "下载",
                            visible=False,
                            elem_classes="btn-secondary",
                        )
                        enter_refine_btn = gr.Button(
                            "边缘修复",
                            visible=False,
                            elem_classes="btn-primary",
                        )

                    # ── 修复模式组件（默认隐藏）──
                    auto_result_editor = gr.ImageEditor(
                        label=None,
                        image_mode="RGBA",
                        type="numpy",
                        height="68vh",
                        canvas_size=(2048, 2048),
                        brush=gr.Brush(
                            default_size=20,
                            colors=["#ff0000"],
                            color_mode="fixed",
                        ),
                        eraser=gr.Eraser(default_size=20),
                        layers=True,
                        transforms=None,
                        elem_classes=["checkerboard", "refine-editor"],
                        interactive=True,
                        show_label=False,
                        visible=False,
                    )
                    editor_actions = gr.Row(elem_classes="preview-actions", visible=False)
                    with editor_actions:
                        apply_refine_btn = gr.Button(
                            "应用修复", variant="primary",
                            elem_classes="btn-primary",
                        )
                        undo_refine_btn = gr.Button(
                            "撤销",
                            elem_classes="btn-secondary",
                        )
                        reset_auto_btn = gr.Button(
                            "重置",
                            elem_classes="btn-secondary",
                        )
                        exit_refine_btn = gr.Button(
                            "退出修复",
                            elem_classes="btn-secondary",
                        )

                    # State for manual refinement
                    original_rgb_state = gr.State(None)
                    auto_rgba_state = gr.State(None)
                    current_rgba_state = gr.State(None)
                    edit_history_state = gr.State([])

        # ==================== Tab 2: 精细选区 ====================
        with gr.Tab("精细选区"):
            with gr.Row():
                # 左栏：控制面板
                with gr.Column(scale=1, elem_classes="control-rail"):
                    gr.Markdown(
                        '<div class="section-title">精细选区 '
                        '<span class="badge">SAM</span></div>'
                    )

                    click_mode = gr.Radio(
                        choices=["正向选取（我要）", "负向排除（不要）"],
                        value="正向选取（我要）",
                        label="点击模式",
                        elem_classes="segment-control",
                    )
                    engine_mode = gr.Radio(
                        choices=list(ENGINE_MODE_MAP.keys()),
                        value="快速模式（MobileSAM）",
                        label="引擎模式",
                        elem_classes="segment-control",
                    )
                    tab2_output_mode = gr.Radio(
                        choices=list(TAB2_OUTPUT_MODES.keys()),
                        value="SAM严格",
                        label="输出模式",
                        elem_classes="segment-control",
                    )

                    use_text_locate = gr.Checkbox(
                        label="启用文本定位",
                        value=False,
                    )
                    canvas_preserve_transparency = gr.Checkbox(
                        label="保护透明/半透明材质",
                        value=False,
                    )
                    canvas_save_debug = gr.Checkbox(
                        label="保存诊断中间结果",
                        value=False,
                    )

                    # 文本定位 UI（条件显示）
                    with gr.Group(
                        visible=False,
                        elem_classes="text-locate-panel",
                    ) as text_locate_group:
                        text_caption = gr.Textbox(
                            label="物体描述",
                            placeholder="例: red car, person, glass bottle",
                            lines=1,
                        )
                        locate_btn = gr.Button(
                            "用文本定位",
                            variant="primary",
                            elem_classes="btn-primary",
                        )

                    cutout_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                        lines=3,
                        elem_classes="status-box",
                    )
                    generate_btn = gr.Button(
                        "开始抠图",
                        variant="primary",
                        elem_classes="btn-primary",
                    )
                    clear_btn = gr.Button(
                        "清除标记",
                        elem_classes="btn-secondary",
                    )

                # 中栏：原图
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown(
                        '<div class="section-title">原图 '
                        '<span class="section-hint">'
                        '点击图片选取区域，绿色=正向，红色=负向</span></div>'
                    )
                    canvas_files = gr.File(
                        label="上传原图",
                        file_types=["image"],
                        elem_classes="upload-area",
                    )
                    canvas_img = gr.Image(
                        label="原图",
                        type="numpy",
                        visible=False,
                        interactive=False,
                        elem_classes="checkerboard",
                    )
                    with gr.Row(elem_classes="preview-actions"):
                        canvas_view_btn = gr.Button(
                            "查看大图",
                            visible=False,
                            elem_classes=["btn-secondary", "preview-open-btn"],
                        )
                    canvas_swap_btn = gr.Button(
                        "清空原图区",
                        visible=False,
                        elem_classes="btn-secondary",
                    )

                # 右栏：效果预览
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown(
                        '<div class="section-title">效果预览 '
                        '<span class="badge">选区结果</span></div>'
                    )
                    result_img = gr.Image(
                        label="效果预览",
                        interactive=False,
                        buttons=[],
                        elem_classes="checkerboard",
                    )
                    with gr.Row(elem_classes="preview-actions"):
                        result_view_btn = gr.Button(
                            "查看大图",
                            visible=False,
                            elem_classes=["btn-secondary", "preview-open-btn"],
                        )
                        result_download_btn = gr.DownloadButton(
                            "下载",
                            visible=False,
                            elem_classes="btn-secondary",
                        )

            # State
            points_state = gr.State([])
            labels_state = gr.State([])
            box_state = gr.State(None)

        # ==================== 事件绑定 ====================

        # --- Tab 1 ---
        auto_files.upload(
            fn=on_auto_upload,
            inputs=[auto_files],
            outputs=[auto_files, auto_input_img, auto_input_view_btn,
                     auto_swap_btn, auto_result_img, auto_result_view_btn,
                     auto_result_download_btn, auto_status],
            queue=False,
            show_progress="hidden",
        )
        auto_swap_btn.click(
            fn=on_auto_clear_source,
            outputs=[auto_files, auto_input_img, auto_input_view_btn,
                     auto_swap_btn, auto_result_img, auto_result_view_btn,
                     auto_result_download_btn, auto_status],
            queue=False,
            show_progress="hidden",
        )
        vitmatte_variant.change(
            fn=on_vitmatte_variant_change,
            inputs=[vitmatte_variant],
            outputs=[process_mode_group],
            queue=False,
            show_progress="hidden",
        )

        # Reset right column to preview mode when starting auto process
        auto_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(value='<div class="section-title">效果预览 <span class="badge">透明背景</span></div>')),
            inputs=[],
            outputs=[auto_result_img, preview_actions,
                     auto_result_editor, editor_actions, result_title],
            queue=False,
            show_progress="hidden",
        )
        auto_btn.click(
            fn=on_auto_process,
            inputs=[auto_files, auto_input_img, detect_transparent,
                    vitmatte_variant, process_mode, save_debug],
            outputs=[auto_input_img, auto_status, auto_result_img,
                     auto_result_view_btn, auto_result_download_btn,
                     original_rgb_state, auto_rgba_state, current_rgba_state,
                     enter_refine_btn],
            stream_every=0.5,
            concurrency_limit=model_concurrency_limit,
            concurrency_id="model-gpu",
        )

        enter_refine_btn.click(
            fn=on_enter_refine_mode,
            inputs=[current_rgba_state],
            outputs=[auto_result_img, preview_actions,
                     auto_result_editor, editor_actions,
                     result_title, enter_refine_btn, auto_status],
        )

        exit_refine_btn.click(
            fn=on_exit_refine_mode,
            inputs=[],
            outputs=[auto_result_img, preview_actions,
                     auto_result_editor, editor_actions,
                     result_title, enter_refine_btn, auto_status],
        )

        apply_refine_btn.click(
            fn=on_apply_refine,
            inputs=[auto_result_editor, original_rgb_state, current_rgba_state,
                    edit_history_state, vitmatte_variant],
            outputs=[auto_result_editor, current_rgba_state,
                     edit_history_state, auto_status],
            concurrency_limit=model_concurrency_limit,
            concurrency_id="model-gpu",
        )

        undo_refine_btn.click(
            fn=on_undo_refine,
            inputs=[edit_history_state, current_rgba_state],
            outputs=[current_rgba_state, edit_history_state,
                     auto_result_editor, auto_status],
        )

        reset_auto_btn.click(
            fn=on_reset_auto,
            inputs=[auto_rgba_state],
            outputs=[current_rgba_state, edit_history_state,
                     auto_result_editor, auto_status],
        )

        # --- Tab 2 ---
        use_text_locate.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[use_text_locate],
            outputs=[text_locate_group],
            queue=False,
            show_progress="hidden",
        )
        engine_mode.change(
            fn=on_engine_mode_change,
            inputs=[canvas_img],
            outputs=[result_img, result_view_btn, result_download_btn,
                     points_state, labels_state, box_state, cutout_status],
            queue=False,
            show_progress="hidden",
        )

        canvas_files.upload(
            fn=on_image_upload,
            inputs=[canvas_files],
            outputs=[canvas_files, canvas_img, canvas_view_btn,
                     canvas_swap_btn, result_view_btn, result_download_btn,
                     points_state, labels_state, box_state, result_img,
                     cutout_status],
            queue=False,
            show_progress="hidden",
        )
        canvas_swap_btn.click(
            fn=on_canvas_clear_source,
            outputs=[canvas_files, canvas_img, canvas_view_btn,
                     canvas_swap_btn, result_view_btn, result_download_btn,
                     points_state, labels_state, box_state, result_img,
                     cutout_status],
            queue=False,
            show_progress="hidden",
        )

        locate_btn.click(
            fn=on_text_locate,
            inputs=[canvas_img, text_caption, engine_mode],
            outputs=[result_img, result_view_btn, result_download_btn,
                     points_state, labels_state, box_state, cutout_status],
            concurrency_limit=model_concurrency_limit,
            concurrency_id="model-gpu",
        )

        canvas_img.select(
            fn=on_image_click,
            inputs=[canvas_img, click_mode, engine_mode, use_text_locate,
                    points_state, labels_state, box_state],
            outputs=[result_img, result_view_btn, result_download_btn,
                     points_state, labels_state, box_state, cutout_status],
            concurrency_limit=model_concurrency_limit,
            concurrency_id="model-gpu",
        )

        generate_btn.click(
            fn=clear_result_preview_on_start,
            inputs=[canvas_img, result_img],
            outputs=[result_img, result_view_btn, result_download_btn],
            queue=False,
            show_progress="hidden",
        )
        generate_btn.click(
            fn=on_generate_cutout,
            inputs=[canvas_img, engine_mode, tab2_output_mode,
                    points_state, labels_state, box_state,
                    canvas_preserve_transparency, canvas_save_debug],
            outputs=[result_img, result_view_btn, result_download_btn,
                     cutout_status],
            stream_every=0.5,
            concurrency_limit=model_concurrency_limit,
            concurrency_id="model-gpu",
        )

        clear_btn.click(
            fn=on_clear_points,
            inputs=[canvas_img],
            outputs=[result_img, result_view_btn, result_download_btn,
                     points_state, labels_state, box_state, cutout_status],
            queue=False,
            show_progress="hidden",
        )

    return demo


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="AI 抠图工具 Web UI")
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=18181,
        help="监听端口（默认 18181）",
    )
    parser.add_argument(
        "-q", "--silent",
        action="store_true",
        help="静默启动：不自动打开浏览器，并减少控制台输出",
    )
    parser.add_argument(
        "--model-concurrency",
        type=int,
        default=None,
        help="模型推理并发数；默认单 session 为 1，多 session 为 2",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=32,
        help="等待队列最大长度（默认 32）",
    )
    parser.add_argument(
        "--max-sam-sessions",
        type=int,
        default=8,
        help="多 session 时最多保留的独立 SAM 会话状态数量（默认 8）",
    )
    parser.add_argument(
        "--multi-session",
        action="store_true",
        help="启用多人/多标签页隔离和模型常驻缓存；默认单 session 低显存模式",
    )
    return parser.parse_args()


# ── __main__ 启动 ───────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_cli_args()
    _startup_log("parse args")
    _configure_runtime(args.multi_session, args.max_sam_sessions)
    _startup_log("configure runtime")
    model_concurrency = (
        args.model_concurrency
        if args.model_concurrency is not None
        else (2 if args.multi_session else 1)
    )
    start_default_model_warmup()
    _startup_log("start warmup thread")
    demo = build_ui(model_concurrency_limit=max(1, model_concurrency))
    _startup_log("warmup overlap checkpoint")
    _startup_log("build ui")
    demo.queue(
        default_concurrency_limit=max(1, model_concurrency),
        max_size=max(1, args.queue_size),
    )
    _startup_log("configure queue")
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=not args.silent,
        quiet=args.silent,
        share=False,
        allowed_paths=[get_output_path()],
        prevent_thread_lock=True,
        theme=gr.themes.Soft(),
        css=APP_CSS,
        js=APP_JS,
    )
    _startup_log("launch gradio")


    # 信号处理
    def _force_exit(*_):
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    # 主线程阻塞
    while True:
        time.sleep(1)
