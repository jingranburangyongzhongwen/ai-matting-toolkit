# ── 导入和全局初始化 ─────────────────────────────────────────────
import gc, hashlib, os, warnings, signal, time, threading
import cv2
import gradio as gr
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*timm.models.*")
warnings.filterwarnings("ignore", message=".*Overwriting.*in registry.*")

from model_manager import (
    ModelManager, free_vram_gb, get_base_path, get_output_path,
    VITMATTE_VARIANTS, VITMATTE_PROCESS_MODES,
)
from engines.rgba_postprocess import make_clean_rgba

os.environ["HF_HOME"] = os.path.join(get_base_path(), "models", "cache")
mgr = ModelManager()
KEEP_RESIDENT_FREE_GB = 6.0
_SAM_IMAGE_FINGERPRINT = None
ENGINE_MODE_MAP = {
    "快速模式（MobileSAM）": "mobile_sam",
    "高精度模式（SAM-HQ）": "sam_hq",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def start_default_model_warmup():
    """Warm up the default RMBG path without blocking the UI server."""
    if os.environ.get("MATTING_PRELOAD_RMBG", "1") == "0":
        return None

    def _worker():
        t0 = time.perf_counter()
        try:
            mgr.preload_rmbg2()
            print(f"[startup warmup] RMBG-2.0 ready in {time.perf_counter() - t0:.2f}s")
        except Exception as exc:
            print(f"[startup warmup] RMBG-2.0 failed: {exc}")

    thread = threading.Thread(target=_worker, name="rmbg2-warmup", daemon=True)
    thread.start()
    return thread


# ── Tab 1 后端：一键抠图 ────────────────────────────────────────
def _unload_sam_and_reset_state():
    """卸载 SAM 并同步清空本模块维护的图像指纹。"""
    global _SAM_IMAGE_FINGERPRINT
    mgr.unload_sam()
    _SAM_IMAGE_FINGERPRINT = None


def on_auto_process(files, source_img, detect_transparent, vitmatte_variant,
                    process_mode, save_debug=False):
    """generator，yield (preview_img, status_text, result_img, result_view_btn)"""
    if not files:
        # 原图区没有内容时完全不改动界面。
        yield gr.update(), gr.update(), gr.update(), gr.update()
        return
    if not _has_source_content(source_img):
        yield gr.update(), "请等待原图预览加载完成", gr.update(), gr.update()
        return

    # 映射 ViTMatte 变体
    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    # 映射推理模式
    refine_mode = VITMATTE_PROCESS_MODES.get(process_mode, "strip")

    needs_vitmatte = variant_key != "none"
    needs_dino = bool(detect_transparent)
    low_vram = free_vram_gb() < KEEP_RESIDENT_FREE_GB

    # 一键抠图必走 RMBG-2.0；低显存时只释放本次不会使用的常驻模型。
    if low_vram:
        _unload_sam_and_reset_state()
        if not needs_dino:
            mgr.unload_grounding_dino()
        if not needs_vitmatte:
            mgr.unload_vitmatte()

    refiner = None
    if needs_vitmatte:
        try:
            mgr.switch_vitmatte(variant_key)
            refiner = mgr.vitmatte
        except FileNotFoundError as e:
            yield gr.update(), f"模型加载失败: {e}", gr.update(), gr.update(visible=False)
            return
        yield gr.update(), f"ViTMatte ({variant_key}) 已加载", gr.update(), gr.update(visible=False)

    # 透明物体检测器
    detector = None
    if detect_transparent:
        detector = mgr.grounding_dino
        yield gr.update(), "Grounding-DINO 已加载，开始处理...", gr.update(), gr.update(visible=False)

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
        yield gr.update(), f"[{idx + 1}/{total}] 正在处理: {fname}", gr.update(), gr.update(visible=(last_result is not None))

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
        yield last_original, f"[{idx + 1}/{total}] RMBG-2.0 推理中: {fname}", gr.update(), gr.update(visible=(last_result is not None))

        result = mgr.rmbg2.remove_background(
            img,
            refiner=refiner,
            transparent_detector=detector,
            refine_mode=refine_mode,
            debug_dir=debug_dir,
        )

        result.save(out_path)

        last_result = result
        # 结果图只在这里传一次；原图不重传。
        yield gr.update(), f"[{idx + 1}/{total}] 完成: {fname} → {os.path.basename(out_path)}", result, gr.update(visible=True)

        # 清理
        del img
        if (idx + 1) % 10 == 0:
            gc.collect()

    if last_result is not None:
        done_msg = f"全部完成，共处理 {total} 张，结果保存在 output/"
        # 汇总只更新状态文字；图片已展示，不再重传。
        yield gr.update(), done_msg, gr.update(), gr.update(visible=True)
    else:
        yield gr.update(), "没有有效图片被处理", gr.update(), gr.update(visible=False)


def on_auto_upload(files):
    """上传后隐藏上传提示，显示原图预览。"""
    if not files:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), None, \
            gr.update(visible=False), "请先上传图片"
    first = files[0] if isinstance(files, list) else files
    try:
        img = Image.open(first).convert("RGB")
        return gr.update(visible=False), gr.update(value=np.array(img), visible=True), \
            gr.update(visible=True), gr.update(visible=True), None, \
            gr.update(visible=False), "图片已上传，点击开始抠图"
    except Exception:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), None, \
            gr.update(visible=False), "图片加载失败"


def on_auto_clear_source():
    """清空原图区：恢复上传提示，隐藏预览和清空按钮。"""
    return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
        gr.update(visible=False), gr.update(visible=False), None, \
        gr.update(visible=False), "请先上传图片"


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
        return gr.update(), gr.update()
    return None, gr.update(visible=False)


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


def _reset_sam_interaction_state():
    """清理 SAM 的交互先验，避免旧 mask/logits 污染下一次标记。"""
    if mgr._sam_engine is None:
        return
    mgr.sam._prev_logits = None
    mgr.sam._prev_npoints = 0
    mgr.sam._cached_mask = None


def _unload_unused_for_tab2_sam_hq(keep_grounding_dino=False):
    """SAM-HQ 低显存准备：只释放 Tab 2 后续不会马上用到的模型。"""
    if free_vram_gb() >= KEEP_RESIDENT_FREE_GB:
        return

    # Tab 2 最终生成必走 RMBG-2.0，不能在这里卸载后又马上重载。
    mgr.unload_vitmatte()
    if not keep_grounding_dino:
        mgr.unload_grounding_dino()


def _ensure_sam_ready(image, engine_mode, keep_grounding_dino=False):
    """确保 SAM 引擎就绪，返回是否切换了引擎"""
    global _SAM_IMAGE_FINGERPRINT
    engine_type = ENGINE_MODE_MAP.get(engine_mode, "mobile_sam")
    if engine_type == "sam_hq":
        _unload_unused_for_tab2_sam_hq(
            keep_grounding_dino=keep_grounding_dino
        )
    switched = mgr.switch_sam(engine_type)
    fingerprint = _image_fingerprint(image)
    if switched or not mgr.sam._image_set or fingerprint != _SAM_IMAGE_FINGERPRINT:
        mgr.sam.set_image(image)
        _SAM_IMAGE_FINGERPRINT = fingerprint
        return True
    return False


def on_image_upload(files):
    """上传后隐藏上传提示，显示画布。"""
    global _SAM_IMAGE_FINGERPRINT
    _SAM_IMAGE_FINGERPRINT = None
    _reset_sam_interaction_state()
    if not files:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            [], [], None, None, "请先上传图片"
    first = files[0] if isinstance(files, list) else files
    try:
        img = Image.open(first).convert("RGB")
        return gr.update(visible=False), gr.update(value=np.array(img), visible=True), \
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), \
            [], [], None, None, "图片已上传，点击图片选取区域或用文本定位"
    except Exception:
        return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            [], [], None, None, "图片加载失败"


def on_canvas_clear_source():
    """清空原图区：恢复上传提示，并清空画布、结果和标记。"""
    global _SAM_IMAGE_FINGERPRINT
    _SAM_IMAGE_FINGERPRINT = None
    _reset_sam_interaction_state()
    return gr.update(value=None, visible=True), gr.update(value=None, visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        [], [], None, None, "请先上传图片"


def on_image_click(image, evt: gr.SelectData, mode, engine_mode, text_locate_enabled,
                   points_state, labels_state, box_state):
    if image is None:
        return image, gr.update(visible=False), points_state, labels_state, box_state, "请先上传图片"
    try:
        x, y = evt.index[0], evt.index[1]
    except Exception:
        return image, gr.update(visible=False), points_state, labels_state, box_state, "无法获取点击坐标"

    label = 1 if mode == "正向选取（我要）" else 0
    new_points = list(points_state) + [[x, y]]
    new_labels = list(labels_state) + [label]

    try:
        _ensure_sam_ready(
            image,
            engine_mode,
            keep_grounding_dino=bool(text_locate_enabled),
        )
        overlay = mgr.sam.predict_and_overlay(
            image, new_points, new_labels, box=box_state
        )
        tag = "正向" if label == 1 else "负向"
        status = f"已添加{tag}标记 ({x}, {y})，共 {len(new_points)} 个点"
        return overlay, gr.update(visible=True), new_points, new_labels, box_state, status
    except Exception as e:
        return image, gr.update(visible=False), new_points, new_labels, box_state, f"预测失败: {e}"


def on_text_locate(image, caption, engine_mode):
    if image is None:
        return None, gr.update(visible=False), [], [], None, "请先上传图片"
    if not caption or not caption.strip():
        return image, gr.update(visible=False), [], [], None, "请输入定位描述"

    try:
        _ensure_sam_ready(image, engine_mode, keep_grounding_dino=True)
        _reset_sam_interaction_state()
        caption_for_dino = caption.strip()
        if not caption_for_dino.endswith("."):
            caption_for_dino += "."
        boxes = mgr.grounding_dino.detect(image, caption=caption_for_dino)
        if not boxes:
            return image, gr.update(visible=False), [], [], None, "未找到匹配物体"

        # 多个框合并为外接矩形，加 margin 给 SAM 足够上下文
        xs = []
        ys = []
        for b in boxes:
            xs.extend([b[0], b[2]])
            ys.extend([b[1], b[3]])
        h, w = image.shape[:2]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        margin = max(30, int(max(bw, bh) * 0.5))
        box = [
            max(0, min(xs) - margin),
            max(0, min(ys) - margin),
            min(w, max(xs) + margin),
            min(h, max(ys) + margin),
        ]

        overlay = mgr.sam.predict_and_overlay(image, [], [], box=box)
        status = f"文本定位: '{caption}' → 框 [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]"
        return overlay, gr.update(visible=True), [], [], box, status
    except Exception as e:
        return image, gr.update(visible=False), [], [], None, f"定位失败: {e}"


def _odd_kernel(value, min_value, max_value):
    """返回 OpenCV 形态学/模糊操作需要的奇数核大小。"""
    value = int(np.clip(int(value), min_value, max_value))
    return value if value % 2 == 1 else value + 1


def _mask_bbox(mask: np.ndarray, image_shape, fallback_box=None):
    """从 SAM mask 取主体 bbox；无 mask 时退回文本定位框。"""
    h, w = image_shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        if fallback_box is None:
            return None
        x1, y1, x2, y2 = fallback_box
        return [max(0, int(x1)), max(0, int(y1)),
                min(w, int(x2)), min(h, int(y2))]
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

    allow_k = _odd_kernel(base_dim * 0.08, 31, 181)
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

    final_alpha = _apply_negative_points(final_alpha, points_state, labels_state, subject_box)
    final_alpha = np.clip(final_alpha, 0.0, 1.0)
    return (final_alpha * 255).round().astype(np.uint8), subject_box, roi_box, quality_notes


def on_generate_cutout(image, engine_mode, points_state,
                       labels_state, box_state, preserve_transparency=False,
                       save_debug=False):
    """generator，yield (result_img, result_view_btn, status_text)"""
    if image is None:
        yield gr.update(), gr.update(), gr.update()
        return
    if not points_state and box_state is None:
        yield gr.update(), gr.update(), "请先标记区域或用文本定位"
        return

    try:
        # SAM 分割（优先用交互 overlay 缓存的 mask，保证一致）
        yield gr.update(), gr.update(), "SAM 分割中..."
        _ensure_sam_ready(image, engine_mode)
        if mgr.sam._cached_mask is not None:
            mask = mgr.sam._cached_mask
        else:
            mask = mgr.sam.predict_mask(points_state, labels_state, box=box_state)

        # 商业级融合：SAM 负责主体先验，RMBG 在扩边 ROI 内重新判断精细 alpha。
        yield gr.update(), gr.update(), "RMBG-2.0 ROI 精修中..."
        alpha, subject_box, roi_box, quality_notes = _sam_guided_rmbg_alpha(
            image, mask, points_state, labels_state, box_state
        )
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
        rx1, ry1, rx2, ry2 = roi_box
        notes = list(quality_notes)
        if preserve_transparency:
            notes.append("透明材质保护")
        if save_debug:
            notes.append(f"诊断目录: {os.path.basename(debug_dir)}")
        note_text = f"\n质量保护: {'、'.join(sorted(set(notes)))}" if notes else ""
        yield result, gr.update(visible=True), (
            f"完成！已保存到 {os.path.basename(out_path)}\n"
            f"SAM主体框: [{sx1},{sy1},{sx2},{sy2}]，RMBG扩边ROI: [{rx1},{ry1},{rx2},{ry2}]"
            f"{note_text}"
        )
    except Exception as e:
        yield gr.update(), gr.update(), f"生成失败: {e}"


def on_clear_points(image):
    if image is None:
        return None, gr.update(visible=False), [], [], None, "请先上传图片"
    _reset_sam_interaction_state()
    return None, gr.update(visible=False), [], [], None, "标记和文本定位已清除"


def on_engine_mode_change(image):
    """切换 SAM 引擎后清空旧选区预览，避免新旧引擎结果混淆。"""
    global _SAM_IMAGE_FINGERPRINT
    _SAM_IMAGE_FINGERPRINT = None
    _reset_sam_interaction_state()
    if image is None:
        return None, gr.update(visible=False), [], [], None, "请先上传图片"
    return None, gr.update(visible=False), [], [], None, "引擎已切换，请重新标记"


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


# ── build_ui() 构建界面 ─────────────────────────────────────────
def build_ui():
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

                # 右栏：效果预览
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown(
                        '<div class="section-title">效果预览 '
                        '<span class="badge">透明背景</span></div>'
                    )
                    auto_result_img = gr.Image(
                        label="效果预览",
                        interactive=False,
                        elem_classes="checkerboard",
                    )
                    with gr.Row(elem_classes="preview-actions"):
                        auto_result_view_btn = gr.Button(
                            "查看大图",
                            visible=False,
                            elem_classes=["btn-secondary", "preview-open-btn"],
                        )

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
                        elem_classes="checkerboard",
                    )
                    with gr.Row(elem_classes="preview-actions"):
                        result_view_btn = gr.Button(
                            "查看大图",
                            visible=False,
                            elem_classes=["btn-secondary", "preview-open-btn"],
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
                     auto_status],
            queue=False,
            show_progress="hidden",
        )
        auto_swap_btn.click(
            fn=on_auto_clear_source,
            outputs=[auto_files, auto_input_img, auto_input_view_btn,
                     auto_swap_btn, auto_result_img, auto_result_view_btn,
                     auto_status],
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

        auto_btn.click(
            fn=clear_result_preview_on_start,
            inputs=[auto_input_img, auto_result_img],
            outputs=[auto_result_img, auto_result_view_btn],
            queue=False,
            show_progress="hidden",
        )
        auto_btn.click(
            fn=on_auto_process,
            inputs=[auto_files, auto_input_img, detect_transparent,
                    vitmatte_variant, process_mode, save_debug],
            outputs=[auto_input_img, auto_status, auto_result_img,
                     auto_result_view_btn],
            stream_every=0.5,
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
            outputs=[result_img, result_view_btn, points_state, labels_state,
                     box_state, cutout_status],
            queue=False,
            show_progress="hidden",
        )

        canvas_files.upload(
            fn=on_image_upload,
            inputs=[canvas_files],
            outputs=[canvas_files, canvas_img, canvas_view_btn,
                     canvas_swap_btn, result_view_btn, points_state,
                     labels_state, box_state, result_img, cutout_status],
            queue=False,
            show_progress="hidden",
        )
        canvas_swap_btn.click(
            fn=on_canvas_clear_source,
            outputs=[canvas_files, canvas_img, canvas_view_btn,
                     canvas_swap_btn, result_view_btn, points_state,
                     labels_state, box_state, result_img, cutout_status],
            queue=False,
            show_progress="hidden",
        )

        locate_btn.click(
            fn=on_text_locate,
            inputs=[canvas_img, text_caption, engine_mode],
            outputs=[result_img, result_view_btn, points_state, labels_state,
                     box_state, cutout_status],
        )

        canvas_img.select(
            fn=on_image_click,
            inputs=[canvas_img, click_mode, engine_mode, use_text_locate,
                    points_state, labels_state, box_state],
            outputs=[result_img, result_view_btn, points_state, labels_state,
                     box_state, cutout_status],
        )

        generate_btn.click(
            fn=clear_result_preview_on_start,
            inputs=[canvas_img, result_img],
            outputs=[result_img, result_view_btn],
            queue=False,
            show_progress="hidden",
        )
        generate_btn.click(
            fn=on_generate_cutout,
            inputs=[canvas_img, engine_mode,
                    points_state, labels_state, box_state,
                    canvas_preserve_transparency, canvas_save_debug],
            outputs=[result_img, result_view_btn, cutout_status],
            stream_every=0.5,
        )

        clear_btn.click(
            fn=on_clear_points,
            inputs=[canvas_img],
            outputs=[result_img, result_view_btn, points_state, labels_state,
                     box_state, cutout_status],
            queue=False,
            show_progress="hidden",
        )

    return demo


# ── __main__ 启动 ───────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        prevent_thread_lock=True,
        theme=gr.themes.Soft(),
        css=APP_CSS,
        js=APP_JS,
    )
    start_default_model_warmup()


    # 信号处理
    def _force_exit(*_):
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    # 主线程阻塞
    while True:
        time.sleep(1)
