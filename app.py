# ── 导入和全局初始化 ─────────────────────────────────────────────
import gc, os, warnings, signal, time
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

os.environ["HF_HOME"] = os.path.join(get_base_path(), "models", "cache")
mgr = ModelManager()
KEEP_RESIDENT_FREE_GB = 6.0
ENGINE_MODE_MAP = {
    "快速模式（MobileSAM）": "mobile_sam",
    "高精度模式（SAM-HQ）": "sam_hq",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ── Tab 1 后端：一键抠图 ────────────────────────────────────────
def on_auto_process(files, detect_transparent, vitmatte_variant, process_mode,
                    save_debug=False):
    """generator，yield (preview_img, status_text, result_img)"""
    if not files:
        yield None, "请先上传图片", None
        return

    # 显存不足时卸载 SAM 和（非透明模式下）Grounding-DINO
    if free_vram_gb() < KEEP_RESIDENT_FREE_GB:
        mgr.unload_sam()
        if not detect_transparent:
            mgr.unload_grounding_dino()

    # 映射 ViTMatte 变体
    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    # 映射推理模式
    refine_mode = VITMATTE_PROCESS_MODES.get(process_mode, "strip")

    refiner = None
    if variant_key != "none":
        try:
            mgr.switch_vitmatte(variant_key)
            refiner = mgr.vitmatte
        except FileNotFoundError as e:
            yield None, f"模型加载失败: {e}", None
            return
        yield None, f"ViTMatte ({variant_key}) 已加载", None

    # 透明物体检测器
    detector = None
    if detect_transparent:
        detector = mgr.grounding_dino
        yield None, "Grounding-DINO 已加载，开始处理...", None

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
        yield None, f"[{idx + 1}/{total}] 正在处理: {fname}", last_result

        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue

        # 调试目录
        debug_dir = None
        if save_debug:
            debug_dir = os.path.join(output_dir, os.path.splitext(fname)[0] + "_debug")

        last_original = np.array(img)
        yield last_original, f"[{idx + 1}/{total}] RMBG-2.0 推理中: {fname}", last_result

        result = mgr.rmbg2.remove_background(
            img,
            refiner=refiner,
            transparent_detector=detector,
            refine_mode=refine_mode,
            debug_dir=debug_dir,
        )

        # 保存到 output/，自动加后缀避免覆盖
        base, ext_out = os.path.splitext(fname)
        out_path = os.path.join(output_dir, base + ".png")
        counter = 1
        while os.path.exists(out_path):
            out_path = os.path.join(output_dir, f"{base}_{counter}.png")
            counter += 1
        result.save(out_path)

        last_result = result
        yield np.array(img), f"[{idx + 1}/{total}] 完成: {fname} → {os.path.basename(out_path)}", result

        # 清理
        del img
        if (idx + 1) % 10 == 0:
            gc.collect()

    if last_result is not None:
        done_msg = f"全部完成，共处理 {total} 张，结果保存在 output/"
        yield last_original, done_msg, last_result
    else:
        yield last_original, "没有有效图片被处理", None


def on_auto_upload(files):
    """隐藏上传区，显示预览图"""
    if not files:
        return gr.update(visible=False), gr.update(), gr.update()
    first = files[0] if isinstance(files, list) else files
    try:
        img = Image.open(first).convert("RGB")
        return gr.update(visible=False), np.array(img), gr.update(visible=True)
    except Exception:
        return gr.update(visible=False), None, gr.update(visible=True)


def on_auto_swap():
    """换图：显示上传区，隐藏预览和换图按钮"""
    return gr.update(visible=True), None, gr.update(visible=False)


# ── Tab 2 后端：精细选区 ────────────────────────────────────────
def _ensure_sam_ready(image, engine_mode):
    """确保 SAM 引擎就绪，返回是否切换了引擎"""
    engine_type = ENGINE_MODE_MAP.get(engine_mode, "mobile_sam")
    switched = mgr.switch_sam(engine_type)
    if switched or not mgr.sam._image_set:
        mgr.sam.set_image(image)
        return True
    return False


def on_image_upload(files):
    """隐藏上传区，显示画布"""
    if not files:
        return gr.update(visible=False), gr.update(), gr.update(), [], None, "请先上传图片"
    first = files[0] if isinstance(files, list) else files
    try:
        img = Image.open(first).convert("RGB")
        return gr.update(visible=False), np.array(img), gr.update(visible=True), \
            [], None, "图片已上传，点击图片选取区域或用文本定位"
    except Exception:
        return gr.update(visible=False), None, gr.update(visible=True), \
            [], None, "图片加载失败"


def on_canvas_swap():
    """换图：显示上传区，清空画布和标记"""
    return gr.update(visible=True), None, gr.update(visible=False), [], [], None


def on_image_click(image, evt: gr.SelectData, mode, engine_mode,
                   points_state, labels_state, box_state):
    if image is None:
        return image, points_state, labels_state, box_state, "请先上传图片"
    try:
        x, y = evt.index[0], evt.index[1]
    except Exception:
        return image, points_state, labels_state, box_state, "无法获取点击坐标"

    label = 1 if mode == "正向选取（我要）" else 0
    new_points = list(points_state) + [[x, y]]
    new_labels = list(labels_state) + [label]

    try:
        _ensure_sam_ready(image, engine_mode)
        overlay = mgr.sam.predict_and_overlay(
            image, new_points, new_labels, box=box_state
        )
        tag = "正向" if label == 1 else "负向"
        status = f"已添加{tag}标记 ({x}, {y})，共 {len(new_points)} 个点"
        return overlay, new_points, new_labels, box_state, status
    except Exception as e:
        return image, new_points, new_labels, box_state, f"预测失败: {e}"


def on_text_locate(image, caption, engine_mode):
    if image is None:
        return None, [], [], None, "请先上传图片"
    if not caption or not caption.strip():
        return image, [], [], None, "请输入定位描述"

    try:
        _ensure_sam_ready(image, engine_mode)
        caption_for_dino = caption.strip()
        if not caption_for_dino.endswith("."):
            caption_for_dino += "."
        boxes = mgr.grounding_dino.detect(image, caption=caption_for_dino)
        if not boxes:
            return image, [], [], None, "未找到匹配物体"

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
        return overlay, [], [], box, status
    except Exception as e:
        return image, [], [], None, f"定位失败: {e}"


def on_generate_cutout(image, engine_mode, points_state,
                       labels_state, box_state):
    """generator，yield (result_img, status_text)"""
    if image is None or (not points_state and box_state is None):
        yield None, "请先上传图片并标记区域"
        return

    try:
        # SAM 分割（优先用交互 overlay 缓存的 mask，保证一致）
        yield gr.update(), "SAM 分割中..."
        _ensure_sam_ready(image, engine_mode)
        if mgr.sam._cached_mask is not None:
            mask = mgr.sam._cached_mask
        else:
            mask = mgr.sam.predict_mask(points_state, labels_state, box=box_state)
        alpha = (mask.astype(np.uint8) * 255)
        rgba = np.dstack([image, alpha])
        result = Image.fromarray(rgba, "RGBA")

        # 保存到 output/
        output_dir = get_output_path()
        out_path = os.path.join(output_dir, "cutout.png")
        counter = 1
        while os.path.exists(out_path):
            out_path = os.path.join(output_dir, f"cutout_{counter}.png")
            counter += 1
        result.save(out_path)

        yield result, f"完成！已保存到 {os.path.basename(out_path)}"
    except Exception as e:
        yield None, f"生成失败: {e}"


def on_clear_points(image):
    if image is None:
        return None, [], [], None, "请先上传图片"
    return image, [], [], None, "标记和文本定位已清除"


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

/* ── 按钮胶囊形 ── */
.btn-primary, .btn-primary button {
    background: linear-gradient(135deg, var(--blue), var(--cyan)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
}
.btn-primary:hover, .btn-primary button:hover {
    filter: brightness(1.1);
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


# ── build_ui() 构建界面 ─────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="全自动抠图") as demo:
        gr.Markdown("# 全自动抠图工具")

        # ==================== Tab 1: 一键抠图 ====================
        with gr.Tab("一键抠图"):
            with gr.Row():
                # 左栏：控制面板
                with gr.Column(scale=1, elem_classes="control-rail"):
                    gr.Markdown('<div class="section-title">一键抠图</div>')

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

                # 中栏：上传 + 原图预览
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown('<div class="section-title">原图</div>')
                    auto_files = gr.File(
                        label="上传图片（支持多张）",
                        file_count="multiple",
                        file_types=["image"],
                        elem_classes="upload-area",
                    )
                    auto_input_img = gr.Image(
                        label="预览",
                        interactive=False,
                        elem_classes="checkerboard",
                    )
                    auto_swap_btn = gr.Button(
                        "换图",
                        visible=False,
                        elem_classes="btn-secondary",
                    )

                # 右栏：结果
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown(
                        '<div class="section-title">结果 '
                        '<span class="badge">透明背景</span></div>'
                    )
                    auto_result_img = gr.Image(
                        label="抠图结果",
                        interactive=False,
                        elem_classes="checkerboard",
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

                    # 文本定位 UI（条件显示）
                    with gr.Group(visible=False) as text_locate_group:
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

                # 中栏：上传 + 画布
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown(
                        '<div class="section-title">画布 '
                        '<span class="badge">Canvas</span>'
                        ' <span style="font-size:0.75em;color:var(--ink-muted);'
                        'font-weight:400;margin-left:8px">'
                        '点击图片选取区域，绿色=正向，红色=负向</span></div>'
                    )
                    canvas_files = gr.File(
                        label="上传图片",
                        file_types=["image"],
                        elem_classes="upload-area",
                    )
                    canvas_img = gr.Image(
                        label="画布",
                        type="numpy",
                        interactive=False,
                        elem_classes="checkerboard",
                    )
                    canvas_swap_btn = gr.Button(
                        "换图",
                        visible=False,
                        elem_classes="btn-secondary",
                    )

                # 右栏：结果
                with gr.Column(scale=4, elem_classes="panel-card"):
                    gr.Markdown(
                        '<div class="section-title">结果 '
                        '<span class="badge">Result</span></div>'
                    )
                    result_img = gr.Image(
                        label="抠图结果",
                        interactive=False,
                        elem_classes="checkerboard",
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
            outputs=[auto_files, auto_input_img, auto_swap_btn],
        )
        auto_swap_btn.click(
            fn=on_auto_swap,
            outputs=[auto_files, auto_input_img, auto_swap_btn],
        )

        auto_btn.click(
            fn=on_auto_process,
            inputs=[auto_files, detect_transparent, vitmatte_variant,
                    process_mode, save_debug],
            outputs=[auto_input_img, auto_status, auto_result_img],
            stream_every=0.5,
        )

        # --- Tab 2 ---
        use_text_locate.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[use_text_locate],
            outputs=[text_locate_group],
        )

        canvas_files.upload(
            fn=on_image_upload,
            inputs=[canvas_files],
            outputs=[canvas_files, canvas_img, canvas_swap_btn,
                     points_state, result_img, cutout_status],
        )
        canvas_swap_btn.click(
            fn=on_canvas_swap,
            outputs=[canvas_files, canvas_img, canvas_swap_btn,
                     points_state, labels_state, box_state],
        )

        locate_btn.click(
            fn=on_text_locate,
            inputs=[canvas_img, text_caption, engine_mode],
            outputs=[result_img, points_state, labels_state, box_state,
                     cutout_status],
        )

        canvas_img.select(
            fn=on_image_click,
            inputs=[canvas_img, click_mode, engine_mode,
                    points_state, labels_state, box_state],
            outputs=[result_img, points_state, labels_state, box_state,
                     cutout_status],
        )

        generate_btn.click(
            fn=on_generate_cutout,
            inputs=[canvas_img, engine_mode,
                    points_state, labels_state, box_state],
            outputs=[result_img, cutout_status],
            stream_every=0.5,
        )

        clear_btn.click(
            fn=on_clear_points,
            inputs=[canvas_img],
            outputs=[canvas_img, points_state, labels_state, box_state,
                     cutout_status],
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
    )

    # 信号处理
    def _force_exit(*_):
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    # 主线程阻塞
    while True:
        time.sleep(1)
