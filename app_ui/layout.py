# ── UI 布局 + CSS/JS ─────────────────────────────────────────────
import gradio as gr

from model_manager import VITMATTE_VARIANTS, VITMATTE_PROCESS_MODES
from app_logic.tab2 import ENGINE_MODE_MAP, TAB2_OUTPUT_MODES, _ensure_auto_choice


# ── CSS ──────────────────────────────────────────────────────────

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

body {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: "Inter", "SF Pro Display", -apple-system, sans-serif;
}

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
.control-rail::-webkit-scrollbar { width: 4px; }
.control-rail::-webkit-scrollbar-thumb { background: var(--ink-muted); border-radius: 2px; }

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

.upload-area {
    border: 2px dashed var(--ink-muted) !important;
    border-radius: var(--radius) !important;
    background: var(--panel-tint) !important;
    transition: border-color 0.2s;
}
.upload-area:hover { border-color: var(--blue) !important; }

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
.image-lightbox-overlay.open { display: flex; }
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
.image-lightbox-close:hover { background: rgba(255,255,255,0.24); }
body.image-lightbox-open { overflow: hidden; }
.preview-actions {
    margin-top: 8px;
    display: flex;
    justify-content: flex-end;
}
.preview-actions button,
.preview-open-btn button {
    font-size: 0.84em !important;
}

/* 左栏按钮行：紧凑并排 */
.control-rail .row > button,
.control-rail .row > .gr-button {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    padding: 6px 8px !important;
    font-size: 0.82em !important;
}

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
.segment-control .wrap label:hover { background: var(--glass) !important; }
.segment-control .wrap label.selected {
    background: var(--panel) !important;
    color: var(--blue) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    font-weight: 600;
}
.segment-control .wrap label input[type="radio"] { display: none !important; }

.status-box textarea {
    background: var(--panel-tint) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-soft) !important;
    font-size: 0.85em;
    resize: none;
}

input[type="checkbox"] { accent-color: var(--blue); }
.hidden { display: none !important; }

@media (max-width: 768px) {
    .control-rail { max-height: none !important; overflow-x: auto; }
}

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


# ── JS ──────────────────────────────────────────────────────────

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

    // 全局粘贴：Ctrl+V 将图片 base64 写入隐藏 Textbox
    document.addEventListener("paste", (event) => {
        const tag = document.activeElement?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        const items = event.clipboardData?.items;
        if (!items) return;
        for (const item of items) {
            if (!item.type.startsWith("image/")) continue;
            const blob = item.getAsFile();
            if (!blob) break;
            event.preventDefault();
            const reader = new FileReader();
            reader.onload = () => {
                // 找到隐藏的 paste textbox 并写入 base64
                const boxes = document.querySelectorAll('textarea[data-testid="textbox"]');
                for (const box of boxes) {
                    if (!box.placeholder?.includes("__paste__")) continue;
                    const setter = Object.getOwnPropertyDescriptor(
                        HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(box, reader.result);
                    box.dispatchEvent(new Event('input', { bubbles: true }));
                    break;
                }
            };
            reader.readAsDataURL(blob);
            break;
        }
    });
})();
"""


# ── Tab 1 布局 ──────────────────────────────────────────────────

def build_tab1_ui():
    """构建 Tab 1 三栏布局，返回组件引用 dict。"""
    comps = {}

    with gr.Tab("一键抠图"):
        with gr.Row():
            # 左栏
            with gr.Column(scale=1, elem_classes="control-rail"):
                gr.Markdown(
                    '<div class="section-title">一键抠图 '
                    '<span class="badge">RMBG-2.0</span></div>'
                )
                comps["detect_transparent"] = gr.Checkbox(
                    label="检测透明物体（玻璃/水滴等）", value=False,
                )
                comps["save_debug"] = gr.Checkbox(
                    label="保存诊断中间结果", value=False,
                )
                comps["vitmatte_variant"] = gr.Radio(
                    choices=list(VITMATTE_VARIANTS.keys()),
                    value="直出", label="精修模型",
                    elem_classes="segment-control",
                )
                with gr.Group(visible=False) as process_mode_group:
                    comps["process_mode"] = gr.Radio(
                        choices=list(VITMATTE_PROCESS_MODES.keys()),
                        value="条带", label="推理模式",
                        elem_classes="segment-control",
                    )
                comps["process_mode_group"] = process_mode_group

                comps["upload_mode"] = gr.Radio(
                    choices=["单张", "批量"], value="单张", label="上传模式",
                    elem_classes="segment-control",
                )

                comps["auto_status"] = gr.Textbox(
                    label="状态", interactive=False, lines=3,
                    elem_classes="status-box",
                )
                comps["auto_btn"] = gr.Button(
                    "开始抠图", variant="primary", elem_classes="btn-primary",
                )

            # 中栏
            with gr.Column(scale=4, elem_classes="panel-card"):
                gr.Markdown(
                    '<div class="section-title">原图 '
                    '<span class="section-hint">上传后在这里确认待处理图片</span></div>'
                )
                comps["auto_single_img"] = gr.Image(
                    sources=["upload", "clipboard"], type="numpy",
                    label="上传或粘贴图片",
                    visible=True, interactive=True,
                    elem_classes="upload-area",
                )
                comps["auto_files"] = gr.File(
                    label="上传原图（支持多张，不支持粘贴）", file_count="multiple",
                    file_types=["image"], elem_classes="upload-area",
                    visible=False,
                )
                comps["paste_box"] = gr.Textbox(
                    placeholder="__paste__", visible=False, max_lines=1,
                )
                comps["auto_input_img"] = gr.Image(
                    label="原图", visible=False, interactive=False,
                    elem_classes="checkerboard",
                )
                with gr.Row(elem_classes="preview-actions"):
                    comps["auto_input_view_btn"] = gr.Button(
                        "查看大图", visible=False,
                        elem_classes=["btn-secondary", "preview-open-btn"],
                    )
                    comps["auto_swap_btn"] = gr.Button(
                        "清空原图区", visible=False, elem_classes="btn-secondary",
                    )

            # 右栏
            with gr.Column(scale=4, elem_classes="panel-card"):
                comps["result_title"] = gr.Markdown(
                    '<div class="section-title">效果预览 '
                    '<span class="badge">透明背景</span></div>'
                )
                comps["auto_result_img"] = gr.Image(
                    label="效果预览", interactive=False, visible=True,
                    buttons=[], elem_classes="checkerboard",
                )
                comps["preview_actions"] = gr.Row(
                    elem_classes="preview-actions", visible=True,
                )
                with comps["preview_actions"]:
                    comps["auto_result_view_btn"] = gr.Button(
                        "查看大图", visible=False,
                        elem_classes=["btn-secondary", "preview-open-btn"],
                    )
                    comps["auto_result_download_btn"] = gr.DownloadButton(
                        "下载", visible=False, elem_classes="btn-secondary",
                    )
                    comps["enter_refine_btn"] = gr.Button(
                        "边缘修复", visible=False, elem_classes="btn-primary",
                    )

                comps["auto_result_editor"] = gr.ImageEditor(
                    label=None, image_mode="RGBA", type="numpy",
                    height="68vh", canvas_size=(2048, 2048),
                    brush=gr.Brush(default_size=20, colors=["#ff0000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=20),
                    layers=True, transforms=None,
                    elem_classes=["checkerboard", "refine-editor"],
                    interactive=True, show_label=False, visible=False,
                )
                comps["editor_actions"] = gr.Row(
                    elem_classes="preview-actions", visible=False,
                )
                with comps["editor_actions"]:
                    comps["apply_refine_btn"] = gr.Button(
                        "应用修复", variant="primary", elem_classes="btn-primary",
                    )
                    comps["undo_refine_btn"] = gr.Button(
                        "撤销", elem_classes="btn-secondary",
                    )
                    comps["reset_auto_btn"] = gr.Button(
                        "重置", elem_classes="btn-secondary",
                    )
                    comps["exit_refine_btn"] = gr.Button(
                        "退出修复", elem_classes="btn-secondary",
                    )

                comps["original_rgb_state"] = gr.State(None)
                comps["auto_rgba_state"] = gr.State(None)
                comps["current_rgba_state"] = gr.State(None)
                comps["edit_history_state"] = gr.State([])

    return comps


# ── Tab 2 布局 ──────────────────────────────────────────────────

def build_tab2_ui():
    """构建 Tab 2 三栏布局，返回组件引用 dict。"""
    comps = {}

    with gr.Tab("精细选区"):
        with gr.Row():
            # 左栏
            with gr.Column(scale=1, elem_classes="control-rail"):
                gr.Markdown(
                    '<div class="section-title">精细选区 '
                    '<span class="badge">SAM</span></div>'
                )
                comps["engine_mode"] = gr.Radio(
                    choices=list(ENGINE_MODE_MAP.keys()),
                    value="高精度", label="引擎模式",
                    elem_classes="segment-control",
                )
                comps["tab2_output_mode"] = gr.Radio(
                    choices=list(TAB2_OUTPUT_MODES.keys()),
                    value="SAM严格", label="输出模式",
                    elem_classes="segment-control",
                )
                comps["canvas_preserve_transparency"] = gr.Checkbox(
                    label="保护透明/半透明材质", value=False,
                )
                comps["canvas_save_debug"] = gr.Checkbox(
                    label="保存诊断中间结果", value=False,
                )
                comps["click_mode"] = gr.Radio(
                    choices=["正向选取（我要）", "负向排除（不要）"],
                    value="正向选取（我要）", label="点击模式",
                    elem_classes="segment-control",
                )
                comps["text_caption"] = gr.Textbox(
                    label=None, placeholder="输入物体名称定位，如 goose, red car",
                    lines=1, container=False,
                    elem_classes="text-locate-input",
                )
                comps["locate_btn"] = gr.Button(
                    "文本定位", variant="primary", elem_classes="btn-primary",
                    size="sm",
                )
                comps["cutout_status"] = gr.Textbox(
                    label="状态", interactive=False, lines=2, max_lines=3,
                    value="点图 — 小物体/精细控制 | 文字 — 知道名称/多候选 | 自动分割 — 不确定位置",
                    elem_classes="status-box",
                )
                comps["auto_seg_btn"] = gr.Button(
                    "自动分割", variant="secondary", elem_classes="btn-secondary",
                )
                with gr.Row():
                    comps["undo_btn"] = gr.Button(
                        "撤销", elem_classes="btn-secondary", min_width=0,
                    )
                    comps["clear_btn"] = gr.Button(
                        "清除标记", elem_classes="btn-secondary", min_width=0,
                    )
                comps["generate_btn"] = gr.Button(
                    "开始抠图", variant="primary", elem_classes="btn-primary",
                )

            # 中栏
            with gr.Column(scale=4, elem_classes="panel-card"):
                gr.Markdown(
                    '<div class="section-title">原图 '
                    '<span class="section-hint">点击图片选取/排除，绿色=正向，红色=负向</span></div>'
                )
                comps["canvas_files"] = gr.Image(
                    sources=["upload", "clipboard"], type="numpy",
                    label="上传原图（支持粘贴 Ctrl+V）",
                    elem_classes="upload-area",
                )
                comps["paste_box"] = gr.Textbox(
                    placeholder="__paste__", visible=False, max_lines=1,
                )
                comps["canvas_img"] = gr.Image(
                    label="原图", type="numpy", visible=False,
                    interactive=False, elem_classes="checkerboard",
                )
                with gr.Row(elem_classes="preview-actions"):
                    comps["canvas_view_btn"] = gr.Button(
                        "查看大图", visible=False,
                        elem_classes=["btn-secondary", "preview-open-btn"],
                    )
                    comps["canvas_swap_btn"] = gr.Button(
                        "清空原图区", visible=False, elem_classes="btn-secondary",
                    )

            # 右栏
            with gr.Column(scale=4, elem_classes="panel-card"):
                comps["canvas_result_title"] = gr.Markdown(
                    '<div class="section-title">选区预览 '
                    '<span class="badge">抠图结果</span></div>'
                )
                comps["result_img"] = gr.Image(
                    label="选区预览", interactive=False, buttons=[],
                    elem_classes="checkerboard",
                )
                comps["canvas_preview_actions"] = gr.Row(
                    elem_classes="preview-actions", visible=True,
                )
                with comps["canvas_preview_actions"]:
                    comps["result_view_btn"] = gr.Button(
                        "查看大图", visible=False,
                        elem_classes=["btn-secondary", "preview-open-btn"],
                    )
                    comps["result_download_btn"] = gr.DownloadButton(
                        "下载", visible=False, elem_classes="btn-secondary",
                    )
                    comps["canvas_enter_refine_btn"] = gr.Button(
                        "边缘修复", visible=False, elem_classes="btn-primary",
                    )
                comps["canvas_result_editor"] = gr.ImageEditor(
                    label=None, image_mode="RGBA", type="numpy",
                    height="68vh", canvas_size=(2048, 2048),
                    brush=gr.Brush(default_size=20, colors=["#ff0000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=20),
                    layers=True, transforms=None,
                    elem_classes=["checkerboard", "refine-editor"],
                    interactive=True, show_label=False, visible=False,
                )
                comps["canvas_editor_actions"] = gr.Row(
                    elem_classes="preview-actions", visible=False,
                )
                with comps["canvas_editor_actions"]:
                    comps["canvas_apply_refine_btn"] = gr.Button(
                        "应用修复", variant="primary", elem_classes="btn-primary",
                    )
                    comps["canvas_undo_refine_btn"] = gr.Button(
                        "撤销", elem_classes="btn-secondary",
                    )
                    comps["canvas_reset_auto_btn"] = gr.Button(
                        "重置", elem_classes="btn-secondary",
                    )
                    comps["canvas_exit_refine_btn"] = gr.Button(
                        "退出修复", elem_classes="btn-secondary",
                    )

        # State
        comps["points_state"] = gr.State([])
        comps["labels_state"] = gr.State([])
        comps["box_state"] = gr.State(None)
        comps["auto_masks_state"] = gr.State([])
        comps["auto_choice_state"] = gr.State(_ensure_auto_choice({"selected": [], "excluded": []}))
        comps["canvas_original_rgb_state"] = gr.State(None)
        comps["canvas_auto_rgba_state"] = gr.State(None)
        comps["canvas_current_rgba_state"] = gr.State(None)
        comps["canvas_edit_history_state"] = gr.State([])
        comps["canvas_refine_variant_state"] = gr.State("Base")

    return comps
