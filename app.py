# ── 入口：初始化 + Tab1 回调 + UI 构建 + CLI ─────────────────────
import argparse, gc, os, signal, time, threading, warnings

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

os.environ["HF_HOME"] = os.path.join(get_base_path(), "models", "cache")
mgr = ModelManager()
_startup_log("initialize globals")

import gradio as gr
_startup_log("import gradio")
import numpy as np
_startup_log("import numpy")
from PIL import Image
_startup_log("import PIL")

# 初始化子模块
from app_logic import tab2, refine
from app_ui.layout import APP_CSS, APP_JS
tab2.init(mgr)
refine.init(mgr)
_startup_log("init submodules")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ── 预热 ────────────────────────────────────────────────────────

def start_default_model_warmup():
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


# ── Tab 1 回调：一键抠图 ────────────────────────────────────────

def _has_source_content(source):
    if source is None:
        return False
    if isinstance(source, np.ndarray):
        return source.size > 0
    if isinstance(source, (list, tuple, set)):
        return len(source) > 0
    return True


def on_auto_process(files, single_img, source_img, detect_transparent, vitmatte_variant,
                    process_mode, save_debug=False):
    yield (gr.update(), "开始处理...", None, gr.update(visible=False), gr.update(visible=False),
           gr.update(), gr.update(), gr.update(), gr.update(visible=False))

    # 统一文件列表：单张模式从 numpy 保存临时文件，批量模式直接用文件列表
    if single_img is not None:
        import tempfile
        img_arr = np.asarray(single_img)
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[2] == 4:
            img_arr = img_arr[:, :, :3]
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(img_arr).save(tmp.name)
        files = [tmp.name]
    if not files:
        yield (gr.update(), "请先上传图片", None, gr.update(visible=False), gr.update(visible=False),
               gr.update(), gr.update(), gr.update(), gr.update(visible=False))
        return
    if not _has_source_content(source_img):
        yield (gr.update(), "请等待原图预览加载完成", None, gr.update(visible=False), gr.update(visible=False),
               gr.update(), gr.update(), gr.update(), gr.update(visible=False))
        return

    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    refine_mode = VITMATTE_PROCESS_MODES.get(process_mode, "strip")
    needs_vitmatte = variant_key != "none"
    needs_dino = bool(detect_transparent)
    should_unload_unused = tab2.should_unload_for_tab1()

    if should_unload_unused:
        tab2._clear_all_sam_contexts()
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

    detector = None
    if detect_transparent:
        detector = mgr.grounding_dino
        yield gr.update(), "Grounding-DINO 已加载，开始处理...", gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(visible=False)

    output_dir = get_output_path()
    total = len(files)
    last_result = None
    last_original = None

    for idx, f in enumerate(files):
        ext = os.path.splitext(str(f))[-1].lower()
        if ext not in VALID_EXTS:
            continue

        fname = os.path.basename(str(f))
        yield gr.update(), f"[{idx + 1}/{total}] 正在处理: {fname}", gr.update(), gr.update(visible=(last_result is not None)), gr.update(visible=(last_result is not None)), gr.update(), gr.update(), gr.update(), gr.update(visible=False)

        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue

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
        yield last_original, f"[{idx + 1}/{total}] RMBG-2.0 推理中: {fname}", gr.update(), gr.update(visible=(last_result is not None)), gr.update(visible=(last_result is not None)), gr.update(), gr.update(), gr.update(), gr.update(visible=False)

        result = mgr.rmbg2.remove_background(
            img, refiner=refiner, transparent_detector=detector,
            refine_mode=refine_mode, debug_dir=debug_dir,
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

        del img
        if (idx + 1) % 10 == 0:
            gc.collect()

    if last_result is not None:
        yield gr.update(), f"全部完成，共处理 {total} 张，结果保存在 output/", gr.update(), gr.update(visible=True), gr.update(visible=True), gr.update(), gr.update(), gr.update(), gr.update(visible=True)
    else:
        yield gr.update(), "没有有效图片被处理", gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(visible=False)


def on_auto_upload(image):
    """接收 numpy 数组（来自 gr.Image 上传/粘贴）。返回 8 值。"""
    if image is None:
        return (gr.update(), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), None,
                gr.update(visible=False), gr.update(value=None, visible=False), "请先上传图片")
    try:
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return (gr.update(), gr.update(value=img, visible=True),
                gr.update(visible=True), gr.update(visible=True), None,
                gr.update(visible=False), gr.update(value=None, visible=False), "图片已上传，点击开始抠图")
    except Exception:
        return (gr.update(), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), None,
                gr.update(visible=False), gr.update(value=None, visible=False), "图片加载失败")


def on_auto_upload_from_file(files):
    """接收文件列表（来自 gr.File 上传）。返回 8 值。"""
    if not files:
        return (gr.update(value=None, visible=True), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), None,
                gr.update(visible=False), gr.update(value=None, visible=False), "请先上传图片")
    first = files[0] if isinstance(files, list) else files
    try:
        img = Image.open(first).convert("RGB")
        return (gr.update(visible=False), gr.update(value=np.array(img), visible=True),
                gr.update(visible=True), gr.update(visible=True), None,
                gr.update(visible=False), gr.update(value=None, visible=False), "图片已上传，点击开始抠图")
    except Exception:
        return (gr.update(value=None, visible=True), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), None,
                gr.update(visible=False), gr.update(value=None, visible=False), "图片加载失败")


def on_auto_clear_source():
    return gr.update(value=None, visible=True), gr.update(value=None), \
        gr.update(value=None, visible=False), \
        gr.update(visible=False), gr.update(visible=False), None, \
        gr.update(visible=False), gr.update(value=None, visible=False), "请先上传图片"


def on_vitmatte_variant_change(vitmatte_variant):
    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    return gr.update(visible=(variant_key != "none"))


def on_upload_mode_change(mode):
    """切换单张/批量上传模式。"""
    if mode == "单张":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


# ── build_ui ────────────────────────────────────────────────────

def build_ui(model_concurrency_limit=2):
    from app_ui.layout import build_tab1_ui, build_tab2_ui
    from app_ui.events import bind_tab1_events, bind_tab2_events

    with gr.Blocks(title="全自动抠图") as demo:
        gr.Markdown("# 全自动抠图工具")
        tab1_comps = build_tab1_ui()
        tab2_comps = build_tab2_ui()
        tab1_callbacks = {
            "on_auto_upload": on_auto_upload,
            "on_auto_upload_from_file": on_auto_upload_from_file,
            "on_auto_clear_source": on_auto_clear_source,
            "on_auto_process": on_auto_process,
            "on_vitmatte_variant_change": on_vitmatte_variant_change,
            "on_upload_mode_change": on_upload_mode_change,
        }
        bind_tab1_events(demo, tab1_comps, model_concurrency_limit, tab1_callbacks)
        bind_tab2_events(demo, tab2_comps, model_concurrency_limit)

    return demo


# ── CLI ─────────────────────────────────────────────────────────

def _parse_cli_args():
    parser = argparse.ArgumentParser(description="AI 抠图工具 Web UI")
    parser.add_argument("-p", "--port", type=int, default=18181, help="监听端口（默认 18181）")
    parser.add_argument("-q", "--silent", action="store_true", help="静默启动")
    parser.add_argument("--model-concurrency", type=int, default=None, help="模型推理并发数")
    parser.add_argument("--queue-size", type=int, default=32, help="等待队列最大长度（默认 32）")
    parser.add_argument("--max-sam-sessions", type=int, default=8, help="多 session 时最多保留的 SAM 会话数")
    parser.add_argument("--multi-session", action="store_true", help="启用多人/多标签页隔离")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    _startup_log("parse args")
    tab2.configure_runtime(args.multi_session, args.max_sam_sessions)
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

    def _force_exit(*_):
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    while True:
        time.sleep(1)
