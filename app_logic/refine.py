# ── 手动边缘修复（Tab1 / Tab2 共用）──────────────────────────────
import os

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from model_manager import VITMATTE_VARIANTS, get_output_path
from engines.manual_refine import refine_manual_edge

_mgr = None


def init(mgr_instance):
    global _mgr
    _mgr = mgr_instance


# ── 内部工具 ─────────────────────────────────────────────────────

def _extract_user_mask(editor_value):
    """Extract painted mask from ImageEditor value. Returns HxW bool, or None."""
    if not editor_value:
        return None
    if isinstance(editor_value, dict):
        layers = editor_value.get("layers")
        if layers and len(layers) > 0:
            layer0 = layers[0]
            if isinstance(layer0, dict):
                layer0 = layer0.get("image", layer0.get("composite"))
            if layer0 is not None and hasattr(layer0, 'ndim') and layer0.ndim >= 3:
                mask = layer0[..., 3] > 30 if layer0.shape[2] == 4 else layer0[..., 0] > 200
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
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


def _empty_editor_value():
    """Return a harmless hidden ImageEditor value for Gradio 6.15 remounts."""
    empty = np.zeros((1, 1, 4), dtype=np.uint8)
    return {"background": empty, "layers": [], "composite": empty}


def _normal_result_title():
    return ('<div class="section-title">效果预览 '
            '<span class="badge">透明背景</span></div>')


def _normal_canvas_result_title():
    return ('<div class="section-title">选区预览 '
            '<span class="badge">抠图结果</span></div>')


def _clear_manual_refine_updates(normal_title=None, preview_visible=True):
    """Return UI/state updates that prevent stale manual-refine state after image changes."""
    if normal_title is None:
        normal_title = _normal_result_title()
    return (
        None,                      # original_rgb_state
        None,                      # auto_rgba_state
        None,                      # current_rgba_state
        [],                        # edit_history_state
        gr.update(visible=False),  # enter_refine_btn
        gr.update(value=_empty_editor_value(), visible="hidden"),  # auto_result_editor
        gr.update(visible=False),  # editor_actions
        gr.update(visible=preview_visible),  # preview_actions
        gr.update(value=normal_title),
    )


def _clear_canvas_manual_refine_updates(preview_visible=True):
    return _clear_manual_refine_updates(_normal_canvas_result_title(), preview_visible)


# ── 回调函数 ─────────────────────────────────────────────────────

def on_enter_refine_mode(current_rgba_state):
    """Switch from preview to editor mode."""
    if current_rgba_state is None:
        return [gr.update()] * 8 + ["请先完成抠图"]
    editor_val = _make_editor_value(current_rgba_state)
    title = ('<div class="section-title">边缘修复 '
             '<span class="badge">画笔涂抹</span> '
             '<span class="section-hint">涂抹污染区域，涂完点应用</span></div>')
    return (
        gr.update(value=None, visible=False),  # auto_result_img
        gr.update(visible=False),       # preview_actions
        gr.update(value=editor_val, visible=True),  # auto_result_editor
        gr.update(visible=True),        # editor_actions
        gr.update(value=title),         # result_title
        gr.update(visible=False),       # enter_refine_btn
        gr.update(visible=False),       # auto_result_view_btn
        gr.update(visible=False),       # auto_result_download_btn
        "修复模式：涂抹绿/蓝边，涂完点「应用修复」",
    )


def on_exit_refine_mode(current_rgba_state):
    """Switch back from editor to preview mode and show the latest refined result."""
    has_result = current_rgba_state is not None
    return (
        gr.update(value=current_rgba_state, visible=True),
        gr.update(visible=True),
        gr.update(value=_empty_editor_value(), visible="hidden"),
        gr.update(visible=False),
        gr.update(value=_normal_result_title()),
        gr.update(visible=has_result),
        gr.update(visible=has_result),
        gr.update(visible=has_result),
        "",
    )


def on_enter_canvas_refine_mode(current_rgba_state):
    """Switch Tab 2 result preview into the shared edge-refine editor."""
    if current_rgba_state is None:
        return [gr.update()] * 8 + ["请先生成抠图结果"]
    editor_val = _make_editor_value(current_rgba_state)
    title = ('<div class="section-title">边缘修复 '
             '<span class="badge">涂抹蒙版</span> '
             '<span class="section-hint">涂抹污染边缘，然后点击应用修复</span></div>')
    return (
        gr.update(value=None, visible=False),  # result_img
        gr.update(visible=False),              # canvas_preview_actions
        gr.update(value=editor_val, visible=True),  # canvas_result_editor
        gr.update(visible=True),               # canvas_editor_actions
        gr.update(value=title),                # canvas_result_title
        gr.update(visible=False),              # canvas_enter_refine_btn
        gr.update(visible=False),              # result_view_btn
        gr.update(visible=False),              # result_download_btn
        "修复模式：涂抹污染边缘区域，然后点击应用修复",
    )


def on_exit_canvas_refine_mode(current_rgba_state):
    """Switch Tab 2 back from editor to preview mode."""
    has_result = current_rgba_state is not None
    return (
        gr.update(value=current_rgba_state, visible=True),
        gr.update(visible=True),
        gr.update(value=_empty_editor_value(), visible="hidden"),
        gr.update(visible=False),
        gr.update(value=_normal_canvas_result_title()),
        gr.update(visible=has_result),
        gr.update(visible=has_result),
        gr.update(visible=has_result),
        "",
    )


def on_apply_refine(auto_result_editor, original_rgb_state, current_rgba_state,
                    edit_history_state, vitmatte_variant, save_debug=False):
    """Apply manual edge refinement."""
    def fail(message):
        return (
            auto_result_editor,
            current_rgba_state,
            edit_history_state,
            gr.update(),
            gr.update(),
            gr.update(),
            message,
        )

    user_mask = _extract_user_mask(auto_result_editor)
    if user_mask is None or not np.any(user_mask):
        return fail("未检测到涂抹区域，请先涂抹污染边缘")

    image_rgb = original_rgb_state
    if image_rgb is None or current_rgba_state is None:
        return fail("缺少原图或当前结果状态")

    variant_key = VITMATTE_VARIANTS.get(vitmatte_variant, "none")
    if variant_key == "none":
        variant_key = "base"
    try:
        refiner = _mgr.get_vitmatte(variant_key)
    except Exception as e:
        return fail(f"ViTMatte 加载失败: {e}")

    try:
        output_dir = get_output_path()
        out_path = os.path.join(output_dir, "refined.png")
        counter = 1
        while os.path.exists(out_path):
            out_path = os.path.join(output_dir, f"refined_{counter}.png")
            counter += 1
        debug_dir = os.path.splitext(out_path)[0] + "_debug" if save_debug else None
        rgba_out, diag = refine_manual_edge(
            image_rgb, current_rgba_state, user_mask, refiner,
            debug_dir=debug_dir,
            verbose=bool(save_debug or os.environ.get("MANUAL_REFINE_DEBUG")),
        )
    except Exception as e:
        return fail(f"修复失败: {e}")

    history = list(edit_history_state or [])
    history.append(current_rgba_state.copy())
    if len(history) > 5:
        history = history[-5:]

    Image.fromarray(rgba_out, "RGBA").save(out_path)

    editor_val = _make_editor_value(rgba_out)
    status_parts = [
        f"accept:{diag.get('accept_pixels', 0)}px",
        f"alpha_delta:{diag.get('alpha_delta_mean', 0):.1f}",
        f"gate:{diag.get('gate_mean', 0):.2f} conf:{diag.get('rgb_conf_mean', 0):.2f}",
    ]
    smooth = diag.get('edge_smooth_score', 0)
    smooth_label = 'bad' if smooth > 0.6 else 'mid' if smooth > 0.3 else 'good'
    status_parts.append(f"smooth:{smooth:.2f}({smooth_label})")
    if diag.get("residue_after_by_alpha"):
        after = diag["residue_after_by_alpha"]
        before = diag.get("residue_before_by_alpha", {})
        status_parts.append(f">=240: {before.get('gte240', 0):.3f}->{after.get('gte240', 0):.3f}")
    status_parts.append(os.path.basename(out_path))

    return (
        gr.update(value=editor_val),
        rgba_out,
        history,
        gr.update(value=rgba_out, visible=False),
        gr.update(visible=False),
        gr.update(value=out_path, visible=False),
        " | ".join(status_parts),
    )


def on_undo_refine(edit_history_state, current_rgba_state):
    """Undo last refinement."""
    history = list(edit_history_state or [])
    if not history:
        editor_update = _make_editor_value(current_rgba_state) if current_rgba_state is not None else gr.update()
        return (
            current_rgba_state,
            edit_history_state,
            editor_update,
            gr.update(value=current_rgba_state, visible=False) if current_rgba_state is not None else gr.update(visible=False),
            gr.update(value=None, visible=False),
            "No refine history to undo.",
        )
    prev = history.pop()
    editor_val = _make_editor_value(prev)
    return (
        prev,
        history,
        gr.update(value=editor_val),
        gr.update(value=prev, visible=False),
        gr.update(value=None, visible=False),
        "Undid last refine.",
    )


def on_reset_auto(auto_rgba_state):
    """Reset to auto result."""
    if auto_rgba_state is None:
        return (None, [], gr.update(), gr.update(visible=False), gr.update(value=None, visible=False), "No auto result to restore.")
    editor_val = _make_editor_value(auto_rgba_state)
    return (
        auto_rgba_state,
        [],
        gr.update(value=editor_val),
        gr.update(value=auto_rgba_state, visible=False),
        gr.update(value=None, visible=False),
        "Reset to auto result.",
    )
