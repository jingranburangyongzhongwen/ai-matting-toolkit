# ── 事件绑定 ─────────────────────────────────────────────────────
import gradio as gr

from app_logic import tab2, refine
from app_logic.refine import (
    _clear_manual_refine_updates, _clear_canvas_manual_refine_updates,
    _normal_result_title,
)


def bind_tab1_events(demo, c, model_concurrency_limit, callbacks):
    """绑定 Tab 1 所有事件。c = tab1 components dict。callbacks = tab1 回调 dict。"""
    # 上传模式切换
    c["upload_mode"].change(
        fn=callbacks["on_upload_mode_change"],
        inputs=[c["upload_mode"]],
        outputs=[c["auto_single_img"], c["auto_files"]],
        queue=False, show_progress="hidden",
    )

    # 单张上传/粘贴
    c["auto_single_img"].change(
        fn=callbacks["on_auto_upload"],
        inputs=[c["auto_single_img"]],
        outputs=[c["auto_single_img"], c["auto_input_img"], c["auto_input_view_btn"],
                 c["auto_swap_btn"], c["auto_result_img"], c["auto_result_view_btn"],
                 c["auto_result_download_btn"], c["auto_status"]],
        queue=False, show_progress="hidden",
    )
    c["auto_single_img"].change(
        fn=_clear_manual_refine_updates,
        inputs=[],
        outputs=[c["original_rgb_state"], c["auto_rgba_state"], c["current_rgba_state"],
                 c["edit_history_state"], c["enter_refine_btn"], c["auto_result_editor"],
                 c["editor_actions"], c["preview_actions"], c["result_title"]],
        queue=False, show_progress="hidden",
    )

    # 粘贴解码 → 写入 Image 组件（触发上面的 change 事件）
    c["paste_box"].change(
        fn=tab2.on_paste_decode,
        inputs=[c["paste_box"]],
        outputs=[c["auto_single_img"]],
        queue=False, show_progress="hidden",
    )

    # 批量上传
    c["auto_files"].upload(
        fn=callbacks["on_auto_upload_from_file"],
        inputs=[c["auto_files"]],
        outputs=[c["auto_files"], c["auto_input_img"], c["auto_input_view_btn"],
                 c["auto_swap_btn"], c["auto_result_img"], c["auto_result_view_btn"],
                 c["auto_result_download_btn"], c["auto_status"]],
        queue=False, show_progress="hidden",
    )
    c["auto_files"].upload(
        fn=_clear_manual_refine_updates,
        inputs=[],
        outputs=[c["original_rgb_state"], c["auto_rgba_state"], c["current_rgba_state"],
                 c["edit_history_state"], c["enter_refine_btn"], c["auto_result_editor"],
                 c["editor_actions"], c["preview_actions"], c["result_title"]],
        queue=False, show_progress="hidden",
    )

    # 清空原图
    c["auto_swap_btn"].click(
        fn=callbacks["on_auto_clear_source"],
        outputs=[c["auto_files"], c["auto_single_img"], c["auto_input_img"], c["auto_input_view_btn"],
                 c["auto_swap_btn"], c["auto_result_img"], c["auto_result_view_btn"],
                 c["auto_result_download_btn"], c["auto_status"]],
        queue=False, show_progress="hidden",
    )
    c["auto_swap_btn"].click(
        fn=_clear_manual_refine_updates,
        inputs=[],
        outputs=[c["original_rgb_state"], c["auto_rgba_state"], c["current_rgba_state"],
                 c["edit_history_state"], c["enter_refine_btn"], c["auto_result_editor"],
                 c["editor_actions"], c["preview_actions"], c["result_title"]],
        queue=False, show_progress="hidden",
    )
    c["vitmatte_variant"].change(
        fn=callbacks["on_vitmatte_variant_change"],
        inputs=[c["vitmatte_variant"]],
        outputs=[c["process_mode_group"]],
        queue=False, show_progress="hidden",
    )

    # Reset right column and manual-refine state when starting auto process
    c["auto_btn"].click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(value=_normal_result_title()),
                    None, None, None, [], gr.update(visible=False)),
        inputs=[],
        outputs=[c["auto_result_img"], c["preview_actions"],
                 c["auto_result_editor"], c["editor_actions"], c["result_title"],
                 c["original_rgb_state"], c["auto_rgba_state"], c["current_rgba_state"],
                 c["edit_history_state"], c["enter_refine_btn"]],
        queue=False, show_progress="hidden",
    )
    c["auto_btn"].click(
        fn=callbacks["on_auto_process"],
        inputs=[c["auto_files"], c["auto_single_img"], c["auto_input_img"], c["detect_transparent"],
                c["vitmatte_variant"], c["process_mode"], c["save_debug"]],
        outputs=[c["auto_input_img"], c["auto_status"], c["auto_result_img"],
                 c["auto_result_view_btn"], c["auto_result_download_btn"],
                 c["original_rgb_state"], c["auto_rgba_state"], c["current_rgba_state"],
                 c["enter_refine_btn"]],
        stream_every=0.5,
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )

    c["enter_refine_btn"].click(
        fn=refine.on_enter_refine_mode,
        inputs=[c["current_rgba_state"]],
        outputs=[c["auto_result_img"], c["preview_actions"],
                 c["auto_result_editor"], c["editor_actions"],
                 c["result_title"], c["enter_refine_btn"], c["auto_status"]],
    )
    c["exit_refine_btn"].click(
        fn=refine.on_exit_refine_mode,
        inputs=[c["current_rgba_state"]],
        outputs=[c["auto_result_img"], c["preview_actions"],
                 c["auto_result_editor"], c["editor_actions"],
                 c["result_title"], c["enter_refine_btn"],
                 c["auto_result_view_btn"], c["auto_result_download_btn"],
                 c["auto_status"]],
    )
    c["apply_refine_btn"].click(
        fn=refine.on_apply_refine,
        inputs=[c["auto_result_editor"], c["original_rgb_state"], c["current_rgba_state"],
                c["edit_history_state"], c["vitmatte_variant"], c["save_debug"]],
        outputs=[c["auto_result_editor"], c["current_rgba_state"],
                 c["edit_history_state"], c["auto_result_img"],
                 c["auto_result_view_btn"], c["auto_result_download_btn"],
                 c["auto_status"]],
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )
    c["undo_refine_btn"].click(
        fn=refine.on_undo_refine,
        inputs=[c["edit_history_state"], c["current_rgba_state"]],
        outputs=[c["current_rgba_state"], c["edit_history_state"],
                 c["auto_result_editor"], c["auto_result_img"],
                 c["auto_result_download_btn"], c["auto_status"]],
    )
    c["reset_auto_btn"].click(
        fn=refine.on_reset_auto,
        inputs=[c["auto_rgba_state"]],
        outputs=[c["current_rgba_state"], c["edit_history_state"],
                 c["auto_result_editor"], c["auto_result_img"],
                 c["auto_result_download_btn"], c["auto_status"]],
    )


def bind_tab2_events(demo, c, model_concurrency_limit):
    """绑定 Tab 2 所有事件。c = tab2 components dict。"""

    c["engine_mode"].change(
        fn=tab2.on_engine_mode_change,
        inputs=[c["canvas_img"]],
        outputs=[c["result_img"], c["result_view_btn"], c["result_download_btn"],
                 c["points_state"], c["labels_state"], c["box_state"],
                 c["auto_masks_state"], c["auto_choice_state"],
                 c["text_caption"], c["cutout_status"]],
        queue=False, show_progress="hidden",
    )
    c["engine_mode"].change(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )

    c["canvas_files"].change(
        fn=tab2.on_image_upload,
        inputs=[c["canvas_files"]],
        outputs=[c["canvas_files"], c["canvas_img"], c["canvas_view_btn"],
                 c["canvas_swap_btn"], c["result_view_btn"], c["result_download_btn"],
                 c["points_state"], c["labels_state"], c["box_state"], c["result_img"],
                 c["auto_masks_state"], c["auto_choice_state"],
                 c["text_caption"], c["cutout_status"]],
        queue=False, show_progress="hidden",
    )
    c["canvas_files"].change(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )
    # 粘贴解码 → 写入 Image 组件（触发上面的 change 事件）
    c["paste_box"].change(
        fn=tab2.on_paste_decode,
        inputs=[c["paste_box"]],
        outputs=[c["canvas_files"]],
        queue=False, show_progress="hidden",
    )
    c["canvas_swap_btn"].click(
        fn=tab2.on_canvas_clear_source,
        outputs=[c["canvas_files"], c["canvas_img"], c["canvas_view_btn"],
                 c["canvas_swap_btn"], c["result_view_btn"], c["result_download_btn"],
                 c["points_state"], c["labels_state"], c["box_state"], c["result_img"],
                 c["auto_masks_state"], c["auto_choice_state"],
                 c["text_caption"], c["cutout_status"]],
        queue=False, show_progress="hidden",
    )
    c["canvas_swap_btn"].click(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )

    c["locate_btn"].click(
        fn=tab2.on_text_locate,
        inputs=[c["canvas_img"], c["text_caption"], c["engine_mode"]],
        outputs=[c["result_img"], c["result_view_btn"], c["result_download_btn"],
                 c["points_state"], c["labels_state"], c["box_state"],
                 c["auto_masks_state"], c["auto_choice_state"],
                 c["cutout_status"]],
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )
    c["locate_btn"].click(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )

    c["canvas_img"].select(
        fn=tab2.on_image_click,
        inputs=[c["canvas_img"], c["click_mode"], c["engine_mode"],
                c["points_state"], c["labels_state"], c["box_state"],
                c["auto_masks_state"], c["auto_choice_state"]],
        outputs=[c["result_img"], c["result_view_btn"], c["result_download_btn"],
                 c["points_state"], c["labels_state"], c["box_state"],
                 c["auto_choice_state"], c["cutout_status"]],
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )
    c["canvas_img"].select(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )

    c["auto_seg_btn"].click(
        fn=tab2.on_auto_segment,
        inputs=[c["canvas_img"], c["engine_mode"],
                c["points_state"], c["labels_state"], c["box_state"]],
        outputs=[c["result_img"], c["points_state"], c["labels_state"], c["box_state"],
                 c["auto_masks_state"], c["auto_choice_state"],
                 c["cutout_status"]],
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )

    c["generate_btn"].click(
        fn=tab2.clear_result_preview_on_start,
        inputs=[c["canvas_img"], c["result_img"]],
        outputs=[c["result_img"], c["result_view_btn"], c["result_download_btn"]],
        queue=False, show_progress="hidden",
    )
    c["generate_btn"].click(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )
    c["generate_btn"].click(
        fn=tab2.on_generate_cutout,
        inputs=[c["canvas_img"], c["engine_mode"], c["tab2_output_mode"],
                c["points_state"], c["labels_state"], c["box_state"],
                c["auto_masks_state"], c["auto_choice_state"],
                c["canvas_preserve_transparency"], c["canvas_save_debug"]],
        outputs=[c["result_img"], c["result_view_btn"], c["result_download_btn"],
                 c["cutout_status"], c["canvas_original_rgb_state"],
                 c["canvas_auto_rgba_state"], c["canvas_current_rgba_state"],
                 c["canvas_edit_history_state"], c["canvas_enter_refine_btn"]],
        stream_every=0.5,
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )

    c["undo_btn"].click(
        fn=tab2.on_undo_point,
        inputs=[c["canvas_img"], c["points_state"], c["labels_state"], c["box_state"],
                c["auto_masks_state"], c["auto_choice_state"], c["engine_mode"]],
        outputs=[c["result_img"], c["result_view_btn"], c["result_download_btn"],
                 c["points_state"], c["labels_state"], c["box_state"],
                 c["auto_choice_state"], c["cutout_status"]],
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )
    c["undo_btn"].click(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )

    c["clear_btn"].click(
        fn=tab2.on_clear_points,
        inputs=[c["canvas_img"]],
        outputs=[c["result_img"], c["result_view_btn"], c["result_download_btn"],
                 c["points_state"], c["labels_state"], c["box_state"],
                 c["auto_masks_state"], c["auto_choice_state"],
                 c["text_caption"], c["cutout_status"]],
        queue=False, show_progress="hidden",
    )
    c["clear_btn"].click(
        fn=_clear_canvas_manual_refine_updates,
        inputs=[],
        outputs=[c["canvas_original_rgb_state"], c["canvas_auto_rgba_state"],
                 c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_enter_refine_btn"], c["canvas_result_editor"],
                 c["canvas_editor_actions"], c["canvas_preview_actions"],
                 c["canvas_result_title"]],
        queue=False, show_progress="hidden",
    )

    c["canvas_enter_refine_btn"].click(
        fn=refine.on_enter_canvas_refine_mode,
        inputs=[c["canvas_current_rgba_state"]],
        outputs=[c["result_img"], c["canvas_preview_actions"],
                 c["canvas_result_editor"], c["canvas_editor_actions"],
                 c["canvas_result_title"], c["canvas_enter_refine_btn"],
                 c["cutout_status"]],
    )
    c["canvas_exit_refine_btn"].click(
        fn=refine.on_exit_canvas_refine_mode,
        inputs=[c["canvas_current_rgba_state"]],
        outputs=[c["result_img"], c["canvas_preview_actions"],
                 c["canvas_result_editor"], c["canvas_editor_actions"],
                 c["canvas_result_title"], c["canvas_enter_refine_btn"],
                 c["result_view_btn"], c["result_download_btn"],
                 c["cutout_status"]],
    )
    c["canvas_apply_refine_btn"].click(
        fn=refine.on_apply_refine,
        inputs=[c["canvas_result_editor"], c["canvas_original_rgb_state"],
                c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                c["canvas_refine_variant_state"], c["canvas_save_debug"]],
        outputs=[c["canvas_result_editor"], c["canvas_current_rgba_state"],
                 c["canvas_edit_history_state"], c["result_img"],
                 c["result_view_btn"], c["result_download_btn"],
                 c["cutout_status"]],
        concurrency_limit=model_concurrency_limit,
        concurrency_id="model-gpu",
    )
    c["canvas_undo_refine_btn"].click(
        fn=refine.on_undo_refine,
        inputs=[c["canvas_edit_history_state"], c["canvas_current_rgba_state"]],
        outputs=[c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_result_editor"], c["result_img"],
                 c["result_download_btn"], c["cutout_status"]],
    )
    c["canvas_reset_auto_btn"].click(
        fn=refine.on_reset_auto,
        inputs=[c["canvas_auto_rgba_state"]],
        outputs=[c["canvas_current_rgba_state"], c["canvas_edit_history_state"],
                 c["canvas_result_editor"], c["result_img"],
                 c["result_download_btn"], c["cutout_status"]],
    )
