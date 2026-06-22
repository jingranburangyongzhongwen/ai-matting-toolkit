"""Headless UI regression for manual-refine tab switching.

Runs a lightweight Gradio app with the real layout/events, fake model callbacks
for cutout generation only, and the real on_apply_refine path for Tab1.
Then drives tab1 -> refine -> apply -> exit -> tab2 -> refine in Playwright.
"""
from __future__ import annotations

import os
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MATTING_PRELOAD_RMBG", "0")
os.environ.setdefault("MATTING_STARTUP_LOG", "0")

import gradio as gr  # noqa: E402
import app  # noqa: E402
from app_logic import tab2  # noqa: E402


def _rgba(color):
    arr = np.zeros((96, 128, 4), dtype=np.uint8)
    arr[..., :3] = color
    yy, xx = np.ogrid[:96, :128]
    mask = (xx - 64) ** 2 + (yy - 48) ** 2 < 36 ** 2
    arr[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return arr


def _rgb(color):
    arr = np.zeros((96, 128, 3), dtype=np.uint8)
    arr[..., :3] = color
    return arr


def fake_auto_process(files, single_img, source_img, detect_transparent, vitmatte_variant, process_mode, save_debug=False):
    time.sleep(0.5)
    rgba = _rgba((20, 180, 120))
    out = ROOT / "output" / "ui-regression-tab1.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out)
    yield (
        gr.update(), "tab1 fake complete", Image.fromarray(rgba, "RGBA"),
        gr.update(visible=True), gr.update(value=str(out), visible=True),
        _rgb((230, 230, 230)), rgba, rgba, gr.update(visible=True),
    )


def fake_generate_cutout(image, engine_mode, output_mode, points_state, labels_state, box_state,
                         auto_masks_state=None, preserve_transparency=False, save_debug=False,
                         image_id=None, selected_auto_mask=None, selected_auto_indices=None,
                         request: gr.Request = None):
    time.sleep(0.5)
    rgba = _rgba((40, 120, 220))
    out = ROOT / "output" / "ui-regression-tab2.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out)
    yield (
        Image.fromarray(rgba, "RGBA"), gr.update(visible=True),
        gr.update(value=str(out), visible=True), "tab2 fake complete",
        _rgb((230, 230, 230)), rgba, rgba, [], gr.update(visible=True),
        gr.update(), gr.update(),
    )


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def paint_refine_editor(page, panel_idx: int) -> None:
    panel = page.locator('[role="tabpanel"]').nth(panel_idx)
    canvas = panel.locator(".refine-editor canvas").first
    canvas.wait_for(state="visible", timeout=10000)
    box = canvas.bounding_box()
    if not box:
        raise AssertionError(f"Tab {panel_idx + 1} refine editor canvas has no bounding box")
    cx = box["x"] + box["width"] * 0.5
    cy = box["y"] + box["height"] * 0.5
    page.mouse.move(cx, cy)
    page.mouse.down()
    for dx in range(0, 120, 3):
        page.mouse.move(cx + dx, cy + (dx % 7))
    page.mouse.up()
    time.sleep(0.8)


def mark_tab2_point(page) -> None:
    from playwright.sync_api import expect

    panel = page.locator('[role="tabpanel"]').nth(1)
    target = panel.locator(".panel-card").first.locator(".checkerboard img, .checkerboard canvas").first
    target.wait_for(state="visible", timeout=10000)
    box = target.bounding_box()
    if not box:
        raise AssertionError("Tab2 source canvas has no bounding box")
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    expect(panel.locator("textarea[disabled]").first).to_have_value(
        re.compile(r"已添加正向标记"), timeout=60000,
    )


def install_fakes():
    app.on_auto_process = fake_auto_process
    tab2.on_generate_cutout = fake_generate_cutout


def main():
    install_fakes()
    port = free_port()
    demo = app.build_ui(model_concurrency_limit=1)
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name="127.0.0.1", server_port=port, inbrowser=False, quiet=True,
        share=False, allowed_paths=[str(ROOT / "output")], prevent_thread_lock=True,
        theme=gr.themes.Soft(), css=app.APP_CSS, js=app.APP_JS,
    )

    from playwright.sync_api import sync_playwright, expect

    url = f"http://127.0.0.1:{port}"
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda msg: errors.append(f"console {msg.type}: {msg.text}") if msg.type in ("error", "warning") else None)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector('button', timeout=60000)

        upload_img = ROOT / "output" / "ui-regression-upload.png"
        upload_img.parent.mkdir(exist_ok=True)
        Image.fromarray(_rgb((180, 210, 240)), "RGB").save(upload_img)
        upload_path = str(upload_img)
        page.locator('[role="tabpanel"]').nth(0).locator('input[type="file"]').first.set_input_files(upload_path)
        page.get_by_role("button", name="开始抠图").first.click()
        expect(page.locator('[role="tabpanel"]').nth(0).locator('textarea[disabled]').first).to_have_value(re.compile("tab1 fake complete"), timeout=20000)
        page.get_by_role("button", name="边缘修复").first.click()
        expect(page.locator('[role="tabpanel"]').nth(0).get_by_role("button", name="应用修复")).to_be_visible(timeout=10000)
        paint_refine_editor(page, panel_idx=0)
        page.locator('[role="tabpanel"]').nth(0).get_by_role("button", name="应用修复").click()
        expect(page.locator('[role="tabpanel"]').nth(0).locator('textarea[disabled]').first).to_have_value(
            re.compile(r"accept:\d+px"), timeout=60000,
        )
        page.get_by_role("button", name="退出修复").click()
        expect(page.get_by_role("button", name="边缘修复").first).to_be_visible(timeout=10000)

        page.get_by_role("tab", name="精细选区").click()
        page.wait_for_selector('input[type="file"]', state="attached", timeout=10000)
        page.locator('[role="tabpanel"]').nth(1).locator('input[type="file"]').first.set_input_files(upload_path)
        expect(page.locator('[role="tabpanel"]').nth(1).locator('textarea[disabled]').first).to_have_value(re.compile("已上传"), timeout=10000)
        mark_tab2_point(page)
        page.locator('[role="tabpanel"]').nth(1).get_by_role("button", name="开始抠图").click()
        expect(page.locator('[role="tabpanel"]').nth(1).locator('textarea[disabled]').first).to_have_value(re.compile("tab2 fake complete"), timeout=20000)
        page.locator('[role="tabpanel"]').nth(1).get_by_role("button", name="边缘修复").click()
        expect(page.locator('[role="tabpanel"]').nth(1).get_by_role("button", name="应用修复")).to_be_visible(timeout=10000)

        visible_editor_images = page.locator('[role="tabpanel"]').nth(1).locator('.refine-editor img, .refine-editor canvas').evaluate_all(
            "els => els.filter(e => { const s = getComputedStyle(e); return e.offsetParent !== null && s.visibility !== 'hidden' && s.display !== 'none' && (e.tagName === 'CANVAS' || e.currentSrc || e.src); }).length"
        )
        tab2_disabled_buttons = page.locator('[role="tabpanel"]').nth(1).locator('button:disabled').evaluate_all(
            "els => els.filter(e => e.offsetParent !== null && getComputedStyle(e).visibility !== 'hidden' && e.textContent.trim()).map(e => e.textContent.trim())"
        )
        page.locator('[role="tabpanel"]').nth(1).get_by_role("button", name="退出修复").click()
        expect(page.locator('[role="tabpanel"]').nth(1).get_by_role("button", name="边缘修复")).to_be_visible(timeout=10000)
        print({"visible_editor_images": visible_editor_images, "tab2_disabled_buttons": tab2_disabled_buttons, "errors": errors[:5]})
        if visible_editor_images < 1:
            raise AssertionError("Tab2 refine editor did not render an image/canvas")
        if tab2_disabled_buttons:
            raise AssertionError("Tab2 has disabled visible buttons: " + repr(tab2_disabled_buttons))
        browser.close()

    demo.close()


if __name__ == "__main__":
    main()
