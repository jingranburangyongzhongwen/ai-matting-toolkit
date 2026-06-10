"""Benchmark Tab2 SAM auto segmentation and click latency.

The script is intentionally result-aware: it records mask hashes so a later
optimization can be compared against the baseline for exact output matches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image


POSITIVE_MODE = "\u6b63\u5411\u9009\u53d6\uff08\u6211\u8981\uff09"
tab2 = None


@dataclass
class FakeSelectData:
    index: tuple[int, int]


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _cuda_memory() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
    except Exception:
        return None


def _time_call(fn, *, sync_cuda: bool = True):
    gc.collect()
    if sync_cuda:
        _cuda_sync()
    t0 = time.perf_counter()
    value = fn()
    if sync_cuda:
        _cuda_sync()
    return time.perf_counter() - t0, value


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _discover_images() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    ignored_parts = {"output", "models", ".git", "__pycache__"}
    images = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel_parts = set(path.relative_to(ROOT).parts)
        if rel_parts & ignored_parts:
            continue
        images.append(path)
    return sorted(images, key=lambda p: str(p.relative_to(ROOT)).lower())


def _load_app_modules():
    global tab2
    if tab2 is not None:
        return tab2
    from app_logic import tab2 as tab2_module

    tab2 = tab2_module
    return tab2


def _engine_mode_for(engine_type: str) -> str:
    tab2_module = _load_app_modules()
    for label, mapped in tab2_module.ENGINE_MODE_MAP.items():
        if mapped == engine_type:
            return label
    raise ValueError(f"Unknown engine type: {engine_type}")


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_hash(mask: np.ndarray) -> str:
    arr = np.ascontiguousarray(mask.astype(np.bool_, copy=False))
    return hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()


def _mask_signature(mask: np.ndarray) -> dict[str, Any]:
    bool_mask = np.asarray(mask, dtype=bool)
    return {
        "shape": list(bool_mask.shape),
        "area": int(bool_mask.sum()),
        "bbox": _mask_bbox(bool_mask),
        "hash": _mask_hash(bool_mask),
    }


def _auto_mask_signature(mask_info: dict[str, Any]) -> dict[str, Any]:
    sig = _mask_signature(mask_info["segmentation"])
    for key in ("area", "predicted_iou", "stability_score"):
        if key in mask_info:
            value = mask_info[key]
            sig[key] = float(value) if isinstance(value, np.floating) else value
    return sig


def _pick_point(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0]
    mid = xs.size // 2
    return [int(xs[mid]), int(ys[mid])]


def _parse_points(raw: str | None, image_shape: tuple[int, int, int]) -> list[list[int]]:
    if raw:
        points = []
        for item in raw.split(";"):
            if not item.strip():
                continue
            x_str, y_str = item.split(",", 1)
            points.append([int(x_str), int(y_str)])
        return points
    h, w = image_shape[:2]
    return [
        [w // 2, h // 2],
        [w // 3, h // 2],
        [(w * 2) // 3, h // 2],
    ]


def _predict_signature(image, engine_mode, points, labels, box_state, auto_masks, auto_choice):
    tab2_module = _load_app_modules()
    ctx = tab2_module._ensure_sam_ready(image, engine_mode, retain=True)
    try:
        mask = tab2_module._predict_tab2_mask(
            ctx,
            points,
            labels,
            box_state,
            auto_masks,
            auto_choice,
            image_shape=image.shape,
        )
        return _mask_signature(mask)
    finally:
        tab2_module._release_sam_context(ctx)


def run_once(args, image: np.ndarray, engine_mode: str) -> dict[str, Any]:
    tab2_module = _load_app_modules()
    timings: dict[str, float] = {}
    result: dict[str, Any] = {
        "timings_sec": timings,
        "cuda_memory_start": _cuda_memory(),
    }

    def ensure_ready():
        return tab2_module._ensure_sam_ready(image, engine_mode, retain=True)

    timings["ensure_ready_current"] , ctx = _time_call(ensure_ready)
    try:
        timings["auto_generate_current"], masks_raw = _time_call(ctx["sam"].auto_segment)
    finally:
        tab2_module._release_sam_context(ctx)

    timings["auto_postprocess"], masks = _time_call(
        lambda: tab2_module._postprocess_auto_masks(masks_raw)
    )
    timings["auto_overlay"], _ = _time_call(
        lambda: tab2_module._draw_auto_segment_overlay(image, masks)
    )

    result["auto"] = {
        "raw_count": len(masks_raw),
        "kept_count": len(masks),
        "signatures": [_auto_mask_signature(m) for m in masks[: args.signature_limit]],
    }

    auto_points = [_pick_point(m["segmentation"]) for m in masks[: args.auto_clicks]]
    points_state: list[list[int]] = []
    labels_state: list[int] = []
    box_state = None
    auto_choice_state = tab2_module._ensure_auto_choice({"selected": [], "excluded": []})
    auto_click_results = []

    for idx, point in enumerate(auto_points):
        def click_auto():
            return tab2_module.on_image_click(
                image,
                FakeSelectData(index=(point[0], point[1])),
                POSITIVE_MODE,
                engine_mode,
                points_state,
                labels_state,
                box_state,
                masks,
                auto_choice_state,
            )

        elapsed, ret = _time_call(click_auto)
        (
            _overlay,
            _view_update,
            _download_update,
            points_state,
            labels_state,
            box_state,
            auto_choice_state,
            status,
        ) = ret
        click_entry = {
            "index": idx,
            "point": point,
            "elapsed_sec": elapsed,
            "points_count": len(points_state or []),
            "selected": list(auto_choice_state.get("selected", [])),
            "excluded": list(auto_choice_state.get("excluded", [])),
            "status": str(status),
        }
        if not args.no_click_signatures:
            sig_time, sig = _time_call(
                lambda: _predict_signature(
                    image,
                    engine_mode,
                    points_state,
                    labels_state,
                    box_state,
                    masks,
                    auto_choice_state,
                )
            )
            click_entry["signature_time_sec"] = sig_time
            click_entry["mask_signature"] = sig
        auto_click_results.append(click_entry)

    result["auto_clicks"] = auto_click_results

    free_click_results = []
    points_state = []
    labels_state = []
    box_state = None
    free_points = _parse_points(args.free_points, image.shape)[: args.free_clicks]
    for idx, point in enumerate(free_points):
        def click_free():
            return tab2_module.on_image_click(
                image,
                FakeSelectData(index=(point[0], point[1])),
                POSITIVE_MODE,
                engine_mode,
                points_state,
                labels_state,
                box_state,
                [],
                tab2_module._ensure_auto_choice({"selected": [], "excluded": []}),
            )

        elapsed, ret = _time_call(click_free)
        (
            _overlay,
            _view_update,
            _download_update,
            points_state,
            labels_state,
            box_state,
            _auto_choice_state,
            status,
        ) = ret
        free_click_results.append(
            {
                "index": idx,
                "point": point,
                "elapsed_sec": elapsed,
                "points_count": len(points_state or []),
                "status": str(status),
            }
        )
    result["free_clicks"] = free_click_results
    result["cuda_memory_end"] = _cuda_memory()
    return result


def compare(left_path: Path, right_path: Path) -> int:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    problems = []

    left_cases = {case["image"]: case for case in left.get("cases", [])}
    right_cases = {case["image"]: case for case in right.get("cases", [])}
    if set(left_cases) != set(right_cases):
        problems.append("image set differs")

    for image_key in sorted(set(left_cases) & set(right_cases)):
        left_run = left_cases[image_key]["runs"][0]
        right_run = right_cases[image_key]["runs"][0]
        if left_run["auto"]["signatures"] != right_run["auto"]["signatures"]:
            problems.append(f"{image_key}: auto mask signatures differ")

        l_clicks = left_run.get("auto_clicks", [])
        r_clicks = right_run.get("auto_clicks", [])
        if len(l_clicks) != len(r_clicks):
            problems.append(f"{image_key}: auto click count differs")
        for idx, (l_item, r_item) in enumerate(zip(l_clicks, r_clicks)):
            if l_item.get("mask_signature") != r_item.get("mask_signature"):
                problems.append(f"{image_key}: auto click {idx} mask signature differs")

    print(f"left:  {left_path}")
    print(f"right: {right_path}")
    if problems:
        print("RESULT: DIFFERENT")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("RESULT: EXACT MATCH for recorded signatures")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Images to benchmark. Defaults to all repo images.",
    )
    parser.add_argument("--engine", choices=["sam_hq", "mobile_sam"], default="sam_hq")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--auto-clicks", type=int, default=3)
    parser.add_argument("--free-clicks", type=int, default=3)
    parser.add_argument("--free-points", help="Semicolon-separated x,y points")
    parser.add_argument("--signature-limit", type=int, default=80)
    parser.add_argument(
        "--no-click-signatures",
        action="store_true",
        help="Skip extra mask signatures for faster timing-only runs.",
    )
    parser.add_argument("--multi-session", action="store_true")
    parser.add_argument("--max-sam-sessions", type=int, default=1)
    parser.add_argument("--out", type=Path, help="Write JSON report")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])

    image_paths = args.images or _discover_images()
    if not image_paths:
        parser.error("no images found; pass image paths explicitly")

    from model_manager import ModelManager

    tab2_module = _load_app_modules()
    mgr = ModelManager()
    tab2_module.init(mgr)
    tab2_module.configure_runtime(args.multi_session, args.max_sam_sessions)
    engine_mode = _engine_mode_for(args.engine)

    default_out = (
        ROOT
        / "tmp"
        / f"sam_tab2_{args.engine}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    out_path = args.out or default_out
    report: dict[str, Any] = {
        "engine": args.engine,
        "repeat": args.repeat,
        "auto_clicks": args.auto_clicks,
        "free_clicks": args.free_clicks,
        "click_signatures": not args.no_click_signatures,
        "cases": [],
    }
    for image_idx, image_path in enumerate(image_paths):
        image_path = image_path.resolve()
        image = _load_image(image_path)
        case = {
            "image": _display_path(image_path),
            "image_shape": list(image.shape),
            "runs": [],
        }
        print(
            f"[bench] image {image_idx + 1}/{len(image_paths)} "
            f"{case['image']} shape={tuple(image.shape)}"
        )
        for run_idx in range(args.repeat):
            print(f"[bench]   run {run_idx + 1}/{args.repeat} engine={args.engine}")
            case["runs"].append(run_once(args, image, engine_mode))
        report["cases"].append(case)

        first = case["runs"][0]
        print("[bench]   timings_sec:")
        for key, value in first["timings_sec"].items():
            print(f"    {key}: {value:.4f}")
        print(f"[bench]   auto masks: {first['auto']['kept_count']} kept")
        if first["auto_clicks"]:
            avg_auto = statistics.mean(c["elapsed_sec"] for c in first["auto_clicks"])
            print(f"[bench]   auto click avg: {avg_auto:.4f}s")
        if first["free_clicks"]:
            avg_free = statistics.mean(c["elapsed_sec"] for c in first["free_clicks"])
            print(f"[bench]   free click avg: {avg_free:.4f}s")

    out_text = json.dumps(report, ensure_ascii=False, indent=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
