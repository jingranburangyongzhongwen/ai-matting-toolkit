"""
端到端抠图质量回归测试。

用法:
    python scripts/benchmark.py save  --input <图片目录>
    python scripts/benchmark.py compare --input <图片目录>
    python scripts/benchmark.py run   --input <图片目录>

管线覆盖:
  Tab1: RMBG-2.0 自动抠图（raw alpha → 后处理 → final RGBA）
  Tab2: SAM 交互抠图（SAM mask → alpha 生成 → 后处理 → final RGBA）

评价分层:
  模型输出: raw alpha 的质量（分布、边缘）
  后处理效果: 后处理对 alpha 的改变量（solid loss、alpha delta）
  最终输出: final RGBA 的合成质量
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from log import get_logger
from model_manager import ModelManager, get_output_path

logger = get_logger("benchmark")
BASELINE_PATH = ROOT / "scripts" / "benchmark_baseline.json"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_images(input_path: str) -> list:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() in VALID_EXTS:
        return [str(p)]
    if p.is_dir():
        files = []
        for ext in VALID_EXTS:
            files.extend(glob.glob(str(p / f"*{ext}")))
            files.extend(glob.glob(str(p / f"*{ext.upper()}")))
        return sorted(set(files))
    return []


# ── 无参考指标 ────────────────────────────────────────────────────

def alpha_dist(alpha: np.ndarray) -> dict:
    total = alpha.size
    return {
        "solid_pct": round(float(np.sum(alpha > 245) / total * 100), 2),
        "transparent_pct": round(float(np.sum(alpha < 10) / total * 100), 2),
        "transition_pct": round(float(np.sum((alpha >= 10) & (alpha <= 245)) / total * 100), 2),
    }


def lap_p95(alpha: np.ndarray) -> float:
    a = alpha.astype(np.float32) / 255.0
    lap = np.abs(cv2.Laplacian(a, cv2.CV_32F))
    edge = (alpha > 10) & (alpha < 245)
    if not np.any(edge):
        return 0.0
    return round(float(np.percentile(lap[edge], 95)), 4)


def fg_preservation(alpha_before: np.ndarray, alpha_after: np.ndarray) -> float:
    """前景保留度：原始 alpha>127 的像素，处理后仍 >127 的比例。"""
    fg = alpha_before > 127
    if not np.any(fg):
        return 1.0
    return round(float(np.mean(alpha_after[fg] > 127)), 4)


def bg_cleanliness(alpha_after: np.ndarray) -> float:
    """背景干净度：alpha<10 的区域越接近 0 越好。"""
    bg = alpha_after < 10
    if not np.any(bg):
        return 1.0
    return round(float(1.0 - np.mean(alpha_after[bg].astype(float) / 255.0)), 4)


def solid_loss(alpha_before: np.ndarray, alpha_after: np.ndarray) -> float:
    """实心区域损失率：原始 >127 的像素，处理后降到 ≤127 的比例。"""
    solid = alpha_before > 127
    if not np.any(solid):
        return 0.0
    lost = np.sum(solid & (alpha_after <= 127))
    return round(float(lost / np.sum(solid) * 100), 3)


def alpha_l1(alpha_before: np.ndarray, alpha_after: np.ndarray) -> float:
    """Alpha 平均绝对变化量（归一化到 0-1）。"""
    return round(float(np.mean(np.abs(alpha_after.astype(float) - alpha_before.astype(float))) / 255.0), 5)


def composite_spill(rgba: np.ndarray) -> float:
    """合成溢色检测：透明区域合成到绿色背景后应接近绿色。"""
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    bg_mask = alpha < 10
    if not np.any(bg_mask):
        return 1.0
    bg_color = np.array([0, 180, 0], dtype=np.float32)
    a_f = alpha.astype(np.float32) / 255.0
    comp = rgb.astype(np.float32) * a_f[..., None] + bg_color * (1 - a_f[..., None])
    diff = np.linalg.norm(comp[bg_mask] - bg_color, axis=1).mean()
    return round(float(1.0 - min(diff / 128.0, 1.0)), 4)


# ── 标准 Matting 指标（需要 Ground Truth） ────────────────────────

def matting_sad(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sum(np.abs(pred.astype(np.float64) - gt.astype(np.float64))) / 1000.0)


def matting_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))


def matting_grad(pred: np.ndarray, gt: np.ndarray) -> float:
    def _grad_mag(a):
        a_f = a.astype(np.float32) / 255.0
        gx = cv2.Sobel(a_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a_f, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    return float(np.mean(np.abs(_grad_mag(pred) - _grad_mag(gt))))


def matting_conn(pred: np.ndarray, gt: np.ndarray, n_levels: int = 10) -> float:
    thresholds = np.linspace(0, 255, n_levels + 1)[1:-1]
    total_err = 0.0
    for t in thresholds:
        pred_bin = (pred >= t).astype(np.uint8)
        gt_bin = (gt >= t).astype(np.uint8)
        n_gt, labels_gt = cv2.connectedComponents(gt_bin)
        n_pred, labels_pred = cv2.connectedComponents(pred_bin)
        for label_id in range(1, n_gt):
            gt_mask = labels_gt == label_id
            overlapping = labels_pred[gt_mask]
            if len(overlapping) == 0:
                continue
            majority = np.bincount(overlapping[overlapping > 0]).argmax()
            total_err += float(np.sum(overlapping != majority))
    return total_err / max(pred.size * len(thresholds), 1)


def compute_matting_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        "SAD": matting_sad(pred, gt),
        "MSE": matting_mse(pred, gt),
        "Grad": matting_grad(pred, gt),
        "Conn": matting_conn(pred, gt),
    }


# ── LPIPS 合成评估 ────────────────────────────────────────────────

_lpips_loss_fn = None

def compute_lpips_score(rgba: np.ndarray) -> dict:
    """LPIPS 合成一致性：同一 alpha 合成到不同背景，感知差异应小。"""
    global _lpips_loss_fn
    try:
        import torch
        local_hub = os.path.join(str(ROOT), "models", "lpips")
        if os.path.isdir(local_hub):
            torch.hub.set_dir(local_hub)
        import lpips
    except ImportError:
        return {"status": "skipped"}

    if _lpips_loss_fn is None:
        _lpips_loss_fn = lpips.LPIPS(net="alex", verbose=False)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    h, w = alpha.shape

    bgs = [np.full((h, w, 3), c, dtype=np.uint8)
           for c in [(0, 0, 0), (255, 255, 255), (128, 128, 128), (0, 128, 255)]]
    a_f = alpha.astype(np.float32) / 255.0
    composites = [np.clip(rgb.astype(np.float32) * a_f[..., None] + bg.astype(np.float32) * (1 - a_f[..., None]), 0, 255).astype(np.uint8)
                  for bg in bgs]

    def to_t(img):
        return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

    scores = []
    with torch.no_grad():
        for i in range(len(composites)):
            for j in range(i + 1, len(composites)):
                scores.append(round(float(_lpips_loss_fn(to_t(composites[i]), to_t(composites[j])).item()), 4))

    return {"pairwise_mean": round(float(np.mean(scores)), 4), "pairwise_max": round(float(np.max(scores)), 4)}


# ── 指标汇总 ──────────────────────────────────────────────────────

def evaluate_raw_alpha(alpha_raw: np.ndarray) -> dict:
    """评价模型原始输出（后处理前）。"""
    return {
        "dist": alpha_dist(alpha_raw),
        "lap_p95": lap_p95(alpha_raw),
    }


def evaluate_postprocess_effect(alpha_raw: np.ndarray, alpha_final: np.ndarray) -> dict:
    """评价后处理对 alpha 的影响。"""
    return {
        "solid_loss_pct": solid_loss(alpha_raw, alpha_final),
        "fg_preservation": fg_preservation(alpha_raw, alpha_final),
        "bg_cleanliness": bg_cleanliness(alpha_final),
        "alpha_l1": alpha_l1(alpha_raw, alpha_final),
        "dist": alpha_dist(alpha_final),
        "lap_p95": lap_p95(alpha_final),
    }


def evaluate_final_output(rgba: np.ndarray) -> dict:
    """评价最终输出的合成质量。"""
    return {
        "spill_score": composite_spill(rgba),
        "lpips": compute_lpips_score(rgba),
    }


# ── Tab1: RMBG-2.0 管线 ──────────────────────────────────────────

def run_tab1(image_path: str, mgr: ModelManager) -> dict:
    """RMBG-2.0 自动抠图：分别评价 raw alpha 和后处理效果。"""
    img = Image.open(image_path).convert("RGB")

    # 获取 raw alpha（后处理前）
    t0 = time.perf_counter()
    alpha_raw = mgr.rmbg2.predict_alpha(img, clean=True, smooth=True)
    raw_time = time.perf_counter() - t0

    # 获取 final RGBA（完整管线含后处理）
    t1 = time.perf_counter()
    rgba_pil = mgr.rmbg2.remove_background(img, refiner=None)
    total_time = time.perf_counter() - t1

    rgba = np.array(rgba_pil.convert("RGBA"))
    alpha_final = rgba[:, :, 3]

    return {
        "raw": evaluate_raw_alpha(alpha_raw),
        "postprocess": evaluate_postprocess_effect(alpha_raw, alpha_final),
        "final": evaluate_final_output(rgba),
        "raw_time_s": round(raw_time, 3),
        "total_time_s": round(total_time, 3),
    }


# ── Tab2: SAM 端到端（auto_segment 不需要手动标点） ───────────────

def run_tab2_e2e(image_path: str, mgr: ModelManager, engine_mode: str = "mobile_sam") -> dict:
    """SAM 端到端：auto_segment → 选最大 mask → _sam_strict_alpha → 后处理。"""
    from app_logic.tab2 import ENGINE_MODE_MAP, _sam_strict_alpha
    from engines.rgba_postprocess import make_clean_rgba

    img_np = np.array(Image.open(image_path).convert("RGB"))
    engine_type = ENGINE_MODE_MAP.get(engine_mode, "mobile_sam")

    # 加载 SAM 模型并 set_image
    engine = mgr.get_sam_engine(engine_type)
    sam = engine.create_session()
    sam.set_image(img_np)

    # auto_segment：自动分割所有主体，不需要手动标点
    t0 = time.perf_counter()
    masks = sam.auto_segment(points_per_side=32)
    auto_time = time.perf_counter() - t0

    if not masks:
        sam.cleanup()
        return {"error": "auto_segment returned no masks"}

    # 选最大的 mask
    best = max(masks, key=lambda m: m.get("area", 0))
    sam_mask = np.asarray(best["segmentation"], dtype=bool)

    # 生成 alpha
    t1 = time.perf_counter()
    alpha_raw, subject_box = _sam_strict_alpha(sam_mask, [], [], None, img_np.shape)
    alpha_time = time.perf_counter() - t1

    # 后处理
    t2 = time.perf_counter()
    rgba = make_clean_rgba(img_np, alpha_raw)
    pp_time = time.perf_counter() - t2

    alpha_final = rgba[:, :, 3]
    sam.cleanup()

    return {
        "raw": evaluate_raw_alpha(alpha_raw),
        "postprocess": evaluate_postprocess_effect(alpha_raw, alpha_final),
        "final": evaluate_final_output(rgba),
        "num_masks": len(masks),
        "auto_time_s": round(auto_time, 3),
        "alpha_time_s": round(alpha_time, 3),
        "pp_time_s": round(pp_time, 3),
    }


# ── 主流程 ────────────────────────────────────────────────────────

def run_benchmark(image_paths: list, mgr: ModelManager) -> dict:
    results = {}

    # Tab1: RMBG-2.0 端到端
    print("\n── Tab1: RMBG-2.0 自动抠图（端到端）──")
    for i, path in enumerate(image_paths):
        name = os.path.basename(path)
        print(f"  [{i + 1}/{len(image_paths)}] {name} ...", end=" ", flush=True)
        try:
            tab1 = run_tab1(path, mgr)
            r = tab1["raw"]["dist"]
            p = tab1["postprocess"]
            print(f"raw_solid={r['solid_pct']:.1f}%  solid_loss={p['solid_loss_pct']:.2f}%  "
                  f"fg={p['fg_preservation']:.3f}  {tab1['total_time_s']:.1f}s")
            results.setdefault(name, {})["tab1"] = tab1
        except Exception as e:
            print(f"ERROR: {e}")
            results.setdefault(name, {})["tab1"] = {"error": str(e)}

    # Tab2: SAM 端到端（两个引擎都跑）
    for engine_name in ("mobile_sam", "sam_hq"):
        print(f"\n── Tab2: SAM 端到端（engine={engine_name}）──")
        for i, path in enumerate(image_paths):
            name = os.path.basename(path)
            print(f"  [{i + 1}/{len(image_paths)}] {name} ...", end=" ", flush=True)
            try:
                tab2 = run_tab2_e2e(path, mgr, engine_mode=engine_name)
                if "error" in tab2:
                    print(f"ERROR: {tab2['error']}")
                else:
                    r = tab2["raw"]["dist"]
                    p = tab2["postprocess"]
                    print(f"masks={tab2['num_masks']}  raw_solid={r['solid_pct']:.1f}%  "
                          f"solid_loss={p['solid_loss_pct']:.2f}%  fg={p['fg_preservation']:.3f}  "
                          f"{tab2['auto_time_s'] + tab2['pp_time_s']:.1f}s")
                results.setdefault(name, {})[f"tab2_{engine_name}"] = tab2
            except Exception as e:
                print(f"ERROR: {e}")
                results.setdefault(name, {})[f"tab2_{engine_name}"] = {"error": str(e)}

    return results


def save_baseline(results: dict):
    data = {
        "version": 3,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_images": len(results),
        "results": results,
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n基线已保存: {BASELINE_PATH}")


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _compare_metrics(name: str, pipeline: str, cur: dict, old: dict) -> list:
    """对比单个管线的指标，返回回归列表。"""
    regressions = []
    if "error" in cur or "error" in old:
        return regressions

    prefix = f"  {name}/{pipeline}"

    # 后处理指标
    cp = cur.get("postprocess", {})
    op = old.get("postprocess", {})
    checks = [
        ("fg_preservation", True, 0.02),
        ("bg_cleanliness", True, 0.01),
        ("solid_loss_pct", False, 0.5),
        ("alpha_l1", False, 0.001),
    ]
    for metric, higher_better, threshold in checks:
        c, o = cp.get(metric), op.get(metric)
        if c is None or o is None:
            continue
        delta = c - o
        status = ""
        if higher_better and delta < -threshold:
            status = "REGRESS"
            regressions.append((f"{name}/{pipeline}", metric, o, c))
        elif not higher_better and delta > threshold:
            status = "REGRESS"
            regressions.append((f"{name}/{pipeline}", metric, o, c))
        elif higher_better and delta > threshold:
            status = "IMPROVE"
        elif not higher_better and delta < -threshold:
            status = "IMPROVE"
        if status or abs(delta) > threshold * 0.3:
            print(f"{prefix:<40} {metric:<18} {o:>8.4f} {c:>8.4f} {delta:>+8.4f} {status:>8}")

    # 耗时监控
    for time_key in ("total_time_s", "pp_time_s", "raw_time_s", "alpha_time_s"):
        c, o = cur.get(time_key), old.get(time_key)
        if c is not None and o is not None and o > 0:
            ratio = c / o
            if ratio > 1.5:
                print(f"{prefix:<40} {time_key:<18} {o:>8.3f} {c:>8.3f} {c - o:>+8.3f} SLOWER")

    return regressions


def compare_with_baseline(current: dict) -> bool:
    baseline = load_baseline()
    if baseline is None:
        print("没有找到基线。先运行: python scripts/benchmark.py save --input <图片目录>")
        return False

    prev = baseline["results"]
    print(f"\n与基线对比（{baseline['timestamp']}）:")
    print(f"{'Image/Pipeline':<40} {'Metric':<18} {'Base':>8} {'Now':>8} {'Delta':>8} {'':>8}")
    print("-" * 90)

    regressions = []
    for fname in current:
        if fname not in prev:
            print(f"  {fname}: 新增（无基线）")
            continue
        for pipeline in ("tab1", "tab2_mobile_sam", "tab2_sam_hq"):
            cur_p = current[fname].get(pipeline)
            old_p = prev[fname].get(pipeline)
            if not cur_p or not old_p:
                continue
            regs = _compare_metrics(fname, pipeline, cur_p, old_p)
            regressions.extend(regs)

    if regressions:
        print(f"\n[WARN] 发现 {len(regressions)} 处回归:")
        for case, metric, old_val, new_val in regressions:
            print(f"  - {case}: {metric} {old_val:.4f} -> {new_val:.4f}")
        return False
    print(f"\n[OK] 无回归。")
    return True


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="端到端抠图质量回归测试")
    parser.add_argument("command", choices=["save", "compare", "run"])
    parser.add_argument("--input", required=True, help="测试图片目录或单张图片")
    args = parser.parse_args()

    image_paths = collect_images(args.input)
    if not image_paths:
        print(f"未找到图片: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print(f"端到端抠图质量测试 ({args.command})")
    print(f"图片数: {len(image_paths)}")
    print("=" * 60)

    mgr = ModelManager()
    current = run_benchmark(image_paths, mgr)

    if args.command == "save":
        save_baseline(current)
    elif args.command == "compare":
        ok = compare_with_baseline(current)
        sys.exit(0 if ok else 1)
    elif args.command == "run":
        print("\n运行完成。")


if __name__ == "__main__":
    main()
