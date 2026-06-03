"""
RMBG-2.0 引擎：自动抠图，配合 ViTMatte 做边缘精细化
"""
import time
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation

from .rgba_postprocess import make_clean_rgba


class RMBG2Engine:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load_model(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            print(f"[RMBG-2.0] loading model on {self.device} ...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.model.to(self.device)
            self.model.eval()
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[VRAM] RMBG-2.0 loaded - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
            print("[RMBG-2.0] model loaded")

    def remove_background(self, image: Image.Image, refiner, transparent_detector=None,
                          refine_mode: str = "auto", debug_dir: str = None) -> Image.Image:
        """推理 → 连通域清理 → ViTMatte 精修 → RGBA 输出
        debug_dir: 传入目录路径时，保存每一步中间结果用于诊断
        """
        img_input = self._preprocess(image)

        t0 = time.perf_counter()
        with self._infer_lock:
            self._load_model()
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = self.model(img_input)
                else:
                    result = self.model(img_input)
            mask_raw = self._postprocess(result, image.size)
        print(f"[RMBG-2.0] inference time {time.perf_counter() - t0:.2f}s")

        img_array = np.array(image.convert("RGB"))
        mask_clean = self._clean_mask(mask_raw)
        mask = self._smooth_edge(mask_clean, img_array)

        if debug_dir:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            self._dump_stats("RMBG原始", mask_raw)
            Image.fromarray(mask_raw, "L").save(os.path.join(debug_dir, "10_rmbg_raw_alpha.png"))
            self._dump_stats("RMBG清理后", mask_clean)
            Image.fromarray(mask_clean, "L").save(os.path.join(debug_dir, "11_rmbg_clean_alpha.png"))
            self._dump_stats("RMBG平滑后", mask)
            Image.fromarray(mask, "L").save(os.path.join(debug_dir, "12_rmbg_smooth_alpha.png"))

        if refiner is not None:
            mask = refiner.refine(image, mask, transparent_detector=transparent_detector,
                                  soft=True, mode=refine_mode, _debug_dir=debug_dir)

        if debug_dir:
            self._dump_stats("后处理输入alpha", mask)
            print(f"[诊断] 中间结果已保存到: {debug_dir}")

        rgba = make_clean_rgba(
            img_array,
            mask,
            debug_dir=debug_dir,
            preserve_transparency=(transparent_detector is not None),
        )
        return Image.fromarray(rgba, "RGBA")

    def predict_alpha(self, image: Image.Image, clean: bool = True,
                      smooth: bool = True) -> np.ndarray:
        """返回 RMBG soft alpha，供交互式 ROI 抠图流程做约束融合。"""
        img_input = self._preprocess(image)
        t0 = time.perf_counter()
        with self._infer_lock:
            self._load_model()
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = self.model(img_input)
                else:
                    result = self.model(img_input)
            alpha = self._postprocess(result, image.size)
        print(f"[RMBG-2.0] ROI inference time {time.perf_counter() - t0:.2f}s")

        if clean:
            alpha = self._clean_mask(alpha)
        if smooth:
            alpha = self._smooth_edge(alpha, np.array(image.convert("RGB")))
        return alpha

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        img = image.convert("RGB").resize((1024, 1024), Image.LANCZOS)
        img_tensor = TF.to_tensor(img).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        return img_tensor.to(self.device)

    def _postprocess(self, result, orig_size: tuple) -> np.ndarray:
        if isinstance(result, (list, tuple)):
            output = result[-1]
        else:
            output = result
        if output.dim() == 4:
            output = output.squeeze(0)
        if output.dim() == 3:
            output = output.squeeze(0)
        mask = torch.sigmoid(output.cpu()).float().numpy()
        mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)
        return (mask * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _dump_stats(name: str, alpha: np.ndarray):
        h, w = alpha.shape
        total = h * w
        bg = np.sum(alpha < 10) / total * 100
        fg = np.sum(alpha > 245) / total * 100
        edge = np.sum((alpha >= 10) & (alpha <= 245)) / total * 100
        mean_a = alpha.mean()
        # 边缘区 alpha 标准差（越大说明边缘越"毛糙"）
        edge_mask = (alpha >= 10) & (alpha <= 245)
        edge_std = alpha[edge_mask].std() if np.any(edge_mask) else 0
        # 最大连通域面积
        n, _, stats, _ = cv2.connectedComponentsWithStats((alpha > 127).astype(np.uint8))
        largest = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
        print(f"[诊断] {name}: {w}x{h} | bg={bg:.1f}% fg={fg:.1f}% edge={edge:.1f}% | "
              f"mean={mean_a:.0f} edge_std={edge_std:.1f} | 最大前景连通域={largest}px")

    @staticmethod
    def _clean_mask(alpha: np.ndarray) -> np.ndarray:
        """连通域清理：去除前景噪点和孤立淡残影，保留内部背景空隙。"""
        h, w = alpha.shape
        total = h * w
        # 阈值：噪点 < 图片面积的 0.02%（1080p≈460px, 4K≈1840px）
        noise_thresh = max(50, int(total * 0.0002))
        # 极低 alpha 在合成时容易变成脏边/残影，直接归零。
        haze_floor = 10
        # 孤立淡残影面积阈值，主要清掉没有实心主体支撑的小片 alpha。
        haze_area_thresh = max(200, int(total * 0.001))

        alpha[alpha < haze_floor] = 0

        bin_mask = (alpha > 127).astype(np.uint8)

        # 去噪点：删除面积 < noise_thresh 的前景连通域
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        noise_labels = [
            i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] < noise_thresh
        ]
        if noise_labels:
            alpha[np.isin(labels, noise_labels)] = 0

        # 清掉没有实心前景支撑的孤立淡 alpha；边缘和发丝若连着主体会保留。
        solid_mask = alpha > 127
        if not np.any(solid_mask):
            return alpha

        support_mask = (alpha > haze_floor).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(support_mask, connectivity=8)
        solid_labels = set(np.unique(labels[solid_mask]).tolist())
        haze_labels = []
        for i in range(1, n):
            if i in solid_labels:
                continue
            vals = alpha[labels == i]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < haze_area_thresh or vals.max() < 120 or vals.mean() < 64:
                haze_labels.append(i)
        if haze_labels:
            alpha[np.isin(labels, haze_labels)] = 0

        return alpha

    @staticmethod
    def _smooth_edge(alpha: np.ndarray, img: np.ndarray = None) -> np.ndarray:
        """
        边缘平滑：只模糊边缘过渡区（10<alpha<245），前景/背景核心区不动。
        避免模糊前景表面产生光晕。
        """
        blurred = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)
        edge = (alpha > 32) & (alpha < 224)
        result = alpha.astype(np.float32)
        result[edge] = result[edge] * 0.65 + blurred[edge] * 0.35
        return np.clip(result, 0, 255).astype(np.uint8)

    def cleanup(self):
        with self._infer_lock, self._load_lock:
            if self.model is not None:
                del self.model
                self.model = None
