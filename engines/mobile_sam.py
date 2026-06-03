"""
MobileSAM 引擎：轻量快速交互式点击分割 + 实时蒙版
缓存 image_embedding 实现秒级响应
"""
import os
import numpy as np
import cv2


class MobileSAMEngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = None
        self.predictor = None
        self.model_path = model_path
        self._image_set = False
        self._original_size = None
        # 交互式细化用的上一次低分辨率 mask 先验
        self._prev_logits = None
        self._prev_npoints = 0
        self._cached_mask = None

    def _load_model(self):
        if self.model is not None:
            return
        print(f"[MobileSAM] 加载模型到 {self.device} ...")
        from mobile_sam import sam_model_registry, SamPredictor

        checkpoint = self._find_checkpoint()
        self.model = sam_model_registry["vit_t"](checkpoint=checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] MobileSAM loaded — allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
        print("[MobileSAM] 模型加载完成")

    def _find_checkpoint(self) -> str:
        """查找 MobileSAM 模型文件"""
        for name in ["mobile_sam.pt", "mobile_sam.pth", "sam_vit_t.pth"]:
            path = os.path.join(self.model_path, name)
            if os.path.exists(path):
                return path
        for f in os.listdir(self.model_path):
            if f.endswith((".pt", ".pth")):
                return os.path.join(self.model_path, f)
        raise FileNotFoundError(
            f"在 {self.model_path} 中未找到 MobileSAM 模型文件"
        )

    def set_image(self, image: np.ndarray):
        self._load_model()
        self._original_size = image.shape[:2]
        self.predictor.set_image(image)
        self._image_set = True
        self._prev_logits = None
        self._prev_npoints = 0
        self._cached_mask = None
        print(f"[MobileSAM] 图像特征已缓存, 尺寸: {self._original_size}")

    def predict_mask(self, point_coords: list, point_labels: list, box=None) -> np.ndarray:
        if not self._image_set:
            raise RuntimeError("请先调用 set_image() 设置图像")
        has_points = len(point_coords) > 0
        if not has_points and box is None:
            self._prev_logits = None
            self._prev_npoints = 0
            return np.zeros(self._original_size, dtype=bool)
        coords = np.array(point_coords) if has_points else None
        labels = np.array(point_labels) if has_points else None
        box_arr = np.array(box) if box is not None else None

        # 交互式细化：有先验就用，让预测结果与交互 overlay 保持一致。
        using_prev = self._prev_logits is not None
        mask_input = self._prev_logits[None, :, :] if using_prev else None

        # multimask 仅用于"首个单正向点"消歧（部件 vs 整体）；有 box / 回喂先验 /
        # 多点时用单 mask 输出，避免 argmax 在候选间跳变。
        single_pos_point = (
            box_arr is None and not using_prev and labels is not None
            and len(labels) == 1 and labels[0] == 1
        )
        masks, scores, low_res = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box_arr,
            mask_input=mask_input,
            multimask_output=single_pos_point,
        )
        idx = int(scores.argmax()) if single_pos_point else 0
        self._prev_logits = low_res[idx]
        self._prev_npoints = len(point_coords)
        self._cached_mask = masks[idx]
        return self._cached_mask

    def predict_and_overlay(
        self,
        image: np.ndarray,
        point_coords: list,
        point_labels: list,
        box=None,
        mask_color: tuple = (255, 0, 0),
        opacity: float = 0.4,
    ) -> np.ndarray:
        mask = self.predict_mask(point_coords, point_labels, box=box)
        overlay = image.copy().astype(np.float32)
        color_array = np.array(mask_color, dtype=np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - opacity) + color_array[c] * opacity,
                overlay[:, :, c],
            )
        return self._draw_points(overlay, point_coords, point_labels)

    def _draw_points(self, image: np.ndarray, point_coords: list, point_labels: list) -> np.ndarray:
        img = image.clip(0, 255).astype(np.uint8).copy()
        for coord, label in zip(point_coords, point_labels):
            x, y = int(coord[0]), int(coord[1])
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 8, (255, 255, 255), 2)
        return img

    def cleanup(self):
        self._image_set = False
        self._original_size = None
        if self.predictor is not None:
            self.predictor.reset_image()
