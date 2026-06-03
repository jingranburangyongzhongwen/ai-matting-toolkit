"""
SAM-HQ 引擎：交互式点击分割 + 实时蒙版（高质量边缘）
缓存 image_embedding 实现秒级响应
"""
import os
import numpy as np
import cv2


class SAMHQEngine:
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
        print(f"[SAM-HQ] 加载模型到 {self.device} ...")
        from segment_anything_hq import sam_model_registry, SamPredictor

        checkpoint = self._find_checkpoint()
        self.model = sam_model_registry["vit_l"](checkpoint=checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] SAM-HQ loaded — allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
        print("[SAM-HQ] 模型加载完成")

    def _find_checkpoint(self) -> str:
        """查找 SAM-HQ 模型文件"""
        for name in [
            "sam_hq_vit_l.pth",
            "sam_hq_vit_b.pth",
            "sam_hq_vit_h.pth",
        ]:
            path = os.path.join(self.model_path, name)
            if os.path.exists(path):
                return path
        for f in os.listdir(self.model_path):
            if f.endswith((".pt", ".pth")):
                return os.path.join(self.model_path, f)
        raise FileNotFoundError(
            f"在 {self.model_path} 中未找到 SAM-HQ 模型文件"
        )

    def set_image(self, image: np.ndarray):
        """
        设置当前图像并缓存 image_embedding
        只在图片上传/切换时调用一次
        Args:
            image: RGB numpy array (H, W, 3)
        """
        self._load_model()
        self._original_size = image.shape[:2]
        self.predictor.set_image(image)
        self._image_set = True
        self._prev_logits = None
        self._prev_npoints = 0
        self._cached_mask = None
        print(f"[SAM-HQ] 图像特征已缓存, 尺寸: {self._original_size}")

    def predict_mask(
        self,
        point_coords: list,
        point_labels: list,
        box=None,
    ) -> np.ndarray:
        """
        根据点击坐标 / 边界框预测 mask（只跑 Decoder，速度极快）
        Args:
            point_coords: [[x1,y1], [x2,y2], ...] 像素坐标
            point_labels: [1, 0, ...] 1=正向选取, 0=负向排除
            box: 可选 [x1, y1, x2, y2] 文本定位给出的初始边界框
        Returns:
            HxW bool numpy array
        """
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

        masks, scores, low_res = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box_arr,
            mask_input=mask_input,
            multimask_output=False,
            hq_token_only=False,
        )
        self._prev_logits = low_res[0]
        self._prev_npoints = len(point_coords)
        self._cached_mask = masks[0]
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
                overlay[:, :, c]
            )

        return self._draw_points(overlay, point_coords, point_labels)

    def _draw_points(
        self,
        image: np.ndarray,
        point_coords: list,
        point_labels: list,
    ) -> np.ndarray:
        """在图上绘制点击标记点（需要 uint8 输入）"""
        img = image.clip(0, 255).astype(np.uint8).copy()
        for coord, label in zip(point_coords, point_labels):
            x, y = int(coord[0]), int(coord[1])
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 8, (255, 255, 255), 2)
        return img

    def cleanup(self):
        """释放资源"""
        self._image_set = False
        self._original_size = None
        if self.predictor is not None:
            self.predictor.reset_image()
