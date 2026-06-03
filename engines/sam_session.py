import cv2
import numpy as np


class BaseSAMSession:
    """Per-browser SAM predictor state sharing read-only model weights."""

    def __init__(
        self,
        model,
        predictor_cls,
        device: str,
        *,
        log_prefix: str,
        enable_single_point_multimask: bool = False,
        predict_kwargs: dict | None = None,
    ):
        self.device = device
        self.model = model
        self.predictor = predictor_cls(model)
        self.log_prefix = log_prefix
        self.enable_single_point_multimask = enable_single_point_multimask
        self.predict_kwargs = dict(predict_kwargs or {})
        self._image_set = False
        self._original_size = None
        self._prev_logits = None
        self._prev_npoints = 0
        self._cached_mask = None

    def set_image(self, image: np.ndarray):
        self._original_size = image.shape[:2]
        self.predictor.set_image(image)
        self._image_set = True
        self._prev_logits = None
        self._prev_npoints = 0
        self._cached_mask = None
        print(f"[{self.log_prefix}] session 图像特征已缓存, 尺寸: {self._original_size}")

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

        using_prev = self._prev_logits is not None
        mask_input = self._prev_logits[None, :, :] if using_prev else None
        single_pos_multimask = self._should_use_single_pos_multimask(
            labels=labels,
            box_arr=box_arr,
            using_prev=using_prev,
        )

        masks, scores, low_res = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box_arr,
            mask_input=mask_input,
            multimask_output=single_pos_multimask,
            **self.predict_kwargs,
        )
        idx = int(scores.argmax()) if single_pos_multimask else 0
        self._prev_logits = low_res[idx]
        self._prev_npoints = len(point_coords)
        self._cached_mask = masks[idx]
        return self._cached_mask

    def _should_use_single_pos_multimask(self, labels, box_arr, using_prev: bool) -> bool:
        if not self.enable_single_point_multimask:
            return False
        return (
            box_arr is None
            and not using_prev
            and labels is not None
            and len(labels) == 1
            and labels[0] == 1
        )

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
