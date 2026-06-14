import cv2
import numpy as np
import torch

from log import get_logger

logger = get_logger(__name__)


def _make_optimized_generator(base_generator_cls, model, session_predictor, **kwargs):
    """Create a SamAutomaticMaskGenerator subclass that skips the redundant
    image-encoder pass for single full-image crops by reusing the session
    predictor's cached embedding.

    Only _process_crop is overridden; the rest of the pipeline (generate →
    _generate_masks → NMS → postprocess → mask encoding) is inherited from the
    stock class, so it automatically tracks any upstream changes.
    """
    # Resolve the AMG utilities from the correct package
    # (segment_anything_hq.utils.amg or mobile_sam.utils.amg)
    import importlib
    _amg_mod = importlib.import_module(
        base_generator_cls.__module__.rsplit(".", 1)[0] + ".utils.amg"
    )

    class _OptimizedAutoMaskGenerator(base_generator_cls):
        def __init__(self, _session_pred, *args, **kw):
            super().__init__(*args, **kw)
            self._session_pred = _session_pred

        def _process_crop(self, image, crop_box, crop_layer_idx, orig_size,
                          multimask_output=True):
            # Only optimize the full-image crop (crop_n_layers=0 default).
            # Multi-crop paths require per-crop re-encoding → use stock logic.
            x0, y0, x1, y1 = crop_box
            is_full_image = (
                x0 == 0 and y0 == 0
                and x1 == orig_size[1] and y1 == orig_size[0]
            )
            if not is_full_image:
                return super()._process_crop(
                    image, crop_box, crop_layer_idx, orig_size,
                )

            cropped_im_size = (y1 - y0, x1 - x0)

            # Swap in session predictor (already encoded) to skip set_image
            saved_pred = self.predictor
            self.predictor = self._session_pred
            saved_state = (
                self._session_pred.is_image_set,
                self._session_pred.features,
                getattr(self._session_pred, "interm_features", None),
                getattr(self._session_pred, "original_size", None),
                getattr(self._session_pred, "input_size", None),
            )

            try:
                points_scale = np.array(cropped_im_size)[None, ::-1]
                points_for_image = self.point_grids[crop_layer_idx] * points_scale

                data = _amg_mod.MaskData()
                for (points,) in _amg_mod.batch_iterator(
                    self.points_per_batch, points_for_image
                ):
                    batch_data = self._process_batch(
                        points, cropped_im_size, crop_box, orig_size,
                    )
                    data.cat(batch_data)
                    del batch_data

                from torchvision.ops.boxes import batched_nms
                keep_by_nms = batched_nms(
                    data["boxes"].float(),
                    data["iou_preds"],
                    torch.zeros_like(data["boxes"][:, 0]),
                    iou_threshold=self.box_nms_thresh,
                )
                data.filter(keep_by_nms)

                data["boxes"] = _amg_mod.uncrop_boxes_xyxy(data["boxes"], crop_box)
                data["points"] = _amg_mod.uncrop_points(data["points"], crop_box)
                data["crop_boxes"] = torch.tensor(
                    [crop_box for _ in range(len(data["rles"]))]
                )
            finally:
                self.predictor = saved_pred
                self._session_pred.is_image_set = saved_state[0]
                self._session_pred.features = saved_state[1]
                if hasattr(self._session_pred, "interm_features"):
                    self._session_pred.interm_features = saved_state[2]
                if saved_state[3] is not None:
                    self._session_pred.original_size = saved_state[3]
                if saved_state[4] is not None:
                    self._session_pred.input_size = saved_state[4]

            return data

    return _OptimizedAutoMaskGenerator(session_predictor, model, **kwargs)


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
        auto_mask_generator_cls=None,
    ):
        self.device = device
        self.model = model
        self.predictor = predictor_cls(model)
        self.log_prefix = log_prefix
        self.enable_single_point_multimask = enable_single_point_multimask
        self.predict_kwargs = dict(predict_kwargs or {})
        self.auto_mask_generator_cls = auto_mask_generator_cls
        self._image_set = False
        self._original_size = None
        self._prev_logits = None
        self._prev_npoints = 0
        self._cached_mask = None
        self._cached_logits = None
        self._cached_generator = None
        self._cached_generator_key = None

    def set_image(self, image: np.ndarray):
        self._original_size = image.shape[:2]
        self._original_image_rgb = image
        with torch.inference_mode():
            self.predictor.set_image(image)
        self._image_set = True
        self._prev_logits = None
        self._prev_npoints = 0
        self._cached_mask = None
        self._cached_logits = None
        logger.info("[%s] session 图像特征已缓存, 尺寸: %s", self.log_prefix, self._original_size)

    def auto_segment(self, **kwargs) -> list:
        """自动分割所有主体，返回按面积降序排列的 mask 列表。"""
        if not self._image_set:
            raise RuntimeError("请先调用 set_image() 设置图像")
        if self.auto_mask_generator_cls is None:
            raise RuntimeError(f"{self.log_prefix} 未配置 auto_mask_generator_cls")
        # Cache generator instance (avoids repeated construction overhead)
        cache_key = tuple(sorted(kwargs.items())) if kwargs else ()
        if self._cached_generator is None or self._cached_generator_key != cache_key:
            self._cached_generator = _make_optimized_generator(
                self.auto_mask_generator_cls, self.model, self.predictor, **kwargs
            )
            self._cached_generator_key = cache_key

        masks = self._cached_generator.generate(self._original_image_rgb)

        masks.sort(key=lambda m: m["area"], reverse=True)
        logger.info("[%s] 自动分割完成, 发现 %d 个主体", self.log_prefix, len(masks))
        return masks

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

        with torch.inference_mode():
            masks, scores, low_res = self.predictor.predict(
                point_coords=coords,
                point_labels=labels,
                box=box_arr,
                mask_input=mask_input,
                multimask_output=single_pos_multimask,
                return_logits=True,
                **self.predict_kwargs,
            )
        idx = int(scores.argmax()) if single_pos_multimask else 0
        best_score = float(scores[idx])
        if best_score < 0.8:
            logger.warning("[%s] SAM mask 质量偏低 (score=%.3f)，可能需要更多标记点",
                           self.log_prefix, best_score)
        self._prev_logits = low_res[idx]
        self._prev_npoints = len(point_coords)
        self._cached_logits = masks[idx]  # 原始 logits（float，保留连续置信度）
        self._cached_mask = masks[idx] > 0  # 二值 mask（向后兼容）
        return self._cached_mask

    def predict_box_batch(self, boxes: list) -> dict:
        """Batch predict independent box-only prompts using the cached image embedding."""
        if not self._image_set:
            raise RuntimeError("请先调用 set_image() 设置图像")

        box_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if box_arr.size == 0:
            mask = np.zeros(self._original_size, dtype=bool)
            self._prev_logits = None
            self._prev_npoints = 0
            self._cached_logits = None
            self._cached_mask = mask
            return {"mask": mask, "logits": None, "low_res": None, "scores": None}

        import torch

        box_torch = torch.as_tensor(box_arr, dtype=torch.float, device=self.predictor.device)
        box_torch = self.predictor.transform.apply_boxes_torch(
            box_torch,
            self._original_size,
        )
        masks, scores, low_res = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=box_torch,
            mask_input=None,
            multimask_output=False,
            return_logits=True,
            **self.predict_kwargs,
        )
        masks_np = masks[:, 0].detach().cpu().numpy()
        scores_np = scores[:, 0].detach().cpu().numpy()
        low_res_np = low_res[:, 0].detach().cpu().numpy()

        combined_logits = masks_np[0] if len(masks_np) == 1 else np.max(masks_np, axis=0)
        combined_mask = combined_logits > 0
        self._cached_logits = combined_logits
        self._cached_mask = combined_mask
        self._prev_logits = low_res_np[0] if len(low_res_np) == 1 else None
        self._prev_npoints = 0
        return {
            "mask": combined_mask,
            "logits": combined_logits,
            "low_res": low_res_np,
            "scores": scores_np,
        }

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
        darkened = cv2.addWeighted(image, 1.0 - opacity,
                                   np.full_like(image, mask_color, dtype=np.uint8),
                                   opacity, 0)
        overlay = np.where(mask[..., None], darkened, image)
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
        self._original_image_rgb = None
        self._cached_logits = None
        self._cached_generator = None
        self._cached_generator_key = None
        if self.predictor is not None:
            self.predictor.reset_image()
