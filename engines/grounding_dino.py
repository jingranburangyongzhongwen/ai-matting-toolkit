"""
Grounding-DINO 透明物体检测引擎
检测玻璃/水滴/灯泡等透明物体，配合 ViTMatte 修正 trimap，
让模型把这些区域当 unknown 来推断 alpha。
"""
import os
import time
import threading

import numpy as np
import torch
from PIL import Image


TRANSPARENT_CAPTION = "glass. lens. crystal. diamond. bubble. bulb. web. grid."


class GroundingDinoDetector:
    HF_REPO = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load_model(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            local_path = self._ensure_local(self.model_path)
            print(f"[Grounding-DINO] loading model on {self.device} ...")
            self.processor = AutoProcessor.from_pretrained(local_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(local_path)
            self.model.to(self.device)
            self.model.eval()
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[VRAM] Grounding-DINO loaded - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
            print("[Grounding-DINO] model loaded")

    @classmethod
    def _ensure_local(cls, path: str) -> str:
        weights_exist = (
            os.path.isfile(os.path.join(path, "model.safetensors")) or
            os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        )
        if weights_exist:
            return path
        print(f"[Grounding-DINO] 本地模型不完整，下载 {cls.HF_REPO} 到 {path} ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=cls.HF_REPO, local_dir=path)
        print("[Grounding-DINO] 下载完成")
        return path

    def detect(
        self,
        image,
        caption: str = TRANSPARENT_CAPTION,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_boxes: int = None,
        return_scores: bool = False,
    ):
        """
        检测图中匹配 caption 的物体。
        默认返回 [[x1,y1,x2,y2], ...]；return_scores=True 时返回
        (boxes, scores)，其中 scores 可能为 None。
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        t0 = time.perf_counter()
        with self._infer_lock:
            self._load_model()
            inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(image.height, image.width)],
            )
        boxes = results[0]["boxes"].cpu().numpy().tolist()
        scores = results[0].get("scores", None)
        has_scores = scores is not None
        scores = scores.cpu().numpy().tolist() if has_scores else None
        if max_boxes is not None and len(boxes) > max_boxes:
            order = (
                np.argsort(np.array(scores))[::-1][:max_boxes]
                if has_scores else np.arange(len(boxes))[:max_boxes]
            )
            boxes = [boxes[i] for i in order]
            if has_scores:
                scores = [scores[i] for i in order]
        print(f"[Grounding-DINO] 检测耗时 {time.perf_counter() - t0:.2f}s，命中 {len(boxes)} 个框")
        if return_scores:
            return boxes, scores
        return boxes

    def cleanup(self):
        with self._infer_lock, self._load_lock:
            if self.model is not None:
                del self.model
                del self.processor
                self.model = None
                self.processor = None
