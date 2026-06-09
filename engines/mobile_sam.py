"""
MobileSAM 引擎：轻量快速交互式点击分割 + 实时蒙版
缓存 image_embedding 实现秒级响应
"""
import os
from .sam_session import BaseSAMSession


class MobileSAMSession(BaseSAMSession):
    def __init__(self, model, predictor_cls, device: str, auto_mask_generator_cls=None):
        super().__init__(
            model,
            predictor_cls,
            device,
            log_prefix="MobileSAM",
            enable_single_point_multimask=True,
            auto_mask_generator_cls=auto_mask_generator_cls,
        )


class MobileSAMEngine:
    def __init__(self, model_path: str, device: str = "cpu", model_type: str = "vit_t"):
        self.device = device
        self.model = None
        self._predictor_cls = None
        self.model_path = model_path
        self.model_type = model_type

    def _load_model(self):
        if self.model is not None:
            return
        print(f"[MobileSAM] 加载 {self.model_type} 模型到 {self.device} ...")
        from mobile_sam import sam_model_registry, SamPredictor

        checkpoint = self._find_checkpoint()
        if self.model_type not in sam_model_registry:
            raise ValueError(f"MobileSAM 不支持 model_type={self.model_type}")
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self._predictor_cls = SamPredictor
        from mobile_sam import SamAutomaticMaskGenerator
        self._auto_mask_generator_cls = SamAutomaticMaskGenerator
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] MobileSAM loaded — allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
        print("[MobileSAM] 模型加载完成")

    def create_session(self):
        """创建独立 predictor/embedding 状态，共享只读模型权重。"""
        self._load_model()
        return MobileSAMSession(
            self.model, self._predictor_cls, self.device,
            auto_mask_generator_cls=self._auto_mask_generator_cls,
        )

    def _find_checkpoint(self) -> str:
        """查找 MobileSAM 模型文件"""
        candidates = {
            "vit_t": ["mobile_sam.pt", "mobile_sam.pth", "sam_vit_t.pth"],
        }
        for name in candidates.get(self.model_type, []):
            path = os.path.join(self.model_path, name)
            if os.path.exists(path):
                self._validate_checkpoint_type(path)
                return path
        for f in os.listdir(self.model_path):
            if f.endswith((".pt", ".pth")):
                path = os.path.join(self.model_path, f)
                self._validate_checkpoint_type(path)
                return path
        raise FileNotFoundError(
            f"在 {self.model_path} 中未找到 MobileSAM 模型文件"
        )

    def _validate_checkpoint_type(self, checkpoint: str):
        """Fail fast if the configured model type cannot match the checkpoint name."""
        name = os.path.basename(checkpoint).lower()
        if self.model_type == "vit_t" and name in {"mobile_sam.pt", "mobile_sam.pth"}:
            return
        if "vit_" in name:
            if self.model_type in name:
                return
            raise ValueError(
                f"MobileSAM 配置 model_type={self.model_type} 与权重文件不匹配: {name}"
            )
        raise ValueError(
            f"无法从 MobileSAM 权重文件名验证 model_type={self.model_type}: {name}；"
            "请使用 mobile_sam.pt 或 sam_vit_t.pth"
        )

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._predictor_cls = None
