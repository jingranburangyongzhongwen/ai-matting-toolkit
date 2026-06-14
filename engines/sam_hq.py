"""
SAM-HQ 引擎：交互式点击分割 + 实时蒙版（高质量边缘）
缓存 image_embedding 实现秒级响应
"""
import os

from log import get_logger
from .sam_session import BaseSAMSession

logger = get_logger(__name__)


class SAMHQSession(BaseSAMSession):
    def __init__(self, model, predictor_cls, device: str, auto_mask_generator_cls=None):
        super().__init__(
            model,
            predictor_cls,
            device,
            log_prefix="SAM-HQ",
            enable_single_point_multimask=True,
            predict_kwargs={"hq_token_only": False},
            auto_mask_generator_cls=auto_mask_generator_cls,
        )


class SAMHQEngine:
    def __init__(self, model_path: str, device: str = "cpu", model_type: str = "vit_l"):
        self.device = device
        self.model = None
        self._predictor_cls = None
        self.model_path = model_path
        self.model_type = model_type

    def _load_model(self):
        if self.model is not None:
            return
        logger.info("加载 %s 模型到 %s ...", self.model_type, self.device)
        from segment_anything_hq import sam_model_registry, SamPredictor

        checkpoint = self._find_checkpoint()
        if self.model_type not in sam_model_registry:
            raise ValueError(f"SAM-HQ 不支持 model_type={self.model_type}")
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self._predictor_cls = SamPredictor
        from segment_anything_hq import SamAutomaticMaskGenerator
        self._auto_mask_generator_cls = SamAutomaticMaskGenerator
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info("VRAM: allocated=%.2fGB, reserved=%.2fGB", allocated, reserved)
        logger.info("模型加载完成")

    def create_session(self):
        """创建独立 predictor/embedding 状态，共享只读模型权重。"""
        self._load_model()
        return SAMHQSession(
            self.model, self._predictor_cls, self.device,
            auto_mask_generator_cls=self._auto_mask_generator_cls,
        )

    def _find_checkpoint(self) -> str:
        """查找 SAM-HQ 模型文件"""
        candidates = {
            "vit_b": ["sam_hq_vit_b.pth"],
            "vit_l": ["sam_hq_vit_l.pth"],
            "vit_h": ["sam_hq_vit_h.pth"],
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
            f"在 {self.model_path} 中未找到 SAM-HQ 模型文件"
        )

    def _validate_checkpoint_type(self, checkpoint: str):
        """Fail fast before PyTorch emits a long shape-mismatch traceback."""
        name = os.path.basename(checkpoint).lower()
        for model_type in ("vit_b", "vit_l", "vit_h"):
            if model_type in name and model_type != self.model_type:
                raise ValueError(
                    f"SAM-HQ 配置 model_type={self.model_type} 与权重文件不匹配: {name}"
                )
            if model_type in name and model_type == self.model_type:
                return
        raise ValueError(
            f"无法从 SAM-HQ 权重文件名验证 model_type={self.model_type}: {name}；"
            "请使用 sam_hq_vit_b/l/h.pth 命名"
        )

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._predictor_cls = None
