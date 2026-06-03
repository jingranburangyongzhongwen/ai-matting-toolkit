"""
SAM-HQ 引擎：交互式点击分割 + 实时蒙版（高质量边缘）
缓存 image_embedding 实现秒级响应
"""
import os
from .sam_session import BaseSAMSession


class SAMHQSession(BaseSAMSession):
    def __init__(self, model, predictor_cls, device: str):
        super().__init__(
            model,
            predictor_cls,
            device,
            log_prefix="SAM-HQ",
            predict_kwargs={"hq_token_only": False},
        )


class SAMHQEngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = None
        self._predictor_cls = None
        self.model_path = model_path

    def _load_model(self):
        if self.model is not None:
            return
        print(f"[SAM-HQ] 加载模型到 {self.device} ...")
        from segment_anything_hq import sam_model_registry, SamPredictor

        checkpoint = self._find_checkpoint()
        self.model = sam_model_registry["vit_l"](checkpoint=checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self._predictor_cls = SamPredictor
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] SAM-HQ loaded — allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
        print("[SAM-HQ] 模型加载完成")

    def create_session(self):
        """创建独立 predictor/embedding 状态，共享只读模型权重。"""
        self._load_model()
        return SAMHQSession(self.model, self._predictor_cls, self.device)

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

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._predictor_cls = None
