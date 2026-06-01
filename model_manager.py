"""
模型管理器：懒加载、卸载、设备检测、路径管理
"""
import gc
import os
import sys
import torch


def get_base_path():
    """获取基础路径（兼容开发模式和 PyInstaller 打包模式）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_models_path():
    """获取模型目录路径"""
    return os.path.join(get_base_path(), "models")


def get_output_path():
    """获取输出目录路径，不存在则自动创建"""
    path = os.path.join(get_base_path(), "output")
    os.makedirs(path, exist_ok=True)
    return path


def get_device():
    """检测可用设备：CUDA > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_vram_gb() -> float:
    """当前 GPU 实际空闲显存(GB，含其它进程占用)。非 CUDA 返回 inf（视为充足，不卸载）。"""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free / 1024**3
    return float("inf")


def clear_gpu_cache():
    """释放 GPU 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] cache cleared — allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")


def log_vram(tag: str = ""):
    """打印当前显存占用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {tag}] allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")


ENGINE_CONFIGS = {
    "mobile_sam": {
        "class": "MobileSAMEngine",
        "module": "engines.mobile_sam",
        "model_path": "mobile_sam",
    },
    "sam_hq": {
        "class": "SAMHQEngine",
        "module": "engines.sam_hq",
        "model_path": "sam_hq",
    },
}

VITMATTE_VARIANTS = {
    "直出": "none",
    "Base": "base",
    "MatAny": "matany",
    "Small": "small",
}

VITMATTE_PROCESS_MODES = {
    "条带": "strip",
    "主体": "subject",
    "边缘": "full",
}


class ModelManager:
    """统一的模型生命周期管理"""

    def __init__(self):
        self.device = get_device()
        self._rmbg2_engine = None
        self._vitmatte_engine = None
        self._vitmatte_variant = None
        self._grounding_dino_engine = None
        self._sam_engine = None
        self._sam_engine_type = None

    @property
    def rmbg2(self):
        """懒加载 RMBG-2.0 引擎"""
        if self._rmbg2_engine is None:
            from engines.rmbg2 import RMBG2Engine
            self._rmbg2_engine = RMBG2Engine(
                model_path=os.path.join(get_models_path(), "rmbg-2.0"),
                device=self.device
            )
        return self._rmbg2_engine

    @property
    def vitmatte(self):
        """懒加载 ViTMatte 精细化引擎（默认 Base，与 Matte-Anything 对齐）"""
        if self._vitmatte_engine is None:
            self._load_vitmatte_engine("base")
        return self._vitmatte_engine

    def _load_vitmatte_engine(self, variant: str):
        from engines.vitmatte import ViTMatteRefiner
        repo_map = {
            "small": ("vitmatte-small", "hustvl/vitmatte-small-distinctions-646"),
            "matany": ("vitmatte-matany", "hustvl/vitmatte-base-distinctions-646"),
            "base": ("vitmatte-base", "hustvl/vitmatte-base-distinctions-646"),
        }
        model_dir, hf_repo = repo_map[variant]
        if variant == "matany":
            matany_path = os.path.join(get_models_path(), model_dir)
            has_weights = (
                os.path.isfile(os.path.join(matany_path, "model.safetensors")) or
                os.path.isfile(os.path.join(matany_path, "ViTMatte_B_DIS.pth"))
            )
            if not has_weights:
                raise FileNotFoundError(
                    "MatAny 需要 detectron2 权重。请下载放到 models/vitmatte-matany/ 目录：\n"
                    "https://drive.google.com/file/d/1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW\n"
                    "文件名: ViTMatte_B_DIS.pth（首次加载自动转换，之后秒加载）"
                )
        self._vitmatte_engine = ViTMatteRefiner(
            model_path=os.path.join(get_models_path(), model_dir),
            device=self.device,
            hf_repo=hf_repo,
            is_matany=(variant == "matany"),
        )
        self._vitmatte_variant = variant

    def switch_vitmatte(self, variant: str) -> bool:
        """切换 ViTMatte 变体（small / matany / base）。返回是否切换了。"""
        if self._vitmatte_variant == variant and self._vitmatte_engine is not None:
            return False
        if self._vitmatte_engine is not None:
            self._vitmatte_engine.cleanup()
            del self._vitmatte_engine
            self._vitmatte_engine = None
            self._vitmatte_variant = None
            clear_gpu_cache()
        self._load_vitmatte_engine(variant)
        return True

    @property
    def grounding_dino(self):
        """懒加载 Grounding-DINO 透明物体检测引擎"""
        if self._grounding_dino_engine is None:
            from engines.grounding_dino import GroundingDinoDetector
            self._grounding_dino_engine = GroundingDinoDetector(
                model_path=os.path.join(get_models_path(), "grounding-dino-tiny"),
                device=self.device,
            )
        return self._grounding_dino_engine

    @property
    def vitmatte_loaded(self) -> bool:
        """ViTMatte 是否已加载（用于 UI 判断要不要显示下载提示）"""
        return self._vitmatte_engine is not None

    @property
    def grounding_dino_loaded(self) -> bool:
        """Grounding-DINO 是否已加载"""
        return self._grounding_dino_engine is not None

    @property
    def sam(self):
        """返回当前激活的 SAM 引擎（默认 MobileSAM）"""
        if self._sam_engine is None:
            self._load_sam_engine("mobile_sam")
        return self._sam_engine

    def _load_sam_engine(self, engine_type: str):
        """加载指定类型的 SAM 引擎"""
        config = ENGINE_CONFIGS[engine_type]
        import importlib
        module = importlib.import_module(config["module"])
        engine_class = getattr(module, config["class"])
        self._sam_engine = engine_class(
            model_path=os.path.join(get_models_path(), config["model_path"]),
            device=self.device,
        )
        self._sam_engine_type = engine_type

    def switch_sam(self, engine_type: str) -> bool:
        """切换 SAM 引擎类型，卸载旧引擎，加载新引擎。返回是否切换了。"""
        if self._sam_engine_type == engine_type and self._sam_engine is not None:
            return False  # 已经是目标引擎，无需切换
        # 卸载旧引擎
        if self._sam_engine is not None:
            self._sam_engine.cleanup()
            del self._sam_engine
            self._sam_engine = None
            self._sam_engine_type = None
            clear_gpu_cache()
        # 加载新引擎
        self._load_sam_engine(engine_type)
        return True

    def unload_rmbg2(self):
        """卸载 RMBG-2.0 释放内存"""
        if self._rmbg2_engine is not None:
            self._rmbg2_engine.cleanup()
            del self._rmbg2_engine
            self._rmbg2_engine = None
            clear_gpu_cache()

    def unload_grounding_dino(self):
        """卸载 Grounding-DINO 释放显存"""
        if self._grounding_dino_engine is not None:
            self._grounding_dino_engine.cleanup()
            del self._grounding_dino_engine
            self._grounding_dino_engine = None
            clear_gpu_cache()

    def unload_vitmatte(self):
        """卸载 ViTMatte 释放显存"""
        if self._vitmatte_engine is not None:
            self._vitmatte_engine.cleanup()
            del self._vitmatte_engine
            self._vitmatte_engine = None
            self._vitmatte_variant = None
            clear_gpu_cache()

    def unload_sam(self):
        """卸载 SAM 引擎释放内存"""
        if self._sam_engine is not None:
            self._sam_engine.cleanup()
            del self._sam_engine
            self._sam_engine = None
            self._sam_engine_type = None
            clear_gpu_cache()
