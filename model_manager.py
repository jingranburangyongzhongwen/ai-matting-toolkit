"""
模型管理器：懒加载、卸载、设备检测、路径管理
"""
import gc
import os
import sys
import threading


def _torch():
    import torch
    return torch


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
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_vram_gb() -> float:
    """当前 GPU 实际空闲显存(GB，含其它进程占用)。非 CUDA 返回 inf（视为充足，不卸载）。"""
    torch = _torch()
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free / 1024**3
    return float("inf")


def clear_gpu_cache():
    """释放 GPU 缓存"""
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] cache cleared — allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")


def log_vram(tag: str = ""):
    """打印当前显存占用"""
    torch = _torch()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {tag}] allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")


ENGINE_CONFIGS = {
    "mobile_sam": {
        "class": "MobileSAMEngine",
        "module": "engines.mobile_sam",
        "model_path": "mobile_sam",
        "model_type": "vit_t",
    },
    "sam_hq": {
        "class": "SAMHQEngine",
        "module": "engines.sam_hq",
        "model_path": "sam_hq",
        "model_type": "vit_l",
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
        self._device = None
        self._model_lock = threading.RLock()
        self._rmbg2_lock = threading.Lock()
        self._rmbg2_engine = None
        self._vitmatte_engines = {}
        self._vitmatte_engine = None
        self._vitmatte_variant = None
        self._grounding_dino_engine = None
        self._sam_models = {}
        self._multi_session_mode = False

    def set_multi_session_mode(self, enabled: bool):
        """单 session 低显存优先；多 session 保留多模型缓存避免互相卸载。"""
        self._multi_session_mode = bool(enabled)

    @property
    def device(self):
        if self._device is None:
            self._device = get_device()
        return self._device

    @property
    def rmbg2(self):
        """Lazy-load the RMBG-2.0 engine."""
        if self._rmbg2_engine is None:
            with self._rmbg2_lock:
                if self._rmbg2_engine is None:
                    from engines.rmbg2 import RMBG2Engine
                    self._rmbg2_engine = RMBG2Engine(
                        model_path=os.path.join(get_models_path(), "rmbg-2.0"),
                        device=self.device,
                    )
        return self._rmbg2_engine

    def preload_rmbg2(self):
        """Load the default RMBG model ahead of the first request."""
        engine = self.rmbg2
        engine._load_model()
        return engine

    @property
    def vitmatte(self):
        """懒加载 ViTMatte 精细化引擎（默认 Base，与 Matte-Anything 对齐）"""
        return self.get_vitmatte("base")

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
        engine = ViTMatteRefiner(
            model_path=os.path.join(get_models_path(), model_dir),
            device=self.device,
            hf_repo=hf_repo,
            is_matany=(variant == "matany"),
        )
        self._vitmatte_engines[variant] = engine
        self._vitmatte_engine = engine
        self._vitmatte_variant = variant

    def get_vitmatte(self, variant: str):
        """获取指定 ViTMatte 变体；单 session 卸载旧变体，多 session 常驻缓存。"""
        if variant == "none":
            return None
        with self._model_lock:
            removed_old = False
            if not self._multi_session_mode:
                for old_variant, old_engine in list(self._vitmatte_engines.items()):
                    if old_variant == variant:
                        continue
                    old_engine.cleanup()
                    del self._vitmatte_engines[old_variant]
                    removed_old = True
                if removed_old:
                    clear_gpu_cache()
            engine = self._vitmatte_engines.get(variant)
            if engine is None:
                self._load_vitmatte_engine(variant)
                engine = self._vitmatte_engines[variant]
            self._vitmatte_engine = engine
            self._vitmatte_variant = variant
            return engine

    def switch_vitmatte(self, variant: str) -> bool:
        """兼容旧调用：加载目标变体，是否卸载旧变体由运行模式决定。"""
        before = self._vitmatte_variant
        self.get_vitmatte(variant)
        return before != variant

    @property
    def grounding_dino(self):
        """懒加载 Grounding-DINO 透明物体检测引擎"""
        with self._model_lock:
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
        return bool(self._vitmatte_engines)

    @property
    def grounding_dino_loaded(self) -> bool:
        """Grounding-DINO 是否已加载"""
        return self._grounding_dino_engine is not None

    def _create_sam_engine(self, engine_type: str):
        """创建指定类型的 SAM 引擎；模型权重可被后续 session predictor 共享。"""
        config = ENGINE_CONFIGS[engine_type]
        import importlib
        module = importlib.import_module(config["module"])
        engine_class = getattr(module, config["class"])
        return engine_class(
            model_path=os.path.join(get_models_path(), config["model_path"]),
            device=self.device,
            model_type=config["model_type"],
        )

    def get_sam_engine(self, engine_type: str):
        """获取某类 SAM 的共享只读模型容器；单 session 仅保留当前类型。"""
        with self._model_lock:
            engine = self._sam_models.get(engine_type)
            if engine is None:
                engine = self._create_sam_engine(engine_type)
                engine._load_model()
                self._sam_models[engine_type] = engine
            return engine

    def unload_rmbg2(self):
        """卸载 RMBG-2.0 释放内存"""
        if self._rmbg2_engine is not None:
            self._rmbg2_engine.cleanup()
            del self._rmbg2_engine
            self._rmbg2_engine = None
            clear_gpu_cache()

    def unload_grounding_dino(self):
        """卸载 Grounding-DINO 释放显存"""
        with self._model_lock:
            if self._grounding_dino_engine is not None:
                self._grounding_dino_engine.cleanup()
                del self._grounding_dino_engine
                self._grounding_dino_engine = None
                clear_gpu_cache()

    def unload_vitmatte(self):
        """卸载 ViTMatte 释放显存"""
        with self._model_lock:
            if self._vitmatte_engines:
                for engine in self._vitmatte_engines.values():
                    engine.cleanup()
                self._vitmatte_engines.clear()
                self._vitmatte_engine = None
                self._vitmatte_variant = None
                clear_gpu_cache()

    def unload_sam(self):
        """卸载 SAM 引擎释放内存"""
        with self._model_lock:
            if self._sam_models:
                for engine in self._sam_models.values():
                    engine.cleanup()
                self._sam_models.clear()
                clear_gpu_cache()

    def unload_sam_engine(self, engine_type: str):
        """卸载某个 SAM 模型容器；调用方需确保没有 session 正在使用它。"""
        with self._model_lock:
            engine = self._sam_models.pop(engine_type, None)
            if engine is not None:
                engine.cleanup()
                clear_gpu_cache()
