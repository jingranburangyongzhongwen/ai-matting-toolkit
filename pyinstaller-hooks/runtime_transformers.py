# 运行前预热 transformers 动态导入链，避免冻结后边缘修复时
# AutoBackbone.from_config() 触发 VitDetBackbone 动态 import 失败。
#
# transformers 的 LazyModule._get_module() 使用相对导入
# importlib.import_module(".models.vitdet.modeling_vitdet", "transformers"),
# 在 PyInstaller 冻结环境中可能失败。修复策略：
#   1. 用绝对路径 importlib.import_module 预热所有子模块（写入 sys.modules 缓存）
#   2. 将关键类直接注册到 transformers 命名空间，绕过 __getattr__
#
# 注意：此文件作为 --runtime-hook 在冻结包中独立运行，无法导入外部模块，
# 因此子模块清单在此内联维护。与 _transformers_modules.py 保持同步。
# 更新时请同步修改：_transformers_modules.py、build.py 的 --hidden-import 列表。

import importlib
import sys
import warnings


def _import_transformers_runtime_modules():
    # ── 阶段 1：预热子模块（绝对导入，确保 sys.modules 缓存命中）──
    # 内联清单，与 _transformers_modules.SUBMODULES 保持同步
    _WARMUP_MODULES = [
        "transformers",
        "transformers.models",
        "transformers.models.auto",
        "transformers.models.auto.auto_factory",
        "transformers.models.auto.configuration_auto",
        "transformers.models.auto.modeling_auto",
        "transformers.models.vitdet",
        "transformers.models.vitdet.modeling_vitdet",
        "transformers.models.vitdet.configuration_vitdet",
        "transformers.models.vitmatte",
        "transformers.models.vitmatte.modeling_vitmatte",
        "transformers.models.vitmatte.configuration_vitmatte",
        "transformers.models.grounding_dino",
        "transformers.models.grounding_dino.modeling_grounding_dino",
        "transformers.models.grounding_dino.processing_grounding_dino",
        "transformers.models.swin",
        "transformers.models.swin.modeling_swin",
        "transformers.models.swin.configuration_swin",
        "transformers.utils.backbone_utils",
    ]

    for mod_name in _WARMUP_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            # 打印警告而非静默跳过，方便诊断打包问题
            warnings.warn(
                f"[runtime_transformers] 预热 {mod_name} 失败: {exc}",
                stacklevel=2,
            )

    # ── 阶段 2：将关键类直接注册到 transformers 命名空间 ──
    # 即使 LazyModule._get_module 的相对导入失败，getattr(transformers, name)
    # 也不会触发 __getattr__，因为属性已存在。
    # 内联注册表，与 _transformers_modules.CLASS_REGISTRY 保持同步
    _CLASS_REGISTRY = {
        # backbone（AutoBackbone.from_config 链路）
        "transformers.models.vitdet.modeling_vitdet": [
            "VitDetBackbone", "VitDetModel", "VitDetPreTrainedModel",
        ],
        "transformers.models.vitdet.configuration_vitdet": [
            "VitDetConfig",
        ],
        # vitmatte
        "transformers.models.vitmatte.modeling_vitmatte": [
            "VitMatteForImageMatting",
        ],
        "transformers.models.vitmatte.configuration_vitmatte": [
            "VitMatteConfig",
        ],
        # grounding dino
        "transformers.models.grounding_dino.modeling_grounding_dino": [
            "GroundingDinoForObjectDetection",
        ],
        "transformers.models.grounding_dino.processing_grounding_dino": [
            "GroundingDinoProcessor",
        ],
        # swin（vitmatte backbone 依赖）
        "transformers.models.swin.modeling_swin": [
            "SwinModel", "SwinPreTrainedModel",
        ],
        "transformers.models.swin.configuration_swin": [
            "SwinConfig",
        ],
    }

    tf_mod = sys.modules.get("transformers")
    if tf_mod is not None:
        for mod_name, class_names in _CLASS_REGISTRY.items():
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            for cls_name in class_names:
                cls_obj = getattr(mod, cls_name, None)
                if cls_obj is not None:
                    # 直接设置属性，覆盖 LazyModule.__getattr__ 的失败路径
                    # （hasattr 在 __getattr__ 抛异常时返回 False，所以必须无条件设置）
                    setattr(tf_mod, cls_name, cls_obj)


_import_transformers_runtime_modules()
