# -*- coding: utf-8 -*-
"""transformers 子模块清单 —— 供 hook / runtime-hook / build.py 共享。

单一数据源，避免三处手动维护同一份列表时产生漂移。
"""

# ── 预热 & 收集所需的子模块 ─────────────────────────────────────────────────
# 包含 Auto 系列、具体模型、backbone 工具等在 PyInstaller 冻结环境中
# 需要显式导入才能正确初始化的子模块。
SUBMODULES: list[str] = [
    # Auto 系列（AutoBackbone.from_config 延迟导入链）
    "transformers.models.auto",
    "transformers.models.auto.auto_factory",
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.modeling_auto",
    # vitdet（backbone）
    "transformers.models.vitdet",
    "transformers.models.vitdet.modeling_vitdet",
    "transformers.models.vitdet.configuration_vitdet",
    # vitmatte（图像抠图）
    "transformers.models.vitmatte",
    "transformers.models.vitmatte.modeling_vitmatte",
    "transformers.models.vitmatte.configuration_vitmatte",
    # grounding_dino（目标检测）
    "transformers.models.grounding_dino",
    "transformers.models.grounding_dino.modeling_grounding_dino",
    "transformers.models.grounding_dino.processing_grounding_dino",
    # swin（vitmatte backbone 依赖）
    "transformers.models.swin",
    "transformers.models.swin.modeling_swin",
    "transformers.models.swin.configuration_swin",
    # backbone 工具
    "transformers.utils.backbone_utils",
]

# ── 需要注册到 transformers 顶层命名空间的类 ─────────────────────────────
# 阶段 2：即使 LazyModule._get_module 的相对导入失败，
# getattr(transformers, name) 也不会触发 __getattr__，因为属性已存在。
CLASS_REGISTRY: dict[str, list[str]] = {
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
