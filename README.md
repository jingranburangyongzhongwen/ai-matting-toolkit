# 全自动抠图工具

本地离线抠图工作站：一键批量 + SAM 精细选区，可直接 `python app.py`，打包后双击 exe 即用，数据不出本机。

## 功能特性

- **离线本地 + 绿色免安装** — 全链路本机推理，无需联网、无需上传云端；打包后无需依赖即可运行
- **一键批量抠图** — 多图拖入，自动识别主体，输出透明 PNG 到 `output/`
- **文本定位 + 精细选区** — 正负向点选/框选，蒙版实时预览；可选中英文描述自动框物（Grounding-DINO → SAM），支持继续加点修正；MobileSAM（快）/ SAM-HQ（高精）双引擎
- **多模型融合** — 精细模式可选 SAM 严格边界，或 SAM 定主体 + RMBG 在 ROI 内约束融合（多主体效果会差）；批量模式 RMBG 全图，可选 ViTMatte 精修软边（大部分情况效果不如直出）
- **显存管理** — 模型按需加载、用完可卸，减轻显卡压力；默认适合单人使用、尽量省显存；多人同时开多个浏览器页时，每人选区分开保存，并限制「同时处理几张」和排队人数，避免显卡一次占满

## 功能说明

### 处理流程

两条路径各自得到全图 alpha，**导出 PNG 前**共用同一套 RGBA 后处理（`engines/rgba_postprocess`）。启动时默认后台预热 RMBG-2.0（`MATTING_PRELOAD_RMBG=0` 可关闭）。

```mermaid
flowchart TB
  subgraph TOP[" "]
    direction LR
    subgraph M1["模式一 · 一键抠图"]
      direction TB
      M1_IN[批量上传 RGB] --> M1_RMBG[RMBG-2.0 全图推理]
      M1_RMBG --> M1_CLEAN[连通域清理 + 边缘平滑]
      M1_CLEAN --> M1_VM{精修 = 直出?}
      M1_VM -->|是| M1_A[全图 alpha]
      M1_VM -->|否| M1_VIT[ViTMatte 精修<br/>条带 / 主体 / 边缘]
      M1_VIT --> M1_A
    end

    subgraph M2["模式二 · 精细选区"]
      direction TB
      M2_UP[上传单张] --> M2_ENG[MobileSAM / SAM-HQ]
      M2_ENG --> M2_INT[打点 / 框选 / 文本定位]
      M2_INT <-->|预览| M2_PRE[SAM 蒙版叠加]
      M2_TL[文本定位] -.-> M2_GD[Grounding-DINO] -.-> M2_INT
      M2_INT -->|开始抠图| M2_MSK[SAM mask + 主体框]
      M2_MSK --> M2_MODE{输出模式?}
      M2_MODE -->|SAM严格| M2_STRICT[SAM 边界收边 + 负向点]
      M2_MODE -->|RMBG精修| M2_ROI[扩边 ROI crop]
      M2_ROI --> M2_RMBG[RMBG predict_alpha]
      M2_RMBG --> M2_EDGE{前景贴 ROI 边?}
      M2_EDGE -->|是| M2_EXP[扩大 ROI 再推理]
      M2_EDGE -->|否| M2_FUSE
      M2_EXP --> M2_FUSE[约束融合 + 负向点]
      M2_STRICT --> M2_A[全图 alpha]
      M2_FUSE --> M2_A
    end
  end

  subgraph PP["共用 · RGBA 后处理"]
    direction LR
    PP_IN[RGB + alpha] --> PP_ANA[拓扑分析] --> PP_PROF[自适应 profile]
    PP_PROF --> PP_TIGHT[收紧光晕带] --> PP_FIX[过切回滚 + 背景方向去色边] --> PP_RGBA[RGBA PNG]
  end

  subgraph MR["可选 · 边缘修复"]
    direction LR
    MR_PAINT[用户涂抹色边] --> MR_MASK[accept_mask]
    MR_MASK --> MR_TRIMAP[spill-aware trimap]
    MR_TRIMAP --> MR_VIT[ViTMatte alpha 重估]
    MR_VIT --> MR_UNMIX[regularized unmix]
    MR_UNMIX --> MR_MERGE[空间羽化 + 置信度合并] --> MR_OUT[修复后 RGBA]
  end

  M1_A --> PP_IN
  M2_A --> PP_IN
  PP_RGBA --> OUT[(output/)]
  PP_RGBA --> MR_PAINT

  M1_DINO[检测透明物体] -.->|仅 ViTMatte 精修时<br/>修正 trimap| M1_VIT
  M1_DINO -.->|直出时仅后处理保护| PP_ANA
  M2_PRES[保护透明材质] -.-> PP_ANA
  DBG[诊断中间结果] -.-> M1_RMBG
  DBG -.-> M2_A
  DBG -.-> PP_ANA
```

**读图说明**

- 模式一 **直出**：RMBG → 清理/平滑 → 后处理；不跑 ViTMatte，也不调用 Grounding-DINO。
- 模式一 **ViTMatte 精修**：在 RMBG alpha 上内部生成窄 unknown trimap 再精修；若同时勾选「检测透明物体」，DINO 用于修正 trimap（非直出路径）。
- 模式一勾选「检测透明物体」且为直出时：仅在后处理阶段加强半透明保护，**不跑 DINO 检测**。
- 模式二 **不走 ViTMatte**；输出二选一：**SAM严格**（纯 SAM 边界，不调 RMBG）或 **RMBG精修**（ROI 内 RMBG + 约束融合）。低显存下 SAM-HQ 与 RMBG 可能同时驻留，建议 ≥8GB 显存或优先 MobileSAM。
- 两种模式导出前都会走自动 RGB 去色边：使用 alpha 外侧背景 seed 估计局部背景方向，只在背景可信且边缘确有背景色残留时修正；透明保护区与细节区保持保守。
- 默认 **单 session 低显存**；多人/多标签页加 `--multi-session`（每页独立 SAM predictor / embedding / 点击先验，SAM 会话 LRU，默认最多 8 个）。

### 模式一：一键抠图

1. 打开 **一键抠图** 标签页
2. 拖入单张或多张图片
3. （可选）勾选「检测透明物体」— 直出时保护半透明区域；配合 ViTMatte 时另用 DINO 修正 trimap
4. （可选）精修模型：默认直出，或 Small / Base / MatAny
5. （可选）勾选「保存诊断中间结果」
6. 点击 **开始抠图** → 结果保存到 `output/`（透明 PNG）
7. 若发丝/边缘有背景色残留，点击 **边缘修复** → 涂抹污染区域 → **应用修复**

### 边缘修复（手动发丝去色边）

当 RMBG 输出的 alpha 在发丝边缘接近实心（alpha≈240-255），但 RGB 仍含背景色（绿幕/蓝幕/红幕残留）时，自动后处理 despill 效果有限。此时可使用**边缘修复**：

1. 一键抠图完成后，右栏出现 **边缘修复** 按钮
2. 进入修复模式，用红色画笔涂抹可见的色边区域
3. 点击 **应用修复** — 系统自动：
   - 构建 accept_mask（仅修复靠近背景的边缘，保护内部）
   - 用 ViTMatte 在窄带内重估 alpha
   - 通过 regularized unmix 恢复前景 RGB（去除背景色污染）
   - 空间羽化 + 置信度门控，平滑合并
4. 支持**撤销**（回退一次）/ **重置**（回到自动结果）/ **退出修复**

```
用户涂抹 → accept_mask → spill-aware trimap → ViTMatte alpha
  → regularized unmix → spatial gate + confidence → 安全合并
```

**本地测试**（无需启动 UI）：

```bash
python scripts/test_manual_refine.py
```

### 直出 vs ViTMatte 精修

| 主体类型 | 推荐 |
|---|---|
| 硬边（产品、硬轮廓） | **直出**，速度最快 |
| 软边（发丝、毛绒、运动模糊） | **ViTMatte**，重建 alpha 过渡 |

- **直出** = RMBG + 后处理；硬边场景通常已足够。
- **ViTMatte** 在硬边上最多追平直出，软边/细节主体收益最大。
- **变体**：Small（省显存）/ Base（更准）/ MatAny（需额外权重）
- **推理模式**：条带（省显存）/ 主体 / 边缘
- ViTMatte 使用模型原生注意力（window+global），不用 strided 近似。

### 模式二：精细选区

1. 打开 **精细选区** 标签页，上传单张图
2. **正向选取** / **负向排除**；引擎选 MobileSAM 或 SAM-HQ
3. 打点预览（仅 SAM）；可选 **文本定位**（DINO 框选 → SAM）
4. 选择 **输出模式**：
   - **SAM严格** — SAM 边界 + 负向点，不调用 RMBG，最快、边界最贴 SAM
   - **RMBG精修** — SAM 定主体，RMBG 在 ROI 内融合 alpha（默认推荐）
5. （可选）「保护透明/半透明材质」— 后处理保留物体内部软 alpha
6. （可选）「保存诊断中间结果」
7. 点击 **开始抠图** → RGBA 后处理 → 保存到 `output/`

## 开发环境

### 系统要求

- Python 3.10+
- NVIDIA GPU（推荐，CUDA）/ Apple Silicon（MPS）/ CPU

### 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

RGBA 后处理不依赖额外 matting solver；RGB 去色边基于 OpenCV/Numpy 的局部背景方向抑制。可不加载模型，先验证后处理诊断链路：

```bash
python scripts/verify_rgb_defringe.py
```

边缘修复自动化测试（需 ViTMatte，GPU ~0.5s/case）：

```bash
python scripts/test_manual_refine.py
```
### 运行

```bash
python app.py
```

默认后台预热 RMBG-2.0。关闭预热：

```bash
# Windows
set MATTING_PRELOAD_RMBG=0 && python app.py

# Linux / macOS
MATTING_PRELOAD_RMBG=0 python app.py
```

> MobileSAM 需从 GitHub 安装；网络受限可手动下载源码后 `pip install .`

### 启动参数

| 参数 | 说明 |
|---|---|
| `-p`, `--port` | 监听端口（默认 `18181`） |
| `-q`, `--silent` | 不自动打开浏览器，减少控制台输出 |
| `--multi-session` | 多标签页 SAM 状态隔离；模型更倾向常驻（偏吞吐） |
| `--max-sam-sessions` | 多 session 下 SAM 缓存上限（默认 `8`，LRU 回收） |
| `--model-concurrency` | GPU 推理并发槽；默认单 session 为 `1`，多 session 为 `2` |
| `--queue-size` | 等待队列长度（默认 `32`） |

示例（本机多人、限并发）：

```bash
python app.py --multi-session --max-sam-sessions 8 --model-concurrency 2 --queue-size 32
```

环境变量 `MATTING_AGGRESSIVE_UNLOAD=1` 可强制单 session 下更积极卸载未用模型（默认单 session 已启用）。

### 下载模型

所有模型放到 `models/` 目录：

```
models/
├── rmbg-2.0/
├── vitmatte-base/
├── vitmatte-small/
├── vitmatte-matany/
├── grounding-dino-tiny/
├── mobile_sam/
│   └── mobile_sam.pt
└── sam_hq/
    └── sam_hq_vit_l.pth
```

#### 一键下载

> 需 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) 并已 `huggingface-cli login`。RMBG-2.0 需在 [模型页](https://huggingface.co/briaai/RMBG-2.0) 申请访问。

```bash
huggingface-cli download briaai/RMBG-2.0 --local-dir models/rmbg-2.0
huggingface-cli download hustvl/vitmatte-base-distinctions-646 --local-dir models/vitmatte-base
huggingface-cli download hustvl/vitmatte-small-distinctions-646 --local-dir models/vitmatte-small
huggingface-cli download IDEA-Research/grounding-dino-tiny --local-dir models/grounding-dino-tiny
huggingface-cli download lkeab/hq-sam sam_hq_vit_l.pth --local-dir models/sam_hq

mkdir -p models/mobile_sam
curl -L -o models/mobile_sam/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

GitHub 困难时 MobileSAM 镜像：

```bash
curl -L -o models/mobile_sam/mobile_sam.pt https://ghproxy.com/https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

#### MatAny（可选）

1. 下载 [ViTMatte_B_DIS.pth](https://drive.google.com/file/d/1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW)
2. 放到 `models/vitmatte-matany/`
3. 首次加载自动转换为 transformers 格式

### 打包为 exe

```bash
build.bat
```

产物目录：`dist/全自动抠图/`（`全自动抠图.exe` + `_internal/` + 可选 `models/`）。PyInstaller 缓存写在 `.pyi-build/`，结束后自动清理。

`pyinstaller-hooks/hook-kornia.py` 将 kornia 以源码打入包内，避免 exe 中 `torch.jit.script` 因缺源码失败。

## 常见问题

**Q: 浏览器没有自动打开？**  
A: 访问 http://127.0.0.1:18181（或 `-p` 指定端口）

**Q: 局域网多人怎么用？**  
A: 默认仅 `127.0.0.1`。多人/多标签页请加 `--multi-session`；超出 `--max-sam-sessions` 后 LRU 回收最久未用的 SAM 缓存。

**Q: 如何退出？**  
A: 关闭控制台窗口或 `Ctrl+C`；只关浏览器标签不会结束服务。

**Q: 杀毒拦截？**  
A: 将整个 `dist/全自动抠图/` 文件夹加入白名单。

**Q: 很慢？**  
A: 首次加载模型较慢；建议使用 NVIDIA GPU。单 session 会在任务间卸载未用模型以省显存。

**Q: 支持哪些格式？**  
A: JPG、PNG、BMP、WEBP、TIFF

## 技术栈

| 组件 | 模型 / 模块 | 用途 |
|---|---|---|
| 自动抠图 | [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | 模式一全图；模式二 ROI alpha |
| 边缘精修 | [ViTMatte](https://huggingface.co/hustvl/vitmatte-base-distinctions-646) | 仅模式一可选（Small/Base/MatAny） |
| 输出净化 | `engines/rgba_postprocess` + `engines/rgb_defringe` | 两模式共用：拓扑收边、背景方向去色边、半透明保护 |
| 手动修复 | `engines/manual_refine` | 涂抹色边 → ViTMatte alpha 重估 + regularized unmix 去背景色污染 |
| 快速选区 | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | 模式二交互与先验 |
| 高精度选区 | [SAM-HQ](https://github.com/SysCV/sam-hq) | 模式二高精度选区 |
| 文本定位 | [Grounding-DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | 模式二框选；模式一 ViTMatte+透明检测时修正 trimap |
| Web UI | [Gradio 6](https://gradio.app/) | 浏览器界面、队列与并发 |

## 许可证

各模型遵循其原始许可证，请参阅对应官方仓库。
