# 全自动抠图工具

一键抠图 + 交互式精细选区，双击即用。

## 功能特性

- **一键批量抠图** — 拖入图片自动去背景，支持批量处理
- **交互式精细选区** — 点击打点选取/排除，实时蒙版预览
- **文本智能定位** — 输入描述自动框选物体（Grounding-DINO，支持中英文）
- **透明物体处理** — 专门优化玻璃、水滴、灯泡等半透明材质
- **双引擎可选** — MobileSAM（快速）/ SAM-HQ（高精度）
- **ViTMatte 精修** — 可选边缘精修，对发丝、毛绒等复杂边缘做 alpha 级优化

## 快速开始（使用打包版本）

1. 双击 `全自动抠图.exe` 启动
2. 程序自动打开浏览器界面
3. 如被杀毒软件拦截，请将整个文件夹添加到白名单

## 功能说明

### 模式一：一键抠图

1. 点击 **一键抠图** 标签页
2. 在原图区拖入单张或多张图片
3. （可选）勾选「检测透明物体」处理玻璃/水滴等材质
4. （可选）选择精修模型（默认直出，可选 Small/Base/MatAny）
5. 点击 **开始抠图**
6. 结果自动保存到 `output/` 文件夹（透明背景 PNG）

### 模式二：精细选区

1. 点击 **精细选区** 标签页
2. 在画布区上传一张图片
3. 选择点击模式：**正向选取**（保留）/ **负向排除**（去掉）
4. 选择引擎：**MobileSAM**（快速）/ **SAM-HQ**（高精度）
5. 在图片上点击打点，红色蒙版实时显示选区
6. （可选）启用文本定位，输入描述自动框选
7. 满意后点击 **开始抠图**

## 开发环境

### 系统要求

- Python 3.10+
- NVIDIA GPU（推荐，支持 CUDA）/ Apple Silicon（MPS）/ CPU

### 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> MobileSAM 需要从 GitHub 安装，如网络受限可手动下载源码后 `pip install .`

### 下载模型

所有模型放到 `models/` 目录下，结构如下：

```
models/
├── rmbg-2.0/              # 自动抠图模型
├── vitmatte-base/         # 边缘精修模型（Base）
├── vitmatte-small/        # 边缘精修模型（Small，省显存）
├── vitmatte-matany/       # 边缘精修模型（MatAny，需额外权重）
├── grounding-dino-tiny/   # 文本定位模型
├── mobile_sam/            # 快速选区模型
│   └── mobile_sam.pt
└── sam_hq/                # 高精度选区模型
    └── sam_hq_vit_l.pth
```

#### 一键下载全部模型

> **前置条件**：已安装 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli)（`pip install huggingface_hub`），并已登录 HuggingFace（`huggingface-cli login`）。RMBG-2.0 需先在 [模型页面](https://huggingface.co/briaai/RMBG-2.0) 申请访问权限（通常秒批）。

```bash
# RMBG-2.0（自动抠图）
huggingface-cli download briaai/RMBG-2.0 --local-dir models/rmbg-2.0

# ViTMatte-Base（边缘精修）
huggingface-cli download hustvl/vitmatte-base-distinctions-646 --local-dir models/vitmatte-base

# ViTMatte-Small（边缘精修，省显存）
huggingface-cli download hustvl/vitmatte-small-distinctions-646 --local-dir models/vitmatte-small

# Grounding-DINO（文本定位）
huggingface-cli download IDEA-Research/grounding-dino-tiny --local-dir models/grounding-dino-tiny

# SAM-HQ（高精度选区）
huggingface-cli download lkeab/hq-sam sam_hq_vit_l.pth --local-dir models/sam_hq

# MobileSAM（快速选区）
mkdir -p models/mobile_sam
curl -L -o models/mobile_sam/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

如果 GitHub 访问困难，MobileSAM 可用镜像：

```bash
curl -L -o models/mobile_sam/mobile_sam.pt https://ghproxy.com/https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

#### MatAny 模型（可选）

MatAny 需要额外的 detectron2 权重：

1. 下载 [ViTMatte_B_DIS.pth](https://drive.google.com/file/d/1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW)
2. 放到 `models/vitmatte-matany/` 目录
3. 首次加载自动转换为 transformers 格式，之后秒加载

### 打包为 exe

```bash
build.bat
```

打包完成后将 `dist/全自动抠图/` 文件夹和 `models/` 文件夹一起分发。

## 常见问题

**Q: 启动后浏览器没有自动打开？**
A: 手动访问 http://127.0.0.1:7860

**Q: 被杀毒软件拦截？**
A: 将整个文件夹添加到杀毒软件白名单

**Q: 处理速度很慢？**
A: 首次使用某个模式需要加载模型，之后会很快。建议使用 NVIDIA GPU。

**Q: 支持哪些图片格式？**
A: JPG、PNG、BMP、WEBP、TIFF

## 技术栈

| 组件 | 模型 | 用途 |
|---|---|---|
| 自动抠图 | [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | 主体识别分割 |
| 边缘精修 | [ViTMatte](https://huggingface.co/hustvl/vitmatte-base-distinctions-646) | Alpha matte 精细化（Small/Base/MatAny） |
| 快速选区 | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | 轻量交互式分割 |
| 高精度选区 | [SAM-HQ](https://github.com/SysCV/sam-hq) | 高质量交互式分割 |
| 文本定位 | [Grounding-DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | 零样本目标检测 |
| Web UI | [Gradio 6](https://gradio.app/) | 浏览器界面 |

## 许可证

本项目使用的模型各自遵循其原始许可证，请参阅各模型的官方仓库了解详情。
