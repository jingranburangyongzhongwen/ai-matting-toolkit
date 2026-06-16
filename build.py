# -*- coding: utf-8 -*-
"""跨平台 PyInstaller 打包脚本 —— 替代 build.bat

用法:
    python build.py              # 默认打包
    python build.py --no-models  # 不复制 models（用于调试）
    python build.py --name MyApp # 自定义产物名称
"""
import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEP = ";" if sys.platform == "win32" else ":"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


# ── 辅助函数 ──────────────────────────────────────────────────────────────────
def info(msg: str) -> None:
    print(f"[信息] {msg}")


def warn(msg: str) -> None:
    print(f"[警告] {msg}")


def error(msg: str) -> None:
    print(f"[错误] {msg}", file=sys.stderr)


def run(cmd: list[str], **kwargs) -> int:
    """执行子命令，实时打印输出，返回退出码。"""
    info(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, **kwargs)
    return result.returncode


def rmtree_safe(path: str) -> None:
    """安全删除目录，失败时打印警告而非抛异常。"""
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError as exc:
            warn(f"无法删除 {path}: {exc}")


def rm_safe(path: str) -> None:
    """安全删除文件。"""
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError as exc:
            warn(f"无法删除 {path}: {exc}")


def python_output(code: str) -> str | None:
    """运行一段 Python 代码并返回 stdout（去掉尾部换行）。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=SCRIPT_DIR,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ── 环境检测 ──────────────────────────────────────────────────────────────────
def check_environment() -> tuple[str, str]:
    """检查 Python / PyInstaller / Gradio 是否就绪，返回 (gradio_path, safehttpx_path)。"""

    # Python 版本
    ver = sys.version_info
    if ver < (3, 10):
        error(f"需要 Python 3.10+，当前 {sys.version}")
        sys.exit(1)
    info(f"Python: {sys.version}")

    # PyInstaller
    try:
        import PyInstaller  # noqa: F401
        info(f"PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        info("正在安装 PyInstaller...")
        rc = run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        if rc != 0:
            error("PyInstaller 安装失败")
            sys.exit(1)

    # Gradio
    gradio_path = python_output(
        "import gradio, os; print(os.path.dirname(gradio.__file__))"
    )
    if not gradio_path:
        error("无法 import gradio，请先安装依赖:\n  pip install -r requirements.txt")
        sys.exit(1)
    info(f"Gradio 路径: {gradio_path}")

    # safehttpx（可选，Gradio 6 依赖）
    safehttpx_path = python_output(
        "import safehttpx, os; print(os.path.dirname(safehttpx.__file__))"
    )

    return gradio_path, safehttpx_path or ""


# ── 清理函数 ──────────────────────────────────────────────────────────────────
def cleanup(work_dir: str, dist_dir: str, app_name: str) -> None:
    """清理所有打包临时产物，保留最终 dist 目录。"""
    info("清理临时文件...")
    rmtree_safe(work_dir)
    rmtree_safe(os.path.join(SCRIPT_DIR, "build"))
    rm_safe(os.path.join(SCRIPT_DIR, f"{app_name}.spec"))

    # 也清理 __pycache__ 中可能残留的打包缓存
    for pycache in glob.glob(os.path.join(SCRIPT_DIR, "**", "__pycache__"), recursive=True):
        # 只删 PyInstaller 中间产物，不动项目本身的 __pycache__
        pass  # 保留项目 __pycache__，避免影响开发

    info("临时文件已清理")


# ── 后处理 ────────────────────────────────────────────────────────────────────
def post_build(dist_dir: str, app_name: str, safehttpx_path: str, copy_models: bool) -> None:
    """PyInstaller 结束后的收尾工作。"""
    app_dir = os.path.join(dist_dir, app_name)
    internal_dir = os.path.join(app_dir, "_internal")

    # 1. safehttpx / groovy version.txt 兜底复制（collect-data 偶发漏打）
    for pkg_name, pkg_path in [("safehttpx", safehttpx_path), ("groovy", "")]:
        if not pkg_path:
            pkg_path = python_output(
                f"import {pkg_name}, os; print(os.path.dirname({pkg_name}.__file__))"
            ) or ""
        if not pkg_path:
            continue
        src = os.path.join(pkg_path, "version.txt")
        if os.path.isfile(src):
            dst_dir = os.path.join(internal_dir, pkg_name)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst_dir)
            info(f"已补充 {pkg_name}/version.txt")

    # 2. 复制 models 目录
    if copy_models:
        models_src = os.path.join(SCRIPT_DIR, "models")
        if os.path.isdir(models_src):
            models_dst = os.path.join(app_dir, "models")
            if os.path.isdir(models_dst):
                shutil.rmtree(models_dst)
            shutil.copytree(models_src, models_dst)
            info("已复制 models 到分发目录")
        else:
            warn("未找到 models/ 目录，请手动复制到 dist/<app_name>/models/")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def build(app_name: str, copy_models: bool) -> None:
    t0 = time.monotonic()
    gradio_path, safehttpx_path = check_environment()

    work_dir = os.path.join(SCRIPT_DIR, ".pyi-build")
    dist_dir = os.path.join(SCRIPT_DIR, "dist")

    info(f"产物名称: {app_name}")
    info(f"中间目录: {work_dir}")
    info(f"输出目录: {dist_dir}")
    info("开始打包...")

    # 清理旧的中间目录
    rmtree_safe(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # 构建 PyInstaller 参数
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--workpath", work_dir,
        "--distpath", dist_dir,
        "--onedir",
        "--name", app_name,
        # 数据文件
        "--add-data", f"engines{SEP}engines",
        "--add-data", f"{os.path.join(gradio_path, 'templates')}{SEP}gradio{os.sep}templates",
        "--add-data", f"README.md{SEP}.",
        # 自定义 hook
        "--additional-hooks-dir", os.path.join(SCRIPT_DIR, "pyinstaller-hooks"),
        # 子模块收集
        "--collect-submodules", "engines",
        "--collect-submodules", "kornia",
        # 隐藏导入
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "transformers",
        "--hidden-import", "safetensors",
        "--hidden-import", "timm",
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "numpy",
        "--hidden-import", "gradio",
        "--hidden-import", "huggingface_hub",
        "--hidden-import", "mobile_sam",
        "--hidden-import", "segment_anything_hq",
        "--hidden-import", "engines.rmbg2",
        "--hidden-import", "engines.vitmatte",
        "--hidden-import", "engines.grounding_dino",
        "--hidden-import", "engines.mobile_sam",
        "--hidden-import", "engines.sam_hq",
        "--hidden-import", "uvicorn",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "fastapi",
        # 整包收集
        "--collect-all", "gradio",
        "--collect-data", "safehttpx",
        "--collect-data", "groovy",
        # 排除不需要的包（避免 Qt 冲突 / skimage 子进程崩溃）
        "--exclude-module", "PySide6",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        "--exclude-module", "skimage",
        # 入口
        "app.py",
    ]

    exit_code = run(cmd)

    # 无论成功失败都清理临时产物
    cleanup(work_dir, dist_dir, app_name)

    if exit_code != 0:
        error("打包失败，请检查上方错误信息")
        sys.exit(exit_code)

    # 后处理
    post_build(dist_dir, app_name, safehttpx_path, copy_models)

    # 打印结果摘要
    app_dir = os.path.join(dist_dir, app_name)
    exe_name = f"{app_name}.exe" if IS_WINDOWS else app_name
    print()
    print("=" * 50)
    print("  打包完成！")
    print("=" * 50)
    print()
    print("输出目录（可整体复制/压缩分发）：")
    print(f"  {app_dir}/")
    print(f"    - {exe_name}        双击运行")
    print(f"    - _internal/        运行时依赖，勿删")
    if copy_models:
        print(f"    - models/           模型权重")
    print()
    if IS_WINDOWS:
        print("若闪退请在 dist 目录下打开 cmd 运行 exe 查看报错。")
    else:
        print(f"若闪退请在终端运行 ./{exe_name} 查看报错。")

    elapsed = time.monotonic() - t0
    mm, ss = divmod(int(elapsed), 60)
    print(f"\n总耗时: {mm}分{ss}秒")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="跨平台 PyInstaller 打包脚本")
    parser.add_argument(
        "--name", default="全自动抠图",
        help="打包产物名称（默认：全自动抠图）",
    )
    parser.add_argument(
        "--no-models", action="store_true",
        help="不复制 models 目录到产物中（用于调试打包流程）",
    )
    args = parser.parse_args()
    build(app_name=args.name, copy_models=not args.no_models)


if __name__ == "__main__":
    main()
