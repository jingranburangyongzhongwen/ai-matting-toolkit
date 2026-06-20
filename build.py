# -*- coding: utf-8 -*-
"""跨平台 PyInstaller 打包脚本 —— 替代 build.bat

用法:
    python build.py              # 默认打包（构建成功后自动清理中间目录）
    python build.py --no-models  # 不复制 models（用于调试）
    python build.py --name MyApp # 自定义产物名称
    python build.py --clean      # 构建前强制清理旧的中间目录
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


def _load_transformers_submodules() -> list[str]:
    """从 pyinstaller-hooks/_transformers_modules.py 读取 SUBMODULES 列表。

    作为单一数据源，避免在 build.py / hook / runtime_hook 三处维护同一份清单。
    """
    import ast

    shared = os.path.join(SCRIPT_DIR, "pyinstaller-hooks", "_transformers_modules.py")
    if not os.path.isfile(shared):
        warn(f"未找到 {shared}，transformers hidden-import 将使用精简列表")
        return ["transformers", "transformers.models"]

    with open(shared, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        # SUBMODULES = [...] 或 SUBMODULES: list[str] = [...]
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if isinstance(target, ast.Name) and target.id == "SUBMODULES" and isinstance(value, ast.List):
                return [
                    elt.value
                    for elt in value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]

    warn("_transformers_modules.py 中未找到 SUBMODULES 定义")
    return ["transformers", "transformers.models"]


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
def cleanup(work_dir: str, dist_dir: str, app_name: str, clean: bool = False) -> None:
    """清理打包临时产物，保留最终 dist 目录。"""
    if clean:
        info("清理临时文件...")
        rmtree_safe(work_dir)
        rmtree_safe(os.path.join(SCRIPT_DIR, "build"))
        rm_safe(os.path.join(SCRIPT_DIR, f"{app_name}.spec"))
        info("临时文件已清理")
    else:
        info("跳过临时目录清理（使用 --clean 可强制清理）")


# ── 后处理 ────────────────────────────────────────────────────────────────────



def _fast_copytree(src: str, dst: str) -> None:
    """用系统 robocopy 复制目录（Windows 多线程，比 shutil.copytree 快 2-3x）。"""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    if sys.platform == "win32":
        info("使用 robocopy 多线程复制...")
        # /E 递归 /MT:16 16线程 /NFL /NDL /NJH /NJS 去掉日志噪音
        result = subprocess.run(
            ["robocopy", src, dst, "/E", "/MT:16", "/NFL", "/NDL", "/NJH", "/NJS"],
            capture_output=True, text=True,
        )
        # robocopy 退出码 0-7 表示成功
        if result.returncode > 7:
            raise RuntimeError(f"robocopy 失败，退出码 {result.returncode}: {result.stderr}")
    else:
        shutil.copytree(src, dst)


def post_build(dist_dir: str, app_name: str, safehttpx_path: str, copy_models: bool) -> None:
    """PyInstaller 结束后的收尾工作。"""
    app_dir = os.path.join(dist_dir, app_name)
    internal_dir = os.path.join(app_dir, "_internal")

    # 0. CUDA DLL 清理 — 已禁用
    # 2026-06-20: PE 导入表分析确认 torch_cuda.dll 静态依赖 cufft64_11.dll，
    # 删除后 Windows DLL 加载器无法解析 → import torch 崩溃。
    # 其余 5 个（cusolverMg / curand / nvrtc.alt / cudnn_heuristic / cufftw）
    # 虽无静态依赖，但可能被 CUDA 运行时 / cuDNN 动态 LoadLibrary，风险不值得省 ~800 MB。
    # _clean_torch_cuda_dlls(internal_dir)

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
            _fast_copytree(models_src, models_dst)
            info("已复制 models 到分发目录")
        else:
            warn("未找到 models/ 目录，请手动复制到 dist/<app_name>/models/")
    else:
        # --no-models 模式：清除上次构建残留的 models 目录
        stale = os.path.join(app_dir, "models")
        if os.path.isdir(stale):
            shutil.rmtree(stale)
            info("已清除残留的 models/ 目录")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def _patch_transformers_auto_docstring() -> str | None:
    """临时修复 transformers auto_docstring 在 PyInstaller 冻结环境中的 IndexError。

    auto_docstring.get_model_name() 假设路径至少有 3 层目录（.../models/{m}/...），
    但在 PyInstaller 中路径可能更短，导致 path.split(sep)[-3] 越界。
    返回原始内容（用于恢复），如果无需修补则返回 None。
    """
    import importlib.util
    spec = importlib.util.find_spec("transformers.utils.auto_docstring")
    if spec is None or spec.origin is None:
        return None
    fpath = spec.origin
    with open(fpath, encoding="utf-8") as f:
        content = f.read()
    old_line = '    if path.split(os.path.sep)[-3] != "models":'
    new_line = '    if len(path.split(os.path.sep)) < 3 or path.split(os.path.sep)[-3] != "models":'
    if old_line not in content:
        error(
            f"无法匹配 auto_docstring 目标代码行，transformers 版本可能已变化。\n"
            f"  文件: {fpath}\n"
            f"  请检查 get_model_name() 是否仍包含以下代码:\n"
            f"  {old_line.strip()}"
        )
        sys.exit(1)
    info("临时修补 transformers auto_docstring.get_model_name() ...")
    patched = content.replace(old_line, new_line, 1)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(patched)
    except OSError as exc:
        error(f"无法写入 {fpath}: {exc}")
        sys.exit(1)
    return content


def _restore_transformers_auto_docstring(original: str) -> None:
    """恢复 transformers auto_docstring 的原始内容。"""
    import importlib.util
    spec = importlib.util.find_spec("transformers.utils.auto_docstring")
    if spec is None or spec.origin is None:
        return
    with open(spec.origin, "w", encoding="utf-8") as f:
        f.write(original)
    info("已恢复 transformers auto_docstring")


def build(app_name: str, copy_models: bool, clean: bool = False) -> None:
    t0 = time.monotonic()
    gradio_path, safehttpx_path = check_environment()

    work_dir = os.path.join(SCRIPT_DIR, ".pyi-build")
    dist_dir = os.path.join(SCRIPT_DIR, "dist")

    info(f"产物名称: {app_name}")
    info(f"中间目录: {work_dir}")
    info(f"输出目录: {dist_dir}")
    info("开始打包...")

    # 清理旧的中间目录（仅在 --clean 时执行）
    if clean:
        rmtree_safe(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # 从共享清单加载 transformers 子模块（单一数据源，避免三处漂移）
    _tf_submodules = _load_transformers_submodules()
    _tf_hidden_imports = []
    for _mod in ["transformers", "transformers.models"] + _tf_submodules:
        _tf_hidden_imports += ["--hidden-import", _mod]

    # 构建 PyInstaller 参数
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--workpath", work_dir,
        "--distpath", dist_dir,
        "--onedir",
        "--noupx",  # 禁用 UPX 压缩，避免大型 DLL 压缩耗时和运行时解压开销
        "--name", app_name,
        # 数据文件
        "--add-data", f"engines{SEP}engines",
        "--add-data", f"{os.path.join(gradio_path, 'templates')}{SEP}gradio{os.sep}templates",
        "--add-data", f"README.md{SEP}.",
        # 自定义 hook
        "--additional-hooks-dir", os.path.join(SCRIPT_DIR, "pyinstaller-hooks"),
        "--runtime-hook", os.path.join(SCRIPT_DIR, "pyinstaller-hooks", "runtime_transformers.py"),
        # 子模块收集
        "--collect-submodules", "engines",
        "--collect-submodules", "kornia",
        # 隐藏导入
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
    ] + _tf_hidden_imports + [
        "--hidden-import", "safetensors",
        "--hidden-import", "timm",
        "--hidden-import", "tqdm",
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
        # transformers 的 dependency_versions_check 在运行时通过 importlib.metadata 检查这些包的版本，
        # 需要将 .dist-info 元数据复制到打包产物中
        "--copy-metadata", "tqdm",
        "--copy-metadata", "regex",
        "--copy-metadata", "requests",
        "--copy-metadata", "packaging",
        "--copy-metadata", "filelock",
        "--copy-metadata", "numpy",
        "--copy-metadata", "huggingface-hub",
        "--copy-metadata", "safetensors",
        "--copy-metadata", "pyyaml",
        "--copy-metadata", "tokenizers",
        "--copy-metadata", "accelerate",
        "--copy-metadata", "torchcodec",
        # 排除不需要的包（避免 Qt 冲突 / skimage 子进程崩溃）
        "--exclude-module", "PySide6",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        "--exclude-module", "skimage",
        # Gradio 传递依赖（本应用不使用）
        "--exclude-module", "panel",           # 108 MB
        "--exclude-module", "bokeh",           # 32 MB
        "--exclude-module", "notebook",        # 63 MB
        "--exclude-module", "jupyterlab",      # 21 MB
        "--exclude-module", "ipykernel",
        "--exclude-module", "IPython",
        # huggingface_hub AWS 依赖（离线推理不需要）
        "--exclude-module", "botocore",        # 106 MB
        "--exclude-module", "boto3",
        # 数据科学栈（本应用不使用）
        "--exclude-module", "pyarrow",         # 18 MB
        "--exclude-module", "sklearn",         # 15 MB
        "--exclude-module", "scipy",           # 35 MB
        "--exclude-module", "dask",
        "--exclude-module", "distributed",
        # 其他不需要的
        "--exclude-module", "llvmlite",        # 88 MB
        "--exclude-module", "av",              # 65 MB
        "--exclude-module", "torchaudio",      # 11 MB
        "--exclude-module", "matplotlib",      # 12 MB
        "--exclude-module", "plotly",          # 13 MB
        "--exclude-module", "sphinx",          # 11 MB
        "--exclude-module", "nltk",            # 15 MB
        "--exclude-module", "statsmodels",     # 7.5 MB
        "--exclude-module", "onnxruntime",     # 33 MB
        # 入口
        "app.py",
    ]

    # 临时修补 transformers auto_docstring，修复 PyInstaller 冻结环境中的 IndexError
    _ad_original = None
    try:
        _ad_original = _patch_transformers_auto_docstring()
        exit_code = run(cmd)
    finally:
        # 无论成功失败都恢复原始文件
        if _ad_original is not None:
            _restore_transformers_auto_docstring(_ad_original)

    # 无论成功失败都清理临时产物（打包成功时删除中间目录释放空间，失败时保留以便排查）
    cleanup(work_dir, dist_dir, app_name, clean=(exit_code == 0))

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
    parser.add_argument(
        "--clean", action="store_true",
        help="构建前强制清理旧的中间目录（默认保留以加速增量打包；构建成功后自动清理）",
    )
    args = parser.parse_args()
    build(app_name=args.name, copy_models=not args.no_models, clean=args.clean)


if __name__ == "__main__":
    main()
