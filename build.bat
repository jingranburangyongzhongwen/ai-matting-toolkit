@echo off
chcp 65001 >nul
echo ========================================
echo   全自动抠图工具 - 打包脚本
echo ========================================
echo.

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查 PyInstaller
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装 PyInstaller...
    pip install pyinstaller
)

REM 获取 Gradio 模板路径
for /f "tokens=*" %%i in ('python -c "import gradio,os; print(os.path.dirname(gradio.__file__))"') do set GRADIO_PATH=%%i

echo [信息] Gradio 路径: %GRADIO_PATH%
echo [信息] 开始打包...
echo.

pyinstaller ^
    --noconfirm ^
    --onedir ^
    --name "全自动抠图" ^
    --add-data "engines;engines" ^
    --add-data "model_manager.py;." ^
    --add-data "%GRADIO_PATH%\templates;gradio\templates" ^
    --hidden-import torch ^
    --hidden-import torchvision ^
    --hidden-import transformers ^
    --hidden-import safetensors ^
    --hidden-import kornia ^
    --hidden-import timm ^
    --hidden-import cv2 ^
    --hidden-import PIL ^
    --hidden-import numpy ^
    --hidden-import gradio ^
    --hidden-import mobile_sam ^
    --hidden-import segment_anything_hq ^
    --hidden-import uvicorn ^
    --hidden-import uvicorn.logging ^
    --hidden-import uvicorn.loops ^
    --hidden-import uvicorn.protocols ^
    --hidden-import uvicorn.protocols.http ^
    --hidden-import uvicorn.protocols.http.auto ^
    --hidden-import uvicorn.protocols.websockets ^
    --hidden-import uvicorn.protocols.websockets.auto ^
    --hidden-import uvicorn.lifespan ^
    --hidden-import uvicorn.lifespan.on ^
    --hidden-import fastapi ^
    --collect-all gradio ^
    app.py

echo.
if errorlevel 1 (
    echo [错误] 打包失败，请检查上方错误信息
) else (
    echo [成功] 打包完成！
    echo 输出目录: dist\全自动抠图\
    echo.
    echo 请将以下内容一起分发：
    echo   1. dist\全自动抠图\ 整个文件夹
    echo   2. models\ 文件夹（包含模型文件）
    echo   3. README.md
    echo.
    echo 建议将上述内容压缩为 zip 分发
)
pause
