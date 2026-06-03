@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYI_WORK=%~dp0.pyi-build"

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

REM 检查 PyInstaller（与上方 python 同一解释器）
python -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装 PyInstaller...
    python -m pip install pyinstaller
)

REM 检查 Gradio 是否已安装，并获取模板路径
for /f "tokens=*" %%i in ('python -c "import gradio,os; print(os.path.dirname(gradio.__file__))" 2^>nul') do set GRADIO_PATH=%%i
if not defined GRADIO_PATH (
    echo [错误] 无法 import gradio，请先安装依赖:
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [信息] Gradio 路径: %GRADIO_PATH%
echo [信息] 中间目录: %PYI_WORK%
echo [信息] 开始打包...
echo.

if exist "%PYI_WORK%" rmdir /s /q "%PYI_WORK%" 2>nul
mkdir "%PYI_WORK%" 2>nul

REM 不使用 --windowed/-w：保留控制台，便于看日志，并用关闭窗口或 Ctrl+C 退出
python -m PyInstaller ^
    --noconfirm ^
    --workpath "%PYI_WORK%" ^
    --distpath "%~dp0dist" ^
    --onedir ^
    --name "全自动抠图" ^
    --add-data "engines;engines" ^
    --add-data "%GRADIO_PATH%\templates;gradio\templates" ^
    --additional-hooks-dir "%~dp0pyinstaller-hooks" ^
    --collect-submodules engines ^
    --collect-submodules kornia ^
    --hidden-import torch ^
    --hidden-import torchvision ^
    --hidden-import transformers ^
    --hidden-import safetensors ^
    --hidden-import timm ^
    --hidden-import cv2 ^
    --hidden-import PIL ^
    --hidden-import numpy ^
    --hidden-import gradio ^
    --hidden-import huggingface_hub ^
    --hidden-import mobile_sam ^
    --hidden-import segment_anything_hq ^
    --hidden-import engines.rmbg2 ^
    --hidden-import engines.vitmatte ^
    --hidden-import engines.grounding_dino ^
    --hidden-import engines.mobile_sam ^
    --hidden-import engines.sam_hq ^
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
    --collect-data safehttpx ^
    --collect-data groovy ^
    app.py
set "PYI_EXIT=%ERRORLEVEL%"

call :cleanup_pyi_temp

if not "%PYI_EXIT%"=="0" (
    echo.
    echo [错误] 打包失败，请检查上方错误信息
    goto :finish
)

(
    REM Gradio 6 依赖 safehttpx.version.txt，collect-data 偶发漏打，此处兜底复制
    for /f "delims=" %%D in ('python -c "import safehttpx, os; print(os.path.dirname(safehttpx.__file__))"') do set "SAFEHTTPX_DIR=%%D"
    if defined SAFEHTTPX_DIR (
        if not exist "dist\全自动抠图\_internal\safehttpx" mkdir "dist\全自动抠图\_internal\safehttpx"
        copy /Y "%SAFEHTTPX_DIR%\version.txt" "dist\全自动抠图\_internal\safehttpx\" >nul 2>&1
    )
    for /f "delims=" %%D in ('python -c "import groovy, os; print(os.path.dirname(groovy.__file__))"') do set "GROOVY_DIR=%%D"
    if defined GROOVY_DIR (
        if exist "%GROOVY_DIR%\version.txt" (
            if not exist "dist\全自动抠图\_internal\groovy" mkdir "dist\全自动抠图\_internal\groovy"
            copy /Y "%GROOVY_DIR%\version.txt" "dist\全自动抠图\_internal\groovy\" >nul 2>&1
        )
    )
    if exist models (
        echo [信息] 复制 models 到分发目录...
        if not exist "dist\全自动抠图\models" mkdir "dist\全自动抠图\models"
        xcopy /E /I /Y /Q models "dist\全自动抠图\models\" >nul
    ) else (
        echo [提示] 未找到项目根目录 models/，请手动复制到 dist/全自动抠图/models/
    )
    echo.
    echo [成功] 打包完成！
    echo.
    echo 只需这一个目录（复制或压缩整个文件夹即可分发）：
    echo   dist/全自动抠图/
    echo     - 全自动抠图.exe      双击运行
    echo     - _internal/          运行时依赖，勿删
    echo     - models/             模型权重（若已复制）
    echo.
    echo 中间文件（.pyi-build/）已自动清理；若闪退请在 dist/全自动抠图/ 打开 cmd 运行 exe 查看报错。
)

:finish
pause
exit /b %PYI_EXIT%

:cleanup_pyi_temp
if exist "%PYI_WORK%" (
    rmdir /s /q "%PYI_WORK%" 2>nul
    if exist "%PYI_WORK%" (
        echo [警告] 无法删除 %PYI_WORK%，请手动删除
    )
)
if exist "%~dp0build" (
    rmdir /s /q "%~dp0build" 2>nul
    if exist "%~dp0build" echo [警告] 无法删除旧 build 目录，请手动删除
)
if exist "%~dp0全自动抠图.spec" del /q "%~dp0全自动抠图.spec" 2>nul
exit /b 0
