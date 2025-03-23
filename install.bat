@echo off
echo 正在安装烟雾环境下的人体目标检测系统...

:: 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python。请安装Python 3.7或更高版本。
    exit /b 1
)

:: 检查pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到pip。请确保pip已正确安装。
    exit /b 1
)

:: 创建虚拟环境（可选）
echo 是否创建虚拟环境? (y/n)
set /p create_venv=
if /i "%create_venv%"=="y" (
    python -m venv venv
    echo 正在激活虚拟环境...
    call venv\Scripts\activate
)

:: 安装依赖
echo 正在安装依赖...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 错误: 安装依赖失败。
    exit /b 1
)

:: 安装项目（可选）
echo 是否安装项目? (y/n)
set /p install_project=
if /i "%install_project%"=="y" (
    python -m pip install -e .
    if %errorlevel% neq 0 (
        echo 错误: 安装项目失败。
        exit /b 1
    )
)

:: 下载预训练模型（如果需要）
echo 是否下载预训练模型? (y/n)
set /p download_models=
if /i "%download_models%"=="y" (
    echo 正在下载预训练模型...
    if not exist models mkdir models
    
    :: 下载YOLOv5s模型
    echo 正在下载YOLOv5s模型...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt' -OutFile 'models/yolov5s.pt'"
    if %errorlevel% neq 0 (
        echo 警告: 下载YOLOv5s模型失败，请手动下载。
    ) else (
        echo YOLOv5s模型下载成功!
    )
    
    :: 下载YOLOv5m模型（可选，更准确但更慢）
    echo 是否下载YOLOv5m模型? (更准确但更慢) (y/n)
    set /p download_medium=
    if /i "%download_medium%"=="y" (
        echo 正在下载YOLOv5m模型...
        powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt' -OutFile 'models/yolov5m.pt'"
        if %errorlevel% neq 0 (
            echo 警告: 下载YOLOv5m模型失败，请手动下载。
        ) else (
            echo YOLOv5m模型下载成功!
        )
    )
)

echo.
echo 安装完成!
echo.
echo 使用示例:
echo python smoke_detection/demo/example.py --input data/images --mode visible --remove_smoke --visualize
echo.
echo 有关更多信息，请参阅README.md和文档。

:: 如果在虚拟环境中，提供退出虚拟环境的信息
if /i "%create_venv%"=="y" (
    echo.
    echo 要退出虚拟环境，请运行: deactivate
)

pause 