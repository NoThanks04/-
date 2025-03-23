#!/bin/bash

echo "正在安装烟雾环境下的人体目标检测系统..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未检测到Python。请安装Python 3.7或更高版本。"
    exit 1
fi

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "错误: 未检测到pip。请确保pip已正确安装。"
    exit 1
fi

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境? (y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    python3 -m venv venv
    echo "正在激活虚拟环境..."
    source venv/bin/activate
fi

# 安装依赖
echo "正在安装依赖..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误: 安装依赖失败。"
    exit 1
fi

# 安装项目（可选）
read -p "是否安装项目? (y/n): " install_project
if [[ $install_project =~ ^[Yy]$ ]]; then
    pip3 install -e .
    if [ $? -ne 0 ]; then
        echo "错误: 安装项目失败。"
        exit 1
    fi
fi

# 下载预训练模型（如果需要）
read -p "是否下载预训练模型? (y/n): " download_models
if [[ $download_models =~ ^[Yy]$ ]]; then
    echo "正在下载预训练模型..."
    mkdir -p models
    
    # 下载YOLOv5s模型
    echo "正在下载YOLOv5s模型..."
    curl -L -o models/yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
    if [ $? -ne 0 ]; then
        echo "警告: 下载YOLOv5s模型失败，请手动下载。"
    else
        echo "YOLOv5s模型下载成功!"
    fi
    
    # 下载YOLOv5m模型（可选，更准确但更慢）
    read -p "是否下载YOLOv5m模型? (更准确但更慢) (y/n): " download_medium
    if [[ $download_medium =~ ^[Yy]$ ]]; then
        echo "正在下载YOLOv5m模型..."
        curl -L -o models/yolov5m.pt https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt
        if [ $? -ne 0 ]; then
            echo "警告: 下载YOLOv5m模型失败，请手动下载。"
        else
            echo "YOLOv5m模型下载成功!"
        fi
    fi
fi

echo
echo "安装完成!"
echo
echo "使用示例:"
echo "python3 smoke_detection/demo/example.py --input data/images --mode visible --remove_smoke --visualize"
echo
echo "有关更多信息，请参阅README.md和文档。"

# 如果在虚拟环境中，提供退出虚拟环境的信息
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo
    echo "要退出虚拟环境，请运行: deactivate"
fi 