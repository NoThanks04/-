# 烟雾环境下的人体目标检测系统

本项目实现了一个面向烟雾环境下的人体目标检测系统，旨在提高搜救任务中对烟雾环境中人员的检测能力。系统集成了烟雾去除、人体检测和多模态融合等多种先进算法。

## 功能特点

- **多模态输入支持**：支持可见光、红外和热成像等多种成像方式
- **烟雾去除算法**：实现了多种烟雾去除算法，如去雾、CLAHE和Retinex等
- **人体检测算法**：集成了YOLOv5等先进的目标检测模型
- **多模态融合**：实现了多种多模态数据融合策略
- **YOLO格式标注支持**：兼容标准YOLO格式的目标检测标注
- **可视化工具**：提供了结果可视化和标注可视化工具

## 安装说明

### 环境要求

- Python 3.7或更高版本
- CUDA支持（可选，用于GPU加速）

### 快速安装

#### Windows

执行安装脚本并按照提示操作：

```bash
install.bat
```

#### Linux/Mac

```bash
chmod +x install.sh  # 添加执行权限（如果需要）
./install.sh
```

### 手动安装步骤

1. 克隆仓库

```bash
git clone https://github.com/username/smoke_detection.git
cd smoke_detection
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 安装项目（可选）

```bash
pip install -e .
```

## 使用方法

### 基本用法

```bash
python smoke_detection/demo/example.py --input data/images --mode visible --remove_smoke --visualize
```

### 使用YOLO格式标注进行评估

```bash
python smoke_detection/demo/example.py --input data/images --labels_dir data/labels --use_annotations --visualize_annotations
```

### 多模态融合

```bash
python smoke_detection/demo/example.py --input data/multimodal --mode fusion --fusion_algo weighted
```

## 项目结构

```
smoke_detection/
├── algorithms/            # 算法实现
│   ├── smoke_removal/     # 烟雾去除算法
│   ├── human_detection/   # 人体检测算法
│   └── fusion/            # 多模态融合算法
├── data/                  # 数据目录
│   ├── visible/           # 可见光图像
│   ├── infrared/          # 红外图像
│   ├── thermal/           # 热成像图像
│   └── labels/            # YOLO格式标注
├── models/                # 预训练模型
├── utils/                 # 工具函数
├── evaluation/            # 评估脚本
├── demo/                  # 演示脚本
│   └── example.py         # 主演示脚本
└── docs/                  # 文档
    ├── usage.md           # 使用文档
    └── annotation_guide.md # 标注指南
```

## 文档

- [使用文档](smoke_detection/docs/usage.md) - 详细的系统使用说明
- [标注指南](smoke_detection/docs/annotation_guide.md) - YOLO格式标注的使用方法

## 标注说明

本项目使用YOLO格式的标注数据。每个图像对应一个同名的`.txt`文件，每行包含一个目标的标注信息：

```
<class_id> <center_x> <center_y> <width> <height>
```

更多信息请参考[标注指南](smoke_detection/docs/annotation_guide.md)。

## 故障排除

- **依赖安装失败**
  - 确保您的Python版本为3.7或更高
  - 尝试使用`pip install --upgrade pip`更新pip
  - 对于特定依赖问题，请参考[requirements.txt](requirements.txt)文件

- **模型加载错误**
  - 确保已下载所需的预训练模型并放置在`models`目录中
  - 检查配置文件中的模型路径设置

- **CUDA相关错误**
  - 确保已安装与您的PyTorch版本兼容的CUDA版本
  - 尝试使用`--device cpu`参数在CPU上运行

## 许可证

MIT License

## 贡献指南

欢迎提交问题和改进建议。请先查阅现有问题，然后再创建新的问题或拉取请求。 