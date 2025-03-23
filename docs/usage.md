# 人体目标判别算法使用文档

## 概述

本系统针对烟雾环境中的人体目标检测，提供了多种算法实现，包括去烟算法、人体检测算法和多模态融合算法。

## 项目结构

```
smoke_detection/
├── algorithms/             # 算法实现
│   ├── smoke_removal/      # 去烟算法
│   ├── human_detection/    # 人体识别算法
│   └── fusion/             # 双模态融合算法
├── utils/                  # 工具函数
├── data/                   # 数据集和处理脚本
├── models/                 # 预训练模型和权重
├── evaluation/             # 评估和测试脚本
├── docs/                   # 文档
└── demo/                   # 演示程序
```

## 环境配置

确保已安装Python 3.7或更高版本，并安装所需依赖：

```bash
pip install -r smoke_detection\requirements.txt
```

## 使用方法

### 1. 准备数据

将数据集放入`data`目录，数据集的组织形式如下：

```
data/
├── infrared/           # 红外图像
├── thermal/            # 热成像图像
├── visible/            # 可见光图像
└── annotations.json    # 标注数据
```

标注数据使用JSON格式，包含每个图像的边界框标注。

### 2. 准备模型

将预训练的YOLOv5模型放入`models`目录：

```
models/
└── yolov5s.pt          # YOLOv5模型权重
```

### 3. 配置设置

根据需求编辑`config.yaml`配置文件，配置包含以下主要部分：

- **数据配置**: 指定数据路径和数据集分割方式
- **去烟算法配置**: 选择去烟算法类型和参数
- **人体检测算法配置**: 选择检测模型和参数
- **多模态融合配置**: 设置融合方法和参数
- **评估配置**: 设置评估指标和参数

### 4. 运行系统

系统支持以下几种运行模式：

#### 演示模式

```bash
python smoke_detection\main.py --mode demo --input <输入视频路径> --output <输出目录> --smoke_removal --fusion
```

参数说明：
- `--input`: 输入视频路径
- `--output`: 输出目录
- `--smoke_removal`: 启用去烟算法
- `--fusion`: 启用多模态融合
- `--device`: 运行设备，可选 'cpu', 'cuda', 'rknn'（默认为'cpu'）

#### 测试模式

```bash
python main.py --mode test --smoke_removal --fusion
```

测试模式会加载测试数据，运行算法，并计算性能指标。

#### 配置自定义测试

```bash
python main.py --config custom_config.yaml --mode test
```

可以通过指定不同的配置文件运行自定义测试。

### 5. 评估结果

测试结果将保存在`evaluation/results`目录下，包括：

- **metrics.json**: 包含精度、召回率、mAP等评估指标
- **pr_curve.png**: 精度-召回率曲线

## 模型和参数调优

### 去烟算法

去烟算法支持多种方法：
- `dehaze`: 去雾算法
- `dark_channel`: 暗通道先验算法
- `histogram_equalization`: 直方图均衡化
- `clahe`: 对比度受限的自适应直方图均衡化

每种方法都有特定参数可以在配置文件中调整，如omega、t0、window_size等。

### 人体检测算法

人体检测支持YOLOv5模型，可以在配置文件中设置以下参数：
- `confidence_threshold`: 置信度阈值
- `nms_threshold`: 非最大抑制IoU阈值
- `input_size`: 输入图像大小

### 多模态融合

融合算法支持三种方法：
- `feature_fusion`: 特征级融合
- `decision_fusion`: 决策级融合
- `hybrid`: 混合融合

可以在配置文件中为不同模态设置权重，以及启用图像对齐。

## 故障排除

1. **模型加载失败**
   - 确保模型文件存在且路径正确
   - 检查是否安装了正确版本的torch和torchvision

2. **GPU加速问题**
   - 确保正确安装了CUDA和cuDNN
   - 使用`--device cuda`参数启用GPU加速

3. **图像对齐错误**
   - 尝试使用不同的对齐方法（homography或affine）
   - 确保不同模态的图像有足够的特征点用于匹配

4. **内存不足**
   - 降低输入图像分辨率
   - 使用批处理而不是一次处理所有图像

## 维护和贡献

如有任何问题或建议，请提交Issue或Pull Request。 