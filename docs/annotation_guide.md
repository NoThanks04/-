# YOLO格式标注指南

本文档描述如何在烟雾环境下人体目标检测项目中使用YOLO格式的标注数据。

## YOLO标注格式

YOLO标注格式是一种常用的目标检测标注格式，每个图像对应一个同名的`.txt`文件，其中每行代表一个目标的标注信息，格式如下：

```
<class_id> <center_x> <center_y> <width> <height>
```

其中:
- `class_id`: 类别ID，通常0表示人类
- `center_x`, `center_y`: 边界框中心点的相对坐标，范围[0,1]
- `width`, `height`: 边界框的相对宽度和高度，范围[0,1]

例如，一个值为`0 0.410417 0.262963 0.139583 0.522222`的标注表示:
- 类别ID为0（人类）
- 边界框中心位于图像宽度的0.410417处（从左到右）和高度的0.262963处（从上到下）
- 边界框宽度为图像宽度的0.139583，高度为图像高度的0.522222

## 目录结构

在本项目中，YOLO格式的标注文件应按以下结构组织:

```
smoke_detection/data/
├── visible/             # 可见光图像
│   ├── train/           # 训练集
│   │   ├── image1.jpg
│   │   ├── ...
│   └── val/             # 验证集
│       ├── image1.jpg
│       ├── ...
├── infrared/            # 红外图像
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── ...
│   └── val/
│       ├── image1.jpg
│       ├── ...
├── thermal/             # 热成像图像
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── ...
│   └── val/
│       ├── image1.jpg
│       ├── ...
└── labels/              # YOLO格式标注文件
    ├── train/
    │   ├── image1.txt
    │   ├── ...
    └── val/
        ├── image1.txt
        ├── ...
```

每个标注文件的名称应与对应的图像文件名相同（除了扩展名）。

## 配置说明

在配置文件`config.yaml`中，设置以下参数以使用YOLO格式标注：

```yaml
data:
  infrared_dir: "data/infrared"
  thermal_dir: "data/thermal"
  visible_dir: "data/visible"
  annotation_file: "data/labels"
  # ...其他配置...
```

## 创建YOLO格式标注

本项目提供了工具脚本`utils/create_annotations.py`来帮助创建YOLO格式的标注文件。

### 方法1: 从图像批量生成标注

该方法会为目录中的每张图像创建一个示例标注文件（图像中央有一个边界框）：

```bash
python utils/create_annotations.py --input_dir data/infrared --output_dir data/labels
```

### 方法2: 手动创建标注文件

您也可以手动创建标注文件，每个文件的格式如下：

```
0 0.410417 0.262963 0.139583 0.522222
0 0.623456 0.789012 0.098765 0.432109
```

每行对应一个目标，第一个数字是类别ID（0表示人类），后面四个数字分别是中心点坐标和宽高（相对于图像尺寸的比例）。

### 方法3: 从绝对坐标转换

如果您有(x1, y1, x2, y2)格式的绝对坐标边界框，可以使用以下公式转换为YOLO格式：

```
center_x = (x1 + x2) / (2 * img_width)
center_y = (y1 + y2) / (2 * img_height)
width = (x2 - x1) / img_width
height = (y2 - y1) / img_height
```

## 可视化标注

为了验证标注是否正确，本项目提供了可视化工具`utils/visualize_annotations.py`，可以将标注绘制到图像上并保存结果。

### 使用方法

```bash
python utils/visualize_annotations.py --image_dir data/infrared --label_dir data/labels --output_dir visualization
```

### 参数说明

- `--image_dir`: 图像目录
- `--label_dir`: YOLO格式标注文件目录
- `--output_dir`: 输出可视化结果的目录 (默认: visualization)
- `--class_names`: 类别名称，用逗号分隔 (默认: person)
- `--color`: 边界框颜色，BGR格式，用逗号分隔 (默认: 255,0,0，即蓝色)
- `--thickness`: 边界框线条粗细 (默认: 2)

### 示例

对红外图像及其标注进行可视化：

```bash
python utils/visualize_annotations.py --image_dir data/infrared --label_dir data/labels --output_dir visualization/infrared
```

对多种类别进行不同颜色的可视化：

```bash
python utils/visualize_annotations.py --image_dir data/visible --label_dir data/labels --output_dir visualization/visible --class_names person,dog,cat --color 255,0,0
```

可视化结果将保存在指定的输出目录中，文件名格式为`<原图像名>_annotated.<原扩展名>`。

## 常见问题

1. **标注文件找不到**
   - 确保标注文件与图像文件同名（除了扩展名）
   - 检查配置文件中的`annotation_file`路径是否正确

2. **标注格式错误**
   - 确保每行有5个数字：类别ID和四个相对坐标
   - 确保所有坐标值都在[0,1]范围内
   - 使用可视化工具检查标注是否正确显示

3. **没有识别到人**
   - 确保类别ID为0（表示人类）
   - 确保边界框坐标正确无误
   - 检查模型是否适用于烟雾环境中的人体检测

4. **多个模态的图像使用相同的标注**
   - 如果红外、热成像和可见光图像已正确对齐，可以使用相同的标注文件
   - 如果图像之间存在偏移，应为每种模态创建单独的标注

## 示例

以下是一个YOLO格式标注文件的示例：

**image1.txt**:
```
0 0.410417 0.262963 0.139583 0.522222
```

这表示`image1.jpg`中有一个人（类别ID为0），边界框中心位于图像宽度的41.0417%和高度的26.2963%处，边界框宽度为图像宽度的13.9583%，高度为图像高度的52.2222%。 