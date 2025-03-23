#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from pathlib import Path


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化YOLO格式标注')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录')
    parser.add_argument('--label_dir', type=str, required=True, help='YOLO格式标注目录')
    parser.add_argument('--output_dir', type=str, default='visualization', help='输出可视化结果的目录')
    parser.add_argument('--class_names', type=str, default='person', help='类别名称，用逗号分隔')
    parser.add_argument('--color', type=str, default='255,0,0', help='边界框颜色，BGR格式，用逗号分隔')
    parser.add_argument('--thickness', type=int, default=2, help='边界框线条粗细')
    return parser.parse_args()


def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """将YOLO格式的相对坐标转换为绝对像素坐标"""
    x1 = int((x_center - width/2) * img_width)
    y1 = int((y_center - height/2) * img_height)
    x2 = int((x_center + width/2) * img_width)
    y2 = int((y_center + height/2) * img_height)
    
    # 确保坐标在图像范围内
    x1 = max(0, min(x1, img_width-1))
    y1 = max(0, min(y1, img_height-1))
    x2 = max(0, min(x2, img_width-1))
    y2 = max(0, min(y2, img_height-1))
    
    return (x1, y1, x2, y2)


def visualize_annotation(image_path, label_path, output_path, class_names, color, thickness):
    """可视化单个图像的标注"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return False
    
    if not os.path.exists(label_path):
        print(f"警告: 标注文件不存在: {label_path}")
        return False
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像: {image_path}")
        return False
    
    img_height, img_width = image.shape[:2]
    
    # 创建图像副本
    vis_image = image.copy()
    
    # 读取标注文件
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print(f"警告: 标注文件为空: {label_path}")
    
    # 遍历每个标注
    for i, line in enumerate(lines):
        line = line.strip().split()
        if len(line) != 5:
            print(f"警告: 标注格式错误，应为5个值: {line}")
            continue
        
        try:
            class_id = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])
            
            # 检查坐标是否在[0,1]范围内
            if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                print(f"警告: 坐标值超出[0,1]范围: {line}")
                continue
            
            # 转换为绝对坐标
            x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            
            # 获取类别名称
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"类别{class_id}"
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制类别标签
            label_size, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, label_size[1])
            cv2.rectangle(vis_image, (x1, label_y - label_size[1]), (x1 + label_size[0], label_y + baseline), color, -1)
            cv2.putText(vis_image, class_name, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except ValueError as e:
            print(f"错误: 无法解析标注值: {line}, 错误: {e}")
            continue
    
    # 保存可视化结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    return True


def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 解析类别名称和颜色
    class_names = args.class_names.split(',')
    try:
        color = tuple(map(int, args.color.split(',')))
    except:
        print("警告: 颜色格式错误，使用默认颜色 (255,0,0)")
        color = (255, 0, 0)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取图像文件列表
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_files.extend(list(Path(args.image_dir).glob(f'*.{ext}')))
        image_files.extend(list(Path(args.image_dir).glob(f'*.{ext.upper()}')))
    
    # 处理每个图像文件
    success_count = 0
    for image_file in image_files:
        # 构建对应的标注文件路径
        label_file = Path(args.label_dir) / f"{image_file.stem}.txt"
        # 构建输出文件路径
        output_file = Path(args.output_dir) / f"{image_file.stem}_annotated{image_file.suffix}"
        
        print(f"处理: {image_file}")
        if visualize_annotation(
            str(image_file), 
            str(label_file), 
            str(output_file), 
            class_names, 
            color, 
            args.thickness
        ):
            success_count += 1
    
    print(f"完成! 成功处理 {success_count}/{len(image_files)} 个图像")
    if success_count > 0:
        print(f"可视化结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main() 