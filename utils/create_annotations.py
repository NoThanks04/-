#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具脚本，用于创建YOLO格式的标注文件
"""

import os
import argparse
import glob
from pathlib import Path
import cv2


def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    """
    将(x1,y1,x2,y2)格式的边界框转换为YOLO格式(center_x, center_y, width, height)
    
    Args:
        x1, y1, x2, y2: 边界框坐标
        img_width, img_height: 图像尺寸
    
    Returns:
        tuple: (center_x, center_y, width, height)，均为相对坐标[0,1]
    """
    # 计算中心点和尺寸
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # 转换为相对坐标
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return center_x, center_y, width, height


def create_yolo_annotation(img_path, output_dir, boxes):
    """
    为图像创建YOLO格式的标注文件
    
    Args:
        img_path (str): 图像路径
        output_dir (str): 输出目录
        boxes (list): 边界框列表，格式为 [x1, y1, x2, y2] 或 [x1, y1, x2, y2, class_id]
    """
    # 读取图像以获取尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像 {img_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # 准备输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备输出文件名
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    
    # 写入标注
    with open(output_path, 'w') as f:
        for box in boxes:
            # 获取边界框信息
            if len(box) >= 5:
                x1, y1, x2, y2, class_id = box[:5]
            else:
                x1, y1, x2, y2 = box[:4]
                class_id = 0  # 默认为人类
            
            # 转换为YOLO格式
            center_x, center_y, width, height = convert_to_yolo_format(
                x1, y1, x2, y2, img_width, img_height
            )
            
            # 写入文件
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"已创建标注文件: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建YOLO格式的标注文件')
    parser.add_argument('--input_dir', type=str, required=True, help='图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='标注输出目录')
    args = parser.parse_args()
    
    # 读取图像列表
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))
    
    if not image_paths:
        print(f"错误: 在目录 {args.input_dir} 中未找到图像")
        return
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 处理每张图像
    for img_path in image_paths:
        # 这里只是一个示例，实际项目中应该从某种来源读取或生成边界框
        # 例如，这里假设每张图像中间有一个人
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 创建一个在图像中心的虚拟边界框
        box_width = w // 3
        box_height = h // 2
        center_x = w // 2
        center_y = h // 2
        
        x1 = center_x - box_width // 2
        y1 = center_y - box_height // 2
        x2 = center_x + box_width // 2
        y2 = center_y + box_height // 2
        
        boxes = [[x1, y1, x2, y2, 0]]  # 类别0表示人
        
        # 创建标注文件
        create_yolo_annotation(img_path, args.output_dir, boxes)
    
    print(f"已完成 {len(image_paths)} 个标注文件的创建")


def batch_convert_from_txt(input_txt, input_img_dir, output_dir):
    """
    从文本文件批量转换标注
    
    Args:
        input_txt (str): 输入文本文件，每行包含一个标注
        input_img_dir (str): 图像目录
        output_dir (str): 输出标注目录
    """
    with open(input_txt, 'r') as f:
        lines = f.readlines()
    
    current_img = None
    current_boxes = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 6:
            continue
        
        img_name = parts[0]
        class_id = int(parts[1])
        center_x = float(parts[2])
        center_y = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])
        
        # 如果是新图像
        if current_img != img_name:
            # 保存前一个图像的标注
            if current_img is not None and current_boxes:
                img_path = os.path.join(input_img_dir, current_img)
                create_yolo_annotation(img_path, output_dir, current_boxes)
            
            # 开始新图像的标注
            current_img = img_name
            current_boxes = []
        
        # 添加边界框
        current_boxes.append([
            center_x - width/2, center_y - height/2,  # x1, y1
            center_x + width/2, center_y + height/2,  # x2, y2
            class_id
        ])
    
    # 保存最后一个图像的标注
    if current_img is not None and current_boxes:
        img_path = os.path.join(input_img_dir, current_img)
        create_yolo_annotation(img_path, output_dir, current_boxes)


if __name__ == "__main__":
    main() 