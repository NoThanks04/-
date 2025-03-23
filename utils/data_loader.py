#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块，负责加载和预处理多模态数据
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class DataLoader:
    """数据加载器，用于加载和预处理数据集"""
    
    def __init__(self, config):
        """
        初始化数据加载器
        
        参数:
            config (dict): 数据配置字典
        """
        # 设置数据路径
        # 兼容旧配置（使用dataset_path）和新配置（使用独立的目录路径）
        if 'dataset_path' in config:
            self.dataset_path = Path(config['dataset_path'])
            self.infrared_dir = self.dataset_path / config.get('infrared_dir', 'infrared')
            self.thermal_dir = self.dataset_path / config.get('thermal_dir', 'thermal')
            self.fusion_dir = self.dataset_path / config.get('visible_dir', 'visible')
        else:
            # 使用新的配置结构
            self.infrared_dir = Path(config.get('infrared_dir', 'data/infrared'))
            self.thermal_dir = Path(config.get('thermal_dir', 'data/thermal'))
            self.fusion_dir = Path(config.get('fusion_dir', 'data/fusion'))
        
        # 设置标注文件路径
        self.annotation_file = config.get('annotation_file', 'labels')
        
        # 设置最大时间差（用于对齐不同模态数据）
        self.max_time_diff = float(config.get('max_time_diff', 1.0))
        
        # 加载数据集
        self.data = self._load_data()
    
    def _load_data(self):
        """
        加载数据集
        
        返回:
            list: 数据项列表
        """
        data = []
        # 加载红外图像
        infrared_images = self._load_images(self.infrared_dir)
        
        # 加载对应的标注
        annotations = self._load_annotations(self.annotation_file)
        
        # 创建数据项
        for img_path in infrared_images:
            item = {
                'infrared': img_path,
                'thermal': self._find_corresponding_image(img_path, self.thermal_dir),
                'fusion': self._find_corresponding_image(img_path, self.fusion_dir),
                'annotation': self._get_annotation(img_path, annotations)
            }
            data.append(item)
        
        return data
    
    def _load_images(self, directory):
        """
        加载指定目录下的所有图像
        
        参数:
            directory (Path): 图像目录
            
        返回:
            list: 图像路径列表
        """
        # 检查目录是否存在
        if not directory.exists():
            print(f"警告: 目录不存在: {directory}")
            return []
        
        # 获取所有图像文件
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(list(directory.glob(ext)))
            image_paths.extend(list(directory.glob(ext.upper())))
        
        # 递归查找子目录中的图像
        for subdir in directory.iterdir():
            if subdir.is_dir():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_paths.extend(list(subdir.glob(ext)))
                    image_paths.extend(list(subdir.glob(ext.upper())))
        
        return sorted(image_paths)
    
    def _load_annotations(self, annotation_path):
        """
        加载标注数据
        
        参数:
            annotation_path (str): 标注文件路径或目录
            
        返回:
            dict: 标注数据字典，键为图像名，值为标注信息
        """
        annotations = {}
        
        # 如果annotation_path是JSON文件，加载JSON
        if os.path.isfile(annotation_path) and annotation_path.endswith('.json'):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
            except Exception as e:
                print(f"错误: 无法加载标注文件 {annotation_path}: {e}")
        # 如果是目录，则假定使用YOLO格式标注
        elif os.path.isdir(annotation_path):
            annotations = self._load_yolo_annotations(annotation_path)
        else:
            print(f"警告: 标注路径 {annotation_path} 不是有效的文件或目录")
        
        return annotations
    
    def _load_yolo_annotations(self, annotation_dir):
        """
        加载YOLO格式的标注
        
        参数:
            annotation_dir (str): 标注目录路径
            
        返回:
            dict: 标注数据字典，键为图像名，值为标注列表
        """
        annotations = {}
        annotation_dir = Path(annotation_dir)
        
        # 确保目录存在
        if not annotation_dir.exists():
            print(f"警告: 标注目录不存在: {annotation_dir}")
            return annotations
        
        # 遍历目录中的所有.txt文件
        for txt_file in annotation_dir.glob('**/*.txt'):
            image_name = txt_file.stem
            annotations[image_name] = self._parse_yolo_annotation_file(txt_file)
        
        return annotations
    
    def _parse_yolo_annotation_file(self, annotation_file):
        """
        解析单个YOLO格式标注文件
        
        参数:
            annotation_file (Path): 标注文件路径
            
        返回:
            list: 边界框标注列表
        """
        bboxes = []
        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # class_id, x_center, y_center, width, height
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    bboxes.append({
                        'class_id': class_id,
                        'center_x': x_center,
                        'center_y': y_center,
                        'width': width,
                        'height': height
                    })
        except Exception as e:
            print(f"错误: 解析标注文件 {annotation_file} 时出错: {e}")
        
        return bboxes
    
    def _convert_relative_to_absolute(self, bbox, img_width, img_height):
        """
        将相对坐标转换为绝对坐标
        
        参数:
            bbox (dict): 边界框信息
            img_width (int): 图像宽度
            img_height (int): 图像高度
            
        返回:
            tuple: (x1, y1, x2, y2) 表示左上角和右下角的绝对坐标
        """
        x_center = bbox['center_x'] * img_width
        y_center = bbox['center_y'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        return (x1, y1, x2, y2)
    
    def _find_corresponding_image(self, img_path, target_dir):
        """
        在目标目录中查找与给定图像对应的图像
        
        参数:
            img_path (Path): 源图像路径
            target_dir (Path): 目标目录
            
        返回:
            Path: 对应图像的路径，如果未找到则返回None
        """
        # 获取图像名称（不包含扩展名）
        img_name = img_path.stem
        
        # 尝试查找相同名称的图像
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
            target_path = target_dir / f"{img_name}{ext}"
            if target_path.exists():
                return target_path
        
        # 如果有子目录，也在子目录中搜索
        for subdir in target_dir.iterdir():
            if subdir.is_dir():
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                    target_path = subdir / f"{img_name}{ext}"
                    if target_path.exists():
                        return target_path
        
        # 未找到对应图像
        return None
    
    def _get_annotation(self, img_path, annotations):
        """
        获取图像的标注信息
        
        参数:
            img_path (Path): 图像路径
            annotations (dict): 标注数据字典
            
        返回:
            list: 标注信息列表，每个元素为一个标注对象
        """
        # 获取图像名称（不包含扩展名）
        img_name = img_path.stem
        
        # 检查是否存在对应的标注
        if img_name in annotations:
            return annotations[img_name]
        
        # 未找到标注
        return []
    
    def __len__(self):
        """
        返回数据集大小
        
        返回:
            int: 数据项数量
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        参数:
            idx (int): 索引
            
        返回:
            dict: 数据项字典
        """
        item = self.data[idx]
        
        # 加载图像
        infrared_img = self._load_image(item['infrared']) if item['infrared'] else None
        thermal_img = self._load_image(item['thermal']) if item['thermal'] else None
        fusion_img = self._load_image(item['fusion']) if item['fusion'] else None
        
        # 获取标注
        annotations = item['annotation']
        
        # 转换标注为绝对坐标（如果是相对坐标）
        if infrared_img is not None and annotations:
            if 'center_x' in annotations[0]:  # 检测是否为YOLO格式
                h, w = infrared_img.shape[:2]
                abs_annotations = []
                for ann in annotations:
                    # 转换为绝对坐标
                    x1, y1, x2, y2 = self._convert_relative_to_absolute(ann, w, h)
                    abs_annotations.append({
                        'class_id': ann['class_id'],
                        'bbox': [x1, y1, x2, y2]
                    })
                annotations = abs_annotations
        
        return {
            'infrared': infrared_img,
            'thermal': thermal_img,
            'fusion': fusion_img,
            'annotation': annotations,
            'infrared_path': str(item['infrared']) if item['infrared'] else None,
            'thermal_path': str(item['thermal']) if item['thermal'] else None,
            'fusion_path': str(item['fusion']) if item['fusion'] else None
        }
    
    def _load_image(self, img_path):
        """
        加载图像
        
        参数:
            img_path (Path): 图像路径
            
        返回:
            ndarray: 图像数据，如果加载失败则返回None
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法加载图像 {img_path}")
            return img
        except Exception as e:
            print(f"错误: 加载图像 {img_path} 时出错: {e}")
            return None

    def _get_timestamp_from_filename(self, filename):
        """
        从文件名中提取时间戳
        
        Args:
            filename (str): 文件名
        
        Returns:
            datetime: 时间戳对象
        """
        # 假设文件名格式为 "frame_YYYYMMDD_HHMMSS.jpg"
        # 实际项目中根据真实文件命名调整
        try:
            parts = filename.split('_')
            date_part = parts[1]
            time_part = parts[2].split('.')[0]
            
            date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
            time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except:
            # 如果无法解析，返回None
            return None
    
    def _find_synchronized_frames(self, ir_frame_path):
        """
        查找与红外帧时间同步的热成像和可见光帧
        
        Args:
            ir_frame_path (Path): 红外帧路径
        
        Returns:
            tuple: (thermal_frame_path, visible_frame_path)，如果找不到对应帧则为None
        """
        ir_filename = os.path.basename(ir_frame_path)
        ir_timestamp = self._get_timestamp_from_filename(ir_filename)
        
        if ir_timestamp is None:
            return None, None
        
        # 查找最接近的热成像帧
        thermal_frame_path = None
        min_thermal_diff = float('inf')
        
        if os.path.exists(self.thermal_dir):
            for thermal_file in os.listdir(self.thermal_dir):
                thermal_timestamp = self._get_timestamp_from_filename(thermal_file)
                if thermal_timestamp is None:
                    continue
                
                time_diff = abs((thermal_timestamp - ir_timestamp).total_seconds())
                if time_diff < min_thermal_diff and time_diff <= self.max_time_diff:
                    min_thermal_diff = time_diff
                    thermal_frame_path = self.thermal_dir / thermal_file
        
        # 查找最接近的可见光帧
        visible_frame_path = None
        min_visible_diff = float('inf')
        
        if os.path.exists(self.fusion_dir):
            for visible_file in os.listdir(self.fusion_dir):
                visible_timestamp = self._get_timestamp_from_filename(visible_file)
                if visible_timestamp is None:
                    continue
                
                time_diff = abs((visible_timestamp - ir_timestamp).total_seconds())
                if time_diff < min_visible_diff and time_diff <= self.max_time_diff:
                    min_visible_diff = time_diff
                    visible_frame_path = self.fusion_dir / visible_file
        
        return thermal_frame_path, visible_frame_path
    
    def load_test_data(self):
        """
        加载测试数据
        
        Returns:
            list: 测试数据列表，每项包含同步的多模态帧和标注
        """
        test_data = []
        
        # 确保红外目录存在
        if not os.path.exists(self.infrared_dir):
            print(f"错误: 找不到红外图像目录 {self.infrared_dir}")
            return test_data
        
        # 遍历红外帧
        for ir_file in os.listdir(self.infrared_dir):
            ir_frame_path = self.infrared_dir / ir_file
            
            # 读取红外帧
            ir_frame = cv2.imread(str(ir_frame_path))
            if ir_frame is None:
                continue
            
            # 查找同步的热成像和可见光帧
            thermal_frame_path, visible_frame_path = self._find_synchronized_frames(ir_frame_path)
            
            # 读取热成像帧
            thermal_frame = None
            if thermal_frame_path is not None:
                thermal_frame = cv2.imread(str(thermal_frame_path))
            
            # 读取可见光帧
            visible_frame = None
            if visible_frame_path is not None:
                visible_frame = cv2.imread(str(visible_frame_path))
            
            # 获取标注
            ground_truth = self._get_annotation(ir_frame_path, self.data)
            
            # 如果找到标注，并且是相对坐标，转换为绝对坐标
            if ground_truth is not None:
                h, w = ir_frame.shape[:2]
                ground_truth = self._convert_relative_to_absolute(ground_truth, w, h)
            
            # 添加到测试数据
            data_item = {
                'ir': ir_frame,
                'ir_path': str(ir_frame_path),
                'ground_truth': ground_truth
            }
            
            if thermal_frame is not None:
                data_item['thermal'] = thermal_frame
                data_item['thermal_path'] = str(thermal_frame_path)
            
            if visible_frame is not None:
                data_item['visible'] = visible_frame
                data_item['visible_path'] = str(visible_frame_path)
            
            test_data.append(data_item)
        
        print(f"已加载 {len(test_data)} 组测试数据")
        return test_data
    
    def load_batch(self, batch_size=1):
        """
        加载一批数据用于训练
        
        Args:
            batch_size (int): 批大小
        
        Returns:
            dict: 包含多模态数据和标注的批次
        """
        # TODO: 实现批处理逻辑用于训练
        raise NotImplementedError("训练批处理功能尚未实现") 