#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人体检测算法模块，支持多种深度学习模型进行人体检测
"""

import os
import time
import cv2
import numpy as np
import torch


class HumanDetector:
    """人体检测器，支持多种模型和设备"""
    
    def __init__(self, config, device='cpu'):
        """
        初始化人体检测器
        
        Args:
            config (dict): 检测器配置
            device (str): 运行设备，可选 'cpu', 'cuda', 'rknn'
        """
        self.config = config
        self.model_name = config['model']
        self.confidence_threshold = config['confidence_threshold']
        self.nms_threshold = config['nms_threshold']
        self.input_size = tuple(config['input_size'])
        self.device = device
        self.max_process_time = config['max_process_time']
        
        # 加载模型
        self.model = self._load_model()
        
        print(f"初始化人体检测器: {self.model_name} (device: {self.device})")
    
    def _load_model(self):
        """
        加载检测模型
        
        Returns:
            object: 加载的模型
        """
        model_path = self.config.get('weights', '')
        
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}")
            return None
        
        if self.model_name == 'yolov5':
            try:
                # 加载YOLOv5模型
                if self.device == 'cpu' or self.device == 'cuda':
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                    model.conf = self.confidence_threshold
                    model.iou = self.nms_threshold
                    model.to(self.device)
                    if self.device == 'cuda' and torch.cuda.is_available():
                        model.cuda()
                    else:
                        model.cpu()
                elif self.device == 'rknn':
                    # 此处应该实现RKNN模型加载
                    # 由于需要特定环境，这里只是占位实现
                    model = None
                    print("RKNN模型加载尚未实现")
                return model
            except Exception as e:
                print(f"加载YOLOv5模型失败: {e}")
                return None
        
        elif self.model_name == 'yolox':
            # 占位实现
            print("YOLOX模型加载尚未实现")
            return None
        
        else:
            print(f"不支持的模型类型: {self.model_name}")
            return None
    
    def detect(self, image):
        """
        检测图像中的人体
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        start_time = time.time()
        
        # 检查模型是否加载成功
        if self.model is None:
            print("警告: 模型未加载，无法进行检测")
            return []
        
        # 根据不同模型进行检测
        if self.model_name == 'yolov5':
            detections = self._detect_yolov5(image)
        elif self.model_name == 'yolox':
            detections = self._detect_yolox(image)
        else:
            detections = []
        
        # 检查处理时间是否超过限制
        process_time = time.time() - start_time
        if process_time > self.max_process_time:
            print(f"警告: 人体检测处理时间({process_time:.3f}s)超过限制({self.max_process_time}s)")
        
        return detections
    
    def _detect_yolov5(self, image):
        """
        使用YOLOv5模型进行检测
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        if self.device == 'cpu' or self.device == 'cuda':
            # 调整图像大小
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 进行推理
            results = self.model([img])
            
            # 提取结果
            detections = []
            predictions = results.pandas().xyxy[0]
            
            # 只保留人类检测结果（类别为0）
            for _, prediction in predictions.iterrows():
                if prediction['class'] == 0 or prediction['name'] == 'person':
                    box = [
                        prediction['xmin'],
                        prediction['ymin'],
                        prediction['xmax'],
                        prediction['ymax'],
                        prediction['confidence']
                    ]
                    detections.append(box)
            
            return detections
        
        elif self.device == 'rknn':
            # 这里应该是RKNN设备上的实现
            # 由于需要特定环境，这里只是占位实现
            print("RKNN设备上的YOLOv5检测尚未实现")
            return []
    
    def _detect_yolox(self, image):
        """
        使用YOLOX模型进行检测
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        # 占位实现
        print("YOLOX检测尚未实现")
        return []
    
    def _preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            numpy.ndarray: 预处理后的图像
        """
        # 调整图像大小
        resized = cv2.resize(image, self.input_size)
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 添加批次维度
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def _apply_nms(self, boxes, scores, iou_threshold=0.45):
        """
        应用非最大抑制
        
        Args:
            boxes (numpy.ndarray): 边界框，形状为 [N, 4]
            scores (numpy.ndarray): 置信度，形状为 [N]
            iou_threshold (float): IoU阈值
        
        Returns:
            list: 保留的边界框索引
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算当前框与其他框的IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep 