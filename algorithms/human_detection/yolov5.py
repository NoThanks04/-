"""
基于YOLOv5的人体检测算法
"""

import cv2
import torch
import numpy as np
from pathlib import Path

class YOLOv5Detector:
    """
    基于YOLOv5的人体检测器
    """
    
    def __init__(self, config=None, device='cpu'):
        """
        初始化YOLOv5检测器
        
        参数:
            config (dict): 检测器配置
            device (str): 运行设备，'cpu'或'cuda'
        """
        self.config = config or {}
        self.device = device
        
        # 获取配置参数
        self.model_path = self.config.get('model_path', 'models/yolov5s.pt')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.4)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.person_class_id = 0  # YOLOv5中人类的ID为0
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """
        加载YOLOv5模型
        """
        try:
            # 使用torch.hub加载模型
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                        path=self.model_path, force_reload=True)
            
            # 设置模型参数
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.classes = [self.person_class_id]  # 只检测人类
            self.model.to(self.device)
            
            print(f"成功加载YOLOv5模型: {self.model_path}")
        except Exception as e:
            print(f"加载YOLOv5模型失败: {str(e)}")
            # 如果加载失败，尝试直接加载本地模型
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                            path=self.model_path, force_reload=True, 
                                            source='local')
                
                # 设置模型参数
                self.model.conf = self.confidence_threshold
                self.model.iou = self.iou_threshold
                self.model.classes = [self.person_class_id]  # 只检测人类
                self.model.to(self.device)
                
                print(f"通过本地加载成功加载YOLOv5模型: {self.model_path}")
            except Exception as e:
                print(f"本地加载YOLOv5模型失败: {str(e)}")
                self.model = None
    
    def detect(self, image):
        """
        检测图像中的人体
        
        参数:
            image (ndarray): 输入图像，BGR格式
            
        返回:
            list: 检测到的人体边界框列表，每个边界框为[x1, y1, x2, y2, confidence]
        """
        # 检查图像是否有效
        if image is None or image.size == 0:
            print("错误: 无效的输入图像")
            return []
        
        # 检查模型是否成功加载
        if self.model is None:
            print("错误: 模型未加载")
            return []
        
        # BGR转RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        results = self.model(rgb_image)
        detections = results.pandas().xyxy[0]  # 获取边界框坐标
        
        # 过滤只保留人类检测结果
        person_detections = detections[detections['class'] == self.person_class_id]
        
        # 构建结果列表
        bboxes = []
        for _, detection in person_detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = float(detection['confidence'])
            bboxes.append([x1, y1, x2, y2, confidence])
        
        return bboxes
    
    def draw_detections(self, image, bboxes):
        """
        在图像上绘制检测结果
        
        参数:
            image (ndarray): 输入图像
            bboxes (list): 边界框列表，每个边界框为[x1, y1, x2, y2, confidence]
            
        返回:
            ndarray: 绘制了检测结果的图像
        """
        result_image = image.copy()
        
        for bbox in bboxes:
            x1, y1, x2, y2, confidence = bbox
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 显示置信度
            label = f"人: {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image 