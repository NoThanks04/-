"""
虚拟检测器模块，用于快速演示
"""

import numpy as np
import random

class DummyDetector:
    """
    虚拟检测器类，生成随机检测框以提高处理速度
    """
    
    def __init__(self, num_boxes=3, confidence_range=(0.5, 0.9)):
        """
        初始化虚拟检测器
        
        参数:
            num_boxes: 每帧生成的检测框数量
            confidence_range: 置信度范围
        """
        self.num_boxes = num_boxes
        self.confidence_range = confidence_range
        self.box_history = []  # 保存历史检测框以保持一定连续性
        
    def detect(self, image):
        """
        检测图像中的人体
        
        参数:
            image: 输入图像
            
        返回:
            list: 检测结果列表，每个检测结果为 [x1, y1, x2, y2, confidence]
        """
        height, width = image.shape[:2]
        
        # 为了保持检测框的连续性，有一定概率使用之前的检测框
        use_history = len(self.box_history) > 0 and random.random() < 0.7
        
        if use_history:
            # 对历史框进行微小偏移以模拟运动
            detections = []
            for box in self.box_history:
                # 添加小的随机偏移
                x_offset = random.randint(-10, 10)
                y_offset = random.randint(-5, 5)
                
                x1 = max(0, min(width - 10, box[0] + x_offset))
                y1 = max(0, min(height - 10, box[1] + y_offset))
                w = box[2] - box[0]
                h = box[3] - box[1]
                x2 = min(width, x1 + w)
                y2 = min(height, y1 + h)
                
                # 稍微改变置信度
                confidence = max(0.3, min(0.95, box[4] + random.uniform(-0.05, 0.05)))
                
                detections.append([x1, y1, x2, y2, confidence])
        else:
            # 生成新的随机检测框
            detections = []
            for _ in range(self.num_boxes):
                # 生成合理大小的边界框
                w = random.randint(width // 10, width // 3)
                h = random.randint(height // 5, height // 2)
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)
                x2 = x1 + w
                y2 = y1 + h
                
                # 生成置信度
                confidence = random.uniform(*self.confidence_range)
                
                detections.append([x1, y1, x2, y2, confidence])
        
        # 更新历史记录
        self.box_history = detections.copy()
        
        return detections 