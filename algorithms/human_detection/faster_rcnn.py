"""
基于Faster R-CNN的人体检测算法
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class FasterRCNNDetector:
    """
    基于Faster R-CNN的人体检测器
    """
    
    def __init__(self, config=None, device='cpu'):
        """
        初始化Faster R-CNN检测器
        
        参数:
            config (dict): 检测器配置
            device (str): 运行设备，'cpu'或'cuda'
        """
        self.config = config or {}
        self.device = device
        
        # 获取配置参数
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.person_class_id = 1  # COCO数据集中人类的ID为1
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """
        加载Faster R-CNN模型
        """
        try:
            # 加载预训练的Faster R-CNN模型
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            print("成功加载Faster R-CNN模型")
        except Exception as e:
            print(f"加载Faster R-CNN模型失败: {str(e)}")
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
        
        # 转换为张量
        image_tensor = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # 执行检测
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # 提取边界框
        bboxes = []
        for i, (boxes, labels, scores) in enumerate(zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])):
            if labels.item() == self.person_class_id and scores.item() >= self.confidence_threshold:
                box = boxes.cpu().numpy()
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                confidence = float(scores.item())
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
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 显示置信度
            label = f"人: {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return result_image 