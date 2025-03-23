"""
基于权重的图像融合算法
"""

import cv2
import numpy as np

class WeightedFusion:
    """
    基于权重的图像融合
    """
    
    def __init__(self, config=None):
        """
        初始化权重融合算法
        
        参数:
            config (dict): 算法配置
        """
        self.config = config or {}
        self.ir_weight = self.config.get('ir_weight', 0.5)
        self.thermal_weight = self.config.get('thermal_weight', 0.5)
        
    def fuse(self, ir_image, thermal_image):
        """
        融合红外和热成像图像
        
        参数:
            ir_image (ndarray): 红外图像
            thermal_image (ndarray): 热成像图像
            
        返回:
            ndarray: 融合后的图像
        """
        # 检查图像是否有效
        if ir_image is None or thermal_image is None:
            print("错误: 无效的输入图像")
            return None
        
        # 确保两张图像尺寸相同
        if ir_image.shape[:2] != thermal_image.shape[:2]:
            print("错误: 两张图像尺寸不一致，需要先调整尺寸")
            # 尝试调整尺寸，以红外图像为基准
            thermal_image = cv2.resize(thermal_image, (ir_image.shape[1], ir_image.shape[0]))
        
        # 确保两张图像为灰度图，如果是彩色图则转为灰度图
        if len(ir_image.shape) == 3:
            ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_image
            
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        # 标准化图像，将灰度值调整到[0,1]范围内
        ir_norm = ir_gray.astype(np.float32) / 255.0
        thermal_norm = thermal_gray.astype(np.float32) / 255.0
        
        # 应用权重融合
        fused_image = self.ir_weight * ir_norm + self.thermal_weight * thermal_norm
        
        # 将结果调整回[0,255]范围并转为8位图像
        fused_image = np.clip(fused_image * 255.0, 0, 255).astype(np.uint8)
        
        # 如果需要，可以应用对比度增强等处理来改善视觉效果
        fused_image = cv2.equalizeHist(fused_image)
        
        # 转为彩色图像用于可视化和后续处理
        fused_color = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
        
        return fused_color
    
    def adjust_weights(self, ir_weight, thermal_weight=None):
        """
        调整融合权重
        
        参数:
            ir_weight (float): 红外图像权重
            thermal_weight (float, optional): 热成像图像权重，如果为None则自动计算为1-ir_weight
        """
        if ir_weight < 0 or ir_weight > 1:
            print("错误: 权重必须在0到1之间")
            return
        
        self.ir_weight = ir_weight
        
        if thermal_weight is None:
            self.thermal_weight = 1.0 - ir_weight
        else:
            if thermal_weight < 0 or thermal_weight > 1:
                print("错误: 权重必须在0到1之间")
                return
            self.thermal_weight = thermal_weight 