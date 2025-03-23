"""
基于CLAHE（对比度受限的自适应直方图均衡化）的烟雾去除算法
"""

import cv2
import numpy as np

class CLAHE:
    """
    基于CLAHE的烟雾去除算法
    """
    
    def __init__(self, config=None):
        """
        初始化CLAHE算法
        
        参数:
            config (dict): 算法配置
        """
        self.config = config or {}
        self.clip_limit = self.config.get('clip_limit', 2.0)
        self.tile_grid_size = self.config.get('tile_grid_size', [8, 8])
        
        # 创建CLAHE对象
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=tuple(self.tile_grid_size)
        )
    
    def process(self, image):
        """
        处理图像（去烟）
        
        参数:
            image (ndarray): 输入图像，BGR格式
            
        返回:
            ndarray: 处理后的图像
        """
        # 检查图像是否有效
        if image is None or image.size == 0:
            print("错误: 无效的输入图像")
            return None
        
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 分离通道
        l, a, b = cv2.split(lab)
        
        # 对亮度通道应用CLAHE
        l_clahe = self.clahe.apply(l)
        
        # 合并通道
        enhanced_lab = cv2.merge([l_clahe, a, b])
        
        # 转换回BGR颜色空间
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result 