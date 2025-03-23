"""
基于注意力机制的图像融合算法
"""

import cv2
import numpy as np

class AttentionFusion:
    """
    基于注意力机制的图像融合
    """
    
    def __init__(self, config=None):
        """
        初始化注意力融合算法
        
        参数:
            config (dict): 算法配置
        """
        self.config = config or {}
        self.kernel_size = self.config.get('kernel_size', 3)
        self.blur_size = self.config.get('blur_size', 5)
        self.alpha = self.config.get('alpha', 0.7)  # 对比度参数
        self.beta = self.config.get('beta', 0.3)    # 亮度参数
        
    def _calculate_spatial_frequency(self, image):
        """
        计算图像的空间频率，用作注意力权重
        
        参数:
            image (ndarray): 输入图像
            
        返回:
            ndarray: 空间频率图
        """
        # 计算行频率
        rows, cols = image.shape
        row_freq = np.zeros_like(image, dtype=np.float32)
        
        for r in range(rows):
            for c in range(1, cols):
                row_freq[r, c] = (image[r, c] - image[r, c-1]) ** 2
        
        # 计算列频率
        col_freq = np.zeros_like(image, dtype=np.float32)
        
        for r in range(1, rows):
            for c in range(cols):
                col_freq[r, c] = (image[r, c] - image[r-1, c]) ** 2
        
        # 计算总的空间频率
        freq = np.sqrt(row_freq + col_freq)
        
        # 归一化
        freq = cv2.normalize(freq, None, 0, 1, cv2.NORM_MINMAX)
        
        return freq
    
    def _calculate_attention_weights(self, ir_image, thermal_image):
        """
        计算注意力权重
        
        参数:
            ir_image (ndarray): 红外图像
            thermal_image (ndarray): 热成像图像
            
        返回:
            tuple: 红外和热成像的注意力权重
        """
        # 计算空间频率
        ir_freq = self._calculate_spatial_frequency(ir_image)
        thermal_freq = self._calculate_spatial_frequency(thermal_image)
        
        # 计算显著性图
        ir_saliency = cv2.GaussianBlur(ir_freq, (self.blur_size, self.blur_size), 0)
        thermal_saliency = cv2.GaussianBlur(thermal_freq, (self.blur_size, self.blur_size), 0)
        
        # 计算注意力权重
        sum_saliency = ir_saliency + thermal_saliency
        sum_saliency = np.where(sum_saliency == 0, 1e-10, sum_saliency)  # 避免除零
        
        ir_weight = ir_saliency / sum_saliency
        thermal_weight = thermal_saliency / sum_saliency
        
        return ir_weight, thermal_weight
    
    def _enhance_contrast(self, image):
        """
        增强图像对比度
        
        参数:
            image (ndarray): 输入图像
            
        返回:
            ndarray: 增强后的图像
        """
        # 使用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
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
        
        # 增强对比度
        ir_enhanced = self._enhance_contrast(ir_gray)
        thermal_enhanced = self._enhance_contrast(thermal_gray)
        
        # 计算注意力权重
        ir_weight, thermal_weight = self._calculate_attention_weights(ir_enhanced, thermal_enhanced)
        
        # 应用注意力融合
        fused_image = ir_weight * ir_enhanced + thermal_weight * thermal_enhanced
        
        # 将结果调整到[0,255]范围
        fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
        
        # 应用对比度和亮度调整
        fused_image = cv2.convertScaleAbs(fused_image, alpha=self.alpha, beta=self.beta)
        
        # 转为彩色图像用于可视化和后续处理
        fused_color = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
        
        return fused_color 