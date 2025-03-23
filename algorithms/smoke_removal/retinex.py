"""
基于Retinex理论的烟雾去除算法
"""

import cv2
import numpy as np

class Retinex:
    """
    基于Retinex理论的烟雾去除算法
    """
    
    def __init__(self, config=None):
        """
        初始化Retinex算法
        
        参数:
            config (dict): 算法配置
        """
        self.config = config or {}
        self.sigma_list = self.config.get('sigma_list', [15, 80, 250])
        self.restore_factor = self.config.get('restore_factor', 6.0)
        self.color_gain = self.config.get('color_gain', 10.0)
        self.color_offset = self.config.get('color_offset', 128)
        
    def _single_scale_retinex(self, image, sigma):
        """
        单尺度Retinex处理
        
        参数:
            image (ndarray): 输入图像
            sigma (float): 高斯模糊的标准差
            
        返回:
            ndarray: 处理后的图像
        """
        # 对图像进行高斯模糊
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # 避免对数运算中的零值
        blurred = np.where(blurred < 1.0, 1.0, blurred)
        image = np.where(image < 1.0, 1.0, image)
        
        # 计算log(图像) - log(模糊图像)
        return np.log10(image) - np.log10(blurred)
    
    def _multi_scale_retinex(self, image):
        """
        多尺度Retinex处理
        
        参数:
            image (ndarray): 输入图像
            
        返回:
            ndarray: 处理后的图像
        """
        retinex = np.zeros_like(image, dtype=np.float32)
        
        # 应用不同尺度的Retinex处理并平均
        for sigma in self.sigma_list:
            retinex += self._single_scale_retinex(image, sigma)
            
        retinex = retinex / len(self.sigma_list)
        
        return retinex
    
    def _color_restoration(self, image, retinex):
        """
        颜色恢复
        
        参数:
            image (ndarray): 原始图像
            retinex (ndarray): Retinex处理后的图像
            
        返回:
            ndarray: 颜色恢复后的图像
        """
        # 计算每个像素的RGB通道总和
        img_sum = np.sum(image, axis=2, keepdims=True)
        img_sum = np.where(img_sum < 1.0, 1.0, img_sum)
        
        # 计算颜色恢复因子
        color_restoration = self.color_gain * (np.log10(image * 1.0 / img_sum) * np.log10(img_sum / 3.0))
        
        # 结合Retinex结果和颜色恢复
        msrcr = retinex + color_restoration
        
        return msrcr
    
    def _gain_offset(self, image):
        """
        增益和偏移调整
        
        参数:
            image (ndarray): 输入图像
            
        返回:
            ndarray: 调整后的图像
        """
        # 应用增益和偏移
        image = image * self.restore_factor
        image = image + self.color_offset
        
        # 裁剪到[0, 255]范围
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    
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
        
        # 将图像转换为浮点型
        img_float = image.astype(np.float32)
        
        # 应用多尺度Retinex
        retinex = self._multi_scale_retinex(img_float)
        
        # 颜色恢复
        msrcr = self._color_restoration(img_float, retinex)
        
        # 应用增益和偏移
        result = self._gain_offset(msrcr)
        
        return result 