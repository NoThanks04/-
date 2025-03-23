"""
基于暗通道先验的去雾算法实现
"""

import cv2
import numpy as np

class Dehaze:
    """
    基于暗通道先验的去雾算法
    
    论文: Single Image Haze Removal Using Dark Channel Prior
    """
    
    def __init__(self, config=None):
        """
        初始化去雾算法
        
        参数:
            config (dict): 算法配置
        """
        self.config = config or {}
        self.omega = self.config.get('omega', 0.95)
        self.t0 = self.config.get('t0', 0.1)
        self.window_size = self.config.get('window_size', 15)
    
    def process(self, image):
        """
        处理图像（去雾）
        
        参数:
            image (ndarray): 输入图像，BGR格式
            
        返回:
            ndarray: 处理后的图像
        """
        # 检查图像是否有效
        if image is None or image.size == 0:
            print("错误: 无效的输入图像")
            return None
        
        # 确保图像是彩色的
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 计算暗通道图像
        dark_channel = self._get_dark_channel(image, self.window_size)
        
        # 估计大气光
        atmosphere = self._estimate_atmosphere(image, dark_channel)
        
        # 估计透射率
        transmission = self._estimate_transmission(image, atmosphere)
        
        # 优化透射率
        refined_transmission = self._refine_transmission(image, transmission)
        
        # 恢复图像
        result = self._recover_image(image, refined_transmission, atmosphere)
        
        return result
    
    def _get_dark_channel(self, image, window_size):
        """
        计算暗通道图像
        
        参数:
            image (ndarray): 输入图像
            window_size (int): 窗口大小
            
        返回:
            ndarray: 暗通道图像
        """
        # 提取最小值通道
        min_channel = np.min(image, axis=2)
        
        # 进行最小值滤波
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def _estimate_atmosphere(self, image, dark_channel):
        """
        估计大气光值
        
        参数:
            image (ndarray): 输入图像
            dark_channel (ndarray): 暗通道图像
            
        返回:
            ndarray: 大气光值
        """
        # 图像尺寸
        height, width = dark_channel.shape
        image_size = height * width
        
        # 选择亮度最高的0.1%像素
        n_pixels = int(max(image_size * 0.001, 1))
        flat_dark = dark_channel.flatten()
        indices = np.argsort(flat_dark)[-n_pixels:]
        
        # 在原图中找出这些点，并计算其平均值作为大气光
        atmosphere = np.zeros(3, dtype=np.float32)
        for i in range(3):
            channel = image[:, :, i].flatten()
            atmosphere[i] = np.max(channel[indices])
        
        return atmosphere
    
    def _estimate_transmission(self, image, atmosphere):
        """
        估计透射率
        
        参数:
            image (ndarray): 输入图像
            atmosphere (ndarray): 大气光值
            
        返回:
            ndarray: 透射率图
        """
        # 归一化图像
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            if atmosphere[i] > 0:
                normalized[:, :, i] = image[:, :, i] / atmosphere[i]
            else:
                normalized[:, :, i] = image[:, :, i]
        
        # 计算暗通道
        dark_channel = self._get_dark_channel(normalized, self.window_size)
        
        # 估计透射率
        transmission = 1 - self.omega * dark_channel
        
        return transmission
    
    def _refine_transmission(self, image, transmission):
        """
        优化透射率图
        
        参数:
            image (ndarray): 输入图像
            transmission (ndarray): 初始透射率图
            
        返回:
            ndarray: 优化后的透射率图
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray) / 255
        
        # 使用导向滤波优化透射率
        refined_transmission = cv2.ximgproc.guidedFilter(
            gray, transmission, r=40, eps=1e-3
        )
        
        # 限制最小透射率
        refined_transmission = np.maximum(refined_transmission, self.t0)
        
        return refined_transmission
    
    def _recover_image(self, image, transmission, atmosphere):
        """
        恢复图像
        
        参数:
            image (ndarray): 输入图像
            transmission (ndarray): 透射率图
            atmosphere (ndarray): 大气光值
            
        返回:
            ndarray: 恢复后的图像
        """
        # 转换为浮点型
        image = image.astype(np.float32)
        
        # 扩展透射率图为3通道
        transmission = np.expand_dims(transmission, axis=2)
        transmission = np.tile(transmission, (1, 1, 3))
        
        # 恢复图像
        result = np.empty_like(image)
        for i in range(3):
            result[:, :, i] = (image[:, :, i] - atmosphere[i]) / transmission[:, :, i] + atmosphere[i]
        
        # 限制值域
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result 