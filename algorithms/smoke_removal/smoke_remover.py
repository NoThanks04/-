#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
去烟算法模块，实现多种烟雾去除算法
"""

import time
import cv2
import numpy as np
from skimage import exposure


class SmokeRemover:
    """烟雾去除算法实现类"""
    
    def __init__(self, config):
        """
        初始化去烟算法
        
        Args:
            config (dict): 算法配置
        """
        self.config = config
        self.method = config['method']
        self.parameters = config['parameters']
        self.enhance = config['enhance']
        self.max_process_time = config['max_process_time']
        
        # 打印配置信息
        print(f"初始化去烟算法: {self.method}")
    
    def remove(self, image):
        """
        应用去烟算法处理图像
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            numpy.ndarray: 处理后的图像
        """
        start_time = time.time()
        
        # 根据方法选择不同的去烟算法
        if self.method == 'dehaze':
            result = self._dehaze(image)
        elif self.method == 'dark_channel':
            result = self._dark_channel_prior(image)
        elif self.method == 'histogram_equalization':
            result = self._histogram_equalization(image)
        elif self.method == 'clahe':
            result = self._clahe(image)
        else:
            print(f"警告: 未知的去烟方法 '{self.method}'，返回原始图像")
            result = image.copy()
        
        # 应用图像增强
        if self.enhance:
            result = self._enhance_image(result)
        
        # 检查处理时间是否超过限制
        process_time = time.time() - start_time
        if process_time > self.max_process_time:
            print(f"警告: 去烟处理时间({process_time:.3f}s)超过限制({self.max_process_time}s)")
        
        return result
    
    def _dehaze(self, image):
        """
        去雾算法（适用于烟雾场景）
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            numpy.ndarray: 去雾处理后的图像
        """
        # 获取参数
        omega = self.parameters.get('omega', 0.95)  # 透射率调整参数
        t0 = self.parameters.get('t0', 0.1)  # 最小透射率
        
        # 转换为浮点型
        img = image.astype(np.float32) / 255.0
        
        # 分离RGB通道
        b, g, r = cv2.split(img)
        
        # 计算暗通道
        dark_channel = np.minimum(np.minimum(r, g), b)
        
        # 使用最大滤波获取大气光
        size = self.parameters.get('window_size', 15)
        kernel = np.ones((size, size), np.float32)
        dark_channel = cv2.erode(dark_channel, kernel)
        
        # 估计大气光
        flat_dark = dark_channel.flatten()
        flat_img = img.reshape(-1, 3)
        indices = np.argsort(flat_dark)[-int(0.001 * len(flat_dark)):]
        atmospheric = np.max(flat_img[indices], axis=0)
        
        # 计算透射率图
        transmission = 1 - omega * dark_channel
        transmission = np.maximum(transmission, t0)
        
        # 去雾处理
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atmospheric[i]) / transmission + atmospheric[i]
        
        # 裁剪到[0,1]范围并转回uint8
        result = np.clip(result, 0, 1) * 255
        result = result.astype(np.uint8)
        
        return result
    
    def _dark_channel_prior(self, image):
        """
        基于暗通道先验的去雾算法
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 获取参数
        size = self.parameters.get('window_size', 15)
        omega = self.parameters.get('omega', 0.95)
        t0 = self.parameters.get('t0', 0.1)
        
        # 归一化图像
        norm_img = image.astype(np.float32) / 255.0
        
        # 获取图像大小
        height, width, _ = norm_img.shape
        
        # 计算暗通道
        dark_channel = np.zeros((height, width))
        for i in range(3):
            dark_channel = np.minimum(dark_channel, norm_img[:, :, i])
        
        # 使用最小值滤波
        kernel = np.ones((size, size), np.uint8)
        dark_channel = cv2.erode(dark_channel, kernel)
        
        # 计算大气光
        num_pixels = height * width
        num_brightest = int(0.001 * num_pixels)
        dark_vec = dark_channel.reshape(num_pixels)
        img_vec = norm_img.reshape(num_pixels, 3)
        indices = np.argsort(dark_vec)
        brightest_indices = indices[num_pixels - num_brightest:num_pixels]
        
        # 从最亮的像素中选择亮度最高的作为大气光
        a_candidates = img_vec[brightest_indices]
        a_brightnesses = np.sum(a_candidates, axis=1)
        a_idx = np.argmax(a_brightnesses)
        A = a_candidates[a_idx]
        
        # 计算透射率
        transmission = 1 - omega * dark_channel
        transmission = np.clip(transmission, t0, 1.0)
        
        # 应用透射率恢复图像
        result = np.zeros_like(norm_img)
        for i in range(3):
            result[:, :, i] = (norm_img[:, :, i] - A[i]) / transmission + A[i]
        
        # 转回uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def _histogram_equalization(self, image):
        """
        直方图均衡化增强
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 对V通道应用直方图均衡化
        v_eq = cv2.equalizeHist(v)
        
        # 合并通道
        hsv_eq = cv2.merge([h, s, v_eq])
        
        # 转回BGR
        result = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _clahe(self, image):
        """
        对比度受限的自适应直方图均衡化
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道应用CLAHE
        l_clahe = clahe.apply(l)
        
        # 合并通道
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # 转回BGR
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _enhance_image(self, image):
        """
        增强图像对比度和亮度
        
        Args:
            image (numpy.ndarray): 输入图像
        
        Returns:
            numpy.ndarray: 增强后的图像
        """
        # 获取增强参数
        contrast = self.enhance.get('contrast', 1.0)
        brightness = self.enhance.get('brightness', 1.0)
        
        # 应用对比度增强
        if contrast != 1.0:
            # 使用自适应对比度增强
            image = exposure.rescale_intensity(image)
            
            # 调整对比度
            mean = np.mean(image, axis=(0, 1))
            image = (image - mean) * contrast + mean
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 应用亮度增强
        if brightness != 1.0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 调整亮度
            v = v * brightness
            v = np.clip(v, 0, 255).astype(np.uint8)
            
            hsv = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image 