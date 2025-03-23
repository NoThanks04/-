#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化工具模块，用于在图像上显示检测结果和处理效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time


def visualize_results(original_frame, processed_frame, detections, score_threshold=0.5):
    """
    可视化检测结果
    
    Args:
        original_frame (numpy.ndarray): 原始输入帧
        processed_frame (numpy.ndarray): 经过去烟处理的帧
        detections (list): 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        score_threshold (float): 置信度阈值，只显示高于此阈值的检测结果
    
    Returns:
        numpy.ndarray: 可视化结果图像
    """
    # 使用处理后的帧作为基础
    if len(processed_frame.shape) == 2:
        result = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    else:
        result = processed_frame.copy()
    
    # 在图像上绘制检测框
    for det in detections:
        if det[4] < score_threshold:
            continue
        
        x1, y1, x2, y2, score = det
        
        # 整数化坐标
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 显示置信度
        conf_text = f"{score:.2f}"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(result, conf_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return result


def save_comparison_images(original_frame, processed_frame, detections, output_dir, filename, score_threshold=0.5):
    """
    保存对比图像
    
    Args:
        original_frame (numpy.ndarray): 原始输入帧
        processed_frame (numpy.ndarray): 经过去烟处理的帧
        detections (list): 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        output_dir (str): 输出目录
        filename (str): 输出文件名
        score_threshold (float): 置信度阈值，只显示高于此阈值的检测结果
    """
    # 创建输出目录
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # 生成可视化结果
    result = visualize_results(original_frame, processed_frame, detections, score_threshold)
    
    # 保存图像
    cv2.imwrite(str(output_path / filename), result)


def plot_multi_modal_comparison(frames_dict, detections, output_path=None):
    """
    绘制多模态对比图
    
    Args:
        frames_dict (dict): 包含不同模态帧的字典，如 {'ir': ir_frame, 'thermal': thermal_frame, 'visible': visible_frame}
        detections (list): 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        output_path (str, optional): 输出路径，如果为None则显示图像而不保存
    
    Returns:
        numpy.ndarray: 可视化结果图像
    """
    # 确定子图数量
    n_frames = len(frames_dict)
    if n_frames == 0:
        return None
    
    # 创建图像
    plt.figure(figsize=(15, 5))
    
    # 显示每个模态的图像
    for i, (modality, frame) in enumerate(frames_dict.items(), 1):
        plt.subplot(1, n_frames, i)
        
        # 如果是灰度图转为RGB
        if len(frame.shape) == 2:
            plt.imshow(frame, cmap='gray')
        else:
            # OpenCV图像是BGR，转为RGB
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 绘制检测框
        for det in detections:
            if det[4] < 0.5:  # 置信度阈值
                continue
                
            x1, y1, x2, y2, score = det
            width = x2 - x1
            height = y2 - y1
            
            # 创建矩形
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, 
                               edgecolor=(0, 1.0 * score, 1.0 * (1-score)), 
                               linewidth=2)
            plt.gca().add_patch(rect)
            
            # 添加置信度标签
            plt.text(x1, y1 - 5, f"{score:.2f}", 
                    color='white', 
                    backgroundcolor=(0, 0, 1.0 * score),
                    fontsize=8)
        
        plt.title(f"{modality.upper()}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    # 将matplotlib图像转换为OpenCV格式
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    return img


def draw_fps_info(frame, fps, process_time):
    """
    在帧上绘制FPS和处理时间信息
    
    Args:
        frame (numpy.ndarray): 输入帧
        fps (float): 每秒帧数
        process_time (float): 处理时间（毫秒）
    
    Returns:
        numpy.ndarray: 添加了信息的帧
    """
    result = frame.copy()
    
    # 绘制FPS信息
    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # 绘制处理时间
    cv2.putText(
        result,
        f"Time: {process_time:.1f} ms",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    return result


class Visualizer:
    """
    可视化工具类，用于在图像上绘制检测结果、处理效果和信息
    """
    
    def __init__(self, config=None):
        """
        初始化可视化工具
        
        参数:
            config (dict): 可视化配置
        """
        self.config = config or {}
        self.vis_config = self.config.get('visualization', {})
        
        # 获取配置参数
        self.show_fps = self.vis_config.get('show_fps', True)
        self.show_smoke_removal = self.vis_config.get('show_smoke_removal', True)
        self.show_human_detection = self.vis_config.get('show_human_detection', True)
        self.font_scale = self.vis_config.get('font_scale', 0.6)
        self.font_thickness = self.vis_config.get('font_thickness', 2)
        self.bbox_thickness = self.vis_config.get('bbox_thickness', 2)
        self.bbox_color = self.vis_config.get('bbox_color', [0, 255, 0])
        
        # 帧率计算
        self.prev_frame_time = 0
        self.fps = 0
    
    def draw_bboxes(self, image, bboxes):
        """在图像上绘制边界框和置信度"""
        result = image.copy()
        
        for bbox in bboxes:
            if len(bbox) >= 5:  # 确保包含置信度
                x1, y1, x2, y2, conf = bbox[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 绘制边界框
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 显示标签和置信度
                conf_text = f"Person: {conf:.2f}"
                # 获取文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                # 绘制文本背景
                cv2.rectangle(result, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1),
                            (0, 255, 0), -1)  # -1表示填充矩形
                # 绘制文本
                cv2.putText(result, conf_text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # 黑色文字
        
        return result
    
    def draw_info(self, image, frame_count=None, smoke_removal=None, fusion=None):
        """保持图像不变，不添加任何信息"""
        return image.copy()
    
    def create_mosaic(self, images, titles=None, rows=1, cols=None):
        """
        创建多图像拼接显示
        
        参数:
            images (list): 图像列表
            titles (list, optional): 图像标题列表
            rows (int, optional): 行数
            cols (int, optional): 列数，如果为None则自动计算
            
        返回:
            ndarray: 拼接后的图像
        """
        # 检查输入
        if not images:
            print("错误: 没有输入图像")
            return None
        
        # 计算行列数
        n = len(images)
        if cols is None:
            cols = (n + rows - 1) // rows
        
        # 确保所有图像尺寸相同
        h, w = images[0].shape[:2]
        for i in range(1, n):
            if images[i].shape[:2] != (h, w):
                images[i] = cv2.resize(images[i], (w, h))
        
        # 创建拼接图像
        mosaic = np.zeros(((h + 30) * rows, w * cols, 3), dtype=np.uint8)
        
        # 填充图像和标题
        for i in range(n):
            r, c = i // cols, i % cols
            y, x = r * (h + 30), c * w
            
            # 确保图像是彩色的
            if len(images[i].shape) == 2 or images[i].shape[2] == 1:
                img = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
            else:
                img = images[i].copy()
            
            mosaic[y+30:y+h+30, x:x+w] = img
            
            # 添加标题
            if titles is not None and i < len(titles):
                cv2.putText(
                    mosaic, 
                    titles[i], 
                    (x + 10, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    self.font_scale, 
                    (255, 255, 255), 
                    self.font_thickness
                )
        
        return mosaic
    
    def draw_comparison(self, original, processed, title_original="原始图像", title_processed="处理后图像"):
        """
        创建图像对比显示
        
        参数:
            original (ndarray): 原始图像
            processed (ndarray): 处理后图像
            title_original (str, optional): 原始图像标题
            title_processed (str, optional): 处理后图像标题
            
        返回:
            ndarray: 对比图像
        """
        return self.create_mosaic(
            [original, processed], 
            [title_original, title_processed], 
            rows=1, 
            cols=2
        )

    @staticmethod
    def visualize_detection(frame, detections, score_threshold=0.5):
        """
        在单帧上可视化检测结果
        """
        result = frame.copy()
        
        for det in detections:
            if det[4] < score_threshold:
                continue
                
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 显示标签和置信度
            conf_text = f"Person: {score:.2f}"
            # 获取文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # 绘制文本背景
            cv2.rectangle(result, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1),
                        (0, 255, 0), -1)
            # 绘制文本
            cv2.putText(result, conf_text, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2) 