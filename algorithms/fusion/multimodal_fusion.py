#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态融合算法模块，实现红外、热成像和可见光数据的融合
"""

import cv2
import numpy as np
import time


class MultiModalFusion:
    """多模态融合算法实现类"""
    
    def __init__(self, config):
        """
        初始化多模态融合算法
        
        Args:
            config (dict): 配置参数
        """
        self.config = config
        self.method = config['method']
        self.weights = config['weights']
        self.alignment_enabled = config['alignment']['enabled']
        self.alignment_method = config['alignment']['method']
        self.confidence_threshold = config['confidence_threshold']
        
        print(f"初始化多模态融合算法: {self.method}")
    
    def detect(self, ir_frame, thermal_frame, visible_frame):
        """
        融合多模态数据进行人体检测
        
        Args:
            ir_frame (numpy.ndarray): 红外图像
            thermal_frame (numpy.ndarray): 热成像图像
            visible_frame (numpy.ndarray): 可见光图像
            
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        start_time = time.time()
        
        # 图像对齐
        if self.alignment_enabled:
            aligned_frames = self._align_frames(ir_frame, thermal_frame, visible_frame)
            ir_frame, thermal_frame, visible_frame = aligned_frames
        
        # 根据融合方法选择不同的实现
        if self.method == 'feature_fusion':
            detections = self._feature_fusion(ir_frame, thermal_frame, visible_frame)
        elif self.method == 'decision_fusion':
            detections = self._decision_fusion(ir_frame, thermal_frame, visible_frame)
        elif self.method == 'hybrid':
            detections = self._hybrid_fusion(ir_frame, thermal_frame, visible_frame)
        else:
            print(f"未知的融合方法: {self.method}")
            detections = []
        
        process_time = time.time() - start_time
        print(f"多模态融合处理时间: {process_time:.3f}秒")
        
        return detections
    
    def _align_frames(self, ir_frame, thermal_frame, visible_frame):
        """
        对齐多模态图像
        
        Args:
            ir_frame (numpy.ndarray): 红外图像
            thermal_frame (numpy.ndarray): 热成像图像
            visible_frame (numpy.ndarray): 可见光图像
            
        Returns:
            tuple: (aligned_ir, aligned_thermal, aligned_visible)
        """
        # 确保所有帧存在且有效
        frames = [ir_frame, thermal_frame, visible_frame]
        if any(frame is None for frame in frames):
            return ir_frame, thermal_frame, visible_frame
        
        # 将图像转为灰度用于特征匹配
        ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY) if len(ir_frame.shape) == 3 else ir_frame
        thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY) if len(thermal_frame.shape) == 3 else thermal_frame
        visible_gray = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2GRAY) if len(visible_frame.shape) == 3 else visible_frame
        
        if self.alignment_method == 'homography':
            # 使用ORB特征检测器
            orb = cv2.ORB_create()
            
            # 检测特征点和描述符
            kp1, des1 = orb.detectAndCompute(ir_gray, None)
            kp2, des2 = orb.detectAndCompute(thermal_gray, None)
            kp3, des3 = orb.detectAndCompute(visible_gray, None)
            
            # 如果特征点太少，返回原始帧
            if len(kp1) < 10 or len(kp2) < 10 or len(kp3) < 10:
                return ir_frame, thermal_frame, visible_frame
            
            # 特征匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # 将热成像和可见光图像与红外图像对齐
            matches12 = bf.match(des1, des2)
            matches13 = bf.match(des1, des3)
            
            # 根据距离排序
            matches12 = sorted(matches12, key=lambda x: x.distance)
            matches13 = sorted(matches13, key=lambda x: x.distance)
            
            # 取前10个最佳匹配
            good_matches12 = matches12[:10]
            good_matches13 = matches13[:10]
            
            # 获取配对的点
            src_pts12 = np.float32([kp1[m.queryIdx].pt for m in good_matches12]).reshape(-1, 1, 2)
            dst_pts12 = np.float32([kp2[m.trainIdx].pt for m in good_matches12]).reshape(-1, 1, 2)
            
            src_pts13 = np.float32([kp1[m.queryIdx].pt for m in good_matches13]).reshape(-1, 1, 2)
            dst_pts13 = np.float32([kp3[m.trainIdx].pt for m in good_matches13]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            H12, _ = cv2.findHomography(dst_pts12, src_pts12, cv2.RANSAC, 5.0)
            H13, _ = cv2.findHomography(dst_pts13, src_pts13, cv2.RANSAC, 5.0)
            
            # 对齐图像
            h, w = ir_frame.shape[:2]
            aligned_thermal = cv2.warpPerspective(thermal_frame, H12, (w, h))
            aligned_visible = cv2.warpPerspective(visible_frame, H13, (w, h))
            
            return ir_frame, aligned_thermal, aligned_visible
            
        elif self.alignment_method == 'affine':
            # 使用仿射变换对齐
            # 提取红外图像和热成像图像的特征点
            MAX_FEATURES = 500
            GOOD_MATCH_PERCENT = 0.15
            
            # 初始化ORB检测器
            orb = cv2.ORB_create(MAX_FEATURES)
            
            # 检测特征点
            kp1, des1 = orb.detectAndCompute(ir_gray, None)
            kp2, des2 = orb.detectAndCompute(thermal_gray, None)
            kp3, des3 = orb.detectAndCompute(visible_gray, None)
            
            # 特征匹配
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches12 = matcher.match(des1, des2, None)
            matches13 = matcher.match(des1, des3, None)
            
            # 根据距离排序
            matches12.sort(key=lambda x: x.distance, reverse=False)
            matches13.sort(key=lambda x: x.distance, reverse=False)
            
            # 移除距离较远的匹配
            numGoodMatches12 = int(len(matches12) * GOOD_MATCH_PERCENT)
            matches12 = matches12[:numGoodMatches12]
            
            numGoodMatches13 = int(len(matches13) * GOOD_MATCH_PERCENT)
            matches13 = matches13[:numGoodMatches13]
            
            # 提取配对点
            points1_12 = np.zeros((len(matches12), 2), dtype=np.float32)
            points2_12 = np.zeros((len(matches12), 2), dtype=np.float32)
            
            for i, match in enumerate(matches12):
                points1_12[i, :] = kp1[match.queryIdx].pt
                points2_12[i, :] = kp2[match.trainIdx].pt
            
            points1_13 = np.zeros((len(matches13), 2), dtype=np.float32)
            points2_13 = np.zeros((len(matches13), 2), dtype=np.float32)
            
            for i, match in enumerate(matches13):
                points1_13[i, :] = kp1[match.queryIdx].pt
                points2_13[i, :] = kp3[match.trainIdx].pt
            
            # 计算仿射变换矩阵
            h, mask = cv2.findHomography(points2_12, points1_12, cv2.RANSAC)
            h2, mask2 = cv2.findHomography(points2_13, points1_13, cv2.RANSAC)
            
            # 应用仿射变换
            height, width = ir_frame.shape[:2]
            aligned_thermal = cv2.warpPerspective(thermal_frame, h, (width, height))
            aligned_visible = cv2.warpPerspective(visible_frame, h2, (width, height))
            
            return ir_frame, aligned_thermal, aligned_visible
        
        else:
            # 如果未知的对齐方法，返回原始帧
            return ir_frame, thermal_frame, visible_frame
    
    def _feature_fusion(self, ir_frame, thermal_frame, visible_frame):
        """
        特征级融合
        
        Args:
            ir_frame (numpy.ndarray): 红外图像
            thermal_frame (numpy.ndarray): 热成像图像
            visible_frame (numpy.ndarray): 可见光图像
            
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        # 为了简化实现，这里使用加权平均作为特征融合的方法
        # 在实际项目中，应该实现更复杂的特征级融合，例如使用卷积网络提取特征
        
        # 确保所有图像具有相同的大小
        h, w = ir_frame.shape[:2]
        if thermal_frame.shape[:2] != (h, w):
            thermal_frame = cv2.resize(thermal_frame, (w, h))
        if visible_frame.shape[:2] != (h, w):
            visible_frame = cv2.resize(visible_frame, (w, h))
        
        # 转换所有图像为灰度
        if len(ir_frame.shape) == 3:
            ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_frame
        
        if len(thermal_frame.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_frame
        
        if len(visible_frame.shape) == 3:
            visible_gray = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2GRAY)
        else:
            visible_gray = visible_frame
        
        # 将灰度图像归一化到[0,1]范围
        ir_norm = ir_gray.astype(np.float32) / 255.0
        thermal_norm = thermal_gray.astype(np.float32) / 255.0
        visible_norm = visible_gray.astype(np.float32) / 255.0
        
        # 加权融合
        ir_weight = self.weights.get('infrared', 0.4)
        thermal_weight = self.weights.get('thermal', 0.3)
        visible_weight = self.weights.get('visible', 0.3)
        
        # 确保权重和为1
        total_weight = ir_weight + thermal_weight + visible_weight
        ir_weight /= total_weight
        thermal_weight /= total_weight
        visible_weight /= total_weight
        
        # 加权融合图像
        fused_image = (ir_norm * ir_weight + 
                        thermal_norm * thermal_weight + 
                        visible_norm * visible_weight)
        
        # 将融合图像转换回uint8
        fused_image = (fused_image * 255).astype(np.uint8)
        
        # 应用简单阈值分割检测人体（实际项目中应该使用更复杂的检测算法）
        # 这里仅作为演示，使用阈值分割和连通区域分析简单检测
        _, thresh = cv2.threshold(fused_image, 150, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选轮廓
        detections = []
        for contour in contours:
            # 计算面积
            area = cv2.contourArea(contour)
            
            # 忽略过小的轮廓
            if area < 500:
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算置信度（这里使用面积的比例作为简单的置信度）
            confidence = min(1.0, area / 10000)
            
            # 如果置信度超过阈值，添加检测结果
            if confidence >= self.confidence_threshold:
                detections.append([x, y, x + w, y + h, confidence])
        
        return detections
    
    def _decision_fusion(self, ir_frame, thermal_frame, visible_frame):
        """
        决策级融合
        
        Args:
            ir_frame (numpy.ndarray): 红外图像
            thermal_frame (numpy.ndarray): 热成像图像
            visible_frame (numpy.ndarray): 可见光图像
            
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        # 简化实现，在实际项目中应该使用独立的检测器处理每个模态
        
        # 对每个模态独立进行人体检测
        # 为简化示例，这里使用简单的边缘检测和轮廓查找
        detections_ir = self._simple_detect(ir_frame)
        detections_thermal = self._simple_detect(thermal_frame)
        detections_visible = self._simple_detect(visible_frame)
        
        # 合并所有检测结果
        all_detections = detections_ir + detections_thermal + detections_visible
        
        # 如果没有检测到目标，返回空列表
        if not all_detections:
            return []
        
        # 使用非最大抑制合并重叠框
        boxes = np.array([d[:4] for d in all_detections])
        scores = np.array([d[4] for d in all_detections])
        
        # 应用非最大抑制
        indices = self._nms(boxes, scores, 0.5)
        
        # 返回NMS后的检测结果
        return [all_detections[i] for i in indices]
    
    def _hybrid_fusion(self, ir_frame, thermal_frame, visible_frame):
        """
        混合融合方法
        
        Args:
            ir_frame (numpy.ndarray): 红外图像
            thermal_frame (numpy.ndarray): 热成像图像
            visible_frame (numpy.ndarray): 可见光图像
            
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        # 实现混合融合，结合特征级和决策级融合的优点
        
        # 先进行特征级融合获取初步检测结果
        feature_detections = self._feature_fusion(ir_frame, thermal_frame, visible_frame)
        
        # 然后进行决策级融合获取另一组检测结果
        decision_detections = self._decision_fusion(ir_frame, thermal_frame, visible_frame)
        
        # 合并两组检测结果
        all_detections = feature_detections + decision_detections
        
        # 如果没有检测到目标，返回空列表
        if not all_detections:
            return []
        
        # 使用非最大抑制合并重叠框
        boxes = np.array([d[:4] for d in all_detections])
        scores = np.array([d[4] for d in all_detections])
        
        # 应用非最大抑制
        indices = self._nms(boxes, scores, 0.5)
        
        # 返回NMS后的检测结果
        return [all_detections[i] for i in indices]
    
    def _simple_detect(self, image):
        """
        使用简单方法检测人体
        
        Args:
            image (numpy.ndarray): 输入图像
            
        Returns:
            list: 检测结果，每个元素为 [x1, y1, x2, y2, confidence]
        """
        # 简化的检测方法，仅用于演示
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 膨胀边缘
        dilated = cv2.dilate(edges, None, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选轮廓
        detections = []
        for contour in contours:
            # 计算面积
            area = cv2.contourArea(contour)
            
            # 忽略过小的轮廓
            if area < 500:
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算长宽比，人体通常是垂直的
            aspect_ratio = h / float(w)
            
            # 筛选可能是人体的轮廓（长宽比大于1.5）
            if aspect_ratio > 1.5:
                # 计算置信度
                confidence = min(1.0, area / 10000)
                
                # 如果置信度超过阈值，添加检测结果
                if confidence >= self.confidence_threshold:
                    detections.append([x, y, x + w, y + h, confidence])
        
        return detections
    
    def _nms(self, boxes, scores, threshold):
        """
        非最大抑制
        
        Args:
            boxes (numpy.ndarray): 边界框数组，形状为 [N, 4]
            scores (numpy.ndarray): 置信度数组，形状为 [N]
            threshold (float): IoU阈值
            
        Returns:
            list: 保留的检测框索引
        """
        # 如果没有框，返回空列表
        if len(boxes) == 0:
            return []
        
        # 转换为浮点数
        boxes = boxes.astype(np.float32)
        
        # 获取坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 计算面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # 按置信度排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算当前框与其余框的IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU小于阈值的框
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return keep 