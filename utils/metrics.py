#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能评估指标计算模块
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    
    Args:
        box1 (list): 第一个框 [x1, y1, x2, y2]
        box2 (list): 第二个框 [x1, y1, x2, y2]
    
    Returns:
        float: IoU值
    """
    # 获取交集区域坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def calculate_precision_recall(results, iou_threshold, confidence_threshold):
    """
    计算指定置信度阈值下的精度和召回率
    
    Args:
        results (list): 检测结果列表
        iou_threshold (float): IoU阈值
        confidence_threshold (float): 置信度阈值
    
    Returns:
        tuple: (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for result in results:
        detections = result['detections']
        ground_truth = result.get('ground_truth', [])
        
        # 筛选置信度高于阈值的检测结果
        valid_detections = [d for d in detections if d[4] >= confidence_threshold]
        
        # 标记已匹配的真值框
        matched_gt = [False] * len(ground_truth)
        
        # 对每个检测结果
        for detection in valid_detections:
            best_iou = 0.0
            best_gt_idx = -1
            
            # 寻找最佳匹配的真值框
            for i, gt_box in enumerate(ground_truth):
                if matched_gt[i]:
                    continue
                
                iou = calculate_iou(detection[:4], gt_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # 如果IoU大于阈值，匹配成功
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt[best_gt_idx] = True
            else:
                false_positives += 1
        
        # 未匹配的真值框为假阴性
        false_negatives += sum(1 for m in matched_gt if not m)
    
    # 计算精度和召回率
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall


def calculate_ap(precisions, recalls):
    """
    计算平均精度(AP)
    
    Args:
        precisions (list): 精度列表
        recalls (list): 召回率列表
    
    Returns:
        float: AP值
    """
    # 按照召回率排序
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    
    # 计算AP (使用插值平均精度计算方法)
    ap = 0.0
    for i in range(len(sorted_recalls) - 1):
        ap += sorted_precisions[i] * (sorted_recalls[i + 1] - sorted_recalls[i])
    
    return ap


def calculate_map(results, config):
    """
    计算mAP (mean Average Precision)
    
    Args:
        results (list): 检测结果列表
        config (dict): 评估配置
    
    Returns:
        dict: 包含mAP@50和mAP@50-95的字典
    """
    iou_thresholds = [0.5]  # mAP@50
    if "map@50-95" in config['map_types']:
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # mAP@50-95
    
    confidence_thresholds = config['confidence_thresholds']
    
    # 存储不同IoU阈值下的AP值
    aps = []
    
    for iou_threshold in iou_thresholds:
        # 计算不同置信度阈值下的精度和召回率
        precisions = []
        recalls = []
        
        for conf_threshold in confidence_thresholds:
            precision, recall = calculate_precision_recall(results, iou_threshold, conf_threshold)
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算该IoU阈值下的AP
        ap = calculate_ap(precisions, recalls)
        aps.append(ap)
    
    # 计算mAP
    map_value = np.mean(aps)
    
    # 如果只计算mAP@50
    map50 = aps[0] if len(aps) > 0 else 0
    
    return {
        "map@50": map50,
        "map@50-95": map_value if len(iou_thresholds) > 1 else map50
    }


def calculate_metrics(results, config):
    """
    计算所有性能指标
    
    Args:
        results (list): 检测结果列表
        config (dict): 评估配置
    
    Returns:
        dict: 包含各项指标的字典
    """
    metrics = {}
    
    # 按照置信度阈值计算精度和召回率
    precision_recalls = []
    iou_threshold = config['iou_threshold']
    confidence_thresholds = config['confidence_thresholds']
    
    for conf_threshold in confidence_thresholds:
        precision, recall = calculate_precision_recall(results, iou_threshold, conf_threshold)
        precision_recalls.append((precision, recall, conf_threshold))
    
    # 找到最佳置信度阈值（基于F1分数）
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_threshold = 0
    
    for precision, recall, threshold in precision_recalls:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = threshold
    
    # 计算mAP
    map_results = calculate_map(results, config)
    
    # 计算平均处理时间
    avg_process_time = np.mean([r['process_time'] for r in results])
    
    # 保存结果
    metrics['precision'] = best_precision
    metrics['recall'] = best_recall
    metrics['f1_score'] = best_f1
    metrics['best_threshold'] = best_threshold
    metrics['map@50'] = map_results['map@50']
    metrics['map@50-95'] = map_results['map@50-95']
    metrics['avg_process_time'] = avg_process_time
    
    # 绘制PR曲线
    if 'output_dir' in config:
        output_dir = Path(config['output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取PR数据
        precisions = [pr[0] for pr in precision_recalls]
        recalls = [pr[1] for pr in precision_recalls]
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        plt.plot(recalls, precisions, 'o-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(output_dir / 'pr_curve.png')
        
        # 保存结果到JSON文件
        with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics 