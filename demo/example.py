#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
演示脚本，展示如何使用人体目标检测系统
"""

import sys
import os
from pathlib import Path
import cv2
import time
import argparse
import yaml
import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from algorithms.smoke_removal import get_smoke_removal_algorithm
from algorithms.human_detection import get_human_detection_algorithm
from algorithms.fusion import get_fusion_algorithm
from utils.visualization import visualize_detection, visualize_multimodal


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='烟雾环境下的人体目标检测示例')
    
    # 输入参数
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入视频/图像路径或目录')
    parser.add_argument('--mode', type=str, default='visible', choices=['visible', 'infrared', 'thermal', 'fusion'], 
                        help='输入模式: 可见光/红外/热成像/融合')
    
    # 处理参数
    parser.add_argument('--remove_smoke', action='store_true', help='是否进行烟雾去除')
    parser.add_argument('--smoke_algo', type=str, default='dehaze', 
                        choices=['dehaze', 'clahe', 'retinex'], help='烟雾去除算法')
    parser.add_argument('--detection_algo', type=str, default='yolov5', 
                        choices=['yolov5', 'faster_rcnn', 'ssd'], help='目标检测算法')
    parser.add_argument('--fusion_algo', type=str, default='weighted', 
                        choices=['weighted', 'attention'], help='多模态融合算法')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--save_video', action='store_true', help='是否保存视频')
    parser.add_argument('--fps', type=int, default=30, help='输出视频的帧率')
    
    # YOLO标注相关参数
    parser.add_argument('--use_annotations', action='store_true', help='是否使用YOLO格式标注进行评估')
    parser.add_argument('--labels_dir', type=str, default='labels', help='YOLO格式标注目录')
    parser.add_argument('--visualize_annotations', action='store_true', help='是否可视化YOLO格式标注')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}，使用默认配置")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_input_files(input_path, mode):
    """获取输入文件列表"""
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    
    # 如果输入是目录，则获取目录中的所有图像/视频文件
    if os.path.isdir(input_path):
        files = []
        # 支持的图像格式
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            files.extend(list(Path(input_path).glob(f'*.{ext}')))
            files.extend(list(Path(input_path).glob(f'*.{ext.upper()}')))
        
        # 支持的视频格式
        for ext in ['mp4', 'avi', 'mov', 'mkv']:
            files.extend(list(Path(input_path).glob(f'*.{ext}')))
            files.extend(list(Path(input_path).glob(f'*.{ext.upper()}')))
        
        if not files:
            raise FileNotFoundError(f"在目录中未找到支持的图像/视频文件: {input_path}")
        
        return sorted(files)
    
    # 如果输入是单个文件
    return [Path(input_path)]


def setup_output_directory(output_dir, mode):
    """设置输出目录"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建特定模式的子目录
    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    
    return mode_dir


def create_video_writer(output_path, fps, frame_size):
    """创建视频写入器"""
    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 创建视频写入器
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def _load_yolo_annotations(annotation_path):
    """加载YOLO格式标注"""
    if not os.path.exists(annotation_path):
        return []
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        line = line.strip().split()
        if len(line) != 5:
            continue
        
        try:
            class_id = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])
            
            annotations.append({
                'class_id': class_id,
                'center_x': x_center,
                'center_y': y_center,
                'width': width,
                'height': height
            })
        except ValueError:
            continue
    
    return annotations


def _convert_yolo_to_bbox(annotation, img_width, img_height):
    """将YOLO格式标注转换为边界框坐标"""
    x_center = annotation['center_x']
    y_center = annotation['center_y']
    width = annotation['width']
    height = annotation['height']
    
    x1 = int((x_center - width/2) * img_width)
    y1 = int((y_center - height/2) * img_height)
    x2 = int((x_center + width/2) * img_width)
    y2 = int((y_center + height/2) * img_height)
    
    # 确保坐标在图像范围内
    x1 = max(0, min(x1, img_width-1))
    y1 = max(0, min(y1, img_height-1))
    x2 = max(0, min(x2, img_width-1))
    y2 = max(0, min(y2, img_height-1))
    
    return [x1, y1, x2, y2]


def _compute_iou(box1, box2):
    """计算两个边界框的IoU（交并比）"""
    # 获取边界框的坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # 检查是否有交集
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # 计算交集面积
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算两个边界框的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    return intersection / union


def _evaluate_detection(detections, annotations, img_width, img_height, iou_threshold=0.5):
    """评估检测结果与标注的匹配程度"""
    if not annotations:
        return 0, 0, 0  # 没有真实标注，无法评估
    
    # 转换标注为边界框坐标
    gt_boxes = [_convert_yolo_to_bbox(anno, img_width, img_height) for anno in annotations]
    
    # 提取检测结果的边界框和置信度
    pred_boxes = [[int(d['bbox'][0]), int(d['bbox'][1]), int(d['bbox'][2]), int(d['bbox'][3])] for d in detections]
    
    # 匹配检测结果和标注
    matched_gt = [False] * len(gt_boxes)
    matched_pred = [False] * len(pred_boxes)
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if matched_gt[j]:
                continue  # 已匹配的标注不再考虑
            
            iou = _compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            matched_gt[best_gt_idx] = True
            matched_pred[i] = True
    
    # 计算评估指标
    tp = sum(matched_pred)  # 正确检测的数量
    fp = len(pred_boxes) - tp  # 误检的数量
    fn = len(gt_boxes) - sum(matched_gt)  # 漏检的数量
    
    return tp, fp, fn


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 获取输入文件列表
    input_files = get_input_files(args.input, args.mode)
    
    # 设置输出目录
    output_dir = setup_output_directory(args.output, args.mode)
    
    # 初始化算法
    # 1. 烟雾去除算法
    smoke_removal = None
    if args.remove_smoke:
        smoke_removal = get_smoke_removal_algorithm(
            algorithm=args.smoke_algo,
            config=config.get('smoke_removal', {})
        )
    
    # 2. 人体检测算法
    human_detection = get_human_detection_algorithm(
        algorithm=args.detection_algo,
        config=config.get('human_detection', {})
    )
    
    # 3. 多模态融合算法（如果需要）
    fusion = None
    if args.mode == 'fusion':
        fusion = get_fusion_algorithm(
            algorithm=args.fusion_algo,
            config=config.get('fusion', {})
        )
    
    # 处理性能指标统计变量
    processing_times = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 是否需要创建视频写入器
    video_writer = None
    
    # 处理每个输入文件
    for input_file in input_files:
        print(f"处理: {input_file}")
        
        # 检查文件是否是视频
        is_video = str(input_file).lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        # 打开输入
        if is_video:
            cap = cv2.VideoCapture(str(input_file))
            if not cap.isOpened():
                print(f"错误: 无法打开视频: {input_file}")
                continue
            
            # 获取视频信息
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = args.fps if args.fps > 0 else int(cap.get(cv2.CAP_PROP_FPS))
            
            # 创建视频写入器
            if args.save_video:
                output_video_path = os.path.join(output_dir, f"{input_file.stem}_result.mp4")
                video_writer = create_video_writer(output_video_path, fps, (frame_width, frame_height))
        else:
            # 读取图像
            frame = cv2.imread(str(input_file))
            if frame is None:
                print(f"错误: 无法读取图像: {input_file}")
                continue
            
            # 获取图像尺寸
            frame_height, frame_width = frame.shape[:2]
        
        # 处理帧
        frame_count = 0
        while True:
            if is_video:
                # 读取视频帧
                ret, frame = cap.read()
                if not ret:
                    break
            
            # 开始计时
            start_time = time.time()
            
            # 1. 烟雾去除（如果需要）
            processed_frame = frame.copy()
            if args.remove_smoke and smoke_removal is not None:
                processed_frame = smoke_removal.process(processed_frame)
            
            # 2. 人体检测
            detections = human_detection.detect(processed_frame)
            
            # 3. 结果可视化
            result_frame = processed_frame.copy()
            if args.visualize:
                result_frame = visualize_detection(result_frame, detections)
            
            # 4. 评估检测结果（如果使用标注）
            if args.use_annotations and not is_video:
                # 构建标注文件路径
                annotation_path = os.path.join(args.labels_dir, f"{input_file.stem}.txt")
                
                # 加载标注
                annotations = _load_yolo_annotations(annotation_path)
                
                # 评估检测结果
                tp, fp, fn = _evaluate_detection(detections, annotations, frame_width, frame_height)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                # 打印当前帧的评估结果
                print(f"  评估结果 - 正确检测: {tp}, 误检: {fp}, 漏检: {fn}")
                
                # 可视化标注（如果需要）
                if args.visualize_annotations:
                    for anno in annotations:
                        # 转换为绝对坐标
                        x1, y1, x2, y2 = _convert_yolo_to_bbox(anno, frame_width, frame_height)
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # 显示处理时间
            if args.visualize:
                cv2.putText(result_frame, f"Time: {processing_time:.3f}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存结果
            if is_video:
                if args.save_video and video_writer is not None:
                    video_writer.write(result_frame)
            else:
                # 保存图像结果
                output_path = os.path.join(output_dir, f"{input_file.stem}_result{input_file.suffix}")
                cv2.imwrite(output_path, result_frame)
            
            # 如果是图像，处理完一次就退出循环
            if not is_video:
                break
            
            frame_count += 1
        
        # 关闭视频相关资源
        if is_video:
            cap.release()
            if args.save_video and video_writer is not None:
                video_writer.release()
                print(f"视频保存到: {output_video_path}")
        else:
            print(f"图像保存到: {output_path}")
    
    # 输出总体性能指标
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"\n平均处理时间: {avg_time:.3f}秒")
    
    # 输出总体评估结果
    if args.use_annotations:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n总体评估结果:")
        print(f"  总正确检测: {total_tp}")
        print(f"  总误检: {total_fp}")
        print(f"  总漏检: {total_fn}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1_score:.4f}")


if __name__ == "__main__":
    main() 