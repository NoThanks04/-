#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
烟雾环境下的人体目标检测系统主入口
"""

import os
import sys
import cv2
import time
import yaml
import argparse
import json
import numpy as np
from pathlib import Path
# import torch

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from algorithms.smoke_removal import get_smoke_removal_algorithm
from algorithms.human_detection import get_human_detection_algorithm
from algorithms.fusion import get_fusion_algorithm
from utils.data_loader import DataLoader
from utils.visualization import Visualizer
from utils.metrics import calculate_metrics


def parse_arguments():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="烟雾环境下的人体目标检测系统")
    
    # 运行模式
    parser.add_argument('--mode', type=str, choices=['demo', 'test', 'train'], default='demo',
                       help="运行模式: demo, test, train")
    
    # 输入/输出路径
    parser.add_argument('--input', type=str, default=None,
                       help="输入视频或图像路径")
    parser.add_argument('--output', type=str, default='results',
                       help="输出路径")
    
    # 功能开关
    parser.add_argument('--smoke_removal', action='store_true',
                       help="启用烟雾去除")
    parser.add_argument('--fusion', action='store_true',
                       help="启用多模态融合")
    
    # 配置和设备
    parser.add_argument('--config', type=str, default='config.yaml',
                       help="配置文件路径")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'rknn'], default='cpu',
                       help="运行设备: cpu, cuda, rknn")
    
    # 性能优化选项
    parser.add_argument('--fast_preview', action='store_true',
                       help="快速预览模式，从视频中间开始处理")
    
    return parser.parse_args()


def load_config(config_path):
    """
    加载配置文件
    
    参数:
        config_path (str): 配置文件路径
        
    返回:
        dict: 配置信息
    """
    try:
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件错误: {e}")
        return {}


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = args.device if args.device else config.get('device', 'cpu')
    
    # 获取性能优化配置
    performance_config = config.get('performance', {})
    skip_model_load = performance_config.get('skip_model_load', False)
    use_dummy_detector = performance_config.get('use_dummy_detector', False)
    
    # 初始化数据加载器
    data_loader = None
    if args.mode == 'test':
        data_loader = DataLoader(config)
    
    # 初始化可视化工具
    visualizer = Visualizer(config)
    
    # 初始化烟雾去除算法
    smoke_remover = None
    if args.smoke_removal:
        smoke_removal_config = config.get('smoke_removal', {})
        algorithm = smoke_removal_config.get('algorithm', 'dehaze')
        smoke_remover = get_smoke_removal_algorithm(algorithm, smoke_removal_config)
    
    # 初始化人体检测算法
    human_detector = None
    if not skip_model_load:
        if use_dummy_detector:
            # 使用虚拟检测器以提高速度
            from algorithms.human_detection.dummy_detector import DummyDetector
            human_detector = DummyDetector()
            print("使用虚拟检测器进行演示")
        else:
            # 使用正常检测器
            human_detection_config = config.get('human_detection', {})
            detector_algorithm = human_detection_config.get('algorithm', 'yolov8')
            model_config = config.get('models', {})
            human_detector = get_human_detection_algorithm(detector_algorithm, model_config, device)
    
    # 初始化融合算法
    fusion_model = None
    if args.fusion and not skip_model_load:
        fusion_config = config.get('fusion', {})
        fusion_algorithm = fusion_config.get('algorithm', 'weighted')
        fusion_model = get_fusion_algorithm(fusion_algorithm, fusion_config)
    
    # 恢复使用YOLOv5模型
    # 示例：假设有一个函数 process_frame(frame) 负责处理每一帧
    # def process_frame(frame):
    #     results = model(frame)
    #     # 处理结果
    #     return results
    
    # 根据模式运行系统
    if args.mode == 'demo':
        run_demo(args, data_loader, smoke_remover, human_detector, fusion_model, visualizer, config)
    elif args.mode == 'test':
        run_test(args, data_loader, smoke_remover, human_detector, fusion_model, config, visualizer)
    else:
        print(f"错误: 未知的运行模式 '{args.mode}'")


def run_demo(args, data_loader, smoke_remover, human_detector, fusion_model, visualizer, config):
    """
    运行演示模式
    
    参数:
        args: 命令行参数
        data_loader: 数据加载器
        smoke_remover: 烟雾去除算法
        human_detector: 人体检测算法
        fusion_model: 融合算法
        visualizer: 可视化工具
        config: 配置信息
    """
    # 检查输入文件
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在")
        return
    
    # 检查输出目录
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开输入视频 '{input_path}'")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取性能优化参数
    performance_config = config.get('performance', {})
    resize_input = performance_config.get('resize_input', False)
    input_width = performance_config.get('input_width', width)
    input_height = performance_config.get('input_height', height)
    process_every_n_frames = performance_config.get('process_every_n_frames', 1)
    disable_visualization = performance_config.get('disable_visualization', False)
    
    print(f"性能设置: 调整输入大小={resize_input}, 输入尺寸={input_width}x{input_height}, 处理帧间隔={process_every_n_frames}, 禁用可视化={disable_visualization}")
    
    # 创建输出视频
    output_path = os.path.join(output_dir, 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理视频
    frame_count = 0
    process_times = []
    last_result_frame = None
    
    print(f"开始处理视频: {input_path}")
    print(f"总帧数: {total_frames}")
    
    # 为了加快处理，预先定位到中间部分（可选）
    if args.fast_preview:
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        frame_count = middle_frame
        print(f"快速预览模式: 跳转到第 {middle_frame} 帧")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 是否处理当前帧
            should_process = (frame_count % process_every_n_frames == 0)
            
            if should_process:
                if (frame_count % 30 == 0) or (frame_count == 1):
                    print(f"处理帧: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
                
                # 调整输入大小以提高速度
                if resize_input:
                    input_frame = cv2.resize(frame, (input_width, input_height))
                else:
                    input_frame = frame.copy()
                
                # 计时开始
                start_time = time.time()
                
                # 处理帧
                try:
                    if human_detector is None:
                        # 如果没有检测器，只进行简单处理
                        if smoke_remover is not None:
                            result_frame = smoke_remover.process(input_frame)
                        else:
                            result_frame = input_frame.copy()
                    else:
                        result_frame = process_frame(input_frame, smoke_remover, human_detector, fusion_model, visualizer, disable_visualization)
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    result_frame = input_frame.copy()
                
                # 如果调整了输入大小，将结果缩放回原始大小
                if resize_input:
                    result_frame = cv2.resize(result_frame, (width, height))
                    
                # 保存上一个处理结果
                last_result_frame = result_frame
                
                # 计时结束
                end_time = time.time()
                process_time = (end_time - start_time) * 1000  # 转换为毫秒
                process_times.append(process_time)
                
                # 如果没有禁用可视化，显示帧率和跳帧信息
                if not disable_visualization:
                    # 显示帧率
                    avg_process_time = sum(process_times[-10:]) / min(len(process_times), 10)
                    fps_text = f"FPS: {1000/avg_process_time:.1f}"
                    cv2.putText(result_frame, fps_text, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 显示跳帧信息
                    if process_every_n_frames > 1:
                        skip_text = f"跳帧: {process_every_n_frames}"
                        cv2.putText(result_frame, skip_text, (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 显示进度信息
                    progress_text = f"Frame: {frame_count}/{total_frames}"
                    cv2.putText(result_frame, progress_text, 
                               (10, height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 0), 2)
                    cv2.putText(result_frame, fps_text, 
                               (10, height - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 0), 2)
            else:
                # 使用上一个处理结果
                if last_result_frame is not None:
                    result_frame = last_result_frame.copy()
                else:
                    result_frame = frame.copy()
            
            # 保存结果
            out.write(result_frame)
            
            # 显示结果
            if not disable_visualization:
                cv2.imshow('Result', result_frame)
                key = cv2.waitKey(1) & 0xFF
                
                # 按q键或Esc键退出
                if key == ord('q') or key == 27:
                    break
                # 按+键增加跳帧
                elif key == ord('+'):
                    process_every_n_frames = min(process_every_n_frames + 1, 10)
                    print(f"跳帧设置为: {process_every_n_frames}")
                # 按-键减少跳帧
                elif key == ord('-'):
                    process_every_n_frames = max(process_every_n_frames - 1, 1)
                    print(f"跳帧设置为: {process_every_n_frames}")
    except Exception as e:
        print(f"处理视频时出错: {e}")
    
    # 统计信息
    if process_times and len(process_times) > 0:
        avg_process_time = sum(process_times) / len(process_times)
        avg_fps = 1000 / avg_process_time if avg_process_time > 0 else 0
        
        print(f"视频处理完成")
        print(f"处理帧数: {len(process_times)}/{total_frames}")
        if avg_process_time > 0:
            print(f"平均处理时间: {avg_process_time:.2f} ms")
            print(f"平均帧率: {avg_fps:.2f} FPS")
    
    print(f"结果已保存到: {output_path}")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def run_test(args, data_loader, smoke_remover, human_detector, fusion_model, config, visualizer):
    """
    运行测试模式
    
    参数:
        args: 命令行参数
        data_loader: 数据加载器
        smoke_remover: 烟雾去除算法
        human_detector: 人体检测算法
        fusion_model: 融合算法
        config: 配置
        visualizer: 可视化工具
    """
    # 检查数据加载器
    if data_loader is None:
        print("错误: 测试模式需要数据加载器")
        return
    
    # 加载测试数据
    test_data = data_loader.load_test_data()
    if not test_data:
        print("错误: 无法加载测试数据")
        return
    
    # 创建输出目录
    output_dir = config.get('output', {}).get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估配置
    eval_config = config.get('evaluation', {})
    
    # 处理每个测试样本
    print(f"开始测试，共 {len(test_data)} 个样本")
    
    results = []
    for i, sample in enumerate(test_data):
        # 打印进度
        if (i+1) % 10 == 0 or i == 0 or i == len(test_data) - 1:
            print(f"处理样本: {i+1}/{len(test_data)} ({(i+1)/len(test_data)*100:.1f}%)")
        
        ir_frame = sample.get('ir')
        thermal_frame = sample.get('thermal')
        visible_frame = sample.get('visible')
        gt_boxes = sample.get('annotations', [])
        
        # 处理帧
        _, detections = process_test_frame(ir_frame, thermal_frame, visible_frame, 
                                          smoke_remover, human_detector, fusion_model, visualizer)
        
        # 添加结果
        results.append({
            'gt_boxes': gt_boxes,
            'detections': detections
        })
        
        # 保存可视化结果
        if config.get('output', {}).get('save_images', False):
            img_path = os.path.join(output_dir, f"sample_{i:04d}.jpg")
            
            # 如果有红外或热成像，展示多模态结果
            if ir_frame is not None or thermal_frame is not None:
                frames_dict = {}
                if ir_frame is not None:
                    frames_dict['ir'] = ir_frame
                if thermal_frame is not None:
                    frames_dict['thermal'] = thermal_frame
                if visible_frame is not None:
                    frames_dict['visible'] = visible_frame
                
                multi_modal_img = visualizer.create_mosaic(
                    list(frames_dict.values()), 
                    list(frames_dict.keys()),
                    rows=1
                )
                
                cv2.imwrite(img_path, multi_modal_img)
            else:
                # 否则展示单一模态结果
                result_frame = visualizer.draw_bboxes(visible_frame, detections)
                cv2.imwrite(img_path, result_frame)
    
    # 评估结果
    metrics = evaluate_results(results, eval_config, output_dir)
    
    # 输出评估结果
    print("\n评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存结果
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)


def process_frame(frame, smoke_remover, human_detector, fusion_model, visualizer, disable_visualization=False):
    """
    处理单个帧
    
    参数:
        frame (ndarray): 输入帧
        smoke_remover: 烟雾去除算法
        human_detector: 人体检测算法
        fusion_model: 融合算法
        visualizer: 可视化工具
        disable_visualization: 是否禁用可视化
        
    返回:
        ndarray: 处理后的帧
    """
    # 1. 烟雾去除
    if smoke_remover is not None:
        processed_frame = smoke_remover.process(frame)
    else:
        processed_frame = frame.copy()
    
    # 2. 人体检测
    detections = human_detector.detect(processed_frame)
    
    # 3. 可视化结果
    if not disable_visualization:
        result_frame = visualizer.draw_bboxes(processed_frame, detections)
        result_frame = visualizer.draw_info(result_frame, smoke_removal=(smoke_remover is not None), fusion=(fusion_model is not None))
    else:
        result_frame = processed_frame
    
    return result_frame


def process_test_frame(ir_frame, thermal_frame, visible_frame, smoke_remover, human_detector, fusion_model, visualizer):
    """
    处理测试帧
    
    参数:
        ir_frame (ndarray): 红外帧
        thermal_frame (ndarray): 热成像帧
        visible_frame (ndarray): 可见光帧
        smoke_remover: 烟雾去除算法
        human_detector: 人体检测算法
        fusion_model: 融合算法
        visualizer: 可视化工具
        
    返回:
        ndarray: 处理后的帧
        list: 检测结果列表
    """
    processed_frame = None
    detections = []
    
    # 1. 多模态融合
    if fusion_model is not None and ir_frame is not None and thermal_frame is not None:
        # 融合红外和热成像
        fused_frame = fusion_model.fuse(ir_frame, thermal_frame)
        
        # 烟雾去除
        if smoke_remover is not None:
            processed_frame = smoke_remover.process(fused_frame)
        else:
            processed_frame = fused_frame
        
        # 人体检测
        detections = human_detector.detect(processed_frame)
        
        # 可视化结果
        result_frame = visualizer.draw_bboxes(processed_frame, detections)
        result_frame = visualizer.draw_info(result_frame, smoke_removal=(smoke_remover is not None), fusion=True)
        
    elif ir_frame is not None:
        # 只有红外图
        if smoke_remover is not None:
            processed_frame = smoke_remover.process(ir_frame)
        else:
            processed_frame = ir_frame.copy()
        
        # 人体检测
        detections = human_detector.detect(processed_frame)
        
        # 可视化结果
        result_frame = visualizer.draw_bboxes(processed_frame, detections)
        result_frame = visualizer.draw_info(result_frame, smoke_removal=(smoke_remover is not None), fusion=False)
        
    else:
        # 使用可见光图像
        if smoke_remover is not None:
            processed_frame = smoke_remover.process(visible_frame)
        else:
            processed_frame = visible_frame.copy()
        
        # 人体检测
        detections = human_detector.detect(processed_frame)
        
        # 可视化结果
        result_frame = visualizer.draw_bboxes(processed_frame, detections)
        result_frame = visualizer.draw_info(result_frame, smoke_removal=(smoke_remover is not None), fusion=False)
    
    return result_frame, detections


def evaluate_results(results, eval_config, output_dir):
    """
    评估检测结果
    
    参数:
        results (list): 结果列表
        eval_config (dict): 评估配置
        output_dir (str): 输出目录
    """
    print("评估结果...")
    
    # 提取评估参数
    iou_threshold = eval_config.get('iou_threshold', 0.5)
    confidence_thresholds = eval_config.get('confidence_thresholds', [0.5])
    
    # 统计指标
    total_gt = sum(len(r['gt_boxes']) for r in results if r['gt_boxes'] is not None)
    
    # 对每个置信度阈值计算结果
    metrics = {}
    for conf_thresh in confidence_thresholds:
        true_positives = 0
        false_positives = 0
        
        for result in results:
            detections = result['detections']
            gt_boxes = result['gt_boxes']
            
            if not gt_boxes or not detections:
                continue
            
            # 筛选超过置信度阈值的检测
            filtered_detections = []
            for det in detections:
                if 'confidence' in det and det['confidence'] >= conf_thresh:
                    filtered_detections.append(det)
            
            # 匹配检测结果和标注
            matched_gt = set()
            for det in filtered_detections:
                det_box = det['bbox']
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue  # 已经匹配过的标注不再考虑
                    
                    # 计算IoU
                    iou = calculate_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
        
        # 计算指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[conf_thresh] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'total_ground_truth': total_gt
        }
    
    # 输出评估结果
    print("\n评估结果:")
    for conf_thresh, metric in metrics.items():
        print(f"置信度阈值 {conf_thresh}:")
        print(f"  精确率: {metric['precision']:.4f}")
        print(f"  召回率: {metric['recall']:.4f}")
        print(f"  F1分数: {metric['f1']:.4f}")
        print(f"  TP: {metric['true_positives']}, FP: {metric['false_positives']}, GT: {metric['total_ground_truth']}")
        print("")
    
    # 保存结果
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"评估结果已保存到: {os.path.join(output_dir, 'metrics.json')}")


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    参数:
        box1 (list): 第一个边界框 [x1, y1, x2, y2]
        box2 (list): 第二个边界框 [x1, y1, x2, y2]
        
    返回:
        float: IoU值
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 检查是否有交集
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area


if __name__ == "__main__":
    main() 