"""
人体检测算法模块
"""

def get_human_detection_algorithm(algorithm="yolov5", config=None, device="cpu"):
    """
    获取人体检测算法实例
    
    参数:
        algorithm (str): 算法名称，支持 'yolov5', 'faster_rcnn', 'ssd'
        config (dict): 算法配置
        device (str): 运行设备，支持 'cpu', 'cuda', 'rknn'
        
    返回:
        object: 人体检测算法实例
    """
    if config is None:
        config = {}
    
    if algorithm == "yolov5":
        from .yolov5 import YOLOv5Detector
        return YOLOv5Detector(config.get('yolov5', {}), device=device)
    
    elif algorithm == "faster_rcnn":
        from .faster_rcnn import FasterRCNNDetector
        return FasterRCNNDetector(config.get('faster_rcnn', {}), device=device)
    
    elif algorithm == "ssd":
        from .ssd import SSDDetector
        return SSDDetector(config.get('ssd', {}), device=device)
    
    else:
        print(f"警告: 未知的人体检测算法 '{algorithm}'，使用默认的YOLOv5算法")
        from .yolov5 import YOLOv5Detector
        return YOLOv5Detector(config.get('yolov5', {}), device=device) 