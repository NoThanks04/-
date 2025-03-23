"""
多模态融合算法模块
"""

def get_fusion_algorithm(algorithm="weighted", config=None):
    """
    获取多模态融合算法实例
    
    参数:
        algorithm (str): 算法名称，支持 'weighted', 'attention'
        config (dict): 算法配置
        
    返回:
        object: 多模态融合算法实例
    """
    if config is None:
        config = {}
    
    if algorithm == "weighted":
        from .weighted_fusion import WeightedFusion
        return WeightedFusion(config.get('weighted', {}))
    
    elif algorithm == "attention":
        from .attention_fusion import AttentionFusion
        return AttentionFusion(config.get('attention', {}))
    
    else:
        print(f"警告: 未知的多模态融合算法 '{algorithm}'，使用默认的加权融合算法")
        from .weighted_fusion import WeightedFusion
        return WeightedFusion(config.get('weighted', {})) 