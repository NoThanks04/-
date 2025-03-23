"""
烟雾去除算法模块
"""

def get_smoke_removal_algorithm(algorithm="dehaze", config=None):
    """
    获取烟雾去除算法实例
    
    参数:
        algorithm (str): 算法名称，支持 'dehaze', 'clahe', 'retinex'
        config (dict): 算法配置
        
    返回:
        object: 烟雾去除算法实例
    """
    if config is None:
        config = {}
    
    if algorithm == "dehaze":
        from .dehaze import Dehaze
        return Dehaze(config.get('dehaze', {}))
    
    elif algorithm == "clahe":
        from .clahe import CLAHE
        return CLAHE(config.get('clahe', {}))
    
    elif algorithm == "retinex":
        from .retinex import Retinex
        return Retinex(config.get('retinex', {}))
    
    else:
        print(f"警告: 未知的烟雾去除算法 '{algorithm}'，使用默认的去雾算法")
        from .dehaze import Dehaze
        return Dehaze(config.get('dehaze', {})) 