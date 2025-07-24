"""
验证工具函数
Validation Utility Functions
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from config.config import Config, DataConfig, ModelConfig, OptimizationConfig

def validate_input(X: np.ndarray, y: Optional[np.ndarray] = None) -> bool:
    """
    验证输入数据
    
    Args:
        X: 特征数据
        y: 目标数据（可选）
        
    Returns:
        验证通过返回True
        
    Raises:
        ValueError: 数据格式错误
    """
    # 检查X的格式
    if not isinstance(X, np.ndarray):
        raise ValueError("X必须是numpy数组")
    
    if len(X.shape) != 2:
        raise ValueError("X必须是二维数组")
    
    if X.shape[0] == 0:
        raise ValueError("X不能为空")
    
    if X.shape[1] == 0:
        raise ValueError("X必须至少有一个特征")
    
    # 检查是否有无穷值或NaN
    if np.any(np.isinf(X)) or np.any(np.isnan(X)):
        raise ValueError("X包含无穷值或NaN")
    
    # 如果提供了y，也进行检查
    if y is not None:
        if not isinstance(y, np.ndarray):
            raise ValueError("y必须是numpy数组")
        
        if len(y.shape) != 1:
            raise ValueError("y必须是一维数组")
        
        if len(y) != X.shape[0]:
            raise ValueError("X和y的样本数必须相同")
        
        if np.any(np.isinf(y)) or np.any(np.isnan(y)):
            raise ValueError("y包含无穷值或NaN")
    
    return True

def validate_config(config: Config) -> bool:
    """
    验证配置对象
    
    Args:
        config: 配置对象
        
    Returns:
        验证通过返回True
        
    Raises:
        ValueError: 配置错误
    """
    # 验证数据配置
    _validate_data_config(config.data)
    
    # 验证模型配置
    _validate_model_config(config.model)
    
    # 验证优化配置
    _validate_optimization_config(config.optimization)
    
    return True

def _validate_data_config(data_config: DataConfig) -> bool:
    """验证数据配置"""
    if not data_config.data_path:
        raise ValueError("数据路径不能为空")
    
    if not data_config.features:
        raise ValueError("特征列表不能为空")
    
    if not data_config.target:
        raise ValueError("目标列不能为空")
    
    if not 0 < data_config.test_size < 1:
        raise ValueError("测试集比例必须在0和1之间")
    
    if data_config.random_state < 0:
        raise ValueError("随机种子必须为非负整数")
    
    if data_config.normalize_method not in ['standard', 'minmax', 'custom']:
        raise ValueError("标准化方法必须是 'standard', 'minmax' 或 'custom'")
    
    return True

def _validate_model_config(model_config: ModelConfig) -> bool:
    """验证模型配置"""
    if model_config.model_type not in ['SVR', 'LSSVR', 'GPR']:
        raise ValueError("模型类型必须是 'SVR', 'LSSVR' 或 'GPR'")
    
    if model_config.kernel not in ['rbf', 'linear', 'poly']:
        raise ValueError("核函数类型必须是 'rbf', 'linear' 或 'poly'")
    
    # 验证SVR参数
    if model_config.C <= 0:
        raise ValueError("C参数必须为正数")
    
    if model_config.epsilon < 0:
        raise ValueError("epsilon参数必须为非负数")
    
    if isinstance(model_config.gamma, (int, float)) and model_config.gamma <= 0:
        raise ValueError("gamma参数必须为正数")
    
    # 验证LSSVR参数
    if model_config.sigma2 <= 0:
        raise ValueError("sigma2参数必须为正数")
    
    # 验证GPR参数
    if model_config.alpha <= 0:
        raise ValueError("alpha参数必须为正数")
    
    if model_config.n_restarts_optimizer < 0:
        raise ValueError("n_restarts_optimizer必须为非负整数")
    
    return True

def _validate_optimization_config(opt_config: OptimizationConfig) -> bool:
    """验证优化配置"""
    if opt_config.optimizer_type not in ['GA', 'PSO']:
        raise ValueError("优化器类型必须是 'GA' 或 'PSO'")
    
    # 验证GA参数
    if opt_config.population_size <= 0:
        raise ValueError("种群大小必须为正整数")
    
    if opt_config.generations <= 0:
        raise ValueError("代数必须为正整数")
    
    if not 0 <= opt_config.crossover_rate <= 1:
        raise ValueError("交叉率必须在0和1之间")
    
    if not 0 <= opt_config.mutation_rate <= 1:
        raise ValueError("变异率必须在0和1之间")
    
    if opt_config.tournament_size <= 0:
        raise ValueError("锦标赛大小必须为正整数")
    
    # 验证PSO参数
    if opt_config.n_particles <= 0:
        raise ValueError("粒子数量必须为正整数")
    
    if opt_config.n_iterations <= 0:
        raise ValueError("迭代次数必须为正整数")
    
    if opt_config.w < 0:
        raise ValueError("惯性权重必须为非负数")
    
    if opt_config.c1 < 0 or opt_config.c2 < 0:
        raise ValueError("学习因子必须为非负数")
    
    if not 0 <= opt_config.w_min <= opt_config.w_max <= 1:
        raise ValueError("惯性权重范围必须满足 0 <= w_min <= w_max <= 1")
    
    # 验证参数边界
    if opt_config.param_bounds:
        for param_name, bounds in opt_config.param_bounds.items():
            if len(bounds) != 2:
                raise ValueError(f"参数 {param_name} 的边界必须是长度为2的元组")
            
            if bounds[0] >= bounds[1]:
                raise ValueError(f"参数 {param_name} 的下界必须小于上界")
    
    return True

def validate_data_file(filepath: str, required_columns: List[str]) -> bool:
    """
    验证数据文件
    
    Args:
        filepath: 文件路径
        required_columns: 必需的列名列表
        
    Returns:
        验证通过返回True
        
    Raises:
        ValueError: 文件格式错误
    """
    import os
    
    if not os.path.exists(filepath):
        raise ValueError(f"数据文件不存在: {filepath}")
    
    try:
        # 尝试读取文件
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")
        
        # 检查必需的列是否存在
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")
        
        # 检查数据是否为空
        if df.empty:
            raise ValueError("数据文件为空")
        
        # 检查是否有过多的缺失值
        for col in required_columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0.5:
                raise ValueError(f"列 {col} 缺失值过多 ({missing_ratio:.1%})")
        
        return True
        
    except Exception as e:
        raise ValueError(f"读取数据文件时出错: {str(e)}")

def validate_prediction_input(ppfd: float, co2: float, t: float, rb: float) -> bool:
    """
    验证预测输入参数
    
    Args:
        ppfd: 光合光子通量密度
        co2: CO2浓度
        t: 温度
        rb: 红蓝光比
        
    Returns:
        验证通过返回True
        
    Raises:
        ValueError: 输入参数错误
    """
    # 检查数据类型
    if not all(isinstance(x, (int, float)) for x in [ppfd, co2, t, rb]):
        raise ValueError("所有输入参数必须是数值类型")
    
    # 检查是否为有限值
    if not all(np.isfinite(x) for x in [ppfd, co2, t, rb]):
        raise ValueError("输入参数不能是无穷值或NaN")
    
    # 检查取值范围（基于实际物理意义）
    if ppfd < 0 or ppfd > 3000:
        raise ValueError(f"PPFD取值范围应在0-3000之间，当前值: {ppfd}")
    
    if co2 < 0 or co2 > 2000:
        raise ValueError(f"CO2浓度取值范围应在0-2000之间，当前值: {co2}")
    
    if t < -10 or t > 60:
        raise ValueError(f"温度取值范围应在-10-60℃之间，当前值: {t}")
    
    if rb < 0 or rb > 10:
        raise ValueError(f"红蓝光比取值范围应在0-10之间，当前值: {rb}")
    
    return True

def check_data_distribution(X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    检查数据分布
    
    Args:
        X: 特征数据
        feature_names: 特征名称列表（可选）
        
    Returns:
        数据分布信息字典
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    distribution_info = {}
    
    for i, feature_name in enumerate(feature_names):
        feature_data = X[:, i]
        
        distribution_info[feature_name] = {
            'mean': np.mean(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'median': np.median(feature_data),
            'q25': np.percentile(feature_data, 25),
            'q75': np.percentile(feature_data, 75),
            'skewness': _calculate_skewness(feature_data),
            'kurtosis': _calculate_kurtosis(feature_data)
        }
    
    return distribution_info

def _calculate_skewness(data: np.ndarray) -> float:
    """计算偏度"""
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 3)

def _calculate_kurtosis(data: np.ndarray) -> float:
    """计算峰度"""
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 4) - 3 