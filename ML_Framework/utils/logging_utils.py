"""
日志工具函数
Logging Utility Functions
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logger(name: str = 'ML_Framework', 
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        format_string: 日志格式字符串（可选）
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，先清除
    if logger.handlers:
        logger.handlers.clear()
    
    # 设置默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_performance(logger: logging.Logger, 
                   model_name: str, 
                   metrics: Dict[str, float],
                   stage: str = "evaluation") -> None:
    """
    记录模型性能指标
    
    Args:
        logger: 日志记录器
        model_name: 模型名称
        metrics: 性能指标字典
        stage: 阶段名称
    """
    logger.info(f"=== {stage.upper()} - {model_name} ===")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.6f}")
    logger.info("=" * 50)

def log_optimization_progress(logger: logging.Logger,
                            optimizer_name: str,
                            iteration: int,
                            best_score: float,
                            current_params: Dict[str, Any]) -> None:
    """
    记录优化进度
    
    Args:
        logger: 日志记录器
        optimizer_name: 优化器名称
        iteration: 迭代次数
        best_score: 最佳分数
        current_params: 当前参数
    """
    logger.info(f"{optimizer_name} - 迭代 {iteration}")
    logger.info(f"  最佳分数: {best_score:.6f}")
    logger.info(f"  当前参数: {current_params}")

def log_data_info(logger: logging.Logger, 
                 data_info: Dict[str, Any]) -> None:
    """
    记录数据信息
    
    Args:
        logger: 日志记录器
        data_info: 数据信息字典
    """
    logger.info("=== 数据信息 ===")
    logger.info(f"样本数量: {data_info.get('total_samples', 'N/A')}")
    logger.info(f"特征列: {data_info.get('features', 'N/A')}")
    logger.info(f"目标列: {data_info.get('target', 'N/A')}")
    
    if 'feature_ranges' in data_info:
        logger.info("特征范围:")
        for feature, range_vals in data_info['feature_ranges'].items():
            logger.info(f"  {feature}: [{range_vals[0]:.3f}, {range_vals[1]:.3f}]")
    
    if 'target_range' in data_info and data_info['target_range']:
        logger.info(f"目标范围: [{data_info['target_range'][0]:.3f}, {data_info['target_range'][1]:.3f}]")

def create_experiment_logger(experiment_name: str, 
                           log_dir: str = "logs") -> logging.Logger:
    """
    创建实验专用的日志记录器
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        
    Returns:
        实验日志记录器
    """
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 设置日志记录器
    logger = setup_logger(
        name=f"Experiment_{experiment_name}",
        log_file=log_filepath,
        format_string='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"实验开始: {experiment_name}")
    logger.info(f"日志文件: {log_filepath}")
    
    return logger 