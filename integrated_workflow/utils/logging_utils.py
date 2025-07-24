#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块
提供日志配置和管理功能
"""

import os
import logging
import logging.handlers
from datetime import datetime


def setup_logging(log_file=None, log_level='INFO', log_format=None):
    """
    设置日志系统
    
    Args:
        log_file (str): 日志文件路径，如果为None则只输出到控制台
        log_level (str): 日志级别，可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_format (str): 日志格式，如果为None则使用默认格式
    """
    # 转换日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 创建文件处理器，使用RotatingFileHandler以限制文件大小
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('paramiko').setLevel(logging.WARNING)
    
    logging.info(f"日志系统初始化完成，级别: {log_level}, 文件: {log_file}")
    
    return root_logger


def get_logger(name):
    """
    获取指定名称的日志记录器
    
    Args:
        name (str): 日志记录器名称
    
    Returns:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)


def log_execution_time(logger, start_time, description="执行"):
    """
    记录执行时间
    
    Args:
        logger (logging.Logger): 日志记录器
        start_time (float): 开始时间（time.time()的返回值）
        description (str): 执行描述
    """
    import time
    execution_time = time.time() - start_time
    logger.info(f"{description}完成，耗时: {execution_time:.2f} 秒")
    return execution_time


def setup_stage_logger(stage_name, log_dir, log_level='INFO'):
    """
    为特定阶段设置日志记录器
    
    Args:
        stage_name (str): 阶段名称
        log_dir (str): 日志目录
        log_level (str): 日志级别
    
    Returns:
        logging.Logger: 阶段日志记录器
    """
    # 创建阶段日志文件路径
    log_file = os.path.join(log_dir, f"{stage_name}.log")
    
    # 获取阶段日志记录器
    logger = logging.getLogger(f"workflow.{stage_name}")
    
    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_system_info(logger):
    """
    记录系统信息
    
    Args:
        logger (logging.Logger): 日志记录器
    """
    import platform
    import sys
    
    logger.info("系统信息:")
    logger.info(f"  操作系统: {platform.system()} {platform.release()}")
    logger.info(f"  Python版本: {platform.python_version()}")
    logger.info(f"  处理器架构: {platform.machine()}")
    
    try:
        import numpy
        logger.info(f"  NumPy版本: {numpy.__version__}")
    except ImportError:
        logger.info("  NumPy: 未安装")
    
    try:
        import pandas
        logger.info(f"  Pandas版本: {pandas.__version__}")
    except ImportError:
        logger.info("  Pandas: 未安装")
    
    try:
        import scipy
        logger.info(f"  SciPy版本: {scipy.__version__}")
    except ImportError:
        logger.info("  SciPy: 未安装")
    
    try:
        import matplotlib
        logger.info(f"  Matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        logger.info("  Matplotlib: 未安装")