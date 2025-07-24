#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置工具模块
提供配置文件验证和处理功能
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_config(config):
    """
    验证配置文件的有效性
    
    Args:
        config (dict): 配置字典
    
    Raises:
        ValueError: 配置无效时抛出异常
    """
    logger.info("开始验证配置文件...")
    
    # 验证必需的顶级配置项
    required_sections = ['general', 'ml_training', 'optimization', 'optimal_point', 'mpc_simulation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"缺少必需的配置节: {section}")
    
    # 验证general配置
    general_config = config['general']
    required_general_keys = ['output_dir', 'log_level', 'enable_checkpoints']
    for key in required_general_keys:
        if key not in general_config:
            raise ValueError(f"general配置中缺少{key}")
    
    # 验证日志级别
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if general_config['log_level'].upper() not in valid_log_levels:
        raise ValueError(f"无效的日志级别: {general_config['log_level']}，有效值为: {', '.join(valid_log_levels)}")
    
    # 验证ml_training配置
    ml_config = config['ml_training']
    required_ml_keys = ['framework_path', 'script_path']
    for key in required_ml_keys:
        if key not in ml_config:
            raise ValueError(f"ml_training配置中缺少{key}")
    
    # 检查ML框架路径是否存在
    ml_framework_path = ml_config['framework_path']
    if not os.path.exists(ml_framework_path):
        logger.warning(f"ML框架路径不存在: {ml_framework_path}，请确保在运行时该路径可用")
    
    # 检查ML脚本是否存在
    ml_script_path = os.path.join(ml_framework_path, ml_config['script_path'])
    if not os.path.exists(ml_script_path):
        logger.warning(f"ML训练脚本不存在: {ml_script_path}，请确保在运行时该脚本可用")
    
    # 验证optimization配置
    opt_config = config['optimization']
    required_opt_keys = ['framework_path', 'script_path']
    for key in required_opt_keys:
        if key not in opt_config:
            raise ValueError(f"optimization配置中缺少{key}")
    
    # 检查优化框架路径是否存在
    opt_framework_path = opt_config['framework_path']
    if not os.path.exists(opt_framework_path):
        logger.warning(f"优化框架路径不存在: {opt_framework_path}，请确保在运行时该路径可用")
    
    # 检查优化脚本是否存在
    opt_script_path = os.path.join(opt_framework_path, opt_config['script_path'])
    if not os.path.exists(opt_script_path):
        logger.warning(f"优化脚本不存在: {opt_script_path}，请确保在运行时该脚本可用")
    
    # 验证optimal_point配置
    opt_point_config = config['optimal_point']
    required_opt_point_keys = ['framework_path', 'script_path']
    for key in required_opt_point_keys:
        if key not in opt_point_config:
            raise ValueError(f"optimal_point配置中缺少{key}")
    
    # 检查最优点框架路径是否存在
    opt_point_framework_path = opt_point_config['framework_path']
    if not os.path.exists(opt_point_framework_path):
        logger.warning(f"最优点框架路径不存在: {opt_point_framework_path}，请确保在运行时该路径可用")
    
    # 检查最优点脚本是否存在
    opt_point_script_path = os.path.join(opt_point_framework_path, opt_point_config['script_path'])
    if not os.path.exists(opt_point_script_path):
        logger.warning(f"最优点脚本不存在: {opt_point_script_path}，请确保在运行时该脚本可用")
    
    # 验证mpc_simulation配置
    mpc_config = config['mpc_simulation']
    required_mpc_keys = ['framework_path', 'script_path']
    for key in required_mpc_keys:
        if key not in mpc_config:
            raise ValueError(f"mpc_simulation配置中缺少{key}")
    
    # 检查MPC框架路径是否存在
    mpc_framework_path = mpc_config['framework_path']
    if not os.path.exists(mpc_framework_path):
        logger.warning(f"MPC框架路径不存在: {mpc_framework_path}，请确保在运行时该路径可用")
    
    # 检查MPC脚本是否存在
    mpc_script_path = os.path.join(mpc_framework_path, mpc_config['script_path'])
    if not os.path.exists(mpc_script_path):
        logger.warning(f"MPC脚本不存在: {mpc_script_path}，请确保在运行时该脚本可用")
    
    # 验证变量和约束
    if 'variables' in opt_config:
        for var_name, var_config in opt_config['variables'].items():
            if 'lower_bound' not in var_config or 'upper_bound' not in var_config:
                raise ValueError(f"变量 {var_name} 缺少上下界定义")
            if var_config['lower_bound'] >= var_config['upper_bound']:
                raise ValueError(f"变量 {var_name} 的下界大于或等于上界")
    
    # 验证MPC约束
    if 'constraints' in mpc_config:
        for var_name, constraint in mpc_config['constraints'].items():
            if 'min' not in constraint or 'max' not in constraint:
                raise ValueError(f"MPC约束 {var_name} 缺少最小值或最大值")
            if constraint['min'] >= constraint['max']:
                raise ValueError(f"MPC约束 {var_name} 的最小值大于或等于最大值")
    
    logger.info("配置文件验证通过")


def load_config_with_defaults(config_path):
    """
    加载配置文件并应用默认值
    
    Args:
        config_path (str): 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    import yaml
    
    # 默认配置
    default_config = {
        'general': {
            'output_dir': 'integrated_workflow/results',
            'log_level': 'INFO',
            'enable_checkpoints': True,
            'checkpoint_frequency': 1
        },
        'ml_training': {
            'framework_path': 'ML_Framework',
            'script_path': 'run_experiment.py',
            'config_file': 'config/config.yaml',
            'model_type': 'LSSVR',
            'test_size': 0.2,
            'random_state': 42
        },
        'optimization': {
            'framework_path': 'pymoo',
            'script_path': 'find_optimal_conditions_multi_model.py',
            'algorithm': 'NSGA2',
            'population_size': 100,
            'generations': 200
        },
        'optimal_point': {
            'framework_path': 'Optimal',
            'script_path': 'fit.py',
            'fitting_method': 'polynomial',
            'polynomial_degree': 3
        },
        'mpc_simulation': {
            'framework_path': 'mpc-farming-master',
            'script_path': 'mpc.py',
            'simulation_duration': 24,
            'time_step': 0.1
        }
    }
    
    # 加载用户配置
    with open(config_path, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f)
    
    # 合并配置
    merged_config = _merge_configs(default_config, user_config)
    
    return merged_config


def _merge_configs(default_config, user_config):
    """
    递归合并配置字典
    
    Args:
        default_config (dict): 默认配置
        user_config (dict): 用户配置
    
    Returns:
        dict: 合并后的配置
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def create_output_directories(config, timestamp=None):
    """
    创建输出目录
    
    Args:
        config (dict): 配置字典
        timestamp (str, optional): 时间戳，用于创建唯一的输出目录
    
    Returns:
        str: 创建的输出目录路径
    """
    base_output_dir = config['general']['output_dir']
    
    # 如果提供了时间戳，创建带时间戳的子目录
    if timestamp:
        output_dir = os.path.join(base_output_dir, f'workflow_{timestamp}')
    else:
        output_dir = base_output_dir
    
    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    subdirs = ['ml_training', 'optimization', 'optimal_point', 'mpc_simulation', 'checkpoints', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    logger.info(f"输出目录创建完成: {output_dir}")
    return output_dir


def save_config_copy(config, output_path):
    """
    保存配置文件的副本
    
    Args:
        config (dict): 配置字典
        output_path (str): 输出路径
    """
    import yaml
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置副本已保存: {output_path}")
    except Exception as e:
        logger.error(f"保存配置副本失败: {str(e)}")


def get_default_config():
    """
    获取默认配置
    
    Returns:
        dict: 默认配置字典
    """
    return {
        'general': {
            'output_dir': 'integrated_workflow/results',
            'log_level': 'INFO',
            'enable_checkpoints': True,
            'checkpoint_frequency': 1
        },
        'ml_training': {
            'framework_path': 'ML_Framework',
            'script_path': 'run_experiment.py',
            'config_file': 'config/config.yaml',
            'model_type': 'LSSVR',
            'test_size': 0.2,
            'random_state': 42
        },
        'optimization': {
            'framework_path': 'pymoo',
            'script_path': 'find_optimal_conditions_multi_model.py',
            'algorithm': 'NSGA2',
            'population_size': 100,
            'generations': 200,
            'variables': {
                'ppfd': {
                    'lower_bound': 100,
                    'upper_bound': 1000
                },
                'r_b_ratio': {
                    'lower_bound': 0.5,
                    'upper_bound': 4.0
                },
                'temperature': {
                    'lower_bound': 15,
                    'upper_bound': 35
                }
            },
            'objectives': [
                'maximize_photosynthesis',
                'minimize_cled_cost'
            ]
        },
        'optimal_point': {
            'framework_path': 'Optimal',
            'script_path': 'fit.py',
            'fitting_method': 'polynomial',
            'polynomial_degree': 3,
            'visualization': {
                'enable': True,
                'format': 'png',
                'dpi': 300
            }
        },
        'mpc_simulation': {
            'framework_path': 'mpc-farming-master',
            'script_path': 'mpc.py',
            'simulation_duration': 24,
            'time_step': 0.1,
            'control_horizon': 6,
            'prediction_horizon': 12,
            'constraints': {
                'ppfd': {
                    'min': 0,
                    'max': 1000
                },
                'temperature': {
                    'min': 15,
                    'max': 35
                }
            }
        }
    }