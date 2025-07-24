"""
文件输入输出工具函数
File I/O Utility Functions
"""

import os
import pickle
import json
import yaml
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union
from datetime import datetime

def save_model(model: Any, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    保存模型到文件
    
    Args:
        model: 要保存的模型
        filepath: 文件路径
        metadata: 元数据（可选）
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 准备保存数据
    save_data = {
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # 保存模型
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"模型已保存到: {filepath}")

def load_model(filepath: str) -> Dict[str, Any]:
    """
    从文件加载模型
    
    Args:
        filepath: 文件路径
        
    Returns:
        包含模型和元数据的字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"模型已从 {filepath} 加载")
    return save_data

def save_results(results: Dict[str, Any], filepath: str, format: str = 'json') -> None:
    """
    保存结果到文件
    
    Args:
        results: 结果字典
        filepath: 文件路径
        format: 文件格式 ('json', 'yaml', 'pickle')
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 添加时间戳
    results_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    if format.lower() == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_with_timestamp, f, indent=2, ensure_ascii=False, default=str)
    elif format.lower() == 'yaml':
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(results_with_timestamp, f, default_flow_style=False, allow_unicode=True)
    elif format.lower() == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results_with_timestamp, f)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    print(f"结果已保存到: {filepath}")

def load_results(filepath: str) -> Dict[str, Any]:
    """
    从文件加载结果
    
    Args:
        filepath: 文件路径
        
    Returns:
        结果字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"结果文件不存在: {filepath}")
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_ext in ['.yaml', '.yml']:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    elif file_ext == '.pkl':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    print(f"结果已从 {filepath} 加载")
    return data

def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    保存DataFrame到文件
    
    Args:
        df: DataFrame
        filepath: 文件路径
        **kwargs: 额外参数
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        df.to_excel(filepath, index=False, **kwargs)
    elif file_ext == '.parquet':
        df.to_parquet(filepath, index=False, **kwargs)
    elif file_ext == '.pickle':
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    print(f"DataFrame已保存到: {filepath}")

def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    """
    从文件加载DataFrame
    
    Args:
        filepath: 文件路径
        **kwargs: 额外参数
        
    Returns:
        DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(filepath, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, **kwargs)
    elif file_ext == '.parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif file_ext == '.pickle':
        df = pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    print(f"DataFrame已从 {filepath} 加载")
    return df

def create_directory(path: str) -> None:
    """
    创建目录
    
    Args:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"目录已创建: {path}")
    else:
        print(f"目录已存在: {path}")

def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        filepath: 文件路径
        
    Returns:
        文件信息字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    stat = os.stat(filepath)
    
    return {
        'path': filepath,
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'accessed': datetime.fromtimestamp(stat.st_atime).isoformat()
    }

def list_files(directory: str, extension: Optional[str] = None) -> list:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        extension: 文件扩展名过滤（可选）
        
    Returns:
        文件列表
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    files = []
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath):
            if extension is None or file.endswith(extension):
                files.append(filepath)
    
    return sorted(files)

def backup_file(filepath: str, backup_dir: Optional[str] = None) -> str:
    """
    备份文件
    
    Args:
        filepath: 原文件路径
        backup_dir: 备份目录（可选）
        
    Returns:
        备份文件路径
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 确定备份目录
    if backup_dir is None:
        backup_dir = os.path.join(os.path.dirname(filepath), 'backup')
    
    create_directory(backup_dir)
    
    # 创建备份文件名
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{name}_{timestamp}{ext}"
    backup_filepath = os.path.join(backup_dir, backup_filename)
    
    # 复制文件
    import shutil
    shutil.copy2(filepath, backup_filepath)
    
    print(f"文件已备份到: {backup_filepath}")
    return backup_filepath

def clean_directory(directory: str, older_than_days: int = 30) -> None:
    """
    清理目录中的旧文件
    
    Args:
        directory: 目录路径
        older_than_days: 删除多少天前的文件
    """
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return
    
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=older_than_days)
    
    deleted_count = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_modified < cutoff_date:
                os.remove(filepath)
                deleted_count += 1
                print(f"已删除: {filepath}")
    
    print(f"共删除 {deleted_count} 个文件")

def safe_filename(filename: str) -> str:
    """
    生成安全的文件名
    
    Args:
        filename: 原文件名
        
    Returns:
        安全的文件名
    """
    # 移除或替换不安全的字符
    import re
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = safe_name.strip('. ')
    
    # 限制长度
    if len(safe_name) > 200:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:200-len(ext)] + ext
    
    return safe_name 