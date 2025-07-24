#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML数据管理器
负责协调数据处理和模型训练的数据流
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path

from integrated_workflow.ml_training.data_processing import DataProcessor

logger = logging.getLogger(__name__)


class MLDataManager:
    """ML数据管理器，负责协调数据处理和模型训练的数据流"""
    
    def __init__(self, config: Dict):
        """
        初始化ML数据管理器
        
        Args:
            config (Dict): 配置字典，包含数据处理和模型训练相关参数
        """
        self.config = config
        self.data_processor = DataProcessor(config.get('data_processing', {}))
        self.output_dir = config.get('output_dir', 'results')
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ML数据管理器初始化完成，输出目录: {self.output_dir}")
    
    def prepare_training_data(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        准备模型训练数据
        
        Args:
            data_path (str, optional): 数据文件路径，如果为None则使用配置中的路径
        
        Returns:
            Dict[str, Any]: 包含训练数据和相关信息的字典
        """
        logger.info("准备模型训练数据...")
        
        # 使用数据处理器准备数据
        X_train_scaled, X_test_scaled, y_train, y_test = self.data_processor.prepare_data(data_path)
        
        # 保存预处理器
        preprocessor_dir = os.path.join(self.output_dir, 'preprocessors')
        os.makedirs(preprocessor_dir, exist_ok=True)
        preprocessor_paths = self.data_processor.save_preprocessor(preprocessor_dir)
        
        # 构建返回结果
        result = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor_paths': preprocessor_paths,
            'feature_names': self.get_feature_names()
        }
        
        logger.info(f"训练数据准备完成，训练集形状: {X_train_scaled.shape}, 测试集形状: {X_test_scaled.shape}")
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            List[str]: 特征名称列表
        """
        # 如果配置中指定了特征列，则使用它们
        if hasattr(self.data_processor, 'feature_columns') and self.data_processor.feature_columns:
            return self.data_processor.feature_columns
        
        # 否则，尝试从数据中获取
        try:
            df = self.data_processor.load_data()
            # 排除目标列
            feature_names = [col for col in df.columns if col != self.data_processor.target_column]
            return feature_names
        except Exception as e:
            logger.warning(f"无法获取特征名称: {str(e)}")
            return []
    
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        加载并预处理数据，但不进行分割
        
        Args:
            data_path (str): 数据文件路径
        
        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        logger.info(f"加载并预处理数据: {data_path}")
        
        # 加载数据
        df = self.data_processor.load_data(data_path)
        
        # 验证数据
        self.data_processor.validate_data(df)
        
        # 预处理数据
        df = self.data_processor.preprocess_data(df)
        
        logger.info(f"数据加载和预处理完成，形状: {df.shape}")
        return df
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征工程
        
        Args:
            df (pd.DataFrame): 原始数据框
        
        Returns:
            pd.DataFrame: 添加新特征后的数据框
        """
        logger.info("应用特征工程...")
        
        # 使用数据处理器的特征工程功能
        df_with_features = self.data_processor.create_features(df)
        
        logger.info(f"特征工程完成，原始形状: {df.shape}, 新形状: {df_with_features.shape}")
        return df_with_features
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据，生成统计信息
        
        Args:
            df (pd.DataFrame): 数据框
        
        Returns:
            Dict[str, Any]: 数据分析结果
        """
        logger.info("分析数据...")
        
        # 基本统计信息
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'summary': {}
        }
        
        # 数值列的统计信息
        for col in stats['numeric_columns']:
            stats['summary'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        # 目标列的分布
        if self.data_processor.target_column in df.columns:
            target_col = self.data_processor.target_column
            stats['target'] = {
                'name': target_col,
                'mean': df[target_col].mean(),
                'std': df[target_col].std(),
                'min': df[target_col].min(),
                'max': df[target_col].max(),
                'median': df[target_col].median()
            }
        
        # 相关性分析
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            
            # 找出与目标高度相关的特征
            if self.data_processor.target_column in corr_matrix.columns:
                target_corr = corr_matrix[self.data_processor.target_column].drop(self.data_processor.target_column)
                stats['target_correlations'] = target_corr.to_dict()
                
                # 高相关性特征
                high_corr_features = target_corr[abs(target_corr) > 0.7]
                if not high_corr_features.empty:
                    stats['high_correlation_features'] = high_corr_features.to_dict()
        
        logger.info("数据分析完成")
        return stats
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> str:
        """
        保存处理后的数据
        
        Args:
            df (pd.DataFrame): 处理后的数据框
            output_path (str): 输出文件路径
        
        Returns:
            str: 保存的文件路径
        """
        logger.info(f"保存处理后的数据: {output_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 根据文件扩展名确定保存格式
        file_ext = os.path.splitext(output_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df.to_csv(output_path, index=False)
            elif file_ext in ['.xls', '.xlsx']:
                df.to_excel(output_path, index=False)
            elif file_ext == '.json':
                df.to_json(output_path, orient='records')
            elif file_ext == '.parquet':
                df.to_parquet(output_path, index=False)
            else:
                # 默认使用CSV格式
                csv_path = os.path.splitext(output_path)[0] + '.csv'
                df.to_csv(csv_path, index=False)
                logger.warning(f"不支持的文件格式: {file_ext}，已保存为CSV: {csv_path}")
                output_path = csv_path
            
            logger.info(f"数据已保存: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}", exc_info=True)
            raise
    
    def load_preprocessor(self, preprocessor_paths: Dict[str, str]) -> None:
        """
        加载预处理器
        
        Args:
            preprocessor_paths (Dict[str, str]): 预处理器文件路径字典
        """
        logger.info("加载预处理器...")
        
        # 使用数据处理器的加载功能
        self.data_processor.load_preprocessor(preprocessor_paths)
        
        logger.info("预处理器加载完成")
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        使用已加载的预处理器转换新数据
        
        Args:
            df (pd.DataFrame): 新数据框
        
        Returns:
            np.ndarray: 转换后的数据
        """
        logger.info("转换新数据...")
        
        if not hasattr(self.data_processor, 'scaler') or self.data_processor.scaler is None:
            raise ValueError("未加载缩放器，请先调用load_preprocessor")
        
        # 预处理数据
        df = self.data_processor.preprocess_data(df)
        
        # 如果有特征选择器，应用它
        if hasattr(self.data_processor, 'feature_selector') and self.data_processor.feature_selector is not None:
            # 获取选择的特征
            selected_features = df.columns[self.data_processor.feature_selector.get_support()]
            df = df[selected_features]
        
        # 应用缩放器
        X_scaled = self.data_processor.scaler.transform(df)
        
        logger.info(f"数据转换完成，形状: {X_scaled.shape}")
        return X_scaled