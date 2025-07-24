#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练组件
负责创建、训练和管理机器学习模型
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union, Type
import pickle
from datetime import datetime
from pathlib import Path

# 导入ML_Framework中的模型类
from ML_Framework.models import BaseModel, LSSVRModel, GPRModel, SVRModel
from integrated_workflow.ml_training.hyperparameter_optimizer import HyperparameterOptimizer
from integrated_workflow.ml_training.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练类，负责创建、训练和管理机器学习模型"""
    
    # 模型类型映射
    MODEL_TYPES = {
        'LSSVR': LSSVRModel,
        'GPR': GPRModel,
        'SVR': SVRModel
    }
    
    def __init__(self, config: Dict):
        """
        初始化模型训练器
        
        Args:
            config (Dict): 配置字典，包含模型训练相关参数
        """
        self.config = config
        self.model_type = config.get('model_type', 'LSSVR').upper()
        self.model_params = config.get('model_params', {})
        self.output_dir = config.get('output_dir', 'results')
        self.model = None
        self.trained = False
        self.evaluator = ModelEvaluator(config.get('evaluation', {}))
        self.optimizer = None
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 验证模型类型
        if self.model_type not in self.MODEL_TYPES:
            raise ValueError(f"不支持的模型类型: {self.model_type}，支持的类型: {list(self.MODEL_TYPES.keys())}")
        
        logger.info(f"模型训练器初始化完成，模型类型: {self.model_type}")
    
    def create_model(self) -> BaseModel:
        """
        创建模型实例
        
        Returns:
            BaseModel: 创建的模型实例
        """
        logger.info(f"创建{self.model_type}模型...")
        
        # 获取模型类
        model_class = self.MODEL_TYPES[self.model_type]
        
        # 创建模型实例
        try:
            model = model_class(**self.model_params)
            logger.info(f"模型创建成功: {model}")
            return model
        except Exception as e:
            logger.error(f"模型创建失败: {str(e)}", exc_info=True)
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> BaseModel:
        """
        训练模型
        
        Args:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
        
        Returns:
            BaseModel: 训练后的模型
        """
        logger.info("开始训练模型...")
        
        # 创建模型（如果尚未创建）
        if self.model is None:
            self.model = self.create_model()
        
        # 训练模型
        try:
            self.model.fit(X_train, y_train)
            self.trained = True
            logger.info("模型训练完成")
            return self.model
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}", exc_info=True)
            raise
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: Optional[np.ndarray] = None, 
                                y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        优化模型超参数
        
        Args:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证目标
        
        Returns:
            Dict[str, Any]: 优化结果，包含最佳参数和性能指标
        """
        logger.info("开始超参数优化...")
        
        # 创建优化器
        optimizer_config = self.config.get('optimization', {})
        self.optimizer = HyperparameterOptimizer(optimizer_config)
        
        # 获取模型类
        model_class = self.MODEL_TYPES[self.model_type]
        
        # 执行优化
        optimization_results = self.optimizer.optimize(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            base_params=self.model_params
        )
        
        # 更新模型参数
        self.model_params.update(optimization_results['best_params'])
        
        # 使用最佳参数创建新模型
        self.model = model_class(**self.model_params)
        
        # 使用全部训练数据重新训练
        self.model.fit(X_train, y_train)
        self.trained = True
        
        logger.info(f"超参数优化完成，最佳参数: {optimization_results['best_params']}")
        return optimization_results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
        
        Returns:
            Dict[str, float]: 性能指标字典
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        logger.info("评估模型性能...")
        metrics = self.evaluator.evaluate(self.model, X_test, y_test)
        
        logger.info(f"模型评估完成: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        return metrics
    
    def save_model(self, model_path: Optional[str] = None, 
                  include_metadata: bool = True) -> str:
        """
        保存模型
        
        Args:
            model_path (str, optional): 模型保存路径，如果为None则使用默认路径
            include_metadata (bool): 是否包含元数据
        
        Returns:
            str: 模型保存路径
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 如果未指定路径，使用默认路径
        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{self.model_type.lower()}_{timestamp}.pkl")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info(f"保存模型到: {model_path}")
        
        # 准备保存的数据
        save_data = {
            'model': self.model
        }
        
        # 如果包含元数据
        if include_metadata:
            save_data.update({
                'model_type': self.model_type,
                'model_params': self.model_params,
                'timestamp': datetime.now().isoformat(),
                'trained': self.trained
            })
        
        # 保存模型
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
            logger.info(f"模型保存成功: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}", exc_info=True)
            raise
    
    def load_model(self, model_path: str) -> BaseModel:
        """
        加载模型
        
        Args:
            model_path (str): 模型文件路径
        
        Returns:
            BaseModel: 加载的模型
        """
        logger.info(f"从{model_path}加载模型...")
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            # 如果是字典格式（包含元数据）
            if isinstance(data, dict) and 'model' in data:
                self.model = data['model']
                self.model_type = data.get('model_type', self.model_type)
                self.model_params = data.get('model_params', self.model_params)
                self.trained = data.get('trained', True)
            else:
                # 如果直接是模型对象
                self.model = data
                self.trained = True
            
            logger.info(f"模型加载成功: {self.model}")
            return self.model
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X (np.ndarray): 预测特征
        
        Returns:
            np.ndarray: 预测结果
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        logger.info(f"使用{self.model_type}模型进行预测...")
        return self.model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        if self.model is None:
            return {
                'model_type': self.model_type,
                'model_params': self.model_params,
                'trained': False
            }
        
        # 获取模型信息
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        
        # 添加基本信息
        info = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'trained': self.trained
        }
        
        # 合并信息
        info.update(model_info)
        
        return info
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        交叉验证模型
        
        Args:
            X (np.ndarray): 特征
            y (np.ndarray): 目标
            cv (int): 交叉验证折数
        
        Returns:
            Dict[str, Any]: 交叉验证结果
        """
        logger.info(f"执行{cv}折交叉验证...")
        
        # 创建模型（如果尚未创建）
        if self.model is None:
            self.model = self.create_model()
        
        # 执行交叉验证
        cv_results = self.evaluator.cross_validate(self.model, X, y, cv=cv)
        
        logger.info(f"交叉验证完成: 平均R²={cv_results['mean_r2']:.4f}, 平均RMSE={cv_results['mean_rmse']:.4f}")
        return cv_results
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          optimize: bool = True) -> Dict[str, Any]:
        """
        训练并评估模型（完整流程）
        
        Args:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
            optimize (bool): 是否执行超参数优化
        
        Returns:
            Dict[str, Any]: 训练和评估结果
        """
        logger.info("开始完整的模型训练和评估流程...")
        
        # 1. 创建基础模型
        base_model = self.create_model()
        
        # 2. 训练基础模型
        base_model.fit(X_train, y_train)
        
        # 3. 评估基础模型
        base_metrics = self.evaluator.evaluate(base_model, X_test, y_test)
        logger.info(f"基础模型性能: R²={base_metrics['r2']:.4f}, RMSE={base_metrics['rmse']:.4f}")
        
        # 4. 如果需要，执行超参数优化
        optimization_results = None
        if optimize:
            optimization_results = self.optimize_hyperparameters(X_train, y_train)
            
            # 5. 评估优化后的模型
            opt_metrics = self.evaluate_model(X_test, y_test)
            logger.info(f"优化后模型性能: R²={opt_metrics['r2']:.4f}, RMSE={opt_metrics['rmse']:.4f}")
            
            # 计算改进
            improvements = {
                'r2_improvement': opt_metrics['r2'] - base_metrics['r2'],
                'rmse_improvement': base_metrics['rmse'] - opt_metrics['rmse']
            }
            
            logger.info(f"性能改进: R²提升{improvements['r2_improvement']:.4f}, RMSE降低{improvements['rmse_improvement']:.4f}")
        else:
            # 如果不优化，使用基础模型
            self.model = base_model
            self.trained = True
            opt_metrics = base_metrics
            improvements = {'r2_improvement': 0.0, 'rmse_improvement': 0.0}
        
        # 6. 保存模型
        model_path = self.save_model()
        
        # 7. 构建结果
        results = {
            'model_type': self.model_type,
            'model_path': model_path,
            'base_metrics': base_metrics,
            'optimized_metrics': opt_metrics,
            'improvements': improvements,
            'optimization_results': optimization_results
        }
        
        logger.info("模型训练和评估流程完成")
        return results