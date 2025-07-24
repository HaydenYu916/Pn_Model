#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估组件
负责评估模型性能和生成可视化结果
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

from ML_Framework.models import BaseModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估类，负责评估模型性能和生成可视化结果"""
    
    def __init__(self, config: Dict):
        """
        初始化模型评估器
        
        Args:
            config (Dict): 配置字典，包含评估相关参数
        """
        self.config = config
        self.output_dir = config.get('output_dir', 'results')
        self.metrics = {}
        self.visualizations = {}
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("模型评估器初始化完成")
    
    def evaluate(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model (BaseModel): 待评估的模型
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
        
        Returns:
            Dict[str, float]: 性能指标字典
        """
        logger.info("评估模型性能...")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 计算相对误差
        relative_error = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        # 保存指标
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'relative_error': relative_error
        }
        
        self.metrics = metrics
        logger.info(f"评估完成: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, 相对误差={relative_error:.2f}%")
        
        return metrics
    
    def cross_validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        交叉验证模型
        
        Args:
            model (BaseModel): 待评估的模型
            X (np.ndarray): 特征
            y (np.ndarray): 目标
            cv (int): 交叉验证折数
        
        Returns:
            Dict[str, Any]: 交叉验证结果
        """
        logger.info(f"执行{cv}折交叉验证...")
        
        # 创建K折交叉验证器
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # 存储每折的指标
        fold_metrics = []
        
        # 执行交叉验证
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # 训练模型
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_fold, y_train_fold)
            
            # 评估模型
            y_pred_fold = model_clone.predict(X_test_fold)
            
            # 计算指标
            mse = mean_squared_error(y_test_fold, y_pred_fold)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_fold, y_pred_fold)
            mae = mean_absolute_error(y_test_fold, y_pred_fold)
            
            # 保存该折的指标
            fold_metrics.append({
                'fold': i + 1,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae
            })
            
            logger.info(f"折 {i+1}/{cv}: R²={r2:.4f}, RMSE={rmse:.4f}")
        
        # 计算平均指标
        mean_metrics = {
            'mean_mse': np.mean([m['mse'] for m in fold_metrics]),
            'mean_rmse': np.mean([m['rmse'] for m in fold_metrics]),
            'mean_r2': np.mean([m['r2'] for m in fold_metrics]),
            'mean_mae': np.mean([m['mae'] for m in fold_metrics]),
            'std_mse': np.std([m['mse'] for m in fold_metrics]),
            'std_rmse': np.std([m['rmse'] for m in fold_metrics]),
            'std_r2': np.std([m['r2'] for m in fold_metrics]),
            'std_mae': np.std([m['mae'] for m in fold_metrics])
        }
        
        # 使用交叉验证预测
        y_cv_pred = cross_val_predict(model, X, y, cv=cv)
        
        # 计算整体指标
        overall_metrics = {
            'cv_mse': mean_squared_error(y, y_cv_pred),
            'cv_rmse': np.sqrt(mean_squared_error(y, y_cv_pred)),
            'cv_r2': r2_score(y, y_cv_pred),
            'cv_mae': mean_absolute_error(y, y_cv_pred)
        }
        
        # 合并结果
        cv_results = {
            'fold_metrics': fold_metrics,
            'mean_metrics': mean_metrics,
            'overall_metrics': overall_metrics
        }
        
        logger.info(f"交叉验证完成: 平均R²={mean_metrics['mean_r2']:.4f}±{mean_metrics['std_r2']:.4f}, "
                   f"平均RMSE={mean_metrics['mean_rmse']:.4f}±{mean_metrics['std_rmse']:.4f}")
        
        return cv_results
    
    def plot_predictions(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, 
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制预测结果图
        
        Args:
            model (BaseModel): 模型
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
            save_path (str, optional): 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        logger.info("绘制预测结果图...")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制散点图
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=50)
        
        # 添加对角线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # 设置标签和标题
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
        
        # 添加R²和RMSE信息
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
               transform=ax.transAxes, fontsize=12, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图已保存: {save_path}")
        
        return fig
    
    def plot_residuals(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, 
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制残差图
        
        Args:
            model (BaseModel): 模型
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
            save_path (str, optional): 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        logger.info("绘制残差图...")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算残差
        residuals = y_test - y_pred
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 残差vs预测值
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 残差直方图
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='k')
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 预测vs实际值
        ax4.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax4.set_xlabel('Actual Values', fontsize=12)
        ax4.set_ylabel('Predicted Values', fontsize=12)
        ax4.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加R²和RMSE信息
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax4.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"残差图已保存: {save_path}")
        
        return fig
    
    def compare_models(self, models: List[BaseModel], X_test: np.ndarray, y_test: np.ndarray, 
                      model_names: Optional[List[str]] = None, 
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        比较多个模型的性能
        
        Args:
            models (List[BaseModel]): 模型列表
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
            model_names (List[str], optional): 模型名称列表
            save_path (str, optional): 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        logger.info(f"比较{len(models)}个模型的性能...")
        
        # 如果未提供模型名称，使用默认名称
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(models))]
        
        # 确保模型名称和模型数量一致
        if len(model_names) != len(models):
            raise ValueError("模型名称列表长度必须与模型列表长度一致")
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 存储每个模型的指标
        model_metrics = []
        
        # 颜色列表
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        # 为每个模型绘制预测vs实际值
        for i, (model, name) in enumerate(zip(models, model_names)):
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # 保存指标
            model_metrics.append({
                'name': name,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae
            })
            
            # 绘制预测vs实际值
            ax1.scatter(y_test, y_pred, alpha=0.6, label=name, color=colors[i])
        
        # 添加对角线
        min_val = min(y_test.min(), min([m.predict(X_test).min() for m in models]))
        max_val = max(y_test.max(), max([m.predict(X_test).max() for m in models]))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('Actual Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制R²对比条形图
        model_names_short = [name[:15] + '...' if len(name) > 15 else name for name in model_names]
        ax2.bar(model_names_short, [m['r2'] for m in model_metrics], color=colors)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('R²', fontsize=12)
        ax2.set_title('R² Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        for i, v in enumerate([m['r2'] for m in model_metrics]):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 绘制RMSE对比条形图
        ax3.bar(model_names_short, [m['rmse'] for m in model_metrics], color=colors)
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('RMSE', fontsize=12)
        ax3.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        for i, v in enumerate([m['rmse'] for m in model_metrics]):
            ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 绘制MAE对比条形图
        ax4.bar(model_names_short, [m['mae'] for m in model_metrics], color=colors)
        ax4.set_xlabel('Model', fontsize=12)
        ax4.set_ylabel('MAE', fontsize=12)
        ax4.set_title('MAE Comparison', fontsize=14, fontweight='bold')
        for i, v in enumerate([m['mae'] for m in model_metrics]):
            ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"模型比较图已保存: {save_path}")
        
        return fig
    
    def generate_report(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, 
                       output_path: str) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            model (BaseModel): 模型
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
            output_path (str): 输出路径
        
        Returns:
            Dict[str, Any]: 报告内容
        """
        logger.info("生成评估报告...")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 评估模型
        metrics = self.evaluate(model, X_test, y_test)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算残差
        residuals = y_test - y_pred
        
        # 生成报告内容
        report = {
            'metrics': metrics,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'residual_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'median': np.median(residuals)
            }
        }
        
        # 保存报告
        with open(output_path, 'w') as f:
            f.write("# 模型评估报告\n\n")
            
            f.write("## 性能指标\n\n")
            f.write(f"- R²: {metrics['r2']:.4f}\n")
            f.write(f"- RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"- MSE: {metrics['mse']:.4f}\n")
            f.write(f"- MAE: {metrics['mae']:.4f}\n")
            f.write(f"- 相对误差: {metrics['relative_error']:.2f}%\n\n")
            
            f.write("## 残差统计\n\n")
            f.write(f"- 均值: {report['residual_stats']['mean']:.4f}\n")
            f.write(f"- 标准差: {report['residual_stats']['std']:.4f}\n")
            f.write(f"- 最小值: {report['residual_stats']['min']:.4f}\n")
            f.write(f"- 最大值: {report['residual_stats']['max']:.4f}\n")
            f.write(f"- 中位数: {report['residual_stats']['median']:.4f}\n\n")
            
            f.write("## 模型信息\n\n")
            for key, value in report['model_info'].items():
                f.write(f"- {key}: {value}\n")
        
        logger.info(f"评估报告已保存: {output_path}")
        
        return report