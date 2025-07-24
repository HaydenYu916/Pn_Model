"""
模型评估器类
Model Evaluator Class
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
import seaborn as sns

from config.config import EvaluationConfig
from models.base_model import BaseModel

class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化评估器
        
        Args:
            config: 评估配置
        """
        self.config = config
        self.results = {}
        
    def evaluate_model(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray,
                      X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            X_train: 训练特征（可选）
            y_train: 训练标签（可选）
            
        Returns:
            评估结果字典
        """
        if not model.fitted:
            raise ValueError("模型未训练")
        
        # 预测
        if hasattr(model, 'predict_with_uncertainty'):
            y_pred, y_std = model.predict_with_uncertainty(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            metrics['mean_std'] = np.mean(y_std)
            metrics['max_std'] = np.max(y_std)
            
            # 计算置信区间覆盖率
            if hasattr(model, 'calculate_confidence_intervals'):
                mean, lower, upper = model.calculate_confidence_intervals(X_test)
                coverage = np.mean((y_test >= lower) & (y_test <= upper))
                metrics['confidence_interval_coverage'] = coverage
        else:
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
        
        # 如果提供了训练数据，也计算训练集指标
        if X_train is not None and y_train is not None:
            if hasattr(model, 'predict_with_uncertainty'):
                y_train_pred, y_train_std = model.predict_with_uncertainty(X_train)
                train_metrics = self._calculate_metrics(y_train, y_train_pred)
                train_metrics['mean_std'] = np.mean(y_train_std)
                train_metrics['max_std'] = np.max(y_train_std)
                
                if hasattr(model, 'calculate_confidence_intervals'):
                    mean, lower, upper = model.calculate_confidence_intervals(X_train)
                    coverage = np.mean((y_train >= lower) & (y_train <= upper))
                    train_metrics['confidence_interval_coverage'] = coverage
            else:
                y_train_pred = model.predict(X_train)
                train_metrics = self._calculate_metrics(y_train, y_train_pred)
            
            # 添加训练集指标
            for key, value in train_metrics.items():
                metrics[f'train_{key}'] = value
        
        # 保存结果
        self.results[model.__class__.__name__] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_test,
            'has_uncertainty': hasattr(model, 'predict_with_uncertainty')
        }
        
        if hasattr(model, 'predict_with_uncertainty'):
            self.results[model.__class__.__name__]['std'] = y_std
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            指标字典
        """
        metrics = {}
        
        if 'mse' in self.config.metrics:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        if 'rmse' in self.config.metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'r2' in self.config.metrics:
            metrics['r2'] = r2_score(y_true, y_pred)
        
        if 'mae' in self.config.metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # 计算其他指标
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        
        return metrics
    
    def cross_validate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        交叉验证评估
        
        Args:
            model: 模型
            X: 特征
            y: 标签
            
        Returns:
            交叉验证结果
        """
        cv_results = {}
        
        # K-fold交叉验证
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        for metric in self.config.metrics:
            if metric == 'rmse':
                scoring = 'neg_mean_squared_error'
                scores = cross_val_score(model.model, X, y, cv=kfold, scoring=scoring)
                cv_results[f'{metric}_cv'] = np.sqrt(-scores)
            elif metric == 'r2':
                scoring = 'r2'
                scores = cross_val_score(model.model, X, y, cv=kfold, scoring=scoring)
                cv_results[f'{metric}_cv'] = scores
            elif metric == 'mae':
                scoring = 'neg_mean_absolute_error'
                scores = cross_val_score(model.model, X, y, cv=kfold, scoring=scoring)
                cv_results[f'{metric}_cv'] = -scores
        
        # 计算统计信息
        for key, scores in cv_results.items():
            cv_results[f'{key}_mean'] = np.mean(scores)
            cv_results[f'{key}_std'] = np.std(scores)
        
        return cv_results
    
    def plot_predictions(self, model_name: str, save_path: Optional[str] = None):
        """
        绘制预测对比图
        
        Args:
            model_name: 模型名称
            save_path: 保存路径
        """
        if model_name not in self.results:
            raise ValueError(f"模型 {model_name} 的结果不存在")
        
        result = self.results[model_name]
        y_true = result['actuals']
        y_pred = result['predictions']
        
        plt.figure(figsize=self.config.plot_size)
        
        # 如果有不确定性估计，添加误差棒
        if result.get('has_uncertainty', False):
            y_std = result['std']
            plt.errorbar(y_true, y_pred, yerr=2*y_std, fmt='o', alpha=0.3, 
                        color='blue', label='Predicted Values (±2σ)')
        else:
            plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predicted Values')
        
        # 完美预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction Line')
        
        # 添加统计信息
        r2 = result['metrics']['r2']
        rmse = result['metrics']['rmse']
        
        if result.get('has_uncertainty', False):
            mean_std = result['metrics']['mean_std']
            coverage = result['metrics'].get('confidence_interval_coverage', 0)
            plt.title(f'{model_name} - Prediction vs Actual\n' + 
                     f'R² = {r2:.4f}, RMSE = {rmse:.4f}\n' +
                     f'Mean σ = {mean_std:.4f}, CI Coverage = {coverage:.2%}')
        else:
            plt.title(f'{model_name} - Prediction vs Actual\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # 关闭图片，不显示
    
    def plot_residuals(self, model_name: str, save_path: Optional[str] = None):
        """
        绘制残差图
        
        Args:
            model_name: 模型名称
            save_path: 保存路径
        """
        if model_name not in self.results:
            raise ValueError(f"模型 {model_name} 的结果不存在")
        
        result = self.results[model_name]
        y_true = result['actuals']
        y_pred = result['predictions']
        residuals = y_true - y_pred
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 残差vs预测值
        if result.get('has_uncertainty', False):
            y_std = result['std']
            ax1.errorbar(y_pred, residuals, yerr=2*y_std, fmt='o', alpha=0.3)
        else:
            ax1.scatter(y_pred, residuals, alpha=0.6)
        
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # 残差分布直方图
        ax2.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # 残差vs实际值
        if result.get('has_uncertainty', False):
            ax4.errorbar(y_true, residuals, yerr=2*y_std, fmt='o', alpha=0.3)
        else:
            ax4.scatter(y_true, residuals, alpha=0.6)
        
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Actual Values')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # 关闭图片，不显示
    
    def plot_uncertainty_analysis(self, model_name: str, save_path: Optional[str] = None):
        """
        绘制不确定性分析图（仅适用于具有不确定性估计的模型）
        
        Args:
            model_name: 模型名称
            save_path: 保存路径
        """
        if model_name not in self.results:
            raise ValueError(f"模型 {model_name} 的结果不存在")
        
        result = self.results[model_name]
        if not result.get('has_uncertainty', False):
            raise ValueError(f"模型 {model_name} 不支持不确定性估计")
        
        y_true = result['actuals']
        y_pred = result['predictions']
        y_std = result['std']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 预测标准差分布
        ax1.hist(y_std, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Prediction Standard Deviation')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Uncertainty')
        ax1.grid(True, alpha=0.3)
        
        # 预测标准差vs预测值
        ax2.scatter(y_pred, y_std, alpha=0.6)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Prediction Standard Deviation')
        ax2.set_title('Uncertainty vs Predictions')
        ax2.grid(True, alpha=0.3)
        
        # 预测标准差vs预测误差
        abs_error = np.abs(y_true - y_pred)
        ax3.scatter(y_std, abs_error, alpha=0.6)
        ax3.plot([0, max(y_std)], [0, max(y_std)], 'r--', label='y=x')
        ax3.set_xlabel('Prediction Standard Deviation')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Uncertainty vs Absolute Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 预测区间覆盖率分析
        sorted_indices = np.argsort(y_std)
        percentiles = np.linspace(0, 100, 10)
        coverages = []
        
        for p in percentiles[1:]:
            idx = sorted_indices[:int(len(sorted_indices) * p/100)]
            lower = y_pred[idx] - 2 * y_std[idx]
            upper = y_pred[idx] + 2 * y_std[idx]
            coverage = np.mean((y_true[idx] >= lower) & (y_true[idx] <= upper))
            coverages.append(coverage)
        
        ax4.plot(percentiles[1:], coverages, 'o-')
        ax4.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
        ax4.set_xlabel('Data Percentile')
        ax4.set_ylabel('95% CI Coverage')
        ax4.set_title('Coverage Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # 关闭图片，不显示
    
    def compare_models(self, models: List[str], save_path: Optional[str] = None):
        """
        比较多个模型的性能
        
        Args:
            models: 模型名称列表
            save_path: 保存路径
        """
        if not models:
            raise ValueError("模型列表不能为空")
        
        # 检查所有模型是否都有结果
        for model_name in models:
            if model_name not in self.results:
                raise ValueError(f"模型 {model_name} 的结果不存在")
        
        # 准备比较数据
        metrics_data = []
        for model_name in models:
            metrics = self.results[model_name]['metrics']
            for metric_name, value in metrics.items():
                if metric_name in self.config.metrics:
                    metrics_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        # 创建数据框
        df = pd.DataFrame(metrics_data)
        
        # 绘制比较图
        plt.figure(figsize=self.config.plot_size)
        sns.barplot(data=df, x='Metric', y='Value', hue='Model')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(self, model_name: str) -> str:
        """
        生成评估报告
        
        Args:
            model_name: 模型名称
            
        Returns:
            评估报告文本
        """
        if model_name not in self.results:
            raise ValueError(f"模型 {model_name} 的结果不存在")
        
        result = self.results[model_name]
        metrics = result['metrics']
        
        report = [
            f"=== {model_name} 评估报告 ===\n",
            "测试集性能:",
            f"  R² Score: {metrics['r2']:.4f}",
            f"  RMSE: {metrics['rmse']:.4f}",
            f"  MAE: {metrics['mae']:.4f}",
            f"  平均残差: {metrics['mean_residual']:.4f}",
            f"  残差标准差: {metrics['std_residual']:.4f}",
            f"  最大误差: {metrics['max_error']:.4f}"
        ]
        
        if result.get('has_uncertainty', False):
            report.extend([
                "\n不确定性估计:",
                f"  平均标准差: {metrics['mean_std']:.4f}",
                f"  最大标准差: {metrics['max_std']:.4f}",
                f"  置信区间覆盖率: {metrics.get('confidence_interval_coverage', 0):.2%}"
            ])
        
        # 添加训练集性能
        train_metrics = {k[6:]: v for k, v in metrics.items() if k.startswith('train_')}
        if train_metrics:
            report.extend([
                "\n训练集性能:",
                f"  R² Score: {train_metrics['r2']:.4f}",
                f"  RMSE: {train_metrics['rmse']:.4f}",
                f"  MAE: {train_metrics['mae']:.4f}"
            ])
        
        return "\n".join(report)
    
    def get_results(self) -> Dict[str, Any]:
        """获取所有评估结果"""
        return self.results
    
    def clear_results(self):
        """清除所有评估结果"""
        self.results = {} 