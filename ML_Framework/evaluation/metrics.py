"""
模型评估指标计算和可视化模块
提供各种机器学习模型的评估指标计算功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算模型评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        y_std: 预测标准差（可选，用于不确定性量化）
        
    Returns:
        包含各种指标的字典
    """
    # 基础回归指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE（平均绝对百分比误差）
    # 避免除零错误
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # 残差统计
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    max_error = np.max(np.abs(residuals))
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'max_error': max_error
    }
    
    # 如果提供了标准差，添加不确定性相关指标
    if y_std is not None:
        metrics['mean_std'] = np.mean(y_std)
        metrics['max_std'] = np.max(y_std)
        
        # 计算置信区间覆盖率（95%）
        lower_bound = y_pred - 1.96 * y_std
        upper_bound = y_pred + 1.96 * y_std
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        metrics['confidence_interval_coverage'] = coverage
    
    return metrics


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, 
                title: str = "Model Prediction Results", 
                save_path: Optional[str] = None,
                figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    绘制模型结果的综合分析图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图标题
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)
    residuals = y_true - y_pred
    
    # 1. 预测vs实际值散点图
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction Line')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Predicted vs Actual Values\nR² = {metrics["r2"]:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差vs预测值
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差分布直方图
    axes[0, 2].hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residual Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Test)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 时间序列图（按索引）
    indices = np.arange(len(y_true))
    axes[1, 1].plot(indices, y_true, label='Actual Values', alpha=0.7)
    axes[1, 1].plot(indices, y_pred, label='Predicted Values', alpha=0.7)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Prediction Sequence Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 指标汇总表
    axes[1, 2].axis('off')
    metrics_text = f"""
    Evaluation Metrics Summary:
    
    R²: {metrics['r2']:.4f}
    RMSE: {metrics['rmse']:.4f}
    MAE: {metrics['mae']:.4f}
    MAPE: {metrics['mape']:.2f}%
    
    Residual Statistics:
    Mean: {metrics['mean_residual']:.4f}
    Std Dev: {metrics['std_residual']:.4f}
    Max Error: {metrics['max_error']:.4f}
    """
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_learning_curve(train_scores: List[float], val_scores: List[float],
                       title: str = "Learning Curve", 
                       save_path: Optional[str] = None) -> None:
    """
    绘制学习曲线
    
    Args:
        train_scores: 训练分数列表
        val_scores: 验证分数列表
        title: 图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    iterations = range(len(train_scores))
    
    plt.plot(iterations, train_scores, 'b-', label='Training Score', linewidth=2)
    plt.plot(iterations, val_scores, 'r-', label='Validation Score', linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                          title: str = "Feature Importance", 
                          save_path: Optional[str] = None) -> None:
    """
    绘制特征重要性图
    
    Args:
        feature_names: 特征名称列表
        importances: 特征重要性数组
        title: 图标题
        save_path: 保存路径
    """
    # 按重要性排序
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix(data: pd.DataFrame, title: str = "Correlation Matrix",
                          save_path: Optional[str] = None) -> None:
    """
    绘制相关性矩阵热力图
    
    Args:
        data: 数据DataFrame
        title: 图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 计算相关性矩阵
    correlation_matrix = data.corr()
    
    # 绘制热力图
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_error_distribution(errors: np.ndarray, title: str = "Error Distribution",
                          save_path: Optional[str] = None) -> None:
    """
    绘制误差分布图
    
    Args:
        errors: 误差数组
        title: 图标题
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 误差直方图
    ax1.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Errors')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution Histogram')
    ax1.grid(True, alpha=0.3)
    
    # 误差箱线图
    ax2.boxplot(errors, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Errors')
    ax2.set_title('Error Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_performance_dashboard(results: Dict[str, Dict[str, Any]], 
                               save_path: Optional[str] = None) -> None:
    """
    创建性能仪表板
    
    Args:
        results: 多个模型的结果字典
        save_path: 保存路径
    """
    if not results:
        print("No results to display")
        return
    
    # 准备数据
    models = list(results.keys())
    metrics_data = []
    
    for model_name, result in results.items():
        for metric_name, value in result['metrics'].items():
            metrics_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(metrics_data)
    
    # 创建仪表板
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Dashboard', fontsize=16)
    
    # 1. R²比较
    r2_data = df[df['Metric'] == 'r2']
    if not r2_data.empty:
        sns.barplot(data=r2_data, x='Model', y='Value', ax=axes[0, 0])
        axes[0, 0].set_title('R² Comparison')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. RMSE比较
    rmse_data = df[df['Metric'] == 'rmse']
    if not rmse_data.empty:
        sns.barplot(data=rmse_data, x='Model', y='Value', ax=axes[0, 1])
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. MAE比较
    mae_data = df[df['Metric'] == 'mae']
    if not mae_data.empty:
        sns.barplot(data=mae_data, x='Model', y='Value', ax=axes[1, 0])
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 综合雷达图
    main_metrics = ['r2', 'rmse', 'mae']
    radar_data = []
    
    for model_name in models:
        model_metrics = []
        for metric in main_metrics:
            metric_data = df[(df['Model'] == model_name) & (df['Metric'] == metric)]
            if not metric_data.empty:
                value = metric_data['Value'].iloc[0]
                # 对于RMSE和MAE，取倒数以便在雷达图中显示（越大越好）
                if metric in ['rmse', 'mae']:
                    value = 1 / (1 + value)
                model_metrics.append(value)
            else:
                model_metrics.append(0)
        radar_data.append(model_metrics)
    
    # 绘制雷达图
    if radar_data:
        angles = np.linspace(0, 2*np.pi, len(main_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 完成圆圈
        
        ax = axes[1, 1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        for i, (model_name, values) in enumerate(zip(models, radar_data)):
            values += values[:1]  # 完成圆圈
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(main_metrics)
        ax.set_title('Comprehensive Performance Radar Chart')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 