#!/usr/bin/env python3
"""
GPR模型优化测试脚本
测试优化后的GPR模型参数配置，减少收敛警告
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.gpr_model import GPRModel
from optimizers.optuna_optimizers import TPEOptimizer
from evaluation.evaluator import ModelEvaluator
from utils.logging_utils import setup_logger
from config.config import Config, DataConfig, EvaluationConfig

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger("GPR_TPE", log_file="examples/logs/gpr_tpe_test.log")
    logger.info("开始GPR模型优化测试")
    
    try:
        # 1. 加载数据
        data_path = "SVR/Data/averaged_data.csv"
        data = pd.read_csv(data_path)
        
        features = ['PPFD', 'CO2', 'T', 'R:B']
        target = 'Pn_avg'
        
        X = data[features].values
        y = data[target].values
        
        # 2. 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"数据加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        # 3. 定义优化配置
        experiment = {
            'name': 'GPR_TPE',
            'model_class': GPRModel,
            'optimizer_class': TPEOptimizer,
            'param_bounds': {
                'alpha': (1e-10, 1.0),
                'length_scale': (1e-5, 1e5),
                'constant_value': (1e-5, 1e5),
                'noise_level': (1e-10, 1.0),
                'n_restarts_optimizer': (5, 20)
            },
            'optimizer_params': {
                'n_trials': 50 
            }
        }
        
        # 4. 执行优化测试
        evaluation_config = EvaluationConfig(
            cv_folds=5,
            metrics=['rmse', 'r2', 'mae'],
            save_plots=True,
            plot_size=(12, 8)
        )
        evaluator = ModelEvaluator(evaluation_config)
        
        logger.info(f"开始测试: {experiment['name']}")
        
        try:
            # 创建优化器
            optimizer = experiment['optimizer_class'](
                param_bounds=experiment['param_bounds'],
                **experiment['optimizer_params']
            )
            
            # 执行优化
            opt_results = optimizer.optimize(
                experiment['model_class'], X_train, y_train
            )
            
            # 使用最佳参数训练模型
            best_params = opt_results['best_params']
            logger.info(f"最佳参数: {best_params}")
            
            optimized_model = experiment['model_class'](**best_params)
            optimized_model.fit(X_train, y_train)
            
            # 评估模型
            metrics = evaluator.evaluate_model(
                optimized_model, X_test, y_test, X_train, y_train
            )
            
            # 保存结果
            results = {
                'best_params': best_params,
                'test_metrics': metrics,
                'optimization_history': opt_results,
                'model': optimized_model
            }
            
            logger.info(f"✅ {experiment['name']} 完成: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
            
            # 获取模型信息
            model_info = optimized_model.get_model_info()
            logger.info(f"模型信息: {model_info}")
            
            # 获取核函数信息
            kernel_info = optimized_model.get_kernel_info()
            logger.info(f"核函数信息: {kernel_info}")
            
            # 打印优化统计信息
            logger.info("=== 优化统计 ===")
            logger.info(f"总试验次数: {opt_results['n_trials']}")
            logger.info(f"完成的试验: {opt_results['n_complete']}")
            logger.info(f"失败的试验: {opt_results['n_fail']}")
            logger.info(f"剪枝的试验: {opt_results['n_pruned']}")
            
        except Exception as e:
            logger.error(f"❌ {experiment['name']} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        logger.info("GPR优化测试完成！")
        
    except Exception as e:
        logger.error(f"GPR优化测试执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 