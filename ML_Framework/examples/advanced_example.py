"""
高级示例 - 多模型多优化器比较
Advanced Example - Multi-Model Multi-Optimizer Comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config, DataConfig, EvaluationConfig
from data_processing import DataProcessor
from models import LSSVRModel, SVRModel, GPRModel
from optimizers import GeneticAlgorithm, ParticleSwarmOptimization
from evaluation import ModelEvaluator
from utils import setup_logger, save_model, save_results, create_experiment_logger

def main():
    """主函数"""
    # 创建实验日志
    logger = create_experiment_logger("AdvancedExample", "examples/logs")
    logger.info("开始高级示例 - 多模型多优化器比较")
    
    try:
        # 1. 配置设置
        config = create_config()
        logger.info("配置创建完成")
        
        # 2. 数据处理
        data_processor = DataProcessor(config.data)
        X_train, X_test, y_train, y_test = data_processor.process_all()
        logger.info("数据处理完成")
        
        # 3. 定义实验组合
        experiments = [
            {
                'name': 'LSSVR_GA',
                'model_class': LSSVRModel,
                'optimizer_class': GeneticAlgorithm,
                'param_bounds': {'gamma': (0.001, 10.0), 'sigma2': (0.1, 100.0)},
                'optimizer_params': {
                    'population_size': 15,
                    'generations': 15,
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.2,
                    'random_state': 42
                }
            },
            {
                'name': 'LSSVR_PSO',
                'model_class': LSSVRModel,
                'optimizer_class': ParticleSwarmOptimization,
                'param_bounds': {'gamma': (0.001, 10.0), 'sigma2': (0.1, 100.0)},
                'optimizer_params': {
                    'n_particles': 15,
                    'n_iterations': 15,
                    'w': 0.9,
                    'c1': 2.0,
                    'c2': 2.0,
                    'random_state': 42
                }
            },
            {
                'name': 'SVR_GA',
                'model_class': SVRModel,
                'optimizer_class': GeneticAlgorithm,
                'param_bounds': {'C': (0.1, 100.0), 'epsilon': (0.001, 1.0), 'gamma': (0.001, 10.0)},
                'optimizer_params': {
                    'population_size': 15,
                    'generations': 15,
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.2,
                    'random_state': 42
                }
            },
            {
                'name': 'SVR_PSO',
                'model_class': SVRModel,
                'optimizer_class': ParticleSwarmOptimization,
                'param_bounds': {'C': (0.1, 100.0), 'epsilon': (0.001, 1.0), 'gamma': (0.001, 10.0)},
                'optimizer_params': {
                    'n_particles': 15,
                    'n_iterations': 15,
                    'w': 0.9,
                    'c1': 2.0,
                    'c2': 2.0,
                    'random_state': 42
                }
            },
            {
                'name': 'GPR_GA',
                'model_class': GPRModel,
                'optimizer_class': GeneticAlgorithm,
                'param_bounds': {
                    'alpha': (1e-6, 1e-3), 
                    'length_scale': (0.1, 10.0),
                    'n_restarts_optimizer': (5, 15),  # 增加重启次数范围
                    'constant_value': (0.5, 2.0)
                },
                'optimizer_params': {
                    'population_size': 15,
                    'generations': 15,
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.2,
                    'random_state': 42
                }
            },
            {
                'name': 'GPR_PSO',
                'model_class': GPRModel,
                'optimizer_class': ParticleSwarmOptimization,
                'param_bounds': {
                    'alpha': (1e-6, 1e-3), 
                    'length_scale': (0.1, 10.0),
                    'n_restarts_optimizer': (5, 15),  # 增加重启次数范围
                    'constant_value': (0.5, 2.0)
                },
                'optimizer_params': {
                    'n_particles': 15,
                    'n_iterations': 15,
                    'w': 0.9,
                    'c1': 2.0,
                    'c2': 2.0,
                    'random_state': 42
                }
            }
        ]
        
        # 4. 执行实验
        all_results = {}
        evaluator = ModelEvaluator(config.evaluation)
        
        for exp in experiments:
            logger.info(f"开始实验: {exp['name']}")
            
            try:
                # 创建优化器
                optimizer = exp['optimizer_class'](
                    param_bounds=exp['param_bounds'],
                    **exp['optimizer_params']
                )
                
                # 执行优化
                opt_results = optimizer.optimize(
                    exp['model_class'], X_train, y_train, n_iterations=15
                )
                
                # 使用最佳参数训练模型
                best_params = opt_results['best_params']
                optimized_model = exp['model_class'](**best_params)
                optimized_model.fit(X_train, y_train)
                
                # 评估模型
                metrics = evaluator.evaluate_model(
                    optimized_model, X_test, y_test, X_train, y_train
                )
                
                # 保存结果
                all_results[exp['name']] = {
                    'model_class': exp['model_class'].__name__,
                    'optimizer_class': exp['optimizer_class'].__name__,
                    'best_params': best_params,
                    'test_metrics': metrics,
                    'optimization_history': opt_results,
                    'model': optimized_model
                }
                
                logger.info(f"✅ {exp['name']} 完成: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {exp['name']} 失败: {str(e)}")
                continue
        
        # 5. 结果分析
        if all_results:
            logger.info("开始结果分析...")
            
            # 找出最佳模型
            best_model_name = None
            best_r2 = -1
            
            for name, result in all_results.items():
                r2 = result['test_metrics']['r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = name
            
            logger.info(f"最佳模型: {best_model_name} (R²={best_r2:.4f})")
            
            # 生成性能对比图
            create_performance_comparison(all_results, logger)
            
            # 保存结果
            save_final_results(all_results, logger)
            
            logger.info("高级示例完成！")
        else:
            logger.error("没有成功的实验结果")
        
    except Exception as e:
        logger.error(f"高级示例执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

def create_config() -> Config:
    """创建配置"""
    data_config = DataConfig(
        data_path="../../SVR/Data/averaged_data.csv",
        features=['PPFD', 'CO2', 'T', 'R:B'],
        target='Pn_avg',
        test_size=0.2,
        random_state=42,
        normalize_method='standard'
    )
    
    evaluation_config = EvaluationConfig(
        cv_folds=5,
        metrics=['rmse', 'r2', 'mae'],
        save_plots=True,
        plot_size=(12, 8)
    )
    
    return Config(data=data_config, evaluation=evaluation_config)

def create_performance_comparison(results, logger):
    """创建性能对比图"""
    try:
        # 准备数据
        names = list(results.keys())
        r2_scores = [results[name]['test_metrics']['r2'] for name in names]
        rmse_scores = [results[name]['test_metrics']['rmse'] for name in names]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² 对比
        bars1 = ax1.bar(names, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_title('R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE 对比
        bars2 = ax2.bar(names, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('RMSE Comparison')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = "examples/plots/performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison chart saved: {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to create performance comparison chart: {str(e)}")

def save_final_results(results, logger):
    """保存最终结果"""
    try:
        # 准备保存的数据
        results_for_save = {}
        for name, result in results.items():
            results_for_save[name] = {
                k: v for k, v in result.items() if k != 'model'
            }
        
        # 保存结果
        save_path = "examples/results/advanced_example_results.json"
        save_results(results_for_save, save_path)
        logger.info(f"结果已保存: {save_path}")
        
        # 保存最佳模型
        best_model_name = max(results.keys(), 
                             key=lambda x: results[x]['test_metrics']['r2'])
        best_model = results[best_model_name]['model']
        
        model_save_path = "examples/models/best_advanced_model.pkl"
        save_model(best_model, model_save_path, 
                  metadata={'best_model_name': best_model_name})
        logger.info(f"最佳模型已保存: {model_save_path}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("examples/logs", exist_ok=True)
    os.makedirs("examples/models", exist_ok=True)
    os.makedirs("examples/results", exist_ok=True)
    os.makedirs("examples/plots", exist_ok=True)
    
    main() 