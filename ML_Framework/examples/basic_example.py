"""
基础使用示例
Basic Usage Example

演示如何使用模块化ML框架进行植物光合作用预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.config import Config, DataConfig, ModelConfig, OptimizationConfig, EvaluationConfig
from data_processing import DataProcessor
from models import LSSVRModel, SVRModel
from optimizers import GeneticAlgorithm
from evaluation import ModelEvaluator
from utils import setup_logger, save_model, save_results

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger("BasicExample", log_file="examples/logs/basic_example.log")
    logger.info("开始基础示例")
    
    try:
        # 1. 创建配置
        config = create_example_config()
        logger.info("配置创建完成")
        
        # 2. 数据处理
        data_processor = DataProcessor(config.data)
        X_train, X_test, y_train, y_test = data_processor.process_all()
        logger.info(f"数据处理完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        # 3. 训练基础模型
        logger.info("开始训练LSSVR模型...")
        lssvr_model = LSSVRModel(gamma=0.5, sigma2=10.0)
        lssvr_model.fit(X_train, y_train)
        logger.info("LSSVR模型训练完成")
        
        # 4. 模型评估
        evaluator = ModelEvaluator(config.evaluation)
        
        # 评估LSSVR模型
        lssvr_metrics = evaluator.evaluate_model(lssvr_model, X_test, y_test, X_train, y_train)
        logger.info(f"LSSVR评估完成: R²={lssvr_metrics['r2']:.4f}, RMSE={lssvr_metrics['rmse']:.4f}")
        
        # 5. 超参数优化示例
        logger.info("开始超参数优化...")
        
        # 定义参数边界
        param_bounds = {
            'gamma': (0.01, 10.0),
            'sigma2': (0.1, 100.0)
        }
        
        # 创建遗传算法优化器
        ga_optimizer = GeneticAlgorithm(
            param_bounds=param_bounds,
            population_size=10,  # 为了演示，使用较小的种群
            generations=20,      # 较少的代数
            random_state=42
        )
        
        # 执行优化
        optimization_results = ga_optimizer.optimize(
            LSSVRModel, X_train, y_train, n_iterations=20
        )
        
        best_params = optimization_results['best_params']
        logger.info(f"优化完成，最佳参数: {best_params}")
        
        # 6. 使用优化参数训练最终模型
        optimized_model = LSSVRModel(**best_params)
        optimized_model.fit(X_train, y_train)
        
        # 评估优化后的模型
        optimized_metrics = evaluator.evaluate_model(
            optimized_model, X_test, y_test, X_train, y_train
        )
        logger.info(f"优化模型评估: R²={optimized_metrics['r2']:.4f}, RMSE={optimized_metrics['rmse']:.4f}")
        
        # 7. 模型对比
        # 训练SVR模型进行对比
        svr_model = SVRModel(C=1.0, epsilon=0.1, gamma=0.001)
        svr_model.fit(X_train, y_train)
        svr_metrics = evaluator.evaluate_model(svr_model, X_test, y_test)
        
        # 比较模型性能
        evaluator.compare_models(['LSSVRModel', 'SVRModel'], 
                                save_path="examples/plots/model_comparison.png")
        
        # 8. 预测示例
        logger.info("进行预测示例...")
        
        # 单个预测
        test_conditions = data_processor.predict_single(ppfd=500, co2=400, t=25, rb=0.8)
        prediction = optimized_model.predict_single(test_conditions)
        logger.info(f"预测示例 (PPFD=500, CO2=400, T=25°C, R:B=0.8): Pn = {prediction:.3f}")
        
        # 批量预测
        batch_conditions = [
            [300, 400, 20, 0.8],
            [600, 400, 25, 0.8],
            [900, 400, 30, 0.8]
        ]
        batch_features = data_processor.predict_batch(batch_conditions)
        batch_predictions = optimized_model.predict_batch(batch_features)
        
        logger.info("批量预测结果:")
        for i, (conditions, pred) in enumerate(zip(batch_conditions, batch_predictions)):
            logger.info(f"  条件{i+1}: PPFD={conditions[0]}, T={conditions[2]}°C → Pn={pred:.3f}")
        
        # 9. 保存结果
        logger.info("保存模型和结果...")
        
        # 保存最佳模型
        save_model(optimized_model, "examples/models/best_lssvr_model.pkl", 
                  metadata={'optimization_results': optimization_results,
                           'test_metrics': optimized_metrics})
        
        # 保存所有结果
        all_results = {
            'models': {
                'lssvr_basic': lssvr_metrics,
                'lssvr_optimized': optimized_metrics,
                'svr_basic': svr_metrics
            },
            'optimization': optimization_results,
            'data_info': data_processor.get_data_info()
        }
        
        save_results(all_results, "examples/results/basic_example_results.json")
        
        logger.info("基础示例完成！")
        
    except Exception as e:
        logger.error(f"示例执行出错: {str(e)}")
        raise

def create_example_config() -> Config:
    """创建示例配置"""
    # 数据配置
    data_config = DataConfig(
        data_path="../SVR/Data/averaged_data.csv",  # 相对于examples目录的路径
        features=['PPFD', 'CO2', 'T', 'R:B'],
        target='Pn_avg',
        test_size=0.2,
        random_state=42,
        normalize_method='standard'
    )
    
    # 模型配置
    model_config = ModelConfig(
        model_type='LSSVR',
        gamma=1.0,
        sigma2=1.0
    )
    
    # 优化配置
    optimization_config = OptimizationConfig(
        optimizer_type='GA',
        population_size=20,
        generations=50,
        param_bounds={
            'gamma': (0.01, 10.0),
            'sigma2': (0.1, 100.0)
        }
    )
    
    # 评估配置
    evaluation_config = EvaluationConfig(
        cv_folds=5,
        metrics=['rmse', 'r2', 'mae'],
        save_plots=True
    )
    
    return Config(
        data=data_config,
        model=model_config,
        optimization=optimization_config,
        evaluation=evaluation_config
    )

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("examples/logs", exist_ok=True)
    os.makedirs("examples/models", exist_ok=True)
    os.makedirs("examples/results", exist_ok=True)
    os.makedirs("examples/plots", exist_ok=True)
    
    main() 