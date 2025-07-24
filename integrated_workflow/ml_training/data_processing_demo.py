#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理组件演示脚本
展示如何使用数据处理组件进行数据加载、预处理、分割和特征工程
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置路径，确保可以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入数据处理组件
from integrated_workflow.ml_training.data_processing import DataProcessor
from integrated_workflow.ml_training.ml_data_manager import MLDataManager
from integrated_workflow.utils.logging_utils import setup_logging


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='数据处理组件演示',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                        help='数据文件路径')
    
    parser.add_argument('--output', type=str, default='integrated_workflow/results/data_processing_demo',
                        help='输出目录路径')
    
    parser.add_argument('--target', type=str, default='Pn',
                        help='目标列名称')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='测试集比例')
    
    parser.add_argument('--random-state', type=int, default=42,
                        help='随机种子')
    
    parser.add_argument('--feature-engineering', action='store_true',
                        help='是否执行特征工程')
    
    parser.add_argument('--log-level', type=str, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', 
                        help='日志级别')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output, 'data_processing.log')
    setup_logging(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("数据处理组件演示")
    logger.info(f"数据文件: {args.data}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"目标列: {args.target}")
    logger.info(f"测试集比例: {args.test_size}")
    logger.info(f"随机种子: {args.random_state}")
    logger.info(f"特征工程: {'是' if args.feature_engineering else '否'}")
    logger.info("="*60)
    
    try:
        # 创建配置
        config = {
            'data_path': args.data,
            'target_column': args.target,
            'test_size': args.test_size,
            'random_state': args.random_state,
            'enable_feature_engineering': args.feature_engineering,
            'handle_outliers': True,
            'outlier_strategy': 'clip',
            'auto_feature_selection': False,
            'scaler_type': 'standard',
            'feature_engineering': {
                'polynomial_features': args.feature_engineering,
                'polynomial_degree': 2,
                'interaction_features': args.feature_engineering,
                'ratio_features': args.feature_engineering
            }
        }
        
        # 保存配置
        config_path = os.path.join(args.output, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"配置已保存: {config_path}")
        
        # 创建数据处理器
        data_processor = DataProcessor(config)
        
        # 加载数据
        logger.info("加载数据...")
        df = data_processor.load_data()
        logger.info(f"数据加载完成，形状: {df.shape}")
        
        # 保存原始数据统计信息
        stats = df.describe()
        stats_path = os.path.join(args.output, 'original_stats.csv')
        stats.to_csv(stats_path)
        logger.info(f"原始数据统计信息已保存: {stats_path}")
        
        # 验证数据
        logger.info("验证数据...")
        data_processor.validate_data(df)
        
        # 如果启用特征工程，创建新特征
        if args.feature_engineering:
            logger.info("执行特征工程...")
            df = data_processor.create_features(df)
            logger.info(f"特征工程完成，新形状: {df.shape}")
            
            # 保存特征工程后的数据
            engineered_path = os.path.join(args.output, 'engineered_data.csv')
            df.to_csv(engineered_path, index=False)
            logger.info(f"特征工程后的数据已保存: {engineered_path}")
        
        # 预处理数据
        logger.info("预处理数据...")
        df = data_processor.preprocess_data(df)
        logger.info(f"预处理完成，形状: {df.shape}")
        
        # 保存预处理后的数据
        processed_path = os.path.join(args.output, 'processed_data.csv')
        df.to_csv(processed_path, index=False)
        logger.info(f"预处理后的数据已保存: {processed_path}")
        
        # 分割数据
        logger.info("分割数据...")
        X_train, X_test, y_train, y_test = data_processor.split_data(df)
        logger.info(f"数据分割完成，训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 标准化数据
        logger.info("标准化数据...")
        X_train_scaled, X_test_scaled = data_processor.normalize_data(X_train, X_test, config['scaler_type'])
        logger.info("数据标准化完成")
        
        # 保存预处理器
        logger.info("保存预处理器...")
        preprocessor_dir = os.path.join(args.output, 'preprocessors')
        os.makedirs(preprocessor_dir, exist_ok=True)
        preprocessor_paths = data_processor.save_preprocessor(preprocessor_dir)
        logger.info(f"预处理器已保存: {preprocessor_paths}")
        
        # 保存分割后的数据
        train_path = os.path.join(args.output, 'train_data.csv')
        test_path = os.path.join(args.output, 'test_data.csv')
        
        # 合并特征和目标
        train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        train_df[args.target] = y_train
        
        test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        test_df[args.target] = y_test
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"训练数据已保存: {train_path}")
        logger.info(f"测试数据已保存: {test_path}")
        
        # 生成数据可视化
        logger.info("生成数据可视化...")
        plots_dir = os.path.join(args.output, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 目标分布图
        plt.figure(figsize=(10, 6))
        plt.hist(df[args.target], bins=30, alpha=0.7)
        plt.title(f'{args.target} 分布')
        plt.xlabel(args.target)
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'target_distribution.png'))
        plt.close()
        
        # 2. 相关性热图
        plt.figure(figsize=(12, 10))
        corr_matrix = df.corr()
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
        plt.colorbar()
        plt.title('特征相关性热图')
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # 3. 训练集和测试集分布比较
        for col in X_train.columns[:5]:  # 只展示前5个特征
            plt.figure(figsize=(10, 6))
            plt.hist(X_train[col], bins=20, alpha=0.5, label='训练集')
            plt.hist(X_test[col], bins=20, alpha=0.5, label='测试集')
            plt.title(f'{col} 在训练集和测试集中的分布')
            plt.xlabel(col)
            plt.ylabel('频率')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, f'{col}_train_test_distribution.png'))
            plt.close()
        
        logger.info(f"数据可视化已保存: {plots_dir}")
        
        # 使用MLDataManager进行完整流程演示
        logger.info("\n" + "="*60)
        logger.info("使用MLDataManager进行完整流程演示")
        
        # 创建MLDataManager
        ml_config = {
            'data_processing': config,
            'output_dir': os.path.join(args.output, 'ml_data_manager')
        }
        
        ml_data_manager = MLDataManager(ml_config)
        
        # 准备训练数据
        logger.info("准备训练数据...")
        training_data = ml_data_manager.prepare_training_data(args.data)
        
        # 分析数据
        logger.info("分析数据...")
        df_for_analysis = ml_data_manager.load_and_preprocess_data(args.data)
        analysis_results = ml_data_manager.analyze_data(df_for_analysis)
        
        # 保存分析结果
        analysis_path = os.path.join(args.output, 'data_analysis.json')
        with open(analysis_path, 'w') as f:
            # 将numpy类型转换为Python原生类型
            analysis_json = {}
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    analysis_json[key] = {k: v if not isinstance(v, np.number) else float(v) for k, v in value.items()}
                else:
                    analysis_json[key] = value
            
            json.dump(analysis_json, f, indent=2)
        
        logger.info(f"数据分析结果已保存: {analysis_path}")
        
        logger.info("数据处理演示完成")
        logger.info("="*60)
        
        print("\n" + "="*60)
        print("数据处理演示完成!")
        print(f"结果保存在: {args.output}")
        print(f"处理后的数据: {processed_path}")
        print(f"数据可视化: {plots_dir}")
        print(f"日志文件: {log_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"数据处理演示失败: {str(e)}", exc_info=True)
        print(f"\n错误: 数据处理演示失败")
        print(f"详细错误信息: {str(e)}")
        print(f"请查看日志文件获取更多信息: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()