#!/usr/bin/env python3
"""
配置驱动的实验运行脚本（论文数据版）
Configuration-driven Experiment Runner for Paper Data

使用YAML配置文件来运行光合作用预测实验（适配论文数据集，无R:B特征）

实验流程：
1. 加载配置文件（YAML）
2. 数据处理（训练/测试分割，标准化）
3. 构建并训练基础模型
4. 评估基础模型
5. 超参数优化（遗传算法/粒子群/IBOA/TPE/CMAES等）
6. 训练优化后的模型
7. 评估优化后的模型
8. 保存结果和模型
9. 生成可视化图表（所有图例使用英文，无R:B相关内容）
"""

import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config, save_config
from data_processing import DataProcessor
from models import SVRModel, LSSVRModel, GPRModel, DGPModel
from optimizers import (
    GeneticAlgorithm, 
    ParticleSwarmOptimization,
    TPEOptimizer,
    OptunaRandomSearch,
    IBOAOptimizer,
    CMAESOptimizer
)
from evaluation import ModelEvaluator
from utils import setup_logger, save_model, save_results

# 模型和优化器映射
MODEL_MAP = {'SVR': SVRModel, 'LSSVR': LSSVRModel, 'GPR': GPRModel, 'DGP': DGPModel}
OPTIMIZER_MAP = {
    'GA': GeneticAlgorithm, 'PSO': ParticleSwarmOptimization, 'TPE': TPEOptimizer,
    'RANDOM': OptunaRandomSearch, 'IBOA': IBOAOptimizer, 'CMAES': CMAESOptimizer
}

def get_model_params(config):
    """获取模型参数"""
    model_type = config.model.model_type.upper()
    if model_type == 'SVR':
        params = {'C': config.model.C, 'gamma': config.model.gamma, 'kernel': config.model.kernel}
        if hasattr(config.model, 'epsilon'): params['epsilon'] = config.model.epsilon
    elif model_type == 'LSSVR':
        params = {'gamma': config.model.gamma, 'sigma2': config.model.sigma2, 'kernel': config.model.kernel}
    elif model_type == 'GPR':
        params = {'alpha': config.model.alpha, 'n_restarts_optimizer': config.model.n_restarts_optimizer}
    elif model_type == 'DGP':
        params = {'n_layers': config.model.n_layers, 'alpha': config.model.dgp_alpha, 'random_state': config.data.random_state}
    return params

def get_optimizer_params(config):
    """获取优化器参数"""
    opt_type = config.optimization.optimizer_type.upper()
    if opt_type == 'GA':
        return {'population_size': config.optimization.population_size, 'generations': config.optimization.generations,
                'crossover_rate': config.optimization.crossover_rate, 'mutation_rate': config.optimization.mutation_rate,
                'tournament_size': config.optimization.tournament_size}
    elif opt_type == 'PSO':
        return {'n_particles': config.optimization.n_particles, 'n_iterations': config.optimization.n_iterations,
                'w': config.optimization.w, 'c1': config.optimization.c1, 'c2': config.optimization.c2}
    elif opt_type == 'IBOA':
        return {'n_butterflies': getattr(config.optimization, 'n_butterflies', 20),
                'sensory_modality': getattr(config.optimization, 'sensory_modality', 0.01),
                'power_exponent': getattr(config.optimization, 'power_exponent', 0.1),
                'switch_probability': getattr(config.optimization, 'switch_probability', 0.8)}
    elif opt_type == 'CMAES':
        return {'n_trials': config.optimization.n_trials, 'population_size': config.optimization.cmaes_population_size}
    else:  # TPE, RANDOM
        return {'n_trials': config.optimization.n_trials}

def run_experiment(config_input):
    """运行实验主函数"""
    # 处理配置输入
    if isinstance(config_input, str):
        config = load_config(config_input)
        config_path = config_input
        print(f"📋 加载配置文件: {config_path}")
    else:
        config = config_input
        config_path = "命令行参数覆盖配置"
        print(f"📋 使用配置对象: {config_path}")
    
    # 设置实验目录 - 添加paper前缀
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config.model.model_type.upper()
    optimizer_type = config.optimization.optimizer_type.upper()
    experiment_name = f"paper_{model_type}_{optimizer_type}_{timestamp}"
    experiment_dir = os.path.join(config.experiment.results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(experiment_dir, config.experiment.log_file)
    logger = setup_logger(
        name=experiment_name,
        log_file=log_file,
        level=config.experiment.log_level
    )
    
    logger.info(f"开始实验: {experiment_name}")
    logger.info(f"实验目录: {experiment_dir}")
    
    # 保存配置副本
    config_copy_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_copy_path)
    logger.info(f"配置已保存至: {config_copy_path}")
    
    try:
        # 1. 数据处理
        logger.info("🔄 开始数据处理...")
        data_processor = DataProcessor(config.data)
        X_train, X_test, y_train, y_test = data_processor.process_all()
        
        data_info = data_processor.get_data_info()
        logger.info(f"数据处理完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        logger.info(f"特征: {data_info['features']}")
        logger.info(f"目标: {data_info['target']}")
        
        # 2. 获取模型和优化器类
        model_class = MODEL_MAP.get(config.model.model_type.upper())
        optimizer_class = OPTIMIZER_MAP.get(config.optimization.optimizer_type.upper())
        
        if model_class is None:
            raise ValueError(f"不支持的模型类型: {config.model.model_type}")
        if optimizer_class is None:
            raise ValueError(f"不支持的优化器类型: {config.optimization.optimizer_type}")
        
        logger.info(f"使用模型: {model_class.__name__}")
        logger.info(f"使用优化器: {optimizer_class.__name__}")
        
        # 3. 创建基础模型
        logger.info("🤖 创建基础模型...")
        model_params = get_model_params(config)
        base_model = model_class(**model_params)
        base_model.fit(X_train, y_train)
        logger.info("基础模型训练完成")
        
        # 4. 模型评估
        logger.info("📊 评估基础模型...")
        evaluator = ModelEvaluator(config.evaluation)
        base_metrics = evaluator.evaluate_model(base_model, X_test, y_test, X_train, y_train)
        
        logger.info(f"基础模型性能:")
        for metric, value in base_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # 5. 超参数优化
        logger.info("⚡ 开始超参数优化...")
        optimizer_params = get_optimizer_params(config)

        # 定义目标函数
        def objective_function(params):
            # 合并参数
            full_params = model_params.copy()
            full_params.update(params)
            # 构建模型
            model = model_class(**full_params)
            model.fit(X_train, y_train)
            # 评估
            metrics = evaluator.evaluate_model(model, X_train, y_train, X_train, y_train)
            # 优化目标：最小化RMSE（如有其它目标可调整）
            return metrics.get('rmse', 0)
        
        optimizer = optimizer_class(
            objective_function=objective_function,
            param_bounds=config.optimization.param_bounds,
            **optimizer_params
        )
        
        # 执行优化
        if config.optimization.optimizer_type.upper() == 'GA':
            optimization_results = optimizer.optimize(
                model_class, X_train, y_train,
                n_trials=config.optimization.generations
            )
        elif config.optimization.optimizer_type.upper() in ['IBOA', 'PSO', 'TPE', 'RANDOM', 'CMAES']:
            optimization_results = optimizer.optimize(
                model_class, X_train, y_train,
                n_trials=config.optimization.n_trials
            )
        
        best_params = optimization_results['best_params']
        best_score = optimization_results['best_score']
        
        logger.info(f"优化完成:")
        logger.info(f"  最佳参数: {best_params}")
        logger.info(f"  最佳分数: {best_score:.4f}")
        
        # 6. 训练优化后的模型
        logger.info("🎯 训练优化后的模型...")
        optimized_model = model_class(**best_params)
        optimized_model.fit(X_train, y_train)
        
        # 评估优化后的模型
        optimized_metrics = evaluator.evaluate_model(
            optimized_model, X_test, y_test, X_train, y_train
        )
        
        logger.info(f"优化后模型性能:")
        for metric, value in optimized_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # 7. 保存结果
        logger.info("💾 保存结果...")
        
        # 保存模型
        models_dir = os.path.join(experiment_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        base_model_path = os.path.join(models_dir, f"base_{config.model.model_type.lower()}_model.pkl")
        optimized_model_path = os.path.join(models_dir, f"optimized_{config.model.model_type.lower()}_model.pkl")
        
        # 保存模型和标准化器
        save_model(base_model, base_model_path, 
                  metadata={'config': config, 'metrics': base_metrics, 'scaler': data_processor.normalizer})
        save_model(optimized_model, optimized_model_path,
                  metadata={'config': config, 'metrics': optimized_metrics, 'optimization': optimization_results, 'scaler': data_processor.normalizer})
        
        logger.info(f"模型已保存: {base_model_path}, {optimized_model_path}")
        
        # 保存所有结果
        all_results = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': timestamp,
                'config': config_path
            },
            'data_info': data_info,
            'models': {
                'base': {
                    'type': config.model.model_type,
                    'params': model_params,
                    'metrics': base_metrics
                },
                'optimized': {
                    'type': config.model.model_type,
                    'params': best_params,
                    'metrics': optimized_metrics
                }
            },
            'optimization': optimization_results,
            'improvement': {
                'r2_improvement': optimized_metrics['r2'] - base_metrics['r2'],
                'rmse_improvement': base_metrics['rmse'] - optimized_metrics['rmse']
            }
        }
        
        results_file = os.path.join(experiment_dir, "results.json")
        save_results(all_results, results_file)
        logger.info(f"结果已保存: {results_file}")
        
        # 8. 生成可视化
        logger.info("📈 生成可视化图表...")
        plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 为评估器添加模型结果
        evaluator.results['Base Model'] = {
            'metrics': base_metrics,
            'actuals': y_test,
            'predictions': base_model.predict(X_test)
        }
        evaluator.results['Optimized Model'] = {
            'metrics': optimized_metrics,
            'actuals': y_test,
            'predictions': optimized_model.predict(X_test)
        }
        
        # 生成基础对比图和预测图
        comparison_path = os.path.join(plots_dir, "model_comparison.png")
        evaluator.compare_models(
            ['Base Model', 'Optimized Model'],
            save_path=comparison_path
        )
        
        prediction_path = os.path.join(plots_dir, "prediction_results.png")
        evaluator.plot_predictions(
            'Optimized Model',
            save_path=prediction_path
        )
        
        # ================= 新可视化部分（仿 run_experiment.py，全部R:B相关改为CO2分组） =================
        logger.info("📊 Generating enhanced visualizations (CO2 as group variable)...")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['text.usetex'] = False
        sns.set_style("whitegrid")
        raw_data = data_processor.data
        feature_names = data_processor.config.features
        target_name = data_processor.config.target
        plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. 温度分层可视化（PPFD vs Pn，按CO2分组）
        logger.info("🌡️ Generating temperature-layered scatter plots (group by CO2)...")
        temperatures = sorted(raw_data['T'].unique())
        for temp in temperatures:
            temp_data = raw_data[raw_data['T'] == temp]
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            co2_levels = sorted(temp_data['CO2'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(co2_levels)))
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            for i, co2 in enumerate(co2_levels):
                co2_data = temp_data[temp_data['CO2'] == co2]
                if len(co2_data) > 0:
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    label_text = f'CO2 {int(co2)} ppm'
                    ax1.scatter(co2_data['PPFD'], co2_data[target_name], c=[color], marker=marker, alpha=0.7, s=50, label=label_text)
            ax1.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12)
            ax1.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
            ax1.set_title(f'Photosynthesis vs PPFD at {temp}°C', fontsize=14, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'avg_temp_{int(temp)}C.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 2. 温度比较图（不同温度下PPFD/CO2 vs Pn，按CO2分组）
        logger.info("🌡️ Generating temperature comparison plots (group by CO2)...")
        co2_values = sorted(raw_data['CO2'].unique())
        for co2 in co2_values:
            co2_data = raw_data[raw_data['CO2'] == co2]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            temp_levels = sorted(co2_data['T'].unique())
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(temp_levels)))
            for i, temp in enumerate(temp_levels):
                temp_data = co2_data[co2_data['T'] == temp]
                ppfd_means = temp_data.groupby('PPFD')[target_name].mean()
                ax1.plot(ppfd_means.index, ppfd_means.values, color=colors[i], linewidth=2, marker='o', label=f'{temp}°C', markersize=6)
            ax1.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12)
            ax1.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
            ax1.set_title(f'Photosynthesis vs PPFD (CO2 = {co2})', fontsize=14, fontweight='bold')
            ax1.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            for i, temp in enumerate(temp_levels):
                temp_data = co2_data[co2_data['T'] == temp]
                co2_means = temp_data.groupby('CO2')[target_name].mean()
                ax2.plot(co2_means.index, co2_means.values, color=colors[i], linewidth=2, marker='s', label=f'{temp}°C', markersize=6)
            ax2.set_xlabel('CO2 (ppm)', fontsize=12)
            ax2.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
            ax2.set_title(f'Photosynthesis vs CO2 (CO2 = {co2})', fontsize=14, fontweight='bold')
            ax2.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'temp_comparison_co2_{co2}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 箱线图（PPFD=1000时，温度/CO2分组）
        logger.info("📦 Generating boxplots (group by T and CO2 at PPFD=1000)...")
        ppfd_1000_data = raw_data[raw_data['PPFD'] == 1000]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=ppfd_1000_data, x='T', y=target_name, ax=ax)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
        ax.set_title('Photosynthesis Distribution at PPFD=1000 umol·m-2·s-1', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "boxplot_ppfd1000_temp_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=ppfd_1000_data, x='CO2', y=target_name, ax=ax)
        ax.set_xlabel('CO2 (ppm)', fontsize=12)
        ax.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
        ax.set_title('Photosynthesis Distribution at PPFD=1000 umol·m-2·s-1', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "boxplot_ppfd1000_co2_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 3D可视化（PPFD, CO2, Pn，T固定为中位数）
        logger.info("🎯 Generating 3D visualizations...")
        fig = plt.figure(figsize=(15, 5))
        ppfd_range = np.linspace(raw_data['PPFD'].min(), raw_data['PPFD'].max(), 50)
        co2_range = np.linspace(raw_data['CO2'].min(), raw_data['CO2'].max(), 50)
        ppfd_mesh, co2_mesh = np.meshgrid(ppfd_range, co2_range)
        temp_val = raw_data['T'].median()
        points_3d = []
        for ppfd in ppfd_range:
            for co2 in co2_range:
                input_point = np.array([[ppfd, co2, temp_val]])
                input_normalized = data_processor.normalizer.transform(pd.DataFrame(input_point, columns=feature_names))
                pred = optimized_model.predict(input_normalized.values)[0]
                points_3d.append(pred)
        pn_mesh = np.array(points_3d).reshape(ppfd_mesh.shape)
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(ppfd_mesh, co2_mesh, pn_mesh, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('PPFD (umol·m-2·s-1)')
        ax1.set_ylabel('CO2 (ppm)')
        ax1.set_zlabel(f'{target_name} (umol·m-2·s-1)')
        ax1.set_title(f'3D Surface: {target_name} vs PPFD & CO2\n(T={temp_val}°C)')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        ax2 = fig.add_subplot(132, projection='3d')
        sample_data = raw_data.sample(min(1000, len(raw_data)))
        scatter = ax2.scatter(sample_data['PPFD'], sample_data['CO2'], sample_data[target_name], c=sample_data[target_name], cmap='viridis', alpha=0.6)
        ax2.set_xlabel('PPFD (umol·m-2·s-1)')
        ax2.set_ylabel('CO2 (ppm)')
        ax2.set_zlabel(f'{target_name} (umol·m-2·s-1)')
        ax2.set_title('3D Scatter: Actual Data')
        fig.colorbar(scatter, ax=ax2, shrink=0.5, aspect=5)
        ax3 = fig.add_subplot(133)
        contour = ax3.contourf(ppfd_mesh, co2_mesh, pn_mesh, levels=20, cmap='viridis')
        ax3.set_xlabel('PPFD (umol·m-2·s-1)')
        ax3.set_ylabel('CO2 (ppm)')
        ax3.set_title(f'Contour: {target_name} vs PPFD & CO2\n(T={temp_val}°C)')
        fig.colorbar(contour, ax=ax3, shrink=0.8, aspect=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "3d_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 论文风格3D表面图（每个温度一张）
        logger.info("📈 Generating paper-style 3D surface plots for each temperature...")
        ppfd_range_paper = np.linspace(0, 1900, 100)
        co2_range_paper = np.linspace(0, 2200, 100)
        ppfd_mesh_paper, co2_mesh_paper = np.meshgrid(ppfd_range_paper, co2_range_paper)
        temp_list = [18, 20, 22, 24, 26, 30]
        for temp_val in temp_list:
            logger.info(f"   Generating T={temp_val}°C 3D surface plot...")
            pn_surface = np.zeros_like(ppfd_mesh_paper)
            for i, ppfd in enumerate(ppfd_range_paper):
                for j, co2 in enumerate(co2_range_paper):
                    input_point = np.array([[ppfd, co2, temp_val]])
                    input_normalized = data_processor.normalizer.transform(pd.DataFrame(input_point, columns=feature_names))
                    pred = optimized_model.predict(input_normalized.values)[0]
                    pn_surface[j, i] = pred
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(ppfd_mesh_paper, co2_mesh_paper, pn_surface, cmap='jet', alpha=0.9, edgecolor='none', linewidth=0, antialiased=True)
            ax.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12, labelpad=10)
            ax.set_ylabel('CO₂ (ppm)', fontsize=12, labelpad=10)
            ax.set_zlabel('Pn (μmol m⁻² s⁻¹)', fontsize=12, labelpad=10)
            ax.set_title(f'Photosynthesis Rate Surface\nT = {temp_val}°C', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, 1900)
            ax.set_ylim(0, 2200)
            ax.invert_xaxis()  # 如需反转可取消注释
            ax.invert_yaxis()
            ax.view_init(elev=30, azim=35)
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label('Pn (μmol m⁻² s⁻¹)', fontsize=12)
            plt.tight_layout()
            filename = f"pn_surface_temp_{temp_val}_paper_style.png"
            plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"   ✅ Saved: {filename}")

        # 6. 单张大范围CO2/PPFD 3D表面图
        logger.info("📊 Generating single 3D surface plot for full CO2/PPFD range...")
        ppfd_range_full = np.linspace(0, 1900, 100)
        co2_range_full = np.linspace(0, 2200, 100)
        ppfd_mesh_full, co2_mesh_full = np.meshgrid(ppfd_range_full, co2_range_full)
        temp_fixed = raw_data['T'].median()
        points_3d = []
        for ppfd in ppfd_range_full:
            for co2 in co2_range_full:
                input_point = np.array([[ppfd, co2, temp_fixed]])
                input_normalized = data_processor.normalizer.transform(pd.DataFrame(input_point, columns=feature_names))
                pred = optimized_model.predict(input_normalized.values)[0]
                points_3d.append(pred)
        pn_mesh_full = np.array(points_3d).reshape(ppfd_mesh_full.shape)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(ppfd_mesh_full, co2_mesh_full, pn_mesh_full, cmap='jet', alpha=0.9, edgecolor='none', linewidth=0, antialiased=True)
        ax.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12, labelpad=10)
        ax.set_ylabel('CO₂ (ppm)', fontsize=12, labelpad=10)
        ax.set_zlabel('Pn (μmol m⁻² s⁻¹)', fontsize=12, labelpad=10)
        ax.set_title(f'Photosynthesis Rate Surface\n(T = {temp_fixed:.1f}°C)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1900)
        ax.set_ylim(0, 2200)
        ax.invert_xaxis()  # 如需反转可取消注释
        ax.invert_yaxis()
        ax.view_init(elev=30, azim=35)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Pn (μmol m⁻² s⁻¹)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pn_surfaces_all_co2_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("   ✅ Saved: pn_surfaces_all_co2_comparison.png")
        # ================= END 新可视化部分 =================
        
        # 生成特征相关性热力图
        logger.info("🔥 生成特征相关性热力图...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        correlation_matrix = raw_data.corr()
        
        # 创建热力图
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成模型性能分析图
        logger.info("📊 生成模型性能分析图...")
        
        # 残差分析
        y_pred = optimized_model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 残差vs预测值
        ax1.scatter(y_pred, residuals, alpha=0.6, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 残差直方图
        ax2.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q图
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 预测vs实际值
        ax4.scatter(y_test, y_pred, alpha=0.6, color='purple')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax4.set_xlabel('Actual Values', fontsize=12)
        ax4.set_ylabel('Predicted Values', fontsize=12)
        ax4.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加R²和RMSE信息
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax4.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "model_performance_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成统计报告
        logger.info("📝 生成统计报告...")
        report_path = os.path.join(plots_dir, "statistical_analysis_results.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("PHOTOSYNTHESIS MODEL STATISTICAL ANALYSIS REPORT (PAPER DATA)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. DATA OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total samples: {len(raw_data)}\n")
            f.write(f"Features: {', '.join(feature_names)}\n")
            f.write(f"Target: {target_name}\n\n")
            
            f.write("2. DATA STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(raw_data.describe().to_string())
            f.write("\n\n")
            
            f.write("3. MODEL PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Base Model R²: {base_metrics['r2']:.4f}\n")
            f.write(f"Optimized Model R²: {optimized_metrics['r2']:.4f}\n")
            f.write(f"R² Improvement: {optimized_metrics['r2'] - base_metrics['r2']:.4f}\n")
            f.write(f"Base Model RMSE: {base_metrics['rmse']:.4f}\n")
            f.write(f"Optimized Model RMSE: {optimized_metrics['rmse']:.4f}\n")
            f.write(f"RMSE Improvement: {base_metrics['rmse'] - optimized_metrics['rmse']:.4f}\n\n")
            
            f.write("4. OPTIMIZATION RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Optimizer: {config.optimization.optimizer_type}\n")
            f.write(f"Best Score: {optimization_results['best_score']:.4f}\n")
            f.write(f"Best Parameters: {optimization_results['best_params']}\n\n")
            
            f.write("5. FEATURE CORRELATIONS\n")
            f.write("-" * 30 + "\n")
            correlations = raw_data.corr()[target_name].sort_values(ascending=False)
            for feature, corr in correlations.items():
                f.write(f"{feature}: {corr:.4f}\n")
        
        logger.info(f"✅ 增强可视化图表生成完成，保存在: {plots_dir}")
        
        # 9. 实验总结
        logger.info("📝 实验总结:")
        logger.info(f"  基础模型 R²: {base_metrics['r2']:.4f}")
        logger.info(f"  优化后模型 R²: {optimized_metrics['r2']:.4f}")
        logger.info(f"  R² 提升: {optimized_metrics['r2'] - base_metrics['r2']:.4f}")
        logger.info(f"  RMSE 改善: {base_metrics['rmse'] - optimized_metrics['rmse']:.4f}")
        
        print(f"\n🎉 实验完成！结果保存在: {experiment_dir}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"实验失败: {str(e)}")
        raise

def main():
    """主函数"""
    # 🔧 在代码中直接指定配置
    CONFIG_FILE = "paper_data_config.yaml"  # 配置文件路径
    MODEL_TYPE = "LSSVR"                   # 模型类型: SVR, LSSVR, GPR, DGP
    OPTIMIZER_TYPE = "GA"                  # 优化器类型: GA, PSO, TPE, RANDOM, IBOA, CMAES

    print(f"🚀 开始光合作用预测实验（论文数据）")
    print(f"📋 配置信息:")
    print(f"   配置文件: {CONFIG_FILE}")
    print(f"   模型类型: {MODEL_TYPE}")
    print(f"   优化器类型: {OPTIMIZER_TYPE}")
    print("-" * 50)

    # 检查配置文件是否存在
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 配置文件不存在: {CONFIG_FILE}")
        return 1

    # 加载配置并设置参数
    config = load_config(CONFIG_FILE)
    config.model.model_type = MODEL_TYPE
    config.optimization.optimizer_type = OPTIMIZER_TYPE

    try:
        run_experiment(config)
        return 0
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 