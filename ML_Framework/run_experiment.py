#!/usr/bin/env python3
"""
光合作用预测实验运行器（功能完整版）

实验流程：
1. 加载配置文件（YAML）
2. 数据处理（训练/测试分割，标准化）
3. 构建并训练基础模型
4. 评估基础模型
5. 超参数优化（遗传算法/粒子群/IBOA/TPE/CMAES等）
6. 训练优化后的模型
7. 评估优化后的模型
8. 保存结果和模型
9. 生成可视化图表（所有图例使用英文）
"""

import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

from config import load_config, save_config
from data_processing import DataProcessor
from models import SVRModel, LSSVRModel, GPRModel, DGPModel
from optimizers import (
    GeneticAlgorithm, ParticleSwarmOptimization, TPEOptimizer,
    OptunaRandomSearch, IBOAOptimizer, CMAESOptimizer
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
    # 获取并行设置
    n_jobs = getattr(config.evaluation, 'n_jobs', -1)
    if opt_type == 'GA':
        return {'population_size': config.optimization.population_size, 'generations': config.optimization.generations,
                'crossover_rate': config.optimization.crossover_rate, 'mutation_rate': config.optimization.mutation_rate,
                'tournament_size': config.optimization.tournament_size, 'n_jobs': n_jobs}
    elif opt_type == 'PSO':
        return {'n_particles': config.optimization.n_particles, 'n_iterations': config.optimization.n_iterations,
                'w': config.optimization.w, 'c1': config.optimization.c1, 'c2': config.optimization.c2, 'n_jobs': n_jobs}
    elif opt_type == 'IBOA':
        return {'n_butterflies': getattr(config.optimization, 'n_butterflies', 20),
                'sensory_modality': getattr(config.optimization, 'sensory_modality', 0.01),
                'power_exponent': getattr(config.optimization, 'power_exponent', 0.1),
                'switch_probability': getattr(config.optimization, 'switch_probability', 0.8), 'n_jobs': n_jobs}
    elif opt_type == 'CMAES':
        return {'n_trials': config.optimization.n_trials, 'population_size': config.optimization.cmaes_population_size, 'n_jobs': n_jobs}
    else:  # TPE, RANDOM
        return {'n_trials': config.optimization.n_trials, 'n_jobs': n_jobs}

def run_experiment(config_input):
    """运行实验主函数"""
    # 处理配置输入
    if isinstance(config_input, str):
        config = load_config(config_input)
        config_path = config_input
    else:
        config = config_input
        config_path = "Command line parameter overrides configuration"
    
    # 设置实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.model.model_type}_{config.optimization.optimizer_type}_{timestamp}"
    exp_dir = os.path.join(config.experiment.results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 设置日志
    import logging
    log_level = getattr(logging, config.experiment.log_level.upper(), logging.INFO)
    logger = setup_logger(name=exp_name, log_file=os.path.join(exp_dir, "experiment.log"), level=log_level)
    logger.info(f"开始实验: {exp_name}")
    save_config(config, os.path.join(exp_dir, "config.yaml"))
    
    try:
        # 1. 数据处理
        logger.info("🔄 开始数据处理...")
        data_proc = DataProcessor(config.data)
        X_train, X_test, y_train, y_test = data_proc.process_all()
        data_info = data_proc.get_data_info()
        logger.info(f"数据处理完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        # 2. 获取模型和优化器类
        model_class = MODEL_MAP.get(config.model.model_type.upper())
        optimizer_class = OPTIMIZER_MAP.get(config.optimization.optimizer_type.upper())
        if not model_class or not optimizer_class:
            raise ValueError(f"不支持的模型类型或优化器类型")
        
        # 3. 构建并训练基础模型
        logger.info("🤖 创建基础模型...")
        model_params = get_model_params(config)
        base_model = model_class(**model_params)
        base_model.fit(X_train, y_train)
        
        # 4. 评估基础模型
        logger.info("📊 评估基础模型...")
        evaluator = ModelEvaluator(config.evaluation)
        base_metrics = evaluator.evaluate_model(base_model, X_test, y_test, X_train, y_train)
        logger.info(f"基础模型性能: R²={base_metrics['r2']:.4f}, RMSE={base_metrics['rmse']:.4f}")
        
        # 设置并行计算
        n_jobs = getattr(config.evaluation, 'n_jobs', -1)
        logger.info(f"🔄 并行计算设置: n_jobs={n_jobs}")
        
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

        # 根据优化器类型分别传参，避免参数冲突
        opt_type = config.optimization.optimizer_type.upper()
        if opt_type in ['GA', 'PSO']:
            optimizer = optimizer_class(
                objective_function,
                config.optimization.param_bounds,
                **optimizer_params
            )
        else:
            optimizer = optimizer_class(
                config.optimization.param_bounds,
                **optimizer_params
            )
        
        # 执行优化
        opt_type = config.optimization.optimizer_type.upper()
        if opt_type == 'GA':
            optimization_results = optimizer.optimize(model_class, X_train, y_train, n_iterations=config.optimization.generations)
        else:
            optimization_results = optimizer.optimize(model_class, X_train, y_train, n_iterations=config.optimization.n_trials)
        
        best_params = optimization_results['best_params']
        logger.info(f"优化完成: 最佳分数={optimization_results['best_score']:.4f}")
        
        # 6. 训练优化后的模型
        logger.info("🎯 训练优化后的模型...")
        opt_model = model_class(**best_params)
        opt_model.fit(X_train, y_train)
        
        # 7. 评估优化后的模型
        opt_metrics = evaluator.evaluate_model(opt_model, X_test, y_test, X_train, y_train)
        logger.info(f"优化后模型性能: R²={opt_metrics['r2']:.4f}, RMSE={opt_metrics['rmse']:.4f}")
        
        # 8. 保存结果和模型
        logger.info("💾 保存结果...")
        models_dir = os.path.join(exp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # 保存模型
        base_model_path = os.path.join(models_dir, f"base_{config.model.model_type.lower()}_model.pkl")
        opt_model_path = os.path.join(models_dir, f"optimized_{config.model.model_type.lower()}_model.pkl")
        save_model(base_model, base_model_path, metadata={'config': config, 'metrics': base_metrics, 'scaler': data_proc.normalizer})
        save_model(opt_model, opt_model_path, metadata={'config': config, 'metrics': opt_metrics, 'optimization': optimization_results, 'scaler': data_proc.normalizer})
        
        # 保存结果
        all_results = {
            'experiment_info': {'name': exp_name, 'timestamp': timestamp, 'config': config_path},
            'data_info': data_info,
            'models': {
                'base': {'type': config.model.model_type, 'params': model_params, 'metrics': base_metrics},
                'optimized': {'type': config.model.model_type, 'params': best_params, 'metrics': opt_metrics}
            },
            'optimization': optimization_results,
            'improvement': {
                'r2_improvement': opt_metrics['r2'] - base_metrics['r2'],
                'rmse_improvement': base_metrics['rmse'] - opt_metrics['rmse']
            }
        }
        save_results(all_results, os.path.join(exp_dir, "results.json"))
        
        # 9. 生成可视化图表
        logger.info("📈 生成可视化图表...")
        plots_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 为评估器添加模型结果
        evaluator.results['Base Model'] = {'metrics': base_metrics, 'actuals': y_test, 'predictions': base_model.predict(X_test)}
        evaluator.results['Optimized Model'] = {'metrics': opt_metrics, 'actuals': y_test, 'predictions': opt_model.predict(X_test)}
        
        # 生成基础对比图和预测图
        evaluator.compare_models(['Base Model', 'Optimized Model'], save_path=os.path.join(plots_dir, "model_comparison.png"))
        evaluator.plot_predictions('Optimized Model', save_path=os.path.join(plots_dir, "prediction_results.png"))
        
        # 9.1 生成增强的可视化图表（类似Result文件夹）
        logger.info("📊 生成增强可视化图表...")
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        
        # 获取原始数据用于可视化
        raw_data = data_proc.data
        feature_names = data_proc.config.features
        target_name = data_proc.config.target
        
        # 生成温度分层可视化（类似avg_temp_*.png）
        logger.info("🌡️ 生成温度分层可视化...")
        # 使用Unicode数学符号（不需要LaTeX）
        plt.rcParams['text.usetex'] = False
        
        temperatures = sorted(raw_data['T'].unique())
        for temp in temperatures:
            temp_data = raw_data[raw_data['T'] == temp]
            
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            
            # 左图：PPFD vs Pn (按CO2、R:B分组，用颜色和形状区分)
            co2_levels = sorted(temp_data['CO2'].unique())
            rb_levels = sorted(temp_data['R:B'].unique())
            
            # 定义CO2对应的形状
            co2_markers = {400: 'o', 800: 's', 600: '^'}
            available_co2 = [co2 for co2 in co2_levels if co2 in co2_markers]
            
            # 使用高区分度的颜色（避免渐变）
            distinct_colors = [
                '#1f77b4',  # 蓝色
                '#ff7f0e',  # 橙色
                '#2ca02c',  # 绿色
                '#d62728',  # 红色
                '#9467bd',  # 紫色
                '#8c564b',  # 棕色
                '#e377c2',  # 粉色
                '#7f7f7f',  # 灰色
                '#bcbd22',  # 黄绿色
                '#17becf'   # 青色
            ]
            
            # 生成颜色映射：每个R:B组合一种颜色（不考虑CO2）
            color_map = {rb: distinct_colors[i % len(distinct_colors)] 
                        for i, rb in enumerate(rb_levels)}
            
            for co2 in available_co2:
                marker = co2_markers.get(co2, 'o')
                co2_data = temp_data[temp_data['CO2'] == co2]
                
                for rb in rb_levels:
                    group_data = co2_data[co2_data['R:B'] == rb]
                    if len(group_data) > 0:
                        color = color_map[rb]
                        # 使用简单ASCII字符避免显示问题
                        label_text = f'CO2 {int(co2)} ppm, R:B {rb}'
                        
                        ax1.scatter(group_data['PPFD'], group_data[target_name], 
                                   c=color, marker=marker, alpha=0.7, s=50, 
                                   label=label_text)
            
            # 使用ASCII字符避免显示问题
            ax1.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12)
            ax1.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
            ax1.set_title(f'Photosynthesis vs PPFD at {temp}°C', fontsize=14, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'avg_temp_{int(temp)}C.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 生成温度比较图（类似temp_comparison_*.png）
        logger.info("🌡️ 生成温度比较图...")
        rb_values = sorted(raw_data['R:B'].unique())
        for rb in rb_values:
            rb_data = raw_data[raw_data['R:B'] == rb]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 左图：不同温度下的PPFD vs Pn
            temp_levels = sorted(rb_data['T'].unique())
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(temp_levels)))
            
            for i, temp in enumerate(temp_levels):
                temp_data = rb_data[rb_data['T'] == temp]
                # 按PPFD分组取平均值
                ppfd_means = temp_data.groupby('PPFD')[target_name].mean()
                ax1.plot(ppfd_means.index, ppfd_means.values, 
                        color=colors[i], linewidth=2, marker='o', 
                        label=f'{temp}°C', markersize=6)
            
            ax1.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12)
            ax1.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
            ax1.set_title(f'Photosynthesis vs PPFD (R:B = {rb})', fontsize=14, fontweight='bold')
            ax1.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 右图：不同温度下的CO2 vs Pn
            for i, temp in enumerate(temp_levels):
                temp_data = rb_data[rb_data['T'] == temp]
                # 按CO2分组取平均值
                co2_means = temp_data.groupby('CO2')[target_name].mean()
                ax2.plot(co2_means.index, co2_means.values, 
                        color=colors[i], linewidth=2, marker='s', 
                        label=f'{temp}°C', markersize=6)
            
            ax2.set_xlabel('CO2 (ppm)', fontsize=12)
            ax2.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
            ax2.set_title(f'Photosynthesis vs CO2 (R:B = {rb})', fontsize=14, fontweight='bold')
            ax2.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'temp_comparison_rb_{rb}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 生成箱线图（类似boxplot_*.png）
        logger.info("📦 生成箱线图...")
        
        # PPFD=1000时的温度比较箱线图
        ppfd_1000_data = raw_data[raw_data['PPFD'] == 1000]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=ppfd_1000_data, x='T', y=target_name, ax=ax)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
        ax.set_title('Photosynthesis Distribution at PPFD=1000 umol·m-2·s-1', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "boxplot_ppfd1000_temp_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # PPFD=1000时的R:B比较箱线图
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=ppfd_1000_data, x='R:B', y=target_name, ax=ax)
        ax.set_xlabel('R:B Ratio', fontsize=12)
        ax.set_ylabel(f'{target_name} (umol·m-2·s-1)', fontsize=12)
        ax.set_title('Photosynthesis Distribution at PPFD=1000 umol·m-2·s-1', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "boxplot_ppfd1000_rb_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成3D可视化
        logger.info("🎯 生成3D可视化...")
        
        # 3D表面图：PPFD, CO2, Pn
        fig = plt.figure(figsize=(15, 5))
        
        # 创建网格数据
        ppfd_range = np.linspace(raw_data['PPFD'].min(), raw_data['PPFD'].max(), 50)
        co2_range = np.linspace(raw_data['CO2'].min(), raw_data['CO2'].max(), 50)
        # 修正网格创建：PPFD作为X轴，CO2作为Y轴
        ppfd_mesh, co2_mesh = np.meshgrid(ppfd_range, co2_range)
        
        # 选择特定温度和R:B值进行可视化
        temp_val = raw_data['T'].median()
        rb_val = raw_data['R:B'].median()
        
        # 使用模型预测3D表面
        points_3d = []
        for ppfd in ppfd_range:
            for co2 in co2_range:
                # 创建输入数据
                input_point = np.array([[ppfd, co2, temp_val, rb_val]])
                # 标准化
                input_normalized = data_proc.normalizer.transform(
                    pd.DataFrame(input_point, columns=feature_names)
                )
                # 转换为numpy数组
                input_normalized_array = input_normalized.values
                # 预测
                pred = opt_model.predict(input_normalized_array)[0]
                points_3d.append(pred)
        
        pn_mesh = np.array(points_3d).reshape(ppfd_mesh.shape)
        
        # 绘制3D表面（修正坐标轴顺序）
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(ppfd_mesh, co2_mesh, pn_mesh, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('PPFD (umol·m-2·s-1)')
        ax1.set_ylabel('CO2 (ppm)')
        ax1.set_zlabel(f'{target_name} (umol·m-2·s-1)')
        ax1.set_title(f'3D Surface: {target_name} vs PPFD & CO2\n(T={temp_val}°C, R:B={rb_val})')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        
        # 3D散点图：实际数据
        ax2 = fig.add_subplot(132, projection='3d')
        sample_data = raw_data.sample(min(1000, len(raw_data)))
        scatter = ax2.scatter(sample_data['PPFD'], sample_data['CO2'], sample_data[target_name], 
                             c=sample_data[target_name], cmap='viridis', alpha=0.6)
        ax2.set_xlabel('PPFD (umol·m-2·s-1)')
        ax2.set_ylabel('CO2 (ppm)')
        ax2.set_zlabel(f'{target_name} (umol·m-2·s-1)')
        ax2.set_title('3D Scatter: Actual Data')
        fig.colorbar(scatter, ax=ax2, shrink=0.5, aspect=5)
        
        # 等高线图（修正坐标轴顺序）
        ax3 = fig.add_subplot(133)
        contour = ax3.contourf(ppfd_mesh, co2_mesh, pn_mesh, levels=20, cmap='viridis')
        ax3.set_xlabel('PPFD (umol·m-2·s-1)')
        ax3.set_ylabel('CO2 (ppm)')
        ax3.set_title(f'Contour: {target_name} vs PPFD & CO2\n(T={temp_val}°C, R:B={rb_val})')
        fig.colorbar(contour, ax=ax3, shrink=0.8, aspect=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "3d_visualization.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成论文Figure 5(b)风格的3D表面图 - 每个R:B比值一张图
        logger.info("📈 生成论文Figure 5(b)风格的3D表面图...")
        
        # 定义PPFD和CO2范围（论文中的范围）
        ppfd_range_paper = np.linspace(0, 1000, 50)
        co2_range_paper = np.linspace(400, 800, 50)
        # 修正网格创建：确保坐标轴值从小到大
        ppfd_mesh_paper, co2_mesh_paper = np.meshgrid(ppfd_range_paper, co2_range_paper)
        
        # 获取所有R:B比值
        rb_values = sorted(raw_data['R:B'].unique())
        temp_fixed = raw_data['T'].median()  # 固定温度
        
        for rb in rb_values:
            logger.info(f"   生成R:B={rb}的3D表面图...")
            
            # 计算该R:B下的Pn表面
            pn_surface = np.zeros_like(ppfd_mesh_paper)
            
            for i, ppfd in enumerate(ppfd_range_paper):
                for j, co2 in enumerate(co2_range_paper):
                    input_point = np.array([[ppfd, co2, temp_fixed, rb]])
                    input_normalized = data_proc.normalizer.transform(
                        pd.DataFrame(input_point, columns=feature_names)
                    )
                    pred = opt_model.predict(input_normalized.values)[0]
                    pn_surface[j, i] = pred  # 注意[j, i]
            
            # 创建3D表面图
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制表面（使用jet colormap以匹配论文风格）
            # 修正坐标轴顺序：X=PPFD, Y=CO2, Z=Pn
            surf = ax.plot_surface(ppfd_mesh_paper, co2_mesh_paper, pn_surface, 
                                 cmap='jet', alpha=0.9, edgecolor='none', 
                                 linewidth=0, antialiased=True)
            
            # 设置坐标轴标签（修正顺序）
            ax.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12, labelpad=10)
            ax.set_ylabel('CO₂ (ppm)', fontsize=12, labelpad=10)
            ax.set_zlabel('Pn (μmol m⁻² s⁻¹)', fontsize=12, labelpad=10)
            
            # 设置标题
            ax.set_title(f'Photosynthesis Rate Surface (R:B = {rb})\nT = {temp_fixed}°C', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # 设置坐标轴范围（修正顺序）
            ax.set_xlim(0, 1000)  # PPFD范围
            ax.set_ylim(400, 800)  # CO2范围
            
            # 确保坐标轴值从小到大排列
            ax.invert_xaxis()  # 反转X轴，确保PPFD从0到1000
            ax.invert_yaxis()  # 反转Y轴，确保CO2从400到800
            
            # 设置视角，确保X轴（PPFD）从左到右，Y轴（CO2）从前到后
            ax.view_init(elev=30, azim=35)
            
            # 添加颜色条
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label('Pn (μmol m⁻² s⁻¹)', fontsize=12)
            
            # 优化布局
            plt.tight_layout()
            
            # 保存图片
            filename = f"pn_surface_rb_{rb:.2f}_paper_style.png"
            plt.savefig(os.path.join(plots_dir, filename), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   ✅ 已保存: {filename}")
        
        # 生成所有R:B比值的对比图（2x3布局）
        logger.info("📊 生成所有R:B比值的对比图...")
        
        # 计算子图布局
        n_rb = len(rb_values)
        if n_rb <= 6:
            rows = 2
            cols = 3
        else:
            rows = int(np.ceil(n_rb / 3))
            cols = 3
        
        fig = plt.figure(figsize=(15, 10))
        
        for idx, rb in enumerate(rb_values):
            if idx >= 6:  # 最多显示6个子图
                break
                
            # 计算该R:B下的Pn表面
            pn_surface = np.zeros_like(ppfd_mesh_paper)
            
            for i, ppfd in enumerate(ppfd_range_paper):
                for j, co2 in enumerate(co2_range_paper):
                    input_point = np.array([[ppfd, co2, temp_fixed, rb]])
                    input_normalized = data_proc.normalizer.transform(
                        pd.DataFrame(input_point, columns=feature_names)
                    )
                    pred = opt_model.predict(input_normalized.values)[0]
                    pn_surface[j, i] = pred  # 注意[j, i]
            
            # 创建子图
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            
            # 绘制表面（修正坐标轴顺序）
            surf = ax.plot_surface(ppfd_mesh_paper, co2_mesh_paper, pn_surface, 
                                 cmap='jet', alpha=0.8, edgecolor='none')
            
            # 设置标签和标题（修正顺序）
            ax.set_xlabel('PPFD', fontsize=10)
            ax.set_ylabel('CO₂', fontsize=10)
            ax.set_zlabel('Pn', fontsize=10)
            ax.set_title(f'R:B = {rb}', fontsize=12, fontweight='bold')
            
            # 确保坐标轴值从小到大排列
            ax.invert_xaxis()  # 反转X轴，确保PPFD从0到1000
            ax.invert_yaxis()  # 反转Y轴，确保CO2从400到800
            
            # 设置视角
            ax.view_init(elev=30, azim=35)
            
            # 设置刻度标签大小
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        # 添加总标题
        fig.suptitle(f'Photosynthesis Rate Surfaces for Different R:B Ratios\n(T = {temp_fixed}°C)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pn_surfaces_all_rb_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("   ✅ 已保存: pn_surfaces_all_rb_comparison.png")
        
        # 生成模型性能分析图
        logger.info("📊 生成模型性能分析图...")
        
        # 特征重要性分析（如果模型支持）
        if hasattr(opt_model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = opt_model.feature_importances_
            feature_names = data_proc.config.features
            
            # 排序
            indices = np.argsort(importances)[::-1]
            
            ax.bar(range(len(importances)), importances[indices])
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Importance', fontsize=12)
            ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "feature_importance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 残差分析
        y_pred = opt_model.predict(X_test)
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
            f.write("PHOTOSYNTHESIS MODEL STATISTICAL ANALYSIS REPORT\n")
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
            f.write(f"Optimized Model R²: {opt_metrics['r2']:.4f}\n")
            f.write(f"R² Improvement: {opt_metrics['r2'] - base_metrics['r2']:.4f}\n")
            f.write(f"Base Model RMSE: {base_metrics['rmse']:.4f}\n")
            f.write(f"Optimized Model RMSE: {opt_metrics['rmse']:.4f}\n")
            f.write(f"RMSE Improvement: {base_metrics['rmse'] - opt_metrics['rmse']:.4f}\n\n")
            
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
        logger.info("📋 生成的图表文件:")
        logger.info("   - 3d_visualization.png: 综合3D可视化")
        logger.info("   - pn_surface_rb_*.png: 每个R:B比值的论文风格3D表面图")
        logger.info("   - pn_surfaces_all_rb_comparison.png: 所有R:B比值对比图")
        logger.info("   - model_performance_analysis.png: 模型性能分析")
        logger.info("   - statistical_analysis_results.txt: 统计分析报告")
        
        # 实验总结
        logger.info("📝 实验总结:")
        logger.info(f"  基础模型 R²: {base_metrics['r2']:.4f}")
        logger.info(f"  优化后模型 R²: {opt_metrics['r2']:.4f}")
        logger.info(f"  R² 提升: {opt_metrics['r2'] - base_metrics['r2']:.4f}")
        logger.info(f"  RMSE 改善: {base_metrics['rmse'] - opt_metrics['rmse']:.4f}")
        
        print(f"\n🎉 实验完成！结果保存在: {exp_dir}")
        return all_results
        
    except Exception as e:
        logger.error(f"实验失败: {str(e)}")
        raise

def main():
    """主函数"""
    # 🔧 在代码中直接指定配置
    CONFIG_FILE = "sample_config.yaml"  # 配置文件路径
    MODEL_TYPE = "GPR"               # 模型类型: SVR, LSSVR, GPR, DGP
    OPTIMIZER_TYPE = "CMAES"              # 优化器类型: GA, PSO, TPE, RANDOM, IBOA, CMAES (TPE最快)
    
    print(f"🚀 开始光合作用预测实验")
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