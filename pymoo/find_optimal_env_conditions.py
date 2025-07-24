#!/usr/bin/env python3
"""
温室环境控制多目标优化器
- 优化变量：PPFD、CO2（温度为输入参数）
- 目标1：最大化Pn（模型预测）
- 目标2：最小化总成本（电费+CO2费，CO2费用用户给定公式）
- 所有参数和模型路径均从env_control_optimization_config.yaml读取
- 结构参考find_optimal_conditions_multi_model.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML_Framework')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import yaml
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
# 使用自定义的NSGA3实现
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'algorithms/moo')))
from nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions

# 导入i-NSGA-II算法
try:
    from algorithms.moo.i_nsga2 import iNSGA2
    INSGA2_AVAILABLE = True
except ImportError:
    INSGA2_AVAILABLE = False
    print("⚠️  警告: i-NSGA-II算法不可用，请确保i_nsga2.py文件存在")

CONFIG_PATH = "env_control_optimization_config.yaml"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calc_cost(ppfd, co2, temp_c, cost_params):
    # 电费
    P = cost_params['k'] * ppfd  # kW
    C_E = P * cost_params['t'] * cost_params['E_price'] * cost_params['n']  # 元
    # CO2费
    T_K = temp_c + 273.15  # 摄氏度转开尔文
    n_CO2 = (cost_params['p'] * cost_params['V'] / (cost_params['R'] * T_K)) * (co2 / 1e6)  # mol
    C_C = n_CO2 * cost_params['M_CO2'] * cost_params['C_price']  # 元
    
    # 基础成本
    base_cost = C_E + C_C
    
    # 通过增加边缘区域成本来引导解集到中间区域
    # 计算PPFD和CO2的归一化位置（0-1之间）
    ppfd_norm = (ppfd - cost_params.get('ppfd_min', 0)) / (cost_params.get('ppfd_max', 1900) - cost_params.get('ppfd_min', 0))
    co2_norm = (co2 - cost_params.get('co2_min', 0)) / (cost_params.get('co2_max', 2200) - cost_params.get('co2_min', 0))
    
    # 计算到中心点的距离（中心点设为0.5, 0.5）
    center_distance = np.sqrt((ppfd_norm - 0.5)**2 + (co2_norm - 0.5)**2)
    
    # 边缘成本增加系数（可调整）
    edge_cost_factor = cost_params.get('edge_cost_factor', 0.5)
    
    # 边缘成本增加：距离中心越远，成本增加越多
    edge_cost_increase = edge_cost_factor * center_distance * base_cost
    
    total_cost = base_cost + edge_cost_increase
    
    return total_cost

class EnvControlProblem(Problem):
    def __init__(self, temp_c, config):
        variables = config['problem']['variables']
        self.ppfd_min = variables['ppfd']['min']
        self.ppfd_max = variables['ppfd']['max']
        self.co2_min = variables['co2']['min']
        self.co2_max = variables['co2']['max']
        self.temp_c = temp_c
        self.cost_params = config['cost']
        self.model_path = config['model']['model_path']
        print(f"[模型路径] 当前Pn模型pkl路径: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model'] if 'model' in model_data else model_data
        self.scaler = model_data['metadata']['scaler'] if 'metadata' in model_data and 'scaler' in model_data['metadata'] else None
        self.feature_names = ['PPFD', 'CO2', 'T']
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([self.ppfd_min, self.co2_min]), xu=np.array([self.ppfd_max, self.co2_max]))
    def _evaluate(self, X, out, *args, **kwargs):
        N = X.shape[0]
        pn_values = np.zeros(N)
        cost_values = np.zeros(N)
        for i in range(N):
            ppfd, co2 = X[i, 0], X[i, 1]
            # Pn预测
            input_df = pd.DataFrame([[ppfd, co2, self.temp_c]], columns=self.feature_names)
            if self.scaler is not None:
                input_scaled = self.scaler.transform(input_df).values
            else:
                input_scaled = input_df.values
            pn_pred = self.model.predict(input_scaled)[0]
            pn_values[i] = pn_pred
            # 成本
            cost = calc_cost(ppfd, co2, self.temp_c, self.cost_params)
            cost_values[i] = cost
        # 目标1最大化Pn，目标2最小化Cost
        out["F"] = np.column_stack([-pn_values, cost_values])
        out["pn_raw"] = pn_values
        out["cost_raw"] = cost_values

def create_algorithm(config):
    algo_config = config['algorithm']
    algorithm_type = algo_config['algorithm_type']
    pop_size = algo_config['population_size']
    
    if algorithm_type == "NSGA2":
        crossover = SBX(prob=algo_config['nsga2']['crossover']['prob'], eta=algo_config['nsga2']['crossover']['eta'])
        mutation = PM(prob=algo_config['nsga2']['mutation']['prob'], eta=algo_config['nsga2']['mutation']['eta'])
        return NSGA2(pop_size=pop_size, crossover=crossover, mutation=mutation, eliminate_duplicates=algo_config.get('eliminate_duplicates', True))
    
    elif algorithm_type == "NSGA3":
        # 为NSGA3生成参考方向
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        
        # 获取NSGA3参数
        nsga3_config = algo_config.get('nsga3', {})
        crossover_config = nsga3_config.get('crossover', {})
        mutation_config = nsga3_config.get('mutation', {})
        
        crossover = SBX(
            prob=crossover_config.get('prob', 0.9), 
            eta=crossover_config.get('eta', 30)
        )
        mutation = PM(
            prob=mutation_config.get('prob', 0.1), 
            eta=mutation_config.get('eta', 20)
        )
        
        return NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=algo_config.get('eliminate_duplicates', True)
        )
    
    elif algorithm_type == "INSGA2":
        if not INSGA2_AVAILABLE:
            raise ValueError("i-NSGA-II算法不可用，请确保i_nsga2.py文件存在")
        params = algo_config.get('insga2', {})
        def get_crossover(cross_config):
            if cross_config.get('type', 'SBX') == 'SBX':
                return SBX(prob=cross_config.get('prob', 0.9), eta=cross_config.get('eta', 20))
            else:
                raise ValueError(f"不支持的交叉算子: {cross_config.get('type')}")
        def get_mutation(mut_config):
            if mut_config.get('type', 'PM') == 'PM':
                return PM(prob=mut_config.get('prob', 0.5), eta=mut_config.get('eta', 20))
            else:
                raise ValueError(f"不支持的变异算子: {mut_config.get('type')}")
        crossover = get_crossover(params.get('crossover', algo_config.get('crossover', {})))
        mutation = get_mutation(params.get('mutation', algo_config.get('mutation', {})))
        return iNSGA2(
            pop_size=pop_size,
            crossover=crossover,
            mutation=mutation
        )
    else:
        raise ValueError(f"不支持的算法类型: {algorithm_type}")

def main():
    print("\n🌱 温室环境控制多目标优化器 (PPFD & CO2, 固定温度)")
    print(f"配置文件: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    # 输入温度（摄氏度）
    temp_c = float(input("请输入温度 (°C, 如24): "))
    print(f"优化温度: {temp_c} °C")
    # 创建优化问题
    problem = EnvControlProblem(temp_c, config)
    # 创建算法
    algorithm = create_algorithm(config)
    generations = config['algorithm']['n_generations']
    algorithm_type = config['algorithm']['algorithm_type']
    print(f"优化算法: {algorithm_type}, 代数: {generations}, 种群: {config['algorithm']['population_size']}")
    # 运行优化
    result = minimize(problem, algorithm, ('n_gen', generations), verbose=True)
    X = result.X
    F = result.F
    pn_values = -F[:, 0]
    cost_values = F[:, 1]
    # Pareto前沿可视化
    plt.figure(figsize=(8,6))
    plt.scatter(cost_values, pn_values, c='b', alpha=0.7)
    plt.xlabel('Total Cost (元)')
    plt.ylabel('Pn (μmol·m⁻²·s⁻¹)')
    plt.title('Pareto Front: Pn vs Total Cost')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results/env_opt_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pareto_front.png"), dpi=300)
    # 保存解集
    df = pd.DataFrame({
        'PPFD': X[:, 0],
        'CO2': X[:, 1],
        'Pn': pn_values,
        'Cost': cost_values
    })
    df.to_csv(os.path.join(out_dir, "pareto_solutions.csv"), index=False)
    print(f"\n✅ 优化完成！结果已保存到: {out_dir}")
    print(f"   - pareto_front.png: Pareto前沿图")
    print(f"   - pareto_solutions.csv: 解集")
    print(f"   - Pn模型路径: {config['model']['model_path']}")
    # 显示最优解
    best_idx = np.argmax(pn_values - cost_values/np.max(cost_values))
    print("\n🌟 推荐解 (Pn与Cost加权最大):")
    print(df.iloc[best_idx])

    # ===== 新增：3D表面图叠加Pareto解集点 =====
    print("\n📈 正在生成Pareto解集3D表面图...")
    # 生成网格
    ppfd_range = np.linspace(0, 1900, 100)
    co2_range = np.linspace(0, 2200, 100)
    ppfd_mesh, co2_mesh = np.meshgrid(ppfd_range, co2_range)
    pn_mesh = np.zeros_like(ppfd_mesh)
    for i in range(ppfd_mesh.shape[0]):
        for j in range(ppfd_mesh.shape[1]):
            ppfd = ppfd_mesh[i, j]
            co2 = co2_mesh[i, j]
            input_df = pd.DataFrame([[ppfd, co2, temp_c]], columns=problem.feature_names)
            if problem.scaler is not None:
                input_scaled = problem.scaler.transform(input_df).values
            else:
                input_scaled = input_df.values
            pn_pred = problem.model.predict(input_scaled)[0]
            pn_mesh[i, j] = pn_pred
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(ppfd_mesh, co2_mesh, pn_mesh, cmap='jet', alpha=0.9, edgecolor='none', linewidth=0, antialiased=True)
    # Pareto解集点
    ax.scatter(X[:, 0], X[:, 1], pn_values, c='black', s=50, label='Pareto Solutions', depthshade=False)
    ax.set_xlabel('PPFD (umol·m-2·s-1)', fontsize=12, labelpad=10)
    ax.set_ylabel('CO₂ (ppm)', fontsize=12, labelpad=10)
    ax.set_zlabel('Pn (μmol m⁻² s⁻¹)', fontsize=12, labelpad=10)
    ax.set_title(f'Photosynthesis Rate Surface\nT = {temp_c}°C', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1900)
    ax.set_ylim(0, 2200)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.view_init(elev=30, azim=35)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Pn (μmol m⁻² s⁻¹)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    surface_path = os.path.join(out_dir, f"pareto_surface_temp_{int(temp_c)}.png")
    plt.savefig(surface_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {surface_path}")

if __name__ == "__main__":
    main() 