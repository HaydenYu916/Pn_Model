#!/usr/bin/env python3
"""
最优种植条件寻找器 - 集成CLED计算（简化结构版本）
Optimal Growing Conditions Finder with CLED Integration (Simplified Structure)

🔧 配置说明：
- 通过修改 moo_optimization_config.yaml 来配置算法和参数
- 支持NSGA2、NSGA3、INSGA2算法
- 支持LSSVR和SVR模型
- 目标：CLED最小化，Pn最大化

使用方法：
python find_optimal_conditions_multi_model.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML_Framework')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import pickle
import yaml
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入pymoo相关模块
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions

# 使用自定义的NSGA3实现
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'algorithms/moo')))
from nsga3 import NSGA3

# 导入i-NSGA-II算法
try:
    from algorithms.moo.i_nsga2 import iNSGA2
    INSGA2_AVAILABLE = True
except ImportError:
    INSGA2_AVAILABLE = False
    print("⚠️  警告: i-NSGA-II算法不可用，请确保i_nsga2.py文件存在")

CONFIG_PATH = "moo_optimization_config.yaml"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calc_cled(ppfd, rb_ratio, cled_params):
    """
    计算CLED值 - 基于论文公式
    公式: Cl = (PPFDLED(t) × S × Ca) / (Eff × 3.6 × 10³)
    返回: mg·m⁻²·s⁻¹
    """
    if ppfd <= 0:
        return 0.0
    
    # 获取计算方法
    calculation_method = cled_params.get('calculation_method', 'standard')
    
    # 根据方法选择对应的参数
    if calculation_method == 'standard':
        params = cled_params['standard']
    elif calculation_method == 'detailed':
        params = cled_params['detailed']
    else:
        raise ValueError(f"不支持的CLED计算方法: {calculation_method}")
    
    # 分解光谱分量
    red_ppfd = ppfd * rb_ratio        # 红光分量
    blue_ppfd = ppfd * (1 - rb_ratio) # 蓝光分量
    
    # 获取参数
    Ca = params['carbon_factor']  # 碳排因子 (kg CO₂/MWh)
    S = params['surface_area']    # 照射面积 (m²)
    conversion_factor = params['conversion_factor']  # 转换因子 (s/h)
    
    # LED光量子效率
    red_efficiency = params['led_efficiency']['red']    # μmol·s⁻¹·W⁻¹
    blue_efficiency = params['led_efficiency']['blue']  # μmol·s⁻¹·W⁻¹
    
    # 系统效率
    if calculation_method == 'standard':
        system_efficiency = params['system_efficiency']
    else:  # detailed
        eff = params['efficiency']
        system_efficiency = eff['driver'] * eff['thermal'] * eff['optical']
    
    # 碳排因子转换
    Ca_g_per_kwh = Ca * 0.001  # kg/MWh → g/kWh
    
    # 红光LED碳排放
    if red_ppfd > 0:
        red_power_density = red_ppfd / red_efficiency  # W/m²
        red_cl_density = red_power_density * Ca_g_per_kwh / 1000  # g CO₂/(h·m²)
    else:
        red_cl_density = 0.0
        
    # 蓝光LED碳排放
    if blue_ppfd > 0:
        blue_power_density = blue_ppfd / blue_efficiency  # W/m²
        blue_cl_density = blue_power_density * Ca_g_per_kwh / 1000  # g CO₂/(h·m²)
    else:
        blue_cl_density = 0.0
    
    # 总碳排放密度
    total_cl_density = (red_cl_density + blue_cl_density) / system_efficiency
    
    # 转换为mg·m⁻²·s⁻¹
    cled = total_cl_density * 1000 / conversion_factor  # mg·m⁻²·s⁻¹
    
    return cled

class OptimalConditionProblem(Problem):
    def __init__(self, fixed_co2, fixed_temp, config):
        variables = config['problem']['variables']
        self.ppfd_min = variables['ppfd']['min']
        self.ppfd_max = variables['ppfd']['max']
        self.rb_min = variables['rb_ratio']['min']
        self.rb_max = variables['rb_ratio']['max']
        self.fixed_co2 = fixed_co2
        self.fixed_temp = fixed_temp
        self.cled_params = config['cled']
        self.model_path = config['model']['model_path']
        
        print(f"[模型路径] 当前Pn模型pkl路径: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model'] if 'model' in model_data else model_data
        self.scaler = model_data['metadata']['scaler'] if 'metadata' in model_data and 'scaler' in model_data['metadata'] else None
        self.feature_names = ['PPFD', 'CO2', 'T', 'R:B']
        
        # 显示CLED计算参数信息
        calculation_method = self.cled_params.get('calculation_method', 'standard')
        print(f"[CLED计算] 方法: {calculation_method}")
        if calculation_method == 'standard':
            params = self.cled_params['standard']
            print(f"[CLED参数] 碳排因子: {params['carbon_factor']} kg CO₂/MWh")
            print(f"[CLED参数] 系统效率: {params['system_efficiency']*100:.1f}%")
        else:
            params = self.cled_params['detailed']
            eff = params['efficiency']
            system_eff = eff['driver'] * eff['thermal'] * eff['optical']
            print(f"[CLED参数] 碳排因子: {params['carbon_factor']} kg CO₂/MWh")
            print(f"[CLED参数] 系统效率: {system_eff*100:.1f}% (驱动:{eff['driver']*100:.0f}% × 热:{eff['thermal']*100:.0f}% × 光学:{eff['optical']*100:.0f}%)")
        
        super().__init__(n_var=2, n_obj=2, n_constr=0, 
                        xl=np.array([self.ppfd_min, self.rb_min]), 
                        xu=np.array([self.ppfd_max, self.rb_max]))
    
    def _evaluate(self, X, out, *args, **kwargs):
        N = X.shape[0]
        cled_values = np.zeros(N)
        pn_values = np.zeros(N)
        
        for i in range(N):
            ppfd, rb_ratio = X[i, 0], X[i, 1]
            
            # CLED计算
            cled = calc_cled(ppfd, rb_ratio, self.cled_params)
            cled_values[i] = cled
            
            # Pn预测
            input_df = pd.DataFrame([[ppfd, self.fixed_co2, self.fixed_temp, rb_ratio]], 
                                  columns=self.feature_names)
            if self.scaler is not None:
                input_scaled = self.scaler.transform(input_df).values
            else:
                input_scaled = input_df.values
            pn_pred = self.model.predict(input_scaled)[0]
            pn_values[i] = pn_pred
        
        # 目标1最小化CLED，目标2最大化Pn（转换为最小化）
        out["F"] = np.column_stack([cled_values, -pn_values])
        out["cled_raw"] = cled_values
        out["pn_raw"] = pn_values

def create_algorithm(config):
    algo_config = config['algorithm']
    algorithm_type = algo_config['algorithm_type']
    pop_size = algo_config['population_size']
    
    if algorithm_type == "NSGA2":
        crossover = SBX(prob=algo_config['nsga2']['crossover']['prob'], 
                       eta=algo_config['nsga2']['crossover']['eta'])
        mutation = PM(prob=algo_config['nsga2']['mutation']['prob'], 
                     eta=algo_config['nsga2']['mutation']['eta'])
        return NSGA2(pop_size=pop_size, crossover=crossover, mutation=mutation, 
                    eliminate_duplicates=algo_config.get('eliminate_duplicates', True))
    
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
    print("\n🌱 最优种植条件寻找器 (PPFD & R:B, 固定CO2和温度)")
    print(f"配置文件: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    
    # 输入固定条件
    co2 = float(input("请输入CO2浓度 (ppm, 如400): "))
    temp = float(input("请输入温度 (°C, 如24): "))
    print(f"优化条件: CO2={co2} ppm, T={temp} °C")
    
    # 创建优化问题
    problem = OptimalConditionProblem(co2, temp, config)
    
    # 创建算法
    algorithm = create_algorithm(config)
    generations = config['algorithm']['n_generations']
    algorithm_type = config['algorithm']['algorithm_type']
    print(f"优化算法: {algorithm_type}, 代数: {generations}, 种群: {config['algorithm']['population_size']}")
    
    # 运行优化
    result = minimize(problem, algorithm, ('n_gen', generations), verbose=True)
    X = result.X
    F = result.F
    cled_values = F[:, 0]
    pn_values = -F[:, 1]
    
    # Pareto前沿可视化
    plt.figure(figsize=(8,6))
    plt.scatter(cled_values, pn_values, c='b', alpha=0.7)
    plt.xlabel('CLED (mg·m⁻²·s⁻¹)')
    plt.ylabel('Pn (μmol·m⁻²·s⁻¹)')
    plt.title('Pareto Front: CLED vs Pn')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(config['model']['model_path']).split('_')[1].upper()
    out_dir = f"results/paper_optimal_conditions_{model_name.lower()}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pareto_front.png"), dpi=300)
    
    # 保存解集
    df = pd.DataFrame({
        'PPFD': X[:, 0],
        'R:B': X[:, 1],
        'CLED': cled_values,
        'Pn': pn_values
    })
    df.to_csv(os.path.join(out_dir, "pareto_solutions.csv"), index=False)
    
    print(f"\n✅ 优化完成！结果已保存到: {out_dir}")
    print(f"   - pareto_front.png: Pareto前沿图")
    print(f"   - pareto_solutions.csv: 解集")
    print(f"   - Pn模型路径: {config['model']['model_path']}")
    
    # 显示最优解
    best_idx = np.argmax(pn_values - cled_values/np.max(cled_values))
    print("\n🌟 推荐解 (Pn与CLED加权最大):")
    print(df.iloc[best_idx])

    # ===== 新增：3D表面图叠加Pareto解集点 =====
    print("\n📈 正在生成Pareto解集3D表面图...")
    # 生成网格
    ppfd_range = np.linspace(problem.ppfd_min, problem.ppfd_max, 50)
    rb_range = np.linspace(problem.rb_min, problem.rb_max, 50)
    ppfd_mesh, rb_mesh = np.meshgrid(ppfd_range, rb_range)
    pn_mesh = np.zeros_like(ppfd_mesh)
    
    for i in range(ppfd_mesh.shape[0]):
        for j in range(ppfd_mesh.shape[1]):
            ppfd = ppfd_mesh[i, j]
            rb = rb_mesh[i, j]
            input_df = pd.DataFrame([[ppfd, co2, temp, rb]], columns=problem.feature_names)
            if problem.scaler is not None:
                input_scaled = problem.scaler.transform(input_df).values
            else:
                input_scaled = input_df.values
            pn_pred = problem.model.predict(input_scaled)[0]
            pn_mesh[i, j] = pn_pred
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(ppfd_mesh, rb_mesh, pn_mesh, cmap='jet', alpha=0.9, 
                          edgecolor='none', linewidth=0, antialiased=True)
    
    # Pareto解集点
    ax.scatter(X[:, 0], X[:, 1], pn_values, c='black', s=50, 
              label='Pareto Solutions', depthshade=False)
    
    ax.set_xlabel('PPFD (μmol·m⁻²·s⁻¹)', fontsize=12, labelpad=10)
    ax.set_ylabel('R:B Ratio', fontsize=12, labelpad=10)
    ax.set_zlabel('Pn (μmol·m⁻²·s⁻¹)', fontsize=12, labelpad=10)
    ax.set_title(f'Photosynthesis Rate Surface\nCO2={co2} ppm, T={temp}°C', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(problem.ppfd_min, problem.ppfd_max)
    ax.set_ylim(problem.rb_min, problem.rb_max)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.view_init(elev=30, azim=35)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Pn (μmol·m⁻²·s⁻¹)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    
    surface_path = os.path.join(out_dir, f"pareto_surface_co2_{int(co2)}_temp_{int(temp)}.png")
    plt.savefig(surface_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {surface_path}")

if __name__ == "__main__":
    main() 