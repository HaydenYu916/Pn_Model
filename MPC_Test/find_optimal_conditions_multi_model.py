#!/usr/bin/env python3
"""
最优种植条件寻找器 - 类结构重构版
Optimal Growing Conditions Finder (Class-based Refactor)

- PnModel: 负责Pn模型加载与预测
- CLEDCalculator: 负责CLED计算
- MultiObjectiveOptimizer: 负责多目标优化流程

使用方法：
python find_optimal_conditions_multi_model.py
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
import pickle
import yaml
import json
warnings.filterwarnings('ignore')

# 统一MPC_Test/results路径
ROOT_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

# 全局MPC优化debug开关
MPC_OPT_DEBUG = False

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML_Framework')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML_Framework/models')))
try:
    from nsga3 import NSGA3
except ImportError:
    NSGA3 = None
try:
    from algorithms.moo.i_nsga2 import iNSGA2
    INSGA2_AVAILABLE = True
except ImportError:
    INSGA2_AVAILABLE = False

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'moo_optimization_config.yaml'))

class PnModel:
    def __init__(self, model_path):
        # 统一模型路径为绝对路径，基于本文件目录
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
        print(f"[PnModel] Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model'] if 'model' in model_data else model_data
        self.scaler = model_data['metadata']['scaler'] if 'metadata' in model_data and 'scaler' in model_data['metadata'] else None
        self.feature_names = ['PPFD', 'CO2', 'T', 'R:B']
    def predict(self, ppfd, co2, temp, rb_ratio):
        input_df = pd.DataFrame([[ppfd, co2, temp, rb_ratio]], columns=self.feature_names)
        if self.scaler is not None:
            input_scaled = self.scaler.transform(input_df).values
        else:
            input_scaled = input_df.values
        return self.model.predict(input_scaled)[0]

class CLEDCalculator:
    def __init__(self, cled_config):
        self.cled_params = cled_config
        self.method = cled_config.get('calculation_method', 'standard')
        if self.method == 'standard':
            params = cled_config['standard']
            print(f"[CLED Params] Method: standard, Carbon factor: {params['carbon_factor']} kg CO₂/MWh, System efficiency: {params['system_efficiency']*100:.1f}%")
        else:
            params = cled_config['detailed']
            eff = params['efficiency']
            system_eff = eff['driver'] * eff['thermal'] * eff['optical']
            print(f"[CLED Params] Method: detailed, Carbon factor: {params['carbon_factor']} kg CO₂/MWh, System efficiency: {system_eff*100:.1f}%")
    def calc(self, ppfd, rb_ratio):
        if ppfd <= 0:
            return 0.0
        method = self.method
        params = self.cled_params[method]
        red_ppfd = ppfd * rb_ratio
        blue_ppfd = ppfd * (1 - rb_ratio)
        Ca = params['carbon_factor']
        S = params['surface_area']
        conversion_factor = params['conversion_factor']
        red_efficiency = params['led_efficiency']['red']
        blue_efficiency = params['led_efficiency']['blue']
        if method == 'standard':
            system_efficiency = params['system_efficiency']
        else:
            eff = params['efficiency']
            system_efficiency = eff['driver'] * eff['thermal'] * eff['optical']
        Ca_g_per_kwh = Ca * 0.001
        if red_ppfd > 0:
            red_power_density = red_ppfd / red_efficiency
            red_cl_density = red_power_density * Ca_g_per_kwh / 1000
        else:
            red_cl_density = 0.0
        if blue_ppfd > 0:
            blue_power_density = blue_ppfd / blue_efficiency
            blue_cl_density = blue_power_density * Ca_g_per_kwh / 1000
        else:
            blue_cl_density = 0.0
        total_cl_density = (red_cl_density + blue_cl_density) / system_efficiency
        cled = total_cl_density * 1000 / conversion_factor
        return cled

class MultiObjectiveOptimizer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.debug_mode = self.config.get('debug_mode', True)
        model_path = self.config['model']['model_path']
        self.pn_model = PnModel(model_path)
        self.cled_calculator = CLEDCalculator(self.config['cled'])
        self.variables = self.config['problem']['variables']
        self.generations = self.config['algorithm']['n_generations']
        self.pop_size = self.config['algorithm']['population_size']
        self.algorithm_type = self.config['algorithm']['algorithm_type']
        # 新增：读取随机种子配置
        experiment_cfg = self.config.get('experiment', {})
        self.use_random_seed = experiment_cfg.get('use_random_seed', True)
        self.random_seed = experiment_cfg.get('random_seed', 42)
    def _create_problem(self, co2, temp):
        pn_model = self.pn_model
        cled_calculator = self.cled_calculator
        ppfd_min = self.variables['ppfd']['min']
        ppfd_max = self.variables['ppfd']['max']
        rb_min = self.variables['rb_ratio']['min']
        rb_max = self.variables['rb_ratio']['max']
        class _Problem(Problem):
            def __init__(self):
                super().__init__(n_var=2, n_obj=2, n_constr=0,
                                 xl=np.array([ppfd_min, rb_min]),
                                 xu=np.array([ppfd_max, rb_max]))
            def _evaluate(self, X, out, *args, **kwargs):
                N = X.shape[0]
                cled_values = np.zeros(N)
                pn_values = np.zeros(N)
                for i in range(N):
                    ppfd, rb_ratio = X[i, 0], X[i, 1]
                    cled = cled_calculator.calc(ppfd, rb_ratio)
                    cled_values[i] = cled
                    pn_pred = pn_model.predict(ppfd, co2, temp, rb_ratio)
                    pn_values[i] = pn_pred
                out["F"] = np.column_stack([cled_values, -pn_values])
                out["cled_raw"] = cled_values
                out["pn_raw"] = pn_values
        return _Problem()
    def _create_algorithm(self):
        algo_config = self.config['algorithm']
        algorithm_type = algo_config['algorithm_type']
        pop_size = algo_config['population_size']
        if algorithm_type == "NSGA2":
            crossover = SBX(prob=algo_config['nsga2']['crossover']['prob'],
                            eta=algo_config['nsga2']['crossover']['eta'])
            mutation = PM(prob=algo_config['nsga2']['mutation']['prob'],
                          eta=algo_config['nsga2']['mutation']['eta'])
            return NSGA2(pop_size=pop_size, crossover=crossover, mutation=mutation,
                         eliminate_duplicates=algo_config.get('eliminate_duplicates', True))
        elif algorithm_type == "NSGA3" and NSGA3 is not None:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
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
        elif algorithm_type == "INSGA2" and INSGA2_AVAILABLE:
            params = algo_config.get('insga2', {})
            def get_crossover(cross_config):
                if cross_config.get('type', 'SBX') == 'SBX':
                    return SBX(prob=cross_config.get('prob', 0.9), eta=cross_config.get('eta', 20))
                else:
                    raise ValueError(f"Unsupported crossover operator: {cross_config.get('type')}")
            def get_mutation(mut_config):
                if mut_config.get('type', 'PM') == 'PM':
                    return PM(prob=mut_config.get('prob', 0.5), eta=mut_config.get('eta', 20))
                else:
                    raise ValueError(f"Unsupported mutation operator: {mut_config.get('type')}")
            crossover = get_crossover(params.get('crossover', algo_config.get('crossover', {})))
            mutation = get_mutation(params.get('mutation', algo_config.get('mutation', {})))
            return iNSGA2(
                pop_size=pop_size,
                crossover=crossover,
                mutation=mutation
            )
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
    def optimize(self, co2, temp, force_debug=None):
        print(f"[DEBUG] optimize called with co2={co2}, temp={temp}")
        # 原有的随机种子设置注释掉
        # import random
        # np.random.seed(42)
        # random.seed(42)
        # 新增：根据配置决定是否设置随机种子
        if self.use_random_seed:
            import random
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        print("🌱 Optimal cultivation condition finder (PPFD & R:B, fixed CO2 and temperature)")
        print(f"Optimization conditions: CO2={co2} ppm, T={temp} °C")
        problem = self._create_problem(co2, temp)
        algorithm = self._create_algorithm()
        print(f"Optimization algorithm: {self.algorithm_type}, generations: {self.generations}, population: {self.pop_size}")
        result = minimize(problem, algorithm, ('n_gen', self.generations), verbose=True)
        X = result.X
        F = result.F
        cled_values = F[:, 0]
        pn_values = -F[:, 1]
        df = pd.DataFrame({
            'PPFD': X[:, 0],
            'R:B': X[:, 1],
            'CLED': cled_values,
            'Pn': pn_values
        })
        best_idx = np.argmax(pn_values - cled_values/np.max(cled_values))
        print("🌟 Recommended solution (Pn and CLED weighted max):")
        print(df.iloc[best_idx])
        # debug_mode为true时所有图片和csv直接保存到results/
        # 只要MPC_OPT_DEBUG为True就生成图片和pareto_solutions.csv
        if MPC_OPT_DEBUG:
            # 新建唯一子目录，优先放在batch_dir下
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            parent_dir = self.batch_dir if self.batch_dir else ROOT_RESULTS_DIR
            subdir = os.path.join(parent_dir, timestamp)
            os.makedirs(subdir, exist_ok=True)
            csv_path = os.path.join(subdir, "pareto_solutions.csv")
            front_path = os.path.join(subdir, "pareto_front.png")
            surface_path = os.path.join(subdir, f"pareto_surface_co2_{co2}_temp_{temp}.png")
            # 保存csv
            df.to_csv(csv_path, index=False)
            # Pareto前沿图
            plt.figure()
            y_col = 'Cost' if 'Cost' in df.columns else ('CLED' if 'CLED' in df.columns else df.columns[1])
            plt.plot(df['Pn'], df[y_col], 'o')
            plt.xlabel('Pn')
            plt.ylabel(y_col)
            plt.title('Pareto Front')
            plt.tight_layout()
            plt.savefig(front_path)
            plt.close()
            # 3D表面图
            if 'CO2' in df.columns and 'Temp' in df.columns:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df['CO2'], df['Temp'], df['PPFD'], c=df[y_col], cmap='viridis')
                ax.set_xlabel('CO2 (ppm)')
                ax.set_ylabel('Temperature (°C)')
                ax.set_zlabel('PPFD (μmol/m²/s)')
                plt.tight_layout()
                plt.savefig(surface_path)
                plt.close()
            # 自动knee分析，生成pareto_fit.png和curvature_analysis.png
            try:
                analyzer = FitKneeAnalyzer(csv_path, debug_mode=True)
                fit_params = analyzer.analyze(output_dir=subdir, co2=co2, temp=temp, ppfd_range=(self.variables['ppfd']['min'], self.variables['ppfd']['max']), rb_range=(self.variables['rb_ratio']['min'], self.variables['rb_ratio']['max']))
                # 只保留fit_knee_parameters.json，移除best_solution.json和knee_point.json
            except Exception as e:
                print(f"[Warning] Knee analysis failed: {e}")
            # 返回时带上本次结果子文件夹路径
            return {'result_dir': subdir, 'fit_knee_parameters': fit_params}
        # 非debug模式返回原结构
        return df

class FitKneeAnalyzer:
    """
    用于对pareto_solutions.csv进行5阶多项式拟合和knee点分析。
    """
    def __init__(self, csv_path, debug_mode=True):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.debug_mode = debug_mode
    def minmax_normalize(self, arr):
        return (arr - np.min(arr)) / (np.max(arr))
    def fifth_order_poly(self, x, c0, c1, c2, c3, c4, c5):
        return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
    def curvature(self, x, y):
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        return d2y / (1 + dy**2)**1.5
    def analyze(self, output_dir=None, co2=None, temp=None, ppfd_range=None, rb_range=None):
        f1 = self.minmax_normalize(self.df['Pn'].values)
        g1 = self.minmax_normalize(self.df['CLED'].values)
        u = f1
        h = g1
        idx = np.argsort(u)
        u_sorted = u[idx]
        h_sorted = h[idx]
        p0 = [h_sorted.min(), 0, 0, 0, 0, 0]
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(self.fifth_order_poly, u_sorted, h_sorted, p0=p0, maxfev=10000)
        u_fit = np.linspace(u_sorted.min(), u_sorted.max(), 2000)
        h_fit = self.fifth_order_poly(u_fit, *popt)
        curv = self.curvature(u_fit, h_fit)
        main_mask = (u_fit >= 0.2) & (u_fit <= 1.0)
        u_main = u_fit[main_mask]
        h_main = h_fit[main_mask]
        curv_main = curv[main_mask]
        knee_idx = np.argmax(curv_main)
        knee_u = u_main[knee_idx]
        knee_h = h_main[knee_idx]
        knee_curv = curv_main[knee_idx]
        orig_idx = np.argmin((u - knee_u)**2 + (h - knee_h)**2)
        orig_row = self.df.iloc[orig_idx]
        import os, json
        if output_dir is None:
            from datetime import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(ROOT_RESULTS_DIR, f"pareto_knee_analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        fit_params = {
            'poly_coeffs': [float(x) for x in popt],
            'poly_formula': f"h(u) = {popt[0]:.6f} + {popt[1]:.6f}u + {popt[2]:.6f}u^2 + {popt[3]:.6f}u^3 + {popt[4]:.6f}u^4 + {popt[5]:.6f}u^5",
            'context': {
                'co2': co2,
                'temp': temp,
                'ppfd_range': ppfd_range,
                'rb_range': rb_range
            },
            'knee_point': {
                'u': float(knee_u),
                'h': float(knee_h),
                'curvature': float(knee_curv),
                'orig_idx': int(orig_idx),
                'PPFD': float(orig_row['PPFD']) if 'PPFD' in orig_row else None,
                'R:B': float(orig_row['R:B']) if 'R:B' in orig_row else None,
                'CLED': float(orig_row['CLED']),
                'Pn': float(orig_row['Pn'])
            }
        }
        with open(os.path.join(output_dir, 'fit_knee_parameters.json'), 'w', encoding='utf-8') as f:
            json.dump(fit_params, f, indent=2, ensure_ascii=False)
        pd.DataFrame({'u': u_main, 'h_fit': h_main, 'curvature': curv_main}).to_csv(os.path.join(output_dir, 'fitted_curve_and_curvature.csv'), index=False)
        if self.debug_mode:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12,5))
            plt.scatter(u, h, c='b', label='Pareto Solutions')
            plt.plot(u_fit, h_fit, 'r-', label='5th Order Polynomial Fit')
            plt.scatter(knee_u, knee_h, c='red', s=120, marker='*', label='Knee Point')
            plt.xlim(0.2, 1.0)
            plt.xlabel('Normalized Photosynthetic Rate Objective $f_1$')
            plt.ylabel('Normalized Cost Objective $g_1$')
            plt.title('Normalized Pareto Front and 5th Order Polynomial Fit')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pareto_fit.png'), dpi=300)
            plt.close()
            plt.figure(figsize=(8,5))
            plt.plot(u_main, curv_main, 'g-', label='Curvature')
            plt.scatter(knee_u, knee_curv, c='red', s=120, marker='*', label='Knee Point')
            plt.xlim(0.2, 1.0)
            plt.xlabel('Normalized Photosynthetic Rate Objective $f_1$')
            plt.ylabel('Curvature')
            plt.title('Curvature Analysis of 5th Order Polynomial Fit')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'curvature_analysis.png'), dpi=300)
            plt.close()
            print('\n===== Knee Point (Normalized) =====')
            print(f'u* = {knee_u:.4f}, h* = {knee_h:.4f}, curvature = {knee_curv:.6f}')
            print('===== Knee Point (Original Data) =====')
            print(f'PPFD = {orig_row["PPFD"]:.2f}, R:B = {orig_row["R:B"]:.4f}, CLED = {orig_row["CLED"]:.2f}, Pn = {orig_row["Pn"]:.4f}')
            print(f'All results saved to: {output_dir}')
        else:
            print("[Debug mode off] Knee analysis images not generated, only fit parameters and data saved.")
        return fit_params

def find_best_ppfd(co2, temp, debug_mode=False, batch_dir=None):
    """
    非交互式接口：输入CO2和温度，返回最佳ppfd、r:b、pn、cled等。
    debug_mode: 可选，覆盖yaml设置。
    output_dir: 保留参数但不再使用。
    save_csv: 是否保存csv结果。
    csv_path: 结果csv文件路径。
    返回: dict，包括最佳ppfd、r:b、pn、cled等。
    """
    global MPC_OPT_DEBUG
    if debug_mode:
        MPC_OPT_DEBUG = True
    else:
        MPC_OPT_DEBUG = False
    optimizer = MultiObjectiveOptimizer(CONFIG_PATH)
    if debug_mode is not None:
        optimizer.debug_mode = debug_mode
    if batch_dir is not None:
        optimizer.batch_dir = batch_dir
    result = optimizer.optimize(co2, temp)
    if isinstance(result, dict) and 'result_dir' in result:
        # debug模式，返回包含knee点的子文件夹路径
        return result
    # 非debug模式只返回None或空dict即可
    return {}

def main():
    co2 = float(input("Please enter CO2 concentration (ppm, e.g., 400): "))
    temp = float(input("Please enter temperature (°C, e.g., 24): "))
    result = find_best_ppfd(co2, temp, save_csv=True)
    print("Optimal solution and input/output appended to csv:", result.get('csv_path', ''))
    import pprint
    pprint.pprint(result)

if __name__ == "__main__":
    main() 