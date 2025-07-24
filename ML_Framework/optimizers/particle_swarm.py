"""
粒子群优化器
用于超参数优化的粒子群算法实现
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from .base_optimizer import BaseOptimizer


@dataclass
class Particle:
    """粒子类"""
    position: Dict[str, float]      # 当前位置（参数值）
    velocity: Dict[str, float]      # 速度
    best_position: Dict[str, float] # 个体最佳位置
    fitness: float = -np.inf        # 当前适应度
    best_fitness: float = -np.inf   # 个体最佳适应度


class ParticleSwarmOptimization(BaseOptimizer):
    """
    粒子群优化器
    
    使用粒子群算法进行超参数优化
    """
    
    def __init__(self, 
                 objective_function,
                 param_bounds: Dict[str, Tuple[float, float]],
                 swarm_size: int = 30,
                 w: float = 0.9,        # 惯性权重
                 c1: float = 2.0,       # 个体学习因子
                 c2: float = 2.0,       # 社会学习因子
                 w_min: float = 0.4,    # 最小惯性权重
                 w_max: float = 0.9,    # 最大惯性权重
                 max_velocity_ratio: float = 0.2,  # 最大速度比例
                 **kwargs):
        """
        初始化粒子群优化器
        
        Args:
            objective_function: 目标函数
            param_bounds: 参数边界
            swarm_size: 粒子群大小
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            w_min: 最小惯性权重
            w_max: 最大惯性权重
            max_velocity_ratio: 最大速度比例
        """
        super().__init__(objective_function, param_bounds, **kwargs)
        
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        self.max_velocity_ratio = max_velocity_ratio
        
        # 粒子群和全局最佳
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # 存储迭代历史
        self.iteration_history = []
        
        # 计算速度限制
        self.velocity_limits = {}
        for param_name, bounds in param_bounds.items():
            min_val = float(bounds[0]) if isinstance(bounds, (list, tuple)) else float(bounds[0])
            max_val = float(bounds[1]) if isinstance(bounds, (list, tuple)) else float(bounds[1])
            max_velocity = (max_val - min_val) * max_velocity_ratio
            self.velocity_limits[param_name] = max_velocity
    
    def optimize(self, model_class, X_train, y_train, n_trials: int = 100, **kwargs) -> Dict[str, Any]:
        """
        统一接口：执行粒子群优化
        Args:
            model_class: 模型类
            X_train: 训练特征
            y_train: 训练目标
            n_trials: 最大迭代次数
        Returns:
            优化结果
        """
        max_iterations = n_trials
        
        print(f"开始粒子群优化...")
        print(f"粒子群大小: {self.swarm_size}, 最大迭代次数: {max_iterations}")
        print(f"惯性权重: {self.w_min}-{self.w_max}, 学习因子: c1={self.c1}, c2={self.c2}")
        
        self._initialize_swarm()
        self._evaluate_swarm_with_model_class(model_class, X_train, y_train)
        self._update_global_best()
        
        # 创建进度条
        pbar = tqdm(total=max_iterations, desc="粒子群优化", ncols=100)
        
        for iteration in range(max_iterations):
            current_w = self.w_max - (self.w_max - self.w_min) * iteration / max_iterations
            for particle in self.swarm:
                self._update_velocity(particle, current_w)
                self._update_position(particle)
            self._evaluate_swarm_with_model_class(model_class, X_train, y_train)
            self._update_personal_best()
            self._update_global_best()
            avg_fitness = np.mean([p.fitness for p in self.swarm])
            
            self.iteration_history.append({
                'iteration': iteration,
                'global_best_fitness': self.global_best_fitness,
                'avg_fitness': avg_fitness,
                'inertia_weight': current_w,
                'global_best_params': self.global_best_position.copy()
            })
            self._record_iteration(iteration, self.global_best_position, self.global_best_fitness)
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'best_fitness': f'{self.global_best_fitness:.4f}',
                'avg_fitness': f'{avg_fitness:.4f}',
                'inertia': f'{current_w:.2f}'
            })
        
        # 关闭进度条
        pbar.close()
        
        print(f"\n粒子群优化完成！")
        print(f"全局最佳适应度: {self.best_score:.6f}")
        print(f"最佳参数: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_evaluations': len(self.history),
            'history': self.iteration_history
        }
    
    def _initialize_swarm(self):
        """初始化粒子群"""
        self.swarm = []
        
        for _ in range(self.swarm_size):
            # 随机生成位置
            position = self._generate_random_params()
            
            # 随机生成初始速度
            velocity = {}
            for param_name in self.param_bounds:
                max_vel = self.velocity_limits[param_name]
                velocity[param_name] = np.random.uniform(-max_vel, max_vel)
            
            # 创建粒子
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy()
            )
            
            self.swarm.append(particle)
    
    def _evaluate_swarm(self, X: np.ndarray, y: np.ndarray):
        """评估粒子群"""
        for particle in self.swarm:
            particle.fitness = self._evaluate_params(particle.position, X, y)
    
    def _evaluate_swarm_with_model_class(self, model_class, X_train, y_train):
        """使用模型类评估粒子群"""
        from sklearn.model_selection import cross_val_score
        
        for particle in self.swarm:
            try:
                # 创建模型并评估
                model = model_class(**particle.position)
                scores = cross_val_score(
                    model, X_train, y_train,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs
                )
                particle.fitness = np.mean(scores)
            except Exception as e:
                if self.verbose > 0:
                    print(f"参数评估失败: {particle.position}, 错误: {str(e)}")
                # 返回很差的分数
                particle.fitness = -np.inf if 'neg_' in self.scoring else np.inf
    
    def _update_velocity(self, particle: Particle, w: float):
        """更新粒子速度"""
        for param_name in self.param_bounds:
            # PSO速度更新公式
            r1, r2 = np.random.random(), np.random.random()
            
            # 惯性项
            inertia = w * particle.velocity[param_name]
            
            # 个体认知项
            cognitive = self.c1 * r1 * (particle.best_position[param_name] - particle.position[param_name])
            
            # 社会学习项
            social = self.c2 * r2 * (self.global_best_position[param_name] - particle.position[param_name])
            
            # 新速度
            new_velocity = inertia + cognitive + social
            
            # 限制速度
            max_vel = self.velocity_limits[param_name]
            particle.velocity[param_name] = np.clip(new_velocity, -max_vel, max_vel)
    
    def _update_position(self, particle: Particle):
        """更新粒子位置"""
        for param_name in self.param_bounds:
            # 更新位置
            new_position = particle.position[param_name] + particle.velocity[param_name]
            
            # 边界处理
            bounds = self.param_bounds[param_name]
            min_val = float(bounds[0]) if isinstance(bounds, (list, tuple)) else float(bounds[0])
            max_val = float(bounds[1]) if isinstance(bounds, (list, tuple)) else float(bounds[1])
            
            if new_position < min_val:
                new_position = min_val
                particle.velocity[param_name] = 0  # 碰撞后速度置零
            elif new_position > max_val:
                new_position = max_val
                particle.velocity[param_name] = 0  # 碰撞后速度置零
            
            particle.position[param_name] = new_position
    
    def _update_personal_best(self):
        """更新个体最佳"""
        for particle in self.swarm:
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
    
    def _update_global_best(self):
        """更新全局最佳"""
        for particle in self.swarm:
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
    
    def plot_iteration_history(self, save_path: str = None):
        """绘制迭代历史"""
        import matplotlib.pyplot as plt
        
        if not self.iteration_history:
            print("No iteration history to plot")
            return
        
        iterations = [h['iteration'] for h in self.iteration_history]
        global_best = [h['global_best_fitness'] for h in self.iteration_history]
        avg_fitness = [h['avg_fitness'] for h in self.iteration_history]
        inertia_weights = [h['inertia_weight'] for h in self.iteration_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 适应度曲线
        ax1.plot(iterations, global_best, 'r-', label='Global Best', linewidth=2)
        ax1.plot(iterations, avg_fitness, 'b-', label='Average Fitness', alpha=0.7)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Particle Swarm Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 惯性权重曲线
        ax2.plot(iterations, inertia_weights, 'g-', label='Inertia Weight', linewidth=2)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Inertia Weight')
        ax2.set_title('Inertia Weight Change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_swarm_diversity(self) -> float:
        """计算粒子群多样性"""
        if not self.swarm:
            return 0.0
        
        # 计算所有粒子之间的平均距离
        distances = []
        for i in range(len(self.swarm)):
            for j in range(i + 1, len(self.swarm)):
                distance = 0
                for param_name in self.param_bounds:
                    val1 = self.swarm[i].position[param_name]
                    val2 = self.swarm[j].position[param_name]
                    min_val, max_val = self.param_bounds[param_name]
                    # 标准化距离
                    normalized_distance = abs(val1 - val2) / (max_val - min_val)
                    distance += normalized_distance ** 2
                distances.append(np.sqrt(distance))
        
        return np.mean(distances) if distances else 0.0
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """获取收敛信息"""
        if not self.iteration_history:
            return {}
        
        # 计算收敛速度
        best_scores = [h['global_best_fitness'] for h in self.iteration_history]
        
        # 找到收敛点（连续10次迭代改进小于阈值）
        convergence_iteration = len(best_scores)
        threshold = 1e-6
        
        for i in range(10, len(best_scores)):
            if all(abs(best_scores[i] - best_scores[i-j]) < threshold for j in range(1, 11)):
                convergence_iteration = i
                break
        
        return {
            'convergence_iteration': convergence_iteration,
            'final_best_score': best_scores[-1],
            'total_iterations': len(best_scores),
            'converged': convergence_iteration < len(best_scores)
        } 