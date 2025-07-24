"""
遗传算法优化器
用于超参数优化的遗传算法实现
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from .base_optimizer import BaseOptimizer


@dataclass
class Individual:
    """个体类"""
    genes: Dict[str, float]  # 基因（参数）
    fitness: float = 0.0     # 适应度


class GeneticAlgorithm(BaseOptimizer):
    """
    遗传算法优化器
    
    使用遗传算法进行超参数优化
    """
    
    def __init__(self, 
                 objective_function,
                 param_bounds: Dict[str, Tuple[float, float]],
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_rate: float = 0.1,
                 tournament_size: int = 3,
                 **kwargs):
        """
        初始化遗传算法
        
        Args:
            objective_function: 目标函数
            param_bounds: 参数边界
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_rate: 精英比例
            tournament_size: 锦标赛大小
        """
        super().__init__(objective_function, param_bounds, **kwargs)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_rate = elite_rate
        self.tournament_size = tournament_size
        
        # 存储每代历史
        self.generation_history = []
        self.population = []
    
    def optimize(self, model_class, X_train, y_train, n_trials: int = 100, **kwargs) -> Dict[str, Any]:
        """
        统一接口：执行遗传算法优化
        Args:
            model_class: 模型类
            X_train: 训练特征
            y_train: 训练目标
            n_trials: 最大代数
        Returns:
            优化结果
        """
        max_generations = n_trials
        
        print(f"开始遗传算法优化...")
        print(f"种群大小: {self.population_size}, 最大代数: {max_generations}")
        print(f"变异率: {self.mutation_rate}, 交叉率: {self.crossover_rate}")
        
        self._initialize_population()
        self._evaluate_population_with_model_class(model_class, X_train, y_train)
        
        # 创建进度条
        pbar = tqdm(total=max_generations, desc="遗传算法优化", ncols=100)
        
        for generation in range(max_generations):
            selected = self._selection()
            offspring = self._crossover(selected)
            self._mutation(offspring)
            self.population = offspring
            self._evaluate_population_with_model_class(model_class, X_train, y_train)
            best_individual = self._get_best_individual()
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            
            self.generation_history.append({
                'generation': generation,
                'best_fitness': best_individual.fitness,
                'avg_fitness': avg_fitness,
                'best_params': best_individual.genes.copy()
            })
            self._record_iteration(generation, best_individual.genes, best_individual.fitness)
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'best_fitness': f'{best_individual.fitness:.4f}',
                'avg_fitness': f'{avg_fitness:.4f}'
            })
        
        # 关闭进度条
        pbar.close()
        
        print(f"\n遗传算法优化完成！")
        print(f"最佳适应度: {self.best_score:.6f}")
        print(f"最佳参数: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_evaluations': len(self.history),
            'history': self.generation_history
        }
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            genes = self._generate_random_params()
            individual = Individual(genes=genes)
            self.population.append(individual)
    
    def _evaluate_population(self, X: np.ndarray, y: np.ndarray):
        """评估种群中所有个体"""
        for individual in self.population:
            individual.fitness = self._evaluate_params(individual.genes, X, y)
    
    def _selection(self) -> List[Individual]:
        """选择操作（锦标赛选择）"""
        selected = []
        
        # 精英保留
        elite_count = int(self.population_size * self.elite_rate)
        if elite_count > 0:
            # 按适应度排序
            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            selected.extend(sorted_population[:elite_count])
        
        # 锦标赛选择填充剩余位置
        while len(selected) < self.population_size:
            tournament = np.random.choice(self.population, self.tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            # 创建副本
            selected.append(Individual(genes=winner.genes.copy(), fitness=winner.fitness))
        
        return selected
    
    def _crossover(self, parents: List[Individual]) -> List[Individual]:
        """交叉操作（单点交叉）"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            if np.random.random() < self.crossover_rate:
                # 执行交叉
                child1_genes = {}
                child2_genes = {}
                
                param_names = list(self.param_bounds.keys())
                crossover_point = np.random.randint(1, len(param_names))
                
                for j, param_name in enumerate(param_names):
                    if j < crossover_point:
                        child1_genes[param_name] = parent1.genes[param_name]
                        child2_genes[param_name] = parent2.genes[param_name]
                    else:
                        child1_genes[param_name] = parent2.genes[param_name]
                        child2_genes[param_name] = parent1.genes[param_name]
                
                offspring.append(Individual(genes=child1_genes))
                offspring.append(Individual(genes=child2_genes))
            else:
                # 不交叉，直接复制
                offspring.append(Individual(genes=parent1.genes.copy()))
                offspring.append(Individual(genes=parent2.genes.copy()))
        
        # 确保数量正确
        return offspring[:self.population_size]
    
    def _mutation(self, individuals: List[Individual]):
        """变异操作（高斯变异）"""
        for individual in individuals:
            for param_name in individual.genes:
                if np.random.random() < self.mutation_rate:
                    # 高斯变异
                    bounds = self.param_bounds[param_name]
                    min_val = float(bounds[0]) if isinstance(bounds, (list, tuple)) else float(bounds[0])
                    max_val = float(bounds[1]) if isinstance(bounds, (list, tuple)) else float(bounds[1])
                    range_val = max_val - min_val
                    
                    # 变异强度为参数范围的10%
                    mutation_strength = range_val * 0.1
                    mutation = np.random.normal(0, mutation_strength)
                    
                    # 应用变异并限制在边界内
                    new_value = individual.genes[param_name] + mutation
                    individual.genes[param_name] = np.clip(new_value, min_val, max_val)
    
    def _evaluate_population_with_model_class(self, model_class, X_train, y_train):
        """使用模型类评估种群中所有个体"""
        from sklearn.model_selection import cross_val_score
        
        for individual in self.population:
            try:
                # 创建模型并评估
                model = model_class(**individual.genes)
                scores = cross_val_score(
                    model, X_train, y_train,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs
                )
                individual.fitness = np.mean(scores)
            except Exception as e:
                if self.verbose > 0:
                    print(f"参数评估失败: {individual.genes}, 错误: {str(e)}")
                # 返回很差的分数
                individual.fitness = -np.inf if 'neg_' in self.scoring else np.inf
    
    def _get_best_individual(self) -> Individual:
        """获取最佳个体"""
        return max(self.population, key=lambda ind: ind.fitness)
    
    def plot_generation_history(self, save_path: str = None):
        """绘制代数历史"""
        import matplotlib.pyplot as plt
        
        if not self.generation_history:
            print("No generation history to plot")
            return
        
        generations = [h['generation'] for h in self.generation_history]
        best_fitness = [h['best_fitness'] for h in self.generation_history]
        avg_fitness = [h['avg_fitness'] for h in self.generation_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(generations, best_fitness, 'r-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Evolution History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_population_diversity(self) -> float:
        """计算种群多样性"""
        if not self.population:
            return 0.0
        
        # 计算所有个体之间的平均距离
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = 0
                for param_name in self.param_bounds:
                    val1 = self.population[i].genes[param_name]
                    val2 = self.population[j].genes[param_name]
                    bounds = self.param_bounds[param_name]
                    min_val = float(bounds[0]) if isinstance(bounds, (list, tuple)) else float(bounds[0])
                    max_val = float(bounds[1]) if isinstance(bounds, (list, tuple)) else float(bounds[1])
                    # 标准化距离
                    normalized_distance = abs(val1 - val2) / (max_val - min_val)
                    distance += normalized_distance ** 2
                distances.append(np.sqrt(distance))
        
        return np.mean(distances) if distances else 0.0 