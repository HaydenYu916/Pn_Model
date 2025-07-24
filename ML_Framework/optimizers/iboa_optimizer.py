"""
IBOA (Improved Butterfly Optimization Algorithm) 优化器
改进蝴蝶优化算法
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import cross_val_score
import logging
from tqdm import tqdm

from .base_optimizer import BaseOptimizer


class Butterfly:
    """蝴蝶个体类"""
    
    def __init__(self, position, fitness=None):
        self.position = np.array(position)
        self.fitness = fitness
        self.best_position = np.array(position)
        self.best_fitness = float('inf') if fitness is None else fitness


class IBOAOptimizer(BaseOptimizer):
    """改进蝴蝶优化算法优化器"""
    
    def __init__(self, param_bounds: Dict[str, List[float]], 
                 n_butterflies: int = 20,
                 n_iterations: int = 100,
                 sensory_modality: float = 0.01,
                 power_exponent: float = 0.1,
                 switch_probability: float = 0.8,
                 **kwargs):
        """
        初始化 IBOA 优化器
        
        Args:
            param_bounds: 参数边界字典
            n_butterflies: 蝴蝶数量
            n_iterations: 迭代次数
            sensory_modality: 感觉模态参数
            power_exponent: 幂指数参数
            switch_probability: 切换概率
        """
        super().__init__(param_bounds, **kwargs)
        self.n_butterflies = n_butterflies
        self.n_iterations = n_iterations
        self.sensory_modality = sensory_modality
        self.power_exponent = power_exponent
        self.switch_probability = switch_probability
        self._sanitize_param_bounds()
        
        # 获取参数名列表
        self.param_names = list(self.param_bounds.keys())
        self.n_params = len(self.param_names)
        
        # 初始化蝴蝶种群
        self.butterflies = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
    def _sanitize_param_bounds(self):
        """确保参数边界为数字类型"""
        for k, v in self.param_bounds.items():
            if isinstance(v, (list, tuple)):
                self.param_bounds[k] = [float(x) if (isinstance(x, float) or '.' in str(x) or 'e' in str(x)) else int(x) for x in v]
    
    def _initialize_butterflies(self):
        """初始化蝴蝶种群"""
        self.butterflies = []
        
        for _ in range(self.n_butterflies):
            # 随机生成蝴蝶位置
            position = []
            for param_name in self.param_names:
                bounds = self.param_bounds[param_name]
                if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                    # 整数参数
                    pos = random.randint(bounds[0], bounds[1])
                else:
                    # 浮点数参数
                    pos = random.uniform(bounds[0], bounds[1])
                position.append(pos)
            
            butterfly = Butterfly(position)
            self.butterflies.append(butterfly)
    
    def _evaluate_fitness(self, butterfly: Butterfly, model_class, X_train, y_train, cv_folds=5):
        """评估蝴蝶适应度"""
        try:
            # 构建参数字典
            params = {}
            for i, param_name in enumerate(self.param_names):
                param_value = butterfly.position[i]
                
                # 根据参数类型进行转换
                bounds = self.param_bounds[param_name]
                if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                    params[param_name] = int(param_value)
                else:
                    params[param_name] = float(param_value)
            
            # 创建模型并评估
            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error')
            fitness = -np.mean(scores)  # 转换为正数（越小越好）
            
            return fitness
            
        except Exception as e:
            logging.warning(f"参数评估失败: {params}, 错误: {e}")
            return float('inf')
    
    def _update_butterfly_position(self, butterfly: Butterfly, iteration: int):
        """更新蝴蝶位置"""
        # 计算感觉模态
        sensory_modality = self.sensory_modality * (0.025 / (1 + iteration))
        
        # 计算香味强度
        fragrance = sensory_modality * (butterfly.fitness ** self.power_exponent)
        
        # 随机数
        r = random.random()
        
        if r < self.switch_probability:
            # 全局搜索阶段
            for i in range(self.n_params):
                param_name = self.param_names[i]
                bounds = self.param_bounds[param_name]
                
                # 使用全局最优位置进行更新
                if self.global_best_position is not None:
                    butterfly.position[i] = self.global_best_position[i] + \
                                          fragrance * (random.random() - 0.5) * 2
                
                # 边界处理
                butterfly.position[i] = np.clip(butterfly.position[i], bounds[0], bounds[1])
        else:
            # 局部搜索阶段
            for i in range(self.n_params):
                param_name = self.param_names[i]
                bounds = self.param_bounds[param_name]
                
                # 随机游走
                butterfly.position[i] += fragrance * (random.random() - 0.5) * 2
                
                # 边界处理
                butterfly.position[i] = np.clip(butterfly.position[i], bounds[0], bounds[1])
    
    def optimize(self, model_class, X_train, y_train, n_trials: int = 100, **kwargs) -> Dict[str, Any]:
        """
        统一接口：执行IBOA优化
        Args:
            model_class: 模型类
            X_train: 训练特征
            y_train: 训练目标
            n_trials: 迭代次数
        Returns:
            优化结果字典
        """
        self.n_iterations = n_trials
        
        print(f"开始 IBOA 优化...")
        print(f"蝴蝶数量: {self.n_butterflies}, 迭代次数: {self.n_iterations}")
        print(f"感觉模态: {self.sensory_modality}, 幂指数: {self.power_exponent}")
        
        self._initialize_butterflies()
        
        # 初始化评估
        for butterfly in self.butterflies:
            butterfly.fitness = self._evaluate_fitness(butterfly, model_class, X_train, y_train)
            if butterfly.fitness < butterfly.best_fitness:
                butterfly.best_position = butterfly.position.copy()
                butterfly.best_fitness = butterfly.fitness
            if butterfly.fitness < self.global_best_fitness:
                self.global_best_position = butterfly.position.copy()
                self.global_best_fitness = butterfly.fitness
        
        optimization_history = []
        
        # 创建进度条
        pbar = tqdm(total=self.n_iterations, desc="IBOA优化", ncols=100)
        
        for iteration in range(self.n_iterations):
            for butterfly in self.butterflies:
                self._update_butterfly_position(butterfly, iteration)
                new_fitness = self._evaluate_fitness(butterfly, model_class, X_train, y_train)
                if new_fitness < butterfly.fitness:
                    butterfly.fitness = new_fitness
                    butterfly.position = butterfly.position.copy()
                    if butterfly.fitness < butterfly.best_fitness:
                        butterfly.best_position = butterfly.position.copy()
                        butterfly.best_fitness = butterfly.fitness
                    if butterfly.fitness < self.global_best_fitness:
                        self.global_best_position = butterfly.position.copy()
                        self.global_best_fitness = butterfly.fitness
            
            optimization_history.append(self.global_best_fitness)
            self._record_iteration(iteration, {name: self.global_best_position[i] for i, name in enumerate(self.param_names)}, self.global_best_fitness)
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'best_fitness': f'{self.global_best_fitness:.4f}'
            })
        
        # 关闭进度条
        pbar.close()
        
        print(f"\nIBOA优化完成！")
        print(f"全局最佳适应度: {self.global_best_fitness:.6f}")
        
        best_params = {name: self.global_best_position[i] for i, name in enumerate(self.param_names)}
        return {
            'best_params': best_params,
            'best_score': self.global_best_fitness,
            'n_evaluations': len(self.history),
            'history': optimization_history
        } 