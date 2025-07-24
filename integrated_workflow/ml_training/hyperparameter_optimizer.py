#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超参数优化组件
负责模型超参数的优化
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union, Type, Callable
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

from ML_Framework.models import BaseModel

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """超参数优化类，负责模型超参数的优化"""
    
    def __init__(self, config: Dict):
        """
        初始化超参数优化器
        
        Args:
            config (Dict): 配置字典，包含优化相关参数
        """
        self.config = config
        self.optimizer_type = config.get('optimizer_type', 'grid').upper()
        self.param_bounds = config.get('param_bounds', {})
        self.n_trials = config.get('n_trials', 50)
        self.cv = config.get('cv', 5)
        self.scoring = config.get('scoring', 'neg_mean_squared_error')
        self.random_state = config.get('random_state', 42)
        self.n_jobs = config.get('n_jobs', -1)
        self.output_dir = config.get('output_dir', 'results')
        
        # 优化结果
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"超参数优化器初始化完成，优化器类型: {self.optimizer_type}")
    
    def optimize(self, model_class: Type[BaseModel], X_train: np.ndarray, y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            model_class (Type[BaseModel]): 模型类
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证目标
            base_params (Dict[str, Any], optional): 基础参数
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        logger.info(f"开始{self.optimizer_type}超参数优化...")
        
        # 合并基础参数
        base_params = base_params or {}
        
        # 根据优化器类型选择优化方法
        if self.optimizer_type == 'GRID':
            return self._grid_search(model_class, X_train, y_train, X_val, y_val, base_params)
        elif self.optimizer_type == 'RANDOM':
            return self._random_search(model_class, X_train, y_train, X_val, y_val, base_params)
        elif self.optimizer_type == 'BAYESIAN':
            return self._bayesian_optimization(model_class, X_train, y_train, X_val, y_val, base_params)
        elif self.optimizer_type == 'GENETIC':
            return self._genetic_algorithm(model_class, X_train, y_train, X_val, y_val, base_params)
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")
    
    def _grid_search(self, model_class: Type[BaseModel], X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                    base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        网格搜索优化
        
        Args:
            model_class (Type[BaseModel]): 模型类
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证目标
            base_params (Dict[str, Any]): 基础参数
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info("执行网格搜索...")
        
        # 准备参数网格
        param_grid = {}
        for param_name, bounds in self.param_bounds.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                # 如果是连续参数，创建离散值列表
                if isinstance(bounds[0], float) or isinstance(bounds[1], float):
                    # 对于浮点数，创建对数均匀分布
                    param_grid[param_name] = np.logspace(
                        np.log10(bounds[0]), np.log10(bounds[1]), 
                        num=min(10, self.n_trials)
                    )
                else:
                    # 对于整数，创建线性均匀分布
                    param_grid[param_name] = np.linspace(
                        bounds[0], bounds[1], 
                        num=min(10, self.n_trials), dtype=int
                    )
            else:
                # 如果已经是列表，直接使用
                param_grid[param_name] = bounds
        
        # 创建基础模型
        base_model = model_class(**base_params)
        
        # 创建网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # 执行搜索
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数和分数
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # 记录优化历史
        self.optimization_history = []
        for i, params in enumerate(grid_search.cv_results_['params']):
            self.optimization_history.append({
                'iteration': i,
                'params': params,
                'score': grid_search.cv_results_['mean_test_score'][i]
            })
        
        logger.info(f"网格搜索完成，最佳分数: {self.best_score:.6f}")
        logger.info(f"最佳参数: {self.best_params}")
        
        # 构建结果
        result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'cv_results': grid_search.cv_results_
        }
        
        return result
    
    def _random_search(self, model_class: Type[BaseModel], X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                      base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        随机搜索优化
        
        Args:
            model_class (Type[BaseModel]): 模型类
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证目标
            base_params (Dict[str, Any]): 基础参数
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, loguniform, randint
        
        logger.info("执行随机搜索...")
        
        # 准备参数分布
        param_distributions = {}
        for param_name, bounds in self.param_bounds.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                # 如果是连续参数，创建分布
                if isinstance(bounds[0], float) or isinstance(bounds[1], float):
                    # 对于浮点数，使用对数均匀分布
                    param_distributions[param_name] = loguniform(bounds[0], bounds[1])
                else:
                    # 对于整数，使用随机整数分布
                    param_distributions[param_name] = randint(bounds[0], bounds[1] + 1)
            else:
                # 如果已经是列表，直接使用
                param_distributions[param_name] = bounds
        
        # 创建基础模型
        base_model = model_class(**base_params)
        
        # 创建随机搜索
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=self.n_trials,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_state,
            return_train_score=True
        )
        
        # 执行搜索
        random_search.fit(X_train, y_train)
        
        # 获取最佳参数和分数
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        # 记录优化历史
        self.optimization_history = []
        for i, params in enumerate(random_search.cv_results_['params']):
            self.optimization_history.append({
                'iteration': i,
                'params': params,
                'score': random_search.cv_results_['mean_test_score'][i]
            })
        
        logger.info(f"随机搜索完成，最佳分数: {self.best_score:.6f}")
        logger.info(f"最佳参数: {self.best_params}")
        
        # 构建结果
        result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'cv_results': random_search.cv_results_
        }
        
        return result
    
    def _bayesian_optimization(self, model_class: Type[BaseModel], X_train: np.ndarray, y_train: np.ndarray,
                             X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                             base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        贝叶斯优化
        
        Args:
            model_class (Type[BaseModel]): 模型类
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证目标
            base_params (Dict[str, Any]): 基础参数
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            import optuna
        except ImportError:
            logger.error("贝叶斯优化需要安装optuna库，请使用pip install optuna安装")
            raise
        
        logger.info("执行贝叶斯优化...")
        
        # 定义目标函数
        def objective(trial):
            # 为每个参数创建建议值
            params = base_params.copy()
            for param_name, bounds in self.param_bounds.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    # 如果是连续参数
                    if isinstance(bounds[0], float) or isinstance(bounds[1], float):
                        # 对于浮点数，使用对数均匀分布
                        params[param_name] = trial.suggest_float(
                            param_name, bounds[0], bounds[1], log=True
                        )
                    else:
                        # 对于整数，使用整数分布
                        params[param_name] = trial.suggest_int(
                            param_name, bounds[0], bounds[1]
                        )
                elif isinstance(bounds, list):
                    # 如果是分类参数
                    params[param_name] = trial.suggest_categorical(param_name, bounds)
            
            # 创建并训练模型
            model = model_class(**params)
            
            # 如果提供了验证集，使用验证集评估
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # 计算分数
                if self.scoring == 'neg_mean_squared_error':
                    from sklearn.metrics import mean_squared_error
                    score = -mean_squared_error(y_val, y_pred)
                elif self.scoring == 'r2':
                    from sklearn.metrics import r2_score
                    score = r2_score(y_val, y_pred)
                else:
                    # 默认使用交叉验证
                    scores = cross_val_score(
                        model, X_train, y_train,
                        scoring=self.scoring,
                        cv=self.cv,
                        n_jobs=self.n_jobs
                    )
                    score = np.mean(scores)
            else:
                # 使用交叉验证
                scores = cross_val_score(
                    model, X_train, y_train,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs
                )
                score = np.mean(scores)
            
            return score
        
        # 创建学习器
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 执行优化
        study.optimize(objective, n_trials=self.n_trials)
        
        # 获取最佳参数和分数
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # 记录优化历史
        self.optimization_history = []
        for i, trial in enumerate(study.trials):
            self.optimization_history.append({
                'iteration': i,
                'params': trial.params,
                'score': trial.value
            })
        
        logger.info(f"贝叶斯优化完成，最佳分数: {self.best_score:.6f}")
        logger.info(f"最佳参数: {self.best_params}")
        
        # 构建结果
        result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'study': study
        }
        
        return result
    
    def _genetic_algorithm(self, model_class: Type[BaseModel], X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                         base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        遗传算法优化
        
        Args:
            model_class (Type[BaseModel]): 模型类
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证目标
            base_params (Dict[str, Any]): 基础参数
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            from ML_Framework.optimizers import GeneticAlgorithm
        except ImportError:
            logger.error("遗传算法优化需要ML_Framework.optimizers.GeneticAlgorithm")
            raise
        
        logger.info("执行遗传算法优化...")
        
        # 定义目标函数
        def objective_function(params):
            # 合并参数
            full_params = base_params.copy()
            full_params.update(params)
            
            # 创建并训练模型
            model = model_class(**full_params)
            
            # 如果提供了验证集，使用验证集评估
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # 计算分数
                if self.scoring == 'neg_mean_squared_error':
                    from sklearn.metrics import mean_squared_error
                    score = -mean_squared_error(y_val, y_pred)
                elif self.scoring == 'r2':
                    from sklearn.metrics import r2_score
                    score = r2_score(y_val, y_pred)
                else:
                    # 默认使用交叉验证
                    scores = cross_val_score(
                        model, X_train, y_train,
                        scoring=self.scoring,
                        cv=self.cv,
                        n_jobs=self.n_jobs
                    )
                    score = np.mean(scores)
            else:
                # 使用交叉验证
                scores = cross_val_score(
                    model, X_train, y_train,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs
                )
                score = np.mean(scores)
            
            return score
        
        # 创建遗传算法优化器
        ga_config = self.config.get('ga_config', {})
        ga = GeneticAlgorithm(
            objective_function=objective_function,
            param_bounds=self.param_bounds,
            population_size=ga_config.get('population_size', 20),
            generations=ga_config.get('generations', self.n_trials),
            crossover_rate=ga_config.get('crossover_rate', 0.8),
            mutation_rate=ga_config.get('mutation_rate', 0.2),
            tournament_size=ga_config.get('tournament_size', 3),
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # 执行优化
        optimization_results = ga.optimize(X_train, y_train, n_iterations=ga_config.get('generations', self.n_trials))
        
        # 获取最佳参数和分数
        self.best_params = optimization_results['best_params']
        self.best_score = optimization_results['best_score']
        
        # 记录优化历史
        self.optimization_history = ga.history
        
        logger.info(f"遗传算法优化完成，最佳分数: {self.best_score:.6f}")
        logger.info(f"最佳参数: {self.best_params}")
        
        # 构建结果
        result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'ga_results': optimization_results
        }
        
        return result
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制优化历史
        
        Args:
            save_path (str, optional): 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        if not self.optimization_history:
            logger.warning("没有优化历史可供绘制")
            return None
        
        logger.info("绘制优化历史...")
        
        # 提取迭代次数和分数
        iterations = [h['iteration'] for h in self.optimization_history]
        scores = [h['score'] for h in self.optimization_history]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制优化历史
        ax.plot(iterations, scores, 'b-', alpha=0.7)
        ax.scatter(iterations, scores, c='b', alpha=0.5)
        
        # 标记最佳分数
        best_iteration = iterations[np.argmax(scores)]
        ax.axhline(y=self.best_score, color='r', linestyle='--', 
                  label=f'Best Score: {self.best_score:.6f}')
        ax.plot(best_iteration, self.best_score, 'ro', markersize=10)
        
        # 设置标签和标题
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.optimizer_type} Optimization History', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        ax.legend()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"优化历史图已保存: {save_path}")
        
        return fig
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最佳参数
        
        Returns:
            Dict[str, Any]: 最佳参数
        """
        return self.best_params
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        获取优化历史
        
        Returns:
            List[Dict[str, Any]]: 优化历史
        """
        return self.optimization_history