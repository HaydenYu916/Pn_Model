"""
Optuna 优化器模块
Optuna Optimizers Module

使用 Optuna 框架实现 TPE 和随机搜索优化器
"""

import numpy as np
import optuna
from typing import Dict, Any, Callable, Optional, List, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import logging
import inspect
from tqdm import tqdm

from .base_optimizer import BaseOptimizer


class TPEOptimizer(BaseOptimizer):
    """Tree-structured Parzen Estimator 优化器"""
    
    def __init__(self, param_bounds: Dict[str, List[float]], 
                 n_trials: int = 100,
                 **kwargs):
        """
        初始化 TPE 优化器
        
        Args:
            param_bounds: 参数边界字典
            n_trials: 试验次数
        """
        super().__init__(param_bounds, **kwargs)
        self.n_trials = n_trials
        self._validate_param_bounds()
    
    def _validate_param_bounds(self):
        """验证参数边界的有效性"""
        for param_name, bounds in self.param_bounds.items():
            if len(bounds) != 2:
                raise ValueError(f"参数 {param_name} 的边界必须包含两个值 [min, max]")
            lower = float(bounds[0])
            upper = float(bounds[1])
            if lower >= upper:
                raise ValueError(f"参数 {param_name} 的下界 ({lower}) 必须小于上界 ({upper})")
    
    def _get_model_param_names(self, model_class):
        """获取模型构造函数参数名列表（不含self）"""
        sig = inspect.signature(model_class.__init__)
        return [p for p in sig.parameters if p != 'self']
    
    def _suggest_param_value(self, trial: optuna.Trial, param_name: str, bounds: List[float]) -> Any:
        """根据参数类型和边界建议参数值"""
        try:
            # 确保边界值是数值类型
            lower_bound = float(bounds[0])
            upper_bound = float(bounds[1])
            
            # 检查是否为整数参数
            if param_name in ['n_restarts_optimizer', 'n_layers']:
                return trial.suggest_int(param_name, int(lower_bound), int(upper_bound))
            elif param_name in ['alpha', 'noise_level', 'length_scale', 'constant_value']:
                # 对这些参数使用对数尺度
                return trial.suggest_float(param_name, lower_bound, upper_bound, log=True)
            else:
                return trial.suggest_float(param_name, lower_bound, upper_bound)
        except Exception as e:
            logging.error(f"参数 {param_name} 取值出错: {str(e)}")
            raise
    
    def _objective(self, trial: optuna.Trial, model_class, X_train, y_train, cv_folds=5):
        """Optuna 目标函数"""
        # 从 trial 中获取参数
        model_param_names = self._get_model_param_names(model_class)
        params = {}
        
        try:
            for param_name in model_param_names:
                if param_name in self.param_bounds:
                    bounds = self.param_bounds[param_name]
                    params[param_name] = self._suggest_param_value(trial, param_name, bounds)
            
            # 创建模型并评估
            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error',
                                   error_score='raise')  # 使用 error_score='raise' 来捕获交叉验证中的错误
            score = -np.mean(scores)  # 转换为最小化问题
            
            # 记录当前参数的性能
            trial.set_user_attr('params', params)
            trial.set_user_attr('cv_scores', scores.tolist())
            
            return score
            
        except Exception as e:
            logging.warning(f"参数评估失败: {params}, 错误: {str(e)}")
            raise optuna.exceptions.TrialPruned()
    
    def optimize(self, model_class, X_train: np.ndarray, y_train: np.ndarray, 
                n_iterations: int = None, **kwargs) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            model_class: 模型类
            X_train: 训练特征
            y_train: 训练目标
            n_iterations: 迭代次数（覆盖默认值）
            
        Returns:
            优化结果字典
        """
        if n_iterations is not None:
            self.n_trials = n_iterations
            
        print(f"开始 TPE 优化...")
        print(f"试验次数: {self.n_trials}")
        print(f"参数边界: {self.param_bounds}")
        
        # 创建 Optuna study，使用 TPE sampler
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=10,  # 增加初始随机试验次数
                multivariate=True,    # 启用多变量 TPE
            )
        )
        
        # 定义目标函数
        objective = lambda trial: self._objective(trial, model_class, X_train, y_train)
        
        try:
            # 创建进度条
            pbar = tqdm(total=self.n_trials, desc="优化进度", ncols=100)
            
            # 定义回调函数来更新进度条
            def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
                pbar.update(1)
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    pbar.set_postfix({
                        'best_score': f'{study.best_value:.4f}',
                        'trial_score': f'{trial.value:.4f}'
                    })
            
            # 执行优化
            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])
            
            # 关闭进度条
            pbar.close()
            
            # 获取最佳结果
            best_params = study.best_params
            best_score = study.best_value
            
            # 获取优化历史
            history = study.trials_dataframe()
            
            # 打印优化结果
            print("\n优化完成！")
            print(f"最佳参数: {best_params}")
            print(f"最佳分数: {best_score:.6f}")
            print(f"总试验次数: {len(study.trials)}")
            print(f"完成的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
            print(f"失败的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))}")
            print(f"剪枝的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_history': history,
                'study': study,
                'n_trials': len(study.trials),
                'n_complete': len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
                'n_fail': len(study.get_trials(states=[optuna.trial.TrialState.FAIL])),
                'n_pruned': len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
            }
            
        except Exception as e:
            logging.error(f"优化过程中出现错误: {str(e)}")
            raise


class OptunaRandomSearch(BaseOptimizer):
    """Optuna 随机搜索优化器（作为基准）"""
    
    def __init__(self, param_bounds: Dict[str, List[float]], 
                 n_trials: int = 100,
                 **kwargs):
        """
        初始化随机搜索优化器
        
        Args:
            param_bounds: 参数边界字典
            n_trials: 试验次数
        """
        super().__init__(param_bounds, **kwargs)
        self.n_trials = n_trials
        self._validate_param_bounds()
    
    def _validate_param_bounds(self):
        """验证参数边界的有效性"""
        for param_name, bounds in self.param_bounds.items():
            if len(bounds) != 2:
                raise ValueError(f"参数 {param_name} 的边界必须包含两个值 [min, max]")
            lower = float(bounds[0])
            upper = float(bounds[1])
            if lower >= upper:
                raise ValueError(f"参数 {param_name} 的下界 ({lower}) 必须小于上界 ({upper})")
    
    def _get_model_param_names(self, model_class):
        """获取模型构造函数参数名列表（不含self）"""
        sig = inspect.signature(model_class.__init__)
        return [p for p in sig.parameters if p != 'self']
    
    def _suggest_param_value(self, trial: optuna.Trial, param_name: str, bounds: List[float]) -> Any:
        """根据参数类型和边界建议参数值"""
        try:
            # 确保边界值是数值类型
            lower_bound = float(bounds[0])
            upper_bound = float(bounds[1])
            
            # 检查是否为整数参数
            if param_name in ['n_restarts_optimizer', 'n_layers']:
                return trial.suggest_int(param_name, int(lower_bound), int(upper_bound))
            elif param_name in ['alpha', 'noise_level', 'length_scale', 'constant_value']:
                # 对这些参数使用对数尺度
                return trial.suggest_float(param_name, lower_bound, upper_bound, log=True)
            else:
                return trial.suggest_float(param_name, lower_bound, upper_bound)
        except Exception as e:
            logging.error(f"参数 {param_name} 取值出错: {str(e)}")
            raise
    
    def _objective(self, trial: optuna.Trial, model_class, X_train, y_train, cv_folds=5):
        """Optuna 目标函数"""
        # 从 trial 中获取参数
        model_param_names = self._get_model_param_names(model_class)
        params = {}
        
        try:
            for param_name in model_param_names:
                if param_name in self.param_bounds:
                    bounds = self.param_bounds[param_name]
                    params[param_name] = self._suggest_param_value(trial, param_name, bounds)
            
            # 创建模型并评估
            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error',
                                   error_score='raise')
            score = -np.mean(scores)
            
            # 记录当前参数的性能
            trial.set_user_attr('params', params)
            trial.set_user_attr('cv_scores', scores.tolist())
            
            return score
            
        except Exception as e:
            logging.warning(f"参数评估失败: {params}, 错误: {str(e)}")
            raise optuna.exceptions.TrialPruned()
    
    def optimize(self, model_class, X_train: np.ndarray, y_train: np.ndarray, 
                n_iterations: int = None, **kwargs) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            model_class: 模型类
            X_train: 训练特征
            y_train: 训练目标
            n_iterations: 迭代次数（覆盖默认值）
            
        Returns:
            优化结果字典
        """
        if n_iterations is not None:
            self.n_trials = n_iterations
            
        print(f"开始随机搜索优化...")
        print(f"试验次数: {self.n_trials}")
        print(f"参数边界: {self.param_bounds}")
        
        # 创建 Optuna study，使用随机采样器
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.RandomSampler(seed=42)
        )
        
        # 定义目标函数
        objective = lambda trial: self._objective(trial, model_class, X_train, y_train)
        
        try:
            # 执行优化
            study.optimize(objective, n_trials=self.n_trials)
            
            # 获取最佳结果
            best_params = study.best_params
            best_score = study.best_value
            
            # 获取优化历史
            history = study.trials_dataframe()
            
            # 打印优化结果
            print("\n优化完成！")
            print(f"最佳参数: {best_params}")
            print(f"最佳分数: {best_score:.6f}")
            print(f"总试验次数: {len(study.trials)}")
            print(f"完成的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
            print(f"失败的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))}")
            print(f"剪枝的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_history': history,
                'study': study,
                'n_trials': len(study.trials),
                'n_complete': len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
                'n_fail': len(study.get_trials(states=[optuna.trial.TrialState.FAIL])),
                'n_pruned': len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
            }
            
        except Exception as e:
            logging.error(f"优化过程中出现错误: {str(e)}")
            raise 


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 优化器"""
    
    def __init__(self, param_bounds: Dict[str, List[float]], 
                 n_trials: int = 100,
                 population_size: int = None,
                 sigma0: float = None,
                 seed: int = 42,
                 **kwargs):
        """
        初始化 CMA-ES 优化器
        
        Args:
            param_bounds: 参数边界字典
            n_trials: 试验次数
            population_size: 种群大小（可选，设置为None则自动确定）
            sigma0: 初始步长（可选，默认为参数范围的1/3）
            seed: 随机种子
        """
        super().__init__(param_bounds, **kwargs)
        self.n_trials = n_trials
        self.population_size = population_size
        self.sigma0 = sigma0
        self.seed = seed
        self._validate_param_bounds()
    
    def _validate_param_bounds(self):
        """验证参数边界的有效性"""
        for param_name, bounds in self.param_bounds.items():
            if len(bounds) != 2:
                raise ValueError(f"参数 {param_name} 的边界必须包含两个值 [min, max]")
            lower = float(bounds[0])
            upper = float(bounds[1])
            if lower >= upper:
                raise ValueError(f"参数 {param_name} 的下界 ({lower}) 必须小于上界 ({upper})")
    
    def _get_model_param_names(self, model_class):
        """获取模型构造函数参数名列表（不含self）"""
        sig = inspect.signature(model_class.__init__)
        return [p for p in sig.parameters if p != 'self']
    
    def _suggest_param_value(self, trial: optuna.Trial, param_name: str, bounds: List[float]) -> Any:
        """根据参数类型和边界建议参数值"""
        try:
            # 确保边界值是数值类型
            lower_bound = float(bounds[0])
            upper_bound = float(bounds[1])
            
            # 检查是否为整数参数
            if param_name in ['n_restarts_optimizer', 'n_layers']:
                return trial.suggest_int(param_name, int(lower_bound), int(upper_bound))
            elif param_name in ['alpha', 'noise_level', 'length_scale', 'constant_value']:
                # 对这些参数使用对数尺度
                return trial.suggest_float(param_name, lower_bound, upper_bound, log=True)
            else:
                return trial.suggest_float(param_name, lower_bound, upper_bound)
        except Exception as e:
            logging.error(f"参数 {param_name} 取值出错: {str(e)}")
            raise
    
    def _objective(self, trial: optuna.Trial, model_class, X_train, y_train, cv_folds=5):
        """Optuna 目标函数"""
        # 从 trial 中获取参数
        model_param_names = self._get_model_param_names(model_class)
        params = {}
        
        try:
            for param_name in model_param_names:
                if param_name in self.param_bounds:
                    bounds = self.param_bounds[param_name]
                    params[param_name] = self._suggest_param_value(trial, param_name, bounds)
            
            # 创建模型并评估
            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error',
                                   error_score='raise')
            score = -np.mean(scores)
            
            # 记录当前参数的性能
            trial.set_user_attr('params', params)
            trial.set_user_attr('cv_scores', scores.tolist())
            
            return score
            
        except Exception as e:
            logging.warning(f"参数评估失败: {params}, 错误: {str(e)}")
            raise optuna.exceptions.TrialPruned()
    
    def optimize(self, model_class, X_train: np.ndarray, y_train: np.ndarray, 
                n_iterations: int = None, **kwargs) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            model_class: 模型类
            X_train: 训练特征
            y_train: 训练目标
            n_iterations: 迭代次数（覆盖默认值）
            
        Returns:
            优化结果字典
        """
        if n_iterations is not None:
            self.n_trials = n_iterations
            
        print(f"开始 CMA-ES 优化...")
        print(f"试验次数: {self.n_trials}")
        print(f"参数边界: {self.param_bounds}")
        print(f"种群大小: {self.population_size}")
        print(f"初始步长: {self.sigma0}")
        
        # 创建 Optuna study，使用 CMA-ES 采样器
        sampler = optuna.samplers.CmaEsSampler(
            seed=self.seed,
            sigma0=self.sigma0
        )
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        # 定义目标函数
        objective = lambda trial: self._objective(trial, model_class, X_train, y_train)
        
        try:
            # 创建进度条
            pbar = tqdm(total=self.n_trials, desc="CMA-ES优化进度", ncols=100)
            
            # 定义回调函数来更新进度条
            def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
                pbar.update(1)
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    pbar.set_postfix({
                        'best_score': f'{study.best_value:.4f}',
                        'trial_score': f'{trial.value:.4f}'
                    })
            
            # 执行优化
            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])
            
            # 关闭进度条
            pbar.close()
            
            # 获取最佳结果
            best_params = study.best_params
            best_score = study.best_value
            
            # 获取优化历史
            history = study.trials_dataframe()
            
            # 打印优化结果
            print("\nCMA-ES 优化完成！")
            print(f"最佳参数: {best_params}")
            print(f"最佳分数: {best_score:.6f}")
            print(f"总试验次数: {len(study.trials)}")
            print(f"完成的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
            print(f"失败的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))}")
            print(f"剪枝的试验次数: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_history': history,
                'study': study,
                'n_trials': len(study.trials),
                'n_complete': len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
                'n_fail': len(study.get_trials(states=[optuna.trial.TrialState.FAIL])),
                'n_pruned': len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
            }
            
        except Exception as e:
            logging.error(f"CMA-ES 优化过程中出现错误: {str(e)}")
            raise 