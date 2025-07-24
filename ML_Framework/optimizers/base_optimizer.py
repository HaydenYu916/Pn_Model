"""
基础优化器抽象类
定义所有优化器的通用接口和方法
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class BaseOptimizer(ABC):
    """
    基础优化器抽象类
    
    所有具体优化器都应该继承此类并实现抽象方法
    """
    
    def __init__(self, 
                 objective_function=None,
                 param_bounds: Dict[str, Tuple[float, float]] = None,
                 scoring: str = 'neg_mean_squared_error',
                 cv: int = 5,
                 random_state: Optional[int] = None,
                 n_jobs: int = -1,
                 verbose: int = 0,
                 **kwargs):
        """
        初始化基础优化器
        
        Args:
            objective_function: 目标函数（模型），可选
            param_bounds: 参数边界字典
            scoring: 评分方法
            cv: 交叉验证折数
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 详细程度
            **kwargs: 其他参数（用于新式优化器的兼容性）
        """
        # 处理参数传递的兼容性问题
        if objective_function is not None and isinstance(objective_function, dict):
            # 如果第一个参数是字典，说明是param_bounds
            param_bounds = objective_function
            objective_function = None
        elif param_bounds is None and isinstance(objective_function, dict):
            # 处理只传递param_bounds的情况
            param_bounds = objective_function
            objective_function = None
        
        self.objective_function = objective_function
        self.param_bounds = param_bounds or {}
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 初始化结果存储
        self.best_params = None
        self.best_score = None
        self.history = []
        
        # 设置随机状态
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                 n_trials: int = 100, **kwargs) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            X: 训练特征
            y: 训练目标
            n_trials: 试验次数
            **kwargs: 其他参数
            
        Returns:
            优化结果字典
        """
        pass
    
    def _evaluate_params(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
        """
        评估参数组合
        
        Args:
            params: 参数字典
            X: 训练特征
            y: 训练目标
            
        Returns:
            评估分数
        """
        from sklearn.model_selection import cross_val_score
        
        try:
            # 设置模型参数
            model = self.objective_function.set_params(**params)
            
            # 交叉验证评估
            scores = cross_val_score(
                model, X, y,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs
            )
            
            return np.mean(scores)
            
        except Exception as e:
            if self.verbose > 0:
                print(f"参数评估失败: {params}, 错误: {str(e)}")
            # 返回很差的分数
            return -np.inf if 'neg_' in self.scoring else np.inf
    
    def _validate_params(self, params: Dict[str, Any]) -> bool:
        """
        验证参数是否在边界内
        
        Args:
            params: 参数字典
            
        Returns:
            是否有效
        """
        for param_name, value in params.items():
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                min_val = float(bounds[0]) if isinstance(bounds, (list, tuple)) else float(bounds[0])
                max_val = float(bounds[1]) if isinstance(bounds, (list, tuple)) else float(bounds[1])
                if value < min_val or value > max_val:
                    return False
        return True
    
    def _clip_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        将参数裁剪到边界内
        
        Args:
            params: 参数字典
            
        Returns:
            裁剪后的参数字典
        """
        clipped_params = {}
        for param_name, value in params.items():
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                min_val = float(bounds[0]) if isinstance(bounds, (list, tuple)) else float(bounds[0])
                max_val = float(bounds[1]) if isinstance(bounds, (list, tuple)) else float(bounds[1])
                clipped_params[param_name] = np.clip(value, min_val, max_val)
            else:
                clipped_params[param_name] = value
        return clipped_params
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """
        生成随机参数组合
        
        Returns:
            随机参数字典
        """
        params = {}
        for param_name, bounds in self.param_bounds.items():
            # 确保边界值是数字类型
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                min_val = float(bounds[0])
                max_val = float(bounds[1])
            else:
                # 如果是元组格式 (min_val, max_val)
                min_val = float(bounds[0])
                max_val = float(bounds[1])
            
            # 生成随机参数值
            params[param_name] = np.random.uniform(min_val, max_val)
        return params
    
    def _record_iteration(self, iteration: int, params: Dict[str, Any], score: float):
        """
        记录迭代结果
        
        Args:
            iteration: 迭代次数
            params: 参数字典
            score: 评估分数
        """
        self.history.append({
            'iteration': iteration,
            'params': params.copy(),
            'score': score
        })
        
        # 更新最佳结果
        if self.best_score is None or self._is_better_score(score, self.best_score):
            self.best_score = score
            self.best_params = params.copy()
    
    def _is_better_score(self, score1: float, score2: float) -> bool:
        """
        判断分数1是否比分数2更好
        
        Args:
            score1: 分数1
            score2: 分数2
            
        Returns:
            是否更好
        """
        if 'neg_' in self.scoring:
            # 负分数，越大越好
            return score1 > score2
        else:
            # 正分数，越小越好
            return score1 < score2
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """
        获取优化历史
        
        Returns:
            优化历史字典
        """
        return {
            'history': self.history,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iterations': len(self.history)
        }
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        if not self.history:
            print("No optimization history to plot")
            return
        
        iterations = [h['iteration'] for h in self.history]
        scores = [h['score'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores, 'b-', alpha=0.7, label='Optimization History')
        plt.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best Score: {self.best_score:.6f}')
        plt.xlabel('Iterations')
        plt.ylabel('Score')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最佳参数
        
        Returns:
            最佳参数字典
        """
        return self.best_params
    
    def get_best_score(self) -> float:
        """
        获取最佳分数
        
        Returns:
            最佳分数
        """
        return self.best_score 