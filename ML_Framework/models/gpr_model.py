"""
GPR 模型类
Gaussian Process Regression Model
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from .base_model import BaseModel

class GPRModel(BaseModel):
    """GPR 模型类"""
    
    def __init__(self, alpha: float = 1e-10, 
                 length_scale: float = 1.0,
                 length_scale_bounds: tuple = (1e-5, 1e5),
                 constant_value: float = 1.0,
                 constant_value_bounds: tuple = (1e-5, 1e5),
                 noise_level: float = 1e-10,
                 noise_level_bounds: tuple = (1e-10, 1.0),
                 n_restarts_optimizer: int = 10,
                 **kwargs):
        """
        初始化GPR模型
        
        Args:
            alpha: 噪声参数，用于数值稳定性
            length_scale: RBF核的长度尺度
            length_scale_bounds: 长度尺度的边界
            constant_value: 常数核的值
            constant_value_bounds: 常数核的边界
            noise_level: 白噪声水平
            noise_level_bounds: 白噪声水平的边界
            n_restarts_optimizer: 优化器重启次数
            **kwargs: 其他参数
        """
        super().__init__(
            alpha=alpha,
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            n_restarts_optimizer=n_restarts_optimizer,
            **kwargs
        )
        self._create_model()
    
    def _create_model(self, **kwargs):
        """创建GPR模型实例"""
        if kwargs:
            self.params.update(kwargs)
        
        # 为每个特征创建独立的长度尺度
        length_scales = np.ones(4) * self.params['length_scale']
        length_scale_bounds = [(self.params['length_scale_bounds'][0], 
                              self.params['length_scale_bounds'][1])] * 4
        
        # 创建复合核函数：常数核 * RBF核 + 白噪声核
        kernel = (C(
            constant_value=self.params['constant_value'], 
            constant_value_bounds=self.params['constant_value_bounds']
        ) * RBF(
            length_scale=length_scales,
            length_scale_bounds=length_scale_bounds
        ) + WhiteKernel(
            noise_level=self.params['noise_level'],
            noise_level_bounds=self.params['noise_level_bounds']
        ))
        
        # 确保n_restarts_optimizer是整数
        n_restarts = int(self.params.get('n_restarts_optimizer', 10))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.params['alpha'],
            n_restarts_optimizer=n_restarts,
            random_state=42,
            normalize_y=True  # 添加y标准化来减少收敛警告
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPRModel':
        """
        训练GPR模型
        
        Args:
            X: 训练特征
            y: 训练目标
            
        Returns:
            训练后的模型
        """
        self.validate_input(X, y)
        
        # 检查数据是否已经标准化
        if not self._is_data_standardized(X):
            print("警告：输入特征似乎未标准化，这可能导致优化器收敛问题")
        
        try:
            self.model.fit(X, y)
            self.fitted = True
            
            # 保存训练历史
            train_metrics = self.evaluate(X, y)
            self.save_training_history(0, train_metrics)
            
            # 打印优化后的核函数参数
            print("优化后的核函数参数:")
            print(self.model.kernel_)
            
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            raise
        
        return self
    
    def _is_data_standardized(self, X: np.ndarray) -> bool:
        """
        检查数据是否已经标准化
        
        Args:
            X: 输入特征
            
        Returns:
            bool: 如果数据已标准化则返回True
        """
        means = np.abs(np.mean(X, axis=0))
        stds = np.std(X, axis=0)
        return np.all(means < 0.1) and np.all(np.abs(stds - 1) < 0.1)
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        预测
        
        Args:
            X: 测试特征
            return_std: 是否返回预测的标准差
            
        Returns:
            预测结果（如果return_std=True，则还包含标准差）
        """
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        self.validate_input(X)
        
        try:
            return self.model.predict(X, return_std=return_std)
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            raise
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测并返回不确定性
        
        Args:
            X: 测试特征
            
        Returns:
            预测结果和标准差
        """
        return self.predict(X, return_std=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        
        if self.fitted:
            kernel_params = self.model.kernel_.get_params()
            # 提取关键参数
            info.update({
                'kernel': str(self.model.kernel_),
                'log_marginal_likelihood': self.model.log_marginal_likelihood_value_,
                'optimized_length_scales': kernel_params.get('k1__k2__length_scale', None),
                'optimized_noise_level': kernel_params.get('k2__noise_level', None),
                'optimized_constant': kernel_params.get('k1__k1__constant_value', None)
            })
        
        return info
    
    def get_kernel_info(self) -> Dict[str, Any]:
        """获取核函数信息"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        kernel_params = self.model.kernel_.get_params()
        return {
            'kernel_type': str(self.model.kernel_),
            'length_scales': kernel_params.get('k1__k2__length_scale', None),
            'noise_level': kernel_params.get('k2__noise_level', None),
            'constant_value': kernel_params.get('k1__k1__constant_value', None),
            'bounds': {
                'length_scale_bounds': kernel_params.get('k1__k2__length_scale_bounds', None),
                'noise_level_bounds': kernel_params.get('k2__noise_level_bounds', None),
                'constant_value_bounds': kernel_params.get('k1__k1__constant_value_bounds', None)
            }
        }
    
    def get_log_marginal_likelihood(self) -> float:
        """获取对数边际似然"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        return self.model.log_marginal_likelihood_value_
    
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        从后验分布中采样
        
        Args:
            X: 输入特征
            n_samples: 采样数量
            random_state: 随机种子
            
        Returns:
            采样结果
        """
        if not self.fitted:
            raise ValueError("模型未训练")
        
        return self.model.sample_y(X, n_samples=n_samples, random_state=random_state)
    
    def calculate_confidence_intervals(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算置信区间
        
        Args:
            X: 输入特征
            confidence: 置信度
            
        Returns:
            预测均值、下界、上界
        """
        if not self.fitted:
            raise ValueError("模型未训练")
        
        mean, std = self.predict(X, return_std=True)
        
        # 计算置信区间
        from scipy import stats
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std
        
        return mean, lower_bound, upper_bound
    
    def update_params(self, **params):
        """更新模型参数"""
        self.params.update(params)
        self._create_model()
        self.fitted = False  # 需要重新训练
    
    def __repr__(self):
        return f"GPRModel(alpha={self.params['alpha']}, n_restarts_optimizer={self.params['n_restarts_optimizer']}, fitted={self.fitted})" 