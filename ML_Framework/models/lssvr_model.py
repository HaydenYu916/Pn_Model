"""
LSSVR 模型类
Least Squares Support Vector Regression Model
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_model import BaseModel

class LSSVRModel(BaseModel):
    """LSSVR 模型类"""
    
    def __init__(self, gamma: float = 1.0, sigma2: float = 1.0, kernel: str = 'rbf', **kwargs):
        """
        初始化LSSVR模型
        
        Args:
            gamma: 核参数
            sigma2: 正则化参数
            kernel: 核函数类型，目前只支持'rbf'
            **kwargs: 其他参数
        """
        super().__init__(gamma=gamma, sigma2=sigma2, kernel=kernel, **kwargs)
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self._create_model()
    
    def _create_model(self, **kwargs):
        """创建LSSVR模型实例"""
        # LSSVR 不使用 scikit-learn，直接实现
        self.model = self
        if kwargs:
            self.params.update(kwargs)
    
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF核函数
        
        Args:
            X1: 第一个数据集
            X2: 第二个数据集
            
        Returns:
            核矩阵
        """
        if self.params['kernel'] == 'rbf':
            # 计算平方欧氏距离
            dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.params['gamma'] * dists)
        else:
            raise ValueError(f"不支持的核函数: {self.params['kernel']}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSSVRModel':
        """
        训练LSSVR模型
        
        Args:
            X: 训练特征
            y: 训练目标
            
        Returns:
            训练后的模型
        """
        self.validate_input(X, y)
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        n_samples = X.shape[0]
        
        # 计算核矩阵
        K = self._kernel_function(X, X)
        
        # 构建系统矩阵 [K + I/sigma2, 1; 1^T, 0]
        A = np.zeros((n_samples + 1, n_samples + 1))
        A[:n_samples, :n_samples] = K + np.eye(n_samples) / self.params['sigma2']
        A[:n_samples, n_samples] = 1
        A[n_samples, :n_samples] = 1
        A[n_samples, n_samples] = 0
        
        # 构建右侧向量 [y; 0]
        b = np.zeros(n_samples + 1)
        b[:n_samples] = y
        b[n_samples] = 0
        
        # 求解线性系统
        try:
            solution = np.linalg.solve(A, b)
            self.alpha = solution[:n_samples]
            self.b = solution[n_samples]
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            solution = np.linalg.pinv(A) @ b
            self.alpha = solution[:n_samples]
            self.b = solution[n_samples]
        
        self.fitted = True
        
        # 保存训练历史
        train_metrics = self.evaluate(X, y)
        self.save_training_history(0, train_metrics)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 测试特征
            
        Returns:
            预测结果
        """
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        self.validate_input(X)
        
        # 计算测试样本与训练样本的核矩阵
        K = self._kernel_function(X, self.X_train)
        
        # 预测
        predictions = np.dot(K, self.alpha) + self.b
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        
        if self.fitted:
            info.update({
                'n_training_samples': len(self.X_train),
                'alpha_range': [self.alpha.min(), self.alpha.max()],
                'bias': self.b,
                'kernel_type': self.params['kernel']
            })
        
        return info
    
    def get_support_vectors_info(self) -> Dict[str, Any]:
        """获取支持向量信息"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        # 在LSSVR中，所有训练样本都是支持向量
        return {
            'n_support_vectors': len(self.X_train),
            'support_vector_ratio': 1.0,  # 100%
            'alpha_stats': {
                'mean': self.alpha.mean(),
                'std': self.alpha.std(),
                'min': self.alpha.min(),
                'max': self.alpha.max()
            }
        }
    
    def get_kernel_matrix(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取核矩阵
        
        Args:
            X: 输入数据，如果为None则使用训练数据
            
        Returns:
            核矩阵
        """
        if not self.fitted:
            raise ValueError("模型未训练")
        
        if X is None:
            X = self.X_train
        
        return self._kernel_function(X, self.X_train)
    
    def update_params(self, **params):
        """更新模型参数并重新训练"""
        if 'gamma' in params:
            self.params['gamma'] = params['gamma']
        if 'sigma2' in params:
            self.params['sigma2'] = params['sigma2']
        if 'kernel' in params:
            self.params['kernel'] = params['kernel']
        
        # 如果模型已训练，需要重新训练
        if self.fitted and self.X_train is not None:
            self.fit(self.X_train, self.y_train)
    
    def __repr__(self):
        return f"LSSVRModel(gamma={self.params['gamma']}, sigma2={self.params['sigma2']}, kernel='{self.params['kernel']}', fitted={self.fitted})" 