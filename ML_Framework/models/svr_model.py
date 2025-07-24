"""
SVR 模型类
Support Vector Regression Model
"""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.svm import SVR
from .base_model import BaseModel

class SVRModel(BaseModel):
    """SVR 模型类"""
    
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, gamma: float = 'scale', 
                 kernel: str = 'rbf', **kwargs):
        """
        初始化SVR模型
        
        Args:
            C: 正则化参数
            epsilon: epsilon管子的宽度
            gamma: 核函数参数
            kernel: 核函数类型
            **kwargs: 其他参数
        """
        super().__init__(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel, **kwargs)
        self._create_model()
    
    def _create_model(self, **kwargs):
        """创建SVR模型实例"""
        if kwargs:
            self.params.update(kwargs)
        
        self.model = SVR(
            C=self.params['C'],
            epsilon=self.params['epsilon'],
            gamma=self.params['gamma'],
            kernel=self.params['kernel']
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRModel':
        """
        训练SVR模型
        
        Args:
            X: 训练特征
            y: 训练目标
            
        Returns:
            训练后的模型
        """
        self.validate_input(X, y)
        
        self.model.fit(X, y)
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
        
        return self.model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        
        if self.fitted:
            info.update({
                'n_support_vectors': len(self.model.support_),
                'support_vector_ratio': len(self.model.support_) / len(self.model.support_vectors_),
                'dual_coef_range': [self.model.dual_coef_.min(), self.model.dual_coef_.max()],
                'intercept': self.model.intercept_,
                'kernel_type': self.params['kernel']
            })
        
        return info
    
    def get_support_vectors_info(self) -> Dict[str, Any]:
        """获取支持向量信息"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        return {
            'n_support_vectors': len(self.model.support_),
            'support_vector_indices': self.model.support_,
            'support_vectors': self.model.support_vectors_,
            'dual_coef': self.model.dual_coef_,
            'support_vector_ratio': len(self.model.support_) / self.model.support_vectors_.shape[0]
        }
    
    def get_decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        获取决策函数值
        
        Args:
            X: 输入数据
            
        Returns:
            决策函数值
        """
        if not self.fitted:
            raise ValueError("模型未训练")
        
        return self.model.decision_function(X)
    
    def update_params(self, **params):
        """更新模型参数"""
        self.params.update(params)
        self._create_model()
        self.fitted = False  # 需要重新训练
    
    def __repr__(self):
        return f"SVRModel(C={self.params['C']}, epsilon={self.params['epsilon']}, gamma={self.params['gamma']}, kernel='{self.params['kernel']}', fitted={self.fitted})" 