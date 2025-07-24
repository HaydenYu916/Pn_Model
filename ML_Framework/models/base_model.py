"""
机器学习模型基类
Base Machine Learning Model
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

class BaseModel(ABC):
    """机器学习模型基类"""
    
    def __init__(self, **kwargs):
        """
        初始化模型
        
        Args:
            **kwargs: 模型参数
        """
        self.model = None
        self.fitted = False
        self.params = kwargs
        self.training_history = []
        
    @abstractmethod
    def _create_model(self, **kwargs):
        """创建模型实例"""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred)
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        """交叉验证"""
        if self.model is None:
            self._create_model(**self.params)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
        
        if scoring == 'neg_mean_squared_error':
            results['cv_rmse'] = np.sqrt(-scores.mean())
        
        return results
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """获取模型参数
        
        Args:
            deep: 是否深度获取参数（兼容sklearn接口）
        """
        return self.params.copy()
    
    def set_params(self, **params):
        """设置模型参数"""
        self.params.update(params)
        if self.model is not None:
            self._create_model(**self.params)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': self.__class__.__name__,
            'fitted': self.fitted,
            'params': self.get_params(),
            'training_history': self.training_history
        }
        
        if hasattr(self.model, 'support_'):
            info['n_support_vectors'] = len(self.model.support_)
        
        return info
    
    def predict_single(self, features: np.ndarray) -> float:
        """单个样本预测"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return self.predict(features)[0]
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """批量预测"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        return self.predict(features)
    
    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """验证输入数据"""
        if not isinstance(X, np.ndarray):
            raise TypeError("X 必须是 numpy 数组")
        
        if len(X.shape) != 2:
            raise ValueError("X 必须是二维数组")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y 必须是 numpy 数组")
            
            if len(y.shape) != 1:
                raise ValueError("y 必须是一维数组")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError("X 和 y 的样本数量必须相同")
    
    def save_training_history(self, epoch: int, metrics: Dict[str, float]):
        """保存训练历史"""
        history_entry = {
            'epoch': epoch,
            'timestamp': pd.Timestamp.now(),
            **metrics
        }
        self.training_history.append(history_entry)
    
    def get_training_history(self) -> pd.DataFrame:
        """获取训练历史"""
        if not self.training_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.training_history)
    
    def reset_training_history(self):
        """重置训练历史"""
        self.training_history = []
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fitted={self.fitted}, params={self.params})" 