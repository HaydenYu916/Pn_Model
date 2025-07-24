"""
标准化器类
Normalizer Classes
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod

class BaseNormalizer(ABC):
    """标准化器基类"""
    
    def __init__(self):
        self.fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'BaseNormalizer':
        """训练标准化器"""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        pass
    
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """反转换数据"""
        pass
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """训练并转换数据"""
        return self.fit(X).transform(X)

class StandardNormalizer(BaseNormalizer):
    """标准化器 (Z-score normalization)"""
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.means_ = None
        self.scales_ = None
    
    def fit(self, X: pd.DataFrame) -> 'StandardNormalizer':
        """训练标准化器"""
        self.feature_names = X.columns.tolist()
        self.scaler.fit(X.values)
        self.means_ = self.scaler.mean_
        self.scales_ = self.scaler.scale_
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("标准化器未训练，请先调用 fit() 方法")
        
        transformed = self.scaler.transform(X.values)
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """反转换数据"""
        if not self.fitted:
            raise ValueError("标准化器未训练，请先调用 fit() 方法")
        
        inverse_transformed = self.scaler.inverse_transform(X.values)
        return pd.DataFrame(inverse_transformed, columns=self.feature_names, index=X.index)
    
    def get_params(self) -> dict:
        """获取标准化参数"""
        if not self.fitted:
            raise ValueError("标准化器未训练")
        
        return {
            'means': self.means_.tolist(),
            'scales': self.scales_.tolist(),
            'feature_names': self.feature_names,
            'method': 'standard'
        }

class MinMaxNormalizer(BaseNormalizer):
    """最小-最大标准化器"""
    
    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__()
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_names = None
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
    
    def fit(self, X: pd.DataFrame) -> 'MinMaxNormalizer':
        """训练标准化器"""
        self.feature_names = X.columns.tolist()
        self.scaler.fit(X.values)
        self.data_min_ = self.scaler.data_min_
        self.data_max_ = self.scaler.data_max_
        self.data_range_ = self.scaler.data_range_
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("标准化器未训练，请先调用 fit() 方法")
        
        transformed = self.scaler.transform(X.values)
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """反转换数据"""
        if not self.fitted:
            raise ValueError("标准化器未训练，请先调用 fit() 方法")
        
        inverse_transformed = self.scaler.inverse_transform(X.values)
        return pd.DataFrame(inverse_transformed, columns=self.feature_names, index=X.index)
    
    def get_params(self) -> dict:
        """获取标准化参数"""
        if not self.fitted:
            raise ValueError("标准化器未训练")
        
        return {
            'data_min': self.data_min_.tolist(),
            'data_max': self.data_max_.tolist(),
            'data_range': self.data_range_.tolist(),
            'feature_range': self.feature_range,
            'feature_names': self.feature_names,
            'method': 'minmax'
        }

class CustomNormalizer(BaseNormalizer):
    """自定义标准化器 (归一化到[-1, 1])"""
    
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.mins_ = None
        self.maxs_ = None
        self.ranges_ = None
    
    def fit(self, X: pd.DataFrame) -> 'CustomNormalizer':
        """训练标准化器"""
        self.feature_names = X.columns.tolist()
        self.mins_ = X.min().values
        self.maxs_ = X.max().values
        self.ranges_ = self.maxs_ - self.mins_
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据到[-1, 1]范围"""
        if not self.fitted:
            raise ValueError("标准化器未训练，请先调用 fit() 方法")
        
        # 自定义归一化公式: x_norm = -1 + 2 * (x - x_min) / (x_max - x_min)
        transformed = -1 + 2 * (X.values - self.mins_) / self.ranges_
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """反转换数据"""
        if not self.fitted:
            raise ValueError("标准化器未训练，请先调用 fit() 方法")
        
        # 反归一化公式: x = x_min + (x_norm + 1) * (x_max - x_min) / 2
        inverse_transformed = self.mins_ + (X.values + 1) * self.ranges_ / 2
        return pd.DataFrame(inverse_transformed, columns=self.feature_names, index=X.index)
    
    def get_params(self) -> dict:
        """获取标准化参数"""
        if not self.fitted:
            raise ValueError("标准化器未训练")
        
        return {
            'mins': self.mins_.tolist(),
            'maxs': self.maxs_.tolist(),
            'ranges': self.ranges_.tolist(),
            'feature_names': self.feature_names,
            'method': 'custom',
            'formula': 'x_norm = -1 + 2 * (x - x_min) / (x_max - x_min)'
        } 