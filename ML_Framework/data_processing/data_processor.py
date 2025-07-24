"""
数据处理器类
Data Processor Class
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .normalizers import StandardNormalizer, MinMaxNormalizer, CustomNormalizer
from config.config import DataConfig

class DataProcessor:
    """数据处理器类，负责数据加载、预处理和划分"""
    
    def __init__(self, config: DataConfig):
        """
        初始化数据处理器
        
        Args:
            config: 数据配置对象
        """
        self.config = config
        self.normalizer = None
        self.data = None
        self.X_raw = None
        self.y_raw = None
        self.X_processed = None
        self.y_processed = None
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径，如果为None则使用配置中的路径
            
        Returns:
            加载的数据DataFrame
        """
        if data_path is None:
            data_path = self.config.data_path
            
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        self.data = pd.read_csv(data_path)
        print(f"数据加载完成: {self.data.shape}")
        # 修正特征和目标列的访问方式
        print(f"特征列: {self.config.features}")
        print(f"目标列: {self.config.target}")
        
        return self.data
    
    def prepare_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和目标变量
        
        Returns:
            特征DataFrame和目标Series
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 检查特征列是否存在
        missing_features = [f for f in self.config.features if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")
        
        # 检查目标列是否存在
        if self.config.target not in self.data.columns:
            raise ValueError(f"缺少目标列: {self.config.target}")
        
        self.X_raw = self.data[self.config.features].copy()
        self.y_raw = self.data[self.config.target].copy()
        
        print(f"特征形状: {self.X_raw.shape}")
        print(f"目标形状: {self.y_raw.shape}")
        print(f"目标范围: [{self.y_raw.min():.2f}, {self.y_raw.max():.2f}]")
        
        return self.X_raw, self.y_raw
    
    def normalize_features(self) -> pd.DataFrame:
        """
        标准化特征
        
        Returns:
            标准化后的特征DataFrame
        """
        if self.X_raw is None:
            raise ValueError("请先准备特征数据")
        
        # 选择标准化方法
        if self.config.normalize_method == "standard":
            self.normalizer = StandardNormalizer()
        elif self.config.normalize_method == "minmax":
            self.normalizer = MinMaxNormalizer()
        elif self.config.normalize_method == "custom":
            self.normalizer = CustomNormalizer()
        else:
            raise ValueError(f"不支持的标准化方法: {self.config.normalize_method}")
        
        # 训练标准化器并转换特征
        self.X_processed = self.normalizer.fit_transform(self.X_raw)
        
        # 验证标准化结果
        print(f"标准化方法: {self.config.normalize_method}")
        print("标准化后特征统计:")
        for col in self.X_processed.columns:
            stats = self.X_processed[col].describe()
            print(f"  {col}: 均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}")
            print(f"    范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
        # 新增调试信息
        print("[调试] 标准化后各列类型:")
        print(self.X_processed.dtypes)
        print("[调试] 标准化后前几行:")
        print(self.X_processed.head())
        return self.X_processed
    
    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练集和测试集
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.X_processed is None or self.y_raw is None:
            raise ValueError("请先处理特征和目标数据")
        # 强制转换为float64，防止类型不兼容
        X_values = np.asarray(self.X_processed.values, dtype=np.float64)
        y_values = np.asarray(self.y_raw.values, dtype=np.float64)
        # 检查NaN/Inf
        if np.any(np.isnan(X_values)) or np.any(np.isinf(X_values)):
            raise ValueError("特征数据包含NaN或Inf")
        if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
            raise ValueError("目标数据包含NaN或Inf")
        X_train, X_test, y_train, y_test = train_test_split(
            X_values,
            y_values,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True
        )
        print(f"数据划分完成:")
        print(f"  训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"  测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
        return X_train, X_test, y_train, y_test
    
    def process_all(self, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        完整的数据处理流程
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("开始数据处理流程...")
        
        # 加载数据
        self.load_data(data_path)
        
        # 准备特征和目标
        self.prepare_features_target()
        
        # 标准化特征
        self.normalize_features()
        
        # 划分数据
        X_train, X_test, y_train, y_test = self.split_data()
        
        print("数据处理完成！")
        return X_train, X_test, y_train, y_test
    
    def predict_single(self, ppfd: float, co2: float, t: float, rb: float) -> np.ndarray:
        """
        对单个样本进行预处理
        
        Args:
            ppfd: 光合光子通量密度
            co2: CO2浓度
            t: 温度
            rb: 红蓝光比
            
        Returns:
            预处理后的特征数组
        """
        if self.normalizer is None:
            raise ValueError("请先训练标准化器")
        
        # 创建输入数据
        input_data = pd.DataFrame({
            'PPFD': [ppfd],
            'CO2': [co2],
            'T': [t],
            'R:B': [rb]
        })
        
        # 标准化
        normalized_data = self.normalizer.transform(input_data)
        
        return normalized_data.values[0]
    
    def predict_batch(self, conditions: list) -> np.ndarray:
        """
        对批量样本进行预处理
        
        Args:
            conditions: 条件列表，每个条件包含 [ppfd, co2, t, rb]
            
        Returns:
            预处理后的特征数组
        """
        if self.normalizer is None:
            raise ValueError("请先训练标准化器")
        
        # 创建输入数据
        input_data = pd.DataFrame(conditions, columns=['PPFD', 'CO2', 'T', 'R:B'])
        
        # 标准化
        normalized_data = self.normalizer.transform(input_data)
        
        return normalized_data.values
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据信息
        
        Returns:
            数据信息字典
        """
        if self.data is None:
            return {}
        
        info = {
            'total_samples': len(self.data),
            'features': self.config.features,
            'target': self.config.target,
            'feature_ranges': {},
            'target_range': [self.y_raw.min(), self.y_raw.max()] if self.y_raw is not None else None,
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        if self.X_raw is not None:
            for col in self.X_raw.columns:
                info['feature_ranges'][col] = [self.X_raw[col].min(), self.X_raw[col].max()]
        
        return info 