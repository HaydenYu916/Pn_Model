#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理组件
提供数据加载、预处理、分割和特征工程功能
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理类，负责数据加载、预处理、分割和特征工程"""
    
    def __init__(self, config: Dict):
        """
        初始化数据处理器
        
        Args:
            config (Dict): 配置字典，包含数据处理相关参数
        """
        self.config = config
        self.data_path = config.get('data_path')
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.target_column = config.get('target_column', 'Pn')
        self.feature_columns = config.get('feature_columns', None)
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
        
        logger.info(f"数据处理器初始化完成，数据路径: {self.data_path}")
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            data_path (str, optional): 数据文件路径，如果为None则使用配置中的路径
        
        Returns:
            pd.DataFrame: 加载的数据框
        
        Raises:
            FileNotFoundError: 数据文件不存在时抛出
            ValueError: 数据格式不正确时抛出
        """
        file_path = data_path or self.data_path
        
        if not file_path:
            raise ValueError("未指定数据文件路径")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        logger.info(f"加载数据文件: {file_path}")
        
        # 根据文件扩展名确定加载方法
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            logger.info(f"数据加载成功，形状: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}", exc_info=True)
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            df (pd.DataFrame): 原始数据框
        
        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        logger.info("开始数据预处理...")
        
        # 检查数据框是否为空
        if df.empty:
            raise ValueError("数据框为空")
        
        # 检查目标列是否存在
        if self.target_column not in df.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不在数据框中")
        
        # 1. 处理缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"检测到缺失值: \n{missing_values[missing_values > 0]}")
            
            # 创建缺失值填充器
            self.imputer = SimpleImputer(strategy='mean')
            
            # 分离特征和目标
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # 填充特征中的缺失值
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # 如果目标列有缺失值，也进行填充
            if y.isnull().sum() > 0:
                y = pd.Series(
                    SimpleImputer(strategy='mean').fit_transform(y.values.reshape(-1, 1)).ravel(),
                    index=y.index
                )
            
            # 重新组合数据框
            df = pd.concat([X_imputed, y], axis=1)
            logger.info("缺失值填充完成")
        
        # 2. 处理异常值
        if self.config.get('handle_outliers', True):
            logger.info("检测和处理异常值...")
            
            # 使用IQR方法检测异常值
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                if not outliers.empty:
                    logger.info(f"列 '{col}' 中检测到 {len(outliers)} 个异常值")
                    
                    # 根据配置决定如何处理异常值
                    outlier_strategy = self.config.get('outlier_strategy', 'clip')
                    
                    if outlier_strategy == 'clip':
                        # 将异常值限制在边界内
                        df[col] = df[col].clip(lower_bound, upper_bound)
                        logger.info(f"列 '{col}' 的异常值已被限制在 [{lower_bound:.2f}, {upper_bound:.2f}] 范围内")
                    elif outlier_strategy == 'remove':
                        # 移除包含异常值的行
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        logger.info(f"已移除包含异常值的 {len(outliers)} 行")
                    elif outlier_strategy == 'mean':
                        # 将异常值替换为均值
                        mean_value = df[col].mean()
                        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = mean_value
                        logger.info(f"列 '{col}' 的异常值已被替换为均值 {mean_value:.2f}")
        
        # 3. 特征选择
        if self.feature_columns:
            # 使用指定的特征列
            all_columns = self.feature_columns + [self.target_column]
            missing_columns = [col for col in all_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"以下指定的列在数据框中不存在: {missing_columns}")
                # 只使用存在的列
                existing_columns = [col for col in all_columns if col in df.columns]
                df = df[existing_columns]
            else:
                df = df[all_columns]
            
            logger.info(f"使用指定的特征列: {self.feature_columns}")
        else:
            # 自动特征选择
            if self.config.get('auto_feature_selection', False):
                logger.info("执行自动特征选择...")
                
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]
                
                # 使用F检验进行特征选择
                k = min(self.config.get('k_best_features', 5), X.shape[1])
                self.feature_selector = SelectKBest(f_regression, k=k)
                self.feature_selector.fit(X, y)
                
                # 获取选择的特征
                selected_features = X.columns[self.feature_selector.get_support()]
                logger.info(f"自动选择的特征: {list(selected_features)}")
                
                # 更新数据框
                df = df[list(selected_features) + [self.target_column]]
        
        logger.info(f"预处理完成，数据形状: {df.shape}")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        分割数据为训练集和测试集
        
        Args:
            df (pd.DataFrame): 预处理后的数据框
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
                (X_train, X_test, y_train, y_test)
        """
        logger.info(f"分割数据为训练集和测试集，测试集比例: {self.test_size}")
        
        # 分离特征和目标
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        logger.info(f"数据分割完成，训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        标准化/归一化数据
        
        Args:
            X_train (pd.DataFrame): 训练特征
            X_test (pd.DataFrame): 测试特征
            scaler_type (str): 缩放器类型，'standard'或'minmax'
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 标准化后的(X_train, X_test)
        """
        logger.info(f"使用 {scaler_type} 缩放器标准化数据")
        
        # 选择缩放器类型
        if scaler_type.lower() == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type.lower() == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的缩放器类型: {scaler_type}")
        
        # 拟合并转换训练数据
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 转换测试数据
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("数据标准化完成")
        return X_train_scaled, X_test_scaled
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建新特征
        
        Args:
            df (pd.DataFrame): 原始数据框
        
        Returns:
            pd.DataFrame: 添加新特征后的数据框
        """
        logger.info("开始特征工程...")
        
        # 复制数据框以避免修改原始数据
        df_new = df.copy()
        
        # 根据配置创建特征
        feature_engineering_config = self.config.get('feature_engineering', {})
        
        # 1. 多项式特征
        if feature_engineering_config.get('polynomial_features', False):
            logger.info("创建多项式特征...")
            
            # 获取要创建多项式特征的列
            poly_columns = feature_engineering_config.get('polynomial_columns', [])
            if not poly_columns:
                # 如果未指定，使用所有数值列（除目标列外）
                poly_columns = [col for col in df_new.select_dtypes(include=[np.number]).columns 
                               if col != self.target_column]
            
            # 创建多项式特征
            degree = feature_engineering_config.get('polynomial_degree', 2)
            
            for col in poly_columns:
                if col in df_new.columns:
                    for d in range(2, degree + 1):
                        df_new[f"{col}^{d}"] = df_new[col] ** d
                        logger.info(f"创建多项式特征: {col}^{d}")
        
        # 2. 交互特征
        if feature_engineering_config.get('interaction_features', False):
            logger.info("创建交互特征...")
            
            # 获取要创建交互特征的列
            interaction_columns = feature_engineering_config.get('interaction_columns', [])
            if not interaction_columns:
                # 如果未指定，使用所有数值列（除目标列外）
                interaction_columns = [col for col in df_new.select_dtypes(include=[np.number]).columns 
                                      if col != self.target_column]
            
            # 创建交互特征
            for i, col1 in enumerate(interaction_columns):
                if col1 not in df_new.columns:
                    continue
                    
                for col2 in interaction_columns[i+1:]:
                    if col2 not in df_new.columns:
                        continue
                        
                    df_new[f"{col1}*{col2}"] = df_new[col1] * df_new[col2]
                    logger.info(f"创建交互特征: {col1}*{col2}")
        
        # 3. 比率特征
        if feature_engineering_config.get('ratio_features', False):
            logger.info("创建比率特征...")
            
            # 获取要创建比率特征的列
            ratio_columns = feature_engineering_config.get('ratio_columns', [])
            if not ratio_columns:
                # 如果未指定，使用所有数值列（除目标列外）
                ratio_columns = [col for col in df_new.select_dtypes(include=[np.number]).columns 
                               if col != self.target_column]
            
            # 创建比率特征
            for i, col1 in enumerate(ratio_columns):
                if col1 not in df_new.columns:
                    continue
                    
                for col2 in ratio_columns[i+1:]:
                    if col2 not in df_new.columns:
                        continue
                    
                    # 避免除以零
                    df_new[f"{col1}/{col2}"] = df_new[col1] / (df_new[col2] + 1e-10)
                    logger.info(f"创建比率特征: {col1}/{col2}")
        
        # 4. 自定义特征
        custom_features = feature_engineering_config.get('custom_features', [])
        for feature in custom_features:
            name = feature.get('name')
            expression = feature.get('expression')
            
            if name and expression:
                try:
                    # 使用eval安全地评估表达式
                    df_new[name] = df_new.eval(expression)
                    logger.info(f"创建自定义特征: {name} = {expression}")
                except Exception as e:
                    logger.error(f"创建自定义特征 '{name}' 失败: {str(e)}")
        
        logger.info(f"特征工程完成，新数据形状: {df_new.shape}")
        return df_new
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据质量和完整性
        
        Args:
            df (pd.DataFrame): 数据框
        
        Returns:
            bool: 数据是否有效
        """
        logger.info("验证数据质量和完整性...")
        
        # 检查数据框是否为空
        if df.empty:
            logger.error("数据框为空")
            return False
        
        # 检查目标列是否存在
        if self.target_column not in df.columns:
            logger.error(f"目标列 '{self.target_column}' 不在数据框中")
            return False
        
        # 检查数据类型
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"列 '{col}' 不是数值类型")
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"检测到缺失值: \n{missing_values[missing_values > 0]}")
        
        # 检查数据范围
        for col in df.select_dtypes(include=[np.number]).columns:
            min_val = df[col].min()
            max_val = df[col].max()
            logger.info(f"列 '{col}' 的范围: [{min_val}, {max_val}]")
            
            # 检查是否有极端值
            if max_val / (min_val + 1e-10) > 1e6:
                logger.warning(f"列 '{col}' 可能存在极端值，范围比例过大")
        
        # 检查目标列的分布
        target_mean = df[self.target_column].mean()
        target_std = df[self.target_column].std()
        logger.info(f"目标列 '{self.target_column}' 的均值: {target_mean:.4f}, 标准差: {target_std:.4f}")
        
        # 检查特征相关性
        if len(df.columns) > 1:
            # 计算与目标的相关性
            correlations = df.corr()[self.target_column].drop(self.target_column)
            
            # 找出高相关性特征
            high_corr_features = correlations[abs(correlations) > 0.7]
            if not high_corr_features.empty:
                logger.info(f"与目标高度相关的特征: \n{high_corr_features}")
            
            # 检查特征间的相关性
            feature_corr = df.drop(columns=[self.target_column]).corr()
            for i in range(len(feature_corr.columns)):
                for j in range(i+1, len(feature_corr.columns)):
                    col1 = feature_corr.columns[i]
                    col2 = feature_corr.columns[j]
                    corr = feature_corr.iloc[i, j]
                    
                    if abs(corr) > 0.9:
                        logger.warning(f"特征 '{col1}' 和 '{col2}' 高度相关 (r={corr:.4f})")
        
        logger.info("数据验证完成")
        return True
    
    def prepare_data(self, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        完整的数据准备流程
        
        Args:
            data_path (str, optional): 数据文件路径，如果为None则使用配置中的路径
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        logger.info("开始完整的数据准备流程...")
        
        # 1. 加载数据
        df = self.load_data(data_path)
        
        # 2. 验证数据
        if not self.validate_data(df):
            raise ValueError("数据验证失败")
        
        # 3. 特征工程（如果启用）
        if self.config.get('enable_feature_engineering', False):
            df = self.create_features(df)
        
        # 4. 预处理数据
        df = self.preprocess_data(df)
        
        # 5. 分割数据
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # 6. 标准化数据
        scaler_type = self.config.get('scaler_type', 'standard')
        X_train_scaled, X_test_scaled = self.normalize_data(X_train, X_test, scaler_type)
        
        logger.info("数据准备流程完成")
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def save_preprocessor(self, output_dir: str) -> Dict[str, str]:
        """
        保存预处理器（缩放器、特征选择器等）
        
        Args:
            output_dir (str): 输出目录
        
        Returns:
            Dict[str, str]: 保存的预处理器文件路径
        """
        import pickle
        import datetime
        
        logger.info(f"保存预处理器到: {output_dir}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # 保存缩放器
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            saved_files['scaler'] = scaler_path
            logger.info(f"缩放器已保存: {scaler_path}")
        
        # 保存特征选择器
        if self.feature_selector is not None:
            selector_path = os.path.join(output_dir, f"feature_selector_{timestamp}.pkl")
            with open(selector_path, 'wb') as f:
                pickle.dump(self.feature_selector, f)
            saved_files['feature_selector'] = selector_path
            logger.info(f"特征选择器已保存: {selector_path}")
        
        # 保存缺失值填充器
        if self.imputer is not None:
            imputer_path = os.path.join(output_dir, f"imputer_{timestamp}.pkl")
            with open(imputer_path, 'wb') as f:
                pickle.dump(self.imputer, f)
            saved_files['imputer'] = imputer_path
            logger.info(f"缺失值填充器已保存: {imputer_path}")
        
        return saved_files
    
    def load_preprocessor(self, preprocessor_paths: Dict[str, str]) -> None:
        """
        加载预处理器
        
        Args:
            preprocessor_paths (Dict[str, str]): 预处理器文件路径字典
        """
        import pickle
        
        logger.info("加载预处理器...")
        
        # 加载缩放器
        if 'scaler' in preprocessor_paths and os.path.exists(preprocessor_paths['scaler']):
            with open(preprocessor_paths['scaler'], 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"缩放器已加载: {preprocessor_paths['scaler']}")
        
        # 加载特征选择器
        if 'feature_selector' in preprocessor_paths and os.path.exists(preprocessor_paths['feature_selector']):
            with open(preprocessor_paths['feature_selector'], 'rb') as f:
                self.feature_selector = pickle.load(f)
            logger.info(f"特征选择器已加载: {preprocessor_paths['feature_selector']}")
        
        # 加载缺失值填充器
        if 'imputer' in preprocessor_paths and os.path.exists(preprocessor_paths['imputer']):
            with open(preprocessor_paths['imputer'], 'rb') as f:
                self.imputer = pickle.load(f)
            logger.info(f"缺失值填充器已加载: {preprocessor_paths['imputer']}")
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型，必须有feature_importances_属性或coef_属性
            feature_names (List[str]): 特征名称列表
        
        Returns:
            pd.DataFrame: 特征重要性数据框
        """
        logger.info("计算特征重要性...")
        
        try:
            # 尝试获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
            else:
                logger.warning("模型没有feature_importances_或coef_属性，无法计算特征重要性")
                return pd.DataFrame()
            
            # 创建特征重要性数据框
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            logger.info(f"特征重要性计算完成，前5个重要特征: \n{importance_df.head()}")
            return importance_df
            
        except Exception as e:
            logger.error(f"计算特征重要性失败: {str(e)}", exc_info=True)
            return pd.DataFrame()