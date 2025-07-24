"""
植物光合作用预测 - 模块化机器学习框架
Photosynthesis Prediction - Modular ML Framework

主要功能:
- 支持多种机器学习模型 (SVR, LSSVR, GPR)
- 支持多种优化算法 (GA, PSO)
- 模块化设计，易于扩展
- 完整的数据处理和模型评估流程
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# 导入主要模块
from .models import SVRModel, LSSVRModel, GPRModel
from .optimizers import GeneticAlgorithm, ParticleSwarmOptimization
from .data_processing import DataProcessor
from .evaluation import ModelEvaluator
from .config.config import load_config
from .utils import save_model, load_model

__all__ = [
    'SVRModel', 'LSSVRModel', 'GPRModel',
    'GeneticAlgorithm', 'ParticleSwarmOptimization',
    'DataProcessor', 'ModelEvaluator',
    'load_config', 'save_model', 'load_model'
] 