"""
配置管理模块
Configuration Management Module
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class DataConfig:
    """数据配置类"""
    data_path: str = "Data/averaged_data.csv"
    features: list = None
    target: str = "Pn_avg"
    test_size: float = 0.2
    random_state: int = 42
    normalize_method: str = "standard"  # "standard", "minmax", "custom"
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['PPFD', 'CO2', 'T', 'R:B']

@dataclass
class ModelConfig:
    """模型配置类"""
    model_type: str = "LSSVR"  # "SVR", "LSSVR", "GPR", "DGP"
    kernel: str = "rbf"
    # SVR 参数
    C: float = 1.0
    epsilon: float = 0.1
    gamma: float = 0.001
    # LSSVR 参数
    sigma2: float = 1.0
    # GPR 参数
    alpha: float = 1e-10
    n_restarts_optimizer: int = 10
    length_scale: float = 1.0
    constant_value: float = 1.0
    noise_level: float = 1e-10
    # DGP 参数
    n_layers: int = 2
    dgp_alpha: float = 1e-6

@dataclass
class OptimizationConfig:
    """优化配置类"""
    optimizer_type: str = "GA"  # "GA", "PSO", "TPE", "RANDOM", "IBOA", "CMAES"
    # GA 参数
    population_size: int = 20
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    # PSO 参数
    n_particles: int = 20
    n_iterations: int = 50
    w: float = 0.9  # 惯性权重
    c1: float = 2.0  # 个体学习因子
    c2: float = 2.0  # 群体学习因子
    # Optuna 优化器参数
    n_trials: int = 50  # 试验次数（用于 TPE, RANDOM 和 CMAES）
    # IBOA 优化器参数
    n_butterflies: int = 20      # 蝴蝶数量
    sensory_modality: float = 0.01 # 感觉模态参数
    power_exponent: float = 0.1    # 幂指数参数
    switch_probability: float = 0.8 # 切换概率
    # CMA-ES 优化器参数
    cmaes_population_size: Optional[int] = None  # CMA-ES种群大小（None表示自动设置）
    cmaes_sigma0: Optional[float] = 0.3         # 初始步长（可选，默认为参数范围的1/3）
    cmaes_seed: Optional[int] = 42              # 随机种子（可选，用于结果复现）
    # 参数边界
    param_bounds: Dict[str, list] = None
    
    def __post_init__(self):
        if self.param_bounds is None:
            self.param_bounds = {
                # SVR参数边界
                'C': [0.1, 100.0],
                'epsilon': [0.001, 1.0],
                'gamma': [1e-6, 1.0],
                # LSSVR参数边界
                'sigma2': [0.1, 100.0],
                # GPR参数边界
                'alpha': [1e-8, 1.0],
                'length_scale': [1e-2, 1e2],
                'constant_value': [1e-2, 1e2],
                'noise_level': [1e-8, 1.0],
                'n_restarts_optimizer': [5, 20]
            }

@dataclass
class EvaluationConfig:
    """评估配置类"""
    cv_folds: int = 5
    metrics: list = None
    plot_size: tuple = (12, 8)
    save_plots: bool = True
    plot_format: str = "png"
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['rmse', 'r2', 'mae']

@dataclass
class ExperimentConfig:
    """实验配置类"""
    name: str = "photosynthesis_prediction"
    results_dir: str = "results"
    models_dir: str = "models"
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    log_file: str = "experiment.log"

@dataclass
class Config:
    """主配置类"""
    data: DataConfig = None
    model: ModelConfig = None
    optimization: OptimizationConfig = None
    evaluation: EvaluationConfig = None
    experiment: ExperimentConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()

def load_config(config_path: str) -> Config:
    """从文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.json'):
            config_dict = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError("不支持的配置文件格式，请使用 .json 或 .yaml")
    
    # 递归转换配置字典
    def dict_to_config(d: Dict[str, Any]) -> Config:
        config = Config()
        if 'data' in d:
            config.data = DataConfig(**d['data'])
        if 'model' in d:
            config.model = ModelConfig(**d['model'])
        if 'optimization' in d:
            config.optimization = OptimizationConfig(**d['optimization'])
        if 'evaluation' in d:
            config.evaluation = EvaluationConfig(**d['evaluation'])
        if 'experiment' in d:
            config.experiment = ExperimentConfig(**d['experiment'])
        return config
    
    return dict_to_config(config_dict)

def save_config(config: Config, config_path: str):
    """保存配置到文件"""
    config_dict = asdict(config)
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.endswith('.json'):
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError("不支持的配置文件格式，请使用 .json 或 .yaml")

def create_default_config() -> Config:
    """创建默认配置"""
    return Config() 