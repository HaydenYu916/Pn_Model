"""
DGPModel: 深度高斯过程模型
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import torch
    import gpytorch
    _has_gpytorch = True
except ImportError:
    _has_gpytorch = False

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

class DGPModel(BaseEstimator, RegressorMixin):
    """
    深度高斯过程（DGP）模型，支持多层GPR近似实现。
    若安装了GPyTorch则可用更高效的实现。
    """
    def __init__(self, n_layers=2, kernel=None, alpha=1e-6, random_state=None, **kwargs):
        self.n_layers = n_layers
        self.kernel = kernel
        self.alpha = alpha
        self.random_state = random_state
        self.kwargs = kwargs
        self.models = []
        self._is_fitted = False
        self.fitted = False  # 添加fitted属性
        self._use_gpytorch = _has_gpytorch

    def fit(self, X, y):
        X_ = np.array(X)
        y_ = np.array(y)
        self.models = []
        X_input = X_
        iterator = range(self.n_layers)
        if _has_tqdm:
            iterator = tqdm(iterator, desc="[DGP] 拟合层", unit="层")
        for i in iterator:
            if self.kernel is None:
                kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e3))
            else:
                kernel = self.kernel
            model = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, random_state=self.random_state, optimizer_params={'maxiter': 2000})
            model.fit(X_input, y_)
            self.models.append(model)
            # 逐层输出作为下一层输入
            X_input = model.predict(X_input).reshape(-1, 1)
        self._is_fitted = True
        self.fitted = True  # 设置fitted属性
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("DGPModel 尚未训练，请先调用 fit 方法。")
        X_input = np.array(X)
        for model in self.models:
            X_input = model.predict(X_input).reshape(-1, 1)
        return X_input.ravel()

    def get_params(self, deep=True):
        return {
            'n_layers': self.n_layers,
            'kernel': self.kernel,
            'alpha': self.alpha,
            'random_state': self.random_state,
            **self.kwargs
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self 