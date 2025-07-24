#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basil光合速率预测器
使用LSSVR模型预测光合速率

模型条件: CO2=400ppm, R:B=0.83
输入特征: PPFD (光量子密度, umol/m2/s) + 温度 (°C)
输出: Pn (光合速率, umol/m2/s)
"""

import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==================== 设置预测变量 ====================
PPFD = 500  # 光量子密度 (umol/m2/s)
TEMPERATURE = 25  # 温度 (°C)
# =====================================================


class PhotosynthesisPredictor:
    """光合速率预测器"""

    def __init__(self):
        # 修正模型文件路径 - 使用相对路径
        self.model_path = "lssvr_ga_trained_model.pkl"
        self.scaler_path = (
            "standard_scaler_lssvr_ppfd_t_co2_400_rb_083.pkl"
        )

        # 模型参数
        self.alpha = None
        self.b = None
        self.X_train = None
        self.gamma = None
        self.sigma2 = None
        self.scaler = None
        self.is_loaded = False

        self.load_model()

    def load_model(self):
        """加载模型"""
        try:
            # 加载LSSVR模型
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)

            model_params = model_data["model_params"]
            self.alpha = np.array(model_params["alpha"])
            self.b = model_params["b"]
            self.X_train = np.array(model_params["X_train"])
            self.gamma = model_params["gamma"]
            self.sigma2 = model_params["sigma2"]

            # 加载标准化器
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            self.is_loaded = True
            print("模型加载成功")
            print("模型条件: CO2=400ppm, R:B=0.83")
            print(f"模型参数: gamma={self.gamma:.4f}, sigma2={self.sigma2:.4f}")

        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
        except Exception as e:
            print(f"加载模型出错: {e}")

    def _rbf_kernel(self, X1, X2):
        """RBF核函数"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        dist_matrix = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return np.exp(-self.gamma * dist_matrix)

    def predict(self, ppfd, temperature):
        """预测光合速率"""
        if not self.is_loaded:
            print("模型未加载")
            return None

        # 输入验证
        if ppfd < 0 or ppfd > 1500:
            print(f"警告: PPFD={ppfd} 超出建议范围 [0, 1000]")
        if temperature < 15 or temperature > 35:
            print(f"警告: 温度={temperature} 超出建议范围 [18, 30]")

        # 标准化
        input_array = np.array([[ppfd, temperature]])
        input_normalized = self.scaler.transform(input_array)

        # 预测
        K = self._rbf_kernel(input_normalized, self.X_train)
        prediction = np.dot(K, self.alpha) + self.b

        return float(prediction[0])


def main():
    """主函数"""
    print("Basil光合速率预测器")
    print("=" * 30)

    predictor = PhotosynthesisPredictor()

    if not predictor.is_loaded:
        return

    print("预测条件:\n")
    print(f"PPFD = {PPFD} umol/m2/s")
    print(f"温度 = {TEMPERATURE} °C")

    result = predictor.predict(PPFD, TEMPERATURE)

    if result is not None:
        print(f"\n预测结果: Pn = {result:.3f} umol/m2/s")


if __name__ == "__main__":
    main()
