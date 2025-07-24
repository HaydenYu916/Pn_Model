import sys
import os
sys.path.append(os.path.abspath('../ML_Framework'))
sys.path.append(os.path.abspath('../ML_Framework/models'))

import pickle
import numpy as np

class PnModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model(model_path)
    def load_model(self, path):
        print(f"加载模型: {path}")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print("模型字典的key:", model.keys())
        print("模型对象类型:", type(model['model']))
        print("模型metadata:", model.get('metadata', '无'))
        return model
    def predict(self, ppfd, co2, rb, temp):
        # 注意特征顺序：['PPFD', 'CO2', 'T', 'R:B']
        X = np.array([[ppfd, co2, temp, rb]])
        scaler = self.model.get('scaler', None)
        if scaler is not None:
            X = scaler.transform(X)
            print("归一化后输入特征:", X)
        else:
            print("未找到scaler，直接用原始输入")
        y = self.model['model'].predict(X)
        print("预测输出:", y)
        return y[0]

if __name__ == "__main__":
    model_path = '../ML_Framework/results/GPR_CMAES_20250717_114342/models/optimized_gpr_model.pkl'
    pn_model = PnModelLoader(model_path)
    ppfd, co2, rb, temp = 500, 400, 0.88, 24
    pn = pn_model.predict(ppfd, co2, rb, temp)
    print(f"Pn预测结果：{pn}")