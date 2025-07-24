import sys
import os
import time
import pandas as pd
from datetime import datetime
import subprocess
import shutil
import yaml
import json

# 自动将 ML_Framework 目录加入 sys.path，兼容 Mac 和 Linux
ml_framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML_Framework'))
if ml_framework_path not in sys.path:
    sys.path.insert(0, ml_framework_path)

# ========== 配置 ==========
CO2_CSV_PATH = os.path.join(os.path.dirname(__file__), 'test.csv')      # CO2 数据路径
TEMP_CSV_PATH = os.path.join(os.path.dirname(__file__), 'riotee.csv')   # 温度数据路径
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
RB_RESULT_CSV = os.path.join(RESULT_DIR, 'rb_result.csv')

# ========== 工具函数 ==========
def get_latest_co2(csv_path):
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    return float(last_row['co2'])  # 用列名最稳妥，或根据实际列名调整

def get_latest_temp(csv_path):
    df = pd.read_csv(csv_path, header=1)  # 指定表头在第二行
    last_row = df.iloc[-1]
    return float(last_row['temperature'])  # 用实际的列名替换

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ========== 主循环 ==========
def main_loop():
    ensure_dir(RESULT_DIR)

    # 检查结果文件是否存在，不存在则写入表头
    if not os.path.exists(RB_RESULT_CSV):
        with open(RB_RESULT_CSV, 'w', encoding='utf-8') as f:
            f.write('id,timestamp,co2,temp,rb,ppfd,pn\n')

    # 读取模型路径并拼接 PYTHONPATH
    config_path = os.path.join(os.path.dirname(__file__), '../pymoo/moo_optimization_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_path = config['model']['model_path']
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # ========== 主循环 ==========
    while True:
        try:
            # 1. 读取最新 CO2 和温度
            co2 = get_latest_co2(CO2_CSV_PATH)
            temp = get_latest_temp(TEMP_CSV_PATH)
            print(f"[{datetime.now()}] 最新CO2: {co2}, 最新温度: {temp}")

            # 2. 调用优化脚本（使用 subprocess 标准输入方式）
            env = os.environ.copy()
            # 设置PYTHONPATH为项目根目录和ML_Framework目录，兼容Mac和Linux
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            ml_framework_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML_Framework'))
            env["PYTHONPATH"] = f"{project_root}:{ml_framework_dir}:" + env.get("PYTHONPATH", "")
            optimize_cmd = [
                "python3",
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../MPC_Test/find_optimal_conditions_multi_model.py"))
            ]
            input_str = f"{co2}\n{temp}\n"
            print("运行优化命令:")
            print(f"PYTHONPATH={env['PYTHONPATH']} python3 ../pymoo/find_optimal_conditions_multi_model.py")

            subprocess.run(optimize_cmd, input=input_str, text=True, check=True, env=env)

            # 3. 找到最新的 pareto_solutions.csv
            results_root = os.path.join(os.path.dirname(__file__), '../pymoo/results')
            all_dirs = [d for d in os.listdir(results_root) if d.startswith('paper_optimal_conditions_')]
            all_dirs = sorted(all_dirs, key=lambda x: os.path.getmtime(os.path.join(results_root, x)), reverse=True)
            latest_dir = os.path.join(results_root, all_dirs[0])
            pareto_csv = os.path.join(latest_dir, 'pareto_solutions.csv')

            # 4. 拟合 knee 点
            fit_cmd = [
                "python3",
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../Optimal/fit.py")),
                pareto_csv,
                latest_dir
            ]
            print(f"运行拟合分析: {' '.join(fit_cmd)}")
            subprocess.run(fit_cmd, check=True)

            # 5. 读取拟合结果并提取 R:B、PPFD、Pn
            fit_json = os.path.join(latest_dir, 'fit_knee_parameters.json')
            with open(fit_json, 'r', encoding='utf-8') as f:
                fit_params = json.load(f)
            knee = fit_params['knee_point']
            rb = knee.get('R:B', None)
            ppfd = knee.get('PPFD', None)
            pn = knee.get('Pn', None)

            # 6. 结果写入 CSV（带 id）
            # 自动获取当前最大 id
            if os.path.exists(RB_RESULT_CSV):
                with open(RB_RESULT_CSV, 'r', encoding='utf-8') as f_id:
                    lines = f_id.readlines()
                    last_id = 0
                    for line in reversed(lines[1:]):  # 跳过表头
                        first_field = line.split(',')[0].strip()
                        if first_field.isdigit():
                            last_id = int(first_field)
                            break
            else:
                last_id = 0
            new_id = last_id + 1
            with open(RB_RESULT_CSV, 'a', encoding='utf-8') as f:
                f.write(f"{new_id},{datetime.now()},{co2},{temp},{rb},{ppfd},{pn}\n")
            print(f"已保存 id={new_id}, r:b={rb}, ppfd={ppfd}, pn={pn} 到 {RB_RESULT_CSV}")
        except Exception as e:
            print(f"[错误] {e}")

        # 7. 等待 1 分钟
        time.sleep(60)

if __name__ == '__main__':
    main_loop()
