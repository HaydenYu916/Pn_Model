import os
import time
import pandas as pd
from datetime import datetime
import subprocess
import shutil

# ========== 配置 ==========
CO2_CSV_PATH = '/home/pi/mpc/results/test.csv'      # CO2数据路径
TEMP_CSV_PATH = '/home/pi/mpc/results/riotee.csv'   # 温度数据路径
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
RB_RESULT_CSV = os.path.join(RESULT_DIR, 'rb_result.csv')

# ========== 工具函数 ==========
def get_latest_value(csv_path):
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    # 假设CO2和温度分别在第一列
    return float(last_row[0])

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ========== 主循环 ==========
def main_loop():
    ensure_dir(RESULT_DIR)
    while True:
        try:
            # 1. 读取最新CO2和温度
            co2 = get_latest_value(CO2_CSV_PATH)
            temp = get_latest_value(TEMP_CSV_PATH)
            print(f"[{datetime.now()}] 最新CO2: {co2}, 最新温度: {temp}")

            # 2. 调用优化脚本
            # 这里用subprocess调用pymoo/find_optimal_conditions_multi_model.py
            # 通过echo输入co2和temp
            optimize_cmd = f"echo '{co2}\n{temp}' | python3 ../pymoo/find_optimal_conditions_multi_model.py"
            print(f"运行优化: {optimize_cmd}")
            subprocess.run(optimize_cmd, shell=True, check=True)

            # 3. 找到最新的pareto_solutions.csv
            results_root = os.path.join(os.path.dirname(__file__), '../pymoo/results')
            all_dirs = [d for d in os.listdir(results_root) if d.startswith('paper_optimal_conditions_')]
            all_dirs = sorted(all_dirs, key=lambda x: os.path.getmtime(os.path.join(results_root, x)), reverse=True)
            latest_dir = os.path.join(results_root, all_dirs[0])
            pareto_csv = os.path.join(latest_dir, 'pareto_solutions.csv')

            # 4. 调用拟合与knee分析
            fit_cmd = f"python3 ../Optimal/fit.py '{pareto_csv}' '{latest_dir}'"
            print(f"运行拟合分析: {fit_cmd}")
            subprocess.run(fit_cmd, shell=True, check=True)

            # 5. 读取拟合结果，提取knee点r:b
            import json
            fit_json = os.path.join(latest_dir, 'fit_knee_parameters.json')
            with open(fit_json, 'r', encoding='utf-8') as f:
                fit_params = json.load(f)
            knee = fit_params['knee_point']
            rb = knee.get('R:B', None)
            # 6. 结果写入CSV
            with open(RB_RESULT_CSV, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now()},{co2},{temp},{rb}\n")
            print(f"已保存r:b={rb}到{RB_RESULT_CSV}")
        except Exception as e:
            print(f"[错误] {e}")
        # 7. 等待1分钟
        time.sleep(60)

if __name__ == '__main__':
    main_loop() 