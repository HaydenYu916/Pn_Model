import numpy as np
import matplotlib.pyplot as plt
import do_mpc
import sys
import os
import datetime
import json
import pandas as pd
from datetime import datetime

# 保证可以import find_best_ppfd
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from find_optimal_conditions_multi_model import find_best_ppfd

# 1. 系统参数
k = 800  # 提升系统极限
tau = 2.0
Ts = 1.0        # 步长1分钟
N_horizon = 20
sim_steps = 100
update_interval = 15  # 15步=15min

# 2. CO2和温度设定（CO2固定，温度有漂移）
CO2_fixed = 400.0
T_base = 24.0
T_noise_std = 5  # 温度噪音标准差
pwm_max = 1.0  # 控制器最大输出

# 设定MPC优化debug开关，True时每次都生成图片和pareto解集
mpc_opt_debug = True

# 生成批次大文件夹名
batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
batch_dir = os.path.join(os.path.dirname(__file__), 'results', f'mpc_batch_{batch_timestamp}')
os.makedirs(batch_dir, exist_ok=True)

# 3. 初始化目标PPFD轨迹（每15步更新一次，初始为400）
ppfd_target = np.ones(sim_steps) * 400
T_traj = np.ones(sim_steps) * T_base
results = []
for i, t in enumerate(range(0, sim_steps, update_interval)):
    actual_len = min(update_interval, sim_steps - t)
    temp_noise = np.random.normal(0, T_noise_std, size=actual_len)
    T_traj[t:t+actual_len] = T_base + temp_noise
    # 用区间最后一步的温度作为优化输入
    T_input = T_traj[t+actual_len-1]
    result = find_best_ppfd(CO2_fixed, T_input, debug_mode=mpc_opt_debug, batch_dir=batch_dir)
    result_dir = result['result_dir']
    with open(os.path.join(result_dir, 'fit_knee_parameters.json'), 'r', encoding='utf-8') as f:
        knee_info = json.load(f)
    knee_ppfd = knee_info['knee_point']['PPFD']
    rb = knee_info['knee_point']['R:B']
    pn = knee_info['knee_point']['Pn']
    cled = knee_info['knee_point']['CLED']
    knee_ppfd = min(knee_ppfd, k * pwm_max)
    ppfd_target[t:t+actual_len] = knee_ppfd
    print(f"[MPC] t={t}min, knee point PPFD (clipped) = {knee_ppfd:.2f}, T_input = {T_input:.2f}")
    results.append({
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'CO2': CO2_fixed,
        'Temp': T_input,  # 记录区间最后一步温度
        'Target_PPFD': knee_ppfd,
        'Actual_PPFD': None,
        'R:B': rb,
        'Pn': pn,
        'CLED': cled
    })

# 4. 创建模型
model_type = 'continuous'
import do_mpc
model = do_mpc.model.Model(model_type)
ppfd = model.set_variable('_x', 'ppfd')
pwm = model.set_variable('_u', 'pwm')
ppfd_ref = model.set_variable('_tvp', 'ppfd_ref')
T_tvp = model.set_variable('_tvp', 'T')  # 新增温度tvp
T_base = 24.0  # 保证T_base一致
beta = 1.0  # 温度扰动系数
model.set_rhs('ppfd', (k * pwm - ppfd) / tau + beta * (T_tvp - T_base))
model.setup()

# 5. 创建MPC控制器
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': N_horizon,
    't_step': Ts,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)
mpc.bounds['lower', '_u', 'pwm'] = 0.0
mpc.bounds['upper', '_u', 'pwm'] = 1.0
mterm = (ppfd - ppfd_ref) ** 2
lterm = (ppfd - ppfd_ref) ** 2 + 0.01 * (pwm) ** 2
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(pwm=0.01)

def tvp_fun(t_now):
    tvp_template = mpc.get_tvp_template()
    for i in range(N_horizon+1):
        step = int(t_now + i)
        if step >= sim_steps:
            step = sim_steps - 1
        tvp_template['_tvp', i, 'ppfd_ref'] = ppfd_target[step]
        tvp_template['_tvp', i, 'T'] = T_traj[step]  # 传递温度
    return tvp_template
mpc.set_tvp_fun(tvp_fun)
mpc.setup()

def tvp_fun_sim(t_now):
    tvp_template = simulator.get_tvp_template()
    step = int(t_now)
    if step >= sim_steps:
        step = sim_steps - 1
    tvp_template['ppfd_ref'] = ppfd_target[step]
    tvp_template['T'] = T_traj[step]  # 传递温度
    return tvp_template

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=Ts)
simulator.set_tvp_fun(tvp_fun_sim)
simulator.setup()

# 6. 闭环仿真（无噪音、无CO2扰动、无事件触发，温度有漂移）
ppfd_log = []
pwm_log = []
T_log = []
x0 = np.array([0.0])
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()

for t in range(sim_steps):
    u0 = mpc.make_step(x0)
    pwm_val = float(u0[0])
    x0 = simulator.make_step(np.array([[pwm_val]]))
    ppfd_model = float(x0[0])
    ppfd_log.append(ppfd_model)
    pwm_log.append(pwm_val)
    T_log.append(T_traj[t])

# 仿真结束后，补全每个区间的实际PPFD
for i, t in enumerate(range(0, sim_steps, update_interval)):
    actual_len = min(update_interval, sim_steps - t)
    if len(ppfd_log) >= t + actual_len:
        results[i]['Actual_PPFD'] = np.mean(ppfd_log[t:t+actual_len])
    else:
        results[i]['Actual_PPFD'] = None

# 7. 结果可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(ppfd_log, label='PPFD')
ax1.plot(ppfd_target, '--', label='Target')
ax1.set_ylabel('PPFD (μmol/m²/s)')
ax1.legend()
ax1.set_title('MPC closed-loop simulation with temperature drift (re-optimizing target PPFD every 15 min)')

ax2.plot(T_log, label='Temperature (with noise)', color='tab:red')
ax2.set_ylabel('Temperature (°C)')
ax2.set_xlabel('Time step (min)')
ax2.legend()
ax2.set_title('Simulated Temperature Trajectory (with noise)')

plt.tight_layout()
plt.show()

    # 保存区间优化结果为csv（追加模式）
csv_path = 'mpc_optimal_results_with_env.csv'
print('csv will be saved to:', os.path.abspath(csv_path))
df_new = pd.DataFrame(results)
print('df_new rows:', len(df_new))

def is_file_empty(path):
    return os.path.exists(path) and os.path.getsize(path) == 0

if os.path.exists(csv_path):
    print('file exists, size before:', os.path.getsize(csv_path))
if os.path.exists(csv_path) and not is_file_empty(csv_path):
    df_old = pd.read_csv(csv_path)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(csv_path, index=False)
else:
    df_new.to_csv(csv_path, index=False)
if os.path.exists(csv_path):
    print('file size after:', os.path.getsize(csv_path))
print('区间优化结果已保存为 mpc_optimal_results_with_env.csv') 