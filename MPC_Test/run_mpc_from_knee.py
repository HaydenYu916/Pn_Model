import numpy as np
import matplotlib.pyplot as plt
import do_mpc
from do_mpc.graphics import Graphics
from matplotlib.animation import FuncAnimation, ImageMagickWriter

# 1. 系统参数
k = 500
tau = 2.0
Ts = 1.0        # 步长1分钟
N_horizon = 20
sim_steps = 100
update_interval = int(5 / Ts)  # 5分钟变一次PWM

# 2. 目标PPFD轨迹
ppfd_target = np.ones(sim_steps) * 400
ppfd_target[30:60] = 300
ppfd_target[60:] = 450

# 3. 创建模型
model_type = 'continuous'
model = do_mpc.model.Model(model_type)

ppfd = model.set_variable('_x', 'ppfd')
pwm = model.set_variable('_u', 'pwm')
ppfd_ref = model.set_variable('_tvp', 'ppfd_ref')

# 动态方程
model.set_rhs('ppfd', (k * pwm - ppfd) / tau)
model.setup()

# 4. 创建MPC控制器
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

# 5. TVP传递函数（严格官方写法）
def tvp_fun(t_now):
    tvp_template = mpc.get_tvp_template()
    for i in range(N_horizon+1):
        step = int(t_now + i)
        if step >= sim_steps:
            step = sim_steps - 1
        tvp_template['_tvp', i, 'ppfd_ref'] = ppfd_target[step]
    return tvp_template

mpc.set_tvp_fun(tvp_fun)
mpc.setup()

def tvp_fun_sim(t_now):
    tvp_template = simulator.get_tvp_template()
    step = int(t_now)
    if step >= sim_steps:
        step = sim_steps - 1
    tvp_template['ppfd_ref'] = ppfd_target[step]
    return tvp_template

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=Ts)
simulator.set_tvp_fun(tvp_fun_sim)
simulator.setup()

# ===== 触发机制参数 =====
delta_T = 0.5         # 温度变化阈值（°C）
delta_CO2 = 100       # CO2变化阈值（ppm）
interval_steps = 5   # 时间间隔（步，1步=1min）
epsilon = 100        # 误差积累阈值（PPFD单位）

# ===== 热漂移参数 =====
alpha = 0.05   # PWM加热系数（每步温度上升量）
beta = 0.01    # 散热系数
T_env = 25.0   # 环境温度
T_ref = 25.0   # 参考温度
# 每升高1°C，PPFD下降0.5%
gamma = 0.005  # 热漂移系数

# ===== 虚拟环境变量初始化 =====
T_now = T_last = T_env
CO2_now = CO2_last = 400.0
last_trigger_time = 0
error_accum = 0

# 8. 闭环仿真（带事件触发+热漂移）
ppfd_log = []
pwm_log = []
T_log = []
CO2_log = []
trigger_log = []

x0 = np.array([0.0])
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()

for t in range(sim_steps):
    # 每步都更新MPC控制量
    u0 = mpc.make_step(x0)
    pwm_val = float(u0[0])
    
    # 2. 更新温度（热漂移）
    T_now = T_now + alpha * pwm_val - beta * (T_now - T_env) + np.random.normal(0, 0.2)
    
    # 3. CO2随机游走
    CO2_now += np.random.normal(0, 10)
    
    # 4. 用仿真器计算理想PPFD
    x0 = simulator.make_step(np.array([[pwm_val]]))
    ppfd_model = float(x0[0])
    
    # 5. 热漂移修正PPFD
    ppfd_actual = ppfd_model * (1 - gamma * (T_now - T_ref))
    
    # 6. 累加误差
    ppfd_ref = ppfd_target[t]
    error_accum += abs(ppfd_ref - ppfd_actual)
    
    # 7. 判断触发条件
    trigger = False
    if abs(T_now - T_last) > delta_T:
        trigger = True
    if abs(CO2_now - CO2_last) > delta_CO2:
        trigger = True
    if (t - last_trigger_time) >= interval_steps:
        trigger = True
    if error_accum > epsilon:
        trigger = True

    if trigger:
        T_last = T_now
        CO2_last = CO2_now
        last_trigger_time = t
        error_accum = 0
        trigger_log.append(t)
    
    ppfd_log.append(ppfd_actual)
    pwm_log.append(pwm_val)
    T_log.append(T_now)
    CO2_log.append(CO2_now)

# 9. 结果可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(ppfd_log, label='PPFD')
plt.plot(ppfd_target, '--', label='Target')
plt.ylabel('PPFD (μmol/m²/s)')
plt.legend()
for t in trigger_log:
    plt.axvline(t, color='r', linestyle=':', alpha=0.3)
plt.title('红线为MPC事件触发点')

plt.subplot(3,1,2)
plt.plot(pwm_log, label='PWM')
plt.ylabel('PWM (0-1)')
plt.legend()
for t in trigger_log:
    plt.axvline(t, color='r', linestyle=':', alpha=0.3)

plt.subplot(3,1,3)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(T_log, label='Temp (°C)', color='tab:blue')
ax2.plot(CO2_log, label='CO2 (ppm)', color='tab:orange')
ax1.set_ylabel('Temp (°C)')
ax2.set_ylabel('CO2 (ppm)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
for t in trigger_log:
    ax1.axvline(t, color='r', linestyle=':', alpha=0.3)
plt.xlabel('Time step (min)')
plt.tight_layout()
plt.show()

# 10. 动态可视化
fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# 1. PPFD
line_ppfd, = ax[0].plot([], [], label='PPFD', color='tab:blue')
line_target, = ax[0].plot([], [], '--', label='Target', color='tab:orange')
ax[0].set_ylabel('PPFD (μmol/m²/s)')
ax[0].legend()

# 2. PWM
line_pwm, = ax[1].plot([], [], label='PWM', color='tab:blue')
ax[1].set_ylabel('PWM (0-1)')
ax[1].legend()

# 3. 温度和CO2（双y轴）
ax3_1 = ax[2]
ax3_2 = ax3_1.twinx()
line_temp, = ax3_1.plot([], [], label='Temp (°C)', color='tab:blue')
line_co2, = ax3_2.plot([], [], label='CO2 (ppm)', color='tab:orange')
ax3_1.set_ylabel('Temp (°C)')
ax3_2.set_ylabel('CO2 (ppm)')
ax3_1.legend(loc='upper left')
ax3_2.legend(loc='upper right')
ax3_1.set_xlabel('Time step (min)')

# 红线为MPC事件触发点
for t in trigger_log:
    ax[0].axvline(t, color='r', linestyle=':', alpha=0.3)
    ax[1].axvline(t, color='r', linestyle=':', alpha=0.3)
    ax3_1.axvline(t, color='r', linestyle=':', alpha=0.3)

fig.suptitle('MPC仿真动态动画')

# 动画更新函数
def update(frame):
    x = np.arange(frame+1)
    line_ppfd.set_data(x, ppfd_log[:frame+1])
    line_target.set_data(x, ppfd_target[:frame+1])
    ax[0].set_xlim(0, sim_steps)
    ax[0].set_ylim(min(ppfd_log+list(ppfd_target))-20, max(ppfd_log+list(ppfd_target))+20)
    
    line_pwm.set_data(x, pwm_log[:frame+1])
    ax[1].set_xlim(0, sim_steps)
    ax[1].set_ylim(0, 1.05)
    
    line_temp.set_data(x, T_log[:frame+1])
    line_co2.set_data(x, CO2_log[:frame+1])
    ax3_1.set_xlim(0, sim_steps)
    ax3_1.set_ylim(min(T_log)-2, max(T_log)+2)
    ax3_2.set_ylim(min(CO2_log)-50, max(CO2_log)+50)
    return line_ppfd, line_target, line_pwm, line_temp, line_co2

anim = FuncAnimation(fig, update, frames=sim_steps, interval=100, blit=False, repeat=False)
plt.show()

# 如需保存为GIF，取消下方注释（需安装ImageMagick）
# gif_writer = ImageMagickWriter(fps=10)
# anim.save('mpc_simulation.gif', writer=gif_writer)
