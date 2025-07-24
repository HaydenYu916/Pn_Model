import numpy as np
from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. PnModelLoader：负责加载pn模型
class PnModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model(model_path)
    def load_model(self, path):
        # TODO: 加载pkl模型，后续可扩展
        print(f"加载模型: {path}")
        return None
    def switch_model(self, new_path):
        self.model_path = new_path
        self.model = self.load_model(new_path)

# 2. ExperimentRunner：根据温度、CO2运行模型，输出解集
class ExperimentRunner:
    def __init__(self, pn_model):
        self.pn_model = pn_model
    def run(self, temp, co2):
        # TODO: 调用run_experiment.py的核心逻辑，输入温度/CO2，输出解集
        print(f"运行实验: temp={temp}, co2={co2}")
        # 返回模拟解集（实际应为模型输出）
        return {'solutions': np.random.rand(10, 3)}

# 3. KneeAnalyzer：分析解集，输出ppfd
class KneeAnalyzer:
    def __init__(self):
        pass
    def analyze(self, solutions):
        # TODO: 调用fit.py的核心逻辑，输入解集，输出ppfd
        print(f"分析解集，获得ppfd")
        # 返回模拟ppfd（实际应为分析结果）
        return float(np.mean(solutions['solutions'][:,0]))

# 4. 主循环
if __name__ == "__main__":
    # 参数
    model_path = '../ML_Framework/results/GPR_CMAES_20250717_114342/models/optimized_gpr_model.pkl'
    sim_steps = 50
    
    # 初始化
    pn_model = PnModelLoader(model_path)
    experiment = ExperimentRunner(pn_model)
    analyzer = KneeAnalyzer()
    
    # ====== MPC仿真参数与初始化 ======
    k = 500
    tau = 2.0
    Ts = 1.0
    N_horizon = 10
    
    # 创建do-mpc模型
    model = Model('continuous')
    ppfd = model.set_variable('_x', 'ppfd')
    pwm = model.set_variable('_u', 'pwm')
    ppfd_ref = model.set_variable('_tvp', 'ppfd_ref')
    model.set_rhs('ppfd', (k * pwm - ppfd) / tau)
    model.setup()

    mpc = MPC(model)
    mpc.set_param(n_horizon=N_horizon, t_step=Ts, store_full_solution=True)
    mpc.bounds['lower', '_u', 'pwm'] = 0.0
    mpc.bounds['upper', '_u', 'pwm'] = 1.0
    mterm = (ppfd - ppfd_ref) ** 2
    lterm = (ppfd - ppfd_ref) ** 2 + 0.01 * (pwm) ** 2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(pwm=0.01)

    # ====== 先定义mpc_target_ppfd，确保tvp_fun可用 ======
    mpc_target_ppfd = 400

    # TVP传递函数（每步都用最新ppfd目标）
    def tvp_fun(t_now):
        tvp_template = mpc.get_tvp_template()
        for i in range(N_horizon+1):
            tvp_template['_tvp', i, 'ppfd_ref'] = mpc_target_ppfd
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    simulator = Simulator(model)
    simulator.set_param(t_step=Ts)
    def tvp_fun_sim(t_now):
        tvp_template = simulator.get_tvp_template()
        tvp_template['ppfd_ref'] = mpc_target_ppfd
        return tvp_template
    simulator.set_tvp_fun(tvp_fun_sim)
    simulator.setup()

    # ====== 热漂移与事件触发参数 ======
    alpha = 0.05   # PWM加热系数
    beta = 0.01    # 散热系数
    T_env = 25.0   # 环境温度
    T_ref = 25.0   # 参考温度
    gamma = 0.005  # 热漂移系数
    delta_T = 0.5         # 温度变化阈值（°C）
    delta_CO2 = 100       # CO2变化阈值（ppm）
    interval_steps = 5    # 时间间隔（步）
    epsilon = 100         # 误差积累阈值

    # 日志
    ppfd_log, pwm_log, T_log, CO2_log, trigger_log = [], [], [], [], []
    T_now = T_last = T_env
    CO2_now = CO2_last = 400.0
    last_trigger_time = 0
    error_accum = 0

    x0 = np.array([0.0])
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()
    # mpc_target_ppfd = 400 # This line is now redundant as it's defined globally

    for t in range(sim_steps):
        # 1. 热漂移：温度随PWM升高
        if t > 0:
            T_now = T_now + alpha * pwm_val - beta * (T_now - T_env) + np.random.normal(0, 0.2)
        else:
            T_now = T_env
        CO2_now += np.random.normal(0, 10)

        # 2. 用当前温度、CO2调用Pn模型和knee分析，得到ppfd目标
        # solutions = experiment.run(T_now, CO2_now)
        # mpc_target_ppfd = analyzer.analyze(solutions)
        # 临时用合理模拟目标
        mpc_target_ppfd = 400 + 30 * np.sin(2 * np.pi * t / 30)

        # 3. 用ppfd作为MPC新目标，运行一步MPC仿真
        u0 = mpc.make_step(x0)
        pwm_val = float(u0[0])
        x0 = simulator.make_step(u0)
        ppfd_model = float(x0[0])
        ppfd_actual = ppfd_model * (1 - gamma * (T_now - T_ref))

        # 4. 事件触发机制
        error_accum += abs(mpc_target_ppfd - ppfd_actual)
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

        # 5. 日志
        print(f"Step {t}: T={T_now:.2f}, CO2={CO2_now:.2f}, ppfd目标={mpc_target_ppfd:.2f}, PWM={pwm_val:.2f}, PPFD={ppfd_actual:.2f}")
        ppfd_log.append(ppfd_actual)
        pwm_log.append(pwm_val)
        T_log.append(T_now)
        CO2_log.append(CO2_now)

    # ====== 动态可视化 ======
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    line_ppfd, = ax[0].plot([], [], label='PPFD', color='tab:blue')
    line_target, = ax[0].plot([], [], '--', label='Target', color='tab:orange')
    ax[0].set_ylabel('PPFD (μmol/m²/s)')
    ax[0].legend()
    line_pwm, = ax[1].plot([], [], label='PWM', color='tab:blue')
    ax[1].set_ylabel('PWM (0-1)')
    ax[1].legend()
    ax3_1 = ax[2]
    ax3_2 = ax3_1.twinx()
    line_temp, = ax3_1.plot([], [], label='Temp (°C)', color='tab:blue')
    line_co2, = ax3_2.plot([], [], label='CO2 (ppm)', color='tab:orange')
    ax3_1.set_ylabel('Temp (°C)')
    ax3_2.set_ylabel('CO2 (ppm)')
    ax3_1.legend(loc='upper left')
    ax3_2.legend(loc='upper right')
    ax3_1.set_xlabel('Time step')
    for t in trigger_log:
        ax[0].axvline(t, color='r', linestyle=':', alpha=0.3)
        ax[1].axvline(t, color='r', linestyle=':', alpha=0.3)
        ax3_1.axvline(t, color='r', linestyle=':', alpha=0.3)
    fig.suptitle('MPC闭环仿真（含热漂移与事件触发）')

    def update(frame):
        x = np.arange(frame+1)
        line_ppfd.set_data(x, ppfd_log[:frame+1])
        line_target.set_data(x, [mpc_target_ppfd]*(frame+1))
        ax[0].set_xlim(0, sim_steps)
        ax[0].set_ylim(min(ppfd_log)-20, max(ppfd_log)+20)
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