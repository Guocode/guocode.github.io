"""Pixel Arena — 两个游戏的轻量物理环境（向量化）

Game 0  RUNNER  双足机器人赛跑：躯干刚体（平动 + 俯仰）+ 两条弹簧腿
Game 1  CAR     失控汽车驾驶：沿赛道 Frenet 坐标 + 随机轮子失控扰动

两个环境共享
  * 动作空间：4 个离散动作
  * 观测维度：14
  * step 频率：1/60 s（2 个 1/120 子步）
因此可以被同一个 MoE 策略网络同时驱动。

本文件中的常量与积分公式与浏览器端 assets/js/env.js 逐行对应，
用固定 seed 的动作序列可做 golden 校验。
"""

import numpy as np

# ================================================================= 共享常量
DT = 1.0 / 120.0
SUBSTEPS = 2
N_OBS = 14
N_ACT = 4
MAX_STEPS = 1500          # 单回合 25 s

ACT_NAMES = {
    0: ["STEP_L", "STEP_R", "LEAN_FWD", "LEAN_BACK"],
    1: ["STEER_L", "STEER_R", "ACCEL", "BRAKE"],
}

# ================================================================= Game 0 RUNNER
G = 20.0
MASS = 1.0
INERTIA = 0.45        # 躯干 + 腿的等效转动惯量
HIP_OFF = 0.22        # 质心到髋的距离
L0 = 0.55
K_LEG = 800.0       # 刚度足够大，双腿/单腿切换时身体只下沉 ~1 cm
C_LEG = 30.0
PUSH = 60.0           # 蹬地推进系数
V_CUT = 5.0           # 推进的速度上限：步频有生理极限，越快蹬地越使不上力
CHARGE_T = 0.30       # 蹬地能量耗尽时间：落地充满，支撑期间衰减
F_MAX = 90.0
# 俯仰控制是"设定目标倾角"的高层指令，由内部伺服持续执行，
# 因此不占用迈腿动作位——否则换步与平衡会互相抢占同一个动作。
KP_PITCH = 14.0       # 伺服比例增益
KD_PITCH = 2.6        # 伺服微分增益
TORQUE_MAX = 6.0      # 伺服力矩上限
C_ANG = 0.30          # 俯仰被动阻尼
PITCH_COUPLE = 0.35   # 蹬地推进对俯仰的耦合系数（躯干与摆臂会抵消大部分）
TILT_FWD = -0.35      # 动作 2：激进前倾，步幅更大更快
TILT_BACK = 0.12      # 动作 3：收身直立，更稳但慢
TILT_INIT = -0.15     # 初始目标倾角
DRAG = 1.4
SWING_T = 0.10          # 摆动相时长 -> 步频 ~10 Hz
STEP_LEN = 0.16         # 基础步幅
STEP_MAX = 0.21         # 步幅上限：脚相对髋的水平偏移受腿长几何约束
STEP_H = 0.15
LEAN_GAIN = 1.2
FALL_ANG = 0.90
MIN_Y = 0.34
V_MAX_RUN = 8.0

W_DIST = 1.0
W_ALIVE = 0.005
W_POSTURE = 0.03
FALL_PENALTY = 1.0

# ================================================================= Game 1 CAR
ACC = 6.0
BRAKE_FORCE = 9.0
CAR_DRAG = 0.22
V_MAX_CAR = 30.0
STEER_RATE = 2.4
STEER_MAX = 0.55
STEER_RETURN = 2.0
WHEELBASE = 2.6
ROAD_HALF = 3.4
KAPPA_SCALE = 80.0
LOOKAHEAD = (0.0, 12.0, 28.0, 50.0)
DISTURB_P = 0.010          # 每步触发"轮子失控"的概率
DISTURB_T_MIN = 0.4
DISTURB_T_MAX = 1.3
D_RATE = 0.95              # 失控偏航角速度 rad/s
D_LAT = 1.30               # 失控横向漂移 m/s
W_CENTER = 0.02            # 偏离车道中心惩罚
CRASH_PENALTY = 1.0
V_STALL = 0.6


def track_curvature(s):
    """赛道曲率 κ(s)：确定性解析函数，JS 端用同一公式。"""
    return 0.018 * np.sin(0.035 * s) + 0.010 * np.sin(0.011 * s + 1.7)


# ================================================================= 基类
class BaseEnv:
    task_id = -1

    def __init__(self, n=64, seed=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.steps = np.zeros(n, dtype=np.int32)
        self.done = np.ones(n, dtype=bool)

    def _roll(self, shape):
        return self.rng.uniform(0.0, 1.0, size=shape)


# ================================================================= Game 0
class RunnerEnv(BaseEnv):
    task_id = 0

    def __init__(self, n=64, seed=None, posture_penalty=True):
        super().__init__(n, seed)
        self.posture_penalty = posture_penalty
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.th = np.zeros(n)
        self.w = np.zeros(n)
        self.stance = np.ones((n, 2), dtype=bool)
        self.fx = np.zeros((n, 2))
        self.fy = np.zeros((n, 2))
        self.sw_t = np.zeros((n, 2))
        self.sw_x0 = np.zeros((n, 2))
        self.charge = np.ones((n, 2))    # 蹬地能量
        self.dragged = np.zeros((n, 2), dtype=bool)  # 该腿是否为被动拖离
        self.th_target = np.full(n, TILT_INIT)   # 目标俯仰角（伺服设定值）

    def _init(self, idx):
        k = len(idx)
        self.x[idx] = 0.0
        self.y[idx] = L0 * 0.94 + HIP_OFF + self.rng.uniform(-0.02, 0.02, size=k)
        self.vx[idx] = 0.0
        self.vy[idx] = 0.0
        self.th[idx] = self.rng.uniform(-0.05, 0.05, size=k)
        self.w[idx] = 0.0
        self.stance[idx] = True
        self.fx[idx, 0] = -0.12
        self.fx[idx, 1] = 0.12
        self.fy[idx] = 0.0
        self.sw_t[idx] = 0.0
        self.sw_x0[idx] = self.fx[idx]
        self.charge[idx] = 1.0
        self.dragged[idx] = False
        self.th_target[idx] = TILT_INIT
        self.done[idx] = False
        self.steps[idx] = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init(np.arange(self.n))
        return self.obs()

    def reset_idx(self, idx):
        """只重置已结束的回合，让未结束的回合跨 rollout 继续。"""
        idx = np.atleast_1d(np.asarray(idx))
        if len(idx):
            self._init(idx)

    def obs(self):
        hip_x = self.x + HIP_OFF * np.sin(self.th)
        u_l = np.clip(self.sw_t[:, 0] / SWING_T, 0.0, 1.0)
        u_r = np.clip(self.sw_t[:, 1] / SWING_T, 0.0, 1.0)
        sw_l = (~self.stance[:, 0]).astype(np.float64)
        sw_r = (~self.stance[:, 1]).astype(np.float64)
        return np.stack([
            self.y / L0,
            self.vx / V_MAX_RUN,
            self.vy / V_MAX_RUN,
            np.sin(self.th),
            np.cos(self.th),
            self.w / 6.0,
            sw_l, u_l * sw_l, (self.fx[:, 0] - hip_x) / L0, self.fy[:, 0] / L0,
            sw_r, u_r * sw_r, (self.fx[:, 1] - hip_x) / L0, self.fy[:, 1] / L0,
        ], axis=1).astype(np.float32)

    def _substep(self, trigger):
        start = self.stance & trigger
        if start.any():
            self.sw_x0 = np.where(start, self.fx, self.sw_x0)
            self.sw_t = np.where(start, 0.0, self.sw_t)
            self.stance = np.where(start, False, self.stance)
            self.dragged = np.where(start, False, self.dragged)

        hip_x = self.x + HIP_OFF * np.sin(self.th)
        hip_y = self.y - HIP_OFF * np.cos(self.th)
        rx, ry = hip_x - self.x, hip_y - self.y

        Fx = np.zeros(self.n)
        Fy = np.zeros(self.n)
        tau = np.zeros(self.n)

        for i in (0, 1):
            st = self.stance[:, i]
            sw = ~st
            dx = hip_x - self.fx[:, i]
            dy = hip_y - self.fy[:, i]
            L = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-3)
            ux, uy = dx / L, dy / L

            Ldot = (self.vx * dx + self.vy * dy) / L
            fmag = np.clip(K_LEG * (L0 - L) - C_LEG * Ldot, -F_MAX, F_MAX)
            # 蹬地推进：模拟腿部肌肉主动伸展。压缩越深蹬地越强，腿伸直后无推力，
            # 因此必须及时换步才能维持速度；落在身后的腿推进更强。
            compress = np.clip((L0 - L) / (0.15 * L0), 0.0, 1.0)
            rear = np.clip(dx / (0.30 * L0), 0.0, 1.0)
            # 水平方向完全由蹬地推进驱动（弹簧只负责垂直支撑），
            # 否则前脚的水平分力会形成无法摆脱的制动，机器人只会倒退。
            # 推进力 ∝ 该腿剩余的蹬地能量，站着不动能量耗尽后就跑不起来。
            fade = np.clip(1.0 - self.vx / V_CUT, 0.0, 1.0)
            fsx = PUSH * self.charge[:, i] * (0.35 + 0.65 * rear) * (0.30 + 0.70 * compress) * fade
            fsy = fmag * uy

            Fx += np.where(st, fsx, 0.0)
            Fy += np.where(st, fsy, 0.0)
            tau += np.where(st, rx * fsy - ry * fsx * PITCH_COUPLE, 0.0)

            # 被动拖离：腿被拉过头时脚自然离地换步（否则弹簧会向下拽倒机器人）。
            # 拖离不重置蹬地能量，所以主动换步才是维持速度的关键。
            other_st = self.stance[:, 1 - i]
            drag = st & (L > L0 * 1.08) & other_st
            self.stance[:, i] = np.where(drag, False, self.stance[:, i])
            self.sw_t[:, i] = np.where(drag, 0.0, self.sw_t[:, i])
            self.sw_x0[:, i] = np.where(drag, self.fx[:, i], self.sw_x0[:, i])
            self.charge[:, i] = np.where(drag, 0.0, self.charge[:, i])
            self.dragged[:, i] = np.where(drag, True, self.dragged[:, i])
            sw = ~self.stance[:, i]

            step_eff = STEP_LEN * (1.0 + LEAN_GAIN * np.maximum(0.0, -self.th)) + 0.05 * self.vx
            step_eff = np.clip(step_eff, 0.05, STEP_MAX)
            target = hip_x + step_eff
            t = np.where(sw, self.sw_t[:, i] + DT, self.sw_t[:, i])
            u = np.clip(t / SWING_T, 0.0, 1.0)
            nx = self.sw_x0[:, i] + (target - self.sw_x0[:, i]) * u
            ny = np.sin(np.pi * u) * STEP_H
            self.fx[:, i] = np.where(sw, nx, self.fx[:, i])
            self.fy[:, i] = np.where(sw, ny, 0.0)
            self.sw_t[:, i] = t
            landed = sw & (u >= 1.0)
            self.stance[:, i] = np.where(landed, True, self.stance[:, i])
            chg = np.maximum(self.charge[:, i] - DT / CHARGE_T, 0.0)
            self.charge[:, i] = np.where(landed, np.where(self.dragged[:, i], 0.0, 1.0), chg)
            self.dragged[:, i] = np.where(landed, False, self.dragged[:, i])

        Fx -= DRAG * self.vx
        # 姿态伺服：追踪玩家设定的目标倾角
        tau_ctrl = np.clip(KP_PITCH * (self.th_target - self.th) - KD_PITCH * self.w,
                           -TORQUE_MAX, TORQUE_MAX)
        tau += tau_ctrl - C_ANG * self.w

        self.w += (tau / INERTIA) * DT
        self.th += self.w * DT
        self.vx += (Fx / MASS) * DT
        self.vy += (Fy / MASS - G) * DT
        self.x += self.vx * DT
        self.y += self.vy * DT

    def step(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        trigger = np.stack([a == 0, a == 1], axis=1)
        self.th_target = np.where(a == 2, TILT_FWD,
                                  np.where(a == 3, TILT_BACK, self.th_target))

        x0 = self.x.copy()
        for _ in range(SUBSTEPS):
            self._substep(trigger)
            trigger = np.zeros_like(trigger)

        fallen = (np.abs(self.th) > FALL_ANG) | (self.y < MIN_Y)
        self.steps += 1
        timeout = self.steps >= MAX_STEPS

        reward = W_DIST * (self.x - x0) + W_ALIVE
        if self.posture_penalty:
            reward -= W_POSTURE * (self.th ** 2)
        reward = np.where(fallen, reward - FALL_PENALTY, reward)

        prev_done = self.done.copy()
        self.done = fallen | timeout
        reward = np.where(prev_done, 0.0, reward)
        info = {"x": self.x.copy(), "fallen": fallen, "timeout": timeout}
        return self.obs(), reward.astype(np.float32), self.done.copy(), info


# ================================================================= Game 1
class CarEnv(BaseEnv):
    task_id = 1

    def __init__(self, n=64, seed=None):
        super().__init__(n, seed)
        self.s = np.zeros(n)
        self.d = np.zeros(n)
        self.psi = np.zeros(n)
        self.v = np.zeros(n)
        self.steer = np.zeros(n)
        self.d_timer = np.zeros(n)
        self.d_dir = np.zeros(n)
        self.d_mag = np.zeros(n)
        self.v_prev = np.zeros(n)

    def _init(self, idx):
        k = len(idx)
        self.s[idx] = 0.0
        self.d[idx] = self.rng.uniform(-0.4, 0.4, size=k)
        self.psi[idx] = self.rng.uniform(-0.05, 0.05, size=k)
        self.v[idx] = 6.0
        self.steer[idx] = 0.0
        self.d_timer[idx] = 0.0
        self.d_dir[idx] = 0.0
        self.d_mag[idx] = 0.0
        self.v_prev[idx] = 6.0
        self.done[idx] = False
        self.steps[idx] = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init(np.arange(self.n))
        return self.obs()

    def reset_idx(self, idx):
        idx = np.atleast_1d(np.asarray(idx))
        if len(idx):
            self._init(idx)

    def obs(self):
        k0 = track_curvature(self.s) * KAPPA_SCALE
        k1 = track_curvature(self.s + LOOKAHEAD[1]) * KAPPA_SCALE
        k2 = track_curvature(self.s + LOOKAHEAD[2]) * KAPPA_SCALE
        k3 = track_curvature(self.s + LOOKAHEAD[3]) * KAPPA_SCALE
        active = (self.d_timer > 0).astype(np.float64)
        return np.stack([
            self.d / ROAD_HALF,
            np.sin(self.psi),
            np.cos(self.psi),
            self.v / V_MAX_CAR,
            self.steer / STEER_MAX,
            k0, k1, k2, k3,
            active,
            self.d_dir * self.d_mag * active,
            self.v * np.sin(self.psi) / V_MAX_CAR,
            np.abs(self.d) / ROAD_HALF,
            (self.v - self.v_prev) / V_MAX_CAR,
        ], axis=1).astype(np.float32)

    def step(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        # 转向输入 + 自动回中
        steer_in = np.where(a == 0, -STEER_RATE, np.where(a == 1, STEER_RATE, 0.0))
        self.steer = np.clip(
            self.steer + (steer_in - STEER_RETURN * self.steer * (steer_in == 0)) * DT * SUBSTEPS,
            -STEER_MAX, STEER_MAX)

        # 失控事件
        idle = self.d_timer <= 0
        fire = idle & (self._roll((self.n,)) < DISTURB_P)
        self.d_timer = np.where(fire, self.rng.uniform(DISTURB_T_MIN, DISTURB_T_MAX, size=self.n), self.d_timer)
        self.d_dir = np.where(fire, self.rng.choice([-1.0, 1.0], size=self.n), self.d_dir)
        self.d_mag = np.where(fire, self.rng.uniform(0.5, 1.0, size=self.n), self.d_mag)
        active = self.d_timer > 0
        self.d_timer = np.maximum(self.d_timer - DT * SUBSTEPS, 0.0)

        s0 = self.s.copy()
        self.v_prev = self.v.copy()
        for _ in range(SUBSTEPS):
            kap = track_curvature(self.s)
            accel = np.where(a == 2, ACC, np.where(a == 3, -BRAKE_FORCE, 0.0))
            self.v = np.clip(self.v + (accel - CAR_DRAG * self.v) * DT, 0.0, V_MAX_CAR)
            yaw = self.v * np.tan(self.steer) / WHEELBASE - self.v * kap
            yaw = yaw + np.where(active, self.d_dir * self.d_mag * D_RATE, 0.0)
            self.psi += yaw * DT
            self.psi = np.clip(self.psi, -1.2, 1.2)
            self.d += (self.v * np.sin(self.psi)
                       + np.where(active, self.d_dir * self.d_mag * D_LAT, 0.0)) * DT
            self.s += self.v * np.cos(self.psi) * DT

        offroad = np.abs(self.d) > ROAD_HALF
        stall = self.v < V_STALL
        self.steps += 1
        timeout = self.steps >= MAX_STEPS

        reward = W_DIST * (self.s - s0) + W_ALIVE - W_CENTER * (self.d / ROAD_HALF) ** 2
        reward = np.where(offroad, reward - CRASH_PENALTY, reward)

        prev_done = self.done.copy()
        self.done = offroad | stall | timeout
        reward = np.where(prev_done, 0.0, reward)
        info = {"x": self.s.copy(), "fallen": offroad, "timeout": timeout}
        return self.obs(), reward.astype(np.float32), self.done.copy(), info


def make_env(task_id, n, seed=None):
    return RunnerEnv(n, seed) if task_id == 0 else CarEnv(n, seed)


# ================================================================= 启发式基线
def runner_heuristic(env):
    """抬起更靠后那条支撑腿（换步），姿态越界时优先修正姿态。"""
    n = env.n
    hip_x = env.x + HIP_OFF * np.sin(env.th)
    hip_y = env.y - HIP_OFF * np.cos(env.th)
    # 必须两腿同时着地才换步，保证任何时刻至少一条腿支撑
    both = env.stance[:, 0] & env.stance[:, 1]
    behind_l = both & (env.fx[:, 0] < env.fx[:, 1])
    behind_r = both & (env.fx[:, 1] <= env.fx[:, 0])
    act = np.where(behind_l, 0, np.where(behind_r, 1, 2))
    act = np.where(env.th > 0.15, 2, act)     # 后仰过度 -> 前倾
    act = np.where(env.th < -0.35, 3, act)    # 前倾过度 -> 后仰
    return act.astype(np.int64)


def car_heuristic(env):
    """车道保持：按横向偏移 + 航向角 + 前瞻曲率反打方向盘"""
    n = env.n
    kap = track_curvature(env.s + 14.0)
    target_psi = -0.20 * (env.d / ROAD_HALF) + 0.9 * WHEELBASE * kap
    err = env.psi - target_psi
    act = np.where(err > 0.015, 0, np.where(err < -0.015, 1, 2))
    act = np.where(env.v > 16.0, 3, act)
    return act.astype(np.int64)


def rollout_stats(env_cls, policy_fn, n_episodes=32, seed=1):
    env = env_cls(n=n_episodes, seed=seed)
    env.reset(seed=seed)
    dist = np.zeros(n_episodes)
    fin = np.zeros(n_episodes, dtype=bool)
    falls = 0
    for _ in range(MAX_STEPS):
        a = policy_fn(env)
        _, _, done, info = env.step(a)
        idx = np.where(done & ~fin)[0]
        for i in idx:
            dist[i] = env.x[i] if env_cls is RunnerEnv else env.s[i]
            fin[i] = True
            if info["fallen"][i]:
                falls += 1
        if fin.all():
            break
    for i in np.where(~fin)[0]:
        dist[i] = env.x[i] if env_cls is RunnerEnv else env.s[i]
    return {"mean": float(dist.mean()), "max": float(dist.max()),
            "min": float(dist.min()), "fail": float(falls / n_episodes)}


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("runner heuristic:", rollout_stats(RunnerEnv, runner_heuristic, 32, 1))
    print("runner random   :", rollout_stats(RunnerEnv, lambda e: e.rng.integers(0, 4, size=e.n), 32, 1))
    print("car heuristic   :", rollout_stats(CarEnv, car_heuristic, 32, 1))
    print("car random      :", rollout_stats(CarEnv, lambda e: e.rng.integers(0, 4, size=e.n), 32, 1))
    print("elapsed %.1fs" % (time.time() - t0))
