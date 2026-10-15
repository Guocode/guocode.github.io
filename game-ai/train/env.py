"""Pixel Arena v2 — 两个游戏的轻量物理环境（向量化·无限模式）

Game 0  RUNNER  独轮平衡机器人：无腿，自动前进，玩家只控身体平衡
Game 1  CAR     高速公路：自动油门，玩家只控左右转向，随机打滑/爆胎

v2 变化
  * 两关均无终点：MAX_STEPS 只是训练用回合上限，浏览器端永不超时
  * RUNNER 去掉腿自由度：躯干立在驱动轮上（倒立摆），前进本身带抖动，
    另有随机平衡噪声（OU 过程）与随机颠簸事件，玩家只有 4 档平衡力矩
  * CAR 去掉油门/刹车：车速自动爬升，玩家只有 4 档转向；
    随机事件分两类——打滑（侧滑 + 转向力下降）与爆胎（强拽偏 + 掉速）
  * 动作空间仍为 4 个离散动作、观测 14 维、步频 1/60 s（2 个 1/120 子步），
    同一个 MoE 策略网络可直接驱动

本文件中的常量与积分公式与浏览器端 assets/js/env.js 逐行对应。
"""

import numpy as np

# ================================================================= 共享常量
DT = 1.0 / 120.0
SUBSTEPS = 2
N_OBS = 14
N_ACT = 4
MAX_STEPS = 3600          # 训练用回合上限 60 s；浏览器端无超时（无限模式）

ACT_NAMES = {
    0: ["LEAN_BACK", "LEAN_FWD", "LEAN_BACK+", "LEAN_FWD+"],
    1: ["STEER_L", "STEER_R", "STEER_L+", "STEER_R+"],
}

# ================================================================= Game 0 RUNNER（独轮平衡）
# 躯干是立在驱动轮上的倒立摆：轮子按预设速度一直向前滚，
# 平衡完全靠玩家对躯干施加力矩。掉下即结束，距离即分数。
K_INV = 3.3             # 倒立摆失稳系数（≈ G / 等效摆长）
INERTIA = 0.45
C_ANG = 0.30            # 俯仰被动阻尼
V_START = 2.2           # 起始车速 m/s
ACC_RAMP = 0.08         # 车速随时间线性爬升
V_JIT = 0.15            # 步频抖动幅度（前进姿势带来的持续抖动）
JIT_FREQ = 1.7          # 抖动频率 Hz
V_MAX_RUN = 6.5
ACC_COUPLE = 0.10       # 加速度对俯仰的耦合（加速时身体后仰）
# 平衡噪声：OU 随机过程，速度越快噪声越大
OU_THETA = 0.8
OU_SIGMA = 2.2
NOISE_GAIN = 0.9
N_SCALE = 3.0           # 观测归一化
# 随机颠簸事件（坑洼/碎石）：短促力矩冲击
BUMP_P = 0.008
BUMP_T_MIN = 0.15
BUMP_T_MAX = 0.45
BUMP_GAIN = 2.6
# 玩家力矩（软/硬各两档，方向由动作决定）
T_SOFT = 1.3
T_HARD = 2.6
FALL_ANG = 0.90
W_NORM = 6.0
A_SCALE = 2.0

# 随机起伏地形：确定性伪随机函数（三个正弦叠加），Python / JS 逐位一致。
# 幅度随里程渐入（前 TER_RAMP 米平缓起步），通过两条路径干扰平衡：
#   1) 竖向加速度 ay 调制等效重力（过凸顶时短暂"失重"，凹坑时超重）
#   2) 坡度直接产生俯仰力矩（上坡后仰、下坡前倾）
TER_A = (0.090, 0.050, 0.022)      # 各分量振幅 m
TER_F = (0.83, 2.17, 3.70)         # 空间频率 rad/m（波长 7.6 / 2.9 / 1.7 m）
TER_PH = (0.0, 1.3, 4.1)           # 相位
TER_RAMP = 40.0                    # 幅度渐入距离
G_EFF = 9.8
K_SLOPE = 1.0                      # 坡度 -> 俯仰力矩系数
S_NORM = 0.20                      # 坡度观测归一化
AY_NORM = 14.0                     # 竖向加速度观测归一化

W_DIST = 1.0
W_ALIVE = 0.005
W_POSTURE = 0.03
FALL_PENALTY = 1.0

# ================================================================= Game 1 CAR（无限公路）
CAR_DRAG = 0.22
V_START_CAR = 8.0       # 起步巡航速度
V_RAMP = 0.004          # 巡航速度随里程爬升（难度递增）
V_MAX_CAR = 26.0
K_RELAX = 0.5           # 实际车速向巡航速度弛豫的速率
STEER_RATE = 2.4
STEER_HARD = 2.0        # 硬转向倍率
STEER_MAX = 0.55
STEER_RETURN = 2.0
WHEELBASE = 2.6
ROAD_HALF = 3.4
KAPPA_SCALE = 80.0
LOOKAHEAD = (0.0, 12.0, 28.0, 50.0)
# 事件机制 v3：打滑由转弯角载荷触发（过猛甩尾，沿当前转向方向继续旋转），
#              爆胎是路面随机 hazards（概率随车速上升）。推力均为冲量式衰减包络，
#              事件结束后有冷却期，杜绝连环事件。
SKID_P = 0.004           # 基础概率/步（零载荷直行时很低）
SKID_LOAD_GAIN = 2.0     # 载荷增益：p = SKID_P * (0.3 + GAIN * load)
SKID_T_MIN, SKID_T_MAX = 0.6, 1.4
SKID_RATE = 0.55         # 偏航角速度 rad/s（×衰减包络）
SKID_LAT = 0.90          # 横向漂移 m/s（×衰减包络）
SKID_GRIP = 0.45         # 打滑时转向执行力度
SKID_DIR_TH = 0.15       # |转向| 超过该值视为"转弯中"，甩尾方向 = 转向方向
# 爆胎（少见、猛烈、掉速且巡航恢复变慢）
BLOW_P = 0.0025
BLOW_V_GAIN = 1.5        # 速度增益：p = BLOW_P * (0.5 + GAIN * v/vmax)
BLOW_T_MIN, BLOW_T_MAX = 0.8, 1.8
BLOW_RATE = 1.05
BLOW_LAT = 1.60
BLOW_GRIP = 0.60
BLOW_VDROP = 0.80        # 爆胎瞬间速度保留比例
BLOW_CRUISE = 0.75       # 事件期间巡航目标折减（爆胎后恢复慢，更真实）
EV_COOLDOWN = 2.0        # 事件结束后的冷却期（s）
D_DIFF_SCALE = 0.05     # 横向位移差分归一化

W_CENTER = 0.02
CRASH_PENALTY = 1.0


def track_curvature(s):
    """赛道曲率 κ(s)：确定性解析函数，JS 端用同一公式。"""
    return 0.018 * np.sin(0.035 * s) + 0.010 * np.sin(0.011 * s + 1.7)


def terrain_h(x):
    """独轮平衡的随机起伏地面高度（确定性，两端一致）。"""
    d = np.minimum(x / TER_RAMP, 1.0)
    return d * (TER_A[0] * np.sin(TER_F[0] * x)
                + TER_A[1] * np.sin(TER_F[1] * x + TER_PH[1])
                + TER_A[2] * np.sin(TER_F[2] * x + TER_PH[2]))


def terrain_slope(x):
    """地面坡度 dh/dx（解析导数）。"""
    d = np.minimum(x / TER_RAMP, 1.0)
    return d * (TER_A[0] * TER_F[0] * np.cos(TER_F[0] * x)
                + TER_A[1] * TER_F[1] * np.cos(TER_F[1] * x + TER_PH[1])
                + TER_A[2] * TER_F[2] * np.cos(TER_F[2] * x + TER_PH[2]))


def terrain_curv(x):
    """地面曲率 d²h/dx²（解析二阶导，凸为负）。"""
    d = np.minimum(x / TER_RAMP, 1.0)
    return -d * (TER_A[0] * TER_F[0] ** 2 * np.sin(TER_F[0] * x)
                 + TER_A[1] * TER_F[1] ** 2 * np.sin(TER_F[1] * x + TER_PH[1])
                 + TER_A[2] * TER_F[2] ** 2 * np.sin(TER_F[2] * x + TER_PH[2]))


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
    """独轮平衡机器人。动作：
       0 后倾(软)  1 前倾(软)  2 后倾(硬)  3 前倾(硬)
       约定 th > 0 为后仰；纠正后仰需要前倾力矩（负力矩）。"""

    task_id = 0

    def __init__(self, n=64, seed=None, posture_penalty=True):
        super().__init__(n, seed)
        self.posture_penalty = posture_penalty
        self.x = np.zeros(n)
        self.th = np.zeros(n)
        self.w = np.zeros(n)
        self.ou = np.zeros(n)
        self.b_timer = np.zeros(n)
        self.b_dur = np.zeros(n)
        self.b_dir = np.zeros(n)
        self.b_mag = np.zeros(n)
        self.phase = np.zeros(n)
        self.vx = np.zeros(n)
        self.ax = np.zeros(n)
        self.vx_prev = np.zeros(n)

    def _init(self, idx):
        k = len(idx)
        self.x[idx] = 0.0
        self.th[idx] = self.rng.uniform(-0.08, 0.08, size=k)
        self.w[idx] = self.rng.uniform(-0.3, 0.3, size=k)
        self.ou[idx] = 0.0
        self.b_timer[idx] = 0.0
        self.b_dur[idx] = 1.0
        self.b_dir[idx] = 0.0
        self.b_mag[idx] = 0.0
        self.phase[idx] = 0.0
        self.vx_prev[idx] = V_START
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

    def _speed_profile(self):
        """预设前进速度（与平衡无关，机器人永远向前滚）：
        随时间爬升 + 步频抖动。返回 (vx, ax)。"""
        t = self.steps * (DT * SUBSTEPS)
        base = np.minimum(V_START + ACC_RAMP * t, V_MAX_RUN)
        vx = base + V_JIT * np.sin(2.0 * np.pi * JIT_FREQ * t)
        ax = (vx - self.vx_prev) / (DT * SUBSTEPS)
        return vx, ax

    def _terrain_state(self, vx, ax):
        """跟随地形的竖向运动学：ay = curv*vx² + slope*ax。"""
        slope = terrain_slope(self.x)
        curv = terrain_curv(self.x)
        ay = curv * vx * vx + slope * ax
        return slope, ay

    def obs(self):
        vx, ax = self._speed_profile()
        slope, ay = self._terrain_state(vx, ax)
        t = self.steps * (DT * SUBSTEPS)
        b_active = (self.b_timer > 0).astype(np.float64)
        b_prog = np.where(self.b_timer > 0,
                          np.clip(1.0 - self.b_timer / np.maximum(self.b_dur, 1e-6), 0.0, 1.0),
                          0.0)
        return np.stack([
            np.sin(self.th),
            np.cos(self.th),
            self.w / W_NORM,
            vx / V_MAX_RUN,
            np.clip(ax / A_SCALE, -1.0, 1.0),
            self.ou / N_SCALE,
            b_active,
            self.b_dir * self.b_mag * b_active,
            b_prog,
            np.abs(self.th),
            self.th * self.w / W_NORM,
            np.clip(slope / S_NORM, -1.0, 1.0),
            np.clip(ay / AY_NORM, -1.0, 1.0),
            np.minimum(t / 60.0, 1.0),
        ], axis=1).astype(np.float32)

    def step(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        tau_cmd = np.where(a == 0, T_SOFT, np.where(a == 1, -T_SOFT,
                           np.where(a == 2, T_HARD, np.where(a == 3, -T_HARD, 0.0))))

        # 随机颠簸事件
        idle = self.b_timer <= 0
        fire = idle & (self._roll((self.n,)) < BUMP_P)
        b_dur_new = self.rng.uniform(BUMP_T_MIN, BUMP_T_MAX, size=self.n)
        self.b_timer = np.where(fire, b_dur_new, self.b_timer)
        self.b_dur = np.where(fire, b_dur_new, self.b_dur)
        self.b_dir = np.where(fire, self.rng.choice([-1.0, 1.0], size=self.n), self.b_dir)
        self.b_mag = np.where(fire, self.rng.uniform(0.5, 1.0, size=self.n), self.b_mag)
        b_active = self.b_timer > 0
        b_u = np.where(b_active, np.clip(1.0 - self.b_timer / np.maximum(self.b_dur, 1e-6), 0.0, 1.0), 0.0)

        vx, ax = self._speed_profile()
        slope, ay = self._terrain_state(vx, ax)
        x0 = self.x.copy()

        for _ in range(SUBSTEPS):
            # OU 平衡噪声（速度越快越大）
            self.ou += (-OU_THETA * self.ou * DT
                        + OU_SIGMA * np.sqrt(DT) * self.rng.standard_normal(self.n))
            tau_noise = NOISE_GAIN * self.ou * (0.35 + 0.65 * vx / V_MAX_RUN)
            tau_bump = np.where(b_active, self.b_dir * self.b_mag * BUMP_GAIN
                                * np.sin(np.pi * np.clip(b_u, 0.0, 1.0)), 0.0)
            # 地形耦合：竖向加速度调制等效重力 + 坡度直接力矩（上坡后仰 / 下坡前倾）
            gfac = np.clip(1.0 + ay / G_EFF, -0.5, 2.5)
            tau = (K_INV * gfac * np.sin(self.th) + tau_cmd / INERTIA
                   + tau_noise + tau_bump + ACC_COUPLE * ax
                   + K_SLOPE * G_EFF * slope - C_ANG * self.w)
            self.w += tau * DT
            self.th += self.w * DT
            self.x += vx * DT
            self.b_timer = np.maximum(self.b_timer - DT, 0.0)

        self.vx = vx
        self.ax = ax
        self.vx_prev = vx

        fallen = np.abs(self.th) > FALL_ANG
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
    """无限公路。动作：
       0 左转(软)  1 右转(软)  2 左转(硬)  3 右转(硬)
       油门全自动（巡航速度随里程爬升），随机事件分打滑 / 爆胎两类。"""

    task_id = 1

    def __init__(self, n=64, seed=None):
        super().__init__(n, seed)
        self.s = np.zeros(n)
        self.d = np.zeros(n)
        self.d_prev = np.zeros(n)
        self.psi = np.zeros(n)
        self.v = np.zeros(n)
        self.steer = np.zeros(n)
        self.ev_timer = np.zeros(n)
        self.ev_dur = np.ones(n)        # 事件总时长（衰减包络分母）
        self.ev_dir = np.zeros(n)
        self.ev_mag = np.zeros(n)
        self.ev_type = np.zeros(n)      # 0 打滑  1 爆胎
        self.ev_cool = np.zeros(n)      # 冷却计时（事件期 + 冷却期）
        self.v = np.zeros(n)

    def _init(self, idx):
        k = len(idx)
        self.s[idx] = 0.0
        self.d[idx] = self.rng.uniform(-0.4, 0.4, size=k)
        self.d_prev[idx] = self.d[idx]
        self.psi[idx] = self.rng.uniform(-0.05, 0.05, size=k)
        self.v[idx] = V_START_CAR
        self.steer[idx] = 0.0
        self.ev_timer[idx] = 0.0
        self.ev_dur[idx] = 1.0
        self.ev_dir[idx] = 0.0
        self.ev_mag[idx] = 0.0
        self.ev_type[idx] = 0.0
        self.ev_cool[idx] = 0.0
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

    def _cruise(self):
        return np.minimum(V_START_CAR + V_RAMP * self.s, V_MAX_CAR)

    def obs(self):
        k0 = track_curvature(self.s) * KAPPA_SCALE
        k1 = track_curvature(self.s + LOOKAHEAD[1]) * KAPPA_SCALE
        k2 = track_curvature(self.s + LOOKAHEAD[2]) * KAPPA_SCALE
        k3 = track_curvature(self.s + LOOKAHEAD[3]) * KAPPA_SCALE
        active = (self.ev_timer > 0).astype(np.float64)
        return np.stack([
            self.d / ROAD_HALF,
            np.sin(self.psi),
            np.cos(self.psi),
            self.v / V_MAX_CAR,
            self.steer / STEER_MAX,
            k0, k1, k2, k3,
            active,
            self.ev_dir * self.ev_mag * active,
            self.ev_type,
            self.v * np.sin(self.psi) / V_MAX_CAR,
            (self.d - self.d_prev) / D_DIFF_SCALE,
        ], axis=1).astype(np.float32)

    def step(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        hard = (a >= 2).astype(np.float64)
        steer_in = np.where((a == 0) | (a == 2), -STEER_RATE * np.where(hard > 0, STEER_HARD, 1.0),
                            np.where((a == 1) | (a == 3), STEER_RATE * np.where(hard > 0, STEER_HARD, 1.0), 0.0))
        # 转向积分 + 自动回中
        self.steer = np.clip(
            self.steer + (steer_in - STEER_RETURN * self.steer * (steer_in == 0)) * DT * SUBSTEPS,
            -STEER_MAX, STEER_MAX)

        # 事件触发：打滑看转弯角载荷（驾驶行为后果），爆胎看车速（高速 hazard）。
        # 随机数消耗顺序与 JS 端一致：r1 触发, r2 时长, r3 方向回退, r4 强度, r5 类型
        r1 = self._roll((self.n,))
        r2 = self.rng.uniform(0.0, 1.0, size=self.n)
        r3 = self.rng.choice([-1.0, 1.0], size=self.n)
        r4 = self.rng.uniform(0.0, 1.0, size=self.n)
        r5 = self.rng.uniform(0.0, 1.0, size=self.n)
        idle = (self.ev_timer <= 0) & (self.ev_cool <= 0)
        load = (np.clip(np.abs(self.steer) / STEER_MAX, 0.0, 1.0)
                * np.clip(self.v / V_MAX_CAR, 0.0, 1.0))
        p_skid = SKID_P * (0.3 + SKID_LOAD_GAIN * load)
        p_blow = BLOW_P * (0.5 + BLOW_V_GAIN * self.v / V_MAX_CAR)
        fire = idle & (r1 < p_skid + p_blow)
        is_blow = r5 < p_blow / (p_skid + p_blow)
        dur = np.where(is_blow,
                       BLOW_T_MIN + r2 * (BLOW_T_MAX - BLOW_T_MIN),
                       SKID_T_MIN + r2 * (SKID_T_MAX - SKID_T_MIN))
        # 甩尾方向：转弯中沿当前转向方向继续旋转（车尾向外甩），接近直行则随机
        skid_dir = np.where(np.abs(self.steer) > SKID_DIR_TH, np.sign(self.steer), r3)
        self.ev_timer = np.where(fire, dur, self.ev_timer)
        self.ev_dur = np.where(fire, dur, self.ev_dur)
        self.ev_dir = np.where(fire, np.where(is_blow, r3, skid_dir), self.ev_dir)
        self.ev_mag = np.where(fire,
                               np.where(is_blow, 0.7 + r4 * 0.3, 0.4 + r4 * 0.6),
                               self.ev_mag)
        self.ev_type = np.where(fire, is_blow.astype(np.float64), self.ev_type)
        self.ev_cool = np.where(fire, dur + EV_COOLDOWN, self.ev_cool)
        active = self.ev_timer > 0
        self.ev_timer = np.maximum(self.ev_timer - DT * SUBSTEPS, 0.0)
        self.ev_cool = np.maximum(self.ev_cool - DT * SUBSTEPS, 0.0)
        # 冲量式衰减包络：事件刚触发时推力最大，随剩余时间线性衰减到 0
        env_u = np.clip(self.ev_timer / np.maximum(self.ev_dur, 1e-6), 0.0, 1.0)

        ev_rate = np.where(self.ev_type > 0, BLOW_RATE, SKID_RATE)
        ev_lat = np.where(self.ev_type > 0, BLOW_LAT, SKID_LAT)
        ev_grip = np.where(self.ev_type > 0, BLOW_GRIP, SKID_GRIP)
        push = np.where(active, self.ev_dir * self.ev_mag * env_u, 0.0)

        v_blow = np.where(fire & (self.ev_type > 0), BLOW_VDROP, 1.0)
        self.v = self.v * v_blow

        s0 = self.s.copy()
        self.d_prev = self.d.copy()
        for _ in range(SUBSTEPS):
            kap = track_curvature(self.s)
            # 爆胎期间巡航目标折减：掉速不会立刻弛豫回去，恢复更慢
            cruise = self._cruise() * np.where(active & (self.ev_type > 0), BLOW_CRUISE, 1.0)
            self.v = np.clip(self.v + (cruise - self.v) * K_RELAX * DT, 0.0, V_MAX_CAR)
            authority = np.where(active, ev_grip, 1.0)
            yaw = (self.v * np.tan(self.steer) * authority / WHEELBASE - self.v * kap
                   + push * ev_rate)
            self.psi += yaw * DT
            self.psi = np.clip(self.psi, -1.2, 1.2)
            self.d += (self.v * np.sin(self.psi) + push * ev_lat) * DT
            self.s += self.v * np.cos(self.psi) * DT

        offroad = np.abs(self.d) > ROAD_HALF
        self.steps += 1
        timeout = self.steps >= MAX_STEPS

        reward = W_DIST * (self.s - s0) + W_ALIVE - W_CENTER * (self.d / ROAD_HALF) ** 2
        reward = np.where(offroad, reward - CRASH_PENALTY, reward)

        prev_done = self.done.copy()
        self.done = offroad | timeout
        reward = np.where(prev_done, 0.0, reward)
        info = {"x": self.s.copy(), "fallen": offroad, "timeout": timeout}
        return self.obs(), reward.astype(np.float32), self.done.copy(), info


def make_env(task_id, n, seed=None):
    return RunnerEnv(n, seed) if task_id == 0 else CarEnv(n, seed)


# ================================================================= 启发式基线
def runner_heuristic(env):
    """倒立摆 PD 平衡：u = KP*th + KD*w > 0 表示正在向后倒，
    需要前倾力矩（负）纠正；误差越大档位越硬。"""
    u = 2.2 * env.th + 0.9 * env.w
    act = np.zeros(env.n, dtype=np.int64)
    act = np.where(u > 0.05, 1, act)          # 后仰 -> 前倾(软)
    act = np.where(u > 0.30, 3, act)          # 严重   -> 前倾(硬)
    act = np.where(u < -0.05, 0, act)         # 前倾 -> 后倾(软)
    act = np.where(u < -0.30, 2, act)         # 严重 -> 后倾(硬)
    return act


def car_heuristic(env):
    """车道保持：偏移 + 航向 + 前瞻曲率反打方向，事件期间提前修正。"""
    kap = track_curvature(env.s + 14.0)
    active = env.ev_timer > 0
    target_psi = -0.20 * (env.d / ROAD_HALF) + 0.9 * WHEELBASE * kap
    # 事件把航向往 dir 方向拽，提前向反方向多打一点（dir>0 推高 psi -> 更需要左转）
    err = env.psi - target_psi + 0.10 * np.where(active, env.ev_dir * env.ev_mag, 0.0)
    act = np.where(err > 0.015, 0, np.where(err < -0.015, 1, 0))
    act = np.where(err > 0.10, 2, act)
    act = np.where(err < -0.10, 3, act)
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
