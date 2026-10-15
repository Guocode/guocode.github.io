/* Pixel Arena v2 — 浏览器端物理环境（无限模式）
 *
 * 常量与积分公式与 train/env.py 逐行对应。
 * 两个游戏共享 2 个离散动作（v4 去掉软/硬档）与 14 维观测，因此能被同一个 MoE 模型驱动。
 * v2：RUNNER 去腿改独轮平衡（自动前进 + 抖动 + 随机噪声/颠簸）；
 *     CAR 去油门刹车（自动巡航 + 只控转向 + 随机打滑/爆胎）。
 *     两关均无终点：step() 永不超时，只有摔倒/冲出路面才结束。
 * v4：去掉 Shift 硬档（动作 4→2），难度下调（噪声/事件更温和、速度上限更低）。
 * 注意：Python 端为向量化实现（批量预生成随机数），浏览器端为单实例流式取数，
 * 因此逐位一致无法保证，采用"统计一致性"校验（同策略下两者成绩分布接近）。
 */

export const DT = 1 / 120;
export const SUBSTEPS = 2;
export const N_OBS = 14;
export const STEP_DT = DT * SUBSTEPS;      // 1/60 s

// ------------------------------------------------- Game 0 RUNNER（独轮平衡）
const K_INV = 3.3, INERTIA = 0.45, C_ANG = 0.30;
const V_START = 2.2, ACC_RAMP = 0.08, V_JIT = 0.15, JIT_FREQ = 1.7, V_MAX_RUN = 5.5;
const ACC_COUPLE = 0.10;
const OU_THETA = 0.8, OU_SIGMA = 1.5, NOISE_GAIN = 0.7, N_SCALE = 3.0;
const BUMP_P = 0.005, BUMP_T_MIN = 0.15, BUMP_T_MAX = 0.45, BUMP_GAIN = 1.8;
const T_LEAN = 1.9, FALL_ANG = 0.90, W_NORM = 6.0, A_SCALE = 2.0;

// 随机起伏地形（确定性伪随机函数，与 train/env.py 逐位一致）
const TER_A = [0.090, 0.050, 0.022];
const TER_F = [0.83, 2.17, 3.70];
const TER_PH = [0.0, 1.3, 4.1];
const TER_RAMP = 40.0;
const G_EFF = 9.8, K_SLOPE = 1.0, S_NORM = 0.20, AY_NORM = 14.0;

export function terrainH(x) {
  const d = Math.min(x / TER_RAMP, 1.0);
  return d * (TER_A[0] * Math.sin(TER_F[0] * x)
            + TER_A[1] * Math.sin(TER_F[1] * x + TER_PH[1])
            + TER_A[2] * Math.sin(TER_F[2] * x + TER_PH[2]));
}
export function terrainSlope(x) {
  const d = Math.min(x / TER_RAMP, 1.0);
  return d * (TER_A[0] * TER_F[0] * Math.cos(TER_F[0] * x)
            + TER_A[1] * TER_F[1] * Math.cos(TER_F[1] * x + TER_PH[1])
            + TER_A[2] * TER_F[2] * Math.cos(TER_F[2] * x + TER_PH[2]));
}
export function terrainCurv(x) {
  const d = Math.min(x / TER_RAMP, 1.0);
  return -d * (TER_A[0] * TER_F[0] * TER_F[0] * Math.sin(TER_F[0] * x)
             + TER_A[1] * TER_F[1] * TER_F[1] * Math.sin(TER_F[1] * x + TER_PH[1])
             + TER_A[2] * TER_F[2] * TER_F[2] * Math.sin(TER_F[2] * x + TER_PH[2]));
}

// ------------------------------------------------- Game 1 CAR（无限公路）
const V_START_CAR = 8.0, V_RAMP = 0.003, V_MAX_CAR = 23.0, K_RELAX = 0.5;
const STEER_RATE = 2.8, STEER_MAX = 0.55, STEER_RETURN = 2.0;
const WHEELBASE = 2.6, ROAD_HALF = 3.4, KAPPA_SCALE = 80.0;
const LOOKAHEAD = [0.0, 12.0, 28.0, 50.0];
// 事件机制 v3：打滑由转弯角载荷触发（过猛甩尾沿转向方向），爆胎概率随车速；
// 推力为冲量式衰减包络；事件结束后有冷却期。v4 降频降幅（人类可玩性）。
const SKID_P = 0.0025, SKID_LOAD_GAIN = 1.2;
const SKID_T_MIN = 0.6, SKID_T_MAX = 1.4;
const SKID_RATE = 0.45, SKID_LAT = 0.70, SKID_GRIP = 0.55;
const SKID_DIR_TH = 0.15;
const BLOW_P = 0.0015, BLOW_V_GAIN = 1.0;
const BLOW_T_MIN = 0.8, BLOW_T_MAX = 1.8;
const BLOW_RATE = 0.85, BLOW_LAT = 1.30, BLOW_GRIP = 0.65, BLOW_VDROP = 0.80;
const BLOW_CRUISE = 0.75;      // 爆胎期间巡航目标折减
const EV_COOLDOWN = 2.0;       // 事件冷却期（s）
const D_DIFF_SCALE = 0.05;

export function trackCurvature(s) {
  return 0.018 * Math.sin(0.035 * s) + 0.010 * Math.sin(0.011 * s + 1.7);
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

// ================================================================ Game 0
export class RunnerEnv {
  constructor(seed = 1) {
    this.taskId = 0;
    this.rng = mulberry32(seed >>> 0);
    this.reset(seed);
  }

  // 标准高斯（Box-Muller，消耗两个均匀数）
  gauss() {
    const u1 = Math.max(this.rng(), 1e-12), u2 = this.rng();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  reset(seed) {
    if (seed !== undefined) this.rng = mulberry32(seed >>> 0);
    this.x = 0.0;
    this.th = (this.rng() - 0.5) * 0.16;
    this.w = (this.rng() - 0.5) * 0.6;
    this.ou = 0.0;
    this.bTimer = 0.0; this.bDur = 1.0; this.bDir = 0.0; this.bMag = 0.0;
    this.vxPrev = V_START;
    this.vx = V_START; this.ax = 0.0;
    this.done = false;
    this.steps = 0;
    this.fell = false;
    return this.obs();
  }

  // 预设前进速度：随时间爬升 + 步频抖动（与平衡无关，永远向前滚）
  _profile() {
    const t = this.steps * STEP_DT;
    const base = Math.min(V_START + ACC_RAMP * t, V_MAX_RUN);
    const vx = base + V_JIT * Math.sin(2.0 * Math.PI * JIT_FREQ * t);
    const ax = (vx - this.vxPrev) / STEP_DT;
    return { vx, ax, t };
  }

  obs() {
    const { vx, ax, t } = this._profile();
    const slope = terrainSlope(this.x);
    const ay = terrainCurv(this.x) * vx * vx + slope * ax;
    const o = new Float32Array(N_OBS);
    const bAct = this.bTimer > 0 ? 1 : 0;
    const bProg = bAct ? clamp(1.0 - this.bTimer / Math.max(this.bDur, 1e-6), 0, 1) : 0.0;
    o[0] = Math.sin(this.th);
    o[1] = Math.cos(this.th);
    o[2] = this.w / W_NORM;
    o[3] = vx / V_MAX_RUN;
    o[4] = clamp(ax / A_SCALE, -1.0, 1.0);
    o[5] = this.ou / N_SCALE;
    o[6] = bAct;
    o[7] = this.bDir * this.bMag * bAct;
    o[8] = bProg;
    o[9] = Math.abs(this.th);
    o[10] = this.th * this.w / W_NORM;
    o[11] = clamp(slope / S_NORM, -1.0, 1.0);
    o[12] = clamp(ay / AY_NORM, -1.0, 1.0);
    o[13] = Math.min(t / 60.0, 1.0);
    return o;
  }

  step(a) {
    // v4：0 后倾 / 1 前倾（原软硬 4 档合并为单档 T_LEAN）；未按键 = 零力矩空转
    const tauCmd = a === 0 ? T_LEAN : a === 1 ? -T_LEAN : 0.0;

    // 随机颠簸事件（坑洼/碎石）
    const r1 = this.rng(), r2 = this.rng(), r3 = this.rng(), r4 = this.rng();
    const idle = this.bTimer <= 0;
    if (idle && r1 < BUMP_P) {
      this.bDur = BUMP_T_MIN + r2 * (BUMP_T_MAX - BUMP_T_MIN);
      this.bTimer = this.bDur;
      this.bDir = r3 < 0.5 ? -1.0 : 1.0;
      this.bMag = 0.5 + r4 * 0.5;
    }
    const bAct = this.bTimer > 0;
    const bU = bAct ? clamp(1.0 - this.bTimer / Math.max(this.bDur, 1e-6), 0, 1) : 0.0;

    const { vx, ax } = this._profile();
    const slope = terrainSlope(this.x);
    const ay = terrainCurv(this.x) * vx * vx + slope * ax;
    const x0 = this.x;

    for (let k = 0; k < SUBSTEPS; k++) {
      // OU 平衡噪声（速度越快越大）
      this.ou += (-OU_THETA * this.ou * DT + OU_SIGMA * Math.sqrt(DT) * this.gauss());
      const tauNoise = NOISE_GAIN * this.ou * (0.35 + 0.65 * vx / V_MAX_RUN);
      const tauBump = bAct ? this.bDir * this.bMag * BUMP_GAIN * Math.sin(Math.PI * bU) : 0.0;
      // 地形耦合：竖向加速度调制等效重力 + 坡度直接力矩（上坡后仰 / 下坡前倾）
      const gfac = clamp(1.0 + ay / G_EFF, -0.5, 2.5);
      const tau = K_INV * gfac * Math.sin(this.th) + tauCmd / INERTIA
                + tauNoise + tauBump + ACC_COUPLE * ax
                + K_SLOPE * G_EFF * slope - C_ANG * this.w;
      this.w += tau * DT;
      this.th += this.w * DT;
      this.x += vx * DT;
      this.bTimer = Math.max(this.bTimer - DT, 0.0);
    }

    this.vx = vx; this.ax = ax; this.vxPrev = vx;
    const fallen = Math.abs(this.th) > FALL_ANG;
    this.steps++;
    this.done = fallen;              // 无限模式：无超时
    this.fell = fallen;
    const reward = 1.0 * (this.x - x0) + 0.005 - 0.03 * this.th * this.th
                 - (fallen ? 1.0 : 0.0);
    return { obs: this.obs(), reward, done: this.done, fell: fallen, x: this.x };
  }
}

// ================================================================ Game 1
export class CarEnv {
  constructor(seed = 1) {
    this.taskId = 1;
    this.rng = mulberry32(seed >>> 0);
    this.reset(seed);
  }

  reset(seed) {
    if (seed !== undefined) this.rng = mulberry32(seed >>> 0);
    this.s = 0.0;
    this.d = (this.rng() - 0.5) * 0.8;
    this.dPrev = this.d;
    this.psi = (this.rng() - 0.5) * 0.1;
    this.v = V_START_CAR;
    this.steer = 0.0;
    this.evTimer = 0.0; this.evDur = 1.0; this.evDir = 0.0; this.evMag = 0.0; this.evType = 0.0;
    this.evCool = 0.0;
    this.done = false;
    this.steps = 0;
    this.fell = false;
    return this.obs();
  }

  _cruise() { return Math.min(V_START_CAR + V_RAMP * this.s, V_MAX_CAR); }

  obs() {
    const o = new Float32Array(N_OBS);
    o[0] = this.d / ROAD_HALF;
    o[1] = Math.sin(this.psi);
    o[2] = Math.cos(this.psi);
    o[3] = this.v / V_MAX_CAR;
    o[4] = this.steer / STEER_MAX;
    o[5] = trackCurvature(this.s) * KAPPA_SCALE;
    o[6] = trackCurvature(this.s + LOOKAHEAD[1]) * KAPPA_SCALE;
    o[7] = trackCurvature(this.s + LOOKAHEAD[2]) * KAPPA_SCALE;
    o[8] = trackCurvature(this.s + LOOKAHEAD[3]) * KAPPA_SCALE;
    const active = this.evTimer > 0 ? 1 : 0;
    o[9] = active;
    o[10] = this.evDir * this.evMag * active;
    o[11] = this.evType;
    o[12] = this.v * Math.sin(this.psi) / V_MAX_CAR;
    o[13] = (this.d - this.dPrev) / D_DIFF_SCALE;
    return o;
  }

  step(a) {
    // v4：0 左转 / 1 右转（单档转向）；未按键 = 自动回中
    const steerIn = a === 0 ? -STEER_RATE : a === 1 ? STEER_RATE : 0.0;
    // 转向积分 + 自动回中
    this.steer = clamp(this.steer + (steerIn - STEER_RETURN * this.steer * (steerIn === 0 ? 1 : 0)) * DT * SUBSTEPS,
                       -STEER_MAX, STEER_MAX);

    // 事件触发：打滑看转弯角载荷（驾驶行为后果），爆胎看车速（高速 hazard）。
    // 随机数消耗顺序与 Python 端一致：r1 触发, r2 时长, r3 方向回退, r4 强度, r5 类型
    const r1 = this.rng(), r2 = this.rng(), r3 = this.rng(), r4 = this.rng(), r5 = this.rng();
    const idle = this.evTimer <= 0 && this.evCool <= 0;
    const load = clamp(Math.abs(this.steer) / STEER_MAX, 0, 1) * clamp(this.v / V_MAX_CAR, 0, 1);
    const pSkid = SKID_P * (0.3 + SKID_LOAD_GAIN * load);
    const pBlow = BLOW_P * (0.5 + BLOW_V_GAIN * this.v / V_MAX_CAR);
    const fire = idle && r1 < pSkid + pBlow;
    let vDrop = 1.0;
    if (fire) {
      const isBlow = r5 < pBlow / (pSkid + pBlow);
      this.evType = isBlow ? 1.0 : 0.0;
      this.evTimer = isBlow ? BLOW_T_MIN + r2 * (BLOW_T_MAX - BLOW_T_MIN)
                            : SKID_T_MIN + r2 * (SKID_T_MAX - SKID_T_MIN);
      this.evDur = this.evTimer;
      // 甩尾方向：转弯中沿当前转向方向继续旋转（车尾向外甩），接近直行则随机
      const rndDir = r3 < 0.5 ? -1.0 : 1.0;
      this.evDir = isBlow ? rndDir
                          : (Math.abs(this.steer) > SKID_DIR_TH ? Math.sign(this.steer) : rndDir);
      this.evMag = isBlow ? 0.7 + r4 * 0.3 : 0.4 + r4 * 0.6;
      this.evCool = this.evTimer + EV_COOLDOWN;
      if (isBlow) vDrop = BLOW_VDROP;
    }
    const active = this.evTimer > 0;
    this.evTimer = Math.max(this.evTimer - DT * SUBSTEPS, 0.0);
    this.evCool = Math.max(this.evCool - DT * SUBSTEPS, 0.0);
    // 冲量式衰减包络：刚触发时推力最大，随剩余时间线性衰减到 0
    const envU = clamp(this.evTimer / Math.max(this.evDur, 1e-6), 0, 1);
    const evRate = this.evType > 0 ? BLOW_RATE : SKID_RATE;
    const evLat = this.evType > 0 ? BLOW_LAT : SKID_LAT;
    const evGrip = this.evType > 0 ? BLOW_GRIP : SKID_GRIP;
    const push = active ? this.evDir * this.evMag * envU : 0.0;

    this.v *= vDrop;

    const s0 = this.s;
    this.dPrev = this.d;
    for (let k = 0; k < SUBSTEPS; k++) {
      const kap = trackCurvature(this.s);
      // 爆胎期间巡航目标折减：掉速不会立刻弛豫回去，恢复更慢
      const cruise = this._cruise() * (active && this.evType > 0 ? BLOW_CRUISE : 1.0);
      this.v = clamp(this.v + (cruise - this.v) * K_RELAX * DT, 0.0, V_MAX_CAR);
      const authority = active ? evGrip : 1.0;
      const yaw = this.v * Math.tan(this.steer) * authority / WHEELBASE - this.v * kap
                + push * evRate;
      this.psi = clamp(this.psi + yaw * DT, -1.2, 1.2);
      this.d += (this.v * Math.sin(this.psi) + push * evLat) * DT;
      this.s += this.v * Math.cos(this.psi) * DT;
    }

    const offroad = Math.abs(this.d) > ROAD_HALF;
    this.steps++;
    this.done = offroad;             // 无限模式：无超时
    this.fell = offroad;
    const reward = 1.0 * (this.s - s0) + 0.005 - 0.02 * Math.pow(this.d / ROAD_HALF, 2)
                 - (offroad ? 1.0 : 0.0);
    return { obs: this.obs(), reward, done: this.done, fell: offroad, x: this.s };
  }
}

export function makeEnv(taskId, seed = 1) {
  return taskId === 0 ? new RunnerEnv(seed) : new CarEnv(seed);
}

export const ACTION_NAMES = [
  ["LEAN_BACK", "LEAN_FWD"],
  ["STEER_L", "STEER_R"],
];

export const KEY_HINTS = [
  ["← 后倾", "→ 前倾"],
  ["← 左打", "→ 右打"],
];

export const CONST = { ROAD_HALF, V_MAX_CAR, V_MAX_RUN, FALL_ANG, STEP_DT };
