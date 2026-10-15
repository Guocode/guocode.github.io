/* Pixel Arena — 浏览器端物理环境
 *
 * 常量与积分公式与 train/env.py 逐行对应。
 * 两个游戏共享 4 个离散动作与 14 维观测，因此能被同一个 MoE 模型驱动。
 * 注意：Python 端为向量化实现（批量预生成随机数），浏览器端为单实例流式取数，
 * 因此逐位一致无法保证，采用"统计一致性"校验（同策略下两者成绩分布接近）。
 */

export const DT = 1 / 120;
export const SUBSTEPS = 2;
export const N_OBS = 14;
export const MAX_STEPS = 1500;

// ---------------------------------------------------------------- Game 0 RUNNER
const G = 20.0, MASS = 1.0, INERTIA = 0.45, HIP_OFF = 0.22, L0 = 0.55;
const K_LEG = 800.0, C_LEG = 30.0, PUSH = 60.0, V_CUT = 5.0, CHARGE_T = 0.30, F_MAX = 90.0;
const KP_PITCH = 14.0, KD_PITCH = 2.6, TORQUE_MAX = 6.0, C_ANG = 0.30, PITCH_COUPLE = 0.35;
const TILT_FWD = -0.35, TILT_BACK = 0.12, TILT_INIT = -0.15;
const DRAG = 1.4, SWING_T = 0.10, STEP_LEN = 0.16, STEP_MAX = 0.21, STEP_H = 0.15;
const LEAN_GAIN = 1.2, FALL_ANG = 0.90, MIN_Y = 0.34, V_MAX_RUN = 8.0;
const FALL_PENALTY = 1.0;

// ---------------------------------------------------------------- Game 1 CAR
const ACC = 6.0, BRAKE_FORCE = 9.0, CAR_DRAG = 0.22, V_MAX_CAR = 30.0;
const STEER_RATE = 2.4, STEER_MAX = 0.55, STEER_RETURN = 2.0, WHEELBASE = 2.6;
const ROAD_HALF = 3.4, KAPPA_SCALE = 80.0;
const LOOKAHEAD = [0.0, 12.0, 28.0, 50.0];
const DISTURB_P = 0.010, DISTURB_T_MIN = 0.4, DISTURB_T_MAX = 1.3;
const D_RATE = 0.95, D_LAT = 1.30, V_STALL = 0.6;
const CRASH_PENALTY = 1.0;

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

  reset(seed) {
    if (seed !== undefined) this.rng = mulberry32(seed >>> 0);
    this.x = 0.0;
    this.y = L0 * 0.94 + HIP_OFF + (this.rng() - 0.5) * 0.04;
    this.vx = 0.0; this.vy = 0.0;
    this.th = (this.rng() - 0.5) * 0.1;
    this.w = 0.0;
    this.stance = [true, true];
    this.fx = [-0.12, 0.12];
    this.fy = [0.0, 0.0];
    this.sw_t = [0.0, 0.0];
    this.sw_x0 = [-0.12, 0.12];
    this.charge = [1.0, 1.0];
    this.dragged = [false, false];
    this.thTarget = TILT_INIT;
    this.done = false;
    this.steps = 0;
    this.fell = false;
    return this.obs();
  }

  obs() {
    const hipX = this.x + HIP_OFF * Math.sin(this.th);
    const o = new Float32Array(N_OBS);
    const ul = clamp(this.sw_t[0] / SWING_T, 0, 1);
    const ur = clamp(this.sw_t[1] / SWING_T, 0, 1);
    const sl = this.stance[0] ? 0 : 1;
    const sr = this.stance[1] ? 0 : 1;
    o[0] = this.y / L0;
    o[1] = this.vx / V_MAX_RUN;
    o[2] = this.vy / V_MAX_RUN;
    o[3] = Math.sin(this.th);
    o[4] = Math.cos(this.th);
    o[5] = this.w / 6.0;
    o[6] = sl; o[7] = ul * sl; o[8] = (this.fx[0] - hipX) / L0; o[9] = this.fy[0] / L0;
    o[10] = sr; o[11] = ur * sr; o[12] = (this.fx[1] - hipX) / L0; o[13] = this.fy[1] / L0;
    return o;
  }

  _substep(trigger) {
    // 主动起摆
    for (let i = 0; i < 2; i++) {
      if (this.stance[i] && trigger[i]) {
        this.sw_x0[i] = this.fx[i];
        this.sw_t[i] = 0.0;
        this.stance[i] = false;
        this.dragged[i] = false;
      }
    }
    const hipX = this.x + HIP_OFF * Math.sin(this.th);
    const hipY = this.y - HIP_OFF * Math.cos(this.th);
    const rx = hipX - this.x, ry = hipY - this.y;
    let Fx = 0.0, Fy = 0.0, tau = 0.0;

    for (let i = 0; i < 2; i++) {
      const st = this.stance[i];
      const dx = hipX - this.fx[i];
      const dy = hipY - this.fy[i];
      const L = Math.max(Math.hypot(dx, dy), 1e-3);
      const ux = dx / L, uy = dy / L;

      const Ldot = (this.vx * dx + this.vy * dy) / L;
      const fmag = clamp(K_LEG * (L0 - L) - C_LEG * Ldot, -F_MAX, F_MAX);
      const compress = clamp((L0 - L) / (0.15 * L0), 0, 1);
      const rear = clamp(dx / (0.30 * L0), 0, 1);
      const fade = clamp(1.0 - this.vx / V_CUT, 0, 1);
      const fsx = PUSH * this.charge[i] * (0.35 + 0.65 * rear) * (0.30 + 0.70 * compress) * fade;
      const fsy = fmag * uy;

      if (st) {
        Fx += fsx; Fy += fsy;
        tau += rx * fsy - ry * fsx * PITCH_COUPLE;
      }
      // 被动拖离
      const otherSt = this.stance[1 - i];
      if (st && L > L0 * 1.08 && otherSt) {
        this.stance[i] = false;
        this.sw_t[i] = 0.0;
        this.sw_x0[i] = this.fx[i];
        this.charge[i] = 0.0;
        this.dragged[i] = true;
      }
      const sw = !this.stance[i];
      let stepEff = STEP_LEN * (1.0 + LEAN_GAIN * Math.max(0.0, -this.th)) + 0.05 * this.vx;
      stepEff = clamp(stepEff, 0.05, STEP_MAX);
      const target = hipX + stepEff;
      const t = sw ? this.sw_t[i] + DT : this.sw_t[i];
      const u = clamp(t / SWING_T, 0, 1);
      if (sw) {
        this.fx[i] = this.sw_x0[i] + (target - this.sw_x0[i]) * u;
        this.fy[i] = Math.sin(Math.PI * u) * STEP_H;
      } else {
        this.fy[i] = 0.0;
      }
      this.sw_t[i] = t;
      const landed = sw && u >= 1.0;
      if (landed) {
        this.stance[i] = true;
        this.charge[i] = this.dragged[i] ? 0.0 : 1.0;
        this.dragged[i] = false;
      } else {
        this.charge[i] = Math.max(this.charge[i] - DT / CHARGE_T, 0.0);
      }
    }

    Fx -= DRAG * this.vx;
    const tauCtrl = clamp(KP_PITCH * (this.thTarget - this.th) - KD_PITCH * this.w,
                          -TORQUE_MAX, TORQUE_MAX);
    tau += tauCtrl - C_ANG * this.w;

    this.w += (tau / INERTIA) * DT;
    this.th += this.w * DT;
    this.vx += (Fx / MASS) * DT;
    this.vy += (Fy / MASS - G) * DT;
    this.x += this.vx * DT;
    this.y += this.vy * DT;
  }

  step(a) {
    const trigger = [a === 0, a === 1];
    if (a === 2) this.thTarget = TILT_FWD;
    else if (a === 3) this.thTarget = TILT_BACK;
    const x0 = this.x;
    for (let k = 0; k < SUBSTEPS; k++) {
      this._substep(k === 0 ? trigger : [false, false]);
    }
    const fallen = Math.abs(this.th) > FALL_ANG || this.y < MIN_Y;
    this.steps++;
    const timeout = this.steps >= MAX_STEPS;
    let reward = 1.0 * (this.x - x0) + 0.005 - 0.03 * this.th * this.th;
    if (fallen) reward -= FALL_PENALTY;
    this.done = fallen || timeout;
    this.fell = fallen;
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
    this.psi = (this.rng() - 0.5) * 0.1;
    this.v = 6.0;
    this.steer = 0.0;
    this.dTimer = 0.0; this.dDir = 0.0; this.dMag = 0.0;
    this.vPrev = 6.0;
    this.done = false;
    this.steps = 0;
    this.fell = false;
    return this.obs();
  }

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
    const active = this.dTimer > 0 ? 1 : 0;
    o[9] = active;
    o[10] = this.dDir * this.dMag * active;
    o[11] = this.v * Math.sin(this.psi) / V_MAX_CAR;
    o[12] = Math.abs(this.d) / ROAD_HALF;
    o[13] = (this.v - this.vPrev) / V_MAX_CAR;
    return o;
  }

  step(a) {
    const steerIn = a === 0 ? -STEER_RATE : a === 1 ? STEER_RATE : 0.0;
    this.steer = clamp(this.steer + (steerIn - STEER_RETURN * this.steer * (steerIn === 0 ? 1 : 0)) * DT * SUBSTEPS,
                       -STEER_MAX, STEER_MAX);

    // 轮子失控事件（随机数消耗顺序与 Python 端保持一致）
    const r1 = this.rng(), r2 = this.rng(), r3 = this.rng(), r4 = this.rng();
    const idle = this.dTimer <= 0;
    const fire = idle && r1 < DISTURB_P;
    if (fire) {
      this.dTimer = DISTURB_T_MIN + r2 * (DISTURB_T_MAX - DISTURB_T_MIN);
      this.dDir = r3 < 0.5 ? -1.0 : 1.0;
      this.dMag = 0.5 + r4 * 0.5;
    }
    const active = this.dTimer > 0;
    this.dTimer = Math.max(this.dTimer - DT * SUBSTEPS, 0.0);

    const s0 = this.s;
    this.vPrev = this.v;
    for (let k = 0; k < SUBSTEPS; k++) {
      const kap = trackCurvature(this.s);
      const accel = a === 2 ? ACC : a === 3 ? -BRAKE_FORCE : 0.0;
      this.v = clamp(this.v + (accel - CAR_DRAG * this.v) * DT, 0.0, V_MAX_CAR);
      let yaw = this.v * Math.tan(this.steer) / WHEELBASE - this.v * kap;
      if (active) yaw += this.dDir * this.dMag * D_RATE;
      this.psi = clamp(this.psi + yaw * DT, -1.2, 1.2);
      let lat = this.v * Math.sin(this.psi);
      if (active) lat += this.dDir * this.dMag * D_LAT;
      this.d += lat * DT;
      this.s += this.v * Math.cos(this.psi) * DT;
    }

    const offroad = Math.abs(this.d) > ROAD_HALF;
    const stall = this.v < V_STALL;
    this.steps++;
    const timeout = this.steps >= MAX_STEPS;
    let reward = 1.0 * (this.s - s0) + 0.005 - 0.02 * Math.pow(this.d / ROAD_HALF, 2);
    if (offroad) reward -= CRASH_PENALTY;
    this.done = offroad || stall || timeout;
    this.fell = offroad;
    return { obs: this.obs(), reward, done: this.done, fell: offroad, x: this.s };
  }
}

export function makeEnv(taskId, seed = 1) {
  return taskId === 0 ? new RunnerEnv(seed) : new CarEnv(seed);
}

export const ACTION_NAMES = [
  ["STEP_L", "STEP_R", "LEAN_FWD", "LEAN_BACK"],
  ["STEER_L", "STEER_R", "ACCEL", "BRAKE"],
];

export const KEY_HINTS = [
  ["← 迈左腿", "→ 迈右腿", "↑ 前倾", "↓ 收身"],
  ["← 左打方向", "→ 右打方向", "↑ 油门", "↓ 刹车"],
];

export const CONST = { L0, HIP_OFF, ROAD_HALF, V_MAX_CAR, V_MAX_RUN, MIN_Y, FALL_ANG };
