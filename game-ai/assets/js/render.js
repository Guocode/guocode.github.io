/* 像素风渲染：低分辨率 Canvas + image-rendering:pixelated 放大。
 * 调色板受限（16 色以内），只画矩形与粗线，配合放大形成像素颗粒感。
 */

import { trackCurvature } from './env.js';

export const PAL = {
  sky: '#141a2e', skyHi: '#1b2340',
  hill: '#1a2b3a', hillHi: '#213548',
  ground: '#252a40', groundHi: '#2e3450',
  brick: '#3a4060', brickHi: '#444b70',
  road: '#383e5a', roadHi: '#414869',
  grass: '#1d3a32', grassHi: '#23443b',
  stripe: '#dfe6ff', stripeDim: '#8f9ac4',
  runner: '#64ffda', runnerHi: '#a8fff0', runnerDark: '#2fae94',
  car: '#ffb454', carHi: '#ffd699', carDark: '#c47f2a',
  danger: '#ff6b6b', warn: '#ffd166',
  dim: '#6b7799',
};

const P = (c) => c;

/* ------------------------------------------------------------ 通用 */
function fill(ctx, x, y, w, h, color) {
  ctx.fillStyle = color;
  ctx.fillRect(Math.round(x), Math.round(y), Math.max(1, Math.round(w)), Math.max(1, Math.round(h)));
}

function thickLine(ctx, x0, y0, x1, y1, w, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = w;
  ctx.lineCap = 'butt';
  ctx.beginPath();
  ctx.moveTo(Math.round(x0) + 0.5, Math.round(y0) + 0.5);
  ctx.lineTo(Math.round(x1) + 0.5, Math.round(y1) + 0.5);
  ctx.stroke();
}

/* ------------------------------------------------------------ Game 0 独轮平衡机器人 */
const RW = 0.18;                      // 轮半径 (m)
export function drawRunner(ctx, env, W, H, t = 0) {
  const S = 62;                       // px per meter
  const camX = env.x - 1.1;
  const gy = H - 30;                  // 地面屏幕 y
  const sx = (wx) => (wx - camX) * S;
  const sy = (wy) => gy - wy * S;

  // 天空 + 远山（视差 0.25）
  fill(ctx, 0, 0, W, gy, PAL.sky);
  fill(ctx, 0, 0, W, 26, PAL.skyHi);
  const par = -camX * 0.25 * S;
  for (let i = 0; i < 8; i++) {
    const bx = ((par * 0.6 + i * 74) % (W + 160)) - 80;
    const bh = 16 + ((i * 37) % 22);
    fill(ctx, bx, gy - bh - 8, 62, bh + 8, i % 2 ? PAL.hill : PAL.hillHi);
  }
  // 云
  const cp = -camX * 0.45 * S;
  for (let i = 0; i < 5; i++) {
    const cx = ((cp + i * 96) % (W + 200)) - 100;
    const cy = 14 + ((i * 29) % 26);
    fill(ctx, cx, cy, 18, 4, PAL.skyHi);
    fill(ctx, cx + 5, cy - 3, 9, 4, PAL.skyHi);
  }

  // 地面
  fill(ctx, 0, gy, W, H - gy, PAL.ground);
  fill(ctx, 0, gy, W, 3, PAL.groundHi);

  // 地面砖块（0.28 m 一块）
  const bs = 0.28 * S;
  const startI = Math.floor(camX / 0.28) - 1;
  for (let i = startI; i < startI + Math.ceil(W / bs) + 3; i++) {
    const bx = sx(i * 0.28);
    if (bx < -bs || bx > W) continue;
    fill(ctx, bx, gy + 3, bs - 1, 5, (i % 2 === 0) ? PAL.brick : PAL.brickHi);
    if (i % 8 === 0) fill(ctx, bx, gy + 9, bs - 1, 3, PAL.brick);
  }

  // 每 10 m 一个里程碑
  const m0 = Math.floor(camX / 10) * 10;
  for (let m = m0; m < m0 + 60; m += 10) {
    const mx = sx(m);
    if (mx < -10 || mx > W + 10) continue;
    fill(ctx, mx, gy - 16, 2, 16, PAL.dim);
    fill(ctx, mx, gy - 20, 12, 5, (m % 50 === 0) ? PAL.warn : PAL.dim);
  }

  // 机器人：驱动轮 + 躯干（倒立摆，无腿）
  const wheelX = sx(env.x), wheelY = sy(RW);
  const spin = env.x / RW;                    // 轮子转角（前进驱动）

  // 独轮（像素圆：外圈 + 辐条随前进旋转）
  ctx.fillStyle = PAL.runnerDark;
  ctx.beginPath();
  ctx.arc(wheelX, wheelY, RW * S, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = PAL.sky;                    // 轮辋内侧
  ctx.beginPath();
  ctx.arc(wheelX, wheelY, RW * S - 3, 0, Math.PI * 2);
  ctx.fill();
  // 辐条（随里程旋转，体现"一直在前进"）
  for (let k = 0; k < 3; k++) {
    const ang = spin + (k * Math.PI * 2) / 3;
    thickLine(ctx, wheelX, wheelY,
              wheelX + Math.cos(ang) * (RW * S - 4), wheelY + Math.sin(ang) * (RW * S - 4),
              2, PAL.runnerDark);
  }
  fill(ctx, wheelX - 2, wheelY - 2, 4, 4, PAL.runnerHi);   // 轮毂

  // 躯干（随俯仰旋转，立在轮轴上）
  ctx.save();
  ctx.translate(wheelX, wheelY);
  ctx.rotate(-env.th);
  fill(ctx, -3, -24, 7, 22, PAL.runner);          // 躯干
  fill(ctx, -3, -24, 7, 4, PAL.runnerHi);
  fill(ctx, -4, -32, 9, 8, PAL.runnerHi);         // 头
  fill(ctx, -1, -30, 2, 2, PAL.sky);              // 眼
  // 手臂：倾斜方向反向摆动，像在找平衡
  const swing = Math.sin(t * 0.02) * 4;
  fill(ctx, 3, -20, 6, 2, PAL.runnerDark);
  fill(ctx, -8, -20 + swing, 6, 2, PAL.runnerDark);
  ctx.restore();

  // 摔倒提示
  if (env.fell) {
    fill(ctx, 0, 0, W, H, 'rgba(255,107,107,0.16)');
  }
  return { gy, S };
}

/* ------------------------------------------------------------ Game 1 汽车驾驶 */
export function drawCar(ctx, env, W, H, t = 0) {
  const HORIZON = Math.round(H * 0.34);
  const FOCAL = 150;
  const CAM_H = 1.15;
  const SEG = 44;
  const DZ = 3.2;

  // 天空
  fill(ctx, 0, 0, W, HORIZON, PAL.sky);
  fill(ctx, 0, 0, W, HORIZON * 0.55, PAL.skyHi);
  // 远山
  for (let i = 0; i < 7; i++) {
    const bx = ((i * 68 - env.s * 0.6) % (W + 140)) - 70;
    const bh = 10 + ((i * 31) % 16);
    fill(ctx, bx, HORIZON - bh, 56, bh, i % 2 ? PAL.hill : PAL.hillHi);
  }
  // 草地
  fill(ctx, 0, HORIZON, W, H - HORIZON, PAL.grass);

  // 由远及近画路段
  let prev = null;
  for (let i = SEG; i >= 1; i--) {
    const z = i * DZ;
    const scale = FOCAL / z;
    const y = HORIZON + CAM_H * scale;
    // 弯道：累积曲率产生的横向偏移
    let off = 0;
    for (let k = 1; k <= i; k++) off += trackCurvature(env.s + k * DZ) * DZ * (k * DZ) * 0.5;
    const cx = W / 2 + (off - env.d) * scale;
    const halfW = 3.4 * scale;
    if (prev) {
      // 路面梯形
      ctx.fillStyle = (i % 2 === 0) ? PAL.road : PAL.roadHi;
      ctx.beginPath();
      ctx.moveTo(prev.cx - prev.halfW, prev.y);
      ctx.lineTo(prev.cx + prev.halfW, prev.y);
      ctx.lineTo(cx + halfW, y);
      ctx.lineTo(cx - halfW, y);
      ctx.closePath();
      ctx.fill();
      // 路肩条纹
      if (i % 2 === 0) {
        fill(ctx, prev.cx - prev.halfW - 2, prev.y, 3, Math.max(1, y - prev.y), PAL.stripe);
        fill(ctx, prev.cx + prev.halfW, prev.y, 3, Math.max(1, y - prev.y), PAL.stripe);
      }
      // 中线
      if (i % 4 < 2) fill(ctx, cx - 1, y, 2, Math.max(1, y - prev.y), PAL.stripeDim);
    }
    prev = { cx, y, halfW };
  }

  // 事件指示：打滑（黄色脉冲）/ 爆胎（红色脉冲 + 更强）
  const active = env.evTimer > 0;
  const isBlow = env.evType > 0;
  if (active) {
    const a = (isBlow ? 0.14 : 0.09) + 0.10 * Math.sin(t * 0.5);
    ctx.fillStyle = isBlow ? `rgba(255,107,107,${a})` : `rgba(255,209,102,${a})`;
    ctx.fillRect(0, 0, W, H);
  }

  // 车（第三人称后视）
  const carY = H - 30;
  const shakeAmp = isBlow ? 3 : 2;
  const shake = active ? Math.round(Math.sin(t * (isBlow ? 1.3 : 0.9)) * shakeAmp) : 0;
  const carX = W / 2 - env.d * (FOCAL / (2.5 * DZ)) * 0.55 + shake;
  const tilt = env.steer * 0.25 + (active ? env.evDir * (isBlow ? 0.14 : 0.10) : 0);
  ctx.save();
  ctx.translate(carX, carY);
  ctx.rotate(tilt);
  fill(ctx, -14, -6, 28, 12, PAL.car);            // 车身
  fill(ctx, -10, -14, 20, 9, PAL.carHi);          // 车顶
  fill(ctx, -7, -12, 15, 6, PAL.sky);             // 后窗
  fill(ctx, -15, 2, 6, 6, PAL.carDark);           // 左轮
  fill(ctx, 9, 2, 6, 6, PAL.carDark);             // 右轮
  fill(ctx, -13, -4, 4, 3, PAL.danger);           // 尾灯
  fill(ctx, 9, -4, 4, 3, PAL.danger);
  if (isBlow) fill(ctx, -15, 2, 6, 6, PAL.sky);   // 爆胎：左前轮瘪了（黑圈）
  ctx.restore();

  // 事件粒子：打滑 = 白色胎烟（贴地喷），爆胎 = 深色碎屑上抛
  if (active) {
    const col = isBlow ? (i) => `rgba(255,107,107,${0.55 - i * 0.09})`
                       : (i) => `rgba(223,230,255,${0.45 - i * 0.07})`;
    for (let i = 0; i < 6; i++) {
      const px = carX + (i % 2 ? 12 : -16) + Math.sin(t * 0.3 + i) * 3;
      const py = isBlow
        ? carY + 4 - ((t * 0.5 + i * 9) % 26)
        : carY + 5 - ((t * 0.35 + i * 7) % 12);
      fill(ctx, px, py, 3, 3, col(i));
    }
  }
  return { HORIZON };
}

/* ------------------------------------------------------------ 姿态指示条（画在 HUD canvas） */
export function drawTiltMeter(ctx, tilt, W, H) {
  fill(ctx, 0, 0, W, H, '#0e1424');
  const cx = W / 2, cy = H / 2;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-tilt);
  fill(ctx, -2, -H * 0.32, 4, H * 0.5, PAL.runner);
  fill(ctx, -6, -H * 0.40, 12, 5, PAL.runnerHi);
  ctx.restore();
  fill(ctx, cx - 1, cy, 3, 3, PAL.dim);
}
