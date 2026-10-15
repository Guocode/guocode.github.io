/* 像素风渲染：低分辨率 Canvas + image-rendering:pixelated 放大。
 * 调色板受限（16 色以内），只画矩形与粗线，配合放大形成像素颗粒感。
 */

import { trackCurvature, terrainH, CONST } from './env.js';

const ROAD_HALF = CONST.ROAD_HALF;

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

  // 起伏地面：逐列贴着地形高度填充（伪随机起伏，与环境物理同函数）
  const COL = 2;
  for (let px = 0; px < W; px += COL) {
    const wx = camX + px / S;
    const top = sy(terrainH(wx));
    const brick = (Math.floor(wx / 0.28) % 2 === 0) ? PAL.brick : PAL.brickHi;
    fill(ctx, px, top + 3, COL, H - top - 3, PAL.ground);
    fill(ctx, px, top, COL, 3, PAL.groundHi);
    fill(ctx, px, top + 3, COL, 5, brick);
  }

  // 每 10 m 一个里程碑（立在起伏地表上）
  const m0 = Math.floor(camX / 10) * 10;
  for (let m = m0; m < m0 + 60; m += 10) {
    const mx = sx(m);
    if (mx < -10 || mx > W + 10) continue;
    const mh = sy(terrainH(m));
    fill(ctx, mx, mh - 16, 2, 16, PAL.dim);
    fill(ctx, mx, mh - 20, 12, 5, (m % 50 === 0) ? PAL.warn : PAL.dim);
  }

  // 机器人：驱动轮 + 躯干（倒立摆，无腿），轮子贴着起伏地表
  const wheelX = sx(env.x), wheelY = sy(terrainH(env.x) + RW);
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
// 梯形面（路段 / 路缘石通用）
function trap(ctx, x0L, x0R, y0, x1L, x1R, y1, col) {
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.moveTo(x0L, y0);
  ctx.lineTo(x0R, y0);
  ctx.lineTo(x1R, y1);
  ctx.lineTo(x1L, y1);
  ctx.closePath();
  ctx.fill();
}

// 水平箭头（像素风）：指出"车正被往哪边拽"
function hArrow(ctx, cx, cy, dir, col) {
  fill(ctx, cx - 7, cy - 1, 14, 2, col);
  if (dir > 0) {
    thickLine(ctx, cx + 1, cy - 4, cx + 7, cy, 2, col);
    thickLine(ctx, cx + 1, cy + 4, cx + 7, cy, 2, col);
  } else {
    thickLine(ctx, cx - 1, cy - 4, cx - 7, cy, 2, col);
    thickLine(ctx, cx - 1, cy + 4, cx - 7, cy, 2, col);
  }
}

export function drawCar(ctx, env, W, H, t = 0) {
  const HORIZON = Math.round(H * 0.34);
  const FOCAL = 150;
  const CAM_H = 2.0;                  // 相机高度：抬高后近景路面不会宽出画面太多
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

  // 由远及近画路段。条纹世界锚定（随里程向玩家滚动，不再固定在屏幕上）：
  // 红白路缘石 = 碰撞边界，压线即冲出路面
  let prev = null;
  for (let i = SEG; i >= 1; i--) {
    const z = i * DZ;
    const scale = FOCAL / z;
    const y = HORIZON + CAM_H * scale;
    // 弯道：累积曲率产生的横向偏移
    let off = 0;
    for (let k = 1; k <= i; k++) off += trackCurvature(env.s + k * DZ) * DZ * (k * DZ) * 0.5;
    const cx = W / 2 + (off - env.d) * scale;
    const halfW = ROAD_HALF * scale;
    if (prev) {
      const ws = env.s + i * DZ;                      // 本段世界里程
      // 路面
      trap(ctx, prev.cx - prev.halfW, prev.cx + prev.halfW, prev.y,
           cx - halfW, cx + halfW, y,
           Math.floor(ws / DZ) % 2 === 0 ? PAL.road : PAL.roadHi);
      // 路缘石（碰撞边界）：路宽尽头，红白相间
      const curbW = Math.max(1.5, 0.55 * scale);
      const curbCol = Math.floor(ws / 2.6) % 2 === 0 ? PAL.danger : PAL.stripe;
      trap(ctx, prev.cx - prev.halfW - curbW, prev.cx - prev.halfW, prev.y,
           cx - halfW - curbW, cx - halfW, y, curbCol);
      trap(ctx, prev.cx + prev.halfW, prev.cx + prev.halfW + curbW, prev.y,
           cx + halfW, cx + halfW + curbW, y, curbCol);
    }
    prev = { cx, y, halfW };
  }

  // 近景延伸：把最近路段一路铺到画面最底部（车就压在路面上）
  {
    const zN = 2.5;
    const scaleN = FOCAL / zN;
    const yN = HORIZON + CAM_H * scaleN;
    const offN = trackCurvature(env.s + zN) * zN * zN * 0.5;
    const cxN = W / 2 + (offN - env.d) * scaleN;
    const halfN = ROAD_HALF * scaleN;
    const wsN = env.s + zN * 1.2;
    trap(ctx, prev.cx - prev.halfW, prev.cx + prev.halfW, prev.y,
         cxN - halfN, cxN + halfN, yN,
         Math.floor(wsN / DZ) % 2 === 0 ? PAL.road : PAL.roadHi);
    const curbN = Math.min(Math.max(2, 0.55 * scaleN), 26);   // 近景路缘封顶，避免巨型斜条
    const w0 = Math.max(1.5, 0.55 * (FOCAL / DZ));   // 衔接最近路段的路缘宽
    const colN = Math.floor(wsN / 2.6) % 2 === 0 ? PAL.danger : PAL.stripe;
    trap(ctx, prev.cx - prev.halfW - w0, prev.cx - prev.halfW, prev.y,
         cxN - halfN - curbN, cxN - halfN, yN, colN);
    trap(ctx, prev.cx + prev.halfW, prev.cx + prev.halfW + w0, prev.y,
         cxN + halfN, cxN + halfN + curbN, yN, colN);
  }

  // 中心虚线：按世界里程带投影（3.2 m 实 + 3.2 m 虚），从近景裁剪面一直画到地平线，
  // 与路段网格解耦，弯道上也是连续对齐的
  {
    const PERIOD = 6.4, DASH = 3.2, ZNEAR = 2.5, ZFAR = SEG * DZ;
    const offAt = (z) => {              // 与路段一致的横向偏移：整段累加 + 部分段
      const m = Math.floor(z / DZ), rem = z - m * DZ;
      let o = 0;
      for (let k = 1; k <= m; k++) o += trackCurvature(env.s + k * DZ) * DZ * (k * DZ) * 0.5;
      if (rem > 0) o += trackCurvature(env.s + m * DZ + rem) * rem * (m * DZ + rem) * 0.5;
      return o;
    };
    for (let kb = Math.floor((env.s + ZNEAR) / PERIOD); kb * PERIOD < env.s + ZFAR; kb++) {
      const z1 = Math.max(kb * PERIOD - env.s, ZNEAR);
      const z2 = Math.min(kb * PERIOD + DASH - env.s, ZFAR);
      if (z2 <= z1) continue;
      const sc1 = FOCAL / z1, y1 = HORIZON + CAM_H * sc1;
      const sc2 = FOCAL / z2, y2 = HORIZON + CAM_H * sc2;
      const c1 = W / 2 + (offAt(z1) - env.d) * sc1;
      const c2 = W / 2 + (offAt(z2) - env.d) * sc2;
      const w1 = Math.min(5, Math.max(1, 0.10 * sc1));
      const w2 = Math.min(5, Math.max(1, 0.10 * sc2));
      trap(ctx, c1 - w1 / 2, c1 + w1 / 2, y1, c2 - w2 / 2, c2 + w2 / 2, y2, PAL.stripe);
    }
  }

  // 远景雾（静态，增强纵深）
  for (let r = 0; r < 6; r++) {
    ctx.fillStyle = `rgba(20,26,46,${(0.26 - r * 0.042).toFixed(3)})`;
    ctx.fillRect(0, HORIZON + r * 2, W, 2);
  }

  // 临边警示：靠近路缘的一侧亮红色边缘（静态强度，随接近程度加深，不闪烁）
  const edgeN = Math.min(Math.abs(env.d) / ROAD_HALF, 1.2);
  if (edgeN > 0.6) {
    const p = Math.min((edgeN - 0.6) / 0.4, 1);
    ctx.fillStyle = `rgba(255,80,80,${(0.05 + 0.22 * p).toFixed(3)})`;
    if (env.d > 0) fill(ctx, W - 30, HORIZON, 30, H - HORIZON, ctx.fillStyle);
    else fill(ctx, 0, HORIZON, 30, H - HORIZON, ctx.fillStyle);
  }

  // 事件反馈：不再全屏闪烁。改为顶部色条 + 车旁方向箭头（指明被拽方向，反向打盘修正）
  const active = env.evTimer > 0;
  const isBlow = env.evType > 0;
  const carY = H - 30;

  // 车（第三人称后视）
  const shake = active ? Math.sin(t * 0.55) * (isBlow ? 1.6 : 1.0) : 0;
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
  if (active && isBlow) {                         // 爆胎轮随方向瘪掉
    if (env.evDir < 0) fill(ctx, -15, 2, 6, 6, PAL.sky);
    else fill(ctx, 9, 2, 6, 6, PAL.sky);
  }
  ctx.restore();

  if (active) {
    // 顶部事件色条（静态）
    fill(ctx, 0, 0, W, 3, isBlow ? PAL.danger : PAL.warn);
    // 方向箭头（车上方 22px）：指向被拽方向
    hArrow(ctx, Math.round(carX), carY - 26, env.evDir > 0 ? 1 : -1,
           isBlow ? PAL.danger : PAL.warn);
    // 事件粒子：打滑 = 白色胎烟（贴地喷），爆胎 = 深色碎屑上抛
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
