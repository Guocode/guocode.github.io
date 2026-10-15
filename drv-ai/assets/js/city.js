/* CityDrive — 城市端到端驾驶环境 JS 镜像
 * 与 train/city.py 物理公式严格一致：
 *   路网/灯态/NPC/导航/自行车模型/碰撞判定
 * 渲染由 index.html 完成，本文件只含环境核心。
 */

// ================================================================ 常量
const G_GRID = 4, B_LEN = 90.0, R_HALF = 3.7, LANE_OFF = 1.85;
const CURB_W = 0.22, LINE_W = 0.16;
const SW_IN = 3.7, SW_OUT = 7.0, BLD_IN = 8.5;
const LIGHT_CYCLE = 16.0, GREEN_T = 6.0, STOP_DIST = 7.0;
const DT = 0.05, MAX_STEPS = 3600;
const N_CARS = 22, N_PEDS = 12, N_BIKES = 6;
const N_NAV = 8, OFF_ROUTE_DIST = 10.0;
const W_PROGRESS = 1.0, W_GOAL = 10.0, W_COLLIDE = 10.0, W_OFFROAD = 5.0;
const W_RED = 2.0, W_REPLAN = 0.2, W_COMFORT_A = 0.02, W_COMFORT_S = 0.02;
const A_MIN = -4.5, A_MAX = 2.0, V_EGO_MAX = 14.0, WHEELBASE = 2.7, STEER_GAIN = 0.42;

// ================================================================ 图 / 导航
function nodeId(i, j) { return i * G_GRID + j; }
function nodeIJ(n) { return [Math.floor(n / G_GRID), n % G_GRID]; }

const NODE_POS = [];
for (let i = 0; i < G_GRID; i++)
  for (let j = 0; j < G_GRID; j++)
    NODE_POS.push([i * B_LEN, j * B_LEN]);

function neighbors(nid) {
  const [i, j] = nodeIJ(nid), out = [];
  if (i > 0) out.push(nodeId(i - 1, j));
  if (i < G_GRID - 1) out.push(nodeId(i + 1, j));
  if (j > 0) out.push(nodeId(i, j - 1));
  if (j < G_GRID - 1) out.push(nodeId(i, j + 1));
  return out;
}

// 有向边表（与 Python 相同生成顺序）
const EDGES = [];
for (let i = 0; i < G_GRID; i++)
  for (let j = 0; j < G_GRID; j++) {
    const cur = nodeId(i, j);
    for (const n of neighbors(cur))
      if (cur < n) { EDGES.push([cur, n]); EDGES.push([n, cur]); }
  }
const N_EDGES = EDGES.length;
const EDGE_ID = new Map();
EDGES.forEach((e, k) => EDGE_ID.set(e[0] + "," + e[1], k));
const EDGE_FROM = EDGES.map(e => e[0]);
const EDGE_DIR = EDGES.map(e => {
  const p = NODE_POS[e[0]], q = NODE_POS[e[1]];
  return [(q[0] - p[0]) / B_LEN, (q[1] - p[1]) / B_LEN];
});
const EDGE_RIGHT = EDGE_DIR.map(d => [d[1], -d[0]]);

function dijkstraRoute(src, dst) {
  const dist = new Map([[src, 0]]), prev = new Map();
  const pq = [[0, src]];
  while (pq.length) {
    pq.sort((a, b) => a[0] - b[0]);
    const [d, u] = pq.shift();
    if (d > (dist.get(u) ?? 1e18)) continue;
    if (u === dst) break;
    for (const v of neighbors(u)) {
      const nd = d + B_LEN;
      if (nd < (dist.get(v) ?? 1e18)) {
        dist.set(v, nd); prev.set(v, u); pq.push([nd, v]);
      }
    }
  }
  const path = [dst];
  while (path[path.length - 1] !== src) path.push(prev.get(path[path.length - 1]));
  return path.reverse();
}

function routePolyline(route) {
  const pts = route.map(n => NODE_POS[n]);
  const seg = [], segLen = [];
  for (let k = 0; k < pts.length - 1; k++) {
    const d = [pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]];
    seg.push(d);
    segLen.push(Math.hypot(d[0], d[1]));
  }
  return { pts, seg, segLen };
}

// ================================================================ 灯态
function lightState(t, travelDir) {
  // travelDir: 0 南北向 1 东西向 → 0 绿 1 黄 2 红
  const ph = Math.floor((t % LIGHT_CYCLE) / (LIGHT_CYCLE / 2));
  const tt = (t % LIGHT_CYCLE) % (LIGHT_CYCLE / 2);
  const yellow = tt >= GREEN_T;
  const myGreen = (ph === 0 && travelDir === 0) || (ph === 1 && travelDir === 1);
  if (myGreen && !yellow) return 0;
  if (myGreen && yellow) return 1;
  return 2;
}

// ================================================================ 路线投影
function projectRoute(x, y, route) {
  const { pts, seg, segLen } = routePolyline(route);
  if (route.length === 1)
    return { prog: 0, dev: Math.hypot(x - pts[0][0], y - pts[0][1]), near: route[0] };
  let bestD = 1e18, bestS = 0, cum = 0;
  for (let k = 0; k < segLen.length; k++) {
    const p0 = pts[k], d = seg[k], L2 = segLen[k] * segLen[k];
    let tt = L2 === 0 ? 0 : ((x - p0[0]) * d[0] + (y - p0[1]) * d[1]) / L2;
    tt = Math.max(0, Math.min(1, tt));
    const px = p0[0] + tt * d[0], py = p0[1] + tt * d[1];
    const dd = Math.hypot(x - px, y - py);
    if (dd < bestD) { bestD = dd; bestS = cum + tt * segLen[k]; }
    cum += segLen[k];
  }
  let near = 0, nd = 1e18;
  NODE_POS.forEach((p, i) => {
    const d = Math.hypot(p[0] - x, p[1] - y);
    if (d < nd) { nd = d; near = i; }
  });
  return { prog: bestS, dev: bestD, near };
}

// ================================================================ 环境
class CityEnv {
  constructor() { this.reset(); }

  reset() {
    this.t = 0.0; this.steps = 0;
    // NPC 车辆
    this.carEdge = []; this.carS = []; this.carV = []; this.carNext = [];
    for (let k = 0; k < N_CARS; k++) {
      this.carEdge.push(Math.floor(Math.random() * N_EDGES));
      this.carS.push(4.0 + Math.random() * (B_LEN - 16.0));
      this.carV.push(4.0 + Math.random() * 3.0);
      this.carNext.push(Math.floor(Math.random() * 3));
    }
    // 行人
    this.pedBlock = [], this.pedSide = [], this.pedP = [], this.pedDir = [];
    for (let k = 0; k < N_PEDS; k++) {
      this.pedBlock.push(Math.floor(Math.random() * (G_GRID - 1) ** 2));
      this.pedSide.push(Math.floor(Math.random() * 4));
      this.pedP.push(4.0 + Math.random() * (B_LEN - 8.0));
      this.pedDir.push(Math.random() < 0.5 ? -1 : 1);
    }
    // 骑车人
    this.bikeEdge = []; this.bikeS = []; this.bikeV = [];
    for (let k = 0; k < N_BIKES; k++) {
      this.bikeEdge.push(Math.floor(Math.random() * N_EDGES));
      this.bikeS.push(4.0 + Math.random() * (B_LEN - 12.0));
      this.bikeV.push(2.5 + Math.random() * 1.5);
    }
    this.npcDirty = true; this.npcCache = null;
    this.spawnGoal(false);
  }

  spawnGoal(keepPos) {
    const nTotal = G_GRID * G_GRID;
    let cur = keepPos ? this._nearestNode() : Math.floor(Math.random() * nTotal);
    let goal = Math.floor(Math.random() * nTotal);
    while (goal === cur) goal = Math.floor(Math.random() * nTotal);
    if (!keepPos) {
      let sx, sy, psi;
      for (let attempt = 0; attempt < 24; attempt++) {
        this.route = dijkstraRoute(cur, goal);
        const startNbr = this.route[1];
        const p = NODE_POS[cur], q = NODE_POS[startNbr];
        const dl = Math.hypot(q[0] - p[0], q[1] - p[1]);
        const dv = [(q[0] - p[0]) / dl, (q[1] - p[1]) / dl];
        sx = p[0] + dv[0] * 6.0 + dv[1] * LANE_OFF;
        sy = p[1] + dv[1] * 6.0 - dv[0] * LANE_OFF;
        psi = Math.atan2(dv[1], dv[0]);
        // 出生避让：6 m 内有 NPC 则换终点重试
        const np = this.npcPositions();
        let ok = true;
        for (let m = 0; m < np.pos.length; m++)
          if (Math.hypot(np.pos[m][0] - sx, np.pos[m][1] - sy) <= 6.0) { ok = false; break; }
        if (ok) break;
        goal = Math.floor(Math.random() * nTotal);
        while (goal === cur) goal = Math.floor(Math.random() * nTotal);
      }
      this.x = sx; this.y = sy; this.psi = psi;
      this.v = 4.0;
    } else {
      this.route = dijkstraRoute(this._nearestNode(), goal);
    }
    const pr = projectRoute(this.x, this.y, this.route);
    this.prog = pr.prog; this.progressPrev = pr.prog; this.dev = pr.dev;
    this.redFlag = true; this.replans = 0; this.goalsDone = 0;
    this.collided = false; this.offroaded = false; this.redRuns = 0;
  }

  _nearestNode() {
    let best = 0, bd = 1e18;
    NODE_POS.forEach((p, i) => {
      const d = Math.hypot(p[0] - this.x, p[1] - this.y);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  // ---------- NPC 位置（类别 0 车 1 骑车 2 行人） ----------
  npcPositions() {
    if (!this.npcDirty && this.npcCache) return this.npcCache;
    const pos = [], cls = [];
    for (let k = 0; k < N_CARS; k++) {
      const e = this.carEdge[k];
      pos.push([
        NODE_POS[EDGE_FROM[e]][0] + EDGE_DIR[e][0] * this.carS[k] + EDGE_RIGHT[e][0] * LANE_OFF,
        NODE_POS[EDGE_FROM[e]][1] + EDGE_DIR[e][1] * this.carS[k] + EDGE_RIGHT[e][1] * LANE_OFF]);
      cls.push(0);
    }
    for (let k = 0; k < N_BIKES; k++) {
      const e = this.bikeEdge[k];
      pos.push([
        NODE_POS[EDGE_FROM[e]][0] + EDGE_DIR[e][0] * this.bikeS[k] + EDGE_RIGHT[e][0] * (R_HALF - 0.8),
        NODE_POS[EDGE_FROM[e]][1] + EDGE_DIR[e][1] * this.bikeS[k] + EDGE_RIGHT[e][1] * (R_HALF - 0.8)]);
      cls.push(1);
    }
    const off = 0.5 * (SW_IN + SW_OUT);   // 5.35 m
    for (let k = 0; k < N_PEDS; k++) {
      const bi = this.pedBlock[k];
      const bx = Math.floor(bi / (G_GRID - 1)) * B_LEN;
      const by = (bi % (G_GRID - 1)) * B_LEN;
      const pp = this.pedP[k];
      let px, py;
      if (this.pedSide[k] === 0) { px = bx + pp; py = by + off; }
      else if (this.pedSide[k] === 1) { px = bx + B_LEN - off; py = by + pp; }
      else if (this.pedSide[k] === 2) { px = bx + pp; py = by + B_LEN - off; }
      else { px = bx + off; py = by + pp; }
      pos.push([px, py]); cls.push(2);
    }
    this.npcCache = { pos, cls };
    this.npcDirty = false;
    return this.npcCache;
  }

  // ---------- NPC 步进 ----------
  _stepNpcs() {
    this.npcDirty = true;
    const t = this.t;
    // 车辆
    for (let k = 0; k < N_CARS; k++) {
      const e = this.carEdge[k];
      const f = EDGES[e][0], ttN = EDGES[e][1];
      const d = EDGE_DIR[e], L = B_LEN;
      const dirCls = Math.abs(d[1]) > Math.abs(d[0]) ? 0 : 1;
      let gap = L;
      for (let m = 0; m < N_CARS; m++) {
        if (m !== k && this.carEdge[m] === e) {
          const ds = this.carS[m] - this.carS[k];
          if (0.5 < ds && ds < gap) gap = ds;
        }
      }
      // 本车纳入跟车避让
      const rx = this.x - NODE_POS[f][0], ry = this.y - NODE_POS[f][1];
      const dsEgo = rx * d[0] + ry * d[1];
      const latEgo = Math.abs(rx * d[1] - ry * d[0]);
      if (latEgo < 2.2 && dsEgo - this.carS[k] > 0.0 && dsEgo - this.carS[k] < gap)
        gap = dsEgo - this.carS[k];
      let vTarget = 7.0;
      const vSafe = Math.sqrt(2.0 * 2.2 * Math.max(gap - 4.0, 0.0));
      vTarget = Math.min(vTarget, vSafe);
      const stopS = L - STOP_DIST;
      if (this.carS[k] < stopS) {
        const ls = lightState(t, dirCls);
        if (ls >= 1) {
          const dd = stopS - this.carS[k];
          if (dd < 22.0)
            vTarget = Math.min(vTarget, Math.sqrt(2.0 * 2.2 * Math.max(dd, 0.0)));
        }
      }
      if (this.carS[k] >= stopS - 0.5 && vTarget < 1.0 && lightState(t, dirCls) >= 1) {
        this.carV[k] = 0.0;
      } else {
        const dv = Math.max(-4 * DT, Math.min(2 * DT, vTarget - this.carV[k]));
        this.carV[k] = Math.max(0.0, Math.min(8.0, this.carV[k] + dv));
      }
      this.carS[k] += this.carV[k] * DT;
      if (this.carS[k] >= L - 1.0 && lightState(t, dirCls) === 0) {
        // 绿灯换边；目标边入口有车/本车则等待
        const nbrs = neighbors(ttN).filter(n => n !== f);
        const nx = nbrs[Math.floor(this.carNext[k] * nbrs.length / 3) % nbrs.length];
        const ne = EDGE_ID.get(ttN + "," + nx);
        const nd = EDGE_DIR[ne], np = NODE_POS[ttN];
        let blocked = false;
        for (let m = 0; m < N_CARS; m++)
          if (m !== k && this.carEdge[m] === ne && this.carS[m] < 9.0) { blocked = true; break; }
        if (!blocked) {
          const ex = this.x - np[0], ey = this.y - np[1];
          const dsE = ex * nd[0] + ey * nd[1];
          const latE = Math.abs(ex * nd[1] - ey * nd[0]);
          if (dsE > 0.0 && dsE < 9.0 && latE < 2.2) blocked = true;
        }
        if (blocked) { this.carS[k] = L - 1.0; this.carV[k] = 0.0; }
        else {
          this.carNext[k] = Math.floor(Math.random() * 3);
          this.carEdge[k] = ne; this.carS[k] = 0.5;
        }
      }
    }
    // 骑车人
    for (let k = 0; k < N_BIKES; k++) {
      this.bikeS[k] += this.bikeV[k] * DT;
      if (this.bikeS[k] >= B_LEN - 1.0) {
        const e = this.bikeEdge[k];
        const f = EDGES[e][0], ttN = EDGES[e][1];
        const nbrs = neighbors(ttN).filter(n => n !== f);
        this.bikeEdge[k] = EDGE_ID.get(ttN + "," + nbrs[Math.floor(Math.random() * nbrs.length)]);
        this.bikeS[k] = 0.5;
      }
    }
    // 行人
    const lim = B_LEN - 8.0;
    for (let k = 0; k < N_PEDS; k++) {
      this.pedP[k] += this.pedDir[k] * 1.3 * DT;
      if (this.pedP[k] > lim && this.pedDir[k] > 0) this.pedDir[k] = -1;
      if (this.pedP[k] < 4.0 && this.pedDir[k] < 0) this.pedDir[k] = 1;
      this.pedP[k] = Math.max(4.0, Math.min(lim, this.pedP[k]));
    }
  }

  // ---------- 导航 ----------
  _navUpdate() {
    let pr = projectRoute(this.x, this.y, this.route);
    if (pr.dev > OFF_ROUTE_DIST) {
      this.route = dijkstraRoute(pr.near, this.route[this.route.length - 1]);
      this.replans++;
      pr = projectRoute(this.x, this.y, this.route);
      this.replanFlag = true;
    } else this.replanFlag = false;
    this.prog = pr.prog; this.dev = pr.dev;
    return pr.prog;
  }

  navInputs() {
    const { pts, seg, segLen } = routePolyline(this.route);
    const total = segLen.reduce((a, b) => a + b, 0);
    const remain = total - this.prog;
    let cum = 0, nxtK = seg.length - 1;
    for (let k = 0; k < segLen.length; k++) {
      if (this.prog <= cum + segLen[k]) { nxtK = k; break; }
      cum += segLen[k];
    }
    const dNext = cum + segLen[nxtK] - this.prog;
    const nodeNext = this.route[nxtK + 1], nodeCur = this.route[nxtK];
    let d1 = [NODE_POS[nodeNext][0] - NODE_POS[nodeCur][0],
              NODE_POS[nodeNext][1] - NODE_POS[nodeCur][1]];
    const l1 = Math.hypot(d1[0], d1[1]) + 1e-9;
    d1 = [d1[0] / l1, d1[1] / l1];
    const h1 = Math.atan2(d1[1], d1[0]);
    const turn1 = ((h1 - this.psi + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    let turn2 = 0.0;
    if (nxtK + 2 <= seg.length - 1) {
      const n2 = this.route[nxtK + 2];
      let d2 = [NODE_POS[n2][0] - NODE_POS[nodeNext][0], NODE_POS[n2][1] - NODE_POS[nodeNext][1]];
      const l2 = Math.hypot(d2[0], d2[1]) + 1e-9;
      d2 = [d2[0] / l2, d2[1] / l2];
      const h2 = Math.atan2(d2[1], d2[0]);
      turn2 = ((h2 - h1 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    }
    const dcur = [Math.cos(this.psi), Math.sin(this.psi)];
    const dirCls = Math.abs(dcur[1]) > Math.abs(dcur[0]) ? 0 : 1;
    const ls = lightState(this.t, dirCls);
    const redAhead = (ls >= 1 && dNext < 30.0) ? 1.0 : 0.0;
    return [
      Math.sin(turn1), Math.cos(turn1), Math.tanh(dNext / 30.0),
      Math.sin(turn2), Math.cos(turn2), Math.tanh(remain / 400.0),
      this.v / V_EGO_MAX, redAhead];
  }

  // ---------- step ----------
  step(aCmd, sCmd) {
    const steer = sCmd * STEER_GAIN;
    const a = Math.max(A_MIN, Math.min(A_MAX, aCmd));
    this.v = Math.max(0.0, Math.min(V_EGO_MAX, this.v + a * DT));
    this.psi += this.v / WHEELBASE * Math.tan(steer) * DT;
    this.x += this.v * Math.cos(this.psi) * DT;
    this.y += this.v * Math.sin(this.psi) * DT;
    this.t += DT; this.steps++;
    this._stepNpcs();
    const progPrev = this.progressPrev;
    const prog = this._navUpdate();
    this.progressPrev = prog;

    let reward = W_PROGRESS * (prog - progPrev)
      - W_COMFORT_A * Math.abs(a) - W_COMFORT_S * sCmd * sCmd;
    let event = "";
    // 碰撞
    const np = this.npcPositions();
    for (let m = 0; m < np.pos.length; m++) {
      const dd = Math.hypot(np.pos[m][0] - this.x, np.pos[m][1] - this.y);
      const thr = np.cls[m] === 0 ? 2.4 : (np.cls[m] === 1 ? 1.9 : 1.3);
      if (dd < thr) { reward -= W_COLLIDE; event = "collision"; this.collided = true; break; }
    }
    // 冲出路面
    if (!event && !this.staticRoad(this.x, this.y)) {
      reward -= W_OFFROAD; event = "offroad"; this.offroaded = true;
    }
    // 闯红灯
    const dcur = [Math.cos(this.psi), Math.sin(this.psi)];
    const dirCls = Math.abs(dcur[1]) > Math.abs(dcur[0]) ? 0 : 1;
    const { seg, segLen } = routePolyline(this.route);
    let cum = 0, nxtK = seg.length - 1;
    for (let k = 0; k < segLen.length; k++) {
      if (this.prog <= cum + segLen[k]) { nxtK = k; break; }
      cum += segLen[k];
    }
    const lp = NODE_POS[this.route[nxtK + 1]];
    const dNext = Math.hypot(lp[0] - this.x, lp[1] - this.y);
    const ls = lightState(this.t, dirCls);
    if (dNext < STOP_DIST && ls >= 1 && !this.redFlag) {
      this.redFlag = true; this.redRuns++;
      if (!event) { reward -= W_RED; event = "red_light"; }
    }
    if (dNext > STOP_DIST + 5) this.redFlag = false;
    // 到达终点
    const remain = segLen.reduce((x, y) => x + y, 0) - this.prog;
    if (remain < 4.0) {
      reward += W_GOAL; this.goalsDone++;
      this.spawnGoal(true);
      event = event || "goal";
    }
    if (this.replanFlag) reward -= W_REPLAN;
    let done = false;
    if (this.steps >= MAX_STEPS) { done = true; event = event || "timeout"; }
    if (event === "collision" || event === "offroad") done = true;
    return { reward, done, event };
  }

  // ---------- 路面查询（与 Python static_sem 的 road 分支一致） ----------
  staticRoad(px, py) {
    const span = (G_GRID - 1) * B_LEN;
    const dxv = Math.abs(px / B_LEN - Math.round(px / B_LEN)) * B_LEN;
    const dyh = Math.abs(py / B_LEN - Math.round(py / B_LEN)) * B_LEN;
    const inV = py >= -R_HALF && py <= span + R_HALF;
    const inH = px >= -R_HALF && px <= span + R_HALF;
    return (dxv <= R_HALF && inV) || (dyh <= R_HALF && inH);
  }

  // ---------- 启发式专家（与 train/city.py expert() 一致） ----------
  expert() {
    const { pts, seg, segLen } = routePolyline(this.route);
    const total = segLen.reduce((x, y) => x + y, 0);
    const look = Math.max(4.5, Math.min(10.0, 0.7 * this.v + 3.5));
    const targetS = Math.min(this.prog + look, total - 0.1);
    let cum = 0, tp = pts[pts.length - 1];
    for (let k = 0; k < segLen.length; k++) {
      if (targetS <= cum + segLen[k]) {
        const tt = (targetS - cum) / Math.max(segLen[k], 1e-6);
        tp = [pts[k][0] + tt * seg[k][0], pts[k][1] + tt * seg[k][1]];
        const nl = Math.max(Math.hypot(seg[k][0], seg[k][1]), 1e-9);
        tp = [tp[0] + seg[k][1] / nl * LANE_OFF, tp[1] - seg[k][0] / nl * LANE_OFF];
        break;
      }
      cum += segLen[k];
    }
    const dx = tp[0] - this.x, dy = tp[1] - this.y;
    const hErr = ((Math.atan2(dy, dx) - this.psi + Math.PI) % (2 * Math.PI)
      + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    const steer = Math.max(-1.0, Math.min(1.0, 1.6 * hErr));
    let vT = 8.0;
    let cum2 = 0, nxtK = seg.length - 1;
    for (let k = 0; k < segLen.length; k++) {
      if (this.prog <= cum2 + segLen[k]) { nxtK = k; break; }
      cum2 += segLen[k];
    }
    const nodeNext = this.route[nxtK + 1];
    const dNext = Math.hypot(NODE_POS[nodeNext][0] - this.x, NODE_POS[nodeNext][1] - this.y);
    // 转向前减速
    if (nxtK + 1 <= seg.length - 1) {
      const a1 = [NODE_POS[nodeNext][0] - NODE_POS[this.route[nxtK]][0],
                  NODE_POS[nodeNext][1] - NODE_POS[this.route[nxtK]][1]];
      const n2 = nxtK + 2 < this.route.length ? this.route[nxtK + 2] : nodeNext;
      const a2 = [NODE_POS[n2][0] - NODE_POS[nodeNext][0], NODE_POS[n2][1] - NODE_POS[nodeNext][1]];
      const ang = Math.acos(Math.max(-1, Math.min(1,
        (a1[0] * a2[0] + a1[1] * a2[1]) /
        (Math.hypot(a1[0], a1[1]) * Math.hypot(a2[0], a2[1]) + 1e-9))));
      if (ang > 0.3 && dNext < 22.0) vT = Math.min(vT, 3.5);
    }
    // 转弯中减速
    const nl = Math.max(Math.hypot(seg[nxtK][0], seg[nxtK][1]), 1e-9);
    const hd = Math.atan2(seg[nxtK][1] / nl, seg[nxtK][0] / nl);
    const hdev = Math.abs(((hd - this.psi + Math.PI) % (2 * Math.PI)
      + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
    if (hdev > 0.35) vT = Math.min(vT, 3.0);
    // 红灯
    const dcur = [Math.cos(this.psi), Math.sin(this.psi)];
    const dirCls = Math.abs(dcur[1]) > Math.abs(dcur[0]) ? 0 : 1;
    if (lightState(this.t, dirCls) >= 1 && dNext < 30.0) {
      const stopAt = dNext - STOP_DIST + 1.5;
      vT = Math.min(vT, Math.sqrt(2.0 * 2.6 * Math.max(stopAt, 0.0)));
    }
    // 前车 / 骑车人
    const np = this.npcPositions();
    const fwd = [Math.cos(this.psi), Math.sin(this.psi)];
    let gap = 1e18;
    for (let m = 0; m < np.pos.length; m++) {
      if (np.cls[m] > 1) continue;
      const rx = np.pos[m][0] - this.x, ry = np.pos[m][1] - this.y;
      const along = rx * fwd[0] + ry * fwd[1];
      const lat = Math.abs(rx * fwd[1] - ry * fwd[0]);
      if (along > 1.0 && along < 20.0 && lat < 2.0 && along < gap) gap = along;
    }
    if (gap < 1e17)
      vT = Math.min(vT, Math.sqrt(2.0 * 4.0 * Math.max(gap - 6.0, 0.0)));
    const a = Math.max(A_MIN, Math.min(A_MAX, 3.0 * (vT - this.v)));
    return [a, steer];
  }
}
