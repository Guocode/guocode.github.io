/* 页面编排：两个游戏共用同一个 MoE 模型，实时可视化专家路由。 */

import { RunnerEnv, CarEnv, MAX_STEPS, ACTION_NAMES, KEY_HINTS } from './env.js';
import { drawRunner, drawCar } from './render.js';
import { loadPolicy } from './moe.js';
import { METRICS } from './metrics.js';

const W = 320, H = 180;
const STEP_DT = 1 / 60;

const $ = (s) => document.querySelector(s);
const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------ 单个游戏实例 */
class Game {
  constructor(taskId, canvasId) {
    this.taskId = taskId;
    this.canvas = el(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.canvas.width = W;
    this.canvas.height = H;
    this.ctx.imageSmoothingEnabled = false;
    this.env = taskId === 0 ? new RunnerEnv(7) : new CarEnv(7);
    this.mode = 'ai';                 // ai | human
    this.best = 0;
    this.lastGate = [0, 0, 0, 0];
    this.lastAction = -1;
    this.finished = false;
    this.finishInfo = '';
  }

  reset(seed = Math.floor(Math.random() * 100000)) {
    this.env.reset(seed);
    this.finished = false;
    this.finishInfo = '';
  }

  advance(policy, humanAction) {
    if (this.finished) return;
    let action, gate = null;
    if (this.mode === 'ai' && policy) {
      const r = policy.act(this.env.obs(), this.taskId);
      action = r.action;
      gate = Array.from(r.gate);
      this.lastGate = gate;
    } else {
      action = humanAction !== undefined ? humanAction : -1;
      if (action < 0) return;          // 人类未按任何键：空转
      this.lastGate = [0, 0, 0, 0];
    }
    this.lastAction = action;
    const out = this.env.step(action);
    if (this.env.x > this.best) this.best = this.env.x;
    if (out.done) {
      this.finished = true;
      this.finishInfo = out.fell ? (this.taskId === 0 ? '摔倒' : '冲出路面')
                                 : '跑完全程';
      setTimeout(() => this.reset(), 1200);
    }
    return gate;
  }

  render(t) {
    const e = this.env;
    if (this.taskId === 0) drawRunner(this.ctx, e, W, H, t);
    else drawCar(this.ctx, e, W, H, t);
  }
}

/* ------------------------------------------------------------ 路由面板 */
function renderGateBars(containerId, gate, active) {
  const box = el(containerId);
  if (!box._built) {
    box.innerHTML = '';
    for (let i = 0; i < 4; i++) {
      const row = document.createElement('div');
      row.className = 'gate-row';
      row.innerHTML = `<span class="gate-name">E${i}</span>
        <span class="gate-track"><i></i></span>
        <span class="gate-val">0.00</span>`;
      box.appendChild(row);
    }
    box._built = true;
  }
  const rows = box.querySelectorAll('.gate-row');
  rows.forEach((r, i) => {
    const v = gate[i] || 0;
    r.querySelector('i').style.width = Math.round(v * 100) + '%';
    r.querySelector('.gate-val').textContent = v.toFixed(2);
    r.classList.toggle('active', v > 0.01);
    r.classList.toggle('muted', !active);
  });
}

/* ------------------------------------------------------------ 主流程 */
async function main() {
  const runner = new Game(0, 'cv-runner');
  const car = new Game(1, 'cv-car');
  const games = [runner, car];

  let policy = null;
  try {
    policy = await loadPolicy('assets/models/moe_weights.json');
    el('model-status').textContent =
      `MoE 已载入 · ${METRICS.params.toLocaleString()} 参数 · ${METRICS.weights_kb} KB`;
    el('model-status').className = 'status ok';
  } catch (err) {
    el('model-status').textContent = '模型未载入：' + err.message + '（仍可手动游玩）';
    el('model-status').className = 'status warn';
    console.warn(err);
  }

  // 模式：0 = AI 同时玩两个 / 1 = 我玩机器人 / 2 = 我玩汽车
  let mode = 0;
  let speed = 1;
  let paused = false;
  const keys = new Set();

  function applyMode() {
    runner.mode = mode === 1 ? 'human' : 'ai';
    car.mode = mode === 2 ? 'human' : 'ai';
    document.querySelectorAll('[data-mode]').forEach(b => {
      b.classList.toggle('on', Number(b.dataset.mode) === mode);
    });
    el('hint-runner').textContent = mode === 1
      ? '你在玩：' + KEY_HINTS[0].join(' / ')
      : 'AI 正在操控（点击"我玩机器人"接管）';
    el('hint-car').textContent = mode === 2
      ? '你在玩：' + KEY_HINTS[1].join(' / ')
      : 'AI 正在操控（点击"我玩汽车"接管）';
    document.querySelectorAll('.stage').forEach((s, i) => {
      s.classList.toggle('human', (i === 0 && mode === 1) || (i === 1 && mode === 2));
    });
  }

  document.querySelectorAll('[data-mode]').forEach(b => {
    b.addEventListener('click', () => { mode = Number(b.dataset.mode); applyMode(); });
  });
  document.querySelectorAll('[data-speed]').forEach(b => {
    b.addEventListener('click', () => {
      speed = Number(b.dataset.speed);
      document.querySelectorAll('[data-speed]').forEach(x =>
        x.classList.toggle('on', Number(x.dataset.speed) === speed));
    });
  });
  el('btn-reset').addEventListener('click', () => {
    games.forEach(g => g.reset());
  });
  el('btn-pause').addEventListener('click', () => {
    paused = !paused;
    el('btn-pause').textContent = paused ? '继续' : '暂停';
  });

  window.addEventListener('keydown', (e) => {
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
      keys.add(e.key);
      e.preventDefault();
    }
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key));

  function humanAction() {
    if (keys.has('ArrowLeft')) return 0;
    if (keys.has('ArrowRight')) return 1;
    if (keys.has('ArrowUp')) return 2;
    if (keys.has('ArrowDown')) return 3;
    return -1;
  }

  applyMode();

  // 主循环：固定 60 Hz 步进
  let acc = 0, last = performance.now(), frame = 0;
  function loop(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    frame++;
    if (!paused) {
      acc += dt;
      let n = 0;
      while (acc >= STEP_DT && n < 8 * speed) {
        acc -= STEP_DT;
        for (let k = 0; k < speed; k++) {
          runner.advance(policy, humanAction());
          car.advance(policy, humanAction());
        }
        n++;
      }
    }
    runner.render(frame);
    car.render(frame);
    updateHud(runner, 'runner');
    updateHud(car, 'car');
    renderGateBars('gate-runner', runner.lastGate, runner.mode === 'ai');
    renderGateBars('gate-car', car.lastGate, car.mode === 'ai');
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

function updateHud(g, tag) {
  const e = g.env;
  const dist = g.taskId === 0 ? e.x : e.s;
  el(`dist-${tag}`).textContent = dist.toFixed(1) + ' m';
  const v = g.taskId === 0 ? e.vx : e.v;
  el(`speed-${tag}`).textContent = (g.taskId === 0 ? v : v * 3.6).toFixed(1) +
    (g.taskId === 0 ? ' m/s' : ' km/h');
  el(`best-${tag}`).textContent = g.best.toFixed(1) + ' m';
  const st = el(`state-${tag}`);
  if (g.finished) { st.textContent = g.finishInfo; st.className = 'state bad'; }
  else if (g.taskId === 1 && e.dTimer > 0) { st.textContent = '轮子失控！'; st.className = 'state warn'; }
  else if (g.taskId === 0 && Math.abs(e.th) > 0.6) { st.textContent = '快摔了'; st.className = 'state warn'; }
  else { st.textContent = '进行中'; st.className = 'state ok'; }
  el(`act-${tag}`).textContent = g.lastAction >= 0 ? ACTION_NAMES[g.taskId][g.lastAction] : '—';
}

/* ------------------------------------------------------------ 静态面板（用训练指标渲染） */
function renderStatic() {
  const m = METRICS;
  el('m-params').textContent = m.params.toLocaleString();
  el('m-size').textContent = m.weights_kb + ' KB';

  const rows = [
    ['MoE (AI)', m.runner.ai.mean, m.runner.ai.fail, m.car.ai.mean, m.car.ai.fail],
    ['启发式规则', m.runner.heuristic.mean, m.runner.heuristic.fail, m.car.heuristic.mean, m.car.heuristic.fail],
    ['随机动作', m.runner.random.mean, m.runner.random.fail, m.car.random.mean, m.car.random.fail],
  ];
  el('score-body').innerHTML = rows.map(r => `
    <tr>
      <td>${r[0]}</td>
      <td class="num">${r[1].toFixed(1)} m</td>
      <td class="num">${(r[2] * 100).toFixed(0)}%</td>
      <td class="num">${r[3].toFixed(1)} m</td>
      <td class="num">${(r[4] * 100).toFixed(0)}%</td>
    </tr>`).join('');

  // 专家路由对比（静态平均）
  const gr = m.runner.ai.gate, gc = m.car.ai.gate;
  el('spec-body').innerHTML = ['', '', '', ''].map((_, i) => `
    <tr>
      <td>Expert ${i}</td>
      <td class="num">${(gr[i] * 100).toFixed(1)}%</td>
      <td><span class="bar"><i style="width:${(gr[i] * 100).toFixed(1)}%"></i></span></td>
      <td class="num">${(gc[i] * 100).toFixed(1)}%</td>
      <td><span class="bar"><i style="width:${(gc[i] * 100).toFixed(1)}%"></i></span></td>
    </tr>`).join('');

  // 架构消融
  if (m.ablation) {
    const a = m.ablation;
    const better = (x, y) => (x >= y ? 'up' : 'down');
    el('ablation-body').innerHTML = `
      <tr>
        <td style="color:var(--teal)">MoE (top-2)</td>
        <td class="num">${m.params.toLocaleString()}</td>
        <td class="num">${m.runner.ai.mean.toFixed(1)} m</td>
        <td class="num">${m.car.ai.mean.toFixed(1)} m</td>
      </tr>
      <tr>
        <td>普通 MLP 对照</td>
        <td class="num">${a.params.toLocaleString()}</td>
        <td class="num">${a.runner.toFixed(1)} m <span class="delta ${better(m.runner.ai.mean, a.runner)}">${(((m.runner.ai.mean - a.runner) / a.runner) * 100).toFixed(0)}%</span></td>
        <td class="num">${a.car.toFixed(1)} m <span class="delta ${better(m.car.ai.mean, a.car)}">${(((m.car.ai.mean - a.car) / a.car) * 100).toFixed(0)}%</span></td>
      </tr>`;
  }

  renderCurve('curve-runner', m.history.runner, '#64ffda', '机器人赛跑（米）');
  renderCurve('curve-car', m.history.car, '#ffb454', '汽车驾驶（米）');
}

function renderCurve(id, data, color, label) {
  const box = el(id);
  if (!data || data.length < 2) {
    box.innerHTML = `<div class="curve-empty">等待训练数据</div>`;
    return;
  }
  const w = box.clientWidth || 520, h = 150, pad = 26;
  const mx = data.length - 1;
  const mn = Math.min(...data), mxx = Math.max(...data);
  const rng = Math.max(mxx - mn, 1e-6);
  const px = (i) => pad + (w - pad * 2) * (i / mx);
  const py = (v) => h - pad - (h - pad * 2) * ((v - mn) / rng);
  const pts = data.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
  box.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none">
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"
                stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
      <text x="${pad}" y="14" fill="#8892b0" font-size="11">${label}</text>
      <text x="${w - pad}" y="14" fill="#8892b0" font-size="11" text-anchor="end">峰值 ${mxx.toFixed(1)}</text>
      <text x="${pad}" y="${h - 6}" fill="#8892b0" font-size="11">${mn.toFixed(1)}</text>
    </svg>`;
}

document.addEventListener('DOMContentLoaded', () => {
  renderStatic();
  main();
  window.addEventListener('resize', () => renderStatic());
});
