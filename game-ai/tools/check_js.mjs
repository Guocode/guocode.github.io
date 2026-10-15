/* 跨端一致性检查（开发用，不参与页面运行）
 *   node tools/check_js.mjs
 * 1) 浏览器端 env.js 与 Python 端 env.py 的启发式成绩是否统计一致
 * 2) 浏览器端 moe.js 手写前向与 PyTorch 前向是否数值一致
 */

import fs from 'node:fs';
import { RunnerEnv, CarEnv, trackCurvature } from '../assets/js/env.js';
import { MoEPolicy } from '../assets/js/moe.js';

const CAP = 3600;   // 与 Python 端 MAX_STEPS 一致（60 s 上限）

// 与 train/env.py 的启发式一致：倒立摆 PD 平衡
function runnerBalance(e) {
  const u = 2.2 * e.th + 0.9 * e.w;
  if (u > 0.30) return 3;
  if (u > 0.05) return 1;
  if (u < -0.30) return 2;
  if (u < -0.05) return 0;
  return 0;
}

// 与 train/env.py 的启发式一致：车道保持 + 事件预修正
function carKeep(e) {
  const kap = trackCurvature(e.s + 14.0);
  const active = e.evTimer > 0;
  const target = -0.20 * (e.d / 3.4) + 0.9 * 2.6 * kap;
  const err = e.psi - target + 0.10 * (active ? e.evDir * e.evMag : 0.0);
  if (err > 0.10) return 2;
  if (err > 0.015) return 0;
  if (err < -0.10) return 3;
  if (err < -0.015) return 1;
  return 0;
}

function run(cls, pol, seeds) {
  const dist = [];
  for (const sd of seeds) {
    const e = new cls(sd);
    for (let s = 0; s < CAP; s++) {
      if (e.step(pol(e)).done) break;
    }
    dist.push(cls === RunnerEnv ? e.x : e.s);
  }
  const m = dist.reduce((a, b) => a + b, 0) / dist.length;
  const sd = Math.sqrt(dist.reduce((a, b) => a + (b - m) ** 2, 0) / dist.length);
  return { mean: m, std: sd, min: Math.min(...dist), max: Math.max(...dist) };
}

const seeds = Array.from({ length: 16 }, (_, i) => i + 1);
const r = run(RunnerEnv, runnerBalance, seeds);
const c = run(CarEnv, carKeep, seeds);
console.log('runner balance    mean %s  std %s  [%s, %s]',
  r.mean.toFixed(2), r.std.toFixed(2), r.min.toFixed(2), r.max.toFixed(2));
console.log('car lane-keeping  mean %s  std %s  [%s, %s]',
  c.mean.toFixed(2), c.std.toFixed(2), c.min.toFixed(2), c.max.toFixed(2));
console.log('  (python ref: runner 274.40±0.00 | car 508.66±8.22)');

// ---------------- MoE 前向一致性 ----------------
const w = JSON.parse(fs.readFileSync('assets/models/moe_weights.json', 'utf8'));
const casesPath = 'train/verify_cases.json';
if (fs.existsSync(casesPath)) {
  const cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
  const p = new MoEPolicy(w);
  let maxErr = 0;
  for (const c2 of cases) {
    const out = p.forward(Float32Array.from(c2.obs), c2.task);
    for (let i = 0; i < 4; i++) {
      maxErr = Math.max(maxErr, Math.abs(out.logits[i] - c2.logits[i]));
      maxErr = Math.max(maxErr, Math.abs(out.gate[i] - c2.gate[i]));
    }
    maxErr = Math.max(maxErr, Math.abs(out.value - c2.value));
  }
  console.log('moe.js vs torch  max abs err = %s  (%s)',
    maxErr.toExponential(2), maxErr < 1e-4 ? 'OK' : 'MISMATCH');
} else {
  console.log('no verify_cases.json, run train/verify.py first');
}
