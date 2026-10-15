/* 跨端一致性检查（开发用，不参与页面运行）
 *   node tools/check_js.mjs
 * 1) 浏览器端 env.js 与 Python 端 env.py 的启发式成绩是否统计一致
 * 2) 浏览器端 moe.js 手写前向与 PyTorch 前向是否数值一致
 */

import fs from 'node:fs';
import { RunnerEnv, CarEnv, MAX_STEPS, trackCurvature } from '../assets/js/env.js';
import { MoEPolicy } from '../assets/js/moe.js';

const ROAD_HALF = 3.4, WHEELBASE = 2.6;

function stepOnly(e) {
  const both = e.stance[0] && e.stance[1];
  if (both) return e.fx[0] < e.fx[1] ? 0 : 1;
  return 2;
}
function carKeep(e) {
  const kap = trackCurvature(e.s + 14.0);
  const target = -0.20 * (e.d / ROAD_HALF) + 0.9 * WHEELBASE * kap;
  const err = e.psi - target;
  let a = err > 0.015 ? 0 : err < -0.015 ? 1 : 2;
  if (e.v > 16.0) a = 3;
  return a;
}

function run(cls, pol, seeds) {
  const dist = [];
  for (const sd of seeds) {
    const e = new cls(sd);
    for (let s = 0; s < MAX_STEPS; s++) {
      if (e.step(pol(e)).done) break;
    }
    dist.push(cls === RunnerEnv ? e.x : e.s);
  }
  const m = dist.reduce((a, b) => a + b, 0) / dist.length;
  const sd = Math.sqrt(dist.reduce((a, b) => a + (b - m) ** 2, 0) / dist.length);
  return { mean: m, std: sd, min: Math.min(...dist), max: Math.max(...dist) };
}

const seeds = Array.from({ length: 16 }, (_, i) => i + 1);
const r = run(RunnerEnv, stepOnly, seeds);
const c = run(CarEnv, carKeep, seeds);
console.log('runner alt-step   mean %s  std %s  [%s, %s]',
  r.mean.toFixed(2), r.std.toFixed(2), r.min.toFixed(2), r.max.toFixed(2));
console.log('car lane-keeping  mean %s  std %s  [%s, %s]',
  c.mean.toFixed(2), c.std.toFixed(2), c.min.toFixed(2), c.max.toFixed(2));
console.log('  (python ref: runner 78.75±0.03 | car 134.88±66.34)');

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
