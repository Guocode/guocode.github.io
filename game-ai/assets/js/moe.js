/* MoE 策略的浏览器端推理（零依赖）。
 *
 * 模型只有 ~22K 参数，直接在 JS 里跑前向即可：省掉 10MB 的 wasm 运行时，
 * 页面加载完就能玩。结构必须与 train/model.py 的 MoEPolicy 完全一致：
 *   trunk(16->64, Tanh) -> trunk(64->64, Tanh)
 *   router(64->4) -> top-2 softmax 门控
 *   4 个 expert(64->64, Tanh)，按门控加权求和
 *   actor(64->2) + critic(64->1)   （v4：动作 4→2，去掉软/硬档）
 */

function b64ToF32(s) {
  const bin = atob(s);
  const len = bin.length;
  const u8 = new Uint8Array(len);
  for (let i = 0; i < len; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

function linear(out, x, W, b, inDim, outDim) {
  for (let i = 0; i < outDim; i++) {
    const off = i * inDim;
    let s = b[i];
    for (let j = 0; j < inDim; j++) s += W[off + j] * x[j];
    out[i] = Math.tanh(s);
  }
}

export class MoEPolicy {
  constructor(weights) {
    this.t = {};
    for (const k in weights) this.t[k] = b64ToF32(weights[k].data);
    this.nExperts = 4;
    this.hidden = 64;
    this.nAct = this.t["actor.bias"].length;   // 动作维度随权重自适应（现为 2）
    this._h = new Float32Array(64);
    this._h2 = new Float32Array(64);
    this._e = new Float32Array(64);
    this._in = new Float32Array(16);
    this._r = new Float32Array(4);
    this._y = new Float32Array(64);
    this._logits = new Float32Array(this.nAct);
    this._gate = new Float32Array(4);
  }

  forward(obs, taskId) {
    const inp = this._in;
    for (let i = 0; i < 14; i++) inp[i] = obs[i];
    inp[14] = taskId === 0 ? 1 : 0;
    inp[15] = taskId === 0 ? 0 : 1;

    linear(this._h, inp, this.t["trunk.0.weight"], this.t["trunk.0.bias"], 16, 64);
    linear(this._h2, this._h, this.t["trunk.2.weight"], this.t["trunk.2.bias"], 64, 64);
    const h = this._h2;

    // router -> top-2 softmax
    const rw = this.t["router.weight"], rb = this.t["router.bias"];
    for (let i = 0; i < 4; i++) {
      let s = rb[i];
      for (let j = 0; j < 64; j++) s += rw[i * 64 + j] * h[j];
      this._r[i] = s;
    }
    let i1 = 0, i2 = 1;
    if (this._r[1] > this._r[0]) { i1 = 1; i2 = 0; }
    for (let i = 2; i < 4; i++) {
      if (this._r[i] > this._r[i1]) { i2 = i1; i1 = i; }
      else if (this._r[i] > this._r[i2]) { i2 = i; }
    }
    const m = Math.max(this._r[i1], this._r[i2]);
    const e1 = Math.exp(this._r[i1] - m), e2 = Math.exp(this._r[i2] - m);
    const sum = e1 + e2;
    const g1 = e1 / sum, g2 = e2 / sum;
    this._gate.fill(0);
    this._gate[i1] = g1;
    this._gate[i2] = g2;

    // 专家加权求和
    const y = this._y;
    y.fill(0);
    for (let e = 0; e < this.nExperts; e++) {
      const g = this._gate[e];
      if (g === 0) continue;
      linear(this._e, h, this.t[`experts.${e}.0.weight`], this.t[`experts.${e}.0.bias`], 64, 64);
      for (let j = 0; j < 64; j++) y[j] += g * this._e[j];
    }

    // actor / critic（线性输出，无激活）
    const aw = this.t["actor.weight"], ab = this.t["actor.bias"];
    for (let i = 0; i < this.nAct; i++) {
      let s = ab[i];
      for (let j = 0; j < 64; j++) s += aw[i * 64 + j] * y[j];
      this._logits[i] = s;
    }
    const cw = this.t["critic.weight"], cb = this.t["critic.bias"];
    let v = cb[0];
    for (let j = 0; j < 64; j++) v += cw[j] * y[j];

    return { logits: this._logits, value: v, gate: this._gate, experts: [i1, i2] };
  }

  act(obs, taskId) {
    const r = this.forward(obs, taskId);
    let best = 0;
    for (let i = 1; i < this.nAct; i++) if (r.logits[i] > r.logits[best]) best = i;
    return { action: best, ...r };
  }
}

export async function loadPolicy(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("weights load failed: " + res.status);
  const w = await res.json();
  return new MoEPolicy(w);
}
