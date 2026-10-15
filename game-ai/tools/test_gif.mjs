/* GIF 编码器单元测试：生成已知图案，交由 Python PIL 解码比对 */
import fs from 'node:fs';
import { GifEncoder, buildLUT, imageToIndices } from '../assets/js/gif.js';

const OUT = 'train/_gif_test.gif';

// 调色板：0 黑 1 白 2 红 3 绿 4 蓝
const pal = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
const W = 32, H = 24;
const enc = new GifEncoder(W, H, pal, 5);

// 帧 1：棋盘格
const f1 = new Uint8Array(W * H);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) f1[y * W + x] = ((x >> 3) + (y >> 3)) % 2;
enc.addFrame(f1);

// 帧 2：竖条纹（红绿蓝）
const f2 = new Uint8Array(W * H);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) f2[y * W + x] = 2 + (x % 3);
enc.addFrame(f2);

// 帧 3：全蓝
enc.addFrame(new Uint8Array(W * H).fill(4));

fs.writeFileSync(OUT, enc.build());
console.log('wrote', OUT, fs.statSync(OUT).size, 'bytes, frames =', enc.frameCount);

// ---- imageToIndices / LUT 路径（与页面录制一致）----
const lut = buildLUT(pal);
const img = new Uint8Array(W * H * 4);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const p = (y * W + x) * 4;
  img[p] = 255; img[p + 1] = 0; img[p + 2] = 0; img[p + 3] = 255;
}
const out = new Uint8Array(W * H);
imageToIndices(img, W, H, W, H, lut, out, 0, W);
const allRed = out.every(v => v === 2);
console.log('imageToIndices pure-red ->', allRed ? 'OK' : 'FAIL');
process.exit(allRed ? 0 : 1);
