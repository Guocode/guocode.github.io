/* 最小 GIF89a 编码器（零依赖）。
 *
 * 渲染本身就限制在 ~23 色的固定调色板内，因此省掉颜色量化这一步：
 * 每帧只需要把 RGB 映射成调色板索引，再走标准 LZW 压缩。
 * 支持多帧、NETSCAPE 循环。
 */

const CLEAR = 256, EOI = 257;

class BitWriter {
  constructor() { this.bytes = []; this.cur = 0; this.nbits = 0; }
  write(code, size) {
    this.cur |= code << this.nbits;
    this.nbits += size;
    while (this.nbits >= 8) {
      this.bytes.push(this.cur & 0xff);
      this.cur >>>= 8;
      this.nbits -= 8;
    }
  }
  flush() {
    if (this.nbits > 0) { this.bytes.push(this.cur & 0xff); this.cur = 0; this.nbits = 0; }
  }
}

function lzwEncode(indices, out) {
  const minCodeSize = 8;
  out.push(minCodeSize);
  const bw = new BitWriter();
  let next = 258, codeSize = 9;
  const dict = new Map();
  const resetDict = () => { dict.clear(); next = 258; codeSize = 9; };

  bw.write(CLEAR, codeSize);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i];
    const key = prefix * 256 + k;
    const hit = dict.get(key);
    if (hit !== undefined) {
      prefix = hit;
    } else {
      bw.write(prefix, codeSize);
      dict.set(key, next);
      next += 1;
      // 码长增长：字典可用码号达到 2^codeSize 时立即加宽（与解码器同步）
      if (next >= (1 << codeSize) && codeSize < 12) codeSize += 1;
      if (next === 4096) {          // 字典满，清空重来
        bw.write(CLEAR, codeSize);
        resetDict();
      }
      prefix = k;
    }
  }
  bw.write(prefix, codeSize);
  bw.write(EOI, codeSize);
  bw.flush();

  const b = bw.bytes;
  for (let i = 0; i < b.length; i += 255) {
    const chunk = b.slice(i, i + 255);
    out.push(chunk.length);
    for (let j = 0; j < chunk.length; j++) out.push(chunk[j]);
  }
  out.push(0);
}

function push16(out, v) { out.push(v & 0xff, (v >> 8) & 0xff); }

export class GifEncoder {
  /**
   * @param w 输出宽  @param h 输出高
   * @param palette 颜色数组，形如 [[r,g,b], ...]，不足 256 色用黑色补齐
   * @param delayCs 每帧延迟（单位 1/100 秒）
   */
  constructor(w, h, palette, delayCs = 5) {
    this.w = w; this.h = h; this.delay = delayCs;
    this.pal = palette.slice();
    while (this.pal.length < 256) this.pal.push([0, 0, 0]);
    this.pal = this.pal.slice(0, 256);
    this.frames = [];
  }

  addFrame(indices) {
    if (indices.length !== this.w * this.h) {
      throw new Error('frame size mismatch: ' + indices.length);
    }
    this.frames.push(Uint8Array.from(indices));
  }

  get frameCount() { return this.frames.length; }

  build() {
    const out = [];
    out.push(0x47, 0x49, 0x46, 0x38, 0x39, 0x61);      // GIF89a
    push16(out, this.w); push16(out, this.h);
    out.push(0xf7, 0, 0);                              // GCT 256 色
    for (const c of this.pal) out.push(c[0], c[1], c[2]);

    // NETSCAPE2.0 循环扩展（无限循环）
    out.push(0x21, 0xff, 11);
    for (const ch of 'NETSCAPE2.0') out.push(ch.charCodeAt(0));
    out.push(3, 1, 0, 0);

    for (const f of this.frames) {
      out.push(0x21, 0xf9, 4);                         // GCE
      out.push(0x04, this.delay & 0xff, (this.delay >> 8) & 0xff, 0x00, 0x00);
      out.push(0x2c);                                  // Image Descriptor
      push16(out, 0); push16(out, 0); push16(out, this.w); push16(out, this.h);
      out.push(0x00);                                  // 无局部调色板
      lzwEncode(f, out);
    }
    out.push(0x3b);                                    // trailer
    return new Uint8Array(out);
  }

  toBlob() {
    return new Blob([this.build()], { type: 'image/gif' });
  }
}

/** 由调色板构建 15-bit(32768) 查找表：RGB → 最近调色板索引 */
export function buildLUT(palette) {
  const lut = new Uint8Array(32768);
  const pr = palette.map(c => c[0]), pg = palette.map(c => c[1]), pb = palette.map(c => c[2]);
  for (let r = 0, i = 0; r < 32; r++) {
    for (let g = 0; g < 32; g++) {
      for (let b = 0; b < 32; b++, i++) {
        const R = r * 8 + 4, G = g * 8 + 4, B = b * 8 + 4;
        let best = 0, bestD = Infinity;
        for (let k = 0; k < palette.length; k++) {
          const dr = R - pr[k], dg = G - pg[k], db = B - pb[k];
          const d = dr * dr + dg * dg + db * db;
          if (d < bestD) { bestD = d; best = k; }
        }
        lut[i] = best;
      }
    }
  }
  return lut;
}

/** 把 ImageData 转成调色板索引，可按 nearest-neighbor 缩放 */
export function imageToIndices(imgData, srcW, srcH, dstW, dstH, lut, out, offsetX, totalW) {
  const d = imgData;
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor(y * srcH / dstH));
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor(x * srcW / dstW));
      const p = (sy * srcW + sx) * 4;
      const idx = lut[((d[p] >> 3) << 10) | ((d[p + 1] >> 3) << 5) | (d[p + 2] >> 3)];
      out[y * totalW + offsetX + x] = idx;
    }
  }
}
