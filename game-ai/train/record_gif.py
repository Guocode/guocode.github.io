"""端到端录制「AI vs HUMAN」对战回放，输出像素风 GIF。

对阵
    AI    = 训练好的 MoE 策略（ckpt/moe.pt）
    HUMAN = 手写启发式规则玩家（普通人类水平的代理）
两组对决使用同一个 seed（同一起跑线），四个环境同步推进。

输出：assets/img/versus.gif（640x360，10 fps，结尾 3 s 成绩定格）
"""

import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from env import (RunnerEnv, CarEnv, MAX_STEPS, HIP_OFF, ROAD_HALF,
                 track_curvature)
from model import MoEPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "ckpt", "moe.pt")
OUT_DIR = os.path.join(ROOT, "assets", "img")
OUT = os.path.join(OUT_DIR, "versus.gif")

SEED = 7
FRAME_EVERY = 6            # 每 6 个 env step 出一帧 -> 10 fps
HOLD_MS = 3000             # 结尾成绩定格

# ------------------------------------------------------------ 调色板（与 render.js 一致）
SKY, SKY_HI = "#141a2e", "#1b2340"
HILL, HILL_HI = "#1a2b3a", "#213548"
GROUND, GROUND_HI = "#252a40", "#2e3450"
BRICK, BRICK_HI = "#3a4060", "#444b70"
ROAD, ROAD_HI = "#383e5a", "#414869"
GRASS, GRASS_HI = "#1d3a32", "#23443b"
STRIPE, STRIPE_DIM = "#dfe6ff", "#8f9ac4"
TEAL, TEAL_HI, TEAL_DK = "#64ffda", "#a8fff0", "#2fae94"
ORANGE, ORANGE_HI, ORANGE_DK = "#ffb454", "#ffd699", "#c47f2a"
DANGER, WARN, DIM, TXT = "#ff6b6b", "#ffd166", "#6b7799", "#e6f1ff"
BG, LABEL_BG = "#0a0f1e", "#0e1424"

W, H = 640, 360
CELL_H = 175
GEO = {  # (ox, oy, w)
    "r_ai": (2, 2, 318), "r_hu": (2, 183, 318),
    "c_ai": (322, 2, 316), "c_hu": (322, 183, 316),
}


def font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


F12, F16, F22 = font(12), font(16), font(22)


# ------------------------------------------------------------ 机器人赛道
def draw_runner(dr, env, ox, oy, w, color, color_hi, color_dk):
    S = 55
    gy = CELL_H - 22
    camx = env.x[0] - 1.1

    def sx(wx):
        return ox + (wx - camx) * S

    def sy(wy):
        return oy + gy - wy * S

    dr.rectangle([ox, oy, ox + w, oy + CELL_H], fill=SKY)
    dr.rectangle([ox, oy, ox + w, oy + 20], fill=SKY_HI)
    par = -camx * 0.25 * S
    for i in range(6):
        bx = ox + ((par * 0.6 + i * 74) % (w + 160)) - 80
        bh = 16 + (i * 37) % 22
        dr.rectangle([bx, oy + gy - bh - 8, bx + 62, oy + gy],
                     fill=HILL if i % 2 else HILL_HI)
    dr.rectangle([ox, oy + gy, ox + w, oy + CELL_H], fill=GROUND)
    dr.rectangle([ox, oy + gy, ox + w, oy + gy + 3], fill=GROUND_HI)
    bs = 0.28 * S
    start_i = int(camx / 0.28) - 1
    for i in range(start_i, start_i + int(w / bs) + 3):
        bx = sx(i * 0.28)
        if bx < ox - bs or bx > ox + w:
            continue
        dr.rectangle([bx, oy + gy + 3, bx + bs - 1, oy + gy + 8],
                     fill=BRICK if i % 2 == 0 else BRICK_HI)
    m0 = int(camx / 10) * 10
    for m in range(m0, m0 + 60, 10):
        mx = sx(m)
        if ox - 10 < mx < ox + w + 10:
            dr.rectangle([mx, oy + gy - 14, mx + 2, oy + gy], fill=DIM)
            dr.rectangle([mx, oy + gy - 18, mx + 10, oy + gy - 14],
                         fill=WARN if m % 50 == 0 else DIM)

    comx, comy = sx(env.x[0]), sy(env.y[0])
    th = env.th[0]
    hipx = comx + math.sin(th) * HIP_OFF * S
    hipy = comy + math.cos(th) * HIP_OFF * S
    for i in (0, 1):
        fx, fy = sx(env.fx[0, i]), sy(env.fy[0, i])
        dr.line([hipx, hipy, fx, fy], fill=color_dk if i == 0 else color, width=3)
        dr.rectangle([fx - 3, fy - 1, fx + 3, fy + 1],
                     fill=color_dk if i == 0 else color_hi)
    L = 26
    topx = hipx - math.sin(th) * L
    topy = hipy - math.cos(th) * L
    dx, dy = topx - hipx, topy - hipy
    n = math.hypot(dx, dy) or 1.0
    px, py = -dy / n * 3.5, dx / n * 3.5
    dr.polygon([(hipx + px, hipy + py), (topx + px, topy + py),
                (topx - px, topy - py), (hipx - px, hipy - py)], fill=color)
    dr.rectangle([topx - 5, topy - 13, topx + 4, topy - 4], fill=color_hi)
    dr.rectangle([topx, topy - 11, topx + 2, topy - 9], fill=SKY)


# ------------------------------------------------------------ 汽车赛道
def draw_car(dr, env, ox, oy, w, color, color_hi, color_dk, frame):
    HORIZON = int(CELL_H * 0.34)
    FOCAL, CAM_H, SEG, DZ = 150, 1.15, 30, 4.0

    dr.rectangle([ox, oy, ox + w, oy + HORIZON], fill=SKY)
    dr.rectangle([ox, oy, ox + w, oy + int(HORIZON * 0.55)], fill=SKY_HI)
    for i in range(6):
        bx = ox + ((i * 68 - env.s[0] * 0.6) % (w + 140)) - 70
        bh = 10 + (i * 31) % 16
        dr.rectangle([bx, oy + HORIZON - bh, bx + 56, oy + HORIZON],
                     fill=HILL if i % 2 else HILL_HI)
    dr.rectangle([ox, oy + HORIZON, ox + w, oy + CELL_H], fill=GRASS)

    prev = None
    for i in range(SEG, 0, -1):
        z = i * DZ
        scale = FOCAL / z
        y = oy + HORIZON + CAM_H * scale
        off = 0.0
        for k in range(1, i + 1):
            off += track_curvature(env.s[0] + k * DZ) * DZ * (k * DZ) * 0.5
        cx = ox + w / 2 + (off - env.d[0]) * scale
        half = 3.4 * scale
        if prev is not None and half > 0.5:
            pts = [(prev[0] - prev[2], prev[1]), (prev[0] + prev[2], prev[1]),
                   (cx + half, y), (cx - half, y)]
            dr.polygon(pts, fill=ROAD if i % 2 == 0 else ROAD_HI)
            if i % 2 == 0 and y - prev[1] >= 1:
                dr.rectangle([prev[0] - prev[2] - 2, prev[1], prev[0] - prev[2] + 1, y], fill=STRIPE)
                dr.rectangle([prev[0] + prev[2], prev[1], prev[0] + prev[2] + 3, y], fill=STRIPE)
            if i % 4 < 2 and y - prev[1] >= 1:
                dr.rectangle([cx - 1, y, cx + 1, y + (y - prev[1])], fill=STRIPE_DIM)
        prev = (cx, y, half)

    active = env.d_timer[0] > 0
    shake = round(math.sin(frame * 0.9) * 2) if active else 0
    car_y = oy + CELL_H - 26
    car_x = ox + w / 2 - env.d[0] * (FOCAL / (2.5 * DZ)) * 0.55 + shake
    dr.rectangle([car_x - 14, car_y - 6, car_x + 14, car_y + 6], fill=color)
    dr.rectangle([car_x - 10, car_y - 14, car_x + 10, car_y - 5], fill=color_hi)
    dr.rectangle([car_x - 7, car_y - 12, car_x + 7, car_y - 6], fill=SKY)
    dr.rectangle([car_x - 15, car_y + 2, car_x - 9, car_y + 8], fill=color_dk)
    dr.rectangle([car_x + 9, car_y + 2, car_x + 15, car_y + 8], fill=color_dk)
    dr.rectangle([car_x - 13, car_y - 4, car_x - 9, car_y - 1], fill=DANGER)
    dr.rectangle([car_x + 9, car_y - 4, car_x + 13, car_y - 1], fill=DANGER)
    if active:
        for i in range(4):
            px = car_x + (12 if i % 2 else -16) + math.sin(frame * 0.3 + i) * 3
            py = car_y + 4 - ((frame * 0.4 + i * 9) % 22)
            dr.rectangle([px, py, px + 3, py + 3], fill=DIM)
        for edge in ((ox, oy, ox + w, oy + 2), (ox, oy + CELL_H - 2, ox + w, oy + CELL_H),
                     (ox, oy, ox + 2, oy + CELL_H), (ox + w - 2, oy, ox + w, oy + CELL_H)):
            dr.rectangle(edge, fill=DANGER)


# ------------------------------------------------------------ 标签条
def draw_label(dr, ox, oy, w, label, dist, color, extra=""):
    dr.rectangle([ox, oy, ox + w, oy + 13], fill=LABEL_BG)
    dr.text((ox + 4, oy + 2), label, fill=color, font=F12)
    txt = dist + ("  " + extra if extra else "")
    dr.text((ox + w - 4, oy + 2), txt, fill=DIM, font=F12, anchor="ra")


def render_frame(envs, frame):
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    r_ai, r_hu = envs["r_ai"], envs["r_hu"]
    c_ai, c_hu = envs["c_ai"], envs["c_hu"]

    for key, env, col, hi, dk, lab in (
            ("r_ai", r_ai, TEAL, TEAL_HI, TEAL_DK, "GAME 00 RUNNER · AI (MoE)"),
            ("r_hu", r_hu, ORANGE, ORANGE_HI, ORANGE_DK, "GAME 00 RUNNER · HUMAN")):
        ox, oy, w = GEO[key]
        draw_runner(dr, env, ox, oy, w, col, hi, dk)
        tag = " CRASHED" if env.done[0] and abs(env.th[0]) > 0.9 else ""
        draw_label(dr, ox, oy, w, lab, "%.1f m" % env.x[0], col, tag)

    for key, env, col, hi, dk, lab in (
            ("c_ai", c_ai, TEAL, TEAL_HI, TEAL_DK, "GAME 01 CAR · AI (MoE)"),
            ("c_hu", c_hu, ORANGE, ORANGE_HI, ORANGE_DK, "GAME 01 CAR · HUMAN")):
        ox, oy, w = GEO[key]
        draw_car(dr, env, ox, oy, w, col, hi, dk, frame)
        tag = " OFF-ROAD" if env.done[0] and abs(env.d[0]) > ROAD_HALF else ""
        draw_label(dr, ox, oy, w, lab, "%.0f m" % env.s[0], col, tag)

    dr.rectangle([0, 177, W, 183], fill=BG)
    return img


def render_scoreboard(d):
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 46], fill=LABEL_BG)
    dr.text((W // 2, 14), "FINAL SCORE - same seed, same start line",
            fill=TXT, font=F22, anchor="ma")
    rows = []
    for name, ai, hu, unit in (("RUNNER", d["r_ai"], d["r_hu"], "%.1f m"),
                               ("CAR", d["c_ai"], d["c_hu"], "%.0f m")):
        win_ai = ai >= hu
        rows.append((name, ai, hu, unit, win_ai))
    y = 80
    for name, ai, hu, unit, win_ai in rows:
        dr.text((70, y), name, fill=DIM, font=F16)
        dr.text((210, y - 6), unit % ai, fill=TEAL, font=F22)
        dr.text((400, y - 6), unit % hu, fill=ORANGE, font=F22)
        dr.text((600, y - 4), "AI WINS" if win_ai else "HUMAN WINS",
                fill=TEAL if win_ai else ORANGE, font=F16, anchor="ra")
        y += 78
    dr.text((W // 2, H - 34), "one MoE model · 22,473 params · trained from scratch in 11 min (CPU)",
            fill=DIM, font=F12, anchor="ma")
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    policy = MoEPolicy(hidden=64, n_experts=4, top_k=2)
    policy.load_state_dict(torch.load(CKPT, map_location="cpu"))
    policy.eval()

    envs = {
        "r_ai": RunnerEnv(1, seed=SEED), "r_hu": RunnerEnv(1, seed=SEED),
        "c_ai": CarEnv(1, seed=SEED), "c_hu": CarEnv(1, seed=SEED),
    }
    for e in envs.values():
        e.reset(seed=SEED)
    oh = {0: torch.zeros(1, 2), 1: torch.zeros(1, 2)}
    oh[0][0, 0] = 1.0
    oh[1][0, 1] = 1.0
    done = {k: False for k in envs}

    frames = []
    for step in range(MAX_STEPS):
        for key, env in envs.items():
            if done[key]:
                continue
            if key.endswith("_ai"):
                tid = 0 if key.startswith("r") else 1
                with torch.no_grad():
                    a = policy.act(torch.as_tensor(env.obs()), oh[tid])[0]
                a = np.array([int(a)])
            else:
                from env import runner_heuristic, car_heuristic
                a = runner_heuristic(env) if key.startswith("r") else car_heuristic(env)
            _, _, d, _ = env.step(a)
            done[key] = bool(d[0])
        if step % FRAME_EVERY == 0:
            frames.append(render_frame(envs, step))
        if all(done.values()):
            break
    frames.append(render_frame(envs, MAX_STEPS))

    dist = {
        "r_ai": envs["r_ai"].x[0], "r_hu": envs["r_hu"].x[0],
        "c_ai": envs["c_ai"].s[0], "c_hu": envs["c_hu"].s[0],
    }
    frames.append(render_scoreboard(dist))

    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)
    size_kb = os.path.getsize(OUT) / 1024.0
    print("frames=%d  size=%.0f KB  -> %s" % (len(frames), size_kb, OUT))
    print("runner  AI %.1f m vs HUMAN %.1f m" % (dist["r_ai"], dist["r_hu"]))
    print("car     AI %.0f m vs HUMAN %.0f m" % (dist["c_ai"], dist["c_hu"]))


if __name__ == "__main__":
    main()
