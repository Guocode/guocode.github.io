"""像素帧渲染器：把两个游戏的环境状态渲染成 (3, 40, 64) 的低分辨率 RGB 帧，
供 CNN 策略网络作纯图像输入。numpy 向量化（按 batch 一次渲染），与页面渲染
风格一致但视角为俯视（信息对 CNN 更友好）：
  - CAR：俯视公路——路面/红白路缘（R 通道）、中线虚线+车头（G）、
         事件侧带+车速条（B）
  - RUNNER：侧视地形——地表填充（R）、独轮机器人身体姿态（G）、
            颠簸事件+角速度条（B）
所有信息"画在画面里"（含必要的 HUD 条），网络只看像素。
"""

import numpy as np

from env import terrain_h, track_curvature

FH, FW = 40, 64          # rows, cols
Z_SPAN = 50.0            # car: 前视 50 m
PX_K, PX_Z0 = 14.0, 3.0  # car: 透视式俯视 px/m = K/(z+z0)（近 4.7，50m 处 0.26）
RUN_M_PER_PX = 0.4       # runner: 侧视 1 px = 0.4 m
LOOKAHEAD_PX = 54        # runner: 从第 10 列向右看 54*0.4 ≈ 21.6 m

# 闭合形式积分系数（track_curvature = a1*sin(b1*s) + a2*sin(b2*s + c2)）
A1, B1 = 0.018, 0.035
A2, B2, C2 = 0.010, 0.011, 1.7


def _kint2(s, z):
    """∫₀^z κ(s+u)·(z-u) du 的闭合形式，s (B,), z (B,H) -> (B,H)。
    路面弯曲导致的横向偏移积累（相对车头方向）。"""
    s0 = s[:, None]
    p0, p1 = B1 * s0, B2 * s0 + C2
    p0z, p1z = B1 * (s0 + z), B2 * (s0 + z) + C2
    i0 = A1 * (np.cos(p0) - np.cos(p0z)) / B1 + A2 * (np.cos(p1) - np.cos(p1z)) / B2
    i1 = (A1 * (-z * np.cos(p0z) / B1 + (np.sin(p0z) - np.sin(p0)) / B1 ** 2)
          + A2 * (-z * np.cos(p1z) / B2 + (np.sin(p1z) - np.sin(p1)) / B2 ** 2))
    return z * i0 - i1


def _rows_z():
    """每行对应的前视距离 (FH,)：row 39 -> 0 m（车底），row 0 -> 50 m。"""
    return (np.arange(FH, dtype=np.float64)[::-1]) * Z_SPAN / (FH - 1)


def render_car(env):
    """env: CarEnv（batch n）。返回 (n, 3, FH, FW) float32。
    透视式俯视：横向比例随距离压缩 PX(z)=K/(z+z0)，保证任何航向下路面都在画面内。"""
    n = env.n
    frm = np.zeros((n, 3, FH, FW), np.float32)
    z = _rows_z()[None, :]                                   # (1,H)
    s = env.s[:, None]
    px = PX_K / (z + PX_Z0)                                  # (1,H) px per meter
    # 路面中心在车体坐标系里的横向位置（右为正）
    e = -(env.d[:, None] + env.psi[:, None] * z + _kint2(env.s, z)) * px
    c_col = FW / 2 + e                                       # (n,H) 路中心列
    cols = np.arange(FW)[None, None, :]                      # (1,1,W)
    # R 通道：路面（含路缘石提亮）
    half = (3.4 * px)[:, :, None]                            # (1,H,1) 路半宽（px）
    dist = np.abs(cols - c_col[:, :, None])
    on_road = dist <= half
    curb = np.abs(dist - half) <= 0.6
    frm[:, 0] = np.where(on_road, 0.65, 0.0)
    frm[:, 0] = np.where(curb & on_road, 1.0, frm[:, 0])
    # G 通道：中线虚线（6.4 m 周期，3.2 m 实）+ 车头亮点
    dash = ((s + z) % 6.4) < 3.2
    near_c = dist <= 0.6
    frm[:, 1] = np.where(dash[:, :, None] & near_c, 1.0, 0.0)
    frm[:, 1, FH - 3:FH, FW // 2 - 1:FW // 2 + 2] = 1.0      # 车身
    # B 通道：事件侧带（爆胎强 / 打滑弱，画在被拽一侧）+ 底部车速条
    active = (env.ev_timer > 0)
    side_r = (env.ev_dir > 0)
    band = np.zeros((n, FW), np.float32)
    for i in range(n):                                        # 事件方向是逐样本标量
        if active[i]:
            lo, hi = (FW // 2 + 1, FW // 2 + 7) if side_r[i] else (FW // 2 - 7, FW // 2 - 1)
            band[i, lo:hi] = env.ev_mag[i] * (1.0 if env.ev_type[i] > 0 else 0.55)
    frm[:, 2, FH - 12:FH - 2, :] = band[:, None, :]
    vbar = np.clip(env.v / 26.0, 0, 1) * 20
    for i in range(n):
        frm[i, 2, 0, :int(round(vbar[i]))] = 0.8
    return frm


def render_runner(env):
    """env: RunnerEnv（batch n）。返回 (n, 3, FH, FW) float32。"""
    n = env.n
    frm = np.zeros((n, 3, FH, FW), np.float32)
    cols = np.arange(FW, dtype=np.float64)
    xw = env.x[:, None] + (cols[None, :] - 10) * RUN_M_PER_PX   # (n,W)
    h = terrain_h(xw)                                           # (n,W)
    row_h = np.round(28 - h * 60).astype(np.int64)              # 地表行号
    for i in range(n):
        for cix in range(FW):
            r0 = max(int(row_h[i, cix]), 0)
            frm[i, 0, r0, cix] = 1.0                            # 地表线
            frm[i, 0, r0 + 1:, cix] = 0.45                      # 地下填充
        # 机器人身体：从轮子位置沿 -th 方向画 12 px 竖杆
        wr = int(row_h[i, 10])
        ct, st = np.cos(env.th[i]), np.sin(env.th[i])
        for k in range(12):
            rr = wr - k * ct
            cc = 10 + k * st * 2.5
            if 0 <= rr < FH - 1 and 0 <= cc < FW:
                frm[i, 1, int(round(rr)), int(round(cc))] = 1.0
        frm[i, 1, wr - 13:wr - 10, 9:12] = 1.0                  # 头
        # B 通道：颠簸事件指示 + 角速度条（底部，从中心向左/右）
        if env.b_timer[i] > 0:
            frm[i, 2, 18:26, 7:14] = 0.8
        wb = int(round(np.clip(env.w[i] / 6.0, -1, 1) * 15))
        if wb >= 0:
            frm[i, 2, 0, 32:32 + wb] = 0.8
        else:
            frm[i, 2, 0, 32 + wb:32] = 0.8
    return frm


def render(task_id, env):
    return render_runner(env) if task_id == 0 else render_car(env)
