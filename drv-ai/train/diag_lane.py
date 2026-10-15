"""诊断：碰撞时本车相对'当前路段正确车道中心'的横向偏差"""
import numpy as np
from city import (CityEnv, _NODE_POS, route_polyline, LANE_OFF)

def lane_dev(env):
    """本车相对路线当前段右侧车道中心的横向偏差（正 = 偏右）"""
    pts, seg, seg_len = route_polyline(env.route)
    # 最近段
    best_d, best_k = 1e18, 0
    for k in range(len(seg_len)):
        p0, d = pts[k], seg[k]
        t = np.clip(((env.x - p0[0]) * d[0] + (env.y - p0[1]) * d[1]) / seg_len[k] ** 2, 0, 1)
        qx, qy = p0[0] + t * d[0], p0[1] + t * d[1]
        dd = np.hypot(env.x - qx, env.y - qy)
        if dd < best_d:
            best_d, best_k = dd, k
    d = seg[best_k] / (np.linalg.norm(seg[best_k]) + 1e-9)
    # 右法向 = (dy, -dx)
    rx, ry = d[1], -d[0]
    cx, cy = pts[best_k][0] + d[0] * 0, pts[best_k][1]  # 中心线起点
    # 本车投影到中心线的垂向量
    vx, vy = env.x - pts[best_k][0], env.y - pts[best_k][1]
    along = vx * d[0] + vy * d[1]
    lat = vx * rx + vy * ry
    dev = lat - LANE_OFF          # 正 = 比车道中心更靠右
    return dev, along

for ep in range(8):
    env = CityEnv(seed=ep)
    for i in range(2400):
        a, s = env.expert()
        (_, nav), r, done, info = env.step(a, s)
        if info["event"] in ("collision", "offroad"):
            dev, along = lane_dev(env)
            print(f"ep{ep} step{i} {info['event']}: lane_dev={dev:+.2f} m "
                  f"(沿段 {along:.1f}) v={env.v:.1f} psi={np.rad2deg(env.psi)%360:.0f} "
                  f"pos=({env.x:.1f},{env.y:.1f})")
            break
        if done:
            break
