"""碰撞诊断：找出与哪类 NPC、相对方位、距离碰撞"""
import numpy as np
from city import CityEnv

for ep in range(8):
    env = CityEnv(seed=ep)
    for i in range(2400):
        a, s = env.expert()
        (_, nav), r, done, info = env.step(a, s)
        if info["event"] == "collision":
            ap, acl = env.npc_positions()
            dd = np.hypot(ap[:, 0] - env.x, ap[:, 1] - env.y)
            hit_car = (acl == 0) & (dd < 2.4)
            hit_bike = (acl == 1) & (dd < 1.9)
            hit_ped = (acl == 2) & (dd < 1.3)
            hit = hit_car | hit_bike | hit_ped
            k = int(np.argmin(dd))
            cls = {0: "car", 1: "bike", 2: "ped"}[int(acl[k])]
            # 相对方位（本车系）
            fwd = np.array([np.cos(env.psi), np.sin(env.psi)])
            rel = ap[k] - np.array([env.x, env.y])
            along = rel @ fwd
            lat = rel[0] * fwd[1] - rel[1] * fwd[0]
            print(f"ep{ep} step{i}: hit {cls} d={dd[k]:.2f} "
                  f"along={along:+.2f} lat={lat:+.2f} v={env.v:.1f} "
                  f"psi={np.rad2deg(env.psi)%360:.0f} pos=({env.x:.0f},{env.y:.0f})")
            break
        if done and info["event"] != "goal":
            break
