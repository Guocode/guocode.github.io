"""追踪碰撞前 12 步：本车速度 / 最近前车距离 / expert 决策"""
import numpy as np
from city import CityEnv  # noqa

for ep in range(8):
    env = CityEnv(seed=ep)
    hist = []
    for i in range(2400):
        ap, acl = env.npc_positions()
        fwd = np.array([np.cos(env.psi), np.sin(env.psi)])
        rel = ap - np.array([env.x, env.y])
        along = rel @ fwd
        lat = np.abs(rel[:, 0] * fwd[1] - rel[:, 1] * fwd[0])
        ahead = (acl <= 1) & (along > 1.0) & (along < 20.0) & (lat < 2.0)
        gap = along[ahead].min() if ahead.any() else 99.0
        a, s = env.expert()
        hist.append((i, env.v, gap, a, s, env.prog))
        (_, nav), r, done, info = env.step(a, s)
        if info["event"] == "collision":
            print(f"=== ep{ep} collision at step {i} ===")
            for row in hist[-12:]:
                print(f"  step{row[0]:4d} v={row[1]:.2f} gap={row[2]:5.2f} "
                      f"a={row[3]:+.2f} s={row[4]:+.2f} prog={row[5]:.1f}")
            break
        if done:
            break
