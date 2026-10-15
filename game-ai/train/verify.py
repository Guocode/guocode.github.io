"""跨端一致性检查用的参考数据生成。

1) 导出一组 (obs, task) 及其 PyTorch 前向结果，供 assets/js/moe.js 比对
2) 打印 Python 端启发式基线的成绩，供浏览器端 env.js 做统计一致性比对
"""

import json
import os

import numpy as np
import torch

from env import RunnerEnv, CarEnv, MAX_STEPS
from model import MoEPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "ckpt", "moe.pt")


def gen_forward_cases(n=12):
    m = MoEPolicy()
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval()
    rng = np.random.default_rng(7)
    cases = []
    for i in range(n):
        obs = (rng.normal(size=14) * 0.8).astype(np.float32)
        tid = i % 2
        t = np.zeros((1, 2), dtype=np.float32)
        t[0, tid] = 1.0
        with torch.no_grad():
            lg, v, g = m.forward(torch.as_tensor(obs[None]), torch.as_tensor(t))
        cases.append({
            "obs": [float(x) for x in obs],
            "task": tid,
            "logits": [float(x) for x in lg[0]],
            "value": float(v[0]),
            "gate": [float(x) for x in g[0]],
        })
    out = os.path.join(ROOT, "train", "verify_cases.json")
    with open(out, "w") as f:
        json.dump(cases, f)
    print("wrote", out)


def step_only(env):
    both = env.stance[:, 0] & env.stance[:, 1]
    sl = both & (env.fx[:, 0] < env.fx[:, 1])
    sr = both & (env.fx[:, 1] <= env.fx[:, 0])
    return np.where(sl, 0, np.where(sr, 1, 2)).astype(np.int64)


def car_keep(env):
    from env import track_curvature, WHEELBASE, ROAD_HALF
    kap = track_curvature(env.s + 14.0)
    target = -0.20 * (env.d / ROAD_HALF) + 0.9 * WHEELBASE * kap
    err = env.psi - target
    act = np.where(err > 0.015, 0, np.where(err < -0.015, 1, 2))
    return np.where(env.v > 16.0, 3, act).astype(np.int64)


def run(env_cls, pol, n=16, seed=1):
    e = env_cls(n=n, seed=seed)
    e.reset(seed=seed)
    dist = np.zeros(n)
    fin = np.zeros(n, dtype=bool)
    for _ in range(MAX_STEPS):
        _, _, done, info = e.step(pol(e))
        idx = np.where(done & ~fin)[0]
        dist[idx] = info["x"][idx]
        fin[idx] = True
        if fin.all():
            break
    dist[~fin] = info["x"][~fin]
    return dist


if __name__ == "__main__":
    gen_forward_cases()
    d = run(RunnerEnv, step_only)
    print("runner alternating-step  mean %.2f  std %.2f  min %.2f  max %.2f"
          % (d.mean(), d.std(), d.min(), d.max()))
    d2 = run(CarEnv, car_keep)
    print("car lane-keeping         mean %.2f  std %.2f  min %.2f  max %.2f"
          % (d2.mean(), d2.std(), d2.min(), d2.max()))
