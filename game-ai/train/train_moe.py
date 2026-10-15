"""多任务 PPO：用一个 MoE 策略同时训练"机器人赛跑"和"失控汽车驾驶"。

两个任务共享同一个网络，靠 router 分流到不同专家组合。
每次更新从两个环境各采样一段 rollout，advantage 按任务分别归一化
（两个任务奖励尺度不同），再合并做 PPO 更新。
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from env import (RunnerEnv, CarEnv, MAX_STEPS, N_OBS,
                 runner_heuristic, car_heuristic)
from model import MoEPolicy, MLPPolicy, moe_balance_loss, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
N_EXPERTS = 4


def task_onehot(task_id, n):
    t = np.zeros((n, 2), dtype=np.float32)
    t[:, task_id] = 1.0
    return t


def collect(env, task_id, policy, T, device):
    n = env.n
    ob = np.zeros((T + 1, n, N_OBS), np.float32)
    ac = np.zeros((T, n), np.int64)
    lp = np.zeros((T, n), np.float32)
    vl = np.zeros((T + 1, n), np.float32)
    rw = np.zeros((T, n), np.float32)
    dn = np.zeros((T, n), np.float32)
    gt = np.zeros((T, n, policy.n_experts), np.float32)
    ep_dist, ep_fail, ep_len = [], [], []
    oh = torch.as_tensor(task_onehot(task_id, n), device=device)

    o = env.obs()
    for t in range(T):
        with torch.no_grad():
            a, logp, _, v, gate = policy.act(
                torch.as_tensor(o, device=device), oh)
        no, r, done, info = env.step(a.cpu().numpy())
        ob[t], ac[t], lp[t], vl[t] = o, a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()
        rw[t], dn[t], gt[t] = r, done.astype(np.float32), gate.cpu().numpy()
        idx = np.where(done)[0]
        if len(idx):
            ep_dist.extend(info["x"][idx].tolist())
            ep_fail.extend(info["fallen"][idx].astype(float).tolist())
            ep_len.extend(env.steps[idx].tolist())
            env.reset_idx(idx)
        o = env.obs()

    with torch.no_grad():
        _, _, _, vlast, _ = policy.act(torch.as_tensor(o, device=device), oh)
    ob[T], vl[T] = o, vlast.cpu().numpy()

    task_col = np.full((T * n,), task_id, np.int64)
    batch = dict(
        obs=ob[:T].reshape(T * n, N_OBS), act=ac.reshape(T * n),
        logp=lp.reshape(T * n), val=vl[:T].reshape(T * n),
        rew=rw.reshape(T * n), done=dn.reshape(T * n),
        gate=gt.reshape(T * n, gt.shape[2]), task=task_col,
        last_val=vl[T],
    )
    stats = dict(
        dist=float(np.mean(ep_dist)) if ep_dist else float("nan"),
        fail=float(np.mean(ep_fail)) if ep_fail else float("nan"),
        length=float(np.mean(ep_len)) if ep_len else float("nan"),
        n_ep=len(ep_dist),
    )
    return batch, stats


def compute_gae(b, gamma, lam):
    T_n = b["rew"].shape[0]
    n = b["last_val"].shape[0]
    rew = b["rew"].reshape(-1, n)
    done = b["done"].reshape(-1, n)
    val = b["val"].reshape(-1, n)
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    gae = np.zeros(n)
    for t in reversed(range(T)):
        nv = b["last_val"] if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nv * (1.0 - done[t]) - val[t]
        gae = delta + gamma * lam * (1.0 - done[t]) * gae
        adv[t] = gae
    ret = adv + val
    return adv.reshape(T_n), ret.reshape(T_n)


def collect_demo(env_cls, task_id, policy_fn, n_samples, seed, n_env=64):
    """用启发式策略采集示范数据。交替迈腿这类组合动作靠随机探索很难发现，
    先用行为克隆把策略带到"会跑"的水平，再交给 PPO 微调。"""
    env = env_cls(n=n_env, seed=seed)
    env.reset(seed=seed)
    obs_list, act_list = [], []
    while len(obs_list) * n_env < n_samples:
        obs_list.append(env.obs().copy())
        a = policy_fn(env)
        act_list.append(a.copy())
        _, _, done, _ = env.step(a)
        idx = np.where(done)[0]
        if len(idx):
            env.reset_idx(idx)
    ob = np.concatenate(obs_list)[:n_samples]
    ac = np.concatenate(act_list)[:n_samples]
    return ob, ac


def warmup(policy, device, n_samples, epochs, lr=1e-3, bs=256, seed=0):
    data = []
    for tid, cls, fn in ((0, RunnerEnv, runner_heuristic), (1, CarEnv, car_heuristic)):
        ob, ac = collect_demo(cls, tid, fn, n_samples, seed=seed + tid)
        data.append((torch.as_tensor(ob, device=device),
                     torch.as_tensor(task_onehot(tid, ob.shape[0]), device=device),
                     torch.as_tensor(ac, device=device)))
    obs = torch.cat([d[0] for d in data])
    task = torch.cat([d[1] for d in data])
    act = torch.cat([d[2] for d in data])
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    N = obs.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot, nacc = 0.0, 0
        for s in range(0, N, bs):
            mb = perm[s:s + bs]
            logits, _, gate = policy.forward(obs[mb], task[mb])
            loss = F.cross_entropy(logits, act[mb]) + 0.01 * moe_balance_loss(gate)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float((logits.argmax(-1) == act[mb]).float().mean())
            nacc += 1
        print("  warmup ep%d  acc %.3f" % (ep + 1, tot / max(nacc, 1)))


def evaluate(env_cls, task_id, policy, device, n_ep=16, seed=0):
    env = env_cls(n=n_ep, seed=seed)
    env.reset(seed=seed)
    oh = torch.as_tensor(task_onehot(task_id, n_ep), device=device)
    dist = np.zeros(n_ep)
    fin = np.zeros(n_ep, dtype=bool)
    gate_acc = np.zeros(policy.n_experts)
    gate_n = 0
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            logits, _, gate = policy.forward(
                torch.as_tensor(env.obs(), device=device), oh)
        a = logits.argmax(-1).cpu().numpy()
        _, _, done, info = env.step(a)
        gate_acc += np.asarray(gate.cpu().numpy()).reshape(-1, gate.shape[-1]).sum(0)
        gate_n += n_ep
        idx = np.where(done & ~fin)[0]
        for i in idx:
            dist[i] = info["x"][i]
            fin[i] = True
        if fin.all():
            break
    dist[~fin] = info["x"][~fin]
    return {
        "mean": float(dist.mean()),
        "max": float(dist.max()),
        "fail": float(np.mean(info["fallen"])),
        "gate": (np.asarray(gate_acc) / max(gate_n, 1)).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--n-env", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.008)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--moe", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--arch", default="moe", choices=["moe", "mlp"])
    ap.add_argument("--warmup-samples", type=int, default=30000)
    ap.add_argument("--warmup-epochs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    policy = (MoEPolicy(hidden=64, n_experts=N_EXPERTS, top_k=2) if args.arch == 'moe'
                else MLPPolicy(hidden=96)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    print("device=%s  params=%d" % (device, count_params(policy)))

    if args.warmup_epochs > 0:
        print("behavior-cloning warmup ...")
        warmup(policy, device, args.warmup_samples, args.warmup_epochs, seed=args.seed)

    envs = {0: RunnerEnv(args.n_env, seed=args.seed),
            1: CarEnv(args.n_env, seed=args.seed + 1)}
    for t, e in envs.items():
        e.reset(seed=args.seed + t)

    hist = {"update": [], "runner": [], "car": [], "loss": [], "gate_runner": [], "gate_car": []}
    best = -1e9
    t0 = time.time()

    for upd in range(1, args.updates + 1):
        batches, stats = [], {}
        for tid in (0, 1):
            b, s = collect(envs[tid], tid, policy, args.horizon, device)
            adv, ret = compute_gae(b, args.gamma, args.lam)
            b["adv"], b["ret"] = adv, ret
            batches.append(b)
            stats[tid] = s

        obs = torch.as_tensor(np.concatenate([b["obs"] for b in batches]), device=device)
        task = torch.as_tensor(np.concatenate([task_onehot(b["task"][0], b["act"].shape[0])
                                               for b in batches]), device=device)
        act = torch.as_tensor(np.concatenate([b["act"] for b in batches]), device=device)
        logp_old = torch.as_tensor(np.concatenate([b["logp"] for b in batches]), device=device)
        ret = torch.as_tensor(np.concatenate([b["ret"] for b in batches]), device=device)
        tk = torch.as_tensor(np.concatenate([b["task"] for b in batches]), device=device)
        adv = np.concatenate([b["adv"] for b in batches])
        # 按任务分别归一化 advantage（奖励尺度不同）
        for t in (0, 1):
            m = (tk.cpu().numpy() == t)
            if m.sum() > 1:
                adv[m] = (adv[m] - adv[m].mean()) / (adv[m].std() + 1e-8)
        adv = torch.as_tensor(adv, device=device, dtype=torch.float32)

        N = obs.shape[0]
        idxs = np.arange(N)
        total_loss = 0.0
        for _ in range(args.epochs):
            np.random.shuffle(idxs)
            for s in range(0, N, args.mb):
                mb = torch.as_tensor(idxs[s:s + args.mb], device=device)
                logits, value, gate = policy.forward(obs[mb], task[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                pg = -torch.min(ratio * adv[mb],
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]).mean()
                vl = F.mse_loss(value, ret[mb])
                el = dist.entropy().mean()
                bl = moe_balance_loss(gate)
                loss = pg + args.vf * vl - args.ent * el + args.moe * bl
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                total_loss += float(loss)

        if upd % args.eval_every == 0 or upd == 1:
            r = evaluate(RunnerEnv, 0, policy, device, seed=100)
            c = evaluate(CarEnv, 1, policy, device, seed=200)
            hist["update"].append(upd)
            hist["runner"].append(r["mean"])
            hist["car"].append(c["mean"])
            hist["loss"].append(total_loss / (args.epochs * (N // args.mb + 1)))
            hist["gate_runner"].append(r["gate"])
            hist["gate_car"].append(c["gate"])
            print("[%4d/%d] loss %7.3f | runner %6.2f m (fail %.0f%%) | car %7.1f m (fail %.0f%%) | %.0fs"
                  % (upd, args.updates, hist["loss"][-1], r["mean"], r["fail"] * 100,
                     c["mean"], c["fail"] * 100, time.time() - t0))
            score = r["mean"] / 80.0 + c["mean"] / 200.0
            if score > best:
                best = score
                torch.save(policy.state_dict(), os.path.join(args.out, args.arch + ".pt"))

    torch.save(policy.state_dict(), os.path.join(args.out, args.arch + "_last.pt"))
    with open(os.path.join(args.out, args.arch + "_history.json"), "w") as f:
        json.dump(hist, f)
    print("best score %.3f -> %s/moe.pt   elapsed %.0fs" % (best, args.out, time.time() - t0))


if __name__ == "__main__":
    main()
