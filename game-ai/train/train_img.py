"""图像输入版多任务 PPO：CNN 编码器 + MoE，只看像素帧学会两个游戏。
训练流程与 train_moe.py 一致（BC warmup + PPO），输入从 14 维状态向量
换成 frames.py 渲染的 (3,40,64) 像素帧。

用法：
  python train_img.py --updates 300 --lr 3e-4
输出：ckpt/cnn.pt + ckpt/cnn_history.json + 多 seed 最终评估
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from env import RunnerEnv, CarEnv, MAX_STEPS, runner_heuristic, car_heuristic
from frames import render
from model_img import CNNMoEPolicy, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
N_EXPERTS = 4
FRAME_SHAPE = (3, 40, 64)


def task_onehot(task_id, n):
    t = np.zeros((n, 2), dtype=np.float32)
    t[:, task_id] = 1.0
    return t


def collect(env, task_id, policy, T, device):
    n = env.n
    ob = np.zeros((T, n) + FRAME_SHAPE, np.float32)
    ac = np.zeros((T, n), np.int64)
    lp = np.zeros((T, n), np.float32)
    vl = np.zeros((T, n), np.float32)
    rw = np.zeros((T, n), np.float32)
    dn = np.zeros((T, n), np.float32)
    gt = np.zeros((T, n, policy.n_experts), np.float32)
    ep_dist, ep_fail, ep_len = [], [], []
    oh = torch.as_tensor(task_onehot(task_id, n), device=device)

    for t in range(T):
        frm = render(task_id, env)
        with torch.no_grad():
            a, logp, _, v, gate = policy.act(
                torch.as_tensor(frm, device=device), oh)
        no, r, done, info = env.step(a.cpu().numpy())
        ob[t], ac[t], lp[t], vl[t] = frm, a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()
        rw[t], dn[t], gt[t] = r, done.astype(np.float32), gate.cpu().numpy()
        idx = np.where(done)[0]
        if len(idx):
            ep_dist.extend(info["x"][idx].tolist())
            ep_fail.extend(info["fallen"][idx].astype(float).tolist())
            ep_len.extend(env.steps[idx].tolist())
            env.reset_idx(idx)

    # 价值引导：截断处的 V(s_t) 用当前帧估算（而非 0）
    with torch.no_grad():
        _, _, _, vlast, _ = policy.act(
            torch.as_tensor(render(task_id, env), device=device), oh)

    task_col = np.full((T * n,), task_id, np.int64)
    batch = dict(
        obs=ob.reshape(T * n, *FRAME_SHAPE), act=ac.reshape(T * n),
        logp=lp.reshape(T * n), val=vl.reshape(T * n),
        rew=rw.reshape(T * n), done=dn.reshape(T * n),
        gate=gt.reshape(T * n, gt.shape[2]), task=task_col,
        last_val=vlast.cpu().numpy().astype(np.float32),
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


def collect_demo(env_cls, task_id, policy_fn, n_samples, seed, n_env=32):
    env = env_cls(n=n_env, seed=seed)
    env.reset(seed=seed)
    frm_list, act_list = [], []
    while len(frm_list) * n_env < n_samples:
        frm_list.append(render(task_id, env))
        a = policy_fn(env)
        act_list.append(a.copy())
        _, _, done, _ = env.step(a)
        idx = np.where(done)[0]
        if len(idx):
            env.reset_idx(idx)
    frm = np.concatenate(frm_list)[:n_samples]
    ac = np.concatenate(act_list)[:n_samples]
    return frm, ac


def warmup(policy, device, n_samples, epochs, lr=1e-3, bs=256, seed=0):
    data = []
    for tid, cls, fn in ((0, RunnerEnv, runner_heuristic), (1, CarEnv, car_heuristic)):
        frm, ac = collect_demo(cls, tid, fn, n_samples, seed=seed + tid)
        data.append((torch.as_tensor(frm, device=device),
                     torch.as_tensor(task_onehot(tid, frm.shape[0]), device=device),
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
            loss = F.cross_entropy(logits, act[mb])
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
                torch.as_tensor(render(task_id, env), device=device), oh)
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


def evaluate_multiseed(env_cls, task_id, policy, device, n_seeds=8, n_ep=16):
    ds, fails = [], []
    for sd in range(1, n_seeds + 1):
        r = evaluate(env_cls, task_id, policy, device, n_ep=n_ep, seed=sd * 100)
        ds.append(r["mean"])
        fails.append(r["fail"])
    return float(np.mean(ds)), float(np.mean(fails))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--n-env", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.008)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--warmup-samples", type=int, default=12000)
    ap.add_argument("--warmup-epochs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cpu")
    os.makedirs(args.out, exist_ok=True)

    policy = CNNMoEPolicy(hidden=48, n_experts=N_EXPERTS, top_k=2).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    print("device=%s  params=%d" % (device, count_params(policy)))

    if args.warmup_epochs > 0:
        print("behavior-cloning warmup (pixel frames) ...")
        warmup(policy, device, args.warmup_samples, args.warmup_epochs, seed=args.seed)

    envs = {0: RunnerEnv(args.n_env, seed=args.seed),
            1: CarEnv(args.n_env, seed=args.seed + 1)}
    for t, e in envs.items():
        e.reset(seed=args.seed + t)

    hist = {"update": [], "runner": [], "car": [], "loss": [],
            "gate_runner": [], "gate_car": []}
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

        obs = torch.as_tensor(
            np.concatenate([b["obs"] for b in batches]), device=device)
        task = torch.as_tensor(np.concatenate(
            [task_onehot(b["task"][0], b["act"].shape[0]) for b in batches]), device=device)
        act = torch.as_tensor(np.concatenate([b["act"] for b in batches]), device=device)
        logp_old = torch.as_tensor(np.concatenate([b["logp"] for b in batches]), device=device)
        ret = torch.as_tensor(np.concatenate([b["ret"] for b in batches]), device=device)
        tk = torch.as_tensor(np.concatenate([b["task"] for b in batches]), device=device)
        adv = np.concatenate([b["adv"] for b in batches])
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
                loss = pg + args.vf * vl - args.ent * el
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
            score = r["mean"] / 150.0 + c["mean"] / 400.0
            if score > best:
                best = score
                torch.save(policy.state_dict(), os.path.join(args.out, "cnn.pt"))
                with open(os.path.join(args.out, "cnn_history.json"), "w") as f:
                    json.dump(hist, f)

    torch.save(policy.state_dict(), os.path.join(args.out, "cnn_last.pt"))
    with open(os.path.join(args.out, "cnn_history.json"), "w") as f:
        json.dump(hist, f)

    rm, rf = evaluate_multiseed(RunnerEnv, 0, policy, device)
    cm, cf = evaluate_multiseed(CarEnv, 1, policy, device)
    print("FINAL multi-seed (8x16): runner %.1f m (fail %.0f%%) | car %.1f m (fail %.0f%%)"
          % (rm, rf * 100, cm, cf * 100))
    with open(os.path.join(args.out, "cnn_final.json"), "w") as f:
        json.dump({"runner": rm, "runner_fail": rf, "car": cm, "car_fail": cf,
                   "params": count_params(policy)}, f)
    print("best score %.3f   elapsed %.0fs" % (best, time.time() - t0))


if __name__ == "__main__":
    main()
