"""导出 MoE 策略为 ONNX，并生成页面使用的指标数据。

产出：
  assets/models/moe.onnx        浏览器端推理模型
  assets/js/metrics.js          评估指标 + 训练曲线 + 专家路由统计（供页面直接引用）
"""

import base64
import json
import os

import numpy as np
import torch

from env import (RunnerEnv, CarEnv, MAX_STEPS, runner_heuristic, car_heuristic)
from model import MoEPolicy, MLPPolicy, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "ckpt", "moe.pt")
OUT_ONNX = os.path.join(ROOT, "assets", "models", "moe.onnx")
OUT_METRICS = os.path.join(ROOT, "assets", "js", "metrics.js")
OUT_WEIGHTS = os.path.join(ROOT, "assets", "models", "moe_weights.json")

N_EXPERTS = 4


def load_policy(ckpt=CKPT, arch="moe"):
    if arch == "mlp":
        m = MLPPolicy(hidden=96)
    else:
        m = MoEPolicy(hidden=64, n_experts=N_EXPERTS, top_k=2)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


def load_history(arch="moe"):
    """优先读 {arch}_history.json，其次通用 history.json（早期版本写入）。"""
    for name in ("%s_history.json" % arch, "history.json"):
        fp = os.path.join(HERE, "ckpt", name)
        if os.path.exists(fp):
            try:
                h = json.load(open(fp))
                if h.get("update"):
                    return h
            except Exception:
                pass
    return {}


def task_onehot(tid, n):
    t = np.zeros((n, 2), dtype=np.float32)
    t[:, tid] = 1.0
    return t


def evaluate_policy(policy, env_cls, tid, n_ep=32, seed=0):
    env = env_cls(n=n_ep, seed=seed)
    env.reset(seed=seed)
    oh = torch.as_tensor(task_onehot(tid, n_ep))
    dist = np.zeros(n_ep)
    fin = np.zeros(n_ep, dtype=bool)
    fails = 0
    gate_acc = np.zeros(N_EXPERTS)
    gate_n = 0
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            logits, _, gate = policy.forward(torch.as_tensor(env.obs()), oh)
        a = logits.argmax(-1).numpy()
        _, _, done, info = env.step(a)
        gate_acc += gate.numpy().sum(0)
        gate_n += n_ep
        idx = np.where(done & ~fin)[0]
        for i in idx:
            dist[i] = info["x"][i]
            fin[i] = True
            if info["fallen"][i]:
                fails += 1
        if fin.all():
            break
    dist[~fin] = info["x"][~fin]
    return {
        "mean": float(dist.mean()),
        "max": float(dist.max()),
        "std": float(dist.std()),
        "fail": float(fails / n_ep),
        "gate": [round(float(g), 4) for g in gate_acc / max(gate_n, 1)],
    }


def evaluate_heuristic(env_cls, policy_fn, n_ep=32, seed=0):
    env = env_cls(n=n_ep, seed=seed)
    env.reset(seed=seed)
    dist = np.zeros(n_ep)
    fin = np.zeros(n_ep, dtype=bool)
    fails = 0
    for _ in range(MAX_STEPS):
        _, _, done, info = env.step(policy_fn(env))
        idx = np.where(done & ~fin)[0]
        for i in idx:
            dist[i] = info["x"][i]
            fin[i] = True
            if info["fallen"][i]:
                fails += 1
        if fin.all():
            break
    dist[~fin] = info["x"][~fin]
    return {"mean": float(dist.mean()), "max": float(dist.max()), "fail": float(fails / n_ep)}


def evaluate_random(env_cls, n_ep=32, seed=0):
    env = env_cls(n=n_ep, seed=seed)
    env.reset(seed=seed)
    dist = np.zeros(n_ep)
    fin = np.zeros(n_ep, dtype=bool)
    fails = 0
    for _ in range(MAX_STEPS):
        _, _, done, info = env.step(env.rng.integers(0, 4, size=n_ep))
        idx = np.where(done & ~fin)[0]
        for i in idx:
            dist[i] = info["x"][i]
            fin[i] = True
            if info["fallen"][i]:
                fails += 1
        if fin.all():
            break
    dist[~fin] = info["x"][~fin]
    return {"mean": float(dist.mean()), "max": float(dist.max()), "fail": float(fails / n_ep)}


def main():
    policy = load_policy()
    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)

    # ---------- 导出 ONNX ----------
    dummy_obs = torch.zeros(1, 14, dtype=torch.float32)
    dummy_task = torch.zeros(1, 2, dtype=torch.float32)
    try:
        torch.onnx.export(
            policy, (dummy_obs, dummy_task), OUT_ONNX,
            input_names=["obs", "task"],
            output_names=["logits", "value", "gate"],
            opset_version=17, dynamo=False,
        )
        size_kb = os.path.getsize(OUT_ONNX) / 1024.0
    except Exception as exc:                      # onnx 未安装时跳过，权重导出不受影响
        print("onnx export skipped:", type(exc).__name__, str(exc)[:120])
        size_kb = 0.0

    # ---------- 导出权重（供浏览器端零依赖推理，模型仅 ~22K 参数） ----------
    sd = policy.state_dict()
    wjson = {}
    for k, v in sd.items():
        arr = v.detach().numpy().astype(np.float32)
        wjson[k] = {"shape": list(arr.shape),
                    "data": base64.b64encode(arr.tobytes()).decode("ascii")}
    with open(OUT_WEIGHTS, "w") as f:
        json.dump(wjson, f)
    w_kb = os.path.getsize(OUT_WEIGHTS) / 1024.0

    # ---------- ONNX 与 PyTorch 一致性 ----------
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        max_err = 0.0
        for _ in range(20):
            o = rng.normal(size=(1, 14)).astype(np.float32)
            t = task_onehot(rng.integers(0, 2), 1)
            with torch.no_grad():
                lg, v, g = policy.forward(torch.as_tensor(o), torch.as_tensor(t))
            ol, ov, og = sess.run(None, {"obs": o, "task": t})
            max_err = max(max_err,
                          float(np.abs(lg.numpy() - ol).max()),
                          float(np.abs(v.numpy() - ov).max()))
        print("onnx vs torch max abs err: %.3e" % max_err)
        onnx_ok = max_err < 1e-4
    except ImportError:
        print("onnxruntime not installed, skip verification")
        max_err, onnx_ok = -1.0, None

    # ---------- 指标 ----------
    run_ai = evaluate_policy(policy, RunnerEnv, 0, seed=100)
    car_ai = evaluate_policy(policy, CarEnv, 1, seed=200)
    run_heu = evaluate_heuristic(RunnerEnv, runner_heuristic, seed=100)
    car_heu = evaluate_heuristic(CarEnv, car_heuristic, seed=200)
    run_rnd = evaluate_random(RunnerEnv, seed=100)
    car_rnd = evaluate_random(CarEnv, seed=200)

    hist = load_history("moe")

    # ---------- 架构消融：等参数量的普通 MLP ----------
    mlp_path = os.path.join(HERE, "ckpt", "mlp.pt")
    ablation = None
    if os.path.exists(mlp_path):
        mlp = load_policy(mlp_path, "mlp")
        ablation = {
            "params": count_params(mlp),
            "runner": round(evaluate_policy(mlp, RunnerEnv, 0, seed=100)["mean"], 2),
            "car": round(evaluate_policy(mlp, CarEnv, 1, seed=200)["mean"], 2),
        }

    metrics = {
        "params": count_params(policy),
        "onnx_kb": round(size_kb, 1),
        "weights_kb": round(w_kb, 1),
        "n_experts": N_EXPERTS,
        "onnx_err": max_err,
        "ablation": ablation,
        "runner": {"ai": run_ai, "heuristic": run_heu, "random": run_rnd},
        "car": {"ai": car_ai, "heuristic": car_heu, "random": car_rnd},
        "history": {
            "update": hist.get("update", []),
            "runner": [round(float(x), 2) for x in hist.get("runner", [])],
            "car": [round(float(x), 2) for x in hist.get("car", [])],
            "loss": [round(float(x), 4) for x in hist.get("loss", [])],
        },
    }
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("// 由 train/export_onnx.py 自动生成，请勿手改\n")
        f.write("export const METRICS = ")
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write(";\n")

    print("params=%d  onnx=%.1f KB  onnx_ok=%s" % (metrics["params"], size_kb, onnx_ok))
    print("runner  ai %.2f m | heuristic %.2f m | random %.2f m"
          % (run_ai["mean"], run_heu["mean"], run_rnd["mean"]))
    print("car     ai %.2f m | heuristic %.2f m | random %.2f m"
          % (car_ai["mean"], car_heu["mean"], car_rnd["mean"]))
    print("gate runner %s" % [round(x, 3) for x in run_ai["gate"]])
    print("gate car    %s" % [round(x, 3) for x in car_ai["gate"]])
    print("wrote", OUT_ONNX, "and", OUT_METRICS)


if __name__ == "__main__":
    main()
