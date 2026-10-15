"""MoE 策略网络：一个模型同时学会玩两个游戏。

结构
    obs(14) + task one-hot(2)
        -> trunk (共享底层)
        -> router -> top-2 门控权重
        -> 4 个 expert FFN（稀疏激活 top-2）
        -> 共享 actor / critic 头

两个游戏状态语义不同，router 会把它们分发到不同的专家组合；
训练结束后统计各任务对每个专家的平均使用率，即为"专家专业化"证据。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_OBS = 14
N_TASK = 2
N_ACT = 4


class MoEPolicy(nn.Module):
    def __init__(self, hidden=64, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.trunk = nn.Sequential(
            nn.Linear(N_OBS + N_TASK, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.router = nn.Linear(hidden, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
            for _ in range(n_experts)
        ])
        self.actor = nn.Linear(hidden, N_ACT)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, task):
        """obs (B,14)  task (B,2) one-hot -> logits, value, gate"""
        h = self.trunk(torch.cat([obs, task], dim=-1))
        logits_r = self.router(h)
        topv, topi = torch.topk(logits_r, self.top_k, dim=-1)
        gate = torch.zeros_like(logits_r).scatter_(
            -1, topi, F.softmax(topv, dim=-1))
        # dense 计算后按门控加权，保持 ONNX 导出简单
        e = torch.stack([ex(h) for ex in self.experts], dim=1)   # (B,E,H)
        y = (gate.unsqueeze(-1) * e).sum(dim=1)
        return self.actor(y), self.critic(y).squeeze(-1), gate

    def act(self, obs, task):
        logits, value, gate = self.forward(obs, task)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), value, gate


class MLPPolicy(nn.Module):
    """对照实验用的普通 MLP（等参数量，无专家、无路由）。
    用来验证在两个任务上 MoE 是否真的比同等规模的稠密网络更好。"""

    def __init__(self, hidden=96):
        super().__init__()
        self.n_experts = 1
        self.trunk = nn.Sequential(
            nn.Linear(N_OBS + N_TASK, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mid = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, N_ACT)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, task):
        h = self.trunk(torch.cat([obs, task], dim=-1))
        y = self.mid(h)
        gate = torch.ones(obs.shape[0], 1, device=obs.device)
        return self.actor(y), self.critic(y).squeeze(-1), gate

    def act(self, obs, task):
        logits, value, gate = self.forward(obs, task)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), value, gate


def moe_balance_loss(gate):
    """Switch Transformer 的辅助损失：鼓励专家负载均衡，避免路由坍缩。"""
    B, E = gate.shape
    f = gate.sum(0) / B                       # 使用频率
    P = (gate > 0).float().mean(0)            # 选中比例
    return E * (f * P).sum()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
