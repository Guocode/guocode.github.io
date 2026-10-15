"""图像输入实验：CNN 编码器 + 同款 MoE 路由（4 专家 top-2）+ actor/critic 头。
与 14 维向量版 MoEPolicy 对照——验证"只看像素"能否学会两个游戏。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACT = 4
N_TASK = 2
N_CH, FH, FW = 3, 40, 64


class CNNMoEPolicy(nn.Module):
    def __init__(self, hidden=48, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(N_CH, 8, 3, stride=2, padding=1), nn.ReLU(),   # 20x32
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),     # 10x16
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),    # 5x8
        )
        self.fc = nn.Sequential(nn.Linear(16 * 5 * 8 + N_TASK, hidden), nn.Tanh())
        self.router = nn.Linear(hidden, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
            for _ in range(n_experts)
        ])
        self.actor = nn.Linear(hidden, N_ACT)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, task):
        """obs (B,3,40,64) float in [0,1]; task (B,2) one-hot -> logits, value, gate"""
        h = self.fc(torch.cat([self.conv(obs).flatten(1), task], dim=-1))
        logits_r = self.router(h)
        topv, topi = torch.topk(logits_r, self.top_k, dim=-1)
        gate = torch.zeros_like(logits_r).scatter_(-1, topi, F.softmax(topv, dim=-1))
        e = torch.stack([ex(h) for ex in self.experts], dim=1)      # (B,E,H)
        y = (gate.unsqueeze(-1) * e).sum(dim=1)
        return self.actor(y), self.critic(y).squeeze(-1), gate

    def act(self, obs, task):
        logits, value, gate = self.forward(obs, task)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), value, gate


def count_params(m):
    return sum(p.numel() for p in m.parameters())
