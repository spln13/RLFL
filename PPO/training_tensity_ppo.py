import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict


class TrainIntensityActor(nn.Module):
    """
    连续动作空间的 Actor 用于动态训练强度输出.
    输入: state (如客户端上轮accuracy, compute time, data size等)
    输出: mean, log_std (通过 Normal(mean, std) 采样)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(TrainIntensityActor, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)  # 动作维度=1 => 训练强度
        self.log_std = nn.Parameter(torch.zeros(1))  # 可学习的对数标准差

    def forward(self, x):
        feat = self.body(x)
        mean = self.mean_head(feat)
        # log_std 是一个可学习的标量
        return mean, self.log_std


class Critic(nn.Module):
    """
    与之前相同: 输入 state, 输出一个 value(标量).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class TrainIntensityPPOAgent:
    """
    用于动态训练强度分配的 PPO（连续动作空间）

    接口:
      - select_action(state): 返回 (action, logprob)
        其中 action ~ Normal(mean, std) 的一个采样,
        一般可映射到相应区间(如 1 ~ 10 epoch).
      - store_transition(state, action, logprob, reward, done):
        将该条经验存入内存.
      - update():
        用 GAE 或简单 discounted return, 多次迭代更新 Actor/Critic
    """

    def __init__(
            self,
            state_dim: int,
            hidden_dim: int = 128,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            gamma: float = 0.99,
            eps_clip: float = 0.2,
            k_epochs: int = 3
    ):
        self.actor = TrainIntensityActor(state_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 经验缓存
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': []
        }

    def select_action(self, state: np.ndarray):
        """
        输入: 单个状态向量 (np.ndarray)
        输出: (action, logprob)
         - action: float, 训练强度(可再做映射到 1~10, etc)
         - logprob: 动作对数概率, 用于后续更新
        """
        # shape = (1, state_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state_t)
        std = torch.exp(log_std)  # shape = [1]

        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()  # reparameter trick
        logprob = dist.log_prob(sample).sum(dim=-1)  # 动作维度=1 => sum不影响结果

        # 例如: action 保持原状, 0~1 => 之后外部clip或映射
        # 也可以在此加sigmoid. 视您需求
        action = sample.item()
        return action, logprob.item()

    def store_transition(self, state, action, logprob, reward, done):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(logprob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def update(self):
        """
        用当前 memory 里的数据进行 PPO 多轮迭代.
        这里示例用简单 discounted return, 您也可换成GAE.
        """
        states = torch.FloatTensor(self.memory['states'])  # shape = [batch, state_dim]
        actions = torch.FloatTensor(self.memory['actions']).unsqueeze(1)  # [batch, 1]
        old_logprobs = torch.FloatTensor(self.memory['logprobs']).unsqueeze(1)  # [batch, 1]

        # (1) 计算 returns
        returns = []
        discounted_reward = 0
        for (r, d) in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
            if d:
                discounted_reward = 0  # 回合结束
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        # (2) advantage
        values = self.critic(states)
        advantage = returns - values.detach()

        # (3) 多次迭代更新
        for _ in range(self.k_epochs):
            # a) 重新计算 logprob
            mean, log_std = self.actor(states)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - old_logprobs)

            # b) PPO clip
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(states)
            critic_loss = nn.MSELoss()(new_values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # (4) 清空 memory
        for k in self.memory.keys():
            self.memory[k] = []

        print("[TrainIntensityPPO] Update finished, memory cleared.\n")
