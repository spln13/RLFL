import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# =============== 1. Actor 与 Critic 网络 ===============

class ContinuousPruneActor(nn.Module):
    """
    连续动作空间的 Actor，用于输出剪枝率。
    - 输入: state (如客户端的性能指标、当前模型大小、精度等)
    - 输出: mean, log_std => 通过 Normal(mean, std) 来采样
    """

    def __init__(self, state_dim, hidden_dim):
        super(ContinuousPruneActor, self).__init__()
        self.fc_body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)  # 只输出1维 => 剪枝率
        # 可以把 log_std 写死，也可以让它变成可学习参数
        self.log_std = nn.Parameter(torch.zeros(1))  # 这里假设可学习

    def forward(self, x):
        feat = self.fc_body(x)
        mean = self.mean_head(feat)
        return mean, self.log_std  # shape: ([batch,1], [1])


class Critic(nn.Module):
    """
    与原先相同: 输入 state, 输出一个 state value。
    """

    def __init__(self, state_dim, hidden_dim):
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


# =============== 2. PPO Agent ===============

class PruningPPOAgent:
    """
    连续动作 PPO，用于输出单维“剪枝率” (0~1).
    - 状态维度: state_dim
    - 动作维度: 1 (表示要剪枝的比例)
    """

    def __init__(
            self,
            state_dim,
            hidden_dim=128,
            actor_lr=1e-4,
            critic_lr=1e-4,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=5
    ):
        self.actor = ContinuousPruneActor(state_dim, hidden_dim)
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
        输入: 单个环境状态, shape = (state_dim,)
        输出: (action, logprob)
        - action: float, 剪枝率(理论上 0~1), 需在外部 clip 或 sigmoid
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        mean, log_std = self.actor(state_t)
        std = torch.exp(log_std)  # [1], 取指数 => 标准差

        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()  # re-parameter trick
        logprob = dist.log_prob(sample).sum(dim=-1)  # 动作空间=1维时，sum不影响结果

        # 可以在此强行 clip 到 [0,1], 或者用 sigmoid
        #   prune_ratio = torch.sigmoid(sample)  # => (0,1)
        #   或者 prune_ratio = torch.clamp(sample, 0, 1)
        #   具体取决于您的需求
        prune_ratio = torch.sigmoid(sample).item()  # 这里示例用 sigmoid

        return prune_ratio, logprob.item()

    def store_transition(self, state, action, logprob, reward, done):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(logprob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def update(self):
        """
        使用存下来的数据进行多次迭代更新
        """
        # 1. 整理数据
        states = torch.FloatTensor(self.memory['states'])
        actions = torch.FloatTensor(self.memory['actions']).unsqueeze(1)  # shape=[batch,1]
        old_logprobs = torch.FloatTensor(self.memory['logprobs']).unsqueeze(1)  # shape=[batch,1]

        # 2. 计算回报(简单 discounted return)
        returns = []
        discounted = 0.0
        for r, d in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
            if d:
                discounted = 0.0
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        # 3. 计算 advantage
        values = self.critic(states)
        advantage = returns - values.detach()

        # 4. 多次迭代更新 Actor, Critic
        for _ in range(self.k_epochs):
            # 4.1 重新计算当前策略下的 log_prob
            mean, log_std = self.actor(states)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions).sum(dim=-1, keepdim=True)  # [batch,1]
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - old_logprobs)

            # 4.2 PPO clip
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()  # 取负做梯度下降
            critic_loss = nn.MSELoss()(self.critic(states), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 5. 清空缓冲
        for k in self.memory.keys():
            self.memory[k] = []


# =============== 3. 简单测试 ===============
if __name__ == "__main__":
    # 假设状态维度是 5
    agent = PruningPPOAgent(state_dim=5, hidden_dim=64)

    # 人为构造一些(状态->动作->回报)的数据
    # 实际使用中，您需要把"剪枝完的模型大小、精度"等作为reward的依据
    for step in range(10):
        state = np.random.randn(5)
        action, logprob = agent.select_action(state)
        # 这里 action 即是剪枝率, (0~1)
        # 根据您的需求来算 reward
        # 比如: reward = (精度差) * -1 + (模型大小减少) * 常数 ... 等
        reward = -abs(0.5 - action)  # 仅示例：离 0.5 越近 reward 越高
        done = False
        agent.store_transition(state, action, logprob, reward, done)

    # 更新 PPO
    agent.update()

    # 下次再 select_action(...) 就会用到更新后的策略
    test_state = np.random.randn(5)
    new_action, _ = agent.select_action(test_state)
    print("Prune ratio after update:", new_action)
