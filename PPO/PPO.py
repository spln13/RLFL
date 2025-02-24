import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


# ============================================================
#  一、离散动作空间的 PPO (PPO1)
#      用于给出模型大小分配 (例如 0=小模型, 1=中模型, 2=大模型...)
# ============================================================

class DiscreteActor(nn.Module):
    """
    离散动作空间的 Actor 网络。
    输入: state (形状 [batch_size, state_dim])
    输出: 每个离散动作的概率分布 (形状 [batch_size, num_actions])
    """

    def __init__(self, state_dim, hidden_dim, num_actions):
        super(DiscreteActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        # 返回各离散动作的 logits
        return self.net(x)


class Critic(nn.Module):
    """
    Critic 网络，用于估计状态价值。
    输入: state
    输出: state_value (标量)
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


class DiscretePPOAgent:
    """
    离散动作空间 PPO：用于模型大小分配。
    接口方法:
      - select_action(state): 给定单个状态, 返回离散动作(模型类别)及其他信息
      - update(): 使用采样好的数据更新Actor-Critic网络
    """

    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            hidden_dim: int = 128,
            actor_lr: float = 1e-3,
            critic_lr: float = 1e-3,
            gamma: float = 0.99,
            eps_clip: float = 0.2,
            k_epochs: int = 3
    ):
        self.actor = DiscreteActor(state_dim, hidden_dim, num_actions)
        self.critic = Critic(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.eps_clip = eps_clip  # PPO clipping 边界
        self.k_epochs = k_epochs  # 一次更新中的迭代次数

        self.logprobs = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        给定单个状态，输出离散动作(模型大小类别)及其对数概率。
        参数:
            state: shape = (state_dim,)
        返回:
            action: int, 动作(模型类别)
            logprob: float, 对数概率
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # shape = [1, state_dim]
        logits = self.actor(state_t)  # [1, num_actions]
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action.item(), logprob.item()

    def store_transition(self, state, action, logprob, reward, done):
        """
        将当前 transition 信息存储到缓冲区，等待后续 update。
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self):
        """
        基于存储的多个 (s, a, logprob, reward, done) 进行 PPO 更新。
        """
        # 将数据转换为 Tensor
        states_t = torch.FloatTensor(self.states)
        actions_t = torch.LongTensor(self.actions)
        old_logprobs_t = torch.FloatTensor(self.logprobs)

        # GAE 或 简单 discounted return
        returns = []
        discounted_reward = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                discounted_reward = 0
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns_t = torch.FloatTensor(returns).unsqueeze(1)  # shape [batch_size, 1]

        # Normalization
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # 计算 advantage
        values = self.critic(states_t)
        advantage = returns_t - values.detach()

        # 更新多次
        for _ in range(self.k_epochs):
            # 重新计算当前策略下的 logprobs
            logits = self.actor(states_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            new_logprobs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            # 价值函数
            new_values = self.critic(states_t)

            # ratio
            ratio = torch.exp(new_logprobs - old_logprobs_t)

            # 计算 actor loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()  # PPO 损失取负号做梯度下降
            # critic loss
            critic_loss = nn.MSELoss()(new_values, returns_t)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # 优化
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 清空 buffer
        self.logprobs.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()


# ============================================================
#  二、连续动作空间的 PPO (PPO2)
#      用于训练强度(可为浮点数或向量)的分配
# ============================================================

class ContinuousActor(nn.Module):
    """
    连续动作空间的 Actor 网络。
    输入: state
    输出: 动作的均值, 标准差(可选)
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ContinuousActor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 分别输出 mean 和 log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习，也可固定

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_head(x)
        # 这里假设对数标准差是可学习参数，因此直接从 self.log_std 读
        log_std = self.log_std
        return mean, log_std


class ContinuousPPOAgent:
    """
    连续动作空间 PPO: 用于训练强度调整。
    接口方法:
      - select_action(state): 给定状态, 输出连续动作 (可对输出进行后处理, 如 softmax 归一化)
      - update(): 使用采样好的数据更新Actor-Critic网络
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 128,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            gamma: float = 0.99,
            eps_clip: float = 0.2,
            k_epochs: int = 3
    ):
        self.actor = ContinuousActor(state_dim, hidden_dim, action_dim)
        self.critic = Critic(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.logprobs = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        给定单个状态, 输出连续动作(训练强度), 并返回对应的 log_prob 以便更新。
        注意: 本示例输出的动作可在外部做 clip 或 softmax 等处理。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state_t)  # shape: [1, action_dim]
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)  # 多维动作时加和

        return action.detach().cpu().numpy()[0], logprob.item()

    def store_transition(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self):
        states_t = torch.FloatTensor(self.states)
        actions_t = torch.FloatTensor(self.actions)
        old_logprobs_t = torch.FloatTensor(self.logprobs).unsqueeze(1)

        # 计算 discounted return
        returns = []
        discounted_reward = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                discounted_reward = 0
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns_t = torch.FloatTensor(returns).unsqueeze(1)

        # Normalization
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Advantage
        values = self.critic(states_t)
        advantage = returns_t - values.detach()

        for _ in range(self.k_epochs):
            mean, log_std = self.actor(states_t)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions_t).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().mean()

            # ratio
            ratio = torch.exp(new_logprobs - old_logprobs_t)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(states_t)
            critic_loss = nn.MSELoss()(new_values, returns_t)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 清空 buffer
        self.logprobs.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()


# ============================================================
#  三、示例: 如何在外部使用上面两个 PPO 模型
# ============================================================

def example_usage_discrete_ppo():
    """
    演示如何使用 DiscretePPOAgent(对应模型大小分配)。
    """
    # 假设状态维度为 5, 动作(模型类别)总数为 3 => 小/中/大模型
    state_dim = 5
    num_actions = 3
    agent = DiscretePPOAgent(state_dim=state_dim, num_actions=num_actions)

    # 在实际使用中, 每一轮会收集一些 (state, action, reward) 数据
    # 这里仅模拟一个简单的示例
    for episode in range(5):
        state = np.random.randn(state_dim)
        action, logprob = agent.select_action(state)
        # 根据实际情况计算 reward, done
        # 例如: reward 可与减少的 straggling latency 负相关, done 视需要而定
        reward = -abs(np.random.randn())  # 随机的示例
        done = False

        agent.store_transition(state, action, logprob, reward, done)

    # 当存储了足够多的数据后, 调用 update() 进行 PPO 参数更新
    agent.update()

    # 之后再次 select_action(...) 即可得到更新后策略的动作(模型类别分配)


def example_usage_continuous_ppo():
    """
    演示如何使用 ContinuousPPOAgent(对应训练强度分配)。
    """
    # 假设状态维度为 6, 连续动作维度为 1(只输出一个训练强度),
    # 也可以把 action_dim 定义大于 1, 需要时再做归一化。
    state_dim = 6
    action_dim = 1
    agent = ContinuousPPOAgent(state_dim=state_dim, action_dim=action_dim)

    for episode in range(5):
        state = np.random.randn(state_dim)
        action, logprob = agent.select_action(state)
        # 这里的 action 即可以看作某种“训练强度”，比如 epoch 数量
        # 若需要总和=固定值，可在外部对整批客户端的 action 做softmax或normalize
        reward = -abs(np.random.randn())  # 仅作示例
        done = False

        agent.store_transition(state, action, logprob, reward, done)

    agent.update()


if __name__ == "__main__":
    # 简单运行示例
    example_usage_discrete_ppo()
    example_usage_continuous_ppo()
