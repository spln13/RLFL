import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def get_action(x, k, s):
#     action = []
#     for i in range(k):
#         action.append(int(x / (s ** (k - 1 - i))))
#         x = x % (s ** (k - 1 - i))
#     action.reverse()
#     return action


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.probs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        # del语句作用在变量上，而不是数据对象上。删除的是变量，而不是数据。
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.probs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_shape, n_latent_var):
        super(ActorCritic, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
        )
        self.head = nn.ModuleList()
        for size in action_shape:
            self.head.append(nn.Linear(64, size))
        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(obs_shape, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        # 为每个可能的离散动作选项，计算相应的 logit，维度变化为: $$(B, N) -> [(B, A_1), ..., (B, A_N)]$$
        logit = [h(x) for h in self.head]
        return logit

    def sample_action(self, logit):
        prob = torch.softmax(logit, dim=-1)
        # print(prob)
        # 构建广义伯努利分布。它的概率质量函数为: $$f(x=i|\boldsymbol{p})=p_i$$
        dist = torch.distributions.Categorical(probs=prob)
        # 为一个 batch 里的每个样本采样一个离散动作，并返回它
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, state, action, epoch, k):
        x = self.encoder(state)
        logit = [h(x) for h in self.head]
        action_probs = torch.softmax(logit[0], dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        # print(state_value)
        state_value = state_value.reshape(epoch, k)
        # print(state_value)
        # print(state_value.size())
        state_value = torch.sum(state_value, dim=1) / k
        state_value = state_value.reshape(epoch, 1)
        # print(state_value)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPOModelSize:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory, epoch, k):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        # 使用stack可以保留两个信息：[1. 序列] 和 [2. 张量矩阵] 信息，属于【扩张再拼接】的函数；
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        # 这里即可以对样本进行多次利用，提高利用率
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, epoch, k)
            dist_entropy = torch.sum(dist_entropy, dim=1) / k

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            ratios = torch.sum(ratios, dim=1) / k

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            # print(advantages)
            # print(ratios)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # print(loss)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(torch.load(path))

    def step_model_size(self, state, action, k, model_size):
        t = []
        for i in range(k):
            t.append(state[i] * model_size[action[i]])
        x = max(t) / min(t)

        if x <= 5:
            return 10 - x
        elif x > 8:
            return -200
        else:
            return -100
