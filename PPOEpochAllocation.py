import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def step_epoch(state, action, k):
    time = []
    for i in range(k):
        time.append(action[i] * 10 * state[i])
    max_time = max(time)
    min_time = min(time)
    # reward = (max_time - min_time) / max_time
    # reward = (std - reward) * 100
    reward = min_time - max_time

    return reward


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


class ActorCriticEpoch(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCriticEpoch, self).__init__()
        # action mean range 控制在0-1之间 且加合为1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # 方差
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        # 手动设置异常
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        # print('action_mean: {}'.format(action_mean))
        # print(self.action_var)
        cov_mat = torch.diag(self.action_var).to(device)
        # print(cov_mat)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        # print('sample_action: {}'.format(action))
        action = torch.clamp(action, 0, 1)
        # print('clamp_action: {}'.format(action))
        # action = F.softmax(action, dim=-1)
        # print('softmax_action: {}'.format(action))
        action_logprob = dist.log_prob(action)
        # print(torch.exp(action_logprob))
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        # torch.diag_embed(input, offset=0, dim1=-2, dim2=-1) → Tensor
        # Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input
        cov_mat = torch.diag_embed(action_var).to(device)
        # 生成一个多元高斯分布矩阵
        dist = MultivariateNormal(action_mean, cov_mat)
        # 我们的目的是要用这个随机的去逼近真正的选择动作action的高斯分布
        action_logprobs = dist.log_prob(action)
        # log_prob 是action在前面那个正太分布的概率的log ，我们相信action是对的 ，
        # 那么我们要求的正态分布曲线中点应该在action这里，所以最大化正太分布的概率的log， 改变mu,sigma得出一条中心点更加在a的正太分布。
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPOEpochAllocation:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCriticEpoch(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCriticEpoch(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
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

        # print("start!")
        # print(rewards)

        # convert list to tensor
        # 使用stack可以保留两个信息：[1. 序列] 和 [2. 张量矩阵] 信息，属于【扩张再拼接】的函数；
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # print(state_values)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # print(ratios)

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            # print(advantages)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

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
