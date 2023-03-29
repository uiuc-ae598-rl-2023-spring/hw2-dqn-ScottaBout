import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import discreteaction_pendulum
import numpy as np
import matplotlib.pyplot as plt


class DQN(nn.Module):
    """

    """

    def __init__(self, in_dim, hid_dim, out_dim):
        super(DQN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.q_network = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=out_dim)
        )

        self.target_q_network = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=out_dim)
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.replay_memory = ReplayMemory(capacity=10000)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4, amsgrad=True)
        self.epsilon = 0.9
        self.batch_size = 128
        self.gamma = 0.95
        self.criterion = nn.SmoothL1Loss()
        self.reward_list = []
        self.tau = 0.005

    def forward(self, input) -> torch.Tensor:
        """
        Forward function for q network
        :param input: state - a tensor
        :return: Q values - a tensor
        """
        if isinstance(input, tuple):
            pass
        elif not isinstance(input, torch.Tensor):
            input = torch.Tensor(input)

        return self.q_network(input)

    def target_forward(self, input) -> torch.Tensor:
        """
        Forward function for target q network
        :param input: state - a tensor
        :return: target Q values - a tensor
        """
        if not isinstance(input, torch.Tensor):
            input = torch.Tensor(input)

        return self.target_q_network(input)

    def learn(self, env: discreteaction_pendulum.Pendulum(), episode_length: int, num_episodes: int) -> None:
        """
        Primary function to train Q and target Q networks
        :param env: pendulum environment
        :param episode_length: int
        :param num_episodes: int -> to experiment with
        :return: Explicitly returns nothing but updates Q and target Q networks and records data for plotting purposes
        """
        for episode in range(num_episodes):
            state = env.reset()
            reward_list = []
            for step in range(episode_length):
                Qs = self.forward(state)
                action = self.epsilon_greedy(env=env, Qs=Qs)
                (next_state, reward, done) = env.step(action)
                reward_list.append(reward)
                state, action, next_state, reward, done = torch.Tensor(state), torch.Tensor([action]), torch.Tensor(next_state), torch.Tensor([reward]), torch.Tensor([done])
                self.replay_memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()
            if episode % 10 == 0:
                print(f'episode = {episode} / {num_episodes}')
                self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.epsilon -= 0.00001
            self.reward_list.append(np.sum(reward_list).mean())
            reward_list.clear()

    def epsilon_greedy(self, env, Qs: torch.Tensor) -> int:
        """
        Epsilon greedy method for selecting an action
        :param env: pendulum environment
        :param Qs: Q values for discretized action space evaluated at recorded state
        :return: action - an integer
        """
        epsilon = np.random.uniform()
        if epsilon > self.epsilon:
            action = int(torch.argmax(Qs))
        else:
            action = int(random.randrange(env.num_actions))

        return action

    def optimize_model(self) -> None:
        """
        Trains Q network
        :return: None
        """
        if len(self.replay_memory) < self.batch_size:
            return
        transitions = self.replay_memory.sample(self.batch_size)
        batch = self.replay_memory.Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action).type(torch.int64)
        reward_batch = torch.stack(batch.reward)

        state_action_values = self.forward(state_batch).gather(1, action_batch)
        expected_state_action_values = reward_batch + self.gamma * torch.max(self.target_forward(state_batch))
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot(self) -> None:
        """
        Plots return over time
        :return: None
        """
        plt.plot(self.reward_list)
        plt.savefig('figures/giwoegfwh.png')
        plt.show()

    def generate_gif(self, env: discreteaction_pendulum.Pendulum()) -> None:
        """
        Generates a gif of an example trajectory with a trained agent
        :param env: pendulum environment
        :return: None
        """
        policy = self.q_network

        # Simulate an episode and save the result as an animated gif
        env.video(policy, filename='figures/bfwbgjgrwerw.gif')


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
