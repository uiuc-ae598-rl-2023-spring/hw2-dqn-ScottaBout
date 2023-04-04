import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import discreteaction_pendulum
import numpy as np
import matplotlib.pyplot as plt
import math

torch.manual_seed(0)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQN(nn.Module):
    """
    DQN agent and methods
    """

    def __init__(self, env, hid_dim):
        super(DQN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env

        self.q_network = nn.Sequential(
            nn.Linear(in_features=self.env.num_states, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=self.env.num_actions)
        )
        self.q_network.to(device=self.device)

        self.target_q_network = nn.Sequential(
            nn.Linear(in_features=self.env.num_states, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=self.env.num_actions)
        )
        self.target_q_network.to(device=self.device)

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.replay_memory = ReplayMemory(capacity=10000)
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, amsgrad=True)
        self.epsilon = 0.9
        self.batch_size = 64
        self.gamma = 0.95
        self.criterion = nn.SmoothL1Loss()
        self.reward_list = []
        self.tau = 0.005
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.steps_done = 0

    def forward(self, input) -> torch.Tensor:
        """
        Forward function for q network
        :param input: state - a tensor
        :return: Q values - a tensor
        """
        if not isinstance(input, torch.Tensor):
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

    def learn(self, episode_length: int, num_episodes: int, replay=True, target=True) -> None:
        """
        Primary function to train Q and target Q networks
        :param episode_length: int
        :param num_episodes: int -> to experiment with
        :param replay: boolean -> True if we want replay, false otherwise
        :param target: boolean -> True if we want to update target q accordingly, false otherwise
        :return: Explicitly returns nothing but updates Q and target Q networks and records data for plotting purposes
        """
        if not replay:
            self.replay_memory = ReplayMemory(capacity=self.batch_size)
        for episode in range(num_episodes):
            state = self.env.reset()
            reward_list = []
            for step in range(episode_length):
                action = self.select_action(state)
                (next_state, reward, done) = self.env.step(action)
                reward_list.append(reward)
                self.replay_memory.push(state, action, next_state, reward, done)
                state = next_state
                if done:
                    state = self.env.reset()
                self.optimize_model()
                if not target:
                    self.target_q_network.load_state_dict(self.q_network.state_dict())
            if (episode % 10 == 0) and target:
                print(f'episode = {episode} / {num_episodes}')
                self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.reward_list.append(np.sum(reward_list).mean())
            reward_list.clear()

    def select_action(self, obs):
        """
        Epsilon greedy select an action
        :param obs:
        :return:
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.get_action(obs)
        else:
            return random.randrange(self.env.num_actions)

    def get_action(self, obs):
        """
        Takes argmax of q values
        :param obs:
        :return:
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        argmax_q = torch.argmax(q_values, dim=1)[0]
        action = argmax_q.detach()

        return action

    def optimize_model(self) -> None:
        """
        Trains Q network
        :return: None
        """
        if len(self.replay_memory) < self.batch_size:
            return
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # convert data to iterable - converting to np array is better than list because torch conversion from list
        # to tensor is extremely slow
        state_batch = np.array(batch.state)
        action_batch = np.asarray(batch.action)
        next_state_batch = np.asarray(batch.next_state)
        reward_batch = np.asarray(batch.reward)
        dones_batch = np.asarray(batch.done)

        # convert data to torch tensor for compatibility with pytorch networks - torch.float32 is the required dtype
        # for pytorch
        state_batch_t = torch.as_tensor(state_batch, dtype=torch.float32)
        action_batch_t = torch.as_tensor(action_batch, dtype=torch.int64).unsqueeze(-1)
        next_state_batch_t = torch.as_tensor(next_state_batch, dtype=torch.float32)
        reward_batch_t = torch.as_tensor(reward_batch, dtype=torch.float32).unsqueeze(-1)
        dones_batch_t = torch.as_tensor(dones_batch, dtype=torch.float32).unsqueeze(-1)

        # generate targets
        target_q_values = self.target_q_network(next_state_batch_t)
        max_target_q_vals = target_q_values.max(dim=1, keepdim=True)[0]
        targets = reward_batch_t + self.gamma * (1 - dones_batch_t) * max_target_q_vals

        # generate predictions
        q_vals = self.q_network(state_batch_t)
        predictions = torch.gather(input=q_vals, dim=1, index=action_batch_t)

        # find loss and optimize
        loss = self.criterion(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_return(self, name=None) -> None:
        """
        Plots return over time
        :return: None
        """
        plt.plot(self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('return')
        if name is not None:
            plt.savefig(f'figures/{name}.png')
        else:
            plt.savefig(f'figures/giwoegfwh.png')
        # plt.show()

    def generate_gif(self, env: discreteaction_pendulum.Pendulum(), name=None) -> None:
        """
        Generates a gif of an example trajectory with a trained agent
        :param env: pendulum environment
        :param name: name to name gif
        :return: None
        """
        policy = self.q_network

        # Simulate an episode and save the result as an animated gif
        if name is not None:
            env.video(policy, filename=f'figures/{name}.gif')
        else:
            env.video(policy, filename='figures/bfwbgjgrwerw.gif')

    def plot_policy(self, name=None) -> None:
        """
        Plots trained policy
        :param name:
        :return:
        """
        pass

    def plot_trajectory(self, name=None) -> None:
        """
        Plots trajectory
        :param name:
        :return:
        """
        s = self.env.reset()
        # print(f's = {torch.Tensor(s)}')

        # Create dict to store data from simulation
        data = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }

        # Simulate until episode is done
        done = False
        while not done:
            a = self.get_action(s)
            # print(f'a = {int(a)}')
            (s, r, done) = self.env.step(a)
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)

        # Parse data from simulation
        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        tau = [self.env._a_to_u(a) for a in data['a']]

        # Plot data and save to png file
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(data['t'], theta, label='theta')
        ax[0].plot(data['t'], thetadot, label='thetadot')
        ax[0].legend()
        ax[1].plot(data['t'][:-1], tau, label='tau')
        ax[1].legend()
        ax[2].plot(data['t'][:-1], data['r'], label='r')
        ax[2].legend()
        ax[2].set_xlabel('time step')
        plt.tight_layout()
        if name is not None:
            plt.savefig(f'figures/{name}.png')

    def plot_fcn(self, pol=False, name=None) -> None:
        """
        Plots either the policy or value
        :param pol:
        :param name:
        :return:
        """
        dim = 500
        theta = np.linspace(-np.pi, np.pi, dim)
        thetadot = np.linspace(-self.env.max_thetadot, self.env.max_thetadot, dim)
        theta_meshgrid, thetadot_meshgrid = np.meshgrid(theta, thetadot)
        policy = np.zeros((dim, dim))
        value_fcn = policy.copy()
        for i in range(dim):
            for j in range(dim):
                s = np.array([theta_meshgrid[i, j], thetadot_meshgrid[i, j]])
                predict = self.q_network(torch.FloatTensor(s)).detach().numpy()
                policy[i, j] = self.env._a_to_u(np.argmax(predict))
                value_fcn[i, j] = np.max(predict)

        fig, (ax) = plt.subplots(1, 1, figsize=(8, 8))
        if pol:
            c = ax.contourf(theta_meshgrid, thetadot_meshgrid, policy)
        else:
            c = ax.contourf(theta_meshgrid, thetadot_meshgrid, value_fcn)
        fig.colorbar(c)
        if pol:
            ax.set_title('Policy', size=10)
        else:
            ax.set_title('Value Function', size=10)
        ax.set_xlabel('Theta', size=8)
        ax.set_ylabel('ThetaDot', size=8)
        ax.grid()
        ax.legend()
        if name is not None:
            fig.savefig(f'figures/{name}.png')


class ReplayMemory(object):
    """
    Replay buffer to store data for q learning
    """

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
