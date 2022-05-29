import random
from collections import deque
from typing import Tuple

import numpy as np
import torch

from .agent import TradingAgent
from ..models.ddqn import DDQN


class DDQNTradingAgent(TradingAgent):

    def __init__(self, state_dim, action_dim, save_dir,
                 exploration_rate_decay=0.9999980,
                 exploration_rate_min=0.1,
                 gamma=0.9,
                 lr=0.001,
                 learn_every=23):
        """
        Constructor of the trading agent
        :param state_dim: dimension of the input state of the neural net
        :param action_dim: dimension of the action space of the environment
        :param save_dir: the directory to save the neural net parameters
        """
        super().__init__(state_dim, action_dim, save_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DDQN(self.state_dim, self.action_dim, 64, 3).float()
        self.net = self.net.to(self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.curr_step = 0

        self.save_every = 2e5  # no. of experiences between saving Agent Net

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.gamma = gamma

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 2.5e4  # min. experiences before training
        self.learn_every = learn_every  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.
        :param state: LazyFrame - a single observation of the current statem dimension is (state_dim)
        :return: action_idx (int): an integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            state = torch.tensor(state).to(self.device)
            state = state.float().unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action: int, reward: float, done: bool):
        """
        Store the experince to self.memory (replay buffer)
        :param state: LazyFrame
        :param next_state: LazyFrame
        :param action: int
        :param reward: float
        :param done: bool
        """
        state = state.__array__()
        next_state = next_state.__array__()

        state = torch.tensor(state).float().to(self.device)
        next_state = torch.tensor(next_state).float().to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        :return: a batch of experiences sample from the memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        """
        Save the model
        """
        save_path = (
                self.save_dir / f"ddqn_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path, )
        print(f"DDQNNet saved to {save_path} at step {self.curr_step}")

    def learn(self) -> Tuple:
        """
        Update the online model sampling a batch for memory
        :return:
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def load_model(self, model, epsilon):
        self.net = model
        self.exploration_rate = epsilon

    def set_current_steps(self, current_steps: int):
        self.curr_step = current_steps
