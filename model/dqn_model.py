import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x = x.float() / 256
        output_conv = self.conv(x).view(x.size()[0], -1)
        output = self.fc(output_conv)
        return output


Transition = collections.namedtuple('Transition', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # in order to acclerate calculate
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        state = self.state
        if np.random.random() < epsilon:
            # take a random action
            action = self.env.action_space.sample()
        else:
            # get a max value aciton from the q-table
            state_vector = torch.tensor(np.array([state], copy=False)).to(device)
            qvals_vector = net(state_vector)
            _, act_v = torch.max(qvals_vector, dim=1)
            action = int(act_v.item())

        # get transition from the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # add transitions into replay buffer for later sample
        trans = Transition(state, action, reward, is_done, new_state)
        self.replay_buffer.append(trans)

        # update state
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
