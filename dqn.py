import collections
import warnings
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from common import args, atari_wrappers, const


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.full_connect = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.convs(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 256
        output_conv = self.convs(x).view(x.size()[0], -1)
        output = self.full_connect(output_conv)
        return output


Transition = collections.namedtuple('Transition', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.experience_source_iter = None
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

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
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            # take a random action
            action = env.action_space.sample()
        else:
            # get a max value aciton from the q-table
            state = np.array([self.state], copy=False)
            state_vector = torch.tensor(state).to(device)
            qvals_vector = net(state_vector)
            _, act_v = torch.max(qvals_vector, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        self.state = new_state

        trans = Transition(self.state, action, reward, is_done, new_state)
        self.replay_buffer.append(trans)

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    # get rid of warnings to have a cleaner terminal logger
    warnings.simplefilter("ignore", category=UserWarning)

    # set args and params
    params = const.HYPERPARAMS['default']
    args = args.get_arg(params)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # to acclerate(by reduce the influence of bigger batch_size)
    env = atari_wrappers.make_atari(params.env, skip_noop=True)
    env = atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)

    # init policyNet and targetNet
    policy_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    # init agent and replayBuffer
    buffer = ReplayBuffer(params.replay_size)
    agent = Agent(env, buffer)

    # training optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    total_rewards = []
    frame = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    while True:
        frame += 1
        epsilon = max(params.epsilon_final, params.epsilon_start - frame / params.epsilon_frames)

        reward = agent.play(policy_net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame - ts_frame) / (time.time() - ts)
            ts_frame = frame
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                      frame, len(total_rewards), m_reward, epsilon,
                      speed
                  ))
        if len(buffer) < params.replay_start:
            continue

        if frame % params.target_net_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss = calc_loss(batch, policy_net, target_net, gamma=params.gamma, device=device)
        loss.backward()
        optimizer.step()
