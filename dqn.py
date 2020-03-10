import warnings
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from common import args, atari_wrappers, const, wrappers
from model import dqn_model as model


def calc_loss(batch, policy_net, target_net, gamma, is_double=False, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    # transfomr np arrary to tensor
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # get values of acitons in state_v
    state_action_values = policy_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = target_net(next_states_v).max(1)[0]
        # done mask
        next_state_values[done_mask] = 0.0
        if is_double:
            next_state_acts = policy_net(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_vals = target_net(next_states_v).gather(1, next_state_acts).squeeze(-1).detach()
        else:
            # not influence the net
            next_state_vals = next_state_values.detach()

    # calculate expect aciton values (Q-table)
    expected_state_action_values = next_state_vals * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    # get rid of warnings to have a cleaner terminal logger
    warnings.simplefilter("ignore", category=UserWarning)

    # set args and params
    params = const.HYPERPARAMS['default']
    args = args.get_arg(params)
    torch.manual_seed(args.seed)

    # init tensorboard
    writer = SummaryWriter(comment="-" + params.env + "-" + args.name)

    device = torch.device("cuda" if args.cuda else "cpu")

    # wappers of env
    # env = atari_wrappers.make_atari(params.env, skip_noop=True)
    # env = atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)
    env = wrappers.make_env(params.env)

    # init policyNet and targetNet
    policy_net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    # init agent and replayBuffer
    buffer = model.ReplayBuffer(params.replay_size)
    agent = model.Agent(env, buffer)

    # training optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    total_rewards = []
    frame = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    # training loop
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
            print("frame-{frame}: finish {game_num} games, reward {reward:.3f}, speed {speed:.2f} f/s".format(
                frame=frame, game_num=len(total_rewards), reward=m_reward, speed=speed
            ))

            # update tensorboard
            writer.add_scalar("reward/iteration", reward, frame)
            writer.add_scalar("reward/avg_100", m_reward, frame)
            writer.add_scalar("indicator/speed", speed, frame)
            writer.add_scalar("indicator/epsilon", epsilon, frame)

            # save best model every 100 frame and chech whether the training is done
            # each game have 1784 frame now
            every_save_epoch = len(total_rewards) % 2
            if every_save_epoch is 0:
                if best_m_reward is None:
                    best_m_reward = m_reward
                elif best_m_reward < m_reward and m_reward > 0:
                    torch.save(policy_net.state_dict(),
                               "{env}-{name}-best_{reward:.0f}.dat".format(env=params.env, name=args.name,
                                                                           reward=m_reward))
                    best_m_reward = m_reward

                if m_reward > params.stop_reward:
                    print("Solved in %d frames!" % frame)
                    break

        # procduce the first batches of transition from scratch
        # apply by agent.play
        if len(buffer) < params.replay_start:
            continue

        # sync target_net
        if frame % params.target_net_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

        optimizer.zero_grad()

        # get a sample batch
        batch = buffer.sample(args.batch_size)
        loss = calc_loss(batch, policy_net, target_net, gamma=params.gamma, is_double=args.double, device=device)
        writer.add_scalar("loss/batch", loss / args.batch_size, frame)
        loss.backward()
        optimizer.step()
