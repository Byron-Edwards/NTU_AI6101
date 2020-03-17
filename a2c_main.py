#!/usr/bin/env python3
import gym
import numpy as np
import argparse

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from model.a2c_model import AtariA2C
from common.Tracker import RewardTracker, TBMeanTracker
from common.atari_wrappers import wrap_a2c
from common.agent import PolicyAgent,ActorCriticAgent
from common.experience import ExperienceSourceFirstLast, unpack_batch

GAME_LIST = {
    "pong": "PongNoFrameskip-v4",
    "boxing": "BoxingNoFrameskip-v4",
    "breakout": "BreakoutNoFrameskip-v4",
    "pacman": "MsPacmanNoFrameskip-v4"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--name", type= str, required=True, help="Name of the run")
    parser.add_argument("--game", default="pong", help="atari game name")

    parser.add_argument("--batchsize", default=32, type=int, help="batch size")
    parser.add_argument("--envsize", default=8, type=int, help="number of environment used synonymously")
    parser.add_argument("--rewardstep", default=4, type=int, help="reward step")
    parser.add_argument("--stopreward", default=20, type=float, help="Name of the run")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--clipgrad", default=0.01, type=float, help="clip of gradient")
    parser.add_argument("--lr", default=0.001, type=float, help="SDG learning rate")
    parser.add_argument("--entropy", default=0.01, type=float, help="the entropy factor add to the gradient loss")

    args = parser.parse_args()

    if torch.cuda.is_available() & ~args.cuda:
        Warning("The cuda device is available while running the script without cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = [wrap_a2c(gym.make(GAME_LIST[args.game]))
            for _ in range(args.envsize)]
    writer = SummaryWriter(comment="-{}-a2c_".format(args.game) + args.name)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ExperienceSourceFirstLast(envs, agent, gamma=args.gamma, steps_count=args.rewardstep)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=1e-3)

    batch = []

    with RewardTracker(writer, args.stopreward, net, args.game) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < args.batchsize:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, args.gamma, args.rewardstep, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(args.batchsize), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = args.entropy * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), args.clipgrad)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)
