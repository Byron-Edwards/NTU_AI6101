import gym
import time
import collections
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from model.a2c_model import AtariA2C
from common.atari_wrappers import wrap_a2c
from common.agent import ArgmaxActionSelector

FPS = 25
GAME_LIST = {
    "pong": "PongNoFrameskip-v4",
    "boxing": "BoxingNoFrameskip-v4",
    "breakout": "BreakoutNoFrameskip-v4",
    "pacman": "MsPacmanNoFrameskip-v4"
}

if __name__ == "__main__":
    # set args and params
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default="boxing", type= str, help="Environment name to use, default=")
    parser.add_argument("-r", "--record", default= "record",help="Directory for video")
    parser.add_argument("--no-vis", default=False, dest='vis', help="Disable visualization", action='store_false')
    args = parser.parse_args()

    if torch.cuda.is_available() & ~args.cuda:
        Warning("The cuda device is available while running the script without cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrap_a2c(gym.make(GAME_LIST[args.env]))

    # record a mp4 for play
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)

    net = AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    # map the loaded tensor location from GPU to CPU
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)
    selector = ArgmaxActionSelector()
    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        probs_v, values_v = net(state_v)
        probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        action = selector(probs).item()
        print(action)
        c[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.vis:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()