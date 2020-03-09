import gym
import time
import collections
import torch
import numpy as np


from model import dqn_model as model
from common import args, atari_wrappers, const

FPS = 25

if __name__ == "__main__":
    # set args and params
    params = const.HYPERPARAMS['default']
    args = args.get_play_arg(params)

    env = atari_wrappers.make_atari(params.env, skip_noop=True)
    env = atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)

    # record a mp4 for play
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)

    net = model.DQN(env.observation_space.shape, env.action_space.n)
    # map the loaded tensor location from GPU to CPU
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
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