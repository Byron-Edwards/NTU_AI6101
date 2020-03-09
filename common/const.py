from types import SimpleNamespace

SEED = 9527

HYPERPARAMS = {
    'default': SimpleNamespace(**{
        'env': "BoxingNoFrameskip-v4",
        'name': 'Boxing',
        'batch_size': 32,
        'learning_rate': 0.0001,
        'seed': SEED,
        'stop_reward': 60.0,
        'replay_size': 100000,
        'replay_start': 10000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 5,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'gamma': 0.99,
    }),
    'optimize': SimpleNamespace(),
}
