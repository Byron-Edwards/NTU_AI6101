import argparse


def get_arg(params):

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-d", "--double", default=False, action="store_true", help="enable double dqn")
    parser.add_argument("-n", "--name", default=params.name, help="training name")
    parser.add_argument("--lr", default=params.learning_rate, help="learning_rate")
    parser.add_argument("--batch_size", default=params.batch_size, help="training batch_size")
    parser.add_argument("--seed", default=params.seed, help="training batch_size")
    parser.add_argument("--model", help="Model file to load")
    args = parser.parse_args()

    return args


def get_play_arg(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=params.env, help="Environment name to use, default=" + params.env)
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=False, dest='vis', help="Disable visualization", action='store_false')
    args = parser.parse_args()

    return args