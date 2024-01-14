import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import gym

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from dreamer.envs.wrappers import *
from dreamer.algorithms.dreamer import Dreamer
from dreamer.utils.utils import load_config


def main(args):
    config = load_config(args.config)

    env = gym.make(config.environment.task_name)
    env = gym.wrappers.ResizeObservation(
        env, (config.environment.height, config.environment.width)
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ChannelFirstEnv(env)

    env = SkipFrame(env, 1)
    env = PixelNormalization(env)

    env.seed(config.environment.seed)

    observation_shape = env.observation_space.shape
    action_size = env.action_space.n

    device = config.operation.device

    agent = Dreamer(observation_shape, action_size, device, config)

    if args.evaluate:
        agent.load()
        agent.play(env)
    else:
        agent.train(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="dmc-walker-walk.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="Evaluate the model with trained weights",
    )
    args = parser.parse_args()
    main(args)
