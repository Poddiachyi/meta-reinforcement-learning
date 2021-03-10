from absl import app
from absl import flags

import numpy as np
import argparse

import dm_alchemy
from dm_env import specs
from dm_alchemy import symbolic_alchemy

from agents.random_agent import RandomAgent


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'docker_image_name', None,
    'Name of the Docker image that contains the Alchemy environment. '
    'If None, uses the default docker image name.')

flags.DEFINE_integer('seed', 42, 'Environment seed.')
flags.DEFINE_string(
    'level_name',
    'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck',
    'Name of Alchemy task to run.')
flags.DEFINE_boolean('symbolic', True, 'Whether to use symbolic env or not')

def main(_):
    params = {}
    if FLAGS.symbolic:
        params = {'seed': FLAGS.seed, 'level_name': FLAGS.level_name}
        env_generator = symbolic_alchemy.get_symbolic_alchemy_level
    else:
        env_settings = dm_alchemy.EnvironmentSettings(seed=FLAGS.seed, level_name=FLAGS.level_name)
        params = {'name': FLAGS.docker_image_name, 'settings': env_settings}
        env_generator = dm_alchemy.load_from_docker

    with env_generator(**params) as env:

        agent = RandomAgent(env.action_spec())

        timestep = env.reset()
        score = 0
        while not timestep.last():
            action = agent.act(timestep)
            timestep = env.step(action)

            if timestep.reward:
                score += timestep.reward
                print('Total score: {:.2f}, reward: {:.2f}'.format(score, timestep.reward))


if __name__ == '__main__':
    app.run(main)