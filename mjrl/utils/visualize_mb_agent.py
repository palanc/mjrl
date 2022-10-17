import gym
import mj_envs
import mjrl.envs
import click 
import os
import gym
import numpy as np
import pickle
import mjrl.samplers.core as sampler
from mjrl.utils.gym_env import GymEnv


DESC = '''
Helper script to visualize model based agent (in mjrl format).\n
USAGE:\n
    Visualizes learned model rollouts by setting environment to sequences of predicted states \n
    $ python utils/visualize_mb_agent --env_name mjrl_swimmer-v0 --policy my_agent.pickle --mode evaluation --episodes 10 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--agent', type=str, help='absolute path of the agent file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=10)

def main(env_name, agent, mode, seed, episodes):
    e = GymEnv(env_name)
    e.set_seed(seed)

    assert agent is not None

    mb_agent = pickle.load(open(agent, 'rb'))

    paths = sampler.sample_data_batch(episodes*mb_agent.env.horizon, 
                                      e, 
                                      mb_agent.policy, 
                                      eval_mode=False, 
                                      base_seed=seed, 
                                      num_cpu=1)

    for p in paths:
        e.reset()
        T = p['observations'].shape[0]
        observations = np.expand_dims(p['observations'],axis=0)
        observations = e.env.env.obsvec2obsdict(observations)
        observations = e.env.env.squeeze_dims(observations)
        qpos, qvel, desired_orien = e.env.env.obs2state(observations)
        for t in range(T):
            e.env.env.set_env_state({'qpos': qpos[t], 'qvel': qvel[t], 'desired_orien': desired_orien[t]})
            e.render()


if __name__ == '__main__':
    main()

