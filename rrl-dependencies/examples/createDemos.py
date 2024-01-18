# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import mj_envs
import click
import gym
from pathlib import Path
import pickle
home = str(Path.home())
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
import mjrl
from mjrl.policies import *
import numpy as np
import os
import rrl

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0', 'tools-v0'}
_mjrl_envs = {'mjrl_peg_insertion-v0', 'mjrl_reacher_7dof-v0'}
DESC = '''
Helper script to create demos.\n
USAGE:\n
    Create demos on the env\n
    $ \n
'''
seed = 123
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--num_demos', type=int, help='Number of demonstrations', default=25)
@click.option('--mode', type=str, help='Mode : evaluation, exploration', default="exploration")
@click.option('--policy', type=str, help='Location to the policy', required=True)

def main(env_name, num_demos, mode, policy):
	#base_dir = home + "/hand_dapg/dapg/"
	base_dir = list(set(rrl.__path__))[0] + "/"
	print("Policy : ", policy)
	pi = pickle.load(open(policy, 'rb'))
	e = GymEnv(env_name)
	e.set_seed(seed)
	demo_paths = []
	for _ in range(num_demos):
		obs = e.reset()
		if env_name in _mj_envs or env_name in _mjrl_envs :
			init_state_dict = e.get_env_state()
		elif env_name in _robel_envs_dclaw :
			init_state_dict = e.env.get_state()
		elif env_name in _robel_envs_dkitty :
			init_state_dict = e.env.get_state()
		else :
			print("Please enter valid environment. Mentioned : ", env_name)
			exit()

		done = False
		new_path = {}
		new_path_obs = obs
		new_path_actions = np.zeros(())
		new_path_rewards = np.zeros(())
		ep_reward = 0
		step = 0
		while not done:
			action = pi.get_action(obs)[0] if mode == 'exploration' else pi.get_action(obs)[1]['evaluation']
			next_obs, reward, done, info = e.step(action)
			ep_reward += reward
			new_path_obs = np.vstack((new_path_obs, next_obs))
			if step == 0:
				new_path_actions = action
				new_path_rewards = reward
			else:
				new_path_actions = np.vstack((new_path_actions, action))
				new_path_rewards = np.vstack((new_path_rewards, reward))
			obs = next_obs
			step += 1
		print("Episode Reward : ", ep_reward)
		new_path_obs = new_path_obs[:-1]
		new_path['observations'] = new_path_obs
		new_path['actions'] = new_path_actions
		new_path['rewards'] = new_path_rewards
		new_path['init_state_dict'] = init_state_dict
		new_path['seed'] = seed
		#print(init_state_dict)
		demo_paths.append(new_path)
	print(len(demo_paths))
	print("Dumping demos at : ", base_dir + 'demonstrations/' + env_name + '_demos{}.pickle'.format(num_demos))
	pickle.dump(demo_paths, open(base_dir + 'demonstrations/' + env_name + '_demos{}.pickle'.format(num_demos), 'wb'))
if __name__ == "__main__":
	main()
