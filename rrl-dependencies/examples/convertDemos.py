# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import numpy as np
import time as timer
import torch
import pickle
import mj_envs
from PIL import Image
import rrl
from rrl.utils import make_env, make_dir
import click
from tqdm import tqdm

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0'}
seed = 123
DEBUG = False

@click.command(help="To convert create new demonstrations with new observations.")
@click.option('--env_name', type=str, help='environment to load', required=True)
@click.option('--encoder_type', type=str, help='Type of encoder.', default="resnet34", required=False)
@click.option('-d', '--demo', type=str, help='Location to the demos', required=True)
@click.option('-c', '--cam', type=str, help='List of cameras', required=True, multiple=True)
@click.option('--hybrid_state', type=bool, help='Attach state with observations', default=True, required=False)
@click.option('--suffix', type=str, help='Attach state with observations', default="", required=False)

def main(env_name, encoder_type, demo, cam, hybrid_state, suffix):
    demo_file = demo
    print("Camera List : ", cam)
    e, _ = make_env(env_name, cam_list=cam, from_pixels=True, encoder_type=encoder_type, hybrid_state=hybrid_state)

    num_cam = len(cam)
    camera_type = cam[0]
    if num_cam > 1:
        camera_type = "multicam"
    enc_suffix = encoder_type

    e.reset()
    demo_paths = pickle.load(open(demo_file, 'rb'))
    keys = [
            "actions",
            "init_state_dict",
            "observations",
            "rewards"
            ]
    time_start = timer.time()
    time_prev = time_start
    new_demo_paths = []
    try :
        e.set_seed(demo_paths[0]['seed'])
    except :
        print("++++++++++++++++++++++++++++ Couldn't find the seed of the demos. Please verify.")
        pass
    print("Number of demo_path : ", len(demo_paths))
    for path in tqdm(demo_paths, desc="Converting demonstrations"):
        obs = e.reset()
        if torch.is_tensor(obs):
            obs = obs.data.cpu().numpy()

        if env_name in _mj_envs :
            e.set_env_state(path["init_state_dict"])
        else :
            print("Please enter valid environment.")

        idx = 0
        new_path = {}
        new_path_obs = obs
        ep_reward = 0
        for action in path["actions"] :
            next_obs, reward, done, goal_achieved = e.step(action)
            if torch.is_tensor(next_obs):
                next_obs = next_obs.data.cpu().numpy()
            ep_reward += reward
            new_path_obs = np.vstack((new_path_obs, next_obs))
            obs = next_obs
            idx += 1

        new_path_obs = new_path_obs[:-1]
        assert new_path_obs.shape[0] == path["observations"].shape[0]
        new_path['observations'] =  new_path_obs
        new_path['actions'] =  path['actions']
        new_path['rewards'] =  path['rewards']
        new_path['init_state_dict'] =  path['init_state_dict']
        new_demo_paths.append(new_path)
        assert type(path["observations"]) == type(new_path["observations"])
        if DEBUG :
            print("Episode Reward : ", ep_reward) # Episode reward need not be equal to original reward due to different versions
            print("Original rewards : ", np.sum(new_path['rewards']))
            print(new_path['observations'].shape)
            print("Time taken for one demo path : ", timer.time() - time_prev)
        time_prev = timer.time()
    assert len(new_demo_paths) == len(demo_paths)
    print("+++++++ Total Time : ", timer.time() - time_start)
    camera_type = camera_type + '_' + enc_suffix
    final_dir = list(set(rrl.__path__))[0] + '/demonstrations/' + camera_type

    make_dir(final_dir)
    final_path = final_dir + '/' + env_name + suffix  + '_demos.pickle'
    pickle.dump(new_demo_paths, open(final_path, 'wb'))
    print("Final Path : ", final_path)

if __name__ == '__main__':
    main()
