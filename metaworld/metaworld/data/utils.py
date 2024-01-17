# Utiulity fucntions for MW
import time

import numpy as np
import gym
import metaworld  # modified mw env with additional gym wrappers
from metaworld.data.dataset import read_trajs
import yaml, os, copy

def download(data_path):  # need azcopy
    import subprocess
    remote_path_root = 'https://rlnexusstorage2.blob.core.windows.net/00-share-data-public/'
    remote_path = os.path.join(remote_path_root, data_path)
    sub_data_path = data_path.split('metaworld/')[1]
    local_path = os.path.join(os.path.expanduser('~'), '.metaworld_data', sub_data_path)
    local_dir = os.path.split(local_path)[0]
    os.makedirs(local_dir, exist_ok=True)
    if not os.path.isfile(local_path):
        subprocess.run(["azcopy", "copy", remote_path, local_dir])
    return local_path


def normalize_reward(reward):
    return (reward-10.0)/10.0


def get_mw_env(task_name,
                cam_height,
                cam_width,
                cam_name,
                goal_cost_reward=False,
                obs_types='states',  #  a tuple of sensor types that may contain 'states', 'images', and 'proprios', e.g. ('images', 'proprios')
                fix_task_sequence=False,
                steps_at_goal=5,
                stop_at_goal=True,
                train_distrib=False,
                use_normalized_reward=True
                ):
    env = metaworld.mw_gym_make(task_name,
                                stop_at_goal=stop_at_goal,
                                steps_at_goal=steps_at_goal,
                                train_distrib=train_distrib,
                                goal_cost_reward=goal_cost_reward,
                                cam_height=cam_height,
                                cam_width=cam_width,
                                depth=False,
                                fix_task_sequence=fix_task_sequence,
                                cam_name=cam_name
                                )

    if use_normalized_reward:
        class RewardNormalizer(gym.RewardWrapper):
            def reward(self, reward):
                return normalize_reward(reward)
        env = RewardNormalizer(env)
    # Wrap the observation
    # If there is one observation type, we return the observation directly.
    # Otherwise, we return a dict.
    if len(obs_types)==1:
        if obs_types[0]=='states':
            obs_type = 'full_states'
        else:
            obs_type = obs_types[0]
        class ObservationWrapper(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                obs = env.reset()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs[obs_type[:-1]] .shape)
            def observation(self, obs):
                return obs[obs_type[:-1]]  # removing `s` at the end
    else: # hybrid, return a dict
        class ObservationWrapper(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)

                self._obs_set = copy.deepcopy(obs_types)
                self._obs_set = set(self._obs_set)
                if 'states' in self._obs_set:
                    self._obs_set.remove('states')
                    self._obs_set.add('full_states')

                obs = env.reset()
                self.observation_space = {k[:-1]: gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs[k[:-1]].shape) \
                                            for k in self._obs_set} # removing `s` at the end
                if 'full_state' in self.observation_space:
                    self.observation_space['state'] = self.observation_space['full_state']
                    del self.observation_space['full_state']

            def observation(self, obs):
                obs = {k[:-1]: obs[k[:-1]] for k in self._obs_set} # removing `s` at the end
                if 'full_state' in obs:
                    obs['state'] = obs['full_state']
                    del obs['full_state']
                return  obs
    return ObservationWrapper(env)


def get_mw_env_and_data(dataset_name,
                        obs_type='states',  #  'states', 'images', or a tuple of sensor types
                        reward_type='original',
                        dataset_dir=None,
                        env_only=False,
                        res=128,
                        fix_task_sequence=False
                        ):

    # We modify the MW env to stop at the goal and shift the reward by -10, so
    # the reward is all non-positive and the reward at the goal is zero.
    MAX_steps_at_goal = 1
    stop_at_goal = True
    train_distrib = False
    obs_set = (obs_type,) if type(obs_type) is str else obs_type

    # Parse dataset_name
    env_name, noise_level = dataset_name.split('-noise')

    cam_name = 'corner' if 'images' in obs_set else None  # turn off rendering if no images are needed.

    # Create gym env
    res = (res, res)
    env = get_mw_env(task_name=env_name,
                        cam_height=res[0],
                        cam_width=res[1],
                        cam_name=cam_name,
                        goal_cost_reward=False,
                        obs_types=obs_set,
                        fix_task_sequence=fix_task_sequence,
                        steps_at_goal=MAX_steps_at_goal,
                        stop_at_goal=stop_at_goal,
                        train_distrib=train_distrib,
                        use_normalized_reward=True
                        )
    if env_only:
        return env

    # Downdload data
    dataset_paths_dict = yaml.safe_load(open(os.path.join(os.path.dirname(__file__),'data_paths.yaml'), 'r'))
    partial_data_path = dataset_paths_dict[dataset_name]
    if dataset_dir is None:
        dataset_path = download(partial_data_path)
    else:
        sub_data_path = partial_data_path.split('metaworld/')[1]
        dataset_path = os.path.join(dataset_dir, sub_data_path)

    # Get dataset
    data = get_raw_qlearning_dataset(dataset_path, reward_type, obs_set)
    data['rewards'] = normalize_reward(data['rewards'])
    return env, data


def get_raw_qlearning_dataset(dataset_path, reward_type, obs_set=('states')):

    obs_set = set(obs_set)

    env_metadata, all_trajs = read_trajs(dataset_path, reward_type)
    print("Concatenating all trajectories into one big dataset...")
    start_extend = time.time()

    # Add timeouts info.
    for traj in all_trajs:
        traj_len = len(traj['rewards'])
        traj['timeouts'] = np.full((traj_len,1), False, dtype=bool)
        traj['terminals'][traj['successes'],0]=True  # done is equal to success.
        if not traj['successes'][-1]: # If not success, it is stopped due to timeout.
            traj['timeouts'][-2:] = True

    # Ignore the last time step
    result = {
        'actions': np.vstack([traj['actions'][:-1] for traj in all_trajs]),
        'rewards': np.vstack([traj['rewards'][:-1] for traj in all_trajs]).flatten(),
        'terminals': np.vstack([traj['terminals'][:-1] for traj in all_trajs]).flatten(),
        'timeouts': np.vstack([traj['timeouts'][:-1] for traj in all_trajs]).flatten(),
    }
    result['observations'] = {}
    result['next_observations'] = {}
    if 'states' in obs_set:  # aka full_states
        result['observations']['state'] = np.vstack([traj['states'][:-1] for traj in all_trajs])
        result['next_observations']['state'] = np.vstack([traj['states'][1:] for traj in all_trajs])

    if 'proprio_states' in obs_set:
        result['observations']['proprio_state'] = np.vstack([traj['proprio_states'][:-1] for traj in all_trajs])
        result['next_observations']['proprio_state'] = np.vstack([traj['proprio_states'][1:] for traj in all_trajs])

    if 'images' in obs_set:
        result['observations']['image'] = np.vstack([traj['observations'][:-1] for traj in all_trajs])
        result['next_observations']['image']=  np.vstack([traj['observations'][1:] for traj in all_trajs])

    if 'depths' in obs_set:
        result['observations']['depth'] = np.vstack([traj['depths'][:-1] for traj in all_trajs]) if env_metadata['has_depth'] else None
        result['next_observations']['depth'] = np.vstack([traj['depths'][1:] for traj in all_trajs]) if env_metadata['has_depth'] else None

    # Flatten the dict if there is only one observation type.
    if len(obs_set)==1:
        result['observations'] = list(result['observations'].values())[0]
        result['next_observations'] = list(result['next_observations'].values())[0]

    end_extend = time.time()
    print(f'Total time concatenating the dataset: {end_extend - start_extend}s.')

    return result
