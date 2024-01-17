import h5py
import json
import numpy as np
import time
from flatten_dict import flatten
import cv2
import os
#from datetime import datetime

MAX_MW_REWARD = 10
GOAL_REWARD = 30

SUBGOAL_REWARD_COEFFICIENTS = {
    'assembly-v2' : [1, 5, 10, GOAL_REWARD],
    'button-press-v2' : [1, 5, 10, GOAL_REWARD],
    'basketball-v2' : [GOAL_REWARD],
    'bin-picking-v2' : [GOAL_REWARD],
    'box-close-v2' : [GOAL_REWARD],
    'button-press-topdown-v2' : [GOAL_REWARD],
    'button-press-topdown-wall-v2' : [GOAL_REWARD],
    'button-press-wall-v2' : [GOAL_REWARD],
    'coffee-button-v2': [GOAL_REWARD],
    'coffee-pull-v2' : [GOAL_REWARD],
    'coffee-push-v2' : [GOAL_REWARD],
    'dial-turn-v2' : [GOAL_REWARD],
    'disassemble-v2' : [GOAL_REWARD],
    'door-close-v2' : [GOAL_REWARD],
    'door-lock-v2' : [GOAL_REWARD],
    'door-open-v2' : [GOAL_REWARD],
    'door-unlock-v2' : [GOAL_REWARD],
    'hand-insert-v2' : [GOAL_REWARD],
    'drawer-close-v2' : [GOAL_REWARD],
    'drawer-open-v2' : [GOAL_REWARD],
    'faucet-open-v2' : [GOAL_REWARD],
    'faucet-close-v2' : [GOAL_REWARD],
    'hammer-v2' : [GOAL_REWARD],
    'handle-press-side-v2' : [GOAL_REWARD],
    'handle-press-v2' : [GOAL_REWARD],
    'handle-pull-side-v2' : [GOAL_REWARD],
    'handle-pull-v2' : [GOAL_REWARD],
    'lever-pull-v2' : [GOAL_REWARD],
    'peg-insert-side-v2' : [GOAL_REWARD],
    'pick-place-wall-v2' : [GOAL_REWARD],
    'pick-out-of-hole-v2' : [GOAL_REWARD],
    'reach-v2' : [GOAL_REWARD],
    'push-back-v2' : [GOAL_REWARD],
    'push-v2' : [GOAL_REWARD],
    'pick-place-v2' : [GOAL_REWARD],
    'plate-slide-v2' : [GOAL_REWARD],
    'plate-slide-side-v2' : [GOAL_REWARD],
    'plate-slide-back-v2' : [GOAL_REWARD],
    'plate-slide-back-side-v2' : [GOAL_REWARD],
    'peg-unplug-side-v2' : [GOAL_REWARD],
    'soccer-v2' : [GOAL_REWARD],
    'stick-push-v2' : [GOAL_REWARD],
    'stick-pull-v2' : [GOAL_REWARD],
    'push-wall-v2' : [GOAL_REWARD],
    'reach-wall-v2' : [GOAL_REWARD],
    'shelf-place-v2' : [GOAL_REWARD],
    'sweep-into-v2' : [GOAL_REWARD],
    'sweep-v2' : [GOAL_REWARD],
    'window-open-v2' : [GOAL_REWARD],
    'window-close-v2' : [GOAL_REWARD]
}


SUBGOAL_BREAKDOWN = {
    'assembly-v2' : ['grasp_success', 'lift_success', 'align_success'],
    'button-press-v2' : ['nearby_success', 'near_button_success', 'button_pressed_success'],
    'basketball-v2' : [],
    'bin-picking-v2' : [],
    'box-close-v2' : [],
    'button-press-topdown-v2' : [],
    'button-press-topdown-wall-v2' : [],
    'button-press-wall-v2' : [],
    'coffee-button-v2': [],
    'coffee-pull-v2' : [],
    'coffee-push-v2' : [],
    'dial-turn-v2' : [],
    'disassemble-v2' : [],
    'door-close-v2' : [],
    'door-lock-v2' : [],
    'door-open-v2' : [],
    'door-unlock-v2' : [],
    'hand-insert-v2' : [],
    'drawer-close-v2' : [],
    'drawer-open-v2' : [],
    'faucet-open-v2' : [],
    'faucet-close-v2' : [],
    'hammer-v2' : [],
    'handle-press-side-v2' : [],
    'handle-press-v2' : [],
    'handle-pull-side-v2' : [],
    'handle-pull-v2' : [],
    'lever-pull-v2' : [],
    'peg-insert-side-v2' : [],
    'pick-place-wall-v2' : [],
    'pick-out-of-hole-v2' : [],
    'reach-v2' : [],
    'push-back-v2' : [],
    'push-v2' : [],
    'pick-place-v2' : [],
    'plate-slide-v2' : [],
    'plate-slide-side-v2' : [],
    'plate-slide-back-v2' : [],
    'plate-slide-back-side-v2' : [],
    'peg-unplug-side-v2' : [],
    'soccer-v2' : [],
    'stick-push-v2' : [],
    'stick-pull-v2' : [],
    'push-wall-v2' : [],
    'reach-wall-v2' : [],
    'shelf-place-v2' : [],
    'sweep-into-v2' : [],
    'sweep-v2' : [],
    'window-open-v2' : [],
    'window-close-v2' : []
}


DTYPES = {
    'full_states': np.float64,
    'proprio_states': np.float64,
    'observations': np.uint8,
    'depths': np.uint16,
    'actions': np.float64,
    'terminals': np.bool_,
    'rewards': np.float64,
    'infos': np.bool_
}


def target_type(data_name):
    mod_data_name = data_name

    if data_name not in DTYPES and data_name.startswith('infos'):
        mod_data_name = 'infos'

    return DTYPES[mod_data_name]


def verify_type(data_name, dtype):
    mod_data_name = data_name

    if data_name not in DTYPES and data_name.startswith('infos'):
        mod_data_name = 'infos'

    assert DTYPES[mod_data_name] == dtype, f'{data_name}\'s np.array data type is {dtype}, but should be {DTYPES[mod_data_name]}'


def check_action(a, act_lim):
        assert (-act_lim <= a).all() and (a <= act_lim).all(), f'Action {a} has entries outside the [{-act_lim}, {act_lim}] range.'


class MWDatasetWriter:
    def __init__(self, data_dir_path, data_file_name, env, task_name, res, camera, write_depth, act_tolerance, success_steps_for_termination, write_data=True):
        # The number of steps with with info/success = True required to trigger episode termination
        self.task_name = task_name
        self.success_steps_for_termination = success_steps_for_termination
        raw_metadata = {
            'task_name' : task_name,
            'horizon' : env.max_path_length,
            'fps': env.metadata["video.frames_per_second"],
            'frame_skip' : env.frame_skip,
            'img_height' : res[0],
            'img_width' : res[1],
            #'img_format' : 'cwh',
            'camera' : camera,
            'has_depth': write_depth,
            'act_tolerance' : act_tolerance,
            'subgoal_breakdown' : SUBGOAL_BREAKDOWN[task_name],
            'success_steps_for_termination' : success_steps_for_termination
        }

        self.write_data = write_data
        self.write_depth = write_depth
        self._act_lim = 1 - act_tolerance

        if self.write_data:
            if not os.path.exists(data_dir_path):
                os.makedirs(data_dir_path)
            data_file_path = os.path.join(data_dir_path, data_file_name)
            self._datafile = h5py.File(data_file_path, 'w')
            self._datafile.attrs["env_metadata"] = json.dumps(raw_metadata, indent=4) # environment info

        self.data = self._reset_data()
        self._num_episodes = 0


    def _reset_data(self):
        data = {
            'full_states': [],
            'proprio_states': [],
            'observations': [],
            'depths': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': []
            }

        for subgoal in SUBGOAL_BREAKDOWN[self.task_name]:
            data['infos/' + subgoal] = []

        return data


    def append_data(self, f, p, o, d, a, r, done, info):
        self.data['full_states'].append(f)
        self.data['proprio_states'].append(p)
        self.data['observations'].append(o)
        if self.write_depth:
            self.data['depths'].append(d)
        check_action(a, self._act_lim)
        self.data['actions'].append(a)
        self.data['rewards'].append(r)
        self.data['terminals'].append(done)
        self.data['infos/goal'].append(info['success'])

        for subgoal in SUBGOAL_BREAKDOWN[self.task_name]:
            self.data['infos/' + subgoal].append(info[subgoal])


    def write_trajectory(self, max_size=None, compression='gzip'):
        if self.write_data:
            np_data = {}
            for k in self.data:
                data = np.array(self.data[k], dtype=target_type(k))

                if max_size is not None:
                    data = data[:max_size]
                np_data[k] = data

            trajectory = self._datafile.create_group('traj_' + str(self._num_episodes))

            for k in np_data:
                trajectory.create_dataset(k, data=np_data[k], compression=compression)

            self._num_episodes += 1

        self.data = self._reset_data()


    def close(self):
        if self.write_data:
            self._datafile.attrs["total"] = self._num_episodes
            self._datafile.close()


class MWVideoWriter:
    def __init__(self, dest_path_root, file_name, fps, res, write_video=True):
        self.write_video = write_video
        if self.write_video:
            if not os.path.exists(dest_path_root):
                os.makedirs(dest_path_root)
            self.writer = cv2.VideoWriter(
                os.path.join(dest_path_root, f'{file_name}.avi'),
                cv2.VideoWriter_fourcc('M','J','P','G'),
                fps,
                res
            )

    def write(self, frame):
        if self.write_video:
            self.writer.write(frame)


def read_trajs(dataset_path, reward_type):
    data = h5py.File(dataset_path, "r")
    env_metadata = json.loads(data.attrs["env_metadata"])
    act_lim = 1 - env_metadata['act_tolerance']
    if 'has_depth' not in env_metadata:
        env_metadata['has_depth'] = False

    # Retrieve the subgoal info for the task whose data was loaded
    subgoals = ['infos/' + key for key in (SUBGOAL_BREAKDOWN.get(env_metadata["task_name"], []) + ['goal'])]
    if reward_type=='subgoal':
        subgoal_coeffs = np.asarray(SUBGOAL_REWARD_COEFFICIENTS.get(env_metadata["task_name"], []))
        assert len(subgoals) == len(subgoal_coeffs), "The number of subgoals, including the goal, and subgoal coefficients must be the same"
    elif reward_type=='sparse':
        subgoal_coeffs_shaped = np.asarray(SUBGOAL_REWARD_COEFFICIENTS.get(env_metadata["task_name"], []))
        subgoal_coeffs = np.zeros_like(subgoal_coeffs_shaped, dtype=np.float32)
        subgoal_coeffs[-1] = subgoal_coeffs_shaped.max()
    elif reward_type=='shaped' or reward_type=='original' or reward_type=='goal-cost':
        pass
    else:
        raise NotImplementedError

    all_trajs = []

    # We are going to concatenate all trajectories for this task, D4RL-style
    # for traj in data.keys():
    import tqdm
    for traj in tqdm.tqdm(data.keys()):

        start_total = time.time()
        # print(f'Processing trajectory {traj}')
        dataset = flatten(data[traj], reducer='path')
        N = dataset['rewards'].shape[0]

        start_tv = time.time()
        if N > 0:
            for k in dataset:
                if k != "depths" or env_metadata['has_depth']:
                    verify_type(k, dataset[k][0].dtype)
        end_tv = time.time()

        total_retrieve = 0
        start_ret = time.time()

        full_state = dataset['full_states']
        proprio_state = dataset['proprio_states']
        obs = dataset['observations']
        depths = dataset['depths'] if env_metadata['has_depth'] else None
        action = dataset['actions']
        reward = dataset['rewards']
        done_bool = dataset['terminals']
        info_goal = dataset['infos/goal']

        end_ret = time.time()
        total_retrieve = (end_ret - start_ret)

        reward_adj = np.array(reward)

        # "Original reward" means the reward as it is in the dataset.
        if not reward_type=='original':
            for i in range(N):
                check_action(action[i], act_lim)
                #TODO: decide whether subgoal rewards should always be *summed*. E.g., what if one subgoal implies another?
                if reward_type=='shaped':
                    reward_adj[i] = reward[i] - MAX_MW_REWARD
                    if info_goal[i]:
                        reward_adj[i] = 0
                elif reward_type in ['sparse', 'subgoal']:
                    subgoals_achieved = np.asarray([dataset[subgoal][i] for subgoal in subgoals], dtype=np.float32)
                    reward_adj[i] = np.dot(subgoal_coeffs, subgoals_achieved) - np.max(subgoal_coeffs)
                elif reward_type=='goal-cost':
                    reward_adj[i] = 0 if info_goal[i] else -1
                else:
                    raise NotImplementedError()
        reward_adj = np.expand_dims(reward_adj, 1)
        done_bool = np.expand_dims(done_bool, 1)

        all_trajs.append({
            'states': full_state,
            'proprio_states': proprio_state,
            'observations': obs,
            'depths': depths,
            'actions': action,
            'rewards': reward_adj,
            'successes': info_goal,
            'terminals': done_bool
        })

        end_total = time.time()
        #print(f'Total processing: {end_total - start_total}s. Total type verification: {end_tv - start_tv}s. Total retrieval: {total_retrieve}s.')

    return env_metadata, all_trajs


def qlearning_dataset(dataset_path, reward_type):
    env_metadata, all_trajs = read_trajs(dataset_path, reward_type)
    print("Concatenating all trajectories into one big dataset...")
    start_extend = time.time()
    result = {
        'states': np.vstack([traj['states'][:-1] for traj in all_trajs]),
        'next_states': np.vstack([traj['states'][1:] for traj in all_trajs]),
        'proprio_states': np.vstack([traj['proprio_states'][:-1] for traj in all_trajs]),
        'next_proprio_states': np.vstack([traj['proprio_states'][1:] for traj in all_trajs]),
        'observations': np.vstack([traj['observations'][:-1] for traj in all_trajs]),
        'next_observations': np.vstack([traj['observations'][1:] for traj in all_trajs]),
        'depths': np.vstack([traj['depths'][:-1] for traj in all_trajs]) if env_metadata['has_depth'] else None,
        'next_depths': np.vstack([traj['depths'][1:] for traj in all_trajs]) if env_metadata['has_depth'] else None,
        'actions': np.vstack([traj['actions'][:-1] for traj in all_trajs]),
        'rewards': np.vstack([traj['rewards'][:-1] for traj in all_trajs]),
        'terminals': np.vstack([traj['terminals'][:-1] for traj in all_trajs]),
    }
    end_extend = time.time()
    #print(f'Total time concatenating the dataset: {end_extend - start_extend}s.')

    boundary = -1
    for i in range(len(all_trajs)):
        boundary += (len(all_trajs[i]['terminals']) - 1)
        result['terminals'][boundary] = True

    return result



class MWQLearningDataset:
    def __init__(self, dataset_path):
        self.data = h5py.File(dataset_path, "r")
