"""Proposal for a simple, understandable MetaWorld API."""
import abc
import pickle
from collections import OrderedDict
from typing import List, NamedTuple, Type

import metaworld.envs.mujoco.env_dict as _env_dict
import numpy as np
import mujoco_py
from metaworld.data.utils import get_mw_env_and_data

EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


class MetaWorldEnv:
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """
    def set_task(self, task: Task) -> None:
        """Set the task.

        Raises:
            ValueError: If task.env_name is different from the current task.

        """

import gym
class GoalDirected(gym.Wrapper):
    def __init__(self, env, steps_at_goal):
        super().__init__(env)
        self._steps_at_goal = 0
        self._max_steps_at_goal = steps_at_goal

    def _check_accomplished(self, info):
        assert 'success' in info, 'Invalid MetaWorld environment: \'success\' key is missing from the info.'
        if info['success']:
            self._steps_at_goal += 1
            if self._steps_at_goal >= self._max_steps_at_goal:
                info['task_accomplished'] = True
            elif info['success']:
                info['task_accomplished'] = False
        else:
            self._steps_at_goal = 0
            info['task_accomplished'] = False

    def reset(self):
        self._steps_at_goal = 0
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._check_accomplished(info)
        # 'done' may be set by self.env for many reasons. e.g., time horizon arrival. Never unset it.
        if info['task_accomplished']:
            done = True
        return observation, reward, done, info

class MWReset(gym.Wrapper):
    def __init__(self, env, train_tasks, fix_task_sequence=False):
        super().__init__(env)
        self._train_tasks = train_tasks
        self._fix_task_sequence = fix_task_sequence
        self._task_index = -1

    def _config_env(self):
        self.env._partially_observable = False
        self.env._freeze_rand_vec = False
        self.env._set_task_called = True

    def eval_reset(self):
        self._task_index = -1

    def reset(self):
        if self._fix_task_sequence:
            self._task_index = (self._task_index + 1) % len(self._train_tasks)
        else: # randomly select a task
            self._task_index = np.random.randint(len(self._train_tasks))
        task = self._train_tasks[self._task_index]

        self.env.set_task(task)
        # ASSUMPTION: env is a "native" (not wrapped) metaworld environment
        self._config_env()
        return self.env.reset()

class MWImgObs(gym.Wrapper):
    def __init__(self, env, cam_height=84, cam_width=84, depth=True, cam_name='corner', device_id=-1):
        self.env = env
        super().__init__(env)
        self._res = (cam_height, cam_width)
        self._camera_name = cam_name
        self._depth = depth
        if cam_name == 'corner2':
            self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        render_context = mujoco_py.MjRenderContextOffscreen(self.env.sim, device_id=device_id)
        self.env.sim.add_render_context(render_context)

    def _render(self, depth=False):
        img = None
        depth_frame = None
        if self._camera_name is not None:
            vis_obs = self.env.sim.render(*self._res, mode='offscreen', camera_name=self._camera_name, depth=depth, device_id=-1)

            if self._depth:
                img = vis_obs[0][:,:,::-1]
                # Logic for converting from zbuffer depth to depth in meters is taken from http://stackoverflow.com/a/6657284/1461210
                extent = self.env.model.stat.extent
                near_plane = self.env.model.vis.map.znear * extent
                far_plane = self.env.model.vis.map.zfar * extent
                depth_frame = near_plane / (1 - vis_obs[1] * (1 - near_plane / far_plane))
                # Convert depth from meters to millimeters and truncate to max uint16.
                depth_frame = (np.minimum(depth_frame * 1000, np.iinfo('uint16').max)).astype(np.uint)
            else:
                img = vis_obs[:,:,::-1]

        return img, depth_frame

    @staticmethod
    def _construct_proprio(state):
        return np.hstack((state[:4], state[18:22]))

    def reset(self):
        state = self.env.reset()
        proprio_state = MWImgObs._construct_proprio(state)
        img, depth_frame = self._render(depth=self._depth)
        return {'proprio_state' : proprio_state, 'full_state' : state,  'image' : img, 'depth' : depth_frame}

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Flip the offscreen-rendered image -- MuJoCo renders offscreen upside down for some reason.
        img, depth_frame = self._render(depth=self._depth)
        # Generate the proprio state from the 39-D full state. A full state consists of 18-D current and prev states and a 3-D goal spec.
        # We need only the eef pose and closed state (4-D) of the current and prev. states plus the goal info, i.e., a 4+4+3 = 11-D vector.
        proprio_state = MWImgObs._construct_proprio(state)
        obs = {'proprio_state' : proprio_state, 'full_state' : state,  'image' : img, 'depth' : depth_frame}
        return obs, reward, done, info

class MWGoalCostReward(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        assert 'success' in info, 'Invalid MetaWorld environment: \'success\' key is missing from the info.'
        reward = 0 if info['success'] else -1
        return observation, reward, done, info

def wrap_mw_env(env, train_tasks, goal_cost_reward: bool, stop_at_goal: bool = False, steps_at_goal: int = 1, cam_height=84, cam_width=84, depth=True, cam_name=None, fix_task_sequence=False, device_id=-1):
    # Wrap the environment to randomly sample a task variation at reset.
    #
    # WARNING: **THIS MUST BE THE FIRST WRAPPER.**
    env = MWReset(env, train_tasks, fix_task_sequence=fix_task_sequence)
    env = MWImgObs(env, cam_height, cam_width, depth, cam_name, device_id=device_id)
    # Wrap the environment to return done when time limit.
    from gym.wrappers import TimeLimit
    env = TimeLimit(env, env.max_path_length)
    if stop_at_goal:
        env = GoalDirected(env, steps_at_goal)
    if goal_cost_reward:
        env = MWGoalCostReward(env)
    return env

class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """
    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks

    def create_train_env(self, task_name, goal_cost_reward=False, stop_at_goal= False, steps_at_goal=1, cam_height=84, cam_width=84, depth=True, cam_name='corner', fix_task_sequence=False, device_id=-1):
        return wrap_mw_env(self.train_classes[task_name](), self.train_tasks, goal_cost_reward, stop_at_goal, steps_at_goal, cam_height, cam_width, depth, cam_name, fix_task_sequence, device_id)

    def create_test_env(self, task_name, goal_cost_reward=False, stop_at_goal= False, steps_at_goal=1, cam_height=84, cam_width=84, depth=True, cam_name='corner', fix_task_sequence=False, device_id=-1):
        return wrap_mw_env(self.test_classes[task_name](), self.test_tasks, goal_cost_reward, stop_at_goal, steps_at_goal, cam_height, cam_width, depth, cam_name, fix_task_sequence, device_id=device_id)

_ML_OVERRIDE = dict(partially_observable=True)
_MT_OVERRIDE = dict(partially_observable=False)

_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    tasks = []
    for (env_name, args) in args_kwargs.items():
        assert len(args['args']) == 0
        env_cls = classes[env_name]
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args['kwargs'].copy()
        del kwargs['task_id']
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
    if seed is not None:
        np.random.set_state(st0)
    return tasks


def _ml1_env_names():
    tasks = list(_env_dict.ML1_V2['train'])
    assert len(tasks) == 50
    return tasks


class ML1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if not env_name in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        _ML_OVERRIDE,
                                        seed=seed)
        self._test_tasks = _make_tasks(
            self._test_classes, {env_name: args_kwargs},
            _ML_OVERRIDE,
            seed=(seed + 1 if seed is not None else seed))

    @property
    def n_train_tasks(self):
        return len(self._train_tasks)

    @property
    def n_test_tasks(self):
        return len(self._test_tasks)


class MT1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if not env_name in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        _MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = []


class ML10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML10_V2['train']
        self._test_classes = _env_dict.ML10_V2['test']
        train_kwargs = _env_dict.ml10_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _ML_OVERRIDE,
                                        seed=seed)
        test_kwargs = _env_dict.ml10_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes, test_kwargs,
                                       _ML_OVERRIDE,
                                       seed=seed)


class ML45(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML45_V2['train']
        self._test_classes = _env_dict.ML45_V2['test']
        train_kwargs = _env_dict.ml45_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _ML_OVERRIDE,
                                        seed=seed)
        test_kwargs = _env_dict.ml45_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes, test_kwargs,
                                       _ML_OVERRIDE,
                                       seed=seed)


class MT10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT10_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT10_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = []


class MT50(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT50_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = []

__all__ = ["ML1", "MT1", "ML10", "MT10", "ML45", "MT50"]


def mw_gym_make(task_name, goal_cost_reward=False, stop_at_goal=False, steps_at_goal=1, cam_height=84, cam_width=84, depth=True, cam_name='corner', train_distrib=True, fix_task_sequence=False, seed=None, device_id=-1):
    if seed is not None:
        ml1 = ML1(task_name, seed=seed) 
    else:
        ml1 = ML1(task_name) # Construct the benchmark, sampling tasks
    if train_distrib:
        env = ml1.create_train_env(task_name,
                                   goal_cost_reward=goal_cost_reward,
                                   stop_at_goal=stop_at_goal,
                                   steps_at_goal=steps_at_goal,
                                   cam_height=cam_height,
                                   cam_width=cam_width,
                                   depth=depth,
                                   cam_name=cam_name,
                                   fix_task_sequence=fix_task_sequence,
                                   device_id=device_id)
    else:
        env = ml1.create_test_env(task_name,
                                  goal_cost_reward=goal_cost_reward,
                                  stop_at_goal=stop_at_goal,
                                  steps_at_goal=steps_at_goal,
                                  cam_height=cam_height,
                                  cam_width=cam_width,
                                  depth=depth,
                                  cam_name=cam_name,
                                  fix_task_sequence=fix_task_sequence,
                                  device_id=device_id)

    env.get_normalized_score = lambda x : x
    return env
