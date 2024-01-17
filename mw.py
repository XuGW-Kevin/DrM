# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import numpy as np
import gym
from gym.wrappers import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld import mw_gym_make
import dm_env
from dm_env import specs
from typing import Any, NamedTuple
from dm_control.suite.wrappers import action_scale
import mujoco_py

class ExtendedTimeStep(NamedTuple):
    done: Any
    reward: Any
    discount: Any
    observation: Any
    state: Any
    action: Any
    success: Any

    def last(self):
        return self.done

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        obs = self._env.reset()
        return self._augment_time_step(np.array(obs), self.prop_state())

    def step(self, action):
        obs, reward, done, extra = self._env.step(action)
        discount = 1.0
        success=extra['success']
        return self._augment_time_step(np.array(obs),
                                       self.prop_state(),
                                       action,
                                       reward,
                                       success,
                                       discount,
                                       done)
    def prop_state(self):
        state = self._env.state
        #return state
        return np.concatenate((state[:4], state[18 : 18 + 4]))
    
    
    def _augment_time_step(self, obs, state, action=None, reward=None, success=False, discount=1.0, done=False):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
            reward = 0.0
            success = 0.0
            discount = 1.0
            done = False
        return ExtendedTimeStep(observation=obs,
                                state=state,
                                action=action,
                                reward=reward,
                                success=success, 
                                discount=discount,
                                done = done)
    
    def state_spec(self):
        return specs.BoundedArray((8,), np.float32, name='state', minimum=0, maximum=255)
    
    def observation_spec(self):
        return specs.BoundedArray(self._env.observation_space.shape, np.uint8, name='observation', minimum=0, maximum=255)

    def action_spec(self):
        return specs.BoundedArray(self._env.action_space.shape, np.float32, name='action', minimum=-1, maximum=1.0)

    def __getattr__(self, name):
        return getattr(self._env, name)

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, img_size, frame_stack, action_repeat):
        super().__init__(env)
        self.env = env
        self._num_frames = frame_stack
        self._action_repeat = action_repeat
        self._frames = deque([], maxlen=self._num_frames)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._num_frames * 3, img_size, img_size),
            dtype=np.uint8,
        )
        self.action_space = self.env.action_space
        self._res = (img_size, img_size)
        self.img_size = img_size


        
    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        return state

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)
    
    def _get_pixel_obs(self, pixel_obs):
        return pixel_obs[:, :, ::-1].transpose(
            2, 0, 1
        )
    
    def reset(self):
        obs = self.env.reset()
        self._state_obs = obs['full_state']
        pixel_obs = self._get_pixel_obs(obs['image'])
        for _ in range(self._num_frames):
            self._frames.append(pixel_obs)
        return self._stacked_obs()

    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            obs, r, _, info = self.env.step(action)
            reward += r
        self._state_obs = obs['full_state']
        pixel_obs = self._get_pixel_obs(obs['image'])
        self._frames.append(pixel_obs)
        return self._stacked_obs(), reward, False, info

    # def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
    #     return self.env.render(offscreen=False, resolution=(width, height), camera_name=self.camera_name).copy() 

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, train=False, cam_name='corner2', img_size=84, episode_length=200):
    env = mw_gym_make(name+'-v2', cam_height=img_size, cam_width=img_size, cam_name=cam_name, train_distrib=train, seed=seed)
    env = MetaWorldWrapper(env, img_size, frame_stack, action_repeat)
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = ExtendedTimeStepWrapper(env)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    return env

