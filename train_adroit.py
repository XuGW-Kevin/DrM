import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import adroit

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import math
import re

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg, obs_sensor_spec=None):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    if obs_sensor_spec is not None:
        cfg.state_dim = obs_sensor_spec.shape[0]
    else:
        cfg.state_dim = 1
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.task_name, str(cfg.seed)])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            wandb.init(project="DrM",
                       group=group_name,
                       name=exp_name,
                       config=cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._nstep = cfg.nstep
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), 
                                self.cfg.agent,
                                self.train_env.observation_sensor_spec())
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = adroit.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.device)
        self.eval_env = adroit.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.device)
        
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                    self.train_env.observation_sensor_spec(),
                    self.train_env.action_spec(),
                    specs.Array((1, ), np.float32, 'reward'),
                    specs.Array((1, ), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            self._nstep,
            self._discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        n_eval_episode = self.cfg.num_eval_episodes
        eval_until_episode = utils.Until(n_eval_episode)
        total_success = 0.0
        while eval_until_episode(episode):
            n_goal_achieved_total = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    observation = time_step.observation
                    action = self.agent.act(observation,
                                            self.global_step,
                                            eval_mode=True,
                                            obs_sensor=time_step.observation_sensor)
                time_step = self.eval_env.step(action)
                n_goal_achieved_total += time_step.n_goal_achieved
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            # here check if success for Adroit tasks. The threshold values come from the mj_envs code
            # e.g. https://github.com/ShahRutav/mj_envs/blob/5ee75c6e294dda47983eb4c60b6dd8f23a3f9aec/mj_envs/hand_manipulation_suite/pen_v0.py
            # can also use the evaluate_success function from Adroit envs, but can be more complicated
            if self.cfg.task_name == 'pen':
                threshold = 20
            else:
                threshold = 25
            if n_goal_achieved_total > threshold:
                total_success += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        success_rate_standard = total_success / n_eval_episode
        episode_reward_standard = total_reward / episode
        episode_length_standard = step * self.cfg.action_repeat / episode

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', episode_reward_standard)
            log('success_rate', success_rate_standard)
            log('episode_length', episode_length_standard)
            log('episode', self.global_episode)
            log('step', self.global_step)


    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                try:
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False,
                                            obs_sensor=time_step.observation_sensor)
                except:
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                ) if self.global_step % self.cfg.update_every_steps == 0 else dict(
                )
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfgs):
    from train_adroit import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfgs)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
