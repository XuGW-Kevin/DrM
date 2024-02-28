import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, feature_dim):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def forward(self, h, state=None):
        state_en = self.state_encoder(state)
        h = h + state_en
        return self.fusion(h)


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim,  state_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        
        self.state_enc = StateEncoder(state_dim, hidden_dim, feature_dim)

        self.apply(utils.weight_init)

    def forward(self, obs, std, obs_sensor=None):
        h = self.trunk(obs)
        h = self.state_enc(h, obs_sensor)
        
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, state_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.state_enc = StateEncoder(state_dim, hidden_dim, feature_dim)
        
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, obs_sensor=None):
        h = self.trunk(obs)
        h = self.state_enc(h, obs_sensor)
        
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    
class VNetwork(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim, state_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.V = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.state_enc = StateEncoder(state_dim, hidden_dim, feature_dim)

        self.apply(utils.weight_init)

    def forward(self, obs, obs_sensor=None):
        h = self.trunk(obs)
        h = self.state_enc(h, obs_sensor)
        v = self.V(h)
        return v


class DrMAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, dormant_threshold,
                 target_dormant_ratio,  dormant_temp,
                 target_lambda, lambda_temp, dormant_perturb_interval,
                 min_perturb_factor, max_perturb_factor, perturb_rate,
                 num_expl_steps, stddev_type, stddev_schedule, stddev_clip,
                 expectile, use_tb, state_dim=None):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_type = stddev_type
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.dormant_threshold = dormant_threshold
        self.target_dormant_ratio = target_dormant_ratio
        self.dormant_temp = dormant_temp
        self.target_lambda = target_lambda
        self.lambda_temp = lambda_temp
        self.dormant_ratio = 1
        self.dormant_perturb_interval = dormant_perturb_interval
        self.min_perturb_factor = min_perturb_factor
        self.max_perturb_factor = max_perturb_factor
        self.perturb_rate = perturb_rate
        self.expectile = expectile
        self.awaken_step = None
        self.state_dim = state_dim

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, state_dim).to(device)
        self.value_predictor = VNetwork(self.encoder.repr_dim, feature_dim,
                                        hidden_dim, state_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, state_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim, state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.predictor_opt = torch.optim.Adam(
            self.value_predictor.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    @property
    def stddev(self):
        return 1 / (1 +
                    math.exp(-self.dormant_temp *
                             (self.dormant_ratio - self.target_dormant_ratio)))

    @property
    def perturb_factor(self):
        return min(max(self.min_perturb_factor, 1 - self.perturb_rate * self.dormant_ratio), self.max_perturb_factor)

    @property
    def lambda_(self):
        return self.target_lambda / (
            1 + math.exp(self.lambda_temp *
                         (self.dormant_ratio - self.target_dormant_ratio)))

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.value_predictor.train(training)

    def act(self, obs, step, eval_mode, obs_sensor=None):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        
        if obs_sensor is not None:
            obs_sensor = torch.as_tensor(obs_sensor, device=self.device)
        
        if self.stddev_type == "drqv2":
            stddev = utils.schedule(self.stddev_schedule, step)
        elif self.stddev_type == "max":
            stddev = max(utils.schedule(self.stddev_schedule, step),
                         self.stddev)
        elif self.stddev_type == "dormant":
            stddev = self.stddev
        elif self.stddev_type == "awake":
            if self.awaken_step == None:
                stddev = self.stddev
            else:
                stddev = max(
                    self.stddev,
                    utils.schedule(self.stddev_schedule,
                                   step - self.awaken_step))
        else:
            raise NotImplementedError(self.stddev_type)

        dist = self.actor(obs, stddev, obs_sensor)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_predictor(self, obs, action, obs_sensor=None):
        metrics = dict()

        if obs_sensor is None:
            Q1, Q2 = self.critic(obs, action)
        else:
            Q1, Q2 = self.critic(obs, action, obs_sensor)
        Q = torch.min(Q1, Q2)
        if obs_sensor is None:
            V = self.value_predictor(obs)
        else:
            V = self.value_predictor(obs, obs_sensor)
        vf_err = V - Q
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 -
                                                                self.expectile)
        predictor_loss = (vf_weight * (vf_err**2)).mean()

        if self.use_tb:
            metrics['predictor_loss'] = predictor_loss.item()

        self.predictor_opt.zero_grad(set_to_none=True)
        predictor_loss.backward()
        self.predictor_opt.step()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, 
                    step, obs_sensor=None, next_obs_sensor=None):
        metrics = dict()

        with torch.no_grad():
            if self.stddev_type == "drqv2":
                stddev = utils.schedule(self.stddev_schedule, step)
            elif self.stddev_type == "max":
                stddev = max(utils.schedule(self.stddev_schedule, step),
                             self.stddev)
            elif self.stddev_type == "dormant":
                stddev = self.stddev
            elif self.stddev_type == "awake":
                if self.awaken_step == None:
                    stddev = self.stddev
                else:
                    stddev = max(
                        self.stddev,
                        utils.schedule(self.stddev_schedule,
                                       step - self.awaken_step))
            else:
                raise NotImplementedError(self.stddev_type)
            if next_obs_sensor is None:
                dist = self.actor(next_obs, stddev)
            else:
                dist = self.actor(next_obs, stddev, next_obs_sensor)
            next_action = dist.sample(clip=self.stddev_clip)
            if next_obs_sensor is None:
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            else:
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action, next_obs_sensor)
            target_V_explore = torch.min(target_Q1, target_Q2)
            if next_obs_sensor is None:
                target_V_exploit = self.value_predictor(next_obs)
            else:
                target_V_exploit = self.value_predictor(next_obs, next_obs_sensor)
            target_V = self.lambda_ * target_V_exploit + (
                1 - self.lambda_) * target_V_explore
            target_Q = reward + (discount * target_V)

        if obs_sensor is None:
            Q1, Q2 = self.critic(obs, action)
        else:
            Q1, Q2 = self.critic(obs, action, obs_sensor)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, obs_sensor):
        metrics = dict()
        if self.stddev_type == "drqv2":
            stddev = utils.schedule(self.stddev_schedule, step)
        elif self.stddev_type == "max":
            stddev = max(utils.schedule(self.stddev_schedule, step),
                         self.stddev)
        elif self.stddev_type == "dormant":
            stddev = self.stddev
        elif self.stddev_type == "awake":
            if self.awaken_step == None:
                stddev = self.stddev
            else:
                stddev = max(
                    self.stddev,
                    utils.schedule(self.stddev_schedule,
                                   step - self.awaken_step))
        else:
            raise NotImplementedError(self.stddev_type)
        if obs_sensor is None:
            dist = self.actor(obs, stddev)
        else:
            dist = self.actor(obs, stddev, obs_sensor)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        if obs_sensor is None:
            Q1, Q2 = self.critic(obs, action)
        else:
            Q1, Q2 = self.critic(obs, action, obs_sensor)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def perturb(self):
        utils.perturb(self.actor, self.actor_opt, self.perturb_factor)
        utils.perturb(self.critic, self.critic_opt, self.perturb_factor)
        utils.perturb(self.critic_target, self.critic_opt, self.perturb_factor)
        utils.perturb(self.encoder, self.encoder_opt, self.perturb_factor)
        utils.perturb(self.value_predictor, self.predictor_opt, self.perturb_factor)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.dormant_perturb_interval == 0:
            self.perturb()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, obs_sensor, next_obs_sensor = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # calculate dormant ratio
        self.dormant_ratio = utils.cal_dormant_ratio(
            self.actor, obs.detach(), 0, obs_sensor, percentage=self.dormant_threshold)
        metrics['actor_dormant_ratio'] = self.dormant_ratio
        
        if self.awaken_step is None and step > self.num_expl_steps and self.dormant_ratio < self.target_dormant_ratio:
            self.awaken_step = step

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # # update predictor
        metrics.update(self.update_predictor(obs.detach(), action, obs_sensor))

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, obs_sensor, next_obs_sensor))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step, obs_sensor))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
