#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import argparse
import torch

## local imports
from .normalizer import *


class Config:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1
    DEFAULT_REPLAY = 'replay'
    PRIORITIZED_REPLAY = 'prioritized_replay'

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        ## tasks
        self.game = None  ## or "env_id", e.g. "unity-reacher-v2"
        self.task_name = None  ## task.game
        self.state_dim = None
        self.action_dim = None
        self.task_fn = None  ## task env func; set either config._task or config.task_fn
        self._task = None  ## task env; set either config._task or config.task_fn
        self.num_workers = 1  ## task env number
        self.env_fn_kwargs = dict()  ## task env func kwargs
        self._eval_env = None
        self.num_workers_eval = 1
        self.env_fn_kwargs_eval = dict()  ## eval env func kwargs
        self.tasks = None
        ## log
        self.tag = 'vanilla'  ## tag for logs
        self.by_episode = False  ## control by step/interval; True: by episode
        self.log_interval = int(1e3)  ## steps
        self.log_level = 0
        self.iteration_log_interval = 30
        ## save models
        self.save_interval = 0  ## save every n steps; 0 = no save
        self.save_after_steps = -1  ## save after training n steps
        self.save_episode_interval = 0 ## save every n episodes
        self.save_after_episodes = -1 ## save after training n episodes
        self.save_filename = None  ## saved torch model file name
        ## eval models
        self.eval_interval = 0  ## eval every n steps; 0 = no eval
        self.eval_episodes = 10  ## eval n episodes 
        self.eval_episode_interval = 0  ## eval every n episodes
        self.eval_after_episodes = -1  ## eval after training n episodes
        ## neural networks
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.target_network_update_freq = None
        self.target_network_mix = 0.001  ## τ (tau), soft update rate
        self.gradient_clip = None
        self.entropy_weight = 0
        self.decaying_lr = False
        self.async_actor = True
        self.double_q = False
        ## step
        self.max_steps = 0  ## task maximum step number
        self.max_episodes = 0 ## task maximum episodes
        self.n_step = 1
        self.random_process_fn = None  ## noise function
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.discount = None  ## λ lambda, Q-value dicount rate
        self.exploration_steps = None
        self.history_length = None
        ## replay
        self.replay_fn = None
        self.replay_type = Config.DEFAULT_REPLAY
        self.warm_up = 0  ## replay after n steps
        self.replay_interval = 1  ## replay per n steps
        self.min_memory_size = None 
        self.mini_batch_size = 64
        self.use_gae = False
        self.gae_tau = 1.0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        ## others
        self.shared_repr = False
        self.noisy_linear = False

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, env):
        self._task = env
        if self.task_name is None: self.task_name = env.game 
        if self.state_dim is None: self.state_dim = env.state_dim 
        if self.action_dim is None: self.action_dim = env.action_dim 

    @property
    def eval_env(self):
        return self._eval_env

    @eval_env.setter
    def eval_env(self, env):
        self._eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.game

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self), indent=4, width=1)

