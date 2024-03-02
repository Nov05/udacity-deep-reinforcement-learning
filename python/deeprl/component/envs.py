# import os
# os.add_dll_directory(r"C:/Users/guido/.mujoco/mjpro150/bin")
# os.add_dll_directory(r"C:/Users/guido/.mujoco/mujoco-py-1.50.1.68/mujoco_py")
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# import os
import gym
import numpy as np
# import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

## local imports
from ..utils import *

# try:
#     import roboschool
# except ImportError as e:
#     print(e)
#     print("You are probably using Windows. Roboschool doesn't work on Windows.")
#     pass

import sys
import warnings
if not sys.warnoptions:  # allow overriding with `-W` option
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')
gym.logger.set_level(40)   



# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed, rank, episode_life=True):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape)==3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


## The original one in baselines is really bad
## baselines\baselines\common\vec_env\dummy_vec_env.py    
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            ## info e.g. {'episodic_return': None}
            obsv, revw, done, info = self.envs[i].step(self.actions[i])
            if done:
                obsv = self.envs[i].reset()
            data.append([obsv, revw, done, info])
        obsvs, revws, dones, infos = zip(*data)
        return obsvs, np.asarray(revws), np.asarray(dones), infos

    def reset(self):
        ## reset all envs, and return next_states
        return [env.reset() for env in self.envs]

    def close(self):
        ## close all envs
        [env.close() for env in self.envs] 


class MLAgentsVecEnv(VecEnv):
    def __init__(self, env):
        self.envs = [env] ## one env is imported in this case

        env = self.envs[0]
        self.brain_name = env.brain_names[0]
        brain = env.brains[self.brain_name]

        num_envs = len(self.envs)
        ## tranlate Unity ML-Agents spaces to gym spaces
        observation_space = Box(float('-inf'), float('inf'), (brain.vector_observation_space_size,), np.float64)
        action_space = Box(-1.0, 1.0, (brain.vector_action_space_size,), np.float32)
        VecEnv.__init__(self, num_envs, observation_space, action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            env_info = self.envs[i].step(self.actions[i])[self.brain_name] 
            obsv, revw, done, info = env_info.vector_observations, env_info.rewards, env_info.local_done, None
            if done:
                obsv = self.envs[i].reset()
            data.append([obsv, revw, done, info])
        obsvs, revws, dones, infos = zip(*data)
        return obsvs, np.asarray(revws), np.asarray(dones), infos

    def reset(self, train_mode=True):
        return [env.reset(train_mode=train_mode) for env in self.envs]

    def close(self):
        [env.close() for env in self.envs]


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 env=None, 
                 is_mlagents=False,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        self.name = name
        if not seed:
            seed = np.random.randint(int(1e9))
        if log_dir:
            mkdir(log_dir)
        if not env:
            env_fns = [make_env(name, seed, i, episode_life) for i in range(num_envs)]

        if is_mlagents: ## Unity ML-Agents
            self.is_mlagents = True
            Wrapper = MLAgentsVecEnv
            self.envs_wrapper = Wrapper([env])
        else:
            self.is_mlagents = False
            if single_process:
                Wrapper = DummyVecEnv
            else:
                Wrapper = SubprocVecEnv
            self.envs_wrapper = Wrapper(env_fns)
            
        self.observation_space = self.envs_wrapper.observation_space
        self.state_dim = int(np.prod(self.observation_space.shape))
        self.action_space = self.envs_wrapper.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'
        
    def reset(self, train_mode=True):
        if self.is_mlagents:
            return self.envs_wrapper.reset(train_mode=train_mode)
        else: 
            return self.envs_wrapper.reset()
        
    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.envs_wrapper.step(actions)
    
    def close(self):
        return self.envs_wrapper.close()


## nov05
if __name__ == '__main__':

    ## in the dir "./python", run "python -m deeprl.component.envs" in terminal
    task = Task('Hopper-v2', num_envs=10, single_process=True) ## multiprocessing doesn't work in Windows
    state = task.reset()
    for _ in range(100):
        actions = [np.random.rand(task.action_space.shape[0])] * task.envs_wrapper.num_envs
        _, _, dones, _ = task.step(actions)
        if np.sum(dones):
            print(dones)
    task.close()

    ## This might be helpful for custom env debugging
    # env_dict = gym.envs.registration.registry.env_specs.copy()
    # for item in env_dict.items():
    #     print(item)