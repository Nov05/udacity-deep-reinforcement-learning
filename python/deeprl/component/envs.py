#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
## added functionalities for the Unity environment, changed by nov05, 2024-03-04



# import os
import gym
import numpy as np
# import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, \
     CloudpickleWrapper, clear_mpi_env_vars
from unityagents.brain import BrainParameters, BrainInfo

## local imports
from ..utils import *
from ..utils.torch_utils import random_seed

import platform
if platform.system()!='Windows':
    ## windows will cause error when importing roboschool  
    ## gym.error.Error: Cannot re-register id: RoboschoolInvertedPendulum-v1
    try:
        import roboschool
        print("roboschool", roboschool.__version__)
    except ImportError as e:
        print(e)
        pass

import sys
import warnings
if not sys.warnoptions:  # allow overriding with `-W` option
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')
gym.logger.set_level(40)   

## added by nov05
import multiprocessing as mp
import dm_control2gym
from unityagents import UnityEnvironment, UnityMultiEnvironment

## wrap the class with a func, or Multiprocessing will throw
## "TypeError: cannot pickle '_thread.lock' object"
def make_unity(**kwargs):
    return lambda:UnityEnvironment(**kwargs)

env_types = {'dm', 'atari', 'gym', 'unity'}
env_fn_mappings = {'dm': dm_control2gym.make,
                   'atari': make_atari,
                   'gym': gym.make,
                   'unity': make_unity}



def get_env_type(game=None, env=None):
    env_type = None
    if game is not None:
        if game.startswith("unity"):
            env_type = 'unity'
        elif game.startswith("dm"):
            env_type = 'dm'
        else:
            env_type = 'gym'
    elif hasattr(gym.envs, 'atari') and \
        isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv):
        env_type = 'atari'
    return env_type
    

# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
## refactored, func for unity added, by nov05
def get_env_fn(game, ## could be called "id", "env_id" in other functions
               env_fn_kwargs=None,
               seed=None, 
               rank=None, 
               episode_life=True):
    
    ## get env type
    env_type =get_env_type(game=game)
    ## add kwargs for different env types
    kwargs = dict()
    if env_type=='unity':
        kwargs.update(env_fn_kwargs)
        if seed: kwargs.update({'seed':seed})
    elif env_type=='dm':
        _, domain, task = game.split('-')
        kwargs.update({'domain_name':domain, 'task_name':task})
    else: ## env_type=='gym'
        kwargs.update({'id':game})

    ## wrap the env
    if env_type=='unity':
        ## The Unity env instance can't be applied with wrapp class.  
        ## Define info['episodic_return'] in return in its superclass, the UnityVecEnv and UnitySubprocVecEnv.
        env_fn = env_fn_mappings[env_type](**kwargs)
    else:
        env = env_fn_mappings[env_type](**kwargs)
        env_type_ = get_env_type(env=env)
        if env_type_=='atari':
            env_type = env_type_
            kwargs.update({'env_id':game})
        random_seed(seed)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env) ## define info['episodic_return'] in return
        if env_type=='atari':
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            if len(env.observation_space.shape)==3:
                env = TransposeImage(env)
                env = FrameStack(env, 4)
        env_fn = lambda:env
    return env_fn, env_type



## This wrapper class adds total_reward to info['episodic_return'] in the return
## of the original environment class.
class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_reward = 0


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.total_reward += reward
        if done:
            info['episodic_return'] = self.total_reward
            self.total_reward = 0
        else:
            info['episodic_return'] = None
        return observation, reward, done, info


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
    '''
    single process, sequentialyl step each environment
    '''
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns] ## create envs
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for env,action in zip(self.envs, self.actions):
            ## info = e.g. {'episodic_return': None}
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            data.append([observation, reward, done, info])
        observations, rewards, dones, infos = zip(*data)
        return observations, np.asarray(rewards), np.asarray(dones), infos

    def reset(self):
        ## reset all envs, and return observations
        return [env.reset() for env in self.envs]

    def close(self):
        ## close all envs  
        [env.close() for env in self.envs] 



def get_unity_spaces(brain_params: BrainParameters): 
    """
    tranlate Unity ML-Agents spaces to gym spaces for compatibility with the deeprl and Baselines packages
    """
    if brain_params.vector_observation_space_type=='continuous':
        observation_space = Box(
            float('-inf'), float('inf'), 
            (brain_params.vector_observation_space_size*brain_params.num_stacked_vector_observations,), 
            np.float64)
    else:
        raise NotImplementedError
    if brain_params.vector_action_space_type=='continuous':
        action_space = Box(-1.0, 1.0, (brain_params.vector_action_space_size,), np.float32)
    else:
        raise NotImplementedError
    return observation_space, action_space



## for Unity envs
def get_return_from_brain_info(env, 
                               brain_name, 
                               phase, 
                               brain_info: BrainInfo):
    if brain_name in ['ReacherBrain', 'TennisBrain']:
        observation = brain_info.vector_observations 
    else:
        raise NotImplementedError
    
    if brain_name in ['ReacherBrain']:
        mode = 'avg'  ## calculate the average of the total scores for each agent
    elif brain_name in ['TennisBrain']:
        mode = 'max'  ## take the maximum of the total scores of each agent
    
    reward, done = brain_info.rewards, brain_info.local_done
    info = {'brain_info': brain_info}
    num_agents = len(brain_info.agents)
    ## env.total_reward: accumulated reward for each agent at the end of each episode
    if env.total_reward is None:
        env.total_reward = [0] * num_agents  
    env.total_reward = np.add(env.total_reward, reward)  ## a list

    if phase=='step':
        if np.any(done): ## one env has multi-agents hence done has multiple values
            if mode=='avg':
                info['episodic_return'] = np.sum(env.total_reward) / num_agents  ## single value
            elif mode=='max':
                info['episodic_return'] = np.max(env.total_reward)  ## single value
            else:
                raise NotImplementedError
            env.total_reward = [0] * num_agents  ## reset episode total reward for each agent
    elif phase=='reset':
        info = {'brain_info': brain_info,
                'episodic_return': 0}

    return observation, reward, done, info



class UnityVecEnv(VecEnv):
    """
    This is a wrapper class for a list of Unity environment instances,
    operating in a single process, stepping enviroments sequentially
    """
    def __init__(self, env_fns=None, train_mode=False):
        self.envs = [fn() for fn in env_fns]
        self.train_mode = train_mode

        env = self.envs[0]
        self.brain_name = env.brain_names[0]
        brain_params = env.brains[self.brain_name]
        self.action_size = brain_params.vector_action_space_size

        ## add total_reward attribute, refer to class OriginalReturnWrapper(gym.Wrapper)
        for env in self.envs:
            env.total_reward = None

        ## reset envs
        _, _, _, infos = self.reset(train_mode=train_mode)
        self.num_agents = len(infos[0]['brain_info'].agents)
        self.actions = None

        self.num_envs = len(self.envs)
        observation_space, action_space = get_unity_spaces(brain_params)
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)


    def step_async(self, actions): ## VecEnv downward func
        self.actions = actions


    def step_wait(self): ## VecEnv downward func
        data = []
        for env,action in zip(self.envs, self.actions):
            brain_info = env.step(action)[self.brain_name]
            observation, reward, done, info = get_return_from_brain_info(env, self.brain_name, 'step', brain_info)  
            data.append([observation, reward, done, info])
        observations, rewards, dones, infos = zip(*data)
        return observations, np.asarray(rewards), np.asarray(dones), infos


    def reset(self, train_mode=None):
        ## reset an env, returning AllBrainInfo
        data = []
        for env in self.envs:
            brain_info = env.reset(train_mode=train_mode)[self.brain_name]
            observation, reward, done, info = get_return_from_brain_info(env, self.brain_name, 'reset', brain_info)
            data.append([observation, reward, done, info])
            print('🟢 Unity environment has been resetted.')
        observations, rewards, dones, infos = zip(*data)
        return observations, np.asarray(rewards), np.asarray(dones), infos


    def close(self):
        [env.close() for env in self.envs]



def unity_worker(remote, parent_remote, env_fn_wrapper, train_mode):
    parent_remote.close()
    env = env_fn_wrapper.x()
    brain_name = env.brain_names[0]
    ## add total_reward attribute, refer to class OriginalReturnWrapper(gym.Wrapper)
    env.total_reward = None
    try:
        while True:
            cmd, data = remote.recv()
            if cmd=='step':
                ## type AllBrainInfo, a dict
                ## e.g. {'ReacherBrain': <unityagents.brain.BrainInfo object at 0x0000022605F2D8A0>}
                brain_info = env.step(data)[brain_name] ## info type ".unityagents.brain.BrainInfo"
                observation, reward, done, info = get_return_from_brain_info(env, brain_name, 'step', brain_info)
                remote.send((observation, reward, done, info))
            elif cmd=='reset':
                brain_info = env.reset(data)[brain_name]
                observation, reward, done, info = get_return_from_brain_info(env, brain_name, 'reset', brain_info)
                remote.send((observation, reward, done, info))
                print('🟢 Unity environment has been resetted.')
            elif cmd=='close':
                remote.close() 
                break
            elif cmd=='get_brain_params':
                brain_params = env.brains[brain_name] ## brain_params: type class BrainParameters
                ## the original value seems to be <class 'google.protobuf.pyext._message.RepeatedScalarContainer'>
                ## and not serializable, which would cause error in Multiprocessing pickling
                brain_params.vector_action_descriptions = ['','','','']
                remote.send(brain_params)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('⚠️ UnitySubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()



class UnitySubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple Unity environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs>1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, train_mode=False, context='spawn'):
        """
        Arguments:
        env_fns: iterable of callables - functions that create environments to run in subprocesses. 
                 Need to be cloud-pickleable
        """
        self.env_fns = env_fns
        self.train_mode = train_mode
        self.context = context

        self.waiting = False
        self.closed = False
        self.viewer = None
        self.num_envs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [ctx.Process(target=unity_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), self.train_mode)) 
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
                print(f"🟢 {p.name} has started.")
        for remote in self.work_remotes:
            remote.close()

        ## get brain parameters
        self.remotes[0].send(('get_brain_params', None))
        brain_params = self.remotes[0].recv()
        self.brain_name = brain_params.brain_name
        observation_space, action_space = get_unity_spaces(brain_params)
        
        ## reset the envs to get num_agents
        _, _, _, infos = self.reset(train_mode=self.train_mode) 
        self.num_agents = len(infos[0]['brain_info'].agents)

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True


    def step_wait(self):
        self._assert_not_closed()
        data = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, infos = zip(*data)
        return np.stack(observations), np.stack(rewards), np.stack(dones), infos


    def reset(self, train_mode=None):
        """
        Reset all Unity environments serially
        """
        self._assert_not_closed()
        data = []
        for remote in self.remotes:
            remote.send(('reset', train_mode))
            data.append(remote.recv())
        observations, rewards, dones, infos = zip(*data)
        return np.stack(observations), np.stack(rewards), np.stack(dones), infos


    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


    def _assert_not_closed(self):
        assert not self.closed, "⚠️ Trying to operate on a UnitySubprocVecEnv object after calling close()"


    def __del__(self):
        if not self.closed:
            self.close()



class Task:
    def __init__(self,
                 game, ## gym or unity game, called "id" or "env_id" in other funcs
                 num_envs=1,
                 env_fn_kwargs=dict(), ## for unity envs
                 train_mode=False, ## for unity envs only
                 seeds=None,
                 single_process=True,
                 log_dir=None,
                 episode_life=True):
        '''
        2024-03-10 added logic for unity
        '''
        ## input parameters
        self.game = game
        self.num_envs = num_envs
        self.env_fn_kwargs = env_fn_kwargs
        self.train_mode = train_mode 
        self.seeds = seeds
        self.log_dir = log_dir
        self.single_process = single_process
        self.episode_life = episode_life

        self.envs = []

        ## make log directory
        if log_dir:
            mkdir(log_dir)

        ## get env type
        self.env_type = None
        if game.startswith('unity'):
            self.env_type = 'unity'

        ## get seeds
        if not self.seeds:
            if self.env_type=='unity':
                self.seeds = [np.random.RandomState().randint(-2147483648, 2147483647) 
                              for _ in range(self.num_envs)]
            else:
                self.seeds = [np.random.RandomState().randint(0, 2**31) 
                              for _ in range(self.num_envs)] 
        ## get env_fns
        self.env_fns = []
        for i in range(self.num_envs):
            if self.env_type=='unity':
                self.env_fn_kwargs.update({'worker_id':i})
            env_fn, self.env_type = get_env_fn(self.game, 
                                               env_fn_kwargs=self.env_fn_kwargs, 
                                               seed=self.seeds[i], 
                                               rank=i, 
                                               episode_life=self.episode_life)
            self.env_fns.append(env_fn)

        ## create envs
        Wrapper, kwargs = None, {'env_fns':self.env_fns}
        if self.env_type in ['unity']:
            if single_process:
                Wrapper = UnityVecEnv
            else:
                Wrapper = UnitySubprocVecEnv
            kwargs['train_mode'] = self.train_mode
        else:
            if single_process:
                Wrapper = DummyVecEnv 
            else:
                Wrapper = SubprocVecEnv
        self.envs_wrapper = Wrapper(**kwargs)
            
        ## get dimensions for different spaces    
        self.observation_space = self.envs_wrapper.observation_space
        self.state_dim = int(np.prod(self.observation_space.shape))
        self.action_space = self.envs_wrapper.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'
  

    def reset(self, train_mode=None):
        ## train_mode is for unity envs only
        kwargs = {}
        if self.env_type in ['unity']:
            if train_mode is not None:
                self.train_mode = train_mode
            kwargs['train_mode'] = self.train_mode
        return self.envs_wrapper.reset(**kwargs)
        

    def step(self, actions):
        if isinstance(self.action_space, Box): 
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.envs_wrapper.step(actions)
    
    
    def close(self):
        return self.envs_wrapper.close()