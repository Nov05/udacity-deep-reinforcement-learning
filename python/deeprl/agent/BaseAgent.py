#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave
import pickle

## local imports
from ..utils import *
from deeprl.component.envs import get_env_type



class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0
        self.env_type = get_env_type(game=self.config.game)  ## added by nov05
        self.total_steps = 0
        self.total_episodes = 0  ## added by nov05
        self.episode_done = False   ## all envs have done an episode, added by nov05


    def close(self):
        try:
            close_obj(self.config._eval_env)
            print(f"🟢 Eval envs {self.config._eval_env} have been closed.") 
        except:
            pass
        try:
            close_obj(self.task)  ## aka. train trask, train envs
            print(f"🟢 Train task {self.task} has been closed.")
        except:
            pass


    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)


    def load(self, filename):
        # state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc:storage) ## cpu
        state_dict = torch.load('%s.model' % filename)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))


    def eval_step(self, state):
        raise NotImplementedError


    def eval_episode(self):
        episodic_returns = []
        while True:
            actions = self.eval_step(self.eval_states)
            self.eval_states, _, dones, infos = self.eval_task.step(actions) ## observations, rewards, dones, infos
            for done,info in zip(dones,infos):
                if np.any(done):  ## there are multiple agents in one Unity env, hence multiple values in "done"
                    episodic_returns.append(info['episodic_return'])
            if len(episodic_returns)==self.eval_task.num_envs:  ## all envs are done
                break 
        return episodic_returns


    def eval_episodes(self, by_episode=False):
        ## self.config.eval_env is a Task instance with a wrapper instance of a list of env instances
        self.eval_task = self.config.eval_env 
        self.eval_states = self._reset_task(self.eval_task, train_mode=False)
        log_info = f"Step {self.total_steps}, evaluating {self.config.eval_episodes} episodes " + \
                   f"in {self.config.num_workers_eval} environments"
        self.logger.info(log_info)
        total_episodic_returns = []
        for i in range(self.config.eval_episodes):
            print(f"Evaluating episode {i}...")
            episodic_returns = self.eval_episode()
            total_episodic_returns.append(np.mean(episodic_returns))
        log_info = f"Step {self.total_steps}, " + \
                   f"episodic_return_test {np.mean(total_episodic_returns):.2f}" + \
                   f"({np.std(total_episodic_returns)/np.sqrt(len(total_episodic_returns)):.2f})"
        if by_episode:
            log_info = f"Episode {self.total_episodes}, " + log_info
        self.logger.info(log_info)
        self.logger.add_scalar('episodic_return_test', np.mean(total_episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(total_episodic_returns),
        }


    def record_online_return(self, info, offset=0, by_episode=False):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps+offset)
                self.logger.info(f"Step {self.total_steps+offset}, episodic_return_train {ret}")
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)   ## recursive
        elif isinstance(info, list) and isinstance(info[0], float): ## for Unity, added by nov05
            ret = np.mean(info)
            self.logger.add_scalar('episodic_return_train', ret, self.total_steps)
            log_info = f"Step {self.total_steps}, episodic_return_train {ret}"
            if by_episode: log_info = f"Episode {self.total_episodes}, " + log_info
            self.logger.info(log_info)
        else:
            raise NotImplementedError
        return {'episodic_return_train': ret}


    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)


    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, _, _, info = env.step(action) ## state, reward, done, info
            ret = info[0]['episodic_return']
            steps += 1
            if ret:
                break


    def record_step(self, state):
        raise NotImplementedError


    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


    ## added by nov05
    def _reshape_for_network(self, data, keep_dim=2):
        data = np.array(data)
        if len(data.shape)>keep_dim:
            if keep_dim>1:
                data = data.reshape(-1, *data.shape[-keep_dim+1:]).tolist()
            else:
                data = data.reshape(-1).tolist()
        elif len(data.shape)<=keep_dim:
            data = data.tolist()
        return data
    

    ## added by nov05
    def _reshape_for_task(self, task, data):
        if self.env_type in ['unity']: ## one env has multiple agents
            data = np.array(data)
            data = data.reshape(task.num_envs, task.envs_wrapper.num_agents, -1)
        return data
    

    ## added by nov05
    def _reset_task(self, task, train_mode=True):
        if self.env_type in ['unity']: ## one env has multiple agents
            states, _, _, _ = task.reset(train_mode=train_mode)  ## observations, rewards, dones, infos
            states = self._reshape_for_network(states, keep_dim=2)
        else:
            states = task.reset()
        return states


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])
