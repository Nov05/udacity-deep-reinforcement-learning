#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
## refactored for multi-envs and multi-agents (in one env) by nov05 in 2024-04



import torch.nn.functional as F
## local imports
from ..network import *
from ..component import *
from .BaseAgent import *



class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task if config.task is not None else config.task_fn()  ## task (with envs)
        self.network = config.network_fn()  ## local neural network (actor and critic)
        self.target_network = config.network_fn()  ## target neural network (actor and critic)
        self.target_network.load_state_dict(self.network.state_dict()) ## initialize target with local
        self.replay = config.replay_fn()  ## replay buffer 
        self.random_process = config.random_process_fn()  ## random states or noise
        self.states = None
        self.total_episodic_returns = []


    def step(self):
        ## reset the task (envs)
        if self.states is None:
            self.random_process.reset_states() ## denoted as 𝒩 in the paper
            self.states = self._reset_task(self.task)
            self.states = self.config.state_normalizer(self.states)

        ## step
        if self.total_steps < self.config.warm_up: ## generate random actions
            actions = self._sample_actions()
        else:  ## get actions from the local network
            self.network.eval()
            with torch.no_grad():
                ## add noise with decay
                actions = to_np(self.network(self.states)) \
                        + self.random_process.sample()*(1/np.sqrt(self.total_episodes+1))  ## denoted as 𝒩_t in the paper
            self.network.train()
        actions = self._reshape_for_task(self.task, actions)
        ## task will clip when step. however, actions have to be clipped here for replay buffer
        actions = np.clip(actions, self.task.action_space.low, self.task.action_space.high)
        next_states, rewards, dones, infos = self.task.step(actions)

        actions_ = self._reshape_for_network(actions, keep_dim=2)
        next_states_ = self._reshape_for_network(next_states, keep_dim=2)
        rewards_ = self._reshape_for_network(rewards, keep_dim=1)
        dones_ = self._reshape_for_network(dones, keep_dim=1)
        next_states_ = self.config.state_normalizer(next_states_)
        rewards_ = self.config.reward_normalizer(rewards_)

        ## update replay buffer
        self.replay.feed(dict(
            state=self.states, 
            action=actions_, 
            reward=rewards_, 
            next_state=next_states_, 
            mask=1-np.asarray(dones_, dtype=np.int32),
        ))
        self.states = next_states_

        ## sample config.mini_batch_size (denoted as S) of transition sequences from the replay buffer
        ## update neural networks
        if self.replay.size() >= self.config.warm_up \
        and self.total_steps%self.config.replay_interval==0:  ## replay every interval steps
            
            transitions = self.replay.sample()  ## batch_size is set in config.replay_fn
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            ## the networks can process data with dimension of (bath_size, observation_size)
            ## target actor (policy network) and critic (value network) forward
            phi_target = self.target_network.feature(next_states)
            a_target = self.target_network.actor(phi_target)  ## get action
            with torch.no_grad():
                q_target = self.target_network.critic(phi_target, a_target)  ## get Q-value
            q_target = q_target * mask * self.config.discount  ## multiply λ-discount rate
            q_target.add_(rewards).detach() ## shape: [mini_batch_size, 1]

            ## local critic forward
            phi = self.network.feature(states)
            q_critic = self.network.critic(phi, actions)  ## expected Q-value
            ## local critic loss and backpropagate
            # critic_loss = (q_critic-q_target).pow(2).mul(0.5).sum(-1).mean(dim=0) ## RMSE, converge faster?
            critic_loss = F.mse_loss(q_critic, q_target)  ## MSE
            self.network.critic_opt.zero_grad() 
            critic_loss.backward()
            self.network.critic_opt.step()  ## optimizer step

            ## local actor forward
            phi = self.network.feature(states)
            a = self.network.actor(phi)
            ## local actor loss and backpropagation
            actor_loss = -self.network.critic(phi.detach(), a).mean(dim=0) 
            self.network.actor_opt.zero_grad()  
            actor_loss.backward()
            self.network.actor_opt.step()  ## optimizer step

            ## update target network
            self.soft_update_network(self.target_network, self.network)

        ## check whether the episode is done
        for done,info in zip(dones,infos):
            if np.any(done):
                self.total_episodic_returns.append(info['episodic_return'])
        if len(self.total_episodic_returns)==self.task.num_envs:
            self.episode_done = True
            self.record_online_return(self.total_episodic_returns, 
                                      by_episode=self.config.by_episode)  ## log train returns
            self.total_episodic_returns = []
            self.total_episodes += 1
        self.total_steps += 1


    def eval_step(self, states):
        if states is None:
            raise Exception("⚠️ \"states\" is None")
        states = self._reshape_for_network(states, keep_dim=2)
        self.config.state_normalizer.set_read_only()
        states = self.config.state_normalizer(states)
        with torch.no_grad():
            actions = to_np(self.network(states))  ## get actions from the local network
        self.config.state_normalizer.unset_read_only()
        actions = self._reshape_for_task(self.eval_task, actions)
        return actions
    

    def soft_update_network(self, target, source):
        ## trg = trg*(1-τ) + src*τ
        tau = self.config.target_network_mix
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data*(1.-tau) + source_param.data*tau)
            

    def _sample_actions(self):
        if self.env_type in ['unity']: ## one env has multiple agents
            actions = []
            for _ in range(self.task.num_envs):
                actions += [self.task.action_space.sample()
                            for _ in range(self.task.envs_wrapper.num_agents)]
        else: ## one env has one agent
            actions = [self.task.action_space.sample()
                       for _ in range(self.task.num_envs)]
        return actions