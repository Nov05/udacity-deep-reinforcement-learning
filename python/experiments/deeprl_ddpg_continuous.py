from torch import nn
import torch.nn.functional as F
import argparse

## local imports
from deeprl import *
from deeprl.utils.misc import rmdir, run_episodes, eval_episodes



## 1. find the config here or in "./deeprl/utils/config.py"
## 2. refer to ".\deeprl_files\examples.py" 
##    $ python -m deeprl_files.examples
## 3. refer to "\tests2\test_deeprl_envs.py"
##    $ python -m tests2.test_deeprl_envs

def ddpg_continuous(**kwargs): 
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    ## train
    config.task = Task(config.game, 
                       num_envs=config.num_workers,
                       env_fn_kwargs=config.env_fn_kwargs, 
                       train_mode=True,
                       single_process=False)
    config.by_episode = True  ## control by episode; if false, by step
    config.max_episodes = 161

    ## eval
    config.eval_env = Task(config.game, 
                           num_envs=config.num_workers_eval,
                           env_fn_kwargs=config.env_fn_kwargs_eval, 
                           train_mode=False,
                           single_process=False)
    config.eval_episodes = num_eval_episodes  ## eval n episodes per interval
    config.eval_episode_interval = 10
    config.eval_after_episodes = 10

    ## save
    config.save_after_episodes = 140 ## save model
    config.save_episode_interval = 5 ## save model

    ## neural network
    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim,  
        config.action_dim,  
        actor_body=FCBody(config.state_dim, (128,128), gate=nn.LeakyReLU, 
                          init_method='uniform_fan_in', 
                          batch_norm=nn.BatchNorm1d,),
        critic_body=FCBody(config.state_dim+config.action_dim, (128,128), gate=nn.LeakyReLU, 
                           init_method='uniform_fan_in', batch_norm=nn.BatchNorm1d),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        ## for the critic optimizer, it seems that 1e-3 won't converge
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5),  
        # batch_norm=nn.BatchNorm1d,
        )
    
    ## replay settings
    config.min_memory_size = int(1e6)
    config.mini_batch_size = 128
    config.replay_fn = lambda: UniformReplay(memory_size=config.min_memory_size, 
                                             batch_size=config.mini_batch_size)
    config.discount = 0.9  ## λ lambda, Q-value discount rate
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    ## before it is warmed up, use random actions, do not sample from buffer or update neural networks
    config.warm_up = int(1e4)  ## can't be 0 steps, or it will create a deadloop in buffer.
    config.replay_interval = 1  ## replay every n steps
    config.target_network_mix = 1e-3  ## τ: soft update rate=0.1%, trg = trg*(1-τ) + src*τ

    if is_training:
        # run_steps(DDPGAgent(config))
        run_episodes(DDPGAgent(config))
    else:
        config.save_filename = save_filename
        eval_episodes(DDPGAgent(config))



if __name__ == '__main__':


    # # Create an ArgumentParser object
    # parser = argparse.ArgumentParser()
    # # Add arguments
    # parser.add_argument('-it', '--is_training', required=True, help='Whether it is a training task')
    # # Parse arguments
    # args = parser.parse_args()
    # # Use the arguments
    # is_training = args.is_training
    is_training = True
  
    ## path to the game file
    # env_file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    env_file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    ## path to the saved torch model file
    save_filename = '.\experiments\ddpg_unity-reacher-v2\DDPGAgent-unity-reacher-v2-remark_ddpg_continuous-run-0-155'  

    set_one_thread() 
    random_seed()

    if is_training:
        # $ python -m tests2.test_rmdir
        ## remove all log and saved files
        try:
            rmdir('data')
        except:
            pass
        mkdir('data\\log')  ## readable logs
        mkdir('data\\tf_log')  ## tensorflow logs
        mkdir('data\\models')  ## trained models
        select_device(0) ## 0: GPU, an non-negative integer is the index of GPU
        num_envs = 1
        num_envs_eval = 3
        offset = 0
        eval_no_graphics = True
        num_eval_episodes = 2
    else:
        select_device(0)  ## -1: CPU
        num_envs = 1
        num_envs_eval = 4
        ## if run train and test in the same terminal session, skip the training port range to avoid potential conflict.
        ## e.g. for training num_envs=1, num_envs_eval=3, hence Unity env port offset=4
        offset = 4  
        eval_no_graphics = False
        num_eval_episodes = 2

    env_fn_kwargs = {'file_name': env_file_name, 'no_graphics': True, 'base_port':5005+offset}
    env_fn_kwargs_eval = {'file_name': env_file_name, 'no_graphics': eval_no_graphics, 
                          'base_port':5005+offset+num_envs}
    ddpg_continuous(game='unity-reacher-v2', 
                    run=0,
                    env_fn_kwargs=env_fn_kwargs,
                    env_fn_kwargs_eval=env_fn_kwargs_eval,
                    num_workers=num_envs,
                    num_workers_eval=num_envs_eval,
                    remark=ddpg_continuous.__name__)
    


## $ python -m experiments.deeprl_ddpg_continuous    <- run this file, train or eval unity reacher
## $ python -m experiments.deeprl_ddpg_plot          <- plot tensorflow log data (tf_log)
## $ python -m deeprl_files.examples                 <- train mujoco reacher
## $ python -m tests2.test_deeprl_envs               <- test unity envs
## $ python -m tests2.test_rmdir                     <- delete logs, models, plots. run with caution

## example network architecture for "unity-reacher-v2"
## check the example log file
'''
DeterministicActorCriticNet(
  (phi_body): DummyBody()
  (actor_body): FCBody(
    (layers): ModuleList(
      (0): Linear(in_features=33, out_features=128, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Linear(in_features=128, out_features=128, bias=True)
      (4): LeakyReLU(negative_slope=0.01)
      (5): Linear(in_features=128, out_features=4, bias=True)
      (6): Tanh()
    )
  )
  (critic_body): FCBody(
    (layers): ModuleList(
      (0): Linear(in_features=37, out_features=128, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Linear(in_features=128, out_features=128, bias=True)
      (4): LeakyReLU(negative_slope=0.01)
      (5): Linear(in_features=128, out_features=1, bias=True)
    )
  )
)
'''