import numpy as np
import pandas as pd
import random
from multiprocess import Process, current_process, cpu_count

## local imports
from unityagents import UnityEnvironment, UnityMultiEnvironment


def test(file_name=None, worker_id=0, seeds=None, train_mode=False,
         no_graphics=False, num_envs=1, env_ids=None, max_steps=0):
    # env = UnityEnvironment(file_name=file_name, worker_id=worker_id,
    #                        no_graphics=no_graphics) ## single env
    env = UnityMultiEnvironment(file_name=file_name, worker_id=worker_id, seeds=seeds,
                                no_graphics=no_graphics, 
                                num_envs=num_envs)
    brain_name = env.brain_names[0] ## there is only one brain for the executables that I have
    brain = env.brains[brain_name]

    if isinstance(env, UnityEnvironment):
        print("Running an instance of UnityEnvironment, which has only one env...")
        env_id = 0
        env_info = env.reset(train_mode=train_mode)[brain_name]
    else:
        print("Running an instance of UnityMultiEnvironment, which has multiple envs...")
        env_id = random.choice(env_ids) if env_ids is not None else 0
        print(f"Displaying env {env_id} info...")
        env_info = env.reset(train_mode=train_mode)[env_id][brain_name]  # reset the environments

    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    action_type = brain.vector_action_space_type
    scores = np.zeros(num_agents)                          # initialize the score (for each agent) 
    if isinstance(env, UnityEnvironment):
        input_dimension = [num_agents]
    else:
        input_dimension = [num_envs, num_agents]
    
    for i in range(max_steps):
        if action_type=='continuous':
            actions = np.random.randn(*(input_dimension+[action_size])) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)              # all actions between -1 and 1
        else: ## discrete
            actions = np.random.randint(0, action_size, 
                                        size=input_dimension) 
        if isinstance(env, UnityEnvironment):
            env_info = env.step(actions)[brain_name]       # send all actions to the environment
        else:
            if i==0: env.step_input_check(vector_actions=actions, env_ids=env_ids) ## check only once
            env_info = env.step(actions, env_ids=env_ids, 
                                input_check=False)[env_id][brain_name] # send all actions to the environment
        rewards = env_info.rewards                               # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
            print(f"An agent in env {env_id} finished an episode!")
            break
    print(f"🟢 Current process {current_process()} "
          f"\nenv {env_id}, total score (averaged over agents) this episode: {np.mean(scores)}")
    env.close()


if __name__ == '__main__':

    max_steps = 10000 ## banana:1000, reacher:10000
    num_envs = 2
    env_ids = None ## None or a list
    train_mode = False 
    no_graphics = False
    seeds = [random.randint(-2147483648, 2147483647) for _ in range(num_envs)] 
    # file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    # file_name = '..\data\Banana_Windows_x86_64\Banana.exe'
    
    print("🟢", current_process())
    print("Number of cpu:", cpu_count()) ## 12
    kwargs = {'file_name':file_name, 'seeds':seeds, 'train_mode':train_mode, 'no_graphics':no_graphics, 
              'num_envs':num_envs, 'max_steps':max_steps}
    p1 = Process(target=test, kwargs=kwargs)
    kwargs['worker_id'] = num_envs * 1
    seeds = [random.randint(-2147483648, 2147483647) for _ in range(num_envs)] ## generate new seeds
    kwargs['seeds'] = seeds
    p2 = Process(target=test, kwargs=kwargs)
    p1.start()
    p2.start()
    p1.join()
    p2.join()

## $ python -m tests2.test_unity_multiprocessing
## kill the terminal each time after execution, or the ports might be stil busy.