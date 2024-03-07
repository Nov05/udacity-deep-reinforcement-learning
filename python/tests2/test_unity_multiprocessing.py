import numpy as np
import pandas as pd

## local imports
from unityagents import UnityEnvironment, MultiUnityEnvironment
# from deeprl import *


def test():
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[0][brain_name]  # reset the environment 
    print(env_info)
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    _ = env_info.vector_observations                       # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    for _ in range(max_steps):
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        # actions = np.random.randint(num_agents, action_size)
        env_info = env.step(actions)[0][brain_name]           # send all actions to the environment
        _ = env_info.rewards                               # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
            print("An agent finished an episode!")
            break
    print(f'🟢 Total score (averaged over agents) this episode: {np.mean(scores)}')
    env.close()


if __name__ == '__main__':

    max_steps = 10000
    # file_name = '..\data\Reacher_Windows_x86_64_1\Reacher.exe'
    file_name = '..\data\Reacher_Windows_x86_64_20\Reacher.exe'
    # file_name = '..\data\Banana_Windows_x86_64\Banana.exe'
    # env = UnityEnvironment(file_name=file_name, no_graphics=False)
    env = MultiUnityEnvironment(file_name=file_name, no_graphics=False, num_envs=4)
    test()

## $ python -m tests2.test_unity_multiprocessing