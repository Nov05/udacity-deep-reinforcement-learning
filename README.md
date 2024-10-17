
## **👉 Setup Python environment for the repo**    

* [notes for env setup](https://gist.github.com/Nov05/36ed6fff08f16f29c364090844eb1d24)  
* [notes for issues](https://gist.github.com/Nov05/1d49183a91456a63e13782e5f49436be?permalink_comment_id=4935583#gistcomment-4935583)

---  

## **👉 Unity enviroment `Tennis` vector game (Project Submission)**  

* check the [project information](https://github.com/Nov05/udacity-deep-reinforcement-learning/tree/master/p3_collab-compet) (multi-agent reinforcement learning (MARL))    
* check the [course notes](https://www.evernote.com/shard/s139/sh/3207cf3f-bcca-a008-c221-45bbd101af72/qBMCR47uxmw1ied7hOLgWCxDfJFWUgoKErH3sbCLoIOTVUIw0x_YVyPiBw)    

✅ **setup Python environment**   
* [notes for env setup](https://gist.github.com/Nov05/36ed6fff08f16f29c364090844eb1d24)  
* [notes for issues](https://gist.github.com/Nov05/1d49183a91456a63e13782e5f49436be?permalink_comment_id=4935583#gistcomment-4935583)

✅ **reference**  
* [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper_files/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)  


---  

## **👉 AlphaZero**  

* [Tic-Tac-Toe notebook](https://nbviewer.org/github/Nov05/udacity-deep-reinforcement-learning/blob/master/alphazero/alphazero-TicTacToe.ipynb)  
  [Tic-Tac-Toe-advanced notebook](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/alphazero/alphazero-TicTacToe-advanced.ipynb)  
  &nbsp;  
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/20240416_alphazero.jpg" width=400>  

---  

## **👉 Unity enviroment `Reacher-v2` vector game (Project Submission)**  

✅ **setup Python environment**   
* [notes for env setup](https://gist.github.com/Nov05/36ed6fff08f16f29c364090844eb1d24)  
* [notes for issues](https://gist.github.com/Nov05/1d49183a91456a63e13782e5f49436be?permalink_comment_id=4935583#gistcomment-4935583)


✅ **entry points**  
* working directory: `$ cd python`   
* [python/experiments/**deeprl_ddpg_continuous.py**](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/experiments/deeprl_ddpg_continuous.py): train and eval  
  `$ python -m experiments.deeprl_ddpg_continuous`  
* [python/experiments/**deeprl_ddpg_plot.py**](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/experiments/deeprl_ddpg_plot.py): plot train and eval scores  
  `$ python -m experiments.deeprl_ddpg_plot`
* launch tensorboard in VSCode: `$ tensorboard --logdir=./data/tf_log`  
  

✅ **Result:** A DDPG model was trained in one Unity-Reacher-v2 environment with 1 agent (1 robot arm) for **155 episodes**, then evaluated in 3 environments (each with 1 agent) parallelly for **150 consecutive episodes** and got an average score of **33.92(0.26)** (0.26 is the standard deviation of scores in different envs). also the trained model is tested to control 20 agents in 4 envs parallelly and got a score of **34.24(0.10)**.    


* evaluation with graphics       
  <img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/2024-04-09_17-35-39_V2.gif?raw=true" width=800>  
  Notes:  
  * the 4 envs and each its own 1 (or 20) agents above were controlled by one single DDPG model at the same time.   
  * observation dimension `[num_envs, num_agents (per env), state_size]` will be converted to `[num_envs*num_agents, state_size]` to pass through the neural networks.   
  * during training, action dimension will be `[mini_batch_size (replay batch), action_size]`;   
           during evaluation, the local network will ouput actions with dimension `[num_envs*num_agents, action_size]`, and it will be converted to `[num_envs, num_agents, action_size]` to step the envs.  


* train and eval scores   
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/20240409_unity-reacher-v2_train_eval_scores.jpg" width=600>
  

* monitor train-eval scores with tensorboard  
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/2024-04-12%2014_34_58-ddpg_unity_reacher_tensorflow.jpg" width=800>


* DDPG neural networks architecture  
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/2024-04-10%2001_13_04-unity-reacher-v2-remark_ddpg_continuous-run-0-240409-123614.log_%20-%20udacity-deep-.jpg" width=500>  


* evaluation result (in 3 envs for 150 consecutive episodes)
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/2024-04-09%2021_48_17-deeprl_ddpg_continuous.py%20-%20udacity-deep-reinforcement-learning%20-%20Visual%20Studio%20.jpg" width=800>  


* saved files (check [the folder](https://github.com/Nov05/udacity-deep-reinforcement-learning/tree/master/python/experiments/ddpg_unity-reacher-v2))  
  * [trained model](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/experiments/ddpg_unity-reacher-v2/DDPGAgent-unity-reacher-v2-remark_ddpg_continuous-run-0-155.model)     
  * [train log](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/experiments/ddpg_unity-reacher-v2/unity-reacher-v2-remark_ddpg_continuous-run-0-240409-123614.log_) (human readable):  
    you can find all the configuration including training **hyperparameters**, **network architecture**, train and eval scores, here.   
  * [tf_log](https://github.com/Nov05/udacity-deep-reinforcement-learning/tree/master/python/experiments/ddpg_unity-reacher-v2/logger-unity-reacher-v2-remark_ddpg_continuous-run-0-240409-123614) (tensorflow log, will be read by the plot modules)
  * [eval log](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/experiments/ddpg_unity-reacher-v2/unity-reacher-v2-remark_ddpg_continuous-run-0-240409-172621.log_) (human readable) 
   

✅ **major efforts in coding**  
* all the code is integrated with `ShangtongZhang`'s [`deeprl`](https://github.com/ShangtongZhang/DeepRL/tree/master/deep_rl) framework which uses some OpenAI `Baselines` functionalities.    
* one task can step multiple envs, either with a single process, or with multiple processes. multiple tasks can be stepped synchronously.
* to enable multiprocessing of Unity environments, the following code has had to be modified.  
  in `python/unityagents/rpc_communicator.py`
  ```python
  class UnityToExternalServicerImplementation(UnityToExternalServicer):
      # parent_conn, child_conn = Pipe() ## removed by nov05
  ...
  class RpcCommunicator(Communicator):
      def initialize(self, inputs: UnityInput) -> UnityOutput: # type: ignore
          try:
              self.unity_to_external = UnityToExternalServicerImplementation()
              self.unity_to_external.parent_conn, self.unity_to_external.child_conn = Pipe() ## added by nov05
  ```
* [Task](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/deeprl/component/envs.py) UML diagram   
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/2024-04-10%2013_06_18-20240410_deeprl_task_uml%20--%20SmartDraw.jpg" width=800>   
  [Agent](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/deeprl/agent/DDPG_agent.py) UML diagram  
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/2024-04-10%2014_20_50-20240410_deeprl_ddpg_agent_uml%20--%20SmartDraw.jpg" width=700>  
  

* **launch multiple Unity environments parallelly (not used in the project)** from an executable file (using Python `Subprocess` and `Multiprocess`, without `MLAgents`)  
  * the major code file [`python\unityagents\environment2.py`](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/unityagents/environment2.py)  
  * check the video of [how to run the code](https://www.youtube.com/watch?v=AYbpY-Wk7N0) ($[`python -m tests2.test_unity_multiprocessing`](https://github.com/Nov05/udacity-deep-reinforcement-learning/blob/master/python/tests2/test_unity_multiprocessing.py))   
  [<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/2024-03-07_08-05-28_reacher_V1-ezgif.com-optimize.gif?raw=true" width=500>](https://www.youtube.com/shorts/z9_dMrkPsz0)  


✅ reference   
* https://arxiv.org/abs/1509.02971  
  <img src="https://raw.githubusercontent.com/Nov05/pictures/master/Udacity/20231221_reinforcement%20learning/20240410_ddpg_arxiv1509.02971.jpg" width=500>  
  
<br>  

---

## **👉 OpenAI Gym's Atari `Pong` pixel game**  

<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/20240219_RL-PPO_pong.gif?raw=true" width=500>

* notebooks 
  * PPO without clipping: [Colab](https://drive.google.com/file/d/17-HyqTB121RjHvJ03GzY81eGxmmpjY3t), [GitHub](https://github.com/Nov05/Google-Colaboratory/blob/master/20240217_pong_REINFORCE.ipynb)   
  * PPO with clipping, [Colab](https://drive.google.com/file/d/1lAvn0_pPyFBnWJ4HPyXfVBhh7qjPo2gP), [GitHub](https://github.com/Nov05/Google-Colaboratory/blob/master/20240218_pong_PPO.ipynb)    

<br>  

---  

## **👉 Unity ML-Agents `Banana Collectors` (Project Submission)**  

<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/p1_navigation_project_submission.gif?raw=true">

1. For this toy game, two `Deep Q-network` methods are tried out. Since the observations (states) are simple (not in pixels), convolutional layers are not in use. And the evaluation results confirm that linear layers are sufficient for solving the problem.   
	* **Double DQN**, with 3 linear layers (hidden dims: 256\*64, later tried with 64\*64)  
	* **Dueling DQN**, with 2 linear layers + 2 split linear layers (hidden dims: 64\*64)  

▪️ The Dueling DQN architecture is displayed as below. 

<table>
<tr>
<th> Dueling Architecture </th>
<th> The green module </th>
</tr>
<tr>
<td>
<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/2024-02-13%2012_08_13-1511.06581.pdf.jpg?raw=true" width=300>  
</td>
<td>
<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/2024-02-13%2012_11_04-1511.06581.pdf.jpg?raw=true" width=300>
</td>
</tr>
</table>  

▪️ Since both the advantage and the value stream propagate gradients to the last convolutional layer in the backward pass, we rescale the combined gradient entering the last convolutional layer by 1/√2. This simple heuristic mildly increases stability.

```Python
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3_adv = nn.Linear(in_features=64, out_features=action_size) ## advantage
        self.layer3_val = nn.Linear(in_features=64, out_features=1) ## state value

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        adv, val = self.layer3_adv(x), self.layer3_val(x)
        return (val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), action_size)) / (2**0.5)
```  

▪️ In addition, we clip the gradients to have their norm less than or equal to 10. This clipping is not standard practice in deep RL, but common in recurrent network training (Bengio et al., 2013).

```Python 
        ## clip the gradients
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.)
        nn.utils.clip_grad_norm_(self.qnetwork_target.parameters(), 10.) 
```

2. The following picture shows the train and eval scores (rewards) for both architectures. Since it is a toy project, trained models are not formally evaluated. We can roughly see that Dueling DQN slightly performs better with **an average score of 17** vs. Double DQN 13 in 10 episodes.  

<img src="https://github.com/Nov05/pictures/blob/master/Udacity/20231221_reinforcement%20learning/p1-project-submission.jpg?raw=true" width=600>  

3. **Project artifacts:** 
	* [All the notebooks](https://gist.github.com/Nov05/4e0ff3edba96928facaff063039c7bce) (trained in Google Colab, evaluated on local machine)  
	* The project folder [`p1_navigation`](https://github.com/Nov05/udacity-deep-reinforcement-learning/tree/master/p1_navigation) (which contains checkpoints `dqn_checkpoint_2000.pth` and `dueling_dqn_checkpoint_2000.pth`)  
	* [Video recording](https://youtu.be/SwAwWLsa9f0?t=35) (which demonstrates how trained models are run on the local machine)  
    * [Project submission repo](https://github.com/Nov05/udacity-drlnd-p1_navigation-submission)  

<br>  

---

## **👉 Logs**  

2024-04-10 p2 Unity Reacher v2 submission   
2024-03-07 Python code to launch multiple Unity environments parallelly from an executable file  
...  
2024-02-14 Banana game project submission  
2024-02-11 Unity MLAgent [Banana env set up](https://gist.github.com/Nov05/bf63ac7e0a2d0f94a635fb3858894cca)  
2024-02-10 repo cloned  

<br>  

---

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Deep Reinforcement Learning Nanodegree

![Trained Agents][image1]

This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  

## Table of Contents

### Tutorials

The tutorials lead you through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.

* [Dynamic Programming](https://github.com/udacity/deep-reinforcement-learning/tree/master/dynamic-programming): Implement Dynamic Programming algorithms such as Policy Evaluation, Policy Improvement, Policy Iteration, and Value Iteration. 
* [Monte Carlo](https://github.com/udacity/deep-reinforcement-learning/tree/master/monte-carlo): Implement Monte Carlo methods for prediction and control. 
* [Temporal-Difference](https://github.com/udacity/deep-reinforcement-learning/tree/master/temporal-difference): Implement Temporal-Difference methods such as Sarsa, Q-Learning, and Expected Sarsa. 
* [Discretization](https://github.com/udacity/deep-reinforcement-learning/tree/master/discretization): Learn how to discretize continuous state spaces, and solve the Mountain Car environment.
* [Tile Coding](https://github.com/udacity/deep-reinforcement-learning/tree/master/tile-coding): Implement a method for discretizing continuous state spaces that enables better generalization.
* [Deep Q-Network](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.
* [Robotics](https://github.com/dusty-nv/jetson-reinforcement): Use a C++ API to train reinforcement learning agents from virtual robotic simulation in 3D. (_External link_)
* [Hill Climbing](https://github.com/udacity/deep-reinforcement-learning/tree/master/hill-climbing): Use hill climbing with adaptive noise scaling to balance a pole on a moving cart.
* [Cross-Entropy Method](https://github.com/udacity/deep-reinforcement-learning/tree/master/cross-entropy): Use the cross-entropy method to train a car to navigate a steep hill.
* [REINFORCE](https://github.com/udacity/deep-reinforcement-learning/tree/master/reinforce): Learn how to use Monte Carlo Policy Gradients to solve a classic control task.
* **Proximal Policy Optimization**: Explore how to use Proximal Policy Optimization (PPO) to solve a classic reinforcement learning task. (_Coming soon!_)
* **Deep Deterministic Policy Gradients**: Explore how to use Deep Deterministic Policy Gradients (DDPG) with OpenAI Gym environments.
  * [Pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum): Use OpenAI Gym's Pendulum environment.
  * [BipedalWalker](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal): Use OpenAI Gym's BipedalWalker environment.
* [Finance](https://github.com/udacity/deep-reinforcement-learning/tree/master/finance): Train an agent to discover optimal trading strategies.

### Labs / Projects

The labs and projects can be found below.  All of the projects use rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents). In the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program, you will receive a review of your project.  These reviews are meant to give you personalized feedback and to tell you what can be improved in your code.

* [The Taxi Problem](https://github.com/udacity/deep-reinforcement-learning/tree/master/lab-taxi): In this lab, you will train a taxi to pick up and drop off passengers.
* [Navigation](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation): In the first project, you will train an agent to collect yellow bananas while avoiding blue bananas.
* [Continuous Control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control): In the second project, you will train an robotic arm to reach target locations.
* [Collaboration and Competition](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet): In the third project, you will train a pair of agents to play tennis! 

### Resources

* [Cheatsheet](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet): You are encouraged to use [this PDF file](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet/cheatsheet.pdf) to guide your study of reinforcement learning. 

## OpenAI Gym Benchmarks

### Classic Control
- `Acrobot-v1` with [Tile Coding](https://github.com/udacity/deep-reinforcement-learning/blob/master/tile-coding/Tile_Coding_Solution.ipynb) and Q-Learning  
- `Cartpole-v0` with [Hill Climbing](https://github.com/udacity/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb) | solved in 13 episodes
- `Cartpole-v0` with [REINFORCE](https://github.com/udacity/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb) | solved in 691 episodes 
- `MountainCarContinuous-v0` with [Cross-Entropy Method](https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb) | solved in 47 iterations
- `MountainCar-v0` with [Uniform-Grid Discretization](https://github.com/udacity/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb) and Q-Learning | solved in <50000 episodes
- `Pendulum-v0` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb)

### Box2d
- `BipedalWalker-v2` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb)
- `CarRacing-v0` with **Deep Q-Networks (DQN)** | _Coming soon!_
- `LunarLander-v2` with [Deep Q-Networks (DQN)](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb) | solved in 1504 episodes

### Toy Text
- `FrozenLake-v0` with [Dynamic Programming](https://github.com/udacity/deep-reinforcement-learning/blob/master/dynamic-programming/Dynamic_Programming_Solution.ipynb)
- `Blackjack-v0` with [Monte Carlo Methods](https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/Monte_Carlo_Solution.ipynb)
- `CliffWalking-v0` with [Temporal-Difference Methods](https://github.com/udacity/deep-reinforcement-learning/blob/master/temporal-difference/Temporal_Difference_Solution.ipynb)

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

## Want to learn more?

<p align="center">Come learn with us in the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
