# Other imports
import sys, os, shutil, time, random
from statistics import stdev

# Import the Cardiff Agent's Directory
from cardiff import *
from cardiff.cage2.Wrappers.BlueTableWrapper import BlueTableWrapper
from cardiff.cage2.Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from cardiff.cage2.Agents.MainAgent import MainAgent

# Import the Mindrake Agent's Directory
from mindrake import *
from mindrake.agents.baseline_sub_agents import BlueTableActionWrapper
from mindrake.agents.baseline_sub_agents import CybORGActionAgent
from mindrake.agents.baseline_sub_agents.CybORGActionAgent import CybORGActionAgent
from mindrake.agents.baseline_sub_agents.loadBanditController import LoadBanditBlueAgent as LoadBlueAgent

# Default import
from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
from CybORG.Agents import B_lineAgent, SleepAgent
from wrapper import CompetitiveWrapper

import gym
from gym import spaces
from gym.spaces import Discrete, MultiBinary

# PPO
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env

# APPO
from ray.rllib.algorithms.appo import APPO, APPOConfig

# IMPALA
from ray.rllib.algorithms.impala import Impala, ImpalaConfig

# DQN
from ray.rllib.algorithms.dqn import DQN, DQNConfig

import numpy as np
from itertools import product
from random import randint, random
from scipy.special import softmax
from statistics import mean

# Flag for setting certain parameters
# Set to TRUE   when running on laptop; mainly for testing setup of code
# Set to FALSE  when running on JupyterHub
laptop = True

# Exploration Rate (Value from 0.0 - 1.0)
# Determines how often the agent should stick to the optimal policy
# 'exploration_rate' = 0 means the agent should always choose the most optimal action
# red_ER = 0
# blu_ER = 0

# Timesteps per game
timesteps = 30

# Agent's training algorithm
algorithm = "ppo"
# algorithm = "impala"
# algorithm = "dqn"

# Set the number of workers, and numGPUs given the laptop flag
workers = 40
ngpus = 1
if(laptop):
    workers = 1
    ngpus = 0

# Training parameters
gae = 1
gamma = 0.99
epochs = 30         
mixer = 0.9         # for training opponent best-response, how many games with current agent policy instead of agent pool

red_lr = 5e-4
red_entropy = 1e-3
red_kl = 1
red_clip_param = 0.3

blue_lr = 5e-4
blue_entropy = 1e-3
blue_kl = 1
blue_clip_param = 0.3

layer_units = 256
model_arch = [layer_units, layer_units]
act_func = "tanh"

experiment_name = "phase1"

# construct the blue and red action spaces
subnets = "Enterprise", "Op", "User"
hostnames = (
    "Enterprise0",
    "Enterprise1",
    "Enterprise2",
    "Op_Host0",
    "Op_Host1",
    "Op_Host2",
    "Op_Server0",
    "User1",
    "User2",
    "User3",
    "User4",)

blue_lone_actions = [["Monitor"]]  # actions with no parameters

blue_host_actions = (
    "Analyse",
    "Remove",
    "Restore",
    "DecoyApache", 
    "DecoyFemitter", 
    "DecoyHarakaSMPT", 
    "DecoySmss", 
    "DecoySSHD", 
    "DecoySvchost", 
    "DecoyTomcat", 
    "DecoyVsftpd",)  # actions with a hostname parameter

red_lone_actions = [["Sleep"], ["Impact"]]  # actions with no parameters
red_network_actions = [
    "DiscoverSystems"]  # actions with a subnet as the parameter
red_host_actions = (
    "DiscoverServices",
    "ExploitServices",
    "PrivilegeEscalate",)

blue_action_list = blue_lone_actions + list(
    product(blue_host_actions, hostnames))

red_action_list = (
    red_lone_actions
    + list(product(red_network_actions, subnets))
    + list(product(red_host_actions, hostnames))) # should be 
cardiff_action_list = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                       132, 2, 15, 24, 25, 26, 27]
mindrake_action_list = list(range(145))

# Batch and mini-batchsizes
b1 = 61440
mb1 = 3840

# Number of actions that red should take in sequence to achieve specific goal
red_action_sequence = 3

# Number of 'timesteps' red gets to take actions to try and achieve the specific action sequence
red_action_tries = pow(len(red_action_list),red_action_sequence) #85184
red_multiplier = 10

b2 = red_action_tries * red_multiplier

print(f'red_action_list is {len(red_action_list)}')
print(f'b2 is {b2}')

mb_scaler = 16
mb2 = b2 // mb_scaler

batch_size = b2
mini_batch_size = mb2

if(laptop):
    batch_size = 100
    mini_batch_size = 10

red_batch_size = batch_size
red_minibatch_size = mini_batch_size
blue_batch_size = batch_size
blue_minibatch_size = mini_batch_size

blue_obs_space = 5*len(hostnames) + timesteps + 1
red_obs_space = len(hostnames) + 3*len(hostnames) + 2*len(subnets) + 2*len(subnets) + 1 + timesteps + 1
cardiff_obs_space = 52
mindrake_obs_space = spaces.Box(-1.0, 1.0, shape=(54,), dtype=np.float32)
mindrake_obs_len = 54

# Declare algorithm configurations after hyperparameters are calculated

# Blu PPO Config
blue_ppo_config = {
    "env": "blue_trainer",
    "num_gpus": ngpus,
    "num_workers": workers,
    "train_batch_size": blue_batch_size,
    "sgd_minibatch_size": blue_minibatch_size,
    'rollout_fragment_length': int(blue_batch_size/workers),
    'num_sgd_iter': epochs,
    'batch_mode': "truncate_episodes",
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers":False},
    "lr": blue_lr,
    "entropy_coeff": blue_entropy,
    "observation_space": MultiBinary(blue_obs_space),
    "action_space": Discrete(len(blue_action_list)),
    "recreate_failed_workers": True,
    'vf_share_layers': False,
    'lambda': gae,
    'gamma': gamma,
    'kl_coeff': blue_kl,
    'kl_target': 0.01,
    'clip_rewards': False,
    'clip_param': blue_clip_param,
    'vf_clip_param': 50.0,
    'vf_loss_coeff': 0.01,
    'log_sys_usage': False,
    'disable_env_checking': True,}

# Red PPO Config
red_ppo_config = {
    "env": "RedTrainer",
    "num_gpus":  ngpus,
    "num_workers": workers,
    "train_batch_size": red_batch_size,
    "sgd_minibatch_size": red_minibatch_size,
    'rollout_fragment_length': int(red_batch_size/workers),
    'num_sgd_iter': epochs,
    'batch_mode': "truncate_episodes",
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers":False},
    "lr": red_lr,
    "entropy_coeff": red_entropy,
    "observation_space": MultiBinary(red_obs_space),
    "action_space": Discrete(len(red_action_list)),
    "recreate_failed_workers": True,
    'vf_share_layers': False,
    'lambda': gae,
    'gamma': gamma,
    'kl_coeff': red_kl,
    'kl_target': 0.01,
    'clip_rewards': False,
    'clip_param': red_clip_param,
    'vf_clip_param': 50.0,
    'vf_loss_coeff': 0.01,
    'log_sys_usage': False,
    'disable_env_checking': True,}

# Blu Opp PPO Config
blu_is_opponent_ppo_config = {
    "num_workers": 0,
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers":False},
    "observation_space": MultiBinary(blue_obs_space),
    "action_space": Discrete(len(blue_action_list)),
    'vf_share_layers': False,
    'log_sys_usage': False,}

# Red Opp PPO Config
red_is_opponent_ppo_config = {
    "num_workers": 0,
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers":False},
    "observation_space": MultiBinary(red_obs_space),
    "action_space": Discrete(len(red_action_list)),
    'vf_share_layers': False,
    'log_sys_usage': False,}

# Blu IMP Config
blue_impala_config = {
    "env": "blue_trainer",
    "num_gpus": ngpus,
    "num_workers": workers,
    "train_batch_size": blue_batch_size,
    # "minibatch_buffer_size": 1,
    "rollout_fragment_length": int(blue_batch_size/workers),
    "num_sgd_iter": epochs,
    "batch_mode": "truncate_episodes",
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "lr": blue_lr,
    "entropy_coeff": blue_entropy,
    "observation_space": MultiBinary(blue_obs_space),
    "action_space": Discrete(len(blue_action_list)),
    "recreate_failed_workers": True,
    "clip_rewards": False,
    "vf_loss_coeff": 0.01,
    "log_sys_usage": False,
    "disable_env_checking": True,

    #########################################
    # Specific Impala Set Parameters        #
    # from impala.py                        #
    # starts at "IMPALA specific settings"  #
    #########################################

    "num_multi_gpu_tower_stacks": 1,
    "vtrace": True,
    "vtrace_drop_last_ts": False,


    # Experience Replay
    # "replay_proportion": 0.5,
    # "replay_buffer_num_slots": blue_batch_size,

    "timeout_s_sampler_manager": 60,
    "learner_queue_timeout": 600,
    "learner_queue_size": 40,}

# Red IMP Config
red_impala_config = {
    "env": "red_trainer",
    "num_gpus": ngpus,
    "num_workers": workers,
    "train_batch_size": red_batch_size,
    # "minibatch_buffer_size": 1,
    "rollout_fragment_length": int(red_batch_size/workers),
    "num_sgd_iter": epochs,
    "batch_mode": "truncate_episodes",
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "lr": red_lr,
    "entropy_coeff": red_entropy,
    "observation_space": MultiBinary(red_obs_space),
    "action_space": Discrete(len(red_action_list)),
    "recreate_failed_workers": True,
    "clip_rewards": False,
    "vf_loss_coeff": 0.01,
    "log_sys_usage": False,
    "disable_env_checking": True,

    #########################################
    # Specific Impala Set Parameters        #
    # from impala.py                        #
    # starts at "IMPALA specific settings"  #
    #########################################

    "num_multi_gpu_tower_stacks": 1,
    "vtrace": True,
    "vtrace_drop_last_ts": False,

    # Experience Replay
    # "replay_proportion": 0.5,
    # "replay_buffer_num_slots": red_batch_size,

    "timeout_s_sampler_manager": 60,
    "learner_queue_timeout": 600,
    "learner_queue_size": 40,}

# Blu Opp IMP Config
blu_is_opponent_impala_config = {
"num_workers": 0,
"num_gpus": 0,
"num_multi_gpu_tower_stacks": 1,
"model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
"observation_space": MultiBinary(blue_obs_space),
"action_space": Discrete(len(blue_action_list)),
"log_sys_usage": False,

# Added
"train_batch_size": blue_batch_size,
# "minibatch_buffer_size": 1,
"rollout_fragment_length": int(blue_batch_size/workers),
"num_sgd_iter": epochs,

"timeout_s_sampler_manager": 60,
"learner_queue_timeout": 600,
"learner_queue_size": 40,}

# Red Opp IMP Config
red_is_opponent_impala_config = {
    "num_workers": 0,
    "num_gpus": 0,
    "num_multi_gpu_tower_stacks": 1,
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "observation_space": MultiBinary(red_obs_space),
    "action_space": Discrete(len(red_action_list)),
    "log_sys_usage": False,

    "train_batch_size": red_batch_size,
    # "minibatch_buffer_size": 2,
    "rollout_fragment_length": int(red_batch_size/workers),
    "num_sgd_iter": epochs,


    "timeout_s_sampler_manager": 60,
    "learner_queue_timeout": 600,
    "learner_queue_size": 40,}

# Blue DQN Config
blue_dqn_config = {
    "env": "blue_trainer",
    "num_gpus": ngpus,
    "num_workers": workers,
    "train_batch_size": blue_batch_size,
    "rollout_fragment_length": int(blue_batch_size/workers),
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "observation_space": MultiBinary(blue_obs_space),
    "action_space": Discrete(len(blue_action_list)),
    "recreate_failed_workers": True,
    "clip_rewards": False,
    "log_sys_usage": False,
    "disable_env_checking": True,

    # Rainbow implementation
    # "n_step": 2,
    # "noisy": True,
    # "num_atoms": 2,
    "v_min": -100.0,
    "v_max": 100.0,}

# Red DQN Config
red_dqn_config = {
    "env": "red_trainer",
    "num_gpus": ngpus,
    "num_workers": workers,
    "train_batch_size": red_batch_size,
    "rollout_fragment_length": int(red_batch_size/workers),
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "observation_space": MultiBinary(red_obs_space),
    "action_space": Discrete(len(red_action_list)),
    "recreate_failed_workers": True,
    "clip_rewards": False,
    "log_sys_usage": False,
    "disable_env_checking": True,

    # Rainbow implementation
    # "n_step": 2,
    # "noisy": True,
    # "num_atoms": 2,
    "v_min": -100.0,
    "v_max": 100.0,}

# Blu Opp DQN Config
blue_is_opponent_dqn_config = {
    "num_workers": 0,
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "observation_space": MultiBinary(blue_obs_space),
    "action_space": Discrete(len(blue_action_list)),
    'log_sys_usage': False,}

# Red Opp DQN Config
red_is_opponent_dqn_config = {
    "num_workers": 0,
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "observation_space": MultiBinary(red_obs_space),
    "action_space": Discrete(len(red_action_list)),
    'log_sys_usage': False,}

class BlueTrainer(gym.Env):
    def __init__(self, env_config):

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        self.action_space = Discrete(len(blue_action_list))
        self.observation_space = MultiBinary(blue_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return  observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        self.red_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),False))
        self.opponent_id = 0
        
    def reset(self):

        # This is how we are selecting the opposing policy from the opponent pool: Uniform Sampling
        pool_file = open(f"./policies/{algorithm}/{timesteps}/red_opponent_pool/pool_size", "r")
        red_pool_size = int(pool_file.read())
        pool_file.close()

        if red_pool_size > 0:
            self.opponent_id = randint(1, red_pool_size)
        else:
            self.opponent_id = 0
        
        path_file = open(f"./policies/{algorithm}/{timesteps}/red_opponent_pool/opponent_red_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        info = {}
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True  

        return (obs, reward, done, info)
    
class BlueOpponent(gym.Env):

    def __init__(self, env_config):

        print("Creating blue opponent")

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        self.action_space = Discrete(len(blue_action_list))
        self.observation_space = MultiBinary(blue_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return  observations

        print("calling competitive wrapper")
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        self.red_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),False))
        self.opponent_id = 0
        
    def reset(self):

        # This is how we are choosing and opposing policy from the opponent pool: Uniform Sampling
        pool_file = open(f"./policies/{algorithm}/{timesteps}/red_competitive_pool/pool_size", "r")
        red_pool_size = int(pool_file.read())
        pool_file.close()

        if ((red_pool_size > 1) and (random() > mixer)):
            self.opponent_id = randint(1, red_pool_size-1)
        else:
            self.opponent_id = red_pool_size
        
        path_file = open(f"./policies/{algorithm}/{timesteps}/red_competitive_pool/competitive_red_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()    
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        info = {}
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True  

        return (obs, reward, done, info)

class DedicatedBlueEnv(gym.Env):
    def __init__(self, env_config):

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        self.action_space = Discrete(len(blue_action_list))
        self.observation_space = MultiBinary(blue_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return  observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        self.red_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),False))
        
    def reset(self):

        path_file = open(f"./policies/{algorithm}/{timesteps}/competitive_red_policy", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()    
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)

class CardiffBlueEnv(gym.Env):
    def __init__(self, env_config):

        print("creating CardiffBlueEnv")

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        # self.action_space = Discrete(len(blue_action_list))                                   # CHANGE

        # this current action space is a list of numbers (most likely enums from the CybORG action space)
        self.action_space = Discrete(len(cardiff_action_list))
        self.observation_space = Discrete(cardiff_obs_space)                                    # CHANGE

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)

        # wrapper to accept red and blue actions, and return  observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        self.red_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),False))
        
    def reset(self):

        path_file = open(f"./policies/{algorithm}/{timesteps}/competitive_red_policy", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()    
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)

class MindrakeBlueEnv(gym.Env):
    def __init__(self, env_config):

        print("creating MindrakeBlueEnv")

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        # self.action_space = Discrete(len(blue_action_list))                                   # CHANGE

        # this current action space is a list of numbers (most likely enums from the CybORG action space)
        self.action_space = Discrete(len(mindrake_action_list))
        
        # self.observation_space = mindrake_obs_space                                    # CHANGE
        self.observation_space = MultiBinary(mindrake_obs_len)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)

        # wrapper to accept red and blue actions, and return  observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        self.red_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),False))
        
    def reset(self):

        path_file = open(f"./policies/{algorithm}/{timesteps}/competitive_red_policy", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()    
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)
    
class RedTrainer(gym.Env):
    def __init__(self, env_config):

        print("Creating red trainer")
        
        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        self.blue_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),True))
        self.opponent_id = 0

    def reset(self):

        pool_file = open(f"./policies/{algorithm}/{timesteps}/blue_opponent_pool/pool_size", "r")
        blue_pool_size = int(pool_file.read())
        pool_file.close()

        if blue_pool_size > 0:
            self.opponent_id = randint(1, blue_pool_size)
        else:
            self.opponent_id = blue_pool_size
        
        path_file = open(f"./policies/{algorithm}/{timesteps}/blue_opponent_pool/opponent_blue_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.blue_opponent.restore(checkpoint_path)
    
        self.blue_obs, obs = self.cyborg.reset()   
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action):

        blue_action = self.blue_opponent.compute_single_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent
        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)

class RedOpponent(gym.Env):
    def __init__(self, env_config):
        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        self.blue_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),True))
        self.opponent_id = 0

    def reset(self):

        pool_file = open(f"./policies/{algorithm}/{timesteps}/blue_competitive_pool/pool_size", "r")
        blue_pool_size = int(pool_file.read())
        pool_file.close()

        if ((blue_pool_size > 1) and (random() > mixer)):
            self.opponent_id = randint(1, blue_pool_size-1)
        else:
            self.opponent_id = blue_pool_size
        
        path_file = open(f"./policies/{algorithm}/{timesteps}/blue_competitive_pool/competitive_blue_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.blue_opponent.restore(checkpoint_path)
    
        self.blue_obs, obs = self.cyborg.reset()   
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action):

        blue_action = self.blue_opponent.compute_single_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent
        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)

class DedicatedRedEnv(gym.Env):
    def __init__(self, env_config):
        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        self.blue_opponent = get_opponent(get_algorithm_select(),get_opponent_config(get_algorithm_select(),True))

    def reset(self):

        path_file = open(f"./policies/{algorithm}/{timesteps}/competitive_blue_policy", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.blue_opponent.restore(checkpoint_path)

        self.blue_obs, obs = self.cyborg.reset()   
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action, verbose=False):

        blue_action = self.blue_opponent.compute_single_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent
        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)

class _DedicatedRedEnv_vs_cardiff(gym.Env):
    def __init__(self, env_config):

        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        self.blue_opponent = MainAgent()

    def reset(self):

        self.blue_opponent = MainAgent()
        self.blue_obs, obs = self.cyborg.reset(cardiff=True)
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action, verbose=False):

        # modified for cardiff
        blue_action = self.blue_opponent.get_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action,
            cardiff=True
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent

        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:

            # based on this logic we won't have to worry about
            # setting back cardiff's inital actions because it get's handled
            # in reset()
            done = True
        
        info = {}

        return (obs, reward, done, info)

# NEED TO UPDATE for 'mindrake' agent!
class _DedicatedRedEnv_vs_mindrake(gym.Env):
    def __init__(self, env_config):

        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        self.blue_opponent = MainAgent()

    def reset(self):

        self.blue_opponent = MainAgent()
        self.blue_obs, obs = self.cyborg.reset(cardiff=True)
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action, verbose=False):

        # modified for cardiff
        blue_action = self.blue_opponent.get_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action,
            cardiff=True
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent

        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:

            # based on this logic we won't have to worry about
            # setting back cardiff's inital actions because it get's handled
            # in reset()
            done = True
        
        info = {}

        return (obs, reward, done, info)
# NEED TO UPDATE for 'mindrake' agent!

def build_cardiff_agent():
    
    select_env = "CardiffBlueEnv"
    register_env(
        select_env,
        lambda config: CardiffBlueEnv(
            env_config={"name": f"{experiment_name}_cardiff_blue"}
        )
    )
    return MainAgent()

def build_mindrake_agent():

    select_env = "MindrakeBlueEnv"
    register_env(
        select_env,
        lambda config: MindrakeBlueEnv(
            env_config={"name": f"{experiment_name}_mindrake_blue"}
        )
    )
    return LoadBlueAgent()

def wrap_mindrake_agent():
    return CybORGActionAgent(config)

def build_blue_agent(fresh, opponent=False, dedicated=False):

    # register the custom environment
    if dedicated:
        select_env = "DedicatedBlueEnv"
        register_env(
            select_env,
            lambda config: DedicatedBlueEnv(
                env_config={"name": f"{experiment_name}_dedicated_blue"}
            )
        )
    elif opponent:
        select_env = "BlueOpponent"
        register_env(
            select_env,
            lambda config: BlueOpponent(
                env_config={"name": f"{experiment_name}_opponent_blue"}
            )
        )
    else:
        select_env = "BlueTrainer"
        register_env(
            select_env,
            lambda config: BlueTrainer(
                env_config={"name": f"{experiment_name}_competitive_blue"}
            )
        )

    # set the RLLib configuration
    blue_config = get_algorithm_config(algorithm,True)

    if dedicated:
        blue_agent = run_algorithm(config=blue_config, env=DedicatedBlueEnv, algorithm_select=algorithm)
        if fresh:
            checkpoint_path = blue_agent.save(checkpoint_dir=f"./policies/{algorithm}/{timesteps}/blue_dedicated_pool/dedicated_blue_0")
            print(checkpoint_path)
            path_file = open(f"./policies/blue_dedicated_pool/{algorithm}/{timesteps}/dedicated_blue_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            path_file = open(f"./policies/blue_dedicated_pool/{algorithm}/{timesteps}/pool_size", "w")
            path_file.write("0")
            path_file.close()
    elif opponent:
        blue_agent = run_algorithm(config=blue_config, env=BlueOpponent, algorithm_select=algorithm)
        if fresh:
            checkpoint_path = blue_agent.save(checkpoint_dir=f"./policies/{algorithm}/{timesteps}/blue_opponent_pool/opponent_blue_0")
            print(checkpoint_path)
            path_file = open(f"./policies/{algorithm}/{timesteps}/blue_opponent_pool/opponent_blue_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            path_file = open(f"./policies/{algorithm}/{timesteps}/blue_opponent_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()
    else:
        blue_agent = run_algorithm(config=blue_config, env=BlueTrainer, algorithm_select=algorithm)
        if fresh:
            checkpoint_path = blue_agent.save(checkpoint_dir=f"./policies/{algorithm}/{timesteps}/blue_competitive_pool/competitive_blue_0")
            print(checkpoint_path)
            path_file = open(f"./policies/{algorithm}/{timesteps}/blue_competitive_pool/competitive_blue_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            path_file = open(f"./policies/{algorithm}/{timesteps}/blue_competitive_pool/pool_size", "w")
            path_file.write("0")
            path_file.close() 
    return blue_agent

def build_red_agent(fresh, opponent=False, dedicated=False, vs_cardiff=False, vs_mindrake=False):
    # register the custom environment
    if vs_cardiff:
        select_env = "DedicatedRedEnv_vs_cardiff"
        register_env(
            select_env,
            lambda config: DedicatedRedEnv_vs_cardiff(
                env_config={"name": f"{experiment_name}_DedicatedRedEnv_vs_cardiff"}
            )
        )
    elif vs_mindrake:
        select_env = "DedicatedRedEnv_vs_mindrake"
        register_env(
            select_env,
            lambda config: DedicatedRedEnv_vs_mindrake(
                env_config={"name": f"{experiment_name}_DedicatedRedEnv_vs_mindrake"}
            )
        )
    elif dedicated:
        select_env = "DedicatedRedEnv"
        register_env(
            select_env,
            lambda config: DedicatedRedEnv(
                env_config={"name": f"{experiment_name}_dedicated_red"}
            )
        )
    elif opponent:
        select_env = "RedOpponent"
        register_env(
            select_env,
            lambda config: RedOpponent(
                env_config={"name": f"{experiment_name}_opponent_red"}
            )
        )    
    else:
        select_env = "RedTrainer"
        register_env(
            select_env,
            lambda config: RedTrainer(
                env_config={"name": f"{experiment_name}_competitive_red"}
            )
        )

    # set the RLLib configuration
    red_config = get_algorithm_config(algorithm,False)

    if vs_cardiff:
        red_agent = run_algorithm(config=red_config, env=_DedicatedRedEnv_vs_cardiff,algorithm_select=algorithm)
        if fresh:
            checkpoint_path = red_agent.save(checkpoint_dir=f"./policies/{algorithm}/{timesteps}/red_dedicated_pool/dedicated_red_0")
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_dedicated_pool/dedicated_red_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            print(checkpoint_path)
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_dedicated_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()    
    elif dedicated:
        red_agent = run_algorithm(config=red_config, env=DedicatedRedEnv,algorithm_select=algorithm)
        if fresh:
            checkpoint_path = red_agent.save(checkpoint_dir=f"./policies/{algorithm}/{timesteps}/red_dedicated_pool/dedicated_red_0")
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_dedicated_pool/dedicated_red_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            print(checkpoint_path)
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_dedicated_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()
    elif opponent:
        red_agent = run_algorithm(config=red_config, env=RedOpponent,algorithm_select=algorithm)
        if fresh:
            checkpoint_path = red_agent.save(checkpoint_dir=f"./policies/{algorithm}/{timesteps}/red_opponent_pool/opponent_red_0")
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_opponent_pool/opponent_red_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            print(checkpoint_path)
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_opponent_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()
    else:
        red_agent = run_algorithm(config=red_config, env=RedTrainer,algorithm_select=algorithm)
        if fresh:
            checkpoint_path = red_agent.save(checkpoint_dir=f"./policies/{algorithm}/{timesteps}/red_competitive_pool/competitive_red_0")
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_competitive_pool/competitive_red_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            print(checkpoint_path)
            path_file = open(f"./policies/{algorithm}/{timesteps}/red_competitive_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()     
    return red_agent

def sample(test_red, test_blue, games=1, verbose=False, show_policy=False, blue_moves=None, red_moves=None, random_blue=False, random_red=False):
    
    base_cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
    
    # wrapper to accept red and blue actions, and return observations
    cyborg = CompetitiveWrapper(env=base_cyborg, turns=timesteps, output_mode="vector")

    scores = []
    max_score = 0
    min_score = float('inf')

    for g in range(games):

        blue_obs, red_obs = cyborg.reset()
        score = 0

        if verbose and (games>1):
            print(f"-------- Game {g+1} --------")
        
        if random_blue:
            blue_moves = []
            for t in range(timesteps):
                blue_moves.append(randint(1, len(blue_action_list)-1))
        
        if random_red:
            red_moves = []
            for t in range(timesteps):
                red_moves.append(randint(1, len(red_action_list)-1))

        for t in range(timesteps):

            if blue_moves is None:
                blue_action, _, blue_extras = test_blue.compute_single_action(blue_obs, full_fetch=True)
            else:
                blue_action = blue_moves[t]
                blue_extras = {'action_dist_inputs':np.zeros(len(blue_action_list)), 'action_prob':1}

            if red_moves is None:
                red_action, _, red_extras = test_red.compute_single_action(red_obs, full_fetch=True)
            else:
                red_action = red_moves[t]
                red_extras = {'action_dist_inputs':np.zeros(len(red_action_list)), 'action_prob':1}

            state = cyborg.step(red_action, blue_action)

            red_reward = -state.reward

            blue_obs = state.blue_observation
            red_obs = state.red_observation

            score += red_reward

            if verbose:
                if 'policy' in blue_extras:
                    blue_policy = blue_extras['policy']
                else:
                    blue_policy = softmax(blue_extras['action_dist_inputs'])
                if 'policy' in red_extras:
                    red_policy = red_extras['policy']
                else:
                    red_policy = softmax(red_extras['action_dist_inputs'])

                print(f'---- Turn {t+1} ----')
                if show_policy:
                    print("Blue policy: ")
                    for a in range(len(blue_action_list)):
                        print(f"{blue_action_list[a]}: {blue_policy[a]*100:0.2f}%")
                print(f"Blue selects {blue_action_list[blue_action]} with probability {blue_extras['action_prob']*100:0.2f}%")
                print()
                if show_policy:
                    print(f"Red Policy: ")
                    for a in range(len(red_action_list)):
                        print(f"{red_action_list[a]}: {red_policy[a]*100:0.2f}%")
                print(f"Red selects {red_action_list[red_action]} with probability {red_extras['action_prob']*100:0.2f}%")
                print()
                print(f'New Red observation: {red_obs}')
                print(cyborg._create_red_table())
                print()
                print(f"Reward: +{red_reward:0.1f}")
                print(f"Score: {score:0.1f}")
                print()
        
        scores.append(score)
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
    
    avg_score = mean(scores)
    if verbose and (games>1):
        print(f'Average Score for {games} Games is {avg_score}')
        print(f'High Score is {max_score}')
        print(f'Low Score is {min_score}')
    
    return(avg_score)

def sample_against_cardiff(test_red, test_blue, games=1, verbose=False):

    # Code to save game interactions if 'games' folder does not exist #
    # Create a folder to save trial games against mindrake at a specific timestep
    game_log_folder = 'games'
    if os.path.exists(game_log_folder):
        pass
    else:
        os.makedirs(game_log_folder)
    if os.path.exists(os.path.join(game_log_folder,'cardiff',str(timesteps))):
        # Clear the old log and create a new one
        shutil.rmtree(os.path.join(game_log_folder,'cardiff',str(timesteps)))
    os.makedirs(os.path.join(game_log_folder,'cardiff',str(timesteps)))
    # Code to save game interactions if 'games' folder does not exist #

    base_cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
    
    # wrapper to accept red and blue actions, and return observations
    cyborg = CompetitiveWrapper(env=base_cyborg, turns=timesteps, output_mode="vector")

    # Keep track of the game and step of where the mindrake agent has encountered an unknown observation
    # Track the Game and the Step
    unknown_obs = {}
    unknown_obs_count = 0

    # Keep track of the number of actions that occur in the game
    actions_taken = {}

    # Keep track of the number of states encountered
    states_encountered = {}

    # Keep track of defenses
    agent_defense = {}

    # new code
    global_bline = {}
    history_bline= {}
    global_meander = {}
    history_meander = {}

    scores = []
    max_score = 0
    min_score = float('inf')
    start_time = time.time()

    for g in range(games):

        # the blue_obs given by 'cyborg.reset()' is not the same as the one given by default cyborg's reset()
        blue_obs, red_obs = cyborg.reset(cardiff=True)
        test_blue.end_episode()
        score = 0

        if verbose and (games>1):
            print(f"-------- Game {g+1} --------")

        # Track the agent defense selected for the game!
        selected_agent_defense = None

        for t in range(timesteps):

            # Set game log file with initial red observation
            if t == 0:
                # Grab initial red observation
                file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),str(g))
                original_stdout = sys.stdout
                try:
                    with open(file, 'a') as file:
                        true_state = cyborg.get_agent_state('True')
                        true_table = true_obs_to_table(true_state,cyborg)
                        sys.stdout = file
                        print(f"Initial Observation")
                        print(true_table)
                        print("\n\n")
                finally:
                    # Restore original stdout no matter what
                    sys.stdout = original_stdout

            # Track the observation of the agent
            key = str(blue_obs)
            key = key.replace('\n', '')
            if key not in states_encountered:
                states_encountered[key] = 1
            else:
                count = states_encountered[key]
                states_encountered[key] = count + 1

            blue_action, agent_to_select = test_blue.get_action(blue_obs)
            translated_action = cyborg.convert_to_original_blue_action(blue_action)
            red_action, _, red_extras = test_red.compute_single_action(red_obs, full_fetch=True)
            if agent_to_select == 0:
                agent_to_select = 'Meander'
            elif agent_to_select == 1:
                agent_to_select = 'BLine'
            else:
                agent_to_select = 'Sleep'

            # Track the agent selection
            # Set the trigger 't' to be 4 bc cardiff uses 3 starting actions before selecting a defense
            if t == 4:
                key = agent_to_select
                selected_agent_defense = key
                if key not in agent_defense:
                    defense_history = [str(f'Game {g}')]
                    defense_score = []
                    agent_defense[key] = (1,defense_score, defense_history)
                else:
                    current_count, current_score, current_history = agent_defense[key]
                    current_game = str(f'Game {g}')
                    current_history.append(current_game)
                    agent_defense[key] = (current_count + 1, current_score, current_history)

            """
            # Track the action
            key = blue_action
            if key not in actions_taken:
                action_history = [str(f'Game {g} Step {t}')]
                actions_taken[key] = (1, translated_action, action_history)
            else:
                count, _, action_history = actions_taken[key]
                current_action = str(f'Game {g} Step {t}')
                action_history.append(current_action)
                actions_taken[key] = (count + 1, translated_action,action_history)
            """
            # Track agent performance based on defense
            key = agent_to_select

            # Track global observations and actions
            if key == 'BLine':
                update_history(history_bline,blue_action,translated_action,g,t)
                update_dictionary(global_bline,blue_obs,blue_action,translated_action,g,t)
            else:
                update_history(history_meander,blue_action,translated_action,g,t)
                update_dictionary(global_meander,blue_obs,blue_action,translated_action,g,t)

            state = cyborg.step(red_action, blue_action, cardiff=True)
            red_reward = -state.reward
            blue_obs = state.blue_observation
            red_obs = state.red_observation
            score += red_reward

            # Update the game log file
            file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),str(g))
            original_stdout = sys.stdout

            try:
                with open(file, 'a') as file:
                    true_state = cyborg.get_agent_state('True')
                    true_table = true_obs_to_table(true_state,cyborg)
                    sys.stdout = file
                    print(f"Step{' ' * 7}\t:{' ' * 2}{t}")
                    print("Blu def.   \t: ",str(agent_to_select))
                    print("Blu last action : ",str(cyborg.get_last_action('Blue')))
                    print("Red last action : ",str(cyborg.get_last_action('Red')))
                    print(true_table)
                    print("Red reward\t:",red_reward)
                    print("Total reward\t:",score)
                    print("\n\n")
            finally:
                sys.stdout = original_stdout

            if verbose:
                print(f'---- Turn {t+1} ----')
                # Cardiff actions are deterministic not stochastic!
                print(f"Blue selects {blue_action_list[cyborg.convert_blue_action(blue_action,cardiff=True)]} with probability {1*100:0.2f}%")
                print()
                print(f"Red selects {red_action_list[red_action]} with probability {red_extras['action_prob']*100:0.2f}%")
                print()
                # print(f'New Red observation: {red_obs}')
                print(f'New Red observation: ')
                print(cyborg._create_red_table())
                print()
                print(f"Reward: +{red_reward:0.1f}")
                print(f"Score: {score:0.1f}")
                print()

        # Record the score!
        current_count, current_score, current_history = agent_defense[selected_agent_defense]
        current_score.append(score)
        agent_defense[selected_agent_defense] = (current_count, current_score, current_history)

        file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),"_scores")
        with open(file, 'a') as file:
            file.write(f'Game {g} - score is {score}\n')

        scores.append(score)
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
    
    avg_score = mean(scores)
    elapsed_time = time.time()-start_time

    # Calculate the % of BLine & Meander Defenses
    num_BLine_defenses = agent_defense['BLine'][0]
    num_Meander_defenses = agent_defense['Meander'][0]
    num_total_defenses = num_BLine_defenses + num_Meander_defenses
    percentage_BLine_defenses = (num_BLine_defenses / num_total_defenses) * 100
    percentage_Meander_defenses = (num_Meander_defenses / num_total_defenses) * 100

    # Calculate the isolated scores of BLine & Meander Defenses
    avg_BLine_score = mean(agent_defense['BLine'][1])
    avg_Meander_score = mean(agent_defense['Meander'][1])
    stddev_BLine_score = stdev(agent_defense['BLine'][1])
    stddev_Meander_score = stdev(agent_defense['Meander'][1])

    # Find minimum and maximum along with their indices
    min_value, min_index = min((val, idx) for idx, val in enumerate(scores))
    max_value, max_index = max((val, idx) for idx, val in enumerate(scores))
    print("\n\nvs_cardiff")
    print(f'Total time is {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
    print(f'Total games played is {games}')
    print(f'Timestpes per game is {timesteps}')
    print(f'BLine defense ({num_BLine_defenses}/{num_total_defenses}) % is {percentage_BLine_defenses}')
    print(f'Meander defense ({num_Meander_defenses}/{num_total_defenses}) % is {percentage_Meander_defenses}')
    print(f'Bline defense average reward is {avg_BLine_score}')
    print(f'Meander defense average reward is {avg_Meander_score}')
    print(f'Standard deviation of Bline defense reward is {stddev_BLine_score}')
    print(f'Standard deviation of Bline defense reward is {stddev_Meander_score}')
    print(f'Avg reward is {avg_score}')
    print(f'Min reward is {min_value} at Game {min_index}')
    print(f'Max reward is {max_value} at Game {max_index}')
    print(f'Standard deviation of {stdev(scores)}')
    print(f'Total number of states_encountered is {len(states_encountered)}')

    print(f'\nBLine Action Count {len(history_bline)}/145\n')
    sorted_bline = sorted(history_bline.items())
    for key, value in sorted_bline:
        num_of_times_action_called = value[0]
        action_name = value[1]
        print(f'called {num_of_times_action_called}\ttimes -\t{action_name}')

    print(f'\nMeander Action Count {len(history_meander)}/145\n')
    sorted_meander = sorted(history_meander.items())
    for key, value in sorted_meander:
        num_of_times_action_called = value[0]
        action_name = value[1]
        print(f'called {num_of_times_action_called}\ttimes -\t{action_name}')

    """
    print(f'Total number of unique actions taken is {len(actions_taken)}/145\n')
    sorted_actions = sorted(actions_taken.items())
    for key, value in sorted_actions:
        num_of_times_action_called = value[0]
        action_name = value[1]
        print(f'called {num_of_times_action_called}\ttimes -\t{action_name}')
    """

    print(f'\nTotal number of games that encountered unknown_obs is {len(unknown_obs)}')
    print(f'Total number of unknown_obs encountered is {unknown_obs_count}\n')
    if(len(unknown_obs) > 0):
        for key, values in unknown_obs.items():
            print(f'{key} encountered {len(values)} unknown_obs')

    # Add the logged information into the game's log file
    file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),"_log")
    with open(file, 'w') as file:
        file.write(f'vs_cardiff\n')
        file.write(f'Total time is {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n')
        file.write(f'Total games played is {games}\n')
        file.write(f'Timestpes per game is {timesteps}\n')
        file.write(f'BLine defense ({num_BLine_defenses}/{num_total_defenses}) % is {percentage_BLine_defenses}\n')
        file.write(f'Meander defense ({num_Meander_defenses}/{num_total_defenses}) % is {percentage_Meander_defenses}\n')
        file.write(f'Bline defense average reward is {avg_BLine_score}\n')
        file.write(f'Meander defense average reward is {avg_Meander_score}\n')
        file.write(f'Standard deviation of Bline defense reward is {stddev_BLine_score}\n')
        file.write(f'Standard deviation of Meander defense reward is {stddev_Meander_score}\n')
        file.write(f'Avg reward is {avg_score}\n')
        file.write(f'Min reward is {min_value} at Game {min_index}\n')
        file.write(f'Max reward is {max_value} at Game {max_index}\n')
        file.write(f'Standard deviation of {stdev(scores)}\n\n')
        file.write(f'Total number of states_encountered is {len(states_encountered)}\n')

        file.write(f'\nBLine Action Count {len(history_bline)}/145\n')
        for key, value in sorted_bline:
            num_of_times_action_called = value[0]
            action_name = value[1]
            file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

        file.write(f'\nMeander Action Count {len(history_meander)}/145\n')
        for key, value in sorted_meander:
            num_of_times_action_called = value[0]
            action_name = value[1]
            file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

        """  
        file.write(f'Total number of unique actions taken is {len(actions_taken)}/145\n\n')
        for key, value in sorted_actions:
            num_of_times_action_called = value[0]
            num_spaces = 0
            if (num_of_times_action_called < 100):
                num_spaces += 1
                if(num_of_times_action_called < 10):
                    num_spaces +=1
            action_name = value[1]
            file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')
        """
    
        file.write(f'\nTotal number of games that encountered unknown_obs is {len(unknown_obs)}')
        file.write(f'\nTotal number of unknown_obs encountered is {unknown_obs_count}\n')
        if(len(unknown_obs) > 0):
            for key, values in unknown_obs.items():
                file.write(f'{key} encountered {len(values)} unknown_obs\n')

    # Track which games have 'BLine' and 'Meander Defenses'
    file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),"_defenses")
    with open(file, 'w') as file:
        for key, value in agent_defense.items():
            file.write(f'{key}\n')
            agent_defense_history = value[2]
            for event in agent_defense_history:
                file.write(f'{event}\n')
            file.write('\n')

    file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),"_states")
    with open(file, 'w') as file:
        for key, value in states_encountered.items():
            file.write(f'{key}\n')

    file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),"_actions")
    with open(file, 'w') as file:

        file.write(f'BLine Action Count {len(history_bline)}/145\n')
        for key, value in sorted_bline:
            num_of_times_action_called = value[0]
            action_name = value[1]
            file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

        file.write(f'\nMeander Action Count {len(history_meander)}/145\n')
        for key, value in sorted_meander:
            num_of_times_action_called = value[0]
            action_name = value[1]
            file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

        file.write("\n\n--BLINE--\n\n")

        for key, value in global_bline.items():
            file.write(f'{key}\n')
            for inside_key, inside_value in value.items():
                action_count = inside_value[0]
                action_name = inside_value[1]
                action_history = inside_value[2]
                file.write(f'{action_name} took place {action_count} times\n')
                for event in action_history:
                    file.write(f'{event}\n')
                file.write('\n')

        file.write("\n\n-MEANDER-\n\n\n\n")

        for key, value in global_meander.items():
            file.write(f'{key}\n')
            for inside_key, inside_value in value.items():
                action_count = inside_value[0]
                action_name = inside_value[1]
                action_history = inside_value[2]
                file.write(f'{action_name} took place {action_count} times\n')
                for event in action_history:
                    file.write(f'{event}\n')
                file.write('\n')

    # Explicit report for the Game and Steps in which the agent encountered the unknown observation
    file = os.path.join(os.getcwd(),game_log_folder,"cardiff",str(timesteps),"_unknown-obs")
    with open(file, 'w') as file:
        for key, values in unknown_obs.items():
            for element in unknown_obs[key]:
                file.write(f'{key} {element}\n')
            file.write(f'\n\n')

    if verbose and (games>1):
        print(f'Average Score for {games} Games is {avg_score}')
        print(f'High Score is {max_score}')
        print(f'Low Score is {min_score}')
    
    # Default return value
    # return(avg_score)
    print(avg_score, '\n')
    print("sample_against_cardiff completed!\n")

def sample_against_mindrake(test_red, test_blue, games=1, verbose=False):

    # Code to save game interactions if 'games' folder does not exist #
    # Create a folder to save trial games against mindrake at a specific timestep
    game_log_folder = 'games'
    if os.path.exists(game_log_folder):
        pass
    else:
        os.makedirs(game_log_folder)

    # Loop through the different exploration rates
    for red_ER in [0.0]:
        for blu_ER in [0.0]:

            # Clear the old log and create a new one
            er_fileName = f'red(ER) is {int(red_ER * 100)}% and blu(ER) is {int(blu_ER * 100)}%'

            if os.path.exists(os.path.join(game_log_folder,'mindrake',str(timesteps),er_fileName)):
                shutil.rmtree(os.path.join(game_log_folder,'mindrake',str(timesteps),er_fileName))
            os.makedirs(os.path.join(game_log_folder,'mindrake',str(timesteps),er_fileName))
            # Code to save game interactions if 'games' folder does not exist #

            base_cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
            
            # wrapper to accept red and blue actions, and return observations
            cyborg = CompetitiveWrapper(env=base_cyborg, turns=timesteps, output_mode="vector")

            # Keep track of the game and step of where the mindrake agent has encountered an unknown observation
            # Track the Game and the Step
            unknown_obs = {}
            unknown_obs_count = 0

            # Keep track of the number of actions that occur in the game
            actions_taken = {}

            # Keep track of the number of states encountered
            states_encountered = {}

            # Keep track of the agent select for performance
            agent_defense = {}

            # new code
            global_bline = {}
            history_bline= {}            
            global_meander = {}
            history_meander = {}

            scores = []
            max_score = 0
            min_score = float('inf')
            start_time = time.time()

            for g in range(games):

                # the blue_obs given by 'cyborg.reset()' is not the same as the one given by default cyborg's reset()
                blue_obs, red_obs = cyborg.reset(mindrake=True)
                test_blue.end_episode()
                score = 0

                if verbose and (games>1):
                    print(f"-------- Game {g+1} --------")

                # Track the agent defense selected for the game!
                selected_agent_defense = None

                for t in range(timesteps):

                    red_action_random = False
                    blu_action_random = False

                    # Set game log file with initial red observation
                    if t == 0:
                        # Grab initial red observation
                        file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,str(g))
                        original_stdout = sys.stdout
                        try:
                            with open(file, 'a') as file:
                                true_state = cyborg.get_agent_state('True')
                                true_table = true_obs_to_table(true_state,cyborg)
                                sys.stdout = file
                                print(f"Initial Observation")
                                print(true_table)
                                print("\n\n")
                        finally:
                            # Restore original stdout no matter what
                            sys.stdout = original_stdout

                    # Track the observation of the agent
                    key = str(blue_obs)
                    key = key.replace('\n', '')
                    if key not in states_encountered:
                        states_encountered[key] = 1
                    else:
                        count = states_encountered[key]
                        states_encountered[key] = count + 1

                    blue_action, agent_to_select = test_blue.get_action(blue_obs,len(mindrake_action_list))
                    translated_action = cyborg.convert_to_original_blue_action(blue_action)
                    red_action, _, red_extras = test_red.compute_single_action(red_obs, full_fetch=True)


                    """
                    # Track the action
                    key = blue_action
                    if key not in actions_taken:
                        action_history = [str(f'Game {g} Step {t}')]
                        actions_taken[key] = (1, translated_action, action_history)
                    else:
                        count, _, action_history = actions_taken[key]
                        current_action = str(f'Game {g} Step {t}')
                        action_history.append(current_action)
                        actions_taken[key] = (count + 1, translated_action,action_history)
                    """

                    # Code to handle exploration action choices
                    if(red_ER > 0):

                        # Generate a random float value between 0.0 (inclusive) and 1.0 (inclusive)
                        rv = randint(1, 100)

                        # If the random float value is less than the exploration rate (i.e .10), then choose a random action
                        if(rv <= red_ER*100):
                            red_action = randint(0,len(red_action_list)-1)
                            red_action_random = True
                    if(blu_ER > 0):

                        # Generate a random float value between 0.0 (inclusive) and 1.0 (inclusive)
                        rv = randint(1, 100)

                        # If the random float value is less than the exploration rate (i.e .10), then choose a random action
                        if(rv <= blu_ER*100):
                            blu_action = randint(0,len(blue_action_list)-1)
                            red_action_random = True

                    # Record Agent Selection
                    if agent_to_select == 0:
                        agent_to_select = 'BLine'
                    elif agent_to_select == 1:
                        agent_to_select = 'Meander'
                    elif agent_to_select == -3:
                        agent_to_select = 'FORCED BLine'
                        key = f'Game {g}'
                        value = f'Step {t}'
                        if key not in unknown_obs:
                            unknown_obs[key] = []
                        unknown_obs[key].append(value)
                        unknown_obs_count += 1
                    elif agent_to_select == -2:
                        agent_to_select = 'FORCED Meander'
                        key = f'Game {g}'
                        value = f'Step {t}'
                        if key not in unknown_obs:
                            unknown_obs[key] = []
                        unknown_obs[key].append(value)
                        unknown_obs_count += 1
                    else:
                        agent_to_select = 'Sleep'

                    # Track agent performance based on defense
                    key = agent_to_select

                    # Note: FORCED agent selections are treated as normal agent selection
                    if agent_to_select == 'FORCED BLine' or agent_to_select == 'FORCED Meander':
                        key = agent_to_select[7:]

                    # Track agent selection
                    if t == 5:
                        selected_agent_defense = key
                        if key not in agent_defense:
                            defense_history = [str(f'Game {g}')]
                            defense_score = []
                            agent_defense[key] = (1,defense_score, defense_history)
                        else:
                            current_count, current_score, current_history = agent_defense[key]
                            current_game = str(f'Game {g}')
                            current_history.append(current_game)
                            agent_defense[key] = (current_count + 1, current_score, current_history)

                    # Track global observations and actions
                    if key == 'BLine':
                        update_history(history_bline,blue_action,translated_action,g,t)
                        update_dictionary(global_bline,blue_obs,blue_action,translated_action,g,t)
                    else:
                        update_history(history_meander,blue_action,translated_action,g,t)
                        update_dictionary(global_meander,blue_obs,blue_action,translated_action,g,t)

                    state = cyborg.step(red_action, blue_action, mindrake=True)
                    red_reward = -state.reward
                    blue_obs = state.blue_observation
                    red_obs = state.red_observation
                    score += red_reward

                    # Update the game log file
                    file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,str(g))
                    original_stdout = sys.stdout

                    try:
                        with open(file, 'a') as file:
                            true_state = cyborg.get_agent_state('True')
                            true_table = true_obs_to_table(true_state,cyborg)
                            sys.stdout = file
                            print(f"Step{' ' * 7}\t:{' ' * 2}{t}")
                            print("Blu def.   \t: ",str(agent_to_select))

                            # Prints to indicate that random action was chosen instead of optimal action
                            if(blu_action_random):
                                print("Blu action \t: (?) ",str(cyborg.get_last_action('Blue')))
                            else:
                                print("Blu action \t: ",str(cyborg.get_last_action('Blue')))

                            if(red_action_random):
                                print("Red action \t: (?) ",str(cyborg.get_last_action('Red')))
                            else:
                                print("Red action \t: ",str(cyborg.get_last_action('Red')))

                            print(true_table)
                            print("Red reward\t:",red_reward)
                            print("Total reward\t:",score)
                            print("\n\n")
                    finally:
                        sys.stdout = original_stdout

                    if verbose:
                        print(f'---- Turn {t+1} ----')
                        # Cardiff actions are deterministic not stochastic! *** NEED UPDATE FOR MINDRAKE HERE ***
                        print(f"Blue selects {blue_action_list[cyborg.convert_blue_action(blue_action,mindrake=True)]} with probability {1*100:0.2f}%")
                        print()
                        print(f"Red selects {red_action_list[red_action]} with probability {red_extras['action_prob']*100:0.2f}%")
                        print()
                        # print(f'New Red observation: {red_obs}')
                        print(f'New Red observation: ')
                        print(cyborg._create_red_table())
                        print()
                        print(f"Reward: +{red_reward:0.1f}")
                        print(f"Score: {score:0.1f}")
                        print()

                # Record the score!
                current_count, current_score, current_history = agent_defense[selected_agent_defense]
                current_score.append(score)
                agent_defense[selected_agent_defense] = (current_count, current_score, current_history)

                # print(f"game {g} score {score}")
                file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,"_scores")
                with open(file, 'a') as file:
                    file.write(f'Game {g} - score is {score}\n')

                scores.append(score)
                if score > max_score:
                    max_score = score
                if score < min_score:
                    min_score = score
            
            avg_score = mean(scores)
            elapsed_time = time.time()-start_time

            # Calculate the % of BLine & Meander Defenses
            ## num_BLine_defenses = agent_defense['BLine'][0]
            ## num_Meander_defenses = agent_defense['Meander'][0]
            ## num_total_defenses = num_BLine_defenses + num_Meander_defenses
            ## percentage_BLine_defenses = (num_BLine_defenses / num_total_defenses) * 100
            ## percentage_Meander_defenses = (num_Meander_defenses / num_total_defenses) * 100
            # print(f'BLine defense ({num_BLine_defenses}/{num_total_defenses}) % is {percentage_BLine_defenses}')
            # print(f'Meander defense ({num_Meander_defenses}/{num_total_defenses}) % is {percentage_Meander_defenses}')

            # Calculate the isolated scores of BLine & Meander Defenses
            ## avg_BLine_score = mean(agent_defense['BLine'][1])
            ## avg_Meander_score = mean(agent_defense['Meander'][1])
            ## stddev_BLine_score = stdev(agent_defense['BLine'][1])
            ## stddev_Meander_score = stdev(agent_defense['Meander'][1])

            # print(f'Bline defense average score is {avg_BLine_score}')
            # print(f'Meander defense average score is {avg_Meander_score}')

            # Find minimum and maximum along with their indices
            min_value, min_index = min((val, idx) for idx, val in enumerate(scores))
            max_value, max_index = max((val, idx) for idx, val in enumerate(scores))
            print("\n\nvs_mindrake")
            print(f'Total time is {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
            print(f'Total games played is {games}')
            print(f'Timestpes per game is {timesteps}')
            ## print(f'BLine defense ({num_BLine_defenses}/{num_total_defenses}) % is {percentage_BLine_defenses}')
            ## print(f'Meander defense ({num_Meander_defenses}/{num_total_defenses}) % is {percentage_Meander_defenses}')
            ## print(f'Bline defense average reward is {avg_BLine_score}')
            ## print(f'Meander defense average reward is {avg_Meander_score}')
            ## print(f'Standard deviation of Bline defense reward is {stddev_BLine_score}')
            ## print(f'Standard deviation of Meander defense reward is {stddev_Meander_score}')
            print(f'Avg reward is {avg_score}')
            print(f'Min reward is {min_value} at Game {min_index}')
            print(f'Max reward is {max_value} at Game {max_index}')
            print(f'Standard deviation of {stdev(scores)}')
            print(f'Total number of states_encountered is {len(states_encountered)}\n')

            ## print(f'BLine Action Count {len(history_bline)}/145\n')
            ## sorted_bline = sorted(history_bline.items())
            ## for key, value in sorted_bline:
            ##     num_of_times_action_called = value[0]
            ##     action_name = value[1]
            ##     print(f'called {num_of_times_action_called}\ttimes -\t{action_name}')

            ## print(f'\nMeander Action Count {len(history_meander)}/145\n')
            ## sorted_meander = sorted(history_meander.items())
            ## for key, value in sorted_meander:
            ##     num_of_times_action_called = value[0]
            ##     action_name = value[1]
            ##     print(f'called {num_of_times_action_called}\ttimes -\t{action_name}')

            print(f'\nTotal number of games that encountered unknown_obs is {len(unknown_obs)}')
            print(f'Total number of unknown_obs encountered is {unknown_obs_count}\n')
            if(len(unknown_obs) > 0):
                for key, values in unknown_obs.items():
                    print(f'{key} encountered {len(values)} unknown_obs')

            # Add the logged information into the game's log file
            file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,"_log")
            with open(file, 'w') as file:
                file.write(f'vs_mindrake\n')
                file.write(f'Total time is {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n')
                file.write(f'Total games played is {games}\n')
                file.write(f'Timestpes per game is {timesteps}\n')
                ## file.write(f'BLine defense ({num_BLine_defenses}/{num_total_defenses}) % is {percentage_BLine_defenses}\n')
                ## file.write(f'Meander defense ({num_Meander_defenses}/{num_total_defenses}) % is {percentage_Meander_defenses}\n')
                ## file.write(f'Bline defense average reward is {avg_BLine_score}\n')
                ## file.write(f'Meander defense average reward is {avg_Meander_score}\n')
                ## file.write(f'Standard deviation of Bline defense reward is {stddev_BLine_score}\n')
                ## file.write(f'Standard deviation of Meander defense reward is {stddev_Meander_score}\n')
                file.write(f'Avg reward is {avg_score}\n')
                file.write(f'Min reward is {min_value} at Game {min_index}\n')
                file.write(f'Max reward is {max_value} at Game {max_index}\n')
                file.write(f'Standard deviation of {stdev(scores)}\n\n')
                file.write(f'Total number of states_encountered is {len(states_encountered)}\n')

                """
                file.write(f'BLine Action Count {len(history_bline)}/145\n')
                for key, value in sorted_bline:
                    num_of_times_action_called = value[0]
                    action_name = value[1]
                    file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

                file.write(f'\nMeander Action Count {len(history_meander)}/145\n')
                for key, value in sorted_meander:
                    num_of_times_action_called = value[0]
                    action_name = value[1]
                    file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

                file.write(f'\nTotal number of games that encountered unknown_obs is {len(unknown_obs)}')
                file.write(f'\nTotal number of unknown_obs encountered is {unknown_obs_count}\n')
                if(len(unknown_obs) > 0):
                    for key, values in unknown_obs.items():
                        file.write(f'{key} encountered {len(values)} unknown_obs\n')
                """

            # Track which games have 'BLine' and 'Meander Defenses'
            file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,"_defenses")
            """
            with open(file, 'w') as file:
                 for key, value in agent_defense.items():
                    file.write(f'{key}\n')
                    agent_defense_history = value[2]
                    for event in agent_defense_history:
                        file.write(f'{event}\n')
                    file.write('\n')
            """

            file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,"_states")
            with open(file, 'w') as file:
                for key, value in states_encountered.items():
                    file.write(f'{key}\n')

            file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,"_actions")
            """
            with open(file, 'w') as file:

                file.write(f'BLine Action Count {len(history_bline)}/145\n')
                for key, value in sorted_bline:
                    num_of_times_action_called = value[0]
                    action_name = value[1]
                    file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

                file.write(f'\nMeander Action Count {len(history_meander)}/145\n')
                for key, value in sorted_meander:
                    num_of_times_action_called = value[0]
                    action_name = value[1]
                    file.write(f'called {num_of_times_action_called}\ttimes -\t{action_name}\n')

                file.write("\n\n--BLINE--\n\n")

                for key, value in global_bline.items():
                    file.write(f'{key}\n')
                    for inside_key, inside_value in value.items():
                        action_count = inside_value[0]
                        action_name = inside_value[1]
                        action_history = inside_value[2]
                        file.write(f'{action_name} took place {action_count} times\n')
                        for event in action_history:
                            file.write(f'{event}\n')
                        file.write('\n')

                file.write("\n\n-MEANDER-\n\n\n\n")

                for key, value in global_meander.items():
                    file.write(f'{key}\n')
                    for inside_key, inside_value in value.items():
                        action_count = inside_value[0]
                        action_name = inside_value[1]
                        action_history = inside_value[2]
                        file.write(f'{action_name} took place {action_count} times\n')
                        for event in action_history:
                            file.write(f'{event}\n')
                        file.write('\n')
            """

            # Explicit report for the Game and Steps in which the agent encountered the unknown observation
            file = os.path.join(os.getcwd(),game_log_folder,"mindrake",str(timesteps),er_fileName,"_unknown-obs")
            with open(file, 'w') as file:
                for key, values in unknown_obs.items():
                    for element in unknown_obs[key]:
                        file.write(f'{key} {element}\n')
                    file.write(f'\n\n')

            if verbose and (games>1):
                print(f'Average Score for {games} Games is {avg_score}')
                print(f'High Score is {max_score}')
                print(f'Low Score is {min_score}')
    
            # Default return without the ER loops
            # return(avg_score)
            print(avg_score,'\n')

    print("sample_against_mindrake completed!\n")

def run_algorithm(config, env, algorithm_select):
    if(algorithm_select == "ppo"):
        return PPO(config=config, env=env)

    elif(algorithm_select == "impala"):
        return Impala(config=config, env=env)

    elif(algorithm_select == "dqn"):
        return DQN(config=config, env=env)
    else:
        raise ValueError("Selected algorithm not implemented!")

def get_algorithm_config(algorithm_select, blue):
    if(algorithm_select == "ppo"):
        if(blue):
            print("selecting blue ppo config")
            return blue_ppo_config
        else:
            print("selecting red ppo config")
            return red_ppo_config
    elif(algorithm_select == "impala"):
        if(blue):
            print("selecting blue impala config")
            return blue_impala_config
        else:
            print("selecting red impala config")
            return red_impala_config
    elif(algorithm_select == "dqn"):
        if(blue):
            print("selecting blue dqn config")
            return blue_dqn_config
        else:
            print("selecting red dqn config")
            return red_dqn_config
    else:
        raise ValueError("Selected algorithm config not implemented!")

def get_opponent_config(algorithm_select, blue):
    if(algorithm_select == "ppo"):
        if(blue):
            return blu_is_opponent_ppo_config
        return red_is_opponent_ppo_config
    elif(algorithm_select == "impala"):
        if(blue):
            return blu_is_opponent_impala_config
        return red_is_opponent_impala_config
    elif(algorithm_select == "dqn"):
        if(blue):
            return blue_is_opponent_dqn_config
        return red_is_opponent_dqn_config 
    else:
        raise ValueError("Selected algorithm config not implemented!")

def get_opponent(algorithm_select, config):
    if(algorithm_select == "ppo"):
        print("opponent is selecting ppo")
        return PPO(config=config)
    elif(algorithm_select == "impala"):
        print("opponent is selecting impala")
        return Impala(config=config)
    elif(algorithm_select == "dqn"):
        print("opponent is selecting dqn")
        return DQN(config=config)
    else:
        raise ValueError("Selected algorithm config not implemented!")

def get_timesteps():
    return timesteps

def get_algorithm_select():
    return algorithm

def update_dictionary(dictionary, key, value, translated_value, game, step):
    k = str(key)
    k = k.replace('\n', '')

    if k not in dictionary:
        dictionary[k] = {}
    if value not in dictionary[k]:
        action_history = [str(f'Game {game} Step {step}')]
        dictionary[k][value] = (1,translated_value,action_history)
    else:
        count, _, action_history = dictionary[k][value]
        current_action = str(f'Game {game} Step {step}')
        action_history.append(current_action)
        dictionary[k][value] = (count + 1, translated_value,action_history)

def update_history(dictionary, key, translated_value, game, step):
    if key not in dictionary:
        action_history = [str(f'Game {game} Step {step}')]
        dictionary[key] = (1, translated_value, action_history)
    else:
        count, _, action_history = dictionary[key]
        current_action = str(f'Game {game} Step {step}')
        action_history.append(current_action)
        dictionary[key] = (count + 1, translated_value,action_history)