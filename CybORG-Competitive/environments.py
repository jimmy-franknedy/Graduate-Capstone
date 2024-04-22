# Import the Cardiff Agent's Directory
from cardiff import *
from cardiff.cage2.Wrappers.BlueTableWrapper import BlueTableWrapper
from cardiff.cage2.Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from cardiff.cage2.Agents.MainAgent import MainAgent

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from wrapper import CompetitiveWrapper

import gym
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
    workers = 4
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
# print("checking blue_action_list: ",blue_action_list)

red_action_list = (
    red_lone_actions
    + list(product(red_network_actions, subnets))
    + list(product(red_host_actions, hostnames)))
cardiff_action_list = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                       132, 2, 15, 24, 25, 26, 27] #27 elements
# print("checking cardiff_action_list: ",cardiff_action_list)

# Batch and mini-batchsizes
b1 = 61440          # original batch size
mb1 = 3840          # ^

# Number of actions that red should take in sequence to achieve specific goal
red_action_sequence = 3

# Number of 'timesteps' red gets to take actions to try and achieve the specific action sequence
red_action_tries = pow(len(red_action_list),red_action_sequence)
red_multiplier = 5

b2 = red_action_tries * red_multiplier          # adjusted batch size given red has 38 possible actions; following same scaling as original
mb_scaler = 10
mb2 = b2 // mb_scaler                                  # ^

batch_size = b2
mini_batch_size = mb2

# print("batch_size: ", batch_size)
# print("mini_batch_size: ", mini_batch_size)
# print("rollout_fragment_length: ",int(batch_size/workers))

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

cardiff_config = {}

# Blu IMP Config
blue_impala_config = {
    "env": "blue_trainer",
    "num_gpus": ngpus,
    "num_workers": workers,
    "train_batch_size": blue_batch_size,
    "minibatch_buffer_size": blue_minibatch_size,
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

    "learner_queue_size": workers,}

# Red IMP Config
red_impala_config = {
    "env": "red_trainer",
    "num_gpus": ngpus,
    "num_workers": workers,
    "train_batch_size": red_batch_size,
    "minibatch_buffer_size": red_minibatch_size,
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

    "learner_queue_size": workers,}

# Blu Opp IMP Config
blu_is_opponent_impala_config = {
    "num_workers": 0,
    "num_gpus": 0,
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "observation_space": MultiBinary(blue_obs_space),
    "action_space": Discrete(len(blue_action_list)),
    "log_sys_usage": False,

    # Added
    "train_batch_size": blue_batch_size,
    "minibatch_buffer_size": blue_minibatch_size,
    "rollout_fragment_length": int(blue_batch_size/workers),}

# Red Opp IMP Config
red_is_opponent_impala_config = {
    "num_workers": 0,
    "num_gpus": 0,
    "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func},
    "observation_space": MultiBinary(red_obs_space),
    "action_space": Discrete(len(red_action_list)),
    "log_sys_usage": False,

    "train_batch_size": red_batch_size,
    "minibatch_buffer_size": red_minibatch_size,
    "rollout_fragment_length": int(red_batch_size/workers),}

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
        # print("self.action_space is ", self.action_space)

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

        print("dont print this")

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

        print("should be here!")

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

def cardiff_wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def build_cardiff_agent():

    # register cardiff environment
    print("creating the blue cardiff agent!")

    # remove later - for testing purposes
    # CardiffBlueEnv(env_config={"name": f"{experiment_name}_cardiff_blue"})
    
    select_env = "CardiffBlueEnv"
    register_env(
        select_env,
        lambda config: CardiffBlueEnv(
            env_config={"name": f"{experiment_name}_cardiff_blue"}
        )
    )
    print("returing...")
    return MainAgent()

def build_blue_agent(fresh, opponent=False, dedicated=False, cardiff=False):

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

def build_red_agent(fresh, opponent=False, dedicated=False, vs_cardiff=False):
    # register the custom environment
    if vs_cardiff:
        select_env = "DedicatedRedEnv_vs_cardiff"
        register_env(
            select_env,
            lambda config: DedicatedRedEnv_vs_cardiff(
                env_config={"name": f"{experiment_name}_DedicatedRedEnv_vs_cardiff"}
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

    base_cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
    # cardiff_cyborg = cardiff_wrap(CybORG(scenario_file="./scenario.yaml", environment="sim", agents={'Red': SleepAgent}))
    
    # wrapper to accept red and blue actions, and return observations
    cyborg = CompetitiveWrapper(env=base_cyborg, turns=timesteps, output_mode="vector")

    scores = []
    max_score = 0
    min_score = float('inf')

    for g in range(games):

        print("g: ",g)

        # the blue_obs given by 'cyborg.reset()' is not the same as the one given by default cyborg's reset()
        blue_obs, red_obs = cyborg.reset(cardiff=True)
        test_blue.end_episode()
        score = 0

        for t in range(timesteps):
            blue_action = test_blue.get_action(blue_obs)
            red_action, _, red_extras = test_red.compute_single_action(red_obs, full_fetch=True)
            print("environments.py - red action is: ", red_action)

            state = cyborg.step(red_action, blue_action, cardiff=True)

            # Logic reminder: Red wants to get as high of a score, while blue wants to get as low of a score
            red_reward = -state.reward

            # double check that blue observation is in right format here
            blue_obs = state.blue_observation
            print(len(blue_obs))


            red_obs = state.red_observation

            score += red_reward

        # print("finished!")
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
    print("calling get_opponent_config()")
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