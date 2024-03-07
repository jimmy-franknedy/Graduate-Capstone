from environments import build_blue_agent, build_red_agent, sample

import ray
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
logger = logging.getLogger(__name__)

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

tolerance = 3 # number of batches without improvement before ending training
generations = 100

# Create Initial Policies
ray.init(ignore_reinit_error=True, log_to_driver=False)
print("pass 1")

blue = build_blue_agent()
print("pass 2")

red = build_red_agent(opponent=True)
print("pass 3")

blue_scores = []
red_scores = []
print("pass 4")

print()
print("+---------------------------------+")
print("| Blue Competitive Training Start |")
print("+---------------------------------+")
print()

for g in range(1, generations+1):

    if (g < 10):
        dashes = 14
    elif (g < 100):
        dashes = 15
    else:
        dashes = 16
    print('+'+'-'*dashes+'+')            
    print(f"| Generation {g} |")
    print('+'+'-'*dashes+'+')
    print()

    blue.restore("./policies/blue_competitive_pool/competitive_blue_0/checkpoint_000000")

    b = 0 # b tracks the batches of training completed
    blue_min = float('inf')
    tol = tolerance
    while True:
        b += 1
        result = blue.train()
        blue_score = -result["sampler_results"]["episode_reward_mean"]
        entropy = result['info']['learner']['default_policy']['learner_stats']['entropy']
        vf_loss = result['info']['learner']['default_policy']['learner_stats']['vf_loss']
        print(f"Batch {b} -- Blue Score: {blue_score:0.2f}    Entropy: {entropy:0.2f}    VF Loss: {vf_loss:0.2f}") 
        if b > 1:
            if (blue_score < blue_min):
                blue_min = blue_score
                tol = tolerance
                checkpoint_path = blue.save(checkpoint_dir=f"./policies/blue_competitive_pool/competitive_blue_{g}")
                path_file = open(f"./policies/blue_competitive_pool/competitive_blue_{g}/checkpoint_path", "w")
                path_file.write(checkpoint_path)
                path_file.close()
            elif(tol > 1):
                tol -= 1
            # when agent is no longer improving, break and save the new competitive agent
            else:
                blue_scores.append(blue_min)
                blue.restore(checkpoint_path)
                print(checkpoint_path)
                break

    pool_size = g
    pool_file = open("./policies/blue_competitive_pool/pool_size", "w")
    pool_file.write(str(pool_size))
    pool_file.close()
    print()

    red.restore("./policies/red_opponent_pool/opponent_red_0/checkpoint_000000")

    b = 0
    red_max = 0
    tol = tolerance
    while True:
        b += 1
        result = red.train()
        red_score = result["sampler_results"]["episode_reward_mean"]
        entropy = result['info']['learner']['default_policy']['learner_stats']['entropy']
        vf_loss = result['info']['learner']['default_policy']['learner_stats']['vf_loss']
        print(f"Batch {b} -- Red Score: {red_score:0.2f}    Entropy: {entropy:0.2f}    VF_loss: {vf_loss:0.2f}")
        if b > 1:
            if (red_score > red_max):
                red_max = red_score
                tol = tolerance
                checkpoint_path = red.save(checkpoint_dir=f"./policies/red_opponent_pool/opponent_red_{g}")   
                path_file = open(f"./policies/red_opponent_pool/opponent_red_{g}/checkpoint_path", "w")
                path_file.write(checkpoint_path)
                path_file.close()
            elif(tol > 1):
                tol -= 1
             # when agent is no longer improving, break and save the new best-response agent
            else:
                red_scores.append(red_max)
                red.restore(checkpoint_path)
                print(checkpoint_path)
                break

    pool_file = open("./policies/red_opponent_pool/pool_size", "w")
    pool_file.write(str(pool_size))
    pool_file.close()
    print()

    print(f'Blue Scores so far {["%.2f" % i for i in blue_scores]}')
    print(f'Red Scores so far {["%.2f" % i for i in red_scores]}')
    print()
    
    print(f'-------- Sample Game for Generation {g} --------')
    sample(red, blue, verbose=True, show_policy=True)
    print()