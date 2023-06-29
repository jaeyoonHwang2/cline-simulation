'''
MIT License
Copyright (c) Chen-Yu Yen - Soheil Abbasloo 2020
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from telnetlib import DM
import threading
import logging
import tensorflow as tf
import sys
from agent import Agent
import os

from tcpCubic import TcpCubic
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import gym
import numpy as np
import time
import random
import datetime
import signal
import pickle
from utils import logger, Params
from envwrapper import Env_Wrapper, TCP_Env_Wrapper
import json
from ns3gym import ns3env
import math

tf.get_logger().setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
parser.add_argument('--eval', action='store_true', default=True, help='default is  %(default)s')
parser.add_argument('--tb_interval', type=int, default=1)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--mem_r', type=int, default = 123456)
parser.add_argument('--mem_w', type=int, default = 12345)
base_path = '../../../Desktop/inferLearner'
job_name = 'actor'  


# parameters from parser
global config
global params
config = parser.parse_args()


# parameters from .json file
params = Params(os.path.join(base_path,'params.json'))
seed = 12 
debug = False  

# parameters for config
startSim = 0.00001
port = 6357
stepTime = 0.000001
simTime = 20
simArgs = {"--duration": simTime, }

# monitoring variables
numSenders = 1
mtp = 20000

cubicAgent = [TcpCubic() for i in range(numSenders)]
print ("before activation")
envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)    
print ("after activation")

obs = envNs3.reset()

Uuid = obs[0] - 1

done = False
info = None

# init. variables

alpha = [1 for i in range (numSenders)]
a = [0 for i in range (numSenders)]
ack_count = [0 for i in range (numSenders)] 
loss_count = [0 for i in range (numSenders)]
rtt_sum = [0 for i in range (numSenders)] 
min_rtt = [0 for i in range (numSenders)]
srtt = [0 for i in range (numSenders)]
timestamp = [0 for i in range (numSenders)] 
interval_cnt = [0 for i in range (numSenders)]
throughput = [0 for i in range (numSenders)]
loss_rate = [0 for i in range (numSenders)]
rtt = [0 for i in range (numSenders)]
epoch = [0 for i in range (numSenders)]
epoch_ = [0 for i in range (numSenders)]
r = [0 for i in range (numSenders)]
ret = [0 for i in range (numSenders)]
action = [[0, 0] for i in range (numSenders)]
orca_init = [True for i in range (numSenders)]
slow_start = [True for i in range (numSenders)]
terminal = [False for i in range (numSenders)]
error_code = [True for i in range (numSenders)]

while True: # one iteration

    Uuid = obs[0] - 1
    
    # increase counts
    ack_count[Uuid] += 1
    if not (obs[11]): loss_count[Uuid] += 1
    rtt_sum[Uuid] += obs[9] / 1000

    if not (min_rtt[Uuid]): min_rtt[Uuid] = obs[9] / 1000 # min_rtt_ms
    else: min_rtt[Uuid] = min(min_rtt[Uuid], obs[9] / 1000)

    if (srtt[Uuid]): srtt[Uuid] = 7/8 * srtt[Uuid] + 1/8 * obs[9] / 1000 # srtt_ms 
    else: srtt[Uuid] = obs[9] / 1000

    # mtp has passed from the last report 
    # -> Orca will take into action
    if (int(obs[2] / mtp) != timestamp[Uuid]):
        
        if not (timestamp[Uuid]): timestamp[Uuid] = int(obs[2] / mtp) - 1
        
        interval_cnt[Uuid] = int(obs[2] / mtp) - timestamp[Uuid]

        # throughput / loss rate / rtt is calculated
        throughput[Uuid] = ack_count[Uuid] * 1500 * 8 / (interval_cnt[Uuid] * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
        loss_rate[Uuid] = loss_count[Uuid] * 1500 * 8 / (interval_cnt[Uuid] * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
        rtt[Uuid] = rtt_sum[Uuid] / ack_count[Uuid] # ms

        # print throughput / loss rate / rtt
        for i in range(interval_cnt[Uuid]):
            print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "throughput", throughput[Uuid])
            print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "loss_rate", loss_rate[Uuid])
            print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "srtt", srtt[Uuid])
        
            ack_count[Uuid] = 0
            loss_count[Uuid] = 0
            rtt_sum[Uuid] = 0

            timestamp[Uuid] = int(obs[2] / mtp)

    if not (srtt): srtt = 20000
    action[Uuid], slow_start[Uuid] = cubicAgent[Uuid].get_action(obs, srtt[Uuid], min_rtt[Uuid], done, info)
    print (Uuid, (obs[2] / 1000000), "cubic_cwnd", action[Uuid][1])

    obs, reward, done, info = envNs3.step(action[Uuid])


    if done: 

        done = False
        info = None
        print ("An episode is over")
        break
