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

from re import U
from telnetlib import DM
import threading
import logging
import tensorflow as tf
import sys
from agent import Agent
import os
from copaAgent import Sender_COPA

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
from ns3gym import ns3env
import math

print("pid", os.getpid())

#def evaluate_TCP(env, agent, epoch, summary_writer, params, s0_rec_buffer, eval_step_counter, envNs3, obs, mtpChecker):

tf.get_logger().setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
parser.add_argument('--eval', action='store_true', default=False, help='default is  %(default)s')
parser.add_argument('--tb_interval', type=int, default=1)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--mem_r', type=int, default = 123456)
parser.add_argument('--mem_w', type=int, default = 12345)
job_name = 'actor'  

# parameters from parser
global config
config = parser.parse_args()

# parameters from .json file
seed = 12 
debug = False  

# parameters for config
startSim = 0.00001
port = 8446
mtp = 20000 # Orca's default mtp is 20000
numSenders = 1
simTime = 20
stepTime = 0.000000001
simArgs = {"--duration": simTime, }
delta = 2

# initialize variables
done = False
info = None
iteration = 0
mtu = 150
epoch = [0 for i in range (numSenders)]
goOrca = [False for i in range (numSenders)]
initOrca = [True for i in range (numSenders)]
cubicActed = [True for i in range (numSenders)]

# monitoring variables
mtpThr = [0 for i in range (numSenders)]
mtpD = [0 for i in range (numSenders)]
mtpChecker = [0 for i in range (numSenders)]
numAcked = [0 for i in range (numSenders)]
numRtt = [0 for i in range (numSenders)]
srtt = [0 for i in range (numSenders)]
aggDelay = [0 for i in range (numSenders)]
dMin = [0 for i in range (numSenders)]
lastTime = [0 for i in range (numSenders)]
isReturn = [0 for i in range (numSenders)]
aggAck = [0 for i in range (numSenders)]
aggRtt = [0 for i in range (numSenders)]
unitAck = 1024 * 1024 
unitRtt = 0

nextTxSeq = [0 for i in range (numSenders)]
nextTxTime = [0 for i in range (numSenders)]
vel = [1 for i in range(numSenders)]
cwnd = [1 for i in range(numSenders)]


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


env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

# COPA
copaAgents = [Sender_COPA(segment_size=mtu - 60, delta = 1 / delta, flow_ID=i, initial_cwnd=cwnd[i], mode='ss', v = vel[i]) for i in range(numSenders)]

ack = [0 for i in range(numSenders)]  

obs = env.reset()

while True:

    Uuid = obs[0]-1

    unitRtt = obs[10]


    ack_count[Uuid] += 1
    if not (obs[11]): loss_count[Uuid] += 1
    rtt_sum[Uuid] += obs[9] / 1000

    if not (min_rtt[Uuid]): min_rtt[Uuid] = obs[9] / 1000 # min_rtt_ms
    else: min_rtt[Uuid] = min(min_rtt[Uuid], obs[9] / 1000)

    if (srtt[Uuid]): srtt[Uuid] = 7/8 * srtt[Uuid] + 1/8 * obs[9] / 1000 # srtt_ms 
    else: srtt[Uuid] = obs[9] / 1000

    if (obs[7]): # ack is recognized            

        # mtp has passed from the last report 
        # -> Orca will take into action
        if (int(obs[2] / mtp) != timestamp[Uuid]):
            
            if not (timestamp[Uuid]): timestamp[Uuid] = int(obs[2] / mtp) - 1
            
            interval_cnt[Uuid] = int(obs[2] / mtp) - timestamp[Uuid]

            # throughput / loss rate / rtt is calculated
            throughput[Uuid] = ack_count[Uuid] * 1500 * 8 / (interval_cnt[Uuid] * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
            if (throughput[Uuid] > 1000): throughput[Uuid] = 1000
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

        if (nextTxTime[Uuid] <= obs[2] - obs[9]): ##### major
        #if (nextTxSeq[Uuid] <= obs[14]):
            copaAgents[Uuid].update_velocity_hwang(obs)   
            print (Uuid, "sendRate", obs[2] / 1000000, (obs[14]-nextTxSeq[Uuid])*8*((obs[6]+60)/obs[6])/((obs[2]-nextTxTime[Uuid])/1000000))
            nextTxSeq[Uuid] = obs[14]             
            nextTxTime[Uuid] = obs[2]
            #nextTxSeq[Uuid] = obs[14] + action[1] 
        action = copaAgents[Uuid].get_action(obs)

    print (Uuid, "copa_action", obs[2] / 1000000, action[1])

    obs, rewardDontUse, done, info = env.step(action)         


    if done: 
        
        done = False
        info = None
        break            

