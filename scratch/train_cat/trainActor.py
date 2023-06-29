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
import socket
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
parser.add_argument('--eval', action='store_true', default=False, help='default is  %(default)s')
parser.add_argument('--tb_interval', type=int, default=1)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--mem_r', type=int, default = 123456)
parser.add_argument('--mem_w', type=int, default = 12345)
base_path = '~/train_learner'
job_name = 'actor'  

# Connect sockets to the learner for (multi-actors sync.)
HOST = '127.0.0.1'
PORT = 9999
actor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
actor_socket.connect((HOST, PORT))
print("Actor socket connected to learner")

# parameters from parser
global config
global params
config = parser.parse_args()

# parameters from .json file
params = Params(os.path.join(base_path,'params.json'))
seed = 12 
debug = False  

# parameters for config
task = 0
startSim = 0.00001
port = 6625+task
stepTime = 0.000001
simArgs = {"--duration":  params.dict['simTime'], }

# monitoring variables
numSenders = 1
mtp = 20000
iteration = 0

alpha = 1
ack_count = 0 
loss_count = 0
rtt_sum = 0 
min_rtt = 0
srtt = 0
timestamp = 0 
interval_cnt = 0
throughput = 0
loss_rate = 0
rtt = 0
epoch = 0
epoch_ = 0
r = 0
ret = 0

if params.dict['single_actor_eval']:
    local_job_device = ''
    shared_job_device = ''
    def is_actor_fn(i): return True
    global_variable_device = '/gpu'
    is_learner = False
    server = tf.train.Server.create_local_server()
    filters = []
    
else:

    local_job_device = '/job:%s/task:%d' % (job_name, task)
    shared_job_device = '/job:learner/task:0'

    is_learner = job_name == 'learner'
    global_variable_device = shared_job_device + '/gpu'

    def is_actor_fn(i): return job_name == 'actor' and i == task

    if params.dict['remote']:
        cluster = tf.train.ClusterSpec({
            'actor': params.dict['actor_ip'][:params.dict['num_actors']],
            'learner': [params.dict['learner_ip']]
        })
    else:
        cluster = tf.train.ClusterSpec({
                'actor': ['localhost:%d' % (4001 + i) for i in range(params.dict['num_actors'])],
                'learner': ['localhost:4000']
            })

    server = tf.train.Server(cluster, job_name=job_name,
                            task_index=task)
    filters = [shared_job_device, local_job_device]

if params.dict['use_TCP']:
    env_str = "TCP"
    env_peek = TCP_Env_Wrapper(env_str, params,use_normalizer=params.dict['use_normalizer'])

else:
    env_str = 'YourEnvironment'
    env_peek =  Env_Wrapper(env_str)

s_dim, a_dim = env_peek.get_dims_info()
action_scale, action_range = env_peek.get_action_info()

if not params.dict['use_TCP']:
    params.dict['state_dim'] = s_dim
if params.dict['recurrent']:
    s_dim = s_dim * params.dict['rec_dim']

if params.dict['use_hard_target'] == True:
    params.dict['tau'] = 1.0

with tf.Graph().as_default(),\
    tf.device(local_job_device + '/cpu'):

    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    actor_op = []
    check_op = []
    now = datetime.datetime.now()
    tfeventdir = os.path.join( base_path, params.dict['logdir'], job_name+str(task) )
    params.dict['train_dir'] = tfeventdir

    if not os.path.exists(tfeventdir):
        os.makedirs(tfeventdir)
    summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

    with tf.device(shared_job_device):

        agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])
        cubicAgent = TcpCubic()

        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
        queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")

    if is_learner:
        with tf.device(params.dict['device']):
            agent.build_learn()

            agent.create_tf_summary()

        if config.load is True and config.eval==False:
            if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
                with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                    replay_memory = pickle.load(fp)

    for i in range(params.dict['num_actors']):
        if is_actor_fn(i):
            env = TCP_Env_Wrapper(env_str, params, config=config, for_init_only=False, use_normalizer=params.dict['use_normalizer'])
            envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

            a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0')
            a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action')
            a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward')
            a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1')
            a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal')
            a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]

            with tf.device(shared_job_device):
                actor_op.append(queue.enqueue(a_buf))

    if is_learner:
        Dequeue_Length = params.dict['dequeue_length']
        dequeue = queue.dequeue_many(Dequeue_Length)

    queuesize_op = queue.size()

    if params.dict['ckptdir'] is not None:
        params.dict['ckptdir'] = os.path.join( base_path, params.dict['ckptdir'])
        print("## checkpoint dir:", params.dict['ckptdir'])
        isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
        print("## checkpoint exists?:", isckpt)
        if isckpt== False:
            print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")
    else:
        params.dict['ckptdir'] = tfeventdir

    tfconfig = tf.ConfigProto(allow_soft_placement=True)

    if params.dict['single_actor_eval']:
        mon_sess = tf.train.SingularMonitoredSession(
            checkpoint_dir=params.dict['ckptdir'])
    else:
        mon_sess = tf.train.MonitoredTrainingSession(master=server.target,
                save_checkpoint_secs=15,
                save_summaries_secs=None,
                save_summaries_steps=None,
                is_chief=is_learner,
                checkpoint_dir=params.dict['ckptdir'],
                config=tfconfig,
                hooks=None)

    agent.assign_sess(mon_sess)
    
    while True:

        obs = envNs3.reset()
        cubicAgent.reset()
        
        done = False
        info = None
        
        s0 = np.zeros([s_dim])
        s1 = np.zeros([s_dim])
        s0_rec_buffer = np.zeros([s_dim]) 
        s1_rec_buffer = np.zeros([s_dim])  
        if params.dict['recurrent']: a = agent.get_action(s0_rec_buffer, not config.eval)
        else: a = agent.get_action(s0, not config.eval)
        a = a[0][0]

        # init. variables
        alpha = 1
        ack_count = 0 
        loss_count = 0
        rtt_sum = 0 
        min_rtt = 0
        srtt = 0
        timestamp = 0 
        interval_cnt = 0
        throughput = 0
        loss_rate = 0
        rtt = 0
        epoch_ = 0
        r = 0
        ret = 0
        action = [0, 0]
        orca_init = True
        slow_start = True

        while True: # one iteration
        
            Uuid = obs[0] - 1
            
            # increase counts
            ack_count += 1
            if not (obs[11]): loss_count += 1
            rtt_sum += obs[9] / 1000

            if not (min_rtt): min_rtt = obs[9] / 1000 # min_rtt_ms
            else: min_rtt = min(min_rtt, obs[9] / 1000)

            if (srtt): srtt = 7/8 * srtt + 1/8 * obs[9] / 1000 # srtt_ms 
            else: srtt = obs[9] / 1000

            if (obs[7]): # ack is recognized            

                # mtp has passed from the last report 
                # -> Orca will take into action
                if (int(obs[2] / mtp) != timestamp):
                    
                    interval_cnt = int(obs[2] / mtp) - timestamp
                    print (timestamp)
                    
                    # throughput / loss rate / rtt is calculated
                    throughput = ack_count * 1500 * 8 / (interval_cnt * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
                    loss_rate = loss_count * 1500 * 8 / (interval_cnt * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
                    rtt = rtt_sum / ack_count # ms

                    # print throughput / loss rate / rtt
                    for i in range(interval_cnt):
                        print (Uuid, (timestamp + 1 + i) * 0.02 + iteration * params.dict['simTime'], "throughput", throughput)
                        print (Uuid, (timestamp + 1 + i) * 0.02 + iteration * params.dict['simTime'], "loss_rate", loss_rate)
                        print (Uuid, (timestamp + 1 + i) * 0.02 + iteration * params.dict['simTime'], "srtt", srtt)
                    
                    if not (slow_start):                                            

                        if (orca_init):
                            orca_init = False

                            s0 = env.reset(obs, throughput, rtt, loss_rate, srtt, min_rtt, ack_count, iteration)
                            s0_rec_buffer = np.zeros([s_dim]) 
                            s1_rec_buffer = np.zeros([s_dim]) 
                            s0_rec_buffer[-1*params.dict['state_dim']:] = s0
                            if params.dict['recurrent']: a = agent.get_action(s0_rec_buffer, not config.eval)                        
                            else: a = agent.get_action(s0, not config.eval)
                            a = a[0][0]
                            alpha = math.pow(4, a)

                        else:

                            # Orca action
                            # epoch: total epochs / epoch_: epochs for this episode
                            epoch += 1
                            epoch_ += 1
                            
                            # return state and reward & erase used states
                            s1, r, terminal, error_code = env.step(obs, throughput, rtt, loss_rate, srtt, min_rtt, ack_count, iteration, eval_= eval)

                            if (epoch_ <= 50): ret = (ret * (epoch_ - 1) + r) / epoch_
                            else: ret = 0.98 * ret + 0.02 * r

                            # print reward / return                    
                            print (Uuid, int(obs[2] / mtp) * 0.02 + iteration * params.dict['simTime'], "reward", r, "epochs", epoch)
                            print (Uuid, int(obs[2] / mtp) * 0.02  + iteration * params.dict['simTime'], "return", ret, "epochs", epoch)

                            if error_code == True:
                            
                                s1_rec_buffer = np.concatenate( (s0_rec_buffer[params.dict['state_dim']:], s1) )
                                if params.dict['recurrent']: a1 = agent.get_action(s1_rec_buffer, not config.eval)
                                else: a1 = agent.get_action(s1,not config.eval)
                                
                                # action is a1 (later used to calculate coefficient k = 4^a1)
                                a1 = a1[0][0]
                                
                            else: 

                                print("TaskID:"+str(task)+"Invalid state received...\n")
                                continue
                                        
                            if params.dict['recurrent']: fd = {a_s0:s0_rec_buffer, a_action:a, a_reward:np.array([r]), a_s1:s1_rec_buffer, a_terminal:np.array([terminal], np.float)}
                            else: fd = {a_s0:s0, a_action:a, a_reward:np.array([r]), a_s1:s1, a_terminal:np.array([terminal], np.float)}

                            if not config.eval: mon_sess.run(actor_op, feed_dict=fd)      

                            s0 = s1
                            a = a1
                            alpha = math.pow(4, a1)
                            print (Uuid, int(obs[2] / mtp) * 0.02 + iteration * params.dict['simTime'], "orca_alpha", alpha)

                            if ((epoch % params.dict['numSample']) == 0):           
                                actor_socket.send(json.dumps({"Wait": 1}).encode())    
                                print ("Waiting for learner to update", time.time())
                                json.loads(actor_socket.recv(1024).decode()).get("Free")
                                print ("Learner has updated", time.time())

                        if params.dict['recurrent']: s0_rec_buffer = s1_rec_buffer

                        if not params.dict['use_TCP'] and (terminal):
                            if agent.actor_noise != None:
                                agent.actor_noise.reset()

                        # just to check CUBIC logic
                        ##alpha = 1

                        action[1] = int(obs[5] * alpha)
                        if (action[1] > 100000): action[1] = 100000
                        if (action[1] < 180): action[1] = 180
                        action[0] = int(1.2 * max(action[1], obs[8]) * 8 / (srtt / 1000) * (obs[6] + 60) / obs[6]) # pacingRate: mss -> mtu
                        print (Uuid, (obs[2] / 1000000) + iteration * params.dict['simTime'], "orca_cwnd", action[1])
                        
                        ack_count = 0
                        loss_count = 0
                        rtt_sum = 0

                        timestamp = int(obs[2] / mtp)
    
                        obs, reward, done, info = envNs3.step(action)

                    
                    else: # slow start
                  
                        action, slow_start = cubicAgent.get_action(obs, srtt, min_rtt, done, info)
                        print (Uuid, (obs[2] / 1000000) + iteration * params.dict['simTime'], "cubic_cwnd", action[1])
                        
                        ack_count = 0
                        loss_count = 0
                        rtt_sum = 0

                        timestamp = int(obs[2] / mtp)
    
                        obs, reward, done, info = envNs3.step(action)

                else: # normal CUBIC
                
                    action, slow_start = cubicAgent.get_action(obs, srtt, min_rtt, done, info)
                    print (Uuid, (obs[2] / 1000000) + iteration * params.dict['simTime'], "cubic_cwnd", action[1])

                    obs, reward, done, info = envNs3.step(action)                    

            else: # no ack (probably loss)
            
                action, slow_start = cubicAgent.get_action(obs, srtt, min_rtt, done, info)
                print (Uuid, (obs[2] / 1000000) + iteration * params.dict['simTime'], "cubic_cwnd", action[1])

                obs, reward, done, info = envNs3.step(action)


            if done: 

                done = False
                info = None
                iteration += 1 
                print ("An episode is over")
                break
